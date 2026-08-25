import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.core import LightningDataModule, LightningModule
    from lightning.pytorch.loggers.logger import Logger

from loguru import logger as log

from nsbi.utils import hydra_utils as utils
from nsbi.utils.lightning_utils import find_latest_checkpoint
from nsbi.utils.ray_utils import parse_dist


def main_function(cfg: DictConfig) -> None:
    """Trains or Evaluation the model.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "medium"))

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    stage = cfg.get("stage", "fit")

    # Directory that `test`/`predict`/`resume` search for checkpoints. An explicit top-level
    # `ckpt_path` wins; otherwise it is derived from the *first* configured ModelCheckpoint's
    # `dirpath` -- conventionally the primary (val_loss) one, so a later checkpoint callback
    # must not overwrite it. If nothing sets `dirpath`, it falls back below to the location
    # Lightning itself would have picked.
    ckpt_path = cfg.get("ckpt_path", None)
    derived_ckpt_path = None

    for key, callback_config in cfg.get("callbacks").items():
        if isinstance(callback_config, DictConfig) and "_target_" in callback_config:
            target = callback_config._target_
            if target == "lightning.pytorch.callbacks.ModelCheckpoint":
                derived_ckpt_path = derived_ckpt_path or callback_config.get("dirpath", None)
            if (
                "RichProgressBar" in target
                and cfg.get("trainer", {}).get("enable_progress_bar", False) is False
            ):
                # remove RichProgressBar callback if progress bar is disabled
                log.info("Removing <{}> callback as progress bar is disabled.", key)
                cfg.callbacks.pop(key)

    ckpt_path = ckpt_path or derived_ckpt_path

    ckpt_file = cfg.get("ckpt_file", None)
    assert not (stage == "finetune" and ckpt_file is None), (
        "In fine-tuning stage, a checkpoint file (ckpt_file) must be provided."
    )

    log.info("Instantiating datamodule <{}>", cfg.datamodule._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    if stage in ["fit", "finetune", "resume"]:
        datamodule.setup("fit")
    else:
        datamodule.setup(stage)

    log.info("Instantiating model <{}>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("do_compile", False):
        compile_kwargs = cfg.compile_kwargs
        log.info("Compiling model with torch.compile()")
        if hasattr(model, "network"):
            model.network = torch.compile(model.network, **compile_kwargs)  # type: ignore
        elif hasattr(model, "model"):
            model.model = torch.compile(model.model, **compile_kwargs)  # type: ignore
        else:
            # If the model does not have a 'network' attribute, compile the model directly.
            # This is useful for models that do not follow the typical structure.
            log.warning("Model does not have a 'network' attribute. Compiling the model directly.")
            # This is a fallback and may not be suitable for all models.
            model = torch.compile(model, **compile_kwargs)  # type: ignore

    log.info("Instantiating loggers...")
    loggers: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))  # type: ignore

    # Use the experiment ID in the ModelCheckpoint callback if it exists.
    if loggers:
        logger = loggers[0]
        filename_suffix = (
            str(logger.experiment.id)  # type: ignore
            if (
                hasattr(logger, "experiment")
                and hasattr(logger.experiment, "id")  # type: ignore
                and logger.experiment.id is not None  # type: ignore
            )
            else ""
        )

        # Prefix the *configured* filename with the experiment ID, when the logger has one
        # (WandB does; CSVLogger does not). Without an ID the configured `filename` is left
        # alone rather than being replaced by a generic epoch/step template.
        if filename_suffix:
            for callback_config in cfg.get("callbacks").values():
                if isinstance(callback_config, DictConfig) and "_target_" in callback_config:
                    if callback_config._target_ == "lightning.pytorch.callbacks.ModelCheckpoint":
                        base = callback_config.get("filename") or "{epoch}-{step}"
                        callback_config.filename = (
                            f"best-{filename_suffix.replace('/', '-')}-{base}"
                        )

    if ckpt_path is None:
        # Nothing configured a ModelCheckpoint `dirpath`, so Lightning picks the directory
        # itself in ModelCheckpoint.__resolve_ckpt_dir: <logger.save_dir>/<logger.name>/
        # version_<N>/checkpoints, or <default_root_dir>/checkpoints when there is no logger.
        # Reconstruct that here minus the version subdirectory -- find_latest_checkpoint()
        # rglobs, so leaving it off finds the previous run's checkpoints instead of the fresh
        # version directory this run just created.
        default_root_dir = cfg.get("trainer", {}).get("default_root_dir", None) or Path.cwd()
        if loggers:
            save_dir = getattr(loggers[0], "save_dir", None) or default_root_dir
            ckpt_path = str(Path(save_dir) / str(loggers[0].name))
        else:
            ckpt_path = str(Path(default_root_dir) / "checkpoints")
        log.info("No ModelCheckpoint dirpath configured; derived ckpt_path: {}", ckpt_path)

    # add TuneReportCallback if using Ray Tune
    if cfg.get("do_hpo_tune", False):
        log.info("Adding Ray Tune callback to report metrics to Ray Tune...")

        if "callbacks" not in cfg:
            cfg.callbacks = {}
        # Metrics to report to Tune are taken from hpo_tune.monitor; each name must
        # match a key logged into trainer.callback_metrics (e.g. "val_l1_energy_ws").
        monitor = cfg.hpo_tune.get("monitor", ["val_loss"])
        cfg.callbacks["ray_tune_report_callback"] = {
            "_target_": "ray.tune.integration.pytorch_lightning.TuneReportCallback",
            "metrics": {name: name for name in monitor},
            "on": "validation_end",
        }
        log.info("Ray Tune callback added.")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    if stage in ("test", "predict") and (
        cfg.trainer.get("devices", 1) != 1 or cfg.trainer.get("num_nodes", 1) != 1
    ):
        # Evaluate on a single device: distributed strategies pad the dataset via
        # DistributedSampler so some samples are evaluated twice, biasing metrics.
        log.info("Forcing devices=1, num_nodes=1 for the {} stage.", stage)
        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1

    log.info("Instantiating trainer <{}>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if (
        cfg.get("suppress_accumulate_grad_warning", False)
        and isinstance(trainer.strategy, L.pytorch.strategies.DDPStrategy)
    ):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }
    utils.log_hyperparameters(object_dict)

    if stage == "finetune":
        log.info("Finetuning the model..")
        log.info("Loading checkpoint from path {}", ckpt_file)
        # check the path and find the best checkpoint.
        model.load_state_dict(torch.load(ckpt_file)["state_dict"])
        ckpt_file = None
    elif stage == "resume":
        ckpt_file = find_latest_checkpoint(ckpt_path) if ckpt_path else None
        if ckpt_file is None:
            raise ValueError(f"No checkpoint found under ckpt_path={ckpt_path!r} to resume from.")
        log.info("Resuming training from checkpoint: {}", ckpt_file)
    elif stage == "fit":
        # Training from scratch: drop any `ckpt_file` left in the config, otherwise the
        # trainer.fit() call below would silently resume from it.
        ckpt_file = None

    if stage in ["fit", "finetune", "resume"]:
        log.info("Starting training!")
        # NB: Lightning's `ckpt_path` argument takes a checkpoint *file*, unlike this module's
        # `ckpt_path` config key, which is the *directory* that file is searched for in.
        trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=ckpt_file,
        )
    elif stage == "test":
        log.info("Starting testing!")
        ckpt_file = find_latest_checkpoint(ckpt_path) if ckpt_path else None
        if ckpt_file:
            log.info("Testing model with checkpoint: {}", ckpt_file)
        else:
            raise ValueError(f"No checkpoint found under ckpt_path={ckpt_path!r} for testing.")

        model.load_state_dict(torch.load(ckpt_file)["state_dict"])
        with torch.no_grad():
            model.eval()
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_file)

    elif stage == "predict":
        log.info("Starting prediction!")
        ckpt_file = find_latest_checkpoint(ckpt_path) if ckpt_path else None
        if ckpt_file:
            log.info("Predicting with model from checkpoint: {}", ckpt_file)
        else:
            raise ValueError(f"No checkpoint found under ckpt_path={ckpt_path!r} for prediction.")

        # model.load_state_dict(torch.load(ckpt_file)["state_dict"])
        trainer.predict(
            model=model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=False,
            ckpt_path="best",
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main_ensemble_function(cfg: DictConfig) -> None:
    """Train an ensemble of models for wifi (w_i f_i) ensembling.

    Dispatches to the single-device path (one Ray task / GPU per member, many members run
    concurrently across the cluster) or the distributed path (each member trained across
    multiple GPUs via a Ray Train TorchTrainer) based on ``ensemble.scaling.strategy`` --
    mirroring how ``main_tune_function`` dispatches on ``hpo_tune.scaling.strategy``. The
    single-device path is the simple/original one; all distributed-specific machinery lives in
    ``nsbi.tools.ensemble_distributed``.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    scaling_cfg = cfg.get("ensemble", {}).get("scaling", {})
    strategy_name = scaling_cfg.get("strategy", None)

    if strategy_name == "single_device" or not strategy_name:
        _main_ensemble_single_device(cfg)
    else:
        from nsbi.tools.ensemble_distributed import main_ensemble_distributed

        main_ensemble_distributed(cfg)


def ensemble_train(config: dict, cfg: DictConfig, base_seed: int, ensemble_dir: Path) -> None:
    """Function trainable for one ensemble member -- the ensemble analogue of ``ray_train``.

    Where ``ray_train`` applies Tune-sampled hyperparameters onto ``cfg.model``, this applies
    the member seed onto ``cfg``: it enables the per-member bootstrap of the training split,
    routes checkpoints/logs to ``member_{i}``, and adds a ``TuneReportCallback`` so the member's
    monitored metric surfaces in the Tune status table (the same table the distributed path
    shows). It then calls ``main_function``, exactly like the HPO trainable -- plain
    single-device Lightning, no Ray Train.

    Tune grids over the per-member seed (mirroring the distributed path's ``member_seed`` grid),
    so ``config`` carries ``member_seed`` and the member index is recovered as
    ``member_seed - base_seed``.

    Each trial receives its own deserialised copy of ``cfg`` (via ``tune.with_parameters``), so
    mutating it here is trial-local and safe -- same contract as ``ray_train``.

    Args:
        config (dict): The per-member values Tune "sampled"; carries ``member_seed``.
        cfg (DictConfig): The full config, bound via ``tune.with_parameters``.
        base_seed (int): Seed of member 0; member ``i`` uses ``base_seed + i``.
        ensemble_dir (Path): ``<ensemble.storage_path>/ensemble``; member ``i`` writes to
            ``member_{i}`` under it.
    """
    member_seed = config["member_seed"]
    member_idx = member_seed - base_seed

    cfg.stage = "fit"
    cfg.do_ensemble_train = False  # each member is a single run
    cfg.seed = member_seed

    # Each member bootstrap-resamples the shared training split with its own seed; val/wi_fit/
    # test stay fixed and shared.
    cfg.datamodule.bootstrap = True
    cfg.datamodule.random_state = member_seed

    # Route each member's checkpoints and logs to its own directory.
    member_dir = ensemble_dir / f"member_{member_idx}"
    cfg.trainer.default_root_dir = str(member_dir)
    logger_cfg = cfg.get("logger")
    if logger_cfg is not None and logger_cfg.get("csv") is not None:
        cfg.logger.csv.save_dir = str(member_dir)
    for cb_name in ("model_checkpoint", "model_checkpoint2"):
        if cfg.get("callbacks", {}).get(cb_name) is not None:
            cfg.callbacks[cb_name].dirpath = str(member_dir / "checkpoints")

    # Report the monitored metric(s) to Tune so the status table shows a metric column (matching
    # the distributed path). ``main_function`` only auto-adds this callback under ``do_hpo_tune``,
    # so we add it explicitly here; default to the checkpoint monitor, widened by
    # ``ensemble.monitor``. TuneReportCallback reports only these keys, keeping the table clean
    # even when the closure-metrics callback logs many per-feature metrics.
    monitor = list(cfg.get("ensemble", {}).get("monitor", []) or [])
    ckpt_cfg = cfg.get("callbacks", {}).get("model_checkpoint")
    if ckpt_cfg is not None and ckpt_cfg.get("monitor") and ckpt_cfg.monitor not in monitor:
        monitor.append(ckpt_cfg.monitor)
    if not monitor:
        monitor = ["val_loss"]
    if "callbacks" not in cfg:
        cfg.callbacks = {}
    cfg.callbacks["ray_tune_report_callback"] = {
        "_target_": "ray.tune.integration.pytorch_lightning.TuneReportCallback",
        "metrics": {name: name for name in monitor},
        "on": "validation_end",
    }

    main_function(cfg)


def _main_ensemble_single_device(cfg: DictConfig) -> None:
    """Single-device ensemble training with Ray Tune (one GPU, or a fraction, per member).

    Mirrors ``_main_tune_single_device``: each member is a Ray Tune function trainable
    (``ensemble_train``) running plain single-device Lightning -- no Ray Train, so no Ray
    strategy and no ``RAY_TRAIN_V2_ENABLED`` flag. Members are enumerated as a Tune grid over the
    per-member seed (``num_samples=1``, so the grid expands to exactly ``ensemble.size`` trials, one
    member each), matching the distributed path, so they show the same status table. Data is
    split/scaled once on the driver; each member bootstrap-resamples the *training* split with its
    own seed and writes to ``<ensemble.storage_path>/ensemble/member_{i}`` (default
    ``~/ray_results/ensemble/member_{i}``) via the Lightning ``ModelCheckpoint`` -- the same
    ``storage_path`` knob the distributed path honors. The held-out ``wi_fit`` and ``test`` splits
    stay fixed and shared for the later weight-fitting and evaluation phases.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    import ray
    from ray import tune

    ensemble_cfg = cfg.get("ensemble", {})
    size = ensemble_cfg.get("size", 16)
    base_seed = ensemble_cfg.get("seed", cfg.get("seed", 0))
    # A member trains on `devices: 1`, so it uses at most one GPU; a fraction < 1 lets Ray pack
    # multiple members onto one physical GPU via the driver's time-slicing (Ray does not enforce
    # VRAM, so co-located members must fit in memory). Mirrors the single-device HPO path.
    gpus_per_member = ensemble_cfg.get("single_device_gpu_fraction_per_member", 1)
    cpus_per_member = ensemble_cfg.get("cpus_per_worker", 2)
    if not 0 < gpus_per_member <= 1:
        raise ValueError(
            "ensemble.single_device_gpu_fraction_per_member must be in (0, 1] "
            f"(got {gpus_per_member}); use a distributed strategy for >1 GPU per member."
        )

    # Prefer connecting to a cluster started externally (e.g. `ray start --head` in the Polaris
    # exec script). If none is running, fall through and let Tune's .fit() auto-start a local one.
    if not ray.is_initialized():
        try:
            ray.init(address="auto", log_to_driver=True)
            log.info("Connected to existing Ray cluster.")
        except ConnectionError:
            log.info("No running Ray cluster found; Tune will auto-start a local one.")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    # Members run headless in worker processes; a progress bar per member would garble the logs.
    cfg.trainer.enable_progress_bar = False

    # Prepare the split/scaled data once on the driver so all members share one partition and
    # don't race to write the same pickles in datamodule.setup("fit").
    log.info("Preparing shared ensemble data (split + scale) once on the driver...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    # Root the members under ``ensemble.storage_path`` (the same knob the distributed path
    # honors), defaulting to Ray's ``~/ray_results`` when unset. Members land in
    # ``<storage_path>/ensemble/member_{i}`` via the Lightning ``ModelCheckpoint`` dirpath.
    storage_path = ensemble_cfg.get("storage_path", None)
    root = Path(storage_path) if storage_path is not None else Path.home() / "ray_results"
    ensemble_dir = root / "ensemble"

    ckpt_callback = cfg.callbacks.model_checkpoint  # provides the trial-ranking metric + mode
    trainable = tune.with_parameters(
        ensemble_train, cfg=cfg, base_seed=base_seed, ensemble_dir=ensemble_dir
    )

    # One Tune trial per member: grid over the per-member seed (num_samples=1 -> exactly `size`
    # trials), mirroring the distributed path's `member_seed` grid. Each trial claims
    # `gpus_per_member` GPU(s), so Tune gates concurrency on real GPU availability and packs
    # fractional members onto one GPU, like the single-device HPO path.
    seeds = [base_seed + i for i in range(size)]

    def _member_name(trial) -> str:
        # Name trials/dirs member_<i> (i = seed - base_seed), mirroring the
        # <storage_path>/ensemble/member_{i} convention and the distributed path's trial naming.
        seed = trial.config["member_seed"]
        return f"member_{seed - base_seed}"

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": cpus_per_member, "gpu": gpus_per_member}),
        param_space={"member_seed": tune.grid_search(seeds)},
        tune_config=tune.TuneConfig(
            num_samples=1,
            metric=ckpt_callback.monitor,
            mode=ckpt_callback.mode,
            trial_name_creator=_member_name,
            trial_dirname_creator=_member_name,
            # Give each member a fresh actor that Ray kills on completion
            reuse_actors=False,
        ),
    )

    log.info(
        "Training {} ensemble members as Ray Tune trials ({} GPU(s) each) under {}...",
        size,
        gpus_per_member,
        ensemble_dir,
    )
    results = tuner.fit()
    # tuner.fit() returns normally even when member trials fail (failures land in the ResultGrid),
    # so check explicitly and raise -- this is what lets a chained do_ensemble_fit run only when
    # the whole ensemble trained, rather than fitting weights on a partial ensemble.
    if results.num_errors:
        raise RuntimeError(
            f"{results.num_errors} of {len(results)} ensemble members errored during training; "
            f"inspect the failed trials under {ensemble_dir} before fitting weights."
        )

    # Record each member's best checkpoint in a manifest the fit reads, so it needn't know this
    # path's on-disk layout. Members here write via the Lightning ModelCheckpoint dirpath to
    # <ensemble_dir>/member_{i}/checkpoints; find_latest_checkpoint picks the current best.
    from nsbi.tools.ensemble_fit import write_member_manifest

    member_ckpts: dict[int, str] = {}
    for i in range(size):
        ckpt = find_latest_checkpoint(ensemble_dir / f"member_{i}" / "checkpoints")
        if ckpt is None:
            raise RuntimeError(
                f"Member {i} reported no error but no checkpoint was found under "
                f"{ensemble_dir / f'member_{i}' / 'checkpoints'}."
            )
        member_ckpts[i] = str(ckpt)
    write_member_manifest(ensemble_dir, member_ckpts)
    log.info("Ensemble training complete: {} members under {}", len(results), ensemble_dir)


def ray_train(config: dict, cfg: DictConfig) -> None:
    """Function to be used by Ray Trainer to launch training.

    Args:
        config (dict): Configuration dictionary for HPO.
        cfg (DictConfig): Original configuration composed by Hydra.
    """
    # update cfg with config assuming all config keys are in cfg.model.
    for k, v in config.items():
        if k in cfg.model:
            cfg.model[k] = v
        else:
            log.warning("Key {} not found in cfg.model. Skipping update.", k)
    # Call the main function
    main_function(cfg)


def main_tune_function(cfg: DictConfig) -> None:
    """Hyperparameter tuning with Ray Tune.

    Dispatches to the single-device path (one GPU per trial) or the distributed path
    (multiple GPUs per trial via Ray TorchTrainer) based on ``hpo_tune.scaling.strategy``.
    The single-device path is the original implementation; all distributed-specific
    machinery lives in ``nsbi.tools.hpo_distributed``.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    scaling_cfg = cfg.hpo_tune.get("scaling", {})
    strategy_name = scaling_cfg.get("strategy", None)

    if strategy_name == "single_device" or not strategy_name:
        _main_tune_single_device(cfg)
    else:
        from nsbi.tools.hpo_distributed import main_tune_distributed

        main_tune_distributed(cfg)


def _main_tune_single_device(cfg: DictConfig) -> None:
    """Single-device hyperparameter tuning with Ray Tune (one GPU per trial).

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    import ray
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator

    # Prefer connecting to a cluster started externally (e.g. `ray start --head` in
    # the exec script). If none is running, fall through and let Tune's .fit()
    # auto-start a local cluster for dev/laptop runs.
    if not ray.is_initialized():
        try:
            ray.init(address="auto", log_to_driver=True)
            log.info("Connected to existing Ray cluster.")
        except ConnectionError:
            log.info("No running Ray cluster found; Tune will auto-start a local one.")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    # disable progress bar for HPO tuning.
    cfg.trainer.enable_progress_bar = False

    # Prepare the split/scaled data once on the driver before trials launch. Trials run
    # as concurrent processes; without this they would all find a cold data_dir and race
    # to write the same pickles in datamodule.setup("fit").
    log.info("Instantiating datamodule <{}>", cfg.datamodule._target_)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    scheduler = hydra.utils.instantiate(cfg.hpo_tune.get("scheduler"))
    ckpt_callback = cfg.callbacks.model_checkpoint
    trainable = tune.with_parameters(ray_train, cfg=cfg)
    search_space = cfg.hpo_tune.get("search_space", {})
    params_space = {}
    for k, v in search_space.items():
        if isinstance(v, str):
            params_space[k] = parse_dist(v)
        else:
            params_space[k] = v

    # printout the param space.
    log.info("Hyperparameter search space: {}", params_space)

    # CPU count is read from hpo_tune.cpus_per_worker (default 2), matching the
    # distributed path. single_device_gpu_fraction_per_trial sets the GPU resource each
    # trial claims (default 1, a whole GPU). A fraction < 1 (e.g. 0.5) lets Ray pack
    # multiple independent single-device trials onto one physical GPU, where they coexist
    # via the GPU driver's default time-slicing. Ray does NOT enforce VRAM, so the
    # co-located trials must fit in memory.
    #
    # TODO: for better isolation/utilization when packing trials, add support for NVIDIA
    # MPS (concurrent kernels on the SMs) or MIG (hardware-partitioned, isolated VRAM)
    # by enabling it in the Polaris launch script. See:
    # https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/?h=mps#running-multiple-processes-per-gpu
    cpus_per_worker = cfg.hpo_tune.get("cpus_per_worker", 2)
    gpu_fraction_per_trial = cfg.hpo_tune.get("single_device_gpu_fraction_per_trial", 1)
    # A single-device trial trains on `devices: 1`, so it can use at most one GPU; values
    # above 1 would over-reserve GPUs it cannot use. Fractions pack trials onto one GPU.
    if not 0 < gpu_fraction_per_trial <= 1:
        raise ValueError(
            "hpo_tune.single_device_gpu_fraction_per_trial must be in (0, 1] "
            f"(got {gpu_fraction_per_trial}); use the distributed strategy for >1 GPU per trial."
        )
    tuner = tune.Tuner(
        tune.with_resources(
            trainable, resources={"cpu": cpus_per_worker, "gpu": gpu_fraction_per_trial}
        ),
        param_space=params_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=cfg.hpo_tune.get("num_samples", 1),
            metric=ckpt_callback.monitor,
            mode=ckpt_callback.mode,
            # Give the search its own seeded RNG so the sampled trials don't depend
            # on whatever else consumed the global NumPy RNG before this point. This
            # makes the trial grid reproducible and identical to the distributed path.
            search_alg=BasicVariantGenerator(random_state=cfg.get("seed")),
            # Name trials/dirs `trial_<id>` so both HPO paths share one convention
            # instead of the default `ray_train_<id>` taken from the trainable name.
            trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            # Fresh actor per trial so DataLoader workers/threads are reclaimed at trial end
            reuse_actors=False,
        ),
    )
    analysis = tuner.fit()
    log.info(f"Best hyperparameters found were: {analysis.get_best_result()}")


def main() -> None:
    """Main function to run the training script."""
    log.remove()
    log.add(sys.stdout, level="INFO")
    log.add(
        "logs/nsbi.log",
        rotation="1 MB",
        retention="10 days",
        level="INFO",
        enqueue=True,
    )

    import argparse

    parser = argparse.ArgumentParser(description="Train a model with OmegaConfig.")
    parser.add_argument(
        "-f", "--config-path", required=True, help="Path to the training configuration file"
    )
    parser.add_argument(
        "-c", "--configs", nargs="*", default=[], help="Additional configurations", action="extend"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    args = parser.parse_args()
    config_path = args.config_path

    if Path(config_path).is_file():
        log.info("Loading configuration from {}", config_path)
        cfg = OmegaConf.load(args.config_path)

        if len(args.configs) > 0:
            log.info("Applying additional configurations: {}", args.configs)
            add_configs = OmegaConf.from_dotlist(args.configs)
            cfg = OmegaConf.merge(cfg, add_configs)

        if not isinstance(cfg, DictConfig):
            raise TypeError("Configuration must be a DictConfig object.")

        if cfg.get("do_hpo_tune", False):
            log.info("Starting hyperparameter tuning with Ray Tune...")
            main_tune_function(cfg)
        elif cfg.get("do_ensemble_train", False):
            log.info("Starting ensemble training...")
            # Raises if any member errored, so the chained fit below is only reached when the
            # whole ensemble trained successfully.
            main_ensemble_function(cfg)
            if cfg.get("do_ensemble_fit", False):
                log.info("Ensemble training succeeded; fitting wifi ensemble weights...")
                from nsbi.tools.ensemble_fit import main_ensemble_fit

                main_ensemble_fit(cfg)
        elif cfg.get("do_ensemble_fit", False):
            log.info("Fitting wifi ensemble weights...")
            from nsbi.tools.ensemble_fit import main_ensemble_fit

            main_ensemble_fit(cfg)
        else:
            log.info("Starting main training/evaluation function...")
            main_function(cfg)
    else:
        log.error("Configuration file {} does not exist.", config_path)
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")


if __name__ == "__main__":
    main()
