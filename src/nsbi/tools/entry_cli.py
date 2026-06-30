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

    for key, callback_config in cfg.get("callbacks").items():
        if isinstance(callback_config, DictConfig) and "_target_" in callback_config:
            target = callback_config._target_
            if target == "lightning.pytorch.callbacks.ModelCheckpoint":
                ckpt_path = callback_config.get("dirpath", None)
            if (
                "RichProgressBar" in target
                and cfg.get("trainer", {}).get("enable_progress_bar", False) is False
            ):
                # remove RichProgressBar callback if progress bar is disabled
                log.info("Removing <{}> callback as progress bar is disabled.", key)
                cfg.callbacks.pop(key)

    ckpt_file = cfg.get("ckpt_file", None)
    assert not (stage == "finetune" and ckpt_file is None), (
        "In fine-tuning stage, a checkpoint file (ckpt_file) must be provided."
    )
    assert not (stage == "resume" and ckpt_path is None), (
        "In resume stage, a checkpoint path (ckpt_path) must be provided."
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

        for callback_config in cfg.get("callbacks").values():
            if isinstance(callback_config, DictConfig) and "_target_" in callback_config:
                if callback_config._target_ == "lightning.pytorch.callbacks.ModelCheckpoint":
                    # Set the filename suffix for ModelCheckpoint
                    ckpt_filename = f"best-{filename_suffix}-" + "{epoch}-{step}"
                    ckpt_filename = ckpt_filename.replace("/", "-")
                    callback_config.filename = ckpt_filename

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
        if ckpt_file:
            log.info("Resuming training from checkpoint: {}", ckpt_file)
    else:
        pass

    if stage in ["fit", "finetune", "resume"]:
        log.info("Starting training!")
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
            raise ValueError("No checkpoint file provided for testing.")

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
            raise ValueError("No checkpoint file provided for prediction.")

        # model.load_state_dict(torch.load(ckpt_file)["state_dict"])
        trainer.predict(
            model=model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=False,
            ckpt_path="best",
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


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
        ),
    )
    analysis = tuner.fit()
    log.info(f"Best hyperparameters found were: {analysis.get_best_result()}")


def main() -> None:
    """Main function to run the training script."""
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
        else:
            log.info("Starting main training/evaluation function...")
            main_function(cfg)
    else:
        log.error("Configuration file {} does not exist.", config_path)
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")


if __name__ == "__main__":
    main()
