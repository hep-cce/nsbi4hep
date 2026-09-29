"""Distributed (multi-GPU / multi-node per trial) Ray Tune HPO path -- Ray Train **V1**.

This is the distributed counterpart to ``_main_tune_single_device`` in ``entry_cli``.
The single-device path gives every trial one GPU; this path gives every trial a group
of GPUs (possibly spanning nodes) that train one model together with a data-parallel
(DDP/FSDP/DeepSpeed) Lightning strategy.

Why Ray Train V1
----------------
Nesting a Ray Train **V2** ``TorchTrainer`` inside a Ray Tune trial is broken by two
open Ray bugs (#53921, #54305): the Tuner is blind to the trainer's resources, and a
V2 ``TorchTrainer`` always builds its *own* placement group separate from Tune's. A
physical GPU cannot live in two placement groups, so the trainer's worker group starves
(``WorkerGroupStartupTimeoutError``) and trials hang. ``with_resources`` makes it worse
by double-booking the GPUs. The maintainers note the V1 interface does not have this bug.

Ray Train **V1** integrates Tune and Train properly: you pass the ``TorchTrainer`` itself
to ``tune.Tuner``, and Tune derives the trial's placement group from the trainer's
``ScalingConfig`` (one trainer-resources bundle + one bundle per worker). Tune and Train
therefore share **one** placement group, which can span nodes. Result: correct resource
accounting, correct concurrency gating, clean reclaim between trials, and multi-node
per-trial training -- none of which the V2 nesting can deliver today.

    ray_trainer = TorchTrainer(
        train_loop_per_worker=partial(_train_func, cfg=cfg),
        scaling_config=ScalingConfig(num_workers=N, use_gpu=True, ...),  # one model, N GPUs
    )
    tune.Tuner(
        ray_trainer,                                       # the trainer ITSELF (V1)
        param_space={"train_loop_config": search_space},   # what Tune samples
        tune_config=...,                                    # ASHA, num_samples, ...
    )

Metrics flow in one hop: ``RayTrainReportCallback`` reports from the Lightning worker to
the Ray Train session, and under the V1 ``Tuner(TorchTrainer(...))`` integration those
metrics surface to Tune automatically for ASHA pruning.

Requirement: ``RAY_TRAIN_V2_ENABLED=0``
---------------------------------------
Ray Train V2 is the default in Ray >= 2.43. The V1 fallback is selected by the
``RAY_TRAIN_V2_ENABLED=0`` environment variable, which is read **at import time**, so it
must be exported in the launch script *before* the Python process starts (before
``ray start``). Setting it from Python is too late. ``main_tune_distributed`` asserts it
is set so a misconfigured launch fails loudly instead of hitting the cryptic
"Improper 'run'" error from the V2 Tuner.

The cluster itself is started outside Python (``ray start --head`` /
``ray start --address`` in the launch script); here we just connect to it with
``ray.init(address="auto")``.
"""

import os
from functools import partial
from typing import TYPE_CHECKING

import hydra
import lightning as L
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from lightning.pytorch.core import LightningDataModule, LightningModule
    from lightning.pytorch.loggers.logger import Logger

from loguru import logger as log

from nsbi.utils import hydra_utils as utils
from nsbi.utils.ray_utils import parse_dist


def _build_ray_strategy(strategy_name: str):
    """Map ``hpo_tune.scaling.strategy`` to a Ray Train Lightning strategy instance."""
    from ray.train.lightning import (
        RayDDPStrategy,
        RayDeepSpeedStrategy,
        RayFSDPStrategy,
    )

    factories = {
        "ddp": lambda: RayDDPStrategy(),
        "fsdp": lambda: RayFSDPStrategy(),
        "deepspeed_stage_1": lambda: RayDeepSpeedStrategy(stage=1),
        "deepspeed_stage_2": lambda: RayDeepSpeedStrategy(stage=2),
        "deepspeed_stage_3": lambda: RayDeepSpeedStrategy(stage=3),
    }
    if strategy_name not in factories:
        raise ValueError(
            f"Unknown distributed strategy '{strategy_name}'. Choose from: {list(factories)}"
        )
    return factories[strategy_name]()


def _build_filtered_report_callback(monitor: set[str]) -> Callback:
    """Build a ``RayTrainReportCallback`` that reports only ``monitor`` metrics to Tune.

    The single-device HPO path reports just ``cfg.hpo_tune.monitor`` (via
    ``TuneReportCallback``), but the stock ``RayTrainReportCallback`` reports *all* of
    ``trainer.callback_metrics`` -- every per-feature closure metric ``monitored_model``
    logs. Ray's new console output (``RAY_AIR_NEW_OUTPUT``, on by default) auto-infers the
    displayed columns from the reported metrics and ignores any user ``CLIReporter``, so the
    only way to match the single-device table is to trim what this path actually reports.

    ``RayTrainReportCallback`` (Ray 2.54) gathers metrics and calls ``ray.train.report``
    inline in ``on_train_epoch_end`` with no override hook, and re-implementing its
    DDP-aware checkpoint/report handling is exactly the fragility this module avoids. So we
    reuse the parent untouched and only intercept its ``ray.train.report`` call -- patched
    transiently and restored in a ``finally`` -- to drop non-monitored metrics. The
    checkpoint and its score metric (e.g. ``val_loss``) are reported as before.

    Args:
        monitor (set[str]): Metric keys to forward to Ray Train/Tune. Must match keys
            logged into ``trainer.callback_metrics`` (e.g. ``"val_loss"``).

    Returns:
        Callback: The configured report callback instance.
    """
    from ray.train.lightning import RayTrainReportCallback

    class _FilteredReportCallback(RayTrainReportCallback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
            import ray.train

            original_report = ray.train.report

            def _report(metrics=None, checkpoint=None, **kwargs):
                if metrics is not None:
                    metrics = {k: v for k, v in metrics.items() if k in monitor}
                return original_report(metrics=metrics, checkpoint=checkpoint, **kwargs)

            ray.train.report = _report
            try:
                super().on_train_epoch_end(trainer, pl_module)
            finally:
                ray.train.report = original_report

    return _FilteredReportCallback()


def _alias_new_output_param_columns(prefix: str = "train_loop_config/") -> None:
    """Drop the ``train_loop_config/`` prefix from parameter columns in Ray's new output.

    The V1 ``Tuner(TorchTrainer(...))`` integration requires the search space to live under
    the ``train_loop_config`` key (that is how Tune forwards sampled hyperparameters into the
    ``TorchTrainer``). Ray's new console output (``RAY_AIR_NEW_OUTPUT``, on by default) infers
    each parameter column by flattening that config, so every column reads
    ``train_loop_config/<name>`` and -- after truncation to 20 chars -- shows
    ``...p_config/n_layers``. The new engine flattens the keys itself and exposes no aliasing
    hook (it also ignores any user ``CLIReporter``), so the single-device table's bare
    ``n_layers`` cannot be matched through the public API while keeping the new engine.

    ``_get_trial_table_data`` uses each param key as BOTH the header label and the dict path
    passed to ``unflattened_lookup(param, trial.config)``, so the header cannot simply be
    renamed without breaking the value lookup. We therefore wrap that function to feed it the
    short names (which become the headers) and, only for the duration of the call, redirect the
    module's ``unflattened_lookup`` to resolve each short name back to its full nested path.
    The redirect is restored in a ``finally`` -- the same transient-monkeypatch pattern
    ``_FilteredReportCallback`` uses for ``ray.train.report``. Best-effort and idempotent: if
    Ray's internals have moved, we log and leave the (prefixed) default columns untouched.

    Args:
        prefix (str): The flattened-config prefix to strip from parameter column names.
    """
    try:
        import ray.tune.experimental.output as out
    except Exception as e:  # pragma: no cover - depends on Ray internals
        log.warning("Could not alias HPO param columns; Ray new-output unavailable ({}).", e)
        return

    if getattr(out, "_nsbi_param_alias_installed", False):
        return
    if not hasattr(out, "_get_trial_table_data") or not hasattr(out, "unflattened_lookup"):
        log.warning("Ray new-output internals changed; leaving param columns prefixed.")
        return

    orig_table = out._get_trial_table_data
    orig_lookup = out.unflattened_lookup

    def _table(trials, param_keys, metric_keys, *args, **kwargs):
        short_keys = [k[len(prefix) :] if k.startswith(prefix) else k for k in param_keys]
        rev = dict(zip(short_keys, param_keys, strict=True))

        def _lookup(flat_key, config, *a, **k):
            return orig_lookup(rev.get(flat_key, flat_key), config, *a, **k)

        out.unflattened_lookup = _lookup
        try:
            return orig_table(trials, short_keys, metric_keys, *args, **kwargs)
        finally:
            out.unflattened_lookup = orig_lookup

    out._get_trial_table_data = _table
    out._nsbi_param_alias_installed = True
    log.info("Aliased distributed HPO param columns to drop the {!r} prefix.", prefix)


def _train_func(config: dict, cfg: DictConfig) -> None:
    """Per-worker training loop, run once inside every Ray Train worker process.

    Mirrors the single-device ``main_function`` worker loop, but builds the Lightning
    ``Trainer`` with a Ray distributed strategy and reports metrics to Ray Train (which,
    under the V1 ``Tuner(TorchTrainer(...))`` integration, forwards them to Tune for ASHA
    pruning).

    Args:
        config (dict): The hyperparameters Tune sampled for this trial (Ray passes the
            trial's ``train_loop_config`` here). Keys are applied onto ``cfg.model``.
        cfg (DictConfig): The full Hydra config, bound via ``functools.partial`` when the
            ``TorchTrainer`` is constructed. Each trial/worker receives its own
            deserialised copy, so mutating it here is safe.
    """
    import tempfile

    from ray.train import get_context
    from ray.train.lightning import (
        RayLightningEnvironment,
        prepare_trainer,
    )

    # Give each trial its own temp root so RayTrainReportCallback's per-epoch checkpoint
    # staging dir (built under tempfile.gettempdir()) is unique per trial. Without this, two
    # trials co-located on one node (num_workers*gpus_per_worker < node GPUs) share
    # /tmp/lightning_checkpoints-...-name=<TorchTrainer> and race on shutil.rmtree ->
    # FileNotFoundError.
    trial_id = get_context().get_trial_id()
    trial_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"nsbi_trial_{trial_id}")
    os.makedirs(trial_tmp, exist_ok=True)
    os.environ["TMPDIR"] = trial_tmp
    tempfile.tempdir = trial_tmp

    # Apply the sampled hyperparameters onto the model config (same contract as the
    # single-device ``ray_train`` in entry_cli).
    for k, v in config.items():
        if k in cfg.model:
            cfg.model[k] = v
        else:
            log.warning("Key {} not found in cfg.model. Skipping update.", k)

    strategy_name = cfg.hpo_tune.get("scaling", {}).get("strategy", "ddp")
    strategy = _build_ray_strategy(strategy_name)

    # Each worker reads the pre-split, pre-scaled data from the shared filesystem
    # (prepared once on the driver in main_tune_distributed before the tuner runs).
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    loggers: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))  # type: ignore
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    # Report only the monitored metrics so the distributed status table matches the
    # single-device path (whose TuneReportCallback reports cfg.hpo_tune.monitor). Always
    # keep the checkpoint score metric so RunConfig's CheckpointConfig can still rank.
    monitor = set(cfg.hpo_tune.get("monitor", ["val_loss"]))
    ckpt_cfg = cfg.callbacks.get("model_checkpoint")
    if ckpt_cfg is not None and ckpt_cfg.get("monitor"):
        monitor.add(ckpt_cfg.monitor)
    callbacks.append(_build_filtered_report_callback(monitor))

    # Ray owns device placement: override trainer.devices and inject the Ray strategy +
    # environment. devices="auto" lets RayLightningEnvironment assign the GPU that Ray
    # reserved for this worker; num_nodes is left to the Ray strategy so a worker group
    # can span nodes.
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        callbacks=callbacks,
        logger=loggers,
        devices="auto",
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)

    log.info("Starting distributed training (strategy={})", strategy_name)
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


def main_tune_distributed(cfg: DictConfig) -> None:
    """Distributed (multi-GPU / multi-node per trial) Ray Tune HPO entrypoint (Train V1).

    Connects to the Ray cluster started by the launch script, prepares the data once on
    the driver, then runs ASHA over a search space where each trial trains a single model
    across ``num_workers * gpus_per_worker`` GPUs via a Ray Train V1 ``TorchTrainer``
    passed directly to ``tune.Tuner`` (so Tune and Train share one placement group).

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # The V1 fallback is read at import time, so it must already be in the environment
    # (set in the exec script before `ray start`). Fail loudly here rather than letting
    # the V2 Tuner reject the trainer with a cryptic "Improper 'run'" error.
    if os.environ.get("RAY_TRAIN_V2_ENABLED") != "0":
        raise RuntimeError(
            "Distributed HPO requires Ray Train V1, but RAY_TRAIN_V2_ENABLED is "
            f"{os.environ.get('RAY_TRAIN_V2_ENABLED')!r}. Export RAY_TRAIN_V2_ENABLED=0 "
            "BEFORE launching (in the launch script, before `ray start`); it "
            "is read at import time, so setting it from Python is too late."
        )

    import ray
    from ray import tune
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.tune.search.basic_variant import BasicVariantGenerator

    # Connect to the externally-started cluster (ray start --head/--address in the exec
    # script) so workers inherit the full shell environment, incl. CUDA paths. Fall back
    # to a local cluster for single-node dev runs.
    try:
        ray.init(address="auto", log_to_driver=True)
        log.info("Connected to existing Ray cluster.")
    except ConnectionError:
        log.info("No running Ray cluster found; starting a local one.")
        # Omit num_cpus/num_gpus so Ray auto-detects resources: CPUs from
        # os.cpu_count(), GPUs from CUDA_VISIBLE_DEVICES / detected devices.
        ray.init(log_to_driver=True)
    log.info("Ray cluster resources: {}", ray.available_resources())

    if cfg.get("seed"):
        L.seed_everything(cfg.seed)
    cfg.trainer.enable_progress_bar = False

    # Prepare data once on the driver: write the split/scaled pickles to the shared
    # data_dir. Every worker then just reads them in datamodule.setup("fit").
    log.info("Preparing data once on driver from <{}>", cfg.datamodule._target_)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    # Parse the search space (strings like "loguniform:1e-5,3e-2" -> Ray distributions).
    search_space = cfg.hpo_tune.get("search_space", {})
    params_space = {
        k: (parse_dist(v) if isinstance(v, str) else v) for k, v in search_space.items()
    }
    log.info("Hyperparameter search space: {}", params_space)

    scheduler = hydra.utils.instantiate(cfg.hpo_tune.get("scheduler"))
    ckpt_callback = cfg.callbacks.model_checkpoint  # provides metric + mode

    scaling_cfg = cfg.hpo_tune.get("scaling", {})
    num_workers = scaling_cfg.get("num_workers", 1)
    gpus_per_worker = scaling_cfg.get("gpus_per_worker", 1)
    cpus_per_worker = cfg.hpo_tune.get("cpus_per_worker", 2)
    gpus_per_trial = num_workers * gpus_per_worker
    # Tune derives the trial's placement group from this ScalingConfig and gates
    # concurrency on real GPU availability, so it won't oversubscribe.
    log.info(
        "Distributed HPO (Train V1): {} GPU(s)/trial ({} workers x {} GPU)",
        gpus_per_trial,
        num_workers,
        gpus_per_worker,
    )

    # V1: build ONE TorchTrainer and hand it to the Tuner. Tune samples
    # `train_loop_config` per trial (via param_space) and overrides it on the trainer;
    # Tune and Train share the single placement group ScalingConfig describes.
    ray_trainer = TorchTrainer(
        train_loop_per_worker=partial(_train_func, cfg=cfg),
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": gpus_per_worker},
        ),
        run_config=RunConfig(
            storage_path=cfg.hpo_tune.get("storage_path", None),
            checkpoint_config=CheckpointConfig(
                num_to_keep=ckpt_callback.get("save_top_k", 1),
                checkpoint_score_attribute=ckpt_callback.monitor,
                checkpoint_score_order=ckpt_callback.mode,
            ),
        ),
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": params_space},
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=cfg.hpo_tune.get("num_samples", 1),
            metric=ckpt_callback.monitor,
            mode=ckpt_callback.mode,
            # Give the search its own seeded RNG so the sampled trials don't depend on
            # the driver-side prepare_data() (or anything else) consuming the global
            # NumPy RNG first. This makes the grid reproducible and identical to the
            # single-device path for the same seed.
            search_alg=BasicVariantGenerator(random_state=cfg.get("seed")),
            # Name trials/dirs `trial_<id>` so both HPO paths share one convention. The
            # V1 Tuner(TorchTrainer(...)) integration would otherwise derive the name
            # from the trainer class and prefix every trial with `TorchTrainer_<id>`.
            trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        ),
    )
    # Strip the required `train_loop_config/` prefix from the new-output table's parameter
    # columns so the status table matches the single-device path (bare `n_layers`, ...).
    _alias_new_output_param_columns()

    analysis = tuner.fit()
    log.info("Best hyperparameters found were: {}", analysis.get_best_result())
