"""Distributed (multi-GPU / multi-node per member) ensemble training -- Ray Train **V1**.

This is the distributed counterpart to ``_main_ensemble_single_device`` in ``entry_cli``.
The single-device path trains each ensemble member on one GPU as a Ray task and runs many
members concurrently across the cluster; this path trains each *individual* member across a
group of GPUs (possibly spanning nodes) with a data-parallel (DDP/FSDP/DeepSpeed) Lightning
strategy. Use it only when a single member is too large or slow for one GPU -- for the small
members typical of wifi ensembling, the single-device path already scales across the cluster.

Orchestration mirrors the distributed HPO path (``hpo_distributed.py``): each member is one
Ray Train **V1** ``TorchTrainer`` handed to a ``tune.Tuner`` whose "search space" is a grid
over the per-member seed (``num_samples=1``, so the grid expands to exactly ``ensemble.size``
trials -- one member each). This reuses the V1 Tune+Train integration -- Tune derives each
member's placement group from the trainer's ``ScalingConfig``, so Tune and Train share one
placement group per member -- and thereby avoids the V2 nesting bugs (#53921, #54305)
documented in ``hpo_distributed.py``. As there, ``RAY_TRAIN_V2_ENABLED=0`` must be exported
*before* the process starts (it is read at import time; the launch script sets it before
``ray start``).

Members differ exactly as in the single-device path: each bootstrap-resamples the shared
training split with its own seed and initializes its network from that seed; val/wi_fit/test
stay fixed and shared. Data is split/scaled once on the driver; every worker only reads the
pickles in ``datamodule.setup("fit")``. Unlike the single-device path (whose members land in
``<data_dir>/ensemble/member_{i}`` via the Lightning ``ModelCheckpoint`` dirpath), checkpoints
here are managed by Ray Train under ``RunConfig.storage_path`` in ``member_{i}`` trial dirs --
the same single-device-vs-distributed storage difference the HPO paths have.
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

from nsbi.tools.hpo_distributed import _build_ray_strategy
from nsbi.utils import hydra_utils as utils


def _train_ensemble_member(config: dict, cfg: DictConfig) -> None:
    """Per-worker training loop for one ensemble member, run in every Ray Train worker process.

    Mirrors ``hpo_distributed._train_func`` but, instead of applying sampled hyperparameters,
    applies the member's seed onto the config: it seeds the RNGs and enables the per-member
    bootstrap of the training split. Builds the Lightning ``Trainer`` with a Ray distributed
    strategy and reports metrics/checkpoints to Ray Train.

    Args:
        config (dict): The per-member values Tune "sampled"; carries ``member_seed`` (Ray
            passes the trial's ``train_loop_config`` here).
        cfg (DictConfig): The full Hydra config, bound via ``functools.partial`` when the
            ``TorchTrainer`` is constructed. Each member/worker receives its own deserialised
            copy, so mutating it here is safe.
    """
    from ray.train.lightning import (
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )

    member_seed = config["member_seed"]
    L.seed_everything(member_seed)

    # Give each member its own temp root so RayTrainReportCallback's per-epoch checkpoint
    # staging dir (built under tempfile.gettempdir()) is unique per member. Without this,
    # members co-located on one node share /tmp/lightning_checkpoints-...-name=<TorchTrainer>
    # and race on shutil.rmtree -> FileNotFoundError
    import tempfile

    member_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"nsbi_member_{member_seed}")
    os.makedirs(member_tmp, exist_ok=True)
    os.environ["TMPDIR"] = member_tmp
    tempfile.tempdir = member_tmp

    # Each member bootstrap-resamples the shared training split with its own seed; val/wi_fit/
    # test stay fixed and shared (same member-diversity contract as the single-device path).
    cfg.seed = member_seed
    cfg.datamodule.bootstrap = True
    cfg.datamodule.random_state = member_seed

    strategy_name = cfg.ensemble.get("scaling", {}).get("strategy", "ddp")
    strategy = _build_ray_strategy(strategy_name)

    # Each worker reads the pre-split, pre-scaled data from the shared filesystem (prepared once
    # on the driver in main_ensemble_distributed before the tuner runs).
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    loggers: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))  # type: ignore
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    # Ray Train's report callback saves a checkpoint and reports metrics each epoch; RunConfig's
    # CheckpointConfig ranks/prunes those checkpoints by the checkpoint monitor.
    callbacks.append(RayTrainReportCallback())

    # Ray owns device placement: override trainer.devices and inject the Ray strategy +
    # environment. devices="auto" lets RayLightningEnvironment assign the GPU that Ray reserved
    # for this worker; num_nodes is left to the Ray strategy so a worker group can span nodes.
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

    log.info("Training ensemble member (seed={}, strategy={})", member_seed, strategy_name)
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


def main_ensemble_distributed(cfg: DictConfig) -> None:
    """Distributed (multi-GPU / multi-node per member) ensemble training entrypoint (Train V1).

    Connects to the Ray cluster started by the launch script, prepares the data once on the
    driver, then trains ``ensemble.size`` members, each across
    ``scaling.num_workers * scaling.gpus_per_worker`` GPUs via a Ray Train V1 ``TorchTrainer``.
    Members are enumerated as a ``tune.Tuner`` grid over the per-member seed so Tune and Train
    share one placement group per member (see the module docstring).

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # The V1 fallback is read at import time, so it must already be in the environment (set in
    # the exec script before `ray start`). Fail loudly here rather than letting the V2 Tuner
    # reject the trainer with a cryptic "Improper 'run'" error.
    if os.environ.get("RAY_TRAIN_V2_ENABLED") != "0":
        raise RuntimeError(
            "Distributed ensemble training requires Ray Train V1, but RAY_TRAIN_V2_ENABLED is "
            f"{os.environ.get('RAY_TRAIN_V2_ENABLED')!r}. Export RAY_TRAIN_V2_ENABLED=0 BEFORE "
            "launching (in the launch script, before `ray start`); it is read at import time, "
            "so setting it from Python is too late."
        )

    import ray
    from ray import tune
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    # Connect to the externally-started cluster (ray start --head/--address in the exec script)
    # so workers inherit the full shell environment, incl. CUDA paths. Fall back to a local
    # cluster for single-node dev runs.
    try:
        ray.init(address="auto", log_to_driver=True)
        log.info("Connected to existing Ray cluster.")
    except ConnectionError:
        log.info("No running Ray cluster found; starting a local one.")
        ray.init(log_to_driver=True)
    log.info("Ray cluster resources: {}", ray.available_resources())

    if cfg.get("seed"):
        L.seed_everything(cfg.seed)
    cfg.trainer.enable_progress_bar = False

    # Prepare data once on the driver: write the split/scaled pickles to the shared data_dir.
    # Every worker then just reads them in datamodule.setup("fit").
    log.info("Preparing data once on driver from <{}>", cfg.datamodule._target_)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    ensemble_cfg = cfg.get("ensemble", {})
    size = ensemble_cfg.get("size", 16)
    base_seed = ensemble_cfg.get("seed", cfg.get("seed", 0))
    scaling_cfg = ensemble_cfg.get("scaling", {})
    num_workers = scaling_cfg.get("num_workers", 1)
    gpus_per_worker = scaling_cfg.get("gpus_per_worker", 1)
    cpus_per_worker = ensemble_cfg.get("cpus_per_worker", 2)
    gpus_per_member = num_workers * gpus_per_worker
    log.info(
        "Distributed ensemble (Train V1): {} member(s), {} GPU(s)/member ({} workers x {} GPU)",
        size,
        gpus_per_member,
        num_workers,
        gpus_per_worker,
    )

    ckpt_callback = cfg.callbacks.model_checkpoint  # provides the checkpoint metric + mode

    # V1: build ONE TorchTrainer and hand it to the Tuner. Tune "samples" the member seed per
    # trial (via param_space) and overrides train_loop_config on the trainer; Tune and Train
    # share the single placement group ScalingConfig describes.
    ray_trainer = TorchTrainer(
        train_loop_per_worker=partial(_train_ensemble_member, cfg=cfg),
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": gpus_per_worker},
        ),
        run_config=RunConfig(
            storage_path=ensemble_cfg.get("storage_path", None),
            checkpoint_config=CheckpointConfig(
                num_to_keep=ckpt_callback.get("save_top_k", 1),
                checkpoint_score_attribute=ckpt_callback.monitor,
                checkpoint_score_order=ckpt_callback.mode,
            ),
        ),
    )

    # Enumerate members as a grid over the per-member seed; num_samples=1 means grid_search
    # expands to exactly `size` trials, one member each. Tune derives each member's placement
    # group from the ScalingConfig and gates concurrency on real GPU availability.
    seeds = [base_seed + i for i in range(size)]

    def _member_name(trial) -> str:
        # Name trials/dirs member_<i> (i = seed - base_seed) so members are easy to find,
        # mirroring the single-device path's <data_dir>/ensemble/member_{i} convention.
        seed = trial.config["train_loop_config"]["member_seed"]
        return f"member_{seed - base_seed}"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": {"member_seed": tune.grid_search(seeds)}},
        tune_config=tune.TuneConfig(
            num_samples=1,
            metric=ckpt_callback.monitor,
            mode=ckpt_callback.mode,
            trial_name_creator=_member_name,
            trial_dirname_creator=_member_name,
        ),
    )

    results = tuner.fit()
    # tuner.fit() returns normally even when member trials fail (failures land in the ResultGrid),
    # so check explicitly and raise -- this is what lets a chained do_ensemble_fit run only when
    # the whole ensemble trained, rather than fitting weights on a partial ensemble.
    if results.num_errors:
        raise RuntimeError(
            f"{results.num_errors} of {len(results)} ensemble members errored during training; "
            "inspect the failed trials before fitting weights."
        )

    # Record each member's best checkpoint in the manifest the fit reads. Unlike the single-device
    # path (fixed Lightning ModelCheckpoint dirpath), checkpoints here live in Ray-generated trial
    # dirs, so we resolve them from the ResultGrid rather than by globbing: the trial dir is named
    # member_{i}, and get_best_checkpoint honors the checkpoint monitor/mode. Ray Train's
    # RayTrainReportCallback saves the Lightning file as CHECKPOINT_NAME inside the checkpoint dir.
    from pathlib import Path

    from ray.train.lightning import RayTrainReportCallback

    from nsbi.tools.ensemble_fit import resolve_ensemble_dir, write_member_manifest

    ckpt_name = getattr(RayTrainReportCallback, "CHECKPOINT_NAME", "checkpoint.ckpt")
    member_ckpts: dict[int, str] = {}
    for result in results:
        member_idx = int(Path(result.path).name.split("_")[-1])
        best = result.get_best_checkpoint(ckpt_callback.monitor, ckpt_callback.mode)
        if best is None:
            raise RuntimeError(
                f"Member {member_idx} reported no error but has no checkpoint in its result "
                f"({result.path}); cannot fit weights without it."
            )
        member_ckpts[member_idx] = str(Path(best.path) / ckpt_name)
    write_member_manifest(resolve_ensemble_dir(ensemble_cfg), member_ckpts)
    log.info("Distributed ensemble training complete: {} member(s).", len(results))
