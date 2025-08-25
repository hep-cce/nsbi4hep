import os
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

# local imports
from loguru import logger as log


def is_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0 or int(os.environ.get("RANK", 0)) == 0


@rank_zero_only
def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config."""
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info("Instantiating logger <%s>", lg_conf._target_)
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        """Wrapper function."""
        outdir = Path(cfg.paths.output_dir)
        log.info("Output dir: %s", outdir)
        Path(outdir / "tensorboard").mkdir(parents=True, exist_ok=True)

        # execute the task
        start_time = time.time()
        try:
            task_func(cfg=cfg)
        except Exception:
            log.exception("")  # save exception to `.log` file
            raise
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time:.3f} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info("Output dir: %s", cfg.paths.output_dir)

    return wrap


@rank_zero_only
def save_file(path: Path, content: str):
    Path(path).write_text(content)


@rank_zero_only
def close_loggers():
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""
    log.info("Closing loggers...")
    if find_spec("wandb"):
        import wandb

        wandb.finish()


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            target = cb_conf._target_
            log.info("Instantiating callback <%s>", target)

            # Skip rank-zero-only callbacks on non-zero ranks
            if not is_rank_zero() and any(
                name in target
                for name in [
                    "RichProgressBar",
                    "ModelCheckpoint",
                    "EarlyStopping",
                ]
            ):
                log.info("Skipping callback <%s> on non-zero rank", target)
                continue
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def _safe_get_nested(config, path: str, default=None):
    """Safely get a nested value from config using dot notation."""
    try:
        keys = path.split(".")
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    - Selective hyperparameters for wandb dashboard

    Selective hyperparameters logged:
    - paths.output_dir -> "output_dir"
    - model.model.* -> parameter name (e.g., "num_embeddings", "embedding_dim")
    - model.optimizer.* -> "optimizer.*" (e.g., "optimizer.lr", "optimizer.weight_decay")
    - model.scheduler, model.scheduler_frequency, model.scheduler_monitor
    - datamodule.* -> parameter name (e.g., "box_size", "batch_size")

    Naming convention: Use parameter name if unique, otherwise add prefix to avoid conflicts.

    Args:
        object_dict: Dictionary containing "cfg", "model", and "trainer" keys
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["version_name"] = cfg.get("version_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Add selective hyperparameters

    # Paths
    output_dir = _safe_get_nested(cfg, "paths.output_dir")
    if output_dir is not None:
        hparams["output_dir"] = output_dir

    # Model.model parameters
    model_params = [
        "num_embeddings",
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "decay",
        "commitment_weight",
        "rotation_trick",
        "codebook_di",
        "use_cosine_sim",
        "codebook_diversity_loss_weight",
    ]
    for param in model_params:
        value = _safe_get_nested(cfg, f"model.model.{param}")
        if value is not None:
            hparams[param] = value

    # Model.optimizer parameters
    optimizer_lr = _safe_get_nested(cfg, "model.optimizer.lr")
    if optimizer_lr is not None:
        hparams["optimizer.lr"] = optimizer_lr

    optimizer_weight_decay = _safe_get_nested(cfg, "model.optimizer.weight_decay")
    if optimizer_weight_decay is not None:
        hparams["optimizer.weight_decay"] = optimizer_weight_decay

    # Model scheduler parameters
    scheduler = _safe_get_nested(cfg, "model.scheduler")
    if scheduler is not None:
        hparams["scheduler"] = scheduler

    scheduler_frequency = _safe_get_nested(cfg, "model.scheduler_frequency")
    if scheduler_frequency is not None:
        hparams["scheduler_frequency"] = scheduler_frequency

    scheduler_monitor = _safe_get_nested(cfg, "model.scheduler_monitor")
    if scheduler_monitor is not None:
        hparams["scheduler_monitor"] = scheduler_monitor

    # Datamodule parameters
    datamodule_params = [
        "box_size",
        "patch_size",
        "batch_size",
        "num_workers",
        "cache_dir",
        "train_data",
        "val_data",
    ]
    for param in datamodule_params:
        value = _safe_get_nested(cfg, f"datamodule.{param}")
        if value is not None:
            hparams[param] = value

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
