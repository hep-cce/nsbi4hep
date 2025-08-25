import operator
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

from nsbi.utils import hydra_utils as utils
from nsbi.utils.lightning_utils import find_latest_checkpoint

from loguru import logger as log

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("sum", operator.add)
OmegaConf.register_new_resolver("prod", operator.mul)
OmegaConf.register_new_resolver("gen_list", lambda x, y: [x] * y)


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

    if not (Path(datamodule.data_dir) / "scaler.pkl").exists():
        datamodule.prepare_data()

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

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating trainer <{}>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

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

    if cfg.trainer.accelerator == "gpu":
        trainer.print(torch.cuda.memory_summary())


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

        main_function(cfg)
    else:
        log.error("Configuration file {} does not exist.", config_path)
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")


if __name__ == "__main__":
    main()
