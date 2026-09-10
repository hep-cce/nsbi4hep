from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger as log

from nsbi.tools.metrics import ReweightingClosureMetric, plot_closure_grid


class ClosureMetricsCallback(Callback):
    def __init__(self, feature_names, plot=False, plot_dir="closure_plots"):
        super().__init__()
        self.feature_names = feature_names
        self.plot = plot
        self.plot_dir = plot_dir

        n_features = len(feature_names)
        self.chi2_metrics = ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="chi2"
        )
        self.ws_metrics = ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="wasserstein"
        )

        self._validation_outputs = []
        self._test_outputs = []
        self._last_val_arrays = None
        self._warned_sharded_plot = False

    def _collect_outputs(self, storage: list, outputs: Any) -> None:
        if outputs is not None:
            storage.append(outputs)

    def _resolve_plot_dir(self, trainer) -> str:
        # anchor a relative plot_dir to the run's root dir so concurrent runs
        # (e.g. ensemble members) don't clobber each other's plots via a shared cwd
        plot_dir = Path(self.plot_dir)
        if not plot_dir.is_absolute() and trainer.default_root_dir:
            plot_dir = Path(trainer.default_root_dir) / plot_dir
        return str(plot_dir)

    def _plot_closure(self, stage, kin, w_pred, w_truth, w_base, output_dir):
        kin = kin.numpy()
        # same auto-binning as ReweightingClosureMetric with binning=None
        binning = []
        for i in range(kin.shape[1]):
            vmin, vmax = np.percentile(kin[:, i], [0.1, 99.9])
            margin = 0.05 * (vmax - vmin)
            binning.append((50, vmin - margin, vmax + margin))

        plot_closure_grid(
            observables=kin,
            observable_names=self.feature_names,
            true_weights=w_truth.numpy(),
            predicted_weights=w_pred.numpy(),
            base_weights=w_base.numpy(),
            binning=binning,
            output_dir=output_dir,
            file_prefix=f"{stage}_closure",
        )

    def _compute_and_log(self, stage, storage, trainer, pl_module):
        if not storage or not isinstance(storage[0], dict) or "kin" not in storage[0]:
            return

        kin = torch.cat([out["kin"] for out in storage]).cpu().detach()
        if kin.shape[1] != len(self.feature_names):
            raise ValueError(
                f"closure_metrics.feature_names has {len(self.feature_names)} entries but the "
                f"model inputs have {kin.shape[1]} columns -- feature_names must list one name "
                "per input feature, in input order."
            )
        w = torch.cat([out["w"] for out in storage]).cpu().detach()
        y = torch.cat([out["y"] for out in storage]).cpu().detach()
        y_hat = torch.cat([out["y_hat"] for out in storage]).cpu().detach()

        w_base = w * (1.0 - y)
        w_truth = w * y

        r_hat = y_hat / (1.0 - y_hat + 1e-8)
        w_pred = w_base * r_hat

        # compute and log closure metrics
        closure_chi2 = self.chi2_metrics(kin, w_pred, w_truth, w_base)
        for idx, (_, v) in enumerate(closure_chi2.items()):
            name = self.feature_names[idx]
            pl_module.log(f"{stage}_{name}_chi2", v, prog_bar=False, sync_dist=True)

        closure_ws = self.ws_metrics(kin, w_pred, w_truth, w_base)

        for idx, (_, v) in enumerate(closure_ws.items()):
            name = self.feature_names[idx]
            pl_module.log(f"{stage}_{name}_ws", v, prog_bar=False, sync_dist=True)

        if self.plot and not trainer.sanity_checking:
            if stage == "test" and trainer.is_global_zero:
                try:
                    self._plot_closure(
                        stage, kin, w_pred, w_truth, w_base, self._resolve_plot_dir(trainer)
                    )
                except Exception:
                    log.exception("Closure plotting failed for stage {}; continuing", stage)
            elif stage == "val":
                self._last_val_arrays = (kin, w_pred, w_truth, w_base)

        # Reset outputs for next epoch
        storage.clear()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_outputs(self._validation_outputs, outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._compute_and_log("val", self._validation_outputs, trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Plot the stashed validation arrays here so the plots on disk always correspond to the
        # newest checkpoint. Clearing the stash means a save only plots arrays a validation loop
        # has not plotted yet, which dedups multiple saves per validation.
        if self._last_val_arrays is None:
            return
        arrays, self._last_val_arrays = self._last_val_arrays, None
        if trainer.is_global_zero:
            if trainer.world_size > 1 and not self._warned_sharded_plot:
                self._warned_sharded_plot = True
                log.warning(
                    "Closure plots are built from rank 0's validation shard only "
                    "(~1/{} of the validation events).",
                    trainer.world_size,
                )
            try:
                self._plot_closure("val", *arrays, self._resolve_plot_dir(trainer))
            except Exception:
                log.exception("Closure plotting failed on checkpoint save; continuing")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect_outputs(self._test_outputs, outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        self._compute_and_log("test", self._test_outputs, trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        self._last_val_arrays = None
