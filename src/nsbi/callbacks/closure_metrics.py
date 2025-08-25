from typing import Any
import torch
import pytorch_lightning as pl

from nsbi.tools.metrics import ReweightingClosureMetric


class ClosureMetricsCallback(pl.Callback):
    def __init__(self, feature_names, chi2_metrics_fn, ws_metrics_fn):
        super().__init__()
        self.feature_names = feature_names

        n_features = len(feature_names)
        self.chi2_metrics = ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="chi2"
        )
        self.ws_metrics = ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="wasserstein"
        )

        self._validation_outputs = []
        self._test_outputs = []

    def _collect_outputs(self, storage: list, outputs: Any) -> None:
        if outputs is not None:
            storage.append(outputs)

    def _compute_and_log(self, stage, storage, pl_module):
        if not storage:
            return

        kin = torch.cat([out["kin"] for out in storage])
        w = torch.cat([out["w"] for out in storage])
        y = torch.cat([out["y"] for out in storage])
        y_hat = torch.cat([out["y_hat"] for out in storage])

        w_base = w * (1.0 - y)
        w_truth = w * y

        r_hat = y_hat / (1.0 - y_hat + 1e-8)
        w_pred = w_base * r_hat

        # compute and log closure metrics
        closure_chi2 = self.chi2_metrics(kin, w_pred, w_truth, w_base)
        for idx, (k, v) in enumerate(closure_chi2.items()):
            name = self.feature_names[idx]
            pl_module.log(f"{stage}_{name}_chi2", v, prog_bar=False, sync_dist=True)

        closure_ws = self.ws_metrics(kin, w_pred, w_truth, w_base)

        for idx, (k, v) in enumerate(closure_ws.items()):
            name = self.feature_names[idx]
            pl_module.log(f"{stage}_{name}_ws", v, prog_bar=False, sync_dist=True)

        # Reset outputs for next epoch
        storage.clear()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._collect_outputs(self._validation_outputs, outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._compute_and_log("val", self._validation_outputs, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._collect_outputs(self._test_outputs, outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        self._compute_and_log("test", self._test_outputs, pl_module)
