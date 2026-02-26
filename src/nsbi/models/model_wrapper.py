import lightning as L
import tools.metrics as tools
import torch


class monitored_model(L.LightningModule):
    def __init__(self, base_module, arg_order=None, feature_names=None, **kwargs):
        super().__init__()
        n_features = kwargs.get("n_features", None)
        if n_features is None:
            raise ValueError("n_features must be specified in kwargs.")

        self.feature_names = feature_names or [f"obs_{i}" for i in range(n_features)]

        if arg_order:
            args = [kwargs[k] for k in arg_order]
            self.base_module = base_module(*args)
        else:
            self.base_module = base_module(**kwargs)

        self.chi2_metrics = tools.ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="chi2"
        )

        self.ws_metrics = tools.ReweightingClosureMetric(
            observables=list(range(n_features)), binning=None, metric="wasserstein"
        )

    def configure_callbacks(self):
        return self.base_module.configure_callbacks()

    def forward(self, x):
        return self.base_module(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.base_module(x).flatten()
        y = y.flatten()
        w = w.flatten()

        loss = (self.base_module.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        if not hasattr(self, "_validation_outputs"):
            return

        kin = torch.cat([out["kin"] for out in self._validation_outputs])
        w = torch.cat([out["w"] for out in self._validation_outputs])
        y = torch.cat([out["y"] for out in self._validation_outputs])
        y_hat = torch.cat([out["y_hat"] for out in self._validation_outputs])

        w_base = w * (1.0 - y)
        w_truth = w * y

        r_hat = y_hat / (1.0 - y_hat + 1e-8)
        w_pred = w_base * r_hat

        # compute and log closure metrics
        closure_chi2 = self.chi2_metrics(kin, w_pred, w_truth, w_base)
        for idx, (_k, v) in enumerate(closure_chi2.items()):
            name = self.feature_names[idx]
            self.log(f"{name}_chi2", v, prog_bar=False, sync_dist=True)

        closure_ws = self.ws_metrics(kin, w_pred, w_truth, w_base)

        for idx, (_k, v) in enumerate(closure_ws.items()):
            name = self.feature_names[idx]
            self.log(f"{name}_ws", v, prog_bar=False, sync_dist=True)

    def on_validation_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        x, y, w, kin = batch
        y_hat = self.base_module(x).flatten()
        y = y.flatten()
        w = w.flatten()
        loss = (self.base_module.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self._validation_outputs.append(
            {
                "val_loss": loss,
                "kin": kin.detach().cpu(),
                "y_hat": y_hat.detach().cpu(),
                "y": y.detach().cpu(),
                "w": w.detach().cpu(),
            }
        )

        return loss

    def predict_step(self, batch, batch_idx):
        return self.base_module.predict_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.base_module.configure_optimizers()
