import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


class TAYLR(L.LightningModule):
    def __init__(self, n_features, n_layers, n_nodes, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate

        layers = []
        layers.append(nn.Sequential(nn.Linear(n_features, n_nodes), nn.SiLU()))
        for _ in range(n_layers):
            layers.append(nn.Sequential(nn.Linear(n_nodes, n_nodes), nn.SiLU()))
        layers.append(nn.Linear(n_nodes, 1))
        self.model = nn.Sequential(*layers)

        def init_weights(node):
            if isinstance(node, nn.Linear):
                torch.nn.init.xavier_uniform_(node.weight)
                node.bias.data.fill_(0.0)

        self.model.apply(init_weights)

        self.loss_fn = nn.MSELoss(reduction="none")

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()

        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=5, filename="{epoch:02d}-{val_loss:.2f}"
            )
        )

        callbacks.append(
            ModelCheckpoint(
                monitor="train_loss",
                mode="min",
                save_top_k=1,
                filename="{epoch:02d}-{train_loss:.2f}",
            )
        )

        callbacks.append(EarlyStopping(monitor="val_loss", patience=20, mode="min"))

        return callbacks

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        w = w.flatten()
        y = y.flatten()
        yhat = self.model(x).flatten()
        loss = (self.loss_fn(yhat, y) * w).sum() / w.sum()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.flatten()
        w = w.flatten()
        yhat = self.model(x).flatten()
        loss = (self.loss_fn(yhat, y) * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return self.model(x).flatten()

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
