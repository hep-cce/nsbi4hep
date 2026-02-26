import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CARL(L.LightningModule):
    def __init__(self, n_features, n_layers, n_nodes, learning_rate):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate

        # MLP with sigmoid output
        layers = []
        layers.append(nn.Sequential(nn.Linear(n_features, n_nodes), nn.SiLU()))
        for _ in range(n_layers):
            layers.append(nn.Sequential(nn.Linear(n_nodes, n_nodes), nn.SiLU()))
        layers.append(nn.Sequential(nn.Linear(n_nodes, 1), nn.Sigmoid()))
        self.model = nn.Sequential(*layers)

        # binary cross-entropy loss
        self.loss_fn = nn.BCELoss(reduction="none")

        # weights initialization
        def xavier_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.model.apply(xavier_init)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()
        loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()
        loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return {
            "val_loss": loss,
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "w": w.detach().cpu(),
            "kin": x.detach().cpu(),
        }

    def predict_step(self, batch, batch_idx):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return self.model(x).flatten()

    def configure_optimizers(self):
        # NAdam optimizer
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
