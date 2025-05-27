import torch
from torch import nn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

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
        self.loss_fn = nn.BCELoss(reduction='none')

        # weights initialization
        def xavier_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        self.model.apply(xavier_init)

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()

        callbacks.append(ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="{epoch:02d}-{val_loss:.2f}"
        ))

        callbacks.append(ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            filename="{epoch:02d}-{train_loss:.2f}"
        ))

        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        ))

        return callbacks

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()
        loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()
        loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, = batch
        return self.model(x).flatten()

    def configure_optimizers(self):
        # NAdam optimizer
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer