import torch
from torch import nn
import lightning as L
import numpy as np

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

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.view(-1)
        w = w.view(-1)
        y_hat = self.model(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        loss = (loss * w).sum() / w.sum()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.view(-1)
        w = w.view(-1)
        y_hat = self.model(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        loss = (loss * w).sum() / w.sum()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, w = batch
        return self.model(x).view(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer 