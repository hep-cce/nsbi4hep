import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L

class CARL(L.LightningModule):
    def __init__(self, n_features, n_layers, n_hidden_nodes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_hidden_nodes),
            nn.SiLU(),
            *[nn.Sequential(nn.Linear(n_hidden_nodes, n_hidden_nodes), nn.SiLU()) for _ in range(n_layers)],
            nn.Linear(n_hidden_nodes, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.view(-1, 1)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.view(-1, 1)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters())
        return optimizer