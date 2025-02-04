import torch
from torch import nn
import lightning as L

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
        self.loss_fn = nn.BCELoss()

        # weights initialization
        def xavier_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        self.model.apply(xavier_init)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).view(-1)
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).view(-1)
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # NAdam optimizer
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer