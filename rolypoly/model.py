import torch
from torch import nn
import lightning as L


def weighted_MSELoss(pred, target, weight):
    mse_loss = nn.MSELoss(reduction='none')
    return torch.sum(weight*mse_loss(pred, target))/torch.sum(weight)

class ROLYPOLY(L.LightningModule):

    def __init__(self, n_features, n_layers, n_nodes, learning_rate):
        super().__init__()

        self.lr = learning_rate

        layers = []
        layers.append(nn.Sequential(nn.Linear(n_features, n_nodes), nn.SiLU()))
        for _ in range(n_layers):
            layers.append(nn.Sequential(nn.Linear(n_nodes, n_nodes), nn.SiLU()))
        layers.append(nn.Sequential(nn.Linear(n_nodes, 1), nn.Linear()))
        self.model = nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.model.apply(init_weights)

        self.loss_fn = weighted_MSELoss

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.model(x).view(-1)
        #print("train_s: ", y_hat)
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y, weight=sample_weight)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.model(x).view(-1)
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y, weight=sample_weight)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer