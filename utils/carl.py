import os
import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from physics.simulation import msq
from models import carl
from datasets import balanced

torch.set_float32_matmul_precision('medium')

def main(args):
    dm = balanced.BalancedDataModule(
                                   numerator_file = args.numerator_events[0],
                                   denominator_file = args.denominator_events[0], 
                                   features = args.features,
                                   sample_size = args.sample_size,
                                   batch_size = args.batch_size,
                                   random_state = args.random_state)

    model_carl = carl.CARL(len(args.features), args.n_layers, args.n_nodes, args.learning_rate)

    # save best-two models based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="checkpoint-carl-{epoch:02d}-{val_loss:.2f}",
    )

    model_best_train_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='train_loss',
        mode='min',
        dirpath='checkpoints/',
        filename='checkpoint-carl-train-{epoch:02d}-{train_loss:.2f}'
    )

    trainer = Trainer(accelerator=args.accelerator, max_epochs=200, callbacks=[model_checkpoint_callback, model_best_train_checkpoint_callback], logger=CSVLogger('.'))

    trainer.fit(model_carl, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CARL model")
    parser.add_argument('--numerator-events', type=str, nargs='+', required=True, help='Numerator sample filepath')
    parser.add_argument('--denominator-events', type=str, nargs='+', required=True, help='Denominator sample filepath')
    parser.add_argument('--features', type=str, nargs='+', default= ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], help='Features to train on')
    parser.add_argument('--n-layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of hidden nodes')
    parser.add_argument('--batch-size', type=int, default=1024, help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Trainer accelerator')
    
    args = parser.parse_args()
    main(args)