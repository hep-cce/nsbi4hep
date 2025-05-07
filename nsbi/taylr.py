import os
import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from physics.simulation import msq
from models import taylr
from datasets import coefficient

torch.set_float32_matmul_precision('medium')

components = {
    'sbi': msq.Component.SBI,
    'sig': msq.Component.SIG,
    'int': msq.Component.INT,
    'bkg': msq.Component.BKG
}

def main(args):
    dm = coefficient.CoefficientDataModule(
                                   file_path = args.events, 
                                   analysis = args.analysis,
                                   features = args.features,
                                   coefficient_index = args.coefficient_index,
                                   scaler_path = 'scaler.pkl',
                                   component = components[args.component],
                                   sample_size = args.sample_size,
                                   batch_size = args.batch_size,
                                   random_state = args.random_state)

    model_taylr = taylr.TAYLR(len(args.features), args.n_layers, args.n_nodes, args.learning_rate)

    # save best-two models based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="checkpoint-taylr-{epoch:02d}-{val_loss:.2f}",
    )

    model_best_train_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='train_loss',
        mode='min',
        dirpath='checkpoints/',
        filename='checkpoint-taylr-train-{epoch:02d}-{train_loss:.2f}'
    )

    trainer = Trainer(accelerator=args.accelerator, max_epochs=300, callbacks=[model_checkpoint_callback, model_best_train_checkpoint_callback], logger=CSVLogger('.'))

    trainer.fit(model_taylr, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TAYLR model")
    parser.add_argument('--events', type=str, required=True, help='Sample filepath')
    parser.add_argument('--analysis', type=str,  default='4l', help='Analysis  to run')
    parser.add_argument('--features', type=str,  nargs='+', default=['l1_pt', 'l1_eta', 'l1_phi', 'l1_energy', 'l2_pt', 'l2_eta', 'l2_phi', 'l2_energy', 'l3_pt', 'l3_eta', 'l3_phi', 'l3_energy', 'l4_pt', 'l4_eta', 'l4_phi', 'l4_energy'], help='Observable system to train on')
    parser.add_argument('--n-layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--max-epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of examples')
    parser.add_argument('--coefficient-index', type=int, default=1, help='Index of polynomial coefficient')
    parser.add_argument('--component', type=str, default='sbi', help='Process component')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Trainer accelerator')
    
    args = parser.parse_args()
    main(args)