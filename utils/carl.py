import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from datasets import BalancedDataModule
from models import CARL

torch.set_float32_matmul_precision('medium')

SAMPLE_FILEPATH = {
    'sig': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sig/events.csv',
    'bkg': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_bkg/events.csv',
    'int': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_int/events.csv',
    'sbi': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sbi/events.csv'
}

def main(args):
    data = BalancedDataModule(signal_file = SAMPLE_FILEPATH[args.signal_process], background_file=SAMPLE_FILEPATH[args.background_process], sample_size = args.sample_size, batch_size = args.batch_size, random_state = args.seed)

    model = CARL(args.n_features, args.n_layers, args.n_nodes)

    # save best-two models based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(accelerator=args.accelerator, max_epochs=args.epochs, callbacks=[model_checkpoint_callback])

    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CARL model")
    parser.add_argument('--n-features', type=int, default=9, help='Number of features')
    parser.add_argument('--n-layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--signal-process', type=str, default='sig', help='Signal process')
    parser.add_argument('--background-process', type=str, default='bkg', help='Background process')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of hidden nodes')
    parser.add_argument('--batch-size', type=int, default=1024, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random state')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Trainer accelerator')
    
    args = parser.parse_args()
    main(args)