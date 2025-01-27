import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from physics.simulation import msq
from alice import dataset, model

torch.set_float32_matmul_precision('medium')

SAMPLE_DIR = '/raven/u/griesemx/ggZZ_work/'

components = {
    'sbi': msq.Component.SBI,
    'sig': msq.Component.SIG,
    'int': msq.Component.INT,
    'bkg': msq.Component.BKG
}

xs = {
    msq.Component.SBI : 1.5569109,
    msq.Component.SIG : 0.15105108,
    msq.Component.INT : -0.22043824,
    msq.Component.BKG : 1.6270497
}

filenames = {
    msq.Component.SBI : 'ggZZ2e2m_sbi.csv',
    msq.Component.SIG : 'ggZZ2e2m_sig.csv',
    msq.Component.INT : 'ggZZ2e2m_int.csv',
    msq.Component.BKG : 'ggZZ2e2m_bkg.csv'
}

def main(args):
    cmp_sig, cmp_bkg = components[args.signal_process], components[args.background_process]

    data = dataset.AliceDataModule(data_dir = SAMPLE_DIR, 
                                   background_file = filenames[cmp_bkg], 
                                   background_xs = xs[cmp_bkg],
                                   signal_component = cmp_sig,
                                   background_component = cmp_bkg,
                                   sample_size = args.sample_size,
                                   batch_size = args.batch_size,
                                   random_state = args.random_state)

    model_alice = model.ALICE(args.n_features, args.n_layers, args.n_nodes, args.learning_rate)

    # save best-two models based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="checkpoint-alice-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(accelerator=args.accelerator, max_epochs=100, callbacks=[model_checkpoint_callback])

    trainer.fit(model_alice, datamodule=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ALICE model")
    parser.add_argument('--n-features', type=int, default=9, help='Number of features')
    parser.add_argument('--n-layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--signal-process', type=str, default='sig', help='Signal process')
    parser.add_argument('--background-process', type=str, default='bkg', help='Background process')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of hidden nodes')
    parser.add_argument('--batch-size', type=int, default=1024, help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Trainer accelerator')
    
    args = parser.parse_args()
    main(args)