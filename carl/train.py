import argparse

from lightning import Trainer

from dataset import BalancedDataModule
from model import CARL

SAMPLE_FILEPATH = {
    'sig': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sig/events.csv',
    'bkg': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_bkg/events.csv',
    'int': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_int/events.csv',
    'sbi': '/raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sbi/events.csv'
}

def main(args):
    dataset = BalancedDataModule(signal_file = SAMPLE_FILEPATH[args.signal_process], background_file=SAMPLE_FILEPATH[args.background_process], sample_size = args.sample_size, batch_size = args.batch_size, random_state = args.random_state)
    model = CARL(args.n_features, args.n_layers, args.n_nodes)
    trainer = Trainer(accelerator="gpu")
    trainer = Trainer(accelerator="gpu")

    trainer.fit(model, train_dataloader=BalancedDataModule().train_dataloader(), val_dataloader=BalancedDataModule().val_dataloader())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CARL model")
    parser.add_argument('--n-features', type=int, default=9, help='Number of features')
    parser.add_argument('--n-layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--n-nodes', type=int, default=100, help='Number of hidden nodes')
    parser.add_argument('--signal-process', type=str, default='sig', help='Signal process')
    parser.add_argument('--background-process', type=str, default='bkg', help='Background process')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of hidden nodes')
    parser.add_argument('--batch-size', type=float, default=32, help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--learning-rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Trainer accelerator')
    
    args = parser.parse_args()
    main(args)