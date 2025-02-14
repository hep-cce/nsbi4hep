import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from physics.hstar import c6
from physics.simulation import mcfm, msq
from physics.hzz import zpair, zz4l

class BalancedDataModule(L.LightningDataModule):

    def __init__(self, numerator_file: str = '', denominator_file: str = '', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], scaler_path = 'scaler.pkl', sample_size = 10000, batch_size: int = 32, random_state: int=None):
        super().__init__()

        self.numerator_file = numerator_file
        self.denominator_file = denominator_file
        self.features = features
        self.scaler_path = scaler_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state

        self.scaler = StandardScaler()

    def prepare_data(self):
        # load samples with enough entries to sufficient size
        sample_numerator = events.from_csv(cross_section=0.1, file_path=self.numerator_file)
        sample_denominator = events.from_csv(cross_section=1.6, file_path=self.denominator_file)

        # apply filters and calculate kinematics
        zcands = zpair.ZPairCandidate(algorithm='leastsquare')
        zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
        h4lcp = zz4l.AngularVariables()
        h4lp4 = zz4l.FourLeptonSystem()
        self.sample_numerator = sample_numerator.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)
        self.sample_denominator = sample_denominator.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)


    def setup(self, stage: str):
        if stage =='fit':
            sample_numerator_train, sample_numerator_val = self.sample_numerator.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)
            sample_denominator_train, sample_denominator_val = self.sample_denominator.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            self.training_data = BalancedDataset(sample_numerator_train, sample_denominator_train, self.features, self.sample_size, self.random_state)
            self.validation_data = BalancedDataset(sample_numerator_val, sample_denominator_val, self.features, self.sample_size, self.random_state)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise DataLoader(self.training_data, batch_size=self.batch_size)

class BalancedDataset(Dataset):
    def __init__(self, sample_sig, sample_bkg, features, sample_size, random_state=None):
        super().__init__()
        sample_sig = sample_sig.unweight(sample_size, random_state=random_state)
        sample_bkg = sample_bkg.unweight(sample_size, random_state=random_state)

        # Get only required features
        X_sig = sample_sig.kinematics[features].to_numpy()
        X_bkg = sample_bkg.kinematics[features].to_numpy()

        self.X = np.concatenate([X_sig, X_bkg])

        self.s = np.concatenate([np.ones(sample_size), np.zeros(sample_size)])

        self.X, self.s = shuffle(self.X, self.s, random_state=random_state)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)