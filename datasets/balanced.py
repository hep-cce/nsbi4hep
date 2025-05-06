import os
import pickle
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from physics.simulation import mcfm, msq
from physics.hzz import zz4l, zz2l2v

class BalancedDataModule(L.LightningDataModule):

    def __init__(self, numerator_file: str = '', denominator_file: str = '', analysis = 'h4l', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], scaler_path = 'scaler.pkl', sample_size = 10000, batch_size: int = 32, random_state: int=None):
        super().__init__()

        self.numerator_file = numerator_file
        self.denominator_file = denominator_file
        self.analysis = analysis
        self.features = features
        self.scaler_path = scaler_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state

        self.scaler = StandardScaler()

    def prepare_data(self):
        events_num = mcfm.from_csv(cross_section=1.0, file_path=self.numerator_file)
        events_denom = mcfm.from_csv(cross_section=1.0, file_path=self.denominator_file)

        if self.analysis == 'h4l':
            events_num = zz4l.analyze(events_num)
            events_denom = zz4l.analyze(events_denom)
        elif self.analysis == 'h2l2v':
            events_num = zz2l2v.analyze(events_num)
            events_denom = zz2l2v.analyze(events_denom)

        with open('events_num.pkl', 'wb') as f:
            pickle.dump(events_num, f)
        with open('events_denom.pkl', 'wb') as f:
            pickle.dump(events_denom, f)

    def setup(self, stage: str):
        if stage =='fit':

            with open('events_num.pkl', 'rb') as fnum:
                events_num = pickle.load(fnum)
            with open('events_denom.pkl', 'rb') as fden:
                events_denom = pickle.load(fden)

            train_size, val_size, test_size = 1.0, 0.1, 0.1
            events_num_train, events_num_val, events_num_test = events_num.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)
            events_denom_train, events_denom_val, events_denom_test = events_denom.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)

            self.training_data = BalancedDataset(events_num_train, events_denom_train, self.features)
            self.validation_data = BalancedDataset(events_num_val, events_denom_val, self.features)
            self.testing_data = BalancedDataset(events_num_test, events_denom_test, self.features)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)
            self.testing_data.X = self.scaler.transform(self.testing_data.X)

            self.training_data.X, self.training_data.s, self.training_data.indices = shuffle(self.training_data.X, self.training_data.s, np.arange(len(self.training_data.s)), random_state=self.random_state)
            self.validation_data.X, self.validation_data.s, self.validation_data.indices = shuffle(self.validation_data.X, self.validation_data.s, np.arange(len(self.validation_data.s)), random_state=self.random_state)
            self.testing_data.X, self.testing_data.s, self.testing_data.indices = shuffle(self.testing_data.X, self.testing_data.s, np.arange(len(self.testing_data.s)), random_state=self.random_state)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class BalancedDataset(Dataset):
    def __init__(self, events_sig, events_bkg, features):
        super().__init__()
        # Get only required features
        X_sig = events_sig.kinematics[features].to_numpy()
        X_bkg = events_bkg.kinematics[features].to_numpy()

        self.X = np.concatenate([X_sig, X_bkg])
        self.s = np.concatenate([np.ones(len(X_sig)), np.zeros(len(X_bkg))])
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
