import os
import pickle
import re

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
        events_numerator = mcfm.from_csv(cross_section=1.0, file_path=self.numerator_file)
        events_denominator = mcfm.from_csv(cross_section=1.0, file_path=self.denominator_file)

        # apply filters and calculate kinematics
        zcands = zpair.ZPairCandidate(algorithm='leastsquare')
        zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
        h4lcp = zz4l.AngularVariables()
        h4lp4 = zz4l.FourLeptonSystem()
        h4l = zz4l.LeptonMomenta()
        events_numerator = events_numerator.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4).calculate(h4l)
        events_denominator = events_denominator.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4).calculate(h4l)

        with open('events_num.pkl', 'wb') as f:
            pickle.dump(events_numerator, f)

        with open('events_den.pkl', 'wb') as f:
            pickle.dump(events_denominator, f)


    def setup(self, stage: str):
        if stage =='fit':
            with open('events_num.pkl', 'rb') as fnum:
                events_num = pickle.load(fnum)

            with open('events_den.pkl', 'rb') as fden:
                events_den = pickle.load(fden)

            train_size, val_size, test_size = 1.0, 0.1, 0.1
            events_numerator_train, events_numerator_val, events_numerator_test = events_num.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)
            events_denominator_train, events_denominator_val, events_denominator_test = events_den.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)

            self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features)
            self.validation_data = BalancedDataset(events_numerator_val, events_denominator_val, self.features)
            self.testing_data = BalancedDataset(events_numerator_test, events_denominator_test, self.features)

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
