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

    def __init__(self, numerator_events: str = '', denominator_events: str = '', analysis = 'h4l', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], sample_size = 10000, batch_size: int = 32, random_state: int=None, data_dir : str = './'):
        super().__init__()

        self.analysis = analysis
        self.features = features

        self.numerator_file = numerator_events
        self.denominator_file = denominator_events

        self.sample_size = sample_size

        self.batch_size = batch_size
        self.random_state = random_state

        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def prepare_data(self):

        events_numerator = mcfm.from_csv(cross_section=1.0, file_path=self.numerator_file)
        events_denominator = mcfm.from_csv(cross_section=1.0, file_path=self.denominator_file)

        if self.analysis == 'h4l':
            events_numerator = zz4l.analyze(events_numerator)
            events_denominator = zz4l.analyze(events_denominator)
        elif self.analysis == 'h2l2v':
            events_numerator = zz2l2v.analyze(events_numerator)
            events_denominator = zz2l2v.analyze(events_denominator)

        train_size, val_size, test_size = 1.0, 0.05, 0.05
        events_numerator_train, events_numerator_val, events_numerator_test = events_numerator.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)
        events_denominator_train, events_denominator_val, events_denominator_test = events_denominator.unweight(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)

        self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features, scaler = None, random_state = self.random_state)
        self.scaler.fit(self.training_data.X)

        # save stuff for later
        with open(os.path.join(self.data_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'wb') as f:
            pickle.dump(events_numerator_train, f)
        with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'wb') as f:
            pickle.dump(events_denominator_train, f)
        with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'wb') as f:
            pickle.dump(events_numerator_val, f)
        with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'wb') as f:
            pickle.dump(events_denominator_val, f)
        with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'wb') as f:
            pickle.dump(events_numerator_test, f)
        with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'wb') as f:
            pickle.dump(events_denominator_test, f)

    def setup(self, stage: str):

        if stage == 'fit':

            with open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'rb') as f:
                events_numerator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'rb') as f:
                events_denominator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'rb') as f:
                events_numerator_val = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'rb') as f:
                events_denominator_val = pickle.load(f)

            self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features, scaler = self.scaler, random_state=self.random_state)
            self.validation_data = BalancedDataset(events_numerator_val, events_denominator_val, self.features, scaler = self.scaler, random_state=self.random_state)

        elif stage == 'test':
            with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'rb') as f:
                events_numerator_test = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'rb') as f:
                events_denominator_test = pickle.load(f)

            self.testing_data = BalancedDataset(events_numerator_test, events_denominator_test, self.features, scaler = self.scaler, random_state=self.random_state)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class BalancedDataset(Dataset):
    def __init__(self, events_numerator = None, events_denominator = None, features = None, scaler = None, random_state = None):
        super().__init__()
        # Get only required features
        X_numerator = events_numerator.kinematics[features].to_numpy()
        X_denominator = events_denominator.kinematics[features].to_numpy()

        self.X = np.concatenate([X_numerator, X_denominator])
        self.s = np.concatenate([np.ones(len(X_numerator)), np.zeros(len(X_denominator))])

        if scaler is not None:
            self.X = scaler.transform(self.X)
        
        self.X, self.s, self.indices = shuffle(self.X, self.s, np.arange(len(self.s)), random_state=random_state)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
