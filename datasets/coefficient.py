import os, pickle

from physics.simulation import mcfm, msq
from physics.hzz import zz4l, zz2l2v
from physics.hstar import c6

import numpy as np
from sklearn.preprocessing import StandardScaler

import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch

class CoefficientDataModule(L.LightningDataModule):

    def __init__(self, events: str = '', analysis : str = None, features : list = None, coefficient : int = None, component : str = 'sbi', sample_size : int = None, batch_size: int = None, random_state: int=None) -> None:
        super().__init__()

        self.file_path = events
        self.analysis = analysis
        self.features = features
        self.component = component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.coefficient_index = coefficient
    
    def prepare_data(self):
        events = mcfm.from_csv(file_path=self.file_path)
        if self.analysis == '4l':
            events = zz4l.analyze(events)
        elif self.analysis == '2l2v':
            events = zz2l2v.analyze(events)

        train_size, val_size, test_size = 1.0, 0.5, 0.5
        events_train, events_val, events_test = events.sample(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)

        with open('events_train.pkl', 'wb') as f:
            pickle.dump(events_train, f)
        with open('events_val.pkl', 'wb') as f:
            pickle.dump(events_val, f)
        with open('events_test.pkl', 'wb') as f:
            pickle.dump(events_test, f)

        scaler_X = StandardScaler()
        scaler_X.fit(events.kinematics[self.features].to_numpy())
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)

        scaler_y = StandardScaler(with_mean=False)
        c6_mod = c6.Modifier(baseline=self.component, events=events_train, c6_points=([-20, -10, 0, 10, 20] if self.component!=msq.Component.INT else [-10, 0, 10]))
        scaler_y.fit(c6_mod.coefficients[:,self.coefficient_index].reshape(-1, 1))
        with open('scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)

    def setup(self, stage: str):
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        
        if stage=='fit':
            with open('scaler_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)
            with open('events_train.pkl', 'rb') as f:
                events_train = pickle.load(f)
            with open('events_val.pkl', 'rb') as f: 
                events_val = pickle.load(f)
            self.training_data = CoefficientDataset(events_train, features=self.features, component=self.component, coefficient_index=self.coefficient_index, scaler_X=scaler_X, scaler_y=scaler_y)
            self.validation_data = CoefficientDataset(events_val, features=self.features, component=self.component, coefficient_index=self.coefficient_index, scaler_X=scaler_X, scaler_y=scaler_y)

        elif stage=='test':
            with open('events_test.pkl', 'rb') as f:
                events_test = pickle.load(f)
            self.testing_data = CoefficientDataset(events_test, features=self.features, component=self.component, coefficient_index=self.coefficient_index, scaler_X=scaler_X)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class CoefficientDataset(Dataset):

    def __init__(self, events, features, coefficient_index, component = msq.Component.SBI, scaler_X = None, scaler_y = None):
        super().__init__()

        c6_mod = c6.Modifier(baseline=component, events=events, c6_points=([-20, -10, 0, 10, 20] if component!=msq.Component.INT else [-10, 0, 10]))
        
        self.X = events.kinematics[features].to_numpy()
        self.y = c6_mod.coefficients[:,coefficient_index]
        self.w = events.weights.to_numpy() * self.y.size / events.weights.sum()

        if scaler_X is not None:
            self.X = scaler_X.transform(self.X)
        if scaler_y is not None:
            self.y = scaler_y.transform(self.y.reshape(-1, 1)).reshape(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32), torch.tensor(self.w[index], dtype=torch.float32)
