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

    def __init__(self, file_path: str = '', analysis : str = None, features : list = None, coefficient_index : int = None, component : msq.Component = msq.Component.SBI, scaler_path : str = 'scaler.pkl', sample_size : int = None, batch_size: int = None, random_state: int=None) -> None:
        super().__init__()

        self.file_path = file_path
        self.analysis = analysis
        self.features = features
        self.component = component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        self.coefficient_index = coefficient_index
    
    def prepare_data(self):
        events = mcfm.from_csv(cross_section=1.0, file_path=self.file_path)
        if self.analysis == '4l':
            events = zz4l.analyze(events)
        elif self.analysis == '2l2v':
            events = zz2l2v.analyze(events)
        with open('events.pkl', 'wb') as f:
            pickle.dump(events, f)

    def setup(self, stage: str):
        with open('events.pkl', 'rb') as fnum:
            events = pickle.load(fnum)
        
        if stage=='fit':

            train_size, val_size, test_size= 1.0, 0.1, 0.1
            events_train, events_val, events_test = events.sample(self.sample_size,random_state=self.random_state).split(train_size=train_size, val_size=val_size, test_size=test_size)

            self.training_data = CoefficientDataset(events_train, features=self.features, coefficient_index=self.coefficient_index, component=self.component)
            self.validation_data = CoefficientDataset(events_val, features=self.features, coefficient_index=self.coefficient_index, component=self.component)
            self.testing_data = CoefficientDataset(events_test, features=self.features, coefficient_index=self.coefficient_index, component=self.component)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)
            self.testing_data.X = self.scaler.transform(self.testing_data.X)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class CoefficientDataset(Dataset):

    def __init__(self, events, features, coefficient_index, component = msq.Component.SBI):
        super().__init__()

        c6_mod = c6.Modifier(baseline=component, events=events, c6_values=[-5,-1,0,1,5]) if component!=msq.Component.INT else c6.Modifier(baseline=component, events=events, c6_values=[-5,0,5])
        
        self.X = events.kinematics[features].to_numpy()
        self.y = c6_mod.coefficients[:,coefficient_index]
        self.w = events.weights.to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32), torch.tensor(self.w[index], dtype=torch.float32)
