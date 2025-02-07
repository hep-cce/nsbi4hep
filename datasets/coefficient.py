import os, pickle

import sys
sys.path.insert(1,'..')

from physics.simulation import events, msq
from physics.hzz import zpair, zz4l
from physics.hstar import c6

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import lightning as L
from torch.utils.data import DataLoader, random_split, Dataset
import torch

class RolypolyDataModule(L.LightningDataModule):

    def __init__(self, filepath: str = '', features = ['mandelstam_s', 'mandelstam_t', 'mandelstam_u'], coefficient_index=1, component = msq.Component.SBI, scaler_path = 'scaler.pkl', sample_size = 10000, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.filepath = filepath
        self.features = features
        self.component = component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        self.coefficient_index = coefficient_index
    
    def prepare_data(self):
        sample = sample.from_csv(cross_section=1.0, file_path=self.filepath)
        
        z_cand = zpair.ZPairCandidate(algorithm='truth')
        z_masses = zpair.ZPairMassWindow(z1=(70,115), z2=(70,115))
        mandelstam = zz4l.MandelstamVariables()

        self.sample = sample.calculate(z_cand).filter(z_masses).calculate(mandelstam)

    def setup(self, stage: str):
        if stage=='fit':

            sample_train, sample_val = self.sample.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            self.training_data = CoefficientDataset(sample_train, features=self.features, coefficient_index=self.coefficient_index, component=self.component, sample_size=self.sample_size, random_state=self.random_state)
            self.validation_data = CoefficientDataset(sample_val, features=self.features, coefficient_index=self.coefficient_index, component=self.component, sample_size=self.sample_size, random_state=self.random_state)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

class CoefficientDataset(Dataset):

    def __init__(self, sample, features, coefficient_index, sample_size, component = msq.Component.SBI, random_state=None):
        super().__init__()
        c6_mod = c6.Modifier(baseline=component, sample=sample, c6_values=[-5,-1,0,1,5]) if component!=msq.Component.INT else c6.Modifier(baseline=component, sample=sample, c6_values=[-5,0,5])
        coefficient = c6_mod.coefficients[:,coefficient_index]
        
        unweighted_indices = sample.weights.sample(n=sample_size, replace=True, weights=sample.weights, random_state=random_state).index
        
        self.X = sample.kinematics[features].to_numpy()[unweighted_indices]
        self.y = coefficient[unweighted_indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)
