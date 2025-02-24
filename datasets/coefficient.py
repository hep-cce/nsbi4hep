import os, pickle

import sys
sys.path.insert(1,'..')

from physics.simulation import mcfm, msq
from physics.hzz import zpair, zz4l
from physics.hstar import c6

from sklearn.preprocessing import StandardScaler

import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch

class RolypolyDataModule(L.LightningDataModule):

    def __init__(self, filepath: str = '', features = ['mandelstam_s', 'mandelstam_t', 'mandelstam_u'], coefficient_index=1, component = msq.Component.SBI, X_scaler_path = 'scaler_X.pkl', y_scaler_path = 'scaler_y.pkl', sample_size = 10000, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.filepath = filepath
        self.features = features
        self.component = component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.X_scaler = StandardScaler()
        self.X_scaler_path = X_scaler_path
        self.y_scaler = StandardScaler()
        self.y_scaler_path = y_scaler_path
        self.coefficient_index = coefficient_index
    
    def prepare_data(self):
        events = mcfm.from_csv(cross_section=1.0, file_path=self.filepath)
        
        z_cand = zpair.ZPairCandidate(algorithm='truth')
        z_masses = zpair.ZPairMassWindow(z1=(70,115), z2=(70,115))
        mandelstam = zz4l.MandelstamVariables()

        self.events = events.calculate(z_cand).filter(z_masses).calculate(mandelstam)

    def setup(self, stage: str):
        if stage=='fit':

            events_train, events_val = self.events.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            self.training_data = CoefficientDataset(events_train, features=self.features, coefficient_index=self.coefficient_index, component=self.component, sample_size=self.sample_size, random_state=self.random_state)
            self.validation_data = CoefficientDataset(events_val, features=self.features, coefficient_index=self.coefficient_index, component=self.component, sample_size=self.sample_size, random_state=self.random_state)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.X_scaler.fit_transform(self.training_data.X)
            with open(self.X_scaler_path, 'wb') as f:
                pickle.dump(self.X_scaler, f)
            self.validation_data.X = self.X_scaler.transform(self.validation_data.X)

            self.training_data.y = self.y_scaler.fit_transform(self.training_data.y)
            with open(self.y_scaler_path, 'wb') as f:
                pickle.dump(self.y_scaler, f)
            self.validation_data.y = self.y_scaler.transform(self.validation_data.y)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

class CoefficientDataset(Dataset):

    def __init__(self, events, features, coefficient_index, sample_size, component = msq.Component.SBI, random_state=None):
        super().__init__()
        c6_mod = c6.Modifier(baseline=component, sample=events, c6_values=[-5,-1,0,1,5]) if component!=msq.Component.INT else c6.Modifier(baseline=component, sample=sample, c6_values=[-5,0,5])
        coefficient = c6_mod.coefficients[:,coefficient_index]
        
        unweighted_indices = events.weights.sample(n=sample_size, replace=True, weights=events.weights, random_state=random_state).index
        
        self.X = events.kinematics[features].to_numpy()[unweighted_indices]
        self.y = coefficient[unweighted_indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)
