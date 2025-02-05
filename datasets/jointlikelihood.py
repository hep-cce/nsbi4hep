import os, pickle

import sys
sys.path.insert(1,'..')

from physics.simulation import msq, sample
from physics.hzz import zpair, zz4l
from physics.hstar import c6

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch

class AliceDataModule(L.LightningDataModule):

    def __init__(self, filepath: str = '', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, scaler_path = 'scaler.pkl', c6_values = None, sample_size = 10000, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.filepath = filepath
        self.features = features
        self.numerator_component = numerator_component
        self.denominator_component = denominator_component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        self.c6_values = c6_values
    
    def prepare_data(self):
        sample_bkg = sample.from_csv(cross_section=1.0, file_path=self.filepath)
        
        z_cand = zpair.ZPairCandidate(algorithm='leastsquare')
        z_masses = zpair.ZPairMassWindow(z1=(70,115), z2=(70,115))
        angles = zz4l.AngularVariables()
        four_lepton_vars = zz4l.FourLeptonSystem()

        self.sample_bkg = sample_bkg.calculate(z_cand).filter(z_masses).calculate(angles).calculate(four_lepton_vars)

    def setup(self, stage: str):
        if stage=='fit':

            sample_bkg_train, sample_bkg_val = self.sample_bkg.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            if self.c6_values is None:
                self.training_data = JointLikelihoodDataset(sample_bkg_train, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
                self.validation_data = JointLikelihoodDataset(sample_bkg_val, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
            else:
                self.training_data = JointLikelihoodParameterizedDataset(sample_bkg_train, features=self.features, c6_values=self.c6_values, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)
                self.validation_data = JointLikelihoodParameterizedDataset(sample_bkg_val, features=self.features, c6_values=self.c6_values, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size, random_state=self.random_state)

            # Apply Scaler to both datasets after fitting to training data
            self.training_data.X = self.scaler.fit_transform(self.training_data.X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.validation_data.X = self.scaler.transform(self.validation_data.X)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

class JointLikelihoodDataset(Dataset):

    def __init__(self, sample, features, sample_size, numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, random_state=None):
        super().__init__()
        sample = sample.unweight(sample_size, random_state=random_state)

        # Get only required features
        self.X = sample.kinematics[features].to_numpy()

        # Get PDF ratios for p(theta_0)/p(theta_1)
        r = sample.probabilities/sample.reweight(numerator=numerator_component, denominator=denominator_component).probabilities

        self.s = (1/(1 + r)).to_numpy()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
    
class JointLikelihoodParameterizedDataset(Dataset):

    def __init__(self, sample_bkg, features, c6_values, sample_size, numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, random_state=None):
        super().__init__()
        c6_mod = c6.Modifier(baseline=numerator_component, sample=sample_bkg, c6_values=[-5,-1,0,1,5]) if numerator_component != msq.Component.INT else c6.Modifier(baseline=numerator_component, sample=sample_bkg, c6_values=[-5,0,5])
        _, c6_probabilities = c6_mod.modify(c6_values)

        X = np.tile(sample_bkg.kinematics[features].to_numpy(), (len(c6_values),1))
        c6_column = np.repeat(c6_values, len(sample_bkg.kinematics), axis=0)[:,np.newaxis]

        X = np.concatenate([X, c6_column], axis=1)

        probabilities_numerator = c6_probabilities.flatten()
        probabilities_denominator = np.tile(sample_bkg.probabilities.to_numpy(), len(c6_values))

        sample_weights = pd.Series((probabilities_numerator + probabilities_denominator)/2*sample_size).reset_index(drop=True)
        
        unweighted_indices = sample_weights.sample(n=sample_size, replace=True, weights=sample_weights, random_state=random_state).index

        s = 1/(1+probabilities_denominator/probabilities_numerator)

        self.X = X[unweighted_indices]
        self.s = s[unweighted_indices]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)
