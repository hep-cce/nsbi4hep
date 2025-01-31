import os, pickle

from physics.simulation import msq, sample
from physics.hzz import zpair, zz4l

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch

class AliceDataModule(L.LightningDataModule):

    def __init__(self, filepath: str = '', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG, scaler_path = 'scaler.pkl', sample_size = 10000, offset=0, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.filepath = background_file
        self.features = features
        self.numerator_component = numerator_component
        self.denominator_component = denominator_component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
    
    def prepare_data(self):
        sample_bkg = sample.from_csv(cross_section=1.0, file_path=self.background_file, n_rows=self.offset+int(self.sample_size*1.2))
        
        z_cand = zpair.ZPairCandidate(algorithm='leastsquare')
        z_masses = zpair.ZPairMassWindow(z1=(70,115), z2=(70,115))
        angles = zz4l.AngularVariables()
        four_lepton_vars = zz4l.FourLeptonSystem()

        self.sample_bkg = sample_bkg.calculate(z_cand).filter(z_masses).calculate(angles).calculate(four_lepton_vars)

    def setup(self, stage: str):
        if stage=='fit':

            sample_bkg_train, sample_bkg_val = self.sample_bkg.shuffle(random_state=self.random_state).split(train_size=0.5, val_size=0.5)

            self.training_data = JointLikelihoodDataset(sample_bkg_train, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size)
            self.validation_data = JointLikelihoodDataset(sample_bkg_val, features=self.features, numerator_component=self.numerator_component, denominator_component=self.denominator_component, sample_size=self.sample_size)

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
        return DataLoader(self.prediction_data, batch_size=self.batch_size)

class JointLikelihoodDataset(Dataset):

    def __init__(self, sample, features, sample_size, numerator_component = msq.Component.SIG, denominator_component = msq.Component.BKG,):
        super().__init__()
        sample = sample.unweight(sample_size)

        # Get only required features
        self.X = sample.kinematics[features].to_numpy()

        # Get PDF ratios for p(theta_0)/p(theta_1)
        r = sample.probabilities/sample.reweight(numerator=self.numerator_component, denominator=self.denominator_component).probabilities
        self.s = (1/(1 + r)).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32)