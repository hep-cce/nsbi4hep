import os
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L

from physics.hstar import c6
from physics.simulation import sample
from physics.hzz import zpair, zz4l

class BalancedDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = '', signal_file: str = '', background_file: str = '', sample_size = 10000, batch_size: int = 32, random_state: int=None):
        super().__init__()

        self.data_dir = data_dir
        self.signal_file = signal_file
        self.background_file = background_file
        self.random_state = random_state
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.scaler = StandardScaler(with_mean=False)

    def prepare_data(self):

        # load samples with enough entries to sufficient size
        sample_sig = sample.from_csv(0.1, os.path.join(self.data_dir, self.signal_file))
        sample_bkg = sample.from_csv(1.6, os.path.join(self.data_dir, self.background_file))

        # apply filters and calculate kinematics
        zcands = zpair.ZPairCandidate(algorithm='leastsquare')
        zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
        h4lcp = zz4l.AngularVariables()
        h4lp4 = zz4l.FourLeptonSystem()
        sample_sig = sample_sig.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)
        sample_bkg = sample_bkg.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)

        # unweight both to *same* (i.e. balanced) sample size
        sample_sig_train = sample_sig.unweight(self.sample_size, random_state=self.random_state)
        sample_sig_val = sample_sig.unweight(self.sample_size, random_state=self.random_state)
        sample_bkg_train = sample_bkg.unweight(self.sample_size, random_state=self.random_state)
        sample_bkg_val = sample_bkg.unweight(self.sample_size, random_state=self.random_state)

        # set kinematics as features
        observables = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']

        # using 4l angular observables + mass + rapidity
        obs_sig_train = sample_sig_train.kinematics[observables].to_numpy()
        obs_sig_val = sample_sig_val.kinematics[observables].to_numpy()
        obs_bkg_train = sample_bkg_train.kinematics[observables].to_numpy()
        obs_bkg_val = sample_bkg_val.kinematics[observables].to_numpy()

        # kinematic features tensors
        X_train = torch.cat([torch.tensor(obs_sig_train,dtype=torch.float32),torch.tensor(obs_bkg_train,dtype=torch.float32)])
        X_val = torch.cat([torch.tensor(obs_sig_val,dtype=torch.float32),torch.tensor(obs_bkg_val,dtype=torch.float32)])

        # scale features
        self.scaler.fit_transform(X_train)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        self.scaler.transform(X_val)

        # y(signal) = 1, y(background) = 0 target tensors
        y_train = torch.cat([torch.tensor(np.ones(self.sample_size),dtype=torch.float32),torch.tensor(np.zeros(self.sample_size),dtype=torch.float32)])
        y_val = torch.cat([torch.tensor(np.ones(self.sample_size),dtype=torch.float32),torch.tensor(np.zeros(self.sample_size),dtype=torch.float32)])

        # create (features, target) tensors
        self.train_data = torch.utils.data.TensorDataset(X_train, y_train)
        self.val_data = torch.utils.data.TensorDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: str):
        pass