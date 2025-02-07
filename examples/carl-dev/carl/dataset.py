import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L

from physics.hstar import c6
from physics.simulation import events
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

        # self.scaler = StandardScaler(with_mean=False)
        self.scaler = StandardScaler()

    def prepare_data(self):

        # load samples with enough entries to sufficient size
        sample_sig = events.from_csv(0.1, os.path.join(self.data_dir, self.signal_file))
        sample_bkg = events.from_csv(1.6, os.path.join(self.data_dir, self.background_file))

        # apply filters and calculate kinematics
        zcands = zpair.ZPairCandidate(algorithm='leastsquare')
        zmasses = zpair.ZPairMassWindow(z1 = (70,115), z2 = (70,115))
        h4lcp = zz4l.AngularVariables()
        h4lp4 = zz4l.FourLeptonSystem()
        sample_sig = sample_sig.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)
        sample_bkg = sample_bkg.calculate(zcands).filter(zmasses).calculate(h4lcp).calculate(h4lp4)

        # unweight signal & background to *same* (i.e. balanced) sample size
        sample_sig_train = sample_sig.unweight(self.sample_size, random_state=self.random_state)
        sample_bkg_train = sample_bkg.unweight(self.sample_size, random_state=self.random_state)
        sample_sig_val = sample_sig.unweight(self.sample_size, random_state=self.random_state)
        sample_bkg_val = sample_bkg.unweight(self.sample_size, random_state=self.random_state)

        # features
        observables = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']

        # 4l angular observables + mass + rapidity
        obs_sig_train = sample_sig_train.kinematics[observables].to_numpy()
        obs_sig_val = sample_sig_val.kinematics[observables].to_numpy()
        obs_bkg_train = sample_bkg_train.kinematics[observables].to_numpy()
        obs_bkg_val = sample_bkg_val.kinematics[observables].to_numpy()

        # signal & background
        X_train = np.concatenate([obs_sig_train,obs_bkg_train])
        X_val = np.concatenate([obs_sig_val,obs_bkg_val])

        # scale
        X_train = self.scaler.fit_transform(X_train)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        X_val = self.scaler.transform(X_val)

        # y(signal) = 1, y(background) = 0 target tensors
        y_train = np.concatenate([np.ones(self.sample_size),np.zeros(self.sample_size)])
        y_val = np.concatenate([np.ones(self.sample_size),np.zeros(self.sample_size)])

        # shuffle features & labels consistently
        X_train, y_train = shuffle(X_train, y_train, random_state=self.random_state) 
        X_val, y_val = shuffle(X_train, y_train, random_state=self.random_state) 

        # create (features, target) tensors
        self.train_data = torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.float32))
        self.val_data = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val,dtype=torch.float32))

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