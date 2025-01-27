import os, pickle

from physics.simulation import msq, sample
from physics.hzz import zpair, zz4l
from physics.hstar import c6

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch


class RolypolyDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = '', sample_file: str = '', sample_xs = 1.6270497, sample_baseline = msq.Component.SBI, coefficient_index = 1, scaler_path = 'scaler.pkl', sample_size = 10000, offset=0, batch_size: int = 32, random_state: int=None) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.sample_file = sample_file
        self.sample_xs = sample_xs
        self.sample_baseline = sample_baseline
        self.coefficient_index = coefficient_index
        self.sample_size = sample_size
        self.offset = offset
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler_path = scaler_path

        self.scaler = StandardScaler()
    
    def prepare_data(self):
        self.data_full = CoefficientDataset(os.path.join(self.data_dir, self.sample_file),
                                            self.sample_xs,
                                            self.sample_baseline,
                                            self.coefficient_index,
                                            self.sample_size,
                                            self.offset)

    def setup(self, stage: str):
        if stage=='fit':
            self.training_data, self.validation_data = self.data_full.split(train_size=0.5, val_size=0.5, shuffle=True, random_state=self.random_state) 

            # Apply Scaler to both datasets after fitting to training data
            X_train = self.scaler.fit_transform(self.training_data.get_X())
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            X_val = self.scaler.transform(self.validation_data.get_X())

            self.training_data, self.validation_data = self.training_data.update_X(X_train), self.validation_data.update_X(X_val)
        
        if stage=='predict':
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            X = self.scaler.transform(self.data_full.get_X())

            self.prediction_data = self.data_full.update_X(X)
            

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.prediction_data, batch_size=self.batch_size)


class CoefficientDataset(Dataset):

    def __init__(self, sample_file: str = '', sample_xs = 1.6270497, sample_baseline = msq.Component.SBI, coefficient_index = 1, sample_size = 10000, offset = 0):
        super().__init__()

        self.sample_file = sample_file
        self.sample_xs = sample_xs
        self.sample_baseline = sample_baseline
        self.coefficient_index = coefficient_index
        self.sample_size = sample_size
        self.offset = offset

        self._create_dataset()

    def _create_dataset(self):
        sample_background = sample.from_csv(cross_section=self.background_xs, file_path=self.background_file, n_rows=self.offset+int(self.sample_size*1.2))
        
        msq_bkg_null = msq.MSQFilter('msq_bkg_sm', 0.0)
        msq_bkg_nan = msq.MSQFilter('msq_bkg_sm', np.nan)

        z_cand = zpair.ZPairCandidate(algorithm='leastsquare')
        z_masses = zpair.ZPairMassWindow(z1=(70,115), z2=(70,115))
        angles = zz4l.AngularVariables()
        four_lepton_vars = zz4l.FourLeptonSystem()

        sample_processed = sample_background.filter(msq_bkg_nan).filter(msq_bkg_null).calculate(z_cand).filter(z_masses).calculate(angles).calculate(four_lepton_vars)[self.offset:self.offset+self.sample_size]

        features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity']

        # Get only required features
        X = sample_processed.kinematics[features].to_numpy()

        c6_mod = c6.Modifier(baseline=self.sample_baseline, sample=sample_processed, c6_values=[-5,-1,0,1,5])
        coefficient = c6_mod.coefficients[self.coefficient_index]

        y = coefficient.to_numpy()

        sample_weights = self.sample_size*sample_processed.probabilities.to_numpy()

        self.data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(sample_weights, dtype=torch.float32))

    def get_X(self):
        return self.data.tensors[0]
    
    def get_y(self):
        return self.data.tensors[1]
    
    def get_weights(self):
        return self.data.tensors[2]
    
    def update_X(self, X):
        ds = self.copy()
        ds.data = TensorDataset(torch.tensor(X, dtype=torch.float32), self.data.tensors[1], self.data.tensors[2])
        return ds

    def update_y(self, y):
        ds = self.copy()
        ds.data = TensorDataset(self.data.tensors[0], torch.tensor(y, dtype=torch.float32), self.data.tensors[2])
        return ds

    def update_weights(self, weights):
        ds = self.copy()
        ds.data = TensorDataset(self.data.tensors[0], self.data.tensors[1], torch.tensor(weights, dtype=torch.float32))
        return ds

    def copy(self):
        ds = object.__new__(CoefficientDataset)
        ds.sample_file = self.sample_file
        ds.sample_xs = self.sample_xs
        ds.sample_baseline = self.sample_baseline
        ds.coefficient_index = self.coefficient_index
        ds.sample_size = self.sample_size
        ds.offset = self.offset

        ds.data = self.data
        return ds
    
    def split(self, train_size=1, val_size=1, test_size=None, random_state=None, shuffle=True):
        if test_size is not None:
            total = train_size + val_size + test_size
            train_size /= total
            val_size /= total
            test_size /= total
                
            split_1_X, X_test, split_1_y, y_test, split_1_wt, weights_test = train_test_split(self.data.tensors[0], self.data.tensors[1], self.data.tensors[2], test_size=test_size, train_size=train_size+val_size, random_state=random_state, shuffle=shuffle)
            X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(split_1_X, split_1_y, split_1_wt, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle)

            ds_train = self.copy()
            ds_train.data = TensorDataset(X_train, y_train, weights_train)

            ds_val = self.copy()
            ds_val.data = TensorDataset(X_val, y_val, weights_val)

            ds_test = self.copy()
            ds_test.data = TensorDataset(X_test, y_test, weights_test)

            return ds_train, ds_val, ds_test

        else:
            total = train_size + val_size
            train_size /= total
            val_size /= total

            X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(self.data.tensors[0], self.data.tensors[1], self.data.tensors[2], test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle)
            
            ds_train = self.copy()
            ds_train.data = TensorDataset(X_train, y_train, weights_train)

            ds_val = self.copy()
            ds_val.data = TensorDataset(X_val, y_val, weights_val)

            return ds_train, ds_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]