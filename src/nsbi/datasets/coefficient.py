import pickle

import lightning as L
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from nsbi.physics.hstar import eft
from nsbi.physics.simulation import mcfm, msq


class CoefficientDataModule(L.LightningDataModule):
    def __init__(
        self,
        events: str = "",
        features: list | None = None,
        coefficient: list | None = None,
        component: str = "sbi",
        sample_size: int | None = None,
        batch_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()

        self.file_path = events
        self.features = features
        self.component = component
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.coefficient_index = coefficient

    def prepare_data(self):
        events = mcfm.from_csv(
            cross_section=None, file_path=self.file_path, kinematics=self.features
        )

        train_size, val_size, test_size = 6, 2, 2
        events_train, events_val, events_test = events.sample(
            self.sample_size, random_state=self.random_state
        ).split(train_size=train_size, val_size=val_size, test_size=test_size)

        with open("events_train.pkl", "wb") as f:
            pickle.dump(events_train, f)
        with open("events_val.pkl", "wb") as f:
            pickle.dump(events_val, f)
        with open("events_test.pkl", "wb") as f:
            pickle.dump(events_test, f)

        scaler_X = StandardScaler()
        scaler_X.fit(events.kinematics[self.features].to_numpy())
        with open("scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_X, f)

        scaler_y = StandardScaler(with_mean=False)
        eft_mod = eft.Modifier(baseline=self.component, events=events_train)
        scaler_y.fit(eft_mod.coefficients[(slice(None), *self.coefficient_index)].reshape(-1, 1))
        with open("scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)

    def setup(self, stage: str):
        with open("scaler_X.pkl", "rb") as f:
            scaler_X = pickle.load(f)

        if stage == "fit":
            with open("scaler_y.pkl", "rb") as f:
                scaler_y = pickle.load(f)
            with open("events_train.pkl", "rb") as f:
                events_train = pickle.load(f)
            with open("events_val.pkl", "rb") as f:
                events_val = pickle.load(f)
            self.training_data = CoefficientDataset(
                events_train,
                features=self.features,
                component=self.component,
                coefficient_index=self.coefficient_index,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
            )
            self.validation_data = CoefficientDataset(
                events_val,
                features=self.features,
                component=self.component,
                coefficient_index=self.coefficient_index,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
            )

        elif stage == "test":
            with open("events_test.pkl", "rb") as f:
                events_test = pickle.load(f)
            self.testing_data = CoefficientDataset(
                events_test,
                features=self.features,
                component=self.component,
                coefficient_index=self.coefficient_index,
                scaler_X=scaler_X,
            )

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)


class CoefficientDataset(Dataset):
    def __init__(
        self,
        events,
        features,
        coefficient_index,
        component=msq.Component.SBI,
        scaler_X=None,
        scaler_y=None,
    ):
        super().__init__()

        eft_mod = eft.Modifier(baseline=component, events=events)

        self.X = events.kinematics[features].to_numpy()
        self.y = eft_mod.coefficients[(slice(None), *coefficient_index)]
        self.w = events.weights.to_numpy() * self.y.size / events.weights.sum()

        if scaler_X is not None:
            self.X = scaler_X.transform(self.X)
        if scaler_y is not None:
            self.y = scaler_y.transform(self.y.reshape(-1, 1)).reshape(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32),
            torch.tensor(self.w[index], dtype=torch.float32),
        )
