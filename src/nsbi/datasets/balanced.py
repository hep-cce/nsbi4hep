import os
import pickle
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from torch.utils.data import DataLoader, Dataset


class BalancedDataModule(L.LightningDataModule):
    def __init__(
        self,
        loader: Any,
        numerator_events: str = "",
        denominator_events: str = "",
        sample_size: int = 10000,
        batch_size: int = 32,
        random_state: int | None = None,
        data_dir: str = "./",
        return_kin_val: bool = False,
        train_size: float = 0.6,
        val_size: float = 0.2,
        wi_fit_size: float = 0.0,
        test_size: float = 0.2,
        bootstrap: bool = False,
        num_workers: int = 8,
        predict_files: list[str] | None = None,
        predict_sample_size: int | None = None,
        predict_loader: Any = None,
    ):
        super().__init__()

        self.loader = loader
        # Predict may read files `loader` cannot e.g. a derived CSV of just features and a weight,
        # rather than full MCFM output. None reuses `loader`.
        self.predict_loader = predict_loader or loader

        self.numerator_file = numerator_events
        self.denominator_file = denominator_events

        # Files scored by the `predict` stage, one output file each. None falls back to the
        # numerator/denominator files, so a bare `stage=predict` scores what was trained on.
        self.predict_files = list(predict_files) if predict_files else None
        # Defaults to all rows, no shuffle to keep the
        # row-for-row correspondence between an input file and its score file.
        self.predict_sample_size = predict_sample_size

        self.sample_size = sample_size

        self.batch_size = batch_size
        self.random_state = random_state

        self.data_dir = data_dir
        self.scaler = StandardScaler()

        self.return_kin_val = return_kin_val
        # The *_size values are relative fractions: they are normalized by their sum in _split,
        # so they need not add up to 1 (all data is used). e.g. 0.6/0.2/0/0.2 and 3/1/0/1 both
        # give a 60/20/0/20 split. Each split is carved via train_test_split, which rejects empty
        # fractions.
        if train_size <= 0 or val_size <= 0 or test_size <= 0 or wi_fit_size < 0:
            raise ValueError(
                "Invalid split sizes: train_size, val_size and test_size must be > 0 and "
                f"wi_fit_size must be >= 0, got train_size={train_size}, val_size={val_size}, "
                f"wi_fit_size={wi_fit_size}, test_size={test_size}"
            )
        self.train_size = train_size
        self.val_size = val_size
        # wi_fit is a held-out split used to fit ensemble (w_i f_i) weights and estimate their
        # covariance on data independent of member training; wi_fit_size == 0 disables it and
        # reduces to the original 3-way train/val/test split.
        self.wi_fit_size = wi_fit_size
        self.test_size = test_size
        # When True, bootstrap-resample the training split (only) per ensemble member.
        self.bootstrap = bootstrap
        self.num_workers = num_workers

    def _split(self, X, w):
        total = self.train_size + self.val_size + self.wi_fit_size + self.test_size
        X_train, X_rest, w_train, w_rest = train_test_split(
            X, w, train_size=self.train_size / total, shuffle=False
        )

        rest_total = self.val_size + self.wi_fit_size + self.test_size
        X_val, X_rest, w_val, w_rest = train_test_split(
            X_rest, w_rest, train_size=self.val_size / rest_total, shuffle=False
        )

        if self.wi_fit_size > 0:
            # Peel off test, leaving wi_fit as the leftover complement (carved last).
            X_test, X_wi_fit, w_test, w_wi_fit = train_test_split(
                X_rest, w_rest,
                train_size=self.test_size / (self.test_size + self.wi_fit_size),
                shuffle=False,
            )
            wi_fit = (X_wi_fit, w_wi_fit)
        else:
            # No wi_fit split: test is the leftover complement (original 3-way behaviour).
            X_test, w_test = X_rest, w_rest
            wi_fit = None

        return (X_train, w_train), (X_val, w_val), wi_fit, (X_test, w_test)

    def _require_data_dir(self):
        if not Path(self.data_dir).is_dir():
            raise NotADirectoryError(
                f"datamodule.data_dir does not exist: {self.data_dir} — create it before running "
                "(it is not created automatically), or point datamodule.data_dir at an existing "
                "directory."
            )

    def prepare_data(self):
        self._require_data_dir()
        X_numerator, w_numerator = self.loader(self.numerator_file, sample_size=self.sample_size, random_state=self.random_state)
        X_denominator, w_denominator = self.loader(self.denominator_file, sample_size=self.sample_size, random_state=self.random_state)

        (
            (X_numerator_train, w_numerator_train),
            (X_numerator_val, w_numerator_val),
            wi_fit_numerator,
            (X_numerator_test, w_numerator_test),
        ) = self._split(X_numerator, w_numerator)
        (
            (X_denominator_train, w_denominator_train),
            (X_denominator_val, w_denominator_val),
            wi_fit_denominator,
            (X_denominator_test, w_denominator_test),
        ) = self._split(X_denominator, w_denominator)

        self.training_data = BalancedDataset(
            X_numerator_train,
            w_numerator_train,
            X_denominator_train,
            w_denominator_train,
            scaler=None,
            random_state=self.random_state,
        )
        self.scaler.fit(self.training_data.X)

        # save stuff for later
        with open(os.path.join(self.data_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(self.data_dir, "events_numerator_train.pkl"), "wb") as f:
            pickle.dump((X_numerator_train, w_numerator_train), f)
        with open(os.path.join(self.data_dir, "events_denominator_train.pkl"), "wb") as f:
            pickle.dump((X_denominator_train, w_denominator_train), f)
        with open(os.path.join(self.data_dir, "events_numerator_val.pkl"), "wb") as f:
            pickle.dump((X_numerator_val, w_numerator_val), f)
        with open(os.path.join(self.data_dir, "events_denominator_val.pkl"), "wb") as f:
            pickle.dump((X_denominator_val, w_denominator_val), f)
        with open(os.path.join(self.data_dir, "events_numerator_test.pkl"), "wb") as f:
            pickle.dump((X_numerator_test, w_numerator_test), f)
        with open(os.path.join(self.data_dir, "events_denominator_test.pkl"), "wb") as f:
            pickle.dump((X_denominator_test, w_denominator_test), f)
        if wi_fit_numerator is not None:
            with open(os.path.join(self.data_dir, "events_numerator_wi_fit.pkl"), "wb") as f:
                pickle.dump(wi_fit_numerator, f)
            with open(os.path.join(self.data_dir, "events_denominator_wi_fit.pkl"), "wb") as f:
                pickle.dump(wi_fit_denominator, f)

    def setup(self, stage: str):
        self._require_data_dir()
        # `predict` never rebuilds the split -- it only needs the training
        # scaler, so a missing one is an error
        if stage != "predict" and not (Path(self.data_dir) / "scaler.pkl").exists():
            self.prepare_data()

        if stage == "fit":
            with open(os.path.join(self.data_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_numerator_train.pkl"), "rb") as f:
                X_numerator_train, w_numerator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_denominator_train.pkl"), "rb") as f:
                X_denominator_train, w_denominator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_numerator_val.pkl"), "rb") as f:
                X_numerator_val, w_numerator_val = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_denominator_val.pkl"), "rb") as f:
                X_denominator_val, w_denominator_val = pickle.load(f)

            if self.bootstrap:
                # Bootstrap-resample the TRAINING split only (val/wi_fit/test stay fixed and
                # shared across ensemble members) so each member sees a different training draw,
                # per the wifi (w_i f_i) ensembling procedure. Seeded by random_state.
                X_numerator_train, w_numerator_train = resample(
                    X_numerator_train, w_numerator_train,
                    replace=True, random_state=self.random_state,
                )
                X_denominator_train, w_denominator_train = resample(
                    X_denominator_train, w_denominator_train,
                    replace=True, random_state=self.random_state,
                )

            self.training_data = BalancedDataset(
                X_numerator_train,
                w_numerator_train,
                X_denominator_train,
                w_denominator_train,
                scaler=self.scaler,
                random_state=self.random_state,
            )
            self.validation_data = BalancedDataset(
                X_numerator_val,
                w_numerator_val,
                X_denominator_val,
                w_denominator_val,
                scaler=self.scaler,
                random_state=self.random_state,
                return_kin=self.return_kin_val,
            )

        elif stage == "test":
            with open(os.path.join(self.data_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_numerator_test.pkl"), "rb") as f:
                X_numerator_test, w_numerator_test = pickle.load(f)
            with open(os.path.join(self.data_dir, "events_denominator_test.pkl"), "rb") as f:
                X_denominator_test, w_denominator_test = pickle.load(f)

            self.testing_data = BalancedDataset(
                X_numerator_test,
                w_numerator_test,
                X_denominator_test,
                w_denominator_test,
                scaler=self.scaler,
                random_state=self.random_state,
            )

        elif stage == "predict":
            scaler_path = Path(self.data_dir) / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"No scaler at {scaler_path}. Prediction must use the scaler the model was "
                    "trained with; run the fit stage first (or point data_dir at its outputs)."
                )
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            files = self.predict_files or [self.numerator_file, self.denominator_file]
            self.predict_file_paths = [str(f) for f in files]
            self.predict_data = []
            for path in self.predict_file_paths:
                X, w = self.predict_loader(
                    path, sample_size=self.predict_sample_size, random_state=self.random_state
                )
                self.predict_data.append(PredictDataset(X, w, scaler=self.scaler, path=path))

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        """One sequential loader per predict file; ``ScoreWriter`` writes one score file each.

        Order is preserved (no sampler shuffling) so batch ``k`` of loader ``j`` holds rows
        ``k * batch_size ...`` of file ``j``.
        """
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for dataset in self.predict_data
        ]


class PredictDataset(Dataset):
    """Unshuffled features + event weights for one input file, scaled with the training scaler.

    Yields ``(x, w)``: models read ``batch[0]``, and ``ScoreWriter`` carries ``batch[1]`` into the
    ``weight`` column, so pairing a score with its event needs no index bookkeeping. ``path``
    records the source file, which ``ScoreWriter`` reads off the dataset to name its output.
    """

    def __init__(self, X, w, scaler=None, path=None):
        super().__init__()

        self.X = scaler.transform(X) if scaler is not None else X
        self.w = w
        self.path = path

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.X[idx], dtype=torch.float32),
            torch.as_tensor(self.w[idx], dtype=torch.float32),
        )


class BalancedDataset(Dataset):
    def __init__(
        self,
        X_numerator,
        w_numerator,
        X_denominator,
        w_denominator,
        scaler=None,
        random_state=None,
        return_kin=False,
    ):
        super().__init__()

        self.X = np.concatenate([X_numerator, X_denominator])
        self.kin = np.concatenate([X_numerator, X_denominator])

        # balanced weights
        w_numerator = w_numerator / w_numerator.sum()
        w_denominator = w_denominator / w_denominator.sum()
        self.w = np.concatenate([w_numerator, w_denominator])

        # numerator = signal = 1, denominator = background = 0
        self.s = np.concatenate([np.ones_like(w_numerator), np.zeros_like(w_denominator)])

        if scaler is not None:
            self.X = scaler.transform(self.X)

        self.return_kin = return_kin
        self.X, self.s, self.w, self.kin = shuffle(
            self.X, self.s, self.w, self.kin, random_state=random_state
        )

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        y = torch.as_tensor(self.s[idx], dtype=torch.float32)
        w = torch.as_tensor(self.w[idx], dtype=torch.float32)

        if self.return_kin:  # ⇢ validation loader
            kin = torch.as_tensor(self.kin[idx], dtype=torch.float32)
            return x, y, w, kin
        return x, y, w
