import pickle

import numpy as np
import pytest

from nsbi.datasets.balanced import BalancedDataModule, BalancedDataset
from sklearn.preprocessing import StandardScaler

N_EVENTS = 160
N_FEATURES = 2
BATCH_SIZE = 32

DENOMINATOR_OFFSET = 1_000.0

SPLIT_3WAY = {"train_size": 0.5, "val_size": 0.25, "wi_fit_size": 0.0, "test_size": 0.25}
SPLIT_4WAY = {"train_size": 0.5, "val_size": 0.25, "wi_fit_size": 0.125, "test_size": 0.125}


def _split_counts(split):
    """Per-hypothesis event counts for a split spec, in (train, val, wi_fit, test) order."""
    total = sum(split.values())
    counts = [
        N_EVENTS * split[k] / total
        for k in ("train_size", "val_size", "wi_fit_size", "test_size")
    ]
    assert all(c == int(c) for c in counts), "split fractions must divide N_EVENTS evenly"
    return tuple(int(c) for c in counts)


def _make_events(offset=0.0, seed=0):
    """Unique-row feature matrix and positive weights."""
    rng = np.random.default_rng(seed)
    X = np.arange(N_EVENTS * N_FEATURES, dtype=np.float64).reshape(N_EVENTS, N_FEATURES) + offset
    w = rng.uniform(0.5, 1.5, size=N_EVENTS)
    return X, w


def _make_event_pair():
    """Numerator and denominator event sets, offset so no feature row is shared."""
    return _make_events(), _make_events(offset=DENOMINATOR_OFFSET, seed=1)


def _make_datamodule(data_dir, **overrides):
    (X_num, w_num), (X_den, w_den) = _make_event_pair()

    def loader(path, sample_size=None, random_state=None):
        return (X_num, w_num) if path == "num" else (X_den, w_den)

    kwargs = dict(
        loader=loader,
        numerator_events="num",
        denominator_events="den",
        data_dir=str(data_dir),
        batch_size=BATCH_SIZE,
        num_workers=0,
        random_state=1,
        **SPLIT_3WAY,
    )
    kwargs.update(overrides)
    return BalancedDataModule(**kwargs)


def test_split_without_wifit(tmp_path):
    dm = _make_datamodule(tmp_path)
    X, w = _make_events()
    n_train, n_val, _, n_test = _split_counts(SPLIT_3WAY)

    (X_tr, w_tr), (X_val, w_val), wi_fit, (X_te, w_te) = dm._split(X, w)

    assert wi_fit is None
    assert (len(X_tr), len(X_val), len(X_te)) == (n_train, n_val, n_test)
    # splits are unshuffled slices that together cover the input exactly
    np.testing.assert_array_equal(np.concatenate([X_tr, X_val, X_te]), X)
    np.testing.assert_array_equal(np.concatenate([w_tr, w_val, w_te]), w)


def test_split_with_wifit(tmp_path):
    dm = _make_datamodule(tmp_path, **SPLIT_4WAY)
    X, w = _make_events()
    n_train, n_val, n_wi_fit, n_test = _split_counts(SPLIT_4WAY)

    (X_tr, _), (X_val, _), (X_wi, w_wi), (X_te, _) = dm._split(X, w)

    assert (len(X_tr), len(X_val), len(X_wi), len(X_te)) == (n_train, n_val, n_wi_fit, n_test)
    # test is peeled off before wi_fit, so the input order is train | val | test | wi_fit
    np.testing.assert_array_equal(np.concatenate([X_tr, X_val, X_te, X_wi]), X)


def test_split_sizes_are_relative_fractions(tmp_path):
    dm_frac = _make_datamodule(tmp_path, **SPLIT_3WAY)
    # the same 2:1:1 ratio as SPLIT_3WAY, unnormalized
    dm_ratio = _make_datamodule(tmp_path, train_size=2, val_size=1, test_size=1)
    X, w = _make_events()

    for part_frac, part_ratio in zip(dm_frac._split(X, w), dm_ratio._split(X, w)):
        if part_frac is None:
            assert part_ratio is None
            continue
        np.testing.assert_array_equal(part_frac[0], part_ratio[0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_size": 0.0},
        {"val_size": 0.0},
        {"test_size": 0.0},
        {"train_size": -1.0},
        {"wi_fit_size": -0.1},
    ],
)
def test_invalid_split_sizes_raise(tmp_path, overrides):
    # train/val/test are carved with train_test_split, which rejects empty fractions, so an
    # unusable split has to fail at construction rather than deep inside prepare_data
    with pytest.raises(ValueError, match="Invalid split sizes"):
        _make_datamodule(tmp_path, **overrides)


def test_dataset_balances_weights_and_labels():
    (X_num, w_num), (X_den, w_den) = _make_event_pair()

    ds = BalancedDataset(X_num, w_num, X_den, w_den, random_state=0)

    assert len(ds) == 2 * N_EVENTS
    # each hypothesis' weights are normalized to sum 1 -> total 2
    np.testing.assert_allclose(ds.w.sum(), 2.0)
    # numerator events labeled 1, denominator 0
    assert ds.s.sum() == N_EVENTS

    x, y, w = ds[0]
    assert x.shape == (N_FEATURES,)
    assert x.dtype == y.dtype == w.dtype
    assert y.item() in (0.0, 1.0)


def test_dataset_return_kin():
    (X_num, w_num), (X_den, w_den) = _make_event_pair()

    ds = BalancedDataset(X_num, w_num, X_den, w_den, random_state=0, return_kin=True)
    item = ds[0]
    assert len(item) == 4


def test_dataset_applies_scaler():
    (X_num, w_num), (X_den, w_den) = _make_event_pair()
    scaler = StandardScaler().fit(np.concatenate([X_num, X_den]))

    ds = BalancedDataset(X_num, w_num, X_den, w_den, scaler=scaler, random_state=0)

    np.testing.assert_allclose(ds.X.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(ds.X.std(axis=0), 1.0, atol=1e-10)


def test_prepare_data_writes_split_pickles(tmp_path):
    dm = _make_datamodule(tmp_path, **SPLIT_4WAY)
    dm.prepare_data()
    n_train = _split_counts(SPLIT_4WAY)[0]

    expected = ["scaler.pkl"] + [
        f"events_{side}_{split}.pkl"
        for side in ("numerator", "denominator")
        for split in ("train", "val", "test", "wi_fit")
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name

    with open(tmp_path / "events_numerator_train.pkl", "rb") as f:
        X_tr, w_tr = pickle.load(f)
    assert X_tr.shape == (n_train, N_FEATURES)
    assert w_tr.shape == (n_train,)


def test_prepare_data_without_wifit_writes_no_wifit_pickles(tmp_path):
    dm = _make_datamodule(tmp_path)
    dm.prepare_data()

    assert not (tmp_path / "events_numerator_wi_fit.pkl").exists()
    assert not (tmp_path / "events_denominator_wi_fit.pkl").exists()


def test_setup_fit_builds_datasets(tmp_path):
    dm = _make_datamodule(tmp_path)
    dm.prepare_data()
    dm.setup("fit")
    n_train, n_val, _, _ = _split_counts(SPLIT_3WAY)

    # each dataset combines both hypotheses' events
    assert len(dm.training_data) == 2 * n_train
    assert len(dm.validation_data) == 2 * n_val

    batch = next(iter(dm.train_dataloader()))
    x, y, w = batch
    assert x.shape == (BATCH_SIZE, N_FEATURES)
    assert y.shape == (BATCH_SIZE,)
    assert w.shape == (BATCH_SIZE,)


def test_setup_test_loads_the_fitted_scaler(tmp_path):
    # a `stage=test` run is a fresh process: the pickles already exist, so setup() skips
    # prepare_data() and nothing fits this instance's scaler -- setup("test") must load the
    # saved one, or scaler.transform raises NotFittedError
    _make_datamodule(tmp_path).prepare_data()

    dm = _make_datamodule(tmp_path)
    dm.setup("test")
    n_test = _split_counts(SPLIT_3WAY)[3]

    assert len(dm.testing_data) == 2 * n_test

    with open(tmp_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(tmp_path / "events_numerator_test.pkl", "rb") as f:
        X_num_test, _ = pickle.load(f)
    with open(tmp_path / "events_denominator_test.pkl", "rb") as f:
        X_den_test, _ = pickle.load(f)
    expected = scaler.transform(np.concatenate([X_num_test, X_den_test]))
    # BalancedDataset shuffles the concatenated rows, so compare the value sets column-wise
    np.testing.assert_allclose(np.sort(dm.testing_data.X, axis=0), np.sort(expected, axis=0))


def test_bootstrap_resamples_training_but_not_validation(tmp_path):
    dm_plain = _make_datamodule(tmp_path, bootstrap=False)
    dm_plain.prepare_data()
    dm_plain.setup("fit")

    dm_boot = _make_datamodule(tmp_path, bootstrap=True)
    dm_boot.setup("fit")  # reuses the pickles dm_plain prepared

    n_combined_train = 2 * _split_counts(SPLIT_3WAY)[0]
    # plain training data keeps every (unique) input row exactly once
    assert len(np.unique(dm_plain.training_data.X, axis=0)) == n_combined_train
    # bootstrap draws with replacement -> duplicates, so fewer unique rows
    assert len(np.unique(dm_boot.training_data.X, axis=0)) < n_combined_train
    assert len(dm_boot.training_data) == n_combined_train

    # validation is untouched by bootstrap and identical across members
    np.testing.assert_allclose(dm_boot.validation_data.X, dm_plain.validation_data.X)


def test_bootstrap_is_seeded_by_random_state(tmp_path):
    dm_prep = _make_datamodule(tmp_path)
    dm_prep.prepare_data()

    def training_X(random_state):
        dm = _make_datamodule(tmp_path, bootstrap=True, random_state=random_state)
        dm.setup("fit")
        return dm.training_data.X

    np.testing.assert_array_equal(training_X(7), training_X(7))
    assert (training_X(7) != training_X(8)).any()
