import numpy as np
import pytest
import torch

from nsbi.tools.metrics import ReweightingClosureMetric


def _w(*values):
    """Weights must be torch tensors: __call__ calls .numpy() on them unconditionally."""
    return torch.tensor(values, dtype=torch.float64)


def test_keys_follow_observable_order():
    """Callers pair values with feature_names by position, so insertion order is a contract."""
    kin = np.array([[0.5, 1.5], [1.5, 0.5]])
    metric = ReweightingClosureMetric([0, 1], [(2, 0.0, 2.0)] * 2, metric="chi2")

    out = metric(kin, _w(1.0, 1.0), _w(1.0, 1.0), _w(1.0, 1.0))

    assert list(out) == ["obs_0_chi2", "obs_1_chi2"]


def test_accepts_tensor_kin():
    kin = torch.tensor([[0.5], [1.5]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="chi2")

    assert "obs_0_chi2" in metric(kin, _w(1.0, 1.0), _w(1.0, 1.0), _w(1.0, 1.0))


def test_chi2_zero_for_perfect_closure():
    kin = np.array([[0.5], [1.5]])
    w = _w(1.0, 2.0)
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="chi2")

    out = metric(kin, w, w, _w(1.0, 1.0))

    assert out["obs_0_chi2"] == pytest.approx(0.0, abs=1e-9)


def test_chi2_hand_computed():
    kin = np.array([[0.5], [1.5]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="chi2")

    out = metric(kin, _w(3.0, 1.0), _w(1.0, 1.0), _w(1.0, 1.0))

    # base = [1, 1] (norm 2); truth = [1, 1]; pred = [3, 1] renormalized to [1.5, 0.5]
    # chi2 = (1.5 - 1)^2 / 1 + (0.5 - 1)^2 / 1 = 0.5
    assert out["obs_0_chi2"] == pytest.approx(0.5, rel=1e-5)


def test_chi2_ignores_overall_normalization_of_prediction():
    """hist_pred is rescaled to hist_base.sum(), so this measures shape closure only."""
    kin = np.array([[0.5], [1.5]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="chi2")

    unscaled = metric(kin, _w(3.0, 1.0), _w(1.0, 1.0), _w(1.0, 1.0))
    scaled = metric(kin, _w(30.0, 10.0), _w(1.0, 1.0), _w(1.0, 1.0))

    assert scaled["obs_0_chi2"] == pytest.approx(unscaled["obs_0_chi2"], rel=1e-6)


def test_explicit_binning_drops_out_of_range_events():
    kin = np.array([[0.5], [1.5], [5.0]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="chi2")

    out = metric(kin, _w(3.0, 1.0, 7.0), _w(1.0, 1.0, 7.0), _w(1.0, 1.0, 7.0))

    # the event at 5.0 falls outside [0, 2] and contributes to no histogram
    assert out["obs_0_chi2"] == pytest.approx(0.5, rel=1e-5)


def test_auto_binning_is_finite_and_closes():
    """Both live call sites construct this with binning=None."""
    rng = np.random.default_rng(0)
    kin = rng.normal(size=(500, 2))
    w = torch.ones(500, dtype=torch.float64)
    metric = ReweightingClosureMetric([0, 1], binning=None, metric="chi2")

    out = metric(kin, w, w, w)

    assert list(out) == ["obs_0_chi2", "obs_1_chi2"]
    assert all(np.isfinite(v) for v in out.values())
    assert out["obs_0_chi2"] == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_zero_for_perfect_closure():
    kin = np.array([[0.5], [1.5]])
    w = _w(1.0, 2.0)
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="wasserstein")

    out = metric(kin, w, w, _w(1.0, 1.0))

    assert out["obs_0_wasserstein"] == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_hand_computed():
    kin = np.array([[0.5], [1.5]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="wasserstein")

    out = metric(kin, _w(1.0, 0.0), _w(0.0, 1.0), _w(1.0, 1.0))

    # all predicted mass at centre 0.5, all truth mass at centre 1.5
    assert out["obs_0_wasserstein"] == pytest.approx(1.0, rel=1e-5)


def test_unsupported_metric_raises_at_call_time_not_construction():
    """A config typo surfaces mid-epoch, not at startup -- pin that so it is a known cost."""
    kin = np.array([[0.5], [1.5]])
    metric = ReweightingClosureMetric([0], [(2, 0.0, 2.0)], metric="kl")

    with pytest.raises(ValueError):
        metric(kin, _w(1.0, 1.0), _w(1.0, 1.0), _w(1.0, 1.0))
