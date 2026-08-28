import math

import pytest

from nsbi.tools.metrics import mean_wasserstein, total_reduced_chi2


def test_total_reduced_chi2():
    chi2_scores = {"a": 2.0, "b": 4.0}
    binning = [(3, 0.0, 1.0), (5, 0.0, 1.0)]  # dof = (3 - 1) + (5 - 1) = 6

    assert total_reduced_chi2(chi2_scores, binning) == pytest.approx(1.0)


def test_total_reduced_chi2_empty_is_nan():
    assert math.isnan(total_reduced_chi2({}, []))


def test_mean_wasserstein():
    assert mean_wasserstein({"a": 1.0, "b": 3.0}) == pytest.approx(2.0)
