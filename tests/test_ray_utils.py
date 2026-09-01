import pytest

from nsbi.utils.ray_utils import _try_num, parse_dist
from ray.tune.search.sample import Categorical, Float, Integer


def test_try_num_int():
    assert _try_num("3") == 3
    assert isinstance(_try_num("3"), int)


def test_try_num_float():
    assert _try_num("3.5") == 3.5
    assert _try_num("1e-5") == 1e-5


def test_try_num_string_passthrough():
    assert _try_num("adam") == "adam"


def test_parse_randint():
    dist = parse_dist("randint:1,10")
    assert isinstance(dist, Integer)
    assert dist.lower == 1
    assert dist.upper == 10


def test_parse_qrandint_quantizes():
    dist = parse_dist("qrandint:0,10,2")
    assert isinstance(dist, Integer)
    samples = [dist.sample() for _ in range(50)]
    assert all(s % 2 == 0 for s in samples)
    assert all(0 <= s <= 10 for s in samples)


def test_parse_uniform():
    dist = parse_dist("uniform:0.1,0.9")
    assert isinstance(dist, Float)
    assert dist.lower == pytest.approx(0.1)
    assert dist.upper == pytest.approx(0.9)


def test_parse_loguniform_samples_in_range():
    dist = parse_dist("loguniform:1e-5,3e-2")
    assert isinstance(dist, Float)
    samples = [dist.sample() for _ in range(50)]
    assert all(1e-5 <= s <= 3e-2 for s in samples)


def test_parse_choice_mixed_types():
    dist = parse_dist("choice:adam,64,0.5")
    assert isinstance(dist, Categorical)
    assert list(dist.categories) == ["adam", 64, 0.5]


def test_parse_grid():
    assert parse_dist("grid:32,64,128") == {"grid_search": [32, 64, 128]}


def test_parse_strips_whitespace_and_empty_parts():
    dist = parse_dist("choice: a , ,b ")
    assert list(dist.categories) == ["a", "b"]


def test_parse_unsupported_kind_raises():
    with pytest.raises(ValueError, match="Unsupported distribution spec"):
        parse_dist("normal:0,1")
