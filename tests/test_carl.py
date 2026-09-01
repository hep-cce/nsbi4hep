import pytest
import torch
import torch.nn.functional as F

from nsbi.models.carl import CARL
from torch.optim.lr_scheduler import ReduceLROnPlateau

N_FEATURES = 3


def _make_model(device=torch.device("cpu")):
    torch.manual_seed(0)
    return CARL(n_features=N_FEATURES, n_layers=2, n_nodes=8, learning_rate=1e-3).to(device)


def _make_batch(device=torch.device("cpu"), n=20):
    torch.manual_seed(1)
    x = torch.randn(n, N_FEATURES, device=device)
    y = torch.randint(0, 2, (n,), device=device).float()
    w = torch.rand(n, device=device) + 0.1
    return x, y, w


def test_forward_shape_and_sigmoid_range(device):
    model = _make_model(device)
    x, _, _ = _make_batch(device)

    out = model(x)

    assert out.shape == (20, 1)
    # sigmoid output: strictly inside (0, 1), required for log[s/(1-s)] downstream
    assert (out > 0).all() and (out < 1).all()


def test_training_step_is_weighted_bce(device):
    model = _make_model(device)
    x, y, w = _make_batch(device)

    loss = model.training_step((x, y, w), 0)

    with torch.no_grad():
        y_hat = model(x).flatten()
        expected = (F.binary_cross_entropy(y_hat, y, reduction="none") * w).sum() / w.sum()
    torch.testing.assert_close(loss.detach(), expected)


@pytest.mark.parametrize(
    ("step", "loss_key"), [("validation_step", "val_loss"), ("test_step", "test_loss")]
)
def test_eval_step_returns_outputs(device, step, loss_key):
    model = _make_model(device)
    x, y, w = _make_batch(device)

    out = getattr(model, step)((x, y, w), 0)

    assert set(out) == {loss_key, "y_hat", "y", "w", "kin"}
    assert out["y_hat"].shape == (20,)
    # eval outputs are moved to cpu for the closure-metrics callback
    torch.testing.assert_close(out["kin"], x.cpu())
    # same weighted BCE as training_step -- the two eval steps are copies of it
    with torch.no_grad():
        y_hat = model(x).flatten()
        expected = (F.binary_cross_entropy(y_hat, y, reduction="none") * w).sum() / w.sum()
    torch.testing.assert_close(out[loss_key].detach(), expected)


def test_predict_step_accepts_tensor_or_tuple_batch(device):
    model = _make_model(device)
    x, _, _ = _make_batch(device)

    from_tensor = model.predict_step(x, 0)
    from_tuple = model.predict_step((x,), 0)

    assert from_tensor.shape == (20,)
    torch.testing.assert_close(from_tensor, from_tuple)


def test_configure_optimizers_monitors_val_loss():
    model = _make_model()

    cfg = model.configure_optimizers()

    assert isinstance(cfg["optimizer"], torch.optim.NAdam)
    assert cfg["optimizer"].param_groups[0]["lr"] == 1e-3
    scheduler_cfg = cfg["lr_scheduler"]
    assert isinstance(scheduler_cfg["scheduler"], ReduceLROnPlateau)
    assert scheduler_cfg["monitor"] == "val_loss"


def test_hyperparameters_are_saved():
    model = _make_model()
    assert model.hparams["n_features"] == N_FEATURES
    assert model.hparams["learning_rate"] == 1e-3
