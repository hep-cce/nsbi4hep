"""Fit wifi (w_i f_i) ensemble weights on the held-out ``wi_fit`` split.

Based on https://github.com/ml4fp/2025-lbnl/blob/main/sessions/day2/ensembling-tutorial/ensembling.ipynb

This is the second phase of wifi ensembling (arXiv:2506.00113), run after the members are
trained by ``main_ensemble_function`` (``do_ensemble_train: true``). It combines the M trained members
into a single log-ratio estimator

    log r_S(x) = sum_i w_i f_i(x) + w_const,

where ``f_i(x) = log[s_i/(1 - s_i)]`` is member i's estimate of ``log r_S`` (``s_i`` is its CARL
sigmoid output) and ``w_const`` is a learned overall offset. 
The weights ``w`` are fit by minimizing the symmetrized
MLC loss on the held-out ``wi_fit`` split -- data independent of member
training, as the fit's asymptotics assume.

The output ``weights.pkl`` holds the fitted ``w``, ready for the analysis phase (the mu
test-statistic adjustment) which lives with the physics analysis, not here.
"""

import json
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger as log
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from nsbi.utils.lightning_utils import find_latest_checkpoint

# Manifest written by the training phase mapping member index -> its best checkpoint path. Both
# training paths (single-device and distributed) write it to <ensemble_dir>/MANIFEST_NAME so the
# fit can locate members without knowing each path's on-disk checkpoint layout.
MANIFEST_NAME = "members.json"


def symmetrized_mlc_loss(
    w: torch.Tensor,
    log_r_n: torch.Tensor,
    log_r_d: torch.Tensor,
    n_weights: torch.Tensor,
    d_weights: torch.Tensor,
) -> torch.Tensor:
    """Symmetrized MLC loss for fitting wifi ensemble weights (arXiv:2506.00113).

    The MLC loss of arXiv:1806.02350 with the extra terms that restore symmetry under swapping the
    numerator/denominator distributions and sending ``log r -> -log r``; it is minimized by the
    true ``log r_S``. ``w @ log_r_n`` is the ensemble's combined ``log r`` on the numerator events
    (rows of ``log_r_n`` are the per-member outputs plus a constant row), and likewise for the
    denominator.

    Args:
        w: Ensemble weights, shape (M + 1,) -- one per member plus a trailing constant weight.
        log_r_n: Per-member log-ratios on numerator events, shape (M + 1, N_n) with a final
            all-ones row for the constant offset.
        log_r_d: Per-member log-ratios on denominator events, shape (M + 1, N_d).
        n_weights: Numerator event weights, shape (N_n,).
        d_weights: Denominator event weights, shape (N_d,).

    Returns:
        Scalar loss tensor.
    """
    n_outs = w @ log_r_n
    d_outs = w @ log_r_d
    return (
        -(n_outs * n_weights).mean()
        + (d_weights * (torch.exp(d_outs) - 1)).mean()
        + (d_weights * d_outs).mean()
        + (n_weights * (torch.exp(-n_outs) - 1)).mean()
    )


def _symmetrized_mlc_hessian(w, log_r_n, log_r_d, n_weights, d_weights):
    """Hessian of the symmetrized MLC loss w.r.t. the weights, shape (M + 1, M + 1).

    Second derivative matrix used to build the sandwich covariance of the fitted ``w`` (the
    ``V`` in ``C = V U V`` of arXiv:2506.00113, App. B).
    """
    n_outs = w @ log_r_n
    d_outs = w @ log_r_d
    return (
        torch.einsum("ia,ja->ija", log_r_d, log_r_d) * torch.exp(d_outs) * d_weights
        + torch.einsum("ia,ja->ija", log_r_n, log_r_n) * torch.exp(-n_outs) * n_weights
    ).sum(-1)


def _symmetrized_mlc_grad_n(w, log_r_n, n_weights):
    """Per-numerator-event score (gradient of the loss' numerator terms), shape (M + 1, N_n)."""
    n_outs = w @ log_r_n
    return (-log_r_n - torch.exp(-n_outs) * log_r_n) * n_weights


def _symmetrized_mlc_grad_d(w, log_r_d, d_weights):
    """Per-denominator-event score (gradient of the loss' denominator terms), shape (M + 1, N_d)."""
    d_outs = w @ log_r_d
    return (log_r_d + torch.exp(d_outs) * log_r_d) * d_weights


def weight_covariance(w, log_r_n, log_r_d, n_weights, d_weights) -> torch.Tensor:
    """Asymptotic covariance of the fitted wifi weights (arXiv:2506.00113, App. B).

    Sandwich estimator ``C = V U V`` where ``V`` is the inverse Hessian of the symmetrized MLC
    loss at the fit and ``U`` is the empirical covariance of the per-event score. This quantifies
    the density-ratio-estimation uncertainty; the downstream mu analysis propagates it into the
    test statistic. Computed in double precision (the inverse/products are ill-conditioned) and
    returned double. Shape (M + 1, M + 1).
    """
    v = torch.linalg.inv(
        _symmetrized_mlc_hessian(w, log_r_n, log_r_d, n_weights, d_weights).double()
    )
    grad_n = _symmetrized_mlc_grad_n(w, log_r_n, n_weights).double()
    grad_d = _symmetrized_mlc_grad_d(w, log_r_d, d_weights).double()
    score_var = grad_n @ grad_n.T + grad_d @ grad_d.T
    return v @ score_var @ v


def _fit_torch_lbfgs(w_init, log_r_n, log_r_d, n_weights, d_weights, max_iter):
    """Fit the ensemble weights with stock ``torch.optim.LBFGS``.

    ``tolerance_grad``/``tolerance_change`` default to 1e-7/1e-9 and are absolute; with the
    small-magnitude objective from raw ~O(1e-8) MCFM event weights they can falsely signal
    convergence after a single step. Zero them and rely on ``max_iter`` + the strong-Wolfe line
    search to decide when the fit is done. Returns the fitted weights (detached).
    """
    w = w_init.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [w],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )
    step = {"i": 0}

    def closure():
        optimizer.zero_grad()
        loss = symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights)
        loss.backward()
        step["i"] += 1
        log.info(
            "L-BFGS iter {}: loss={:.8f} |grad|max={:.3e}",
            step["i"],
            loss.item(),
            w.grad.abs().max().item(),
        )
        return loss

    optimizer.step(closure)
    return w.detach()


def _fit_torchmin(w_init, log_r_n, log_r_d, n_weights, d_weights, max_iter):
    """Fit the ensemble weights with pytorch-minimize's L-BFGS.

    ``torchmin.minimize(method="l-bfgs")`` uses a scale-robust termination test plus a strong-Wolfe
    line search, so unlike stock ``torch.optim.LBFGS`` it converges on the raw ~O(1e-8) MCFM weight
    scale with no tolerance or weight-rescale fixes -- this is the path the ``ensembling.ipynb``
    tutorial uses. Requires the ``pytorch-minimize`` package (imported lazily so the default
    ``torch_lbfgs`` path carries no dependency on it). Returns the fitted weights (detached).
    """
    from torchmin import minimize

    result = minimize(
        lambda w: symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights),
        w_init,
        method="l-bfgs",
        options={"max_iter": max_iter, "disp": False},
    )
    log.info(
        "torchmin l-bfgs: success={} n_iter={} final loss={:.8f}",
        getattr(result, "success", "?"),
        getattr(result, "nit", "?"),
        float(result.fun),
    )
    return result.x.detach()


def _predict_member(model, X: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    """Run a trained member over (already-scaled) features and return its sigmoid outputs ``s``.

    Mirrors ``CARL.predict_step`` (the sigmoid output), batched to avoid holding the whole
    ``wi_fit`` split on the GPU at once. Outputs are kept on ``device`` so the downstream fit runs
    there too.
    """
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    model.eval()
    model.to(device)
    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            outs.append(model(xb.to(device)).flatten())
    return torch.cat(outs)


def resolve_ensemble_dir(ensemble_cfg: DictConfig) -> Path:
    """Locate the canonical ``<storage_path>/ensemble`` directory for this ensemble.

    Shared by both training paths and the fit so they agree on one location for the member
    manifest (and, for the single-device path, the member checkpoints themselves). Defaults to
    Ray's ``~/ray_results`` when ``storage_path`` is unset -- the same default both Ray Tune and
    Ray Train use, so training and the fit stay in lockstep.
    """
    storage_path = ensemble_cfg.get("storage_path", None)
    root = Path(storage_path) if storage_path is not None else Path.home() / "ray_results"
    return root / "ensemble"


def write_member_manifest(ensemble_dir: Path, member_ckpts: dict[int, str]) -> Path:
    """Write the member-index -> best-checkpoint-path manifest for the fit to read.

    Called by each training path once all members have finished, recording wherever that path
    stored each member's best checkpoint (a Lightning ``ModelCheckpoint`` dir for the single-device
    path, a Ray Train checkpoint dir for the distributed path). Paths are absolute so the fit needs
    no knowledge of either layout.

    Args:
        ensemble_dir: Canonical ensemble directory (from ``resolve_ensemble_dir``).
        member_ckpts: Mapping of member index to the absolute path of its best ``.ckpt`` file.

    Returns:
        The path the manifest was written to.
    """
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ensemble_dir / MANIFEST_NAME
    # Keys are stringified for JSON; the fit reads them back by member index.
    payload = {"members": {str(i): str(p) for i, p in member_ckpts.items()}}
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote member manifest ({} members) to {}", len(member_ckpts), manifest_path)
    return manifest_path


def load_member_ckpts(ensemble_dir: Path, size: int) -> list[str]:
    """Resolve the ``size`` member checkpoint paths, preferring the manifest.

    Uses ``<ensemble_dir>/members.json`` when present (the layout-agnostic path written by
    training). Falls back to globbing ``member_{i}/checkpoints`` with ``find_latest_checkpoint``
    for single-device ensembles trained before manifests existed. Raises if any member is missing.

    Args:
        ensemble_dir: Canonical ensemble directory (from ``resolve_ensemble_dir``).
        size: Number of members M expected (indices ``0..M-1``).

    Returns:
        Member checkpoint paths ordered by member index.
    """
    manifest_path = ensemble_dir / MANIFEST_NAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            members = json.load(f)["members"]
        ckpts = []
        for i in range(size):
            path = members.get(str(i))
            if path is None or not Path(path).exists():
                raise FileNotFoundError(
                    f"Member {i} is missing or its checkpoint {path!r} (from {manifest_path}) does "
                    "not exist. Was training completed for every member?"
                )
            ckpts.append(path)
        return ckpts

    # Legacy fallback: single-device layout, <ensemble_dir>/member_{i}/checkpoints/*.ckpt.
    log.info("No manifest at {}; falling back to per-member checkpoint globbing.", manifest_path)
    ckpts = []
    for i in range(size):
        ckpt_dir = ensemble_dir / f"member_{i}" / "checkpoints"
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found for member {i} under {ckpt_dir}, and no manifest at "
                f"{manifest_path}. Was training completed for every member?"
            )
        ckpts.append(str(ckpt))
    return ckpts


def main_ensemble_fit(cfg: DictConfig) -> None:
    """Fit the wifi ensemble weights on the held-out ``wi_fit`` split and save them.

    Loads each trained member's checkpoint, evaluates it on the ``wi_fit`` numerator/denominator
    events, builds the per-member log-ratio matrices (plus a constant row), and minimizes the
    symmetrized MLC loss with full-batch L-BFGS. Everything (predictions, log-ratio matrices,
    weights, the fit itself) runs on ``device`` -- the GPU when available -- since each L-BFGS
    iteration sweeps every ``wi_fit`` event. Writes the fitted ``w`` to ``weights.pkl`` under the
    ensemble directory.

    Requires that training was run with ``datamodule.wi_fit_size > 0`` so the ``wi_fit`` pickles
    exist. Members are selected per directory with ``find_latest_checkpoint`` -- the same selector
    ``main_function`` uses for the ``test``/``predict`` stages.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    ensemble_cfg = cfg.get("ensemble", {})
    size = ensemble_cfg.get("size", 16)
    fit_cfg = ensemble_cfg.get("fit", {})
    max_iter = fit_cfg.get("max_iter", 1000)
    # eps guards log[s/(1 - s)] against s -> 1
    eps = fit_cfg.get("eps", 1e-7)

    data_dir = cfg.datamodule.get("data_dir", "./")
    batch_size = cfg.datamodule.get("batch_size", 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_dir = resolve_ensemble_dir(ensemble_cfg)
    log.info("Fitting wifi weights for {} members under {} (device: {})", size, ensemble_dir, device)

    # Load the shared scaler and the held-out wi_fit split reserved in training. Members were
    # trained on scaler-transformed features, so the wi_fit features must be transformed too.
    scaler_path = Path(data_dir) / "scaler.pkl"
    n_path = Path(data_dir) / "events_numerator_wi_fit.pkl"
    d_path = Path(data_dir) / "events_denominator_wi_fit.pkl"
    for p in (scaler_path, n_path, d_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file {p} not found. Run training (do_ensemble_train: true) with "
                "datamodule.wi_fit_size > 0 to reserve the wi_fit split before fitting weights."
            )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(n_path, "rb") as f:
        X_n, w_n = pickle.load(f)
    with open(d_path, "rb") as f:
        X_d, w_d = pickle.load(f)

    X_n = scaler.transform(X_n)
    X_d = scaler.transform(X_d)

    # Resolve every member's checkpoint via the manifest training wrote (layout-agnostic across the
    # single-device and distributed paths), then evaluate on the wi_fit split. r = s/(1-s+eps).
    member_ckpts = load_member_ckpts(ensemble_dir, size)
    log_r_n_rows = []
    log_r_d_rows = []
    for i, ckpt in enumerate(member_ckpts):
        log.info("Member {}: loading {}", i, ckpt)
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

        s_n = _predict_member(model, X_n, batch_size, device)
        s_d = _predict_member(model, X_d, batch_size, device)
        log_r_n_rows.append(torch.log(s_n / (1 - s_n + eps)))
        log_r_d_rows.append(torch.log(s_d / (1 - s_d + eps)))

    # Stack members into (M, N) and append the all-ones constant row -> (M + 1, N). The trailing
    # weight w[-1] then acts as an overall additive offset on log r.
    log_r_n = torch.ones((size + 1, log_r_n_rows[0].shape[0]), device=device)
    log_r_n[:-1, :] = torch.stack(log_r_n_rows, dim=0)
    log_r_d = torch.ones((size + 1, log_r_d_rows[0].shape[0]), device=device)
    log_r_d[:-1, :] = torch.stack(log_r_d_rows, dim=0)

    n_weights = torch.as_tensor(np.asarray(w_n), dtype=torch.float32, device=device)
    d_weights = torch.as_tensor(np.asarray(w_d), dtype=torch.float32, device=device)

    rescale_weights = False
    if rescale_weights:
        weight_scale = torch.cat([n_weights, d_weights]).mean()
        n_weights = n_weights / weight_scale
        d_weights = d_weights / weight_scale
        log.info(
            "Rescaled wi_fit weights by 1/{:.3e} (common factor; leaves w* unchanged)",
            weight_scale.item(),
        )

    # Start L-BFGS from uniform 1/M member weights with the constant offset at 0, 
    # then descend on the symmetrized MLC loss.
    w_init = torch.ones(size + 1, device=device) / size
    w_init[-1] = 0.0

    # Snapshot the init loss so we can report how far the fit actually moved the weights: if |Δw|max
    # and the loss decrease are both ~0, the members' uniform average was already (near-)optimal and
    # the fit had nothing to do
    with torch.no_grad():
        init_loss = symmetrized_mlc_loss(w_init, log_r_n, log_r_d, n_weights, d_weights)
    log.info("Initial symmetrized MLC loss: {:.6f}", init_loss.item())

    # Which L-BFGS backend fits the weights
    optimizer_backend = "torchmin"
    if optimizer_backend == "torchmin":
        w = _fit_torchmin(w_init, log_r_n, log_r_d, n_weights, d_weights, max_iter)
    elif optimizer_backend == "torch_lbfgs":
        w = _fit_torch_lbfgs(w_init, log_r_n, log_r_d, n_weights, d_weights, max_iter)
    else:
        raise ValueError(f"Unknown fit optimizer_backend {optimizer_backend!r}")

    with torch.no_grad():
        final_loss = symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights)
        max_dw = (w - w_init).abs().max().item()
    log.info("Final symmetrized MLC loss: {:.6f}", final_loss.item())
    log.info(
        "Fit moved weights by |Δw|max={:.3e} (loss Δ={:.3e})",
        max_dw,
        (final_loss - init_loss).item(),
    )

    # Asymptotic covariance of the fitted weights (V U V), on the same wi_fit log-ratio matrices
    # the fit used. It can only be computed here, where the wi_fit split is in scope.
    with torch.no_grad():
        cov = weight_covariance(w, log_r_n, log_r_d, n_weights, d_weights)
    log.info(
        "Weight covariance: max|C|={:.3e}, trace={:.3e}",
        cov.abs().max().item(),
        cov.diag().sum().item(),
    )

    w_fitted = w.detach().cpu().numpy()
    out_path = ensemble_dir / "weights.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "w": w_fitted,  # (M + 1,): M member weights + trailing constant offset
                "cov": cov.detach().cpu().numpy(),  # (M + 1, M + 1): asymptotic covariance of w
                "size": size,  # M; member i corresponds to w[i]
                "eps": eps,  # reuse when reconstructing r = s / (1 - s + eps) downstream
            },
            f,
        )
    log.info("Saved ensemble weights + covariance to {}", out_path)
