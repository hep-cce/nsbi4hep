"""Fit wifi (w_i f_i) ensemble weights on the held-out ``wi_fit`` split.

Based on https://github.com/ml4fp/2025-lbnl/blob/main/sessions/day2/ensembling-tutorial/ensembling.ipynb

This is the second phase of wifi ensembling (arXiv:2506.00113), run after the members are
trained by ``main_ensemble_function`` (``do_ensemble_train: true``). It combines the M trained
members into a single log-ratio estimator

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

import numpy as np
import torch
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torchmin import minimize

from nsbi.tools.predict import build_model_from_checkpoint
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
    # Contract over events directly rather than forming the (M+1, M+1, N) outer-product stack and
    # summing it: same result, O(M^2) memory instead of O(M^2 N)
    c_d = torch.exp(d_outs) * d_weights
    c_n = torch.exp(-n_outs) * n_weights
    return (log_r_d * c_d) @ log_r_d.T + (log_r_n * c_n) @ log_r_n.T


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


def _fit_torchmin(w_init, log_r_n, log_r_d, n_weights, d_weights, method, options):
    """Fit the ensemble weights with a pytorch-minimize solver.

    Returns the fitted weights (detached).
    """
    log.info("torchmin method={} options={}", method, options)
    result = minimize(
        lambda w: symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights),
        w_init,
        method=method,
        options=options,
    )
    success = getattr(result, "success", None)
    log.info(
        "torchmin {}: success={} n_iter={} final loss={:.8f}",
        method,
        "?" if success is None else success,
        getattr(result, "nit", "?"),
        float(result.fun),
    )
    # A failed solve may not necessarily mean a bad fit
    w_fit = result.x.detach()
    grad = torch.autograd.functional.jacobian(
        lambda w: symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights),
        w_fit,
    )
    log.info("Gradient norm at solution: {:.3e}", float(grad.norm()))
    if success is False:
        log.warning(
            "torchmin {} did not converge (status={!r}: {}). Using the last iterate; check the "
            "gradient norm above and |dw|max below before trusting the weights.",
            method,
            getattr(result, "status", "?"),
            getattr(result, "message", "?"),
        )
    return w_fit


# Termination-tolerance options across the torchmin solver families. The tolerance check tightens
# only the ones the user actually set in ensemble.fit.options
TOLERANCE_KEYS = ("gtol", "xtol", "ftol", "tol")

# Parameters smaller than this fraction of the largest weight are excluded from the percent-change
_PCT_MAGNITUDE_FLOOR = 1e-6


def _check_fit_tolerances(
    w: torch.Tensor,
    w_init: torch.Tensor,
    log_r_n: torch.Tensor,
    log_r_d: torch.Tensor,
    n_weights: torch.Tensor,
    d_weights: torch.Tensor,
    method: str,
    options: dict,
    factor: float,
    pct_threshold: float,
) -> None:
    """Re-fit with tighter tolerances and report whether the weights changed significantly.

    Diagnostic for whether ``ensemble.fit.options``' termination tolerances are tight enough for
    this objective: divides every tolerance the user set by ``factor`` and solves again from the
    same init. Restarting from ``w_init`` rather than from ``w`` is deliberate -- continuing from
    the fitted point would just resume the same descent and under-report the difference, whereas a
    clean re-run tests where the stopping rule actually lands.

    The verdict is the largest per-parameter change relative to the configured fit,
    ``100 * |w_tight_i - w_i| / |w_i|``, over the parameters whose magnitude is at least
    ``_PCT_MAGNITUDE_FLOOR`` of the largest weight. The configured fit is the denominator because
    it is the one being validated (and saved) -- the re-fit is diagnostic only and is discarded.

    Skipped with a warning if the user set no tolerance option at all, since there is then nothing
    to tighten.

    Args:
        w: Weights from the configured fit, shape (M + 1,).
        w_init: The init the configured fit started from, shape (M + 1,).
        log_r_n: Per-member log-ratios on numerator events, shape (M + 1, N_n).
        log_r_d: Per-member log-ratios on denominator events, shape (M + 1, N_d).
        n_weights: Numerator event weights, shape (N_n,).
        d_weights: Denominator event weights, shape (N_d,).
        method: torchmin solver used for the configured fit.
        options: The configured fit's options, forwarded with its tolerances tightened.
        factor: Divisor applied to each tolerance option (10 => 10x tighter).
        pct_threshold: Warn if any weight changes by more than this many percent.
    """
    tightened = {k: v / factor for k, v in options.items() if k in TOLERANCE_KEYS}
    if not tightened:
        log.warning(
            "Tolerance check requested but ensemble.fit.options sets none of {}; nothing to "
            "tighten, skipping. Set an explicit tolerance to enable the check.",
            TOLERANCE_KEYS,
        )
        return

    log.info("Tolerance check: re-fitting with {}x tighter {}", factor, tightened)
    w_tight = _fit_torchmin(
        w_init, log_r_n, log_r_d, n_weights, d_weights, method, {**options, **tightened}
    )

    with torch.no_grad():
        loss = symmetrized_mlc_loss(w, log_r_n, log_r_d, n_weights, d_weights).item()
        loss_tight = symmetrized_mlc_loss(w_tight, log_r_n, log_r_d, n_weights, d_weights).item()

    w_np = np.asarray(w.detach().cpu().numpy(), dtype=float)
    w_tight_np = np.asarray(w_tight.detach().cpu().numpy(), dtype=float)
    dw = np.abs(w_tight_np - w_np)
    scale = np.abs(w_np).max()

    # Scale-relative change and loss change: always well-defined, so they stay informative even for
    # the near-zero parameters the per-parameter percent below has to exclude.
    log.info(
        "Tolerance check: |dw|max={:.3e} ({:.3f}% of max|w|), loss delta={:.3e} ({:.3f}%)",
        dw.max(),
        100.0 * dw.max() / scale if scale > 0 else float("inf"),
        loss_tight - loss,
        100.0 * abs(loss_tight - loss) / abs(loss) if loss != 0 else float("inf"),
    )

    keep = np.abs(w_np) >= _PCT_MAGNITUDE_FLOOR * scale
    if not keep.any():
        log.warning("Tolerance check: every fitted weight is ~0; no relative change to compare.")
        return
    # -inf on the excluded parameters so they cannot win the argmax below.
    pct = np.full(w_np.shape, -np.inf)
    pct[keep] = 100.0 * dw[keep] / np.abs(w_np[keep])
    arg = int(np.argmax(pct))
    log.info(
        "Tolerance check: max relative change {:.3f}% at parameter {} ({} excluded as ~0)",
        pct[arg],
        arg,
        int((~keep).sum()),
    )
    if pct[arg] > pct_threshold:
        log.warning(
            "Tolerance check FAILED: parameter {} changed {:.3f}% (> {}%) when tolerances were "
            "tightened {}x -- the configured tolerances may not be tight enough for this "
            "objective. Check the two torchmin summaries above: a re-fit that stopped on max_iter "
            "rather than on tolerance is truncated, not disagreeing.",
            arg,
            pct[arg],
            pct_threshold,
            factor,
        )
        # Dump both fits side by side so the disagreement can be traced to specific members rather
        # than inferred from the single worst number. Only on failure -- a passing check has
        # nothing to debug.
        log.warning("Per-parameter comparison (configured fit -> {}x tighter re-fit):", factor)
        n_params = w_np.size
        for i in range(n_params):
            label = "const" if i == n_params - 1 else str(i)
            change = f"{pct[i]:.3f}%" if keep[i] else "excluded (~0)"
            flag = "  <-- over threshold" if keep[i] and pct[i] > pct_threshold else ""
            log.warning(
                "  w[{:>5}] {:+.9f} -> {:+.9f}  diff={:+.3e}  {}{}",
                label,
                w_np[i],
                w_tight_np[i],
                w_tight_np[i] - w_np[i],
                change,
                flag,
            )
    else:
        log.info(
            "Tolerance check passed: every weight changed < {}% under {}x tighter tolerances.",
            pct_threshold,
            factor,
        )


def _log_fit_summary(w: torch.Tensor, cov: torch.Tensor, size: int) -> None:
    """Log the fitted weights, uncertainties, and sanity checks.

    Args:
        w: Fitted weights, shape (M + 1,) -- M member weights plus the constant offset.
        cov: Asymptotic covariance of ``w`` from ``weight_covariance``, shape (M + 1, M + 1).
        size: Number of members M.
    """
    w_np = np.asarray(w.detach().cpu().numpy(), dtype=float)
    cov_np = np.asarray(cov.detach().cpu().numpy(), dtype=float)
    n = size + 1

    if not np.all(np.isfinite(w_np)):
        log.warning("Fitted weights contain non-finite values (NaN/Inf): {}", w_np)
    if not np.all(np.isfinite(cov_np)):
        log.warning("Weight covariance contains non-finite values (NaN/Inf).")

    diag = cov_np.diagonal() if cov_np.shape == (n, n) else np.full(n, np.nan)
    err = np.sqrt(np.clip(diag, 0.0, None))

    log.info("Fitted weights (+/- sqrt of covariance diagonal):")
    for i in range(size):
        log.info("  w[{:>2}] = {:+.6f} +/- {:.6f}", i, w_np[i], err[i])
    log.info("  w[const] = {:+.6f} +/- {:.6f}", w_np[size], err[size])

    member_w = w_np[:size]
    log.info(
        "Member weight summary: sum={:+.6f} mean={:+.6f} std={:.6f} min={:+.6f} max={:+.6f}",
        member_w.sum(),
        member_w.mean(),
        member_w.std(),
        member_w.min(),
        member_w.max(),
    )

    # The fit starts from uniform 1/M members with a zero offset. Weights still sitting there mean
    # the optimizer never took a step -- the fit ran but produced nothing.
    w_init = np.ones(n) / size
    w_init[-1] = 0.0
    if np.allclose(w_np, w_init, atol=1e-6):
        log.warning(
            "Fitted weights equal the optimizer init (uniform 1/M, offset 0) -- the optimizer did "
            "not move; treat this as a failed/no-op fit, not a real result."
        )
    elif np.allclose(member_w, 1.0 / size, atol=1e-6):
        log.warning(
            "Member weights are all ~1/M though the offset moved -- inspect the fit closely."
        )

    if cov_np.shape != (n, n):
        log.warning("Weight covariance shape {} != ({}, {}).", cov_np.shape, n, n)
        return


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
    method = fit_cfg.get("method", "l-bfgs")
    options: dict = {}
    user_options = fit_cfg.get("options", None)
    if user_options is not None:
        options.update(
            OmegaConf.to_container(user_options, resolve=True)
            if OmegaConf.is_config(user_options)
            else dict(user_options)
        )
    # eps guards log[s/(1 - s)] against s -> 1
    eps = fit_cfg.get("eps", 1e-7)
    rescale_fit_weights = fit_cfg.get("rescale_fit_weights", False)
    # Opt-in diagnostic: re-fit with tighter tolerances and compare (see _check_fit_tolerances).
    tolerance_check = fit_cfg.get("tolerance_check", False)
    tolerance_check_factor = fit_cfg.get("tolerance_check_factor", 10.0)
    tolerance_check_pct = fit_cfg.get("tolerance_check_pct", 5.0)
    fit_dtype = getattr(torch, str(fit_cfg.get("dtype", "float64")))

    data_dir = cfg.datamodule.get("data_dir", "./")
    batch_size = cfg.datamodule.get("batch_size", 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_dir = resolve_ensemble_dir(ensemble_cfg)
    log.info(
        "Fitting wifi weights for {} members under {} (device: {})", size, ensemble_dir, device
    )

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
        # Built from each checkpoint's stored hyperparameters, not from cfg.model
        model = build_model_from_checkpoint(cfg, ckpt, warn_on_mismatch=(i == 0))

        # Cast to the fit dtype
        s_n = _predict_member(model, X_n, batch_size, device).to(fit_dtype)
        s_d = _predict_member(model, X_d, batch_size, device).to(fit_dtype)
        log_r_n_rows.append(torch.log(s_n / (1 - s_n + eps)))
        log_r_d_rows.append(torch.log(s_d / (1 - s_d + eps)))

    # Stack members into (M, N) and append the all-ones constant row -> (M + 1, N). The trailing
    # weight w[-1] then acts as an overall additive offset on log r.
    log_r_n = torch.ones((size + 1, log_r_n_rows[0].shape[0]), device=device, dtype=fit_dtype)
    log_r_n[:-1, :] = torch.stack(log_r_n_rows, dim=0)
    log_r_d = torch.ones((size + 1, log_r_d_rows[0].shape[0]), device=device, dtype=fit_dtype)
    log_r_d[:-1, :] = torch.stack(log_r_d_rows, dim=0)

    n_weights = torch.as_tensor(np.asarray(w_n), dtype=fit_dtype, device=device)
    d_weights = torch.as_tensor(np.asarray(w_d), dtype=fit_dtype, device=device)

    if rescale_fit_weights:
        weight_scale = torch.cat([n_weights, d_weights]).mean()
        n_weights = n_weights / weight_scale
        d_weights = d_weights / weight_scale
        log.info(
            "Rescaled wi_fit weights by 1/{:.3e} (common factor; leaves w* unchanged)",
            weight_scale.item(),
        )

    # Start L-BFGS from uniform 1/M member weights with the constant offset at 0,
    # then descend on the symmetrized MLC loss.
    w_init = torch.ones(size + 1, device=device, dtype=fit_dtype) / size
    w_init[-1] = 0.0

    # Snapshot the init loss so we can report how far the fit actually moved the weights: if |Δw|max
    # and the loss decrease are both ~0, the members' uniform average was already (near-)optimal and
    # the fit had nothing to do
    with torch.no_grad():
        init_loss = symmetrized_mlc_loss(w_init, log_r_n, log_r_d, n_weights, d_weights)
    log.info("Initial symmetrized MLC loss: {:.6f}", init_loss.item())

    w = _fit_torchmin(w_init, log_r_n, log_r_d, n_weights, d_weights, method, options)

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

    _log_fit_summary(w, cov, size)

    # Diagnostic only, and after everything the saved result depends on: w above stays what gets
    # written to weights.pkl regardless of what the tighter re-fit finds.
    if tolerance_check:
        _check_fit_tolerances(
            w,
            w_init,
            log_r_n,
            log_r_d,
            n_weights,
            d_weights,
            method,
            options,
            tolerance_check_factor,
            tolerance_check_pct,
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
