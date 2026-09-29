#!/usr/bin/env python
"""Reproduce the mu confidence-interval plot from https://github.com/ml4fp/2025-lbnl/blob/main/sessions/day2/ensembling-tutorial/ensembling.ipynb.

Standalone downstream analysis: it takes an ensemble trained + fit by the framework
(do_ensemble_train + do_ensemble_fit -> member checkpoints + weights.pkl + scaler.pkl) and runs the
signal-strength (mu) test-statistic scan, then
saves the [0, 1]-zoomed -2 log lambda curve with the 1sigma/2sigma lines and the interpolated
interval bounds.

Ensemble size and every network's architecture are read from the artifacts themselves
(weights.pkl records the member count; Lightning checkpoints store the CARL hyperparameters),
so only paths and physics choices are passed on the command line.

Example:
    python3 mu_inference_example.py \
        --ensemble-dir /home/nkang/test_notebook/storage/ensemble \
        --ensemble-scaler /home/nkang/test_notebook/data/scaler.pkl \
        --sbi-ckpt /home/nkang/wifi_fit_bundle/sbi_base_model/epoch=76-train_loss=0.69.ckpt \
        --sbi-scaler /home/nkang/wifi_fit_bundle/sbi_base_model/scaler.pkl \
        --obs-csv /eagle/ScalingHEPAI/test_Nathan/wifi_data/obs_data/mu_x.csv \
        --xs-json /eagle/ScalingHEPAI/test_Nathan/wifi_data/xsecs/ggzz4l_xs.json
"""

import argparse
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset

from nsbi.models.carl import CARL

FEATURES = [
    "l1_pt",
    "l1_eta",
    "l1_phi",
    "l1_energy",
    "l2_pt",
    "l2_eta",
    "l2_phi",
    "l2_energy",
    "l3_pt",
    "l3_eta",
    "l3_phi",
    "l3_energy",
    "l4_pt",
    "l4_eta",
    "l4_phi",
    "l4_energy",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and hyperparameters for the mu scan."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Framework ensemble (trained + fit by do_ensemble_train/do_ensemble_fit).
    p.add_argument(
        "--ensemble-dir", required=True, help="<storage_path>/ensemble (members + weights.pkl)"
    )
    p.add_argument(
        "--ensemble-scaler", required=True, help="scaler.pkl the members were trained with"
    )
    # Separately trained SBI/bkg network (the r_SBI estimator; a bigger CARL) + its own scaler.
    p.add_argument("--sbi-ckpt", required=True, help="checkpoint for the SBI/bkg CARL network")
    p.add_argument("--sbi-scaler", required=True, help="scaler pickle for the SBI/bkg network")
    # Physics data.
    p.add_argument("--obs-csv", required=True, help="observed events CSV (features + column 'n')")
    p.add_argument("--xs-json", required=True, help="cross-section JSON (keys sig/int/sbi/bkg)")
    p.add_argument("--lumi", type=float, default=300.0, help="integrated luminosity [1/fb]")
    p.add_argument("--mu-max", type=float, default=4.0, help="upper edge of the mu scan grid")
    p.add_argument(
        "--mu-points", type=int, default=401, help="number of grid points in the mu scan"
    )
    p.add_argument("--out", default="mu_interval_framework.png", help="output plot path")
    p.add_argument(
        "--fixed-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude negative-density events once, for all mu, instead of letting "
        "nansum drop a different set at each mu (default: on)",
    )
    p.add_argument(
        "--mask-mu-range",
        default="0,1",
        metavar="LO,HI",
        help="mu range the fixed mask keeps pole-free (default 0,1)",
    )
    p.add_argument(
        "--list-masked",
        action="store_true",
        help="print each masked event's ratios and its negative-density mu window",
    )
    # The adjustment is evaluated at a single mu and is sensitive to it.
    p.add_argument(
        "--mu-hat",
        type=float,
        default=None,
        help="evaluate the adjustment here instead of at the refined MLE",
    )
    p.add_argument(
        "--scan-adjustment",
        nargs="?",
        const="0.20,0.35,31",
        default=None,
        metavar="LO,HI,N",
        help="tabulate the adjustment across a mu range",
    )
    return p.parse_args()


def load_scaler(path: str):
    """Unpickle a fitted sklearn StandardScaler and check it matches the FEATURES list.

    The scaler was fit on a bare array, so it stores only a feature count — the ordering of
    FEATURES is still an implicit contract with the training config; this catches count drift.
    """
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"{path} was fit on {scaler.n_features_in_} features, "
            f"but this script's FEATURES list has {len(FEATURES)}"
        )
    return scaler


def member_ckpts(ensemble_dir: str, size: int) -> list[str]:
    """Resolve the ``size`` member checkpoints, preferring the manifest, else newest-by-ctime.

    Mirrors the framework fit's ``load_member_ckpts``: use ``members.json`` when present (the
    same checkpoints the weight fit combined), otherwise glob ``member_{i}/checkpoints/*.ckpt``
    and take the most recent, matching ``find_latest_checkpoint``.
    """
    manifest = os.path.join(ensemble_dir, "members.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            members = json.load(f)["members"]
        return [members[str(i)] for i in range(size)]
    ckpts = []
    for i in range(size):
        matches = glob.glob(os.path.join(ensemble_dir, f"member_{i}", "checkpoints", "*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint for member {i} under {ensemble_dir}")
        ckpts.append(max(matches, key=os.path.getctime))
    return ckpts


def load_carl(ckpt_path: str) -> CARL:
    """Load a CARL checkpoint directly from its stored hyperparameters and state dict.

    Bypasses ``CARL.load_from_checkpoint`` because checkpoints trained under LightningCLI carry an
    ``_instantiator`` key in their hyperparameters, which makes Lightning route construction
    through jsonargparse (and crash when ``jsonargparse[signatures]`` isn't installed).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams.pop("_instantiator", None)
    model = CARL(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    return model


def predict(model: CARL, scaler, X: np.ndarray, batch_size: int = 1024) -> torch.Tensor:
    """Scale features with ``scaler`` and return the model's sigmoid outputs, batched."""
    dl = DataLoader(
        TensorDataset(torch.tensor(scaler.transform(X), dtype=torch.float32)),
        batch_size=batch_size,
    )
    model.eval()
    with torch.no_grad():
        return torch.cat([model(xb).flatten() for (xb,) in dl])


def refine_mu_hat(log_likelihood, w, mu: torch.Tensor, index: int) -> torch.Tensor:
    """Locate the shape-only MLE between the grid points bracketing ``index``.

    The adjustment is evaluated *at* ``mu_hat`` and can vary sharply near the minimum, so a bounded
    1-D minimization removes the grid quantization.

    Args:
        log_likelihood: Shape log-likelihood of ``(mu, w)``; ``t_shape = -2 * log_likelihood``.
        w: Fitted ensemble weights, held fixed.
        mu: The scan grid.
        index: ``argmin`` of the shape statistic over that grid.

    Returns:
        The refined ``mu_hat`` as a scalar tensor matching ``mu``'s dtype.
    """
    # A unimodal function sampled on a grid has its true minimum between the argmin's neighbours.
    lo = float(mu[max(index - 1, 0)])
    hi = float(mu[min(index + 1, len(mu) - 1)])
    if not hi > lo:
        # argmin sat on a grid edge, so there is no bracket to search.
        return mu[index]

    def nll(value: float) -> float:
        return -float(log_likelihood(torch.tensor(value, dtype=mu.dtype), w))

    result = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    return torch.tensor(float(result.x), dtype=mu.dtype)


def positive_density_mask(r_S, r_sbi, xs_sig, xs_sbi, xs_bkg, mu_lo: float, mu_hi: float):
    """Events whose predicted density stays positive across ``[mu_lo, mu_hi]``.

    The numerator of ``r_mu`` is, with ``u = sqrt(mu)``,

        N(u) = xs_sig*r_S*u^2 + (xs_sbi*r_sbi - xs_sig*r_S - xs_bkg)*u + xs_bkg

    and ``m_sig = mu - sqrt(mu)`` is negative for mu in (0, 1), so a signal-like enough event has a
    mu window where ``N`` goes negative, ``log`` returns NaN and ``nansum`` drops it -- a different
    event set at every mu.

    Args:
        r_S: Ensembled signal likelihood ratio per event.
        r_sbi: SBI/bkg ratio per event.
        xs_sig, xs_sbi, xs_bkg: Cross sections.
        mu_lo: Lower edge of the range to keep pole-free.
        mu_hi: Upper edge of it.

    Returns:
        Boolean mask of events to keep.
    """
    return density_minimum(r_S, r_sbi, xs_sig, xs_sbi, xs_bkg, mu_lo, mu_hi)[1] > 0


def density_minimum(r_S, r_sbi, xs_sig, xs_sbi, xs_bkg, mu_lo: float, mu_hi: float):
    """Smallest predicted density each event reaches over ``[mu_lo, mu_hi]``, and where.

    The single place the masking criterion is evaluated: ``positive_density_mask`` takes its sign
    and ``report_masked_events`` prints its value.

    Args:
        r_S: Ensembled signal likelihood ratio per event.
        r_sbi: SBI/bkg ratio per event.
        xs_sig, xs_sbi, xs_bkg: Cross sections.
        mu_lo: Lower edge of the range.
        mu_hi: Upper edge of it.

    Returns:
        ``(mu_min, n_min)`` -- where in mu the density bottoms out, and its value there.
    """
    a = xs_sig * r_S
    b = xs_sbi * r_sbi - xs_sig * r_S - xs_bkg
    # Clamping the vertex into the interval locates the minimum in every case.
    u_star = torch.clamp(-b / (2 * a), min=float(mu_lo) ** 0.5, max=float(mu_hi) ** 0.5)
    return u_star**2, a * u_star**2 + b * u_star + xs_bkg


def report_masked_events(
    keep, log_r_members, r_S, r_sbi, n_obs, xs_sig, xs_sbi, xs_bkg, mask_lo, mask_hi
) -> None:
    """Print, per excluded event, the density evaluation that excluded it.

    ``N_min``/``mu_min`` come from ``density_minimum``, so the table shows the decision itself.
    ``mu_lo``/``mu_hi`` are the roots of the same quadratic -- the window over which the density is
    negative -- sorted so the crossings appear in the order ``mu`` passes them. ``max f_i`` against
    ``log r_S`` separates causes: near ``log(1/eps)`` a saturated member is driving the combination,
    while a value of order ``log r_S`` means the members agree the event is signal-like.

    Args:
        keep: Mask from ``positive_density_mask``; the report covers its complement.
        log_r_members: Per-member log-ratios, shape ``(M, N_obs)``.
        r_S: Ensembled ratio per event.
        r_sbi: SBI/bkg ratio per event.
        n_obs: Per-event count/weight.
        xs_sig, xs_sbi, xs_bkg: Cross sections.
        mask_lo, mask_hi: The range the mask was applied over.
    """
    idx = torch.nonzero(~keep).flatten()
    if not len(idx):
        return

    # The criterion itself: masked exactly when this minimum is <= 0.
    mu_min, n_min = density_minimum(r_S[idx], r_sbi[idx], xs_sig, xs_sbi, xs_bkg, mask_lo, mask_hi)
    # Context: the roots bounding the negative window.
    a = xs_sig * r_S[idx]
    b = xs_sbi * r_sbi[idx] - xs_sig * r_S[idx] - xs_bkg
    root = torch.sqrt(torch.clamp(b**2 - 4 * a * xs_bkg, min=0.0))
    mu_lo = ((-b - root) / (2 * a)) ** 2
    mu_hi = ((-b + root) / (2 * a)) ** 2
    log_r_S = torch.log(r_S[idx])
    max_f = log_r_members[:, idx].max(dim=0).values

    print(
        f"\nmasked when min N(u) <= 0 over mu in [{mask_lo}, {mask_hi}], u = sqrt(mu):\n"
        f"  N(u) = {xs_sig:.6g}*r_S*u^2 + ({xs_sbi:.6g}*r_sbi - {xs_sig:.6g}*r_S - {xs_bkg:.6g})*u"
        f" + {xs_bkg:.6g}"
    )
    print(
        f"\n{'row':>8} {'n':>9} {'log r_S':>9} {'r_S':>11} {'r_sbi':>9} {'max f_i':>8} "
        f"{'mu_min':>8} {'N_min':>11} {'mu_lo':>8} {'mu_hi':>8}"
    )
    for k in torch.argsort(mu_lo):
        print(
            f"{int(idx[k]):8d} {float(n_obs[idx[k]]):9.5f} {float(log_r_S[k]):9.4f} "
            f"{float(r_S[idx[k]]):11.4g} {float(r_sbi[idx[k]]):9.4g} {float(max_f[k]):8.3f} "
            f"{float(mu_min[k]):8.4f} {float(n_min[k]):11.4g} "
            f"{float(mu_lo[k]):8.4f} {float(mu_hi[k]):8.4f}"
        )
    print()


def adjustment_at(log_likelihood, w, cov_t, mu_val) -> tuple[float, float, float]:
    """The uncertainty adjustment and its two factors at one ``mu``.

    ``sigma^2_MLE = -1 / d^2 logL/dmu^2`` is the statistical variance of ``mu_hat``, and ``A_i =
    d^2 logL / (dmu dw_i)`` how the mu-score responds to each weight, so ``A^T C A`` is the variance
    the weight uncertainty injects and the product is a variance ratio
    ``1 + sigma^2 * A^T C A = Var_total / Var_stat``.

    Args:
        log_likelihood: Shape log-likelihood as a function of ``(mu, w)``.
        w: Fitted ensemble weights.
        cov_t: Their covariance, float64.
        mu_val: Scalar tensor to evaluate at.

    Returns:
        ``(sigma2_mle, A^T C A, adjustment)`` as plain floats.
    """
    sigma2 = -1.0 / torch.func.hessian(log_likelihood, argnums=0)(mu_val, w)
    a_i = torch.func.jacfwd(torch.func.jacrev(log_likelihood, argnums=0), argnums=1)(mu_val, w)
    a_i64 = a_i.double()
    quad = (a_i64 @ cov_t @ a_i64.T)[0, 0]
    return float(sigma2), float(quad), float(1 + sigma2 * quad)


def scan_adjustment(log_likelihood, w, cov_t, dtype, spec: str) -> None:
    """Tabulate the adjustment across a range of ``mu``.

    Both factors vary with ``mu``, so this says whether ``mu_hat`` sits somewhere flat enough for a
    single-point value to be meaningful.

    Args:
        log_likelihood: Shape log-likelihood as a function of ``(mu, w)``.
        w: Fitted ensemble weights.
        cov_t: Their covariance, float64.
        dtype: dtype of the mu grid, so the scan points match it.
        spec: ``"lo,hi,n"``.
    """
    lo, hi, count = spec.split(",")
    # t_shape is the objective mu_hat minimizes; the other columns are only what the adjustment
    # would be at each mu.
    print(f"\n{'mu':>9}  {'t_shape':>12}  {'sigma2_mle':>12}  {'a^T C a':>12}  {'adjustment':>12}")
    for value in np.linspace(float(lo), float(hi), int(count)):
        mu_val = torch.tensor(float(value), dtype=dtype)
        t_shape_at = -2 * float(log_likelihood(mu_val, w))
        sigma2, quad, adj = adjustment_at(log_likelihood, w, cov_t, mu_val)
        print(f"{value:9.5f}  {t_shape_at:12.6g}  {sigma2:12.6g}  {quad:12.6g}  {adj:12.6g}")
    print()


def interval_bounds(t: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Shift ``t`` to its minimum and read off the 1-sigma (t = 1) interval by interpolation.

    Returns the shifted curve (min 0) plus the lower/upper mu where it crosses 1. Mirrors the
    notebook: interpolate on each side of the minimum, so a monotone-past-the-edge side clamps to
    the grid boundary rather than extrapolating.
    """
    t = t - t.min()
    k = int(t.argmin())
    lower = -np.interp(-1, -t[:k].numpy(), -mu[:k].numpy())
    upper = np.interp(1, t[k:].numpy(), mu[k:].numpy())
    return t, float(lower), float(upper)


def plot_curve(mu: torch.Tensor, t: torch.Tensor, lower: float, upper: float, out: str, title=None):
    """Save the [0, 1]-zoomed -2 log lambda curve with the 1/2-sigma lines and interval bounds."""
    plt.figure()
    plt.plot(mu.numpy(), t.numpy(), color="C0", label=r"$-2\log\lambda$")
    plt.xlim(0, 1.0)
    plt.ylim(0, 10)
    plt.axhline(1.0, color="tab:green", linestyle="--", label=r"$1\sigma$")
    plt.axhline(4.0, color="tab:orange", linestyle="--", label=r"$2\sigma$")
    plt.axvline(lower, color="tab:red", linestyle="--", label=f"Lower bound: {lower:.4f}")
    plt.axvline(upper, color="tab:purple", linestyle="--", label=f"Upper bound: {upper:.4f}")
    if title:
        plt.title(title)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$-2\log\lambda$")
    plt.legend()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the mu scan and save the confidence-interval plot."""
    args = parse_args()

    # --- Framework ensemble: fitted weights, scaler, members ---------------------------------
    # weights.pkl is loaded first: its "size" says how many members the fit combined, so the
    # checkpoints resolved below are exactly the ones w was fit against.
    with open(os.path.join(args.ensemble_dir, "weights.pkl"), "rb") as f:
        wf = pickle.load(f)
    w_fw = torch.tensor(wf["w"], dtype=torch.float32)  # (M + 1,): members + constant offset
    eps = wf["eps"]
    size = wf["size"]

    # The members were trained on features transformed by THIS scaler. Architecture comes from
    # each checkpoint's stored hyperparameters (CARL calls save_hyperparameters()).
    scaler_fw = load_scaler(args.ensemble_scaler)
    models_fw = [load_carl(ckpt) for ckpt in member_ckpts(args.ensemble_dir, size)]

    # --- SBI/bkg network (r_SBI estimator) + its own scaler -----------------------------------
    sbi = load_carl(args.sbi_ckpt)
    scaler_sbi = load_scaler(args.sbi_scaler)

    # --- Observed data + cross sections -------------------------------------------------------
    obs = pd.read_csv(args.obs_csv)
    X_obs = obs[FEATURES].to_numpy()
    n_obs = torch.tensor(obs["n"].to_numpy(), dtype=torch.float32)
    with open(args.xs_json) as f:
        xs = json.load(f)
    # Plain Python floats (not np.float64) so the autodiff below stays in float32.
    xs_sig, xs_int = float(np.prod(xs["sig"])), float(np.prod(xs["int"]))
    xs_sbi, xs_bkg = float(np.prod(xs["sbi"])), float(np.prod(xs["bkg"]))

    mu = torch.linspace(0.0, args.mu_max, args.mu_points)

    # --- Ensembled log r_S ---------------------------------------------------------------------
    s = torch.stack([predict(m, scaler_fw, X_obs) for m in models_fw], dim=0)  # (M, N_obs)
    log_r_members = torch.log(s / (1 - s + eps))  # (M, N_obs): per-member log r on the obs data
    r_S = torch.exp(w_fw[:-1] @ log_r_members + w_fw[-1])  # wifi ensemble
    s_sbi = predict(sbi, scaler_sbi, X_obs)
    r_sbi = s_sbi / (1 - s_sbi)

    # Excluded once, up front, so the event set does not change with mu (see positive_density_mask).
    if args.fixed_mask:
        mask_lo, mask_hi = (float(v) for v in args.mask_mu_range.split(","))
        keep = positive_density_mask(r_S, r_sbi, xs_sig, xs_sbi, xs_bkg, mask_lo, mask_hi)
        n_drop = int((~keep).sum())
        if n_drop:
            print(
                f"fixed mask: excluding {n_drop} of {len(keep)} events with a negative predicted "
                f"density somewhere in mu = [{mask_lo}, {mask_hi}] "
                f"({float(n_obs[~keep].sum()):.4g} of sum(n) = {float(n_obs.sum()):.4g})"
            )
            if args.list_masked:
                report_masked_events(
                    keep,
                    log_r_members,
                    r_S,
                    r_sbi,
                    n_obs,
                    xs_sig,
                    xs_sbi,
                    xs_bkg,
                    mask_lo,
                    mask_hi,
                )
            log_r_members = log_r_members[:, keep]
            r_S, r_sbi, n_obs = r_S[keep], r_sbi[keep], n_obs[keep]

    # Rate term: Poisson NLL of the observed count under the expected yield nu(mu)
    nu_sbi_mu = xs_sig * args.lumi * mu + xs_int * args.lumi * torch.sqrt(mu) + xs_bkg * args.lumi
    t_rate = nu_sbi_mu - n_obs.sum() * torch.log(nu_sbi_mu)

    # --- The shape term --------------------------------------------------------------------------
    mult_sig, mult_sbi, mult_bkg = mu - torch.sqrt(mu), torch.sqrt(mu), 1 - torch.sqrt(mu)
    r_mu = (
        xs_sig * mult_sig[None, :] * r_S[:, None]
        + xs_sbi * mult_sbi[None, :] * r_sbi[:, None]
        + xs_bkg * mult_bkg[None, :]
    ) / (xs_sig * mu + xs_int * torch.sqrt(mu) + xs_bkg)
    # nansum drops events whose predicted density is negative at this mu: inert under --fixed-mask,
    # and what makes the event set mu-dependent without it.
    t_shape = -2 * torch.nansum(n_obs[:, None] * torch.log(r_mu), dim=0)

    # Two output paths derived from --out: the before/after-adjustment curves.
    base, ext = os.path.splitext(args.out)
    out_before, out_after = f"{base}_before{ext}", f"{base}_after{ext}"

    # --- Before: unadjusted interval (ignores r_S estimation uncertainty) ----------------------
    t_before, lo0, hi0 = interval_bounds(t_rate + t_shape, mu)
    print(f"[before] Lower {lo0:.4f}  Upper {hi0:.4f}  stddev {(hi0 - lo0) / 2:.4f}")
    plot_curve(mu, t_before, lo0, hi0, out_before, title="wifi ensemble (before adjustment)")
    print(f"Saved before-adjustment plot to {out_before}")

    # --- After: propagate the fitted-weight covariance into the test statistic -----------------
    # (arXiv:2506.00113, App. B). Inflate t_shape by 1 + sigma^2_MLE * A^T C A, where C is the
    # weight covariance from the fit, sigma^2_MLE is the naive (r_S-known) mu resolution, and A_i
    # is the mixed mu-w_i second derivative of the log-likelihood at mu_hat. Without C we can only
    # report the (over-confident) before-adjustment interval.
    cov = wf.get("cov")
    if cov is None:
        print(
            "weights.pkl has no 'cov'; re-run the fit to enable the uncertainty adjustment. "
            "Only the before-adjustment plot was produced."
        )
        return

    # C comes out of the fit in float64 (weight_covariance computes the V U V sandwich in double
    # because the inverses are ill-conditioned); keep it there rather than throwing that away for
    # the product.
    cov_t = torch.tensor(cov, dtype=torch.float64)

    def log_likelihood(mu_val, w):
        """Observed log-likelihood as a function of mu and the weights, for autodiff at mu_hat."""
        r_s = torch.exp(w[:-1] @ log_r_members + w[-1])  # (N_obs,)
        m_sig, m_sbi, m_bkg = mu_val - mu_val**0.5, mu_val**0.5, 1 - mu_val**0.5
        r = (xs_sig * m_sig * r_s[:, None] + xs_sbi * m_sbi * r_sbi[:, None] + xs_bkg * m_bkg) / (
            xs_sig * mu_val + xs_int * mu_val**0.5 + xs_bkg
        )
        return torch.nansum(n_obs[:, None] * torch.log(r), dim=0)

    # Refined off the grid: the adjustment is more sensitive to mu_hat than the grid spacing can
    # resolve (see refine_mu_hat). --mu-hat overrides it to compare runs at the same point.
    mu_index = int(t_shape.argmin())
    if args.mu_hat is not None:
        mu_hat = torch.tensor(float(args.mu_hat), dtype=mu.dtype)
    else:
        mu_hat = refine_mu_hat(log_likelihood, w_fw, mu, mu_index)

    # Report the factors, not just the product, so a disagreement between runs says which moved.
    sigma2_mle, quad, adj = adjustment_at(log_likelihood, w_fw, cov_t, mu_hat)
    print(
        f"mu_hat = {float(mu_hat):.6f} (grid {float(mu[mu_index]):.4f}"
        f"{', forced' if args.mu_hat is not None else ''})  "
        f"sigma2_mle = {sigma2_mle:.6g}  a^T C a = {quad:.6g}"
    )

    if args.scan_adjustment:
        scan_adjustment(log_likelihood, w_fw, cov_t, mu.dtype, args.scan_adjustment)

    t_after, lo1, hi1 = interval_bounds(t_rate + t_shape / adj, mu)
    print(f"[after]  Lower {lo1:.4f}  Upper {hi1:.4f}  stddev {(hi1 - lo1) / 2:.4f}")
    widening = (hi1 - lo1) / (hi0 - lo0) - 1
    print(f"adjustment = {adj:.4f}  ->  interval widened by {widening * 100:.2f}%")
    plot_curve(mu, t_after, lo1, hi1, out_after, title="wifi ensemble (uncertainty-adjusted)")
    print(f"Saved after-adjustment plot to {out_after}")


if __name__ == "__main__":
    main()
