#!/usr/bin/env python
"""Run the mu confidence-interval scan from precomputed score CSVs, with no model checkpoints.

Same analysis as ``mu_inference_example.py`` -- and it should reproduce that script's numbers on
the same events -- but it consumes the output of ``stage: predict`` instead of loading networks
and running inference itself. Nothing here needs a checkpoint, a scaler: the only inputs
are two score files, the fitted ``weights.pkl``, and the cross sections.

The network evaluations are the only expensive step and they do not depend on mu, the luminosity,
the cross sections, or the fitted weights -- so scoring once lets the scan be re-run freely.

Both score files come from ``stage: predict`` over the SAME observed-events file:

  --ensemble-scores   predict.use_ensemble=true  ->  score_0..score_{M-1}, weight
  --sbi-scores        a single SBI/bkg checkpoint ->  score, weight

The ``weight`` column is the per-row multiplicity the likelihood needs (``weight_column: "n"`` for
a counts/Asimov dataset, 1.0 per event for real data), so ``--n-column`` defaults to it. Rows are
matched positionally, as predict guarantees (``predict_sample_size: null`` preserves input order);
row counts and weight columns are cross-checked to catch a mismatched pair.

Example:
    python3 mu_inference_from_scores.py \
        --ensemble-scores /eagle/.../predictions/mu_x_scores.csv \
        --sbi-scores /eagle/.../predictions_sbi/mu_x_scores.csv \
        --weights /home/nkang/test_notebook/storage/ensemble/weights.pkl \
        --xs-json /eagle/ScalingHEPAI/test_Nathan/wifi_data/xsecs/ggzz4l_xs.json
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar


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
    a = xs_sig * r_S
    b = xs_sbi * r_sbi - xs_sig * r_S - xs_bkg
    # Clamping the vertex into the interval locates the minimum in every case.
    u_star = torch.clamp(-b / (2 * a), min=float(mu_lo) ** 0.5, max=float(mu_hi) ** 0.5)
    return a * u_star**2 + b * u_star + xs_bkg > 0


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


def parse_args() -> argparse.Namespace:
    """Parse the score files, fitted weights, and physics inputs for the mu scan."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Precomputed scores (stage: predict) -- one file per network, same observed events.
    p.add_argument(
        "--ensemble-scores", required=True, help="predict output with score_0..score_{M-1}"
    )
    p.add_argument("--sbi-scores", required=True, help="predict output for the SBI/bkg net (score)")
    # Framework fit artifact (do_ensemble_fit).
    p.add_argument(
        "--weights", required=True, help="weights.pkl from do_ensemble_fit (w, cov, eps, size)"
    )
    # Physics data.
    p.add_argument("--xs-json", required=True, help="cross-section JSON (keys sig/int/sbi/bkg)")
    p.add_argument(
        "--n-column", default="weight", help="column holding the per-row event count/weight"
    )
    p.add_argument("--lumi", type=float, default=300.0, help="integrated luminosity [1/fb]")
    p.add_argument("--mu-max", type=float, default=4.0, help="upper edge of the mu scan grid")
    p.add_argument(
        "--mu-points", type=int, default=401, help="number of grid points in the mu scan"
    )
    p.add_argument("--out", default="mu_interval_from_scores.png", help="output plot path")
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


def load_member_scores(path: str, size: int) -> tuple[pd.DataFrame, torch.Tensor]:
    """Read an ensemble predict file and return its frame plus the ``(M, N_obs)`` member outputs.

    The member columns are read by index, ``score_0 .. score_{size-1}``, so they line up with
    ``w[i]`` from the fit regardless of the order pandas hands back the columns in.
    """
    df = pd.read_csv(path)
    columns = [f"score_{i}" for i in range(size)]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing member column(s) {missing}. "
            f"weights.pkl was fit for {size} members; "
            "was this file written by a predict run over the same ensemble?"
        )
    # (N, M) as stored -> (M, N), the orientation the fit and the loss work in.
    s = torch.tensor(df[columns].to_numpy().T, dtype=torch.float32)
    return df, s


def load_single_scores(path: str) -> tuple[pd.DataFrame, torch.Tensor]:
    """Read a single-model predict file and return its frame plus the ``(N_obs,)`` scores."""
    df = pd.read_csv(path)
    if "score" not in df.columns:
        raise KeyError(f"{path} has no 'score' column; expected a single-model predict output.")
    return df, torch.tensor(df["score"].to_numpy(), dtype=torch.float32)


def check_aligned(ens: pd.DataFrame, sbi: pd.DataFrame, n_column: str) -> None:
    """Fail loudly if the two score files cannot be the same events in the same order.

    The join is positional, so a silent mismatch would produce a plausible-looking but meaningless
    curve. Row count catches different files; the weight column catches same-length files scored
    from different inputs.
    """
    if len(ens) != len(sbi):
        raise ValueError(
            f"Score files have different row counts ({len(ens)} vs {len(sbi)}); "
            "they must be predict outputs over the same observed-events file."
        )
    if n_column in ens.columns and n_column in sbi.columns:
        if not np.allclose(ens[n_column].to_numpy(), sbi[n_column].to_numpy()):
            raise ValueError(
                f"The '{n_column}' columns of the two score files differ, so they were not scored "
                "from the same input rows."
            )


def interval_bounds(t: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Shift ``t`` to its minimum and read off the 1-sigma (t = 1) interval by interpolation.

    Returns the shifted curve (min 0) plus the lower/upper mu where it crosses 1. Interpolates on
    each side of the minimum, so a monotone-past-the-edge side clamps to the grid boundary rather
    than extrapolating. Same as ``mu_inference_example.interval_bounds``.
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
    """Run the mu scan from score files and save the confidence-interval plots."""
    args = parse_args()

    # --- Fitted ensemble weights ---------------------------------------------------------------
    # weights.pkl is the only fit artifact needed: "size" says how many member columns to read,
    # "eps" reproduces the guard the weights were fit under, and "cov" drives the adjustment.
    with open(args.weights, "rb") as f:
        wf = pickle.load(f)
    w_fw = torch.tensor(wf["w"], dtype=torch.float32)  # (M + 1,): members + constant offset
    eps = wf["eps"]
    size = wf["size"]

    # --- Precomputed scores ---------------------------------------------------------------------
    ens_df, s = load_member_scores(args.ensemble_scores, size)  # (M, N_obs)
    sbi_df, s_sbi = load_single_scores(args.sbi_scores)  # (N_obs,)
    check_aligned(ens_df, sbi_df, args.n_column)

    if args.n_column not in ens_df.columns:
        raise KeyError(
            f"{args.ensemble_scores} has no '{args.n_column}' column "
            "to use as the per-event count. Point predict's loader at the count column "
            "(weight_column: 'n'), use weight_column: null for unit-weight real data, "
            "or pass --n-column."
        )
    n_obs = torch.tensor(ens_df[args.n_column].to_numpy(), dtype=torch.float32)

    with open(args.xs_json) as f:
        xs = json.load(f)
    # Plain Python floats (not np.float64) so the autodiff below stays in float32.
    xs_sig, xs_int = float(np.prod(xs["sig"])), float(np.prod(xs["int"]))
    xs_sbi, xs_bkg = float(np.prod(xs["sbi"])), float(np.prod(xs["bkg"]))

    print(f"{len(ens_df)} events, {size} members, sum(n) = {float(n_obs.sum()):.2f}")

    mu = torch.linspace(0.0, args.mu_max, args.mu_points)

    # --- Ensembled log r_S ------------------------------------------------------------------------
    # The wifi combination happens here rather than at predict time: the score files hold each
    # member's raw output, so a re-fit changes only w and eps below, not the scored events.
    log_r_members = torch.log(s / (1 - s + eps))  # (M, N_obs)
    r_S = torch.exp(w_fw[:-1] @ log_r_members + w_fw[-1])
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
            log_r_members = log_r_members[:, keep]
            r_S, r_sbi, n_obs = r_S[keep], r_sbi[keep], n_obs[keep]

    # Rate term: Poisson NLL of the observed count under the expected yield nu(mu)
    nu_sbi_mu = xs_sig * args.lumi * mu + xs_int * args.lumi * torch.sqrt(mu) + xs_bkg * args.lumi
    t_rate = nu_sbi_mu - n_obs.sum() * torch.log(nu_sbi_mu)

    # --- The shape term ---------------------------------------------------------------------------
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

    # --- Before: unadjusted interval (ignores r_S estimation uncertainty) ------------------------
    t_before, lo0, hi0 = interval_bounds(t_rate + t_shape, mu)
    print(f"[before] Lower {lo0:.4f}  Upper {hi0:.4f}  stddev {(hi0 - lo0) / 2:.4f}")
    plot_curve(mu, t_before, lo0, hi0, out_before, title="wifi ensemble (before adjustment)")
    print(f"Saved before-adjustment plot to {out_before}")

    # --- After: propagate the fitted-weight covariance into the test statistic --------------------
    # (arXiv:2506.00113, App. B). Inflate t_shape by 1 + sigma^2_MLE * A^T C A, where C is the
    # weight covariance from the fit, sigma^2_MLE is the naive (r_S-known) mu resolution, and A_i
    # is the mixed mu-w_i second derivative of the log-likelihood at mu_hat. The autodiff below is
    # over (mu, w) only -- log_r_members is a constant -- which is why no network is needed here.
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
