#!/usr/bin/env python
"""Reproduce the mu confidence-interval plot from https://github.com/ml4fp/2025-lbnl/blob/main/sessions/day2/ensembling-tutorial/ensembling.ipynb.

Standalone downstream analysis: it takes an ensemble trained + fit by the framework
(do_ensemble_train + do_ensemble_fit -> member checkpoints + weights.pkl + scaler.pkl) and runs the
signal-strength (mu) test-statistic scan, then
saves the [0, 1]-zoomed -2 log lambda curve with the 1sigma/2sigma lines and the interpolated
interval bounds.

Example:
    python3 mu_inference_example.py \
        --ensemble-dir /home/nkang/test_notebook/storage/ensemble \
        --ensemble-scaler /home/nkang/test_notebook/data/scaler.pkl \
        --size 8 \
        --n-layers 2 \
        --n-nodes 8 \
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
from torch.utils.data import DataLoader, TensorDataset

from nsbi.models.carl import CARL

FEATURES = [
    "l1_pt", "l1_eta", "l1_phi", "l1_energy",
    "l2_pt", "l2_eta", "l2_phi", "l2_energy",
    "l3_pt", "l3_eta", "l3_phi", "l3_energy",
    "l4_pt", "l4_eta", "l4_phi", "l4_energy",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and hyperparameters for the mu scan."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Framework ensemble (trained + fit by do_ensemble_train/do_ensemble_fit).
    p.add_argument("--ensemble-dir", required=True, help="<storage_path>/ensemble (members + weights.pkl)")
    p.add_argument("--ensemble-scaler", required=True, help="scaler.pkl the members were trained with")
    p.add_argument("--size", type=int, default=16, help="number of ensemble members M")
    p.add_argument("--n-layers", type=int, default=3, help="member CARL n_layers (must match training)")
    p.add_argument("--n-nodes", type=int, default=64, help="member CARL n_nodes (must match training)")
    # Separately trained SBI/bkg network (the r_SBI estimator; a bigger CARL) + its own scaler.
    p.add_argument("--sbi-ckpt", required=True, help="checkpoint for the SBI/bkg CARL network")
    p.add_argument("--sbi-scaler", required=True, help="scaler pickle for the SBI/bkg network")
    p.add_argument("--sbi-layers", type=int, default=16, help="SBI network n_layers")
    p.add_argument("--sbi-nodes", type=int, default=1024, help="SBI network n_nodes")
    # Physics data.
    p.add_argument("--obs-csv", required=True, help="observed events CSV (features + column 'n')")
    p.add_argument("--xs-json", required=True, help="cross-section JSON (keys sig/int/sbi/bkg)")
    p.add_argument("--lumi", type=float, default=300.0, help="integrated luminosity [1/fb]")
    p.add_argument("--mu-max", type=float, default=4.0, help="upper edge of the mu scan grid")
    p.add_argument("--mu-points", type=int, default=401, help="number of grid points in the mu scan")
    p.add_argument("--out", default="mu_interval_framework.png", help="output plot path")
    return p.parse_args()


def load_scaler(path: str):
    """Unpickle a fitted sklearn StandardScaler."""
    with open(path, "rb") as f:
        return pickle.load(f)


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


def predict(model: CARL, scaler, X: np.ndarray, batch_size: int = 1024) -> torch.Tensor:
    """Scale features with ``scaler`` and return the model's sigmoid outputs, batched."""
    dl = DataLoader(
        TensorDataset(torch.tensor(scaler.transform(X), dtype=torch.float32)),
        batch_size=batch_size,
    )
    model.eval()
    with torch.no_grad():
        return torch.cat([model(xb).flatten() for (xb,) in dl])


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
    plt.axvline(lower, color="tab:red", linestyle="--", label=f"Lower bound: {lower:.2f}")
    plt.axvline(upper, color="tab:purple", linestyle="--", label=f"Upper bound: {upper:.2f}")
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

    # --- Framework ensemble: scaler, members, fitted weights ---------------------------------
    # The members were trained on features transformed by THIS scaler
    scaler_fw = load_scaler(args.ensemble_scaler)
    models_fw = []
    for ckpt in member_ckpts(args.ensemble_dir, args.size):
        m = CARL(n_features=len(FEATURES), n_layers=args.n_layers, n_nodes=args.n_nodes, learning_rate=1e-3)
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        models_fw.append(m)
    with open(os.path.join(args.ensemble_dir, "weights.pkl"), "rb") as f:
        wf = pickle.load(f)
    w_fw = torch.tensor(wf["w"], dtype=torch.float32)  # (M + 1,): members + constant offset
    eps = wf["eps"]

    # --- SBI/bkg network (r_SBI estimator) + its own scaler -----------------------------------
    sbi = CARL(n_features=len(FEATURES), n_layers=args.sbi_layers, n_nodes=args.sbi_nodes, learning_rate=1e-5)
    sbi.load_state_dict(torch.load(args.sbi_ckpt, map_location="cpu")["state_dict"])
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

    # Rate term: Poisson NLL of the observed count under the expected yield nu(mu)
    nu_sbi_mu = xs_sig * args.lumi * mu + xs_int * args.lumi * torch.sqrt(mu) + xs_bkg * args.lumi
    t_rate = nu_sbi_mu - n_obs.sum() * torch.log(nu_sbi_mu)

    # --- Ensembled log r_S and the shape term -------------------------------------------------
    s = torch.stack([predict(m, scaler_fw, X_obs) for m in models_fw], dim=0)  # (M, N_obs)
    r_S = torch.exp(w_fw[:-1] @ torch.log(s / (1 - s + eps)) + w_fw[-1])       # wifi ensemble
    s_sbi = predict(sbi, scaler_sbi, X_obs)
    r_sbi = s_sbi / (1 - s_sbi)

    mult_sig, mult_sbi, mult_bkg = mu - torch.sqrt(mu), torch.sqrt(mu), 1 - torch.sqrt(mu)
    r_mu = (
        xs_sig * mult_sig[None, :] * r_S[:, None]
        + xs_sbi * mult_sbi[None, :] * r_sbi[:, None]
        + xs_bkg * mult_bkg[None, :]
    ) / (xs_sig * mu + xs_int * torch.sqrt(mu) + xs_bkg)
    # nansum: a bad member can push r_mu <= 0 for some events; those log terms are dropped.
    t_shape = -2 * torch.nansum(n_obs[:, None] * torch.log(r_mu), dim=0)

    # Two output paths derived from --out: the before/after-adjustment curves.
    base, ext = os.path.splitext(args.out)
    out_before, out_after = f"{base}_before{ext}", f"{base}_after{ext}"

    # --- Before: unadjusted interval (ignores r_S estimation uncertainty) ----------------------
    t_before, lo0, hi0 = interval_bounds(t_rate + t_shape, mu)
    print(f"[before] Lower {lo0:.2f}  Upper {hi0:.2f}  stddev {(hi0 - lo0) / 2:.2f}")
    plot_curve(mu, t_before, lo0, hi0, out_before, title="wifi ensemble (before adjustment)")
    print(f"Saved before-adjustment plot to {out_before}")

    # --- After: propagate the fitted-weight covariance into the test statistic -----------------
    # (arXiv:2506.00113, App. B). Inflate t_shape by 1 + sigma^2_MLE * A^T C A, where C is the
    # weight covariance from the fit, sigma^2_MLE is the naive (r_S-known) mu resolution, and A_i
    # is the mixed mu-w_i second derivative of the log-likelihood at mu_hat. Without C we can only
    # report the (over-confident) before-adjustment interval.
    cov = wf.get("cov")
    if cov is None:
        print("weights.pkl has no 'cov'; re-run the fit to enable the uncertainty adjustment. "
              "Only the before-adjustment plot was produced.")
        return

    cov_t = torch.tensor(cov, dtype=torch.float32)  # notebook casts C to float for the product
    log_r_members = torch.log(s / (1 - s + eps))    # (M, N_obs): per-member log r on the obs data

    def log_likelihood(mu_val, w):
        """Observed log-likelihood as a function of mu and the weights, for autodiff at mu_hat."""
        r_s = torch.exp(w[:-1] @ log_r_members + w[-1])  # (N_obs,)
        m_sig, m_sbi, m_bkg = mu_val - mu_val**0.5, mu_val**0.5, 1 - mu_val**0.5
        r = (xs_sig * m_sig * r_s[:, None] + xs_sbi * m_sbi * r_sbi[:, None] + xs_bkg * m_bkg) / (
            xs_sig * mu_val + xs_int * mu_val**0.5 + xs_bkg
        )
        return torch.nansum(n_obs[:, None] * torch.log(r), dim=0)

    mu_hat = mu[int(t_shape.argmin())]
    # sigma^2_MLE = -1 / d^2 logL/dmu^2 ; A_i = d^2 logL / (dmu dw_i), both at (mu_hat, w).
    sigma2_mle = -1.0 / torch.func.hessian(log_likelihood, argnums=0)(mu_hat, w_fw)
    a_i = torch.func.jacfwd(torch.func.jacrev(log_likelihood, argnums=0), argnums=1)(mu_hat, w_fw)
    adjustment = 1 + sigma2_mle * (a_i @ cov_t @ a_i.T)[0, 0]
    adj = float(adjustment)

    t_after, lo1, hi1 = interval_bounds(t_rate + t_shape / adjustment, mu)
    print(f"[after]  Lower {lo1:.2f}  Upper {hi1:.2f}  stddev {(hi1 - lo1) / 2:.2f}")
    #
    widening = (hi1 - lo1) / (hi0 - lo0) - 1
    print(f"adjustment = {adj:.4f}  ->  interval widened by {widening * 100:.0f}%")
    plot_curve(mu, t_after, lo1, hi1, out_after, title="wifi ensemble (uncertainty-adjusted)")
    print(f"Saved after-adjustment plot to {out_after}")


if __name__ == "__main__":
    main()
