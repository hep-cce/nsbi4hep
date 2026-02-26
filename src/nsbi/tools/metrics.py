import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


# remove?
def total_reduced_chi2(chi2_scores: dict, binning: list):
    total_chi2 = 0.0
    total_dof = 0

    for i, key in enumerate(chi2_scores):
        score = chi2_scores[key]
        nbins = binning[i][0]
        dof = nbins - 1
        total_chi2 += score
        total_dof += dof

    return total_chi2 / total_dof if total_dof > 0 else float("nan")


def mean_wasserstein(wstein_scores: dict):
    wstein_values = []

    for key in wstein_scores:
        wstein_values.append(wstein_scores[key])
    avg_wstein = np.mean(wstein_values)
    return avg_wstein


# need to generalize to arb features
def plot_closure_grid(
    observables,
    observable_names,
    true_weights,
    predicted_weights,
    base_weights,
    binning,
    ncols=3,
    ratio_ylim=(0.8, 1.2),
    log_scale=False,
    figsize=(16, 16),
    title_prefix="Reweighting closure:",
    output_dir="closure_plots",
    file_prefix="closure",
):
    nvars = observables.shape[1]
    nrows = (nvars + ncols - 1) // ncols

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=figsize, sharex=False)
    axes = np.array(axes).reshape(nrows * 2, ncols)

    for i in range(nvars):
        row = (i // ncols) * 2
        col = i % ncols

        values = observables[:, i]

        name = observable_names[i]
        nbins, low, high = binning[i]
        bins = np.linspace(low, high, nbins + 1)

        hist_base, bin_edges = np.histogram(values, bins=bins, weights=base_weights)
        hist_truth, _ = np.histogram(values, bins=bin_edges, weights=true_weights)
        hist_pred, _ = np.histogram(values, bins=bin_edges, weights=predicted_weights)

        norm_factor = hist_base.sum()
        hist_truth = hist_truth * norm_factor / (hist_truth.sum() + 1e-8)
        hist_pred = hist_pred * norm_factor / (hist_pred.sum() + 1e-8)

        centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        # Main plot
        ax_main.step(centers, hist_base, where="mid", label="BKG", color="black", linestyle="--")
        ax_main.step(centers, hist_truth, where="mid", label="BKG->SBI (truth)", color="blue")
        ax_main.step(centers, hist_pred, where="mid", label="BKG->SBI (NN prediction)", color="red")
        if log_scale:
            ax_main.set_yscale("log")
        ax_main.set_title(f"{title_prefix} {name}")
        ax_main.grid(True)

        # Ratio plot
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_truth = np.nan_to_num(hist_truth / hist_base, nan=0, posinf=0, neginf=0)
            ratio_pred = np.nan_to_num(hist_pred / hist_base, nan=0, posinf=0, neginf=0)

        ax_ratio.axhline(1.0, color="gray", linestyle="--")
        ax_ratio.step(centers, ratio_truth, where="mid", color="blue")
        ax_ratio.step(centers, ratio_pred, where="mid", color="red")
        ax_ratio.set_ylim(*ratio_ylim)
        ax_ratio.set_xlabel(name)
        ax_ratio.grid(True)

    # Turn off unused axes
    for j in range(nvars, nrows * ncols):
        axes[(j // ncols) * 2, j % ncols].axis("off")
        axes[(j // ncols) * 2 + 1, j % ncols].axis("off")

    fig.tight_layout()
    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    output_path = os.path.join(output_dir, f"{file_prefix}.png")
    plt.savefig(output_path)
    plt.close(fig)


class ReweightingClosureMetric:
    def __init__(self, observables, binning, metric="chi2"):
        self.observables = observables
        if binning is not None:
            self.binning = binning

        else:
            self.binning = []
            for _obs in self.observables:
                self.binning.append(None)

        self.metric = metric

    def __call__(self, kin, weights_pred, weights_truth, weights_base):
        results = {}
        kin = kin.numpy() if not isinstance(kin, np.ndarray) else kin
        weights_pred = weights_pred.numpy()
        weights_truth = weights_truth.numpy()
        weights_base = weights_base.numpy()

        for i, obs in enumerate(self.observables):
            name = f"obs_{obs}" if isinstance(obs, int) else str(obs)
            values = kin[:, obs] if isinstance(obs, int) else kin[obs]

            if self.binning[i] is None:
                vmin, vmax = np.percentile(values, [0.1, 99.9])
                margin = 0.05 * (vmax - vmin)
                bins = np.linspace(vmin - margin, vmax + margin, 51)
            else:
                bins = np.linspace(self.binning[i][1], self.binning[i][2], self.binning[i][0] + 1)

            hist_base, _ = np.histogram(values, bins=bins, weights=weights_base)
            hist_truth, _ = np.histogram(values, bins=bins, weights=weights_truth)
            hist_pred, _ = np.histogram(values, bins=bins, weights=weights_pred)

            norm_factor = hist_base.sum()
            hist_truth = hist_truth * norm_factor / (hist_truth.sum() + 1e-8)
            hist_pred = hist_pred * norm_factor / (hist_pred.sum() + 1e-8)

            if self.metric == "chi2":
                with np.errstate(divide="ignore", invalid="ignore"):
                    chi2 = np.nan_to_num((hist_pred - hist_truth) ** 2 / (hist_truth + 1e-6)).sum()
                    results[f"{name}_chi2"] = chi2
            elif self.metric == "wasserstein":
                centers = 0.5 * (bins[:-1] + bins[1:])
                wdist = wasserstein_distance(centers, centers, hist_pred, hist_truth)
                results[f"{name}_wasserstein"] = wdist
            else:
                raise ValueError("Unsupported metric type.")

        return results
