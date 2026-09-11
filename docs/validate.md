# Validating a trained model

A likelihood-ratio estimator cannot be judged by just checking the classifier accuracy — a
well-trained CARL model can sit near 0.5 everywhere if the two hypotheses overlap heavily. Instead
we can validate the model by checking whether the learned ratio **reweights one sample into the
other**.

Taking the denominator events with their weights `w_base`, multiply by the estimated ratio

$$
\hat{r}(x) = \frac{\hat{s}(x)}{1 - \hat{s}(x)}
$$

and compare the resulting distribution against the numerator events (`w_truth`). If the estimator is
accurate, the reweighted denominator and the true numerator agree feature by feature. This is a
*closure* test: it uses only the samples you already have, so it runs every validation epoch at no
extra cost.

## Closure metrics

`ClosureMetricsCallback` (`src/nsbi/callbacks/closure_metrics.py`) performs that comparison:

```yaml
callbacks:
  closure_metrics:
    _target_: nsbi.callbacks.closure_metrics.ClosureMetricsCallback
    plot: false
    plot_dir: "closure_plots"   # relative paths are anchored to trainer.default_root_dir
    feature_names: ["l1_pt", "l1_eta", ..., "l4_energy"]  # example config's names, one per column
```

For each feature it histograms `w_base`, `w_truth` and `w_pred = w_base * r̂` into 50 bins spanning
the 0.1–99.9 percentile range (plus a 5% margin), renormalizes the truth and prediction histograms
to the base sum, and computes two distances:

| Metric | Definition |
| --- | --- |
| χ² | `Σ_bins (pred - truth)² / (truth + 1e-6)` |
| Wasserstein | `Δ · Σ_bins \|CDF_pred - CDF_truth\|`, each CDF normalized to 1 |

Both are logged into `trainer.callback_metrics` **per feature**, as `{stage}_{feature}_chi2` and
`{stage}_{feature}_ws` — e.g. `val_l1_energy_ws`, `test_l3_pt_chi2`. There is no aggregate closure
metric, so any `hpo_tune.monitor` or `ModelCheckpoint` entry must name a specific feature.

`feature_names` must list one name per input column, **in input order**. The callback raises if the
count disagrees with the model's input width; it cannot detect a wrong *order*, which silently
mislabels every metric and plot. Keep it in sync with the column order your datamodule loader
returns.

## Closure plots

With `plot: true` the callback writes a grid figure, one panel per feature with a prediction/base
ratio panel under it, to `<plot_dir>/{stage}_closure.png`:

- **Validation** plots are written when a checkpoint is saved, so the figure on disk always
  corresponds to the newest checkpoint rather than to an arbitrary epoch.
- **Test** plots are written at the end of the test epoch.

Each panel shows three curves: the denominator (black, dashed), the numerator truth (blue), and the
NN-reweighted denominator (red). Good closure means red on top of blue, with the ratio panel flat
at 1.

Plotting failures are caught and logged rather than killing the run. Under a distributed strategy
the plot is built from rank 0's validation shard only.

## Evaluating on the test split

While the validation split drives early stopping and checkpoint selection during training, a final
evaluation can be done on the untouched test split:

```bash
uv run nsbi -f configs/conf_tune_carl.yaml -c stage=test
```

This resolves the latest checkpoint under `ckpt_path` (see
[train.md](train.md#where-checkpoints-go-and-how-they-are-found)), runs `trainer.test`, and logs
`test_loss` plus the full set of `test_*_chi2` / `test_*_ws` metrics. The stage is forced onto a
single device to avoid issues when using distributed samplers.

The test split should come from the same `data_dir` partition the model trained on.

With the `logger.csv` block from the example config, metrics land in `metrics.csv` under
`<save_dir>/lightning_logs/version_*/`.

## Checking an ensemble fit

The weight fit (`do_ensemble_fit: true`) validates itself as it runs; the checks go to stdout and
`logs/nsbi.log`. You should read these lines before trusting a fitted ensemble:

- **Initial and final symmetrized MLC loss**, and `|Δw|max`. If both the loss change and `|Δw|max`
  are ≈ 0, it may mean the optimizer never took a step.
- **An explicit warning when the weights equal the uniform init** (`1/M`, offset 0). A weaker
  warning fires when the member weights are all ~1/M but the offset *did* move.
- **Per-weight values**, `w[ 0] = +0.123456 +/- 0.004321`, approximate uncertainties from the square
  root of the covariance diagonal, plus a summary (sum / mean / std / min / max) of the member
  weights.

`disp` in `ensemble.fit.options` can be adjusted to increase verbosity during the minimization.

### Is the tolerance tight enough?

`ensemble.fit.tolerance_check: true` (off by default) will re-solve from the same initialization
with every tolerance you set divided by `tolerance_check_factor` (default 10), then reports the
largest relative weight change, warning with a per-parameter comparison if it exceeds
`tolerance_check_pct` (default 5). It is a diagnostic only: the saved `weights.pkl` is always from
the configured fit, never from the re-fit. If the tightened solve moves the weights appreciably,
your tolerances may be too loose for this objective's scale.

## Downstream example: signal-strength scan

> **Note:** This example is adapted from the ML4FP 2025 ensembling tutorial
> ([ensembling.ipynb](https://github.com/ml4fp/2025-lbnl/blob/main/sessions/day2/ensembling-tutorial/ensembling.ipynb)),
> and **is meant to be run on that tutorial's datasets** — the checkpoints, scaler, observed-data
> CSV and cross-section JSON it expects all come from there.

`scripts/mu_inference_example.py` is a standalone example of consuming a trained + fitted ensemble
for a physics measurement. It is not part of the framework — it shows how the framework's outputs
can be utilized downstream.

```bash
uv run python scripts/mu_inference_example.py \
    --ensemble-dir  /path/to/storage/ensemble \
    --ensemble-scaler /path/to/data/scaler.pkl \
    --sbi-ckpt      /path/to/sbi_model/epoch=76-train_loss=0.69.ckpt \
    --sbi-scaler    /path/to/sbi_model/scaler.pkl \
    --obs-csv       /path/to/obs_data/mu_x.csv \
    --xs-json       /path/to/xsecs/ggzz4l_xs.json
```

It reads the member count from `weights.pkl` and each network's architecture from its checkpoint
hyperparameters. It then scans the signal strength μ, combining a Poisson rate term with a per-event
shape term built from the ensembled ratio, and writes **two** plots of `-2 log λ(μ)` with the 1σ/2σ
lines and the interpolated interval bounds:

- `*_before.png` — the naive interval, treating the learned ratio as exact.
- `*_after.png` — the same scan with the fitted-weight covariance propagated into the test statistic
  ([arXiv:2506.00113](https://arxiv.org/abs/2506.00113)), which adjusts the interval by the amount
  printed to stdout.

Note that this example needs a second, separately trained network — an SBI/background CARL model
with its own scaler — in addition to the ensemble.
