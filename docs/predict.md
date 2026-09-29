# Prediction

`stage: predict` scores **input files** — not a split — and writes one score file per input.

```bash
uv run nsbi -f configs/conf_tune_carl.yaml -c stage=predict ckpt_path=/path/to/checkpoints
```

## What it does

`datamodule.setup("predict")` ignores the train / val / test partition entirely. It loads the
`StandardScaler` saved at `data_dir/scaler.pkl` — the one the model was trained with — and builds
one sequential, unshuffled dataset per input file. Score row *k* is input row *k*, so a score file
joins back to its input by row number.

Like `test`, the stage is forced onto `devices=1, num_nodes=1` to avoid using a distributed sampler.

`float32_matmul_precision` defaults to `"highest"` here and to `"medium"` for every other stage; an
explicit value in the config overrides this.

## Which files get scored

```yaml
datamodule:
  predict_files: null          # list of paths; null = numerator_events + denominator_events
  predict_sample_size: null    # null = every row, in file order
  predict_loader: null         # null = reuse the training `loader`
```

`predict_files` is a plain list, so one run can score several files; each gets its own output.
`null` falls back to the configured numerator and denominator.

`predict_sample_size` sets how many events the loader returns; `null` means every row.

`predict_loader` can be specified as needed if the `loader` used for training cannot be used for
processing the `predict_files`; `null` reuses the training `loader`.

## What gets written

```yaml
predict:
  out_dir: "predictions"   # relative paths are anchored to datamodule.data_dir
  format: "csv"            # see FORMATS in callbacks/prediction_writer.py
  use_ensemble: false      # score with every trained ensemble member instead of one checkpoint
```

Scores are streamed to disk by `ScoreWriter`, a `BasePredictionWriter` that `main_function` injects
into `cfg.callbacks` for this stage only.

An absolute `out_dir` is used as given. A relative one is anchored to `datamodule.data_dir` — where
the scaler and the split pickles already live.

`predict.format` selects a backend from the `FORMATS` registry (currently only `"csv"` exists).
Adding a format is a backend class plus a registry entry, with no change to `ScoreWriter` itself.

An output file is named from its input's stem, the writer's `suffix` (`_scores` by default), and the
extension the `format` backend declares — so using the `format: "csv"` an input `foo.csv` produces
`<out_dir>/foo_scores.csv`. Two inputs sharing a stem are disambiguated by dataloader index
(`foo_1_scores.csv`) rather than one silently overwriting the other.

One column per model that ran is written, plus the event weight read straight from the input:

| Run | Columns |
| --- | --- |
| Single model | `score, weight` |
| Ensemble (`use_ensemble: true`) | `score_0, …, score_{M-1}, weight` |

`score_i` is member *i*'s sigmoid output `s_i(x)` — the same quantity `score` holds for a single
model.

## Scoring with one model or with an ensemble

By default the stage resolves the most recently created checkpoint under `ckpt_path` (the same
directory search the `test` stage uses — see
[train.md](train.md#where-checkpoints-go-and-how-they-are-found)) and raises if it finds none.

`predict.use_ensemble: true` instead ignores `ckpt_path` entirely and assembles a `MemberEnsemble`
from the member manifest written by `do_ensemble_train`.

### Model configuration from checkpoint

Every checkpoint loaded here — a single model or an ensemble member — is rebuilt by
`build_model_from_checkpoint` from the hyperparameters the checkpoint itself stored (`CARL` calls
`save_hyperparameters`). Keys `cfg.model` disagrees about are warned about individually and the
checkpoint's value is used; anything the checkpoint did not record falls back to the config.

## Examples

```bash
# Every event of the configured numerator + denominator, with one trained model
uv run nsbi -f configs/conf_tune_carl.yaml -c \
    stage=predict ckpt_path=/path/to/checkpoints

# The same events, scored by every member of a trained ensemble (one score_i column each)
uv run nsbi -f configs/conf_tune_carl.yaml -c \
    stage=predict predict.use_ensemble=true ensemble.storage_path=/path/to/storage

# An observed-events CSV (features + a weight column), written somewhere other than
# <data_dir>/predictions -- with its own loader, since the training one expects more columns.
# `features`, `momentum_columns`, `component_columns` and `weight_column` are arguments of the
# example MCFM loader named on the `_target_` line
uv run nsbi -f configs/conf_tune_carl.yaml -c \
    stage=predict predict.use_ensemble=true \
    datamodule.predict_files=[/path/to/obs_data/mu_x.csv] \
    datamodule.predict_loader._target_=nsbi.examples.physics.simulation.mcfm.load_arrays_ordered \
    datamodule.predict_loader._partial_=true \
    datamodule.predict_loader.features=[l1_pt,l1_eta,...,l4_energy] \
    datamodule.predict_loader.momentum_columns=[] \
    datamodule.predict_loader.component_columns=[] \
    datamodule.predict_loader.weight_column=n \
    predict.out_dir=/path/to/scores
```

## Downstream example: signal-strength scan using scores

`scripts/mu_inference_from_scores.py` runs the same signal-strength scan as the example in
[validate.md](validate.md#downstream-example-signal-strength-scan), but from score files rather than
model checkpoints. Since the network evaluations already happened at predict time, the scan can be
run without needing to specify checkpoints.

It takes two predict runs over the **same** observed-events file — an ensemble one
(`score_0 … score_{M-1}`) and a single SBI/background checkpoint (`score`) — plus the `weights.pkl`
from `do_ensemble_fit` and the cross-section JSON:

```bash
uv run python scripts/mu_inference_from_scores.py \
    --ensemble-scores /path/to/predictions/mu_x_scores.csv \
    --sbi-scores      /path/to/predictions_sbi/mu_x_scores.csv \
    --weights         /path/to/storage/ensemble/weights.pkl \
    --xs-json         /path/to/xsecs/ggzz4l_xs.json
```
