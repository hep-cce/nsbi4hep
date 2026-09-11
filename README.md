# Scaling Neural Simulation-Based Inference at High Performance Computing Centers for LHC analysis

This framework covers the steps related to training, validating and evaluating the neural
likelihood-ratio estimators at scale for an NSBI analysis. Downstream tasks such as the statistical
inference are not currently intended to be included as part of the framework.

## Features

- **One config file.** A single OmegaConf YAML describes the data, model, trainer, loggers
  and callbacks; every field is overridable from the command line.
- **Physics closure metrics.** Per-feature χ² and Wasserstein distances between reweighted and
  target distributions, logged every validation epoch.
- **Hyperparameter optimization.** Ray Tune, with single/fractional GPU per trial or multiple using
  distributed training.
- ***w<sub>i</sub>f<sub>i</sub>* ensembling.** Train multiple bootstrapped ensemble members
  (*f<sub>i</sub>*), then fit the weights (*w<sub>i</sub>*) of a linear combination of their
  classifier scores on a held-out data split by optimizing for the maximum likelihood classifier
  loss ([arXiv:2506.00113](https://arxiv.org/abs/2506.00113)).

## Installation

We use [`uv`](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

```bash
uv venv
uv sync                  # or `make sync` to include the dev group and all extras
source .venv/bin/activate
```

## Quickstart

Everything goes through the `nsbi` entry point, which takes a config **file** and optional
`key=value` overrides:

```bash
# Train a model on your own numerator/denominator CSVs
uv run nsbi -f configs/conf_tune_carl.yaml -c \
    datamodule.numerator_events=/path/to/signal.csv \
    datamodule.denominator_events=/path/to/background.csv \
    datamodule.data_dir=/path/to/mydatadir

# Evaluate the trained model on the held-out test split
uv run nsbi -f configs/conf_tune_carl.yaml -c stage=test

# Hyperparameter search instead of a single run
uv run nsbi -f configs/conf_tune_carl.yaml -c do_hpo_tune=true

uv run nsbi --help
```

## CLI

| Flag | Meaning |
| --- | --- |
| `-f`, `--config-path` | Path to one OmegaConf YAML file. Required. |
| `-c`, `--configs` | Zero or more `key=value` overrides, merged onto the loaded config. |

| Key | Effect |
| --- | --- |
| `stage` | `fit` \| `finetune` \| `resume` \| `test` \| `predict` |
| `do_hpo_tune` | Ray Tune hyperparameter search |
| `do_ensemble_train` | Train an ensemble of members |
| `do_ensemble_fit` | Fit *w<sub>i</sub>f<sub>i</sub>* ensemble weights over trained members |

## Documentation

- **[docs/train.md](docs/train.md)** — config schema, data format and splits, the training stages,
  HPO, ensembling.
- **[docs/validate.md](docs/validate.md)** — closure metrics, evaluating a trained model, reading
  the ensemble fit output.
- **[docs/predict.md](docs/predict.md)** — writing classifier scores for arbitrary input files to
  disk, with one model or with every ensemble member.

## References

- A. Held, J. Sandesara, "Introduction to NSBI",
  [link](https://indico.cern.ch/event/1656822/contributions/6963531/attachments/3277049/5855409/20260519_SBI_intro.pdf)
- ATLAS Collaboration, "An implementation of NSBI in ATLAS",
  [arXiv:2412.01600](https://arxiv.org/abs/2412.01600)
- Ensembling for NSBI (*w<sub>i</sub>f<sub>i</sub>*),
  [arXiv:2506.00113](https://arxiv.org/abs/2506.00113)
