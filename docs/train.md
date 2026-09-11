# Training

This page covers how to configure and launch training: the config file, the input data, the run
stages, hyperparameter optimization, ensembling. For validating a trained model, see
[validate.md](validate.md).

## The config file

A run is described by a single OmegaConf YAML file — `configs/conf_tune_carl.yaml` is the reference
example. To change behavior you either edit your specified YAML or pass `-c key=value` overrides.

```bash
uv run nsbi -f configs/conf_tune_carl.yaml -c stage=test
```

Top-level keys fall into two kinds.

**Switches** — plain values that select what the run does:

```yaml
seed: 42                    # seeds torch / numpy / random via L.seed_everything
stage: "fit"                # fit | finetune | resume | test | predict
ckpt_path: null             # directory searched for checkpoints by resume / test / predict
ckpt_file: null             # a single checkpoint file; required by finetune, ignored by fit
do_hpo_tune: false          # Ray Tune hyperparameter search
do_ensemble_train: false    # train an ensemble of members
do_ensemble_fit: false      # fit wifi ensemble weights
do_compile: false           # torch.compile the network
compile_kwargs: {...}       # forwarded to torch.compile when do_compile
suppress_accumulate_grad_warning: false   # silence accumulate-grad stream-mismatch for DDP
```

**Component blocks** — each has a `_target_` naming a Python class or function, and the rest of the
keys are its constructor arguments. `hydra.utils.instantiate` builds them at run time:

| Block | Builds |
| --- | --- |
| `datamodule` | A Lightning `LightningDataModule` (e.g. `BalancedDataModule`) |
| `model` | A Lightning module (e.g. `CARL`) |
| `trainer` | `lightning.pytorch.Trainer` |
| `logger` | One entry per logger (CSV, WandB, …) |
| `callbacks` | One entry per callback (checkpointing, early stopping, closure metrics, …) |
| `hpo_tune.scheduler` | The Ray Tune trial scheduler (e.g. ASHA) |

Because these are ordinary constructor arguments, anything the underlying class accepts is
configurable without touching the framework.

`ensemble`, `hpo_tune` and `predict` are nested blocks of plain values rather than `_target_`
components; they are covered in the ensembling and HPO sections below.

## The example physics package

`src/nsbi/examples/physics/` is an example HEP domain package shipped alongside the framework rather
than part of it. The example path — `BalancedDataModule` with CARL — reaches it only through a
`_target_` in the config, so it is a template to copy or replace.

## Input data

The `datamodule` block builds the data pipeline, and its keys are simply the constructor arguments
of whatever class `datamodule._target_` specifies.

### BalancedDataModule

Reads two files, one per hypothesis, and serves balanced `(x, y, w)` batches to a classifier such as
CARL: `x` the feature vector for one event, `y` its label (1 = numerator, 0 = denominator), and `w`
its per-event weight.

#### The loader

`datamodule.loader._target_` can be any callable returning arrays `(X, w)` — a feature matrix of
shape `(n_events, n_features)` and a weight vector of shape `(n_events,)`. It is instantiated with
`_partial_: true`, so the datamodule calls it as `loader(path, sample_size=..., random_state=...)`.

| Argument | What the datamodule passes |
| --- | --- |
| `path` | `numerator_events` / `denominator_events`, or one entry of `predict_files` |
| `sample_size` | `sample_size`, or `predict_sample_size` for predict; `null` means every event |
| `random_state` | `random_state`; accept it even if nothing in the loader is random |

Every other key under `datamodule.loader` is an argument of the callable itself, bound by
`_partial_` and never inspected by the framework.

```yaml
datamodule:
  loader:
    _target_: nsbi.examples.physics.simulation.mcfm.load_arrays  # an example loader, not a default
    _partial_: true
    features: ["l1_pt", "l1_eta", ..., "l4_energy"]  # an argument of load_arrays, not a framework key
```

The `predict` stage joins score row *k* to input row *k*, so a loader used there must return rows in
input order. The framework does not check this. Among the example loaders in
`nsbi.examples.physics`, `load_arrays` samples randomly when `sample_size` is set and thus cannot be
used for predict with a non-null `predict_sample_size`, while `load_arrays_ordered` takes the same
arguments and keeps input row order.

#### Numerator and denominator

The two hypotheses are supplied as two separate files — the numerator (label 1) and the denominator
(label 0) of the ratio the classifier learns:

```yaml
datamodule:
  numerator_events: "/path/to/signal.csv"       # label 1
  denominator_events: "/path/to/background.csv" # label 0
  sample_size: 10_000_000                       # events sampled from each file (null = all)
```

Within each split, the two classes' weights are separately normalized to sum to 1 hence why it is a
*balanced* datamodule.

#### Splits

```yaml
datamodule:
  train_size: 0.6
  val_size: 0.2
  wi_fit_size: 0.0   # > 0 carves out the extra split used by ensemble weight-fitting
  test_size: 0.2
  data_dir: "/path/to/mydatadir"
```

The four sizes are **relative fractions**, normalized by their sum — they need not add to 1.
`0.6/0.2/0/0.2` and `3/1/0/1` describe the same 60/20/0/20 split. `train_size`, `val_size` and
`test_size` must all be `> 0`; `wi_fit_size: 0` disables the fourth split and reduces to the usual
three-way split.

`prepare_data()` performs the split once, fits a `StandardScaler` on the training features, and
writes both to `data_dir`:

```text
data_dir/
├── scaler.pkl
├── events_numerator_train.pkl      events_denominator_train.pkl
├── events_numerator_val.pkl        events_denominator_val.pkl
├── events_numerator_test.pkl       events_denominator_test.pkl
└── events_numerator_wi_fit.pkl     events_denominator_wi_fit.pkl   (only if wi_fit_size > 0)
```

These pickle files are used as a cache. `setup()` regenerates them only when `scaler.pkl` is absent.
If you change fields like `sample_size` or the split fractions then you must clear `data_dir` (or
point at a new one), otherwise you will silently keep using the previously generated cache.

`data_dir` must already exist. The datamodule checks it at the start of every stage and raises
`NotADirectoryError` if it doesn't exist.

`bootstrap: true` bootstrap-resamples the *training* split only, seeded by `random_state`; the val /
wi_fit / test splits stay fixed. This is what makes ensemble members differ, and the ensemble
tranining enables it for you automatically.

## The model

The `model` block builds the Lightning module that is trained, and its keys are simply the
constructor arguments of whatever class `model._target_` specifies.

### CARL

`src/nsbi/models/carl.py` — an MLP classifier trained to separate the numerator from the denominator
hypothesis. Its sigmoid output `s(x)` is the likelihood-ratio estimate, read off as
`log r(x) = log[ s(x) / (1 - s(x)) ]`.

It consumes the `(x, y, w)` batches the datamodule provides.

#### Configuration

```yaml
model:
  _target_: nsbi.models.carl.CARL
  n_features: 16
  n_layers: 16
  n_nodes: 1024
  learning_rate: 1e-4
```

#### Architecture

An input `Linear(n_features, n_nodes)` + SiLU, then `n_layers` hidden `Linear(n_nodes, n_nodes)` +
SiLU blocks, then `Linear(n_nodes, 1)` + sigmoid. All `Linear` layers get Xavier-uniform weights and
zero bias.

#### Loss and optimizer

Weighted binary cross-entropy, `(BCE * w).sum() / w.sum()` — using the same expression for train /
val / test, logged as `train_loss` / `val_loss` / `test_loss`. NAdam at `learning_rate`, with
`ReduceLROnPlateau` on `val_loss` (factor 0.1, patience 5, stepped per epoch).

`validation_step` and `test_step` return a dict per batch — `y_hat` (the model output `s(x)`), `y`,
`w`, and `kin` (the input feature matrix `x`) — alongside the loss. Lightning passes it to
`on_validation_batch_end` / `on_test_batch_end`, which is how `ClosureMetricsCallback` gets its
inputs; see [validate.md](validate.md). `training_step` returns only the loss.

#### Feature-count consistency

The following three things should agree:

1. the number of columns in the `X` your datamodule loader returns;
2. `model.n_features`;
3. `callbacks.closure_metrics.feature_names` — one name per column, **in the same order**.

## Running a single training

```bash
uv run nsbi -f configs/conf_tune_carl.yaml
```

### Stages

| `stage` | What happens |
| --- | --- |
| `fit` | Train from scratch. Any `ckpt_file` left in the config is ignored, so you cannot accidentally resume. |
| `finetune` | Load weights from `ckpt_file` (**required**) into a fresh optimizer and train. |
| `resume` | Find the latest checkpoint under `ckpt_path` and continue training, restoring optimizer and epoch. |
| `test` | Load the latest checkpoint under `ckpt_path` and run `trainer.test` on the test split. |
| `predict` | Score input files with the latest checkpoint under `ckpt_path` — or with every ensemble member (`predict.use_ensemble: true`) — writing one score file per input. See [predict.md](predict.md). |

`test` and `predict` are forced onto `devices=1, num_nodes=1` to avoid issues with distributed
samplers.

### Where checkpoints go, and how they are found

`ckpt_path` is a **directory** to search (unlike Lightning's own `ckpt_path` argument, which takes a
file). It is resolved in this order:

1. an explicit top-level `ckpt_path` in the config;
2. the `dirpath` of the **first** configured `ModelCheckpoint` callback
3. otherwise, reconstructed from the logger (`<logger.save_dir>/<logger.name>`) or, with no logger,
   from `<trainer.default_root_dir>/checkpoints` — where `default_root_dir` itself falls back to the
   working directory when unset.

Searching is recursive and picks the **most recently created** `.ckpt` file. If a logger exposes an
experiment id (WandB), it is prefixed onto the configured checkpoint filename so runs do not
collide.

### Loggers and callbacks

The example config uses `CSVLogger` and four callbacks: two `ModelCheckpoint`s (best 5 by
`val_loss`, best 1 by `train_loss`), `EarlyStopping` on `val_loss` with patience 20, and
`ClosureMetricsCallback`.

## Hyperparameter optimization

Set `do_hpo_tune: true` and configure the `hpo_tune` block:

```yaml
hpo_tune:
  num_samples: 20
  scheduler:
    _target_: ray.tune.schedulers.ASHAScheduler
    max_t: 50
    grace_period: 5
    reduction_factor: 2
  search_space:
    n_layers: "randint:5,15"
    n_nodes: "qrandint:100,1000,4"
    learning_rate: "loguniform:1e-5,3e-2"
  monitor: ["val_loss", "val_l1_energy_ws"]
  cpus_per_worker: 2
  single_device_gpu_fraction_per_trial: 1
  storage_path: null      # distributed strategies only: Ray Train RunConfig storage root
  scaling:
    strategy: "single_device"
```

**Search space keys must exist under `model:`** — sampled values are applied onto `cfg.model`, and
anything else is skipped with a warning. The string values are parsed by
`nsbi.utils.ray_utils.parse_dist`:

| Spec | Ray Tune equivalent |
| --- | --- |
| `randint:5,15` | `tune.randint(5, 15)` |
| `qrandint:100,1000,4` | `tune.qrandint(100, 1000, 4)` |
| `uniform:0,1` | `tune.uniform(0, 1)` |
| `loguniform:1e-5,3e-2` | `tune.loguniform(1e-5, 3e-2)` |
| `choice:a,b,c` | `tune.choice([...])` (numeric-looking entries are converted) |
| `grid:1,2,3` | `tune.grid_search([...])` |

`hpo_tune.monitor` lists the metrics reported back to Tune; every name must match a key that lands
in `trainer.callback_metrics` (e.g. `val_loss`, or a closure metric like `val_l1_energy_ws`). Trials
are *ranked* by `callbacks.model_checkpoint.monitor` / `.mode`, and when using ASHA it prunes on the
reported metrics.

Two execution paths, selected by `hpo_tune.scaling.strategy`:

- **`single_device`** (or unset) — one trial per GPU. `single_device_gpu_fraction_per_trial` must be
  in `(0, 1]`; a fraction lets Ray pack several trials onto one physical GPU (Ray does not enforce
  VRAM, so they must actually fit).
- **`ddp` / `fsdp` / `deepspeed_stage_{i}`** — each trial trains across
  `scaling.num_workers × scaling.gpus_per_worker` GPUs via a Ray Train `TorchTrainer`. **This path
  currently requires `RAY_TRAIN_V2_ENABLED=0` exported before the process starts** (it is read at
  import time). Note only DDP has been validated to work as the other strategies involving model
  sharding are not worth looking into unless the model size is large.

The run connects to an already-running Ray cluster via `ray.init(address="auto")` if there is one,
and otherwise starts a local one.

## Ensembling (*w<sub>i</sub>f<sub>i</sub>*)

Implements *w<sub>i</sub>f<sub>i</sub>* ensembling
([arXiv:2506.00113](https://arxiv.org/abs/2506.00113)): train M independent CARL members on
bootstrapped training draws, then fit a linear combination of their log-ratios on a held-out split.
The combined estimator is

$$
\log r(x) = \sum_i w_i f_i(x) + w_\text{const},
\qquad
f_i(x) = \log\!\left[\frac{s_i(x)}{1 - s_i(x)}\right]
$$

where

- $x$ — one event, as the feature vector the datamodule loader returns.
- $r(x)$ — the ensemble's estimate of the likelihood ratio between the numerator and denominator
  hypotheses.
- $i$ — the member index, running over the $M$ trained members.
- $s_i(x)$ — the sigmoid output of member $i$: the estimated probability that the event came from
  the numerator rather than the denominator hypothesis.
- $f_i(x)$ — that member's log-ratio, the logit of $s_i(x)$.
- $w_i$ — the fitted weight on member $i$.
- $w_\text{const}$ — a single fitted offset.

It is a two-phase pipeline. **Training requires `datamodule.wi_fit_size > 0`**, or the fit will have
no data to fit on.

```yaml
ensemble:
  size: 16
  cpus_per_worker: 2
  single_device_gpu_fraction_per_member: 1
  storage_path: /path/to/storage    # members land in <storage_path>/ensemble/member_{i}
  monitor: ["val_loss"]             # single_device only: extra metrics in the Tune status table
  scaling:
    strategy: "single_device"       # or ddp
    num_workers: 1
    gpus_per_worker: 1
  fit:
    eps: 1e-7                       # guards log[s/(1-s)] as s -> 1
    dtype: "float64"
    rescale_fit_weights: false
    method: "l-bfgs"
    options:
      max_iter: 1000
      gtol: 1e-10
      xtol: 1e-10
      disp: 1
    tolerance_check: false          # opt-in diagnostic: re-fit with tighter tolerances
    tolerance_check_factor: 10.0    # divides each tolerance above
    tolerance_check_pct: 5.0        # warn if any weight moves by more than this percent
```

### Phase 1 — train the members (`do_ensemble_train: true`)

Members are enumerated as a Ray Tune grid over the per-member seed (`ensemble.seed` or `seed`, plus
the member index), one trial per member. Each member sets `datamodule.bootstrap=true` with its own
`random_state`, so it trains on a different resampled draw while val / wi_fit / test stay fixed and
shared.

The `scaling.strategy` dispatch mirrors HPO: `single_device` gives each member one GPU (or a
fraction, via `single_device_gpu_fraction_per_member` in `(0, 1]`), while the distributed strategies
train each *individual* member across `num_workers × gpus_per_worker` GPUs — and carry the same
`RAY_TRAIN_V2_ENABLED=0` requirement.

When every member has finished, a manifest `<storage_path>/ensemble/members.json` maps member index
→ checkpoint path. This decouples the fit from each path's on-disk layout (Lightning
`ModelCheckpoint` directories for `single_device`, Ray Train checkpoints for distributed).

What the manifest records differs by path. The distributed path asks Ray Train for the best
checkpoint by `callbacks.model_checkpoint`'s `monitor` / `mode`. The single-device path instead
records whichever `.ckpt` under the member's directory was **created last**
(`find_latest_checkpoint`).

> **Note:** Perhaps this checkpoint selection can be synchronized at some point.

### Phase 2 — fit the weights (`do_ensemble_fit: true`)

Every member is loaded from the manifest and evaluated on the scaler-transformed `wi_fit` split. The
ensemble weights `w` (M member weights plus a constant offset — unrelated to the per-event weights
the members train on) minimize the symmetrized MLC loss, solved full-batch with `torchmin.minimize`
from [`pytorch-minimize`](https://github.com/rfeinman/pytorch-minimize).

Notes on configuring the solver:

- Nothing is defaulted on the framework side. `method` and `options` are forwarded verbatim to
  `torchmin.minimize`, so anything the chosen method accepts is settable. Whatever `options` omits
  runs using torchmin's own defaults.
- `dtype` sets the precision only for the fit.
- `rescale_fit_weights: true` divides the `wi_fit` event weights by their common mean for the fit
  only.

The fit also computes the asymptotic sandwich covariance of the ensemble weights, and writes
`<storage_path>/ensemble/weights.pkl` containing `w` (shape `M+1`), `cov` (`M+1 × M+1`), `size` (M),
and `eps`.

### Running both phases

Enabling both flags chains them in one job — the fit runs only if every member trained:

```bash
uv run nsbi -f configs/conf_tune_carl.yaml -c \
    do_ensemble_train=true do_ensemble_fit=true \
    ensemble.size=16 ensemble.storage_path=/path/to/storage \
    datamodule.wi_fit_size=0.1 datamodule.test_size=0.1
```

Set only `do_ensemble_fit=true` to re-fit weights over an already-trained ensemble — useful when
tuning the solver options without having to do a full retrain.

## Running on an HPC batch system

> **Note:** The directory `scripts/` currently holds examples for Polaris at ALCF, which schedules
> with PBS and should be treated as a template to copy and adapt rather than a general-purpose
> launcher. **A configurable launcher for arbitrary sites is on the [TODO](../TODO.md).**

The framework itself is site-agnostic — a single training run is plain PyTorch Lightning, and the
Ray-enabled paths (distributed HPO, ensemble training) connect to whatever Ray cluster is already
running.

### Example: Polaris (PBS)

The Polaris scripts come in pairs — a `qsub_*` job script that requests nodes and sets network
environment, and an `exec_*` script that runs on each node:

| Job script | Exec script | Purpose |
| --- | --- | --- |
| `qsub_nsbi_polaris.sh` | `exec_nsbi_polaris.sh` | Single training run |
| `qsub_nsbi_polaris_hpo.sh` | `exec_nsbi_polaris_hpo.sh` | Distributed HPO |
| `qsub_nsbi_polaris_ensemble.sh` | `exec_nsbi_polaris_ensemble.sh` | Ensemble train + fit |

**The paths inside are user-specific and must be edited before use** — `NSBI_SOURCE_PATH`,
`CONFIG_YAML`, the data CSVs, `DATA_DIR`, `STORAGE_PATH`, and the PBS `-A` / `-o` / `-e` directives.

The multi-node scripts follow one pattern: `mpiexec` launches one rank per node; the rank with
`PALS_NODEID == 0` starts the Ray head (`ray start --head`) and then runs the CLI, while the other
ranks join with `ray start --address=$MASTER_ADDR:$RAY_PORT` and idle until the head goes away.

Each `qsub_*` script exports the fabric settings before launching — `NCCL_SOCKET_IFNAME=hsn`, the
other `NCCL_*` tuning, and the `FI_CXI_*` / `FI_MR_*` libfabric variables for Slingshot — and in the
multi-node scripts `mpiexec` forwards them to every rank it launches. Each `exec_*` rank then adds:

- the ALCF HTTP proxy (needed for outbound traffic);
- `TMPDIR=/tmp`, which avoids Ray's `OSError: AF_UNIX path too long`;
- `RAY_TRAIN_V2_ENABLED=0` in the HPO and ensemble scripts, since the distributed paths need it.

## Examples of common overrides

```bash
# Point at different data without editing the YAML
-c datamodule.numerator_events=a.csv datamodule.denominator_events=b.csv datamodule.data_dir=/scratch

# Smoke test on CPU with a tiny model
-c trainer.accelerator=cpu trainer.max_epochs=1 datamodule.sample_size=10000 model.n_layers=2 model.n_nodes=32

# Resume the run whose checkpoints live in a known directory
-c stage=resume ckpt_path=/path/to/checkpoints

# Fine-tune from one specific checkpoint
-c stage=finetune ckpt_file=/path/to/epoch=76-val_loss=0.50.ckpt
```

Note that dotlist overrides *create* keys that do not exist rather than erroring, so a typo such as
`model.learningrate=1e-3` is silently ignored by the model.
