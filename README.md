# Scaling Neural Simulation-Based Inference at High Performance Computing Centers for LHC analysis


## Installation
We use `uv` to manage our Python environment and dependencies.

```bash
uv venv
uv sync --with dev,docs
```

### Install the project
You can now set up the project's environment by navigating to your cloned repository's directory, installing the environment dependencies, and activating the environment:

```bash
poetry install --with dev,docs

eval $(poetry env activate)

## Overview of packages in this repo

- `physics.simulation` : load MCFM datasets as a set of DataFrames and manipulate them in physically/statistically-coherent manner.
- `hzz.zpair, hzz.zz4l` implementations of physics analysis/processing of datasets.
- `hstar.c6` modify the predicted cross-section and probabilities of simulated events under Higgs self-coupling modification scenarios.

## Training NSBI models

### ALICE
```sh
python alice.py --accelerator gpu --events ggZZ2e2m_bkg/events.csv --numerator-process sig --denominator-process bkg --sample-size 100000 --n-layers 1000 --n-layers 10 --batch-size 128 --seed 42
```

### CARL
```sh
python carl.py --features "l1_pt" "l1_eta" "l1_phi" "l1_energy" "l2_pt" "l2_eta" "l2_phi" "l2_energy" "l3_pt" "l3_eta" "l3_phi" "l3_energy" "l4_pt" "l4_eta" "l4_phi" "l4_energy" --numerator-events ${HIGGS_BASEDIR}/../data/qqZZ2e2m.csv --denominator-events ${HIGGS_BASEDIR}/../data/ggZZ2e2m_sbi.csv --n-nodes 1000 --n-layers 10 --sample-size 1000000 --learning-rate 1e-5 --batch-size 1024
```

## Examine outputs

See `notebooks/plot-alice-outputs.ipynb`

## Reweight diagnostic

See `notebooks/plot-alice-reweight.ipynb`

## Expected sensitivity from ground-truth vs. NSBI

(Coming soon!)
