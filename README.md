# Higgs off-shell interpretation

## Requirements

- MCFM datasets:
    - $gg\to ZZ$ continuum background
    - $gg\to h^{\ast}\to ZZ$ signal
    - $(gg\to h^{\ast}\to ZZ) \times (gg\to ZZ)$ signal-background interference
    - $gg(\to h^{\ast})\to ZZ$ signal+background+interference

## Setup

### Installing `poetry`
To install [`poetry`](https://python-poetry.org/docs/), you first need to install `pipx` in your `$HOME` directory:

```bash
/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python -m pip install --user pipx
```

Then, `poetry` can be installed with the following:

```bash
pipx install poetry
```

Lastly, you can use the python installed at NERSC:

```bash
poetry env use /global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python
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
