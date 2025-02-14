# Higgs off-shell interpretation

## Requirements

- MCFM datasets:
    - $gg\to ZZ$ continuum background
    - $gg\to h^{\ast}\to ZZ$ signal
    - $(gg\to h^{\ast}\to ZZ) \times (gg\to ZZ)$ signal-background interference
    - $gg(\to h^{\ast})\to ZZ$ signal+background+interference

## Setup

The Python virtual environment can be setup as follows:

```sh
pip install -r requirements
pip install -e .
```

## Overview of packages in this repo

- `physics.simulation` : load MCFM datasets as a set of DataFrames and manipulate them in physically/statistically-coherent manner.
- `hzz.zpair, hzz.zz4l` implementations of physics analysis/processing of datasets.
- `hstar.c6` modify the predicted cross-section and probabilities of simulated events under Higgs self-coupling modification scenarios.

## Training NSBI models

```sh
python alice.py --accelerator gpu --events ggZZ2e2m_bkg/events.csv --numerator-process sig --denominator-process bkg --sample-size 100000 --n-layers 1000 --n-layers 10 --batch-size 128 --seed 42
```

## Examine outputs

See `notebooks/plot-alice-outputs.ipynb`

## Reweight diagnostic

See `notebooks/plot-alice-reweight.ipynb`

## Expected sensitivity from ground-truth vs. NSBI

(Coming soon!)
