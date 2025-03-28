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

### ALICE
```sh
python alice.py --accelerator gpu --events ggZZ2e2m_bkg/events.csv --numerator-process sig --denominator-process bkg --sample-size 100000 --n-layers 1000 --n-layers 10 --batch-size 128 --seed 42
```

### CARL
```sh
python carl.py --features "l1_pt" "l1_eta" "l1_phi" "l1_energy" "l2_pt" "l2_eta" "l2_phi" "l2_energy" "l3_pt" "l3_eta" "l3_phi" "l3_energy" "l4_pt" "l4_eta" "l4_phi" "l4_energy" --numerator-events ${HIGGS_BASEDIR}/../data/qqZZ2e2m.csv --denominator-events ${HIGGS_BASEDIR}/../data/ggZZ2e2m_sbi.csv --n-nodes 2000 --sample-size 2500000 --learning-rate 7e-6 --batch-size 2048
```

## Examine outputs

See `notebooks/plot-alice-outputs.ipynb`

## Reweight diagnostic

See `notebooks/plot-alice-reweight.ipynb`

## Expected sensitivity from ground-truth vs. NSBI

(Coming soon!)
