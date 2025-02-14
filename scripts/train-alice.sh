#!/usr/bin/env bash

python alice.py --accelerator gpu --events ggZZ2e2m_bkg/events.csv --numerator-process sig --denominator-process bkg --sample-size 100000 --n-layers 1000 --n-layers 10 --batch-size 128 --seed 42