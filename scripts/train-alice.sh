#!/usr/bin/env bash

python carl.py --accelerator gpu --sample-size 100000 --batch-size 128 --seed 42

# python carl.py train --accelerator gpu --sample-size 100000 --batch-size 128 --seed 42