#!/usr/bin/env bash

python rolypoly.py --accelerator gpu --sample-size 1000000 --batch-size 128 --features mandelstam_s mandelstam_t mandelstam_u --events /raven/u/taepa/mcfm/MCFM-10.3/Bin/ggZZ2e2m_sbi/events.csv --component sbi