#!/usr/bin/env bash

 python -m nsbi.carl fit \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/qqZZ/analyzed.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/analyzed.csv' \
    --data.sample_size 10_000_000 \
    --data.batch_size 2048 \
    --model.learning_rate 2e-5 \
    --model.n_layers 20 \
    --model.n_nodes 100 \
    --trainer.devices 8 \
    --trainer.max_epochs 500 \
    --seed_everything 42