#!/usr/bin/env bash

 python -m nsbi.carl fit \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "met", "met_phi"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/*l*v/qq*/analyzed.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz2l2v/ggZZ_sbi/analyzed.csv' \
    --data.sample_size 10_000_000 \
    --data.batch_size 2048 \
    --model.learning_rate 1e-4 \
    --model.n_layers 20 \
    --model.n_nodes 100 \
    --trainer.devices 1 \
    --trainer.max_epochs 500 \
    --seed_everything 42