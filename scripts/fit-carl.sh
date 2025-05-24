#!/usr/bin/env bash

 python -m nsbi.carl fit \
    --data.analysis 4l \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz2l2v/qqZZ/events_*.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz2l2v/ggZZ_sbi/events_*.csv' \
    --data.sample_size 3_000_000 \
    --data.batch_size 1024 \
    --model.learning_rate 1e-5 \
    --model.n_layers 20 \
    --model.n_nodes 1000 \
    --trainer.devices 1 \
    --trainer.max_epochs 300 \
    --seed_everything 42

 python -m nsbi.carl fit \
    --data.analysis 2l2v \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "met", "met_phi"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz2l2v/qqZZ/events_*.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz2l2v/ggZZ_sbi/events_*.csv' \
    --data.sample_size 3_000_000 \
    --data.batch_size 1024 \
    --model.learning_rate 1e-5 \
    --model.n_layers 20 \
    --model.n_nodes 1000 \
    --trainer.devices 1 \
    --trainer.max_epochs 300 \
    --seed_everything 42