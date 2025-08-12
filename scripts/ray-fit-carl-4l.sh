#!/usr/bin/env bash
 python -m hpo.tune_carl \
    --resources.gpu_per_trial 0.5 --resources.cpu_per_trial 4 \
    --space.n_layers randint:6,12 \
    --space.n_nodes qrandint:128,1024,128 \
    --space.learning_rate loguniform:1e-5,1e-2 \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
    --data.numerator_events '/home/idies/workspace/Storage/dbollweg/persistent/zz4l/zz4l/qqZZ/analyzed.csv' \
    --data.denominator_events '/home/idies/workspace/Storage/dbollweg/persistent/zz4l/zz4l/ggZZ_sbi/analyzed.csv' \
    --data.sample_size 10_000_000 \
    --data.batch_size 2048 \
    --trainer.devices 1 \
    --trainer.max_epochs 5