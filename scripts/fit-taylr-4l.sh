#!/bin/bash

for i in {1..4}; do
   j=0
   k=0
   echo "c6_${i}_ct_${j}_cg_${k}"
   dir="taylr_c6_${i}_ct_${j}_cg_${k}"
   coeff="[${i},${j},${k}]"
   mkdir -p "$dir" && cd "$dir"
   python -m nsbi.taylr fit \
      --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
      --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
      --data.component sbi \
      --data.coefficient "$coeff" \
      --data.sample_size 6_000_000 \
      --data.batch_size 1024 \
      --model.learning_rate 1e-3 \
      --model.n_layers 10 \
      --model.n_nodes 100 \
      --trainer.accelerator gpu \
      --trainer.devices 4 \
      --trainer.max_epochs 500 \
      --seed_everything 42
   cd ..
done

for j in {1..2}; do
   i=0
   k=0
   echo "c6_${i}_ct_${j}_cg_${k}"
   dir="taylr_c6_${i}_ct_${j}_cg_${k}"
   coeff="[${i},${j},${k}]"
   mkdir -p "$dir" && cd "$dir"
   python -m nsbi.taylr fit \
      --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
      --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
      --data.component sbi \
      --data.coefficient "$coeff" \
      --data.sample_size 6_000_000 \
      --data.batch_size 1024 \
      --model.learning_rate 1e-3 \
      --model.n_layers 10 \
      --model.n_nodes 100 \
      --trainer.accelerator gpu \
      --trainer.devices 4 \
      --trainer.max_epochs 500 \
      --seed_everything 42
   cd ..
done

for k in {1..2}; do
   i=0
   j=0
   echo "c6_${i}_ct_${j}_cg_${k}"
   dir="taylr_c6_${i}_ct_${j}_cg_${k}"
   coeff="[${i},${j},${k}]"
   mkdir -p "$dir" && cd "$dir"
   python -m nsbi.taylr fit \
      --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
      --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
      --data.component sbi \
      --data.coefficient "$coeff" \
      --data.sample_size 6_000_000 \
      --data.batch_size 1024 \
      --model.learning_rate 1e-3 \
      --model.n_layers 10 \
      --model.n_nodes 100 \
      --trainer.accelerator gpu \
      --trainer.devices 4 \
      --trainer.max_epochs 500 \
      --seed_everything 42
   cd ..
done

for i in {1..2}; do
   j=1
   k=0
   echo "c6_${i}_ct_${j}_cg_${k}"
   dir="taylr_c6_${i}_ct_${j}_cg_${k}"
   coeff="[${i},${j},${k}]"
   mkdir -p "$dir" && cd "$dir"
   python -m nsbi.taylr fit \
      --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
      --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
      --data.component sbi \
      --data.coefficient "$coeff" \
      --data.sample_size 6_000_000 \
      --data.batch_size 1024 \
      --model.learning_rate 1e-3 \
      --model.n_layers 10 \
      --model.n_nodes 100 \
      --trainer.accelerator gpu \
      --trainer.devices 4 \
      --trainer.max_epochs 500 \
      --seed_everything 42
   cd ..
done

for i in {1..2}; do
   j=0
   k=1
   echo "c6_${i}_ct_${j}_cg_${k}"
   dir="taylr_c6_${i}_ct_${j}_cg_${k}"
   coeff="[${i},${j},${k}]"
   mkdir -p "$dir" && cd "$dir"
   python -m nsbi.taylr fit \
      --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
      --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
      --data.component sbi \
      --data.coefficient "$coeff" \
      --data.sample_size 6_000_000 \
      --data.batch_size 1024 \
      --model.learning_rate 1e-3 \
      --model.n_layers 10 \
      --model.n_nodes 100 \
      --trainer.accelerator gpu \
      --trainer.devices 4 \
      --trainer.max_epochs 500 \
      --seed_everything 42
   cd ..
done

i=0
j=1
k=1
echo "c6_${i}_ct_${j}_cg_${k}"
dir="taylr_c6_${i}_ct_${j}_cg_${k}"
coeff="[${i},${j},${k}]"
mkdir -p "$dir" && cd "$dir"
python -m nsbi.taylr fit \
   --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
   --data.events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/events_analyzed.csv' \
   --data.component sbi \
   --data.coefficient "$coeff" \
   --data.sample_size 6_000_000 \
   --data.batch_size 1024 \
   --model.learning_rate 1e-3 \
   --model.n_layers 10 \
   --model.n_nodes 100 \
   --trainer.accelerator gpu \
   --trainer.devices 4 \
   --trainer.max_epochs 500 \
   --seed_everything 42
cd ..