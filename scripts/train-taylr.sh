#!/usr/bin/env bash

python -m train.taylr --accelerator gpu --sample-size 1200000 --batch-size 128 --learning-rate 1e-4 --coefficient 1 --analysis 4l --features l1_pt l1_eta l1_phi l1_energy l2_pt l2_eta l2_phi l2_energy l3_pt l3_eta l3_phi l3_energy l4_pt l4_eta l4_phi l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python -m train.taylr --accelerator gpu --sample-size 1200000 --batch-size 128 --learning-rate 1e-4 --coefficient 2 --analysis 4l --features l1_pt l1_eta l1_phi l1_energy l2_pt l2_eta l2_phi l2_energy l3_pt l3_eta l3_phi l3_energy l4_pt l4_eta l4_phi l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python -m train.taylr --accelerator gpu --sample-size 1200000 --batch-size 128 --learning-rate 1e-4 --coefficient 3 --analysis 4l --features l1_pt l1_eta l1_phi l1_energy l2_pt l2_eta l2_phi l2_energy l3_pt l3_eta l3_phi l3_energy l4_pt l4_eta l4_phi l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python -m train.taylr --accelerator gpu --sample-size 1200000 --batch-size 128 --learning-rate 1e-4 --coefficient 4 --analysis 4l --features l1_pt l1_eta l1_phi l1_energy l2_pt l2_eta l2_phi l2_energy l3_pt l3_eta l3_phi l3_energy l4_pt l4_eta l4_phi l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi