#!/usr/bin/env bash

python rolypoly.py --accelerator gpu --sample-size 1000000 --batch-size 128 --coefficient 1 --features l1_pt l2_pt l3_pt l4_pt l1_eta l2_eta l3_eta l4_eta l1_phi l2_phi l3_phi l4_phi l1_energy l2_energy l3_energy l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python rolypoly.py --accelerator gpu --sample-size 1000000 --batch-size 128 --coefficient 2 --features l1_pt l2_pt l3_pt l4_pt l1_eta l2_eta l3_eta l4_eta l1_phi l2_phi l3_phi l4_phi l1_energy l2_energy l3_energy l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python rolypoly.py --accelerator gpu --sample-size 1000000 --batch-size 128 --coefficient 3 --features l1_pt l2_pt l3_pt l4_pt l1_eta l2_eta l3_eta l4_eta l1_phi l2_phi l3_phi l4_phi l1_energy l2_energy l3_energy l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi

python rolypoly.py --accelerator gpu --sample-size 1000000 --batch-size 128 --coefficient 4 --features l1_pt l2_pt l3_pt l4_pt l1_eta l2_eta l3_eta l4_eta l1_phi l2_phi l3_phi l4_phi l1_energy l2_energy l3_energy l4_energy --events /ptmp/mpp/taepa/higgs-offshell-interpretation/data/ggZZ2e2m_sbi/events.csv --component sbi