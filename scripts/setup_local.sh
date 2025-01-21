#!/usr/bin/env bash

module purge

python3.10 -m venv venv_tf
source venv_tf/bin/activate

pip install tensorflow[and-cuda]==2.14
pip install numpy==1.23.5
pip install vector hist scikit-learn pandas matplotlib ipykernel