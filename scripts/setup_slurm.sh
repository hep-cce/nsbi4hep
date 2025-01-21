#!/usr/bin/env bash

module purge
module load anaconda/3/2023.03 
module load tensorflow/gpu-cuda-12.1/2.14.0 protobuf/4.24.0 mkl/2023.1 cuda/12.1 cudnn/8.9.2 nccl/2.18.3 tensorrt/8.6.1 tensorboard/2.13.0 keras/2.14.0 keras-preprocessing/1.1.2

python -m venv --system-site-packages venv
source venv/bin/activate
pip install vector
deactivate