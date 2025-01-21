
# module load python-waterboa/2024.06

module purge
module load anaconda/3/2023.03 
module load pytorch/gpu-cuda-12.1/2.2.0 pytorch-lightning/2.1.2 protobuf/4.24.0 mkl/2023.1 nccl/2.18.3 tensorrt/8.6.1 tensorboard/2.13.0 

python -m venv --system-site-packages venv
source venv/bin/activate
pip install lightning
pip install vector
deactivate