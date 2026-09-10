#!/bin/bash

NODES=$1
RANKS_PER_NODE=$2
CPUS_PER_RANK=$3

NSBI_SOURCE_PATH=/home/nkang/workdir/nsbi4hep
CONFIG_YAML=/home/nkang/workdir/nsbi4hep/configs/conf_tune_carl.yaml

### User specific setup above

cd $NSBI_SOURCE_PATH

#module use /soft/modulefiles
#module load conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base
source $NSBI_SOURCE_PATH/.venv/bin/activate

NODELIST=$(cat $PBS_NODEFILE | sort | uniq)
MASTER_ADDR=$(echo $NODELIST | cut -d ' ' -f 1)

#export OMP_NUM_THREADS=1
#export MASTER_PORT=29500
#export WORLD_SIZE=$PMI_SIZE
#export NODE_RANK=$PALS_NODEID
#export LOCAL_RANK=$PMI_LOCAL_RANK
#echo $NODE_RANK $LOCAL_RANK $WORLD_SIZE $MASTER_ADDR $MASTER_PORT

# Avoid OSError: AF_UNIX path too long
# https://docs.alcf.anl.gov/polaris/known-issues/?h=af+unix#set-tmpdir-to-avoid-af_unix-path-too-long-error
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1

uv run --active nsbi -f $CONFIG_YAML
