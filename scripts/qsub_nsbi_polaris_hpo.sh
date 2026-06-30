#!/bin/bash
#PBS -A ScalingHEPAI
#PBS -l filesystems=home:eagle
#PBS -l place=scatter:excl
#PBS -l select=1:system=polaris:ncpus=64:ngpus=4
#PBS -l walltime=10:00
#PBS -N nsbi_job
#PBS -o /home/nkang
#PBS -e /home/nkang
#PBS -q debug

EXEC_PATH=/home/nkang/exec_nsbi_polaris_hpo.sh

### User specific setup above

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
# export NCCL_NET="AWS Libfabric"
export NCCL_SOCKET_IFNAME=hsn

export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# MPI topology
NODES=$(wc -l < "${PBS_NODEFILE}")
RANKS_PER_NODE=4
TOTAL_RANKS=$((NODES * RANKS_PER_NODE))
CPUS_PER_RANK=$((64 / RANKS_PER_NODE))

# --cpu-bind depth needed for setting cpus per rank
# https://docs.alcf.anl.gov/polaris/data-science/frameworks/pytorch/#multi-gpu-multi-node-scale-up
EXEC_ARGS="$NODES $RANKS_PER_NODE $CPUS_PER_RANK"
mpiexec -hostfile $PBS_NODEFILE -n $NODES -ppn 1 --cpu-bind depth -d 64 $EXEC_PATH $EXEC_ARGS
