#!/usr/bin/env bash
#SBATCH --job-name=zz4l/ggZZ_int
#SBATCH --output=zz4l/ggZZ_int/%j.out
#SBATCH --error=zz4l/ggZZ_int/%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=general

module purge
module load gcc/14

export OMP_STACKSIZE=16000
export OMP_NUM_THREADS=24

./mcfm zz4l/input_ggZZ_int.ini -general%runstring=zz4l -general%rundir=zz4l/ggZZ_int
