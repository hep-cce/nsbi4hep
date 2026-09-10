#!/bin/bash

NODES=$1
RANKS_PER_NODE=$2
CPUS_PER_RANK=$3

NSBI_SOURCE_PATH=/home/nkang/workdir/nsbi4hep
CONFIG_YAML=/home/nkang/workdir/nsbi4hep/configs/conf_tune_carl.yaml

STRATEGY=single_device
ENSEMBLE_SIZE=16
SEED=0
NUMERATOR_CSV=/lus/eagle/projects/ScalingHEPAI/test_Nathan/wifi_data/sig_small.csv
DENOMINATOR_CSV=/lus/eagle/projects/ScalingHEPAI/test_Nathan/wifi_data/bkg_small.csv
DATA_DIR=/home/nkang/test_notebook/data
STORAGE_PATH=/home/nkang/test_notebook/storage
mkdir -p $DATA_DIR $STORAGE_PATH

### User specific setup above

cd $NSBI_SOURCE_PATH

#module use /soft/modulefiles
#module load conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base
source $NSBI_SOURCE_PATH/.venv/bin/activate

# proxy settings
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

MASTER_ADDR=$(head -1 $PBS_NODEFILE)
RAY_PORT=6379

# Avoid OSError: AF_UNIX path too long
# https://docs.alcf.anl.gov/polaris/known-issues/?h=af+unix#set-tmpdir-to-avoid-af_unix-path-too-long-error
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export RAY_enable_worker_prestart=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# The distributed ensemble path (STRATEGY=ddp/fsdp/deepspeed_*) uses the Ray Train V1
export RAY_TRAIN_V2_ENABLED=0

uv run --active ray stop --force 2>/dev/null || true

if [ "$PALS_NODEID" == "0" ]; then
    uv run --active ray start --head --port=$RAY_PORT --temp-dir=/tmp/ray_session
    if [ "$NODES" -gt 1 ]; then
        sleep 5
    fi

    uv run --active nsbi -f $CONFIG_YAML -c \
        do_ensemble_train=true \
        do_ensemble_fit=true \
        seed=$SEED \
	ensemble.fit.tolerance_check=true \
	ensemble.fit.tolerance_check_pct=5 \
	ensemble.fit.rescale_fit_weights=false \
	ensemble.fit.method="l-bfgs" \
	ensemble.fit.options.disp=2 \
	ensemble.fit.options.gtol=1e-10 \
	ensemble.fit.options.xtol=1e-10 \
        ensemble.size=$ENSEMBLE_SIZE \
        ensemble.scaling.strategy=$STRATEGY \
        ensemble.storage_path=$STORAGE_PATH \
        model.n_layers=3 \
        model.n_nodes=64 \
        model.learning_rate=1e-3 \
        datamodule.numerator_events=$NUMERATOR_CSV \
        datamodule.denominator_events=$DENOMINATOR_CSV \
        datamodule.sample_size=100000 \
        datamodule.batch_size=1024 \
        datamodule.data_dir=$DATA_DIR \
        datamodule.train_size=0.3 \
        datamodule.val_size=0.2 \
        datamodule.wi_fit_size=0.3 \
        datamodule.test_size=0.2 \
        trainer.max_epochs=40

    uv run --active ray stop --force
else
    uv run --active ray start --address=$MASTER_ADDR:$RAY_PORT --temp-dir=/tmp/ray_session

    # Keep the worker alive only while the head is up
    sleep 30
    while uv run --active ray status --address=$MASTER_ADDR:$RAY_PORT >/dev/null 2>&1; do
        sleep 10
    done
fi
