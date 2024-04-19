#!/bin/bash
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=40g
#SBATCH --time=2-00:00
#SBATCH --container-image=registry.webis.de\#code-lib/public-images/yb63tadu/active-learning-lab-v2:2.0.0
#SBATCH --container-name=active-learning-lab-v2
#SBATCH --container-workdir=/mnt/ceph/storage/data-tmp/current/yb63tadu
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/yb63tadu:/mnt/ceph/storage/data-tmp/current/yb63tadu
#SBATCH --container-env=TORCH_HOME,TORCH_HOME_CACHE,HF_HOME,HF_DATASETS_CACHE,HF_MODULE_CACHE,HUGGINGFACE_HUB_CACHE,MXNET_HOME,GENSIM_DATA_DIR,MLFLOW_TRACKING_URI,TMP_DIR
#SBATCH --container-writable

DATASET=$1
CLASSIFIER_NAME=$2
QUERY_STRATEGY=$3

SCRATCH=/var/tmp/$USER

ENTRY_POINT="active_learning_lab.experiments.active_learning.active_learning_runner"
CODE_DIR="/mnt/ceph/storage/data-tmp/current/yb63tadu/active-learning-lab-v2"

MLFLOW_TRACKING_URI="/mnt/ceph/storage/data-tmp/current/yb63tadu/mlruns"
TMP_DIR="/var/tmp/yb63tadu/active-learning-lab-v2"

TORCH_HOME="$TMP_DIR/.torch-home"
TORCH_HOME_CACHE="$TMP_DIR/.torch-home/.cache"
HF_HOME="$TMP_DIR/.hf-home"
HF_DATASETS_CACHE="$TMP_DIR/.hf-datasets"
HF_MODULE_CACHE="$TMP_DIR/.hf-modules"
HUGGINGFACE_HUB_CACHE="$TMP_DIR/.hf-hub"
MXNET_HOME="$TMP_DIR/.hf-modules"
GENSIM_DATA_DIR="$TMP_DIR/.gensim-data"

MKDIR_CMD="mkdir -p $TMP_DIR $HF_HOME $HF_DATASETS_CACHE $HF_MODULE_CACHE $HUGGINGFACE_HUB_CACHE $MXNET_HOME $GENSIM_DATA_DIR $TORCH_HOME $TORCH_HOME_CACHE"

INNER_CMD="cd $CODE_DIR && PYTHONPATH=$CODE_DIR:/opt/conda/lib/python3.8/site-packages/"
INNER_CMD="${INNER_CMD} MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
INNER_CMD="${INNER_CMD} TMP_DIR=$TMP_DIR"
INNER_CMD="${INNER_CMD} TORCH_HOME=$TORCH_HOME"
INNER_CMD="${INNER_CMD} TORCH_HOME_CACHE=$TORCH_HOME_CACHE"
INNER_CMD="${INNER_CMD} HF_HOME=$HF_HOME"
INNER_CMD="${INNER_CMD} HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
INNER_CMD="${INNER_CMD} HF_MODULE_CACHE=$HF_MODULE_CACHE"
INNER_CMD="${INNER_CMD} HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
INNER_CMD="${INNER_CMD} MXNET_HOME=$MXNET_HOME"
INNER_CMD="${INNER_CMD} GENSIM_DATA_DIR=$GENSIM_DATA_DIR"
INNER_CMD="${INNER_CMD} LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtinfo.so.6"
INNER_CMD="${INNER_CMD} python3 -m ${ENTRY_POINT} active_learning_lab.config.active_learning.yb_coresets "
INNER_CMD="${INNER_CMD} --dataset_name $DATASET "
INNER_CMD="${INNER_CMD} --classifier_name $CLASSIFIER_NAME "
INNER_CMD="${INNER_CMD} --query_strategy $QUERY_STRATEGY "
INNER_CMD="${INNER_CMD} ${@:4}"

echo $INNER_CMD
bash -c "conda install -c conda-forge ncurses && $MKDIR_CMD && $INNER_CMD"