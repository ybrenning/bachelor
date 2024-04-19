#!/bin/bash

DATASET=$1

MAIN_DIR="/mnt/ceph/storage/data-tmp/current/yb63tadu"
TMP_DIR="/var/tmp/yb63tadu/active-learning-lab-v2"
CODE_DIR="/mnt/ceph/storage/data-tmp/current/yb63tadu/active-learning-lab-v2"

HF_HOME="$TMP_DIR/.hf-home"
HF_DATASETS_CACHE="$TMP_DIR/.hf-datasets"
HF_MODULE_CACHE="$TMP_DIR/.hf-modules"
HUGGINGFACE_HUB_CACHE="$TMP_DIR/.hf-hub"
MXNET_HOME="$TMP_DIR/.hf-modules"
GENSIM_DATA_DIR="$TMP_DIR/.gensim-data"
srun mkdir -p $TMP_DIR $HF_HOME $HF_DATASETS_CACHE $HF_MODULE_CACHE $HUGGINGFACE_HUB_CACHE $MXNET_HOME $GENSIM_DATA_DIR

INNER_CMD="PYTHONPATH=$CODE_DIR:/opt/conda/lib/python3.8/site-packages/"
INNER_CMD="${INNER_CMD} cd $CODE_DIR && python3 dr.py"
echo $INNER_CMD

srun --container-image=registry.webis.de#code-lib/public-images/active-learning-lab-v2:1.11.0 \
  --container-name=active-learning-lab-v2 \
  --container-mounts=$MAIN_DIR:$MAIN_DIR,$TMP_DIR:$TMP_DIR \
  --export=ALL,HF_HOME=$HF_HOME,HF_DATASETS_CACHE=$HF_DATASETS_CACHE,HF_MODULE_CACHE=$HF_MODULE_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,MXNET_HOME=$MXNET_HOME,GENSIM_DATA_DIR=$GENSIM_DATA_DIR,SMALL_TEXT_TEMP=$TMP_DIR \
  --mem=$MEM \
  bash -c "$INNER_CMD"
