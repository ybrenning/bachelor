# Active Learning Lab v2

A framework for running active learning and/or text classification experiments.

## Requirements

Python 3.7+  
Some methods will require a CUDA GPU

## Docker

(Im Hauptverzeichnis: active-learning-lab-v2-2023)

bash docker/build.sh

## Installation

```bash
pip -r requirements.txt
```

For certain features you might need optional requirements:

```bash
pip -r requirements-optional.txt
```

**[!]** [Make sure that the installed PyTorch and torchtext versions match](https://github.com/pytorch/text#installation)
(and that the installed PyTorch version matches your local CUDA version).

## Usage

While this framework can potentially support multiple types of experiments, 
in the following we only describe the active learning portion (located at `active_learning_lab/experiments/active_learning`):

The working dir is expected to be top-level directory of this project. *(You could change this but then the following commands might not work without alterations.)**

### Prerequisites

This application uses [mlflow](https://www.mlflow.org/) to log results. 
Before the first run you must create a mlflow experiment:

```bash
# replace [NEW_EXPERIMENT_NAME] by some string for example by new_experiment
mlflow experiments create -n [NEW_EXPERIMENT_NAME]
```

On creation you will be shown the experiment ID (which you could also lookup later). 
The  results will later be written to `mlruns/[experiment ID]`.

### General

The general syntax is as follows:

```bash
python -m active_learning_lab.experiments.active_learning.active_learning_runner [config_file] [arguments]
```

This is an argument parser and you can simple pass `-h` to print the full help:

```bash
python -m active_learning_lab.experiments.active_learning.active_learning_runner -h
```

### Examples

```bash
python -m active_learning_lab/experiments/active_learning/active_learning_runner active_learning_lab.config.active_learning.test
--dataset_name
ag-news
--classifier_name
transformer
--dataset_kwargs
max_length=60
--query_strategy
lc-ent
```

### Possible Values

| Parameter | Values | 
| --------- | ------ |
| dataset_name | ag-news, cr, mr, subj, trec-6  (*) |
| classifier_name | svm, kimcnn, transformer | 
| query_strategy | TODO | 

(*) Some others as well but I am not sure if they are complete, 
i.e. they might not work for every type of features.

### Inspecting the Results

The results are similar to the predecessor of this project (v1) which was partly used 
for [webis-de/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers](https://github.com/webis-de/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers/blob/main/USAGE.md#inspecting-the-results) where the results are described as well.
