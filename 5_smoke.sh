#!/bin/bash

# Check if the dataset is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset>"
  echo "Options for dataset: mayoieeg, dreams"
  exit 1
fi

# Dataset
dataset=$1

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

qsub -v "args=$data_args --patience 5 --model cdil --optuna_study SD-optuna-smoke2-metacentrum-$dataset --optuna_timeout 3000 --optuna_params mode@categorical@detection_only,shared_bottleneck,separate_bottleneck hidden_size@int@50@70" 1_run_instance.sh
