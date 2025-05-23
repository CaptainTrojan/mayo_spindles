#!/bin/bash

# Check if the dataset, number of repeats, optuna study, and aggregate runs are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
  echo "Usage: $0 <dataset> <number_of_parallel_samplers> <optuna_study> <aggregate_runs>"
  echo "Options for dataset: mayoieeg, dreams"
  exit 1
fi

# Dataset, number of repeats, optuna study, and aggregate runs
dataset=$1
repeats=$2
optuna_study=$3
aggregate_runs=$4

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data hdf5_data_corrected"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

for ((i=0; i<repeats; i++))
do
    qsub -v "args=\"$data_args --epochs 1000 --patience 60 --model cdil --aggregate_runs $aggregate_runs --optuna_study $optuna_study --optuna_params mode@categorical@detection_only,shared_bottleneck,separate_bottleneck end_dropout@float@0.0@0.5 conv_dropout@float@0.0@0.5 hidden_size@int@50@100\"" 1_run_instance.sh
done
