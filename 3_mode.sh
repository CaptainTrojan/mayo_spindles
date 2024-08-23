#!/bin/bash

# Check if the dataset, number of repeats, and optuna study are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <dataset> <number_of_parallel_samplers> <mode> <optuna_study> "
  echo "Options for dataset: mayoieeg, dreams"
  echo "Options for mode: detection_only, shared_bottleneck, separate_bottleneck"
  exit 1
fi

# Dataset, number of repeats, and optuna study
dataset=$1
repeats=$2
mode=$3
optuna_study=$4

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi


# Set mode options
if [ "$mode" == "detection_only" ]; then
  mode_args="--hidden_size 57 --end_dropout 0.34 --conv_dropout: 0.33 --mode detection_only"
elif [ "$mode" == "shared_bottleneck" ]; then
  mode_args="--hidden_size 57 --end_dropout 0.158 --conv_dropout: 0.062 --mode shared_bottleneck"
elif [ "$mode" == "separate_bottleneck" ]; then
  mode_args="--hidden_size 60 --end_dropout 0.124 --conv_dropout: 0.35 --mode separate_bottleneck"
else
  echo "Invalid mode option. Choose either 'detection_only', 'shared_bottleneck', or 'separate_bottleneck'."
  exit 1
fi

for ((i=0; i<repeats; i++))
do
    qsub -v "args=\"$data_args --epochs 1000 --patience 60 --model cdil --optuna_study $optuna_study $mode_args\"" 1_run_instance.sh
done
