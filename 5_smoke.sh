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
  data_args="--data \$SCRATCHDIR/mayo_spindles/hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data \$SCRATCHDIR/mayo_spindles/DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

qsub -v "args=$data_args --patience 5 --model cdil" 1_run_instance.sh