#!/bin/bash

# Check if the dataset and number of repeats are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <dataset> <number_of_parallel_samplers>"
  echo "Options for dataset: mayoieeg, dreams"
  exit 1
fi

# Dataset and number of repeats
dataset=$1
repeats=$2

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data \$SCRATCHDIR/mayo_spindles/hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data \$SCRATCHDIR/mayo_spindles/DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

for ((i=0; i<repeats; i++))
do
    qsub -v "args=$data_args --epochs 1000 --patience 60 --model cdil" 1_run_instance.sh
done
