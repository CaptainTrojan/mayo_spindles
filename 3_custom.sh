#!/bin/bash

# Check if the dataset, number of repeats, and other arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <dataset> <number_of_repeats> <other_args>"
  echo "Options for dataset: mayoieeg, dreams"
  exit 1
fi

# Dataset, number of repeats, and other arguments
dataset=$1
repeats=$2
other_args=$3

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

for ((i=0; i<repeats; i++))
do
    qsub -v "args=\"$data_args $other_args\"" 1_run_instance.sh
done