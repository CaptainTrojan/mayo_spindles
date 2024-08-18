#!/bin/bash

# Check if the dataset and number of repeats are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <dataset> <number_of_repeats>"
  echo "Options for dataset: mayoieeg, dreams"
  exit 1
fi

# Dataset and number of repeats
dataset=$1
repeats=$2

# Set data path and additional arguments based on the dataset
if [ "$dataset" == "mayoieeg" ]; then
  data_args="--data hdf5_data"
elif [ "$dataset" == "dreams" ]; then
  data_args="--data DREAMS_HDF5 --annotator_spec any"
else
  echo "Invalid dataset option. Choose either 'mayoieeg' or 'dreams'."
  exit 1
fi

# models=("mlp" "cnn" "rnn" "gru" "lstm" "cdil" "autoformer" "crossformer" "dlinear" "etsformer" "fedformer" "film" "fret")
models=("cdil")
share_bottleneck=("True" "False")
hidden_sizes=(32 64 128)
conv_dropouts=(0.0 0.2 0.5)
end_dropouts=(0.0 0.2 0.5)

for model in "${models[@]}"
do
    for shb in "${share_bottleneck[@]}"
    do
        for hidden_size in "${hidden_sizes[@]}"
        do
            for conv_dropout in "${conv_dropouts[@]}"
            do
                for end_dropout in "${end_dropouts[@]}"
                do
                    for ((i=0; i<repeats; i++))
                    do
                        qsub -v "args=$data_args --epochs 1000 --patience 60 --model $model --share_bottleneck $shb --hidden_size $hidden_size --conv_dropout $conv_dropout --end_dropout $end_dropout" 1_run_instance.sh
                    done
                done
            done
        done
    done
done
