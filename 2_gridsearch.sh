#!/bin/bash

# Check if the number of repeats is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_repeats>"
  exit 1
fi

# Number of repeats
repeats=$1

# models=("mlp" "cnn" "rnn" "gru" "lstm" "cdil" "autoformer" "crossformer" "dlinear" "etsformer" "fedformer" "film" "fret")
models=("cdil")
share_bottleneck=("True" "False")

for model in "${models[@]}"
do
    for shb in "${share_bottleneck[@]}"
    do
        for ((i=0; i<repeats; i++))
        do
            qsub -v "args=--epochs 1000 --patience 100 --model $model --share_bottleneck $shb" 1_run_instance.sh
        done
    done
done