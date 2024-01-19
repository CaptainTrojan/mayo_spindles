#!/bin/bash

models=("mlp" "cnn" "rnn" "gru" "lstm" "cdil" "autoformer" "crossformer" "dlinear" "etsformer" "fedformer" "film" "fret")
filter_bandwidths=("True" "False")

for model in "${models[@]}"
do
    for filter_bandwidth in "${filter_bandwidths[@]}"
    do
        qsub -v "args=--model $model --epochs 1000 --filter_bandwidth $filter_bandwidth" 1_run_instance.sh
    done
done

