#!/bin/bash

models=("mlp" "cnn" "rnn" "gru" "lstm" "cdil" "autoformer" "crossformer" "dlinear" "etsformer" "fedformer" "film" "frets" "informer" "itransformer" "lightts" "micn" "nonstationary_transformer" "patchtst" "pyraformer" "reformer" "tide" "timesnet" "transformer")

for model in "${models[@]}"
do
    echo qsub -v "args=--model $model --epochs 1000" 1_run_instance.sh
done

