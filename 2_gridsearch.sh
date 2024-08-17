#!/bin/bash

# models=("mlp" "cnn" "rnn" "gru" "lstm" "cdil" "autoformer" "crossformer" "dlinear" "etsformer" "fedformer" "film" "fret")
models=("cdil")
share_bottleneck=("True" "False")

for model in "${models[@]}"
do
    for shb in "${share_bottleneck[@]}"
    do
        qsub -v "args=--epochs 1000 --model $model --share_bottleneck $shb" 1_run_instance.sh
    done
done

