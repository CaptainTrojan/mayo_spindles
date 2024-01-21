#!/bin/bash

models=("cdil" "cdil_rnn" "cdil_gru" "cdil_lstm")
filter_bandwidths=("True" "False")
avg_window_sizes=("5" "10" "20")

for model in "${models[@]}"
do
    for filter_bandwidth in "${filter_bandwidths[@]}"
    do
        for avg_window_size in "${avg_window_sizes[@]}"
        do
            qsub -v "args=--model $model --epochs 1000 --filter_bandwidth $filter_bandwidth --avg_window_size $avg_window_size" 1_run_instance.sh
        done
    done
done

