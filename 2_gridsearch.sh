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
                        qsub -v "args=--epochs 1000 --patience 100 --model $model --share_bottleneck $shb --hidden_size $hidden_size --conv_dropout $conv_dropout --end_dropout $end_dropout" 1_run_instance.sh
                    done
                done
            done
        done
    done
done