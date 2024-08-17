#!/bin/bash

qsub -v "args=--epochs 5 --model cdil" 1_run_instance.sh

