#!/bin/bash

qsub -v "args=--patience 5 --model cdil" 1_run_instance.sh

