#!/bin/bash
#PBS -N mayo-spindles
#PBS -l select=1:ncpus=10:ngpus=1:gpu_mem=10gb:mem=64gb:scratch_local=60gb
#PBS -l walltime=23:59:00
#PBS -q gpu
#PBS -m n

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

cd /storage/plzen1/home/sejakm/mayo_spindles
cp -r /storage/plzen1/home/sejakm/mayo_spindles $SCRATCHDIR/mayo_spindles
cd $SCRATCHDIR

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $SCRATCHDIR/conda
. $SCRATCHDIR/conda/etc/profile.d/conda.sh
conda activate base
export TMPDIR=$SCRATCHDIR
export PYTHONPATH=$PYTHONPATH:$SCRATCHDIR/mayo_spindles
export WANDB_API_KEY=42f902b34c4d27b2d2887fbb261df5ed89594e58
cd mayo_spindles
pip install -r requirements.txt

ls -alh

python run.py --data $SCRATCHDIR/mayo_spindles/data $args

clean_scratch
