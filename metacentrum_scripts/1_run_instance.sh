#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:ngpus=1:mem=8gb:scratch_local=20gb:cuda_version=12.4
#PBS -N spindle-detector
#PBS -m n

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

cd /storage/plzen1/home/sejakm/mayo_spindles
cp -r /storage/plzen1/home/sejakm/mayo_spindles $SCRATCHDIR/mayo_spindles
cd $SCRATCHDIR

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $SCRATCHDIR/conda
. $SCRATCHDIR/conda/etc/profile.d/conda.sh
conda create -n p311 -c conda-forge python=3.11 cmake
conda activate p311
export TMPDIR=$SCRATCHDIR
export PYTHONPATH=$PYTHONPATH:$SCRATCHDIR/mayo_spindles:$SCRATCHDIR/mayo_spindles/mayo_spindles
export WANDB_API_KEY=42f902b34c4d27b2d2887fbb261df5ed89594e58
export POSTGRES_PW=31optuna42rocks
cd mayo_spindles
pip install -r requirements.txt

ls -alh

python mayo_spindles/run.py --num_workers 0 $args

clean_scratch
