#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id ac882c7e33fa0c62a9a8f27837b2d74e3563288a"
cd $TMPDIR/code
git checkout ac882c7e33fa0c62a9a8f27837b2d74e3563288a


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate
unset PYTHONPATH    
export PYTHONNOUSERSITE=1 

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_27/kaggle_challenge_sdi/configs/tmpakkjm2g9-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
