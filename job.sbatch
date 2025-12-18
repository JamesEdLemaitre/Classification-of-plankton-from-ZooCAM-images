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

echo "Checking out the correct version of the code commit_id 4fe8d8a2fb5b23748bc55f4246d55613eef6ff9c"
cd $TMPDIR/code
git checkout 4fe8d8a2fb5b23748bc55f4246d55613eef6ff9c


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_23/Desktop/Deep learning/Projet/kaggle_challenge_sdi/configs/tmpvc64klft-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
