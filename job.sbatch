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

echo "Checking out the correct version of the code commit_id f2a056f9336adbde19e90355da60795e1632ab12"
cd $TMPDIR/code
git checkout f2a056f9336adbde19e90355da60795e1632ab12


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate
unset PYTHONPATH    
export PYTHONNOUSERSITE=1 

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_27/kaggle_challenge_sdi/configs/tmpx39_s3h1-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
