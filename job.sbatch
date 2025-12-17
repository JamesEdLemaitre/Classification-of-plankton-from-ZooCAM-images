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
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude .git --exclude __pycache__ --exclude "*.pyc" --exclude venv . $TMPDIR/code

echo "Checking out the correct version of the code commit_id d067336ff63e555072f4efa4a2059476a3a6636c"
cd $TMPDIR/code
git checkout d067336ff63e555072f4efa4a2059476a3a6636c

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main config-sample.yaml train

if [[ $? != 0 ]]; then
    exit -1
fi
