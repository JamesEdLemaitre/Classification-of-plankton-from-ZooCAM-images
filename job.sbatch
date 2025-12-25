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
rsync -r --exclude logs --exclude configs --exclude denoising --exclude logslurms . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 333f698b5115743ac8d4d6c1c5c8aaef15bfb15c"
cd $TMPDIR/code
git checkout 333f698b5115743ac8d4d6c1c5c8aaef15bfb15c


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


echo "Training"
python -m torchtmpl.main configs/tmp7micsulx-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
