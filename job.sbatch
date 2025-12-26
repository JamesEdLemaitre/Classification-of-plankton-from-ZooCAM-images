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
rsync -r --exclude logs --exclude denoising --exclude logslurms . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 8e2625a98a1a5a81bef545979f582993ead71d54"
cd $TMPDIR/code
git checkout 8e2625a98a1a5a81bef545979f582993ead71d54


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


echo "Training"
python -m torchtmpl.main configs/tmpfcs1mhhf-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
