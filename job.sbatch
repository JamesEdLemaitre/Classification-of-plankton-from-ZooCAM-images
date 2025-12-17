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

echo "Copying the config file"
cp /usr/users/sdim/sdim_23/Desktop/Deep learning/Projet/pytorch_template_code-main/configs/tmpe3g9cyu9-config.yml $TMPDIR/code/config_to_use.yaml

echo "Checking out the correct version of the code commit_id 20cb5d63bab6d96014e1a7c5472ec95002757c01"
cd $TMPDIR/code
git checkout 20cb5d63bab6d96014e1a7c5472ec95002757c01


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main config_to_use.yaml train

if [[ $? != 0 ]]; then
    exit -1
fi
