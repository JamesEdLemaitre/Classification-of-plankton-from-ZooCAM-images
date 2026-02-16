#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir="$(pwd)"
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude denoising --exclude logslurms . $TMPDIR/code

# --exclude logs

echo "Checking out the correct version of the code commit_id 85daac413e17874fdbeb08a0bd2d819022e5cf3b"
# on se place dans le répertoire d'un noeud de calcul, on était à la base a la racine du projet
cd $TMPDIR/code
git checkout 85daac413e17874fdbeb08a0bd2d819022e5cf3b


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


#IL FAUT PAS DE DOSSIER CONFIGS dans le chemin je pense, si parce qu'enfaite on créer des configs temp qui sont dans le dossier configs
echo "Training"
python -m torchtmpl.main configs/tmpw96egolm-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi

echo "Generating submission file"
python -m torchtmpl.main configs/tmpw96egolm-config.yml test

echo "Copying results back"
mkdir -p "$current_dir/logs"
mkdir -p "$current_dir/results"

if [ -d logs ]; then
    rsync -r logs/ "$current_dir/logs/"
else
    echo "Warning: logs directory not found"
fi

if [ -d results ]; then
    rsync -r results/ "$current_dir/results/"
else
    echo "Warning: results directory not found"
fi

