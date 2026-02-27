#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(commit_id, configpath, nruns):
    # Utiliser seulement le nom du fichier
    config_filename = os.path.basename(configpath)
    return f"""#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --exclude=sh[12,03]
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir="$(pwd)"
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

# echo "Copying dataset"
# rsync -r /mounts/datasets/datasets/2025-ZooCamChallenge/ $TMPDIR/data/

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
#rsync -r --exclude denoising --exclude logslurms . $TMPDIR/code
rsync -r --exclude denoising --exclude logslurms \
    --filter='+ logs/RGB/***' \
    --filter='- logs/**' \
    . $TMPDIR/code

echo "Checking out the correct version of the code commit_id {commit_id}"
# on se place dans le répertoire d'un noeud de calcul, on était à la base a la racine du projet
cd $TMPDIR/code
git checkout {commit_id}


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


#IL FAUT PAS DE DOSSIER CONFIGS dans le chemin je pense, si parce qu'enfaite on créer des configs temp qui sont dans le dossier configs
echo "Training"
python -m src.torchtmpl.main configs/{config_filename} train

if [[ $? != 0 ]]; then
    exit -1
fi

echo "Generating submission file"
python -m src.torchtmpl.main configs/{config_filename} test

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

"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | grep -v '^logslurms/' | wc -l) + $(git diff --name-only --cached | grep -v '^logslurms/'  | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [2, 3]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f'cp "{configpath}" "{tmp_configfilepath}"')

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_configfilepath, nruns))

