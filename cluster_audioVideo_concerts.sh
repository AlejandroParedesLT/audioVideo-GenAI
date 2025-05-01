#!/bin/bash
#SBATCH -t 2:00:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out
#SBATCH --signal=B:SIGTERM@1800

srun hostname
srun date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/finalCS590-text2audiovideo/venv
#mkdir -p $VENV_DIR

#srun python3 -m venv $VENV_DIR
srun bash -c "nvidia-smi"

srun bash -c "source $VENV_DIR/bin/activate && bash ./ssh_scripts/multimodal_train_concerts.sh"

# srun bash -c "source $VENV_DIR/bin/activate && bash ./ssh_scripts/multimodal_sample_sr_concerts.sh"

# srun bash -c "source $VENV_DIR/bin/activate && bash ./ssh_scripts/evaluation/exec_eval.py"