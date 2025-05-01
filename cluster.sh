#!/bin/bash
#SBATCH -t 4:00:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out
#SBATCH --signal=B:SIGTERM@1800

srun hostname
srun date

# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/finalCS590-text2audiovideo/text2Movie/audiovid_venv

export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

# srun bash -c "source \$HOME/finalCS590-text2audiovideo/text2Movie/.env && source \$VENV_DIR/bin/activate && huggingface-cli login --token \$HF_TOKEN && wandb login \$WANDB_TOKEN && diffusers.sh"

# srun bash -c "source $HOME/finalCS590-text2audiovideo/text2Movie/.env && source $VENV_DIR/bin/activate && huggingface-cli login --token $HF_TOKEN && wandb login $WANDB_TOKEN && multi_generation.py"
srun bash -c "source $VENV_DIR/bin/activate && python ./multi_generation.py"