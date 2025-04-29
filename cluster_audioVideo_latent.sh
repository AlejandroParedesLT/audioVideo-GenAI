#!/bin/bash
#SBATCH -t 1:00:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out
#SBATCH --signal=B:SIGTERM@1800

srun hostname
srun date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/finalCS590-text2audiovideo/venv
#mkdir -p $VENV_DIR


export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

#srun python3 -m venv $VENV_DIR
srun bash -c "nvidia-smi"
srun bash -c "source \$HOME/finalCS590-text2audiovideo/text2Movie/.env && source \$VENV_DIR/bin/activate && huggingface-cli login --token \$HF_TOKEN && wandb login \$WANDB_TOKEN && ./ssh_scripts/multimodal_train_latent.sh"