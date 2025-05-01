#!/bin/bash
#SBATCH -t 1:00:00  # time requested in hour:minute:second
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
#mkdir -p $VENV_DIR


export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

srun bash -c "source \$HOME/finalCS590-text2audiovideo/text2Movie/.env"\
    " && source \$VENV_DIR/bin/activate && huggingface-cli login --token"\
    " \$HF_TOKEN && wandb login \$WANDB_TOKEN && "\
    " python ./src/inference.py --mmvideo_id=THUDM/CogVideoX-2b"\
    " --mmaudio_variant=large_44k_v2"\
    " --prompt='Relistic concert of a rock and roll band doing a live show in front of a crowd' "\
    " --duration=8.0"\
    " --output=./output"\
    " --num_steps=25"\
    " --seed=43" \
    "--video_name rock_roll_concert"

