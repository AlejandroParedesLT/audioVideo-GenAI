#!/bin/bash
#SBATCH -t 1:00:00  # time requested in hour:minute:second
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000 #24 a6000, v100, a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out

srun hostname
srun date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/final_project_distillLLM/venv

srun bash -c "source $VENV_DIR/bin/activate && jupyter notebook --no-browser --port=8888 --ip=0.0.0.0"

# sbatch submit_llm.sh && JID=`squeu -u $USER -h -o%A` && sleep 5 && head slurm=$JID.out --lines=25