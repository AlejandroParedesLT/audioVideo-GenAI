#!/bin/bash
#SBATCH -t 0:30:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --constraint=a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out
#SBATCH --signal=B:SIGTERM@1800

srun --ntasks-per-node=1 hostname
srun --ntasks-per-node=1 date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/finalCS590-text2audiovideo/venv
srun --ntasks-per-node=1 bash -c "nvidia-smi"

if [ -f "./ssh_scripts/test_multinode.sh" ]; then
    echo "File exists."
else
    echo "File does not exist."
fi

srun bash -c "source $VENV_DIR/bin/activate && bash /home/users/ap794/finalCS590-text2audiovideo/MM-Diffusion/ssh_scripts/test_multinode.sh"