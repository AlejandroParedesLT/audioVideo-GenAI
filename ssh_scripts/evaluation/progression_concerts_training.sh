#!/bin/bash

#SBATCH -t 1:00:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out
#SBATCH --signal=B:SIGTERM@1800

base_path=${1:-"."}

for MODEL_NAME in model002000 model004000 model006000 model008000 model010000 model012000 model014000 model016000 model018000 model020000
do
    echo "Running inference for MODEL_NAME: $MODEL_NAME"
    bash ./ssh_scripts/evaluation/progression_concerts_training.sh $MODEL_NAME $base_path
done


# # Iterate through the model names
# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     echo "Running inference for MODEL_NAME: $MODEL_NAME"
#     MODEL_NAME=$MODEL_NAME bash ./ssh_scripts/evaluation/progression_concerts_training.sh
# done



