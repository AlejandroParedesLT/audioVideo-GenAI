#!/bin/bash

echo "Starting the distributed training script..."

# Dynamically discover the node rank and master address
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NNODES=${SLURM_NNODES}
MASTER_PORT=${3-2012}
GPUS_PER_NODE=4
# Setup the distributed training command with torchrun
DISTRIBUTED_ARGS="--nproc_per_node=$GPUS_PER_NODE \
                  --nnodes=$NNODES \
                  --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR \
                  --master_port=$MASTER_PORT"

# Running the job with torchrun
CMD="torchrun ${DISTRIBUTED_ARGS} distributed_training.py --rank $NODE_RANK --world_size $(($NNODES * $GPUS_PER_NODE))"
${CMD}
