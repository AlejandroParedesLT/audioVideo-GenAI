#! /bin/bash

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${2-2012}
NNODES=${SLURM_NNODES}
NODE_RANK=${SLURM_NODEID}
GPUS_PER_NODE=4
BASE_PATH=${1-"."}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


#################64 x 64 uncondition###########################################################
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.1 
--video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--video_size 16,3,64,64 --audio_size 1,25600 --learn_sigma False --num_channels 128
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True 
--use_scale_shift_norm True --num_workers 4"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 0.0001 --batch_size 4 
--devices 0,1,2,3 --log_interval 100 --save_interval 500 --use_db False " #--schedule_sampler loss-second-moment


DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 100 --save_type mp4 --sample_fn ddpm" 

# Modify the following pathes to your own paths
# DATA_DIR="./data10/AIST++_crop/train/"
# OUTPUT_DIR="./data10/debug"
DATA_DIR="./data10/call_of_duty/train/"
OUTPUT_DIR="./data10/call_of_duty/debug/"

#export NCCL_P2P_DISABLE=1

# NUM_GPUS=2
# -n $NUM_GPUS



# mpiexec --use-hwthread-cpus -n $NUM_GPUS python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS
# mpiexec --use-hwthread-cpus  python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$NUM_GPUS py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS

# --rdzv_backend=nccl --nproc_per_node=$NUM_GPUS

#torchrun py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS --rdzv_backend=c10d --rdzv_endpoint=localhost:29500

#torchrun --nnodes=1 --nproc_per_node=2 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS


# torchrun --nproc_per_node=$NUM_GPUS --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS

#mpiexec --hostfile ./hosts.txt -n $NUM_GPUS python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS


export NCCL_DEBUG=INFO
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
#CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/py_scripts/multimodal_train_multiprocessing.py  --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS"
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/py_scripts/multimodal_train_multiprocessing.py  --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS"
${CMD}
