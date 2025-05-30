#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
BASE_PATH=${1-"."}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True  --video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--video_size 16,3,64,64 --audio_size 1,25600 --learn_sigma False --num_channels 128 
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True
--use_scale_shift_norm True"

#  --use_fp16 True


SRMODEL_FLAGS="--sr_attention_resolutions 8,16,32  --large_size 256  
--small_size 64 --sr_learn_sigma True 
--sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 
--sr_resblock_updown True --use_fp16 True --sr_use_scale_shift_norm True"

# Modify --devices according your GPU number
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 64 --save_type mp4  --devices 0
--batch_size 4   --is_strict True --sample_fn dpm_solver"

SR_DIFFUSION_FLAGS="--sr_diffusion_steps 1000  --sr_sample_fn ddim  --sr_timestep_respacing ddim25"

# Modify the following paths to your own paths

# model002000 model004000 model006000 model008000 model010000 model012000 model014000 model016000 model018000 model020000

MULTIMODAL_MODEL_PATH="./data10/concerts_audiovideo_dataset/debug/model020000.pt"

# /data10/call_of_duty/debug/model005000.pt
# /model007000.pt

SR_MODEL_PATH="./data10/models/AIST++_SR.pt"
OUT_DIR="./data10/concerts_audiovideo_dataset/model020000"
REF_PATH="./data10/concerts_audiovideo_dataset/unittest"

# --use-hwthread-cpus
export NCCL_DEBUG=INFO
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# NUM_GPUS=1
# mpiexec -n $NUM_GPUS python3 py_scripts/multimodal_sample_sr_multiprocessing.py  \
# $MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS --ref_path ${REF_PATH} \
# --output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH}  --sr_model_path ${SR_MODEL_PATH} 

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/py_scripts/multimodal_sample_sr_multiprocessing.py $MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS --ref_path ${REF_PATH} --output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH}  --sr_model_path ${SR_MODEL_PATH}"  
${CMD}
