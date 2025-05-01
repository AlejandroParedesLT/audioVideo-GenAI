#!/bin/bash

srun hostname
srun date

export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

GPU_IDS="0,1,2,3"

accelerate launch --gpu_ids $GPU_IDS ./src/utils/train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --cache_dir /dev/shm/hf-home \
  --instance_data_root /home/users/ap794/finalCS590-text2audiovideo/MM-Diffusion/data10/call_of_duty/train \
  --dataset_name my-awesome-name/my-awesome-dataset \
  --caption_column TEXT_DESCRIPTION \
  --video_column VIDEO_PATH \
  --id_token <ID_TOKEN> \
  --validation_prompt "<ID_TOKEN> Spiderman swinging over buildings:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 10 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision fp16 \
  --output_dir /cogvideox-lora \
  --height 480 --width 720 --fps 8 --max_num_frames 49 --skip_frames_start 0 --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer Adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb