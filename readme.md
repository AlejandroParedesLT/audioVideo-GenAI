# Multimodal Diffusion Model for Audio-Video Generation

This repository contains an implementation of a diffusion-based model for generating synchronized audio-video content. The project leverages distributed training for scalability and supports super-resolution enhancement of generated videos.

## Overview

This multimodal diffusion model is designed to:
- Generate synchronized video and audio pairs
- Train on distributed systems using PyTorch's DistributedDataParallel
- Support super-resolution enhancement of generated videos
- Provide various sampling methods for quality and efficiency

## Features

- **Multimodal Generation**: Creates coherent audio-video pairs where content is synchronized
- **Distributed Training**: Scales across multiple GPUs using NCCL backend
- **Flexible Sampling Methods**: Supports various diffusion samplers including:
  - DPM-Solver
  - DPM-Solver++ (with adaptive stepping)
  - DDPM
  - DDIM
- **Super-Resolution**: Optional video enhancement to upscale low-resolution outputs
- **Evaluation**: Built-in metrics for content quality assessment

## Installation

```bash
# Clone the repository
git clone https://github.com/AlejandroParedesLT/audioVideo-GenAI.git
cd multimodal-diffusion

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- PyTorch with CUDA support
- torchvision
- einops
- debugpy (for debugging)
- numpy
- scikit-image
- matplotlib

## Usage

### Training

To train the multimodal diffusion model:

```bash
torchrun --nproc_per_node=N multimodal_train_multiprocessing.py \
  --data_dir /path/to/data \
  --output_dir /path/to/output \
  --batch_size 16 \
  --num_workers 4 \
  --video_size 16,3,64,64 \
  --audio_size 16,1,44100 \
  --video_fps 10 \
  --audio_fps 16000 \
  --lr 1e-4 \
  --t_lr 1e-4 \
  --save_interval 10000 \
  --log_interval 100 \
  --sample_fn dpm_solver
```

Where `N` is the number of GPUs to use.

### Sampling/Inference

To generate audio-video pairs with the trained model:

```bash
torchrun --nproc_per_node=N multimodal_sample_sr_multiprocessing.py \
  --multimodal_model_path /path/to/model.pt \
  --output_dir /path/to/output \
  --batch_size 16 \
  --sample_fn dpm_solver \
  --sr_sample_fn dpm_solver \
  --video_size 16,3,64,64 \
  --audio_size 16,1,44100 \
  --video_fps 10 \
  --audio_fps 16000 \
  --all_save_num 1024
```

To run with super-resolution enhancement, add the SR model path:

```bash
torchrun --nproc_per_node=N multimodal_sample_sr_multiprocessing.py \
  --multimodal_model_path /path/to/model.pt \
  --sr_model_path /path/to/sr_model.pt \
  --large_size 256 \
  --output_dir /path/to/output \
  --batch_size 16 \
  --sample_fn dpm_solver \
  --sr_sample_fn dpm_solver \
  --video_size 16,3,64,64 \
  --audio_size 16,1,44100 \
  --video_fps 10 \
  --audio_fps 16000 \
  --all_save_num 1024
```

## Parameters

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_dir` | Directory containing training data | - |
| `output_dir` | Directory for saving checkpoints and logs | - |
| `batch_size` | Batch size for training | 1 |
| `video_size` | Video dimensions (frames,channels,height,width) | - |
| `audio_size` | Audio dimensions (frames,channels,samples) | - |
| `video_fps` | Video frames per second | 10 |
| `audio_fps` | Audio sample rate | 16000 |
| `lr` | Base learning rate | 0.0 |
| `t_lr` | Transformer learning rate | 1e-4 |
| `sample_fn` | Sampling method (dpm_solver, ddpm, ddim) | dpm_solver |
| `save_interval` | Steps between model checkpoints | 10000 |
| `log_interval` | Steps between logging | 100 |
| `use_fp16` | Use half precision for training | False |

### Sampling Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `multimodal_model_path` | Path to trained model checkpoint(s) | - |
| `sr_model_path` | Path to super-resolution model | - |
| `output_dir` | Directory for saving generated samples | - |
| `batch_size` | Batch size for sampling | 16 |
| `all_save_num` | Total number of samples to generate | 1024 |
| `large_size` | Size for super-resolution output | - |
| `sample_fn` | Sampling method for the multimodal model | dpm_solver |
| `sr_sample_fn` | Sampling method for super-resolution | dpm_solver |
| `save_type` | Format for saving videos (mp4, gif, etc.) | mp4 |

## Structure

Generated samples are organized as follows:

```
output_dir/
  └── model_name/
      ├── original/         # Original resolution videos
      ├── sr_mp4/           # Super-resolution enhanced videos
      ├── audios/           # Extracted audio files
      └── img/              # Individual video frames
```

## Sampling Methods

The model supports different diffusion sampling methods:

1. **DPM-Solver**: Fast sampling with better quality/speed trade-off
   - Default steps: 20
   - Order: 2-3

2. **DPM-Solver++**: Enhanced version with thresholding and adaptive stepping
   - Supports predict_x0 mode
   - Adaptive step size

3. **DDPM**: Original diffusion sampling method
   - Slower but sometimes more stable

4. **DDIM**: Denoising Diffusion Implicit Models
   - Faster than DDPM with controllable quality

## Distributed Training

The model uses PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training:

- NCCL backend for GPU communication
- Process group initialization for coordination
- Local rank assignment for device management

## Super-Resolution

The optional super-resolution model can enhance the quality of generated videos:

- Processes each frame independently
- Can upscale to arbitrary resolutions (specified by `large_size`)
- Uses the same diffusion sampling methods as the main model

## Acknowledgements

This implementation is from the paper

@misc{ruan2023mmdiffusionlearningmultimodaldiffusion,
      title={MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation}, 
      author={Ludan Ruan and Yiyang Ma and Huan Yang and Huiguo He and Bei Liu and Jianlong Fu and Nicholas Jing Yuan and Qin Jin and Baining Guo},
      year={2023},
      eprint={2212.09478},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.09478}, 
}

@inproceedings{ruan2022mmdiffusion,
author = {Ruan, Ludan and Ma, Yiyang and Yang, Huan and He, Huiguo and Liu, Bei and Fu, Jianlong and Yuan, Nicholas Jing and Jin, Qin and Guo, Baining},
title = {MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation},
year	= {2023},
booktitle	= {CVPR},
}