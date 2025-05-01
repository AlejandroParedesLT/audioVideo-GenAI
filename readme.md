# Diffusion Models for Joint and Sequential Audio-Video Generation 

This repository provides an implementation of advanced multimodal diffusion frameworks for synchronized audio-video generation, including MM-Diffusion and a novel two-step sequential pipeline combining CogVideoX for video and MM-Audio for audio generation.

The work builds upon cutting-edge research in multimodal generative modeling and proposes new methods, datasets, and evaluation benchmarks for high-fidelity and temporally aligned audio-visual synthesis.

🚀 Highlights
📦 Two new datasets released:

🎮 Call of Duty Game Dataset (13 hrs)

🎤 Concerts Around the Globe Dataset (64 hrs)

🌀 MM-Diffusion trained from scratch on curated datasets for joint audio-video generation

🧩 Latent MM-Diffusion experiments using pretrained audio and video VAE backbones

🔁 Two-step text → video → audio pipeline using CogVideoX and MM-Audio for aligned synthesis

📊 Evaluated with Fréchet Audio Distance (FAD) and Fréchet Video Distance (FVD)

## Overview

The joint multimodal diffusion model (MM-Diffusion) is designed to:
- Generate synchronized video and audio pairs
- Train on distributed systems using PyTorch's DistributedDataParallel
- Support super-resolution enhancement of generated videos
- Provide various sampling methods for quality and efficiency

## Features

- **Multimodal Generation**: Creates coherent audio-video pairs where content is synchronized
- **Distributed Training**: Scales across multiple GPUs using NCCL backend
- **Super-Resolution**: Optional video enhancement to upscale low-resolution outputs
- **Evaluation**: Built-in metrics for content quality assessment

## Installation

```bash
# Clone the repository
git clone https://github.com/AlejandroParedesLT/audioVideo-GenAI.git
cd audioVideo-GenAI

# Install dependencies Ideally use two different virtual environments
pip install -r requirements_sequentialDiffusion.txt
pip install -r requirements_unconditionalDiffusion.txt
```

## Usage

### Training

For each of the .sh files named cluster replace the virtual environment directory: export VENV_DIR=$HOME/finalCS590-text2audiovideo/venv according to your needs

To train the multimodal diffusion model:

```bash
sbatch cluster_audioVideo_concerts.sh && JID=`squeue -u $USER -h -o%A` && sleep 5 && head slurm-$JID.out --lines=25
```

### Sampling/Inference

To generate audio-video pairs with the trained model simply uncomment the following line in the file cluster_audioVideo_concerts.sh:

```bash
# srun bash -c "source $VENV_DIR/bin/activate && bash ./ssh_scripts/multimodal_sample_sr_concerts.sh"
```

To run the two-step audio video generation:

```bash
sbatch cluster_audioVideo_concerts.sh && JID=`squeue -u $USER -h -o%A` && sleep 5 && head slurm-$JID.out --lines=25
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

This implementation was forked from the paper

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
