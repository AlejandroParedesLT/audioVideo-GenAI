# #@title Import required libraries
# import argparse
# import itertools
# import math
# import os
# from contextlib import nullcontext
# import random

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch.utils.data import Dataset

# import PIL
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
# from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.optimization import get_scheduler
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
# from PIL import Image
# from torchvision import transforms
# from tqdm.auto import tqdm
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# import bitsandbytes as bnb

# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows*cols

#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     grid_w, grid_h = grid.size
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     return grid

# from dataclasses import dataclass

# @dataclass
# class TrainingConfig:
#     image_size = 128  # the generated image resolution
#     train_batch_size = 16
#     eval_batch_size = 16  # how many images to sample during evaluation
#     num_epochs = 50
#     gradient_accumulation_steps = 1
#     learning_rate = 1e-4
#     lr_warmup_steps = 500
#     save_image_epochs = 10
#     save_model_epochs = 30
#     mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
#     output_dir = 'ddpm-butterflies-128'  # the model namy locally and on the HF Hub

#     push_to_hub = True  # whether to upload the saved model to the HF Hub
#     hub_private_repo = False  
#     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
#     seed = 0

# config = TrainingConfig()


# from datasets import load_dataset

# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")

# # Feel free to try other datasets from https://hf.co/huggan/ too! 
# # Here's is a dataset of flower photos:
# # config.dataset_name = "huggan/flowers-102-categories"
# # dataset = load_dataset(config.dataset_name, split="train")

# # Or just load images from a local folder!
# # config.dataset_name = "imagefolder"
# # dataset = load_dataset(config.dataset_name, data_dir="path/to/folder")

