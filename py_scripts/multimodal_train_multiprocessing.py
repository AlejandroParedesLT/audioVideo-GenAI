"""
Train a diffusion model on audio-video pairs.
"""
import sys
import os
import argparse
import subprocess
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import debugpy  # Import debugpy for debugging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_diffusion import dist_util, logger
from mm_diffusion.multimodal_datasets_torchDistrib import load_data
from mm_diffusion.resample import create_named_schedule_sampler
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from mm_diffusion.multimodal_train_util import TrainLoop
from mm_diffusion.common import set_seed_logger_random

def setup():
    """Initialize the distributed backend"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def load_training_data(args):
    """Load dataset with distributed support."""
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        video_size=args.video_size,
        audio_size=args.audio_size,
        num_workers=args.num_workers,
        video_fps=args.video_fps,
        audio_fps=args.audio_fps
    )

    for video_batch, audio_batch in data:
        gt_batch = {"video": video_batch, "audio": audio_batch}
        yield gt_batch


def main():

    local_rank = setup()
    print(local_rank)
    print('World Size perceived from torch distributed: ', torch.distributed.get_world_size())
    args = create_argparser().parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]
    logger.configure(args.output_dir)
    print('Print devices')
    print(args.devices)
    dist_util.setup_dist(args.devices)
    
    args = set_seed_logger_random(args)

    if local_rank == 0:
        logger.log("Creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Fix suggested here: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    #model = DDP(model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if local_rank == 0:
        logger.log("Creating data loader...")

    data = load_training_data(args)

    if local_rank == 0:
        dist_util.check_nvidia_smi()

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        save_type=args.save_type,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        lr=args.lr,
        t_lr=args.t_lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_db=args.use_db,
        sample_fn=args.sample_fn,
        video_fps=args.video_fps,
        audio_fps=args.audio_fps,
    ).run_loop()

    cleanup()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.0,
        t_lr=1e-4,
        seed=42,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        num_workers=0,
        save_type="mp4",
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        devices=None,
        save_interval=10000,
        output_dir="",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_db=False,
        sample_fn="dpm_solver",
        frame_gap=1,
        video_fps=10,
        audio_fps=16000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
