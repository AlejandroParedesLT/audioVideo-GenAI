import torch
import pytest
import functools
# from mm_diffusion.ALEJANDRO_multimodal_script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
# )

from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)

from mm_diffusion.resample import create_named_schedule_sampler
from mm_diffusion.fp16_util import MixedPrecisionTrainer
from mm_diffusion.train_util import log_loss_dict
from mm_diffusion import dist_util
from mm_diffusion.multimodal_datasets_torchDistrib import load_data
import torch
import torch.distributed as dist
from mm_diffusion.multimodal_train_util import TrainLoop
import os
# /home/users/ap794/finalCS590-text2audiovideo/MM-Diffusion/data10/call_of_duty/test/_huIixkio2c_chunk1.mp4
# Manually initiate huggingface login
from huggingface_hub import login
from dotenv import load_dotenv
# Load environment variables from .env fil
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))



@pytest.mark.parametrize("video_dir", ["./data10/concerts_audiovideo_dataset/unittest"])
def test_model_pipeline_step(video_dir):
    args = dict(
        video_size=[3, 16, 64, 64],
        audio_size=[1, 25600],
        learn_sigma=False,
        num_channels=64,
        num_res_blocks=2,
        channel_mult="1,2,2",
        num_heads=4,
        num_head_channels=32,
        num_heads_upsample=1,
        cross_attention_resolutions="2,4,8",
        cross_attention_windows="1,4,8",
        cross_attention_shift=True,
        video_attention_resolutions="2,4,8",
        audio_attention_resolutions="-1",
        dropout=0.1,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="100",  # model: "100" respacing
        
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,  # NOTE: commented in bash for main model, but used True in SR model

        # --- Super-Resolution Model flags (SRMODEL_FLAGS) ---
        sr_attention_resolutions="8,16,32",
        large_size=256,
        small_size=64,
        sr_learn_sigma=True,
        sr_num_channels=192,
        sr_num_heads=4,
        sr_num_res_blocks=2,
        sr_resblock_updown=True,
        sr_use_scale_shift_norm=True,
        
        # --- Super-Resolution Diffusion flags (SR_DIFFUSION_FLAGS) ---
        sr_diffusion_steps=1000,
        sr_sample_fn="ddim",
        sr_timestep_respacing="ddim25",

        # --- Training/Inference flags (DIFFUSION_FLAGS) ---
        all_save_num=64,
        save_type="mp4",
        devices="0",         # You hardcoded devices=0 (single GPU, or multi with MPI later)
        batch_size=4,
        is_strict=True,
        sample_fn="dpm_solver",

    )
    print("args", args)
    def init_distributed_mode():
        # Initialize the default process group (assuming NCCL backend for multi-GPU)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(dist.get_rank())  # Set the GPU for this rank
        # Set necessary environment variables for distributed setup
    os.environ["RANK"] = "0"  # Rank of this process (in multi-GPU setup)
    os.environ["WORLD_SIZE"] = "1"  # Total number of processes (1 for single-node)
    os.environ["MASTER_ADDR"] = "localhost"  # Address of the master node
    os.environ["MASTER_PORT"] = "29500"  # Port number for communication

    # Initialize the distributed environment (in your test setup)
    init_distributed_mode()

    batch_size = 2
    seconds = 1.6
    video_fps = 10
    audio_fps = 16000

    args = create_argparser().parse_args()
    def load_training_data(args):
        """Load dataset with distributed support."""
        dataset = load_data(
            data_dir=video_dir,
            batch_size=batch_size,
            video_size=args['video_size'],
            audio_size=args['audio_size'],
            frame_gap=1,
            random_flip=True,
            num_workers=4,
            deterministic=True,
            video_fps=video_fps,
            audio_fps=audio_fps
        )

        for video_batch, audio_batch in dataset:
            gt_batch = {"video": video_batch, "audio": audio_batch}
            yield gt_batch

    data = load_training_data(args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
        )
    model.to(dist_util.dev())
    # model.train()

    # # Prepare utilities
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False)
    # schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    # microbatch = batch_size

    # batch = next(dataset)
    # print(batch)
    # # Move data to device
    # # batch = {k: v.to(dist_util.dev()) for k, v in batch.items()}
    # # cond = {k: v.to(dist_util.dev()) for k, v in batch.items() if k != "video"}

    # # batch_len = batch['video'].shape[0]
    # def forward_backward(batch, cond, microbatch):
    #     batch = {k:v.to(dist_util.dev()) \
    #         for k, v in batch.items()}

    #     cond = {k:v.to(dist_util.dev()) \
    #         for k, v in cond.items()}

    #     batch_len = batch['video'].shape[0]
               
    #     for i in range(0, batch_len, microbatch):
    #         micro = {
    #             k: v[i : i + microbatch]
    #             for k, v in batch.items()
    #         }
            
    #         micro_cond = {
    #             k: v[i : i + microbatch]
    #             for k, v in cond.items()
    #         }
            
    #         last_batch = (i + microbatch) >= batch_len
    #         t, weights = schedule_sampler.sample(batch_size, dist_util.dev())
            
           
    #         compute_losses = functools.partial(
    #         diffusion.multimodal_training_losses,
    #         model,
    #         micro,
    #         t,
    #         model_kwargs=micro_cond,
    #         )
          
    #         # if last_batch or not self.use_ddp:
    #         losses = compute_losses()

    #         loss = (losses["loss"] * weights).mean()
    #         mp_trainer.backward(loss)
         
    #     # if isinstance(self.schedule_sampler, LossAwareSampler):
    #     #     self.schedule_sampler.update_with_local_losses(
    #     #             t, losses["loss"].detach()
    #     #         )

    #     log_loss_dict(
    #             diffusion, t, {k: v * weights for k, v in losses.items()}
    #         )
           
    #     return losses
    # # Run forward, backward, and step
    # losses = forward_backward(batch, {}, microbatch)
    # optimizer.step()
    # optimizer.zero_grad()
    # loss = losses["loss"].mean()

    # print("✅ Test passed: Full pipeline forward+backward+step ran successfully")
    # assert loss.item() > 0 and not torch.isnan(loss), "Loss should be positive and finite"


    ##################
    ### Prod env


    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        save_type="mp4",
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=2,
        resume_checkpoint="",
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
    import argparse
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    # Run the test
    # pytest.main([__file__])
    test_model_pipeline_step("./data10/concerts_audiovideo_dataset/unittest")



    # import librosa
    # # Load an example audio file
    # audio_path = librosa.example('trumpet')
    # wv, sr = librosa.load(audio_path, sr=44100)
    # # Dimension of the audio
    # print(len(wv))
    # print(wv)
    # print(type(wv))
    # from music2latent import EncoderDecoder
    # encdec = EncoderDecoder()
    # latent = encdec.encode(wv)
    # # latent has shape (batch_size/audio_channels, dim (64), sequence_length)
    # print(latent.shape)
    # wv_rec = encdec.decode(latent)