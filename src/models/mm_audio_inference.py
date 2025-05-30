import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@torch.inference_mode()
def audio_infenrence(
    variant: str = 'large_44k_v2',
    video: Path = None,
    prompt: str = '',
    negative_prompt: str = '',
    output: Path = Path('./output'),
    num_steps: int = 25,
    duration: float = 8.0,
    cfg_strength: float = 4.5,
    skip_video_composite: bool = False,
    mask_away_clip: bool = False,
    full_precision: bool = False,
    seed: int = 42,
):
    setup_eval_logging()
    if variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {variant}')
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    if video:
        video_path: Path = Path(video).expanduser()
    else:
        raise ValueError(f'Not a valid path: {video_path}')
    prompt: str = prompt
    negative_prompt: str = negative_prompt
    output_dir: str = output.expanduser()
    seed: int = seed
    num_steps: int = num_steps
    duration: float = duration
    cfg_strength: float = cfg_strength
    skip_video_composite: bool = skip_video_composite
    mask_away_clip: bool = mask_away_clip

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = sync_frames = None

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f'Audio saved to {save_path}')
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    del net
    del feature_utils
    del fm
    del audio
    del clip_frames
    del sync_frames
    del video_info
    del video_path
    del save_path

if __name__ == '__main__':
    audio_infenrence()
