import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from argparse import ArgumentParser
from pathlib import Path
from models.mm_audio_inference import audio_infenrence
import os
# pydotenv
from dotenv import load_dotenv
load_dotenv()
home_dir = Path.home()

def main():
    print("Starting Arg Parser...")
    parser = ArgumentParser()
    parser.add_argument('--mmvideo_id', type=str, help='Model type for video generation', default="THUDM/CogVideoX-2b")
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--mmaudio_variant',
                        type=str,
                        default='large_44k_v2',
                        help='small_16k, small_44k, medium_44k, large_44k, large_44k_v2')
    # parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--video_name', help='Name of the generated video', default='myvideo')

    args = parser.parse_args()

    print("Starting Video Generation...")
    print(args)
    device_map = "balanced" #, device_map=device_map
    pipe = CogVideoXPipeline.from_pretrained(args.mmvideo_id, torch_dtype=torch.float16, token=os.getenv("HUGGINGFACE_TOKEN"))
    print("Loaded Video model...")
    # pipe.load_lora_weights("/path/to/lora/weights", adapter_name="cogvideox-lora") # Or,
    #pipe.load_lora_weights("my-awesome-hf-username/my-awesome-lora-name", adapter_name="cogvideox-lora") # If loading from the HF Hub
    pipe.to("cuda")

    # Assuming lora_alpha=32 and rank=64 for training. If different, set accordingly
    #pipe.set_adapters(["cogvideox-lora"], [32 / 64])

    # prompt = "Generate a cinematic video of a person cooking a traditional meal in a warmly lit rustic kitchen during golden hour. The person is in their early 30s, with medium-length curly hair tied back in a loose bun, wearing a beige linen apron over a dark green shirt. The kitchen is cozy and lived-in, with wooden countertops, hanging copper pots, potted herbs on the windowsill, and sunlight streaming through a window, casting soft shadows."
    frames = pipe(args.prompt, guidance_scale=6, use_dynamic_cfg=True).frames[0]
    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent

    # Combine with the desired output subdir and filename
    output_dir = (script_dir / args.output).resolve()

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Final video path
    video_name = args.video_name if args.video_name.endswith(".mp4") else f"{args.video_name}.mp4"
    write_file_destination = output_dir / video_name
    
    print(f'Exporting video to {write_file_destination}')
    export_to_video(frames, write_file_destination, fps=8)
    del pipe
    print(f"Video generated and saved to {write_file_destination}")
    print("Starting audio inference...")
    audio_infenrence(
        variant=args.mmaudio_variant,
        video=write_file_destination,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output=output_dir,
        num_steps=args.num_steps,
        duration=  args.duration,
        cfg_strength=args.cfg_strength,
        skip_video_composite=args.skip_video_composite,
        mask_away_clip=args.mask_away_clip,
        full_precision=args.full_precision,
        seed=42,
    )

    print(f"The result is stored at {args.output}/{args.video_name}")


if __name__ == '__main__':
    main()