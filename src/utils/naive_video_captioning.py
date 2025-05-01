import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import transformers
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from PIL import Image

import builtins
import typing

# Hack to make `List` available globally if it's missing in module scope
builtins.List = typing.List

# Init dotenv
from dotenv import load_dotenv
load_dotenv()
# Login to HuggingFace
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_data_root", type=str, required=True, help="Base folder where video files are located."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to where models are stored.")
    parser.add_argument(
        "--output_path", type=str, default="video_dataset.csv", help="File path where dataset csv should be stored."
    )
    parser.add_argument("--max_new_tokens", type=int, default=226, help="Maximum number of new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()


SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.
- Part of the prompt should be a description of the scene, and part of it should be a description of the action.
""".strip()

# You responses should just be the video generation prompt. Here are examples:
#- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
#- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"


USER_PROMPT = """
Could you generate a prompt for a video generation model for the following video summary:

```
{0}
```

Please limit the prompt to [{1}] words.
""".strip()

QUESTION = """
Describe the video. You should pay close attention to every detail in the video and describe it in as much detail as possible.
""".strip()

MAX_NUM_FRAMES = 49


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_video(video_path: str):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps())
        sample_fps = max(1, sample_fps // 2)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print("num frames:", len(frames))
        return frames
    except Exception as e:
        print(f"[WARNING] Skipping file `{video_path}` due to error: {e}")
        return None


@torch.no_grad()
def main(args: Dict[str, Any]):
    set_seed(args.seed)

    video_files = [file for file in os.listdir(args.instance_data_root) if file.endswith(".mp4")]
    video_files = [os.path.join(args.instance_data_root, file) for file in video_files]
    video_descriptions = {}

    # model_id = "openbmb/MiniCPM-V-2_6"
    model_id = "marianna13/llava-phi-2-3b"
    # from transformers import pipeline
    # pipe = pipeline("text-generation", model="marianna13/llava-phi-2-3b", trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, token=os.getenv("HUGGINGFACE_TOKEN")
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))

    for filepath in video_files:
        print(f"Generating video summary for file: `{filepath}`")
        frames = encode_video(filepath)
        print(f"Number of frames: {len(frames)}")
        # msgs = [{"role": "user", "content": frames + [QUESTION]}]

        msgs = [{"role": "user", "content": QUESTION}]

        params = {
            "use_image_id": False,
            "max_slice_nums": 10,
        }

        # description = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
        description = model.chat(image=frames, msgs=msgs, tokenizer=tokenizer, **params)
        print(f"Video summary: {description}")
        video_descriptions[filepath] = {"summary": description}

    del model
    del tokenizer
    model = None
    tokenizer = None
    torch.cuda.empty_cache()

    #model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        model_kwargs={
            "local_files_only": True,
            "cache_dir": args.cache_dir,
            "torch_dtype": torch.bfloat16,
        },
    )

    for filepath in video_files:
        print(f"Generating captions for file: `{filepath}`")

        for prompt_type, num_words in [("short_prompt", 25), ("prompt", 75), ("verbose_prompt", 125)]:
            user_prompt = USER_PROMPT.format(video_descriptions[filepath]["summary"], num_words)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=226,
            )
            video_descriptions[filepath][prompt_type] = outputs[0]["generated_text"][-1]["content"]

    df_data = []
    for filepath, description in video_descriptions.items():
        relative_path = os.path.relpath(filepath, args.instance_data_root)
        df_data.append(
            {
                "path": relative_path,
                "short_prompt": description.get("short_prompt", ""),
                "prompt": description.get("prompt", ""),
                "verbose_prompt": description.get("verbose_prompt", ""),
                "summary": description.get("summary", ""),
            }
        )

    df = pd.DataFrame(df_data)
    df.to_csv(args.output_path)


if __name__ == "__main__":
    args = get_args()
    main(args)

    # /home/users/ap794/finalCS590-text2audiovideo/MM-Diffusion/data10/concerts_audiovideo_dataset/unittest