import subprocess
import os

# Path to your prompts file
PROMPTS_FILE = "./tools/prompts_concerts.txt"
OUTPUT_DIR = "./output"
BASE_COMMAND = [
    "python", "./src/inference.py",
    "--mmvideo_id=THUDM/CogVideoX-2b",
    "--mmaudio_variant=large_44k_v2",
    "--duration=8.0",
    "--output=" + OUTPUT_DIR,
    "--num_steps=25",
    "--seed=43"
]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read prompts from file
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Run inference for each prompt
for idx, prompt in enumerate(prompts, start=1):
    video_name = f"video_{idx:02d}"
    cmd = BASE_COMMAND + [
        f"--prompt={prompt}",
        f"--video_name={video_name}"
    ]
    print(f"Running inference for: {video_name}")
    subprocess.run(cmd)
