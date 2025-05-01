# preload_dataset.py

import os
import subprocess

def main():
    target_dir = "/dev/shm/CallOfDuty-AudioVideo-Dataset"

    # Check if dataset already exists
    if os.path.exists(target_dir):
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    print(f"Downloading dataset to {target_dir}...")

    # Clone the repo into /dev/shm
    subprocess.run([
        "git", "clone", 
        "https://huggingface.co/datasets/alejandroparedeslatorre/CallOfDuty-AudioVideo-Dataset", 
        target_dir
    ], check=True)

    print(f"Dataset successfully downloaded to {target_dir}.")

if __name__ == "__main__":
    main()
