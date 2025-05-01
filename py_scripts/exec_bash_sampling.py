import subprocess

# List of model names
model_names = [
    "model002000", "model004000", "model006000", "model008000", "model010000",
    "model012000", "model014000", "model016000", "model018000", "model020000"
]

# Base path to pass as an argument
base_path = "/home/users/ap794/finalCS590-text2audiovideo/MM-Diffusion/"

# Loop through each model name
for model_name in model_names:
    print(f"Running inference for MODEL_NAME: {model_name}")
    
    # Call the bash script with the model name and base path
    try:
        subprocess.run(
            [f"bash ./ssh_scripts/evaluation/multimodal_sample_sr_concerts_multicond.sh {model_name} {base_path}"],
            check=True  # This ensures that an exception is raised if the command fails
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running for {model_name}: {e}")
