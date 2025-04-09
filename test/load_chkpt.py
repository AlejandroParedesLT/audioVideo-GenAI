import torch

checkpoint = torch.load('./data10/call_of_duty/debug/opt002000.pt', map_location="cpu")

# Print the main keys stored in the checkpoint
print("Checkpoint keys:")
for key, value in checkpoint.items():
    print(f"- {key}")

# If it's a state_dict, inspect the model parameter names
if "state_dict" in checkpoint:
    print("\nModel state_dict keys:")
    for key in checkpoint["state_dict"].keys():
        print(f"- {key}")
else:
    print("\nDirect state_dict keys:")
    for key in checkpoint.keys():
        print(f"- {key}")
