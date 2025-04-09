import os
import torch
import torch.distributed as dist

def main():
    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # Get local rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set GPU for each process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Print test message
    print(f"Process {rank}/{world_size} running on GPU {local_rank} ({torch.cuda.get_device_name(device)})")

    # Simple tensor test
    tensor = torch.tensor([rank], device=device)
    dist.all_reduce(tensor)
    
    print(f"Process {rank}: Tensor after all_reduce: {tensor.item()}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
