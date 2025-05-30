# """
# Helpers for distributed training.
# """

# import io
# import os
# import socket
# import blobfile as bf
# #from mpi4py import MPI
# import torch as th
# import torch.distributed as dist

# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).

# GPUS_PER_NODE = 2

# def setup_dist(devices=None):
#     """
#     Setup a distributed process group.
#     """
#     global GPUS_PER_NODE
#     print("GPUs per node",GPUS_PER_NODE)
#     if dist.is_initialized():
#         return

#     if devices.startswith("G"):
#         GPUS_PER_NODE = int(devices[1:])
#         os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#     else:
#         devices_list=devices.split(',')
#         print("devices listed here", devices_list)
#         GPUS_PER_NODE = len(devices_list)
#         os.environ["CUDA_VISIBLE_DEVICES"] =  f"{devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]}"
        
      
#     comm = MPI.COMM_WORLD

#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)

#     dist.init_process_group(backend=backend, init_method="env://")




# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """

#     if th.cuda.is_available():
#         return th.device("cuda")
#     return th.device("cpu")

# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     with bf.BlobFile(path, "rb") as f:
#         data = f.read()
       
#     return th.load(io.BytesIO(data), **kwargs)

# def sync_params(params):
#     """
#     Synchronize a sequence of Tensors across ranks from rank 0.
#     """
#     for p in params:
#         with th.no_grad():
#             dist.broadcast(p, 0)
    
# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()



######################################################################################
import io
import os
import socket
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).

def setup_dist(devices=None):
    """
    Setup a distributed process group using PyTorch distributed without MPI.
    """
    #global GPUS_PER_NODE
    GPUS_PER_NODE = len(devices.split(','))
    print("GPUs per node", GPUS_PER_NODE)

    if dist.is_initialized():
        return

    if devices.startswith("G"):
        GPUS_PER_NODE = int(devices[1:])
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{th.distributed.get_rank() % GPUS_PER_NODE}"
    else:
        devices_list = devices.split(',')
        print("devices listed here", devices_list)
        GPUS_PER_NODE = len(devices_list)
        #os.environ["CUDA_VISIBLE_DEVICES"] = f"{devices_list[th.distributed.get_rank() % GPUS_PER_NODE]}"

    # Set the backend for distributed training.
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = hostname
    os.environ["RANK"] = str(th.distributed.get_rank())
    os.environ["WORLD_SIZE"] = str(th.distributed.get_world_size())

    port = _find_free_port()
    os.environ["MASTER_PORT"] = str(port)
    print('This is being executed')
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across distributed ranks.
    """
    with open(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def check_nvidia_smi():
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)