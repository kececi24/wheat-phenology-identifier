import os

import torch
import torch.distributed as dist

def initialize_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    return device, dist.get_rank(), dist.get_world_size()


if __name__ == "__main__":
    device, rank, world_size = initialize_distributed()

    try:
        value = torch.ones(1, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

        assert value.item() == world_size
        print(f"rank={rank}, device={device}, world_size={world_size}, sum={value.item()}", flush=True)
    finally:
        dist.destroy_process_group()