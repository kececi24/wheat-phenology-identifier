import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

def initialize_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl")

        device = torch.device("cuda", local_rank)
        return device, dist.get_rank(), dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    rank = 0
    world_size = 1
    return device, rank, world_size

def make_train_sampler(dataset, rank, world_size, seed):
    if world_size == 1:
        return None

    return DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True
    )

def shard_temporal_model(model, world_size):
    if world_size == 1:
        return model

    if not dist.is_initialized():
        raise RuntimeError("Initialize distributed communication first")

    if not isinstance(model.temporal, torch.nn.TransformerEncoder):
        raise ValueError("This FSDP2 implementation only supports transformer only")

    scalar_names = [name for name, parameter in model.named_parameters() if parameter.ndim == 0]
    if scalar_names:
        raise ValueError(f"Reshape scalar parameters first: {scalar_names}")

    mesh = init_device_mesh(
        "cuda",
        mesh_shape={world_size},
        mesh_dim_names=("shard",)
    )

    for block in model.temporal.layers:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)

    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    return model

if __name__ == "__main__":
    device, rank, world_size = initialize_distributed()

    try:
        value = torch.ones(1, device=device)
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

        assert value.item() == world_size
        print(f"rank={rank}, device={device}, world_size={world_size}, sum={value.item()}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()