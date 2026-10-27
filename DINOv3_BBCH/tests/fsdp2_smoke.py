import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from distributed_runtime import (
    initialize_distributed,
    shard_temporal_model
)
from multiscale_phenology import SingleStreamEmbeddingTemporalTransformer

device, rank, world_size = initialize_distributed()

try:
    torch.manual_seed(42)

    model = SingleStreamEmbeddingTemporalTransformer(
        feature_dim=32,
        embed_dim=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_norm_first=True,
        temporal_feature_dim=1,
        location_feature_dim=3
    ).to(device)

    model = shard_temporal_model(model, world_size)

    if world_size > 1:
        assert all(isinstance(parameter, DTensor) for parameter in model.parameters())

    parameter = model.temporal.layers[0].linear1.weight
    local_shape = (
        parameter.to_local().shape if isinstance(parameter, DTensor)
        else parameter.shape
    )

    print(
        f"rank={rank}, global_shape={tuple(parameter.shape)}",
        f"local_shape={tuple(local_shape)}",
        flush=True
    )
finally:
    if dist.is_initialized():
        dist.destroy_process_group()