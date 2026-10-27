import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from distributed_runtime import (
    initialize_distributed,
    shard_temporal_model,
    read_gate_value
)
from multiscale_phenology import SingleStreamEmbeddingTemporalTransformer

device, rank, world_size = initialize_distributed()

def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        foreach=True
    )

    torch.manual_seed(1000 + rank)

    batch_size = 2
    window_days = 5
    tiles = 4

    features = torch.randn(
        batch_size, window_days, 32,
        device=device
    )
    mask = torch.ones(
        batch_size, window_days,
        dtype=torch.bool,
        device=device
    )
    temporal_features = torch.randn(
        batch_size, window_days, 1,
        device=device
    )
    location_features = torch.randn(
        batch_size, 3,
        device=device
    )
    targets = torch.arange(batch_size, device=device)
    weight = model.temporal.layers[0].linear1.weight

    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        before = local_tensor(weight).detach().clone()

        logits = model(
            macro=features,
            micro=features,
            mask=mask,
            temporal_features=temporal_features,
            location_features=location_features
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)

        assert torch.isfinite(loss).item(), "Non-finite training loss"

        loss.backward()
        optimizer.step()

        changed = torch.tensor(
            int(not torch.equal(before, local_tensor(weight).detach())),
            device=device
        )
        if dist.is_initialized():
            dist.all_reduce(changed, op=dist.ReduceOp.MAX)

        assert changed.item() == 1, "No rank updated the checked weight"

        print(f"rank={rank}, step={step}, loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits = model(
            macro=features,
            micro=features,
            mask=mask,
            temporal_features=temporal_features,
            location_features=location_features
        )

    assert logits.shape[0] == batch_size
    assert torch.isfinite(logits).all().item()

    print(f"rank={rank}: training and evaluation smoke test passed", flush=True)

    calender_gate = read_gate_value(model.temporal_feature_gate_logit)
    location_gate = read_gate_value(model.location_gate_logit)

    assert calender_gate is not None
    assert 0.0 < calender_gate < 1.0
    assert location_gate is not None
    assert 0.0 < location_gate < 1.0

    if rank == 0:
        print(
            f"Gate logging passed: "
            f"calender={calender_gate:.6f},"
            f"location={location_gate:.6f}",
            flush=True
        )

finally:
    if dist.is_initialized():
        dist.destroy_process_group()