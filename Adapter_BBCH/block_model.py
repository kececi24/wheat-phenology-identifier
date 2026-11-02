"""DINOv3 block PEFT. Original backbone parameters always remain frozen."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from DINOv3_BBCH.multiscale_phenology import (
    HierarchicalDenseTilePooler, compact_dense_tokens,
)

METHODS = ("none", "lora", "standard", "transition")
CHECKPOINT_FORMAT = "bbch_block_adapter_v1"


def make_extractor(backbone_name, revision=None):
    """Load the public pretrained base; checkpoints pin its resolved HF commit."""
    from transformers import AutoImageProcessor, AutoModel
    from DINOv3_BBCH.multiscale_phenology import get_huggingface_token

    kwargs = {"revision": revision} if revision else {}
    token = get_huggingface_token() if get_huggingface_token else None
    if token:
        kwargs["token"] = token
    backbone = AutoModel.from_pretrained(backbone_name, **kwargs)
    if backbone.config.model_type != "dinov3_vit":
        raise ValueError("This experiment supports Hugging Face DINOv3 ViT only")
    processor_kwargs = dict(kwargs)
    resolved = getattr(backbone.config, "_commit_hash", None)
    if resolved:
        processor_kwargs["revision"] = resolved
    processor = AutoImageProcessor.from_pretrained(backbone_name, **processor_kwargs)
    return DenseExtractor(backbone, processor)


class DenseExtractor(nn.Module):
    def __init__(self, backbone, processor=None):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.out_dim = backbone.config.hidden_size
        self.num_register_tokens = backbone.config.num_register_tokens
        patch = backbone.config.patch_size
        if isinstance(patch, (tuple, list)):
            if len(set(patch)) != 1:
                raise ValueError("Only square DINOv3 patches are supported")
            patch = patch[0]
        self.patch_size = int(patch)

    def forward_dense(self, pixels, grid_size=2, include_cls=True):
        output = self.backbone(pixel_values=pixels)
        return compact_dense_tokens(
            output.last_hidden_state, pixels.shape[-2], pixels.shape[-1],
            self.patch_size, self.num_register_tokens, grid_size, include_cls)


def selected_blocks(extractor, last_blocks):
    blocks = getattr(extractor.backbone, "layer", None)
    if blocks is None:
        blocks = getattr(getattr(extractor.backbone, "model", None), "layer", None)
    if not isinstance(blocks, nn.ModuleList) or not 1 <= last_blocks <= len(blocks):
        raise ValueError("Expected DINOv3 layer/model.layer and a valid last_blocks count")
    return blocks, list(range(len(blocks) - last_blocks, len(blocks)))


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x, with frozen W and initially zero B."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("DINOv3 q_proj/v_proj must be Linear modules")
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class Bottleneck(nn.Module):
    """Parallel FFN branch; the transition variant learns A0 + E."""
    def __init__(self, dim, rank, basis=None):
        super().__init__()
        if basis is None:
            self.down = nn.Parameter(torch.empty(rank, dim))
            nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
            self.register_buffer("prior", torch.zeros(rank, dim))
            self.has_prior = False
        else:
            if basis.shape != (rank, dim) or not torch.isfinite(basis).all():
                raise ValueError("Transition prior must be finite [rank, hidden_size]")
            self.down = nn.Parameter(torch.zeros(rank, dim))  # E
            self.register_buffer("prior", basis.detach().float().clone())
            self.has_prior = True
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, z):
        return self.up(F.gelu(F.linear(z, self.prior + self.down)))

    def penalty(self):
        return self.down.square().sum() if self.has_prior else self.down.new_zeros(())


class BlockWithAdapter(nn.Module):
    """Preserve HF forward/rotary attention and add an unscaled FFN residual.

    norm2 is the exact input of the pretrained FFN. The new branch is added
    after the original block, outside its pretrained LayerScale/drop-path.
    Only eager single-device training is supported (no checkpoint recomputation).
    """
    def __init__(self, original, adapter):
        super().__init__()
        self.original = original
        self.adapter = adapter
        self._z = None
        self.original.norm2.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._z = output

    def forward(self, *args, **kwargs):
        self._z = None
        try:
            output = self.original(*args, **kwargs)
            if not isinstance(output, torch.Tensor) or self._z is None:
                raise RuntimeError("Unsupported DINOv3 block forward contract")
            return output + self.adapter(self._z)
        finally:
            self._z = None  # never retain a graph between image batches


@contextmanager
def capture_layer_inputs(extractor, last_blocks):
    """Temporary hooks for training-only prior estimation, before any PEFT."""
    blocks, indices = selected_blocks(extractor, last_blocks)
    captured, handles = {}, []
    for index in indices:
        def capture(module, inputs, output, index=index):
            captured[index] = output.detach()
        handles.append(blocks[index].norm2.register_forward_hook(capture))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class BlockAdapterClassifier(nn.Module):
    def __init__(self, extractor, method="none", last_blocks=2, rank=6,
                 lora_rank=3, lora_alpha=3.0, hidden_dim=128, dropout=0.2,
                 dense_grid_size=2, tile_batch_size=8, bases=None):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown block adapter method: {method}")
        if min(rank, lora_rank, hidden_dim, dense_grid_size, tile_batch_size) < 1:
            raise ValueError("Adapter/model dimensions must be positive")
        if method == "transition" and bases is None:
            raise ValueError("The transition adapter requires training-only per-layer bases")
        self.extractor = extractor
        extractor.requires_grad_(False)
        self.method = method
        self.grid = dense_grid_size
        self.tile_batch_size = tile_batch_size
        dim = extractor.out_dim
        # Construct common components BEFORE adapters: same seed => same head.
        self.pooler = HierarchicalDenseTilePooler(dim, dense_grid_size ** 2 + 1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 7))
        blocks, indices = selected_blocks(extractor, last_blocks)
        self.block_indices = indices
        for index in indices:
            if not hasattr(blocks[index], "norm2") or not hasattr(blocks[index], "attention"):
                raise ValueError("Unsupported DINOv3 block; expected norm2 and attention")
            if method in {"standard", "transition"}:
                basis = bases[index] if method == "transition" else None
                blocks[index] = BlockWithAdapter(blocks[index], Bottleneck(dim, rank, basis))
            elif method == "lora":
                attention = blocks[index].attention
                for name in ("q_proj", "v_proj"):
                    setattr(attention, name, LoRALinear(getattr(attention, name), lora_rank, lora_alpha))
        self.config = dict(method=method, last_blocks=last_blocks, rank=rank,
                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                           hidden_dim=hidden_dim, dropout=dropout,
                           dense_grid_size=dense_grid_size, tile_batch_size=tile_batch_size)

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbone has deterministic drop-path/dropout in every arm.
        # Adapter branches contain no dropout; pooling/head remain trainable.
        self.extractor.eval()
        return self

    def prior_penalty(self):
        terms = [module.penalty() for module in self.modules()
                 if isinstance(module, Bottleneck) and module.has_prior]
        return torch.stack(terms).mean() if terms else next(self.head.parameters()).new_zeros(())

    def forward(self, tiles, tile_mask):
        # Encode real tiles only. Chunking bounds each backbone invocation;
        # autograd still retains adapted-layer graphs until the image backward.
        batch, count = tiles.shape[:2]
        if tile_mask.shape != (batch, count) or tile_mask.dtype != torch.bool or not tile_mask.any(1).all():
            raise ValueError("Every image must have a nonempty boolean tile mask")
        valid = tiles.flatten(0, 1)[tile_mask.flatten()]
        chunks = []
        for pixels in valid.split(self.tile_batch_size):
            chunks.append(self.extractor.forward_dense(pixels, self.grid, True))
        dense = torch.cat(chunks)
        padded = dense.new_zeros(batch * count, dense.shape[1], dense.shape[2])
        padded[tile_mask.flatten()] = dense
        dense = padded.reshape(batch, count, dense.shape[1], dense.shape[2]).unsqueeze(1)
        return self.head(self.pooler(dense, tile_mask.unsqueeze(1)).squeeze(1))

    def parameter_counts(self):
        adapter = sum(p.numel() for p in self.extractor.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"adapter_parameters": adapter, "head_pool_parameters": total - adapter,
                "trainable_parameters": total}

    def adaptation_state(self):
        # Frozen public weights are intentionally not duplicated per fold/arm.
        keys = {name for name, p in self.named_parameters() if p.requires_grad}
        keys.update(name for name, _ in self.named_buffers()
                    if name.endswith("adapter.prior"))
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                if key in keys}

    def load_adaptation_state(self, state):
        expected = set(self.adaptation_state())
        if set(state) != expected:
            raise ValueError(f"Checkpoint adapter keys differ: {set(state) ^ expected}")
        self.load_state_dict(state, strict=False)


def from_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a transformer-block adapter checkpoint")
    source = checkpoint["backbone"]
    extractor = make_extractor(source["name"], source["revision"])
    config = checkpoint["model_config"]
    _, indices = selected_blocks(extractor, config["last_blocks"])
    bases = {index: torch.zeros(config["rank"], extractor.out_dim) for index in indices}
    model = BlockAdapterClassifier(extractor, bases=bases, **config)
    model.load_adaptation_state(checkpoint["model"])
    return model.to(device).eval()
