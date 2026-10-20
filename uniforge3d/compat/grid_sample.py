"""Sparse-voxel grid sampling: replacement for ``flex_gemm.ops.grid_sample.grid_sample_3d``.

Mirrors FlexGEMM's CUDA kernel (``hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight``):
voxel ``i`` has its centre at ``i + 0.5``; trilinear sampling visits the 8 voxels around
``floor(q - 0.5)``, weights only the occupied ones and renormalises by their weight sum, so a
query with no occupied neighbour returns zeros.
"""

from __future__ import annotations

import torch

from . import native
from .comfy_backend import comfy

_CORNERS = torch.tensor(
    [[(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=torch.long
)
# Queries per chunk; bounds the [chunk, 8, C] gather.
_CHUNK = 1 << 20


def _flat_keys(batch: torch.Tensor, xyz: torch.Tensor, W: int, H: int, D: int) -> torch.Tensor:
    return ((batch * W + xyz[..., 0]) * H + xyz[..., 1]) * D + xyz[..., 2]


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape,
    grid: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Sample ``feats`` [N, C] at voxel coordinates ``coords`` [N, 4] (b, x, y, z).

    ``shape`` ends with (W, H, D); ``grid`` is [B, L, 3] in voxel units. Returns [B, L, C].
    """
    if mode not in ("nearest", "trilinear"):
        raise ValueError(f"Unsupported grid_sample_3d mode: {mode}")
    flex_gemm = native.flex_gemm()
    if flex_gemm is not None:
        from flex_gemm.ops.grid_sample import grid_sample_3d as native_grid_sample
        return native_grid_sample(feats, coords, shape, grid, mode=mode)
    if feats.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 4 or grid.ndim != 3:
        raise ValueError(
            f"Expected feats [N,C], coords [N,4], grid [B,L,3]; got "
            f"{tuple(feats.shape)}, {tuple(coords.shape)}, {tuple(grid.shape)}"
        )
    W, H, D = (int(s) for s in tuple(shape)[-3:])
    B, L = grid.shape[:2]
    C = feats.shape[1]
    device = feats.device

    coords = coords.to(device=device, dtype=torch.long)
    hashmap = comfy().TorchHashMap(
        _flat_keys(coords[:, 0], coords[:, 1:], W, H, D),
        torch.arange(coords.shape[0], dtype=torch.int32, device=device),
    )
    bounds = torch.tensor([W, H, D], device=device)
    queries = grid.to(device=device, dtype=torch.float32).reshape(-1, 3)
    query_batch = torch.arange(B, device=device).repeat_interleave(L)
    out = torch.zeros(queries.shape[0], C, device=device, dtype=feats.dtype)

    for start in range(0, queries.shape[0], _CHUNK):
        q = queries[start:start + _CHUNK]
        b = query_batch[start:start + _CHUNK]
        if mode == "nearest":
            cells = q.floor().long()
            inside = ((cells >= 0) & (cells < bounds)).all(-1)
            index = hashmap.lookup_flat(
                torch.where(inside, _flat_keys(b, cells, W, H, D), torch.full_like(b, -1))
            ).long()
            found = index >= 0
            out[start:start + q.shape[0]][found] = feats[index[found]]
            continue

        cells = (q - 0.5).floor().long()[:, None, :] + _CORNERS.to(device)[None]  # [n, 8, 3]
        inside = ((cells >= 0) & (cells < bounds)).all(-1)
        keys = torch.where(inside, _flat_keys(b[:, None], cells, W, H, D), torch.full_like(inside, -1, dtype=torch.long))
        index = hashmap.lookup_flat(keys.reshape(-1)).view(-1, 8).long()
        weight = (1 - (q[:, None, :] - cells.float() - 0.5).abs()).prod(-1)
        weight = torch.where(index >= 0, weight, torch.zeros_like(weight))
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        gathered = feats[index.clamp_min(0)]                                     # [n, 8, C]
        out[start:start + q.shape[0]] = (gathered * weight[..., None].to(feats.dtype)).sum(1)

    return out.view(B, L, C)
