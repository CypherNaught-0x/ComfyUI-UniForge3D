"""UV-space rasterization: the part of nvdiffrast UniForge3D uses for texture baking.

Upstream rasterizes ``uv * 2 - 1`` with ``nvdiffrast.rasterize`` and then interpolates vertex
positions. Texel row ``i`` / column ``j`` covers ``v = (i + 0.5) / size``, ``u = (j + 0.5) / size``
in both nvdiffrast's convention and ComfyUI's tiled point-in-triangle rasterizer used here.
"""

from __future__ import annotations

import torch

from .comfy_backend import comfy


def rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, size: int):
    """Return per-texel (face index [S,S], barycentrics [S,S,3], coverage [S,S]) on the GPU."""
    return comfy().rasterize_uv_barycentric(
        faces.detach().cpu().numpy(), uvs.detach().float().cpu().numpy(), int(size),
    )


def interpolate(attr: torch.Tensor, faces: torch.Tensor, face_index: torch.Tensor,
                barycentric: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Barycentric interpolation of per-vertex ``attr`` [N,C] into a [S,S,C] map (zeros outside)."""
    device = face_index.device
    out = torch.zeros((*mask.shape, attr.shape[1]), device=device, dtype=attr.dtype)
    if mask.any():
        corners = attr.to(device)[faces.to(device).long()[face_index[mask]]]   # [K, 3, C]
        out[mask] = (barycentric[mask][:, :, None].to(attr.dtype) * corners).sum(1)
    return out
