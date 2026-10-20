"""Submanifold sparse conv backed by ComfyUI's pure-torch gather+GEMM (FlexGEMM replacement).

Same weight layout (Co, Kd, Kh, Kw, Ci), neighbour ordering and spatial-cache key as the
``flex_gemm`` backend, so released checkpoints load unchanged.
"""
import torch

from .. import SparseTensor
from ....compat.comfy_backend import comfy
from .conv_flex_gemm import (  # noqa: F401  (identical parameter initialisation)
    sparse_conv3d_init,
    sparse_inverse_conv3d_init,
    sparse_inverse_conv3d_forward,
)


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    feats = x.feats
    weight = self.weight.to(feats.dtype)
    bias = self.bias.to(feats.dtype) if self.bias is not None else None
    out, neighbor_cache_ = comfy().sparse_submanifold_conv3d(
        feats, x.coords, tuple(x.spatial_shape), weight, bias, neighbor_cache, self.dilation,
    )
    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
    return x.replace(out)
