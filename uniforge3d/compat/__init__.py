"""Pure-PyTorch stand-ins for the CUDA-only extensions UniForge3D was built on.

Upstream depends on four NVIDIA-only packages. On ROCm they are replaced as follows:

=================  ===================================================================
FlexGEMM           ``conv_torch`` sparse-conv backend (ComfyUI's gather+GEMM submanifold
                   conv) and :func:`grid_sample.grid_sample_3d` (port of the CUDA kernel).
o-voxel            ComfyUI's ``flexible_dual_grid_to_mesh`` (same signature/algorithm).
CuMesh             :mod:`.cumesh`, a facade over ComfyUI's pure-torch mesh library
                   (QEM decimation, narrow-band DC remesh, hole filling, BVH, UV unwrap).
nvdiffrast         :mod:`.rasterize`, UV-space rasterization + attribute interpolation.
=================  ===================================================================

``flash_attn``, ``peft`` and ``easydict`` are replaced by an SDPA attention backend,
:mod:`.lora` and :class:`.attrdict.AttrDict`. The ComfyUI implementations are imported
lazily from the running ComfyUI install, so ComfyUI must be importable.
"""

from .comfy_backend import check_backend

__all__ = ["check_backend"]
