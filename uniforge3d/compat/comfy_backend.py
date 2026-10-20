"""Single place that reaches into ComfyUI internals, so API drift fails loudly and early."""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import SimpleNamespace

# (module, attribute) pairs the port relies on. Several are private helpers of
# comfy_extras.nodes_mesh_postprocess; check_backend() reports any that disappear.
_REQUIRED = {
    "TorchHashMap": ("comfy.ldm.trellis2.flexgemm", "TorchHashMap"),
    "sparse_submanifold_conv3d": ("comfy.ldm.trellis2.flexgemm", "sparse_submanifold_conv3d"),
    "flexible_dual_grid_to_mesh": ("comfy.ldm.trellis2.vae", "flexible_dual_grid_to_mesh"),
    "QEMConfig": ("comfy_extras.mesh3d.postprocess.qem_decimate", "QEMConfig"),
    "qem_simplify": ("comfy_extras.mesh3d.postprocess.qem_decimate", "qem_simplify"),
    "compute_vertex_normals": ("comfy_extras.mesh3d.postprocess.qem_decimate", "_compute_vertex_normals"),
    "remesh_narrow_band_dc": ("comfy_extras.mesh3d.postprocess.remesh", "remesh_narrow_band_dc"),
    "fill_holes": ("comfy_extras.nodes_mesh_postprocess", "fill_holes_v2_fn"),
    "build_triangle_bvh": ("comfy_extras.nodes_mesh_postprocess", "_build_triangle_bvh"),
    "closest_points_on_mesh_bvh": ("comfy_extras.nodes_mesh_postprocess", "_closest_points_on_mesh_bvh"),
    "uv_unwrap": ("comfy_extras.nodes_mesh_postprocess", "_uv_unwrap"),
    "rasterize_uv_barycentric": ("comfy_extras.nodes_mesh_postprocess", "_rasterize_uv_barycentric"),
}


@lru_cache(maxsize=1)
def comfy() -> SimpleNamespace:
    """Resolve every required ComfyUI symbol, raising one error that lists all missing ones."""
    resolved, missing = {}, []
    for name, (module, attribute) in _REQUIRED.items():
        try:
            resolved[name] = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as error:
            missing.append(f"{module}.{attribute} ({type(error).__name__}: {error})")
    if missing:
        raise ImportError(
            "UniForge3D's ROCm port needs ComfyUI's native TRELLIS.2/mesh implementation; "
            "missing: " + "; ".join(missing)
        )
    return SimpleNamespace(**resolved)


def check_backend() -> None:
    comfy()
