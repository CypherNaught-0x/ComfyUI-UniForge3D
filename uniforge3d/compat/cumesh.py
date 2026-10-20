"""Facade with the subset of the ``cumesh`` API that UniForge3D calls, backed by ComfyUI.

When a natively-built ``cumesh`` is importable (see :mod:`.native`), each operation is delegated
to it, which reproduces upstream exactly. Otherwise only the operations on UniForge3D's code
paths are provided; anything else raises
NotImplementedError rather than silently doing something different. Parameter mapping:

* ``simplify`` -> ComfyUI QEM, ``placement_mode="midpoint"`` (ComfyUI's CuMesh-faithful preset).
* ``fill_holes`` -> ComfyUI's GPU hole filler (same perimeter threshold; it pre-welds vertices).
* ``remeshing.remesh_narrow_band_dc`` -> ComfyUI's narrow-band DC in ``udf`` mode with every
  component filter off, i.e. CuMesh's (UDF - eps) isosurface. ``drop_inner_shell=True`` is an
  opt-in extra that removes the inside-out offset shell a closed input produces.
* ``uv_unwrap`` -> ComfyUI's chart segmentation/parameterization/packing (``pec`` segmenter).
  It is a different algorithm from CuMesh's xatlas-based atlas, so charts differ; texel density
  is controlled by ``uv_resolution``.
* ``cuBVH.unsigned_distance`` -> ComfyUI's pure-torch LBVH closest-point query.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from . import native
from .comfy_backend import comfy


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _drop_unused_vertices(vertices: torch.Tensor, faces: torch.Tensor):
    used, inverse = torch.unique(faces.reshape(-1).long(), return_inverse=True)
    return vertices[used], inverse.view(-1, 3).int()


def _native_mesh(vertices: torch.Tensor, faces: torch.Tensor):
    mesh = native.cumesh().CuMesh()
    mesh.init(vertices, faces)
    return mesh


class CuMesh:
    def __init__(self) -> None:
        self.vertices = None
        self.faces = None
        self._normals = None

    def init(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        device = vertices.device if vertices.is_cuda else _device()
        self.vertices = vertices.detach().to(device=device, dtype=torch.float32).contiguous()
        self.faces = faces.detach().to(device=device, dtype=torch.int32).contiguous()
        self._normals = None

    def read(self):
        return self.vertices, self.faces

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def _set(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        self.init(vertices, faces)

    def simplify(self, target: int, verbose: bool = False, options: dict | None = None) -> None:
        if self.num_faces <= int(target):
            return
        if native.cumesh() is not None:
            mesh = _native_mesh(self.vertices, self.faces)
            mesh.simplify(int(target), verbose=verbose, options=options or {})
            self._set(*mesh.read())
            return
        backend = comfy()
        vertices, faces, _ = backend.qem_simplify(
            self.vertices, self.faces.long(), int(target), config=backend.QEMConfig(placement_mode="midpoint"),
        )
        self._set(vertices, faces)

    def fill_holes(self, max_hole_perimeter: float = 3e-2) -> None:
        if native.cumesh() is not None:
            mesh = _native_mesh(self.vertices, self.faces)
            mesh.fill_holes(max_hole_perimeter=float(max_hole_perimeter))
            self._set(*mesh.read())
            return
        vertices, faces, _ = comfy().fill_holes(self.vertices, self.faces.long(), max_perimeter=float(max_hole_perimeter))
        self._set(vertices, faces)

    def remove_faces(self, face_mask: torch.Tensor) -> None:
        keep = ~face_mask.to(device=self.faces.device, dtype=torch.bool)
        self._set(*_drop_unused_vertices(self.vertices, self.faces[keep]))

    def compute_vertex_normals(self) -> None:
        if native.cumesh() is not None:
            mesh = _native_mesh(self.vertices, self.faces)
            mesh.compute_vertex_normals()
            self._normals = mesh.read_vertex_normals()
            return
        self._normals = comfy().compute_vertex_normals(self.vertices, self.faces, weld=False)

    def read_vertex_normals(self) -> torch.Tensor:
        if self._normals is None:
            raise RuntimeError("compute_vertex_normals() must be called first")
        return self._normals

    def uv_unwrap(self, compute_charts_kwargs=None, return_vmaps: bool = False, verbose: bool = False,
                  uv_resolution: int = 2048, padding: int = 2):
        """Return (vertices, faces, uvs[, vmaps]); seam vertices are duplicated as in CuMesh."""
        if native.cumesh() is not None:
            mesh = _native_mesh(self.vertices, self.faces)
            return mesh.uv_unwrap(compute_charts_kwargs=dict(compute_charts_kwargs or {}),
                                  return_vmaps=return_vmaps, verbose=verbose)
        vmapping, indices, uvs = comfy().uv_unwrap(
            self.vertices, self.faces.long(), "pec", int(uv_resolution), int(padding), 0.0,
        )
        device = self.vertices.device
        vmaps = torch.from_numpy(np.ascontiguousarray(vmapping).astype(np.int64)).to(device)
        out = (
            self.vertices[vmaps],
            torch.from_numpy(np.ascontiguousarray(indices).astype(np.int32)).to(device),
            torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).to(device),
        )
        return (*out, vmaps) if return_vmaps else out

    def __getattr__(self, name):
        # Only reached for attributes not defined above (e.g. repair_non_manifold_edges).
        if name.startswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(f"cumesh.CuMesh.{name} is not used by UniForge3D's port and is not implemented")


class cuBVH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        device = vertices.device if vertices.is_cuda else _device()
        self.native = None
        if native.cumesh() is not None:
            self.native = native.cumesh().cuBVH(vertices.to(device).float(), faces.to(device).int())
            return
        self.triangles = vertices.to(device=device, dtype=torch.float32)[faces.to(device).long()]
        self.bvh = comfy().build_triangle_bvh(self.triangles)

    def unsigned_distance(self, positions: torch.Tensor, return_uvw: bool = False):
        if self.native is not None:
            return self.native.unsigned_distance(positions, return_uvw=return_uvw)
        prefix = positions.shape[:-1]
        query = positions.reshape(-1, 3).to(device=self.triangles.device, dtype=torch.float32)
        closest, face_id = comfy().closest_points_on_mesh_bvh(query, self.triangles, self.bvh, return_face=True)
        distance = (closest - query).norm(dim=-1)
        uvw = _barycentric(closest, self.triangles[face_id]) if return_uvw else None
        return (
            distance.view(*prefix),
            face_id.view(*prefix),
            uvw.view(*prefix, 3) if uvw is not None else None,
        )


def _barycentric(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    a, b, c = triangles.unbind(1)
    v0, v1, v2 = b - a, c - a, points - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack([1 - v - w, v, w], dim=-1)


def _remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh=None,
    drop_inner_shell: bool = False,
):
    if native.cumesh() is not None and not drop_inner_shell:
        return native.cumesh().remeshing.remesh_narrow_band_dc(
            vertices.float().contiguous(), faces.int().contiguous(), center=center, scale=scale,
            resolution=int(resolution), band=band, project_back=project_back, verbose=verbose,
            bvh=bvh.native if isinstance(bvh, cuBVH) else None,
        )
    new_vertices, new_faces, _ = comfy().remesh_narrow_band_dc(
        vertices.float(), faces.long(),
        resolution=int(resolution), band=float(band), project_back=float(project_back),
        qef=False, sign_mode="udf",
        drop_small_components=0.0,
        drop_inverted_components=bool(drop_inner_shell),
        drop_enclosed_components=False,
        scale=float(scale), center=center,
    )
    return new_vertices.float(), new_faces.int()


remeshing = SimpleNamespace(remesh_narrow_band_dc=_remesh_narrow_band_dc)
