"""ComfyUI nodes for UniForge3D (multi-view images -> 3D) on the native ComfyUI/ROCm stack.

The nodes only add functionality: nothing in ComfyUI or other node packs is patched. Weights are
read from ``models/uniforge3d/<name>/`` (``pipeline.json``, ``uniforge3d/**.pt`` and
``external/<repo_id>/<file>`` for the TRELLIS checkpoints); DINOv3 comes from a ClipVision input.
"""

from __future__ import annotations

import copy
import json
import logging
import os

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import Types

from .uniforge3d import UniForge3DImageTo3DPipeline
from .uniforge3d.compat import check_backend
from .uniforge3d.models.dino import ComfyDinoV3Encoder
from .uniforge3d.pipelines.uniforge3d_image_to_3d import OUTPUT_MODES
from .uniforge3d.utils.preprocessing import preprocess_rgba, stack_views

log = logging.getLogger("UniForge3D")

FOLDER = "uniforge3d"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

MASK_MODES = ("inverted (LoadImage alpha)", "foreground", "none (opaque input)")
MAX_VIEWS = 6
STAGE_NAMES = ("DINOv3 features", "Sparse structure", "Shape 512", "Shape 1024", "Decode 1024", "PBR texture", "done")


def _model_dirs() -> list[str]:
    names = []
    for root in folder_paths.get_folder_paths(FOLDER):
        if os.path.isdir(root):
            names += [d for d in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, d, "pipeline.json"))]
    return names or ["(place weights in models/uniforge3d/<name>/)"]


def _model_root(name: str) -> str:
    for root in folder_paths.get_folder_paths(FOLDER):
        if os.path.isfile(os.path.join(root, name, "pipeline.json")):
            return os.path.join(root, name)
    raise FileNotFoundError(f"No models/uniforge3d/{name}/pipeline.json")


class UniForge3DLoader:
    """Configures a pipeline. Weights load lazily, one stage at a time, when Generate runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_dirs(),),
                "clip_vision": ("CLIP_VISION", {"tooltip": "DINOv3 ViT-L/16 (e.g. dino_v3_L_naf_fp32), the same one TRELLIS.2/Pixal3D use."}),
                "keep_models_in_ram": ("BOOLEAN", {"default": False, "tooltip": "Keep stage weights (~31 GB fp32) in system RAM between runs instead of reloading from disk."}),
            }
        }

    RETURN_TYPES = ("UNIFORGE3D_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "3d/UniForge3D"

    def load(self, model, clip_vision, keep_models_in_ram):
        check_backend()
        root = _model_root(model)
        device = comfy.model_management.get_torch_device()

        def reserve(bytes_needed: int) -> None:
            comfy.model_management.free_memory(bytes_needed, device)

        pipeline = UniForge3DImageTo3DPipeline.from_config(
            os.path.join(root, "pipeline.json"), weights_root=root, device=device,
            low_vram=not keep_models_in_ram, local_files_only=True,
            dino=ComfyDinoV3Encoder(clip_vision), reserve_memory=reserve,
        )
        return (pipeline,)


def _views_to_rgba(images: torch.Tensor, masks: torch.Tensor | None, mask_mode: str) -> list[np.ndarray]:
    """IMAGE batch [B,H,W,C] + MASK [B,H,W] -> float RGBA views whose alpha is the foreground."""
    rgb = images[..., :3].float().clamp(0, 1).cpu().numpy()
    if mask_mode == MASK_MODES[2]:
        alpha = np.ones(rgb.shape[:3], dtype=np.float32)
    elif masks is None:
        raise ValueError("UniForge3D needs a mask per view (connect LoadImage's MASK, or choose mask_mode 'none').")
    else:
        alpha = masks.float().cpu().numpy()
        if alpha.ndim == 2:
            alpha = alpha[None]
        if mask_mode == MASK_MODES[0]:
            alpha = 1.0 - alpha       # LoadImage's MASK is 1 where the PNG is transparent
        if alpha.shape[:3] != rgb.shape[:3]:
            if alpha.shape[1:3] == (64, 64) and not alpha.any():
                raise ValueError("This view has no alpha channel (LoadImage returned an empty 64x64 mask). "
                                 "Use RGBA images, a background-removal mask with mask_mode 'foreground', "
                                 "or mask_mode 'none (opaque input)'.")
            raise ValueError(f"Mask {tuple(alpha.shape)} does not match image {tuple(rgb.shape[:3])}")
    return [np.concatenate([rgb[i], alpha[i][..., None]], axis=-1) for i in range(rgb.shape[0])]


class UniForge3DViews:
    """Collects up to six views of any size (with their masks) for UniForge3D Generate."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_VIEWS + 1):
            optional[f"image_{i}"] = ("IMAGE",)
            optional[f"mask_{i}"] = ("MASK",)
        return {
            "required": {
                "mask_mode": (MASK_MODES, {"default": MASK_MODES[0], "tooltip": "inverted: LoadImage's MASK from an RGBA PNG. foreground: 1 = object (background-removal nodes). none: whole image is the object."}),
            },
            "optional": {"previous": ("UNIFORGE3D_VIEWS", {"tooltip": "Chain for more than six views."}), **optional},
        }

    RETURN_TYPES = ("UNIFORGE3D_VIEWS",)
    FUNCTION = "collect"
    CATEGORY = "3d/UniForge3D"

    def collect(self, mask_mode, previous=None, **kwargs):
        views = list(previous or [])
        for i in range(1, MAX_VIEWS + 1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                views += _views_to_rgba(image, kwargs.get(f"mask_{i}"), mask_mode)
        if not views:
            raise ValueError("Connect at least one view image")
        return (views,)


def _to_comfy_mesh(mesh) -> Types.MESH:
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))[None]
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int32))[None]
    normals = torch.from_numpy(np.asarray(mesh.vertex_normals, dtype=np.float32))[None]
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if getattr(visual, "uv", None) is None or material is None:
        return Types.MESH(vertices=vertices, faces=faces, normals=normals)
    uvs = np.asarray(visual.uv, dtype=np.float32).copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]       # trimesh stores OpenGL-style V; ComfyUI MESH uses glTF V
    base = np.asarray(material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
    mr = np.asarray(material.metallicRoughnessTexture.convert("RGB"), dtype=np.float32) / 255.0
    return Types.MESH(
        vertices=vertices, faces=faces, normals=normals, uvs=torch.from_numpy(uvs)[None],
        texture=torch.from_numpy(base)[None], metallic_roughness=torch.from_numpy(mr)[None],
    )


class UniForge3DGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("UNIFORGE3D_PIPELINE",),
                "views": ("UNIFORGE3D_VIEWS", {"tooltip": "Unposed views of one object from UniForge3D Views; the first is the anchor view."}),
                "output": (OUTPUT_MODES, {"default": "textured"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "decimation_target": ("INT", {"default": 1_000_000, "min": 10_000, "max": 5_000_000, "step": 10_000}),
            },
            "optional": {
                "structure_steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "0 = pipeline.json default (12)."}),
                "shape_steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Shape 512 and 1024 sampling steps; 0 = pipeline.json default (20)."}),
                "texture_steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "0 = pipeline.json default (20)."}),
                "drop_inner_shell": ("BOOLEAN", {"default": False, "tooltip": "Remove the inside-out offset shell the UDF remesh leaves inside closed parts (not in upstream)."}),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate"
    CATEGORY = "3d/UniForge3D"

    def generate(self, pipeline, views, output, seed, texture_size, decimation_target, structure_steps=0, shape_steps=0, texture_steps=0, drop_inner_shell=False):
        config = copy.deepcopy(pipeline.config)
        overrides = {"sparse_structure": structure_steps, "structured_latent_512": shape_steps,
                     "structured_latent_1024": shape_steps, "texture": texture_steps}
        for stage, steps in overrides.items():
            if steps:
                config[stage]["sampler"]["params"]["steps"] = int(steps)
        config["texture"]["texture_size"] = int(texture_size)
        config["texture"]["decimation_target"] = int(decimation_target)
        config["mesh_postprocess"]["decimation_target"] = int(decimation_target)
        config["mesh_postprocess"]["drop_inner_shell"] = bool(drop_inner_shell)

        batch = stack_views([preprocess_rgba(view, config["image_preprocess"], f"view {i}") for i, view in enumerate(views)])

        bar = comfy.utils.ProgressBar(len(STAGE_NAMES) - 1)

        def progress(stage: str) -> None:
            comfy.model_management.throw_exception_if_processing_interrupted()
            log.info("UniForge3D: %s", stage)
            bar.update_absolute(STAGE_NAMES.index(stage))

        original = pipeline.config
        pipeline.config = config
        devices = list(range(torch.cuda.device_count())) if pipeline.device.type == "cuda" else []
        try:
            # The pipeline seeds the global RNGs; restore them so later nodes in this process
            # (e.g. Pixal3D) see the same RNG state as if UniForge3D had not run.
            with torch.random.fork_rng(devices=devices):
                mesh = pipeline.generate(batch, seed=int(seed), output=output, progress=progress)
        finally:
            pipeline.config = original
        bar.update_absolute(len(STAGE_NAMES) - 1)
        log.info("UniForge3D %s: %d vertices, %d faces", output, len(mesh.vertices), len(mesh.faces))
        return (_to_comfy_mesh(mesh),)


NODE_CLASS_MAPPINGS = {
    "UniForge3DLoader": UniForge3DLoader,
    "UniForge3DViews": UniForge3DViews,
    "UniForge3DGenerate": UniForge3DGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniForge3DLoader": "UniForge3D Loader",
    "UniForge3DViews": "UniForge3D Views",
    "UniForge3DGenerate": "UniForge3D Generate (multi-view)",
}
