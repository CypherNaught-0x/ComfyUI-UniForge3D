

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Sequence

import torch
import trimesh
from huggingface_hub import snapshot_download

from ..models.dino import DinoV3Encoder
from ..models.loaders import load_model, resolve_checkpoint
from ..utils.device import release_cuda
from ..utils.mesh import (
    display_to_model,
    export_glb,
    normalize_to_cube,
    official_shape_only_postprocess,
    surface_voxelize,
)
from ..utils.preprocessing import BackgroundRemover, InputBatch, build_input_batch

OUTPUT_MODES = ("geometry_512", "geometry_1024", "textured")
# Approximate peak device memory per stage, passed to the memory hook (fp32 weights + activations).
_STAGE_MEMORY_GB = {"structure": 14, "shape512": 16, "shape1024": 20, "texture": 28, "decode": 8}
from .shape import decode_shape_latent, sample_shape_latent
from .sparse_structure import decode_sparse_structure, decoded_coordinates, sample_sparse_structure
from .texture import (
    decode_shape_and_texture,
    denormalize_shape_latent,
    remesh_and_bake_texture,
    sample_texture,
)
from .types import NormalizedShapeLatent, SparseStructure


class UniForge3DImageTo3DPipeline:


    def __init__(
        self,
        config: dict,
        *,
        device: int | torch.device = 0,
        low_vram: bool = True,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        dino: torch.nn.Module | None = None,
        remove_background: Callable | None = None,
        reserve_memory: Callable[[int], None] | None = None,
    ) -> None:
        """Models are built per stage on first use; with ``low_vram`` they are freed afterwards.

        ``dino`` and ``remove_background`` replace the Hub DINOv3 / BRIA RMBG models (e.g. with
        ComfyUI-managed ones). ``reserve_memory(bytes)`` is called before each stage so a host
        application can evict its own models first.
        """
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device("cuda", int(device))
        self.low_vram = low_vram
        self._download = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        self._models: dict[str, torch.nn.Module] = {}
        self._dino = dino
        self._remove_background = remove_background
        self._reserve_memory = reserve_memory

    @property
    def dino(self) -> torch.nn.Module:
        if self._dino is None:
            self._dino = DinoV3Encoder(self.config["external_models"]["dino"], **self._download).eval().requires_grad_(False)
        return self._dino

    @property
    def remove_background(self) -> Callable:
        if self._remove_background is None:
            self._remove_background = BackgroundRemover(
                self.config["image_preprocess"]["rembg_model"], self.device, **self._download,
            )
        return self._remove_background

    def _model(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            with torch.device("cpu"):
                self._models[name] = load_model(self.config, name)
        return self._models[name]

    def _reserve(self, stage: str) -> None:
        if self._reserve_memory is not None:
            self._reserve_memory(_STAGE_MEMORY_GB[stage] * 1024 ** 3)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: int = 0, low_vram: bool = True,
        revision: str | None = None, cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a local model directory or a Hugging Face repository."""
        root = Path(path).expanduser()
        if not root.is_dir():
            root = Path(snapshot_download(
                str(path), revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["pipeline.json", "uniforge3d/**/*.pt"],
            ))
        return cls.from_config(
            root / "pipeline.json", weights_root=root, device=device,
            low_vram=low_vram, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_config(
        cls, config_path: str | Path, *, device: int | torch.device = 0, low_vram: bool = True,
        weights_root: str | Path | None = None,
        cache_dir: str | Path | None = None, local_files_only: bool = False,
        **kwargs,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a JSON config; local weights default to the repository's weights/."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        root = (Path(weights_root).expanduser() if weights_root is not None
                else Path(__file__).resolve().parents[2] / "weights")
        options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
            weights = config[stage]["weights"]
            for key in ("checkpoint", "adapter_checkpoint", "dit_base", "vggt_base", "base_checkpoint"):
                if key in weights:
                    weights[key] = resolve_checkpoint(weights[key], root, **options)
            if "decoder" in config[stage]:
                decoder = config[stage]["decoder"]
                decoder["checkpoint"] = resolve_checkpoint(
                    decoder["checkpoint"], root, decoder=True, **options,
                )
        for key in ("shape_decoder_checkpoint", "texture_decoder_checkpoint"):
            config["texture"][key] = resolve_checkpoint(
                config["texture"][key], root, decoder=True, **options,
            )
        vggt = config["conditioner"]["vggt_feature_extractor"]
        vggt["checkpoint_path"] = resolve_checkpoint(vggt["checkpoint_path"], root, **options)
        return cls(config, device=device, low_vram=low_vram, **options, **kwargs)

    @contextmanager
    def _use_models(self, *names: str):
        """Place the named models on the device; with low_vram, free them afterwards."""
        try:
            models = [self._model(name) for name in names]
            for model in models:
                model.to(self.device)
            yield models
        finally:
            for name in names:
                model = self._models.get(name)
                if model is None:
                    continue
                if self.low_vram:
                    del self._models[name]
                else:
                    model.to("cpu")
            release_cuda()

    def _preprocess_images(self, images: Sequence[str | Path]) -> InputBatch:

        with ExitStack() as stack:
            background_loaded = False

            def remove_background(image):
                nonlocal background_loaded
                remover = self.remove_background
                if not background_loaded and hasattr(remover, "model"):
                    remover.model.to(self.device)
                    stack.callback(remover.model.to, "cpu")
                    background_loaded = True
                return remover(image)

            return build_input_batch(
                images,
                self.config["image_preprocess"],
                remove_background,
            )

    @torch.no_grad()
    def _encode_dino(self, batch: InputBatch) -> dict[str, torch.Tensor]:
        """Encode once per resolution and share 512px features across stages."""
        dino = self.dino
        managed = isinstance(dino, DinoV3Encoder)
        if managed:
            dino.to(self.device)
        try:
            images512 = batch.dino_images512.to(self.device, dtype=torch.float32)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                structure_features = dino(images512).cpu()
                shape1024_features = dino(
                    batch.dino_images1024.to(self.device, dtype=torch.float32)
                ).cpu()
        finally:
            if managed and self.low_vram:
                dino.to("cpu")
                release_cuda()
        return {
            "sparse_structure": structure_features,
            "shape512": structure_features,
            "shape1024": shape1024_features,
        }

    def _generate_structure(
        self,
        batch: InputBatch,
        dino_features: torch.Tensor,
    ) -> SparseStructure:
        sampling = self.config["sparse_structure"]["sampler"]
        self._reserve("structure")
        dit = self._model("structure_model").dit
        resolution = dit.resolution
        noise = torch.randn(
            (1, dit.in_channels, resolution, resolution, resolution),
            device="cpu",
        ).to(self.device)
        with self._use_models("structure_model") as (structure_model,):
            vggt_images = batch.vggt_images.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                latent = sample_sparse_structure(
                    structure_model,
                    noise,
                    vggt_images,
                    dino_features.to(self.device),
                    **sampling["args"],
                    **sampling["params"],
                )
        with self._use_models("structure_decoder") as (structure_decoder,):
            occupancy_logits = decode_sparse_structure(
                structure_decoder,
                latent,
                self.config["sparse_structure"]["decoder"].get("normalization"),
            )
        coords32, coords64 = decoded_coordinates(occupancy_logits)
        return SparseStructure(coords32.cpu(), coords64.cpu())

    def _sample_shape(
        self,
        model_name: str,
        config: dict,
        batch: InputBatch,
        *,
        latent_coords: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        dino_features: torch.Tensor,
        return_view_assignment: bool,
    ):
        latent_coords = latent_coords.int().contiguous()
        structure_conditioned_coords = structure_conditioned_coords.int().contiguous()
        with self._use_models(model_name, "conditioner") as (model, conditioner):
            conditioning = config["conditioning"]
            latent_coords = latent_coords.to(device=self.device, dtype=torch.int32)
            condition = conditioner.encode(
                model.adapter,
                dino_features=dino_features.to(device=self.device, dtype=torch.float32),
                vggt_images=batch.vggt_images.to(device=self.device, dtype=torch.float32),
                structure_conditioned_coords=structure_conditioned_coords,
                input_masks=batch.input_masks.to(device=self.device, dtype=torch.bool),
                latent_coords=latent_coords,
                router_grid=int(conditioning["grid_resolution"]),
            )
            latent = sample_shape_latent(
                model,
                latent_coords,
                condition,
                config["sampler"],
                description=f"Shape-{config['decoder']['resolution']} sampling",
            )
            view_assignment = (
                {"selected_view": condition["cond"]["selected_view"].cpu()}
                if return_view_assignment else None
            )
        if not torch.isfinite(latent.feats).all():
            raise RuntimeError("Shape sampling produced non-finite latent features")
        return latent, view_assignment

    def _generate_shape512(
        self,
        batch: InputBatch,
        structure: SparseStructure,
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        self._reserve("shape512")
        latent, _ = self._sample_shape(
            "shape512_model",
            self.config["structured_latent_512"],
            batch,
            latent_coords=structure.coords32,
            structure_conditioned_coords=structure.coords64,
            dino_features=dino_features,
            return_view_assignment=False,
        )
        with self._use_models("shape512_decoder") as (shape512_decoder,):
            mesh = decode_shape_latent(
                shape512_decoder,
                latent,
                self.config["structured_latent_512"]["decoder"].get("normalization"),
            )[0]
        return official_shape_only_postprocess(
            mesh.vertices,
            mesh.faces,
            **self._shape_postprocess(),
        )

    def _generate_shape1024(
        self,
        batch: InputBatch,
        coords64: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> tuple[NormalizedShapeLatent, dict[str, torch.Tensor]]:
        self._reserve("shape1024")
        latent, view_assignment = self._sample_shape(
            "shape1024_model",
            self.config["structured_latent_1024"],
            batch,
            latent_coords=coords64,
            structure_conditioned_coords=coords64,
            dino_features=dino_features,
            return_view_assignment=True,
        )
        if view_assignment is None:
            raise RuntimeError("Shape1024 must provide its texture routing result")
        shape = NormalizedShapeLatent(coords=latent.coords.cpu(), feats=latent.feats.cpu())
        return shape, view_assignment

    def _shape512_to_coords64(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Convert the already-remeshed Shape-512 surface to coords64."""
        config = self.config["structured_latent_handoff"]
        model_vertices = display_to_model(mesh.vertices)
        normalized_vertices, _ = normalize_to_cube(model_vertices, config["target_max_side"])
        xyz = torch.from_numpy(
            surface_voxelize(normalized_vertices, mesh.faces, config["surface_voxel_resolution"])
        )
        return torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz.int()), dim=1)

    def _generate_texture(
        self,
        batch: InputBatch,
        normalized_shape_latent: NormalizedShapeLatent,
        view_assignment: dict[str, torch.Tensor],
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        config = self.config["texture"]
        self._reserve("texture")
        with self._use_models(
            "texture_model", "shape1024_decoder", "texture_decoder", "conditioner",
        ) as (texture_model, shape1024_decoder, texture_decoder, conditioner):
            shape_vae_latent = denormalize_shape_latent(
                config,
                normalized_shape_latent,
                self.device,
            )
            texture_vae_latent = sample_texture(
                texture_model,
                config,
                conditioner,
                shape_vae_latent,
                batch,
                dino_features,
                view_assignment,
                self.device,
            )
            reference_mesh, pbr_voxel = decode_shape_and_texture(
                shape1024_decoder, texture_decoder, shape_vae_latent, texture_vae_latent
            )
            del shape_vae_latent, texture_vae_latent
        return remesh_and_bake_texture(
            config,
            self.config["mesh_postprocess"],
            reference_mesh,
            pbr_voxel,
            drop_inner_shell=self.config["mesh_postprocess"].get("drop_inner_shell", False),
        )

    def _decode_shape1024(self, normalized_shape_latent: NormalizedShapeLatent) -> trimesh.Trimesh:
        """Geometry-only 1024 output: decode the Shape1024 latent and apply the 512 postprocess."""
        self._reserve("decode")
        with self._use_models("shape1024_decoder") as (shape1024_decoder,):
            latent = denormalize_shape_latent(self.config["texture"], normalized_shape_latent, self.device)
            mesh = shape1024_decoder(latent)[0]
        return official_shape_only_postprocess(mesh.vertices, mesh.faces, **self._shape_postprocess())

    def _shape_postprocess(self) -> dict:
        keys = ("grid_size", "pre_simplify_target", "decimation_target", "remesh_band", "remesh_project", "drop_inner_shell")
        return {key: value for key, value in self.config["mesh_postprocess"].items() if key in keys}

    @torch.no_grad()
    def generate(
        self,
        batch: InputBatch,
        *,
        seed: int | None = None,
        output: str = "textured",
        progress: Callable[[str], None] | None = None,
    ) -> trimesh.Trimesh:
        """Run the stages needed for ``output`` (one of OUTPUT_MODES) on a preprocessed batch."""
        if output not in OUTPUT_MODES:
            raise ValueError(f"output must be one of {OUTPUT_MODES}, got {output!r}")
        seed = self.config["default_seed"] if seed is None else seed
        report = progress or (lambda stage: None)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        try:
            # Share one DINO model while preserving each stage's original precision.
            report("DINOv3 features")
            dino_features = self._encode_dino(batch)

            report("Sparse structure")
            sparse_structure = self._generate_structure(batch, dino_features["sparse_structure"])
            report("Shape 512")
            shape512_mesh = self._generate_shape512(
                batch, sparse_structure, dino_features["shape512"]
            )
            if output == "geometry_512":
                return shape512_mesh

            structure_conditioned_coords = self._shape512_to_coords64(shape512_mesh)
            report("Shape 1024")
            shape1024_latent, view_assignment = self._generate_shape1024(
                batch,
                structure_conditioned_coords,
                dino_features["shape1024"],
            )
            if output == "geometry_1024":
                report("Decode 1024")
                return self._decode_shape1024(shape1024_latent)

            # Texture reuses the Shape1024 latent and its per-token view routing.
            report("PBR texture")
            return self._generate_texture(
                batch,
                shape1024_latent,
                view_assignment,
                dino_features["shape1024"],
            )
        finally:
            release_cuda()

    @torch.no_grad()
    def run(
        self,
        images: Sequence[str | Path],
        output: str | Path,
        *,
        seed: int | None = None,
        overwrite: bool = False,
    ) -> Path:

        output_path = Path(output).expanduser().resolve()
        if output_path.exists() and (not overwrite or not output_path.is_file()):
            raise FileExistsError(
                f"Output already exists: {output_path}; use --overwrite to recompute"
            )
        batch = self._preprocess_images(images)
        textured_mesh = self.generate(batch, seed=seed, output="textured")
        return export_glb(textured_mesh, output_path, overwrite=overwrite)
