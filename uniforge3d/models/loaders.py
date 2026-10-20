"""Initialize models and load original training weights over their pretrained bases."""

import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, load_model

from ..compat.attrdict import AttrDict as edict
from ..compat.lora import merge_peft_state_dict

from . import __getattr__ as get_model_class
from .conditioning import StructuredLatentConditioner
from .dino import DinoV3Encoder
from .lora import attach_stage1_lora
from .structured_latent_flow import VGGTConditionedSLatFlowModel
from .vaes.flexi_dual_grid import FlexiDualGridVaeDecoder
from .vaes.sparse_structure import SparseStructureDecoder
from .vaes.sparse_unet import SparseUnetVaeDecoder


def resolve_checkpoint(
    checkpoint: str | Path | dict[str, str],
    root: Path,
    *,
    decoder: bool = False,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a local checkpoint or an explicit Hub reference."""
    if isinstance(checkpoint, (str, Path)):
        return str(root / Path(checkpoint).expanduser())
    filename = checkpoint["filename"]
    # Pre-fetched copies laid out as <root>/external/<repo_id>/<filename> avoid any Hub access.
    local = root / "external" / checkpoint["repo_id"] / filename
    if decoder and Path(f"{local}.json").is_file() and Path(f"{local}.safetensors").is_file():
        return str(local)
    if not decoder and local.is_file():
        return str(local)
    options = {
        "repo_id": checkpoint["repo_id"],
        "revision": checkpoint.get("revision"),
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if decoder:
        folder = snapshot_download(
            **options,
            allow_patterns=[f"{filename}.json", f"{filename}.safetensors"],
        )
        return str(Path(folder) / filename)
    return hf_hub_download(filename=filename, **options)


def _decoder_args(checkpoint: str | Path, expected_name: str) -> dict[str, Any]:
    """Read and validate the constructor arguments beside a decoder checkpoint."""
    config_path = Path(f"{checkpoint}.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["name"] != expected_name:
        raise ValueError(
            f"Expected {expected_name} in {config_path}, found {config['name']}"
        )
    return config["args"]

def _load_decoder_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Strictly load the Safetensors file beside a decoder configuration."""
    weights_path = Path(f"{checkpoint}.safetensors")
    load_model(model, weights_path, strict=True, device="cpu")
    model.eval().requires_grad_(False)

def load_shape_decoder(
    checkpoint: str | Path,
    resolution: int,
) -> FlexiDualGridVaeDecoder:
    """Construct the shape decoder and load its frozen inference weights."""
    model = FlexiDualGridVaeDecoder(
        **_decoder_args(checkpoint, "FlexiDualGridVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    model.set_resolution(int(resolution))
    return model

def load_sparse_structure_decoder(checkpoint: str | Path) -> SparseStructureDecoder:
    model = SparseStructureDecoder(**_decoder_args(checkpoint, "SparseStructureDecoder"))
    _load_decoder_weights(model, checkpoint)
    return model

def load_texture_decoder(checkpoint: str | Path) -> SparseUnetVaeDecoder:
    """Construct the PBR decoder and load its frozen inference weights."""
    model = SparseUnetVaeDecoder(
        **_decoder_args(checkpoint, "SparseUnetVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    return model

def _read_weights(checkpoint: str) -> dict[str, torch.Tensor]:
    if str(checkpoint).endswith(".safetensors"):
        return load_file(checkpoint, device="cpu")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    return state["model"] if "model" in state else state


def _load_trainable_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """A compact checkpoint must cover every trainable parameter over its loaded base."""
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = required - state.keys()
    if missing:
        raise RuntimeError(f"Missing training weights: {sorted(missing)[:20]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected training weights: {result.unexpected_keys[:20]}")


def _load_structure_weights(model: torch.nn.Module, weights: dict) -> None:
    # Bases are loaded into FP32 modules, just as in the training constructor.
    dit = _read_weights(weights["dit_base"])
    # RoPE phases are deterministic buffers, absent from the official base file.
    dit.setdefault("rope_phases", model.dit.rope_phases)
    model.dit.load_state_dict(dit, strict=True)
    vggt = _read_weights(weights["vggt_base"])
    vggt = {name.removeprefix("model."): value for name, value in vggt.items()}
    # The published model omits the unused camera head and mask token.
    vggt = {
        name: value for name, value in vggt.items()
        if not name.startswith("camera_head.") and name != "aggregator.patch_embed.mask_token"
    }
    model.vggt.load_state_dict(vggt, strict=True)
    attach_stage1_lora(model.dit, **weights["dit_lora"])
    attach_stage1_lora(model.vggt, **weights["vggt_lora"])

    state = _read_weights(weights["checkpoint"])
    state = {
        re.sub(r"^fusion_groups\.(\d+)\.f_from_z\.", r"dit_to_vggt.\1.", name): value
        for name, value in state.items()
        if not re.match(r"^fusion_groups\.\d+\.z_from_f\.", name)
    }
    _load_trainable_weights(model, state)
    model.eval().requires_grad_(False)


def _load_flow_model(spec: edict, weights: dict):
    if spec.name != "VGGTConditionedSLatFlowModel":
        raise ValueError(f"Unsupported flow model: {spec.name}")
    model = VGGTConditionedSLatFlowModel(**spec.args)
    if "base_checkpoint" in weights:
        raise NotImplementedError("Adapter-only checkpoints over a separate base are not supported")
    model.denoiser.requires_grad_(False)
    # PEFT-format checkpoint, merged into the plain denoiser instead of wrapping it with peft.
    state = merge_peft_state_dict(_read_weights(weights["checkpoint"]), **weights["lora"])
    model.denoiser.load_state_dict(state, strict=True)
    model.adapter.load_state_dict(_read_weights(weights["adapter_checkpoint"]), strict=True)
    model.eval().requires_grad_(False)
    return model


def _load_structure_model(config: dict[str, Any]):
    structure = edict(config["sparse_structure"])
    model = get_model_class(structure.model.name)(**structure.model.args)
    _load_structure_weights(model, structure.weights)
    return model


def _load_shape_decoder_for(stage: str):
    def load(config: dict[str, Any]):
        if stage == "texture":
            return load_shape_decoder(config["texture"]["shape_decoder_checkpoint"], int(config["texture"]["resolution"]))
        decoder = config[stage]["decoder"]
        return load_shape_decoder(decoder["checkpoint"], int(decoder["resolution"]))
    return load


def _load_flow_model_for(stage: str):
    def load(config: dict[str, Any]):
        spec = edict(config[stage])
        return _load_flow_model(spec.model, spec.weights)
    return load


# Name -> loader, so a pipeline can build only the models one stage needs.
MODEL_LOADERS = {
    "conditioner": lambda config: StructuredLatentConditioner(**config["conditioner"]),
    "structure_model": _load_structure_model,
    "structure_decoder": lambda config: load_sparse_structure_decoder(config["sparse_structure"]["decoder"]["checkpoint"]),
    "shape512_model": _load_flow_model_for("structured_latent_512"),
    "shape512_decoder": _load_shape_decoder_for("structured_latent_512"),
    "shape1024_model": _load_flow_model_for("structured_latent_1024"),
    "shape1024_decoder": _load_shape_decoder_for("texture"),
    "texture_model": _load_flow_model_for("texture"),
    "texture_decoder": lambda config: load_texture_decoder(config["texture"]["texture_decoder_checkpoint"]),
}


def load_model(config: dict[str, Any], name: str):
    return MODEL_LOADERS[name](config)


def load_models(
    config: dict[str, Any], *, cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    dino = DinoV3Encoder(
        config["external_models"]["dino"],
        cache_dir=cache_dir, local_files_only=local_files_only,
    ).eval().requires_grad_(False)
    return {"dino": dino, **{name: load_model(config, name) for name in MODEL_LOADERS}}
