"""UniForge3D for ComfyUI on ROCm/AMD (and any PyTorch GPU): no CUDA-only extensions required."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
