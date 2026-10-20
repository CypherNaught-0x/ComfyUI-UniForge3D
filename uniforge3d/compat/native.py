"""Optional natively-built extensions (FlexGEMM, CuMesh), used when importable.

FlexGEMM builds for ROCm unmodified and CuMesh with ``patches/cumesh-hip.patch`` (see
docs/AMD_PORT.md). On gfx1151 they are 4-17x faster than the pure-torch fallbacks for the ops
UniForge3D uses. ``UNIFORGE3D_NATIVE_EXT=0`` forces the pure-torch path; ``=1`` makes a missing
extension an error instead of a silent fallback.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache

log = logging.getLogger("UniForge3D")


def _mode() -> str:
    mode = os.environ.get("UNIFORGE3D_NATIVE_EXT", "auto").lower()
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"UNIFORGE3D_NATIVE_EXT must be auto, 0 or 1, got {mode!r}")
    return mode


@lru_cache(maxsize=None)
def module(name: str):
    """The native module ``name`` (e.g. ``"cumesh"``), or None when disabled/unavailable."""
    mode = _mode()
    if mode == "0":
        return None
    try:
        loaded = importlib.import_module(name)
    except ImportError as error:
        if mode == "1":
            raise ImportError(f"UNIFORGE3D_NATIVE_EXT=1 but {name} is not importable: {error}") from error
        return None
    log.info("UniForge3D: using native %s from %s", name, getattr(loaded, "__file__", "?"))
    return loaded


def flex_gemm():
    return module("flex_gemm")


def cumesh():
    return module("cumesh")
