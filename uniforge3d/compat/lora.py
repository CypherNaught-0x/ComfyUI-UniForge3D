"""Load PEFT-format LoRA checkpoints without the ``peft`` package.

The released stage-2/3 denoisers are full ``get_peft_model`` state dicts
(``base_model.model.<module>.base_layer.weight`` plus ``lora_A/lora_B.default.weight``).
The adapters are merged into the base weights at load time: ``W += (alpha / r) * B @ A``.
With dropout 0 and no bias adaptation (the released config) this equals PEFT's forward pass
up to floating-point rounding, and removes the extra matmuls at inference.
"""

from __future__ import annotations

import torch

_PREFIX = "base_model.model."


def merge_peft_state_dict(state: dict[str, torch.Tensor], *, r: int, lora_alpha: float,
                          bias: str = "none", lora_dropout: float = 0.0, **_unused) -> dict[str, torch.Tensor]:
    if bias != "none":
        raise NotImplementedError(f"PEFT bias={bias!r} is not supported")
    scaling = float(lora_alpha) / int(r)
    merged: dict[str, torch.Tensor] = {}
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith(_PREFIX):
            raise ValueError(f"Unexpected non-PEFT key {key!r}")
        key = key[len(_PREFIX):]
        if ".lora_A." in key or ".lora_B." in key:
            module, rest = key.split(".lora_", 1)
            adapters.setdefault(module, {})[rest[0]] = value
        else:
            merged[key.replace(".base_layer.", ".")] = value
    for module, pair in adapters.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete LoRA pair for {module}")
        weight_key = f"{module}.weight"
        base = merged[weight_key]
        delta = pair["B"].float() @ pair["A"].float()
        merged[weight_key] = (base.float() + scaling * delta).to(base.dtype)
    return merged
