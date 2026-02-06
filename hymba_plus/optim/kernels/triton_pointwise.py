from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.jit
    def _gate_mul_kernel(x_ptr, gate_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        out = x * tl.sigmoid(gate)
        tl.store(out_ptr + offsets, out, mask=mask)


def gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors")

    out = torch.empty_like(x)
    n_elements = x.numel()
    block = 1024
    grid = (triton.cdiv(n_elements, block),)
    _gate_mul_kernel[grid](
        x,
        gate,
        out,
        n_elements,
        BLOCK=block,
    )
    return out
