# Hymba+

Hymba+ is a **modular hybrid Transformer–SSM architecture** with runnable core components, training scripts, and a validation notebook. This repository is intentionally honest: heavy kernels (FlashAttention‑3, Triton, FP8) are **not** fully included, but extension points and configuration are wired for future integration.

## Highlights

- Modular components (attention, SSM, MLP, fusion, norm, embeddings).
- Hybrid blocks (Transformer / Mamba-like / Hybrid).
- Deterministic nested YAML → dataclass configuration loader.
- Scripts for quick smoke tests (pretrain/SFT/RL/eval/export).
- Validation notebook with intermediate outputs and basic visualization.
- Optional Triton gate-fusion kernel for the SSM gating path when CUDA + Triton are available.
- Flash attention path wired via PyTorch SDP when enabled (requires CUDA).
- Nanochat-style training loop primitives (cosine LR, warmup, grad accumulation, bf16 autocast).

## Quick start

```bash
python scripts/evaluate.py --config configs/hymba_plus.yaml
```

## Validation notebook

Open `notebooks/01_validation.ipynb` to verify:
- model construction
- forward pass
- parameter counts
- basic logits visualization

## Repository status

This is a **starter scaffold**. It runs, but it is **not** optimized for speed. See `ARCHITECTURE.md` for the critical review and 2026‑era roadmap.
