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
- GQA-compatible attention and vectorized MoE (batched expert matmul).
- Nanochat-style training loop primitives (cosine LR, warmup, grad accumulation, bf16 autocast, grad clipping).
Hymba+ is a **modular hybrid Transformer–SSM architecture** that targets flexible composition, modern training recipes, and production-ready optimization. This repository provides a **clean skeleton** plus a **critical review** of the supplied design docs, updated to 2026-era best practices.

## What's in this repo

- **`ARCHITECTURE.md`**: consolidated design + 2026-era updates and correctness cautions.
- **`hymba_plus/`**: minimal, working registry + config loader + model scaffolding.
- **`configs/`**: YAML configs matching the dataclass loader (nested → flat mapping).
- **`training/`**: placeholders for pretrain/SFT/RL with clear extension points.

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
python -c "from hymba_plus.core.config import HymbaPlusConfig; cfg = HymbaPlusConfig.from_yaml('configs/hymba_plus.yaml'); print(cfg)"
```

## Repository status

This repository intentionally **starts minimal**: the structure is production-aligned, but heavy kernels (FlashAttention3/Triton/FP8) are **not implemented**. The goal is to avoid false claims and to provide **clear extension points** for real kernels and training loops.

See `ARCHITECTURE.md` for the full critique and an implementation roadmap.
