# Hymba+

Hymba+ is a **modular hybrid Transformer–SSM architecture** that targets flexible composition, modern training recipes, and production-ready optimization. This repository provides a **clean skeleton** plus a **critical review** of the supplied design docs, updated to 2026-era best practices.

## What's in this repo

- **`ARCHITECTURE.md`**: consolidated design + 2026-era updates and correctness cautions.
- **`hymba_plus/`**: minimal, working registry + config loader + model scaffolding.
- **`configs/`**: YAML configs matching the dataclass loader (nested → flat mapping).
- **`training/`**: placeholders for pretrain/SFT/RL with clear extension points.

## Quick start

```bash
python -c "from hymba_plus.core.config import HymbaPlusConfig; cfg = HymbaPlusConfig.from_yaml('configs/hymba_plus.yaml'); print(cfg)"
```

## Repository status

This repository intentionally **starts minimal**: the structure is production-aligned, but heavy kernels (FlashAttention3/Triton/FP8) are **not implemented**. The goal is to avoid false claims and to provide **clear extension points** for real kernels and training loops.

See `ARCHITECTURE.md` for the full critique and an implementation roadmap.
