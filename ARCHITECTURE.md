# Hymba+ Architecture (2026 Review + Build Plan)

> **Positioning**: This document consolidates the provided design notes into a critical, production‑oriented plan. It flags mismatches, missing pieces, and 2026‑era best practices that must be present for a credible implementation.

## 1) Executive assessment

The original documents describe a highly modular hybrid model (Transformer + SSM) with rich recipes, kernels, and RL alignment. However, **most of those claims are aspirational**: there is no guarantee of kernel availability, config compatibility, cache correctness, or training stability. A production‑grade Hymba+ must explicitly address correctness (cache/positioning), config reproducibility, and performance realism (vectorized MoE, real SSM kernels).

This repository is built to:
1. **Avoid false claims** (no stubbed “FlashAttention3” without real kernels),
2. **Make config and model wiring deterministic**, and
3. Provide **extension points** for true 2026‑level optimizations (FP8, kernel fusion, cache compression).

## 2) Critical review of existing docs (what’s correct vs. risky)

### ✅ Valid architecture ideas
- **Registry‑based component composition** is correct and necessary for modularity.
- **Recipe system** is a good UX for switching between pure Transformer, pure SSM, and hybrid.
- **Hybrid ratio** is useful if it is implemented *explicitly* (interleaved vs. parallel blocks).
- **GRPO/DPO** are appropriate alignment choices when data availability differs.

### ⚠️ Risks and gaps
1. **Cache + meta tokens correctness**
   - Meta tokens must be injected **only on the first step** of generation; otherwise KV caches corrupt.
   - Position IDs must be **offset by cached length** to maintain RoPE correctness.

2. **Configuration mismatch**
   - Nested YAML structures do not align with most dataclass loaders. This causes silent config drift.
   - A **deterministic loader** that maps nested YAML → flat dataclass is required.

3. **Hybrid block semantics**
   - If “sliding‑window attention” is auto‑applied to non‑global layers, then a “pure Transformer” is not actually pure.
   - Hybrid blocks need explicit `block_pattern`: `parallel`, `interleaved`, or `custom`.

4. **Performance realism**
   - Any SSM that loops over sequence positions in Python is non‑viable.
   - MoE routing must be **vectorized** (token grouping + batched expert calls). Python loops are unacceptable.

5. **Kernel claims**
   - FlashAttention‑3, Triton kernels, FP8, and cache compression must only be advertised **if implemented**.
   - Otherwise, expose them as optional future features, **not enabled by default**.

## 3) 2026-era best practices that must be included

### 3.1 Correctness + reproducibility
- **Deterministic config loader** with versioned schema.
- **Cache‑aware position IDs** and **meta token gating**.
- **Strict shape guards** (head dims, rotary dims, Mamba state dims).
- **Seeded training** and dataset manifest logging.

### 3.2 Architecture enhancements
- **None‑component stubs** (NoneAttention/NoneSSM/NoneFusion) to remove special‑case logic.
- **Block pattern support**:
  - `parallel`: attention + SSM fused per layer
  - `interleaved`: attention and SSM blocks alternate
  - `custom`: layer‑wise toggles
- **RoPE scaling variants**: linear / NTK / YaRN for long context.
- **KV cache compression** (MLA cache or quantized cache) optional path.

### 3.3 Performance pathways
- **SSM**: chunked scan or fused kernel (Triton/CUDA). Avoid Python loops.
- **MoE**: vectorized routing + optional fused kernels.
- **FlashAttention‑3**: use only with hardware‑valid FP8/FP16 paths and gated by runtime checks.
- **Current repository** includes a minimal Triton gate‑fusion kernel as a first step toward SSM kernelization.
- **Current repository** routes FlashAttention through PyTorch SDP, which selects Flash kernels on supported CUDA builds.

### 3.4 Training + alignment
- **Stage 0**: data curation and tokenizer lock‑in (single tokenizer across chat + time‑series).
- **Stage 1**: pretrain with curriculum on sequence length and mixture weights.
- **Stage 2**: SFT with structured tool‑call schemas.
- **Stage 3**: GRPO (verifiable) or DPO (preference). Use length/style penalties.
- **Stage 4**: distillation (logit + sequence). Preserve specialized heads.
- **Stage 5**: PTQ/QAT + post‑quant alignment.

## 4) This repository: what is implemented now

This repo intentionally implements **only the core scaffolding** to make the system honest and buildable:

- **Registry**: minimal, deterministic component registry.
- **Config**: dataclass‑based config + nested YAML loader.
- **Model skeleton**: a minimal `HymbaPlus` class that wires config and registry.
- **Clear extension points** for kernels, MoE, and training.
- **Triton gate‑fusion kernel** used in the SSM gating path when available.
- **Flash attention path** via PyTorch SDP for CUDA builds (not a full FA3 implementation).
- **Nanochat‑style training loop primitives** (cosine schedule, warmup, bf16 autocast, grad accumulation, grad clipping).
- **Vectorized MoE** with batched expert matmul (no Python token loops).

Everything else is left as **explicit TODOs**, not marketing claims.

## 5) Implementation roadmap (priority order)

### P0 — Correctness
- Cache/meta‑token gating + position ID offsets.
- Strict config validation (head dims, rope dims).
- Unit tests for cache correctness and config parsing.

### P1 — Architecture completeness
- `block_pattern` support.
- None‑components.
- Time‑series tokenizer + forecaster head.

### P2 — Performance
- SSM kernel (chunked or fused).
- MoE routing vectorization.
- Optional FlashAttention‑3 and cache compression.

### P3 — Evaluation
- LM eval harness, time‑series benchmarks, and tool‑call accuracy tests.

---

## Appendix: structure of this repo

```
/ARCHITECTURE.md
/README.md
/hymba_plus
  /core           # config, registry, base classes
  /components     # component stubs
  /blocks         # block stubs
  /models         # model skeletons
/configs          # nested YAML configs (loader supported)
/training         # placeholders
```
