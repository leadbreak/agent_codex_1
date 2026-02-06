from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AttentionConfig:
    type: str = "standard"
    num_heads: int = 8
    num_kv_heads: int = 4
    dropout: float = 0.0
    use_flash: bool = False
    flash_version: int = 2
    global_layers: Optional[list[int]] = None
    local_type: Optional[str] = None
    window_size: Optional[int] = None


@dataclass
class SSMConfig:
    type: str = "mamba"
    expand: int = 2
    d_state: int = 128
    d_conv: int = 4
    dropout: float = 0.0


@dataclass
class FusionConfig:
    type: str = "average"
    init_beta: float = 0.5
    learnable: bool = True


@dataclass
class MLPConfig:
    type: str = "swiglu"
    expand: int = 4
    dropout: float = 0.0
    n_experts: int = 0
    top_k: int = 2
    balance_mode: Optional[str] = None


@dataclass
class NormConfig:
    type: str = "rmsnorm"
    eps: float = 1e-6


@dataclass
class EmbeddingConfig:
    type: str = "token"
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None
    rope_scale_factor: float = 1.0
    max_position_embeddings: int = 2048
    num_meta_tokens: int = 0


@dataclass
class ArchitectureConfig:
    transformer_ratio: float = 0.5
    block_pattern: str = "parallel"
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ssm: SSMConfig = field(default_factory=SSMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class QuantizationConfig:
    enabled: bool = False
    method: str = "fp8"


@dataclass
class KernelConfig:
    use_triton: bool = False
    use_flash_attention: bool = False


@dataclass
class MemoryConfig:
    gradient_checkpointing: bool = False
    kv_cache_compression: bool = False


@dataclass
class OptimizationConfig:
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    kernels: KernelConfig = field(default_factory=KernelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class TrainingStageConfig:
    batch_size: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    max_steps: int = 0
    epochs: int = 0
    group_size: int = 0
    method: str = ""


@dataclass
class TrainingConfig:
    stage: str = "pretrain"
    pretrain: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    sft: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    rl: TrainingStageConfig = field(default_factory=TrainingStageConfig)


@dataclass
class HymbaPlusConfig:
    name: str = "hymba_plus"
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 32000
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """구성 값의 기본 유효성을 검사합니다."""
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model은 n_heads로 나누어 떨어져야 합니다.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads는 n_kv_heads로 나누어 떨어져야 합니다.")
        if self.architecture.embedding.rope_scale_factor <= 0:
            raise ValueError("rope_scale_factor는 0보다 커야 합니다.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HymbaPlusConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HymbaPlusConfig":
        model = data.get("model", {})
        architecture = data.get("architecture", {})
        optimization = data.get("optimization", {})
        training = data.get("training", {})

        return cls(
            name=model.get("name", cls.name),
            d_model=model.get("d_model", cls.d_model),
            n_layers=model.get("n_layers", cls.n_layers),
            n_heads=model.get("n_heads", cls.n_heads),
            n_kv_heads=model.get("n_kv_heads", cls.n_kv_heads),
            vocab_size=model.get("vocab_size", cls.vocab_size),
            architecture=_build_architecture(architecture),
            optimization=_build_optimization(optimization),
            training=_build_training(training),
        )


def _build_architecture(data: Dict[str, Any]) -> ArchitectureConfig:
    return ArchitectureConfig(
        transformer_ratio=data.get("transformer_ratio", 0.5),
        block_pattern=data.get("block_pattern", "parallel"),
        attention=AttentionConfig(**data.get("attention", {})),
        ssm=SSMConfig(**data.get("ssm", {})),
        fusion=FusionConfig(**data.get("fusion", {})),
        mlp=MLPConfig(**data.get("mlp", {})),
        norm=NormConfig(**data.get("norm", {})),
        embedding=EmbeddingConfig(**data.get("embedding", {})),
    )


def _build_optimization(data: Dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(
        quantization=QuantizationConfig(**data.get("quantization", {})),
        kernels=KernelConfig(**data.get("kernels", {})),
        memory=MemoryConfig(**data.get("memory", {})),
    )


def _build_training(data: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        stage=data.get("stage", "pretrain"),
        pretrain=TrainingStageConfig(**data.get("pretrain", {})),
        sft=TrainingStageConfig(**data.get("sft", {})),
        rl=TrainingStageConfig(**data.get("rl", {})),
    )
