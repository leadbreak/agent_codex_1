from hymba_plus.core.registry import Registry
from hymba_plus.components.norm.layernorm import LayerNorm
from hymba_plus.components.norm.rmsnorm import RMSNorm

NORM_REGISTRY = Registry("norm")

NORM_REGISTRY.register("rmsnorm")(RMSNorm)
NORM_REGISTRY.register("layernorm")(LayerNorm)

__all__ = ["NORM_REGISTRY", "RMSNorm", "LayerNorm"]
