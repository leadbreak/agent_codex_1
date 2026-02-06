from hymba_plus.core.registry import Registry
from hymba_plus.components.attention.standard import StandardAttention

ATTENTION_REGISTRY = Registry("attention")

ATTENTION_REGISTRY.register("standard")(StandardAttention)

__all__ = ["ATTENTION_REGISTRY", "StandardAttention"]
