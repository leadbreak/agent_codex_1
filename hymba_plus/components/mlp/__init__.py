from hymba_plus.core.registry import Registry
from hymba_plus.components.mlp.geglu import GeGLU
from hymba_plus.components.mlp.moe import MoE
from hymba_plus.components.mlp.swiglu import SwiGLU

MLP_REGISTRY = Registry("mlp")

MLP_REGISTRY.register("swiglu")(SwiGLU)
MLP_REGISTRY.register("geglu")(GeGLU)
MLP_REGISTRY.register("moe")(MoE)
MLP_REGISTRY.register("sparse_moe")(MoE)

__all__ = ["MLP_REGISTRY", "SwiGLU", "GeGLU", "MoE"]
