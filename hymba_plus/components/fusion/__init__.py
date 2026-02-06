from hymba_plus.core.registry import Registry
from hymba_plus.components.fusion.adaptive import AdaptiveFusion
from hymba_plus.components.fusion.average import AverageFusion
from hymba_plus.components.fusion.gated import GatedFusion
from hymba_plus.components.fusion.learnable import LearnableFusion

FUSION_REGISTRY = Registry("fusion")

FUSION_REGISTRY.register("average")(AverageFusion)
FUSION_REGISTRY.register("learnable")(LearnableFusion)
FUSION_REGISTRY.register("gated")(GatedFusion)
FUSION_REGISTRY.register("adaptive")(AdaptiveFusion)

__all__ = ["FUSION_REGISTRY", "AverageFusion", "LearnableFusion", "GatedFusion", "AdaptiveFusion"]
