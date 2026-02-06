from hymba_plus.core.registry import Registry
from hymba_plus.components.ssm.simple_ssm import SimpleSSM

SSM_REGISTRY = Registry("ssm")

SSM_REGISTRY.register("mamba")(SimpleSSM)
SSM_REGISTRY.register("mamba2")(SimpleSSM)

__all__ = ["SSM_REGISTRY", "SimpleSSM"]
