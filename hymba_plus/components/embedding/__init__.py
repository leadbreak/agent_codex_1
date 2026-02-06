from hymba_plus.core.registry import Registry
from hymba_plus.components.embedding.token import TokenEmbedding
from hymba_plus.components.embedding.rope import RotaryEmbedding

EMBEDDING_REGISTRY = Registry("embedding")

EMBEDDING_REGISTRY.register("token")(TokenEmbedding)

__all__ = ["EMBEDDING_REGISTRY", "TokenEmbedding", "RotaryEmbedding"]
