# flake8: noqa
from .layers import ResLinear, LinearProjection, AdaptiveWeight, SlidingWindowAttention
from .neural_memory import NeuralMemory
from .llama_memory_layers import LlamaMemoryAsLayer
from .utils import l2_norm, flatten_and_expand
