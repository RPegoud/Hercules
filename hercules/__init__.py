# flake8: noqa
from hercules.layers import (
    ResLinear,
    LinearProjection,
    AdaptiveWeight,
    SlidingWindowAttention,
)
from hercules.neural_memory import NeuralMemory
from hercules.llama_memory_layers import LlamaMemoryAsLayer, inject_memory_module
from hercules.utils import l2_norm, flatten_and_expand
from hercules.logger import log_config, log_memory_model
from hercules.memory_llama import MemoryLlama
