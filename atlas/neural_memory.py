from layers import ResLinear, LinearProjection, AdaptiveLR, SlidingWindowAttention
import torch
import torch.nn as nn
from torch.func import functional_call, grad
from tensordict import TensorDict


class NeuralMemory(nn.Module):
    def __init__(
        self,
        layer_size: int,
        input_dim: int,
        n_hidden_layers: int,
        learning_rate: float,
        weight_decay: float,
        max_adaptive_lr: float,
        meta_memory_dim: int,
        num_attention_heads: int,
        attention_window_size: int,
        device: str,  # TODO: pass variables to device before/after mha if needed
    ) -> None:
        # TODO: add chunking
        # TODO: add multihead processing
        # TODO: add momentum & past surprises

        # DONE: add SWA
        # DONE: add persistent memory
        # DONE: add adaptive learning rate

        super(NeuralMemory, self).__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_adaptive_lr = max_adaptive_lr
        self.meta_memory_dim = meta_memory_dim
        self.num_attention_heads = num_attention_heads
        self.attention_window_size = attention_window_size
        self.device = device

        self.lmm = ResLinear(input_dim, n_hidden_layers)
        self.key_projection = LinearProjection(input_dim, layer_size)
        self.query_projection = LinearProjection(input_dim, layer_size)
        self.value_projection = LinearProjection(input_dim, layer_size)
        self.adaptive_lr_projection = AdaptiveLR(
            input_dim, 1, self.max_adaptive_lr
        )  # TODO: modify out_dim when adding chuncking
        self.meta_memory = nn.Parameter(torch.randn(meta_memory_dim, input_dim))

        self.optimizer = torch.optim.AdamW(
            self.lmm.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )
        self.swa = SlidingWindowAttention(
            input_dim, num_attention_heads, attention_window_size, device
        )


    def _associative_loss(self, params, inputs, targets, weights) -> float:
        preds = functional_call(self.lmm, params, inputs)
        loss = torch.pow(preds - targets, 2).mean(dim=-1)
        weighted_loss = loss * weights.squeeze()
        return weighted_loss.sum(), loss

    def _inject_meta_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        meta_memory = self.meta_memory.expand(batch_size, -1, -1)
        meta_x = torch.concat([meta_memory, x], dim=1)
        return meta_x
    
    def _discard_meta_memory(self, x: torch.Tensor):
        return x[:, self.meta_memory_dim: , :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        params = self.lmm.named_parameters()

        x = self._inject_meta_memory(x)

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        adaptive_lr = self.adaptive_lr_projection(x)

        grad_fn = grad(self._associative_loss, has_aux=True)
        grads, _ = grad_fn(dict(params), keys, values, adaptive_lr)
        for name, param in self.lmm.named_parameters():
            if grads[name] is not None:
                param.grad = grads[name]
        self.optimizer.step()

        surprises = TensorDict(grads).mul(-1)
        retrieved = self.lmm(queries)
        retrieved = self.swa(retrieved)

        output = self._discard_meta_memory(retrieved)

        return output
