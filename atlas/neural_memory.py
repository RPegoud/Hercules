from .layers import ResLinear, LinearProjection, AdaptiveLR, SlidingWindowAttention
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
        n_chunks: int,
    ) -> None:
        # TODO: add chunking
        # TODO: vectorize the loss
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
        self.n_chunks = n_chunks

        self.lmm = ResLinear(input_dim, n_hidden_layers)
        self.key_projection = LinearProjection(input_dim, layer_size, n_chunks)
        self.query_projection = LinearProjection(input_dim, layer_size, 1)
        self.value_projection = LinearProjection(input_dim, layer_size, n_chunks)
        self.adaptive_lr_projection = AdaptiveLR(
            input_dim, 1, n_chunks, max_adaptive_lr
        )
        self.meta_memory = nn.Parameter(torch.randn(meta_memory_dim, input_dim))

        self.optimizer = torch.optim.AdamW(
            self.lmm.parameters(), learning_rate, weight_decay=weight_decay
        )
        self.swa = SlidingWindowAttention(
            input_dim, num_attention_heads, attention_window_size
        )

    def _associative_loss(self, params, inputs, targets, weights) -> float:
        preds = functional_call(self.lmm, params, inputs)
        loss = torch.pow(preds - targets, 2).mean(dim=-1)
        print(inputs.shape, weights.shape, targets.shape, loss.shape)
        weighted_loss = loss * weights.squeeze()
        print(weighted_loss.shape)
        return weighted_loss.sum(), loss

    def _inject_meta_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        meta_memory = self.meta_memory.expand(batch_size, -1, -1)
        meta_x = torch.concat([meta_memory, x], dim=1)
        return meta_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        params = self.lmm.named_parameters()

        x = self._inject_meta_memory(x)

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        adaptive_lr = self.adaptive_lr_projection(x)

        grad_fn = grad(self._associative_loss, has_aux=True)
        per_chunk_grad_fn = vmap(grad_fn, in_dims=(None, 2, 2, 2))
        grads, _ = per_chunk_grad_fn(dict(params), keys, values, adaptive_lr)
        grads = TensorDict(grads)
        grads = grads.apply(lambda g: g.mean(0) if g.ndim == 3 else g)
        surprises = grads.mul(-1)  # TODO: store surprises

        for name, param in self.lmm.named_parameters():
            if grads.get(name) is not None:
                param.grad = grads.get(name)
        self.optimizer.step()

        retrieved = self.lmm(queries)
        retrieved = self.swa(retrieved)

        output = retrieved[:, self.meta_memory_dim :, :]  # discard meta-memory

        return output
