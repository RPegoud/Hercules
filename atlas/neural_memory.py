from .layers import ResLinear, LinearProjection, AdaptiveLR, SlidingWindowAttention
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from tensordict import TensorDict
import torch.nn.functional as F


class NeuralMemory(nn.Module):

    def __init__(
        self,
        layer_size: int,
        input_dim: int,
        n_layers: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        max_adaptive_lr: float,
        meta_memory_dim: int,
        num_attention_heads: int,
        attention_window_size: int,
        n_chunks: int,
    ) -> None:
        # TODO: implement associative scan kernel
        # TODO: add learned gating

        # DONE: add SWA
        # DONE: add persistent memory
        # DONE: add adaptive learning rate
        # DONE: add chunking
        # DONE: pad inputs and replace n_chunks by chunk_size
        # DONE: vectorize the loss
        # DONE: add momentum, past surprises

        super(NeuralMemory, self).__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_adaptive_lr = max_adaptive_lr
        self.meta_memory_dim = meta_memory_dim
        self.num_attention_heads = num_attention_heads
        self.attention_window_size = attention_window_size
        self.n_chunks = n_chunks
        self.added_padding = None

        self.lmm = ResLinear(input_dim, n_layers)
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

        self.register_buffer(
            "surprises",
            torch.zeros((self.n_layers, self.n_chunks, self.input_dim, self.input_dim)),
        )

    def _pad_to_chunk_size(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        print(seq_len, self.n_chunks)
        if not seq_len % self.n_chunks == 0:
            pad = (seq_len // self.n_chunks) * self.n_chunks + self.n_chunks - seq_len
            self.added_padding = -pad
            x = F.pad(x, (0, 0, 0, pad))
        return x

    def _inject_meta_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        meta_memory = self.meta_memory.expand(batch_size, -1, -1)
        meta_x = torch.concat([meta_memory, x], dim=1)
        return meta_x

    def _associative_memory_loss(self, params, inputs, targets, weights) -> float:
        preds = functional_call(self.lmm, params, inputs)
        loss = torch.pow(preds - targets, 2).mean(dim=-1)
        weighted_loss = loss * weights.squeeze()
        return weighted_loss.sum(), loss

    @torch.no_grad
    def _compute_surprises(
        self, theta_t: torch.Tensor, per_sample_grads: TensorDict
    ) -> None:
        eta = torch.full_like(theta_t, self.momentum)  # TODO: add learned gate
        eta_prod = torch.cumprod(eta, dim=1)
        for layer_idx in range(self.n_layers):
            layer_id = f"weights.{layer_idx}"
            if layer_id in per_sample_grads:
                u_t = per_sample_grads[layer_id]  # (n_chunks, input_dim, input_dim)
                # Vectorized S_t = -sum(theta_t * u_t * eta_prod)
                theta_u = (
                    theta_t.unsqueeze(-1).unsqueeze(-1) * u_t
                )  # (batch_size, n_chunks, input_dim, input_dim)
                weighted_theta_u = theta_u * eta_prod.unsqueeze(-1).unsqueeze(-1)
                S_t = -torch.cumsum(
                    weighted_theta_u.mean(dim=0), dim=0
                )  # (n_chunks, input_dim, input_dim)
                self.surprises[layer_idx] = S_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        params = self.lmm.named_parameters()

        x = self._inject_meta_memory(x)
        x = self._pad_to_chunk_size(x)

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        adaptive_lr = self.adaptive_lr_projection(x)

        grad_fn = grad(self._associative_memory_loss, has_aux=True)
        per_chunk_grad_fn = vmap(grad_fn, in_dims=(None, 2, 2, 2))
        per_sample_grads, _ = per_chunk_grad_fn(dict(params), keys, values, adaptive_lr)
        per_sample_grads = TensorDict(per_sample_grads)

        theta_t = adaptive_lr.mean(dim=1).squeeze(-1)
        self._compute_surprises(theta_t, per_sample_grads)

        for idx, (name, param) in enumerate(self.lmm.named_parameters()):
            if per_sample_grads.get(name) is not None:
                param.grad = (per_sample_grads.get(name) + self.surprises[idx]).mean(0)
        self.optimizer.step()

        retrieved = self.lmm(queries.squeeze())
        retrieved = self.swa(retrieved)

        # discard meta-memory and padding
        output = retrieved[:, self.meta_memory_dim : self.added_padding, :]

        return output
