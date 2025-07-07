import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.func import functional_call, vmap

from .layers import AdaptiveWeight, LinearProjection, ResLinear, SlidingWindowAttention


def flatten_and_expand(x: torch.Tensor, n: int):
    """Flattens a tensor and adds `n` trailing dimensions."""
    return x.view(-1, *([1] * n))


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class NeuralMemory(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        max_adaptive_lr: float,
        meta_memory_dim: int,
        num_attention_heads: int,
        attention_window_size: int,
        conv_kernel_size: int,
        n_chunks: int,
    ) -> None:
        super(NeuralMemory, self).__init__()
        self.meta_memory_dim = meta_memory_dim
        self.n_chunks = n_chunks
        self.added_padding = None

        self.memory_module = ResLinear(
            input_dim, hidden_dim, output_dim, n_hidden_layers
        )
        self.key_projection = LinearProjection(
            input_dim, output_dim, n_chunks, conv_kernel_size
        )
        self.query_projection = LinearProjection(
            input_dim, output_dim, 1, conv_kernel_size
        )
        self.value_projection = LinearProjection(
            input_dim, output_dim, n_chunks, conv_kernel_size
        )

        # α: forget gate
        self.adaptive_forget_projection = AdaptiveWeight(
            input_dim, 1, n_chunks, max_weight=1
        )
        # θ: data-dependent learning rate
        self.adaptive_lr_projection = AdaptiveWeight(
            input_dim, 1, n_chunks, max_weight=max_adaptive_lr
        )
        # η: data-dependent surprise momentum
        self.adaptive_momentum_projection = AdaptiveWeight(
            input_dim, 1, n_chunks, max_weight=1
        )

        self.meta_memory = nn.Parameter(torch.randn(meta_memory_dim, input_dim))
        nn.init.xavier_uniform_(self.meta_memory)

        self.swa = SlidingWindowAttention(
            input_dim, num_attention_heads, attention_window_size
        )

        self.momentum_states = nn.ParameterDict()
        self.register_buffer("last_associative_loss", torch.tensor(0.0))
        self._initialize_momentum_buffers()

    def _initialize_momentum_buffers(self):
        """Initialize momentum states as buffers for each LMM parameter."""
        for name, param in self.memory_module.named_parameters():
            self.register_buffer(
                f"momentum_{name.replace('.', '_')}", torch.zeros_like(param)
            )

    def _get_momentum_dict(self, batch_size, device):
        """Return a dictionary of momentum states, reshaped for batch processing."""
        momentum_dict = {}
        for name, param in self.memory_module.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            momentum = getattr(self, buffer_name)
            # Expand to match batch size
            momentum = momentum.expand(batch_size, *momentum.shape).to(device)
            momentum_dict[name] = momentum
        return TensorDict(momentum_dict)

    def _pad_to_chunk_size(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
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

    def _associative_memory_loss(
        self,
        params: TensorDict,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        adaptive_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = functional_call(self.memory_module, params, inputs)
        loss = F.mse_loss(preds, targets, reduction="none").mean(dim=-1)
        weighted_loss = loss * adaptive_lr.squeeze()
        total_loss = weighted_loss.sum()
        return total_loss, loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.memory_module.named_parameters()

        x = self._inject_meta_memory(x)
        x = self._pad_to_chunk_size(x)

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        q, k = l2_norm(q), l2_norm(k)
        momentum_dict = self._get_momentum_dict(x.size(0), x.device)

        adaptive_lr = self.adaptive_lr_projection(x)
        adaptive_forget = self.adaptive_forget_projection(x)
        adaptive_momentum = self.adaptive_momentum_projection(x)

        grad_fn = torch.func.grad(self._associative_memory_loss, has_aux=True)
        per_chunk_grad_fn = vmap(grad_fn, in_dims=(None, 1, 1, 1))
        per_chunk_grads, per_chunk_loss = per_chunk_grad_fn(
            dict(params), k, v, adaptive_lr
        )
        per_chunk_grads = TensorDict(per_chunk_grads)
        self.last_associative_loss = per_chunk_loss.detach().mean()
        

        with torch.no_grad():  # Disable grad for test-time updates
            for name, param in self.memory_module.named_parameters():
                if per_chunk_grads.get(name) is not None:
                    grad = per_chunk_grads[name].mean(dim=0)  # average per chunk
                    momentum = momentum_dict[name].clone()

                    alpha_t = adaptive_forget.mean(dim=[1, 2, 3])
                    theta_t = adaptive_lr.mean(dim=[1, 2, 3])
                    eta_t = adaptive_momentum.mean(dim=[1, 2, 3])

                    alpha_t = flatten_and_expand(alpha_t, grad.dim())
                    eta_t = flatten_and_expand(eta_t, grad.dim())
                    theta_t = flatten_and_expand(theta_t, grad.dim())

                    # Momentum update: S_t = η_t * S_{t-1} - θ_t * grad
                    momentum.mul_(eta_t)
                    momentum.sub_(theta_t * grad)

                    # Parameter update: M_t = (1 - α_t) * M_{t-1} + S_t
                    new_param = param * (1.0 - alpha_t) + momentum
                    param.data.copy_(new_param.mean(0))

                    buffer_name = f"momentum_{name.replace('.', '_')}"
                    getattr(self, buffer_name).copy_(momentum.mean(dim=0))

        retrieved = self.memory_module(q.squeeze())
        retrieved = F.silu(retrieved)
        retrieved = self.swa(retrieved)

        # discard meta-memory and padding
        output = retrieved[:, self.meta_memory_dim : self.added_padding, :]

        return output
