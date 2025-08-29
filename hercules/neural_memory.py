from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.func import functional_call, vmap

from .layers import AdaptiveWeight, LinearProjection, SlidingWindowAttention, ResLinear


def flatten_and_expand(x: torch.Tensor, n: int):
    """Flattens a tensor and adds `n` trailing dimensions."""
    return x.view(-1, *([1] * n))


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class NeuralMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_depth: int,
        mlp_expansion_factor: int,
        max_adaptive_lr: float,
        n_chunks: int,
    ) -> None:
        super(NeuralMemory, self).__init__()
        self.n_chunks = n_chunks
        self.added_padding = None

        self.memory_network = ResLinear(hidden_size, mlp_depth, mlp_expansion_factor)

        self.query_projection = LinearProjection(hidden_size, hidden_size, 1)
        self.key_projection = LinearProjection(hidden_size, hidden_size, n_chunks)
        self.value_projection = LinearProjection(hidden_size, hidden_size, n_chunks)

        # α: forget gate
        self.adaptive_forget_projection = AdaptiveWeight(
            hidden_size, 1, n_chunks, max_weight=1
        )
        # θ: data-dependent learning rate
        self.adaptive_lr_projection = AdaptiveWeight(
            hidden_size, 1, n_chunks, max_weight=max_adaptive_lr
        )
        # η: data-dependent surprise momentum
        self.adaptive_momentum_projection = AdaptiveWeight(
            hidden_size, 1, n_chunks, max_weight=1
        )

        self.momentum_states = nn.ParameterDict()
        self._initialize_momentum_buffers()

    @property
    def gate_parameters(self) -> List[nn.Parameter]:
        """
        The forget gate, adaptive learning rate and momentum are trained with ``backward()`` at train time
        The memory network (residual MLP) is trained at inference time using a custom logic that's
        not compatible with ``backward()``
        """
        return [
            p
            for n, p in self.named_parameters()
            if not n.startswith("memory_network.") and p.requires_grad
        ]

    def _initialize_momentum_buffers(self) -> None:
        """Initialize momentum states as buffers for each memory module parameter."""
        for name, param in self.memory_network.named_parameters():
            self.register_buffer(
                f"momentum_{name.replace('.', '_')}", torch.zeros_like(param)
            )

    def _get_momentum_dict(self, batch_size) -> TensorDict:
        """Return a dictionary of momentum states, reshaped for batch processing."""
        momentum_dict = {}
        for name, _ in self.memory_network.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            momentum = getattr(self, buffer_name)
            # Expand to match batch size
            momentum = momentum.expand(batch_size, *momentum.shape)
            momentum_dict[name] = momentum
        return TensorDict(momentum_dict)

    def _pad_to_chunk_size(self, x: torch.Tensor) -> torch.Tensor:
        """Adds necessary right padding for the sequence to be split in chunks."""
        seq_len = x.size(1)
        if not seq_len % self.n_chunks == 0:
            pad = (seq_len // self.n_chunks) * self.n_chunks + self.n_chunks - seq_len
            self.added_padding = -pad
            x = F.pad(x, (0, 0, 0, pad))
        return x

    @staticmethod
    def _add_batch_dim(x: torch.Tensor) -> torch.Tensor:
        """Adds a batch dimension to the input if needed"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def _associative_memory_loss(
        self,
        params: TensorDict,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        adaptive_lr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = functional_call(self.memory_network, params, inputs)
        loss = F.mse_loss(preds, targets, reduction="none").mean(dim=-1)
        weighted_loss = loss * adaptive_lr.squeeze()
        total_loss = weighted_loss.sum()
        return total_loss, loss

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            params_dict = dict(self.memory_network.named_parameters())
            x = self._add_batch_dim(x)
            q = self.query_projection(x, is_generating=False).squeeze()
            q = l2_norm(q)
            retrieved = functional_call(self.memory_network, params_dict, q)

        return retrieved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._add_batch_dim(x)

        is_generating = x.shape[1] == 1

        params_dict = dict(self.memory_network.named_parameters())

        q = self.query_projection(x, is_generating)
        q = l2_norm(q)

        # when the llm is generating, avoid the k,v computation and the update
        if not is_generating:
            x = self._pad_to_chunk_size(x)
            q = q.squeeze(1)
            v = self.value_projection(x)
            k = self.key_projection(x)
            k = l2_norm(k)

            momentum_dict = self._get_momentum_dict(x.size(0))

            adaptive_lr = self.adaptive_lr_projection(x)
            adaptive_forget = self.adaptive_forget_projection(x)
            adaptive_momentum = self.adaptive_momentum_projection(x)

            grad_fn = torch.func.grad(self._associative_memory_loss, has_aux=True)
            per_chunk_grad_fn = vmap(grad_fn, in_dims=(None, 1, 1, 1))
            per_chunk_grads, per_chunk_loss = per_chunk_grad_fn(
                params_dict, k, v, adaptive_lr
            )
            per_chunk_grads = TensorDict(per_chunk_grads)

            temp_updated_params = {}
            for name, param in self.memory_network.named_parameters():
                grad = per_chunk_grads[name].mean(dim=0)
                momentum = momentum_dict[name]

                alpha_t = adaptive_forget.mean(dim=[1, 2, 3])
                theta_t = adaptive_lr.mean(dim=[1, 2, 3])
                eta_t = adaptive_momentum.mean(dim=[1, 2, 3])

                alpha_t = flatten_and_expand(alpha_t, grad.dim())
                eta_t = flatten_and_expand(eta_t, grad.dim())
                theta_t = flatten_and_expand(theta_t, grad.dim())

                momentum = momentum.mul(eta_t).sub(theta_t * grad)
                new_param = param * (1.0 - alpha_t) + momentum
                temp_updated_params[name] = new_param.mean(0)

                with torch.no_grad():
                    buffer_name = f"momentum_{name.replace('.', '_')}"
                    getattr(self, buffer_name).copy_(momentum.mean(dim=0))

            updated_params_dict = temp_updated_params

        else:
            updated_params_dict = dict(self.memory_network.named_parameters())

        retrieved = functional_call(self.memory_network, updated_params_dict, q)

        return retrieved
