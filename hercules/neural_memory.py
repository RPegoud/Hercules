from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.func import functional_call, vmap

from .layers import AdaptiveWeight, LinearProjection, ResLinear


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
    ) -> None:
        super(NeuralMemory, self).__init__()
        self.added_padding = None

        self.memory_module = ResLinear(hidden_size, mlp_depth, mlp_expansion_factor)

        self.query_projection = LinearProjection(hidden_size, 1)
        self.key_projection = LinearProjection(hidden_size, 1)
        self.value_projection = LinearProjection(hidden_size, 1)

        # α: forget gate
        self.adaptive_forget_projection = AdaptiveWeight(hidden_size, 1, max_weight=1.0)
        # θ: data-dependent learning rate
        self.adaptive_lr_projection = AdaptiveWeight(
            hidden_size, 1, max_weight=max_adaptive_lr
        )
        # η: data-dependent surprise momentum
        self.adaptive_momentum_projection = AdaptiveWeight(hidden_size, 1, max_weight=1)

        self.momentum_states = nn.ParameterDict()
        self._initialize_momentum_buffers()

    @property
    def gate_parameters(self) -> List[nn.Parameter]:
        """
        The forget gate, adaptive learning rate and momentum are trained with ``backward()`` at train time
        The memory module in itself (residual MLP) is trained at inference time using a custom logic that's
        not compatible with ``backward()``
        """
        return [
            p
            for n, p in self.named_parameters()
            if not n.startswith("memory_module.") and p.requires_grad
        ]

    def _initialize_momentum_buffers(self) -> None:
        """Initialize momentum states as buffers for each memory module parameter."""
        for name, param in self.memory_module.named_parameters():
            self.register_buffer(
                f"momentum_{name.replace('.', '_')}", torch.zeros_like(param)
            )

    @staticmethod
    def _add_batch_dim(x: torch.Tensor) -> torch.Tensor:
        """Adds a batch dimension to the input if missing."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def _get_momentum_dict(self, batch_size: int) -> TensorDict:
        """Return a dictionary of momentum states, reshaped for batch processing."""
        momentum_dict = {}
        for name, _ in self.memory_module.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            momentum = getattr(self, buffer_name)
            momentum = momentum.expand(batch_size, *momentum.shape)
            momentum_dict[name] = momentum
        return TensorDict(momentum_dict)

    def _associative_memory_loss(
        self,
        params: TensorDict,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = functional_call(self.memory_module, params, k_t)
        return F.mse_loss(preds, v_t)

    def _update_step(
        self,
        x_t: torch.Tensor,
        params_t: dict[str, torch.Tensor],
        momentum_t: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        v_t = self.value_projection(x_t)
        k_t = self.key_projection(x_t)
        k_t = l2_norm(k_t)

        lr_t = self.adaptive_lr_projection(x_t)
        forget_t = self.adaptive_forget_projection(x_t)
        momentum_t = self.adaptive_momentum_projection(x_t)

        grads = torch.func.grad(self._associative_memory_loss)(params_t, k_t, v_t)

        new_params = {}
        new_momentum = {}

        for name, param in params_t.items():
            grad = grads[name]
            updated_momentum = momentum_t[name].mul(momentum_t).sub(lr_t * grad)
            updated_param = param * (1.0 - forget_t) + updated_momentum

            new_params[name] = updated_param
            new_momentum[name] = updated_momentum

        return new_params, new_momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._add_batch_dim(x)
        is_generating = x.size(1) == 1
        batch_size = x.size(0)

        q = self.query_projection(x, is_generating)
        q = l2_norm(q)

        initial_params = {
            name: p.detach() for name, p in self.memory_module.named_parameters()
        }

        if not is_generating:
            batched_params = {
                name: p.detach().expand(batch_size, *p.shape)
                for name, p in initial_params.items()
            }
            batched_momentum = self._get_momentum_dict(batch_size)
            batch_update_fn = vmap(self._update_step, in_dims=(0, 0, 0))

            print(x.shape)

            for t in range(x.size(1)):
                batched_params, batched_momentum = batch_update_fn(
                    x[:, t, :, :], batched_params, batched_momentum
                )

            updated_params_dict = {
                name: p.mean(dim=0) for name, p in batched_params.items()
            }

            with torch.no_grad():
                final_momentum = {
                    name: m.mean(dim=0) for name, m in batched_momentum.items()
                }
                for name, _ in self.memory_module.named_parameters():
                    buffer_name = f"momentum_{name.replace('.', '_')}"
                    getattr(self, buffer_name).copy_(final_momentum[name])

        else:
            updated_params_dict = initial_params

        retrieved = functional_call(self.memory_module, updated_params_dict, q)

        return retrieved
