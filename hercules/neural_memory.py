from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.func import functional_call, vmap
from colorama import Style, Fore
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

        self.memory_network = ResLinear(hidden_size, mlp_depth, mlp_expansion_factor)

        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.adaptive_forget_projection = nn.Linear(hidden_size, 1)
        self.adaptive_lr_projection = nn.Linear(hidden_size, 1)
        self.adaptive_momentum_projection = nn.Linear(hidden_size, 1)

        self.max_adaptive_lr = max_adaptive_lr
        self._initialize_momentum_buffers()

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

    @staticmethod
    def _maybe_add_batch(x: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension if the input is 2-dimensional."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def _associative_memory_loss(
        self,
        params: TensorDict,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = functional_call(self.memory_network, params, inputs)
        loss = F.mse_loss(preds, targets, reduction="none").mean(dim=-1)
        total_loss = loss.sum()
        return total_loss, loss

    @torch.no_grad()
    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_add_batch(x)
        memory_net_params = dict(self.memory_network.named_parameters())
        q = l2_norm(self.query_projection(x))
        return functional_call(self.memory_network, memory_net_params, q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_add_batch(x)
        batch_size, seq_len, _ = x.shape
        is_generating = seq_len == 1

        if is_generating:
            q = l2_norm(self.query_projection(x))
            current_params = dict(self.memory_network.named_parameters())
            return functional_call(self.memory_network, current_params, q)

        outputs = []

        current_params = {
            name: param.expand(batch_size, *param.shape)
            for name, param in self.memory_network.named_parameters()
        }
        current_momentum = self._get_momentum_dict(batch_size)

        q = l2_norm(F.silu(self.query_projection(x)))
        k = l2_norm(F.silu(self.key_projection(x)))
        v = F.silu(self.value_projection(x))

        theta_t = torch.sigmoid(self.adaptive_lr_projection(x)) * self.max_adaptive_lr
        alpha_t = torch.sigmoid(self.adaptive_forget_projection(x))
        eta_t = torch.sigmoid(self.adaptive_momentum_projection(x))

        for t in range(seq_len):
            updated_params_functional = {}
            next_momentum_functional = {}

            grad_fn = torch.func.grad(self._associative_memory_loss, has_aux=True)
            per_batch_grads, _ = vmap(grad_fn, in_dims=(0, 0, 0))(
                current_params, k[:, t], v[:, t]
            )

            for name, param in self.memory_network.named_parameters():
                grad = per_batch_grads[name]
                momentum = current_momentum[name]

                alpha = alpha_t[:, t]
                theta = theta_t[:, t]
                eta = eta_t[:, t]

                # Expand gates for broadcasting with parameter shapes
                alpha_b = flatten_and_expand(alpha, param.dim())
                theta_b = flatten_and_expand(theta, param.dim())
                eta_b = flatten_and_expand(eta, param.dim())

                next_momentum = momentum.mul(eta_b) - theta_b * grad
                next_param = param * (1.0 - alpha_b) + next_momentum

                updated_params_functional[name] = next_param
                next_momentum_functional[name] = next_momentum

            q_t = q[:, t].unsqueeze(1)  # Add sequence dim back
            output_t = functional_call(
                self.memory_network, updated_params_functional, q_t
            )
            outputs.append(output_t)

            # truncate the gradient history
            current_params = {
                k: v.detach() for k, v in updated_params_functional.items()
            }
            current_momentum = {
                k: v.detach() for k, v in next_momentum_functional.items()
            }

        with torch.no_grad():
            for name, param in self.memory_network.named_parameters():
                # average the final state across the batch dimension before saving
                param.copy_(current_params[name].mean(0))
                buffer_name = f"momentum_{name.replace('.', '_')}"
                getattr(self, buffer_name).copy_(current_momentum[name].mean(0))

        return torch.cat(outputs, dim=1)
