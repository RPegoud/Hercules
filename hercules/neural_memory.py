import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.func import functional_call, vmap, grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import OrderedDict
import math


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class MemoryNetwork(nn.Module):
    """Deep MLP with residual connections and SiLU activation."""

    def __init__(self, hidden_size: int, depth: int, expansion_factor: int):
        super().__init__()
        dims = [
            hidden_size,
            *((hidden_size * expansion_factor,) * (depth - 1)),
            hidden_size,
        ]
        layer_sizes = zip(dims[:-1], dims[1:])
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(dim_in, dim_out))
                for dim_in, dim_out in layer_sizes
            ]
        )

        self.projections = nn.ParameterList(
            [
                (
                    nn.Parameter(torch.eye(in_dim, out_dim))
                    if in_dim == out_dim
                    else nn.Parameter(torch.randn(in_dim, out_dim))
                )
                for in_dim, out_dim in layer_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, (w, p) in enumerate(zip(self.weights, self.projections)):
            if idx != 0:
                x = F.silu(x)
            residual = x
            x = x @ w + residual @ p
        return x


class NeuralMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_depth: int,
        mlp_expansion_factor: int,
        max_adaptive_lr: float,
    ) -> None:
        super(NeuralMemory, self).__init__()

        self.memory_network = MemoryNetwork(
            hidden_size, mlp_depth, mlp_expansion_factor
        )
        self.param_shapes = OrderedDict(
            {n: p.shape for n, p in self.memory_network.named_parameters()}
        )
        self.param_names = [n for n, _ in self.memory_network.named_parameters()]
        self.register_buffer(
            "momentum_buffer",
            parameters_to_vector(
                [torch.zeros_like(p) for _, p in self.memory_network.named_parameters()]
            ),
        )

        self.d_q = hidden_size
        self.d_k = hidden_size
        self.d_v = hidden_size

        # q, k, v, α, θ, η projections
        self.fused_proj = nn.Linear(
            hidden_size,
            self.d_q + self.d_k + self.d_v + 3,
            bias=False,
        )

        self.max_adaptive_lr = max_adaptive_lr

    @staticmethod
    def _maybe_add_batch(x: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension if the input is 2-dimensional."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _unpack_vector_to_tensors(vec: torch.Tensor, param_shapes: list[torch.Size]):
        """Given a 1D Tensor `vec`, yield tensors matching `param_shapes` in order."""
        tensors = []
        offset = 0
        for shape in param_shapes:
            n = math.prod(shape)
            view = vec[offset : offset + n]
            tensors.append(view.view(shape))
            offset += n
        return tensors

    def _vector_to_ordered_dict(
        self, x: torch.Tensor
    ) -> OrderedDict[str, torch.Tensor]:
        """Converts a 1D Tensor to a dictionary matching `memory_network` parameters."""
        return OrderedDict(
            zip(
                self.param_shapes.keys(),
                self._unpack_vector_to_tensors(x, self.param_shapes.values()),
            )
        )

    def _get_fused_projection(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        fused_proj = self.fused_proj(x)
        q, k, v, alpha, theta, eta = torch.split(
            fused_proj,
            [self.d_q, self.d_k, self.d_v, 1, 1, 1],
            dim=-1,
        )

        q = l2_norm(F.silu(q))
        k = l2_norm(F.silu(k))
        v = F.silu(v)
        alpha = torch.sigmoid(alpha)
        theta = torch.sigmoid(theta) * self.max_adaptive_lr
        eta = torch.sigmoid(eta)

        return q, k, v, alpha, theta, eta

    def get_query_projection(self, x: torch.Tensor) -> torch.Tensor:
        W_q = self.fused_proj.weight[: self.d_q]
        q = F.linear(x, W_q)
        return l2_norm(F.silu(q))

    def _associative_memory_loss(
        self,
        params: TensorDict,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        preds = functional_call(self.memory_network, params, inputs)
        loss = F.mse_loss(preds, targets, reduction="none").mean(dim=-1)
        return loss.sum()

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._maybe_add_batch(x)
            memory_net_params = dict(self.memory_network.named_parameters())
            q = self.get_query_projection(x)
            return functional_call(self.memory_network, memory_net_params, q)

    def reset_memory(self) -> None:
        """Reset the state of the Memory Network and the Momentum buffer."""
        with torch.no_grad():
            for param in self.memory_network.parameters():
                param.zero_()
            self.momentum_buffer.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_add_batch(x)
        B, T, _ = x.shape
        is_generating = T == 1

        # if the LLM is generating, return a retrieved vector without memory update
        if is_generating:
            q = self.get_query_projection(x)
            params = dict(self.memory_network.named_parameters())
            return functional_call(self.memory_network, params, q)

        outputs = []

        params_0 = parameters_to_vector(self.memory_network.parameters())
        momentum_0 = self.momentum_buffer.detach()
        P = params_0.numel()
        current_params = params_0.unsqueeze(0).expand(B, P).contiguous()
        current_momentum = momentum_0.unsqueeze(0).expand(B, P).contiguous()

        q, k, v, alpha, theta, eta = self._get_fused_projection(x)

        def _loss_fn(param_vec, key, val):
            param_dict = self._vector_to_ordered_dict(param_vec)
            return self._associative_memory_loss(param_dict, key, val)

        def _retrieve_fn(param_vec, query):
            param_dict = self._vector_to_ordered_dict(param_vec)
            return functional_call(self.memory_network, param_dict, query)

        grad_fn = grad(_loss_fn)

        for t in range(T):
            grads = vmap(grad_fn, in_dims=(0, 0, 0))(current_params, k[:, t], v[:, t])
            grads = grads.view(B, -1)  # flatten to (B, P)

            alpha_b, theta_b, eta_b = map(lambda x: x[:, t], (alpha, theta, eta))

            next_momentum = current_momentum * eta_b - theta_b * grads
            next_params = current_params * (1 - alpha_b) + next_momentum

            q_t = q[:, t]
            output_t = vmap(_retrieve_fn, in_dims=(0, 0))(next_params, q_t)
            outputs.append(output_t)

            current_params = next_params.detach()
            current_momentum = next_momentum.detach()

        with torch.no_grad():
            updated_params = current_params.mean(0)
            updated_momentum = current_momentum.mean(0)

            vector_to_parameters(updated_params, self.memory_network.parameters())
            self.momentum_buffer.copy_(updated_momentum)

        return torch.stack(outputs, dim=1)
