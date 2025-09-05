from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from .layers import ResLinear
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import OrderedDict
from einops import rearrange


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

        self.memory_network = ResLinear(hidden_size, mlp_depth, mlp_expansion_factor)
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

        self.n_chunks = n_chunks
        self.max_adaptive_lr = max_adaptive_lr

    @staticmethod
    def _maybe_add_batch(x: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension if the input is 2-dimensional."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

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

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._maybe_add_batch(x)
            memory_net_params = dict(self.memory_network.named_parameters())
            q = self.get_query_projection(x)
            return functional_call(self.memory_network, memory_net_params, q)

    def reset_memory(self) -> None:
        with torch.no_grad():
            for param in self.memory_network.parameters():
                param.zero_()

    def scan_update_chunk(
        self,
        W0_dict: dict[str, torch.Tensor],
        M0_dict: dict[str, torch.Tensor],
        grads_dict: dict[str, torch.Tensor],
        alpha: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        W_all_dict = {}
        M_all_dict = {}

        def _reshape_to_chunk_dim(x: torch.Tensor) -> torch.Tensor:
            return x.view(T_chunk, *((1,) * len(P_dims)))

        for name in W0_dict.keys():
            W0, M0, grads = W0_dict[name], M0_dict[name], grads_dict[name]
            T_chunk, P_dims = grads.shape[0], grads.shape[1:]
            alpha_r, theta_r, eta_r = map(_reshape_to_chunk_dim, (alpha, theta, eta))
            M0_b = M0.unsqueeze(0)

            # momentum update
            eta_prods = torch.cumprod(
                torch.cat([torch.ones_like(eta_r[:1]), eta_r], dim=0), dim=0
            )
            scaled_inputs_M = (-theta_r * grads) / eta_prods[1:]
            sum_M = torch.cumsum(
                torch.cat([M0_b / eta_prods[:1], scaled_inputs_M], dim=0), dim=0
            )
            M_all = sum_M[1:] * eta_prods[1:]
            M_all_dict[name] = M_all

            # memory update
            W0_b = W0.unsqueeze(0)
            decay = 1.0 - alpha_r
            decay_prods = torch.cumprod(
                torch.cat([torch.ones_like(decay[:1]), decay], dim=0), dim=0
            )
            scaled_inputs_W = M_all / decay_prods[1:]
            sum_W = torch.cumsum(
                torch.cat([W0_b / decay_prods[:1], scaled_inputs_W], dim=0), dim=0
            )
            W_all = sum_W[1:] * decay_prods[1:]
            W_all_dict[name] = W_all
        return W_all_dict, M_all_dict

    # @torch.compile
    def process_sequence_in_chunks(
        self,
        initial_W: dict[str, torch.Tensor],
        initial_M: dict[str, torch.Tensor],
        all_grads: dict[str, torch.Tensor],
        all_alpha: torch.Tensor,
        all_theta: torch.Tensor,
        all_eta: torch.Tensor,
        update_fn: callable,
    ) -> dict[str, torch.Tensor]:
        """
        Processes the entire sequence by iterating sequentially through chunks.
        Operations within each chunk are parallelized over the batch and time dimensions.

        Args:
            initial_W (dict): Dict of initial model parameters, shape (B, *param_shape).
            initial_M (dict): Dict of initial momentum, shape (B, *param_shape).
            all_grads (dict): Dict of grads for all chunks, shape (B, C, T_chunk, *param_shape).
            all_alpha (torch.Tensor): Alpha values for all chunks, shape (B, C, T_chunk, 1).
            all_theta (torch.Tensor): Theta values for all chunks, shape (B, C, T_chunk, 1).
            all_eta (torch.Tensor): Eta values for all chunks, shape (B, C, T_chunk, 1).

        Returns:
            W_state (dict): The final parameter state after the last chunk,
            averaged over the batch dimension, shape (*param_shape).
        """
        C = all_alpha.shape[1]

        W_state = initial_W
        M_state = initial_M

        for c in range(C):
            grads_c = {name: all_grads[name][:, c, ...] for name in all_grads}
            alpha_c = all_alpha[:, c, ...]
            theta_c = all_theta[:, c, ...]
            eta_c = all_eta[:, c, ...]

            W_all_in_chunk, M_all_in_chunk = update_fn(
                W_state, M_state, grads_c, alpha_c, theta_c, eta_c
            )

            W_state = {name: W_all_in_chunk[name][:, -1] for name in W_all_in_chunk}
            M_state = {name: M_all_in_chunk[name][:, -1] for name in M_all_in_chunk}

        updated_params = {
            name: torch.mean(tensor, dim=0) for name, tensor in W_state.items()
        }

        return updated_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def _chunk_and_reshape(x: torch.Tensor) -> torch.Tensor:
            x = torch.chunk(x, C, dim=1)
            x = rearrange(torch.stack(x), "c b t f -> b c t f")
            return x

        def _associative_memory_loss(params_dict, k, v):
            preds = functional_call(self.memory_network, params_dict, k)
            return F.mse_loss(preds, v)

        x = self._maybe_add_batch(x)
        B, T, _ = x.shape
        C = self.n_chunks

        is_generating = T == 1
        params = dict(self.memory_network.named_parameters())

        # if the LLM is generating, return a retrieved vector without memory update
        if is_generating:
            q = self.get_query_projection(x)
            return functional_call(self.memory_network, params, q)

        # --- forward pass with update ---

        batched_params = {
            name: p.unsqueeze(0).unsqueeze(0).expand(B, C, *p.shape)
            for name, p in params.items()
        }  # batched params
        batched_momentums = {
            name: torch.zeros_like(p).unsqueeze(0).unsqueeze(0).expand(B, C, *p.shape)
            for name, p in params.items()
        }  # batched momentums

        q, k, v, alpha, theta, eta = self._get_fused_projection(x)
        x, q, k, v, alpha, theta, eta = map(
            _chunk_and_reshape, (x, q, k, v, alpha, theta, eta)
        )

        grad_fn = grad(_associative_memory_loss, argnums=0)
        params_in_dims = {name: 0 for name in batched_params}

        grad_over_time = vmap(
            grad_fn, in_dims=(None, 0, 0)
        )  # broadcast params for each step
        grad_over_chunks = vmap(grad_over_time, in_dims=(params_in_dims, 0, 0))
        grad_over_batch = vmap(grad_over_chunks, in_dims=(params_in_dims, 0, 0))

        grads = grad_over_batch(batched_params, k, v)

        params_in_dims = {name: 0 for name in batched_params}
        grads_in_dims = {name: 0 for name in grads}

        batched_scan_for_one_chunk = vmap(
            self.scan_update_chunk,
            in_dims=(params_in_dims, params_in_dims, grads_in_dims, 0, 0, 0),
        )
        initial_W = {name: p[:, 0, ...] for name, p in batched_params.items()}
        initial_M = {name: m[:, 0, ...] for name, m in batched_momentums.items()}

        updated_params = self.process_sequence_in_chunks(
            initial_W,
            initial_M,
            grads,
            alpha,
            theta,
            eta,
            update_fn=batched_scan_for_one_chunk,
        )

        with torch.no_grad():
            vector_to_parameters(updated_params, self.memory_network.parameters())

        return self.retrieve(x)
