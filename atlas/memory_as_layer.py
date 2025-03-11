from layers import ResLinear, LinearProjection, AdaptiveLR
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
    ) -> None:
        # TODO: add chunking
        # TODO: add multihead processing
        # TODO: add adaptive learning rate, momentum
        # TODO: add persistent memory
        super(NeuralMemory, self).__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_adaptive_lr = max_adaptive_lr

        self.lmm = ResLinear(input_dim, n_hidden_layers)
        self.key_projection = LinearProjection(input_dim, layer_size)
        self.query_projection = LinearProjection(input_dim, layer_size)
        self.value_projection = LinearProjection(input_dim, layer_size)
        self.adaptive_lr_projection = AdaptiveLR(
            input_dim, 1, self.max_adaptive_lr
        )  # TODO: modify out dim when adding chuncking

        self.optimizer = torch.optim.AdamW(
            self.lmm.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )

    def _associative_loss(self, params, inputs, targets, weights) -> float:
        pred = functional_call(self.lmm, params, inputs)
        loss = torch.pow(pred - targets, 2).mean(dim=-1)
        weighted_loss = loss * weights
        return weighted_loss.sum(), loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()

        params = self.lmm.named_parameters()
        keys = self.key_projection(x)
        queries = self.query_projection(x)
        values = self.value_projection(x)
        adaptive_lr = self.adaptive_lr_projection(x).view(-1)

        keys_flat = keys.view(-1, self.layer_size)
        values_flat = values.view(-1, self.layer_size)

        grad_fn = grad(self._associative_loss, has_aux=True)
        grads, _ = grad_fn(dict(params), keys_flat, values_flat, adaptive_lr)
        for name, param in self.lmm.named_parameters():
            if grads[name] is not None:
                param.grad = grads[name]
        self.optimizer.step()

        surprises = TensorDict(grads).mul(-1)
        retrieved = self.lmm(queries)

        return retrieved, surprises
