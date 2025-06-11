import torch.nn as nn
import torch


class LlamaMemoryAsLayer(nn.Module):
    def __init__(self, original_layer, lmm):
        super().__init__()
        self.original_layer = original_layer
        self.lmm = lmm

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        with torch.set_grad_enabled(self.training):
            output = self.original_layer(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        attn_output = output[0]
        mal_output = self.lmm(attn_output)
        return (mal_output,) + output[1:]
