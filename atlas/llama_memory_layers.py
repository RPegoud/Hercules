import torch.nn as nn


class LlamaMemoryAsLayer(nn.Module):
    def __init__(self, original_layer, mal):
        super().__init__()
        self.original_layer = original_layer
        self.mal = mal

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        output = self.original_layer(
            hidden_states, attention_mask=attention_mask, **kwargs
        )
        attn_output = output[0]
        print(attn_output.shape)
        mal_output = self.mal(attn_output)
        return (mal_output,) + output[1:]
