from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    Apply dense transformation, dropout, and residual connection.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    # Apply dense transformation
    transformation = dense_layer(output)
    # Apply dropout to transformed output of each sub-layer
    transformation = dropout(transformation)
    # Apply residual connection
    return input + transformation


  def forward(self, hidden_states, attention_mask):
    """
    GPT-2 uses Pre-LayerNorm architecture:
      1. LayerNorm -> Self-Attention -> Dropout -> Residual
      2. LayerNorm -> FFN -> Dropout -> Residual
    """
    # go through steps of part one
    # Apply layer norm
    normed_hidden_states = self.attention_layer_norm(hidden_states)
    # self-attention
    attention_output = self.self_attention(normed_hidden_states, attention_mask)
    # dense projection, dropout, and residual connection
    #use self.attention_dropout which drops with probability p
    hidden_states = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)

    #go through steps of part two
    # Apply layer norm
    normed_hidden_states = self.out_layer_norm(hidden_states)
    # intermediate dense layer with GELU activation
    intermediate_output = self.interm_af(self.interm_dense(normed_hidden_states))
    # Apply output dense, dropout, and residual connection
    hidden_states = self.add(hidden_states, intermediate_output, self.out_dense, self.out_dropout)

    return hidden_states

