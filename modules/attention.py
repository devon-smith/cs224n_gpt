import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    Compute scaled dot-product attention with causal masking.

    Args:
      key: [bs, num_attention_heads, seq_len, attention_head_size]
      query: [bs, num_attention_heads, seq_len, attention_head_size]
      value: [bs, num_attention_heads, seq_len, attention_head_size]
      attention_mask: [bs, 1, 1, seq_len] - padding mask (0 for padding, large negative for real tokens after processing)

    Returns:
      attn_output: [bs, seq_len, hidden_state]
    """
    # Get dimensions
    bs, num_heads, seq_len, head_size = query.size()

    # Compute attention scores: Q * K^T / sqrt(d_k)
    # [bs, num_heads, seq_len, head_size] @ [bs, num_heads, head_size, seq_len] -> [bs, num_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / (head_size ** 0.5)

    # Create causal mask (upper triangular) to prevent attending to future tokens
    # Mask positions where j > i (future positions)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1).bool()
    attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

    # Apply the padding attention mask (already in the form where padding positions have large negative values)
    attention_scores = attention_scores + attention_mask

    # Apply softmax to get attention weights
    attention_probs = torch.softmax(attention_scores, dim=-1)

    # Apply dropout to attention weights
    attention_probs = self.dropout(attention_probs)

    # Compute attention output: attention_probs @ V
    # [bs, num_heads, seq_len, seq_len] @ [bs, num_heads, seq_len, head_size] -> [bs, num_heads, seq_len, head_size]
    attn_output = torch.matmul(attention_probs, value)

    # Reshape back to [bs, seq_len, hidden_state]
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')

    return attn_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
