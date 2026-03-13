import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # attention variant configuration
    # Options: 'standard', 'flash', 'sliding_window', 'mixed', 'gqa'
    self.attention_type = getattr(config, 'attention_type', 'standard')
    self.window_size = getattr(config, 'window_size', 128)
    # Number of leading tokens that attend globally (Longformer-style).
    # Only used when attention_type == 'sliding_window'.
    self.num_global_tokens = getattr(config, 'num_global_tokens', 0)

    # Track layer index for mixed local/global attention.
    # Each CausalSelfAttention increments a counter stored on the config so that
    # even layers use global (full causal) and odd layers use sliding window.
    if not hasattr(config, '_layer_counter'):
      config._layer_counter = 0
    self.layer_idx = config._layer_counter
    config._layer_counter += 1

    # GQA: num_kv_heads can be fewer than num_attention_heads
    # For standard/flash/sliding_window/mixed, num_kv_heads == num_attention_heads.
    self.num_kv_heads = getattr(config, 'num_kv_heads', self.num_attention_heads)
    if not self.num_kv_heads:  # treat 0 or None as "use all heads" (no group query attention)
      self.num_kv_heads = self.num_attention_heads
    assert self.num_attention_heads % self.num_kv_heads == 0, \
      "num_attention_heads must be divisible by num_kv_heads"
    self.kv_all_head_size = self.num_kv_heads * self.attention_head_size

    # Query always uses the full number of attention heads
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    # Key and value use num_kv_heads (equal to num_attention_heads unless GQA)
    self.key = nn.Linear(config.hidden_size, self.kv_all_head_size)
    self.value = nn.Linear(config.hidden_size, self.kv_all_head_size)

    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer, num_heads):
    """Project x with linear_layer and split into num_heads heads."""
    proj = linear_layer(x)
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=num_heads)
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    Dispatch to the correct attention variant based on self.attention_type.

    Args:
      key:            [bs, num_attention_heads, seq_len, attention_head_size]
      query:          [bs, num_attention_heads, seq_len, attention_head_size]
      value:          [bs, num_attention_heads, seq_len, attention_head_size]
      attention_mask: [bs, 1, 1, seq_len]  (0 for real tokens, -10000 for padding)

    Returns:
      attn_output: [bs, seq_len, hidden_size]
    """
    bs, num_heads, seq_len, head_size = query.size()

    # Mixed local / global attention
    #alternate layes between sliding window (local) and full causal (global)
    # even layers = full causal (global), odd layers = sliding window (local)
    if self.attention_type == 'mixed':
      effective_type = 'sliding_window' if (self.layer_idx % 2 == 1) else 'standard'
    else:
      effective_type = self.attention_type

    # Flash Attention
    # F.scaled_dot_product_attention selects Flash Attention automatically
    # on CUDA. is_causal=True handles the causal mask internally.
    if effective_type == 'flash':
      attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=True,
      )
      attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
      return attn_output

    # Memory-efficient sliding window: O(n * w) instead of O(n^2)
    if effective_type == 'sliding_window':
      from modules.sliding_window_attention import sliding_window_attention
      attn_output = sliding_window_attention(
        query, key, value,
        window_size=self.window_size,
        attention_mask=attention_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        num_global_tokens=self.num_global_tokens,
        training=self.training,
      )
      attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
      return attn_output

    # Standard full causal attention: O(n^2)
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / (head_size ** 0.5)

    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1
    ).bool()
    attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

    attention_scores = attention_scores + attention_mask

    attention_probs = torch.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    attn_output = torch.matmul(attention_probs, value)
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
    return attn_output

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_size]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_size]
    """
    query_layer = self.transform(hidden_states, self.query, self.num_attention_heads)
    key_layer   = self.transform(hidden_states, self.key,   self.num_kv_heads)
    value_layer = self.transform(hidden_states, self.value, self.num_kv_heads)

    # GQA: expand K/V heads to match the number of Q heads via repetition.
    # e.g. with num_attention_heads=12 and num_kv_heads=4, each KV head is
    # shared by 3 query heads.
    if self.num_kv_heads != self.num_attention_heads:
      groups = self.num_attention_heads // self.num_kv_heads
      key_layer   = key_layer.repeat_interleave(groups, dim=1)
      value_layer = value_layer.repeat_interleave(groups, dim=1)

    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
