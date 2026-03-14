import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn


def _efficient_sliding_window_attention(query, key, value, window_size, attention_mask, dropout_module, training):
  """
  O(T·w) memory sliding window attention using a loop over window offsets.

  Each token attends to the `window_size` most recent tokens (including itself).
  Key and value tensors are padded once at the start; each loop iteration takes
  a contiguous stride-1 slice — no hidden .contiguous() copies.

  Args:
    query, key, value: [bs, heads, T, d]
    attention_mask:    [bs, 1, 1, T]  (0=real, large-negative=padding)
  Returns:
    output: [bs, heads, T, d]
  """
  bs, heads, T, d = query.shape
  pad_len = window_size - 1

  # Pad K and V at the start of the sequence dimension
  k_padded = F.pad(key,   (0, 0, pad_len, 0))  # [bs, heads, T+W-1, d]
  v_padded = F.pad(value, (0, 0, pad_len, 0))

  # Pre-allocate scores [bs, heads, T, W]
  # scores[:, :, t, w] = dot(q_t, k_{t - W + 1 + w}) / sqrt(d)
  scores = torch.empty(bs, heads, T, window_size, device=query.device, dtype=query.dtype)
  for w in range(window_size):
    k_slice = k_padded[:, :, w:w + T, :]        # [bs, heads, T, d] — contiguous view
    scores[:, :, :, w] = (query * k_slice).sum(-1) / (d ** 0.5)

  # Causal mask: window position w for token t is before sequence start when
  # t - W + 1 + w < 0, i.e., w < pad_len - t
  causal_mask = torch.zeros(T, window_size, device=query.device, dtype=torch.bool)
  for t in range(min(pad_len, T)):
    n_invalid = pad_len - t
    causal_mask[t, :n_invalid] = True
  scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

  # Padding mask: map attention_mask [bs, 1, 1, T] to [bs, 1, T, W] by looking
  # up the key position each window slot corresponds to
  key_positions = (
    torch.arange(T, device=query.device).unsqueeze(1) - pad_len
    + torch.arange(window_size, device=query.device).unsqueeze(0)
  ).clamp(0, T - 1)                              # [T, W]
  attn_mask_flat = attention_mask.squeeze(2)     # [bs, 1, T]
  pad_mask = attn_mask_flat[:, :, key_positions] # [bs, 1, T, W]
  pad_mask = pad_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
  scores = scores + pad_mask

  probs = torch.softmax(scores, dim=-1)
  if training:
    probs = dropout_module(probs)

  # Weighted sum of values — same contiguous-slice loop pattern
  output = torch.zeros(bs, heads, T, d, device=query.device, dtype=query.dtype)
  for w in range(window_size):
    v_slice = v_padded[:, :, w:w + T, :]        # [bs, heads, T, d] — contiguous view
    output += probs[:, :, :, w:w + 1] * v_slice

  return output  # [bs, heads, T, d]


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # attention variant configuration where can specify 'standard', 'flash', 'sliding_window', 'mixed', 'gqa'
    self.attention_type = getattr(config, 'attention_type', 'standard')
    self.window_size = getattr(config, 'window_size', 128)
    # Number of leading tokens that attend globally (Longformer-style).
    # Only used when attention_type == 'sliding_window'.
    self.num_global_tokens = getattr(config, 'num_global_tokens', 0)

    # Track layer index for mixed local/global attention.
    # Each CausalSelfAttention increments a counter stored on the config so that
    # even layers use global (full causal) and odd layers use sliding window
    if not hasattr(config, '_layer_counter'):
      config._layer_counter = 0
    self.layer_idx = config._layer_counter
    config._layer_counter += 1

    # GQA: num_kv_heads can be fewer than num_attention_heads
    # For standard/flash/sliding_window/mixed, num_kv_heads == num_attention_heads
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
    _, _, seq_len, head_size = query.size()

    # Mixed local / global attention
    # alternate layers between sliding window (local) and full causal (global)
    # even layers = full causal (global), odd layers = sliding window (local)
    if self.attention_type == 'mixed':
      effective_type = 'sliding_window' if (self.layer_idx % 2 == 1) else 'standard'
    else:
      effective_type = self.attention_type

    # Flash Attention
    # F.scaled_dot_product_attention selects Flash Attention automatically
    # on CUDA. is_causal=True handles the causal mask internally
    if effective_type == 'flash':
      attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=True,
      )
      attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
      return attn_output

    # Efficient sliding window: O(T·w) memory via contiguous-slice loop
    if effective_type == 'sliding_window':
      attn_output = _efficient_sliding_window_attention(
        query, key, value,
        window_size=self.window_size,
        attention_mask=attention_mask,
        dropout_module=self.dropout,
        training=self.training,
      )
      attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
      return attn_output

    # Standard / Sliding Window Masked (non-flash path)
    # Scaled dot-product scores: [bs, num_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / (head_size ** 0.5)

    # Causal mask: block future positions (j > i)
    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1
    ).bool()

    # Sliding window (mask-based, O(n²) memory): build full n×n matrix, mask out-of-window positions
    if effective_type == 'sliding_window_masked':
      window_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=-(self.window_size + 1)
      ).bool()
      causal_mask = causal_mask | window_mask

    attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

    # Padding mask (already formatted as large negatives for padding positions)
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
    if self.num_kv_heads != self.num_attention_heads:
      groups = self.num_attention_heads // self.num_kv_heads
      key_layer   = key_layer.repeat_interleave(groups, dim=1)
      value_layer = value_layer.repeat_interleave(groups, dim=1)

    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
