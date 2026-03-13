"""
Memory-efficient sliding window causal attention using unfold.

Instead of materializing the full [seq_len, seq_len] attention matrix and
masking out-of-window positions, this module only computes scores within each
token's local window using torch.Tensor.unfold(), reducing attention memory
from O(n^2) to O(n * w).

Optional Longformer-style global tokens: the first `num_global_tokens`
positions attend to all causal positions and are attended to by all tokens.
"""

import torch

# Chunk size for processing sliding window attention along the sequence
# dimension to avoid materializing large non-contiguous unfolded views.
_SEQ_CHUNK = 64


def sliding_window_attention(
    query, key, value, window_size, attention_mask,
    dropout_p=0.0, num_global_tokens=0, training=False,
):
  """
  Memory-efficient sliding window causal attention.

  Args:
    query:  [bs, heads, seq_len, head_dim]
    key:    [bs, heads, seq_len, head_dim]
    value:  [bs, heads, seq_len, head_dim]
    window_size: int, number of past positions each token attends to
                 (including itself). Total window = window_size positions.
    attention_mask: [bs, 1, 1, seq_len] additive mask (0=real, large neg=pad)
    dropout_p: dropout probability for attention weights
    num_global_tokens: first g positions use full causal attention and are
                       visible to all other tokens
    training: whether in training mode (for dropout)

  Returns:
    output: [bs, heads, seq_len, head_dim]
  """
  bs, heads, seq_len, head_dim = query.size()
  g = min(num_global_tokens, seq_len)

  if g == 0:
    return _local_attention(query, key, value, window_size,
                            attention_mask, dropout_p, training)

  # Split into global tokens (first g) and local tokens (rest)
  q_global = query[:, :, :g, :]    # [bs, heads, g, head_dim]
  q_local = query[:, :, g:, :]     # [bs, heads, seq_len-g, head_dim]
  local_len = seq_len - g

  # --- Global token rows: full causal attention over positions 0..g-1 ---
  # Global tokens attend to all previous global tokens (standard causal)
  global_scores = torch.matmul(q_global, key[:, :, :g, :].transpose(-1, -2))
  global_scores = global_scores / (head_dim ** 0.5)
  # Causal mask for global tokens: block j > i
  causal_g = torch.triu(
      torch.ones(g, g, device=query.device, dtype=torch.bool), diagonal=1
  )
  global_scores = global_scores.masked_fill(causal_g, float('-inf'))
  # Apply padding mask for global positions
  global_scores = global_scores + attention_mask[:, :, :, :g]
  global_probs = torch.softmax(global_scores, dim=-1)
  if training and dropout_p > 0:
    global_probs = torch.nn.functional.dropout(global_probs, p=dropout_p)
  global_out = torch.matmul(global_probs, value[:, :, :g, :])  # [bs, h, g, d]

  # --- Local token rows: attend to global tokens + local window ---
  if local_len == 0:
    return global_out

  # Part A: local tokens attending to global tokens
  # [bs, heads, local_len, g]
  scores_to_global = torch.matmul(q_local, key[:, :, :g, :].transpose(-1, -2))
  scores_to_global = scores_to_global / (head_dim ** 0.5)
  # Apply padding mask for global key positions
  scores_to_global = scores_to_global + attention_mask[:, :, :, :g]

  # Part B: local tokens attending to local window (unfold-based)
  k_local = key[:, :, g:, :]
  v_local = value[:, :, g:, :]
  pad_mask_local = attention_mask[:, :, :, g:]  # [bs, 1, 1, local_len]

  # Pad key/value on the left so each position has window_size keys available
  pad_len = window_size - 1
  k_padded = torch.nn.functional.pad(k_local, (0, 0, pad_len, 0))
  v_padded = torch.nn.functional.pad(v_local, (0, 0, pad_len, 0))
  # Pad the mask similarly
  mask_padded = torch.nn.functional.pad(
      pad_mask_local, (pad_len, 0), value=-1e9
  )  # [bs, 1, 1, pad_len + local_len]

  # Unfold to get windows: [bs, heads, local_len, window_size, head_dim]
  k_windows = k_padded.unfold(2, window_size, 1)   # [..., head_dim, window_size]
  k_windows = k_windows.transpose(-1, -2)           # [..., window_size, head_dim]
  v_windows = v_padded.unfold(2, window_size, 1)
  v_windows = v_windows.transpose(-1, -2)

  # Unfold mask: [bs, 1, local_len, window_size]
  mask_windows = mask_padded.unfold(3, window_size, 1)  # [bs, 1, 1, local_len, ws]
  mask_windows = mask_windows.squeeze(2)                 # [bs, 1, local_len, ws]

  # Compute local scores in chunks: [bs, heads, local_len, window_size]
  scores_local = torch.empty(bs, heads, local_len, window_size,
                              device=query.device, dtype=query.dtype)
  for _s in range(0, local_len, _SEQ_CHUNK):
    _e = min(_s + _SEQ_CHUNK, local_len)
    scores_local[:, :, _s:_e, :] = torch.einsum(
        'bhqd, bhqwd -> bhqw',
        q_local[:, :, _s:_e, :],
        k_windows[:, :, _s:_e, :, :].contiguous(),
    )
  scores_local = scores_local / (head_dim ** 0.5)

  # Build causal mask within the window
  # For query at local position i, key at window position j corresponds to
  # absolute local position (i - window_size + 1 + j).
  # Block if absolute position > i (future) or < 0 (before sequence start)
  pos_q = torch.arange(local_len, device=query.device).unsqueeze(1)  # [local_len, 1]
  pos_k = pos_q - (window_size - 1) + torch.arange(
      window_size, device=query.device
  ).unsqueeze(0)  # [local_len, window_size]
  window_causal_mask = (pos_k > pos_q) | (pos_k < 0)  # True = blocked
  scores_local = scores_local.masked_fill(window_causal_mask, float('-inf'))

  # Apply padding mask within window
  scores_local = scores_local + mask_windows

  # Concatenate global + local scores: [bs, heads, local_len, g + window_size]
  all_scores = torch.cat([scores_to_global, scores_local], dim=-1)
  all_probs = torch.softmax(all_scores, dim=-1)
  # Guard against NaN from all-masked rows
  all_probs = torch.nan_to_num(all_probs, nan=0.0)
  if training and dropout_p > 0:
    all_probs = torch.nn.functional.dropout(all_probs, p=dropout_p)

  # Split probs and compute weighted sum
  probs_global_part = all_probs[:, :, :, :g]         # [bs, h, local_len, g]
  probs_local_part = all_probs[:, :, :, g:]           # [bs, h, local_len, ws]

  out_from_global = torch.matmul(
      probs_global_part, value[:, :, :g, :]
  )  # [bs, h, local_len, d]
  # Chunked weighted sum for local part
  out_from_local = torch.empty(bs, heads, local_len, head_dim,
                                device=query.device, dtype=query.dtype)
  for _s in range(0, local_len, _SEQ_CHUNK):
    _e = min(_s + _SEQ_CHUNK, local_len)
    out_from_local[:, :, _s:_e, :] = torch.einsum(
        'bhqw, bhqwd -> bhqd',
        probs_local_part[:, :, _s:_e, :],
        v_windows[:, :, _s:_e, :, :].contiguous(),
    )  # [bs, h, local_len, d]
  local_out = out_from_global + out_from_local

  # Reassemble: [bs, heads, seq_len, head_dim]
  return torch.cat([global_out, local_out], dim=2)


def _local_attention(query, key, value, window_size, attention_mask,
                     dropout_p=0.0, training=False):
  """
  Pure sliding window causal attention without global tokens.

  Only materializes [bs, heads, seq_len, window_size] instead of
  [bs, heads, seq_len, seq_len].
  """
  bs, heads, seq_len, head_dim = query.size()

  # Pad key/value on the left so each position has window_size keys
  pad_len = window_size - 1
  k_padded = torch.nn.functional.pad(key, (0, 0, pad_len, 0))
  v_padded = torch.nn.functional.pad(value, (0, 0, pad_len, 0))
  # Pad attention mask
  mask_padded = torch.nn.functional.pad(
      attention_mask, (pad_len, 0), value=-1e9
  )  # [bs, 1, 1, pad_len + seq_len]

  # Unfold into windows: [bs, heads, seq_len, head_dim, window_size]
  k_windows = k_padded.unfold(2, window_size, 1).transpose(-1, -2)
  v_windows = v_padded.unfold(2, window_size, 1).transpose(-1, -2)
  # [bs, heads, seq_len, window_size, head_dim]

  # Unfold mask: [bs, 1, seq_len, window_size]
  mask_windows = mask_padded.unfold(3, window_size, 1).squeeze(2)

  # Compute scores in chunks to avoid materializing full contiguous k_windows
  # [bs, heads, seq_len, window_size]
  scores = torch.empty(bs, heads, seq_len, window_size,
                        device=query.device, dtype=query.dtype)
  for _s in range(0, seq_len, _SEQ_CHUNK):
    _e = min(_s + _SEQ_CHUNK, seq_len)
    scores[:, :, _s:_e, :] = torch.einsum(
        'bhqd, bhqwd -> bhqw',
        query[:, :, _s:_e, :],
        k_windows[:, :, _s:_e, :, :].contiguous(),
    )
  scores = scores / (head_dim ** 0.5)

  # Causal mask within window
  pos_q = torch.arange(seq_len, device=query.device).unsqueeze(1)
  pos_k = pos_q - (window_size - 1) + torch.arange(
      window_size, device=query.device
  ).unsqueeze(0)
  window_causal_mask = (pos_k > pos_q) | (pos_k < 0)
  scores = scores.masked_fill(window_causal_mask, float('-inf'))

  # Apply padding mask
  scores = scores + mask_windows

  probs = torch.softmax(scores, dim=-1)
  probs = torch.nan_to_num(probs, nan=0.0)
  if training and dropout_p > 0:
    probs = torch.nn.functional.dropout(probs, p=dropout_p)

  # Weighted sum in chunks: [bs, heads, seq_len, head_dim]
  output = torch.empty(bs, heads, seq_len, head_dim,
                        device=query.device, dtype=query.dtype)
  for _s in range(0, seq_len, _SEQ_CHUNK):
    _e = min(_s + _SEQ_CHUNK, seq_len)
    output[:, :, _s:_e, :] = torch.einsum(
        'bhqw, bhqwd -> bhqd',
        probs[:, :, _s:_e, :],
        v_windows[:, :, _s:_e, :, :].contiguous(),
    )
  return output
