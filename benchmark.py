"""
Benchmarks speed and memory for each attention variant.
Run in Colab: python benchmark.py

Two benchmarks are reported:
  1. Full forward pass: wall-clock time and peak GPU memory for a full GPT-2
     forward pass. Peak memory is dominated by model weights (~517 MB) so
     attention-level savings are not visible here.
  2. Attention-only: measures memory allocated by a single CausalSelfAttention
     call in isolation, before and after, to isolate the score-matrix footprint.
"""

import torch
import time
from models.gpt2 import GPT2Model
from modules.attention import CausalSelfAttention
from config import GPT2Config

VARIANTS = [
  ('standard',                         {}),
  ('flash',                            {}),
  ('sliding_window_masked (w=64)',     {'attention_type': 'sliding_window_masked', 'window_size': 64}),
  ('sliding_window_efficient (w=64)',  {'attention_type': 'sliding_window',        'window_size': 64}),
  ('sliding_window_masked (w=128)',    {'attention_type': 'sliding_window_masked', 'window_size': 128}),
  ('sliding_window_efficient (w=128)', {'attention_type': 'sliding_window',        'window_size': 128}),
  ('mixed (w=64)',                     {'attention_type': 'mixed',                 'window_size': 64}),
  ('gqa_4heads',                       {'attention_type': 'standard', 'num_kv_heads': 4}),
  ('gqa_flash',                        {'attention_type': 'flash',    'num_kv_heads': 4}),
]

SEQ_LENS    = [128, 256, 512, 1024]
BATCH_SIZE  = 4
NUM_RUNS    = 30
WARMUP_RUNS = 5


def benchmark(label, attention_type, seq_len, extra_kwargs):
  config = GPT2Config(attention_type=attention_type, **extra_kwargs)
  model = GPT2Model(config).cuda().eval()

  input_ids = torch.randint(0, 50257, (BATCH_SIZE, seq_len)).cuda()
  mask = torch.ones(BATCH_SIZE, seq_len).cuda()

  for _ in range(WARMUP_RUNS):
    with torch.no_grad():
      model(input_ids, mask)

  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  start = time.time()

  for _ in range(NUM_RUNS):
    with torch.no_grad():
      model(input_ids, mask)

  torch.cuda.synchronize()
  elapsed_ms = (time.time() - start) / NUM_RUNS * 1000
  memory_gb  = torch.cuda.max_memory_allocated() / 1e9

  print(f"{label:30s} | seq={seq_len:4d} | {elapsed_ms:6.1f} ms | {memory_gb:.3f} GB")
  return elapsed_ms, memory_gb


def benchmark_attention_only(label, attention_type, seq_len, extra_kwargs):
  """
  Measures memory allocated by a single attention layer call in isolation.
  Computes before/after torch.cuda.memory_allocated() to capture only the
  tensors live during the attention computation, excluding model weights.
  """
  hidden_size = 768
  config = GPT2Config(
    hidden_size=hidden_size,
    attention_type=attention_type,
    **extra_kwargs
  )
  attn = CausalSelfAttention(config).cuda().eval()

  # [bs, seq_len, hidden_size] input, [bs, 1, 1, seq_len] mask
  x    = torch.randn(BATCH_SIZE, seq_len, hidden_size, device='cuda')
  mask = torch.zeros(BATCH_SIZE, 1, 1, seq_len, device='cuda')

  # Warmup
  for _ in range(WARMUP_RUNS):
    with torch.no_grad():
      attn(x, mask)

  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()

  # Measure memory delta across a single call
  before_mb = torch.cuda.memory_allocated() / 1e6
  with torch.no_grad():
    attn(x, mask)
  torch.cuda.synchronize()
  peak_mb   = torch.cuda.max_memory_allocated() / 1e6
  delta_mb  = peak_mb - before_mb

  print(f"{label:30s} | seq={seq_len:4d} | peak={peak_mb:7.1f} MB | delta={delta_mb:6.1f} MB")
  return peak_mb, delta_mb


if __name__ == '__main__':
  if not torch.cuda.is_available():
    print("CUDA not available — benchmark requires a GPU.")
    exit(1)

  print("=" * 75)
  print("FULL FORWARD PASS BENCHMARK")
  print("=" * 75)
  print(f"{'Variant':30s} | {'Seq':>6} | {'Time':>8} | {'Memory':>9}")
  print("-" * 65)

  for seq_len in SEQ_LENS:
    for label, kwargs in VARIANTS:
      attn_type = kwargs.pop('attention_type', label.split('_')[0])
      benchmark(label, attn_type, seq_len, kwargs)
    print()

  print()
  print("=" * 75)
  print("ATTENTION-ONLY MEMORY BENCHMARK (isolates score matrix footprint)")
  print("=" * 75)
  print(f"{'Variant':30s} | {'Seq':>6} | {'Peak MB':>9} | {'Delta MB':>9}")
  print("-" * 65)

  attn_variants = [
    ('standard',                          'standard',                {}),
    ('sliding_window_masked (w=64)',      'sliding_window_masked',   {'window_size': 64}),
    ('sliding_window_efficient (w=64)',   'sliding_window',          {'window_size': 64}),
    ('sliding_window_masked (w=128)',     'sliding_window_masked',   {'window_size': 128}),
    ('sliding_window_efficient (w=128)',  'sliding_window',          {'window_size': 128}),
  ]

  for seq_len in SEQ_LENS:
    for label, attn_type, kwargs in attn_variants:
      benchmark_attention_only(label, attn_type, seq_len, dict(kwargs))
    print()
