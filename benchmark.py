"""
Benchmarks speed and memory for each attention variant.
Run in Colab: python benchmark.py
"""

import csv
import torch
import time
from models.gpt2 import GPT2Model
from config import GPT2Config

VARIANTS = [
  ('standard',             {'attention_type': 'standard'}),
  ('flash',                {'attention_type': 'flash'}),
  ('sliding_window_w64',   {'attention_type': 'sliding_window', 'window_size': 64}),
  ('sliding_window_w128',  {'attention_type': 'sliding_window', 'window_size': 128}),
  ('sw_w128_g2',           {'attention_type': 'sliding_window', 'window_size': 128,
                            'num_global_tokens': 2}),
  ('sw_w128_g8',           {'attention_type': 'sliding_window', 'window_size': 128,
                            'num_global_tokens': 8}),
  ('mixed_w64',            {'attention_type': 'mixed', 'window_size': 64}),
  ('gqa_4heads',           {'attention_type': 'standard', 'num_kv_heads': 4}),
  ('gqa_flash',            {'attention_type': 'flash',    'num_kv_heads': 4}),
]

SEQ_LENS    = [128, 256, 512, 1024]
BATCH_SIZE  = 4
NUM_RUNS    = 30
WARMUP_RUNS = 5


def benchmark(label, seq_len, config_kwargs):
  config = GPT2Config(**config_kwargs)
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

  print(f"{label:25s} | seq={seq_len:4d} | {elapsed_ms:6.1f} ms | {memory_gb:.3f} GB")
  return elapsed_ms, memory_gb


if __name__ == '__main__':
  if not torch.cuda.is_available():
    print("CUDA not available — benchmark requires a GPU.")
    exit(1)

  print(f"{'Variant':25s} | {'Seq':>6} | {'Time':>8} | {'Memory':>9}")
  print("-" * 60)

  rows = []
  for seq_len in SEQ_LENS:
    for label, kwargs in VARIANTS:
      elapsed_ms, memory_gb = benchmark(label, seq_len, kwargs)
      rows.append({'variant': label, 'seq_len': seq_len,
                   'time_ms': round(elapsed_ms, 2), 'memory_gb': round(memory_gb, 4)})
    print()

  with open('benchmark_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['variant', 'seq_len', 'time_ms', 'memory_gb'])
    writer.writeheader()
    writer.writerows(rows)
  print("Wrote benchmark_results.csv")
