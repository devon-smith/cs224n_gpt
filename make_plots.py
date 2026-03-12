import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})
os.makedirs('figures', exist_ok=True)

# ── Benchmark data ────────────────────────────────────────────────────────────
seq_lens = [128, 256, 512, 1024]

time_ms = {
    'Standard':            [11.4, 23.5, 55.6, 141.3],
    'Flash':               [10.4, 19.4, 41.3,  86.6],
    'Sliding Window':      [11.4, 23.5, 55.7, 141.6],
    'Mixed':               [11.5, 23.6, 55.5, 141.2],
    'GQA (4 KV heads)':    [10.7, 21.9, 51.3, 134.7],
    'GQA + Flash':         [ 9.8, 21.9, 51.4, 134.9],
}

memory_gb = {
    'Standard':            [0.534, 0.562, 0.665, 1.018],
    'Flash':               [0.535, 0.554, 0.595, 0.677],
    'Sliding Window':      [0.535, 0.564, 0.665, 1.018],
    'Mixed':               [0.535, 0.564, 0.665, 1.019],
    'GQA (4 KV heads)':    [0.497, 0.526, 0.627, 0.980],
    'GQA + Flash':         [0.497, 0.526, 0.627, 0.980],
}

colors = {
    'Standard':         '#2c7bb6',
    'Flash':            '#d7191c',
    'Sliding Window':   '#1a9641',
    'Mixed':            '#ff7f00',
    'GQA (4 KV heads)': '#984ea3',
    'GQA + Flash':      '#a65628',
}
markers = {
    'Standard': 'o', 'Flash': 's', 'Sliding Window': '^',
    'Mixed': 'D', 'GQA (4 KV heads)': 'v', 'GQA + Flash': 'P',
}

# ── Plot 1: Benchmark line plots ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for name in time_ms:
    ax1.plot(seq_lens, time_ms[name], label=name,
             color=colors[name], marker=markers[name],
             linewidth=2, markersize=6)
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Time per Forward Pass (ms)')
ax1.set_title('Latency vs. Sequence Length')
ax1.set_xticks(seq_lens)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

for name in memory_gb:
    ax2.plot(seq_lens, memory_gb[name], label=name,
             color=colors[name], marker=markers[name],
             linewidth=2, markersize=6)
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Peak GPU Memory (GB)')
ax2.set_title('Memory Usage vs. Sequence Length')
ax2.set_xticks(seq_lens)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/benchmark.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/benchmark.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/benchmark.pdf + .png")

# ── Plot 2: CFIMDB + SST bar chart ────────────────────────────────────────────
variants     = ['Last-linear\n(standard)', 'Full-model\n(standard)', 'Full-model\n(flash)',
                'Full-model\n(sliding\nwindow)', 'Full-model\n(mixed)']
sst_acc      = [0.460, 0.513, 0.512, 0.515, 0.512]
cfimdb_acc   = [0.861, 0.971, 0.976, 0.873, 0.947]

x = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, sst_acc,    width, label='SST',    color='#2c7bb6', alpha=0.85)
bars2 = ax.bar(x + width/2, cfimdb_acc, width, label='CFIMDB', color='#d7191c', alpha=0.85)

ax.set_ylabel('Dev Accuracy')
ax.set_title('Sentiment Classification Accuracy by Attention Variant')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9)
ax.set_ylim(0.3, 1.05)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Value labels on bars
for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/sentiment_accuracy.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/sentiment_accuracy.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/sentiment_accuracy.pdf + .png")
