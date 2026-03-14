import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})

seq_lens = [128, 256, 512, 1024]

data = {
    'Standard':              {'time': [11.4, 23.5, 55.6, 141.3], 'mem': [0.534, 0.562, 0.665, 1.018]},
    'Flash':                 {'time': [10.4, 19.4, 41.3,  86.6], 'mem': [0.535, 0.554, 0.595, 0.677]},
    'Sliding Window (w=64)': {'time': [11.4, 23.5, 55.7, 141.6], 'mem': [0.535, 0.564, 0.665, 1.018]},
    'Sliding Window (w=128)':{'time': [11.4, 23.5, 55.7, 141.5], 'mem': [0.535, 0.564, 0.665, 1.018]},
    'Mixed (w=64)':          {'time': [11.5, 23.6, 55.5, 141.2], 'mem': [0.535, 0.564, 0.665, 1.019]},
    'GQA (4 KV heads)':      {'time': [10.7, 21.9, 51.3, 134.7], 'mem': [0.497, 0.526, 0.627, 0.980]},
    'GQA + Flash':           {'time': [ 9.8, 21.9, 51.4, 134.9], 'mem': [0.497, 0.526, 0.627, 0.980]},
}

colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf']
markers   = ['o', 's', '^', 'v', 'D', 'P', 'X']
linestyles = ['-', '-', '--', ':', '--', '-', '-']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, metric, ylabel, title in [
    (axes[0], 'time', 'Latency (ms)',        'Forward Pass Latency'),
    (axes[1], 'mem',  'Peak GPU Memory (GB)', 'Peak GPU Memory'),
]:
    for i, (label, vals) in enumerate(data.items()):
        ax.plot(seq_lens, vals[metric],
                label=label,
                color=colors[i],
                marker=markers[i],
                linestyle=linestyles[i],
                linewidth=1.8,
                markersize=6)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(seq_lens)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=9, loc='upper left')
axes[1].legend(fontsize=9, loc='upper left')

plt.suptitle('Efficiency Benchmark on NVIDIA A10G GPU', fontsize=13, y=1.01)
plt.tight_layout()

plt.savefig('figures/benchmark_split.pdf', bbox_inches='tight')
plt.savefig('figures/benchmark_split.png', bbox_inches='tight', dpi=150)
print('Saved figures/benchmark_split.pdf and .png')
