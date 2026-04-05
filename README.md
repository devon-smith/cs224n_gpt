# GPT-2 From Scratch

A ground-up implementation of GPT-2, fine-tuned on three downstream tasks — and then an excuse to go deep on attention mechanisms and figure out which ones are actually worth the hype.

## What This Is

We built GPT-2 from scratch — multi-head causal self-attention, transformer layers with pre-LayerNorm, residual connections, AdamW, pretrained weight loading, the whole thing. Then we pointed it at three tasks to see how it holds up:

- **Sentiment Classification** — fine-tuning on SST and CFIMDB datasets, both last-layer-only and full-model. CFIMDB hits 0.97–0.98 with full fine-tuning. SST is harder and less generous with training data, landing around 0.46–0.52 across variants.
- **Paraphrase Detection** — cloze-style classification on the Quora paraphrase dataset, where the model reads a sentence pair and predicts "yes" or "no." Straightforward in theory, fiddly in practice.
- **Sonnet Generation** — autoregressive language modeling to generate Shakespeare-style sonnets. The results are... recognizably Shakespearean, which is about all you can ask from a model this size.

## The Interesting Part: Attention Variants

Beyond task accuracy, we benchmarked multiple attention mechanisms across sequence lengths 128–1024, measuring latency and peak memory. This is where most of the engineering effort went.

| Variant | What It Does |
|---|---|
| Standard | Full causal self-attention, O(T²). The baseline everyone's trying to beat. |
| Flash | PyTorch's `scaled_dot_product_attention` with `is_causal=True`. Spoiler: it wins. |
| Sliding Window | Efficient O(T·w) using `unfold()` + chunked einsum. Doesn't look at the full sequence — just a local window. |
| Mixed | Alternates full causal (even layers) and sliding window (odd layers). Best of both, in theory. |
| GQA | Grouped query attention — reduces KV heads from 12→4, paired with flash. Fewer heads, less memory. |
| Global Tokens | First *g* tokens attend globally; the rest attend locally within a window. A compromise that actually works. |

## Key Results

- **Flash attention** is fastest and most memory-efficient end-to-end — 54ms / 673MB at seq=1024 vs. 78ms / 1014MB for standard. Not close.
- **Sliding window** achieves ~70% reduction in attention score matrix memory (143MB vs 466MB at seq=1024, w=128). The trick was using `unfold()` instead of materializing the full N×N matrix, which we learned the hard way matters.
- **CFIMDB** accuracy hits 0.97–0.98 with full-model fine-tuning. SST is tougher — 0.46–0.52 across variants.

## Infrastructure

- `benchmark.py` — profiles latency and peak memory per variant/sequence length, outputs CSV
- `make_plots.py` — generates figures for benchmark results, attention memory scaling, and sentiment accuracy
- `modal_train.py` — A100-80GB training pipeline on Modal (sentiment → paraphrase → sonnet generation)
- Gradient checkpointing for memory-heavy attention variants
- `torch.no_grad()` guards on evaluation paths, because forgetting those once is enough

## Credits

- [GPT-2 (OpenAI)](https://openai.com/research/better-language-models)
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
- [Quora Question Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- [Modal](https://modal.com/)
- [PyTorch](https://pytorch.org/)
