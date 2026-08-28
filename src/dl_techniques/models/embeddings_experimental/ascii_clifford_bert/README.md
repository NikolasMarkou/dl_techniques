# ascii_clifford_bert

The study's attention-free arm: the same BERT skeleton as `ascii_bert`, with
self-attention replaced by bidirectional sequence-mode `CliffordNetBlock`
mixing — a shifted geometric product along the channel axis plus a depthwise
convolutional context branch. There is no attention anywhere in this arm.

## Variants

Depth- and width-matched to `ascii_bert` so the size axis lines up.

| Variant | hidden | layers | shifts | K | params (`max_position_embeddings=128`) |
|---|---:|---:|---|---:|---:|
| tiny | 128 | 4 | [1,2] | 7 | 499,072 |
| small | 256 | 6 | [1,2,4] | 7 | 3,630,336 |
| base | 512 | 8 | [1,2,4,8] | 7 | 23,272,960 |

Matched depth and width does **not** mean matched parameters — see the counts.

## What is interesting, and what to watch

**Cost is linear in sequence length.** Nothing here builds an `S x S` matrix. At
character granularity, where sequences are ~5x longer than their sub-word
equivalents, that is the point of the arm.

**Token mixing is local, and its span is a design parameter.** The geometric
product mixes *channels*, not positions, so all cross-token mixing comes from
the two stacked depthwise convolutions per block:

```
span = num_layers * 2 * (context_kernel_size - 1) + 1
```

At the layer's default `K=3`, four layers see **17 characters** — a couple of
words. This package therefore defaults to `K=7` and warns when the span is
shorter than `max_position_embeddings`. `use_global_context=True` (a cumulative
mean) has unbounded reach and is the other lever.

**Padding is not neutral.** `supports_masking` is `False`. The wrapper zeroes
masked positions, which bounds the effect but cannot remove it, and
`use_global_context` defaults to `False` because that is the setting whose
padding hazard is boundary-local rather than global. Train stage 1 on packed
sequences; bucket by length wherever padding is unavoidable. The family README
carries the measured numbers.

```python
from dl_techniques.models.embeddings_experimental.ascii_clifford_bert import (
    create_ascii_clifford_bert,
)

model = create_ascii_clifford_bert("small", pooling_strategy="mean")
print(model.receptive_field)  # 73 characters at small/K=7
```

`pretrained=True` raises `NotImplementedError` — no weights are distributed.
