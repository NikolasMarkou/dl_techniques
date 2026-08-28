# ascii_convnext_bert

The study's convolutional arm: the same BERT skeleton as `ascii_bert`, with
self-attention replaced by a ConvNeXt V1 block applied along the sequence axis —
depthwise convolution, normalization, pointwise expansion by 4, activation,
pointwise contraction, LayerScale.

## Why this arm exists

It is the control that makes the other comparison interpretable. The Clifford
arm differs from the transformer baseline in **two** ways at once: it is
convolutional rather than attentional, **and** it mixes channels through a
geometric product. This arm is convolutional *without* the geometric product, so
the three together separate those two effects. Without it, any
Clifford-vs-transformer result is confounded.

## Variants

Depth- and width-matched to the other two arms.

| Variant | hidden | layers | K | span | params (`max_position_embeddings=128`) |
|---|---:|---:|---:|---:|---:|
| tiny | 128 | 4 | 7 | 25 | 562,048 |
| small | 256 | 6 | 7 | 37 | 3,229,440 |
| base | 512 | 8 | 7 | 57 | 16,961,024 |

All three arms at `small`: transformer 4,797,696 · Clifford 3,630,336 ·
ConvNeXt 3,229,440. **Matched depth and width does not mean matched parameters.**

The FFN width is not a knob: `ConvNextV1Block` fixes the expansion at 4x, which
happens to match the baseline arm's intermediate size.

## Two things that differ from the Clifford arm

**Its receptive field is half.** A ConvNeXt block applies a *single* depthwise
convolution where `CliffordNetBlock` applies two, so at equal depth and kernel:

```
convnext:  num_layers * (K - 1) + 1
clifford:  num_layers * 2 * (K - 1) + 1     = 2 * convnext - 1
```

Matching the two arms on `kernel_size` does **not** match them on span. Use
`conv_receptive_field` for this arm — the two formulas are separate functions
rather than one with a flag, because applying the wrong factor shows up only as
a mediocre metric.

**LayerScale starts at 1.0**, not the Clifford block's `1e-5`, so this block
contributes at full magnitude from the first step rather than easing in. That
also means the padding hazard is visible here at initialization, where the
Clifford arm's is damped five orders below float32 noise.

## Shared caveats

The residual is **external** — `ConvNextV1Block.call` returns the update only —
and padding is **not neutral**, because a same-padded depthwise convolution pulls
zero padding into the receptive field of real positions. Both are handled by the
shared wrapper and by training stage 1 on packed sequences; see the family README.

```python
from dl_techniques.models.embeddings_experimental.ascii_convnext_bert import (
    create_ascii_convnext_bert,
)

model = create_ascii_convnext_bert("small", pooling_strategy="mean")
print(model.receptive_field)  # 37
```

`pretrained=True` raises `NotImplementedError` — no weights are distributed.
