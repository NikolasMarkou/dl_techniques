# ascii_convnext_v2_bert

The same block as [`ascii_convnext_bert`](../ascii_convnext_bert/) with **one**
addition: Global Response Normalization after the activation. That single
difference is the point — paired against the V1 arm, this isolates what GRN
does with depth, width, kernel, expansion and everything else held fixed.

GRN scores each channel by its L2 magnitude over the sequence, divides by the
mean score across channels, and reweights. It is channel competition with no
parameters beyond a `gamma` and a `beta`, and it is the whole of what separates
ConvNeXt V2 from V1.

## Variants

| Variant | hidden | layers | K | params | vs V1 |
|---|---:|---:|---:|---:|---:|
| tiny | 128 | 4 | 7 | 566,144 | +4,096 |
| small | 256 | 6 | 7 | 3,241,728 | +12,288 |
| base | 512 | 8 | 7 | 16,993,792 | +32,768 |

The gap is *exactly* the GRN parameters — asserted by a test, because a larger
gap would mean something else changed too and the pair would stop being a
controlled comparison.

## The one thing to know before trusting a number from this arm

**GRN makes it sensitive to padding LENGTH, and only after training.**

GRN reduces over the sequence, so whatever sits in the padded region enters
every real position's normalizer. The wrapper zeroes masked positions, and GRN's
score is an L2 **sum** — `sqrt(sum(x**2) + eps)` — so exact zeros contribute
nothing. At initialization every bias is zero, so the padded region stays
exactly zero through `conv_1`, the norm, `conv_2` and the activation (each
verified), and pad length is exactly inert.

Training moves the biases off zero. Measured on a 6-token prefix,
`hidden_size=32`, 2 blocks, `K=3`, pad-8 against pad-12:

| arm | at initialization | with non-zero biases |
|---|---:|---:|
| `convnext` (V1) | 0.000e+00 | 0.000e+00 |
| `convnext_v2` | 0.000e+00 | **3.215e-03** |

So **a probe of a freshly constructed model reports this arm as padding-safe,
and is wrong about the model that is actually trained.** Batch composition then
changes every sequence's embedding. This is the same shape of trap as LayerScale
hiding the Clifford arm's boundary effect, and it has the same answer: stage 1
trains on packed sequences carrying no padding at all, and anywhere padding is
unavoidable, bucket by length.

The V1 arm is the control — inert either way, which is what attributes the
effect to GRN rather than to the convolution.

## Everything else

Shared with the V1 arm: linear cost in sequence length, a local receptive field
of `num_layers * (K - 1) + 1`, an external residual, and the boundary-effect
half of the padding hazard. See that arm's README and the family README.

```python
from dl_techniques.models.embeddings_experimental.ascii_convnext_v2_bert import (
    create_ascii_convnext_v2_bert,
)

model = create_ascii_convnext_v2_bert("small", pooling_strategy="mean")
```

`pretrained=True` raises `NotImplementedError` — no weights are distributed.
