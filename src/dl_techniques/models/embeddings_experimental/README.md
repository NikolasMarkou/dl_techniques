# embeddings_experimental

Text-embedding encoders over a character-level ASCII vocabulary, built to be
compared against each other.

This is a **study**, not a model. Every arm shares one skeleton — the same
embeddings, the same depth/width ladder, the same pooling options, the same
heads — and differs **only** in its sequence-mixing block, so a difference in a
reported metric is attributable to the block rather than to the plumbing.

## Arms

| Package | Block | Attention? | Cost in sequence length |
|---|---|---|---|
| `ascii_bert` | `TransformerLayer` (multi-head self-attention) | yes | quadratic |
| `ascii_clifford_bert` | `CliffordNetBlock`, bidirectional sequence mode | no | linear |
| `shared` | the skeleton and the block registry — not an arm | — | — |

Adding an arm is one `BLOCK_REGISTRY` entry plus a thin leaf package. It is
never a second copy of the encoder.

## Measured parameter counts

At `max_position_embeddings=128`, ASCII vocabulary (101 ids):

| Variant | hidden / layers | `ascii_bert` | `ascii_clifford_bert` |
|---|---|---:|---:|
| tiny | 128 / 4 | 822,656 | 499,072 |
| small | 256 / 6 | 4,797,696 | 3,630,336 |
| base | 512 / 8 | 25,337,344 | 23,272,960 |

**Equal variant names do not mean equal parameter counts.** The arms are depth-
and width-matched, not parameter-matched, which is why the study reports the
parameter column beside every result.

## Two caveats that decide how the study must be run

**1. Padding is not neutral for the Clifford arm.** `CliffordNetBlock` sets
`supports_masking = False`. Three separate facts, each measured and each pinned
by a test — conflating them produces a guard that reports the opposite of the
truth:

- The hazard is the **presence** of padding, not its length. With the global
  branch off, pad-8 vs pad-12 is **exactly 0.0**. A guard comparing two padded
  lengths measures zero and wrongly concludes the arm is padding-safe.
- Only `use_global_context=True` makes pad **length** matter (8.160e-03 at
  gamma 1.0), because its cumulative mean runs over the whole padded sequence.
  Hence the default is `False`.
- **LayerScale hides all of it at initialization**: 1.490e-07 at the default
  `layer_scale_init=1e-5` versus 1.082e-02 at gamma 1.0. Gamma is *learned*, so
  a smoke test at init reports a padding-safe arm that becomes
  padding-sensitive during training.

The study's answer is to remove padding rather than fake a mask: stage 1 trains
on **packed** fixed-length sequences. The transformer arm honours its mask and
measures exactly 0.0 on the same comparison.

**2. Token mixing in the Clifford arm is local.** The geometric product rolls
along the *channel* axis, so all cross-token mixing comes from the two depthwise
convolutions per block: `num_layers * 2 * (K - 1) + 1` characters. At the
layer's default `K=3` a 4-block stack sees **17 characters**. This arm therefore
defaults to `K=7`, not 3, and warns when the span is shorter than
`max_position_embeddings`.

## Usage

```python
from dl_techniques.models.embeddings_experimental.ascii_bert import create_ascii_bert

model = create_ascii_bert("small", pooling_strategy="mean")
model.build((None, 256))
out = model({"input_ids": ids})
# out: last_hidden_state (B, S, H), attention_mask (B, S), pooled_output (B, P)
```

Training and the sweep live in `src/train/embeddings_experimental/`.
