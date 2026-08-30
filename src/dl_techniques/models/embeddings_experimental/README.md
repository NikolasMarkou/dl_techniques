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
| `ascii_convnext_bert` | `ConvNextV1Block` along the sequence axis | no | linear |
| `shared` | the skeleton and the block registry — not an arm | — | — |

Adding an arm is one `BLOCK_REGISTRY` entry plus a thin leaf package. It is
never a second copy of the encoder.

**This file describes the architectures only.** The study that trains and
compares them lives in `src/train/embeddings_experimental/`: its
[`README.md`](../../../train/embeddings_experimental/README.md) explains the
harness and defines every metric, and
[`RESULTS.md`](../../../train/embeddings_experimental/RESULTS.md) carries the
measured outcomes. The parameter counts below stay here because they are a
property of the models rather than a result.

## Measured parameter counts

At `max_position_embeddings=128`, ASCII vocabulary (101 ids):

| Variant | hidden / layers | `ascii_bert` | `ascii_clifford_bert` | `ascii_convnext_bert` |
|---|---|---:|---:|---:|
| tiny | 128 / 4 | 822,656 | 499,072 | 562,048 |
| small | 256 / 6 | 4,797,696 | 3,630,336 | 3,229,440 |
| base | 512 / 8 | 25,337,344 | 23,272,960 | 16,961,024 |

**Equal variant names do not mean equal parameter counts.** The arms are depth-
and width-matched, not parameter-matched, which is why the study reports the
parameter column beside every result.

## Four caveats that decide how the study must be run

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
measures exactly 0.0 on the same comparison. The ConvNeXt arm has the same
hazard for the same reason (a same-padded depthwise convolution), but shows it
at initialization rather than hiding it, because its LayerScale starts at 1.0.

> **The V2 arm is gone, and this hazard is why.** `ascii_convnext_v2_bert` was
> withdrawn from the study on 2026-08-30. GRN reduces over the sequence axis, so
> pad **length** entered every real position -- but only once training moved
> biases off zero, because GRN's score is an L2 sum and exact zeros contribute
> nothing. Measured pad-8 vs pad-12: `convnext` 0.000e+00 both at init and with
> non-zero biases; `convnext_v2` 0.000e+00 at init and **3.215e-03** with
> non-zero biases. A freshly constructed V2 model looked padding-safe and was
> not. See caveat 2 for the token-mixing half of the same defect.

**1b. The arms are chosen to separate effects, not just to add data points.**
The Clifford arm differs from the baseline in two ways at once — convolutional
instead of attentional, *and* a geometric product for channel mixing.
`ascii_convnext_bert` is convolutional without the geometric product, so those
two come apart. Each addition buys one comparison that the previous set could
not make.

A fourth arm, `ascii_convnext_v2_bert`, was added to isolate channel competition
(it differed from the V1 arm by *exactly* Global Response Normalization) and
**withdrawn on 2026-08-30**: GRN turned out to mix tokens globally, so the arm
did not differ from V1 only in channel competition, and the convolutional arms
no longer shared a fixed span. See caveat 2.

**2. Token mixing in both convolutional arms is local — and must stay so.** The geometric product rolls
along the *channel* axis, so all cross-token mixing comes from the two depthwise
convolutions per block. The two arms do **not** share a formula:

```
clifford:  num_layers * 2 * (K - 1) + 1     (two convs per block)
convnext:  num_layers * (K - 1) + 1         (one conv per block)
```

so matching them on `kernel_size` does not match them on span — the Clifford
span is `2 * convnext - 1`. At the Clifford layer's default `K=3` a 4-block
stack sees **17 characters**. Both arms therefore default to `K=7`, and both
warn when the span is shorter than `max_position_embeddings`.

> **Why there is no V2 arm.** `ascii_convnext_v2_bert` was withdrawn on
> 2026-08-30 because these formulas did not bound it. The same GRN that made it
> pad-length-sensitive under caveat 1 also mixes *tokens*: reducing over the
> sequence axis makes every position a function of every other, so the arm had
> no finite receptive field. Measured on trained encoders, perturbing position 0
> and reading `max |delta|` at position `d`:
>
> | arm | span formula | d=25 | d=60 | d=120 |
> |---|---|---:|---:|---:|
> | `ascii_clifford_bert` | 49 | 0.000 | 0.000 | 0.000 |
> | `ascii_convnext_bert` | 25 | 0.000 | 0.000 | 0.000 |
> | `ascii_convnext_v2_bert` | 25 (**did not hold**) | 7.210e-02 | 8.378e-02 | 3.989e-02 |
>
> It moved position 60 by more than the attention arm did. A study whose arms
> are meant to differ only in the sequence-mixing block, with the convolutional
> arms sharing a fixed span, could not contain it. The 28 V2 cells already under
> `results/` are left in place and stay loadable -- `ConvNextEncoderBlock` still
> accepts `version="v2"` for exactly that reason -- but the arm cannot be
> constructed from a study config any more. Pinned by
> `tests/test_models/test_embeddings_shared/test_the_arms_differ_in_reach.py`,
> which now requires EVERY convolutional arm to be exactly inert beyond its span.

**3. The encoder's positional default is not the study's.** `EmbeddingEncoder`
defaults to `position_embedding_type='learned'`; the study overrides it. This is
not cosmetic — a learned table starts at essentially the word table's norm
(0.1985 against 0.1987) and is *abandoned* during training, shrinking to 0.1612
while the word table grows to 0.3283. `ascii_bert` cannot bootstrap
position-dependent attention from that and converges to a bag of characters:
reordering its whole context moves the output 0.83% of activation scale while
replacing the context moves 52.58%.

The three attention-free arms are far less sensitive, because their blocks carry
positional structure in the architecture rather than in the embedding. That
asymmetry is the point: **switching to sinusoidal is worth −1.20 nats to
`ascii_bert` and costs the convolutional arms +0.07 to +0.11.** It also interacts
with `max_seq_length`, and it moves retrieval and language-modelling quality in
opposite directions. Any comparison across these arms must state which setting it
used. The numbers are in
[`RESULTS.md`](../../../train/embeddings_experimental/RESULTS.md).

**4. The blocks are equally regularized only since 2026-08-30.** Until then
`CliffordEncoderBlock` had no dropout at all — `build_clifford_block` declared no
such parameter and `CliffordNetBlock` has none of its own — while the ConvNeXt
blocks carried `dropout_rate` and `TransformerLayer` carried two dropout sites.
The fix lives in this package's wrapper rather than in `CliffordNetBlock`, which
is shared with other packages: dropout is applied to the update before the
external residual add, the same position the ConvNeXt blocks use, so parameter
counts are unchanged.

Repairing it exposed a second asymmetry: `AsciiBert.attention_probs_dropout_rate`
was hard-defaulted to 0.1 and ignored `hidden_dropout_rate`, so the study's
dropout knob only partly controlled that arm. It now follows `hidden_dropout_rate`
unless passed explicitly. Both are pinned by
`tests/test_models/test_embeddings_shared/test_every_arm_is_equally_regularized.py`.
**Every run reported in `RESULTS.md` predates this**, so a clifford cell trained
after the fix is not comparable to one trained before it.

## Usage

```python
from dl_techniques.models.embeddings_experimental.ascii_bert import create_ascii_bert

model = create_ascii_bert("small", pooling_strategy="mean")
model.build((None, 256))
out = model({"input_ids": ids})
# out: last_hidden_state (B, S, H), attention_mask (B, S), pooled_output (B, P)
```

Training and the sweep live in `src/train/embeddings_experimental/`.
