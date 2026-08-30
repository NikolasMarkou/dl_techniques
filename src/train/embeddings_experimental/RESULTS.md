# Embeddings study — results

Measured outcomes of the `embeddings_experimental` study. This file holds the
**numbers a run produced**; `README.md` beside it explains what each metric means
and how the harness works, and
`src/dl_techniques/models/embeddings_experimental/README.md` describes the
architectures themselves.

## TL;DR

At a 3000-step character-level budget, **all three attention-free arms beat the
transformer baseline on every quality metric, with fewer parameters.** The
baseline has learned character frequencies and almost no context.

> **Correction, 2026-08-29.** This section previously read "which is what a
> transformer looks like this early at character granularity". That was wrong,
> and the investigation below says why: the baseline is not early, it is
> **converged to a zero-context solution and has stopped moving** (−0.008 nats
> per 1000 steps). Two configuration choices cause it, and both are fixable. See
> *Why the transformer arm was stuck*.

**No comparison here is statistically significant, and none is claimed to be.**
This was a one-seed pipeline check; every paired comparison is correctly labelled
`UNDERPOWERED`.

## Run 1 — smoke, 2026-08-29

Four arms, `tiny` variant, `mean` pooling, **one seed**, 3000 MLM steps + 1000
contrastive steps, `max_seq_length=256`, batch 32/64, on one RTX 4090.
Wall clock: **20 minutes** for all four cells including evaluation.

```bash
python -m train.embeddings_experimental.sweep \
    --variants tiny --pooling mean --seeds 0 --gpu 0 \
    --sweep-root results/embeddings_smoke \
    --trainer-arg=--steps-per-epoch --trainer-arg=3000 \
    --trainer-arg=--contrastive-steps-per-epoch --trainer-arg=1000
```

### Headline

| | bert | clifford | convnext | convnext_v2 |
|---|---:|---:|---:|---:|
| parameters | 839,040 | 515,456 | 578,432 | 582,528 |
| **SQuAD MRR@10** *(primary)* | 0.0246 | 0.0538 | 0.0604 | **0.0647** |
| SQuAD recall@1 | 1.15% | 3.45% | 3.85% | **4.25%** |
| SQuAD recall@10 | 6.40% | 11.05% | 11.80% | **12.55%** |
| SST-2 probe accuracy | 58.94% | **63.99%** | 62.50% | 63.65% |
| MLM val loss *(nats)* | 2.838 | 1.324 | 1.179 | **1.153** |
| MLM val accuracy | 23.28% | 61.68% | 66.04% | **66.67%** |

Chance levels: SQuAD recall@1 **0.048%** (1 of 2,067 paragraphs), recall@10
**0.484%**, SST-2 **50.92%** (majority class). So retrieval runs 24-88x chance —
real signal, but still far from useful in absolute terms.

### Diagnostics

No verdicts attach to these. They explain *why* the headline moved.

| | bert | clifford | convnext | convnext_v2 |
|---|---:|---:|---:|---:|
| anisotropy | 0.429 | 0.616 | 0.134 | **0.132** |
| effective rank *(of 128)* | 35.5 | **78.6** | 66.2 | 70.8 |
| alignment *(lower better)* | 1.065 | **0.872** | 1.446 | 1.445 |
| uniformity *(lower better)* | −2.029 | −1.420 | −3.116 | **−3.161** |
| median rank *(of 2,067)* | 457 | 322 | 285 | **274** |

## Two things worth reading twice

**The baseline has learned letter frequencies and little else.** A uniform guess
over the 101-id vocabulary costs `ln(101)` = **4.615 nats**; knowing only this
corpus's character frequencies, with no context at all, costs **3.130 nats**
(measured over 1.03M characters). `ascii_bert` sits at **2.838** — just 0.29 nats
below the context-free baseline — while the convolutional arms reach 1.15-1.32.

The mechanism is plausible rather than mysterious: at character granularity a
depthwise convolution has exactly the right prior for "predict this character
from its neighbours", while attention has to learn that from scratch. **This is a
statement about training speed at this budget, not about final quality.** A
longer run could close or reverse it.

**Alignment and uniformity show the trade-off, not a winner.** `clifford` has the
*best* alignment (0.872) and the *worst* uniformity (−1.420); `convnext_v2` is
the mirror image (1.445 / −3.161). Neither is simply better — they sit at
different points on the same curve, which is also why `clifford` has the highest
anisotropy of the four.

## What this run cannot support

- **Any significance claim.** One seed. The primary family (3 non-baseline arms,
  Holm-corrected) has a floor of **7 seeds**; the secondary family (18 tests, BH)
  needs **10**. All 21 comparisons are labelled `UNDERPOWERED`, which means the
  question was not askable, not that the arms are equivalent.
- **Any absolute claim.** SQuAD contexts *are* Wikipedia paragraphs and the MLM
  corpus is Wikipedia, so there is genuine train/eval overlap. Every arm shares
  it, so the *relative* ordering survives; the absolute numbers do not.
- **Any claim about full-passage retrieval.** Contexts average 780 characters
  against a 256-character window, so this is prefix matching.

## Speed, measured

Per-step cost after subtracting a ~54 s fixed overhead (Wikipedia load, model
build, checkpoint save), at `tiny` / 256 characters / batch 32:

| arm | ms/step |
|---|---:|
| convnext_v2 | **11** |
| ascii_bert | 31 |
| clifford | 65 |

The arms differ, so the run is **not** bottlenecked on the Python data
generator. Attention pays its `O(S²)` cost at 256 characters; a single depthwise
convolution is the cheapest thing here.

## Next run

7 seeds x 4 arms at the same shape clears the primary family's floor and would
let the ordering above be tested rather than merely observed. At 20 minutes per
seed-sweep of four cells, that is roughly 2.5 hours.


## Why the transformer arm was stuck — 2026-08-29

The baseline's 512-context run moved validation loss **−0.0041 nats** against its
256-context run despite doubling both context and steps. That is a plateau, not
slow progress, so it was worth a proper investigation.

### The arm had converged to a zero-context solution

Recovering the instantaneous per-batch loss (Keras logs the epoch running mean,
which hides this) shows the baseline reaching 2.878 by step ~400 and then going
flat at **−0.0081 nats per 1000 steps**. Extrapolated, it needs ~200,000 further
steps to reach where the convolutional arms already are.

Where it stops is not arbitrary. MLM corrupts 15% of positions: 80% become
`[MASK]`, 10% become a random token, and 10% are **left unchanged** — and those
last are free to any model that can copy its input. So a model that predicts the
corpus unigram everywhere and copies the unchanged tokens scores:

| | predicted | measured |
|---|---:|---:|
| loss | `0.9 x 3.1135` = **2.8022** | 2.8307–2.8331 |
| accuracy | `0.9 x 0.1532 + 0.1` = **0.2379** | 0.2321–0.2322 |

(3.1135 nats is the unigram entropy of the exact packed stream, top-id frequency
0.1532, measured over 3,121,152 ids.) The arm lands just short of that ideal. It
is not using context at all.

**Confirmed directly.** Perturbing the trained model's position-0 hidden state:

| context change | mean \|delta\| | as % of activation scale |
|---|---:|---:|
| **reordered** (identical multiset) | 0.006561 | **0.83%** |
| **replaced** (different multiset) | 0.416745 | **52.58%** |

A 63x ratio. The model reads *which* characters are in the window and not
*where* they are — it is a bag of characters. Consistently, its position
embedding table **shrank** during training (mean row norm 0.1987 → 0.1612) while
the word table **grew** (0.1987 → 0.3283).

### Four hypotheses that turned out to be wrong

Each was tested at the same shape (tiny, 256 ctx, 3000 steps, seed 0):

| change | val loss | verdict |
|---|---:|---|
| baseline, replicating the study's 2.8382 | 2.8331 | gate passed |
| pre-LN instead of post-LN | 2.8321 | **inert** |
| warmup 0.06 → 0.10 (the house value) | 2.8325 | **inert** |
| pre-LN *and* warmup 0.10 | 2.8327 | **inert** |
| dropout 0.1 → 0.0 | 2.8307 | **inert** |
| all three together | 2.8312 | **inert** |
| weight decay 0.01 → 0.0 | 2.8331 | **inert** |

Seven configurations spanning **0.0024 nats**. Also refuted earlier by direct
measurement: the block is not missing a residual (`TransformerLayer` adds its own
two), and attention is neither dead nor mask-inverted (it moves information
2.106e-03 across 25 positions; the convolutional arms correctly move exactly
0.000 beyond their spans).

### Two things that do work, and they compound

| | learned positions | sinusoidal positions |
|---|---:|---:|
| **256 context** | 2.8331 | **1.9832** |
| **64 context** | **1.8542** | **1.6641** |

Tokens per step are held constant (32x256 = 128x64 = 8192), so these are matched
budgets.

**Context length is the larger effect.** Shortening context from 256 to 64
improves the learned-position model by **0.98 nats**. A model given *less*
information doing dramatically *better* is direct evidence of attention dilution:
with a near-uniform softmax over 256 positions each neighbour contributes ~1/256
of the attended value, so every position receives roughly the sequence mean.

**The positional signal is the second effect.** Fixed sinusoidal encodings buy
**0.85 nats** at 256 context and 0.19 at 64. The learned table is initialized at
`initializer_range=0.02` — mean row norm ~0.2 against sinusoidal's ~8 — and never
grows, so the model cannot bootstrap position-dependent attention. Disabling
weight decay does not rescue it, so this is the initial scale and structure of
the signal, not its decay.

Convergence slopes over the last 1500 steps, which separate "stuck" from "slow":

| run | slope, nats/1000 steps |
|---|---:|
| baseline (learned, 256) | **−0.0081** |
| sinusoidal, 256 | −0.1827 |
| learned, 64 | −0.1474 |
| sinusoidal, 64 | −0.0736 |
| clifford control, 256 | −0.0676 |

The baseline is the only run that has stopped. Every fix is still descending 9–22x
faster than it, so none of the numbers above is its converged value.

### A separate defect this exposed: the arms are not equally regularized

Independent of the cause, the comparison carries a confound:

| arm | dropout inside the block |
|---|---|
| `ascii_bert` | attention-probability 0.1 **and** FFN 0.1 |
| `ascii_convnext_bert` / `_v2` | 0.1 |
| `ascii_clifford_bert` | **none** |

`AsciiCliffordBert` never puts `dropout_rate` in its `block_config` and
`build_clifford_block` has no such parameter, so that arm trains unregularized.
Measured above as **inert** as a cause (2.8307 at dropout 0.0), but it is still an
uncontrolled difference between arms and should be equalized before the four-arm
comparison is restated.

### The fix is not a free win, and the three conv arms pay an identical price

Switching `position_embedding_type` to `'sinusoidal'`, measured at 256 context /
3000 steps / seed 0, one A/B per arm:

| arm | learned | sinusoidal | delta |
|---|---:|---:|---:|
| `ascii_bert` | 2.8334 | **1.9832** | **−0.8502** |
| `ascii_clifford_bert` | 1.3314 | 1.4732 | +0.1417 |
| `ascii_convnext_bert` | 1.1882 | 1.3248 | +0.1366 |
| `ascii_convnext_v2_bert` | 1.1633 | 1.2989 | +0.1356 |

The three convolutional arms lose **+0.136 to +0.142** — a spread of 0.006 nats
across three different blocks. That tightness says one shared mechanism, not
noise: a sinusoidal table has mean row norm ~9.3 against the learned table's
~0.2, so after the embedding LayerNorm it dominates, and the token signal
measured through the same probe drops from 11.07 to 0.42. A block that already
has positional structure gains nothing from the position signal and pays for the
compressed token signal.

**The study keeps one setting across all arms anyway.** Its design is one encoder
with the block swapped; letting each arm pick its own position encoding would
confound block with position type and stop it answering the question it asks.
This cost is reported, not tuned away.

Note what this does and does not change. The ordering is unaffected — the
transformer is still last (1.98 against 1.30–1.47) — but the gap narrows from
**1.67 nats to 0.51**. Roughly two thirds of the headline result in "Run 1" was
the baseline's broken positional signal rather than the blocks.

### What this does not settle

One seed per configuration at 3000 steps. This identifies a **mechanism**, not an
effect size, and no run here is converged. The completed 28-cell study is
unaffected — but its transformer arm should be read as "a transformer with a
weak positional signal, diluted over 512 positions", not as "a transformer".

Reproduction scripts are throwaway and live outside the repo; the configuration
changes they test are `position_embedding_type="sinusoidal"` and
`max_seq_length`, both already exposed by `EmbeddingEncoder`.


## Run 2 — 512 context, sinusoidal positions, 28 cells, 2026-08-30

A full rerun of Run 1's grid with `position_embedding_type='sinusoidal'`, 7
seeds x 4 arms. Because Run 1 predates that field, **it is the learned-position
arm of the same factorial** — every number below is paired cell-by-cell on
(model, variant, pooling, seed), not compared as two independent means.

### MLM: the fix is worth 1.20 nats to the transformer and costs the others ~0.08

| arm | learned | sinusoidal | paired delta (n=7) |
|---|---:|---:|---:|
| `ascii_bert` | 2.8248 ±0.008 | **1.6218** ±0.025 | **−1.2030** ±0.024 |
| `ascii_clifford_bert` | 1.0693 ±0.019 | 1.1746 ±0.009 | +0.1053 ±0.019 |
| `ascii_convnext_bert` | 0.9213 ±0.019 | 0.9933 ±0.009 | +0.0719 ±0.021 |
| `ascii_convnext_v2_bert` | **0.8973** ±0.016 | 0.9756 ±0.010 | +0.0782 ±0.019 |

Every arm moves the same direction on all 7 of its 7 seeds, and the transformer's
effect is ~50x its own seed spread.

**The block ordering is unchanged** — `convnext_v2 < convnext < clifford < bert`
under both settings — so Run 1's central conclusion survives. What changes is the
size of the claim: the transformer's deficit against the best arm falls from
**1.9275 to 0.6463 nats, closing 66% of it.** Two thirds of Run 1's headline was
the baseline's broken positional signal, not the blocks.

### Retrieval: every arm gets dramatically worse, and it is not anisotropy

| arm | R@1 learned | R@1 sinusoidal | ratio |
|---|---:|---:|---:|
| `ascii_bert` | 0.0129 | 0.0019 | **x0.14** |
| `ascii_clifford_bert` | 0.0587 | 0.0061 | **x0.10** |
| `ascii_convnext_bert` | 0.0594 | 0.0170 | x0.29 |
| `ascii_convnext_v2_bert` | 0.0681 | 0.0507 | x0.75 |

So the setting that makes the language model far better makes the **embeddings**
— this study's actual deliverable — 1.3x to 10x worse. `sst2_probe_accuracy`
moves by at most 2% in either direction for every arm, so content remains
linearly decodable; it is specifically cosine retrieval that collapses.

**The obvious explanation is wrong, and was refuted three separate ways.** A
sinusoidal table has row norm ~9.3 against a learned table's ~0.2 and is
identical for every sequence of a given length, so mean-pooling should turn it
into a large shared offset — which would inflate anisotropy and break cosine
similarity without destroying information. Anisotropy did rise (`ascii_bert`
0.298 -> 0.745, `cos_to_centroid` 0.705 -> 0.931). But:

1. **Centering does not recover retrieval.** Subtracting the pool mean drives
   anisotropy to −0.0002 and `cos_to_centroid` to 0.0001 — a complete fix of the
   offset — and R@1 stays at 0.0040, R@10 at 0.0120, unchanged to four decimals.
2. **Clifford collapsed 10x with essentially no anisotropy change** (0.494 ->
   0.465, slightly *better*) at seed 0.
3. **Across arms the two quantities are anti-correlated.** Clifford has the
   *smallest* anisotropy rise (x1.26) and the *largest* retrieval loss (x0.10);
   convnext_v2 has a larger anisotropy rise (x1.45) and by far the smallest loss
   (x0.75).

| arm | anisotropy ratio | R@1 ratio |
|---|---:|---:|
| `ascii_bert` | x2.50 | x0.14 |
| `ascii_clifford_bert` | **x1.26** | **x0.10** |
| `ascii_convnext_bert` | x1.63 | x0.29 |
| `ascii_convnext_v2_bert` | **x1.45** | **x0.75** |

The information is genuinely absent from the pooled vector, not merely hidden
behind a shift. **Why** is not established — a plausible story is that a model
handed a strong positional signal solves MLM with position-indexed local
structure instead of building content summaries, and mean-pooling then has less
content to average. That is untested speculation; the effect is measured, the
cause is not.

### What to take from this

`position_embedding_type` is a genuine trade-off axis for this study, not a bug
with a fix. It buys a large amount of the pretraining objective and sells a large
amount of the downstream embedding quality, and the exchange rate differs sharply
by block (convnext_v2 barely pays; clifford pays most). Whichever default the
study settles on, **it must be reported as a choice with a cost**, and the arm
comparison must be read at a fixed setting.

Absolute caution: every retrieval number here is near chance (0.00048). The best
cell, `convnext_v2` with learned positions at 0.0681, is ~141x chance and still
under 7%. These are 6000-step `tiny` models.


## The largest retrieval win here is free, and the study was not measuring it

Chasing the *cause* of the sinusoidal retrieval collapse turned up something
bigger than the question. Two hypotheses died first, and both deaths matter:

1. **Shared offset.** Refuted by centering (above): anisotropy goes to −0.0002
   and R@1 does not move.
2. **Variance drowning.** SST-2 (a linear probe, free to reweight dimensions) is
   unchanged while cosine similarity (which weights dimensions equally)
   collapses — the signature of content surviving in low-variance directions.
   Refuted **for the sinusoidal models**: ZCA whitening makes them *worse*
   (`ascii_bert` 0.0040 -> 0.0013). The content really is gone there.

But hypothesis 2 is **correct for the learned-position convolutional arms**, and
the effect is large. Full 2000-query protocol, 3 seeds per cell:

| arm | learned raw | learned + ZCA | gain | sinusoidal raw | sinusoidal + ZCA |
|---|---:|---:|---:|---:|---:|
| `ascii_bert` | 0.0102 | 0.0102 | **1.00x** | 0.0040 | 0.0013 |
| `ascii_clifford_bert` | 0.0598 | **0.2178** | 3.64x | 0.0435 | 0.0212 |
| `ascii_convnext_bert` | 0.0590 | **0.2075** | 3.52x | 0.0280 | 0.0047 |
| `ascii_convnext_v2_bert` | 0.0647 | **0.2110** | 3.26x | 0.0535 | 0.0297 |

**9 of 9 learned conv cells gain (3.09x–3.97x, no exceptions); 12 of 12
sinusoidal cells lose.** `ascii_bert` with learned positions gains exactly
nothing — which is its own confirmation, since that model is a bag of characters
with no drowned content structure to recover.

**It is not transductive.** Fitting the whitening on 3000 SST-2 sentences —
movie reviews, a different domain from the Wikipedia paragraphs being retrieved —
and applying it unchanged keeps ~90% of the gain:

| arm | raw | ZCA fit on the pool | ZCA fit on SST-2 |
|---|---:|---:|---:|
| `ascii_clifford_bert` | 0.0575 | 0.2145 | **0.1865** |
| `ascii_convnext_bert` | 0.0595 | 0.2060 | **0.1925** |
| `ascii_convnext_v2_bert` | 0.0620 | 0.2085 | **0.1940** |

So the transform is a property of the embedding space, not of the retrieval
corpus. It can be fitted once on arbitrary text and shipped with the encoder; no
index-time corpus access is required and no retraining is involved.

This is the known BERT-whitening result (Su et al., 2021) reproducing on
character-level encoders, which is a reason to trust it rather than a reason to
discount it.

**Consequences for how this study reports retrieval.** Every `squad_*` number in
Run 1 and Run 2 is a *raw* cosine number, and raw cosine understates these
encoders by ~3.3x. The study's best reported cell (`convnext_v2`, learned,
0.0681) is really ~0.21 under a transform that costs nothing. Any future
comparison should quote both, because the ordering is not preserved: raw ranks
`convnext_v2 > convnext ~ clifford`, whitened ranks them within 0.01 of each
other, and the transform's *benefit* is what separates the arms from the
transformer baseline.

References:
    - Su et al., 2021. Whitening Sentence Representations for Better Semantics
      and Faster Retrieval. (https://arxiv.org/abs/2103.15316)


## Run 3 — 64 context, sinusoidal, 28 cells, 2026-08-30

Same grid at `max_seq_length=64` with `mlm_batch_size=256`, so tokens per step
are **16384, identical to Run 2's 32 x 512**. Matching tokens is what separates
"shorter context helps" from "less training hurts"; the price is that batch size
is no longer constant across runs, which is the lesser confound.

### Context length is an interaction, not a main effect

Paired 512 -> 64 at sinusoidal, n=7 per arm:

| arm | delta | |
|---|---:|---|
| `ascii_bert` | **−0.3220** ±0.024 | helped, ~13x its seed spread |
| `ascii_clifford_bert` | +0.0170 ±0.009 | mildly hurt |
| `ascii_convnext_bert` | +0.0314 ±0.004 | mildly hurt |
| `ascii_convnext_v2_bert` | +0.0306 ±0.004 | mildly hurt |

Shorter context helps **only** the transformer, and mildly hurts all three
convolutional arms — the dilution prediction, confirmed on 4/4 arms. A
near-uniform softmax over `S` positions gives each neighbour ~`1/S` of the
attended value, so more positions means more dilution; a convolution has a fixed
span, so extra context is a small amount of extra material to model.

### The cumulative effect on the baseline, and on the study's claim

| `ascii_bert` | val loss |
|---|---:|
| 512, learned — as originally reported | 2.8248 |
| 512, sinusoidal | 1.6218 (−1.20) |
| 64, sinusoidal | **1.2999** (−0.32 more) |

**−1.52 nats from two configuration changes and zero architecture changes.** The
transformer's deficit against the best convolutional arm falls from **1.9275 to
0.2937 — 85% of it closed.**

The convolutional arms still win, on every configuration tested, and that
conclusion is robust. But "attention-free blocks beat the transformer by 1.9
nats" was substantially a statement about a misconfigured baseline. The
defensible version is ~0.3 nats.

### The sinusoidal retrieval penalty is a LONG-CONTEXT effect

Run 2 reported that sinusoidal positions cost 1.3x-10x retrieval. That is not a
property of sinusoidal encodings — it is a property of sinusoidal encodings **at
512 context**:

| arm, SQuAD R@1 | 512 learned | 512 sinusoidal | 64 sinusoidal |
|---|---:|---:|---:|
| `ascii_bert` | 0.0129 | 0.0019 | **0.0261** |
| `ascii_clifford_bert` | 0.0587 | 0.0061 | 0.0531 |
| `ascii_convnext_bert` | 0.0594 | 0.0170 | 0.0511 |
| `ascii_convnext_v2_bert` | 0.0681 | 0.0507 | 0.0575 |

At 64 the penalty nearly vanishes, and `ascii_bert` reaches its best retrieval of
any configuration measured — **double** its original. `sst2_probe_accuracy` stays
within 0.03 everywhere, as it has throughout.

**So the two families want opposite configurations.** The transformer is best at
64 + sinusoidal on *both* endpoints simultaneously (MLM 1.2999, R@1 0.0261). The
convolutional arms are best at 512 + learned on both (MLM 0.897-1.069, R@1
0.0587-0.0681). There is no single setting that is best for every arm, which
means a fixed-setting four-arm comparison necessarily handicaps someone — and
which setting is chosen decides part of the answer.

This does not undo Run 2's finding that the mechanism is not anisotropy; centering
and whitening refuted that independently of context length. It narrows *when* the
penalty applies.
