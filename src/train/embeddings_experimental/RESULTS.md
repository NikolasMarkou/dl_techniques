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

### What this does not settle

One seed per configuration at 3000 steps. This identifies a **mechanism**, not an
effect size, and no run here is converged. The completed 28-cell study is
unaffected — but its transformer arm should be read as "a transformer with a
weak positional signal, diluted over 512 positions", not as "a transformer".

Reproduction scripts are throwaway and live outside the repo; the configuration
changes they test are `position_embedding_type="sinusoidal"` and
`max_seq_length`, both already exposed by `EmbeddingEncoder`.
