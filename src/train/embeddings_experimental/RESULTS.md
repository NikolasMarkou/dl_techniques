# Embeddings study — results

Measured outcomes of the `embeddings_experimental` study. This file holds the
**numbers a run produced**; `README.md` beside it explains what each metric means
and how the harness works, and
`src/dl_techniques/models/embeddings_experimental/README.md` describes the
architectures themselves.

## TL;DR

At a 3000-step character-level budget, **all three attention-free arms beat the
transformer baseline on every quality metric, with fewer parameters.** The
baseline is not broken — it has learned character frequencies and little context,
which is what a transformer looks like this early at character granularity.

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
