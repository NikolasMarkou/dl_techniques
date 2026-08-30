# Embeddings study — results

Measured outcomes of the `embeddings_experimental` study. This file holds **the
numbers**. [`README.md`](README.md) beside it explains the harness and defines
every metric; [`models/embeddings_experimental/README.md`](../../dl_techniques/models/embeddings_experimental/README.md)
describes the architectures.

Nothing here is pushed as settled that has not been measured, and where a claim
in an earlier version of this file was later falsified, the correction is stated
by name rather than quietly edited out.

---

## TL;DR

**1. The convolutional arms beat the transformer in all 16 cells of the
context x position factorial**, and at every arm's own best setting. That is the
study's central result and it is robust.

**2. Most of the original margin was configuration, not architecture.** The
transformer's deficit against the best arm was first reported as **1.9275 nats**.
Repair its positional encoding **or** shorten its context — either alone
recovers most of it — and with both it is **0.2937**. Roughly **85% of the
headline gap was a misconfigured baseline.** The defensible claim is ~0.3 nats,
not ~1.9.

**3. The two model families want opposite settings, so no fixed setting is
neutral.** The transformer's best cell is 64 + sinusoidal, on *both* endpoints at
once; all three convolutional arms peak at 512 + learned, on both. Any
fixed-setting four-arm comparison handicaps someone, and the choice of setting
decides part of the answer. The strong two-way interaction behind this is
**specific to the attention arm** — the convolutional arms show only a small
same-signed penalty from each setting.

**4. Every raw retrieval number below understates these encoders by ~3.3x.** ZCA
whitening lifts the learned-position convolutional arms from R@1 ~0.060 to ~0.21,
on 9 of 9 cells, with no retraining.

---

## The runs

| | context | positions | cells | status |
|---|---|---|---:|---|
| Run 1 | 256 | learned | 4 | 1-seed smoke, **superseded** |
| Run 2a | 512 | learned | 28 | complete (the original full study) |
| Run 2b | 512 | sinusoidal | 28 | complete |
| Run 3 | 64 | sinusoidal | 28 | complete |
| Run 4 | 64 | learned | 28 | complete — closes the 2x2 |

Runs 2-4 are 4 arms x 7 seeds at `tiny`/`mean`, 6000 MLM + 2000 contrastive
steps, `max_train_samples=60000`, and are paired cell-by-cell on
`(model, variant, pooling, seed)` — never compared as two independent means.

Run 3 uses `mlm_batch_size=256` so tokens per step are **16384, identical to
Run 2's 32 x 512**. Matching tokens is what separates "shorter context helps"
from "less training hurts". The price is that batch size is not constant across
runs; that is the lesser confound and it is deliberate.

```bash
python -m train.embeddings_experimental.sweep \
    --variants tiny --pooling mean --seeds 0 1 2 3 4 5 6 --gpu 0 \
    --sweep-root results/embeddings_study_64_sinusoidal \
    --trainer-arg=--max-seq-length=64 --trainer-arg=--mlm-batch-size=256 \
    --trainer-arg=--steps-per-epoch=6000 \
    --trainer-arg=--contrastive-steps-per-epoch=2000 \
    --trainer-arg=--max-train-samples=60000 \
    --trainer-arg=--position-embedding-type=sinusoidal
```

---

## MLM validation loss (nats), mean ± sd over 7 seeds

| arm | 512 / learned | 512 / sinusoidal | 64 / sinusoidal | 64 / learned |
|---|---:|---:|---:|---:|
| `ascii_bert` | 2.8248 ±0.008 | 1.6218 ±0.025 | **1.2999** ±0.012 | 1.4285 ±0.055 |
| `ascii_clifford_bert` | **1.0693** ±0.019 | 1.1746 ±0.009 | 1.1916 ±0.007 | 1.1192 ±0.006 |
| `ascii_convnext_bert` | **0.9213** ±0.019 | 0.9933 ±0.009 | 1.0246 ±0.009 | 0.9750 ±0.008 |
| `ascii_convnext_v2_bert` | **0.8973** ±0.016 | 0.9756 ±0.010 | 1.0061 ±0.008 | 0.9549 ±0.008 |

All 16 cells are n=7. **Bold marks each arm's best configuration** — and the
transformer's is a different cell from every other arm's.

Reference points: a uniform guess over the 101-id vocabulary costs `ln(101)` =
**4.615 nats**; predicting from character frequencies alone, with no context,
costs **3.1135 nats** (measured on the exact packed stream, 3,121,152 ids). The
block ordering — `convnext_v2 < convnext < clifford < bert` — is **identical
under all three configurations.**

### The two settings substitute for each other — for the transformer only

The completed 2x2, n=7 in every one of its four cells:

| `ascii_bert` | learned | sinusoidal |
|---|---:|---:|
| **512 context** | 2.8248 ±0.008 | 1.6218 ±0.025 |
| **64 context** | **1.4285** ±0.055 | **1.2999** ±0.012 |

| effect | applied alone | applied second | attenuation |
|---|---:|---:|---:|
| positions, learned → sinusoidal | **−1.2030** *(at 512)* | −0.1287 *(at 64)* | **9.3x** |
| context, 512 → 64 | **−1.3963** *(at learned)* | −0.3220 *(at sinusoidal)* | **4.3x** |

Each change is worth ~1.2–1.4 nats on its own and ~0.13–0.32 once the other is in
place. **Either alone recovers most of the collapse**, and 64 + learned (1.4285)
is slightly better than 512 + sinusoidal (1.6218), so shortening the context is
marginally the stronger single fix.

> **Correction, 2026-08-30.** An earlier version of this section was titled *"The
> two effects compound"* and reported the path 2.8248 → 1.6218 → 1.2999 as
> "−1.52 nats from two configuration changes". The arithmetic was right and the
> framing was wrong: it named the positional encoding as the defect and context
> length as a secondary interaction, purely because they were found in that
> order. **Neither is the cause.** The failure requires long context *and* a weak
> positional signal together.

**The strong interaction is specific to the attention arm.** Run 4 makes this
checkable — the same two effects, measured at both levels of the other, for all
four arms:

| arm | positions @512 | positions @64 | context @learned | context @sinusoidal |
|---|---:|---:|---:|---:|
| `ascii_bert` | **−1.2030** | −0.1287 | **−1.3963** | −0.3220 |
| `ascii_clifford_bert` | +0.1053 | +0.0725 | +0.0499 | +0.0170 |
| `ascii_convnext_bert` | +0.0719 | +0.0496 | +0.0536 | +0.0314 |
| `ascii_convnext_v2_bert` | +0.0782 | +0.0512 | +0.0576 | +0.0306 |

Two qualitatively different regimes. For the transformer both settings are **large
fixes that substitute** — attenuating 9.3x and 4.3x. For the three convolutional
arms both are **small penalties** (+0.02 to +0.11, always the same sign) that
attenuate only ~1.5–2x. The convolutional arms have no failure to rescue, so
there is nothing for a second fix to be redundant with; they simply pay a little
for each setting, and pay slightly less for the second.

**The mechanism this implies.** Attention over `S` positions with a near-uniform
softmax gives each neighbour ~`1/S` of the attended value. A positional signal
that starts at the token signal's magnitude and then *shrinks* during training
(row norm 0.1987 → 0.1612, while the word table grows to 0.3283) cannot overcome
that dilution at `S` = 512. Cut the dilution (64 positions) **or** raise the
positional signal (a sinusoidal table, ~40x larger) and attention becomes usable.
Doing both adds little, because the first already fixed it. A convolution has
neither problem — its span is fixed and its positional structure is in the block
— which is exactly why it shows no such interaction.

### Both settings act on the arms in opposite directions

Positional encoding, paired at 512 context, n=7:

| arm | sinusoidal − learned |
|---|---:|
| `ascii_bert` | **−1.2030** ±0.024 |
| `ascii_clifford_bert` | +0.1053 ±0.019 |
| `ascii_convnext_bert` | +0.0719 ±0.021 |
| `ascii_convnext_v2_bert` | +0.0782 ±0.019 |

Context length, paired at sinusoidal, n=7:

| arm | 64 − 512 |
|---|---:|
| `ascii_bert` | **−0.3220** ±0.024 |
| `ascii_clifford_bert` | +0.0170 ±0.009 |
| `ascii_convnext_bert` | +0.0314 ±0.004 |
| `ascii_convnext_v2_bert` | +0.0306 ±0.004 |

All 7 of 7 seeds agree in every arm of both tables. **Every change that helps the
transformer mildly hurts all three convolutional arms** — they already carry
positional structure in the block, so a large sinusoidal table only compresses
their token signal (11.07 → 0.42 through the embedding LayerNorm), and a fixed
convolutional span means extra context is simply a little more material to model,
not dilution.

That is why finding 3 in the TL;DR holds: there is no setting that is neutral.

### The consequence for the study's central claim

The transformer's deficit against the best convolutional arm:

| comparison | deficit | gap closed |
|---|---:|---:|
| 512 + learned, as originally reported | **1.9275** | — |
| both arms at 64 + sinusoidal | **0.2938** | **85%** |
| **each arm at its own best cell** (bert 1.2999, `convnext_v2` 0.8973) | **0.4026** | **79%** |

The second row is the like-for-like comparison at a shared setting; the third
lets every arm use the configuration that suits it, which is the harder test and
the fairer one. Both are reported because they answer different questions, and
quoting only the flattering one would overstate the repair.

Either way, **~80% of the headline gap was a misconfigured baseline**, closed
with configuration changes and no architecture change. The convolutional arms
still win — on all four configurations and at every arm's own best — but the
defensible size of that claim is ~0.3–0.4 nats, not ~1.9.

---

## Retrieval — SQuAD v1.1, recall@1, mean over 7 seeds

Chance is **0.00048** (1 of 2,067 unique paragraphs).

| arm | 512 / learned | 512 / sinusoidal | 64 / sinusoidal | 64 / learned |
|---|---:|---:|---:|---:|
| `ascii_bert` | 0.0129 ±0.003 | 0.0019 ±0.001 | **0.0261** ±0.004 | 0.0109 ±0.004 |
| `ascii_clifford_bert` | **0.0587** ±0.009 | 0.0061 ±0.001 | 0.0531 ±0.004 | 0.0533 ±0.004 |
| `ascii_convnext_bert` | **0.0594** ±0.004 | 0.0170 ±0.007 | 0.0511 ±0.004 | 0.0574 ±0.005 |
| `ascii_convnext_v2_bert` | **0.0681** ±0.003 | 0.0507 ±0.007 | 0.0575 ±0.006 | 0.0613 ±0.005 |

**The sinusoidal retrieval penalty is a long-context effect, not a property of
sinusoidal encodings.** At 512 it is severe (up to 10x); at 64 it nearly
vanishes, and `ascii_bert` reaches its best retrieval of any configuration
measured — double its original.

SST-2 probe accuracy stays within 0.03 of 0.60 in **every** cell of every run, so
content remains linearly decodable throughout. What moves is specifically cosine
retrieval. That gap — a linear probe steady while cosine collapses — is the
signature that led to the whitening result below.

---

## The largest retrieval win here is free

Full 2000-query protocol, 3 seeds per cell, ZCA whitening fitted on the context
pool (no queries, no labels):

| arm | learned raw | learned + ZCA | gain | sinusoidal raw | sinusoidal + ZCA |
|---|---:|---:|---:|---:|---:|
| `ascii_bert` | 0.0102 | 0.0102 | **1.00x** | 0.0040 | 0.0013 |
| `ascii_clifford_bert` | 0.0598 | **0.2178** | 3.64x | 0.0435 | 0.0212 |
| `ascii_convnext_bert` | 0.0590 | **0.2075** | 3.52x | 0.0280 | 0.0047 |
| `ascii_convnext_v2_bert` | 0.0647 | **0.2110** | 3.26x | 0.0535 | 0.0297 |

**9 of 9 learned convolutional cells gain (3.09x–3.97x, no exceptions); 12 of 12
sinusoidal cells lose.** `ascii_bert` with learned positions gains exactly
nothing — its own confirmation, since that model is a bag of characters with no
drowned structure to recover. Whitening therefore works as a **diagnostic** as
much as a fix: it helps only where low-variance directions carry content.

**It is not transductive.** Fitting the transform on 3000 SST-2 sentences —
movie reviews, a different domain from the Wikipedia paragraphs being retrieved —
keeps ~90% of the gain:

| arm | raw | ZCA fit on the pool | ZCA fit on SST-2 |
|---|---:|---:|---:|
| `ascii_clifford_bert` | 0.0575 | 0.2145 | **0.1865** |
| `ascii_convnext_bert` | 0.0595 | 0.2060 | **0.1925** |
| `ascii_convnext_v2_bert` | 0.0620 | 0.2085 | **0.1940** |

So it is a property of the embedding space, not of the corpus: fit once on
arbitrary text, ship it with the encoder, no index-time corpus access. This is
the known BERT-whitening result (Su et al., 2021) reproducing on character-level
encoders — a reason to trust it rather than to discount it.

**Consequence for this file.** Every `squad_*` number above is raw cosine and
understates these encoders by ~3.3x. The ordering also changes: raw ranks
`convnext_v2` clearly first; whitened puts the three convolutional arms within
0.01 of each other. Quote both.

---

## Why the transformer arm was stuck

The diagnosis behind finding 2, kept because it is reusable.

**It had converged, not stalled early.** Doubling both context and steps moved
validation loss by **−0.0041 nats**. Recovering the instantaneous per-batch loss
(Keras logs the epoch running mean, which hides this) shows the arm reaching
2.878 by step ~400 and then going flat at **−0.0081 nats per 1000 steps** —
~200,000 further steps to reach where the convolutional arms already were.

**Where it stopped is exactly derivable.** MLM leaves 10% of selected tokens
unchanged, which are free to copy. A model that predicts the corpus unigram and
copies those scores:

| | predicted | measured |
|---|---:|---:|
| loss | `0.9 x 3.1135` = **2.8022** | 2.8307–2.8331 |
| accuracy | `0.9 x 0.1532 + 0.1` = **0.2379** | 0.2321–0.2322 |

**Confirmed directly.** Perturbing the trained model's position-0 hidden state:
reordering the whole context (identical multiset) moved it **0.83%** of
activation scale; replacing the context moved **52.58%**. A 63x ratio — the model
read *which* characters were present, not *where*. Consistently, its position
table **shrank** during training (mean row norm 0.1987 → 0.1612) while the word
table **grew** (0.1987 → 0.3283).

---

## What was refuted

Every one of these was a plausible, stated hypothesis that measurement killed.
They are recorded because knowing what is *not* the cause is most of the value.

| hypothesis | how it died |
|---|---|
| **Missing external residual** | `TransformerLayer` is self-contained — it adds its own two residuals. Unlike the conv blocks, which are transform-only and get an external add in their wrappers. |
| **Dead attention / inverted mask** | Attention moves information 25 positions: **2.106e-03**, while both conv arms measure **exactly 0.000** beyond their spans. Stage 1 is packed, so the mask is all-ones anyway. |
| **post-LN, warmup, dropout, weight decay** | Seven configurations — pre-LN, warmup 0.10, dropout 0.0, weight decay 0.0, and combinations — span **0.0024 nats**. All inert. |
| **Shared offset / anisotropy** (retrieval) | Centering drives anisotropy to −0.0002 and `cos_to_centroid` to 0.0001 — a complete fix — and R@1 does not move. Clifford also lost 10x with anisotropy slightly *improving*, and across arms the anisotropy rise is *anti-correlated* with the damage. |
| **Variance drowning** (retrieval, sinusoidal) | ZCA whitening makes sinusoidal models **worse** (0.0040 → 0.0013). The content is genuinely absent there — though this hypothesis is *correct* for the learned-position conv arms, which is where the 3.3x came from. |
| **"The two effects compound"** — this file's own framing | Corrected 2026-08-30. Naming the positional encoding as the defect and context length as secondary reflected the order they were found, not the data. Each setting is worth ~1.2–1.4 nats alone and ~0.13–0.32 second; they **substitute**. The collapse needs both conditions together, so neither is primary. |
| **"A longer run could close or reverse it"** — Run 1's own explanation | Stated in this file's first version as "a statement about training speed at this budget". False: the arm had stopped moving. The convolutional prior is a real advantage, but it is not what produced Run 1's number. |

---

## Known defects and confounds

**The arms are not equally regularized.** `ascii_bert` carries attention-probability
dropout 0.1 *and* FFN dropout 0.1; the convnext arms carry 0.1; `ascii_clifford_bert`
carries **none** — it never puts `dropout_rate` in its `block_config` and
`build_clifford_block` has no such parameter. Measured **inert** as a cause
(2.8307 at dropout 0.0), but it is an uncontrolled difference between arms and
should be equalized before the comparison is restated. Not yet fixed: it requires
a layer change, not a config change.

**Train/eval overlap.** SQuAD contexts *are* Wikipedia paragraphs and the MLM
corpus is Wikipedia. Every arm shares the overlap, so the ordering survives; the
absolute numbers do not.

**Prefix matching at 512.** SQuAD contexts average ~780 characters, so a 512- or
64-character window sees a prefix, not a passage.

**Absolute quality is low.** The best raw cell is `convnext_v2`/learned at R@1
0.0681 — 141x chance, and still under 7%. Whitened it is ~0.21. These are
6000-step `tiny` models.

---

## Run 1 — the original smoke test (superseded)

Kept because the sections above correct it by name. Four arms, `tiny`, `mean`,
**one seed**, 3000 MLM + 1000 contrastive steps, `max_seq_length=256`, 20 minutes
for all four cells on one RTX 4090.

| | bert | clifford | convnext | convnext_v2 |
|---|---:|---:|---:|---:|
| parameters | 839,040 | 515,456 | 578,432 | 582,528 |
| SQuAD MRR@10 | 0.0246 | 0.0538 | 0.0604 | **0.0647** |
| MLM val loss | 2.838 | 1.324 | 1.179 | **1.153** |
| MLM val accuracy | 23.28% | 61.68% | 66.04% | **66.67%** |

Every comparison in it is `UNDERPOWERED` — the primary family (3 non-baseline
arms, Holm-corrected) has a floor of **7 seeds**, the secondary family (18 tests,
BH) needs **10**. That means the question was not askable, not that the arms were
equivalent.

**Speed**, after subtracting ~54 s fixed overhead, at `tiny` / 256 / batch 32:
`convnext_v2` **11 ms/step**, `ascii_bert` 31, `clifford` 65. The arms differ, so
the run was not bottlenecked on the Python packing generator.

---

## References

- Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding. (https://arxiv.org/abs/1810.04805)
- Gao et al., 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings.
  (https://arxiv.org/abs/2104.08821)
- Su et al., 2021. Whitening Sentence Representations for Better Semantics and
  Faster Retrieval. (https://arxiv.org/abs/2103.15316)
- Vaswani et al., 2017. Attention Is All You Need.
  (https://arxiv.org/abs/1706.03762)
