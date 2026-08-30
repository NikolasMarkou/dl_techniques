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
transformer's deficit against the best arm was first reported as **1.9035 nats**.
Repair its positional encoding **or** shorten its context — either alone
recovers most of it. At each arm's own best cell the deficit is **0.3785**, a
**80%** repair; averaged over all four configurations it is 0.8152, a 57%
repair. (The 86% figure this file used to lead with compares the two extreme
corners of the 2x2 — see the note in "The consequence for the study's central
claim".) The defensible claim is ~0.4 nats, not ~1.9.

**3. The two model families want opposite settings, so no fixed setting is
neutral.** The transformer's best cell is 64 + sinusoidal, on *both* endpoints at
once; all three convolutional arms peak at 512 + learned, on both. Any
fixed-setting four-arm comparison handicaps someone, and the choice of setting
decides part of the answer. The strong two-way interaction behind this is
**specific to the attention arm** — the convolutional arms show only a small
same-signed penalty from each setting.

**4. The sinusoidal retrieval collapse involves a length artifact, but length
drift is one contributor, not the cause.** A pooled vector does depend on
sequence length, and more so under sinusoidal positions. But on **natural text**
learned-position models drift almost as much (cosine 0.53 against sinusoidal's
0.39 at L=512) while retrieving **9.7x better**, so drift does not separate them;
the dramatic 0.9705-vs-0.3805 figure comes from a repeated-sentence probe where
content is held identical by construction. Swapping to a length-invariant readout
on the same trained weights **does** help — clifford 3.9x, p=0.0156, 7/7 seeds —
but reaches only 41% of the learned-position baseline. For retrieval, use
**learned positions**.

**5. Every raw retrieval number below understates these encoders by ~3.9x.**
ZCA whitening lifts the learned-position convolutional arms from R@1 ~0.059 to
~0.23, on **14 of 14 cells**, with no retraining.

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

All 16 cells are n=7. **Bold marks each arm's best configuration** — and the
transformer's is a different cell from every other arm's.

Reference points: a uniform guess over the 101-id vocabulary costs `ln(101)` =
**4.615 nats**; predicting from character frequencies alone, with no context,
costs **3.1135 nats** (measured on the exact packed stream, 3,121,152 ids). The
block ordering — `convnext < clifford < bert` — is **identical
under all four configurations** — 28 of 28 cells, per seed.

### The two settings substitute for each other — for the transformer only

The completed 2x2, n=7 in every one of its four cells:

| `ascii_bert` | learned | sinusoidal |
|---|---:|---:|
| **512 context** | 2.8248 ±0.008 | 1.6218 ±0.025 |
| **64 context** | **1.4285** ±0.055 | **1.2999** ±0.012 |

| effect | applied alone | applied second |
|---|---:|---:|
| positions, learned → sinusoidal | **−1.2030** *(at 512)* | −0.1287 *(at 64)* |
| context, 512 → 64 | **−1.3963** *(at learned)* | −0.3220 *(at sinusoidal)* |

Each change is worth ~1.2–1.4 nats on its own and ~0.13–0.32 once the other is in
place. **Either alone recovers most of the collapse**, and 64 + learned (1.4285)
is slightly better than 512 + sinusoidal (1.6218).

> **Correction, 2026-08-30 (review).** This table used to carry an
> "attenuation" column reporting **9.3x** for positions and **4.3x** for
> context, and the text inferred from the difference that "shortening the
> context is marginally the stronger single fix". **There are not two
> attenuations.** In a 2x2 the interaction contrast is symmetric — which factor
> you call "first" cannot change it — and re-derived paired by seed it is one
> number:
>
> ```
> positions interaction  −1.0743 ±0.0651   7/7 seeds   p=0.0156
> context   interaction  −1.0743 ±0.0651   7/7 seeds   p=0.0156
> ```
>
> 9.3x and 4.3x are that single interaction divided by two different main
> effects. The interaction is large and real; there is just one of it, and no
> ordering claim follows from the ratio.

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

Two qualitatively different regimes. For the transformer both settings are **large
fixes that substitute**: one interaction of **−1.0743 ±0.0651**, 7/7 seeds,
p=0.0156. For the three convolutional arms both are **small penalties** (+0.02 to
+0.11, always the same sign).

Whether the convolutional arms show *any* interaction is a weaker claim than this
file used to make. Re-derived, paired by seed:

| arm | interaction | seeds agreeing | p |
|---|---:|---|---:|
| `ascii_clifford_bert` | +0.0328 ±0.0203 | 7/7 | 0.0156 |
| `ascii_convnext_bert` | +0.0223 ±0.0244 | 5/7 | **0.0938** |

Under this study's own Holm family of three non-baseline arms, **neither convnext
arm's attenuation survives**; only clifford's does. An earlier version of this
paragraph stated "attenuate only ~1.5–2x" for all three as established fact.
What is established is that the convolutional arms have no *large* interaction —
not that they have a small one.

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

Context length, paired at sinusoidal, n=7:

| arm | 64 − 512 |
|---|---:|
| `ascii_bert` | **−0.3220** ±0.024 |
| `ascii_clifford_bert` | +0.0170 ±0.009 |
| `ascii_convnext_bert` | +0.0314 ±0.004 |

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
| 512 + learned, as originally reported | **1.9035** | — |
| both arms at 64 + sinusoidal | **0.2753** | 86% |
| **each arm at its own best cell** (bert 1.2999, `convnext` 0.9213) | **0.3785** | **80%** |
| averaged over all four configurations | **0.8152** | 57% |

The third row is the one to quote. It lets every arm use the configuration that
suits it, which is the harder test and the fairer one.

> **Read the 85% row with care.** Both its numbers are corners of the 2x2, and
> in opposite directions. `512/learned` is simultaneously `ascii_convnext_bert`'s
> **best** cell and `ascii_bert`'s **worst**; `64/sinusoidal` is `convnext`'s **worst**
> and `ascii_bert`'s **best**. So 85% is the ratio between the maximum-spread and
> minimum-spread corners of the design, and it is the largest number the data
> can be made to yield. Averaged over all four configurations the deficit is
> 0.8152 and the gap closed is **57%**. The honest range is 57%–86%, with 80%
> the defensible single figure.

So **most of the headline gap was a misconfigured baseline**, closed
with configuration changes and no architecture change. The convolutional arms
still win — on all four configurations and at every arm's own best — but the
defensible size of that claim is ~0.4–0.8 nats, not ~1.9.

---

## Retrieval — SQuAD v1.1, recall@1, mean over 7 seeds

Chance is **0.00048** (1 of 2,067 unique paragraphs).

| arm | 512 / learned | 512 / sinusoidal | 64 / sinusoidal | 64 / learned |
|---|---:|---:|---:|---:|
| `ascii_bert` | 0.0129 ±0.003 | 0.0019 ±0.001 | **0.0261** ±0.004 | 0.0109 ±0.004 |
| `ascii_clifford_bert` | **0.0587** ±0.009 | 0.0061 ±0.001 | 0.0531 ±0.004 | 0.0533 ±0.004 |
| `ascii_convnext_bert` | **0.0594** ±0.004 | 0.0170 ±0.007 | 0.0511 ±0.004 | 0.0574 ±0.005 |

**The sinusoidal retrieval penalty is a long-context effect, not a property of
sinusoidal encodings.** At 512 it is severe (up to 10x); at 64 it nearly
vanishes, and `ascii_bert` reaches its best retrieval of any configuration
measured — double its original.

SST-2 probe accuracy spans **0.5252–0.6514** over all 118 cells and
**0.5767–0.6355** at the 7-seed arm-configuration mean, so content remains
linearly decodable throughout while cosine retrieval moves by up to 10x. What
moves is specifically cosine retrieval.

> **Correction, 2026-08-30 (review).** This used to read "stays within 0.03 of
> 0.60 in **every** cell of every run". That bound is wrong at both readings of
> "cell": **30 of 118** individual cells fall outside [0.57, 0.63], and so do
> **2 of 16** arm-configuration means. The conclusion is unaffected — the probe
> moves ~0.06 across everything while retrieval moves 10x — but the probe is
> also not arm-neutral, which the old phrasing implied: `ascii_bert` sits at
> 0.577–0.593 in every configuration and the convolutional arms at 0.61–0.64.

That gap — a linear probe steady while cosine collapses — is the signature that
led to the whitening result below, and eventually to the cause.

### The cause: a sinusoidal model's pooled embedding depends on LENGTH

Earlier revisions of this file said the effect was measured and the cause was
not established. It is now established, and it is not really about retrieval.

Encode **one sentence repeated** to different real lengths, so the content
distribution is identical and only length changes. Cosine against the 64-
character version:

| | L=64 | L=128 | L=256 | L=512 |
|---|---:|---:|---:|---:|
| `ascii_bert` / learned | 1.0000 | 0.9895 | 0.9794 | **0.9705** |
| `ascii_bert` / **sinusoidal** | 1.0000 | 0.9433 | 0.6441 | **0.3805** |
| `ascii_clifford_bert` / learned | 1.0000 | 0.9789 | 0.9733 | **0.9666** |
| `ascii_clifford_bert` / **sinusoidal** | 1.0000 | 0.9629 | 0.8899 | **0.7521** |

Under **this probe** learned-position models are essentially length-invariant
across an 8x range, and sinusoidal ones are not: the same content at 512
characters is nearly orthogonal to itself at 64 (0.3805). Mean-pooling averages
positions `0..L-1`, and with a positional table of row norm exactly 8.0 at
`d`=128 that mean is dominated by a term that differs with `L`.

**SQuAD queries average 59 characters; contexts average 774.** So a query and its
own answer land in different regions of the space partly *because of the length
difference*, before content is considered.

> **Correction, 2026-08-30 (review). The 0.9705-vs-0.3805 separation is an
> artifact of the repeated-sentence probe, and it does not survive on natural
> text.** In that probe the content is literally identical at every length, so
> the positional term is the only thing that *can* move the vector. Re-run on 20
> real SQuAD contexts, prefix-truncated, cosine against the same text at L=64:
>
> | model | readout | repeated @512 | **natural text @512** |
> |---|---|---:|---:|
> | `ascii_bert` / sinusoidal | mean | 0.4690 | **0.3879 ±0.040** |
> | `ascii_bert` / **learned** | mean | 0.9518 | **0.5294 ±0.190** |
> | `ascii_clifford_bert` / sinusoidal | mean | 0.6918 | **0.4314 ±0.133** |
> | `ascii_clifford_bert` / **learned** | mean | 0.9691 | **0.5666 ±0.137** |
>
> The learned-vs-sinusoidal gap collapses from ~0.48 to **~0.14**.
> Learned-position models drift nearly as much as sinusoidal ones on the kind of
> text being retrieved — while retrieving **9.7x better** on the same cells
> (clifford 0.0587 against 0.0061). **A 0.14 difference in cosine cannot account
> for an order-of-magnitude retrieval gap**, so the claim below that this single
> mechanism accounts for every observation is withdrawn.
>
> What survives: length drift is real, it is worse under sinusoidal positions,
> and `max` pooling genuinely removes it — on natural text and for every arm,
> `max` scores 0.977–0.979 at L=512 where `mean` scores 0.39–0.57.

That mechanism accounts for several observations, and is stated here as a
contributor rather than as the cause:

| observation | explanation |
|---|---|
| retrieval collapses, up to 10x | query/document length mismatch, ~59 vs ~774 |
| SST-2 probe unmoved | one narrow length band, so the offset is a constant a linear probe absorbs |
| centering does nothing | the offset varies per text; centering removes one global mean |
| whitening makes it worse | same reason, plus it amplifies the low-variance directions where content survives |
| the 64-context run shows no penalty | packed training makes every sequence exactly 64 — no mismatch to expose |
| MLM ordering unaffected in all 16 cells | MLM is scored per position and never pooled, so it never touches the readout |

It also predicts the truncation curve that found it. Re-measured 2026-08-30 at
3 seeds, truncating **both** queries and contexts to a common length:

| arm / positions | L=64 | L=128 | L=256 | L=512 |
|---|---:|---:|---:|---:|
| `ascii_clifford_bert` / **sinusoidal** | 0.0505 | 0.0532 | 0.0443 | **0.0055** |
| `ascii_clifford_bert` / learned | 0.0515 | 0.0592 | 0.0595 | 0.0612 |

The clifford row reproduces the curve this file reported (0.0560, 0.0640, 0.0560,
0.0060) in shape and magnitude.

> **But the curve does not isolate length mismatch.** Queries average 59
> characters, so "truncating to L" leaves them untouched at every L; only the
> contexts grow. The mismatch therefore rises monotonically — 1x, 2x, 4.3x, 8.7x
> — while retrieval stays flat to L=256 and then collapses, which is a threshold
> rather than the smooth dependence a length-proportional offset predicts. More
> decisively, the now-withdrawn V2 arm carried **exactly the same query/context
> length mismatch at 512 and did not collapse** (0.0523, level across all four
> lengths, against clifford's 0.0055). An arm with identical mismatch and no
> damage means mismatch alone is not sufficient. That arm is no longer in the
> study — see "The V2 arm was withdrawn" below — but the measurement stands and
> is the reason this section reports a contributor rather than a cause.

**Two hypotheses died on the way, and the second death was the informative one.**
A padding-mismatch version of this story is wrong: encoding the same text padded
to 128, 256 and 512 gives identical embeddings on **all three** arms (max |Δ|
1.8e-07 to 3.6e-07), because pooling is mask-aware. Ruling that out is what
forced the distinction between padded length and real length, which is the one
that matters.

> **This held for three arms and not for the fourth**, which is part of why the
> fourth is gone. The withdrawn V2 arm moved by **1.663e-01** (cosine 0.9863)
> across pad widths 128/256/512 on identical real content, because
> `GlobalResponseNormalization` reduces over axes `1..N-2` — the sequence axis,
> once the block lifts to `(B, 1, L, D)`. It was immaterial to the numbers in
> this file only because length-sorted batching holds evaluation pad fractions
> at 1-3%, which is a property of the evaluation and not of the arm.
> `tests/test_models/test_embeddings_shared/test_the_arms_differ_in_reach.py`
> now requires pad-width inertness of every arm in the registry.

### It is the readout, and `max` pooling fixes it

Nothing above says sinusoidal positions are bad. It says **`mean` pooling is the
wrong readout for an encoder with a large positional signal** — and the readout
can be swapped on the *same trained weights*, with no retraining. Same probe,
same checkpoints, cosine against the 64-character version:

| model | `mean` | `cls` | `last` | `max` |
|---|---:|---:|---:|---:|
| `ascii_bert` / sinusoidal | **0.3805** | 0.7810 | 0.2482 | **0.9693** |
| `ascii_clifford_bert` / sinusoidal | 0.7521 | 1.0000 | 0.0248 | **0.9963** |
| `ascii_bert` / learned | 0.9705 | 0.9948 | 0.9911 | 0.9945 |
| `ascii_clifford_bert` / learned | 0.9666 | 1.0000 | 0.2636 | 0.9984 |

`max` is length-invariant on both sinusoidal models where `mean` collapses,
because a per-dimension extremum does not accumulate a positional mean that grows
with `L`.

**Read the `cls` column carefully.** `ascii_clifford_bert` scores exactly 1.0000
at every length, which is *not* readout invariance: that block's span is 49
tokens, so position 0 cannot see beyond it and the rest of the sequence is
literally unreachable. `ascii_bert`, whose attention does see the whole
sequence, gives 0.7810 — the honest number for `cls`.

**`max` was not in the study's pooling axis.** `EmbeddingEncoder` has supported
it since the beginning (`SUPPORTED_POOLING`), but `POOLING_STRATEGIES` listed
only `cls`, `mean` and `attention`, and the trainer's `--pooling-strategy`
validates against that tuple — so the readout that fixes this was unreachable
from the study. Added 2026-08-30.

### Swapping the readout DOES help — but not enough. Measured twice.

This section previously concluded that the fix was refuted. **It was testing a
different experiment from the one the prediction names**, and the correct test
does not refute it.

**Test 1 — the prediction as stated: swap the readout on the same trained
weights, no retraining.** Mean-trained 512/sinusoidal checkpoints, readout
swapped at evaluation, **n=7**, paired by seed:

| arm | `mean` (as trained) | `max` (swapped) | `cls` (swapped) | paired `max` − `mean` |
|---|---:|---:|---:|---|
| `ascii_bert` | 0.00186 | **0.00293** | 0.00157 | +0.00107, p=0.1875, 5/7 |
| `ascii_clifford_bert` | 0.00607 | **0.02386** | 0.01000 | **+0.01779, p=0.0156, 7/7** |

Chance is 0.00048. `ascii_clifford_bert` improves **293%**, significant at the
n=7 floor with every seed agreeing. `ascii_bert` improves slightly and
non-significantly — and, importantly, **does not fall to chance**: 0.00293 is 6x
chance and better than its own `mean` readout.

**Test 2 — cells RETRAINED at `pooling=max`** (a separate sweep root, its own
`encoder.keras`, stage 2 trained under `max`), 3 seeds:

| arm | sinusoidal + `mean` (n=7) | sinusoidal + `max`, retrained (n=3) | learned + `mean` (n=7) |
|---|---:|---:|---:|
| `ascii_bert` | 0.00186 | 0.00033 | 0.01286 |
| `ascii_clifford_bert` | 0.00607 | 0.02167 | 0.05871 |

> **Correction, 2026-08-30 (review).** The old text read the second table as
> refuting the prediction — *"`max` pooling ... drives `ascii_bert` to chance"*.
> Three problems:
>
> 1. **It is not the stated experiment.** This file defines the fix as a readout
>    "swapped on the *same trained weights*, with no retraining". Retraining
>    under `max` changes the stage-2 contrastive objective as well as the
>    readout. Done as stated, bert goes to 0.00293, not 0.00033.
> 2. **n=3 cannot refute anything here.** By this study's own power rule the
>    smallest reachable p at n=3 is **0.250**, and both paired comparisons on the
>    shared seeds 0–2 return exactly that floor (bert −0.00067, clifford
>    +0.01617). The study labels such comparisons `UNDERPOWERED` everywhere else.
> 3. **bert's retrained cells are 1, 0 and 1 hits out of 2000.** "3% of its
>    learned baseline" is a ratio on single-digit counts.
>
> What survives is the weaker half, and it is still worth having: **neither
> readout reaches the learned-position baseline** — clifford recovers to 41% of
> it (0.0239 against 0.0587), bert to 23% (0.0029 against 0.0129). So
> length-invariance is **necessary and not sufficient**, exactly as this section
> always said. What is withdrawn is "does not fix it" and "falls to chance".

A per-dimension extremum over 128 dimensions does discard information, and how
much depends on the arm — but on the stated test it costs less than the drift it
removes for clifford, and roughly breaks even for bert.

Whitening does not rescue the retrained `max` cells — it lowers them (clifford
0.0217 raw to 0.0065 whitened), consistent with `max` already having flattened
the variance structure the transform exploits.

**Practical recommendation, unchanged.** For retrieval use **learned positions
with `mean` pooling**: it beats every sinusoidal configuration measured, with or
without a swapped readout, and it whitens to ~0.23. The sinusoidal positional
repair belongs to the language-modelling objective. `max` is worth reaching for
when a sinusoidal encoder is already trained and retraining is not an option —
it recovers a significant fraction for the clifford arm at zero cost.

Two cells of this run first failed with a CUDA OOM in the contrastive stage and
were re-run to completion on an idle GPU with no config change; the numbers above
are all from clean cells.

---

## The largest retrieval win here is free

Full 2000-query protocol, **7 seeds per cell**, ZCA whitening fitted on the
context pool (embeddings only — no queries, no labels). Re-measured 2026-08-30
from the saved encoders; see the correction below for what this replaces.

| arm | learned raw | learned + ZCA | gain | cells gaining |
|---|---:|---:|---:|---|
| `ascii_bert` | 0.0129 | 0.0084 | **0.65x** | 0 / 7 |
| `ascii_clifford_bert` | 0.0587 | **0.2304** | **3.92x** | 7 / 7 |
| `ascii_convnext_bert` | 0.0594 | **0.2268** | **3.82x** | 7 / 7 |

| arm | sinusoidal raw | sinusoidal + ZCA | gain | cells gaining |
|---|---:|---:|---:|---|
| `ascii_bert` | 0.0019 | 0.0010 | 0.54x | 1 / 7 |
| `ascii_clifford_bert` | 0.0061 | **0.0469** | **7.72x** | **7 / 7** |
| `ascii_convnext_bert` | 0.0170 | 0.0059 | 0.34x | 0 / 7 |

**All 14 learned convolutional cells gain — 3.82x and 3.92x by arm, 3.16x to
5.19x by cell, no exceptions.** That is the headline result and it is stronger
than the 3-seed version this file used to carry.

Whitening works as a **diagnostic** as much as a fix: it helps only where
low-variance directions carry content.

> **Correction, 2026-08-30 (review).** Three things in the previous version of
> this section were wrong, and one had no source.
>
> 1. **"`ascii_bert` with learned positions gains exactly nothing (1.00x) — its
>    own confirmation."** At 7 seeds that arm **loses 35%** (0.0129 → 0.0084),
>    with 0 of 7 cells gaining; the per-cell gains are 0.87, 1.00, 0.62, 0.78,
>    0.52, 0.13, 0.95. The "exactly 1.00x" was seed 1 alone. The point it was
>    making — that whitening finds nothing in a bag of characters — is if
>    anything stronger, but it was a coincidence read as a result.
> 2. **"12 of 12 sinusoidal cells lose."** `ascii_clifford_bert`/sinusoidal
>    **gains 7.72x on 7 of 7 cells** — the largest whitening gain anywhere in
>    this study — and it was recorded as a loss.
> 3. **The old table's numbers are not re-derivable.** `squad_whitened_*` was
>    added by `8915409e2`, *after* Runs 1–4, so the four 28-cell roots carry no
>    whitened keys at all and both whitening tables came from an ad-hoc
>    measurement whose artifacts were not kept. The old "sinusoidal raw" column
>    matched **no stored root** at 3 or 7 seeds (it gave clifford 0.0435 against
>    0.0061 at 512/sinusoidal and 0.0531 at 64/sinusoidal).
>
> The tables above are re-derived from the saved `encoder.keras` of every cell,
> with the whitening path first checked against the shipped code on the 6
> maxpool cells that *do* carry stored `squad_whitened_*` values — **6 of 6
> identical**.

**It is not transductive**, and this is now measured by a route that exists in
the code. Fitting the transform on a random **half** of the context pool, and
separately on the **queries** — short questions rather than long paragraphs, a
different length and register — at 512/learned, 7 seeds:

| arm | raw | fit on the pool | fit on half the pool | fit on the queries |
|---|---:|---:|---:|---:|
| `ascii_clifford_bert` | 0.0587 | 0.2304 | **0.2135** | **0.2582** |
| `ascii_convnext_bert` | 0.0594 | 0.2268 | **0.2119** | **0.2557** |

A half-size fit keeps ~93% of the gain and a fit on a different text
distribution *beats* fitting on the pool.

> **Correction, 2026-08-30 (review).** This paragraph used to cite a fit on
> 3000 SST-2 sentences keeping "~90% of the gain". **That measurement has no
> code anywhere in the study** — `evaluate_embeddings.py` only ever fits on the
> context pool — so it could not be reproduced. The conclusion it supported is
> confirmed above by two fits that can be.

So it is a property of the embedding space, not of the corpus: fit once on
arbitrary text, ship it with the encoder, no index-time corpus access. This is
the known BERT-whitening result (Su et al., 2021) reproducing on character-level
encoders — a reason to trust it rather than to discount it.

**Consequence for this file.** Every `squad_*` number above is raw cosine and
understates these learned-position convolutional encoders by ~3.9x. The
ordering also changes: raw ranks `convnext` and `clifford` a hair apart
(0.0594 against 0.0587); whitened they swap, and sit within 0.004 of each other
(0.2268 against 0.2304). Neither ordering is meaningful. Quote both.

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

| | predicted | measured (diagnostic runs) | Run 2a, 7 seeds |
|---|---:|---:|---:|
| loss | `0.9 x 3.1135` = **2.8022** | 2.8307–2.8331 | **2.8128–2.8366** |
| accuracy | `0.9 x 0.1532 + 0.1` = **0.2379** | 0.2321–0.2322 | **0.2314–0.2351** |

The middle column is from the 256-context diagnostic runs, not from the 28-cell
study; it carried no label saying so until 2026-08-30, and its bands are much
tighter than the study's own cells. The right-hand column is Run 2a re-derived.
The prediction holds against both.

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
| **Dead attention / inverted mask** | Attention moves information 25 positions: **2.106e-03**, while both convolutional arms measure **exactly 0.000** beyond their spans. Stage 1 is packed, so the mask is all-ones anyway. (A fourth arm failed this and was withdrawn — see "The V2 arm was withdrawn".) |
| **post-LN, warmup, dropout, weight decay** | Seven configurations — pre-LN, warmup 0.10, dropout 0.0, weight decay 0.0, and combinations — span **0.0024 nats**. All inert. |
| **Shared offset / anisotropy** (retrieval) | Centering drives anisotropy to −0.0001 — a complete fix — and R@1 does not move (re-verified 2026-08-30 at 7 seeds: `ascii_bert` 0.0129 → 0.0124, `ascii_clifford_bert` 0.0587 → 0.0606). **The two supporting facts this row used to give were both wrong** and are withdrawn: clifford's anisotropy *worsens* under sinusoidal positions (0.2821 → 0.3568), it does not "slightly improve"; and across arms the anisotropy rise is **positively** correlated with the damage (r = +0.324, n=4), not anti-correlated. The refutation rests on the centering result alone, which holds. |
| **Variance drowning** (retrieval, sinusoidal) | ZCA whitening makes **two of the three** sinusoidal arms worse (`ascii_bert` 0.0019 → 0.0010, `convnext` 0.0170 → 0.0059). Correct for the learned-position conv arms — that is where the ~3.9x came from — but wrong for those two. NOT wrong for `ascii_clifford_bert`/sinusoidal, which **gains 7.72x on 7 of 7 cells**; an earlier revision of this row claimed all 12 sinusoidal cells lose. An earlier revision concluded from this that "the content is genuinely absent"; that was premature. The content is present and the vector is displaced by a length-dependent offset, which no global transform can undo. |
| **"A length-invariant readout will fix retrieval"** — my own prediction | **Partly survives; the earlier refutation was of the wrong experiment.** Swapped on the same trained weights as the prediction states (n=7, paired), `max` improves `ascii_clifford_bert` **3.9x** (0.00607 → 0.02386, p=0.0156, 7/7 seeds) and moves `ascii_bert` to 0.00293 — 6x chance, *better* than `mean`, **not** to chance. The "falls to chance (0.00033)" figure came from cells RETRAINED under `max` at n=3, where this study's own power rule makes the minimum reachable p 0.250. What holds is that neither readout reaches the learned-position baseline (41% and 23%), so length-invariance is necessary and not sufficient. |
| **"Padding-length mismatch"** (retrieval) | Encoding one text padded to 128, 256 and 512 gives identical embeddings — pooling is mask-aware, so padding cannot be the cause. Ruling it out forced the distinction between padded and REAL length, which is the actual mechanism. |
| **"The two effects compound"** — this file's own framing | Corrected 2026-08-30. Naming the positional encoding as the defect and context length as secondary reflected the order they were found, not the data. Each setting is worth ~1.2–1.4 nats alone and ~0.13–0.32 second; they **substitute**. The collapse needs both conditions together, so neither is primary. |
| **"A longer run could close or reverse it"** — Run 1's own explanation | Stated in this file's first version as "a statement about training speed at this budget". False: the arm had stopped moving. The convolutional prior is a real advantage, but it is not what produced Run 1's number. |

---

## Known defects and confounds

**The V2 arm was withdrawn — 2026-08-30.** The study ran with a fourth arm,
`ascii_convnext_v2_bert` (ConvNeXt V1 plus Global Response Normalization), and
it was the best arm in every table. It has been removed, and every number in
this file is now a three-arm number.

The reason is that it was not the thing the study said it was. This file
explains the settings asymmetry with "a convolution has a fixed span", and that
was false of the V2 arm: GRN scores each channel by its L2 magnitude over axes
`1..N-2`, which is the **sequence axis** once `ConvNextEncoderBlock` lifts to
`(B, 1, L, D)`. Perturbing position 0 of a trained encoder and reading
`max |Δ|` at position `d`:

| arm | nominal span | d=10 | d=25 | d=60 | d=120 |
|---|---|---:|---:|---:|---:|
| `ascii_bert` | full | 3.130e-02 | 4.188e-02 | 2.661e-02 | 1.696e-02 |
| `ascii_clifford_bert` | 49 | 2.041e-01 | 0.000 | 0.000 | 0.000 |
| `ascii_convnext_bert` | 25 | 2.664e-03 | 0.000 | 0.000 | 0.000 |
| `ascii_convnext_v2_bert` | 25 | 3.484e-02 | **7.210e-02** | **8.378e-02** | **3.989e-02** |

It moved position 60 by more than the attention arm did. The arms were meant to
differ only in the sequence-mixing block, with the convolutional arms sharing a
fixed span; an arm with an always-on global branch is a second uncontrolled
difference, and it was the arm carrying the study's best result. The same GRN
made it the only arm whose pooled embedding depended on the pad **width**. It
was withdrawn rather than explained away.

**What that costs, stated plainly.** The best arm becomes `ascii_convnext_bert`,
and the headline numbers move a little: the original deficit 1.9275 → **1.9035**,
the each-arm-at-its-best deficit 0.4025 → **0.3785** (79% → **80%**), best raw
R@1 0.0681 → **0.0594**, whitening 3.34x–3.92x → **3.82x and 3.92x**. No
conclusion changes and no cell was re-run: each cell is an independent run, so
the surviving arms' numbers are exactly what they were. The block ordering
`convnext < clifford < bert` still holds in all four configurations, 28 of 28
(configuration, seed) pairs.

**The 28 V2 cells under `results/` are left in place**, and stay loadable —
`ConvNextEncoderBlock` still accepts `version="v2"` for that reason. What is gone
is the ability to construct that arm from a study config. The invariant is pinned
by `tests/test_models/test_embeddings_shared/test_the_arms_differ_in_reach.py`,
which requires every convolutional arm to be exactly inert beyond its span and
uses `ascii_bert` as the positive control that the probe can see global mixing at
all.

**The study runs at its statistical power floor.** The primary endpoint
(`eval_squad_mrr_at_10`) is `BETTER` for all three convolutional arms in all four
configurations. But with n=7 the smallest reachable two-sided sign-flip p is
`2/2^7` = **0.0156**, and Holm across the three non-baseline arms multiplies the
first by 3, so **0.0469 is the smallest adjusted p this design can ever
produce**. The measured values are 0.0469 (Run 2a) and 0.0474 (the other three
roots), against a 0.05 bar. Every arm shows perfect 7/7 sign agreement — the
result is as strong as the design can show, and it clears by 0.003. The same
effect judged against the secondary family of 18 (which `README.md` says needs 10
seeds) would not clear.

**Run 2a has no committed verdicts.** `report.py` was never run on
`results/embeddings_study_512`, so that root has only `all_runs.json` — no
`summary.md`, `headline_summary.csv` or `paired_summary.csv`. The 0.0469 above
was recomputed for this file rather than read from an artifact.

**Stage 2 is packed, and its docstring says it is not.** `data.py` states "Stage
2 cannot pack — contrastive learning needs whole sentences", but
`run_contrastive_stage` calls `build_packed_mlm_dataset`. So the contrastive
stage trains on packed fixed-length windows too, which means **no model in this
study has ever seen a sequence of any length other than `max_seq_length`**. That
leaves "the embedding drifts with length" and "no model was ever trained
off-length" unseparated by any measurement here — a live alternative to the
length-drift account above.

**Training frames no `[CLS]`; evaluation frames one.** The packed MLM stream is
`[SEP]`-delimited character runs with no `[CLS]` anywhere, while `embed_texts`
frames every text as `[CLS] + content + [SEP]`. Any `cls`-pooling number is
therefore read off a token the encoder never saw in that position during
training.

**Train/eval overlap.** SQuAD contexts *are* Wikipedia paragraphs and the MLM
corpus is Wikipedia. Every arm shares the overlap, so the ordering survives; the
absolute numbers do not.

**Prefix matching at 512.** SQuAD contexts average ~780 characters, so a 512- or
64-character window sees a prefix, not a passage.

**Absolute quality is low.** The best raw cell is `convnext`/learned at R@1
0.0594 — 123x chance, and still under 6%. Whitened it is ~0.23. These are
6000-step `tiny` models.

---

## Run 1 — the original smoke test (superseded)

Kept because the sections above correct it by name. Four arms, `tiny`, `mean`,
**one seed**, 3000 MLM + 1000 contrastive steps, `max_seq_length=256`, 20 minutes
for all four cells on one RTX 4090 (a fourth arm, since withdrawn, ran too).

| | bert | clifford | convnext |
|---|---:|---:|---:|
| parameters | 839,040 | 515,456 | 578,432 |
| SQuAD MRR@10 | 0.0246 | 0.0538 | **0.0604** |
| MLM val loss | 2.838 | 1.324 | **1.179** |
| MLM val accuracy | 23.28% | 61.68% | **66.04%** |

Every comparison in it is `UNDERPOWERED` — the primary family (3 non-baseline
arms, Holm-corrected) has a floor of **7 seeds**, the secondary family (18 tests,
BH) needs **10**. That means the question was not askable, not that the arms were
equivalent.

**Speed**, after subtracting ~54 s fixed overhead, at `tiny` / 256 / batch 32:
`ascii_convnext_bert` **12 ms/step**, `ascii_bert` 31, `clifford` 65. The arms
differ, so the run was not bottlenecked on the Python packing generator.

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
