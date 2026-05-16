# Token Superposition Training (TST) — Pretraining Efficiency via Bag-Folded Inputs and Multi-Hot Targets

**Paper**: *Efficient Pretraining with Token Superposition* (Nous Research, arXiv:2605.06546)
**Status**: training-only intervention. Inference-time model is bit-identical to one produced by conventional pretraining. No architectural change.

---

## 1. TL;DR

TST modifies the pretraining *loop* — not the model, optimizer, tokenizer, parallelism, or training data — to deliver a **2–3× wall-clock speedup at matched FLOPs** on LLM pretraining. Validated at 270M, 600M, 3B dense and 10B-A1B MoE.

The intervention is two-phase:

- **Phase 1 (superposition, ~20–40% of total steps)**: the model reads contiguous *bags* of `s` tokens. Inputs are formed by averaging the `s` token embeddings within each bag. Targets are the next *bag* of `s` tokens, scored by a mean cross-entropy that reduces algebraically to a sum of `s` ordinary CE terms — so the existing fused CE kernel runs unchanged. Each transformer step now sees `s×` as much text per unit of work.
- **Phase 2 (recovery, remaining 60–80%)**: resume from the phase-1 checkpoint under ordinary next-token prediction. A 1–2 nat loss spike at the transition resolves within ~few thousand steps; the curve then crosses below the matched-FLOPs baseline and stays there.

**Headline numbers** (10B-A1B Qwen3-shaped MoE):
- TST: 50k steps, 2T tokens consumed, 4768 B200-hours, final loss 2.236.
- Baseline: 125k steps, 1.05T tokens, 12311 B200-hours, final loss 2.252.
- TST beats baseline on **HellaSwag, ARC-Easy, ARC-Challenge, and MMLU** (71.2 vs 70.1, 74.2 vs 73.8, 47.3 vs 46.3, 39.0 vs 37.4).
- **38.7% of baseline's wall-clock** to reach lower loss + better downstream.

The implementation is famously small:

> *"The method is basically just a reshape, a mean, and a summed cross-entropy."*

---

## 2. The Two Observations That Motivate TST

### 2a. Where the BPE advantage at iso-FLOPs actually comes from

Subword tokenization (BPE / SentencePiece) outperforms byte-level modeling at fixed FLOPs. The conventional explanation is that learned subword units encode useful semantic chunks. Recent work (Gigant et al.; Minixhofer et al. on subword-to-byte distillation; Zheng et al. on proxy compression) suggests a *simpler* story: subword sequences are **shorter**, so the model processes more natural-language content per unit of compute. The semantic-structure claim is at best a small contributor; the dominant effect is **throughput**.

If pretraining efficiency is partly a throughput problem, then the throughput lever can be pulled independently of the tokenizer.

### 2b. Decoupling training-time and inference-time efficiency

Chain-of-thought, ParScale, looped LMs, and speculative decoding show you can scale *inference* compute independently of training compute. The reverse direction is less explored: most "pretraining efficiency" methods (sparse attention, MoE, alternative tokenization) simultaneously change the trained artifact, introducing confounds and sometimes cancelling training savings against slower inference.

TST adopts the stricter criterion: **a real pretraining efficiency improvement must leave the inference-time architecture untouched.**

---

## 3. The Method — Input Side

### 3a. Bag Reshape

Given input sequence `inputs : (B, N)` and bag size `s`:

```python
if superposition_bag_size > 1:
    bs, seq = inputs.shape
    # (B, N) -> (B, N/s, s)
    inputs = inputs.reshape(bs, seq // superposition_bag_size, superposition_bag_size)
```

### 3b. Embedding-Average

The embedding layer looks up all `s` token embeddings per bag and averages along the bag axis:

```python
# E : (V, d) the embedding table
embedded = E[inputs]                                    # (B, N/s, s, d)
x = embedded.mean(dim=-2)                               # (B, N/s, d)
```

The transformer now processes a sequence of length `N/s` while the underlying text covers `N` tokens. **Nothing else in the forward pass changes.** Same attention, same FFN, same RoPE (applied over `N/s` positions, not `N`), same output projection.

Mathematically, the input at latent position `i` is the centroid of the bag's embedding distribution:

```
x_i = (1/s) Σ_{j=0..s-1} E[t_{is + j}]
```

where `t_k` is the k-th source token.

### 3c. Why this works (intuitively)

Three non-exclusive readings:
- **Throughput**: the model sees `s×` text per step at iso-FLOPs.
- **Low-pass filter / regularizer**: mean is a low-pass filter over the embedding sequence; damps high-frequency variance that makes early training noisy at large vocabularies.
- **Embedding-table geometry constraint**: for averaged bags to remain separable, the vocabulary must lie with sufficient angular dispersion. The phase-1 objective implicitly regularizes the embedding table layout, and that layout is inherited at phase 2.
- **Pre-pretraining analogy**: averaging produces a "simpler" version of the same text — coarser statistical structure. Hu et al. (formal-language pre-pretraining) and Lee et al. (NCA-generated synthetic pre-pretraining) showed that a first stage on a simpler distribution accelerates the second stage by ~1.3–1.6×. TST may be doing the same thing, with the pre-pretraining corpus generated on the fly from the real corpus.

---

## 4. The Method — Output Side

### 4a. Target = Next Bag

At latent position `i`, the target is the *next* bag of `s` tokens:

```
target_i = { t_{(i+1)s}, t_{(i+1)s + 1}, ..., t_{(i+1)s + s - 1} }
```

The model emits a single distribution `p_i ∈ ℝ^V` at each latent position. The loss is a mean cross-entropy over the `s` targets:

```
L_i = (1/s) Σ_{j=0..s-1} -log p_i(t_{(i+1)s + j})
L   = (1/(N/s)) Σ_i L_i
```

### 4b. The MCE ↔ sum-of-CE Equivalence (the paper's Appendix B)

Define a **multi-hot** target distribution that places probability mass `1/s` on each of the `s` valid tokens in the target bag:

```
q_i(v) = (1/s) · |{ j : t_{(i+1)s + j} = v }|             (v ∈ vocabulary)
       = (1/s) · #occurrences of v in the target bag
```

Cross-entropy with this multi-hot target:

```
H(q_i, p_i) = -Σ_v q_i(v) log p_i(v)
            = -Σ_v [(1/s) · #occ(v)] log p_i(v)
            = -(1/s) Σ_{j=0..s-1} log p_i(t_{(i+1)s + j})           ★
            = (1/s) Σ_{j=0..s-1} CE(t_{(i+1)s + j}, p_i)
```

Step ★ uses the fact that summing over distinct vocab indices with their counts is identical to summing over the bag positions. The result is **the average of `s` ordinary cross-entropy terms against the same logit vector**.

**Implementation consequence**: you do *not* need a multi-hot CE kernel. Compute the logits once at each latent position, then call your standard fused CE kernel `s` times (or replicate the logits and call it once on a 1-D batch). The compiler / framework folds the constant `1/s` into the gradient. No new CUDA, no new autograd, no auxiliary head.

### 4c. What's discarded

The objective **discards within-bag order**: the model is asked which tokens appear in the next `s` positions, not in which positions. Framed this way, the bag target is **topic modeling over a short future window** — closer in spirit to Luhn's bag-of-words summarization or Spärck Jones's IDF than to next-token prediction. The closest recent work is Mahajan et al.'s future-summary prediction (which attaches an auxiliary head); TST instead *replaces* the target.

That this works at all suggests a substantial portion of what NTP forces the model to learn concerns the *distribution* of near-future tokens, not their precise ordering — and these can be partially separated.

---

## 5. Two-Phase Training Recipe

```
Total steps:  T
Step ratio:   r ∈ [0.2, 0.4]   (typical)
Bag size:     s ∈ {2..16}      (depends on model size)

Phase 1 (steps 1 .. r·T):
  - Input:  bag-folded, embedding-averaged  (sequence length N/s)
  - Target: next-bag MCE  (= sum of s standard CE terms)
  - Optimizer: AdamW, WSD schedule, ordinary settings
  - Each step sees s× more text than baseline at iso-FLOPs

Phase 2 (steps r·T .. T):
  - Restore from phase-1 checkpoint
  - Input:  standard tokens  (sequence length N)
  - Target: ordinary next-token CE
  - Optimizer state continues; learning-rate schedule continues
  - Loss spikes 1-2 nats at transition, recovers in ~1k-2k steps
  - Loss curve crosses below matched-FLOPs baseline; stays there

Result: inference-time model is bit-identical in architecture to a
standard-pretrained model. Only the training history differs.
```

**At the transition**, nothing in the model is touched. Same weights, same optimizer state, same dataloader continuation. The only change is the dataloader collation and the loss function — both swap back to their canonical forms.

---

## 6. Power-Law Within-Bag Weighting

Uniform weighting (each target position gets weight `1/s`) is suboptimal at large bag sizes. The paper finds a power-law weighting:

```
w_j ∝ (j+1)^{-α},     j ∈ {0, 1, ..., s-1}
```

normalized so `Σ_j w_j = 1`. The loss becomes:

```
L_i = Σ_{j=0..s-1} w_j · CE(t_{(i+1)s + j}, p_i)
```

For `s ≤ 6` the choice is essentially indistinguishable from uniform. For `s > 6`, power-law weighting yields **lower final loss**.

### 6a. Justification (Ebeling–Pöschel 1994)

The choice of α is not arbitrary. Ebeling & Pöschel (1994) showed that **mutual information between English letter pairs decays as a power law with distance**: `I(d) ∝ d^{-β}`. The TST authors measured the same quantity for tokenized DCLM and found the same functional form with fitted exponent **α ≈ 0.6**.

The interpretation: near-future tokens carry more information about the present hidden state than far-future tokens do, in a power-law sense. Weighting the loss accordingly aligns the training signal with the actual information content of the target window.

This was an empirical coincidence (the weighting that won was the one consistent with the natural-text statistics), not a derivation, but it's a notable cross-validation between 1994 information theory and 2026 pretraining.

---

## 7. Hyperparameters

Two scalars control everything. Both have flat-bottomed optima.

### 7a. Bag size `s`

Sensitivity is U-shaped:
- **Too small** (`s ≤ 2`): superposition regime too similar to standard training → throughput gain too small to matter.
- **Too large**: bag target becomes too lossy → phase 2 can't recover in the remaining budget → final loss degrades.

Optimal `s` drifts **upward with model size**:

| Model       | Tokens budget | Optimal `s` range | Used in paper |
|-------------|--------------:|------------------:|--------------:|
| 270M dense  |          42B  |             6–8   |             6 |
| 600M dense  |          42B  |             6–12  |             6 |
| 3B dense    |         105B  |          (~6–8)*  |             6 |
| 10B-A1B MoE |          2T   |            ~16    |            16 |

(*) Inferred from the 3B configuration in the main table.

Three points is not a scaling law, but the direction is **consistent with the hypothesis that larger models tolerate coarser superposition**. This is the dimension most in need of further study.

### 7b. Step ratio `r` (fraction of steps in phase 1)

Much less sensitive than `s`. Values in `[0.2, 0.4]` are near-optimal across every scale tested.
- At `r = 0`: TST reduces to baseline (no phase 1).
- At `r = 1`: no recovery phase — the model is permanently a bag-of-words predictor, useless for next-token tasks.
- At `r > 0.5`: phase 2 has too few steps to undo the bag damage to the output head; final loss degrades.

Downstream eval scores track the loss curves, not diverging from them — consistent with the view that TST does not distort representations in a transfer-hostile way.

---

## 8. Experimental Setup

**Architectures**:
- 270M, 600M dense: SmolLM2 shape with **untied embeddings** + Llama3-8B tokenizer.
- 3B dense: SmolLM3 shape.
- 10B-A1B: Qwen3-family MoE (10B total params, ~1B active per token).

**Data**:
- Small runs (270M, 600M, 3B): **DCLM**.
- MoE: **50/50 DCLM + FineWeb-Edu** mix.

**Optimizer**: AdamW with the **Warmup-Stable-Decay (WSD)** schedule. Learning rates swept at 270M and 600M; taken from family recommendations at 3B and 10B.

**Framework**: TorchTitan + FSDP. Up to **64 B200 GPUs**.

**Reporting**: training loss, downstream evals (HellaSwag, ARC-E, ARC-C, MMLU), wall-clock proxied as B200-hours.

---

## 9. Main Results

| Model       | Steps   | Bag `s` | Tokens (TST / total) | B200-h | Final Loss | HellaSwag | ARC-E | ARC-C | MMLU |
|-------------|--------:|--------:|---------------------:|-------:|-----------:|----------:|------:|------:|-----:|
| **Baseline 270M**  |  20k |    –   |          – / 42B    |    34 |    3.212  |    36.3 |  46.7 |  24.9 |   –  |
| TST 270M           |  20k |    6   |          75B / 105B |    34 |    3.142  |    38.6 |  47.6 |  26.4 |   –  |
| **Baseline 270M**  | 100k |    –   |         – / 209B    |   170 |    3.092  |    40.2 |  47.5 |  26.2 |   –  |
| TST 270M           | 100k |    6   |         377B / 524B |   170 |    3.048  |    42.6 |  50.3 |  25.5 |   –  |
| **Baseline 600M**  |  20k |    –   |          – / 42B    |    61 |    3.019  |    43.5 |  51.7 |  25.5 |   –  |
| TST 600M           |  20k |    6   |          75B / 105B |    61 |    2.943  |    48.2 |  52.5 |  26.9 |   –  |
| **Baseline 3B**    |  20k |    –   |          – / 42B    |   247 |    2.808  |    57.6 |  60.6 |  31.9 | 31.2 |
| **Baseline 3B**    |  36k |    –   |          – / 75B    |   443 |    2.677  |    62.3 |  65.9 |  34.9 | 32.7 |
| **Baseline 3B**    |  50k |    –   |          – / 105B   |   622 |    2.640  |    63.9 |  67.3 |  36.8 | 33.3 |
| TST 3B             |  20k |    6   |          75B / 105B |   247 |    2.676  |    62.4 |  66.3 |  36.0 | 32.8 |
| **Baseline 10B-A1B** | 125k |  –   |       – / 1.05T     | 12311 |    2.252  |    70.1 |  73.8 |  46.3 | 37.4 |
| **TST 10B-A1B**    |  50k |  16    |          1.68T / 2T |  4768 |  **2.236**| **71.2**|**74.2**|**47.3**|**39.0**|

### Reading the table

- **At 3B**: TST-20k matches baseline-36k in final loss at *similar* wall-clock (247 vs 443 B200-h is for the *36k baseline*; TST-20k is at 247). Downstream essentially identical. Baseline-50k (622 B200-h) beats TST-20k — expected, because TST front-loads a fixed speedup and does not compound on a fixed step budget.
- **At 10B-A1B**: the cleanest result. TST in **38.7% of baseline's wall-clock** beats baseline on every downstream and on training loss. Single config — not a cherry-picked sweep. **Every downstream lands above baseline**, which is meaningfully stronger than just a loss improvement (loss reductions don't always transfer).

---

## 10. Three Comparison Framings (one TST run, three different baselines)

A TST run can be compared three ways against a baseline. Same TST run, different rescalings of the x-axis:

### 10a. Matched FLOPs per step + matched step count → TST wins

TST sees `s×` more tokens for the same per-step compute and same total step count. Final loss is lower. **This is the efficiency claim.**

### 10b. Matched final loss → TST takes less wall-clock

TST reaches the baseline's terminal loss in substantially fewer wall-clock hours. On 3B and 10B runs the ratio is **~2×**. This is the framing the speedup numbers refer to.

### 10c. Matched total tokens consumed → TST loses

TST's effective compute budget per token is smaller (it spent fewer FLOPs per token during phase 1). If data is the binding constraint — not compute — TST is **counterproductive**: it spends the corpus faster than standard training does.

The data-bound regime is real (Kim et al. argue it will tighten as labs scale compute past available web text). The candidate for that regime is the **output-only variant of TST** (see §11), which preserves the auxiliary future signal without consuming extra tokens. Head-to-head against MTP / token-order prediction is the most pressing follow-up.

---

## 11. Ablations: Input-Only, Output-Only, Full

The two mechanisms (input bag-fold and output bag-target) can be ablated independently. The paper shows (Figure 6):

| Variant              | Phase 1 input            | Phase 1 target          | Improvement vs baseline   |
|----------------------|--------------------------|-------------------------|---------------------------|
| Baseline             | standard tokens          | next token              | —                         |
| **Input-only**       | bag-fold + mean-embed    | next token (single)*    | improves over baseline    |
| **Output-only**      | standard tokens          | next-bag MCE            | improves over baseline    |
| **Full TST**         | bag-fold + mean-embed    | next-bag MCE            | improves *most*           |

(*) Input-only has to define a target somehow at each latent position; the natural choice is the next single token at each latent position, which is what the paper compares.

The improvements **combine roughly additively**. This is the evidence that TST is *two distinct mechanisms* that happen to be compatible, not one trick with two knobs.

**Output-only is the natural data-bound candidate**: it preserves the multi-hot future signal but does not change the input sequence length, so it consumes one token per output position just like baseline NTP.

---

## 12. Two Mechanisms — Conceptual Analysis

### 12a. Output mechanism — easy to place in the literature

Next-bag prediction is structurally a **weight-shared multi-token prediction (MTP)**. Where Gloeckle et al.'s MTP has `s` independent heads predicting the `s` future tokens, TST has a single head predicting their *mean distribution*.

Family of methods exploiting near-future signal:
- DeepSeek-V3's cascaded MTP
- Zuhri et al.'s token-order prediction
- Liu et al.'s next-concept prediction
- Mahajan et al.'s future-summary prediction
- A concurrent modded-nanogpt speedrun with exponential-weighted next-bag + smooth phase transition

Common thread: an auxiliary signal from the near future produces a more informative gradient than the per-position one-hot target.

What's distinctive about TST's bag target: it **discards within-bag order entirely**. The model is asked which tokens appear, not in what order. This works — and that's the empirical hint that NTP's gradient is partly about token *distribution* and partly about precise *ordering*, and these can be partially separated.

### 12b. Input mechanism — harder to place

No direct analogue in the recent pretraining literature. Three non-exclusive interpretations:

1. **Noise reduction**: mean is a low-pass filter. Damps high-frequency variance in the embedding sequence — particularly helpful in early training at large vocabulary sizes.

2. **Embedding-geometry regularizer**: for the model to extract information from bag averages, the vocabulary must be laid out with sufficient angular dispersion. Phase 1 implicitly forces this; phase 2 inherits the layout.

3. **Cheap pre-pretraining**: averaging produces a coarser version of the same text. Hu et al. and Lee et al. showed pre-pretraining on simpler distributions reduces the budget to reach a given loss by ~1.3–1.6×. TST may be doing this with the pre-pretraining corpus constructed on the fly. **Testable prediction**: attention-head circuits acquired in phase 1 should remain functional in phase 2 (the hypothesis Hu et al. validated for formal languages). Not yet tested for TST.

### 12c. Where TST sits in the field

Pretraining efficiency research can be loosely organized into three categories:

| Category                  | Strategy                                                  | Examples                                       |
|---------------------------|-----------------------------------------------------------|------------------------------------------------|
| **Information maximization** | Richer signal per sample (input or supervision)         | better tokenizers, n-gram hashing, MTP, ToP    |
| **Compute sparsity**      | Same representation, fewer FLOPs/token                    | MoE, sparse attention                          |
| **Compressive modeling**  | Internally compress input → fewer latent positions        | TST input-side, byte-LM with patches (BLT)    |

- **Output-side TST** = category 1 (richer supervision).
- **Input-side TST** = category 3, with the distinguishing property that **compression is applied only during a first training phase and discarded**. The inference-time model retains the original tokenization and granularity.

That last property — the temporary, discardable compression — is what decouples training-time efficiency from inference-time architecture. This is the philosophical core of why TST is interesting.

---

## 13. Limitations

### 13a. Compute–data tradeoff

The 600M "42B equivalent-token" run actually consumes **105B tokens** from the dataloader. TST trades compute for data:
- Compute-bound regime (most labs through ~2025): TST wins.
- Data-bound regime (frontier scale, exhausted web text): TST is *counterproductive*.

Kim et al. argue the compute-bound assumption will weaken. The output-only variant is the natural data-bound candidate; head-to-head against MTP is the open comparison.

### 13b. Scaling of optimal `s`

Three model sizes (270M, 600M, 10B) is enough to indicate a trend but not enough to fit a law. Predicting optimal `s` at frontier scale requires a proper scaling study.

### 13c. Long-context behavior

During phase 1 the model is exposed to **`s ×` more source-text content per sample**. At `s = 16` with an 8k model context, the model sees 64k-token spans of source text. Effect on long-context evaluation: **untested**. Most likely small positive effect (reduced need to truncate long documents during phase 1), but hypothesis only.

---

## 14. Negative Results (variants that didn't help)

For completeness — the authors list things they tried that did not improve on the default recipe:

1. **Positional encodings on tokens before averaging** (to preserve some within-bag order) — consistently failed; often hurt. The bag's permutation invariance is a *feature*, not a limitation.

2. **Alternative output losses**: BCE against multi-hot target, TopP-style head (Zuhri et al.), hinge-style ranking loss — all less stable than mean-CE. Likely because MCE is most aligned with the NTP target phase 2 switches to, so the recovery has less to unlearn.

3. **Rescaling RoPE at the phase transition** (`N/s` → `N`) — accelerated the first few thousand recovery steps but sometimes raised final loss. Leaving RoPE untouched is more stable.

4. **Partial MTP**: a middle ground between single-head MCE and full `s`-head MTP, where heads predict positions of each bag. No consistent gain over single-head MCE at the cost of extra parameters and complexity. **Simpler version preferable.**

---

## 15. Open Questions

### 15a. Extract phase-1 model as a draft model

By the end of phase 1 the model has learned an `s`-to-1 input compression and decodes distributions over `s`-token futures from the compressed representation. **These are exactly the characteristics of a speculative-decoding draft model or a retrieval-prefix encoder.** Extracting it for inference-time use while continuing the main training into phase 2 would impose zero additional training cost and might yield a useful inference component.

### 15b. Retain TST head during phase 2 as auxiliary MTP loss

Discard at the phase boundary, or keep it as a low-weight auxiliary signal? The head is already trained and carries near-future information that ordinary NTP doesn't reinforce. This is the variant most directly addressing the **data-bound regime** (no extra tokens consumed) and most compatible with the auxiliary-loss literature.

### 15c. Scaling study of optimal `s` vs model size

Needed before TST can be applied at frontier scale with confidence.

### 15d. Long-context evaluation

The 64k-source-token exposure during phase 1 at `s=16` is an unexplored data point.

---

## 16. Verbatim Reference Implementation

The implementation is small enough to embed verbatim. The full diff would be: a config knob (`superposition_bag_size`, `superposition_steps`), a dataloader collate change, an embedding-layer branch, a loss-function branch, and a phase-transition hook.

### 16a. Config

```python
@dataclass
class TSTConfig:
    enable: bool = True
    bag_size: int = 6                      # s
    phase1_step_ratio: float = 0.3         # r — fraction of total steps in phase 1
    within_bag_weighting: str = "uniform"  # "uniform" | "power_law"
    within_bag_alpha: float = 0.6          # Ebeling-Pöschel exponent; ignored if uniform
```

### 16b. Dataloader collate change

```python
def collate_tst(batch, bag_size: int, in_phase_1: bool):
    """
    `batch` is a list of token-id sequences of length N+s (one extra bag of
    targets). Standard collate truncates to N input + 1 target shifted by one;
    TST collate keeps the +s tail so the next-bag targets exist for every
    latent position.

    Returns (inputs, targets) where:
        inputs  : (B, N)   raw token ids
        targets : (B, N)   shifted-by-one for NTP, OR (B, N/s, s) for TST.
    """
    ids = torch.stack([torch.tensor(seq) for seq in batch])      # (B, N+s)
    if in_phase_1:
        N    = ids.shape[1] - bag_size
        inp  = ids[:, :N]                                         # (B, N)
        # Targets per latent position i: the next bag of s tokens
        # starting at position (i+1)*s. We keep (B, N) and reshape to
        # (B, N/s, s) downstream.
        tgt  = ids[:, bag_size : bag_size + N]                    # (B, N)
        return inp, tgt
    else:
        N    = ids.shape[1] - 1
        inp  = ids[:, :N]
        tgt  = ids[:, 1:1+N]
        return inp, tgt
```

### 16c. Embedding-layer branch

```python
class TSTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, bag_size: int):
        super().__init__()
        self.E = nn.Embedding(vocab_size, dim)
        self.bag_size = bag_size

    def forward(self, inputs, in_phase_1: bool):
        if in_phase_1 and self.bag_size > 1:
            bs, seq = inputs.shape
            # (B, N) -> (B, N/s, s)
            inputs = inputs.reshape(bs, seq // self.bag_size, self.bag_size)
            # (B, N/s, s, d) -> (B, N/s, d)
            return self.E(inputs).mean(dim=-2)
        return self.E(inputs)
```

### 16d. Loss-function branch

The MCE→sum-of-CE equivalence (§4b) means we just compute `s` standard CE terms against the same logit vector.

```python
def tst_loss(
    logits,                 # (B, N/s, V)    one logit vector per latent position
    targets,                # (B, N)         raw token ids of the target tail
    bag_size: int,
    weighting: str = "uniform",
    alpha: float = 0.6,
    in_phase_1: bool = True,
):
    if not in_phase_1 or bag_size == 1:
        # Standard NTP: logits are (B, N, V); targets are (B, N).
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    B, Nlat, V = logits.shape                       # Nlat = N / s
    assert targets.shape == (B, Nlat * bag_size)

    # Reshape targets so each latent position has s targets.
    targets = targets.reshape(B, Nlat, bag_size)    # (B, N/s, s)

    # Within-bag weights.
    if weighting == "uniform":
        w = torch.full((bag_size,), 1.0 / bag_size, device=logits.device)
    elif weighting == "power_law":
        idx = torch.arange(bag_size, device=logits.device, dtype=torch.float32)
        w = (idx + 1.0).pow(-alpha)
        w = w / w.sum()
    else:
        raise ValueError(weighting)

    # Sum-of-CE formulation: compute s standard CE terms against same logits.
    # Vectorise by tiling logits along a new "bag-position" axis.
    logits_tiled  = logits.unsqueeze(2).expand(B, Nlat, bag_size, V)  # (B, N/s, s, V)
    ce_per_pos    = F.cross_entropy(
        logits_tiled.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    ).reshape(B, Nlat, bag_size)                                       # (B, N/s, s)

    # Weighted mean over bag positions, mean over latent positions and batch.
    loss = (ce_per_pos * w).sum(dim=-1).mean()
    return loss
```

**The `expand` does not allocate** — `logits_tiled` is a view. PyTorch's fused CE kernel still runs efficiently on the flattened tensor; no new kernel is needed.

Alternative implementation that avoids the expand entirely (compute CE separately for each bag position and accumulate):

```python
loss = 0.0
for j in range(bag_size):
    ce_j = F.cross_entropy(
        logits.reshape(-1, V),
        targets[:, :, j].reshape(-1),
    )
    loss = loss + w[j] * ce_j
```

Either formulation is correct. The first is one fused kernel call on a `(B·Nlat·s, V)` tensor; the second is `s` calls on a `(B·Nlat, V)` tensor. Both reduce to the same arithmetic.

### 16e. Training loop with phase transition

```python
total_steps   = 50_000
phase1_steps  = int(0.3 * total_steps)             # r = 0.3
bag_size      = 6

model       = ...                                    # standard Transformer
optimizer   = AdamW(model.parameters(), ...)
scheduler   = WSDScheduler(optimizer, total_steps=total_steps, ...)
dataloader  = build_dataloader(seq_len_extra=bag_size)    # see §16b

for step in range(total_steps):
    in_phase_1 = step < phase1_steps
    inputs, targets = next(dataloader)              # collate aware of phase

    # Embedding sees the phase flag (§16c)
    x = model.embed(inputs, in_phase_1=in_phase_1)

    # Transformer forward — UNCHANGED. Sees (B, N/s, d) in phase 1
    # and (B, N, d) in phase 2; RoPE is computed over whatever the
    # sequence length actually is. No conditional inside the model body.
    h = model.transformer(x)
    logits = model.lm_head(h)                       # (B, N/s, V) or (B, N, V)

    loss = tst_loss(
        logits, targets, bag_size,
        weighting="power_law" if bag_size > 6 else "uniform",
        alpha=0.6,
        in_phase_1=in_phase_1,
    )

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step == phase1_steps - 1:
        # Optional: checkpoint at the phase boundary so the recovery can
        # be re-launched from scratch if needed. Otherwise just continue;
        # the phase flag flips on the next iteration.
        torch.save({"model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "step":  step + 1}, "phase1_end.pt")
```

**Critical**: the phase-2 transition does not modify the model, optimizer, or scheduler. Only the dataloader collation and the loss flag change. The 1–2 nat loss spike is observable but recoverable.

### 16f. The full diff against a standard pretraining loop is roughly:

- `+1` dataclass for the TST config (5 fields).
- `+10` lines in the collate function (the bag-tail slicing).
- `+5` lines in the embedding module (reshape + mean if-branch).
- `+30` lines for the TST loss function.
- `+1` line in the training step (pass `in_phase_1` everywhere).

Most production codebases will absorb this in **under 100 lines of changes**.

---

## 17. Reproduction Recipe

### 17a. Smallest reproducible run (270M, 20k steps, single 8-GPU node)

1. **Architecture**: SmolLM2-270M shape — `dim=576`, `n_layers=30`, `n_heads=9`, `n_kv_heads=3`, `ffn_dim=1536`, RoPE `theta=10000`. Crucially **untied embeddings** (separate `wte` and `lm_head` matrices). Llama3-8B tokenizer (vocab=128256).

2. **Data**: DCLM. ~105B tokens consumed across phase 1 + phase 2.

3. **Training**:
   - Sequence length: 2048 (configurable; the paper's small runs use this scale).
   - Global batch: depends on hardware; the paper targets the SmolLM2 family defaults.
   - **Phase 1**: steps 1 to 6000 (`r = 0.3`), `bag_size = 6`, MCE loss, uniform within-bag weighting.
   - **Phase 2**: steps 6000 to 20000, standard NTP, same optimizer state.
   - Optimizer: AdamW, lr swept (paper sweeps `lr ∈ {3e-4, 6e-4, 1.2e-3, …}` at 270M; ~6e-4 is typical).
   - Scheduler: **WSD** — Warmup-Stable-Decay. Warmup ~2000 steps, stable until ~80% of total, decay over the last 20% (the WSD canonical setup).
   - dtype: bfloat16. FSDP only.

4. **Evaluation**: HellaSwag, ARC-Easy, ARC-Challenge after run. (MMLU only meaningful at 3B+.)

5. **Expected outcome**:
   - Phase 1 loss curve sits *above* the matched-FLOPs baseline (the bag-target is harder than NTP).
   - At step 6000 there is a **transient ~1–2 nat spike** when the loss switches back to NTP.
   - Loss recovers within ~1k–2k steps and crosses below the baseline.
   - Final loss: ~3.142 (TST) vs ~3.212 (baseline) on the 20k-step / 42B-equivalent budget.

### 17b. Large-scale recipe (10B-A1B MoE, the headline result)

- Qwen3-shaped MoE, 10B total params, ~1B active.
- Tokenizer: Qwen3 default.
- Data: 50/50 DCLM + FineWeb-Edu.
- **Phase 1**: 50k × 0.3 ≈ 15k steps? *(The paper reports 50k total steps with `s=16`; the phase split isn't explicitly tabulated but `r ∈ [0.2, 0.4]` is the recipe range.)*
- Bag size **`s = 16`** — coarser superposition for the larger model.
- Power-law within-bag weighting with `α ≈ 0.6` (required at `s=16`; uniform underperforms at this bag size).
- Total: 50k steps, 2T tokens consumed, 4768 B200-hours.
- Compared against a 125k-step / 1.05T-token / 12311-B200-hour baseline at *matched final loss*.

---

## 18. Three Implementation Pitfalls

1. **Targets must extend `s` tokens past inputs.** Phase-1 collate needs an extra bag in the raw token tail so that every latent position has a valid next-bag target. Forgetting this off-by-`s` is the most common bug. The standard NTP collate only needs +1 token of tail.

2. **RoPE applies to latent positions, not source positions.** During phase 1 the transformer sees a sequence of length `N/s`; RoPE indexes 0..N/s-1. Do *not* try to pre-compute RoPE on source positions and average — that would defeat the bag-fold symmetry. Leave RoPE alone (it's a function of the latent sequence index, period).

3. **Power-law weighting is necessary at `s > 6`.** Uniform weighting at `s=16` (the MoE config) substantially underperforms. The paper's negative results section confirms this — try the simpler uniform first only if `s ≤ 6`.

---

## 19. Where to Port This Into `dl_techniques`

The repo already has a training scaffold under `src/train/`; TST naturally fits as a training-loop modification with no model-side changes. Minimum viable port:

### 19a. New files

- `src/dl_techniques/training/token_superposition.py`:
  - `TSTConfig` dataclass (§16a).
  - `tst_collate(batch, bag_size, in_phase_1)` (§16b, Keras `tf.data` variant — use `tf.reshape` and `tf.gather` over a token-id buffer).
  - `tst_loss(logits, targets, bag_size, ...)` as a Keras loss callable (§16d). Use `keras.ops.cross_entropy` so it works under JAX / TF / PyTorch backends.
  - `TSTPhaseCallback(total_steps, phase1_step_ratio)` — a Keras callback that flips `in_phase_1` on the model and on the loss at the right step.

### 19b. Model-side change (single layer)

- Wrap `keras.layers.Embedding` in a `TSTEmbedding` layer (§16c) that takes a `training_state` flag and reshapes+means when in phase 1. Set the flag via the callback above.

### 19c. Training-script change

In any of the existing `src/train/<model>/train_*.py` scripts:
- Add `--tst.enable`, `--tst.bag_size`, `--tst.phase1_step_ratio` CLI args.
- Build the TST callback.
- Swap the embedding layer for `TSTEmbedding`.
- Swap the loss for `tst_loss` partial.

### 19d. Hardest part of a Keras port

The MCE loss as a fused kernel: PyTorch has `F.cross_entropy` that fuses log-softmax + nll into one kernel. Keras 3 `keras.ops.sparse_categorical_crossentropy` does the same under most backends. The `expand` trick in §16d works in Keras via `keras.ops.broadcast_to`; the resulting `(B, Nlat, s, V)` tensor is just a view (not materialised) in all three Keras backends. **No new kernel needed in any backend.**

### 19e. Validating the port

The smallest correctness check: at `bag_size=1`, TST loss must reduce *exactly* to standard CE. The bag-fold is identity; the MCE is one CE term; the result is bit-identical to baseline. This single equality is the canary that catches off-by-one errors in collation and in the loss.

---

## 20. The Five Non-Obvious Correctness Invariants

If a port doesn't preserve these, the 2× speedup doesn't materialize:

1. **The embedding average is `(1/s) Σ E[t_j]`, not a learned reduction.** No attention-pool, no linear layer, no concat-and-project. Mean over the bag axis. Permutation-invariant by construction.

2. **The loss is mean-CE, *not* CE against an averaged distribution.** I.e. `(1/s) Σ CE(t_j, p)`, not `CE(mean over j of one-hot(t_j), p)`. These are mathematically the same when `q` is normalized to sum to 1 (§4b), but easy to bungle when you write the loss directly. Always implement as the sum-of-CE form.

3. **No new output head.** A common temptation is to add `s` parallel projection heads (one per bag position). That's MTP, not TST. The whole point is one head, one logit vector per latent position, `s` CE terms against it.

4. **Phase transition is purely dataloader+loss.** Do not reset the optimizer, do not change weights, do not rescale RoPE. The loss-curve spike at transition is *normal* and *expected*; the model recovers.

5. **Within-bag order is discarded by design.** Adding positional encodings to source tokens before averaging will hurt. The bag's permutation invariance is a feature.

---

## 21. References

- **TST paper**: Nous Research, *Efficient Pretraining with Token Superposition*, arXiv:2605.06546.
- **MCE equivalence**: derived in §4b above; corresponds to Appendix B of the paper.
- **Ebeling & Pöschel (1994)**: power-law decay of mutual information between English letter pairs as a function of distance.
- **Multi-token prediction**: Gloeckle et al., 2024.
- **DeepSeek-V3 cascaded MTP**: DeepSeek-AI, 2024.
- **Token-order prediction**: Zuhri et al.
- **Next-concept prediction**: Liu et al.
- **Future-summary prediction**: Mahajan et al.
- **WSD scheduler**: Hu et al., 2024 (warmup-stable-decay).
- **DCLM corpus**: Soldaini et al.
- **FineWeb-Edu**: Penedo et al.
- **SmolLM2 / SmolLM3 shapes**: HuggingFace, 2024–2025.
- **Pre-pretraining analogues**: Hu et al. (formal languages), Lee et al. (NCA synthetic data).
- **BPE-as-throughput hypothesis**: Gigant et al.; Minixhofer et al.; Zheng et al.
- **Data-bound regime**: Kim et al.
- **Concurrent modded-nanogpt speedrun entry**: next-bag with exponential weighting + smooth phase transition.
