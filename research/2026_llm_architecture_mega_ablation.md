# I Tried Every LLM Architecture Change Since 2020

**A self-contained edition of Nikita Pavlichenko's X Article (published 30 October 2026), with every referenced technique, model, paper and tool expanded.**

| | |
|---|---|
| Author | Nikita Pavlichenko (@nv_pavlichenko), pre-training at Cohere, previously head of Mellum at JetBrains |
| Original venue | X (Twitter) Article, 30 Oct 2026 |
| Framework | bux, `github.com/pilot7747/bux` (JAX, Apache-2.0) |
| Open data | Hugging Face `pavlichenko/bux-timeline` (CC BY 4.0) |
| Scale | 1,129 training runs, about 1.1T training tokens, 5,232 vLLM serving measurements |

> **Editorial notes.** Part I reproduces the article's content in full, reorganized for reading without the original plots. The plots themselves were images and are not reproduced; their numbers are carried in the text and tables. Part II expands every technique and reference the article links to. Part III is a bibliography. Reference expansions for 2026 techniques are taken from the bux repository's own documentation and experiment files (which cite the source of every arm), since many of them post-date widely available literature.

---

## Contents

- [Part I. The Study](#part-i-the-study)
  - [1. TL;DR](#1-tldr)
  - [2. Motivation](#2-motivation)
  - [3. Setup](#3-setup)
  - [4. How to read the results](#4-how-to-read-the-results)
  - [5. Individual changes](#5-individual-changes)
  - [6. Stacking round 1](#6-stacking-round-1)
  - [7. Stacking round 2](#7-stacking-round-2)
  - [8. Stacking round 3](#8-stacking-round-3)
  - [9. The final architecture](#9-the-final-architecture)
  - [10. Full architectures compared](#10-full-architectures-compared)
  - [11. Summary of conclusions](#11-summary-of-conclusions)
  - [12. Side note: how good AI is at research](#12-side-note-how-good-ai-is-at-research)
  - [13. Reproduction](#13-reproduction)
- [Part II. Expanded References](#part-ii-expanded-references)
- [Part III. Bibliography](#part-iii-bibliography)

---

# Part I. The Study

## 1. TL;DR

- The best architecture change since GPT-3 is **fine-grained Mixture-of-Experts (MoE)**. Linear attention and hyper-connections also work.
- Not literally every change, but close.
- Most other changes (norms, MLPs, positions, residual tricks) are worth 1-2% at best once evaluated at a realistic serving size, often nothing.
- On a late-2026 model, only about 2% of predicted loss is left for architecture to win; in 2021 it was about 11%.

## 2. Motivation

Architecture research is high-prestige work: it takes up a large share of every tech report, and carries a mystique (the image of researchers reading raw tensor dumps, spotting norm blowups, and inventing things like manifold-constrained hyper-connections).

Comparing architectures properly is hard for three reasons:

1. **No clear definition of "better".** Better loss per parameter? Cheaper to serve? Cheaper to train? Some combination?
2. **Results are setup-dependent.** Signals flip with how much long context matters, the data, the hardware, and the shape of the inference load.
3. **Results transfer loosely with scale.** Every new nanoGPT-speedrun record invites the question "does it scale?".

There is also a view that architecture does not matter much and that data and the pre-/post-training setup are where the wins are: rumors that frontier labs train very boring transformers, "pre-training is dead", and the "old data + modern arch vs. modern data + old arch" debate.

The goal: put six years of open-lab architecture decisions into one setup, test each change alone, then stack the winners to see how much survives. The author had planned this "mega-ablation" for over a year; coding agents made it feasible in reasonable time. All models were trained with **bux**, the author's framework, which defines an architecture, trains it, exports it to vLLM, and measures both quality and inference speed.

## 3. Setup

### 3.1 Data

- **Code:** individual code files and whole Python repositories from **Stack-Edu**.
- **Web:** **FineWeb-Edu**.
- **Tokenizer:** GPT-NeoX (EleutherAI/gpt-neox-20b), vocabulary 50,304.

### 3.2 Model shapes

- Every model keeps **GPT-3's proportions**: one layer per 64 of width (`layers = d_model / 64`), attention heads of dimension 64.
- Up to four widths: `d_model` = 512, 768, 1024, 1280.
- For the GPT-3 baseline that is **25M to 394M non-embedding parameters**.

### 3.3 Token budget

- At a given width every model sees **the same tokens**: 20 per non-embedding parameter of the GPT-3 baseline at that width (the Chinchilla ratio).
- Budgets: **0.5B, 1.7B, 4.0B and 7.9B tokens** for d=512, 768, 1024, 1280.

### 3.4 Training recipe

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.1 |
| Schedule | Warmup-Stable-Decay (WSD): 5% warmup, 20% decay (sqrt shape) |
| Auxiliary loss | z-loss 1e-4 |
| Batch size | DeepSeek's batch-size scaling law: 48 to 288 rows of 2k tokens |
| Precision of serving | FP8 weights, activations and KV cache |

### 3.5 Long-context curriculum

| Phase | Share of steps | Row length | Data mix |
|---|---|---|---|
| Main | first 90% | 2k tokens | 60% code, 40% web |
| Extension | last 10% | 16k tokens | 50% whole repositories, 30% code, 20% web |

At the switch, every full-attention layer that uses RoPE gets **YaRN** with an 8x factor (2k to 16k).

### 3.6 Hardware

- NVIDIA **B300** GPUs.
- Smallest run: about 10 minutes on one GPU. Largest: about 14 hours on four.

### 3.7 Learning-rate sweep

- Each architecture gets a short sweep at d=512: **1e-3, 2e-3, 4e-3**, plus **8e-3** for changes that claim to help stability.
- A parabola is fit in log learning rate; its minimum is reused at wider models.

### 3.8 Quality metric

- **Bits per byte (BPB)** on held-out data, weighted: **0.5 x code files + 0.3 x whole repositories + 0.2 x web text**.
- Repositories are read in 16k-token windows, so long context counts.
- Lower is better. "-2%" means 2% fewer bits per byte.

### 3.9 Noise control

- Changing only the seed moves the objective by about **0.7%**.
- Runs with the same seed see the same data in the same order, so comparisons always pair runs that share a seed. This cuts the noise of a comparison to about **0.3%**.

### 3.10 Equal serving cost

Motivation (from the author's work on Mellum at JetBrains): the question is how good a model can be **within a fixed inference budget**. Every architecture gets the same serving budget; if a change makes serving cheaper, the savings buy a bigger model.

**Serving baseline:**

- GPT-3 6.7B shape: `d_model` 4096, 32 layers, full multi-head attention.
- vLLM 0.32 with bux's plugin, FP8 weights/activations/KV cache, one B300.
- **Workload** (approximates a coding agent): each session gets a shared 2k-token prefix, 14k tokens of its own context, then 4 turns in which the model writes 512 tokens and receives a 2k-token tool result. Prefix caching is on.
- **Throughput metric:** sessions per hour per GPU while median time per output token stays under 30 ms.
- Baseline: **1,924 sessions/h**. Each architecture must match this, and the study measures how large a model it can afford at that rate.

### 3.11 From small models to a prediction

Per architecture, two to four small models are trained: d=512 and d=768 for every change, plus d=1024 and d=1280 for promising ones. A scaling law then predicts quality at the size that matches the serving budget.

**Step 1: measure the target inference size.** Each architecture is exported from bux without weights at 3-4 sizes and timed in vLLM under the same load. Interpolation gives the size that serves exactly 1,924 sessions/h per GPU: the architecture's *target*.

| Model | Target size at equal serving cost |
|---|---|
| 2021 reference (GPT-3 + RoPE) | 3.7B active |
| GQA variant | 13.8B |
| 2026 KDA hybrid | 11.4B active (201B total) |

An analytic cost model was tried first. It was right on the median but far off wherever kernels matter (sparse attention, MLA), so it is used only to choose which sizes to time.

**Step 2: fit the law.**

$$L(N, D) = E + A \cdot N^{-\alpha} + B \cdot D^{-\beta}$$

- `N`: active non-embedding parameters. `D`: training tokens.
- `E` (entropy of the data) is fixed from an earlier ladder on the same data.
- `B` and `beta` are shared across all families (same data).
- `A` is always family-specific. `alpha` is family-specific only with 3+ trained widths; otherwise it is borrowed from the parent architecture.

**Step 3: report two numbers per change.**

- **"Worth N times its parameters"**: compares models at the trained sizes.
- **"Change at equal serving cost"**: reads the law at the target size and **1T tokens**.

These two often disagree.

### 3.12 Author's disclaimer

Do not treat this as a rigorous scientific study:

- Serving cost depends heavily on how well vLLM is optimized for each architecture.
- The scaling ladder is tiny and the extrapolation to the target size is long.
- Data and benchmarks are toy-scale; the training setup is far from production (e.g., context extension only to 16k).
- Not all hyperparameters were tuned, nor all method options swept.
- Much of the code was written by Claude Opus 5.5.
- It was intended as a fun evening project.

## 4. How to read the results

### 4.1 Plot panels (described)

Each result figure in the original has up to three panels:

1. **Measured:** the change vs. the model without it, same width and token budget. Bars go d=512, 768, 1024 (light to dark).
2. **Serving:** how many active parameters can be served at the baseline's cost. A dashed line marks the model before the change.
3. **Predicted:** the scaling-law prediction at that serving size after 1T tokens. Blue: the whole predicted range is below zero (improvement). Red: entirely above zero. Gray: range crosses zero. **Hollow dots** mark changes trained at only two widths that borrow the parent's scaling exponent; treat them with particular caution.

All "loss change at equal serving cost" figures below come from the third panel (an extrapolation, with all the caveats above).

### 4.2 The three reference models

| Name | Definition | Role |
|---|---|---|
| **2021 reference** | GPT-3 with RoPE instead of learned positions | Baseline for all individual changes |
| **2026 hybrid** | Small version of a late-2026 architecture: 3 KDA layers per global attention layer, fine-grained MoE, QK-norm, gated attention, MTP | Starting point for stacking |
| **Candidate** | 2026 hybrid + hyper-connections (HC, 4 residual copies) + value residual (V) + Inkling's key-value conv (K) on the global layers | Base for later stacking, sparse attention and Canon experiments |
| **Final model** | Candidate with its dense FFN moved from the first layer to the last | Result of the study |

## 5. Individual changes

### 5.1 Overview ("political coordinates" plot)

Each change to GPT-3 + RoPE is a dot: further right means better quality per parameter; further up means cheaper to serve (a bigger model for the same money).

- **Fine-grained MoE** and **linear attention** stand out.
  - With MoE, an active parameter is worth about **4-5 dense parameters**.
  - Linear-attention hybrids improve quality per parameter **and** are **2-7x cheaper to serve**.
- **GQA, MQA, MLA and sliding windows** mainly cut the KV cache. Quality per parameter is slightly worse, but a **4-9x bigger model** can be served. In this setup, a good trade.
- **Everything else** clusters together. Norms, MLPs, positions and residual changes barely affect serving cost. Some let a small model match one with 1.2-2x the parameters, but at the target serving size that becomes a 1-2% improvement at best, often nothing.

**Best individual changes** (predictions vs. GPT-3 + RoPE, equal serving cost, 1T tokens):

| Change | Predicted change in loss |
|---|---|
| LatentMoE | -5.7% |
| DeepSeek-V3 MoE (128 experts, sigmoid routing) | -5.5% |
| KDA (one global attention layer in 8*) | -4.8% |
| KDA in 3 of 4 layers | -4.7% |
| DeepSeekMoE (64 experts + shared) | -4.2% |
| Gated DeltaNet in 3 of 4 layers | -4.2% |
| MLA (at DeepSeek's dimensions) | -4.1% |
| 512-token sliding windows (one global layer in 8*) | -3.9% |
| Mamba-2 in 3 of 4 layers | -3.6% |
| GQA (4 query heads per KV head) | -3.2% |

\* One global layer in 8 at d=512, in 6 at d=768.

### 5.2 Positional encodings

All rows compare against GPT-3's learned position table. Positions do not change serving cost.

**RoPE went against the author's priors.**

| Width | RoPE vs. learned positions |
|---|---|
| d=512 | -6.9% |
| d=768 | -4.0% |
| d=1024 | -2.3% |
| Equal serving cost (predicted) | +0.5% (range -1.0% to +2.1%) |

The author questions the setup rather than concluding learned positions are fine: their repository loss swings from 0.65 to 0.80 across learning rates. GPT-3 + RoPE is used as the reference anyway.

**RoPE variants barely matter.** Base 500k, partial RoPE, iRoPE, and raising the base at the 16k switch all land within 1% of plain RoPE at trained sizes. BPB is probably not the right tool to separate them; something like the RULER long-context benchmark would be.

**Beware the hollow -2.5% dots.** Those variants were trained at only two widths, so the fit assumes their advantage over the reference is constant with size. Plain RoPE, trained at four widths, visibly shrinks; the variants likely do too. Base 500k illustrates the effect: it helps at every trained width but shrinks fastest, and its prediction ends at +0.5%.

**YaRN matters for using the longer context.** Without it, RoPE beats learned positions by 5.3% in the first 2k tokens of a repository but only 0.6% past 8k.

**Removing positions (NoPE).**

- Everywhere: the model breaks (+100% at d=512).
- Only in the global layers of a hybrid: fine, because local layers still provide position information. 2026 hybrids do this; adding partial RoPE back to their global layers is slightly worse (+0.2%).
- Conclusion: some positional inductive bias is needed, but not in every layer (a reproduction of a known result).

### 5.3 Normalization

Most changes here were introduced for speed or stability, and show **no quality gain**: RMSNorm, removing biases, sandwich norms, logit soft caps.

**QK-norm is the exception.**

| Width | QK-norm |
|---|---|
| d=512 | -1.1% |
| d=768 | -0.9% |
| d=1024 | -0.9% |
| d=1280 | -0.8% |
| Equal serving cost | -0.8% (range crosses zero) |

- The reference model has no loss spikes, so the gain is not from fixing an obvious instability.
- But attention sharpens very early: by step 500, the largest attention logit is **71 at lr 2e-3** and **351 at 4e-3**. With QK-norm it is **12-25**.
- **Ablating the learned per-channel scales** of the query/key RMSNorms removes the gain: +0.3% and +0.6%, slightly worse than no QK-norm. Letting the model *learn* how sharp attention should be is the useful part. Soft-capping logits to the same range does not substitute for those scales.

**Other norm results:**

- **Post-norm alone**, `x + norm(f(x))`: much worse (+19%, +27%). With QK-norm, as in OLMo 2, it is fine.
- **DyT** does not work with the usual initialization. With embeddings initialized at std 0.02, `tanh(alpha x)` gets tiny inputs; 2 of 3 seeds diverge at d=512 (+74%). Multiplying embeddings by sqrt(d) helps but is still 3.1% worse.

**Learning-rate tolerance.** QK-norm handles a higher learning rate well: one tested learning rate above its optimum barely hurts. Sandwich norms, gated attention and hyper-connections blow up at 8e-3.

### 5.4 MLP variants

All three alternatives beat the reference's plain **GELU MLP (4x wide)** at equal parameter count. Gated MLPs are narrower (8/3 width) to keep parameters equal; serving cost barely changes.

| Variant | d=512 | d=768 | d=1024 | Equal serving cost |
|---|---|---|---|---|
| Squared ReLU | | | | -1.5% |
| SwiGLU | -0.07% | -0.66% | -0.92% | -1.4% |
| GeGLU | | | | -0.5% |

- SwiGLU's d=512 result would look like noise if only the smallest model were run.
- **Squared ReLU wins in the dense model but not as an expert**: inside the 2026 hybrid's experts it is 0.7% worse, even with 5% more active parameters. A good dense MLP is not automatically a good expert (a second width would be needed to rule it out).

### 5.5 Softmax attention variants

| Variant | Quality per parameter | Equal serving cost |
|---|---|---|
| GQA | about +1.1% at every width | -3.2% |
| MQA | +1.6 to +2.5% | -2.5% |
| MLA | within noise | -4.1% |

- They cut the KV cache 4x or more, so the same money buys a **4-9x bigger model**. That is why they win.
- **MLA caveat:** trained with a latent of 256 and 32 rotated dimensions, which runs about 2x slower in vLLM than DeepSeek's 512 + 64 (better kernels). DeepSeek's dimensions were used for the serving comparison. For small MLA models you may need custom kernels.

**Gated attention** (sigmoid gate on each head's output):

- Alone: -2.1%, -1.4%, -1.5% at the three widths.
- With QK-norm: -3.6% and -2.6%, slightly more than the sum of separate gains.
- Pair at equal serving cost: -2.0%.

**Inkling's key-value conv** (4-tap causal conv over keys and values): -1.0%, -0.7%, -0.7%. See [5.9](#59-short-convolutions).

**No measurable difference:** attention sinks, heads of 128.

**Context length changes the pick** (largest servable model at equal cost):

| Variant | 16-24k context | 256k context |
|---|---|---|
| MQA | 33B (largest) | 8B |
| GQA | 14B | 6B |
| Sliding windows, KDA, Mamba-2 | about 14-16B | about 14-16B (stable) |

### 5.6 Hybrids: sliding windows vs. linear attention

**Linear attention is "meta" here.** Replacing 3 of 4 attention layers:

| Mixer (3 of 4 layers) | Measured (d=512 / 768 / 1024) | Served params vs. baseline | Equal serving cost |
|---|---|---|---|
| KDA | -3.7% / -3.4% / -3.4% | 3.6x | -4.7% |
| Gated DeltaNet | | | -4.2% |
| Mamba-2 | | | -3.6% |
| RWKV-7 | | | -2.6% |

**Sliding-window attention (SWA)** helps through serving cost only:

- 512-token windows in 3 of 4 layers: quality per parameter 0.9-1.7% worse, but the larger affordable model is predicted **-2.8%**.
- 128-token windows: cheaper still, **-3.6%**.

**Most of linear attention's lead disappears in full models.** Once MoE, QK-norm and gates are added to both the KDA and SWA models, KDA is only **0.3% better** at 16-24k and **0.45% better** at 256k (equal serving cost). Most of the gain comes from shared components. This may be why Western open labs have not rushed to adopt linear attention.

**KDA ratio:** 3-in-4 vs. all-but-one-in-8 (one in 6 at d=768) are within half a point at trained sizes; predictions -4.8% vs. -4.7%. More KDA layers allow more active parameters, but the quality trade cancels it. KDA in every other layer does worse.

**2048-token sliding window failure mode:** only 0.9% worse in the first 2k tokens of a repository, but **9.3% worse past 8k**. For the first 90% of training a 2048 window on 2048-token rows is identical to full attention; at the 16k phase those layers suddenly see only 2048 of 16k keys.

**Gated DeltaNet-2** is best per parameter at d=512 (-4.7%), but vLLM runs it at **0.42x KDA's speed**, so its equal-cost prediction falls to -2.9%. Worth revisiting with better kernels.

**LFM2's short conv** is cheaper still but loses more quality (see 5.9).

### 5.7 MoE variants

Every MoE uses the **same MLP compute per token** as the dense model.

**Fine-grained MoE is the easiest win in the study.**

| Variant | d=512 | d=768 | d=1024 | Equal serving cost |
|---|---|---|---|---|
| DeepSeek-V3 style (128 experts, sigmoid routing) | -4.9% | -5.6% | -6.2% | -5.5% |
| LatentMoE | -4.2% | | -6.25% | -5.7% (best single change) |
| Mixtral style (8 experts, top-2) | about half of DeepSeek-V3's gain | | | |
| DeepSeekMoE (64 experts + shared) | | | | -4.2% |

- Unlike most small changes, fine-grained MoE's gain **grows with width**.
- **LatentMoE** (Nemotron 3, Kimi K3): projects each token to half the model width, runs it through 12 of 256 experts, projects back. Router and shared expert stay at full width. This allows twice as many experts as the DeepSeek-V3-like model at the same total parameters. Behind at d=512 (-4.2% vs. -4.9%), tied at d=1024 (-6.25% vs. -6.23%).
- **Serving cost per active parameter is higher for MoE**: with 32 concurrent agent sessions, each decode step reads many experts' weights. DeepSeek-V3 MoE affords 3.0B active (33B total) vs. 3.7B dense, and still wins by 5.5%.

**Routing details matter little.** On the 2026 sliding-window model, none of these made a measurable difference: softmax with balance loss instead of sigmoid with bias, removing the shared expert, SMEBU, sqrt-softplus scores.

**On the 2026 hybrid:**

| Change | Effect |
|---|---|
| Experts 256 -> 128 (still 7 per token) | halves expert weights read during decode; +0.5% at d=512 |
| Zero-compute "null" experts | +0.2% |
| Squared-ReLU experts | +0.7% |

**LatentMoE did not carry over to the candidate** (which already has fine-grained experts): 512 experts, 12 per token, at half width gave -0.13% at d=512 (3 seeds) and -0.11% at d=768 (2 seeds), within noise. Serving drops to 0.88-0.94x the candidate's sessions per GPU, about 9% fewer affordable active parameters. On the 2021 reference, LatentMoE's big win was over a *dense* MLP. No refit was done, but with no measured gain and a smaller servable model, it is hard to see a win. **The final model keeps ordinary experts.**

Tuning caveat: on the 2021 reference, LatentMoE's best learning rate was the lowest tried (1e-3), but 2e-3 was only 0.02% worse.

### 5.8 Short convolutions

Four kinds appear in the study:

1. **Inkling's key-value conv:** 4-tap causal conv over an attention layer's keys and values.
2. **Canon layers** (Allen-Zhu, *Physics of Language Models* 4.1): 4-tap conv with its own skip, before the mixer (Canon-A) or before the MLP (Canon-C).
3. **Convs inside linear attention:** KDA and Gated DeltaNet run a 4-tap conv over their own queries, keys and values (part of the mixer; not ablated separately).
4. **LFM2's gated short conv:** an entire mixer layer built around a 3-tap conv, replacing attention or linear attention.

**Inkling's KV conv on the 2021 reference:** -1.0%, -0.7%, -0.7%. Reduces servable size about 4% (3.7B to 3.5B). Equal serving cost: -0.3% (range crosses zero). With QK-norm: -2.3% and -1.6% (about additive), -1.2% at equal serving cost. On the 2026 hybrid: -0.7% on the global layers. It is "K" in stacking round 1 and stays in the final model.

**Canon layers on the candidate** (d=512 only; Canon-A's serving cost is an analytic estimate):

| Variant | Effect |
|---|---|
| Canon-C | +0.07% |
| Canon-A | +0.78% |
| Both | +0.43% |
| Both, with AttnRes replacing HC | +0.59% |

The candidate already has convs in every layer (inside KDA in 3 of 4 layers, over keys and values in the rest). Canon's claims (reasoning depth, NoPE matching RoPE) concern models without those convs, which was not tested. Not a general verdict on Canon.

**LFM2 short conv** (`short_conv` in bux: project input to B, C, u; compute `C * conv(B * u)` with 3 taps):

*Replacing 3 of 4 attention layers in the 2021 reference:* loss +2.1% and +2.3%, but very cheap to serve: 17B affordable parameters vs. KDA's 13.3B. Prediction -2.7%, about equal to sliding windows (-2.8%), well behind KDA (-4.7%).

*Replacing the candidate's KDA layers:* much worse: +11.5% at d=512, +8.3% at d=768, +14% past 8k tokens. It serves about 1.8x the sessions per GPU at equal width, has 20% fewer active parameters, and a state of only **0.9 MB per sequence vs. KDA's 82 MB**. At equal serving cost this only raises the model from 9.1B to 10.4B, not enough. Prediction **+1.45%**, similar at every context up to 256k.

*LFM2's own layout* (GQA in 3 of 8 layers, as LFM2-1.2B): +8.5% and +6.5%. Adding RoPE to global layers: +7.3% and +5.8%. Conclusion: at these sizes a short conv cannot replace what KDA keeps in its state.

### 5.9 Residual streams

| Change | Measured (d=512 / 768 / 1024) | Serving effect | Equal serving cost |
|---|---|---|---|
| Hyper-connections (4 streams) | -2.4% / -2.2% / -1.3% | -10% sessions/GPU on reference (FP8, fused kernels); -20% on 2026 hybrid | +0.2% on reference |
| AttnRes | -1.6% / -1.2% / -1.2% | small | -0.7% |
| Value residual | -2.1% / -0.8% / -1.0% | none | 0.0% |
| mHC | d=512 only on reference (analytic serving cost) | on 2026 hybrid, 19% fewer parameters than HC | |

- On the reference, HC's serving cost cancels its gain. **On the 2026 hybrid HC became the biggest addition of round 1**, paying for its serving cost. Stopping at the 2021 reference would have wrongly ruled it out.
- **Two bugs** initially broke results: an initialization bug collapsed mHC's four streams into one; the gated residual (`gr`) trained to loss 2.33 (vs. about 0.73 for similar runs) because an XLA rewrite on GPUs halved its residual branch. Reported results are after both fixes.

### 5.10 Sparse attention

Sparse attention was put on the **candidate's global layers** (one in four). Training and quality stop at 16k; 64k-256k columns only measure serving speed of the same model.

- **DSA, QSA and MSA** do not measurably hurt quality, here or on the 2021 reference with every layer sparse.
- **CSA** (DeepSeek-V4's compressed sparse attention) is the exception: **+8.3%** on the reference.

**The cost problem at 16-24k:** dense FP8 attention is already cheap and only 1 layer in 4 uses it, so running the indexer costs more than sparsity saves. Affordable size relative to the candidate: **0.82x (DSA), 0.72x (QSA), 0.70x (MSA)**.

**At longer contexts:** at 256k, QSA allows a 1.05x larger model. **IndexShare** (indexer in one global layer of every 2 or 4, picks reused by the others) reaches **1.33-1.37x at 128k** (noisiest measurements, plus or minus 10%). All quality predictions at equal serving cost are within 0.3% of the candidate.

**Index sharing stability:** sharing an index across four layers failed on 1 of 3 seeds: at d=768 that seed was 6.5% worse past 8k because the single indexer drifted. **IndexCache's cross-layer distillation loss** fixed it, with no measurable quality change on all three seeds.

**Prefill anecdote:** at 175k tokens, DSA's prefill indexer alone took **3.2 s**, while dense FP8 attention took **2.3 s**. Choosing tokens cost more than attending to all of them.

### 5.11 Other changes

- **Multi-token prediction (MTP):** no measurable BPB improvement. Mellum2's pre-training ablations showed the same; gains appeared only on actual benchmarks. MTP's speculative-decoding benefit is not measured here.
- **Engram** (DeepSeek's lookup memory): no help at these sizes.
- **Untying the output head:** worse by 2.3% and 1.1%.
- **Parallel attention and MLP:** worse by 1.0% and 1.4%. Possibly worth it if training speed is the priority.
- **Deeper and thinner:** same quality per parameter, but more layers serve slower, so the prediction at equal serving cost is 1.5% worse.

## 6. Stacking round 1

**Question:** does a change still help once the model already has everything else?

**Starting model: the 2026 KDA hybrid**, a small version of the architecture used by Kimi K3, GLM-5.3-Flash, Qwen3.8 and Nemotron 3:

- 3 KDA layers per global attention layer.
- Global layers: no positional encoding, QK-norm, output gate, 4 query heads per KV head.
- First FFN dense; the rest use 7-of-256 experts plus a shared expert.
- One MTP module.
- Already **15.6% better than GPT-3 at d=512**, predicted **8.5% better** at equal serving cost.

**Six factors tested** (things the hybrid lacks):

| Factor | Options |
|---|---|
| Residual stream | plain, hyper-connections (hc), AttnRes |
| V | value residual on global layers |
| K | key-value conv on global layers |
| F | squared-ReLU experts |
| P | partial RoPE on global layers |
| S | DSA indexer |

Full factorial = 3 x 2^5 = 96 models. Instead, **16 combinations at d=512 with 3 seeds each**, balanced so each change can be compared with and without (a **fractional factorial design**).

**Results:**

| Factor | Effect |
|---|---|
| HC | -1.8% |
| AttnRes | -1.1% |
| K | -0.7% |
| V | -0.4% |
| F, P, S | worse or no measurable difference |

- Gains appear additive; no interactions larger than noise.
- The statistical fit suggested keeping F; three confirmation runs gave -0.03%, so F was dropped.

**Choice: HC + V + K = "the candidate".** vs. the hybrid: -2.6%, -2.2%, -1.9%, -1.8% at d=512 / 768 / 1024 / 1280. The improvement barely shrinks at the last width.

Candidate structure: HC widens the residual stream to four lanes; `v=@v1` saves the first global layer's values and `v=$v1` mixes them into later global layers; each global layer has the 4-tap KV conv; the MTP module keeps the hybrid's attention.

**Serving-adjusted:** HC makes serving more expensive: **9.1B active** affordable vs. the hybrid's **11.4B**. Predicted improvement **-0.44%**. If the candidate's four widths get their own scaling exponent (instead of sharing one with the roughly 60 variants built on it), it becomes **-0.96%**. The prediction is fit-sensitive.

**HC + V vs. HC + V + K:** prediction -0.8% without the conv, -0.4% with. Serving size barely changes (9.2B vs. 9.1B), and the conv helps 0.3-0.7% at every trained width. The difference comes from joint fitting; the author calls it a tie.

**mHC instead of HC:** same quality, better stability at high learning rates, but costs more to serve: 7.4B active vs. 9.1B.

## 7. Stacking round 2

Four more changes on the candidate (`stack3`):

| Label | Change |
|---|---|
| E | Embedding multiplied by sqrt(d) (Gemma, T5) |
| G | GatedNorm: low-rank gate after every norm (Qwen) |
| O | Norm on attention heads' outputs |
| L | Learned value-residual mix |

- All four together: **-1.2% at d=512**, **-0.1% at d=768** (essentially gone).
- **Embedding scaling flips**: from -0.73% to +0.79%.
- **Only GatedNorm keeps a gain**: -0.71% to -0.49%.

**Optimizer comparison (kept out of all stacks):** **Muon** improves by 2.2% then 1.8%, holding up far better than most small architecture changes.

**Rule adopted:** require a second width before adding any change that helps by less than about 1% at d=512.

## 8. Stacking round 3

New 2026 layers on the candidate, then combinations:

| Run | Contents |
|---|---|
| stack4 | GatedNorm + value embeddings + learned XSA + differential attention |
| stack5 | copy pointer + GatedNorm |
| stack6 | copy pointer + GatedNorm + value embeddings + learned XSA + differential attention |

**Copy pointer (CPLM, from modded-nanogpt)**: adds a "copy" option to the output head so the model can put probability on copying tokens from earlier in the document. Largest small-scale gain of these additions: **-4.1% (d=512), -2.5% (d=768), -1.4% (d=1024)**. Each width step keeps about 60% of the previous gain.

**stack5 (copy pointer + GatedNorm):**

| Width | d=512 | d=768 | d=1024 | d=1280 |
|---|---|---|---|---|
| vs. candidate | -4.8% | -2.8% | -1.8% | -1.6% |

- Gains add fully at d=512 and about 95% at d=768.
- Shrinkage slows at d=1280: 89% of the d=1024 gain retained vs. about 60% between earlier widths.
- Costlier to serve: **7.9B active** vs. the candidate's 9.1B.

**Lost their gains by d=768:** document-local copy, value embeddings, learned XSA, differential attention.

**Worse:**

- **K = V**: +1.2%, and vLLM still caches both K and V, so no serving savings.
- **MSAR**: +0.5% and serves **58% fewer** sessions per GPU.
- **Null experts, paired heads, Canon layers**: worse.

**At equal serving cost:** copy pointer -0.4%, stack5 -0.5% vs. the candidate. Too close to other options to pick a winner from the fit.

### 8.1 The gradient-spike finding

- The candidate **clips 9-11% of steps** at gradient norm 1.0, vs. **0.08%** for the hybrid.
- Hyper-connections were the first suspect, but the largest gradient came from **the first layer's dense FFN**.

| Fix | Spikes | Quality |
|---|---|---|
| `dense0`: replace dense layer with routed experts | mostly removed | unchanged |
| `dense_last`: move dense layer to the end | removed | -0.42%, -0.41%, -0.33% at d=512/768/1024; all 8 seeds improve; same serving cost |

With GatedNorm already in the model, spikes are gone and moving the dense layer gives no further gain.

## 9. The final architecture

**Definition:** the 2026 KDA hybrid + hyper-connections + value residual + key-value conv on global layers, with the **dense FFN moved to the last layer**.

**Prediction:** **1.0% better than the hybrid** and **9.4% better than GPT-3**, at equal FP8 serving cost after 1T tokens.

### 9.1 Final model vs. stack5

| Metric | Final model | stack5 |
|---|---|---|
| Predicted vs. hybrid | -0.98% | -0.95% (with its d=1280 run) |
| Predicted vs. candidate (bootstrap range) | -0.54% (-0.90 to -0.24) | -0.51% (-0.92 to -0.21) |
| Extra layers | none | copy pointer head |
| Serving cost | same as candidate | 14% smaller servable model |
| Gradient spikes | fixed | fixed (by GatedNorm) |
| d=1280 runs in fit | no | yes |

**Why the dense-layer move was preferred:** no added layers, same serving cost, fixes the gradient spikes, and helped at every width with every seed. The copy pointer benefits from repeated text (plentiful in code repositories) but shrinks the servable model by 14%. **Combining both** (stack5 with the dense layer last) adds nothing: -0.06% at d=512, -0.05% at d=768.

### 9.2 Final architecture specification (bux)

The final architecture: the dense FFN is in the last decoder block; all earlier FFNs use MoE. HC, the value residual, the KV conv and the MTP module are unchanged.

```plaintext
# The final architecture: the late-2026 KDA hybrid with hyper-connections, a value
# residual and a key-value conv on its global layers, and its one dense FFN moved
# from the first layer to the last. Shown at d=1024; any d_model >= 768 works.
d_model    = 1024
vocab_size = 50304
layers     = d_model // 64

norm  = rmsnorm(eps=1e-5)
delta = kda(num_heads=d_model // 128, num_v_heads=d_model // 64, head_dim=128,
            gate_lower_bound=-5)
attn  = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 256, qk_norm=norm,
            gated=true)
glob  = attn(kv_conv_size=4)              # NoPE global layer + 4-tap key-value conv

swiglu_width = (8 * d_model + 96) // 192 * 64
dense   = gated_mlp(intermediate_size=swiglu_width, activation=swish)
experts = moe(num_experts=256, top_k=7, num_shared_experts=1, router=sigmoid,
              routed_scaling_factor=2.5, load_balancing=aux_free,
              aux_loss_coef=1e-4, aux_loss_scope=sequence,
              expert=gated_mlp(intermediate_size=(swiglu_width + 256) // 512 * 64,
                               activation=swish))

layer(mixer, ffn=experts) = [norm mixer] [norm ffn]

# Three KDA layers per global layer. Every global layer after the first mixes the
# first one's values into its own (the value residual: @v1 writes, $v1 reads).
first = layer(delta)^3 layer(glob(v=@v1))
group = layer(delta)^3 layer(glob(v=$v1))
last  = layer(delta)^3 layer(glob(v=$v1), dense)

shared out = head
mtp = $h nextn(norm=norm) layer(attn) norm out   # one MTP module

model    = embed first group^(layers // 4 - 2) last norm @h out mtp
residual = hc(expansion=4)                       # 4 copies of the residual stream

# Training: code and web in rows of 2k, then the last 10% in rows of 16k with YaRN.
tokenizer = tokenizer(name="EleutherAI/gpt-neox-20b")
web       = text(path="data/fineweb-edu/sample/10BT/*.parquet")
code      = text(path="data/stack-edu/files/train/*.parquet")
repos     = text(path="data/stack-edu/repos/train/*.parquet")

data       = 0.6 * code + 0.4 * web
seq_len    = 2048
batch      = 192
budget     = 4.03B                                # tokens: d=1024's rung
steps      = budget // (batch * seq_len)
curriculum = phase(at=0.9, seq_len=16384, batch=batch // 8,
                   data=0.5 * repos + 0.3 * code + 0.2 * web,
                   rope_scaling=yarn(factor=8, original_max_len=2048))

optimizer = adamw(lr=2e-3, weight_decay=0.1)
schedule  = wsd(warmup=0.05, decay=0.2, shape=sqrt)
loss      = loss(z=1e-4)
```

**Reading the bux notation:** `[f]` means `x + f(x)` (a residual branch), so `[norm mixer] [norm ffn]` is a pre-norm transformer block. `^n` repeats. `@name` writes to a named port; `$name` reads from it. `shared` makes a layer instance reused (the MTP module shares the output head). `|` (not used here) creates a sweep.

### 9.3 Why four models above the final one in the plot do not count

The final-ranking plot shows four models at -1.05% to -1.17%: stack5 without a dense layer, stack5 with its dense layer last, stack6, and the copy pointer with a 4k window. **All are hollow dots** (trained at d=512 and d=768 only), so the fit gives them the candidate's scaling exponent and effectively assumes their small-model advantage persists (the same problem as the -2.5% RoPE variants).

- The two stack5 variants only move its single dense layer and are within noise of stack5 at trained sizes: -0.14% / -0.02% without the dense layer; -0.06% / -0.05% with it last.
- stack5 itself has four sizes; its advantage over the candidate shrinks from 4.8% (d=512) to 1.6% (d=1280), and the fit predicts -0.95%. **Fit on its two smallest sizes alone, stack5 predicts -1.23%**, right where its variants sit.
- Among models with at least three trained sizes, the best are the final model and stack5.

More generally, the fit should not be trusted to rank the top models. The final model, stack5, the candidate, HC + V, mHC and the copy pointer all lie within about half a percentage point, and their order depends on how few-width models are fit.

## 10. Full architectures compared

**Predicted improvement over GPT-3** at equal serving cost and 1T tokens:

| Architecture | Change in loss |
|---|---|
| Llama 3 | -4.7% |
| DeepSeek-V3 | -8.4% |
| Qwen3-Next | -8.5% |
| 2026 hybrid | -8.5% |
| Final model | -9.4% |

Most of the improvement was already present in **DeepSeek-V3** (fine-grained MoE and MLA). The 2025-26 changes look much bigger on small models than at the serving size.

### 10.1 Where the loss goes

The scaling law decomposes loss into (1) data entropy, (2) a limited-data penalty, (3) a limited-parameter penalty.

| Model (1T tokens) | Limited-data share | Limited-parameter share | Rest |
|---|---|---|---|
| 2026 hybrid | about 8% | about 2% | data entropy |
| 2021 reference | | about 11% | |

Architecture can only reduce the parameter term. Even eliminating it entirely would improve the hybrid by under 2%. This is how a change that makes a model match one with 3x the parameters can end up worth only half a percent.

**Small-model gains overstate serving-size gains by 1.5-3x.** Example: the hybrid beats GPT-3 by 15.6% (d=512), 12.2% (d=1024), and a predicted 8.5% at equal serving cost.

### 10.2 Training cost view

If data is plentiful:

- To reach the hybrid's predicted loss after 1T tokens, the **hybrid needs 1.46x the B300 GPU-hours of the candidate**.
- **DeepSeek-V3 reaches GPT-3's loss 2.3x faster** than GPT-3. Part of that is training efficiency: **0.26 MFU** vs. **0.16** for the 2026 hybrids.

### 10.3 FP8 quality check

On trained d=1024 checkpoints, FP8 weights and KV cache raise code BPB by **0.07%** and repository BPB by **0.10%**: small.

### 10.4 Long contexts

The predicted order holds when serving is measured at contexts up to 256k.

For long-context serving, the author would add **sparse attention on the global layers**: DSA with GLM-5.2's **IndexShare** (one global layer in four runs the indexer; the next three reuse its picks), trained with **IndexCache's cross-layer loss** (without it, 1 of 3 seeds fell apart past 8k; with it, neutral on all three).

| Context | Sessions per GPU vs. candidate (same width) |
|---|---|
| 16-24k | 0.88-0.92x (does not pay) |
| 64k | 1.03x |
| 128k | 1.40x |
| 256k | 1.10x |

At equal serving cost this is worth only about 0.1 percentage point even at 128k: a long-context serving option, not an architecture win. Claims about quality at 256k would require a 256k extension phase.

### 10.5 MLA, revisited

The final architecture is uncannily close to **Kimi K3**, except for MLA. MLA's absence is mainly due to the small scale and vLLM not serving small MLA shapes fast enough; Opus 5.5 also gave up on MLA too quickly after noting this.

Second attempt: MLA in the candidate's global layers at the shape vLLM serves well on B300 (a **576-wide latent**, FlashInfer's MLA kernel):

- Quality **1.2-1.4% worse** at d=512, on all six seed pairs. Partly because bux's MLA cannot take the candidate's value residual, KV conv or QK-norm.
- Serving does not compensate: despite a **3.1x smaller KV cache** in global layers, it handles **0.78-0.89x** the candidate's sessions per GPU at 16-24k and breaks even at 256k.

MLA would have a good chance at a larger target size, but the GQA vs. MLA gap does not appear large.

## 11. Summary of conclusions

1. **For pure intelligence per parameter from softmax attention, plain full MHA wins**: it beats GQA, MQA and sliding windows; MLA only ties it. It is painfully slow to serve, which is the only reason the others win. The biggest per-parameter wins since 2020 are **MoE and linear attention**, counting only active parameters.
2. **Linear attention works**, especially once kernels are not a bottleneck. Its edge over SWA in complete models is small, so the choice should come from stronger signals: real benchmarks, post-training behavior, etc.
3. **Residual-stream changes work**, though hyper-connections only pay for their serving cost on the 2026 hybrid, not the 2021 reference.
4. **Architecture research progress is real** and visible even at small scale.
5. **Caveat on priorities:** a state-of-the-art training setup on a mediocre architecture can beat the best architecture on a mediocre setup. Swapping to Muon alone buys about as much as a full round of architecture stacking; data changes likely similar. Architecture research is a privilege of labs that have everything else sorted out.

## 12. Side note: how good AI is at research

- **Strengths of Opus 5.5:** implemented an architecture from its tech report in under 30 minutes (the author estimates a day by hand); debugged complex XLA and kernel issues; understands good research practice; good at web search.
- **Weakness:** missed obvious combinations even after seeing that the individual changes were strong. Asked why, it would say "yeah, I guess worth trying". This can feel like gaslighting: when a very capable system skips the obvious, you start doubting yourself.
- The author expects AI to improve at research as it did at coding.
- Tooling: Claude Code and Codex running inside JetBrains' Air in PyCharm.

## 13. Reproduction

| Resource | Location |
|---|---|
| Framework | bux (`github.com/pilot7747/bux`) |
| Every `.bux` file of the study | `examples/experiments/timeline/` |
| Training recipe | `timeline_code.bux` |
| Walkthrough | `docs/tutorials/timeline.md` |
| Scaling-law fit | `scripts/timeline/fit.py` |
| Train one family's ladder | `bux launch examples/experiments/timeline/dense_last.bux` |
| Export for vLLM | `bux export examples/experiments/timeline/dense_last.bux OUT` |
| Open data | Hugging Face `pavlichenko/bux-timeline`, CC BY 4.0: all 1,129 runs, each with a self-contained `.bux`, rerun commands and logged curves; 5,232 vLLM serving measurements; both refits with inputs. Browsable online (Hugging Face Space) or offline via the pack's `index.html` |
| Training curves | W&B project |

---

# Part II. Expanded References

Each entry gives: what the technique is, where it comes from, and how the study used it. Bracketed numbers point to the bibliography in Part III.

## A. Data, tokenizer and training recipe

**Stack-Edu.** A filtered, education-quality subset of The Stack v2 (permissively licensed source code), built by Hugging Face for SmolLM2 by scoring files with an educational-value classifier [B1]. The study uses individual files and whole Python repositories (the repository split supplies 16k-token long-context data).

**FineWeb-Edu.** A subset of the FineWeb web corpus filtered by an LLM-trained classifier for educational content [B2]. Supplies the web portion; the 10BT sample is used.

**GPT-NeoX tokenizer.** The BPE tokenizer of EleutherAI's GPT-NeoX-20B [B3], padded vocabulary 50,304.

**Chinchilla ratio and scaling-law form.** Hoffmann et al. showed compute-optimal training uses roughly 20 tokens per parameter and fit loss as `L(N, D) = E + A/N^alpha + B/D^beta` [B4]. The study uses both: 20 tokens per non-embedding parameter for each rung, and this exact functional form for extrapolation.

**AdamW.** Adam with decoupled weight decay [B5].

**Warmup-Stable-Decay (WSD) schedule.** Linear warmup, a long constant-learning-rate plateau, then a short decay phase; popularized by MiniCPM [B6]. Lets one checkpoint at the plateau be decayed at any budget. Here: 5% warmup, 20% decay with a square-root shape.

**z-loss.** An auxiliary penalty on the squared log of the softmax normalizer, `1e-4 * log(Z)^2`, that keeps output logits from drifting large; introduced in PaLM [B7] and used widely since.

**DeepSeek batch-size scaling law.** DeepSeek LLM fit optimal batch size and learning rate as power laws of compute [B8]. The study sets batch size (48 to 288 rows of 2k) from the batch law.

**Bits per byte (BPB).** Cross-entropy normalized by UTF-8 bytes instead of tokens, making results tokenizer-independent.

**Muon.** An optimizer for 2D weight matrices that orthogonalizes the momentum update with a few Newton-Schulz iterations [B9]. Popularized by the modded-nanogpt speedrun and later adopted at scale (e.g., Kimi). The study reports it as an optimizer comparison outside all architecture stacks: -2.2% then -1.8%.

**Fractional factorial design.** A statistical design that runs a chosen subset of all factor combinations so each main effect can still be estimated (with some effects aliased with higher-order interactions). Round 1 runs 16 of 96 combinations; round 2's `stack3` uses a resolution IV half-fraction of 2^4 (no main effect aliased with any two-factor interaction).

**Bootstrap range.** Confidence interval from resampling runs; used to compare the final model and stack5.

## B. Serving stack

**vLLM.** The open-source LLM inference engine built around PagedAttention, which stores the KV cache in fixed-size pages [B10]. The study uses vLLM 0.32 with a bux plugin that serves bux-exported models, FP8 weights, activations and KV cache, and **prefix caching** (reuse of KV cache for a shared prompt prefix, here the 2k shared prefix).

**KV cache.** Keys and values stored for every past token in every attention layer. It dominates decode memory and bandwidth at long contexts, which is why GQA/MQA/MLA/SWA/linear attention can buy large models at equal cost.

**FlashInfer.** A library of attention kernels for serving; its MLA kernel was used in the MLA revisit.

**NVIDIA B300.** Blackwell Ultra datacenter GPU.

**MFU (model FLOPs utilization).** Achieved FLOPs as a share of peak hardware FLOPs.

**Speculative decoding.** A small or auxiliary predictor drafts several tokens that the main model verifies in one pass. MTP modules can serve as drafters; not measured here.

## C. Positional encodings

**Learned absolute positions.** A trainable embedding per position, added to the token embedding (GPT-2, GPT-3) [B11].

**RoPE (Rotary Position Embedding).** Rotates pairs of query/key dimensions by an angle proportional to position, so the attention dot product depends only on relative offset [B12]. Adopted by GPT-J, GPT-NeoX, Llama and most later models. The study's "2021 reference" is GPT-3 + RoPE.

**RoPE base (theta).** Sets the rotation wavelengths. A larger base (Llama 3 uses 500k) rotates more slowly, which helps long context [B13].

**Partial RoPE.** Rotating only a fraction of each head's dimensions (GPT-J/NeoX; Qwen3-Next rotates a quarter at base 10M).

**ABF (Adjusted Base Frequency).** Raising RoPE's base when extending context (Xiong et al., 2023; Llama 3) [B14]. Tested as an alternative to YaRN at the 16k switch.

**YaRN.** A RoPE context-extension method that interpolates frequencies non-uniformly (high frequencies kept, low frequencies stretched, "NTK-by-parts") and adds an attention temperature correction [B15]. Here applied with factor 8 (2k to 16k) at the start of the long phase.

**NoPE (no positional encoding).** Causal decoders can infer position from the causal mask alone [B16]. Breaks the model if applied everywhere here, but is standard in the global layers of 2026 hybrids (Kimi Linear, Kimi K3, Nemotron 3), where local or recurrent layers supply order.

**iRoPE.** Llama 4's interleaving of NoPE layers (every 4th) with RoPE layers, aimed at long context [B17]. SmolLM3 uses a similar scheme.

**Inkling's relative bias.** Thinking Machines' 2026 multimodal MoE "Inkling" uses no RoPE; instead each head adds a data-dependent bias `<W_h x_i, P[i - j]>` from a learned table of offsets (1024 here). Tested as a position arm.

**RULER.** A synthetic long-context benchmark (retrieval, multi-hop tracing, aggregation) [B18]; suggested by the author as a better discriminator of RoPE variants than BPB.

## D. Normalization

**LayerNorm vs. RMSNorm.** LayerNorm subtracts the mean and divides by standard deviation with gain and bias. RMSNorm divides by the root mean square only, with a gain [B19]; cheaper, used since Llama.

**Removing biases.** PaLM removed biases from all projections and norms for stability and throughput [B7].

**Pre-norm / post-norm.** Pre-norm applies the norm to a branch's input (`x + f(norm(x))`), the GPT-2 onward default. The tested post-branch norm `x + norm(f(x))` is OLMo 2's reordered norm [B20].

**Sandwich norm.** Normalizing both the input and the output of a branch, `x + norm(f(norm(x)))`; from CogView [B21], used in Gemma 2/3. Trinity (Arcee, 2026) initializes output gains at 1/sqrt(layers).

**Logit soft-capping.** `cap * tanh(logits / cap)` on attention or final logits to prevent extreme values; Gemma 2 [B22] (cap 30 in the study's arm).

**QK-norm.** Normalizing each query and key head (RMSNorm with a learned per-channel gain, before RoPE) so attention logits cannot grow unboundedly. Introduced by Henry et al. [B23], scaled up in ViT-22B [B24]; Wortsman et al. linked unbounded "attention logit growth" to instability [B25]. Used in OLMo 2 and Qwen3. The study's key finding: the gain comes from the **learned gains**, not normalization per se.

**DyT (Dynamic Tanh).** Replaces normalization with `gamma * tanh(alpha * x) + beta` [B26]. Fails here with standard 0.02 embedding init.

**GatedNorm.** A low-rank gate multiplying every branch's norm output: `y * sigmoid(W2 swish(W1 y))` with rank 16. From Qwen (arXiv 2601.22966) and Qwen3.8-Next (arXiv 2608.30320), claimed to reduce loss spikes and activation outliers. The only round-2 change that held its gain; also removes the candidate's gradient spikes.

**Output norm (per-head).** Normalizing each attention head's output before the output gate; from the Forgetting Transformer's "Pro" block [B27], also in Muse Spark (2026). "O" in round 2.

**Embedding times sqrt(d).** Scaling token embeddings by sqrt(d_model), as in T5 and Gemma. "E" in round 2; flipped from helping to hurting.

## E. MLP variants

**GELU MLP.** GPT-style two-layer MLP, hidden width 4x, GELU activation, with biases.

**GLU variants (SwiGLU, GeGLU).** `(act(x W) * (x V)) W2`, a gated MLP [B28]. SwiGLU uses SiLU (Swish) as the gate (PaLM, Llama); GeGLU uses GELU (Gemma). Hidden width 8/3 of the stream to match parameters.

**Squared ReLU.** `max(0, x)^2`, found by architecture search in Primer [B29]; used in modded-nanogpt and NVIDIA Nemotron-H / Nemotron 3.

## F. Softmax attention variants

**MHA (multi-head attention).** Every query head has its own key and value head [B30]. Best per parameter in this study, slowest to serve.

**MQA (multi-query attention).** All query heads share one key/value head [B31]; minimal KV cache (PaLM).

**GQA (grouped-query attention).** Groups of query heads share a key/value head [B32]; here 4 query heads per KV head (Llama 2 70B, Llama 3).

**MLA (multi-head latent attention).** DeepSeek-V2's compression of keys and values into a shared low-rank latent per token (512 dims in DeepSeek) plus a small decoupled RoPE component (64 dims), with up-projections absorbed at inference [B33]. Cuts KV cache while keeping per-head expressiveness.

**Gated attention.** A sigmoid gate on each head's output, computed from the layer input, applied after scaled-dot-product attention and before the output projection ("G1" in Qiu et al.) [B34]. Adds non-linearity, sparsity, and removes the "attention sink" phenomenon. Used in Qwen3-Next, Qwen3.8, Kimi K3.

**HyGA.** A 2026 extension of gated attention (Zhou et al., arXiv 2608.11805) where each head's gate also reads the head's own output (H-gate) and a cross-head gate multiplies it (C-gate). Tested in round 3's new-layer sweep.

**Attention sinks.** A learned per-head logit that lets attention place mass "nowhere"; from StreamingLLM's observation that models dump attention on initial tokens [B35], adopted as learned sinks in gpt-oss. No measurable effect here.

**Heads of 128.** Halving head count while doubling head dimension (Llama 2/3, Mistral). No effect here.

**Sliding-window attention (SWA).** Each layer attends only to the last W tokens (512 or 128 here), bounding KV cache; used in Mistral, Gemma 2/3, gpt-oss, interleaved with global layers.

**Inkling's key-value conv.** A 4-tap causal depthwise convolution applied to keys and values before attention, so each key/value mixes in its three predecessors. From Thinking Machines' Inkling (2026), which uses short causal convs pervasively. "K" in the final model.

**XSA (Exclusive Self-Attention).** Removes from each head's output the component along the token's own value vector, so attention carries only what context adds (Zhai, Apple, 2026, arXiv 2603.09078); modded-nanogpt uses a "learned" version where the removed share `tanh(a)` is learned per head starting from 0. Gain lost by d=768.

**Differential attention (DIFF V2).** Each output head is the difference of two attention maps, `a1 - sigmoid(x W) * a2`, letting a head's output cancel toward zero (Ye et al., Microsoft, 2026). V1 [B36] used a fixed lambda and a per-head norm. Gain lost by d=768.

**Paired heads.** From modded-nanogpt record 58: pairs of heads are interleaved along the sequence so one query sees both heads' keys in one softmax; doubles attention FLOPs. Worse here.

**K = V (values from keys).** Gemma 4's global layers derive values from the key projection (RMS-normalized, no gain), with no value projection; in principle halves KV cache. +1.2% here, and vLLM still caches both.

**Value embeddings.** A per-token learned table added to attention values in some layers, gated per KV head (modded-nanogpt record 14, nanochat). Gain lost by d=768.

## G. Linear attention and state-space models

**Linear attention (general).** Replaces softmax attention with a recurrence over a fixed-size state matrix, giving constant memory per sequence at decode and linear-time training with chunked kernels. Variants differ in how the state is updated and decayed.

**Delta rule / DeltaNet.** The state is updated by an error-correcting write `S <- S + beta (v - S k) k^T` instead of a pure additive write, enabling overwriting (Schlag et al., 2021 [B37]; parallelized by Yang et al., 2024 [B38]).

**Gated DeltaNet (GDN).** Delta rule plus a data-dependent scalar decay per head [B39]. Used in Qwen3-Next and Qwen3.8 (3 GDN layers per gated attention layer).

**KDA (Kimi Delta Attention).** Gated DeltaNet with a **per-channel** data-dependent decay instead of per-head, from Kimi Linear (Moonshot AI) [B40], scaled up in Kimi K3. In the study: query/key heads of 128, twice as many value heads, decay bounded below at -5 (`gate_lower_bound=-5`), 4-tap convs on q, k, v. State is about 82 MB per sequence at the target size.

**Gated DeltaNet-2 (GDN-2).** Decouples KDA's single step into separate per-channel erase and write gates (Hatamizadeh et al., NVIDIA, 2026). Best per parameter at d=512 but slow in vLLM (0.42x KDA).

**Mamba-2 (SSD).** A selective state-space model reformulated as structured state-space duality with per-head scalar decay [B41]; used in Jamba, Zamba2, Nemotron-H and Nemotron 3.

**RWKV-7 "Goose".** An RNN with a generalized delta rule and vector-valued gating [B42].

**LFM2 short conv.** Liquid AI's LFM2 family (2025) uses a double-gated short convolution `C * conv(B * u)` with 3 taps as the main mixer, with GQA in a minority of layers (3 of 8 in LFM2-1.2B; 3 of 4 in MoE variants) [B43]. Its state is just the last two inputs (0.9 MB per sequence).

**Canon layers.** Allen-Zhu's *Physics of Language Models, Part 4.1* (arXiv 2512.17351) proposes short causal depthwise convs with their own residual skip (`y + conv(y)`) at points in each block (A: before attention; B: on q/k/v; C: before the MLP), claiming better reasoning depth and NoPE matching RoPE [B44]. The candidate already has the B-type convs.

**Hybrid ratio.** "3 of 4" means three linear/local layers per global attention layer.

## H. Mixture of Experts

**Sparse MoE (GShard, Switch, Mixtral).** Replace the MLP with N experts and a router that picks top-k per token; compute scales with k, parameters with N [B45, B46]. Mixtral: 8 experts, top-2, softmax router, balance loss [B47].

**Fine-grained MoE / DeepSeekMoE.** Split experts into many smaller ones (more combinations of specialists) and add always-on **shared experts** for common knowledge [B48]. Study arm: 64 experts of d/2, top-7, plus one shared.

**DeepSeek-V3 MoE.** Fine-grained experts with **sigmoid** router scores (normalized over the selected experts), a `routed_scaling_factor`, and **auxiliary-loss-free load balancing**: a per-expert bias added only for selection, nudged up or down each step by load imbalance, plus a tiny sequence-wise balance loss [B49, B50]. Study arm: 128 experts. The 2026 hybrid uses 256 experts, top-7, one shared expert, routed scaling 2.5, aux-loss coefficient 1e-4.

**Active vs. total parameters.** Active parameters run per token; total includes all experts. The 2026 hybrid at target: 11.4B active, 201B total.

**LatentMoE.** Projects tokens into a narrower latent (here d/2) before the routed experts and back after, allowing more experts at equal parameters; router and shared expert remain full width. From NVIDIA Nemotron 3, also Kimi K3 and Muse Spark [B51].

**Softmax router + balance loss.** The classic scheme kept by Qwen3-Next/Qwen3.8 and Mellum2.

**sqrt-softplus scores.** DeepSeek-V4 scores experts by `sqrt(softplus(logit))` in place of sigmoid.

**SMEBU.** Arcee Trinity Large's "soft-clamped momentum expert bias update": the aux-free bias moves by `lambda * tanh(kappa * (mean_load - load) / mean_load)`, centered and smoothed with momentum, so it stops oscillating around balance.

**Zero-compute ("null") experts.** Identity experts the router can choose, so easy tokens use fewer real experts; from Meituan's LongCat-Flash [B52].

## I. Residual stream variants

**Hyper-connections (HC).** Keep n copies (here 4) of the residual stream; each branch reads a learned (and input-dependent) mix of copies and writes back with learned weights, while copies are also mixed among themselves [B53]. Raises per-layer memory traffic, hence -10% to -20% serving throughput in FP8.

**mHC (manifold-constrained hyper-connections).** DeepSeek's variant that projects the stream-mixing matrix onto doubly stochastic matrices via Sinkhorn iterations, preserving signal magnitude for stability at scale [B54]; used in DeepSeek-V4.1 and GLM-5.3-Flash. Here: same quality as HC, better high-LR stability, more serving cost (7.4B vs. 9.1B).

**AttnRes (Attention Residuals).** Moonshot/Kimi (2026, Kimi K3): instead of summing all earlier branch outputs, each branch reads a learned softmax over depth of the embedding and earlier block outputs, countering "pre-norm dilution".

**MSAR (multi-stream attention residuals).** AttnRes in four streams per source (the source plus causal convs over 4, 8, 12 tokens), mixed by sigmoid gates (Qin et al., 2026, arXiv 2610.13630). +0.5% and 58% fewer sessions per GPU here.

**Value residual (ResFormer).** Later attention layers mix the first layer's value vectors into their own: `a * v + b * v1` with a = b = 0.5 [B55]. "Learnable-ResFormer" learns a and b ("L" in round 2). Also in RWKV-7 and modded-nanogpt. "V" in the final model.

**Gated residual (gr).** Qwen3.8-Flash-Next's residual algebra: a gated read of 4 normalized streams and a weighted write to each, replacing pre-norms. Initially broken by an XLA rewrite bug.

## J. Sparse attention

**DSA (DeepSeek Sparse Attention).** From DeepSeek-V3.2: a small "lightning indexer" (few heads, ReLU dot products, learned per-head weights) scores every past key, keeps the top-k (2048), and the main attention attends only to those [B56]. The indexer is trained by a distillation loss toward the main attention.

**QSA.** Qwen3.8-Flash-Next's block variant: the indexer scores mean-pooled blocks of 4 keys and keeps the best 512 blocks plus the query's own block.

**MSA (MiniMax Sparse Attention).** From MiniMax-M3: picks blocks (128 keys, max-pooled scores) separately for each KV head's group of query heads.

**CSA (Compressed Sparse Attention).** From DeepSeek-V4: each query attends to a short sliding window of per-token latents plus an indexer-selected subset of **compressed** latents (pooled groups of tokens). Was the one sparse method that hurt quality here (+8.3%).

**IndexShare / IndexCache.** GLM-5.2's IndexShare runs the indexer only in some layers and lets following layers reuse its top-k picks (after IndexCache, Bai et al., 2026). IndexCache's **cross-layer (multi-layer) distillation loss** trains the shared indexer toward the attention of every layer that reuses its picks, preventing drift.

**Prefill vs. decode.** Prefill processes the prompt in parallel; decode generates token by token. The indexer's cost is especially visible in long prefills (3.2 s vs. 2.3 s for dense at 175k).

## K. Heads, memory and miscellaneous

**Multi-token prediction (MTP).** Predicting several future tokens per position. Gloeckle et al. used parallel heads [B57]; DeepSeek-V3 uses sequential MTP modules sharing the embedding and output head [B49]. The study uses one module (weight 0.3), consistent with Qwen3.8, GLM-5.3 and MiMo.

**Tied vs. untied output head.** Tied: the output projection reuses the input embedding matrix. Untying was worse here.

**Engram.** DeepSeek's (2026, arXiv 2601.07372) conditional memory: hashed 2- and 3-gram embeddings of input tokens, gated by the hidden state and added to the stream; used in DeepSeek-V4.1 and Qwen3.8-Flash-Next.

**Copy pointer (CPLM).** From modded-nanogpt PR 379: the output vocabulary gets an extra `<copy>` row; its probability `a` is routed to a 128-dim pointer attention over earlier positions in the document, so `p = p_lm (1 - a) + a p_copy`. Exploits repetition in code. Large small-scale gain, shrinks with width.

**Document-local copy feature.** modded-nanogpt PR 378: finds the longest earlier match (1 to 32 tokens) of the current context in the document and adds an embedding of the token that followed it.

**Parallel blocks.** Attention and MLP read the same normalized input and are added together (GPT-J, PaLM); faster, slightly worse here.

**Deep and thin.** MobileLLM's finding that small models benefit from depth at fixed parameters [B58]. Equal quality here but slower to serve.

**Dense first layer / `dense_last`.** DeepSeek-V3 style MoE models keep the first layer(s) dense. The study found that dense first layer causes the candidate's gradient spikes and that moving it to the last layer removes them with a small quality gain.

**Gradient clipping at 1.0.** Rescales gradients whose global norm exceeds 1.0; the share of clipped steps diagnoses spikes.

## L. Models and architectures referenced

| Model | Lab | Relevance |
|---|---|---|
| GPT-3 | OpenAI, 2020 [B11] | Baseline shape; 6.7B shape for serving baseline |
| Llama 3 | Meta, 2024 [B13] | GQA, RoPE base 500k; full-architecture comparison (-4.7%) |
| Llama 4 | Meta, 2025 | iRoPE |
| PaLM | Google, 2022 [B7] | No biases, MQA, z-loss, parallel blocks |
| Gemma 2 / 3 / 4 | Google | Sandwich norm, soft caps, SWA, sqrt(d) embeddings, K=V |
| OLMo 2 | AI2, 2024 [B20] | Post-branch norm with QK-norm |
| DeepSeek-V2 / V3 / V3.2 / V4 / V4.1 | DeepSeek | MLA, fine-grained MoE, aux-free balancing, MTP, DSA, CSA, sqrt-softplus routing, mHC, Engram |
| Qwen3-Next / Qwen3.8 / Qwen3.8-Flash-Next | Alibaba | Gated DeltaNet hybrid, gated attention, QSA, gated residual, GatedNorm |
| Kimi Linear / Kimi K3 | Moonshot AI | KDA hybrid, NoPE global layers, AttnRes, LatentMoE; final architecture is "uncannily close" to K3 (K3 is Kimi Linear scaled to 93 layers) |
| GLM-5.2 / GLM-5.3(-Flash) | Zhipu AI (Z.ai) | IndexShare, KDA-based hybrid flash model |
| Nemotron-H / Nemotron 3 | NVIDIA | Mamba hybrids, squared ReLU, LatentMoE |
| MiniMax-M3 | MiniMax | MSA |
| LFM2 | Liquid AI, 2025 | Short-conv hybrid |
| Inkling | Thinking Machines, 2026 | KV conv, relative bias, pervasive short convs |
| Trinity Large | Arcee, 2026 | SMEBU, sandwich-norm init |
| LongCat-Flash | Meituan, 2025 | Zero-compute experts; embedding norm |
| Muse Spark | 2026 | Output norm, LatentMoE norms |
| gpt-oss | OpenAI, 2025 | Learned sinks, SWA |
| Mellum / Mellum2 | JetBrains | Author's former code-model project; MTP ablations, serving-budget framing |
| modded-nanogpt | Keller Jordan and community | GPT-2 speedrun; source of Muon, value embeddings, XSA-learned, paired heads, CPLM, doc-copy |

## M. Tools

**bux.** The author's JAX/Flax NNX framework for pre-training ablations. A model is an algebra over the residual stream in a `.bux` file (`[f]` = `x + f(x)`; `^n` repeats; `@`/`$` write/read named ports; `|` sweeps). Supports FSDP/tensor/expert/context/pipeline parallelism, 62 transcribed architectures, export to transformers and a vLLM plugin. Apache-2.0.

**XLA.** The compiler under JAX; an XLA rewrite on GPUs caused the gated-residual bug.

**Claude Opus 5.5, Claude Code, Codex.** Coding agents used to write most of the study's code, run inside JetBrains' Air in PyCharm.

**Weights & Biases (W&B).** Experiment tracking for training curves.

---

# Part III. Bibliography

| # | Reference | Link |
|---|---|---|
| B1 | Allal et al. (2025), *SmolLM2: When Smol Goes Big* (Stack-Edu) | arxiv.org/abs/2502.02737 |
| B2 | Penedo et al. (2024), *The FineWeb Datasets* (FineWeb-Edu) | arxiv.org/abs/2406.17557 |
| B3 | Black et al. (2022), *GPT-NeoX-20B* | arxiv.org/abs/2204.06745 |
| B4 | Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* | arxiv.org/abs/2203.15556 |
| B5 | Loshchilov and Hutter (2019), *Decoupled Weight Decay Regularization* | arxiv.org/abs/1711.05101 |
| B6 | Hu et al. (2024), *MiniCPM* (WSD schedule) | arxiv.org/abs/2404.06395 |
| B7 | Chowdhery et al. (2022), *PaLM* | arxiv.org/abs/2204.02311 |
| B8 | DeepSeek-AI (2024), *DeepSeek LLM* (batch-size and LR scaling laws) | arxiv.org/abs/2401.02954 |
| B9 | Jordan et al. (2024), *Muon: An optimizer for hidden layers in neural networks* | kellerjordan.github.io/posts/muon |
| B10 | Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) | arxiv.org/abs/2309.06180 |
| B11 | Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3) | arxiv.org/abs/2005.14165 |
| B12 | Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* | arxiv.org/abs/2104.09864 |
| B13 | Meta (2024), *The Llama 3 Herd of Models* | arxiv.org/abs/2407.21783 |
| B14 | Xiong et al. (2023), *Effective Long-Context Scaling of Foundation Models* (ABF) | arxiv.org/abs/2309.16039 |
| B15 | Peng et al. (2023), *YaRN: Efficient Context Window Extension of LLMs* | arxiv.org/abs/2309.00071 |
| B16 | Kazemnejad et al. (2023), *The Impact of Positional Encoding on Length Generalization in Transformers* (NoPE) | arxiv.org/abs/2305.19466 |
| B17 | Meta (2025), Llama 4 release (iRoPE) | ai.meta.com/blog/llama-4-multimodal-intelligence |
| B18 | Hsieh et al. (2024), *RULER: What's the Real Context Size of Your Long-Context LMs?* | arxiv.org/abs/2404.06654 |
| B19 | Zhang and Sennrich (2019), *Root Mean Square Layer Normalization* | arxiv.org/abs/1910.07467 |
| B20 | OLMo Team (2024), *2 OLMo 2 Furious* | arxiv.org/abs/2501.00656 |
| B21 | Ding et al. (2021), *CogView* (sandwich LayerNorm) | arxiv.org/abs/2105.13290 |
| B22 | Gemma Team (2024), *Gemma 2* | arxiv.org/abs/2408.00118 |
| B23 | Henry et al. (2020), *Query-Key Normalization for Transformers* | arxiv.org/abs/2010.04245 |
| B24 | Dehghani et al. (2023), *Scaling Vision Transformers to 22 Billion Parameters* | arxiv.org/abs/2302.05442 |
| B25 | Wortsman et al. (2023), *Small-scale proxies for large-scale Transformer training instabilities* | arxiv.org/abs/2309.14322 |
| B26 | Zhu et al. (2025), *Transformers without Normalization* (DyT) | arxiv.org/abs/2503.10622 |
| B27 | Lin et al. (2025), *Forgetting Transformer* | arxiv.org/abs/2503.02130 |
| B28 | Shazeer (2020), *GLU Variants Improve Transformer* | arxiv.org/abs/2002.05202 |
| B29 | So et al. (2021), *Primer: Searching for Efficient Transformers* | arxiv.org/abs/2109.08668 |
| B30 | Vaswani et al. (2017), *Attention Is All You Need* | arxiv.org/abs/1706.03762 |
| B31 | Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* (MQA) | arxiv.org/abs/1911.02150 |
| B32 | Ainslie et al. (2023), *GQA* | arxiv.org/abs/2305.13245 |
| B33 | DeepSeek-AI (2024), *DeepSeek-V2* (MLA) | arxiv.org/abs/2405.04434 |
| B34 | Qiu et al. (2025), *Gated Attention for Large Language Models* | arxiv.org/abs/2505.06708 |
| B35 | Xiao et al. (2023), *Efficient Streaming Language Models with Attention Sinks* | arxiv.org/abs/2309.17453 |
| B36 | Ye et al. (2024), *Differential Transformer* | arxiv.org/abs/2410.05258 |
| B37 | Schlag et al. (2021), *Linear Transformers Are Secretly Fast Weight Programmers* | arxiv.org/abs/2102.11174 |
| B38 | Yang et al. (2024), *Parallelizing Linear Transformers with the Delta Rule* | arxiv.org/abs/2406.06484 |
| B39 | Yang, Kautz and Hatamizadeh (2025), *Gated Delta Networks* (ICLR) | arxiv.org/abs/2412.06464 |
| B40 | Kimi Team (2025), *Kimi Linear* (KDA) | arxiv.org/abs/2510.26692 |
| B41 | Dao and Gu (2024), *Transformers are SSMs* (Mamba-2) | arxiv.org/abs/2405.21060 |
| B42 | Peng et al. (2025), *RWKV-7 "Goose"* | arxiv.org/abs/2503.14456 |
| B43 | Liquid AI (2025), LFM2 | liquid.ai |
| B44 | Allen-Zhu (2025), *Physics of Language Models: Part 4.1* (Canon layers) | arxiv.org/abs/2512.17351 |
| B45 | Lepikhin et al. (2020), *GShard* | arxiv.org/abs/2006.16668 |
| B46 | Fedus et al. (2021), *Switch Transformers* | arxiv.org/abs/2101.03961 |
| B47 | Jiang et al. (2024), *Mixtral of Experts* | arxiv.org/abs/2401.04088 |
| B48 | Dai et al. (2024), *DeepSeekMoE* | arxiv.org/abs/2401.06066 |
| B49 | DeepSeek-AI (2024), *DeepSeek-V3 Technical Report* | arxiv.org/abs/2412.19437 |
| B50 | Wang et al. (2024), *Auxiliary-Loss-Free Load Balancing Strategy for MoE* | arxiv.org/abs/2408.15664 |
| B51 | NVIDIA (2025), Nemotron 3 (LatentMoE) | research.nvidia.com |
| B52 | Meituan LongCat Team (2025), *LongCat-Flash Technical Report* | arxiv.org/abs/2509.01322 |
| B53 | Zhu et al. (2025), *Hyper-Connections* (ICLR) | arxiv.org/abs/2409.19606 |
| B54 | DeepSeek-AI (2025), *mHC: Manifold-Constrained Hyper-Connections* | arxiv.org/abs/2512.24880 |
| B55 | Zhou et al. (2024), *Value Residual Learning* (ResFormer) | arxiv.org/abs/2410.17897 |
| B56 | DeepSeek-AI (2025), DeepSeek-V3.2 (DSA) | github.com/deepseek-ai |
| B57 | Gloeckle et al. (2024), *Better & Faster LLMs via Multi-token Prediction* | arxiv.org/abs/2404.19737 |
| B58 | Liu et al. (2024), *MobileLLM* | arxiv.org/abs/2402.14905 |

**2026 sources cited by the study's experiment files** (identifiers as given in the bux repository):

| Technique | Source |
|---|---|
| GatedNorm | Qwen, arXiv 2601.22966; Qwen3.8-Next, arXiv 2608.30320 |
| Engram | DeepSeek, arXiv 2601.07372 |
| Embedding norm / sqrt(d) result | LongCat, arXiv 2601.21204 |
| XSA | Zhai (Apple), arXiv 2603.09078 |
| HyGA | Zhou et al., arXiv 2608.11805 |
| MSAR | Qin et al., arXiv 2610.13630 |
| mHC-lite (mHC init) | arXiv 2601.05732 |
| Differential Transformer V2 | Ye et al. (Microsoft), 2026 |
| Gated DeltaNet-2 | Hatamizadeh et al., 2026 |
| Attention Residuals | Kimi (Moonshot AI), 2026 |
| IndexCache / IndexShare | Bai et al., 2026; GLM-5.2 |
| CPLM copy pointer | modded-nanogpt PR 379 |
| Document-local copy | modded-nanogpt PR 378 |
| Paired heads | modded-nanogpt record 58 |
| Value embeddings | modded-nanogpt record 14 |
| Cross-layer KV sharing (CLA, tested in ideas arm) | Brandon et al., arXiv 2405.12981 |
| NorMuon / cautious weight decay (optimizer arms) | arXiv 2510.05491; arXiv 2510.12402 |

**Study resources:** bux, `github.com/pilot7747/bux`; open data, `huggingface.co/spaces/pavlichenko/bux-timeline` and the `pavlichenko/bux-timeline` dataset (CC BY 4.0).