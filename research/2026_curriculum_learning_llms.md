# Curriculum Learning for Decoder-Only LLMs

A reference on how training-data ordering is used (and not used) in modern decoder-only language model training. Covers difficulty metrics, static and adaptive strategies, the interaction with learning-rate schedules, and what the scaling-era evidence actually supports.

---

## 1. The Core Idea

Standard LLM pretraining shuffles the corpus uniformly. **Curriculum Learning (CL)** replaces uniform sampling with a structured order, typically easy → hard, inspired by how humans learn. The hypothesized benefits:

- Faster convergence in early training.
- Better conditioning of representations (avoid bad local minima).
- Improved generalization.
- More stable training on heterogeneous corpora.

The empirical reality is more nuanced: gains are real but small at scale, sensitive to interactions with the optimizer, and often subsumed by simpler data-mixture reweighting. The one form of CL that *is* now standard practice is **multi-stage pretraining** (bulk web data → high-quality mid-training), which most labs do not even label as CL.

---

## 2. Anatomy of a Curriculum

A curriculum is fully specified by three components.

### 2.1 Difficulty metric (how to rank examples)

| Metric family | Signal | Notes |
|---|---|---|
| Sequence length | Shorter = easier | Cheap. Learns local dependencies first, long-range later. |
| Vocabulary rarity | Common words = easier | Needs token-frequency pass. |
| Syntactic complexity | Parse-tree depth, Flesch-Kincaid | Needs a parser; expensive at web scale. |
| Linguistic acquisition | Age-of-Acquisition, Verb Variation | Cognitively motivated; limited gains. |
| Loss / perplexity | Model's own loss on each sample | Self-paced; requires a reference model. |
| Attention scores | Attention entropy or spread | Found most effective among prompt-based signals for 7B models. |
| Prompt length | Input-token count | Weak but cheap. |
| Data quality score | Heuristic, model-based classifier | The dominant signal in modern pipelines. |
| Task-specific | Code: cyclomatic complexity, Halstead; Math: annotated difficulty levels | Strong when labels exist. |

### 2.2 Pacing function (how fast to raise difficulty)

- **Step function**: hard cutoffs between stages (multi-stage pretraining).
- **Linear / root / geometric**: gradually expand the allowed difficulty percentile per step.
- **Self-paced**: the model's current loss determines the next batch's difficulty.
- **Adaptive / bandit-based**: online policy selects categories by estimated learning gain.

### 2.3 Scoring granularity

- **Per-example**: rank every document.
- **Per-bucket**: split into easy/medium/hard bins (most common).
- **Per-source**: assign each corpus source a tier (web, books, code, curated).

---

## 3. Strategies for Decoder-Only Pretraining

### 3.1 Static curriculum
Sort the entire corpus once by a difficulty score, train monotonically. Conceptually clean; rarely used at scale because it removes all late-training exposure to easy data and interacts badly with LR decay (§5).

### 3.2 Pacing-function curriculum
Fix an easy-to-hard ranking, but at step $t$ sample uniformly from the bottom $f(t)$ percentile of difficulty, where $f$ grows from some initial fraction to 1. Preserves some easy-sample exposure late in training.

### 3.3 Multi-stage pretraining (the de facto standard)
- **Stage 1**: train on bulk, heterogeneous web data (CommonCrawl, RedPajama, FineWeb).
- **Stage 2 ("mid-training")**: shift the mixture to high-quality curated sources (textbooks, code, math, filtered web).
- Optionally **Stage 3**: short annealing on the highest-quality slice.

Adopted by OLMo 2, Phi-3, DeepSeek, and most frontier labs. Labeled as "data-mixture scheduling" more often than "curriculum learning" but is structurally identical.

### 3.4 Length-based curriculum
Train with short context windows first (e.g., 2k), then extend to longer (8k, 32k, 128k+). Combines well with RoPE scaling / YaRN / position interpolation methods for long-context extension. Standard practice.

### 3.5 Objective-level curriculum
Not strictly data ordering, but related: interleave auxiliary objectives early (span corruption, FIM, masked next-token) and converge to pure next-token later. Common in code LLMs (StarCoder, DeepSeek-Coder).

### 3.6 Self-paced learning
Use the model's own per-example loss as the difficulty score. A batch is constructed from samples whose loss falls in a moving window. Risks: feedback loops, amplifying noisy samples, instability under distributed training.

### 3.7 Automatic / Self-Evolving Curriculum (RL fine-tuning)
Relevant for reasoning post-training (o1-style, RLVR, GRPO). Frame curriculum selection as a non-stationary multi-armed bandit:
- **Arms** = difficulty levels or problem categories (e.g., MATH levels 1-5).
- **Reward signal** = gradient norm, advantage magnitude, or pass-rate improvement (proxies for *learning gain*, not raw task reward).
- **Policy** = Thompson sampling or UCB over arms.

Adapts online as the model's capability changes. Outperforms fixed-difficulty and random baselines on MATH500, AMC, AIME in recent work.

### 3.8 Instruction-tuning curriculum
Random first epoch, then structured ordering from epoch 2+ (loss-based, attention-based, or prompt-length-based). Reported gains are small (~1-2 points) and dataset-dependent.

---

## 4. Representative Published Results

| Work | Setting | Metric(s) used | Result |
|---|---|---|---|
| Campos 2021 | BERT-style pretraining, GLUE | Multiple linguistic curricula | No compelling evidence of improvement. |
| Kim & Lee 2024 | Mistral-7B, Gemma-7B instruction tuning | Prompt length, attention, loss | Attention-based ordering gave the largest (still modest) gain. |
| Nair et al. 2024 | 1M-param GPT on Python code | Code complexity (Overall Metric) | Novel CL schedule beats baseline on code execution; also improved Code Llama 7B fine-tuning. |
| Elgaar & Rumshisky 2026 | Pythia 14M-410M, 300B tokens | AoA, word frequency, Verb Variation | Curricula reorder *within-phase* exposure but do not change the shared sequence of latent training phases. |
| SEC 2025 | RL fine-tuning on MATH | MAB over difficulty levels, gradient-norm reward | Beats random and fixed-difficulty curricula on MATH500/AMC/AIME. |
| LR-decay paper 2025 | Pretraining with quality curriculum | Quality score, varying LR schedules | Curriculum beats random under constant LR; advantage shrinks under standard cosine decay. |

---

## 5. The Learning-Rate Interaction (Critical)

A recurring finding: **a quality-ascending curriculum is incompatible with a standard decaying LR schedule.** The model sees the most valuable data when the LR has decayed to near zero, so the highest-quality gradients barely move the weights.

Mitigations that have been shown to recover the curriculum advantage:

1. **Moderate LR decay** — keep the final LR at a non-trivial fraction of peak.
2. **WSD schedule (Warmup–Stable–Decay)** with the high-quality stage aligned to the Stable phase and a short decay.
3. **Constant LR + weight averaging** — EMA or SMA over checkpoints, assigning non-decreasing weights to later (higher-quality) checkpoints. Outperforms WMA under a data curriculum.
4. **Decouple the schedule from the curriculum** — use cyclic or restart schedules across stages.

Ignoring this interaction is the single most common reason published curriculum results look weak.

---

## 6. Why CL Gains Are Modest at Scale

1. **Capacity substitutes for ordering.** Large models have enough capacity that gradient noise from random sampling is not the bottleneck; data quality is.
2. **Scale dominates curricula.** As shown on Pythia up to 410M, curriculum mainly reorders *within-phase* exposure; the latent training trajectory is shared across orderings.
3. **Data mixture weighting is simpler and more effective.** Most frontier labs tune per-source mixing weights (and schedule them across stages) rather than per-example orderings.
4. **Defining difficulty is hard.** A bad metric produces a detrimental curriculum. Attention-score and loss-based signals require a reference model, adding pipeline cost.
5. **Risk of forgetting.** If easy data appears only early, the model may regress on it; large capacity mitigates but does not eliminate this.

---

## 7. Practical Recipe (2024–2026 Consensus)

The strategies that consistently pay off for decoder-only LLMs:

1. **Two- or three-stage data curriculum**: bulk web → high-quality mid-training → short annealing on the cleanest slice.
2. **Length curriculum**: pretrain at a short context window, extend post-hoc with RoPE scaling and a short long-context stage.
3. **WSD learning-rate schedule** with the high-quality stage aligned to Stable + a moderate Decay.
4. **Weight averaging** (EMA/SMA) across late checkpoints, weighted toward the high-quality stage.
5. **Automatic curriculum (bandit-based) during RL fine-tuning** on reasoning tasks; aligns training batches with current model capability.
6. **Skip fine-grained per-example curricula during pretraining.** The engineering cost exceeds the measured gain above ~1B parameters.

---

## 8. Open Questions

- Does curriculum actually change the *learning trajectory* or only *within-phase exposure*? Current evidence up to 1B parameters favors the latter.
- Is there a curriculum that provably changes scaling-law constants rather than just the prefactor?
- Can automatic curricula (SEC-style) transfer from RL fine-tuning back to pretraining at scale?
- How should curriculum design interact with Muon/Shampoo-style optimizers whose effective LR dynamics differ from AdamW?
- What is the right curriculum for multi-modal decoder-only models (interleaved text/image/video tokens)?

---

## 9. Glossary

- **Curriculum Learning (CL)**: training on structured, non-random data order.
- **Pacing function**: mapping from training step to allowed difficulty range.
- **Self-paced learning**: curriculum driven by the model's own loss.
- **Mid-training**: second-stage pretraining on a curated, high-quality mixture.
- **WSD**: Warmup–Stable–Decay learning-rate schedule.
- **EMA / SMA / WMA**: exponential / simple / weighted moving average of checkpoint weights.
- **SEC (Self-Evolving Curriculum)**: bandit-based online curriculum for RL fine-tuning.
- **RLVR**: RL on verifiable rewards (math, code).
- **FIM**: Fill-in-the-Middle data reformatting.