Here is the refined guide. It has been structured into clean Markdown, with inline citations, specific publication dates, and callout boxes explaining the technical terminology.

***

# Pretraining Trajectories for Small LLMs (100M – 1.5B params)

A reference note compiling documented training trajectories, token budgets, and capability milestones for **decoder-only** language models in the 100M – 1.5B parameter range. Compiled from public training runs and scaling-law literature.

*Document Date: 2026-05-08*

> 📝 **Term Explained: Decoder-only Language Model**
> A neural network architecture (like GPT) that processes text left-to-right to predict the next word. It uses "self-attention" to look only at past context, making it ideal for text generation.

---

## 1. The Three Token-Budget Regimes

A single number ("optimal tokens per parameter") is misleading because three different scaling regimes coexist in the literature. 

> 📝 **Term Explained: D/N Ratio**
> Represents the ratio of **D**ataset size (in tokens) to **N**etwork size (in parameters). For example, training a 100M parameter model on 2B tokens gives a D/N ratio of 20×.

| Regime | Tokens / Param (D/N) | Optimization target | Examples |
|---|---|---|---|
| **Kaplan / GPT-3 era** *(Jan 2020)* | ~1.7× | Compute-optimal under old scaling assumptions *(Kaplan et al., 2020)*. | GPT-3 (175B params → 300B tokens) |
| **Chinchilla compute-optimal** *(Mar 2022)* | **~20×** | Minimum loss for fixed training **FLOPs** *(Hoffmann et al., 2022)*. | Chinchilla (70B → 1.4T), Pythia, Cerebras-GPT |
| **Modern over-training (inference-optimal)** *(2023-2024)* | **200× – 3000×** | Best loss per **inference FLOP**. Emphasizes making a model small for cheap deployment *(Dubey et al., 2024)*. | Llama-3 (70B → 15T, ~215×), TinyLlama (1.1B → 3T, ~2727×) |

> 📝 **Term Explained: FLOPs (Floating Point Operations)**
> A measure of computational work. *Training FLOPs* refers to the compute cost to train the model once. *Inference FLOPs* refers to the compute cost to run the model for a user. Overtraining spends more *training FLOPs* to minimize *inference FLOPs* later.

**Why this matters for a 100M – 1.5B model:**

*   If you only care about saving training compute: stop at ~20× params (Chinchilla).
*   If the model will be served / fine-tuned and inference cost matters: train to **at least 100×**, ideally more. Small models benefit disproportionately from over-training because their inference cost is fixed and small, while their capability ceiling is the dominant constraint.
*   For a **131M model**, the bands translate to:
    *   *Chinchilla floor:* **~2.6B tokens** (Do not fine-tune before reaching this).
    *   *Inference-optimal target:* **20B – 50B tokens**.
    *   *"TinyLlama-style" saturation:* **300B – 1T tokens** (Capabilities still increase, but with sharp diminishing returns).

---

## 2. Documented Reference Runs (100M – 1.5B)

### 2.1 GPT-2 Small (124M) — *Feb 2019*

| Metric | Value |
|---|---|
| **Tokens trained** | ~10B (WebText dataset) |
| **Tokens/param** | ~80× |
| **WikiText-103 Perplexity** | **37.5** (zero-shot) |
| **LAMBADA accuracy** | ~46% |

> 📝 **Terms Explained:**
> *   **Perplexity (pp):** A metric of how "surprised" a model is by a dataset. Lower is better. A perplexity of ~37 is the canonical baseline for a 124M dense decoder on diverse web text.
> *   **Zero-shot:** Evaluating the model without giving it any prior examples of the task.
> *   **LAMBADA:** A dataset that tests a model's ability to capture long-range context by asking it to predict the final word of a paragraph *(Paperno et al., 2016)*.

*Note:* Andrej Karpathy's `llm.c` reproduction (April 2024) reaches comparable validation loss on the FineWeb dataset at 10B tokens in ~90 minutes using 8×H100 GPUs.

### 2.2 Pythia Suite (EleutherAI) — *Feb 2023*

All Pythia models trained on **the Pile, 300B tokens** (≈1 epoch standard / 1.5 epochs deduped) with a uniform batch size of 2M tokens *(Biderman et al., 2023)*. 

| Model | Params | D/N ratio | Train cross-entropy (final) | Notes |
|---|---|---|---|---|
| pythia-70m | 70M | 4286× | ~3.8 | Severely over-trained for size |
| pythia-160m | 160M | 1875× | ~3.2 | |
| pythia-410m | 410M | 732× | ~2.8 | |
| pythia-1b | 1.0B | 300× | ~2.55 | |
| pythia-1.4b | 1.4B | 214× | ~2.45 | Closest to Llama-3-style overtraining ratio |

*(Training-loss values are approximate, derived from Pythia paper Fig. 2). The suite's primary value is **uniform data ordering**, making capability-vs-tokens curves directly comparable across scales.*

### 2.3 Cerebras-GPT — *Mar 2023*

Designed as a **strict Chinchilla-optimal** family trained on the Pile *(Dey et al., 2023)*.

| Model | Params | Tokens | D/N ratio |
|---|---|---|---|
| Cerebras-GPT-111M | 111M | 2.2B | 20× |
| Cerebras-GPT-256M | 256M | 5.1B | 20× |
| Cerebras-GPT-590M | 590M | 11.8B | 20× |
| Cerebras-GPT-1.3B | 1.3B | 26.0B | 20× |

These represent the canonical **"compute-optimal floor"** reference. You should not stop training below these numbers because the loss is still descending rapidly.

### 2.4 TinyLlama 1.1B — *Jan 2024*

Trained on **3T tokens** (SlimPajama + StarCoder datasets), representing extreme overtraining with D/N ≈ 2727× *(Zhang et al., 2024)*. 

| Step | Tokens | Capability Milestone |
|---|---|---|
| 50k | 105B | "Fluent base" emerging |
| 240k | 503B | Factual recall improving steadily |
| 480k | 1.0T | Strong base for Fine-Tuning (FT) |
| 715k | 1.5T | Beats Pythia-1.4B (which trained 5× less) on most benchmarks |
| 955k | 2.0T | Knees (inflection points) of most downstream curves |
| 1195k | 2.5T | Diminishing returns in evals |
| 1431k | 3.0T | Published final checkpoint |

It surpasses OPT-1.3B and Pythia-1.4B downstream by ~step 715k (1.5T tokens). Once the model consumes roughly **1300× its parameter count**, it overtakes a Chinchilla-optimal peer of the same size class.

### 2.5 OLMo-1B (Allen AI) — *Feb 2024*

Trained on **~2T tokens** from the Dolma dataset *(Groeneveld et al., 2024)*.

*   **Params:** 1.0B
*   **Tokens:** ~2T
*   **D/N Ratio:** ~2000×
*   **Final train loss:** ~2.6

OLMo provides the cleanest publicly available comparison point for full intermediate checkpoints at the 1B scale.

---

## 3. Capability Emergence Milestones

Synthesized from probe-style analyses *(Pythia learning-dynamics paper; TinyLlama eval log; OLMo training notes; Hewitt et al., 2024)* for 100M – 1.5B dense decoders trained on diverse web data.

> 📝 **Terms Explained:**
> *   **Repetition collapse:** A common failure mode early in training where the model gets stuck repeating a single word or phrase indefinitely.
> *   **Distinct-2:** A metric measuring the percentage of unique two-word phrases (bigrams) generated. High distinct-2 means the model has stopped repeating itself.
> *   **LoRA (Low-Rank Adaptation):** A lightweight fine-tuning technique that updates only a tiny fraction of weights *(Hu et al., 2021)*, making adaptation cheap.
> *   **Instruction-following:** The model's ability to act as a chatbot/assistant following explicit user commands (e.g., "Summarize this article").

| Milestone | Tokens (130M) | Tokens (1.5B) | Notes |
|---|---|---|---|
| **No NaNs / Loss descending cleanly** | < 0.05B | < 0.1B | If you can't reach this in 1 hour, your run is broken. |
| **Repetition collapse exits** | 0.1 – 0.3B | 0.3 – 1.0B | Distinct-2 metric climbs above ~0.7. |
| **Grammatical multi-clause output** | 0.5 – 1.5B | 1.0 – 3.0B | Clean English, but may abruptly switch topics. |
| **Topical coherence within paragraph** | 1.0 – 3.0B | 3.0 – 10B | "Wikipedia-flavoured hallucination"—well-formed but factually unreliable. |
| **Chinchilla floor (D/N ≈ 20)** | 2.6B | 30.0B | Below this, Fine-Tuning (FT) recovers poorly. Always stop later. |
| **Domain / Style FT becomes reliable** | 2.0 – 10B | 30.0 – 100B | LoRA on niche text yields clean transfer. |
| **Supervised task FT (NER, summaries)** | 10.0 – 50B | 100 – 300B | This is the workable target for 100M – 500M models. |
| **Instruction-following, even fragile** | **Not reliable** | 100 – 500B | Below ~1.5B params, instruction tuning is brittle regardless of pretraining *(Hewitt et al., 2024)*. |
| **Diminishing returns on pretraining** | 50.0 – 100B | 1.0 – 2.0T | Subsequent passes (epochs) over the *same* corpus flatten out. |
| **TinyLlama-style saturation** | 300B – 1.0T | 2.0 – 3.0T | Requires vast, diverse corpora (SlimPajama / FineWeb). Wikipedia alone is too small. |

---

## 4. Loss-Curve Shape (Qualitative)

Loss vs. log(tokens) for a small dense decoder follows a roughly piecewise-linear shape on log-log axes (meaning both the X and Y axes use a logarithmic scale).

```text
loss
 ^
 |\
 | \                         (1) Initial drop (~1B–10B tokens), steep slope
 |  \
 |   \____                   (2) Fluency plateau (slope ~ -0.05 per decade)
 |        \____              (3) Factual / world-knowledge regime
 |             \_____        (4) Data-bottleneck plateau (corpus exhausted)
 +-------------------> log10(tokens)
```

Region transitions for a **130M model** on Wikipedia-class data:

| Transition | Token Count | Visible Signal |
|---|---|---|
| (1) → (2) | 0.5 – 1B | Slope inflection; perplexity drops from ~1000 to ~80. |
| (2) → (3) | 5 – 20B | Probes produce recognizable proper nouns, dates, and syntactic structures. |
| (3) → (4) | 30 – 100B | Validation loss flattens *(if using Wikipedia only)*; need more diverse data to break through. |

*For a 1.5B model, the same shape holds, but each transition shifts right by ~10× tokens.*

---

## 5. Practical Recommendations

### Picking a Token Budget

| Goal | Minimum Tokens | Recommended | Ceiling on Wikipedia-only |
|---|---|---|---|
| **Architecture sanity check** | 0.5B | 1 – 2B | — |
| **Baseline for FT comparisons** | 2 × Chinchilla | 5 – 10 × Chinchilla | ~50B (1 epoch ≈ 3B) |
| **Domain FT release-grade base** | 50 × params | 100 – 200 × params | ~50B |
| **Supervised-task FT release-grade**| 100 × params | 200 – 500 × params | Needs additional web data |
| **Instruction-following base** | *Wait for ≥1.5B*| 500 – 2000 × params | Needs SlimPajama/FineWeb |

### Picking a Model Size

If your final use case is fine-tuning (FT), **buy capabilities with tokens**, rather than parameters:
*   A **130M model trained on 30B tokens** beats a **500M model trained on 3B tokens** for almost any downstream FT task.
*   **1.5B params** is generally the smallest scale at which instruction tuning produces non-fragile chat behavior.
*   Below 500M, expect a useful but narrow tool—excellent at a single classification/extraction task after FT, but poor at general open-ended chat.

### When to Stop Training

Watch for three independent stop signals:
1.  **Validation loss flattens for ≥10% of total budget:** The corpus is likely exhausted. More epochs won't help; only *diverse* data will.
2.  **Distinct-2 metric drops:** An early sign of memorization or repetition collapse, especially after epoch 2 on a small corpus.
3.  **Downstream eval plateaus:** Evaluate periodically on benchmarks like HellaSwag or ARC-easy. This is the only signal that strictly correlates with fine-tuning readiness.

### Case Study: Managing an In-Progress Run
*Scenario: You have a 131M param model at 1.3B tokens, Perplexity ≈ 50, approaching the end of Epoch 1 on a 2.76B token dataset.*

*   **Action:** Pretrain through the end of Epoch 2 (≈ 5.5B tokens). This gets the model clearly past the Chinchilla floor.
*   **Action:** Save the checkpoint with the best validation loss to serve as your FT base.
*   **Action:** Skip instruction tuning at this size; rely on targeted task fine-tuning.
*   **Action:** If testing a novel architecture against a standard baseline, ensure you train the baseline **on the exact same dataloader to the same step count**. Anything else introduces confounds.

---

## 6. Open Questions for Novel Architectures

When testing non-standard architectures (e.g., WaveFieldLLM, JEPA-LM, CALM), the published trajectories above are **upper bounds you must match, not targets you can claim to have beaten**. 

> 📝 **Term Explained: Confound**
> An outside variable that affects your results. If your novel model beats GPT-2, but your model trained on high-quality 2024 web data while GPT-2 trained on 2019 Reddit data, the data is a *confound*. You don't know if the architecture is better, or just the dataset.

*   **Loss vs. Tokens:** Match Pythia-160M's curve at the same D/N ratio on a comparable corpus *before* claiming the architecture "works."
*   **Compute-Optimality:** Match Cerebras-GPT-111M at exactly 20× tokens/param *before* claiming your model is more compute-efficient.
*   **Inference-Optimality:** Beat TinyLlama-1.1B's 1.5T-token checkpoint on downstream evals *before* claiming scaling laws favor your design.

The temptation is to declare success on perplexity alone. However, published loss curves are within 10–20% of one another at matched scale, so simply "tracking the curve" only confirms the architecture is **not broken**. Claims of true architectural advantage require **parameter-, data-, and step-matched baselines**.

---

## Sources

*   **Kaplan et al. (Jan 2020)** — *Scaling Laws for Neural Language Models* (arXiv:2001.08361)
*   **Hoffmann et al. (Mar 2022)** — *Training Compute-Optimal Large Language Models [Chinchilla]* (arXiv:2203.15556)
*   **Paperno et al. (2016)** — *The LAMBADA dataset: Word prediction requiring a broad discourse context* (arXiv:1606.06031)
*   **Epoch AI (Apr 2024)** — *Chinchilla Scaling: A Replication Attempt* (arXiv:2404.10102)
*   **Biderman et al. (Apr 2023)** — *Pythia: A Suite for Analyzing LLMs Across Training and Scaling* (arXiv:2304.01373). Checkpoints: [EleutherAI/pythia](https://github.com/EleutherAI/pythia)
*   **Dey et al. (Apr 2023)** — *Cerebras-GPT: A Family of Open, Compute-efficient, Large Language Models* (arXiv:2304.03208)
*   **Zhang et al. (Jan 2024)** — *TinyLlama: An Open-Source Small Language Model* (arXiv:2401.02385). Checkpoints: [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
*   **Groeneveld et al. (Feb 2024)** — *OLMo: Accelerating the Science of Language Models* (arXiv:2402.00838). Convergence notes via [allenai/OLMo #642](https://github.com/allenai/OLMo/issues/642)
*   **Dubey et al. (Jul 2024)** — *The Llama 3 Herd of Models* (arXiv:2407.09298)
*   **Hewitt, Liu, Manning, Liang (2024)** — *Instruction Following Without Instruction Tuning* ([Stanford CS PDF](https://cs.stanford.edu/~nfliu/papers/hewitt-liu-manning-liang.arxiv2024.pdf))
*   **Lin, Gou, et al. (Apr 2024)** — *Rho-1: Not All Tokens Are What You Need* (arXiv:2404.07965)
*   *(Various)* — *Balancing Continuous Pre-Training and Instruction Fine-Tuning* (arXiv:2410.10739)
*   *(Various)* — *Pre-training Small Base LMs with Fewer Tokens* (arXiv:2404.08634)
*   *(Various)* — *Characterizing Learning Curves During LM Pre-Training* (TACL 2024)
*   **Karpathy (2024)** — *llm.c — Reproducing GPT-2 (124M) on FineWeb 10B* ([GitHub Discussion](https://github.com/karpathy/llm.c/discussions/481))
*   **Databricks (2023)** — *How Long Should You Train Your Language Model?* ([Blog Post](https://www.databricks.com/blog/how-long-should-you-train-your-language-model))
*   **Brenndoerfer (2023)** — *Chinchilla Scaling Laws Explainer* ([Article](https://mbrenndoerfer.com/writing/chinchilla-scaling-laws-compute-optimal-llm-training))
