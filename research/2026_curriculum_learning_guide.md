# Curriculum Learning for Large Language Models

Oct 19, 2026 · @Nikolas Markou

A reference on how training-data ordering is used across the decoder-only LLM lifecycle: pretraining, mid-training, instruction tuning, RL fine-tuning and agent training. It covers difficulty metrics, pacing, static and adaptive strategies, the interaction with learning-rate schedules, a tested reference implementation, and what the evidence actually supports. Claims carry their evidence level; where a practice is a hypothesis rather than a finding, the text says so.

**Bottom line**

1. **Standard and well supported:** multi-stage data schedules (bulk web data, then curated mid-training, then a short anneal) and sequence-length schedules. Labs rarely call these curricula; structurally they are.
2. **Supported at small scale:** per-example difficulty curricula in pretraining speed up early convergence and, used as a warmup before random sampling, gave end-task gains of up to 3.5% (Zhang et al. 2026). Published pretraining evidence stops at roughly 1 to 1.5B parameters.
3. **A critical interaction:** quality-ascending curricula lose most of their advantage under standard learning-rate decay. Moderate decay plus checkpoint averaging recovers it (Luo et al. 2026).
4. **The strongest growth area:** adaptive curricula in RL fine-tuning on verifiable tasks, which pick problems by estimated learning gain (Chen et al. 2025; Do et al. 2026).
5. **No universal direction:** whether easy-to-hard or hard-to-easy wins depends on model strength and task difficulty (Jia et al. 2026).

## 1. The Core Idea

Standard LLM training samples data uniformly at random. **Curriculum learning (CL)** replaces uniform sampling with a structured order, classically from easy to hard, by analogy with how people learn (Bengio et al. 2009). The hypothesized benefits are faster early convergence, better-conditioned optimization, improved generalization, and more stable training on heterogeneous corpora.

### 1.1 Anatomy of a curriculum

A curriculum is fully specified by four choices.

1. **Difficulty measure.** A function D(x) that ranks training examples, buckets or sources.
2. **Pacing function.** A map from training progress to the range of difficulty the sampler may draw from.
3. **Scoring granularity.** Per example, per bucket (easy, medium, hard bins), or per source (web, books, code, curated).
4. **Adaptivity.** Static, fixed before training, or dynamic, driven by the model's current state.

### 1.2 Lineage

| Idea | Origin | Contribution |
| --- | --- | --- |
| Curriculum learning | Bengio et al. 2009 | Easy-to-hard ordering, framed as a continuation method |
| Self-paced learning | Kumar, Packer and Koller 2010 | The model's own loss selects which examples to train on |
| Automated curriculum learning | Graves et al. 2017 | A bandit over tasks, rewarded by learning progress rather than task reward |
| Competence-based curriculum | Platanios et al. 2019 | A square-root competence schedule for translation; the origin of root pacing |
| Pacing and transfer scoring | Hacohen and Weinshall 2019 | Pacing as a tunable hyperparameter; difficulty scored by a pretrained teacher |
| When do curricula work? | Wu, Dyer and Neyshabur 2021 | Random order matches curricula on standard benchmarks; gains appear mainly under tight compute budgets or noisy labels |

### 1.3 Five things called curriculum

The term covers practices with very different evidence behind them. Keeping them apart is the single most useful habit when reading the literature.

| Form | What is ordered | Status at LLM scale |
| --- | --- | --- |
| Per-example difficulty ordering | Individual documents, by a difficulty score | The classical form; modest, inconsistent gains |
| Multi-stage data schedules | Mixtures of sources, by quality or domain | Standard practice, usually called mid-training or annealing rather than CL |
| Sequence-length schedules | Context length | Standard practice |
| Adaptive selection in RL fine-tuning | Prompts or problem categories, by estimated learning gain | Growing, with consistent reported gains |
| Data selection and reweighting | Nothing: it changes what is seen, not when | Not a curriculum, but often the source of gains attributed to one |

## 2. Difficulty Metrics

A difficulty metric is useful only if it tracks what the model finds hard. Human-centred metrics are cheap and interpretable; model-centred metrics track the learner but need a reference model, and sometimes a full surrogate training run.

### 2.1 Metric families

| Family | Signal | Cost | Evidence and caveats |
| --- | --- | --- | --- |
| Information density | Compression ratio | Very cheap | Among the most effective pretraining signals (Zhang et al. 2026); confounded with length on short texts |
| Lexical diversity | MTLD | Cheap | Among the most effective pretraining signals (Zhang et al. 2026) |
| Readability | Flesch Reading Ease, grade level | Cheap | Effective in pretraining (Zhang et al. 2026); English-centric |
| Sequence length | Token count | Free | Valuable as a compute schedule (Section 4.2), weak as a difficulty proxy |
| Word frequency, developmental | Token rarity, age of acquisition, verb variation | Cheap given a frequency pass or lexicon | Cognitively motivated; limited effect at small scale (Elgaar and Rumshisky 2026) |
| Syntactic complexity | Parse depth, clause count | Expensive at web scale | Requires a parser |
| Reference-model perplexity | Perplexity under a reference model | One forward pass per document | Conflates hard with noisy; filter for quality first |
| Training-data influence | Average influence on other examples (TracIn-style) | High: needs a surrogate training run | Over 10 points above random order in low-data pretraining (Schoenegger et al. 2025) |
| Attention statistics | Variance of attention scores over the prompt | One forward pass | Best of three signals for 7B instruction tuning, with slight gains (Kim and Lee 2024) |
| Prompt length | Input token count | Free | Weak but cheap |
| Reasoning depth | Step count in a teacher's chain of thought | One teacher generation per example | Proposed with testable hypotheses, not yet validated (Jung and Jung 2025) |
| Task-native labels | Math difficulty levels; cyclomatic complexity and Halstead difficulty for code | Free when labels exist | Strong when available (Naïr et al. 2024) |
| Data quality score | Heuristic or classifier-based quality | Already computed in modern pipelines | The dominant ordering signal in practice; it measures value, not difficulty |

### 2.2 Difficulty is not quality

Much of the literature, and most production practice, orders data by **quality** rather than **difficulty**, and the two often run in opposite directions. Curated textbooks are high quality and easy to read; raw web text is low quality and often hard, because noise is hard to predict. A "low to high quality" mid-training schedule is therefore frequently hard-to-easy in readability terms. Results from one axis do not transfer to the other.

Model-perceived difficulty also moves during training: an example that is hard early can be easy later, and the reverse (the "difficulty shift" of Zhang et al. 2025). Static scores go stale, which is the main argument for adaptive curricula (Sections 6.2 and 6.3).

### 2.3 Formal definitions

Perplexity under a reference model:

```latex
D_{\mathrm{ppl}}(x) = \exp\Big(-\frac{1}{N}\sum_{i=1}^{N}\log P_{\mathrm{ref}}(x_i \mid x_{<i})\Big)
```

Information density, where higher means less redundant and therefore harder:

```latex
D_{\mathrm{comp}}(x) = \frac{|\mathrm{compress}(x)|}{|x|}
```

Reasoning depth, counted on a teacher's chain of thought under fixed formatting:

```latex
\mathrm{DoT}(x) = \big|\,\mathrm{steps}\big(\mathrm{CoT}_{\mathrm{teacher}}(x)\big)\big|
```

**Check the direction.** Some papers report the inverse, |x| / |compress(x)|, as the "compression ratio"; there, higher means more redundant and easier. Sorting on the wrong orientation silently turns a curriculum into an anti-curriculum.

**Validate the metric before using it.** Early in training, correlate the metric with the model's own loss on a held-out sample. A metric uncorrelated with loss produces a curriculum in name only.

## 3. Pacing Functions and Scheduling

### 3.1 Pace in quantiles, not raw scores

Let τ ∈ \[0, 1\] be training progress, the fraction of planned steps or tokens. The pacing function returns a **competence** c(τ) ∈ \[c₀, 1\]: the fraction of the difficulty-sorted data the sampler may draw from, starting from an initial fraction c₀ (Platanios et al. 2019).

Pacing on quantiles rather than on raw difficulty thresholds matters. A threshold such as D\_min + g(τ)(D\_max − D\_min) depends on the score scale: one extreme outlier stretches the range so that most data unlocks almost at once, or almost never. Quantile pacing is invariant to the scale and skew of any metric, so the same schedule works for compression ratio, perplexity or influence.

The four common shapes, each running from c₀ at τ = 0 to 1 at τ = 1:

```latex
\begin{aligned}
\text{linear:}\quad & c(\tau) = c_0 + (1-c_0)\,\tau \\
\text{quadratic:}\quad & c(\tau) = c_0 + (1-c_0)\,\tau^2 \\
\text{inverse quadratic:}\quad & c(\tau) = c_0 + (1-c_0)\,\big(1-(1-\tau)^2\big) \\
\text{root:}\quad & c(\tau) = \min\Big(1,\ \sqrt{\tau\,(1-c_0^2) + c_0^2}\Big)
\end{aligned}
```

Competence at five points of training, with c₀ = 0.1 (computed with the reference implementation in Section 12):

| Pacing | τ = 0 | 0.25 | 0.5 | 0.75 | 1 | Character |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.10 | 0.33 | 0.55 | 0.78 | 1.00 | Uniform unlocking; the baseline to beat |
| Quadratic | 0.10 | 0.16 | 0.33 | 0.61 | 1.00 | Long easy phase, late rush to hard data |
| Inverse quadratic | 0.10 | 0.49 | 0.78 | 0.94 | 1.00 | Fast early diversity, slow phase-in of the hardest data |
| Root | 0.10 | 0.51 | 0.71 | 0.87 | 1.00 | Aggressive early unlocking |

Other pacing families: **step functions** with hard stage boundaries (multi-stage pretraining), **self-paced** windows in which the model's current loss decides the next batch (Kumar et al. 2010), and **adaptive** policies that select categories by estimated learning gain (Section 6.3). The best shape interacts with the metric and the budget; treat pacing as a hyperparameter tuned against a random-order baseline (Hacohen and Weinshall 2019).

### 3.2 Bucket scheduling

Partition the data into K difficulty quantiles B₁, …, B\_K and sample bucket k at progress τ with probability proportional to a weight w\_k(τ) that shifts from favouring low k early toward uniform, or toward high k, late. This is quantile pacing at coarse granularity. It is the most common implementation because it maps onto sharded storage: each bucket can be a set of shards.

### 3.3 Phased, paced, interleaved and warmup schedules

| Strategy | Description | Strength | Risk |
| --- | --- | --- | --- |
| Phased (vanilla) | Each difficulty level in turn, monotonically | Clear progression | Forgetting early data; no late diversity; worst interaction with LR decay |
| Paced sampling | Uniform sampling from the unlocked easy prefix | Easy data stays visible | Hard data is scarce early |
| Interleaved | All difficulties always present, in shifting ratios | Diversity throughout | Weaker ordering signal |
| Curriculum as warmup | Curriculum for the first part of training, then uniform | Largest sustained gains in a 200-model study (Zhang et al. 2026) | Warmup length is a new hyperparameter |
| Replay | A fixed share of each batch drawn uniformly from all data | Cheap insurance against forgetting | Dilutes the curriculum |

## 4. Pretraining Curricula

### 4.1 Multi-stage data schedules: the de facto standard

1. **Bulk stage.** Heterogeneous web-scale data (CommonCrawl derivatives such as FineWeb, DCLM, RedPajama), with code and math already present.
2. **Mid-training.** The mixture shifts toward curated sources: textbooks, code, math, heavily filtered web and synthetic data. OLMo 2 trains its final phase on a dedicated high-quality mix (Team OLMo 2024); Phi-3 moves from filtered web to more heavily filtered web plus synthetic data (Abdin et al. 2024).
3. **Anneal.** A short learning-rate decay over the highest-quality slice. Llama 3 upsampled high-quality code and math while annealing at the end of pretraining, and used small annealing runs to value candidate datasets (Grattafiori et al. 2024). Upsampling target domains only at the end of training gives outsized gains for its cost (Blakeney et al. 2024).

This is a quality schedule over sources at stage granularity (Section 2.2), usually labelled data-mixture scheduling or mid-training rather than curriculum learning. Its payoff depends on the learning-rate schedule (Section 5).

**Domains are rarely introduced in sequence.** A progression such as general text, then technical documents, then papers, then code has no published support at scale. Code and math are usually present from the start and upweighted later; a domain introduced only at the end receives little training at a high learning rate.

**Keep replay in later stages.** Retaining a share of earlier-stage data limits forgetting; in continual pretraining, replay combined with learning-rate re-warming matches retraining from scratch at a fraction of the cost (Ibrahim et al. 2024).

An illustrative stage configuration, as fractions of the token budget:

```python
# Illustrative only: real fractions and mixes are tuned per model.
stages = [
    {"name": "bulk",   "budget": 0.85, "lr": "stable",
     "mix": {"web": 0.80, "code": 0.12, "math": 0.04, "books": 0.04}},
    {"name": "mid",    "budget": 0.12, "lr": "stable",
     "mix": {"filtered_web": 0.40, "code": 0.20, "math": 0.15,
             "textbooks": 0.10, "synthetic": 0.05, "bulk_replay": 0.10}},
    {"name": "anneal", "budget": 0.03, "lr": "decay",
     "mix": {"best_slice": 0.80, "bulk_replay": 0.20}},
]
```

### 4.2 Sequence-length schedules

Short contexts first, longer later. Staged short-then-long training speeds up language model training (Press et al. 2021). Sequence-length warmup stabilizes GPT training and allows larger batches and learning rates (Li et al. 2022). A block-size curriculum speeds up BERT pretraining (Nagatsuka et al. 2021). Variable-length bucketing with a length curriculum trains an 8k-context model at the cost of a 2k-context baseline (Pouransari et al. 2024).

**Long-context extension** is the same idea at the end of training: pretrain at a short window, then run a short stage at long context with rotary-position scaling, such as position interpolation (Chen et al. 2023) or YaRN (Peng et al. 2023). The data matters more than the volume: a few billion tokens of length-upsampled data suffice for 128K context (Fu et al. 2024). Llama 3 extended from 8K to 128K in stages (Grattafiori et al. 2024); DeepSeek-V3 from 4K to 32K to 128K with YaRN (DeepSeek-AI 2024).

**Keep tokens per step constant.** Shrink the batch as sequences grow. Otherwise the length schedule is also an unannounced batch-size schedule, and comparisons with a fixed-length baseline are confounded.

### 4.3 Per-example difficulty curricula

Score every document, sort into quantiles, and sample by a pacing function (Section 3). The largest systematic study trained over 200 models on up to 100B tokens with three strategies and six metrics (Zhang et al. 2026). Curricula reached baseline performance in 18 to 45% fewer steps; used as a warmup before random sampling, they gave sustained gains of up to 3.5%. Compression ratio, MTLD and Flesch Reading Ease were the most effective signals.

Model-centred difficulty can do better in small-data regimes. Ordering by training-data influence beat random order by over 10 points on BabyLM-scale pretraining, but the influence scores require a surrogate model trained first in random order (Schoenegger et al. 2025).

The counter-evidence is substantial. Linguistically motivated curricula showed no compelling benefit for GLUE transfer (Campos 2021). On Pythia-family models, frequency, age-of-acquisition and verb-variation curricula traversed the same latent training phases as random order; they changed which data the model saw within each phase and how long it stayed there, not the phases themselves (Elgaar and Rumshisky 2026).

### 4.4 Objective-level schedules

Fill-in-the-middle (FIM) training in code models is often described as an objective curriculum. It is not: FIM is a data transformation applied at a fixed rate throughout pretraining. At a 50% rate it costs little left-to-right ability (Bavarian et al. 2022), and StarCoder and DeepSeek-Coder apply it at a fixed rate (Li et al. 2023; Guo et al. 2024). Scheduling objectives, for example denoising early and pure next-token prediction late, is a plausible design with little published evidence at LLM scale.

### 4.5 Growing the model with the data

Curriculum-Guided Layer Scaling (CGLS) adds layers progressively while raising data difficulty (Singh et al. 2025). At about 100M parameters the curriculum moves from synthetic short stories to web text; at 1.2B it moves from general to highly technical content, stratified with a classifier over DCLM. It improved zero-shot results on knowledge-intensive and reasoning benchmarks such as PIQA and ARC, at up to 1.2B parameters and 20B tokens. The paper's central claim is that progressive stacking pays off when paired with the curriculum.

```
for stage k = 1 .. n:
    grow the model: add layers, so L(k) > L(k-1)
    train the new layers on data subset D(k), harder than D(k-1),
        with earlier layers frozen
    unfreeze and train the full model briefly
```

## 5. The Learning-Rate Interaction

**A quality-ascending curriculum conflicts with a decaying learning rate.** The model meets its most valuable data when the learning rate has decayed toward zero, so the best gradients barely move the weights.

The effect is measured. With a constant learning rate, a quality-ascending curriculum substantially outperforms random shuffling; under standard decay schedules most of that advantage disappears (Luo et al. 2026, 1.5B-parameter models on 30B tokens). The authors identify this as a critical reason earlier curriculum studies reported limited gains.

Four mitigations, in order of evidence:

1. **Moderate decay.** End training at a learning rate only moderately below the peak, so late, high-quality data still moves the weights (Luo et al. 2026).
2. **Replace decay with checkpoint averaging.** Keep the learning rate high and average the final few checkpoints, weighted toward the latest, high-quality stage (Curriculum Model Averaging; Luo et al. 2026). Combined with moderate decay, it improved the average benchmark score by 1.64% over random shuffling without extra data refinement. Averaging helped more under a curriculum than under uniform ordering. It extends earlier work on weight averaging at high learning rates (Sanyal et al. 2024).
3. **Warmup-stable-decay (WSD).** Keep the learning rate flat through a long stable phase and confine decay to a short final phase (Hu et al. 2024; Hägele et al. 2024). Most curriculum data then arrives at a high learning rate. Placing the highest-quality data in the decay phase is common practice; keep that decay short or moderate so the data is not wasted.
4. **Re-warm at stage boundaries.** Raise the learning rate again when a new data stage begins, then decay. The evidence comes from continual pretraining, where re-warming plus replay matched retraining from scratch (Ibrahim et al. 2024).

**Design the data schedule and the learning-rate schedule together.** A curriculum evaluated under a schedule tuned for random order is evaluated under the wrong schedule.

## 6. Post-Training Curricula

### 6.1 Static curricula for instruction tuning

Instruction data can be graded by complexity. Tree-Instruct raises an instruction's complexity by adding nodes to its semantic tree, producing graded levels, and the authors report benefits from training easy to hard (Zhao et al. 2023). Kim and Lee (2024) trained Mistral-7B and Gemma-7B with a random first epoch, then from the second epoch ordered data by prompt length, attention-score variance or loss. Curricula improved slightly on random shuffling, with the attention criterion best.

An illustrative tiering for synthesized instruction data:

| Tier | Content |
| --- | --- |
| 1 | Single-step factual questions |
| 2 | Multi-step reasoning with explicit steps |
| 3 | Complex instructions with implicit requirements |
| 4 | Ambiguous or open-ended tasks |

For reasoning data, depth of thought counts the steps in a teacher's chain of thought: "What is 2+2?" has depth 1; "If x = 3, what is 2x + 1?" has depth 2; a word problem needing setup, equations, solving and checking has depth 5 or more. The proposal is explicit that its premise is a hypothesis to be tested (Jung and Jung 2025).

### 6.2 Competence-aware instruction tuning

Static metrics produce rigid curricula that cannot follow the model's changing ability. CAMPUS adds dynamic selection among sub-curricula, competence-aware adjustment of the schedule, and several difficulty perspectives at once (Li et al. 2025). On LLaMA-7B and 13B, evaluated on GSM8K, HumanEval and MT-Bench, it averaged 38.2 against 36.5 for random shuffling, at a cost of more training time (54 against 47 hours at 7B).

A simple competence-aware rule in the same spirit: track held-out accuracy per difficulty bucket and sample most from buckets near the **frontier**, where accuracy is neither near 0 nor near 1. Weighting buckets purely by their gap to a target accuracy over-samples data the model cannot yet learn from.

### 6.3 Adaptive curricula for RL fine-tuning

In group-based policy optimization on verifiable rewards (GRPO and relatives), the learning signal concentrates on problems the model sometimes solves. If every rollout for a prompt gets the same reward, all solved or all failed, the group's advantages are zero and so is its gradient. Every adaptive method below exploits this in some form.

| Method | Unit selected | Signal | Selection rule |
| --- | --- | --- | --- |
| Dynamic sampling (Yu et al. 2025) | Prompt | Group accuracy strictly between 0 and 1 | Filter out zero-advantage groups |
| Kimi k1.5 (Kimi Team 2025) | Prompt | Difficulty and per-problem success rate | Easy-to-hard sampling plus prioritized resampling of failed problems |
| SEC (Chen et al. 2025) | Category: difficulty level or problem type | Mean absolute advantage, a proxy for learning gain | Non-stationary bandit, TD(0) value updates, softmax selection |
| SPaCe (Do et al. 2026) | Cluster of semantically similar problems | Solve rates and learning progress | Thompson-sampling bandit over clusters |

SEC improved on a random curriculum across planning, inductive reasoning and mathematics, generalized better to harder out-of-distribution problems, and kept skills balanced when training on several domains at once. In its controlled comparison, a hard-to-easy order on the Countdown task did much worse than random. SPaCe, first released as SPaRFT, matched or beat strong baselines with up to 100 times fewer training samples, and needed both clustering and adaptive selection to do so.

Two cautions. These are RL fine-tuning results on verifiable reasoning rewards, not RLHF on human preferences. And "fewer samples" is not "less compute": rollouts, scoring and bandit updates have their own cost.

The lineage runs back to learning-progress bandits (Graves et al. 2017) and teacher-student curriculum learning (Matiisen et al. 2020); Portelas et al. (2020) survey the family.

## 7. Domain Curricula

Evidence here is thin. Most orderings in this section are design hypotheses; run them as experiments against a baseline that mixes all domains from the start.

### 7.1 Multilingual

Hypothesized orderings: high-resource to low-resource languages, similar to distant languages, shared script to different scripts.

The risk is structural: a language introduced late gets little training at a high learning rate, and a long monolingual first phase leaves less budget for everything after it. Most multilingual pretraining therefore schedules the **sampling distribution** across languages rather than the order in which they appear, for example by temperature sampling, or by UniMax, which caps how often low-resource data is repeated (Chung et al. 2023). A schedule over the sampling temperature is the mixture-level analogue of a language curriculum.

### 7.2 Code

The clearest result is small-scale. An Overall Metric averaging cyclomatic complexity and Halstead difficulty, used in three-stage schedules, significantly improved code-execution accuracy in 1M-parameter models, with smaller effects on code completion; the curriculum also helped a fine-tuned Code Llama 7B on execution (Naïr et al. 2024).

A progression from natural-language descriptions to pseudocode, simple scripts and then complex programs is plausible and untested at scale. In production pretraining, code is present from the start and FIM is applied at a fixed rate (Section 4.4).

### 7.3 Scientific and technical domains

The hypothesis: encyclopedic summaries, then textbooks, then review articles, then primary research. What is used in practice is the mid-training form, upweighting curated technical text late in training (Section 4.1). The closest tested instance is the 1.2B CGLS setting, which moves from general to highly technical content (Singh et al. 2025).

## 8. Curricula for Agents

Three different things travel under this heading: training curricula for agent policies, which have RL evidence behind them; deployment staging, which is a governance practice rather than a learning method; and design hypotheses not yet tested on LLM agents. Keep them apart.

### 8.1 What the evidence supports

The RL literature's strongest result is **automatic task generation at the frontier of the agent's ability**:

- **Unsupervised environment design** generates or selects environments the agent can almost solve: PAIRED (Dennis et al. 2020), Prioritized Level Replay, which replays levels with high estimated learning potential (Jiang et al. 2021), and POET (Wang et al. 2019).
- **Reverse curricula** start near the goal and move the start state outward, which helps long-horizon tasks with sparse rewards (Florensa et al. 2017).
- **For LLM web agents**, WebRL generates new training tasks from the agent's own failures, a self-evolving online curriculum, and substantially raised open models' success rates on WebArena-Lite (Qi et al. 2024).

The principle is the one in Section 6.3: train where success is neither certain nor impossible.

### 8.2 Hypothesized progressions

Plausible, and untested at LLM scale:

| Dimension | Progression |
| --- | --- |
| Planning horizon | Single step → 2 to 3 steps → 5 to 10 steps → long horizon |
| Environment | Deterministic sandbox → simplified real environment → full environment with noise |
| Multi-agent | Single agent → cooperative pairs → small teams → mixed-motive → adversarial |
| Error recovery | Avoid obvious errors → recover from minor failures → correct hallucinated actions → detect subtle drift |

### 8.3 Deployment staging is not a training curriculum

Unlocking an agent's action space in steps (read-only operations, then simple edits, then execution and deployment, then autonomy with safety checks) is sound deployment practice. It governs what the agent is permitted to do, not how it learns. Whether training on a restricted action space first also helps learning is a separate claim that needs its own evidence. Evaluate safety properties at each autonomy level rather than assuming they carry over from the previous one.

## 9. Representative Published Results

Ordered by year. Note the scale column: almost all pretraining evidence comes from models of 1.5B parameters or fewer.

| Work | Setting | Signal | Result |
| --- | --- | --- | --- |
| Campos 2021 | Language model pretraining, GLUE transfer | Linguistically motivated curricula | No compelling evidence of improvement |
| Nagatsuka et al. 2021 | BERT pretraining | Input block size | Faster convergence, better downstream results |
| Wu et al. 2021 | Image classification | Curricula, anti-curricula and random order | No benefit in standard settings; benefits under tight budgets or noisy labels |
| Li et al. 2022 | GPT pretraining | Sequence-length warmup | Stable training at larger batch sizes and learning rates |
| Kim and Lee 2024 | Mistral-7B and Gemma-7B instruction tuning | Prompt length, attention variance, loss | Slight gains; attention-based ordering best |
| Naïr et al. 2024 | 1M-parameter code models; Code Llama 7B fine-tuning | Cyclomatic complexity and Halstead difficulty | Significant gains on code execution, smaller on completion |
| Pouransari et al. 2024 | Decoder-only pretraining | Variable sequence-length buckets | 8k-context model at the cost of a 2k-context baseline |
| Chen et al. 2025 (SEC) | RL fine-tuning on planning, inductive reasoning and math | Bandit over categories, absolute-advantage reward | Beats a random curriculum; better out-of-distribution generalization |
| Li et al. 2025 (CAMPUS) | LLaMA-7B and 13B instruction tuning | Competence-aware, multiple difficulty perspectives | Average 38.2 against 36.5 for random order, with longer training time |
| Schoenegger et al. 2025 | BabyLM-scale pretraining | Training-data influence | Over 10 points above random order |
| Singh et al. 2025 (CGLS) | 100M and 1.2B pretraining with layer growth | Synthetic-to-web; general-to-technical | Better zero-shot results on knowledge-intensive QA |
| Zhang et al. 2026 | Over 200 models, up to 100B tokens | Six metrics, three strategies | 18 to 45% fewer steps to baseline; up to 3.5% sustained gain as warmup |
| Luo et al. 2026 | 1.5B models, 30B tokens | Quality-ascending order under several LR schedules | Advantage disappears under standard decay; +1.64% with moderate decay and averaging |
| Elgaar and Rumshisky 2026 | Pythia-family pretraining | Word frequency, age of acquisition, verb variation | Same latent training phases as random order; within-phase exposure differs |
| Jia et al. 2026 | Math reasoning with Llama, Mistral and Gemma 7 to 8B models | Five metric families, forward and reverse order | No universal winner; direction depends on model strength and task difficulty |
| Do et al. 2026 (SPaCe) | RL fine-tuning on reasoning benchmarks | Semantic-difficulty clusters, Thompson sampling | Comparable or better accuracy with up to 100 times fewer samples |

## 10. Limits, Anti-Curricula and Pitfalls

### 10.1 Why gains are modest at scale

1. **The optimizer schedule undoes the curriculum.** Under standard decay, late high-value data arrives at a near-zero learning rate (Section 5).
2. **Curricula reorder exposure; they do not change the trajectory.** Orderings traverse the same latent training phases as random order and differ in how long the model spends in each (Elgaar and Rumshisky 2026).
3. **Selection and reweighting carry most of the value.** Per-source mixture weights (Xie et al. 2023; Ye et al. 2024) and quality filtering change what the model sees; scheduled across stages, they capture the ordering effect that matters.
4. **Difficulty is hard to define and moves during training.** A bad metric produces a harmful curriculum, and a good static one goes stale (Section 2.2).
5. **Forgetting.** Data seen only early can regress later; capacity mitigates this but does not remove it.
6. **Curricula help most where random order is weak:** tight budgets, noisy labels and small data (Wu et al. 2021; Schoenegger et al. 2025). Frontier pretraining, with clean data and full budgets, rarely meets those conditions.

### 10.2 Direction: easy-to-hard, hard-to-easy, or neither

No direction wins universally. In a controlled study on mathematical reasoning, stronger models benefited from easy-to-hard order on simpler tasks, while weaker models or harder tasks often favoured hard-to-easy, and prioritizing samples on which the model's decisions were uncertain helped further (Jia et al. 2026). In a different setting, a hard-to-easy order on Countdown did much worse than random (Chen et al. 2025).

Treat direction as a hyperparameter, or sidestep it: adaptive methods that target the frontier of current ability (Section 6.3) do not need a fixed direction.

### 10.3 Pitfalls

| Issue | Cause | Mitigation |
| --- | --- | --- |
| Advantage vanishes by the end of training | High-value data arrives at a near-zero learning rate | Moderate decay, checkpoint averaging, WSD (Section 5) |
| Stalled convergence | Metric misaligned with what the model finds hard | Validate the metric against early training loss |
| Silent anti-curriculum | Metric orientation inverted | Check the direction on a labelled sample |
| Ending on noise | Perplexity ranks garbage as hardest | Filter for quality before scoring difficulty |
| Forgetting | Early data absent late in training | Replay or interleaving |
| Confounded comparison | Different tokens per step or different data seen | Constant tokens per step; matched data, tokens and schedule; several seeds |
| Overhead | Scoring a web-scale corpus | Cheap metrics, precomputed once, bucketed by shard |
| Stale difficulty | Difficulty shifts as the model learns | Re-score periodically, or use an adaptive method |

### 10.4 When to use what

| Scenario | Curriculum type | Expected benefit | Evidence |
| --- | --- | --- | --- |
| Long-context models | Sequence-length schedule | Large compute savings | Strong |
| Large-scale pretraining | Multi-stage quality schedule, co-designed with the LR schedule | Reliable gains; standard practice | Strong |
| Pretraining on a tight budget | Per-example warmup on cheap metrics | Faster early convergence, small lasting gains | Moderate, up to about 1.5B |
| Low-data pretraining | Influence-based ordering | Large gains possible | Moderate, BabyLM scale |
| RL fine-tuning on verifiable tasks | Zero-advantage filtering; bandit over categories | Sample efficiency, better generalization | Moderate to strong |
| Instruction tuning | Static or competence-aware ordering | Slight, dataset-dependent | Weak to moderate |
| Domain adaptation and continual pretraining | New-domain stage with replay and LR re-warming | Adaptation without forgetting; the replay, not the ordering, prevents forgetting | Moderate |

## 11. Practical Recipe

The practices that hold up for decoder-only LLMs, in the order a training run meets them.

**Pretraining**

1. **Run a two- or three-stage data schedule:** bulk web data, then a curated mid-training mix, then a short anneal on the cleanest slice. Keep a share of earlier-stage data in every later stage.
2. **Schedule sequence length:** train the bulk stage at a short window, then extend in a final long-context stage with rotary-position scaling on length-upsampled data. Hold tokens per step constant throughout.
3. **Co-design the learning-rate schedule with the data schedule:** use WSD or a moderate decay, and consider averaging the final checkpoints with weight on the last stage.
4. **On small budgets, try a curriculum warmup:** order by compression ratio, MTLD or readability for the first part of training, then switch to uniform sampling.
5. **Treat fine-grained per-example curricula at frontier scale as unproven.** Published pretraining evidence ends at roughly 1.5B parameters; validate before paying the scoring cost.

**Post-training**

6. **In RL fine-tuning, drop zero-advantage prompts,** and consider a bandit over difficulty categories rewarded by absolute advantage.
7. **In instruction tuning, settle data quality first,** then try ordering; expect slight gains.

**Evaluation discipline**

8. **Compare against random order with matched data, tokens, learning-rate schedule and seeds.** Report both early-convergence and final metrics, and test both directions.

## 12. Reference Implementation

A compact, dependency-light implementation of the components above, tested with Python 3 and NumPy. Design choices worth copying:

- **Difficulty is oriented so that higher means harder,** and texts too short for compression to be meaningful are left unscored rather than ranked as hard.
- **Pacing works in quantile space** with an initial fraction c₀, so it is independent of the metric's scale.
- **Warmup and replay are first-class options:** the curriculum can stop part-way through training, and a fixed share of every batch can be drawn uniformly.
- **Sampling is vectorized and seeded** through one random generator, so runs are reproducible.
- **The length schedule holds tokens per step constant.**
- **The RL components** are an SEC-style advantage bandit and the zero-advantage filter.

```python
"""Reference implementation: difficulty scoring, quantile pacing, and
adaptive category selection for curriculum training."""
from __future__ import annotations

import math
import zlib
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- scoring
def compression_difficulty(text: str, min_bytes: int = 256) -> float:
    """Information density: compressed size / raw size.

    Higher = less redundant = treated as harder. Texts shorter than
    min_bytes return NaN, because zlib's fixed overhead makes short texts
    look incompressible and would confound difficulty with length.
    """
    raw = text.encode("utf-8")
    if len(raw) < min_bytes:
        return float("nan")
    return len(zlib.compress(raw, 9)) / len(raw)


def perplexity_difficulty(nll_per_token: Sequence[float]) -> float:
    """Perplexity from per-token negative log-likelihoods of a reference
    model: exp(mean NLL). High values mix genuinely hard text with noise;
    filter for quality before using this as difficulty."""
    return float(math.exp(np.mean(nll_per_token)))


# ---------------------------------------------------------------- pacing
def pacing(tau: float, kind: str = "root", c0: float = 0.1) -> float:
    """Competence c(tau): the fraction of the difficulty-sorted data that is
    available at training progress tau in [0, 1]. Pacing operates on
    quantiles, not raw scores, so it is robust to skewed score scales."""
    tau = min(max(tau, 0.0), 1.0)
    if kind == "linear":
        g = tau
    elif kind == "quadratic":           # slow start
        g = tau ** 2
    elif kind == "inverse_quadratic":   # fast start, slow finish
        g = 1.0 - (1.0 - tau) ** 2
    elif kind == "root":                # Platanios et al. (2019)
        return min(1.0, math.sqrt(tau * (1.0 - c0 ** 2) + c0 ** 2))
    else:
        raise ValueError(f"unknown pacing: {kind}")
    return c0 + (1.0 - c0) * g


class CurriculumSampler:
    """Samples example indices according to a quantile-paced curriculum.

    curriculum_fraction: share of the run under the curriculum; after it,
        sampling is uniform ("curriculum as warmup").
    replay: share of each batch drawn uniformly from all data, which keeps
        hard data visible early and easy data visible late.
    """

    def __init__(self, scores: Sequence[float], total_steps: int,
                 kind: str = "root", c0: float = 0.1,
                 curriculum_fraction: float = 1.0, replay: float = 0.0,
                 seed: int = 0):
        s = np.asarray(scores, dtype=float)
        s = np.where(np.isnan(s), np.nanmedian(s), s)  # unscored -> median
        self.order = np.argsort(s, kind="stable")       # easy -> hard
        self.n = len(self.order)
        if self.n == 0:
            raise ValueError("empty dataset")
        self.total_steps = total_steps
        self.kind, self.c0 = kind, c0
        self.curriculum_fraction = curriculum_fraction
        self.replay = replay
        self.rng = np.random.default_rng(seed)

    def competence(self, step: int) -> float:
        tau = step / max(1, self.total_steps)
        if tau >= self.curriculum_fraction:
            return 1.0
        return pacing(tau / self.curriculum_fraction, self.kind, self.c0)

    def sample(self, step: int, batch_size: int) -> np.ndarray:
        k = max(1, int(round(self.competence(step) * self.n)))
        n_replay = self.rng.binomial(batch_size, self.replay)
        paced = self.order[self.rng.integers(0, k, batch_size - n_replay)]
        replayed = self.rng.integers(0, self.n, n_replay)
        batch = np.concatenate([paced, replayed])
        self.rng.shuffle(batch)
        return batch


# ---------------------------------------------------------------- length
def length_schedule(step: int, total_steps: int, tokens_per_batch: int,
                    min_len: int = 512, max_len: int = 8192,
                    multiple: int = 128) -> tuple[int, int]:
    """Sequence-length warmup at a constant number of tokens per step.
    Returns (sequence_length, sequences_per_batch)."""
    tau = min(max(step / max(1, total_steps), 0.0), 1.0)
    length = min_len + tau * (max_len - min_len)
    length = int(max(min_len, min(max_len, round(length / multiple) * multiple)))
    return length, max(1, tokens_per_batch // length)


# ---------------------------------------------------------------- adaptive
class AdvantageBandit:
    """SEC-style curriculum for RL fine-tuning: arms are problem categories
    (e.g. difficulty levels); the reward for an arm is the mean absolute
    advantage of its rollouts, a proxy for learning gain. Values are
    updated with TD(0) and arms chosen by softmax (Boltzmann) sampling."""

    def __init__(self, n_arms: int, lr: float = 0.5,
                 temperature: float = 0.2, seed: int = 0):
        self.q = np.zeros(n_arms)
        self.lr, self.temperature = lr, temperature
        self.rng = np.random.default_rng(seed)

    def probs(self) -> np.ndarray:
        z = self.q / self.temperature
        z = np.exp(z - z.max())
        return z / z.sum()

    def select(self, k: int) -> np.ndarray:
        return self.rng.choice(len(self.q), size=k, p=self.probs())

    def update(self, arm: int, advantages: Sequence[float]) -> None:
        reward = float(np.mean(np.abs(advantages)))
        self.q[arm] += self.lr * (reward - self.q[arm])


def zero_advantage_filter(group_rewards: Sequence[Sequence[float]]) -> list[int]:
    """Indices of prompts whose rollout group has non-constant reward.
    In GRPO-style training, a group with identical rewards (all solved or
    all failed) has zero advantage and contributes no gradient."""
    return [i for i, g in enumerate(group_rewards) if np.ptp(g) > 0]
```

**Usage.** A root-paced curriculum used as a warmup over the first 30% of training, with 20% replay, plus the length schedule at 1M tokens per step:

```python
scores = [compression_difficulty(doc) for doc in corpus]
sampler = CurriculumSampler(scores, total_steps=100_000, kind="root",
                            curriculum_fraction=0.3, replay=0.2, seed=0)

for step in range(100_000):
    seq_len, n_seqs = length_schedule(step, 100_000, tokens_per_batch=1 << 20)
    batch_ids = sampler.sample(step, n_seqs)
    # pack corpus[batch_ids] into sequences of seq_len tokens, then train
```

In the test run, the length schedule held tokens per step at about one million while moving from 512 tokens × 2,048 sequences to 8,192 × 128, and the bandit concentrated 66% of its probability on the arm with the largest advantages after 200 updates.

## 13. Open Questions

- **Trajectory or exposure?** Does a curriculum change the learning trajectory, or only the exposure within shared training phases? Current evidence, from models of a few hundred million parameters, favours exposure.
- **Scale.** Do per-example pretraining gains survive beyond about 1.5B parameters once the learning-rate schedule is co-designed?
- **Scaling laws.** Can any curriculum change scaling-law exponents rather than only constants?
- **Transfer of adaptive methods.** Can learning-gain bandits such as SEC move from RL fine-tuning back into pretraining at scale?
- **Optimizers.** How do curricula interact with Muon- or Shampoo-style optimizers, whose effective learning-rate dynamics differ from AdamW?
- **Objectives.** Are scheduled objective curricula, such as denoising early and next-token prediction late, worth their complexity at scale?
- **Stale difficulty.** How can difficulty be re-estimated cheaply as it shifts during training?
- **Multimodality.** What is the right curriculum for decoder-only models trained on interleaved text, image and video tokens?

## Glossary

| Term | Meaning |
| --- | --- |
| Curriculum learning (CL) | Training on a structured, non-random data order |
| Anti-curriculum | Hard-to-easy order |
| Pacing function | Map from training progress to the share of the difficulty-sorted data available |
| Competence | The value of the pacing function: the unlocked fraction of the data |
| Self-paced learning | Curriculum driven by the model's own current loss |
| Mid-training | A second pretraining stage on a curated, high-quality mixture |
| Annealing | A short final learning-rate decay, usually on the highest-quality data |
| WSD | Warmup-stable-decay learning-rate schedule |
| Checkpoint averaging | Averaging the weights of several checkpoints (simple, exponential or weighted); CMA applies it under a curriculum |
| GRPO | Group-relative policy optimization: advantages computed relative to other rollouts for the same prompt |
| RLVR | RL on verifiable rewards, such as math answers and code tests |
| Zero-advantage group | A rollout group with identical rewards, which contributes no gradient |
| SEC | Self-Evolving Curriculum: a bandit over problem categories rewarded by absolute advantage |
| SPaCe | Self-paced RL fine-tuning with clustering and a bandit over clusters (formerly SPaRFT) |
| CAMPUS | Competence-aware, multi-perspective curriculum for instruction tuning |
| CGLS | Curriculum-Guided Layer Scaling: growing model depth with data difficulty |
| FIM | Fill-in-the-middle data transformation for code models |
| MTLD | Measure of Textual Lexical Diversity |
| DoT | Depth of thought: step count in a teacher's chain of thought |
| UED | Unsupervised environment design: generating training environments at the agent's frontier |

## References

Linked entries were checked against the source. Unlinked entries were checked from memory and should be confirmed before external publication.

- Abdin, M. et al. (2024). Phi-3 Technical Report.
- Bavarian, M. et al. (2022). Efficient Training of Language Models to Fill in the Middle.
- Bengio, Y., Louradour, J., Collobert, R. and Weston, J. (2009). Curriculum Learning. ICML.
- Blakeney, C. et al. (2024). Does Your Data Spark Joy? Performance Gains from Domain Upsampling at the End of Training.
- Campos, D. (2021). Curriculum Learning for Language Modeling.
- Chen, S. et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation.
- Chen, X. et al. (2025). [Self-Evolving Curriculum for LLM Reasoning](https://arxiv.org/abs/2505.14970v4).
- Chung, H. W. et al. (2023). UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining. ICLR.
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report.
- Dennis, M. et al. (2020). Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design. NeurIPS.
- Do, D., Nguyen, M., Venkatesh, S. and Le, H. (2026). [SPaCe: Unlocking Sample-Efficient Large Language Models Training With Self-Pace Curriculum Learning](https://arxiv.org/html/2508.05015v1). Findings of ACL. First released as SPaRFT: Self-Paced Reinforcement Fine-Tuning for Large Language Models.
- Elgaar, M. and Rumshisky, A. (2026). [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics](https://arxiv.org/pdf/2601.21698).
- Florensa, C. et al. (2017). Reverse Curriculum Generation for Reinforcement Learning. CoRL.
- Fu, Y. et al. (2024). Data Engineering for Scaling Language Models to 128K Context. ICML.
- Grattafiori, A. et al. (2024). The Llama 3 Herd of Models.
- Graves, A., Bellemare, M. G., Menick, J., Munos, R. and Kavukcuoglu, K. (2017). Automated Curriculum Learning for Neural Networks. ICML.
- Guo, D. et al. (2024). DeepSeek-Coder: When the Large Language Model Meets Programming.
- Hacohen, G. and Weinshall, D. (2019). On the Power of Curriculum Learning in Training Deep Networks. ICML.
- Hägele, A. et al. (2024). Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS.
- Hu, S. et al. (2024). MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.
- Ibrahim, A. et al. (2024). Simple and Scalable Strategies to Continually Pre-train Large Language Models. TMLR.
- Jia, Y., Zhang, C., Diao, X., Yuan, X., Ouyang, Z., Ma, C. and Vosoughi, S. (2026). [What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning](https://aclanthology.org/2026.acl-long.1591/). ACL.
- Jiang, M., Grefenstette, E. and Rocktäschel, T. (2021). Prioritized Level Replay. ICML.
- Jung and Jung (2025). [Reasoning Steps as Curriculum: Using Depth of Thought as a Difficulty Signal for Tuning LLMs](https://arxiv.org/html/2508.18279v1).
- Kim, J. and Lee, J. (2024). [Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning](https://arxiv.org/abs/2405.07490v1).
- Kimi Team (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs.
- Kumar, M. P., Packer, B. and Koller, D. (2010). Self-Paced Learning for Latent Variable Models. NeurIPS.
- Li, C., Zhang, M. and He, Y. (2022). The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models. NeurIPS.
- Li, R. et al. (2023). StarCoder: May the Source Be with You!
- Li, Y. et al. (2025). [Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning](https://arxiv.org/pdf/2509.13790). Findings of EMNLP.
- Luo, K. et al. (2026). [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](https://mlanthology.org/iclr/2026/luo2026iclr-learning/). ICLR.
- Matiisen, T., Oliver, A., Cohen, T. and Schulman, J. (2020). Teacher-Student Curriculum Learning. IEEE TNNLS.
- Nagatsuka, K. et al. (2021). Pre-training a BERT with Curriculum Learning by Increasing Block-Size of Input Text. RANLP.
- Naïr, M., Yamani, K., Lhadj, L. and Baghdadi, R. (2024). [Curriculum Learning for Small Code Language Models](https://aclanthology.org/2024.acl-srw.44). ACL Student Research Workshop.
- Peng, B. et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models.
- Platanios, E. A. et al. (2019). Competence-based Curriculum Learning for Neural Machine Translation. NAACL.
- Portelas, R. et al. (2020). Automatic Curriculum Learning for Deep RL: A Short Survey. IJCAI.
- Pouransari, H. et al. (2024). Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum. NeurIPS.
- Press, O., Smith, N. A. and Lewis, M. (2021). Shortformer: Better Language Modeling Using Shorter Inputs. ACL.
- Qi, Z. et al. (2024). WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning.
- Sanyal, S. et al. (2024). Early Weight Averaging Meets High Learning Rates for LLM Pre-training.
- Schoenegger, L., Thoma, L., Blevins, T. and Roth, B. (2025). [Influence-driven Curriculum Learning for Pre-training on Limited Data](https://aclanthology.org/2025.babylm-main.26). BabyLM Workshop.
- Singh, K., Band, N. and Adeli, E. (2025). [Curriculum-Guided Layer Scaling for Language Model Pretraining](https://arxiv.org/abs/2506.11389v2).
- Team OLMo (2024). 2 OLMo 2 Furious.
- Wang, R. et al. (2019). Paired Open-Ended Trailblazer (POET).
- Wu, X., Dyer, E. and Neyshabur, B. (2021). When Do Curricula Work? ICLR.
- Xie, S. M. et al. (2023). DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. NeurIPS.
- Ye, J. et al. (2024). Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.
- Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
- Zhang, E. et al. (2025). [Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation](https://arxiv.org/abs/2505.08364v1).
- Zhang, Y., Mohamed, A., Abdine, H., Shang, G. and Vazirgiannis, M. (2026). [Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://aclanthology.org/2026.eacl-long.271). EACL.
- Zhao, Y. et al. (2023). A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment.
