# LLM Training Metrics: A Comprehensive Reference

State-of-the-art LLM training is a multi-stage pipeline. Each stage -- pretraining, supervised fine-tuning (SFT), and post-training alignment (RLHF/RLVR) -- exposes a distinct set of metrics that engineers monitor to diagnose health, measure efficiency, and track capability progress.

---

## 1. Pretraining Metrics

### 1.1 Loss Metrics

| Metric | Description | Target |
|---|---|---|
| **Training Loss** | Cross-entropy over next-token predictions on training batches | Decreasing monotonically |
| **Validation Loss** | Cross-entropy on held-out data (test split) | Should track training loss; divergence signals overfitting |
| **Per-domain Loss** | Loss broken out by data source (code, web, books, math, etc.) | Balanced or intentionally weighted |
| **Bits per Character (BPC)** | Loss expressed in bits; language-model equivalent of BPC = loss / ln(2) | Lower = better |

The primary training objective is **average negative log-likelihood per token** (nats/token). Under Chinchilla and Kaplan scaling laws, this loss is the canonical measure of pretraining progress against compute budget.

Loss spikes are a critical failure signal. Sudden jumps of 10x or more above baseline, often correlated with gradient norm explosions, indicate instability requiring checkpoint rollback or learning rate intervention.

### 1.2 Perplexity

Perplexity = exp(loss). It quantifies how "surprised" the model is by the test set. Directly interpretable: a perplexity of 10 means the model assigns roughly equal probability to 10 tokens at every step. Reported on domain-specific test sets (e.g., Penn Treebank, Pile, C4 subsets) to track capability across text types.

### 1.3 Gradient Metrics

| Metric | Description | Warning Signs |
|---|---|---|
| **Gradient Norm (L2)** | Euclidean norm of all parameter gradients concatenated | Sudden spikes → exploding gradients; near-zero → vanishing gradients |
| **Per-layer Gradient Norm** | Norm per transformer block | Reveals which layers are over/under-learning |
| **Gradient Variance** | Rolling variance of gradient norms over a time window | High variance → unstable optimization landscape |
| **Gradient Clipping Frequency** | % of steps where clipping activates (clip_grad_norm threshold hit) | High clipping rate → learning rate too large or batch too noisy |

Gradient clipping (typically at max norm = 1.0) is near-universal in SOTA pretraining. The norm is logged before clipping to preserve the diagnostic signal.

### 1.4 Optimizer State Metrics

| Metric | Description |
|---|---|
| **Learning Rate (LR)** | Current LR from schedule (warmup → cosine decay or WSD) |
| **LR Warmup Progress** | Steps elapsed in linear warmup phase |
| **Weight Norm** | L2 norm of parameter tensors per layer; large growth can signal instability |
| **Adam Second Moment (v)** | Moving average of squared gradients; reveals which parameters are receiving the most signal |
| **Weight Update Ratio** | ||Δw|| / ||w|| per layer; ratio that is too large = destructive updates |

### 1.5 Tokens and Data Pipeline Metrics

| Metric | Description |
|---|---|
| **Tokens Seen / Total Tokens** | Cumulative tokens consumed; primary "x-axis" of training |
| **Tokens per Second (TPS)** | Training throughput in tokens/sec |
| **Global Batch Size** | Total tokens per gradient step across all GPUs |
| **Data Source Distribution** | Fraction of each data domain per batch (web/code/books/math/multilingual) |
| **Data Deduplication Rate** | % of training examples filtered by exact/near-dedup |
| **Repeat Ratio** | How many times each epoch the data has been seen; repeated data degrades performance measurably |

---

## 2. Hardware and Infrastructure Efficiency Metrics

These metrics govern how efficiently the cluster executes math, distinct from how well the model learns.

### 2.1 Compute Utilization

| Metric | Formula | Typical Range |
|---|---|---|
| **Model FLOPs Utilization (MFU)** | (Model FLOPs × observed TPS) / (Peak hardware FLOPs) | 30–50% on A100/H100; 38–43% reported for LLaMA 3.1 |
| **Model Bandwidth Utilization (MBU)** | (Weight bytes transferred × TPS) / Peak memory bandwidth | Used in memory-bound (decode) regime |
| **Hardware FLOP/s** | Sustained floating point ops per second across the cluster | Compared against theoretical peak (e.g., 312 TFLOPs BF16 per A100) |
| **GPU Utilization (%)** | % time GPU compute cores are active | Low utilization = communication or I/O bound |
| **GPU Memory Usage** | Bytes allocated vs. total HBM | OOM threshold monitoring |
| **GPU Memory Bandwidth Utilization** | Fraction of peak HBM bandwidth consumed | |

MFU is the principal compute-efficiency KPI. It is hardware-agnostic and directly comparable across GPU generations and cluster sizes. Google introduced MFU in the PaLM paper; values approaching 50%+ require Flash Attention and high-quality parallelism tuning.

### 2.2 Distributed Training Efficiency

| Metric | Description |
|---|---|
| **Goodput** | Fraction of wall-clock time spent doing productive forward/backward computation (excluding comms, checkpointing, restarts) |
| **Badput** | Complement of goodput; time lost to failures, stragglers, or idle bubbles |
| **Pipeline Bubble Fraction** | Idle time in pipeline-parallel schedules (e.g., GPipe, 1F1B); lower = better |
| **All-Reduce Time** | Time spent synchronizing gradients across data-parallel replicas |
| **Communication/Compute Overlap** | % of comms that overlap with compute; 100% = no stall |
| **Checkpoint Save/Restore Time** | Wall-clock overhead for saving and loading model state |
| **Step Time** | Wall-clock time per gradient update step |
| **Infra Goodput** | Fraction of potential training capacity not lost to hardware failures |

### 2.3 Memory Metrics

| Metric | Description |
|---|---|
| **Activation Memory** | Memory consumed by stored activations for backward pass |
| **Optimizer State Memory** | Adam keeps two FP32 moment tensors per parameter (~12 bytes/param); dominant in large models |
| **KV Cache Size (during training evals)** | Memory used for attention key/value store during long-context passes |
| **Peak Memory Per GPU** | Maximum HBM allocation; determines max batch size and sequence length |

---

## 3. Supervised Fine-Tuning (SFT) Metrics

SFT uses the same cross-entropy objective as pretraining but on high-quality instruction/demonstration data.

| Metric | Description |
|---|---|
| **SFT Training Loss** | Cross-entropy on instruction-following pairs |
| **SFT Validation Loss** | Held-out demonstration loss; overfitting visible quickly given small dataset sizes |
| **Per-category Loss** | Loss broken out by task type (code, summarization, Q&A, etc.) |
| **Response Length Distribution** | Mean/std of output token counts; collapse (all short) or explosion (runaway) signals problems |
| **Perplexity on SFT Prompts** | How confidently the model follows instruction format |
| **Exact Match / ROUGE / BLEU** | Token-overlap proxies for generation quality on structured tasks |

Overfitting risk is high in SFT because datasets are typically 1–100K examples versus billions of pretraining tokens. Early stopping on validation loss is standard.

---

## 4. RLHF / PPO Post-Training Metrics

RLHF uses Proximal Policy Optimization to align the SFT model to human preferences via a reward model. Metrics here are qualitatively different from pretraining.

### 4.1 Reward and Objective Metrics

| Metric | Description | Notes |
|---|---|---|
| **RLHF Reward (score)** | Mean score from reward model across batch | Primary optimization target; should increase over training |
| **KL Divergence (Policy vs. Reference)** | KL(π_θ ∥ π_SFT); how far the trained policy has drifted from the SFT baseline | If too high → reward hacking, incoherence; target range is task-dependent |
| **Non-Score Reward** | β × KL penalty component subtracted from raw score | Encodes the "price" of diverging from the SFT model |
| **Net RLHF Objective** | score − β·KL; the actual optimization target | Should increase while KL stays bounded |
| **Value Loss** | MSE between critic's predicted return and actual discounted returns | Tracks quality of the value function; a diverging critic destabilizes PPO |
| **Policy Loss** | PPO clipped surrogate loss | Should decrease; very large values = policy collapsing or exploding |
| **Entropy** | Mean entropy of the policy distribution | Monitors exploration; collapse to near-zero entropy → degenerate policy |

### 4.2 PPO-Specific Stability Metrics

| Metric | Description | Warning Signs |
|---|---|---|
| **Clip Fraction (Policy)** | % of policy updates clipped by PPO's ε-clipping | >30% frequently = learning rate too high or ε too small |
| **Approx KL (consecutive policies)** | KL between consecutive mini-batch policy updates within a PPO epoch | Very large = update too aggressive |
| **Advantage Estimates** | Normalized advantages (GAE); variance indicates signal quality | Collapsing variance = reward model saturated |
| **Value Clip Fraction** | % of value function updates clipped | |
| **EOS Token Count** | Number of completions that hit end-of-sequence naturally | Sharp drops → length collapse |
| **Response Length** | Mean output length per step | Sudden shortening is a reward hacking signal (model learns short answers game the reward model) |

### 4.3 Reward Hacking Signals

Reward hacking is one of the most pernicious failure modes in RLHF. Indicators include:

- Reward increasing while human eval / win rate stays flat or drops
- KL divergence exploding (model finding high-reward out-of-distribution outputs)
- Response length collapsing (gaming brevity-biased reward models)
- Repetitive or gibberish outputs that receive high proxy rewards

---

## 5. RLVR / GRPO Metrics (Reasoning Model Training)

Reinforcement Learning with Verifiable Rewards (RLVR), used in DeepSeek-R1, o1/o3, and successor reasoning models, replaces the learned reward model with a deterministic verifier. GRPO (Group Relative Policy Optimization) eliminates the critic/value model entirely.

| Metric | Description |
|---|---|
| **Verifiable Accuracy** | Binary correct/incorrect rate on math, code, or logic tasks as determined by symbolic verifiers |
| **Pass@k** | Probability at least one of k sampled responses is correct; measures exploration breadth |
| **KL Divergence (GRPO)** | KL term added directly to policy loss; regularizes group-relative updates |
| **Group Relative Reward** | Reward for a response normalized by the mean reward across a group of responses to the same prompt |
| **Reasoning Trace Length** | Mean number of tokens in chain-of-thought traces; increases as model learns to "think longer" |
| **Thinking Token Budget Utilization** | For test-time scaling models: how much of the allocated thinking budget is used |
| **Format Compliance Rate** | % of outputs that follow expected XML/JSON/chain-of-thought format (critical for rule-based reward parsing) |
| **Reward Variance Within Group** | GRPO's signal quality; high variance = model has diverse strategies; near-zero = all responses equally good/bad |
| **Process Reward Signal (if PRM used)** | Step-level rewards from a process reward model rather than only outcome rewards |

Karpathy notes that RLVR "allowed for much longer optimization" versus SFT/RLHF and created a new "test-time compute" scaling law: more thinking tokens at inference = better performance.

---

## 6. Evaluation / Capability Metrics (Tracked During and After Training)

These are not per-step logged metrics but are evaluated at checkpoints to track capability trajectory.

### 6.1 General Capability Benchmarks (as of 2025–2026)

| Benchmark | What It Measures |
|---|---|
| **MMLU-Pro** | Multi-task language understanding (10-choice, harder than original MMLU which is saturated) |
| **GPQA Diamond** | Graduate-level expert science Q&A; highly resistant to lookup |
| **HLE (Humanity's Last Exam)** | 2,500 expert-curated questions across all academic domains; frontier-model difficulty |
| **ARC-AGI** | Abstract visual reasoning; tests out-of-distribution generalization |
| **FrontierMath** | Research-grade math problems; frontier models were below 2% at release |
| **AIME / AMC** | Competition mathematics; multi-step numerical reasoning |
| **LiveCodeBench** | Code problems released after model training cutoff; prevents memorization contamination |
| **SWE-Bench Verified** | Real GitHub issues requiring autonomous resolution; gold standard for agentic coding |

### 6.2 Domain-Specific Evals

| Domain | Common Metrics |
|---|---|
| **Code** | Pass@1, Pass@10 on HumanEval, MBPP, SWE-Bench; functional correctness via unit tests |
| **Math** | Exact answer match on GSM8K, MATH, AIME; symbolic equivalence checking |
| **Factuality** | SimpleQA, FACTS Grounding; hallucination rate on verifiable claims |
| **Instruction Following** | IFEval; % of constraints satisfied in structured prompts |
| **Long Context** | RULER, Needle-in-a-Haystack; retrieval accuracy at various context depths |
| **Safety / Alignment** | Refusal rates, jailbreak resistance, TruthfulQA, BBQ bias |
| **Multilingual** | Cross-lingual transfer accuracy; per-language perplexity |

### 6.3 Human Preference Metrics

| Metric | Description |
|---|---|
| **Win Rate** | % of head-to-head comparisons where annotators prefer the model's output over a reference |
| **ELO / Arena Score** | Comparative ranking from LM Arena / Chatbot Arena (crowd-sourced battles) |
| **GPT-4-as-Judge Score** | Automated preference proxy using a strong model as evaluator |
| **MTBench Score** | Multi-turn instruction following scored 1–10 by GPT-4 |

---

## 7. Scaling Law Metrics

Scaling law research tracks how metrics change as resources scale, enabling budget decisions before committing to full runs.

| Metric | Role in Scaling Laws |
|---|---|
| **Pretraining Loss vs. Compute (C)** | Chinchilla / Kaplan law: L ∝ C^-α; fit to small runs, extrapolated to target |
| **Tokens Seen (D) vs. Model Parameters (N)** | Optimal ratio: N_opt ∝ C^0.5, D_opt ∝ C^0.5 (Hoffmann et al. 2022) |
| **Loss at Convergence vs. Data Repeats** | Repeated data degrades performance; scaling laws quantify the penalty |
| **Downstream Task Accuracy vs. Pretraining Loss** | For predictable tasks, pretraining loss is a reliable proxy; used to select checkpoints |
| **FLOPs per Token** | Typically ~6N for dense transformers (forward + backward); key budget metric |
| **Effective Parameter Count** | Adjusted for numerical precision; lower precision = lower effective capacity |

---

## 8. Summary Table: Metric by Training Stage

| Stage | Primary Metrics | Critical Alarms |
|---|---|---|
| **Pretraining** | Training/val loss, gradient norm, LR, MFU, TPS, tokens seen | Loss spike, grad norm explosion, NaN loss, MFU drop |
| **Mid-Training** (long context, domain tune) | Per-domain loss, long-context perplexity, loss on held-out domain test sets | Catastrophic forgetting on general domains |
| **SFT** | SFT loss, val loss, response length distribution, format compliance | Overfitting (val loss diverges), length collapse |
| **RLHF / PPO** | RLHF reward, KL divergence, clip fraction, entropy, value loss | KL explosion, reward hacking, entropy collapse |
| **RLVR / GRPO** | Verifiable accuracy, pass@k, trace length, format compliance | Reward hacking via format gaming, zero variance groups |
| **All stages** | GPU utilization, goodput, memory usage, checkpoint health | OOM, stale checkpoints, infra failure spikes |

---

## 9. Tooling Context

Modern labs log the above to experiment tracking platforms (Weights & Biases, TensorBoard, MLflow) at frequencies ranging from every step (loss, grad norm) to every few hundred steps (benchmark evals). Infrastructure metrics are typically surfaced via Prometheus + Grafana with DCGM for GPU telemetry. Training pipelines implement automated alerts and checkpoint rollback for loss spikes and gradient explosions.
