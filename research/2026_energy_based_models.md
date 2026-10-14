# Energy-Based Models Family: Integration Research (2025–2026)

**Scope:** How to bring the 2025–2026 energy-based reasoning family into `dl_techniques` as first-class Keras 3 citizens.
**Coverage:** EBT, Equilibrium Reasoners (EqR), Energy-guided Recursive Model (ERM), EBRM latent-trajectory planning, Compositional Energy Minimization + PEM, MERITED metacognition, NRGPT, plus foundations (LeCun 2006, Du 2022/2024, Hoover ET 2023).
**Non-goal:** No implementation in this document. No `.py` changes. This is the design input for the build phase.
**House docs this builds on:** `research/2026_keras_custom_models_instructions_v2.md` (mandatory authoring guide), `research/2026_jepa.md` (§1.3 frames JEPA as EBM), `src/dl_techniques/models/AGENTS.md`, `src/dl_techniques/layers/AGENTS.md`, `src/train/AGENTS.md`, `REPO_MAP.md`.
**Locked decisions (2026-10-14):** family `models/general_purpose/ebm/` (subfamily container, sam-style nesting) · first port EBM-LM (EBT for language) · single trainer `train/ebm/` with per-member scripts.

---

## Table of Contents

1. [Why this family, why now](#1-why-this-family-why-now)
2. [EBM primer distilled for this repo](#2-ebm-primer-distilled-for-this-repo) + [Math and operations required to implement](#21-math-and-operations-required-to-implement-normative)
3. [Paper-by-paper digest](#3-paper-by-paper-digest)
4. [What the repo already owns](#4-what-the-repo-already-owns)
5. [Gaps: what does not exist yet](#5-gaps-what-does-not-exist-yet)
6. [Where the family should live](#6-where-the-family-should-live)
7. [Keras 3 / TF porting notes](#7-keras-3--tf-porting-notes)
8. [Training and inference design](#8-training-and-inference-design)
9. [Data and benchmarks](#9-data-and-benchmarks)
10. [Test and guard plan](#10-test-and-guard-plan)
11. [Phased roadmap](#11-phased-roadmap)
12. [Open decisions](#12-open-decisions)
13. [References](#13-references)

---

## 1. Why this family, why now

Three trends converge:

1. **Test-time scaling without verifiers.** EBT, EqR, ERM all scale compute at inference by running more optimizer/fixed-point steps, not by sampling more tokens and reranking with an external reward model. This matches the repo's existing ACT line (HRM/TRM/PTRM) but replaces the Q-halting heuristic with a scalar energy that is itself learned.
2. **One mechanism across modalities.** EBT shows the same `E(x, y_hat)` + gradient-descent predictor working for text and image denoising. The repo is modality-split (`models/vision`, `models/language`, `models/graph`); an energy head + sampler that is modality-agnostic would be the first shared reasoning substrate.
3. **Composability.** Compositional Energy Minimization and JEPA §1.3 both treat energies as composable (sum landscapes at test time). No other reasoning head in the repo composes this way. Q-heads do not add.

Practical payoff if ported well: Sudoku / Maze / PPBench solvers that reuse the tiny TRM/HRM trunks the repo already trains, plus a language EBT that reuses `TextDecoder` / `TransformerLayer` instead of forking a new stack.

---

## 2. EBM primer distilled for this repo

An EBM assigns a scalar `E(x, y)` — low for compatible pairs, high otherwise — and predicts by minimizing it:

- `y* = argmin_y E(x, y)` via gradient descent, Langevin dynamics, or fixed-point iteration.
- Training shapes the landscape: push down energy on positives, push up on negatives (contrastive divergence, NCE, score matching, or supervised energy regression + contrastive shaping).
- Sampling uses the same landscape: Langevin `y <- y - alpha*dE/dy + sqrt(2*alpha*T)*eps`, parallel tempering across temperatures, best-of-N + energy rerank.

Three energy shapes in the 2026 line:

| Shape | Form | Exemplar |
|---|---|---|
| State energy | `E(g) = E_ATT(g) + E_HN(g)` over token states, descent in representation space | Hoover ET 2023, repo `EnergyTransformer` |
| Verifier energy | `E(x, y_hat)` over input + candidate prediction, descent in output / embedding space | EBT |
| Trajectory energy | `E(h_x, z_1:T) = sum compat + sum transition + smoothness` over a latent rollout | EBRM |

Two scaling axes (EqR language): **depth** (more solver steps) and **width** (more stochastic trajectories from different inits, aggregated). Both must be first-class inference knobs, not training constants.

Failure mode to design against (EBRM §4): latent drift — decoder trained on encoder outputs `h_x` but evaluated on planner outputs `z_T` that live elsewhere. Fix is dual-path decoder training + anchoring, not more planner steps.

### 2.1 Math and operations required to implement (normative)

This subsection is the implementability checklist. A build that cannot tick every row is not ready. Notation: `B` batch, `N` tokens/positions, `D` dim, `K` memories, `T` thinking steps, `x` input, `y` candidate prediction, `z_1:T` latent trajectory, `h_x` encoder output, `theta` weights, `alpha` step size, `eps ~ N(0,I)`.

**A. Gibbs form + MLE gradient (why contrastive samples are unavoidable).**

- `p_theta(y|x) = exp(-E_theta(x,y)) / Z_theta(x)`, `Z_theta(x) = sum_y exp(-E_theta(x,y))` (intractable for text/vision).
- MLE gradient: `d/dtheta log p = -dE(x,y_pos)/dtheta + E_{y~p}[dE(x,y)/dtheta]`. The second term needs samples from the model itself — hence Langevin / tempered sampling at training time, not just inference. Any design that trains the energy head with CE only and no negatives is not shaping the landscape, only shifting it.

**B. State energy (Hoover ET — already in repo, reuse exactly).**

- Per-step: `g = EnergyLayerNorm(x)`, `update = attn.update(g) + hopfield.update(g)` where each `update == -dE/dg` in closed form (no autodiff in this block), `x <- x + alpha * update`, repeat `T` steps reusing one block's weights.
- Attention term: token-mixing energy `E_ATT(g)` whose negative gradient mixes tokens along edges (graph: restricted to adjacency via rank-3 mask; vision: dense). `EnergyAttention` has no value matrix by construction.
- Hopfield term per token: `h_{nk} = sum_d xi_{kd} g_{nd}`, `E_HN = -sum_n sum_k G(h_{nk})` with matched pair `G' = r`:
  - `relu`: `r(h) = relu(h)`, `E = -0.5 * sum relu(h)^2`.
  - `softmax`: `r(h) = softmax_k(beta*h)`, `E = -(1/beta) * sum_n logsumexp_k(beta*h)`, softmax over memory axis `k`, never token axis `n`.
- Pairing one activation's energy with the other's gradient factor is a silent break (runs, trains, stops descending). Pin by asserting `update == -grad(energy)` numerically per term.
- Output contract: tokens `(B, N, D)` + optional trace `(B, T+1)` float32 (never fp16 — trace magnitude overflows fp16; ET D-011).

**C. Verifier energy + thinking loop (EBT / EBM-LM core).**

- Head: scalar `E_theta(x, y_hat)` from a `Dense(1)` over joint features (backbone hidden + candidate embedding). One scalar per sample `(B,)`, float32.
- Inner loop (differentiable): `y_0 = init(x)` (prior / amortized guess / noise), `y_{t+1} = y_t - alpha * dE/dy_t` for `t = 0..N-1`, predict `y_N`. For discrete tokens, descend in embedding/logit space and project (`argmax` / sample) only after convergence — never differentiate through the projection.
- Through-optimization training: outer tape over `theta`, inner tape(s) over `y_t`. Loss on `y_N` (task loss) + energy shaping (row D). Backprop through the unrolled chain needs Hessian-vector products `d/dtheta (dE/dy)`; nested `tf.GradientTape` yields HVPs without materializing Hessians, cost linear in model size and linear in N memory. Truncate (stop-gradient on all but last K inner steps) only as a documented ablation — it biases the meta-gradient.
- Second-order + XLA + fp16 is the riskiest intersection: keep energies/scores in float32, cast trunk I/O at block boundaries (ET backbone precedent).

**D. Energy-shaping losses (all contrastive; CE alone is insufficient).**

- Margin: `L = max(0, m + E(x,y_pos) - E(x,y_neg))`, per-sample `(B,)`.
- NCE / InfoNCE over K negatives: `L = E(x,y_pos) + logsumexp over (negatives plus positive)` up to constants; hard negatives required (random negatives give a flat landscape that still descends on positives).
- Supervised + contrastive composite (EBT/EBRM recipe): `L = L_task(y_N, y_true) + lambda_e * L_energy(E_pos, E_negs)`. Log the two terms split (Pattern 8), never as one number.
- Regularizers that belong beside the loss: L2 on energies (prevents runaway scale that breaks composition), entropy/confidence penalties only if batch-level semantics are asserted explicitly (see `losses/AGENTS.md` premature-scalar rule).

**E. Langevin + tempering samplers (training negatives AND inference width).**

- Unadjusted Langevin (ULA) per step: `y <- y - alpha*dE/dy + sqrt(2*alpha*T) * eps`, `eps ~ N(0,I)`, temperature `T` (`T=1` canonical; repo `SGLD` exposes it as `noise_scale^2`). `alpha -> 0`, steps `-> inf` samples `p(y|x)`; finite runs are approximate — document steps and schedule.
- SGLD-as-optimizer identity (repo `sgld_optimizer.py`): `w <- w - lr*g + s*sqrt(2*lr)*eps`. Zero LR must give zero noise (diagnostic arm); `s=0` must equal SGD bit-for-bit (ablation arm).
- Parallel tempering (ERM): ladders `T_1 < ... < T_M`, run chains in parallel, swap states `i,j` with Metropolis probability `min(1, exp((E_i - E_j)(1/T_i - 1/T_j)))`. High-T chains explore, low-T chains exploit; swaps move good basins down. Assert swap acceptance in `(0,1)` strictly — 0 means ladders too far apart, 1 means redundant.
- Inference aggregators: min-energy select, tempered rerank, or mean/majority vote over K trajectories. Score all with `pass_at_k` (row H), not mean accuracy.

**F. Fixed-point / attractor iteration (EqR).**

- Dynamics `z_{t+1} = f_theta(z_t, x)` (Transformer block(s) reused, or ET-style descent step). Fixed point `z* = f(z*, x)`.
- Depth scaling: iterate to `||z_{t+1} - z_t|| < tol` or max steps; report steps-to-convergence histogram (1–5 easy, long tail hard). Never fix N globally — it defeats adaptive compute.
- Convergence-gated exit replaces Q-halting: halt on energy delta / residual norm, not `q_halt > q_continue`. Keep the HRM/TRM `ops.where`-freeze + static-trip-count structure for `tf.function`.
- Breadth scaling: K random inits (or noise injections à la PTRM `z_L`), aggregate. Width and depth are orthogonal knobs; sweep the grid, not one axis.

**G. Hopfield-guided recursion (ERM).**

- Memories of valid local/global structures: tied matrix `xi (K, D)` (same form as row B). Trajectory score `E_traj = E_local(z_1:T) + E_global(z_T)` with the same `G/r` matched pairs.
- Selection: keep lowest-energy of K rollouts; depth range: the N minimizing energy on validation (deeper is NOT monotonic — assert the U-curve exists before claiming a depth).
- Language side: same scorer as a perplexity reranker (`-1.74%` claim is rerank/select gain, not a new LM head).

**H. Structured trajectory energy (EBRM — highest ceremony).**

- `E(h_x, z_1:T) = sum_t C(h_x, z_t) + sum_t T(z_t, z_{t+1}) + S(z_1:T)`:
  - `C`: per-step compatibility (e.g. `||proj(z_t) - h_x||^2` or learned scorer),
  - `T`: transition consistency (`||z_{t+1} - g(z_t)||^2` for learned step `g`),
  - `S`: smoothness (`sum_t ||z_{t+1} - z_t||^2`, weight `lambda_s`).
- Inference: GD or Langevin over `z` (row E), decode answer from `z_T` only.
- Mandatory companions (the paper's negative result): latent anchor `lambda_a * ||z_T - h_x||^2` (or per-step variant) + dual-path decoder training (decoder sees both `h_x` and planner `z_T` during training). Without both, expect the 95→56 CNF collapse. Ablate anchor weight over at least an order of magnitude; assert `||z_T - h_x||` moves monotonically in it.
- Diagnostics: per-step decode accuracy, `||z_t - h_x||` trace, gradient-norm decomposition per term (C vs T vs S) — log all three, not energy alone.

**I. Composition (PEM).**

- Global energy is the SUM of subproblem energies: `E_global(y) = sum_i E_i(x_i, y_{|i})`. Inference minimizes the sum jointly (Parallel Energy Minimization: parallel per-subproblem descents + periodic consensus). Heads must share scale (no per-head LayerNorm-on-energy that breaks additivity); document the summation contract in each head docstring.

**J. Metacognition (MERITED, last).**

- IBL allocator: instances `(uncertainty u, compute c, correct?)` with activation `A_i = log(freq) - decay * recency + noise`, retrieval probability `softmax(A/T)`, blended value `sum p_i * outcome_i`. Pre-response inputs are `E(x, y_0)` and initial slope `E_0 - E_1` (cheap, no full rollout). Do not build before rows C–E are calibrated — an allocator over uncalibrated energies allocates randomly.

**K. Evaluation math that must be reused.**

- `pass@k` unbiased (Chen et al.): `1 - C(n-c_i, k)/C(n, k)` per problem, mean over problems; `>0` marks solved; plug-in (`any of n correct`) is an upper bound for `k<n` — report one, not both (`metrics/pass_at_k.py` owns both; do not reimplement).
- Calibration: bin energies, plot `P(correct | E in bin)`; assert monotone decreasing (higher energy → less often correct) before any IBL/threshold use.
- Cost accounting per run: forward passes, wall time, (N, K, temperatures) — the family's claim is accuracy-per-compute, never accuracy alone.

---

## 3. Paper-by-paper digest

### 3.1 EBT — Energy-Based Transformers are Scalable Learners and Thinkers (2507.02092, ICLR 2026 Oral)

- **Claim:** Train `E(x, y_hat)` explicitly; predict by gradient descent until convergence. Works for discrete text and continuous vision with one mechanism. No verifier, no RL reward, unsupervised pretraining suffices.
- **Numbers to quote:** up to 35% higher scaling rate vs Transformer++ on data/batch/params/FLOPs/depth; +29% System-2 gain on language; beats Diffusion Transformers on denoising with fewer forward passes; better downstream at same-or-worse pretraining loss (generalization claim).
- **Mechanism:** Outer loop over weights, inner loop `y <- y - alpha*dE/dy` for N steps, differentiable through the inner loop (second-order terms via Hessian-vector products, linear in model size). Inference-time knob N.
- **Port takeaway:** This is a `train_step` override + scalar head design, not a new attention type. Keep the backbone (`TransformerLayer` / `TextDecoder`) unchanged; add `Dense(1)` energy head. Inner graph must stay differentiable.
- **Sources:** paper `https://arxiv.org/abs/2507.02092`, code `https://github.com/alexiglad/EBT`, site `https://energy-based-transformers.github.io`.

### 3.2 EqR — Equilibrium Reasoners: Learning Attractors (2605.21488, ICML 2026)

- **Claim:** Generalizable reasoning = learning task-conditioned attractors. Fixed points of latent dynamics correspond to valid solutions. Test-time gains correlate with convergence strength.
- **Numbers:** Sudoku-Extreme 2.6% feedforward → 99%+ with up to 40k-layer-equivalent unrolling. Easy cases converge in 1–5 steps; hard cases use massive scaling. Adaptive compute falls out of convergence, not a learned halt head.
- **Mechanism:** Depth (more iterations) + breadth (aggregate stochastic trajectories from multiple random inits). No external verifier, no task priors.
- **Port takeaway:** Closest to repo's HRM/TRM carry loops, but halting signal is energy/convergence delta, not `q_halt > q_continue`. Reuse carry-freeze pattern (`ops.where`, static trip count for `tf.function`), replace Q-head with convergence monitor. Breadth = PTRM-style parallel rollouts + aggregation.
- **Sources:** `https://arxiv.org/abs/2605.21488`.

### 3.3 ERM — Energy-guided Recursive Model (2607.10128)

- **Claim:** Recursive models lack a principled trajectory-selection and depth-selection criterion. ERM adds Hopfield-type memories of valid local/global structures that score candidate trajectories intrinsically.
- **Numbers:** Sudoku 98.97%, PPBench 88.04%, Maze 99.30% (best recursive at time of writing); RedPajama-V2 perplexity −1.74% with marginal overhead. Beats PTRM and EqR on those three puzzle benches. Deeper ≠ better; energy suggests effective depth range.
- **Mechanism:** Explicit Hopfield energies guide candidate selection + enable parallel tempering for exploration. Language side is a perplexity win, not a new tokenizer.
- **Port takeaway:** Smallest delta from current repo: keep TRM/PTRM/HRM trunk exactly, add an energy scorer over `(z_H, z_L)` trajectories + tempered sampler. This is the recommended first build (see §11).
- **Sources:** `https://arxiv.org/abs/2607.10128`.

### 3.4 EBRM — Reasoning as Energy Minimization over Structured Latent Trajectories (2603.28248)

- **Claim:** Reasoning = gradient-based optimization of `z_1:T` under `E(h_x, z)` decomposing into per-step compatibility + pairwise transition consistency + trajectory smoothness. Train supervised encoder-decoder + contrastive energy shaping with hard negatives; infer by GD/Langevin over `z`, decode from `z_T`.
- **Negative result (load-bearing):** On CNF logic, latent planning drops accuracy ~95% → ~56%. Cause: distribution mismatch (decoder trained on `h_x`, tested on drifted `z_T`). Diagnosed via per-step decoding, drift tracking, gradient decomposition.
- **Fixes proposed:** Dual-path decoder training (train decoder on both encoder outputs and planner outputs) + latent anchoring (penalty keeping `z` near `h_x` manifold).
- **Ablation protocol (reuse):** six parts — component contributions, trajectory length, planner dynamics, initialization, decoder training distribution, anchor weight.
- **Port takeaway:** Do not port planner without dual-path decoder + anchor. The repo's `CausalLanguageModel` / `MaskedLanguageModel` wrappers assume decoder trains on backbone outputs only; EBRM needs a second training path.
- **Sources:** `https://arxiv.org/abs/2603.28248`, code `https://github.com/Shr1ftyy/ebrm`.

### 3.5 Compositional Energy Minimization (NeurIPS 2025)

- **Claim:** Learn landscapes over subproblems, sum them at test time into a global landscape. Parallel Energy Minimization (PEM) as the solver.
- **Port takeaway:** Only composable reasoning head in the survey. Design energy heads so energies add (shared scale, no per-head normalization that breaks summation). No repo equivalent; new capability.
- **Sources:** site `https://alexoarga.github.io/compositional_reasoning/`.

### 3.6 MERITED — Metacognitive Reasoning with IBL + Energy Dynamics (2610.00399)

- **Claim:** EBMs give interpretable uncertainty + dynamic compute, but do not decide *how much* compute to allocate. MERITED adds an Instance-Based Learning model over past (uncertainty → cost/accuracy) instances to set compute pre-response.
- **Artifacts:** 191M reasoning EBM trained + open weights; IBL allocator implementation.
- **Port takeaway:** Phase-4 work. Allocator is a callback / inference wrapper, not a model change. Depends on calibrated energy→uncertainty mapping first. Do not build before EBT/ERM energies are stable.
- **Sources:** `https://arxiv.org/abs/2610.00399`.

### 3.7 Related 2026 line (cite, do not port yet)

- Attractor models for language and reasoning (2605.12466), Flow reasoning models (2606.29150), Fractal basins trap latent reasoning (2609.04963), Fixed-Point Reasoners (2606.18206), NRGPT (2512.16762, ICLR 2026 — minimal GPT→EBM edit, inference as landscape exploration, resists overfitting on Shakespeare/ListOPS/OpenWebText).

### 3.8 Foundations (must cite in new modules)

- LeCun et al. 2006 Tutorial on Energy-Based Learning; Du et al. ICML 2022 iterative reasoning via energy minimization + 2024 energy diffusion; Hoover et al. NeurIPS 2023 Energy Transformer; Ramsauer 2020 Modern Hopfield; Krotov-Hopfield 2016 dense associative memory.
- Overviews: Shekhar *EBMs: From Basics to LLMs* (`https://www.shashankshekhar.com/blog/ebm-talk`), Logical Intelligence *EBMs for Reasoning* (`https://logicalintelligence.com/blog/energy-based-models-for-reasoning`), Yilun Du lab page (`https://yilundu.github.io/`).
- Keras/TF templates: `https://github.com/raoulnormand/mnist_distribution` (Langevin + contrastive loop), Foster *Generative Deep Learning* 2nd ed. EBM chapter.

---

## 4. What the repo already owns

This section is the reuse inventory. Every entry was read at HEAD; paths are import paths.

### 4.1 Energy descent blocks (reuse directly, do not fork)

- `dl_techniques.layers.transformers.energy_transformer.EnergyTransformer` + `HopfieldNetwork` (`src/dl_techniques/layers/transformers/energy_transformer.py`): `T` steps of `x <- x + alpha*update` where `update == -dE/dg`, closed-form gradients in `keras.ops`, no autodiff. `HopfieldNetwork` is intentionally NOT in the FFN factory (D-002 anchor) — do not register it there.
- `dl_techniques.layers.attention.energy_attention.EnergyAttention` (`src/dl_techniques/layers/attention/energy_attention.py`): no value matrix; `call` returns closed-form negative gradient of token-mixing energy. Factory-registered as `energy` among 35 attention types (`src/dl_techniques/layers/attention/factory.py`, `ATTENTION_REGISTRY`, `create_attention_layer`).
- `dl_techniques.layers.norms.energy_layer_norm.EnergyLayerNorm` (`src/dl_techniques/layers/norms/`): scalar gamma + vector delta; output is `dL/dx` of a Lagrangian with PSD Hessian — the descent guarantee. Factory reachable via `create_normalization_layer` (`src/dl_techniques/layers/norms/factory.py`).
- `dl_techniques.layers.transformers` package (`src/dl_techniques/layers/transformers/__init__.py`, `__all__` curated): `TransformerLayer`, `TransformerDecoderLayer`, `TextDecoder`, `TextEncoder`, `VisionEncoder`, `PatchCausalTransformer`, plus `create_transformer_block` / `TRANSFORMER_BLOCK_REGISTRY` (`factory.py`). EBT language trunk should compose these, not mint new blocks.
- Precedent that a bespoke block composes factories rather than reimplementing them: `models/language/topolm/` `TopoLMBlock` (uses `create_attention_layer` / `create_ffn_layer` / `create_normalization_layer`), documented in `REPO_MAP.md` § model/trainer/test triangle and `src/dl_techniques/models/language/AGENTS.md`.

### 4.2 Model packages that are already energy / recursive (extend, do not duplicate)

- `dl_techniques.models.vision.energy_transformer` (`src/dl_techniques/models/vision/energy_transformer/model.py`): `EnergyTransformerBackbone` + `EnergyTransformerMIM` + `EnergyTransformerClassifier`. House shape (variant tables `tiny/small/base`, `BACKBONE_NAME="et_backbone"`, factories `create_energy_transformer_*`, `return_energy` float32 probe, warm-start by name). Copy this shape for any new EBM vision head.
- `dl_techniques.models.graph.graph_energy_transformer` (`src/dl_techniques/models/graph/graph_energy_transformer/model.py`): same descent on graphs, adjacency as rank-3 `attention_mask`, `GRAPH_BACKBONE_NAME` convention. Proves ET block is modality-agnostic.
- `dl_techniques.models.language.hrm.HierarchicalReasoningModel` (`src/dl_techniques/models/language/hrm/model.py`): two-timescale `z_h`/`z_l`, `HierarchicalReasoningCore` (`src/dl_techniques/layers/reasoning/hrm_reasoning_core.py` + `hrm_reasoning_module.py`), carry dict + `ops.where` freeze + static trip count for `tf.function`, Q-halting with Bellman target, `from_variant` built-and-uncompiled, `pretrained=True` raises. ERM/EqR language work starts here.
- `dl_techniques.models.language.trm.TRM` (`src/dl_techniques/models/language/trm/model.py` + `components.py::TRMInner`): single-step `call(carry, batch)`, stop-gradient across steps, `no_act_continue` vs Q-learning halting branches, `create_trm` builds inner so `H_init`/`L_init` exist. ERM's Hopfield scorer taps `z_H`/`z_L` here.
- `dl_techniques.models.language.ptrm.PTRM` (`src/dl_techniques/models/language/ptrm/model.py` + `inference.py` + `factory.py`): identical trunk to TRM, K parallel stochastic rollouts with Gaussian noise in `z_L`, select by `q_halt`. This is the width-axis precedent for EqR breadth + ERM tempering; energy selection replaces `q_halt` selection.
- `dl_techniques.models.vision.lewm.LeWM` (`src/dl_techniques/models/vision/lewm/model.py` + `config.py`, `embedder.py`, `predictor.py`, `projector.py`): JEPA-style world model, `add_loss`-only training (`fit` with `loss=None`), SIGReg anti-collapse (`dl_techniques.regularizers.sigreg`). EBRM anchoring / anti-collapse should look at SIGReg first.
- `dl_techniques.models.common.power_sampling` (`src/dl_techniques/models/common/power_sampling/` with `config.py`, `sampler.py`, `forward.py`, `ops.py`, `protocols.py`): model-agnostic inference-time sampling for causal LMs. Natural home for best-of-N + energy rerank + tempered sampler wrappers.
- `dl_techniques.models.common.masked_language_model` (`src/dl_techniques/models/common/masked_language_model/clm.py`, `mlm.py`): `CausalLanguageModel` / `MaskedLanguageModel` wrappers, `aggregate_backbone_losses` flag for `add_loss` backbones (TopoLM precedent). EBT language training will need the same flag pattern.

### 4.3 Losses, regularizers, optimizers, metrics (compose first)

- Losses (`src/dl_techniques/losses/`, ~52 modules, `AnyLoss` registry in `any_loss.py`): `hrm_loss.py::StableMaxCrossEntropy` (per-token `(batch, seq_len)` shape is deliberate, not a scalar defect) + `HRMLoss`/`create_hrm_loss` (LM + Q-halt + Q-continue composite). New `ebt_loss` / `hopfield_energy_loss` / `trajectory_energy_loss` belong here, following the `(batch,)`-per-sample rule in `losses/AGENTS.md` and `@register_dl_technique("dl_techniques.losses.<module>")`.
- Regularizers (`src/dl_techniques/regularizers/`): `sigreg.py::SIGRegLayer` (hyperspherical-energy anti-collapse, used by LeWM), `soft_orthogonal.py`, `srip.py`, `entropy_regularizer.py`. EBRM smoothness / anchoring terms should be evaluated as regularizers before becoming losses.
- Optimizers (`src/dl_techniques/optimization/`): `sgld_optimizer.py::SGLD` (`w - lr*g + s*sqrt(2*lr)*eps`, SeedGenerator, graph-safe) wired as `"sgld"` in `optimizer.py::optimizer_builder` (`OptimizerType`); `vsgd_optimizer.py::VSGD`, `gefen_optimizer.py::Gefen`, `muon_optimizer.py::Muon` (direct construct only). Schedules via `schedule.py::schedule_builder` / `learning_rate_schedule_builder` + `WarmupSchedule`. Langevin inference reuses SGLD; parallel tempering needs a new sampler, not a new optimizer.
- Metrics (`src/dl_techniques/metrics/`): `pass_at_k.py` (unbiased + plug-in estimators over `(n_problems, n_samples)` outcome matrix, plain functions not streaming `Metric`, consumed by `optimization/ssp`) — the metric for best-of-N / breadth evaluation. `perplexity_metric.py`, `hrm_metrics.py`, `topographic.py` (equivariance + CapCorr, plain NumPy), `spatial_autocorrelation.py` (Moran's I). Energy calibration / convergence-trace metrics belong here as plain functions where they need whole trajectories.
- Utils (`src/dl_techniques/utils/`): `logger.py` (mandatory, no prints), `keras_registration.py::register_dl_technique` (mandatory, family-stripped keys, never bare `register_keras_serializable`), `weight_transfer.py` (`load_weights_from_checkpoint` partial vs `load_weights_or_raise` whole, `load_backbone_from_checkpoint` language rule, `pretrained_path_or_raise`), `sequence_length_guard.py` (refuse over-long sequences on every path; every guarded language model lists `RecordedBuildShapeMixin` first), `tensors.py` (`l2_normalize` scale-safe, `canonical_binary_input_shape`), `masking/` (canonical mask factory), `drop_path.py`, `activation_serialization.py`.

### 4.4 Factories that must be reused (factory-first policy)

Per `src/dl_techniques/layers/AGENTS.md` § Layer Reuse Policy, check in order and only then mint a bespoke layer:

1. `create_attention_layer` (35 types), `create_ffn_layer` (22 types), `create_normalization_layer` (19 types), `create_embedding_layer` (15 types), `create_activation_layer` (24 types), `create_sequence_pooling_layer`, `create_logic_layer`, `create_mixture_layer`, `create_head(domain, ...)`, direct-import `TransformerLayer`.
2. Standalone layers (`capsules.py`, `dense_block.py`, `mothnet_blocks.py`, etc.).
3. New layer only in the right `layers/` subpackage + factory registration + tests in `tests/test_layers/`.

Energy-specific duck-typed trio convention (`energy(g) -> (B,)`, `update(g) -> (B,N,D)`, `call -> update`) is by convention, not a base class — new energy terms must honor it so `EnergyTransformer` can compose them.

### 4.5 Trainers and data paths (copy, do not re-derive)

- `src/train/hrm/train_hrm.py::HRMTrainer`: custom GradientTape ACT loop, owns optimizer via `optimizer_builder` + `learning_rate_schedule_builder`, `HRMLoss` + `HRMMetrics`. Template for any energy-guided ACT trainer.
- `src/train/AGENTS.md` Patterns 3 (NLP pretrain/finetune via `train.common.nlp`), 6 (byte LM via `datasets/byte_lm.py`), 8 (auxiliary `add_loss` + `aggregate_backbone_losses=True` + split loss logging, TopoLM exemplar). EBT language training is Pattern 3 + Pattern 8 (energy term rides `add_loss`).
- `src/train/common/`: `setup_gpu`, `create_callbacks` / `create_nlp_callbacks`, `resolve_monitor_mode` (token registry, never substring), `optimizer_builder` (note: clipping keys are `clipvalue`/`clipnorm`/`global_clipnorm`, literal `clipnorm` key otherwise silently ignored; name hard-coded `AdamW`), `load_dataset` (mnist/cifar/imagenet only), `compare_runs`, `StepCheckpointCallback`, `set_seeds`, run-dir contract (`config.json`, `run.log`, `training_log.csv`, `training_history.json`, `best_model.keras`, `final_model.keras`, `results_summary.json`, `visualizations/`, `model_analysis/`).
- Datasets (`src/dl_techniques/datasets/`): `nlp.py::load_wikipedia_train_val` (+ `byte_lm.py` byte sibling), `arc/` (ARC reasoning corpus — closest in-tree data to Sudoku/Maze-style puzzles), `vision/masked_patches.py::make_masked_patch_map_fn` (MIM map fn), `simple_2d.py` (synthetic 2D generators for landscape debugging), `graphs/`, `time_series/`. No Sudoku/Maze/PPBench loader exists yet — new loaders belong here, NumPy-first, TF/TFDS lazily imported.
- Callbacks: reusable in `src/dl_techniques/callbacks/`, shared trainer-side in `src/train/common/callbacks.py`, tightly-coupled beside the thing they serve. Energy-trace logging / convergence early-stopping are new callbacks, not trainer inline code.

---

## 5. Gaps: what does not exist yet

1. **Scalar verifier head.** No `Dense(1)` energy head over `(x, y_hat)` with a `return_energy` float32 contract. ET's `return_energy` is a representation-space trace, not an input-candidate verifier.
2. **Nested-tape thinking loop.** No inner `y <- y - alpha*dE/dy` with outer weight updates, no HVP helper, no `train_step` override that trains through the inner optimization. HRM/TRM loops are carry recurrences with stop-gradient, not gradient-through-optimization.
3. **Trajectory energy.** No `compat + transition + smoothness` decomposition over `z_1:T`, no dual-path decoder, no latent anchor.
4. **Hopfield scorer for recursive trajectories.** No memory-matrix scorer over `(z_H, z_L)` rollouts, no energy-selected best-of-K (PTRM selects by `q_halt`, not energy).
5. **Tempered / Langevin samplers at inference.** SGLD exists as an optimizer, not as an inference-time sampler over `y`/`z` with temperature ladders. No parallel tempering, no PEM, no energy rerank wrapper in `power_sampling`.
6. **Energy-aware losses.** No contrastive energy shaping loss with hard negatives, no EBT-style supervised+contrastive composite, no calibration of energy→uncertainty.
7. **Puzzle benchmarks.** No Sudoku / Maze / PPBench / CNF loaders, no pass@k-over-energy evaluation harness, no energy-vs-accuracy calibration plots.
8. **Metacognitive allocator.** No IBL-style compute allocator; no energy-threshold early-exit callback.

---

## 6. Where the family should live

**LOCKED 2026-10-14: `models/general_purpose/ebm/` as a sam-style subfamily container, first member EBM-LM, single trainer `train/ebm/`.** The alternatives below are retained as the rejected record.

Follow the house rules: leaf package per architecture under a family, family `__init__.py` docstring-only (except `time_series/`), import from leaf, `@register_dl_technique` with family stripped, `create_*` factory + `__all__` + `README.md` per leaf, Sphinx/reST docstrings for new `models/` code (exemplar `models/language/bert/model.py`), `research/2026_keras_custom_models_instructions_v2.md` before any new layer or model.

### 6.1 Locked layout

```text
src/dl_techniques/models/general_purpose/ebm/      # NEW subfamily container (5th nesting)
├── __init__.py                                     # docstring only, lists members, exports nothing
├── ebm_lm/                                         # FIRST PORT: EBT-for-language (verifier energy + thinking loop)
│   ├── __init__.py                                 # create_ebm_lm + __all__
│   ├── model.py                                    # EbmLm (+ backbone), MODEL_VARIANTS, factory
│   ├── energy_head.py                              # scalar Dense(1) verifier head (or layers/energy/ reuse)
│   └── README.md
├── erm/                                            # later: energy-guided recursive scorer over TRM/HRM trunk
├── eqr/                                            # later: fixed-point iterator + convergence + breadth
└── ebrm/                                           # later: trajectory planner + dual-path decoder + anchor

src/train/ebm/                                      # SINGLE trainer dir (bfunet precedent: many scripts, one dir)
├── __init__.py
├── common.py                                       # shared EBM trainer pieces (optimizer, run-dir, callbacks)
├── train_ebm_lm.py                                 # first script (Pattern 3 + Pattern 8)
├── train_erm.py                                    # later
└── README.md                                       # flags + run-dir contract per script
```

Why this shape:

- `general_purpose/` today holds `hkan/`, `kan/`, `mothnet/`, `power_mlp/` (see `models/general_purpose/__init__.py`), all modality-agnostic. EBM verifier/trajectory energies are modality-agnostic, so this is the honest filing — not `language/`, which would misfile the vision/graph reuse the same heads will get.
- Sam-style nesting precedent: `vision_language/sam/` (`sam1/`, `sam2/`, `sam3/`) is a docstring-only container whose leaves each apply the house module shape per inner architecture (`models/AGENTS.md` § When the shape does not apply: multi-model families). `general_purpose/ebm/` becomes the 5th such nesting (after `vision/image_restoration`, `vision/keypoints`, `vision/super_resolution`, `vision_language/sam`). Each EBM member (`ebm_lm`, `erm`, `eqr`, `ebrm`) gets its own `MODEL_VARIANTS` (or documented no-variants justification), factory, `README.md`, and flat test dir.
- Registration consequence: `models/AGENTS.md` strips the 13 family dirs + 4 subfamily containers from the `@register_dl_technique` package string. Adding `ebm` as a 5th container means the strip list must grow by one (`ebm` alongside `sam`), e.g. leaf `general_purpose/ebm/ebm_lm/model.py` registers as `dl_techniques.models.ebm_lm.model` (or `dl_techniques.models.ebm.ebm_lm.model` if the team prefers keeping the container — decide at build time and pin it; precedent `power_mlp/model.py` registers as `dl_techniques.models.power_mlp.model`, family stripped). Record the choice in the leaf README and never change it after the first checkpoint.
- Import rule unchanged: `from dl_techniques.models.general_purpose.ebm_lm import create_ebm_lm` (leaf import; never from the family or container). Container `__init__.py` stays docstring-only so importing one member does not construct the whole family (the `vision/` 50-package eager-import reason in `models/README.md`).
- Trainer precedent: `src/train/bfunet/` holds several `train_*.py` scripts sharing `common.py`; `src/train/convunext/` vs `bfunet/train_convunext_denoiser.py` shows when to split (shared `common.py` keeps it together). All EBM members share the energy loss/metric/sampler substrate, so one `train/ebm/` dir. Script naming obeys `train_<model>.py`, never `train.py` (shadows the `train` package). Runner: `MPLBACKEND=Agg .venv/bin/python -m train.ebm.train_ebm_lm`.
- Tests stay flat per ruling: `tests/test_models/test_ebm_lm/`, `test_erm/`, `test_eqr/`, `test_ebrm/` (never nested), `tests/test_layers/test_energy/`, `tests/test_losses/test_ebt*/`, trainer tests `tests/test_train/test_ebm/`.

### 6.2 Member responsibilities (locked order: EBM-LM first)

- `ebm/ebm_lm/` — **FIRST PORT.** `TextDecoder` trunk (reuse `layers/transformers/text_decoder.py`, do not fork) + scalar `Dense(1)` verifier head `E(x, y_hat)` + differentiable thinking loop (`y <- y - alpha*dE/dy`, N steps, call-time knob). Override `train_step` (outer weights / inner candidate tapes). See §7–§8.
- `ebm/erm/` — thin wrapper over frozen-or-joint TRM/HRM trunk (`models/language/trm`, `models/language/hrm`, selection logic from `models/language/ptrm/inference.py`) + Hopfield trajectory scorer + tempered selector. Smallest delta; second build.
- `ebm/eqr/` — fixed-point iterator + convergence monitor (energy delta, replaces `q_halt`) + breadth aggregator (K inits, `pass_at_k` scoring). Shares sampler with ERM.
- `ebm/ebrm/` — trajectory planner (`compat + transition + smoothness`) + dual-path decoder + anchor. Last model build (drift risk, see §3.4).
- `layers/energy/` (new subpackage, exports-nothing like `reasoning/`) — shared terms: scalar verifier head, compat/transition/smoothness terms, Hopfield scorer, anchor penalty. Honor the `energy/update/call` duck-typed trio. Direct import (precedent `dynamic_chunking/`, `blt/`); add a local factory only if interchangeable terms emerge.
- `losses/ebt_loss.py`, `losses/trajectory_energy_loss.py`, `losses/hopfield_energy_loss.py` — energy-shaping losses, `(batch,)` shape, registered.
- `models/common/power_sampling/` extensions — energy rerank, tempered sampling, convergence early-exit as inference wrappers, not model changes.
- `datasets/puzzles/` (new under `datasets/`) — Sudoku, Maze, PPBench, CNF loaders. NumPy-first, matching `arc/` precedent.
- `datasets/puzzles/` (new under `datasets/`) — Sudoku, Maze, PPBench, CNF loaders. NumPy-first, matching `arc/` precedent.
- `tests/test_models/test_ebm_lm/`, `test_erm/`, `test_eqr/`, `test_ebrm/` (flat, per ruling — never `test_vision/test_*`), plus `tests/test_layers/test_energy/`, `tests/test_losses/test_ebt*/`, trainer tests in `tests/test_train/test_ebm/` (single dir matching `train/ebm/`).

What NOT to do: do not add a `models/`-level re-export; do not register `HopfieldNetwork` as an FFN type; do not invent `MODEL_VARIANTS` without genuine named variants; do not convert docstring styles wholesale; do not delete `# DECISION` anchors.

---

## 7. Keras 3 / TF porting notes

Distilled from the user's porting notes plus house constraints in `models/GUIDE.md` and the authoring guide:

- **Nested tapes for thinking.** Outer `tf.GradientTape` over weights, inner tape over candidate `y` (or latent `z`), `y <- y - alpha*dE/dy` for N steps. Keep inner graph differentiable for EBT-style through-optimization training. Wrap the inner loop in `tf.function(jit_compile=True)`; verify on GPU (XLA-on-GPU failures are invisible to a CPU suite per `models/GUIDE.md`).
- **Hessian-vector products scale linearly.** TF nested tapes give HVPs without materializing Hessians. Test memory vs N (steps) explicitly — backward memory is linear in N, same as ET's `num_steps` caveat.
- **Subclass `keras.Model`, override `train_step`.** Keep energy head as scalar `Dense(1)`. Return split losses (task vs energy vs anchor) so a rising curve can be attributed — TopoLM Pattern 8 precedent (`SpatialLossLogger` analog).
- **Precision.** Energy traces stay float32 even under `mixed_float16` (ET D-011 precedent: `EnergyLayerNorm` backward overflows fp16; heads consuming traces must be float32). Pin with a mixed-precision test.
- **Build discipline.** Unconditional sublayer creation in `__init__`/`build`, `compute_output_shape` on every custom layer, `get_config` round-trips every ctor arg, `from_config` only when it does real work, `pretrained=True` raises `NotImplementedError` when no weights ship. `build()` before first `call` where carry reset reads `H_init`/`L_init` (HRM D-005 / TRM precedent).
- **Graph safety.** No Python `break` on tensor predicates under `tf.function` — freeze with `ops.where` + static trip count (HRM/TRM precedent). No eager ops at module scope. `keras.ops` over raw `tf` except genuinely unmigratable sites (FFT/SVD). Central `dl_techniques.utils.logger`, never `print`, never log inside `call()`.
- **Masking and causality.** EBT text head must pass the future-leak probe if wrapped by `CausalLanguageModel` (see `models/language/AGENTS.md`). Energy attention over padded positions needs explicit keep masks — `mask_zero=True` metadata is dropped with a warning, not honored.
- **Serialization.** `@register_dl_technique("dl_techniques.models.<name>.<module>")` (family stripped), never bare `register_keras_serializable`. Backbone naming stable for warm-starts (`BACKBONE_NAME` / `GRAPH_BACKBONE_NAME` precedent). Verify `.keras` round-trip + weight transfer via `weight_transfer.py` helpers, not `load_weights(by_name=True)` (broken on `.keras` in Keras 3.8).
- **Second-order + XLA + mixed precision is the riskiest intersection.** Plan a dedicated probe: tiny EBT, N=2 inner steps, `jit_compile=True`, `mixed_float16`, CPU + GPU, assert finite energies, finite grads, and descending energies across inner steps.

---

## 8. Training and inference design

### Training

- **EBT-LM (`ebm/ebm_lm`, LOCKED first port):** supervised next-token/CE + contrastive energy shaping (positives down, hard negatives up). Batches are plain tokenized text via `train.common.nlp` (`preprocess_clm_dataset`, `estimate_clm_steps_per_epoch`, `create_warmup_lr_schedule`); masking inside `train_step` where needed. Auxiliary energy term rides `add_loss` → head built with `aggregate_backbone_losses=True`. Trainer `train/ebm/train_ebm_lm.py` (Pattern 3 + Pattern 8).
- **ERM on TRM/HRM:** keep existing ACT loop (`train/hrm/train_hrm.py` shape, bespoke carry loop, `HRMLoss` + `StableMaxCrossEntropy`); add Hopfield energy scorer trained jointly or post-hoc on frozen trunk rollouts (ablate both). Exploration via existing `halt_exploration_prob` + new tempered noise.
- **EBRM:** supervised encoder-decoder pretraining first, then contrastive shaping with hard negatives, then dual-path decoder finetuning (encoder outputs + planner outputs) + anchor weight sweep. Six-part ablation from the paper, in order.
- **Optimizer/schedule:** `optimizer_builder` + `learning_rate_schedule_builder` (AdamW default, SGLD for Langevin phases). Never double weight decay (AdamW decay XOR `kernel_regularizer`, not both).

### Inference

- **Depth knob:** N thinking / fixed-point steps, exposed as a call-time argument (like ET `num_steps`), not baked into config. Convergence monitor (energy delta) drives adaptive exit.
- **Width knob:** K stochastic trajectories (PTRM precedent: noise in `z_L`; EqR: multiple inits; ERM: tempering) + energy selection (min-energy or tempered rerank). Evaluate with `pass_at_k` over the pool (`metrics/pass_at_k.py`), not mean accuracy.
- **Composition:** energies sum across subproblems (PEM). Keep head scales compatible; document the summation contract in the head's docstring.
- **Metacognition (later):** IBL allocator as an inference wrapper mapping pre-response uncertainty (initial energy / energy slope) to (N, K, temperature). Log (uncertainty, cost, correctness) instances per run to bootstrap it.

---

## 9. Data and benchmarks

- **Puzzles:** Sudoku, Sudoku-Extreme, Maze, PPBench, CNF logic — new loaders in `datasets/` (propose `datasets/puzzles/`). Start from `datasets/arc/` conventions; seed everything; return ground-truth factors alongside sequences where CapCorr-style analysis applies (precedent: `datasets/vision/transform_sequences.py` returns the varied factor).
- **Language:** Wikipedia packed-CLM (`datasets/nlp.py` + `train.common.nlp`) for EBT pretraining; RedPajama-V2 slice for ERM perplexity replication (−1.74% claim); Shakespeare/ListOPS tiny suites for NRGPT-style smoke tests.
- **Vision (optional, later):** masked-patch MIM via `datasets/vision/masked_patches.py` if EBT-vision is pursued; ET-MIM already covers the backbone.
- **What to report per run:** task accuracy + `pass@k` curve + energy-vs-correctness calibration + steps-to-convergence histogram + cost (forward passes / wall time). Single-monitor runs (`val_accuracy` only) will hide the compute/accuracy trade the family exists to expose.

---

## 10. Test and guard plan

Per `models/AGENTS.md` verification conventions + `models/GUIDE.md` §8 (write guards that can fail; prove RED by injecting the defect):

- **Contract:** `smoke_contract_oracle.assert_contract_rejects_a_broken_forward` (requires `AssertionError`, not `Exception`).
- **Knobs:** `knob_sensitivity_oracle` — structural knobs (depth/heads/steps) pinned on weight-shape signatures; value knobs (alpha, anchor weight, temperature) on output differences.
- **Energy descent:** assert energies strictly decrease across inner steps on a fixed batch (float32, CPU + GPU + XLA); RED proof by flipping the update sign.
- **Langevin / tempering:** assert noise scale `sqrt(2*lr)` wiring (zero LR → zero noise; `noise_scale=0` → SGD-identical); assert temperature ladder actually diversifies trajectories (pairwise distance > epsilon).
- **Selection:** assert min-energy selection beats random selection on a synthetic landscape; assert `pass@k` monotonic in K.
- **Drift (EBRM):** assert decoder trained dual-path outperforms encoder-only-path on planner outputs; assert anchor weight moves `||z_T - h_x||` monotonically.
- **Serialization:** `.keras` round-trip + `load_weights_or_raise` whole-restore + `load_weights_from_checkpoint` warm-start (ET backbone-name precedent); colliding bare names pinned by `test_the_legacy_alias_namespace_has_no_collisions.py`.
- **Precision:** `dtype_policy` fixture over float32 / mixed_float16 / float64; `tf32_disabled` where needed; energy trace stays float32 under fp16 heads.
- **Docs:** `tests/test_docs_make_no_unearned_performance_claims.py` — credit papers for numbers, never this code until measured.
- **Collection:** `pytest tests/test_models/ -q --collect-only` after any public-surface change; scope runs to touched modules (full suite ~1.5h, never parallel pytest jobs).

---

## 11. Phased roadmap

**Locked order: EBM-LM first, single `train/ebm/` trainer.** ERM second (reuses the substrate EBM-LM proves).

| Phase | Delivers | Reuses | Risk |
|---|---|---|---|
| 0 — Doc (this file) | Design consensus + locked §6/§12 | — | Low |
| 1 — Shared energy substrate | `layers/energy/` terms + `losses/` energy losses + energy-trace callback + puzzle loaders | ET trio, SIGReg, SGLD, `pass_at_k`, `arc/` | Low-medium |
| 2 — EBM-LM (first model, LOCKED) | `models/general_purpose/ebm/ebm_lm/` + `train/ebm/train_ebm_lm.py` (Pattern 3 + Pattern 8), nested-tape `train_step`, XLA + mixed-precision probes, tiny/model variants | `TextDecoder`, `TransformerLayer`, CLM wrapper, `StableMaxCrossEntropy` shape precedent | High — second-order + XLA intersection; keep N=2 first, Shakespeare/ListOPS smoke before Wikipedia |
| 3 — ERM on TRM/HRM | `models/general_purpose/ebm/erm/` + `train/ebm/train_erm.py` on Sudoku/Maze, energy selection replacing `q_halt` selection | TRM/PTRM trunk + `ptrm/inference.py`, HRM trainer loop, Phase-1 scorer | Medium — proves energy selection without a new backbone |
| 4 — EqR breadth + composition | `models/general_purpose/ebm/eqr/` convergence monitor + width aggregator + PEM summation contract | PTRM K-rollouts, `power_sampling/`, `pass_at_k` | Medium |
| 5 — EBRM planner | `models/general_purpose/ebm/ebrm/` trajectory planner + dual-path decoder + anchor sweep + six-part ablation | LeWM `add_loss` pattern, EBM-LM head | High — drift risk; do not start before Phase 2 energies are stable |
| 6 — MERITED allocator | IBL compute allocator + calibration harness | Phase 2–5 run logs as instance store | Low (wrapper only) but last |

Each phase lands a leaf `README.md`, flat tests (`tests/test_models/test_ebm_lm/` first), trainer script in `train/ebm/`, and a `results_summary.json` with the §9 metrics — no phase claims a paper number until measured in this repo. Update `models/README.md` leaf census (currently 107 leaves) + `general_purpose/__init__.py` docstring member list + `models/AGENTS.md` counts on landing (re-derive with the `find` commands, never transcribe).

---

## 12. Open decisions

**Decided 2026-10-14 (locked, §6.1 is authoritative):**

1. **Family placement: `general_purpose/ebm/` — DECIDED.** Sam-style subfamily container under `general_purpose/` (5th nesting). Rejected per-modality `language/ebt` + `vision/ebt` split: energies are modality-agnostic and the split would duplicate the scorer/sampler substrate across families. Container `__init__.py` docstring-only; leaves `ebm_lm/`, `erm/`, `eqr/`, `ebrm/`; strip-list in `models/AGENTS.md` gains `ebm`.
2. **First model: EBM-LM (EBT-for-language) — DECIDED.** Rejected ERM-first despite its lower risk: EBM-LM proves the nested-tape + verifier-head + `train_step` substrate every later member reuses. ERM follows in Phase 3 on the proven substrate. Keep N=2 + Shakespeare/ListOPS smoke first.
3. **Trainer: single `train/ebm/` — DECIDED.** Rejected `train/ebt/` + `train/erm/` split: members share losses/metrics/samplers, so one dir with `common.py` + per-member `train_*.py` (bfunet precedent). First script `train/ebm/train_ebm_lm.py` (Pattern 3 + Pattern 8); do not fork `train/hrm/`, copy its ACT loop shape where needed.
4. **Sampler home:** extend `models/common/power_sampling/` vs new `layers/energy/samplers.py`? Recommendation: extend `power_sampling/` (model-agnostic inference is its charter).
5. **Puzzle data scope:** minimal Sudoku/Maze loaders vs full PPBench/CNF upfront? Recommendation: Sudoku + Maze first, PPBench/CNF in Phase 5.
6. **EBT backbone:** `TextDecoder` + `Dense(1)` vs bespoke verifier stack? Recommendation: `TextDecoder` reuse; bespoke only if the energy/gradient taps need pre-add branch tensors (TopoLM precedent).

---

## 13. References

**Core (2025–2026):**

- Gladstone et al. 2025. Energy-Based Transformers are Scalable Learners and Thinkers. ICLR 2026 Oral. `https://arxiv.org/abs/2507.02092` — code `https://github.com/alexiglad/EBT` — site `https://energy-based-transformers.github.io`.
- Huang, Geng, Kolter 2026. Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning. ICML 2026. `https://arxiv.org/abs/2605.21488`.
- Zhao, Tang 2026. Energy-guided Recursive Model. `https://arxiv.org/abs/2607.10128`.
- Johansson 2026. Reasoning as Energy Minimization over Structured Latent Trajectories (EBRM). `https://arxiv.org/abs/2603.28248` — code `https://github.com/Shr1ftyy/ebrm`.
- Oarga et al. 2025. Compositional Energy Minimization. NeurIPS 2025. `https://alexoarga.github.io/compositional_reasoning/`.
- Malloy et al. 2026. MERITED: Metacognitive Reasoning in EBMs using IBLT. `https://arxiv.org/abs/2610.00399`.
- Dehmamy et al. 2025/2026. NRGPT: An Energy-based Alternative for GPT. ICLR 2026. `https://arxiv.org/abs/2512.16762`.
- Solve the loop: Attractor models (2605.12466); Flow reasoning models (2606.29150); Fractal basins trap latent reasoning (2609.04963); Fixed-Point Reasoners (2606.18206).

**Foundations:**

- LeCun et al. 2006. A Tutorial on Energy-Based Learning.
- Du et al. 2022. Learning iterative reasoning through energy minimization (ICML); 2024 energy diffusion (`arXiv:2406.11179`).
- Hoover et al. 2023. Energy Transformer (NeurIPS). `https://arxiv.org/abs/2302.07253`.
- Ramsauer et al. 2020. Hopfield Networks is All You Need. `https://arxiv.org/abs/2008.02217`; Krotov & Hopfield 2016 Dense Associative Memory. `https://arxiv.org/abs/1606.01164`.

**Overviews:**

- Shekhar 2026. Energy-Based Models: From Basics to LLMs. `https://www.shashankshekhar.com/blog/ebm-talk`.
- Logical Intelligence 2026. Energy-Based Models for Reasoning. `https://logicalintelligence.com/blog/energy-based-models-for-reasoning`.
- Du lab overview. `https://yilundu.github.io/`.

**Code references:**

- `https://github.com/alexiglad/EBT` (PyTorch EBT — port reference).
- `https://github.com/Shr1ftyy/ebrm` (PyTorch EBRM — Langevin inference + best-of-N).
- `https://github.com/raoulnormand/mnist_distribution` (TF 2.8 / Keras 2.8 Langevin + contrastive loop template).
- Foster, Generative Deep Learning 2nd ed., EBM chapter (Keras energy + Langevin).
- Sghaier et al. 2026. PTRM. `https://arxiv.org/abs/2605.19943`; Jolicoeur-Martineau 2025 TRM. `https://arxiv.org/abs/2510.04871`; Wang et al. 2025 HRM. `https://arxiv.org/abs/2506.21734`.
