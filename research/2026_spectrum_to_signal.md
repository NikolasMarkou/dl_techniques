# Spectrum-to-Signal Principle (SSP) — the paper, and what was actually built

**Source:** Sen Xu, Yi Zhou, Wei Wang, Jixin Min, Zhibin Yin, Yingwei Dai, Shixi Liu,
Lianyu Pang, Yirong Chen, Junlin Zhang. *"Tiny Model, Big Logic: Diversity-Driven
Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B."* Weibo, technical
report, 7 Nov 2025.
<https://github.com/WeiboAI/VibeThinker> · <https://huggingface.co/WeiboAI/VibeThinker-1.5B>

Built as `src/dl_techniques/optimization/ssp/` (+ `src/dl_techniques/metrics/pass_at_k.py`).

---

## 1. The claim

Decouple SFT and RL into two phases with *different* objectives:

| Phase | Stage | Objective | Metric |
|---|---|---|---|
| **Spectrum** | SFT | maximise `Pass@K` — build a broad candidate space | coverage, not accuracy |
| **Signal** | RL | maximise probability of the correct paths *within* that space | reward on an existing spectrum |

The paper's argument is that the prevailing implicit convention — pick the SFT
checkpoint with the best `Pass@1`, then RL the same metric — artificially caps what RL
can achieve, because it narrows the candidate space before the amplification stage has
anything to amplify.

Reported outcome: a 1.5B dense model (Qwen2.5-Math-1.5B base) beating DeepSeek-R1-0120
(671B) on AIME24 (80.3 vs 79.8), AIME25 (74.4 vs 70.0) and HMMT25 (50.4 vs 41.7), for
~$7,800 in post-training (3.9K H800-hours).

## 2. What the paper specifies, in full

### 2.1 Spectrum phase

**Domain-Aware Diversity Probing.** Partition the domain into `N` subdomains
`S = {S_1..S_N}` (the paper uses algebra / geometry / calculus / statistics). Build a
probing set `D_i` per subdomain. During SFT, evaluate intermediate checkpoints `M_t`
(every `k` steps) on each `D_i` with Pass@K, giving `P_i(t)`, and keep

```
M_i* = argmax_t P_i(t)
```

**Expert Model Fusion.** Combine the `N` diversity-maximising specialists:

```
M_merge = Σ_{i=1}^{N} w_i · M_i*,    w_i ≥ 0,  Σ_i w_i = 1
```

The paper uses the uniform case `w_i = 1/N`, and reports that the fused model attains
state-of-the-art on **both** Pass@K and Pass@1 — the claim being that optimising the
spectrum does not cost single-shot accuracy, and that a broader spectrum may even
reinforce the most correct pathways.

### 2.2 Signal phase — MGPO

For query `q`, sample `G` rollouts with rewards `r_i`. The empirical success rate is

```
p_c(q) = (1/G) Σ_i 1[r_i = 1]
```

The premise: a problem's training value peaks where the policy is *maximally
uncertain* about it, which for a binary outcome is `p_c = 0.5`. Rather than using
Shannon entropy `H(q)` directly, the paper defines a distance from that ideal state and
weights by its exponential negative:

```
w_ME(p_c) = exp(-λ · D_ME(p_c ‖ p_0)),   p_0 = 0.5
```

applied directly to the GRPO advantage:

```
A'_j(q) = w_ME(p_c(q)) · A_j(q)
```

`λ = 0` recovers standard GRPO. The paper describes this as an implicit curriculum that
steers gradient updates toward the most ambiguous questions.

## 3. Two corrections

### 3.1 The printed `D_ME` is not a KL divergence

The report prints

```
D_ME(p_c ‖ p_0) = p_c · log(p_c / (1 − p_c)) + (1 − p_c) · log(p_0 / (1 − p_0))
```

The first denominator should be `p_0`, not `(1 − p_c)`. As printed:

* it is **asymmetric** in a way a divergence between two Bernoulli distributions need
  not be, and not in a way that reflects either distribution;
* it is **negative** at `p_c = 0.1, p_0 = 0.5`: `0.1·log(0.1/0.9) + 0.9·log(1) ≈ −0.22`;
* it does **not** vanish at `p_c = p_0` for general `p_0` — it vanishes at
  `p_c = 0.5` only by coincidence, since `0.5·log(1) + 0.5·log(1) = 0`;
* so it cannot be the "distance from the ideal maximum-entropy state" the surrounding
  prose describes.

**Implemented instead** — the Bernoulli KL:

```
D_ME(p ‖ p_0) = p·log(p/p_0) + (1−p)·log((1−p)/(1−p_0))
```

which is zero exactly at `p = p_0`, non-negative everywhere, and `log 2` at either
endpoint for `p_0 = 0.5`. That is precisely the quantity the prose describes.

### 3.2 The closed form is the interesting part

For two Bernoullis the KL *is* the entropy deficit:

```
D_KL(p ‖ p_0) = −H(p) + H(p, p_0),    H(p, p_0) = −p·log p_0 − (1−p)·log(1−p_0)
```

At `p_0 = 0.5` the cross-entropy is the constant `log 2`, so

```
D_ME(p ‖ 0.5) = log 2 − H(p)        w_ME(p) = 2^(−λ) · e^(λ·H(p))
```

`w_ME` is therefore **strictly monotone increasing in binary entropy**, with three exact
properties the test suite pins:

| | |
|---|---|
| `w_ME(0.5)` | `= 1.0` — maximum weight at maximum uncertainty |
| `w_ME(0) = w_ME(1)` | `= 2^(−λ)` — exponentially separated floor |
| `λ = 0` | `≡ 1` — weighting vanishes, plain GRPO recovered bit-for-bit |

This is not a transcription detail — it is what makes the weight cheap (two transcendentals
and an `exp`), monotone by inspection, and diagnosable. Nothing in the paper derives it.

### 3.3 The low-probability-trace term is not implemented

The report also claims MGPO "modif[ies] the advantage calculation to incentivize the
increased generation probability of low-probability yet correct reasoning traces sampled
during rollouts."

**No formula for this term appears anywhere in the paper.** Only the advantage
reweighting is specified. It is therefore *not* implemented here, and no such behaviour
should be attributed to this code.

## 4. What was generalised, and why it is legitimate

### 4.1 `Pass@K` → any verifier

The paper's Pass@K assumes a binary verifier (unit tests, answer match). The
implementation takes any `(n_problems, n_samples)` outcome matrix — binary, or continuous
in `[0,1]` — because coverage is a property of a *candidate pool*, not of a task family.
`> 0` marks solved; threshold partial credit upstream.

The estimator is Chen et al. 2021's unbiased `1 − C(n−c,k)/C(n,k)`, evaluated through
`gammaln` and verified against exact `math.comb` to `4e-15` over 500 random matrices ×
every valid `k`. Two details worth recording:

* the degenerate region is `n − c < k`, **not** `≤`. At `n − c == k` the falling
  factorial lands on `1`, not `0`; the wider test scored a never-solved row as `1.0`,
  i.e. a model that solved nothing reported as perfect coverage;
* `k > n_samples` **raises** rather than clamping, since clamping returns `1.0` for every
  problem.

### 4.2 The Signal half is not RL-bound

`w_ME` is a per-group weight vector. It multiplies *whatever* advantage you have:

* group-relative policy optimisation (GRPO advantages),
* RLOO / any group-baseline estimator,
* **a per-example curriculum weight in ordinary supervised training** — the "group"
  being a set of sampled targets for one input, and `p_c` the fraction the model
  currently solves.

That last use needs no RL machinery and is already built:
`spectrum_sampling_weights(mode="max_entropy")` calls `max_entropy_weight` **verbatim**,
so the checkpoint-side and data-side criteria cannot drift apart. A test pins them
together to float32 precision on identical inputs.

The contrast with the two naive modes is the argument for the criterion:

| `mode` | peaks at | effect |
|---|---|---|
| `proportional` | coverage 1.0 | up-weights what the model already solves best |
| `inverse` | coverage 0.0 | points at the unreachable, where reward is noise |
| `max_entropy` | coverage 0.5 | the ambiguous middle |

### 4.3 Fusion: two better-known siblings

Uniform `w_i = 1/N` is the paper's scheme and the default. Also shipped, because they
cost nothing and are strictly more capable when a score or a base is available:

* **score-softmax weights** — `w_i = softmax(P_i / T)` over measured Pass@K, so capacity
  tracks how broad each specialist actually was;
* **task arithmetic** — `base + c·Σ w_i(M_i − base)`;
* **greedy soup** — add candidates while a held-out score rises.

A derived result, pinned by test: with a **shared** base and `c = 1`, task arithmetic
and linear fusion are the *same function*, because `Σ w_i = 1`. The `mode` knob bites
only across different ancestors or at `c ≠ 1` — which is exactly the paper's setting, so
for specialists fine-tuned from one ancestor the knob is inert by algebra.

## 5. Three findings that would otherwise look like bugs

### 5.1 The batch mean of the objective is identically zero

A group-relative advantage satisfies `Σ_i A_i = 0` by construction, and the max-entropy
weight is **constant across a group** (it depends only on that group's `p_c`). Hence

```
Σ_i w(p_c)·A_i = w(p_c)·Σ_i A_i = 0
```

for every group, for every `λ`, including `λ = 0`. Consequences:

* a logged MGPO loss pinned at `0.000` is **not broken**;
* `normalize_weights` cannot change the batch mean, only the scale;
* the reweighting is invisible in any batch-mean scalar and shows up only in the
  **gradient** — in how strongly each group pulls relative to the others.

So log the per-response vector (`loss.call(...)`, which returns `(B,)`) or the weights.
`MGPOObjective.__call__` reduces to a scalar; `call` does not.

### 5.2 `spectrum_gain` cannot detect a degenerate selection

`spectrum_gain` is defined as (mean of per-subdomain argmax scores) − (best column mean).
An implementation that ignored subdomains and returned the best-overall checkpoint for
every column yields **exactly 0.0** too, because the best-overall row's mean *is* the
mean of its own per-column values. So a gain of zero cannot distinguish "nothing to gain"
from "never looked". The discriminating statistic is the count of subdomains where the
selection is a true column argmax: 1 of 3 for the degenerate answer, 3 of 3 for the real
one. Pinned by test.

### 5.3 Keras 3 never passes `sample_weight` to `call`

`keras.losses.Loss.__call__` invokes `self.call(y_true, y_pred)` — two arguments — and
applies `sample_weight` itself, afterwards, inside `reduce_weighted_values`. A shape
check written inside `call` is therefore **dead code**. This shipped in the first draft
of `MGPOObjective` as a validation branch that could never fire; it is now an
`__call__` override, following `losses/dino_loss.py`.

Consequences for the design:

* `call` returns `(B,)`, satisfying `losses/AGENTS.md`'s one-value-per-sample rule, and
  Keras' `sum_over_batch_size` then divides by `B` — which is the paper's outer `1/G`;
* the **advantage cannot be handed over as `sample_weight`**, because Keras multiplies
  by it after `call` returns and the value is already advantage-weighted. Worse, the PPO
  clip is not linear in the advantage — `min(r·A, clip(r)·A)` selects a different branch
  depending on `sign(A)` — so it cannot be factored out at all. Hence the packed
  `(B, T, 2)` `y_true` (`[..., 0]` = reference log-probs, `[..., 1]` = advantage).

## 6. What is NOT implemented, and what is untested

**Not implemented:**

* an SSP trainer, rollout sampler or verifier — none exists in this repository, and none
  is shipped here. `MGPOObjective` is the objective, not the algorithm;
* the low-probability-trace term (§3.3) — unspecified in the source;
* probe-set construction ("a capable LLM builds a probing set per subdomain") — a data
  pipeline concern;
* Pass@K evaluation against real benchmarks.

**Not validated:** every function here is arithmetic whose behaviour is pinned by tests
(300 in `test_ssp/`, 63 in `test_pass_at_k.py`). Whether the principle *improves a
downstream metric* is the paper's empirical claim, and **it has not been reproduced
here**. No benchmark, A/B run, or performance number is claimed anywhere in this
package. `enable=False` exists precisely so that claim can be measured against itself
when someone runs it.

## 7. Files

| Path | Contents |
|---|---|
| `src/dl_techniques/metrics/pass_at_k.py` | `pass_at_k`, `pass_at_k_curve`, `per_problem_pass_at_k`, `solved_counts` |
| `src/dl_techniques/optimization/ssp/spectrum.py` | `spectrum_profile`, `select_specialists`, `spectrum_sampling_weights` |
| `src/dl_techniques/optimization/ssp/fusion.py` | `fuse_specialists`, `fusion_weights_from_scores`, `uniform_weights`, `score_softmax_weights`, `greedy_soup` |
| `src/dl_techniques/optimization/ssp/signal.py` | entropy, `max_entropy_deviation`, `max_entropy_weight`, advantages, `mgpo_surrogate`, `MGPOObjective` |
| `src/dl_techniques/optimization/ssp/config.py` | `SSPStrategyConfig`, `SSPStrategy`, `ssp_builder` |
| `src/dl_techniques/optimization/ssp/README.md` | principle, closed forms, gotchas |
| `src/dl_techniques/optimization/ssp/AGENTS.md` | module map, structural properties, authoring rules |
| `tests/test_optimization/test_ssp/` | 300 tests, with RED proofs per file |
| `tests/test_metrics/test_pass_at_k.py` | 63 tests |

## References

- Xu, S. et al. (2025). Tiny Model, Big Logic. Weibo technical report. — the SSP
  framework, Domain-Aware Diversity Probing, Expert Model Fusion, MGPO.
- Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code.
  arXiv:2107.03374 — the unbiased `pass@k` estimator.
- Dalal, U. et al. (2025). Leveraging LLM Inconsistency to Boost Pass@k Performance.
  arXiv:2505.12938 — the diversity / Pass@K link SSP relies on.
- Shao, Z. et al. (2024). DeepSeekMath. arXiv:2402.03300 — GRPO.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. — the clipped
  surrogate.
- Wortsman, T. et al. (2022). Model soups. arXiv:2203.05482 — greedy soup, and uniform
  averaging.
- Ilharco, G. et al. (2023). Editing models with task arithmetic. arXiv:2212.04089 —
  task-arithmetic fusion.
