# `dl_techniques.optimization.ssp`

The **Spectrum-to-Signal Principle**: a two-phase contract for a
supervised-then-reinforcement pipeline.

| Phase | Stage | Question it answers | Implementation |
|---|---|---|---|
| **Spectrum** | SFT | *How broad is this model's repertoire?* | `spectrum_profile`, `select_specialists`, `fuse_specialists` |
| **Signal** | RL | *Which groups is the policy still unsure about?* | `max_entropy_weight`, `mgpo_advantages`, `MGPOObjective` |

The claim binding them: a checkpoint selected for breadth is a better precondition for
the second phase than one selected for first-guess accuracy, because the second phase
can only amplify a signal that already exists somewhere in the candidate space.

Introduced as the organising principle of **VibeThinker-1.5B** (Xu et al., 2025,
Weibo), which reports a 1.5B model beating DeepSeek-R1-0120 (671B) on AIME24/25 and
HMMT25. Those are *the paper's* numbers; nothing here has been trained, benchmarked
or A/B-tested in this repository.

## What is actually general here

Neither phase is bound to language models, to GRPO, or to RL at all.

**The Spectrum half** takes any `(n_problems, n_samples)` outcome matrix — unit
tests, rubric scores, a verifier's pass/fail — and any set of checkpoints whose
weights can be averaged.

**The Signal half** produces a per-group weight vector that multiplies *whatever
advantage you already have*:

```python
weights * group_relative_advantage(...)        # group-relative policy optimisation
weights * rloo_advantage(...)                  # any group-baseline estimator
weights * np.ones(len(dataset))                # a per-example curriculum weight,
                                               # in ordinary supervised training
```

That last one needs no RL machinery at all, and it is already built:
`spectrum_sampling_weights` calls `max_entropy_weight` **verbatim**, so the
data-side and checkpoint-side criteria cannot drift apart.

```python
import numpy as np
from dl_techniques.optimization.ssp import spectrum_sampling_weights

# Per-example coverage: the fraction of sampled targets the model currently solves.
coverage = np.array([0.0, 0.5, 1.0, 0.5])
probabilities = spectrum_sampling_weights(coverage, mode="max_entropy", lam=2.0)
# [0.064 0.340 0.255 0.340]  -- peaks at coverage 0.5, the ambiguous middle
```

Compare the two naive alternatives, which is the reason the middle matters:

| `mode` | peaks at | what it does |
|---|---|---|
| `"proportional"` | coverage **1.0** | up-weights what the model already solves best |
| `"inverse"` | coverage **0.0** | points at what it cannot reach, where the reward is noise |
| `"max_entropy"` | coverage **0.5** | the ambiguous middle |

## Quick start

```python
from dl_techniques.optimization import ssp_builder

ssp = ssp_builder({
    "type": "ssp_v1",
    "config": {
        "enable": True,          # master switch; OFF is a strict no-op
        "pass_at_k": 8,          # breadth is measured at this k
        "fusion_weight_scheme": "score_softmax",   # or the paper's "uniform"
        "mgpo_lambda": 2.0,      # max-entropy sharpening; 0.0 disables weighting
    },
})
```

`enable=False` is the **A/B control**: sampling weights become uniform, fusion
weights become uniform, the profile's headline `pass@k` becomes `pass@1`, and
`advantages()` returns the plain group-relative advantage. A recipe can carry the
config unconditionally and flip one flag to measure the principle against itself.

## Entry points

| Import from `dl_techniques.optimization.ssp` | Returns |
|---|---|
| `ssp_builder(config)` | `SSPStrategy` |
| `SSPStrategyConfig(**kwargs)` / `.get_config()` | config object |
| `spectrum_profile(outcomes, ks=None, estimator="unbiased")` | breadth + diagnostics dict |
| `select_specialists(score_matrix, subdomains=None)` | per-subdomain `argmax` + `spectrum_gain` |
| `spectrum_sampling_weights(coverage, mode, lam, p0)` | normalised probability vector |
| `fuse_specialists(specialists, weights, mode, base, coefficient)` | **new** arrays |
| `fusion_weights_from_scores(scores, scheme, temperature)` | weight vector |
| `uniform_weights(n)` / `score_softmax_weights(scores, temperature)` | weight vectors |
| `greedy_soup(pool, score_fn, max_size)` | `(indices, uniform weights)` |
| `binary_entropy(p)` | Bernoulli entropy in nats |
| `max_entropy_deviation(p, p0=0.5)` | `KL(Bern(p) ‖ Bern(p0))` |
| `max_entropy_weight(p, lam=1.0, p0=0.5)` | `exp(-λ · D_ME)` |
| `group_success_rate(rewards, group_axis=-1)` | `p_c` per group |
| `group_relative_advantages(rewards, ...)` | `(r − μ)/(σ + ε)` |
| `mgpo_advantages(rewards, lam, p0, ...)` | `w_ME(p_c) · A_j` |
| `mgpo_surrogate(log_probs, old_log_probs, advantages, ...)` | PPO clipped objective |
| `pack_mgpo_targets(old_log_probs, advantage)` / `unpack_mgpo_targets(y_true)` | `(B, T, 2)` packing |
| `MGPOObjective(clip_eps=0.2)` | a `keras.losses.Loss` |

`pass@k` itself lives in **`dl_techniques.metrics.pass_at_k`** and is imported here,
not duplicated — it is a metric, and one implementation of it is the point:

```python
from dl_techniques.metrics.pass_at_k import pass_at_k, pass_at_k_curve
```

## The weight, in closed form

For a Bernoulli, the KL against a Bernoulli *is* the entropy deficit:

```
D_ME(p ‖ 0.5) = ln2 − H(p)        w_ME(p) = 2^(−λ) · e^(λ·H(p))
```

So `w_ME` is strictly monotone in binary entropy, with three properties the tests pin
exactly:

| | |
|---|---|
| `w_ME(0.5)` | `1.0` — maximum weight at maximum uncertainty |
| `w_ME(0) = w_ME(1)` | `2^(−λ)` — exponentially separated floor |
| `λ = 0` | `≡ 1` — the weighting vanishes and plain GRPO is recovered |

`p_c = 1/N · Σᵢ 1[rᵢ = 1]` for a binary verifier, and `p_c = mean(r)` for a continuous
reward in `[0,1]`. The Bernoulli induced by that mean is the right object either way,
so no special case is needed downstream.

## Two corrections to the source, both documented at the call site

1. **The printed `D_ME` is not a KL divergence.** The report prints
   `p·log(p/(1−p)) + (1−p)·log(p0/(1−p0))`; the first denominator should be `p0`. As
   printed the expression is asymmetric and is *negative* at `p = 0.1`, so it cannot
   be the "distance from the ideal maximum-entropy state" its own prose describes.
   `max_entropy_deviation` implements the Bernoulli KL.
2. **The low-probability-trace term is not implemented, because the paper does not
   specify it.** The report also credits MGPO with "incentiviz[ing] the increased
   generation probability of low-probability yet correct reasoning traces". No
   formula for that appears anywhere. Only the advantage reweighting — which *is*
   specified — exists here, and no such behaviour should be attributed to this code.

## Gotchas

- **`spectrum_gain` cannot detect a degenerate selection.** A `select_specialists`
  that returned the best-overall checkpoint for every subdomain reports a gain of
  exactly `0.0`, identical to a legitimate "one checkpoint dominates" result. Check
  `checkpoint_index` against the per-column maxima too.
- **The batch mean of `MGPOObjective` is `0.0` for every `λ`, always.** A
  group-relative advantage sums to zero within a group, and the weight is constant
  across a group, so the weighted sum does too. The value is structurally incapable
  of showing the weighting; the **gradient** is where it acts. Log the per-response
  vector (`loss.call(...)`) or the weights, never the mean.
- **`__call__` reduces, `call` does not.** `loss(packed, new)` returns a scalar;
  `loss.call(packed, new)` returns the `(B,)` per-response vector that `sample_weight`
  selects rows from.
- **The advantage is per RESPONSE, not per ROLLOUT.** `(B, G)` from
  `mgpo_advantages` is one value per rollout and has no meaning against one
  response's token axis. Flatten: `old_log_probs.reshape(-1, T)` with
  `advantage.reshape(-1)`. The error message names this case.
- **`sample_weight` must be `(B,)`, not `(B, T)`.** A token-shaped weight is folded
  into the mean Keras already takes over responses, double-counting the padding
  mask. Pass a `token_mask` through `mgpo_surrogate` instead.
- **Fusion never mutates and never aliases.** There is deliberately no `set_weights`
  wrapper. Uniform fusion of identical specialists is the **bit-exact** identity, so
  `atol=0` comparisons against the original hold.
- **`k > n_samples` raises** rather than clamping; clamping would return `1.0` for
  every problem.
- **Quote one `pass@k` estimator.** The plug-in (`pass@n`) is an upper bound for
  `k < n`; quoting both is a false comparison, not belt-and-braces.

## See also

- `AGENTS.md` in this directory — module map and authoring rules.
- `../AGENTS.md` — the `sample_weight`/`call()` contract and the `DEFAULT_*` rule.
- `research/2026_spectrum_to_signal.md` — the paper, the two corrections, and what is
  untested here.
- Tests: `tests/test_optimization/test_ssp/`, `tests/test_metrics/test_pass_at_k.py`.
