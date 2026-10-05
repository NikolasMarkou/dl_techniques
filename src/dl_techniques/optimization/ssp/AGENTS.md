# SSP Subpackage

`dl_techniques.optimization.ssp` — the Spectrum-to-Signal Principle: two phases, four
operations, one config.

## Layout

| Module | Holds |
|---|---|
| `spectrum.py` | `spectrum_profile`, `select_specialists`, `spectrum_sampling_weights` — the Spectrum phase's selection |
| `fusion.py` | `fuse_specialists`, `fusion_weights_from_scores`, `uniform_weights`, `score_softmax_weights`, `greedy_soup` — consolidating specialists into one model |
| `signal.py` | `binary_entropy`, `max_entropy_deviation`, `max_entropy_weight`, `group_success_rate`, `group_relative_advantages`, `mgpo_advantages`, `mgpo_surrogate`, `pack_mgpo_targets` / `unpack_mgpo_targets`, `MGPOObjective` |
| `config.py` | `SSPStrategyConfig`, `SSPStrategy`, `SSPType`, `ssp_builder` |
| `README.md` | The principle, its closed forms, and its gotchas |

`pass@k` is **not** here. It lives in `dl_techniques.metrics.pass_at_k` and is imported
by `spectrum.py`, because it is a metric and one implementation of it is the point.
Do not add a second `pass_at_k` under this package.

## Module boundaries that are load-bearing

**`spectrum.py` is numpy; `signal.py` is `keras.ops`.** `spectrum.py` holds arrays of
probe outcomes and checkpoint scores, and runs outside any graph. `signal.py`'s weight
has to work inside a traced `tf.function` for the RL use, so it is backend-agnostic.
`spectrum_sampling_weights` is the single seam between them, and it calls
`signal.max_entropy_weight` **directly** rather than reimplementing the weighting — the
"uncertainty is valuable" criterion must have one implementation, or the checkpoint-side
and data-side criteria drift apart silently. A test pins the two to float32 precision
on the same inputs.

**`fusion.py` is numpy and mutates nothing.** Every function returns new arrays; there
is deliberately no `set_weights` convenience wrapper, because a merge that writes into a
live model is unrecoverable the moment two inputs alias it.

**`config.py` orchestrates, it does not run.** There is no SSP trainer in this
repository. The Signal phase needs a rollout sampler and a verifier, and neither exists
here, so `SSPStrategy` is a set of configured functions rather than a pipeline.

## Structural properties — read before "fixing" one of them

Each is pinned by a test; each looks like a bug the first time you hit it.

1. **`enable=False` is a strict no-op** on all four operations: uniform sampling weights,
   uniform fusion weights, `pass@1` as the headline `pass@k`, and the plain
   group-relative advantage. It is the A/B control the principle is measured against,
   so a switch that merely softened the weighting would make every comparison
   meaningless.

2. **`spectrum_gain == 0.0` does not mean "nothing to gain".** A
   `select_specialists` that ignored subdomains and returned the best-overall checkpoint
   for every column also reports exactly `0.0`, because the best-overall row's mean is
   the mean of its own per-column values. Check `checkpoint_index` against the
   per-column maxima too; the count of subdomains where the selection is a true column
   argmax is the discriminating statistic.

3. **The batch mean of `MGPOObjective` is `0.0` for every `lambda`.** A group-relative
   advantage sums to zero within a group and the max-entropy weight is constant across
   a group, so the weighted sum does too. The value is structurally incapable of
   showing the weighting; the **gradient** is where it acts. Log the per-response vector
   (`loss.call(...)`) or the weights, never the mean loss.

4. **Uniform fusion of identical specialists is the bit-exact identity** (a short
   circuit in `fuse_specialists`). Averaging N copies of an array in float64 rounds, so
   the naive result lands *within* an ulp of the input rather than on it, and every
   downstream `atol=0` comparison would inherit the discrepancy.

5. **float64 accumulation bounds, but does not remove, order dependence.** Float
   addition is not associative in any precision. MEASURED: reordering five specialists
   moves the fused weights by at most `1e-6` relative. That bound — not bit-equality —
   is the guarantee, and claiming more would be false.

## Authoring rules

- `constants.py` (the parent package) is the single source of truth for `DEFAULT_*`.
  `config.py` imports its defaults and re-declares none;
  `test_config.py::test_every_default_matches_the_constants_module` re-derives all
  sixteen.
- New enumerated fields get their member list exported from the module that defines
  them (`FUSION_MODES`, `FUSION_WEIGHT_SCHEMES`, `SAMPLING_MODES`, `TOKEN_REDUCTIONS`,
  `ZERO_VARIANCE_POLICIES`) and validated through `_one_of`. A test asserts the
  dispatcher's member set equals the declared list.
- Numeric fields are validated at **construction**, so a bad recipe fails when it is
  parsed rather than three stages into a run.
- A string field is normalised (strip + lower) before its membership check.
- `keras.losses.Loss.call` takes `(y_true, y_pred)` in Keras 3 — **never** a third
  `sample_weight`. Validate anything about `sample_weight` in an `__call__` override.

## RED proofs

Every guard class named `TestTheGuardsActuallyGoRed` in
`tests/test_optimization/test_ssp/` injects the specific defect its file claims to
catch and asserts the oracle or the guard disagrees. The shared instruments live in
`ssp_oracle.py`, which carries **no `test_` prefix** so pytest does not collect it.

The oracles are written longhand and share no code with the implementation. That matters
most for the Bernoulli KL: it is evaluated from the *defining two-term sum*, not from
`ln 2 - H(p)`. Deriving the quantity under test from the identity it is supposed to
satisfy would make every identity check a tautology.

## Testing

`tests/test_optimization/test_ssp/` (300 tests) and
`tests/test_metrics/test_pass_at_k.py` (63). Scoped runs:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_optimization/test_ssp/ -q
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_metrics/test_pass_at_k.py -q
```

Do not run them in parallel with anything else — this suite has ordering-sensitive
failures under contention (`src/dl_techniques/AGENTS.md` § Testing).
