# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **TreeTransformer (`models/tree_transformer/`)** is structurally sound — save/load + gradient flow + MLM-wrapper integration all correct. Four real bugs fixed in plan_3c3ed037: B-1 fp16 NaN (dtype-aware mask sentinel `-1e4` under float16, plus fp32 cast on GroupAttention DP log/matmul/exp block); B-3 explicit `attention_mask` honored in dict input; B-4 `load_pretrained_weights` via `weight_transfer.load_weights_from_checkpoint` (Keras 3.8 `by_name=True` broken); B-5 `PRETRAINED_WEIGHTS={}` + `NotImplementedError` (no public checkpoints). Trainer `src/train/tree_transformer/{pretrain,finetune}.py` mirrors `bert/`. Anchor: `model.py:318` D-001. **Trainer config MUST pass `pad_token_id=config.pad_token_id` (tiktoken cl100k_base = 100266) to encoder — model default 0 is silent semantic bug.** Aligned to `bert/`/`resnet/` conventions in plan_0a5779e8: bare `create_tree_transformer(variant, ...)` factory added, `__init__.py` trimmed to 3 names (`TreeTransformer`, `create_tree_transformer`, `create_tree_transformer_with_head`; internal layer classes remain importable from `.model` for `nam/` consumers), and `from_variant(pretrained=True)` now raises `NotImplementedError` loudly instead of silently random-initializing (D-001 anchor at `model.py:1133`, narrowed try/except to `(IOError, OSError, ValueError)`).
- **TinyRecursiveModel (`models/tiny_recursive_model/`)** — save/load clean. B-3 Q-learn lookahead `training=False` + `keras.ops.stop_gradient` on `target_q`; B-5 inference halts on learned signal. `hrm_loss`/`HRMMetrics` API-compatible with TRM output schema. Anchor: `model.py:370` D-001 (plan_e6309bd5).
- **`keras.ops.expand_dims(axis=tuple)` works** on Keras 3.8 / TF 2.18 eager + `@tf.function` (B-1 false-positive in plan_e6309bd5).
- **DepthAnything** is now full-feature — real ViT encoder, DPTDecoder linear default + `upsample_factor`, weight-shared frozen teacher via `clone_model`, semi-sup `train_step` (FAL + L1 pseudo-label stop-gradient), on-step EMA via `TeacherEMACallback`, `from_pretrained_encoder(path)`, `StrongAugmentation` + dynamic cutmix.
- **Keras 3 / TF 2.18 idioms**: `keras.random.*` (NOT `keras.ops.random.*`); `keras.ops.*` for backend-agnostic ops; `@keras.saving.register_keras_serializable()` + `get_config()` round-trip; `dl_techniques.utils.logger` only.
- **Save/load on subclassed Models wrapping inner Models**: weights drop unless outer class overrides `save_own_variables` / `load_own_variables` (D-004).
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken** — use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay footgun**: never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **CLM training**: use `train.common.nlp.estimate_clm_steps_per_epoch`; `min_article_length=0` correct for packed pipelines.
- **Two-optimizer differential-LR**: register one with `super().compile`; apply second manually inside `train_step` via name-prefix variable routing (leading-component match).
- **`keras.ops.cond` traces BOTH branches under `tf.function`** — multiply-by-zero for compute-amount differences.
- **Frozen state in layers**: `add_weight(trainable=False)` or numpy-on-self — never plain tensors in `build()` (FuncGraph dead-tensor).
- **BERT (`models/bert/`)** aligned to resnet/tree_transformer template in plan_9357982a: `create_bert` bare-encoder factory added; `__init__.py` trimmed to 3-name surface `{BERT, create_bert, create_bert_with_head}` (drop `create_nlp_head` re-export); `_download_weights` raises `NotImplementedError`, `from_variant` try/except narrowed to `(IOError, OSError, ValueError)` (D-001 anchor at `bert.py:687`); docstring/README path fixed to `dl_techniques.layers.nlp_heads`. 28/28 pytest PASS, 0 fix attempts.
- **AccUNet** requires H,W divisible by 16; validation in `call()` raising `ValueError` (plan_bdb2c84d D-001/D-002).
- **`SegmentationWrapperLoss`** is canonical save/load-friendly segmentation loss; `compile=False` workaround removed (plan_17633038 D-002).

### Key Decisions
- **D-001 plan_3c3ed037 (TreeTransformer bundle)**: 4 model bugs + Pattern-3 trainer in one iteration — 5 new files / +950 LOC at the cost of 2 over file-budget; trainer depends on Step 5 re-exports and Step 2 attention_mask honoring, so splitting would force pinning to broken imports.
- **D-001 plan_e6309bd5 (TRM bundle)**: bug fixes + factory + tests + trainer in one plan — at cost of larger review surface; B-5 testable only with same harness as trainer eval path.
- **Pseudo-label loss**: plain L1 + `stop_gradient`, not `compute_loss` against synthetic mask (plan_54e6e303 D-002).
- **Encoder weight-loading**: keep `--pretrained-encoder-weights` + `--init-from` distinct (plan_54e6e303 D-003).
- **D-004 (save_own_variables override)**: canonical Keras-3 fix when `.keras` round-trip drops sub-Model weights.
- **D-003 (Keras-3 canonical train_step)**: `compute_loss(x,y,y_pred)` adds `self.losses` internally — no manual regularization addition.
- **D-005 (StrongAugmentation graph-mode safety)**: symbolic gate; `keras.random.*` not `keras.ops.random.*`.
- **CLM metrics architecture**: math in `dl_techniques/metrics/`; list in `train/common/nlp/build_clm_metrics()`; fresh instances each call.
- **`current_phase` / `_global_step` counters**: `add_weight(trainable=False, dtype="float32")` — int32 fails CPU/GPU device placement.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-06-03_5c8c6d19
### Index

| # | Finding | File | Key facts |
|---|---------|------|-----------|
| 1 | CCNets train folder current state | `findings/ccnets-current-state.md` | 6 scripts + `CLAUDE.md`, no `README.md`; all 6 violate `train_<model>.py`; no argparse in 4; architectures defined inline; cross-script imports between train files |
| 2 | Canonical train-folder conventions | `findings/train-conventions.md` | `train_<task>.py` naming; `README.md` (not subfolder `CLAUDE.md`); models imported from `dl_techniques.models.*`; `main()`+argparse via `create_base_argument_parser`; `setup_gpu(args.gpu)`/`set_seeds(args.seed)`; 12-point checklist |
| 3 | CCNets model package vs train scripts | `findings/ccnets-models.md` | `models/ccnets/` is framework-only (orchestrator/trainer/config/losses), zero architectures; 17 classes in `mnist.py`, 7 in `cifar100.py`, text classes in `text_sentiment.py`; factories `create_mnist_ccnet`/`create_cifar100_ccnet` live in train scripts |

### Key Constraints

### HARD
- CCNet framework contracts (PRINCIPLES_CCNETS.md P1-P11): three-network design `explainer(x)->(mu,log_var)`, `reasoner(x,e)->y`, `producer(y,e)->x_hat`; differentiable label projection `Dense(use_bias=False)` not `Embedding`; variational Explainer. Must be preserved byte-for-byte across any move.
- `src/train/CLAUDE.md:42` — scripts must be named `train_<model>.py`, never bare nouns (`mnist.py` shadows package names).
- Model architecture classes must NOT live in train scripts — belong in `dl_techniques/models/<pkg>/`.
- `CCNetTrainer` owns a CUSTOM training loop (manual GradientTapes, per-network optimizers, KL annealing) — `model.fit()` is NOT used, so `train.common.create_callbacks()` cannot wrap it directly. This is an intrinsic deviation, not neglect.
- `setup_gpu(args.gpu)` must receive the parsed `--gpu` arg.

### SOFT
- Provide `README.md` (not subfolder `CLAUDE.md`) matching the convex/cliffordnet README structure.
- Provide `main()` + argparse with at least `--gpu/--epochs/--batch-size`.
- Call `set_seeds(args.seed)` consistently (currently only 2 of 6 scripts).
- Use `save_config_json` / `validate_model_loading` post-training where feasible.

### GHOST
- Subfolder-level `CLAUDE.md` — only `src/train/CLAUDE.md` should be the AI-instruction layer; no other train subfolder has its own `CLAUDE.md`. The ccnets `CLAUDE.md` content is mostly findings/results that belong in a README.
- `dynamic_weighting` flag in `CCNetConfig` is deprecated (`base.py:76`), stays `False`.

### Structural insight (drives the plan)

Moving architectures into `dl_techniques/models/ccnets/` SOLVES the cross-script-import fragility: today `cifar100.py`<-`mnist.py`, `cifar100_hybrid.py`<-`cifar100.py`, `baseline_comparison.py`/`latent_sweep.py`<-`mnist.py`. Once classes live in the model package, every train script imports cleanly from `dl_techniques.models.ccnets.*` and renaming becomes safe. Therefore: **architecture migration must precede script renaming.**

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-03_bc986e52
### Index

| # | Finding | File |
|---|---------|------|
| F1 | Authoritative WeightWatcher reference (verbatim formulas) | findings/ww-authoritative-reference.md |
| F2 | Core math: MP edge, Tracy-Widom, ERG, naming, phases vs WW (orchestrator source-read) | findings/core-math-vs-ww.md |
| F3 | ERG/detX boundary + alpha MLE vs WW (explorer) | findings/erg-alpha-vs-ww.md |
| F4 | Metric names/values + helpers vs WW (explorer) | findings/metrics-naming-vs-ww.md |

### Verified divergences from authoritative WeightWatcher (ranked)

**HIGH — correctness bugs revealed by ground-truth source:**
- **R1: MP edge uses `(1+√Q)²`, WW uses `(1+1/√Q)²`** (Q=N/M, N=larger). `spectral_metrics.py:504-505`. Factor-((1+√Q)/(1+1/√Q))² error for rectangular layers (Q=4 → 9σ² vs 2.25σ²). Both λ+ and λ−. **Pre-existing** (predates prior plan). Two-source confirmed (orchestrator + WW `calc_lambda_plus/minus`).
- **R2: ERG boundary is computed by a BROKEN mechanism.** `compute_erg_condition` (`:378-383`) builds `cumulative_log = cumsum(log(ascending evals))` — a NON-MONOTONIC array — then `searchsorted(cumulative_log, 0.0)` (searchsorted requires sorted input → result undefined/wrong). WW's authoritative algorithm is the DESCENDING product loop `largest idx where prod(evals[idx:])<1.0`. The analyzer ALREADY has the correct loop in `compute_detX_constraint` (`:~1043`) but `compute_erg_condition` does not use it. Net: the `erg_lambda_min` (and thus Δλ_min, which prior-plan only sign-fixed) is computed on a broken boundary. Two-source confirmed.
- **R3: Tracy-Widom threshold structure is not authoritative.** WW: `bulk_max + √[(1/√Q)·bulk_max^(2/3)·M^(-2/3)]`, no c_TW. Analyzer (`:512-513`, prior D-E): `mp_lambda_plus + c_TW·σ²·M^(-2/3)`. Prior D-E got base=M/exp=-2/3 but not the `(1/√Q)`, the `bulk_max^(2/3)`, the overall √, or the no-c_TW. Couples to R1 (uses the edge).

**MED — naming/coverage divergences (value often already correct):**
- **R4: α̂ naming inverted vs WW.** WW canonical = `alpha_weighted` (= α·log₁₀λ_max); WW has NO `alpha_hat` and NO `/N` variant. Prior plan (D-F) made `alpha_hat` canonical and labeled `alpha_weighted` "deprecated" — backwards vs WW. VALUE is correct (un-normalized) and matches WW; only the canonical-name/deprecation diverges. `alpha_hat_normalized` (/N) has zero WW basis. `spectral_metrics.py:~639`, `constants.py:84-85`.
- **R5: MetricNames coverage gaps.** `MP_SOFTRANK` appears renamed to `WW_SOFTRANK` (dead placeholder, no impl); `MATRIX_RANK` absent; `NORM`/`SPECTRAL_NORM` bare literals not in MetricNames. `constants.py`.

**LOW / enhancement:**
- **R6: mp_softrank metric missing** (WW: λ+/λ_max). Value-add (prior plan deferred as D-J).
- **R7: small-N (<20) bias-corrected alpha branch missing** (WW: `1+(n-1)/s`, `J=D_ks−0.868/√n`). Standard MLE applied regardless → upward-biased α on thin tails.
- **R8: phase "ideal" band + "over-regularized" term are SETOL-only, not WW.** WW: plain over-trained(<2)/under-trained(>6), term "over-trained". Prior plan D-C ideal band [2.0,2.1) and D-D "over-regularization" are SETOL-paper choices. Surface: keep SETOL framing or revert toward WW under the "WW authoritative" instruction.
- **R9: NORM-layer analysis gap** — recognized but `get_layer_weights_and_bias` returns nothing for NORM (`spectral_utils.py:~141`). WW supports NORM.
- (No-action) matrix_entropy EPSILON handling differs negligibly; conv reshape `(H·W·C_in, C_out)` vs WW transpose — SVD-invariant.

### Confirmed-correct (authoritative agreement — do NOT touch):
- `rescale_eigenvalues` byte-identical to WW (re-confirms prior-plan Correction C1 ghost). 
- Alpha MLE `1+n/(Σlog−n·log_xmin)` + KS-argmin xmin == WW.
- `sigma=(α−1)/√N_tail` == WW. `stable_rank=Σλ/maxλ` == WW. `matrix_rank` tol == WW. norm/log_norm/log_alpha_norm == WW.
- Δλ_min units (xmin·wscale² vs rescaled boundary) consistent; sign (prior fix) matches SETOL §7.3 — but the boundary it subtracts is R2-broken.

### Key Constraints

HARD:
- Authoritative source = Martin's WeightWatcher (user-declared). Where SETOL.md and WW conflict on MECHANISM, WW wins (R1,R2,R3). Where they conflict on FRAMING/terminology (R8) it is a user choice.
- `compute_detX_constraint` already exists and is WW-correct → R2 fix is reuse, not new code (DRY).
- Tests: `tests/test_analyzer/test_spectral_metrics.py` only (full suite ~1.5h forbidden). Logger-only, Keras 3, MPLBACKEND=Agg.
- Some divergences are PRE-EXISTING bugs (R1, R2) not introduced by the prior plan — fixing them is in-scope for "reconcile to authoritative" but changes long-standing behavior; trap/ERG tests may encode the old (wrong) numbers as contracts.

SOFT:
- R4 naming realignment (alpha_weighted canonical) is reversible and low-risk but touches the metric just changed last plan — confirm intent.
- R6/R7/R9 are additions (scope-expanding); R8 is a framing reversal.

GHOST:
- "Prior-plan D-E fully fixed Tracy-Widom" — false; R3 shows it was partial. "Prior-plan Δλ_min fix made the ERG diagnostic correct" — false; R2 shows the boundary is broken underneath.

### Corrections
- (none yet this plan)

## plan_2026-06-03_9e82787d
### Index

| # | Finding | File | Detail |
|---|---------|------|--------|
| F1 | Core spectral math audit (spectral_metrics.py, spectral_utils.py) vs SETOL | findings/spectral-core-math.md | Alpha/Clauset COMPLIANT; Δλ_min abs() bug; TW exponent; α̂ naming; R-transform missing |
| F2 | Orchestration + visualization audit (spectral_analyzer.py, spectral_visualizer.py) | findings/spectral-orchestration-viz.md | Funnel plot missing; α̂ not in summary; ideal band [2,2.5); MP overlay absent |
| F3 | Package integration audit (model_analyzer, config, constants, data_types, utils) | findings/package-integration.md | α<2 mislabeled "memorization"; α̂/α_weighted naming; summary metric omissions |

### Verified Discrepancies (source-read by orchestrator, ranked)

Severity HIGH:
- **D-A: `Δλ_min` sign destroyed** — `spectral_metrics.py:383` `delta_lambda_min = float(abs(...))`. SETOL §7.3 requires SIGNED value (<0 = over-regularized). The `abs()` makes the over-regularization diagnostic impossible. Unit handling around it (`xmin*wscale²` vs rescaled `erg_lambda_min`) is otherwise correct. **HARD constraint: fix is a 1-token removal of `abs()`.**
- **D-B: α̂ (AlphaHat) absent from model-level summary** — `constants.py:122-128` `SPECTRAL_DEFAULT_SUMMARY_METRICS` lists ALPHA + LOG_SPECTRAL_NORM but omits ALPHA_HAT. SETOL §2.4/§13 make ⟨α̂⟩ the primary model-quality metric. Computed per-layer but never aggregated.

Severity MEDIUM:
- **D-C: "ideal" phase band is [2.0, 2.5)** — `spectral_metrics.py:417-418`. SETOL §3.2/§13: Ideal = critical point α≈2, not a 0.5-wide band. classify_learning_phase thresholds (2.5/4.0/6.0) also diverge from SETOL's clean HT band (2,6) AND from this file's own docstring (which says "2.0<α<4.0 Good"). Internally inconsistent.
- **D-D: α<2 mislabeled "overfitting/memorization"** — `model_analyzer.py:39-40` docstring. SETOL §7.2 defines α<2 as Over-Regularization (glassy, compensatory), mechanistically distinct from memorization. Interpretation error in docs.
- **D-E: Tracy-Widom exponent** — `spectral_metrics.py:447,493` uses `N^(-1/3)`. SETOL §2.3 states Δ_TW ~ O(M^(-2/3)). Need to confirm code's `N`/`M` orientation before claiming exact exponent error; at minimum the magnitude differs → trap detection sensitivity off.
- **D-F: α̂ field naming/convention confusion** — `spectral_metrics.py:615-625`. Two fields: `alpha_weighted`=α·log₁₀(σ²_max) (WeightWatcher convention, matches §8/§10.5 validation refs); `alpha_hat`=α·log₁₀(σ²_max/N) (SETOL theory X=(1/N)WᵀW). Per §10.2 NEITHER is strictly wrong, but the naming is misleading and which one `MetricNames.ALPHA_HAT` exports matters for downstream consumers.
- **D-G: Funnel diagnostic (α, Δλ_min) plot missing** — `spectral_visualizer.py`. SETOL §8.2/§10.4 central diagnostic. `erg_delta_lambda_min` computed+stored but never plotted. (Depends on D-A to be meaningful.)
- **D-H: MP bulk overlay absent from per-layer ESD plot** — `spectral_visualizer.py:338-361`. MP envelope (λ-, λ+) only drawn in trap overlay (gated on randomize), not on the standard per-layer ESD diagnostic.

Severity LOW / enhancement:
- **D-I: Free cumulants / R-transform / Layer Quality Q̄² entirely absent** — SETOL §5.4, §6. Large theory chunk; "computational R-transform Layer Quality" listed in §10.5 as a WW capability. Enhancement, not a correctness bug.
- **D-J: MP SoftRank R_MP missing** — SETOL §2.4 metrics table. `stable_rank` exists but is a different quantity.
- **D-K: Missing universality classes** — RandomLike, Bulk+Spikes, Rank-Collapse not in `classify_learning_phase`. SETOL §2.2 6-class table.
- **D-L: α<2 recommendation says "early stopping/regularization"** — `spectral_analyzer.py:342`. SETOL §13 action table says "Reduce LR; check for Correlation Traps".

### Key Constraints

HARD:
- Code already COMPLIANT and must stay so: Clauset MLE α (joint xmin/KS), Conv2D reshape (H·W·C_in, C_out), BN/Dropout skipping, bias exclusion, D_KS, correlation-trap randomize protocol, ERG trace-normalization (wscale) is the legit §10.2 correction.
- `Δλ_min` fix is the single highest-value, lowest-risk change (remove `abs()`).
- Tests live in `tests/test_analyzer/` — must scope pytest there (full suite ~1.5h forbidden as regression check).
- No `print`; use `dl_techniques.utils.logger`. Keras 3 idioms. `MPLBACKEND=Agg` for any plot-generating code.

SOFT:
- WeightWatcher-convention vs SETOL-theory α̂ normalization is a project-preference choice (§10.2 sanctions both) — needs user intent, not unilateral change.
- Whether to ADD missing theory (R-transform Q̄², missing phases, funnel plot, MP overlay) vs only FIX existing-but-wrong code is a scope decision for the user.

GHOST:
- "ERG must target det=1 in the rescaling step" — NOT a real constraint; trace-normalization then det-check on rescaled evals is the correct WW/SETOL pattern. (See Correction C1.)

### Corrections

- **[CORRECTED iter-0] C1**: findings/spectral-core-math.md ranks "ERG rescaling targets mean=1 not det=1" as severity-2 DISCREPANCY. Downgraded after orchestrator source-read of `rescale_eigenvalues` (spectral_metrics.py:984-996) + `compute_erg_condition` (340-391): the Σλ→N trace-normalization IS the `wscale` correction SETOL §10.2 explicitly sanctions ("ERG calculation applies wscale correction internally"); det=1 / Σln≈0 is then evaluated on the rescaled eigenvalues. This is the canonical WeightWatcher pattern, NOT a discrepancy. The only real ERG-path bug is D-A (the `abs()` on Δλ_min).

## plan_2026-06-03_bf1e592d
### Index
| # | Finding | File | Key refs |
|---|---------|------|----------|
| F1 | ConvNeXt train-script structure + exact `stochastic_mode` plumbing points | `findings/train-scripts.md` | v1:102, v2:106, v2_mae:21-26; factory `**kwargs` chain confirmed |
| F2 | Experiment-harness / `compare_runs` reuse patterns + CIFAR-10 choice | `findings/experiment-harness.md` | sweep.py, compare_runs.py:198, callbacks.py:85 (CSVLogger) |
| F3 | Pre-PLAN design decisions (DN-1..DN-5): direct-kwarg plumbing, seeding, subprocess driver, snapshot-diff | `findings/design-notes.md` | DN-1..DN-5 |

### Key Constraints
- [HARD] `stochastic_mode` ∈ {`depth`,`gradient`}; constructor raises `ValueError` otherwise (`convnext_v1.py:182`, `convnext_v2.py:199`) → CLI `choices=['depth','gradient']`.
- [HARD] V2-MAE `create_convnext_encoder` (`train_convnext_v2_mae.py:21`) has a fixed signature, no `**kwargs` → must add `stochastic_mode` param explicitly + thread it.
- [HARD] Subprocess cells must hard-set `CUDA_VISIBLE_DEVICES` + `MPLBACKEND=Agg` (not setdefault); serial only (no parallel GPU); results under repo-root `results/`.
- [HARD] Train scripts must keep `train_<model>.py` naming (shadowing `train` package breaks imports).
- [SOFT] Existing scripts already emit `training_log.csv` via `create_callbacks` → `compare_runs` works directly on their dirs; no custom report writer needed.
- [GHOST] No `--stochastic-mode` or `--seed` flag exists in any convnext script; both are net-new, convnext-local additions (do NOT widen the shared base parser).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
