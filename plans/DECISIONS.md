# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed from 628 lines on 2026-05-13 (refreshed after plan_2026-05-13_a1c9a52d close — merged layers/ntm/ into layers/memory/, deleted ntm package; no new active constraints introduced, but note new constraint below). Read full content below for details on each plan's decisions.*

### Active Constraints (anchored, do-not-break)
- **3-name encoder public surface** (`<Model>`, `create_<model>`, `create_<model>_with_head`) — locked in tree_transformer/bert/cliffordnet; gpt2 is 2-name (LM head intrinsic); cliffordnet now hosts 4+3 names (multiple model classes).
- **`_download_weights` raises `NotImplementedError`** + **`from_variant` narrow `except (IOError, OSError, ValueError)`** — no silent random-init fallback. Anchored in tree_transformer, bert, gpt2, vit, cliffordnet, cliffordnet/embedding_unet.
- **`pad_token_id=<tokenizer_pad>` must be wired from trainer config to encoder ctor** (silent semantic bug otherwise). tiktoken cl100k_base pad = 100266; gpt2 enc pad differs.
- **Output dict key `"logits"`** + **`prepare_dict_keyed_compile(model, output_key="logits")`** required for every Pattern-3 CLM trainer before `model.compile`.
- **`build_clm_metrics(encoding_name, ignore_index)`** — required metric floor for every CLM trainer (replaces bare `["accuracy"]`).
- **`SegmentationWrapperLoss`** is the canonical save/load-friendly seg loss; no more `compile=False` workarounds in trainers.
- **`save_own_variables`/`load_own_variables`** on outer Model classes wrapping inner Models (DepthAnything pattern) — required for `.keras` round-trip when sub-Model weights would otherwise re-initialize.
- **memory_bank dual-optimizer**: register one optimizer with `super().compile`, apply second manually; prefix split via `name.split('/')[0].startswith(p)` (leading-component, NOT substring).
- **U-Net `.keras` round-trip tolerance is atol=1e-4** (not 1e-5) on fp32 GPU due to reduction-order noise. Applies to lmunet + embedding_unet + AccUNet.
- **`dl_techniques.layers.ntm` no longer exists** — all NTM / MANN / SOM imports go through `dl_techniques.layers.memory` (plan_2026-05-13_a1c9a52d D-002). Top-level (`NTMCell`, `NTMConfig`, `create_ntm`, `MannLayer`, `SOM2dLayer`, `SOMLayer`, `SoftSOMLayer`) and deep-submodule paths both supported.

### Failed Approaches (do NOT retry)
- "Modify `lmunet.py` in place with a `causal` flag" — REJECTED (plan_632605aa D-001); also "modify Clifford block classes with `causal` flag" — REJECTED. Sibling-stack additive file is correct.
- `keras.ops.cond` for runtime branch skipping inside `call()` — both branches trace under TF; use multiply-by-zero (plan_0f39a086 D-003).
- Mocking the database in tests / using `compile=False` to dodge a custom-loss round-trip bug — both are workarounds, not fixes (LESSONS).
- SimCSE / contrastive sentence-pair training as iter-1 for an encoder package — explicit deferral pattern (plan_632605aa D-003; plan_146ae899 — staged plans only).
- LR sweep on "smooth-train + cliff-val + sub-random val" signature — that fingerprint = data-pipeline divergence, NOT hparams (plan_f2d29729 D-006/D-007).

### Decision-Anchor Conventions
- Format: `# DECISION plan_<id>/D-NNN: <one-line>` at point of impact. Block, hash, double-dash variants supported. Unqualified `D-NNN` anchors from old plans are tolerated but WARN; new code MUST use qualified form.
- 5 triggers: failure-driven, non-obvious, rejected-alternative, constraint-workaround, 3-strike.
- Anchor at impact site (not at decision definition). One anchor per impact site, even if shared with sibling decision.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-06-03_5c8c6d19
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_5c8c6d19/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: All CCNet task architectures live inside train scripts with train-to-train cross-imports (cifar100←mnist, cifar100_hybrid←cifar100, baseline/latent_sweep←mnist); the model pkg is framework-only. Findings show moving architectures into the model pkg dissolves the import fragility.
**Decision**: Migrate all architectures + shared blocks + factories into `dl_techniques/models/ccnets/` (blocks.py + architectures/{mnist,cifar100,text}.py) BEFORE renaming train scripts to thin `train_/run_` wrappers.
**Trade-off**: Architecture-first ordering and a new `architectures/` namespace **at the cost of** exceeding the default 3-file complexity budget (~13 files added / 8 deleted).
**Reasoning**: Per findings.md structural insight, renaming before migration would leave broken cross-imports; migrating first makes renames safe. The `architectures/` sub-package clears the earned-abstraction rule (3 concrete task modules). Alternative (leave architectures in train scripts, only rename) rejected: it cannot eliminate the train-to-train imports that motivate the consolidation.

### D-002 | EXPLORE → PLAN | 2026-06-03
**Context**: Data-prep helpers (`prepare_mnist_data` etc.) are training-side; `run_baseline_comparison`/`run_latent_sweep` need both architecture (now in model pkg) and data-prep.
**Decision**: Factories (`create_*_ccnet`) move to the model pkg; data-prep helpers stay in `train_mnist.py`; the two `run_*` drivers import data-prep from `train.ccnets.train_mnist`.
**Trade-off**: One residual `run_* → train_mnist` train-to-train edge **at the cost of** not achieving literal zero train-to-train imports.
**Reasoning**: Data-prep is genuinely training-side and does not belong in the library; the single sibling-import edge matches the canonical multi-script pattern (convnext `run_stochastic_comparison.py`). Alternative (`train/ccnets/data.py`) adds a file for one shared helper with no second non-train consumer — fails use-before-reuse.

### D-003 | EXPLORE → PLAN | 2026-06-03
**Context**: USER-LOCKED #4 fixes the `CCNetTrainer` custom GradientTape loop + `EarlyStoppingCallback` in place; `create_callbacks()` cannot wrap a non-`fit()` loop.
**Decision**: Leave `CCNetTrainer` and `EarlyStoppingCallback` untouched; consolidation covers naming, docs, architecture placement, and entry-point plumbing only.
**Trade-off**: Convention-parity on structure/docs/CLI **at the cost of** NOT adopting `create_callbacks`/`load_dataset` standardization.
**Reasoning**: The custom loop is intrinsic to the CCNet paradigm (SYSTEM.md custom train_step invariants); forcing `model.fit()` would be a rewrite, not a consolidation, and is explicitly out of scope per the user.

### D-004 | EXECUTE step-2 | 2026-06-03
**Context**: Migrating `create_mnist_ccnet` out of `train/ccnets/mnist.py`. The original factory took the train-side `ExperimentConfig` dataclass and read both `config.model` (architecture) and `config.training` (loss_fn / learning_rates / weights). The architecture classes' `from_config` rebuild a `ModelConfig`, so `ModelConfig` itself had to move into the model package; but importing the full `ExperimentConfig`/`TrainingConfig` would create a `models -> train` dependency, which is forbidden (A2/A4).
**Decision**: Move the architecture-only `ModelConfig` dataclass into `architectures/mnist.py` and change `create_mnist_ccnet` to take `model_config: ModelConfig` plus the CCNetConfig-relevant training fields as explicit keyword args (defaults identical to the old `TrainingConfig`). Keep the bias-free `Dense` label projection in `MNISTProducer` verbatim (anchored).
**Trade-off**: Factory signature changes from `(ExperimentConfig)` to `(ModelConfig, *, loss_fn=..., learning_rates=..., ...)` **at the cost of** the three current callers (`latent_sweep`, `baseline_comparison`, plus the train script) needing updated call sites in Step 6.
**Reasoning**: The model package must not import `train.*`; passing training params explicitly with matching defaults preserves behavior byte-for-byte while severing the train dependency (A4). Alternative (copy `ExperimentConfig` into the model pkg) rejected: drags training-only fields (epochs, augmentation, viz, early-stopping) into the library. The `Dense(use_bias=False)` anchor records why `Embedding` must never replace it (CCNet Invariant 1 / P4).
**Anchor-Refs**: `src/dl_techniques/models/ccnets/architectures/mnist.py:256-266`

### D-005 | EXECUTE step-3 | 2026-06-03
**Context**: Migrating `create_cifar100_ccnet` (from `cifar100.py`) and `create_hybrid_ccnet` (from `cifar100_hybrid.py`) into the model package. Both train-side factories took the train `ExperimentConfig` and were byte-for-byte identical except the final orchestrator class (`CCNetOrchestrator` vs `HybridCCNetOrchestrator`, which additionally passes `perceptual_model=reasoner` + `latent_weight`). Following D-004, the architecture-only `ModelConfig` moves with the module and the factory takes explicit training kwargs.
**Decision**: Provide a single decoupled `create_cifar100_ccnet(model_config, *, loss_fn='l1', ..., hybrid=False, latent_weight=LATENT_WEIGHT)`; when `hybrid=True` it returns the `HybridCCNetOrchestrator` (phi = the Reasoner). Defaults match the old cifar100 `TrainingConfig` (`loss_fn='l1'`, lr 3e-4 each, explainer kl 1e-3, reasoner reconstruction 0.1, producer 1.0/1.0). `HybridCCNetOrchestrator` moves into `architectures/cifar100.py` since it orchestrates the cifar100 nets.
**Trade-off**: One factory with a `hybrid` flag **at the cost of** not preserving the two separate factory names; the two Step-6 train wrappers (`cifar100`, `cifar100_hybrid`) call the same function with/without `hybrid=True`.
**Reasoning**: The two factories duplicated every line except the orchestrator class — reuse-before-write (DRY) favors one parametrized factory over copy-paste. The model package must not import `train.*` (A4); explicit kwargs with matching defaults preserve behavior. Alternative (two factory functions) rejected: it re-introduces the duplication the migration is meant to remove. The bias-free `Dense` label projection in `Cifar100Producer` is preserved verbatim (CCNet Invariant 1) with an inline note rather than a `# DECISION` anchor (the D-004 anchor in mnist.py already records the full rationale; no second anchor needed).

### D-006 | EXECUTE step-4 | 2026-06-03
**Context**: Migrating `create_sentiment_ccnet` (from `train/ccnets/text_sentiment.py`) into the model package. The original `create_sentiment_ccnet(config: ExperimentConfig)` read `config.model` (architecture) and `config.training` (learning_rates / gradient_clip_norm / explainer_weights / reasoner_weights / producer_weights), and the network classes' `from_config` rebuild a text-specific `ModelConfig`. Following D-004/D-005, importing the train-side `ExperimentConfig`/`TrainingConfig` would create a forbidden `models -> train` dependency (A2/A4). The text task differs from the image tasks: its `ModelConfig` is text-specific (`vocab_size`, `max_len`, `producer_type`, transformer dims), the orchestrator chosen at build time is token-space (`TextCCNetOrchestrator` for non-AR, `ARTextCCNetOrchestrator` for AR), and the AR/non-AR Producer split is selected by `model_config.producer_type` — not a separate `hybrid=` flag as in D-005.
**Decision**: Move the text-specific architecture-only `ModelConfig` into `architectures/text.py` and rename/decouple the factory to `create_text_ccnet(model_config: ModelConfig, *, learning_rates=None, gradient_clip_norm=1.0, explainer_weights=None, reasoner_weights=None, producer_weights=None)`. Each `None` default is replaced inside the body with the exact dict literal from the old `TrainingConfig` (lrs 3e-4 each; explainer inf 1.0/gen 1.0/kl 1e-3; reasoner inf 1.0/recon 0.1; producer gen 1.0/recon 1.0). Both Producer variants (`SentimentProducer` non-AR, `ARSentimentProducer` AR) and both orchestrators (`TextCCNetOrchestrator`, `ARTextCCNetOrchestrator`) are migrated verbatim; the AR/non-AR branch stays driven by `model_config.producer_type` exactly as in the source.
**Trade-off**: Factory signature changes from `(ExperimentConfig)` to `(ModelConfig, *, ...)` and is renamed `create_sentiment_ccnet` -> `create_text_ccnet` **at the cost of** the Step-6 text train wrapper needing an updated call site (build `ModelConfig` from its `ExperimentConfig.model`, pass `config.training.*` as kwargs).
**Reasoning**: The model package must not import `train.*` (A4); explicit kwargs with matching defaults preserve behavior byte-for-byte while severing the dependency. The `reasoner_weights` reconstruction default is `0.1` (text-specific, NOT the framework's `1.0`) — this is a deliberate task tuning preserved exactly. Data loading, decoding, plotting and evaluation helpers (`prepare_imdb_data`, `build_decoder`, `evaluate_and_report`, `plot_*`) were NOT migrated — they are training-side and stay in the (Step-6) train script. The bias-free `Dense` label projection in both producers is preserved verbatim (CCNet Invariant 1) with inline P4 notes; the D-004 anchor in mnist.py already records the full rationale, so no new `# DECISION` anchor is placed here. Alternative (copy `ExperimentConfig` into the model pkg) rejected for the same reason as D-004: it drags training-only fields into the library.

### D-007 | EXECUTE step-6 | 2026-06-03
**Context**: Step 6 rewrites the six train scripts as thin wrappers. Two cross-script edges remain after architecture migration: (1) `run_baseline_comparison`/`run_latent_sweep` need MNIST data-prep (`prepare_mnist_data`) which is training-side and lives in `train_mnist.py` (the D-002 edge); (2) `train_cifar100_hybrid` needs CIFAR-100 data prep + eval + plotting + the training config dataclasses, all of which are training-side and live in `train_cifar100.py`. Duplicating the CIFAR-100 data/eval helpers into the hybrid script would copy ~120 lines verbatim and violate reuse-before-write.
**Decision**: Sanction data-prep train-to-train edges of the same class as D-002: `run_baseline_comparison`/`run_latent_sweep` import data-prep (and the training config dataclasses) from `train.ccnets.train_mnist`; `train_cifar100_hybrid` imports `prepare_cifar100`/`evaluate`/`plot_history`/`ExperimentConfig` from `train.ccnets.train_cifar100`. ALL architecture (networks + `create_*_ccnet` factories + the `HybridCCNetOrchestrator`) is imported from `dl_techniques.models.ccnets`, never from a train script. Result: zero architecture train-to-train edges; only data-prep/training-config sibling imports remain.
**Trade-off**: Two sibling train-to-train data-prep edges (`run_* → train_mnist`, `train_cifar100_hybrid → train_cifar100`) **at the cost of** not achieving literal zero train-to-train imports.
**Reasoning**: This extends D-002's rationale: data prep, evaluation, and training-side config are genuinely training-side and do not belong in the library (A4 forbids `models → train`). Both edges match the canonical multi-script sibling-import pattern (convnext `run_stochastic_comparison.py`). The architecture edges — the ones that motivated the consolidation and were forbidden — are fully eliminated. Alternative (a shared `train/ccnets/data.py`) rejected: same use-before-reuse failure noted in D-002 (one helper, no second non-train consumer per dataset).

### D-008 | REFLECT iter-1 | 2026-06-03
**Context**: All 8 steps complete + independently verified by ip-verifier. 8/9 success criteria fully green. SC8 (full ccnets suite) shows 1 failing test, `test_orchestrator.py::TestTrainStep::test_training_reduces_total_error`. Full `validate-plan.mjs` reports 14 ERRORs + 57 WARNs.
**Decision**: Route to CLOSE. The SC8 failure and ALL 14 validate-plan ERRORs are pre-existing debt OUTSIDE this plan's changed files, not regressions.
**Trade-off**: Accept "8/9 SC + non-zero validator errors" **at the cost of** not having a literally all-green suite/validator — because forcing those green means fixing unrelated pre-existing debt (an unseeded flaky test + orphaned anchors from sliding-window-trimmed prior plans) that this plan did not create.
**Reasoning / evidence**:
- SC8 flaky test: `git diff 0c95ace6..HEAD -- tests/test_models/test_ccnets/test_orchestrator.py` is EMPTY (file untouched by this plan); last commit touching it (e5ca9c16) predates the plan; isolation reruns 2/3 fail (no random seed, 15-step convergence assertion). Pre-existing flaky, not a regression. New `test_architectures.py` = 28/28 green (SC7 round-trips included).
- 14 validate-plan ERRORs are all `anchor-orphan`/`anchor-unknown-plan` in OTHER files from OTHER plans (lighthouse_attention, lpips_loss, depth_anything, lewm, gpt2, nam, rms_variants, bdd100k, token_superposition, train_benchmark) — none in ccnets files. Repo-wide anchor debt, not introduced here.
- WARNs: `anchor-unqualified` (other plans' bare D-NNN colliding with our numbering — false positives), `anchor-refs` missing on our D-001..D-007 (cosmetic — anchors are in code, refs line optional), `changelog-malformed` op format (cosmetic), `presentation-contract-unlogged` (PC-PLAN WAS emitted to user; just no token logged in state.md).
**6 Simplification Checks**:
1. Essential vs accidental: `architectures/` subpackage is essential (3 task modules + framework separation). PASS.
2. Smaller? Net -1024 train-side LOC; D-005 merged duplicate hybrid factory; D-007 avoided ~120 LOC dup. Already minimized. PASS.
3. Junior-dev test: thin wrappers importing a documented model package >> 1112-line train scripts. PASS.
4. DRY: duplication reduced, not added. PASS.
5. YAGNI: no speculative abstractions; `hybrid` flag has 2 real call sites. PASS.
6. Net lines: strongly net-negative train-side; model-pkg gains are relocations. PASS.
No simplification blockers.
**Anchor-Refs**: (REFLECT summary; no new code anchor)

## plan_2026-06-03_bc986e52
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_bc986e52/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: Findings R1-R9 (findings.md) measured nine divergences of `analyzer/spectral_metrics.py`/`constants.py`/`spectral_utils.py` from Charles Martin's WeightWatcher source (findings/ww-authoritative-reference.md, GROUND TRUTH). Some are pre-existing correctness bugs (R1 MP edge, R2 ERG boundary), some are naming/coverage (R4/R5/R6), some are framing (R8). The prior plan (plan_2026-06-03_9e82787d) made several SETOL-paper choices that conflict with WW.
**Decision**: Reconcile every divergence (R1-R9) to the WeightWatcher source. Where SETOL.md and WW conflict on MECHANISM **or** TERMINOLOGY, WW wins (user-approved).
**Trade-off**: Authoritative WW conformance + two pre-existing-bug fixes + 2 new metrics **at the cost of** reversing four prior-plan decisions (D-C/D-D/partial D-E/D-F) and deliberately rewriting several pre-existing test contracts that encoded the old (SETOL/wrong) numbers.
**Reasoning**: The user explicitly declared WW the authoritative source for ALL mechanism and terminology. Pre-existing bugs (R1 factor-((1+√Q)/(1+1/√Q))² error; R2 broken `searchsorted` on a non-monotonic array) silently corrupt trap detection and the ERG diagnostic for every rectangular/non-trivial layer; "reconcile to authoritative" mandates fixing them. Tests scoped to `tests/test_analyzer/test_spectral_metrics.py` only (full suite ~1.5h forbidden, per LESSONS + CLAUDE.md).

### D-002 | EXPLORE → PLAN | 2026-06-03
**Context**: `detect_correlation_trap` (`spectral_metrics.py:501-513`) computes the MP edge with `(1+√Q)²` (R1, pre-existing bug; WW uses `(1+1/√Q)²`, Q=N/M, N=larger) and the TW threshold as `c_TW·σ²·M^(-2/3)` (prior D-E; R3 shows it lacks the `1/√Q` scale, the `bulk_max^(2/3)` factor, and the overall √). WW authoritative (§A/§B): `bulk_max=(σ(1+1/√Q))²`; `TW=(1/√Q)·bulk_max^(2/3)·M^(-2/3)`; `threshold=bulk_max+√TW`, NO c_TW.
**Decision**: Adopt the WW-exact edge and TW forms. Keep `c_TW` as a MULTIPLIER on `√TW` (`threshold = bulk_max + c_TW·√TW`) so `test_custom_tw_factor` monotonicity still holds, but change its default to **1.0** so default behavior is WW-exact. Update `SPECTRAL_TW_SAFETY_FACTOR = 1.0`.
**Trade-off**: WW-exact default behavior + retained tunable knob **at the cost of** a deliberate `c_TW` default change (2.5→1.0) — any consumer relying on the old 2.5 default gets a more sensitive threshold.
**Reasoning**: The multiplier preserves the existing `c_TW` parameter API and its only test contract (monotonicity). Default 1.0 makes the at-default output byte-match WW. Pure WW has no c_TW, so 1.0 is the WW-faithful default. SUPERSEDES the TW portion of prior-plan D-E (M^(-2/3) base alone was insufficient).
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:505-510` (re-anchored from prior D-007)

### D-003 | EXPLORE → PLAN | 2026-06-03
**Context**: `spectral_metrics.py:508` carries `# DECISION plan_2026-06-03_9e82787d/D-007` asserting the TW scaling is `M^(-2/3)` "per SETOL §2.3" and "Do NOT revert to N^(-1/3)". Step 1 (D-002) replaces that whole TW block with the WW-authoritative form, so the SETOL-framed anchor is stale.
**Decision**: Remove the prior D-007 anchor comment and replace it with a fresh `# DECISION plan_2026-06-03_bc986e52/D-002` anchor whose comment cites the WW form (`bulk_max + c_TW·√[(1/√Q)·bulk_max^(2/3)·M^(-2/3)]`, c_TW default 1.0). Likewise update the docstring formula block (~lines 460-464).
**Trade-off**: A single authoritative anchor **at the cost of** retiring a prior-plan anchor (the validator may flag the removed `9e82787d/D-007` until that plan is retired; this is inherited debt per LESSONS, not introduced).
**Reasoning**: In-code anchors MUST be numeric `D-NNN` matching the OWNING plan's decisions.md (LESSONS: in-code anchor IDs). Since this plan rewrites the code, the anchor must point here.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:513-518`

### D-004 | EXPLORE → PLAN | 2026-06-03
**Context**: `compute_erg_condition` (`:380-393`) computes the ERG boundary with `cumsum(log(ascending)) + searchsorted(.,0.0)` — searchsorted requires a sorted (monotonic) input, but a cumulative-log array is NOT monotonic (R2; explorer counter-example in erg-alpha-vs-ww.md Q1 shows it returns the OPPOSITE boundary). The analyzer ALREADY has the WW-correct descending-product loop in `compute_detX_constraint` (`:1028-1049`), which returns the COUNT of tail eigenvalues and rescales internally.
**Decision**: Replace the broken block by reusing `compute_detX_constraint`. Algorithm: on the rescaled ECS evals, derive the tail count, convert to an ascending-sorted boundary index `idx = len(rescaled) - count`, set `erg_lambda_min = sorted_ascending_rescaled[idx]`, keep `delta_lambda_min = float(xmin * wscale * wscale - erg_lambda_min)` (signed, prior D-006 sign fix STAYS). Keep `erg_log_det = Σ ln(rescaled tail)` and `erg_satisfied` unchanged.
**Trade-off**: DRY reuse of the authoritative loop + a real-bug fix **at the cost of** a small double-rescale guard: `compute_detX_constraint` rescales internally, so it must be fed the already-rescaled ECS evals (rescaling rescaled evals is near-idempotent since Σλ→N is already satisfied) OR fed the raw ECS evals with the caller deriving the boundary from the count — plan uses the count-to-index mapping on the caller's own already-rescaled array to avoid double work.
**Reasoning**: `compute_detX_constraint` is byte-equivalent to WW's `detX_constraint` (log-space, strictly safer than WW's raw `np.prod`). Reusing it (not reimplementing) honors the HARD DRY constraint. The prior signed-Δλ_min fix (D-006) is correct vs SETOL §7.3 and survives — only the BROKEN boundary underneath it is replaced.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:380-394`

### D-005 | EXPLORE → PLAN | 2026-06-03
**Context**: WW's sole canonical name for α̂ = α·log10(λ_max) is `alpha_weighted` (no `alpha_hat`, no `/N` variant anywhere in WW — findings R4/§E). Prior-plan D-F made `alpha_hat` canonical and labeled `alpha_weighted` "DEPRECATED alias" — backwards vs WW. The VALUE is correct; only the canonical-name/deprecation diverges.
**Decision**: Make `alpha_weighted` the CANONICAL name. Keep `alpha_hat` as an ALIAS equal to `alpha_weighted` (documented "SETOL-paper notation α̂ for alpha_weighted"); flip the "deprecated" label OFF `alpha_weighted`. Keep `alpha_hat_normalized` (/N) but DOCUMENT it as a non-WeightWatcher SETOL-theory extra. Prefer `MetricNames.ALPHA_WEIGHTED` as the WW-canonical summary key in `SPECTRAL_DEFAULT_SUMMARY_METRICS`, keeping `ALPHA_HAT` too for back-compat.
**Trade-off**: WW-canonical public naming **at the cost of** reversing the metric-naming decision made just last plan (D-F) — a second naming flip on the same field.
**Reasoning**: The user named WW the authoritative tie-breaker for terminology, and LESSONS notes that "which key is canonical" is a project-intent decision now explicitly resolved toward WW. Aliasing preserves both keys' VALUES, so no numeric output changes; existing `alpha_hat≈6.0` / `alpha_hat_normalized≈3·log10(2)` assertions survive. SUPERSEDES prior-plan D-F.

### D-006 | EXPLORE → PLAN | 2026-06-03
**Context**: `MetricNames.WW_SOFTRANK='ww_softrank'` exists and is WRITTEN at `spectral_analyzer.py:234,238` — but the value there is `max(rand_evals)/max(evals)` (a randomization-ratio), NOT WW's `mp_softrank = λ_plus/λ_max`. So `WW_SOFTRANK` is a mis-named live metric, and the real WW `mp_softrank` is absent. `MATRIX_RANK`, `NORM`, `SPECTRAL_NORM` are missing as named constants (bare literals).
**Decision**: Rename `WW_SOFTRANK`→`MP_SOFTRANK='mp_softrank'` (WW's name) and reserve it for the REAL WW metric added in Step 4. Move the existing `max(rand_evals)/max(evals)` randomization ratio to a distinct, correctly-named constant (`RAND_SV_RATIO='rand_sv_ratio'`) so it is not conflated with WW's mp_softrank. Add `MATRIX_RANK='matrix_rank'`, `NORM='norm'`, `SPECTRAL_NORM='spectral_norm'`.
**Trade-off**: Clean WW-aligned metric vocabulary **at the cost of** renaming one existing emitted column key (`ww_softrank`→`rand_sv_ratio`) — a back-compat break for any DataFrame consumer reading `ww_softrank`.
**Reasoning**: Leaving `WW_SOFTRANK` bound to a non-WW formula AND adding a second WW `mp_softrank` would create two confusingly-named ratio metrics. Separating them (WW name for WW formula; descriptive name for the rand ratio) is the only WW-faithful resolution. The `ww_softrank` key was added recently and has no documented external consumer.

### D-007 | EXPLORE → PLAN | 2026-06-03
**Context**: WW `mp_soft_rank(evals, num_spikes) = λ_plus/λ_max` with spikes removed first (§E). Absent from the analyzer (prior plan deferred as D-J).
**Decision**: Add `mp_softrank(evals, num_spikes=0)` computing `λ_plus(after removing num_spikes top evals) / λ_max`, wire into the per-layer metric dict under `MetricNames.MP_SOFTRANK`. `num_spikes` sourced from trap detection when available, else 0.
**Trade-off**: WW metric parity (+1 metric, net positive lines) **at the cost of** the Complexity-Budget line charge — acknowledged conformance cost, no new file/abstraction.
**Reasoning**: Single concrete call site; no abstraction introduced (a plain module function alongside the other metric helpers). Depends on Step 1's λ_plus form being WW-exact for consistency.

### D-008 | EXPLORE → PLAN | 2026-06-03
**Context**: WW switches to bias-corrected `alpha_bc = 1 + (n-1)/s` with objective `J = D_ks - 0.868/√n` for tail `N < 20` (SMALL_N_CUTOFF, §D). The analyzer applies standard MLE regardless → upward-biased α on thin tails.
**Decision**: Add a small-N branch inside `fit_powerlaw`: when the tail length `n < 20`, select xmin by argmin of the penalized objective `J = D_ks − 0.868/√n` and report the bias-corrected `alpha_bc = 1 + (n-1)/s` (s = sum-log-ratio). Standard branch (n≥20) unchanged.
**Trade-off**: WW-faithful thin-tail α **at the cost of** added branch complexity in the hot `fit_powerlaw` loop.
**Reasoning**: Existing tests use N≥100 (standard branch) → unaffected; a new N<20 test pins the new branch. SMALL_N_CUTOFF=20 is WW's constant.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:261-303`

### D-009 | EXPLORE → PLAN | 2026-06-03
**Context**: WW has only OVER_TRAINED_THRESH=2.0 / UNDER_TRAINED_THRESH=6.0 and labels α<2 literally "over-trained" (§F). Prior plan introduced an "ideal" band [2.0,2.1) (D-C, `SPECTRAL_IDEAL_ALPHA_BAND`) and the term "over-regularized" for α<2 (D-D). Both are SETOL-paper choices with no WW basis.
**Decision**: `classify_learning_phase` → WW scheme: α<0 "failed"; 0≤α<2.0 "over-trained"; 2.0≤α≤6.0 "good"; α>6.0 "under-trained". REMOVE "ideal"/"over-regularized"/"fair" labels and DELETE `SPECTRAL_IDEAL_ALPHA_BAND`. Update `model_analyzer.py` docstring (α<2 → "over-trained"), `spectral_analyzer.py` recommendations (WW phrasing; keep "check for correlation traps" guidance), and neutralize SETOL "ideal/critical target" framing in the funnel-plot docstring/labels.
**Trade-off**: WW-faithful terminology and the disappearance of a project-specific critical-point band **at the cost of** reversing two prior-plan decisions AND deliberately rewriting `TestClassifyLearningPhase` (which asserts the old SETOL labels: 1.5→"over-regularized", 2.0→"ideal").
**Reasoning**: User declared WW authoritative for terminology too. The narrow [2.0,2.1) band was an admitted prior-plan invention. New test contract: 1.5→"over-trained", 2.0/2.4/3.0/3.9/5.0/6.0→"good", 7.0→"under-trained", -1.0→"failed". SUPERSEDES prior-plan D-C and D-D.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:448-477` (classify_learning_phase + D-009 anchor at :458)

### D-010 | EXPLORE → PLAN | 2026-06-03
**Context**: WW's LAYER_TYPES_SUPPORTED includes NORM (1-D gamma vector). `get_layer_weights_and_bias` (`spectral_utils.py:141`) does NOT list NORM in its handled types → silently returns `has_weights=False`. Spectral ESD of a 1-D vector is degenerate (a single "matrix" of shape (d,1) → one singular value), so a full power-law fit is not meaningful.
**Decision (LAST step, most uncertain)**: MINIMAL fix — make the NORM skip EXPLICIT with a one-line `logger.debug` reason ("NORM layer <name>: 1-D scale/gamma params have degenerate ESD; spectral analysis skipped") rather than a silent fall-through. Do NOT attempt full NORM ESD analysis. If even this minimal change would balloon (>10 lines or needs new infra), STOP and recommend deferring R9 to a follow-up plan (Pre-Mortem trigger).
**Trade-off**: Observability of the skip (no silent drop) **at the cost of** NOT achieving full WW NORM-ESD parity — deferred as out-of-scope-if-it-balloons.
**Reasoning**: A 1-D gamma vector's ESD is genuinely degenerate; replicating WW's NORM handling fully is uncertain and risks scope blow-up. Explicit logging is the bankable minimum. Placed LAST so Steps 1-6 are committed before touching the most uncertain area.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_utils.py:161-171` (NORM explicit-skip branch + D-010 anchor at :162)

### D-011 | REFLECT iter-1 | 2026-06-03
**Context**: All 10 success criteria (C1-C10) verified PASS (ip-verifier, 30 checks); pytest 56/56 spectral + 69/69 full analyzer dir; zero scope drift (8 files, all in plan set); diff review clean (no debug artifacts, no reintroduced `(1+√Q)`/`searchsorted`/`abs()`-on-Δλ_min). Root-cause analysis skipped (no failures). EXTENDED checks skipped per protocol (iteration 1). validate-plan: zero errors INTRODUCED by this plan (19 remaining are inherited repo-wide debt in untouched files).
**Decision**: Recommend CLOSE. One orchestrator-fixed bookkeeping issue surfaced during REFLECT (see D-012).
**Trade-off**: Ship full WW conformance (R1-R9) now **at the cost of** the two "Not Verified" gaps (mp_softrank/small-N branch correctness spot-checked via fixtures, not cross-validated digit-for-digit against WW's own outputs on real model weights).
**Reasoning**: Both new metrics mirror WW's formulas verbatim and pass targeted unit tests; numeric parity against live WW runs would need WW installed + a shared model, out of scope. The two HIGH bug fixes (R1 MP edge, R2 ERG boundary) are independently confirmed (P-3 boundary hand-verified; trap tests green). Rejected alternative: hold CLOSE for a WW-vs-analyzer numeric harness (disproportionate to the conformance goal).

### D-012 | REFLECT iter-1 | 2026-06-03
**Context**: validate-plan initially reported 5 `anchor-unknown-plan` ERRORs on THIS plan's own bc986e52/D-002,003,004,008,010 anchors — not because the anchors or decisions.md entries were missing, but because the plan-writer wrote D-NNN headers with a 4th `| <title>` segment (`## D-NNN | EXPLORE → PLAN | <date> | <title>`). The validator's header regex `^## D-(\d{3}) \| (.+) \| (\d{4}-\d{2}-\d{2})$` anchors at the DATE, so 4-segment headers parse to ZERO entries → the active plan registered no decision IDs → every qualified anchor reported "unknown plan".
**Decision**: Orchestrator reformatted all `## D-NNN` headers in decisions.md to the canonical 3-segment `## D-NNN | <context> | <date>` (dropping the redundant 4th title segment; entry bodies retain the full content). plans/ is gitignored so no committed code changed.
**Trade-off**: Anchor validator now resolves all bc986e52/D-NNN **at the cost of** losing the one-line title summary from each header (content preserved in bodies).
**Reasoning**: The 3-segment header is the validator's contract; the 4-segment form is a silent footgun. After the fix, errors dropped 74→19 with zero introduced. LESSON for CLOSE: decisions.md D-NNN headers MUST end at the date — no trailing `| title`.

## plan_2026-06-03_9e82787d
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_9e82787d/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: Scope confirmed as Tier 1+2 (correctness + reporting/naming), explicitly excluding D-I (R-transform/Q̄²), D-J (MP SoftRank), D-K (missing universality classes). EXPLORE proposed adding new theory; user rejected.
**Decision**: Plan only D-A,B,C,D,E,F,G,H,L; treat D-I/J/K as out-of-scope and the ERG "det=1" claim as a GHOST (Correction C1).
**Trade-off**: A focused, low-risk, mergeable compliance fix **at the cost of** leaving the analyzer short of full SETOL §5.4/§6/§2.2 coverage (no free cumulants, no MP SoftRank, no 6-class universality table).
**Reasoning**: The excluded items are large theory additions with no existing-but-wrong code to correct; they are enhancements, not bug fixes. Bundling them would inflate blast radius and complexity budget. Rejected alternative: full-spec rewrite in one plan.

### D-002 | EXPLORE → PLAN | 2026-06-03
**Context**: D-F naming confusion — code has `alpha_weighted`=α·log₁₀(λ_max) (WW) and `alpha_hat`=α·log₁₀(λ_max/N) (/N theory). SETOL §10.2 sanctions both normalizations; the choice is project preference. User decided `MetricNames.ALPHA_HAT` must expose the WeightWatcher (un-normalized) convention.
**Decision**: `alpha_hat` key becomes the WW un-normalized value; introduce `alpha_hat_normalized` for the /N variant; keep `alpha_weighted` as a deprecated alias equal to `alpha_hat`.
**Trade-off**: WW-comparable α̂ as the primary metric + zero-break for `alpha_weighted` consumers **at the cost of** inverting what the `alpha_hat` key holds (a silent value change for any consumer that wanted the /N value) and carrying a deprecated alias.
**Reasoning**: WW-convention α̂ is the comparable cross-architecture metric per §2.4/§13 and matches the reference tool; the alias preserves DataFrame/report back-compat. Risk is contained by the mandatory consumer `rg` trace (Step 1.3) gated by STOP-IF #1. Rejected alternative: rename keys outright (breaks consumers) or leave as-is (misleading `MetricNames.ALPHA_HAT`).

### D-003 | EXPLORE → PLAN | 2026-06-03
**Context**: D-C phase bands are internally inconsistent (docstring says "2.0<α<4.0 Good" AND "α≈2 Ideal"; impl uses [2.0,2.5) ideal, 2.5-4 good, 4-6 fair). SETOL §3.2/§13 define Ideal as the critical point α≈2.
**Decision**: Narrow "ideal" to a documented band `SPECTRAL_IDEAL_ALPHA_BAND=(2.0,2.1)`; collapse the 4-6 "fair" sub-split into "good" so bands are self-consistent with the existing 2.0/6.0 threshold constants.
**Trade-off**: A single self-consistent, spec-aligned band scheme **at the cost of** dropping the informal "fair" label (a reporting granularity loss) and updating one test assertion (2.4 → "good").
**Reasoning**: Phase strings are informal display labels not keyed on by code logic (package-integration §3c), so collapsing "fair" is behavior-safe. The narrow ideal band restores the §3.2 critical-point semantics. Rejected alternative: keep [2.0,2.5) ideal (contradicts spec) or keep "fair" (re-introduces the docstring/impl split). If user wants "fair" retained, it is a one-line band tweak.

### D-004 | EXPLORE → PLAN | 2026-06-03
**Context**: D-G/D-H require new plot methods (funnel, MP-overlay). Complexity budget targets net-zero lines, but these are feature-bearing spec-mandated diagnostics with a single call site each.
**Decision**: Add the two plot paths as methods on the existing `SpectralVisualizer`; accept a net-positive line delta concentrated in Step 5.
**Trade-off**: Spec-mandated visual diagnostics (funnel convergence + MP bulk envelope) **at the cost of** a positive net-line delta and two single-call-site methods (earned-abstraction exception: spec-required, not speculative).
**Reasoning**: These are review-compliance features, not refactors; the earned-abstraction rule's single-call-site caution is waived because the methods implement named SETOL diagnostics (§8.2/§10.4, §10.1) rather than speculative generality. Verification is a headless generate-without-error smoke (LESSONS: smoke != correctness), since no pixel-level assertion is meaningful. Rejected alternative: defer viz to a follow-up plan (LESSONS: doc/feature belongs with the code change).

### D-005 | EXECUTE Step 1 surprise | 2026-06-03
**Context**: Step 1 consumer trace surfaced `src/dl_techniques/analyzer/README.md:~255`, which documents `alpha_hat` as the /N-normalized version and `alpha_weighted` as "legacy". Step 1 inverted those semantics, so the README prose is now stale (doc-drift introduced by this plan). Not in the original Files To Modify list.
**Decision**: Fold a README prose correction into Step 4 (interpretation/doc-text step) rather than spawn a follow-up; add `analyzer/README.md` to scope.
**Trade-off**: Keeping docs consistent with code in the same plan **at the cost of** a minor scope expansion (1 extra file, +1 doc edit in Step 4).
**Reasoning**: LESSONS — "Doc updates belong in the same plan as the code change they describe." The README is the user-facing description of the exact metric semantics Step 1 changed; leaving it stale would make the plan "complete" while user-visible docs are wrong. Step 4 is already the text-correctness step, so it groups cleanly. STOP-IF #1 was NOT triggered — README is prose, not a code consumer reading the value as a contract.

### D-006 | EXECUTE Step 2 (D-A) | 2026-06-03
**Context**: `compute_erg_condition` wrapped Δλ_min in `abs()` (spectral_metrics.py:383), destroying the sign that SETOL §7.3 makes the over-regularization diagnostic (<0 over-regularized, ≈0 ideal, >0 normal). Unit handling (`xmin*wscale²` vs rescaled boundary) was already correct.
**Decision**: Remove `abs()` so `erg_delta_lambda_min` is signed; anchor the invariant in-code.
**Trade-off**: A working over-regularization diagnostic **at the cost of** any latent consumer that assumed a non-negative Δλ_min (none found; ERG tests assert key-presence only).
**Reasoning**: One-token fix restoring spec semantics; the in-code `# DECISION .../D-006` anchor at spectral_metrics.py:386 warns against re-adding `abs()` as a spurious "safety" guard (decision-anchoring trigger: code whose obvious-looking "fix" would reintroduce the bug).
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:386-389`

### D-007 | EXECUTE Step 2 (D-E) | 2026-06-03
**Context**: `detect_correlation_trap` scaled Tracy-Widom fluctuations as `N^(-1/3)` (spectral_metrics.py:493); SETOL §2.3 specifies Δ_TW ~ O(M^(-2/3)) (base = min dimension, exponent -2/3). Both base and exponent differed.
**Decision**: Change to `M ** (-2.0/3.0)`; anchor the invariant in-code.
**Trade-off**: Spec-faithful trap-detection sensitivity **at the cost of** a different (smaller) Δ_TW threshold than before, slightly changing how aggressively spikes are flagged — verified to keep Δ_TW>0 so existing relative-comparison trap tests still pass.
**Reasoning**: Follows SETOL.md literally (§2.3 writes M^(-2/3)); the `# DECISION .../D-007` anchor at spectral_metrics.py:500 warns against reverting to N^(-1/3). If the user's RMT intent is the Johnstone N^(-2/3) sample-size scaling instead, that is a one-line base swap — noted for REFLECT.
**Anchor-Refs**: `src/dl_techniques/analyzer/spectral_metrics.py:500-503`

### D-008 | REFLECT iter-1 | 2026-06-03
**Context**: All 7 success criteria (C1-C7) verified PASS on first attempt; scoped pytest 50/50 green; scope drift = none (7 files, all in plan's Files-To-Modify incl. README via D-005); diff review clean (no debug artifacts, no abs() reintroduction, no new print, no new files). Root-cause analysis skipped (no failures). EXTENDED checks (adversarial review, devil's advocate, convergence, prediction accuracy) skipped per protocol — iteration 1.
**Decision**: Recommend CLOSE. Surface two items to the user before closing: (1) D-007 open question — SETOL.md literally writes Tracy-Widom O(M^(-2/3)) and the code now follows it, but the classical Johnstone scaling is N^(-2/3) on sample size; if the user's RMT intent differs it is a one-line base swap. (2) validate-plan.mjs shows 22 repo-wide anchor ERRORs, ALL in files this plan did not touch (inherited debt per LESSONS: "gate is zero INTRODUCED errors"); this plan introduced zero.
**Trade-off**: Closing now ships the verified Tier 1+2 compliance fix **at the cost of** leaving Tier 3 (R-transform Q̄², MP SoftRank, full 6-class universality) and the inherited repo-wide anchor debt for separate plans.
**Reasoning**: Simplification checks pass — added code is spec-mandated (funnel/MP-overlay), `SPECTRAL_IDEAL_ALPHA_BAND` centralizes the band (DRY), the `alpha_weighted` alias is justified back-compat, no speculative generality (YAGNI). Net +~130 lines concentrated in Step 5 viz, acknowledged in the complexity budget. Rejected alternative: bundle Tier 3 or the anchor-debt cleanup (would balloon scope past the user-approved boundary).

## plan_2026-06-03_bf1e592d
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_bf1e592d/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: Both ConvNeXt constructors already accept `stochastic_mode` and `create_convnext_v*` forward `**kwargs` cleanly (F1 §4). The knob can be surfaced either by mutating `create_model_config` (the per-dataset config dict) or by passing it as an explicit named kwarg at the factory call site.
**Decision**: Thread `stochastic_mode=args.stochastic_mode` DIRECTLY at the `create_convnext_v*` call site; do NOT inject it into `create_model_config`.
**Trade-off**: One explicit kwarg per call site **at the cost of** the per-script consistency of having all model knobs flow through `create_model_config`.
**Reasoning**: Keeps `create_model_config`'s pure `dataset -> config` contract intact (it is not a dataset-derived value; it is an experiment axis). Mutating the dict would entangle an A/B experiment knob with dataset defaults. Rejected: dict injection.

### D-002 | EXPLORE → PLAN | 2026-06-03
**Context**: None of the 3 convnext scripts currently seed. A depth-vs-gradient A/B whose only intended difference is the regularizer is confounded by random init + data order if unseeded (DN-2).
**Decision**: Add a convnext-local `--seed` (default 42) and call `set_seeds(args.seed)` at the top of each `train_model`/training entry, before dataset load and model build; do NOT widen the shared base parser.
**Trade-off**: Per-script duplication of the seed flag **at the cost of** not centralizing it in `create_base_argument_parser`.
**Reasoning**: LESSONS: don't widen shared infra (50+ trainers) for one concern; blast radius stays convnext-only. Rejected: adding `--seed` to the shared parser.

### D-003 | EXPLORE → PLAN | 2026-06-03
**Context**: The "choose between the 2" deliverable needs to train both modes and compare. The repo has a full subprocess-sweep harness (`rms_variants_train`) writing per-cell `results.csv`, AND a simpler `compare_runs(run_a, run_b, ...)` that reads `training_log.csv` directly. The existing train scripts already emit `training_log.csv` via `create_callbacks`/CSVLogger (DN-3, F2 §7).
**Decision**: Build ONE thin driver `run_stochastic_comparison.py` that subprocess-wraps the production train scripts verbatim (one per mode, serial), discovers each run dir via snapshot-diff of `results/`, then calls `compare_runs`. No per-cell trainer, no custom report writer.
**Trade-off**: Reuse of the production training path + minimal new code (1 file, 0 abstractions) **at the cost of** the richer aggregated `all_runs.csv`/`summary.md` artifacts the full sweep harness produces.
**Reasoning**: A 2-cell comparison does not earn the sweep harness's machinery (earned-abstraction rule). Subprocess isolation prevents TF/Keras cross-run state contamination. `compare_runs` already emits `comparison.md` + curves, which is sufficient to choose. Rejected: cloning `rms_variants_train/sweep.py` (per-cell `results.csv` trainer) — over-built for N=2.

### D-004 | EXPLORE → PLAN | 2026-06-03
**Context**: Train scripts build `results/{prefix}_{model_name}_{timestamp}` internally with no output-dir override; overriding would touch `create_callbacks` (shared by 50+ trainers, out of scope) — DN-5.
**Decision**: Discover each subprocess's run dir by snapshotting `set(os.listdir('results'))` before launch and diffing after; assert exactly one new dir, fail loudly otherwise.
**Trade-off**: Robustness under the no-parallel-GPU constraint + zero shared-infra edits **at the cost of** relying on serial execution and no concurrent unrelated writes to `results/` during a run.
**Reasoning**: Serial-only is already a hard constraint (single-GPU). Snapshot-diff keeps blast radius to the driver. Rejected: adding an `--output-dir` override to `create_callbacks`.

### D-005 | REFLECT (iter-1) | 2026-06-03
**Context**: All 4 implementation steps shipped (ef9c8ae6, 90763c5a, 0065d802, 6732b889). Verification ran the full non-training suite: 7/7 SC PASS (AST parse, `--help` smoke on all 4 modules, grep of call sites / `set_seeds` / env hard-set / `compare_runs`, git-diff scope guard). Diff review clean (no TODO/print/breakpoint/commented-out code). `validate-plan.mjs` reports 34 ERRORs — ALL inherited repo-wide anchor debt (convnext_patch_vae_v2, bdd100k_video, routing_probabilities, lighthouse_attention, gpt2, nam) in files this plan never touched; ZERO introduced by this plan (the 4 changed files carry no `# DECISION` anchors).
**Decision**: Recommend CLOSE. No regressions (default `stochastic_mode='depth'` is behavior-preserving; all edits additive), no scope drift (4 planned = 4 changed), no simplification blockers (0 new abstractions).
**Trade-off**: Closing on static verification (AST + `--help` + grep + diff) **at the cost of** an end-to-end driver run (subprocess → snapshot-diff → compare_runs), which is deliberately deferred to the user (GPU time).
**Reasoning**: The deliverable is the selectable knob + the driver, not a trained result. The default path is behavior-preserving and the new code is a thin, read-verified subprocess loop reusing the production training path and the existing `compare_runs`. Devil's-advocate (not required at iter-1, noted anyway): the one un-run path is the snapshot-diff dir discovery — mitigated by the exactly-one-new-dir `SystemExit` guard and the serial-only constraint. Inherited validator ERRORs are pre-existing debt (LESSONS: gate is zero-introduced, not zero-repo-wide), addressable via `bootstrap.mjs retire` in a separate plan.
**Anchor-Refs**: none (no in-code anchors introduced).

### D-006 | EXECUTE (iter-2, surprise discovery during experiment run) | 2026-06-03
**Context**: Running the actual experiment surfaced a pre-existing bug NOT in this plan's original scope: `train_convnext_v1.py` compiled with `SparseCategoricalCrossentropy(from_logits=False)`, but the ConvNeXtV1 classifier head is a bare `Dense(num_classes)` with no softmax (`convnext_v1.py:335-341`) — the model emits LOGITS. Result: init loss ~9.5 (vs ln(10)=2.30), saturated gradients, val_accuracy pinned at 0.10 (random) for all 30 epochs. `train_convnext_v2.py` already used `from_logits=True`. Empirically confirmed: model output rows sum to 0.19/0.64/0.96 (not 1); `from_logits=True` loss=2.245 at init.
**Decision**: Flip v1's loss to `from_logits=True` (one line). Fixes convergence: re-run reached val_acc 0.52 by epoch 30.
**Trade-off**: A one-line correctness fix touching a file already in this plan's scope **at the cost of** widening the plan from pure-plumbing to also fixing a latent training bug — justified because the experiment is unusable without it and the user directed an autonomous fix.
**Reasoning**: The model is logits-by-design (matches v2 + standard ConvNeXt). The trainer was wrong, not the model. Rejected: adding softmax to the model head (would change model contract, break v2/tests/other consumers, and double-apply softmax under v2's from_logits=True).
**Anchor-Refs**: none introduced.

### D-007 | EXECUTE (iter-2, surprise during experiment run) | 2026-06-03
**Context**: The comparison driver, when launched on the same GPU as the trainer, SIGABRT'd the trainer with a fatal XLA allocator error (`Check failed: h != kInvalidChunkHandle`) on an otherwise-free 24GB GPU0 (worked nondeterministically on GPU1). Cause: the driver imports TF (via `compare_runs` → logger) and held a GPU context on the trainer's GPU, fragmenting/starving the trainer's allocator.
**Decision**: Force the driver process CPU-only by setting `os.environ['CUDA_VISIBLE_DEVICES']=''` at module top, before the TF-triggering import. The child training subprocess still gets its GPU via the hard-set `env['CUDA_VISIBLE_DEVICES']=str(args.gpu)` override.
**Trade-off**: Driver runs CPU-only (a few harmless `cuInit NO_DEVICE` log lines) **at the cost of** the driver no longer being able to do GPU work — which it never needed (orchestration + pandas/matplotlib comparison only).
**Reasoning**: A driver holding a GPU context next to its own training subprocess is an anti-pattern. Rejected: relying on TF memory-growth coexistence (nondeterministic, already failed once).
**Anchor-Refs**: none introduced.
