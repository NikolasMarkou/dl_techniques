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

## plan_2026-06-04_d4ef81f1
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-04_d4ef81f1/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-04
**Context**: `sampling.py` is a single top-level file imported by ≥3 sites via `from dl_techniques.layers.sampling import Sampling`. Two factory placements exist in the repo: package promotion (dominant, ~10 layers) and inline `create_*` in a top-level file (1 precedent, `sparse_autoencoder.py:823`).
**Decision**: Add the factory INLINE to `layers/sampling.py` (not promote to a `sampling/` package), mirroring the `sequence_pooling/factory.py` 4-function surface.
**Trade-off**: zero import-site churn and a smaller diff **at the cost of** diverging from the dominant package-promotion convention (the factory lives in the layer file rather than a sibling `factory.py`).
**Reasoning**: Promotion would break every `from ...sampling import Sampling` caller and balloon the diff; the two samplers have only 1-2 ctor params each, so a full package is disproportionate. The inline precedent exists. Rejected: package promotion (import-sweep cost, out of complexity budget).
**Anchor-Refs**: `src/dl_techniques/layers/sampling.py:455-470`

### D-002 | EXPLORE → PLAN | 2026-06-04
**Context**: Three behaviors are wanted but `HypersphereSampling.build` hard-rejects `z_log_var` last-dim != 1, and the Gaussian KL is the wrong prior for a sphere. The factory could expose 2 or 3 types.
**Decision**: Factory exposes exactly 2 LAYER types (`gaussian`→`Sampling`, `hypersphere`→`HypersphereSampling`); the third behavior (`hypersphere_faithful`) is a VAE-level encoder+loss MODE that reuses the `hypersphere` layer, selected by a `VAE.sampling_type` knob ∈ {gaussian, hypersphere_controlled, hypersphere_faithful}.
**Trade-off**: a clean 2-type factory + a model-level mode branch **at the cost of** the factory type-set not being 1:1 with the user-facing mode-set (one extra layer of indirection between `sampling_type` and the layer type).
**Reasoning**: The faithful mode differs from controlled only in the ENCODER head ([B,1] radius vs mean-reduced [B,1]) and the LOSS (radius-variance KL vs Gaussian KL) — both are VAE concerns, not layer concerns. Keeping them out of the factory keeps the layer registry honest (2 real classes). Matches the locked design resolved with the user. Rejected: a 3rd factory type (would encode model-level loss/encoder choices into a layer factory).

### D-003 | EXPLORE → PLAN | 2026-06-04
**Context**: The hypersphere prior's correct regularizer is a vMF/wrapped-normal KL; the user resolved (AskUserQuestion, `findings.md` "Design Decision") to ship a simplified radius-variance KL instead, documented as NOT a full vMF S-VAE.
**Decision**: For `hypersphere_faithful`, REPLACE the Gaussian KL with `kl = mean(0.5·(exp(rlv_clip) − rlv_clip − 1))` (`rlv_clip = clip(radius_log_var, -20, 20)`); no direction-KL term (uniform-sphere prior; radius mean fixed at 1.0 by the layer).
**Trade-off**: a tractable, numerically-stable, geometry-aligned regularizer that ships now **at the cost of** theoretical completeness (it is not the exact vMF KL; the direction distribution is unregularized).
**Reasoning**: User-locked scope. The radius-variance KL is the standard 1-D Gaussian-Gaussian KL on the radius noise and is float32-stable under [-20,20] clipping (mirrors the existing `_compute_kl_loss` clip). Documented as a simplification in the model docstring. Rejected: full vMF S-VAE (out of scope; user declined).
**Anchor-Refs**: `src/dl_techniques/models/vae/model.py:865`

### D-004 | EXPLORE → PLAN | 2026-06-04
**Context**: `self.decoder` is extracted at `VAE.__init__` via `self.get_layer("vae_sampling").output` (`model.py:242-244`). Changing the sampler layer name or its output rank breaks `decode()`/`sample()` and the trainer's interpolation plots.
**Decision**: Name the sampler `"vae_sampling"` in ALL three modes (both `Sampling` and `HypersphereSampling` instances), so the decoder extraction line is untouched.
**Trade-off**: zero change to the decoder-extraction code path **at the cost of** the layer name no longer signaling which sampler is active (the active mode is recoverable only from `get_config`).
**Reasoning**: `HypersphereSampling` emits `[B,latent_dim]` in every mode, identical in rank/shape to `Sampling`, so the extraction is shape-safe. Preserving the name is the lowest-risk integration. Rejected: per-mode layer names (would require branching the decoder-extraction line — more surface, more risk).
**Anchor-Refs**: `src/dl_techniques/models/vae/model.py:293`

### D-005 | EXPLORE → PLAN | 2026-06-04
**Context**: The full 3-arm A/B exceeds 2 min wall-clock. LESSONS: `run_in_background` from a sub-agent dies when the sub-agent exits.
**Decision**: ip-executor runs ONLY the short `--smoke` verification (step 5); the ORCHESTRATOR (main thread) launches the full `run_sampler_comparison.py` run (step 6) serially on GPU1.
**Trade-off**: reliable completion of the long run **at the cost of** the orchestrator being blocked on the full run instead of delegating it.
**Reasoning**: A sub-agent-launched background job would be killed on sub-agent exit, losing the run. Step 6 was added at the PLAN handoff and carries its own Pre-Mortem (per LESSONS: handoff-appended steps need their own falsification entry). Rejected: delegating the full run to an ip-executor (would die mid-run).

### D-006 | EXECUTE surprise (step 3) | 2026-06-04
**Context**: The step-3 smoke run surfaced a PRE-EXISTING latent bug: `train_vae.py`'s `plot_training_history` -> `train.common.evaluation:184 generate_training_curves` hard-requires a `'loss'` history key, but the VAE custom train_step emits `total_loss`/`val_total_loss` (no plain `loss`). `KeyError: 'loss'` fired AFTER all artifacts were written (RC!=0). This would crash EVERY real VAE run (not just smoke) and would trip the step-4 driver's non-zero-returncode SystemExit.
**Decision**: Fix LOCALLY in `train_vae.py` with a <10-line history-key alias (`total_loss`->`loss`, `val_total_loss`->`val_loss`) before calling the plot util; do NOT modify the shared `train.common.evaluation` util.
**Trade-off**: A local shim **at the cost of** leaving the shared util's rigid `'loss'` assumption unfixed for other custom-loss models. Acceptable: LESSONS — "do NOT refactor shared infra (`train.common`) for one trainer's concern"; the shared util has many callers and a broad fix is out of this plan's scope.
**Reasoning**: Surfaced as a step-3 surprise, not a plan falsification — the fix is in-scope (the trainer must run for the comparison) and within the leash (local, <10 lines, no new approach). Flagged for REFLECT/LESSONS as a reusable gotcha (custom-loss VAE + `generate_training_curves`). Anchor: not anchored (simple alias, no design rationale needed at the call site).

### D-007 | EXECUTE surprise (step 5 smoke) | 2026-06-04
**Context**: The step-5 smoke revealed `create_callbacks(include_analyzer=True)` (default) attaches a per-epoch WeightWatcher/ModelAnalyzer callback (~80s/epoch per LESSONS) that dominates wall-clock. It is irrelevant to a SAMPLER comparison (which compares total_loss/recon/KL curves, not spectral metrics). The plan's A4 wall-clock assumption did not account for it; a full multi-epoch x3-arm run with the analyzer ON would be ~40+ min.
**Decision**: Add a `--no-epoch-analyzer` flag to `train_vae.py` (threads `include_analyzer=not args.no_epoch_analyzer`; DEFAULT keeps analyzer ON = preserves existing behavior), and have `run_sampler_comparison.py` ALWAYS pass `--no-epoch-analyzer` to its arms (the comparison never needs per-epoch spectral analysis).
**Trade-off**: A fast, focused comparison run **at the cost of** one more CLI flag + dropping per-epoch spectral plots for the comparison arms. Acceptable: matches the repo's established convnext `--no-epoch-analyzer` convention (LESSONS: "make it throttleable for sweeps"); end-of-training artifacts + the loss/recon/KL comparison are unaffected.
**Reasoning**: Smaller blast radius than changing the default; the flag is reusable. Surfaced as a step-5 surprise; in-scope (makes step 6 practical), within leash.

### D-008 | REFLECT (iter 1) | 2026-06-04
**Context**: All 6 steps executed; no leash hit beyond the in-step D-006/D-007 surprises. Independent verifier: criteria 6/6 PASS — test_sampling 55/55, test_vae 61/61, no regression, exactly 6 files changed, 0 introduced validator ERRORs, all 3 plan-id anchors (D-001/D-003/D-004) resolve. No Pre-Mortem STOP-IF fired (faithful loss finite, decoder extraction OK all modes, save/load OK, no full-run arm crash). Full A/B (mnist, 20ep): val_total_loss gaussian=0.2273, hypersphere_controlled=0.26143, hypersphere_faithful=0.20815.
**Decision**: Route to CLOSE (pending user confirmation).
**Trade-off**: Closing now **at the cost of** the comparison being a single-dataset (mnist), single-seed, latent_dim=2, 20-epoch snapshot — not a multi-seed/multi-dataset study. Acceptable: the user asked to build + run the comparison; the harness (factory + 3-mode VAE + driver) is reusable for deeper sweeps later.
**Reasoning**: 6 Simplification Checks clear — inline factory (no new package), single `sampling_type` branch (no wrapper cascade), simplified radius-only faithful KL (numerically stable, documented as not-a-vMF-S-VAE). The result is scientifically coherent: faithful (proper radius reg) slightly beats the Gaussian baseline on total loss; controlled (sphere sampler + mismatched Gaussian KL) is worst — exactly the mis-regularization caveat flagged at design time.

## plan_2026-06-04_a114f829
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-04_a114f829/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-04
**Context**: A new thin-shell hypersphere sampler is needed alongside the existing Gaussian-ball `Sampling` layer. `layers/sampling.py` already hosts `Sampling` (5-method Keras-3 pattern, raw-int seed). The interface and formula were locked with the user (direction = normalize(z_mean+eps), radius = thin Gaussian shell at `radius` with `z_log_var` thickness, `z_log_var` shape `[B,1]`).
**Decision**: Add `HypersphereSampling` as a SIBLING class in the existing `src/dl_techniques/layers/sampling.py` (not a new file), mirroring `Sampling`'s structure exactly, and cover it with a `TestHypersphereSampling` class in the existing test file.
**Trade-off**: A growing single file with two sampler classes **at the cost of** one-file-per-class granularity. Justified: both are tiny stateless reparameterization layers sharing the same skeleton, conventions, and import path; co-location matches the user's explicit "sibling in the same file" instruction and the repo's standalone-layer-file convention.
**Reasoning**: Keeps the import surface (`from dl_techniques.layers.sampling import ...`) and serialization keys stable; no `__init__.py` export needed (empty by convention). Rejected alternative: new file `hypersphere_sampling.py` (contradicts the locked instruction and fragments two near-identical samplers).

### D-002 | EXPLORE → PLAN | 2026-06-04
**Context**: The explorer flagged `keras.random.SeedGenerator` as a possible need for per-call stochasticity, but the sibling `Sampling` deliberately uses a raw int `seed` and the existing save/load test relies on seed=42 reproducibility.
**Decision**: Mirror the sibling exactly — store a raw int `seed` and pass it directly to `keras.random.normal`; treat `SeedGenerator` as a rejected GHOST.
**Trade-off**: Same-seed-same-output determinism (test reproducibility) **at the cost of** the per-call fresh-randomness that `SeedGenerator` would provide. Acceptable: the sibling's contract is the spec here, and the locked design needs stochastic-but-reproducible sampling, which raw-int seed delivers exactly.
**Reasoning**: Re-opening SeedGenerator is gated behind an explicit Pre-Mortem STOP IF (save/load array-equal fails while determinism passes); not pre-emptively adopted.
**Anchor-Refs**: `src/dl_techniques/layers/sampling.py:396`

### D-003 | EXPLORE → PLAN | 2026-06-04
**Context**: `ops.normalize(x, axis=-1)` is the repo's canonical runtime L2-normalize idiom, but Keras 3.8 zero-safety and return-type (tensor vs `(normalized, norm)` tuple) are unverified; the degenerate `z_mean+eps==0` row could produce NaN.
**Decision**: Empirically verify `ops.normalize` as the FIRST substep of EXECUTE; use it if it is a zero-safe single tensor, else fall back to the manual `x / ops.maximum(norm, eps)` idiom (`polar_initializer.py:78-97`).
**Trade-off**: A small upfront verification cost **at the cost of** committing to a normalize mechanism before EXECUTE. Acceptable: the observable invariant (`||z||≈radius`) is mechanism-agnostic, so the fallback is drop-in and blocks no downstream step.
**Reasoning**: Avoids shipping a silent NaN footgun on the degenerate direction; the choice is empirically grounded rather than assumed.

### D-004 | PLAN approval refinement | 2026-06-04
**Context**: At PLAN approval the user specified the degenerate-direction behavior: when `z_mean + eps` is exactly the zero vector, the unit direction must resolve to the canonical basis vector `e_0 = [1, 0, ..., 0]` (so `z = radius * e_0`), not a zero/NaN/eps-floored near-zero. This SUPERSEDES the D-003 "verify ops.normalize, else eps-fallback" plan for the degenerate branch.
**Decision**: Use manual normalize-with-branch unconditionally: `u = where(norm < eps0, e_0, g / maximum(norm, eps0))`, with `e_0` a one-hot first-basis vector broadcast to `[B, D]`. Bare `ops.normalize` is no longer sufficient (it cannot emit `e_0`), so the D-003 empirical check is repurposed to validate the manual path + exact `e_0` on a zero row.
**Trade-off**: A deterministic, well-defined direction on the measure-zero degenerate set **at the cost of** a slightly more verbose normalize (explicit `where` + `one_hot`) than a one-line `ops.normalize`. Acceptable: removes the only NaN risk and the only user-escalation Pre-Mortem in the plan; the extra ops are cheap and standard.
**Reasoning**: `e_0` is a fixed, gradient-stable target (constant w.r.t. inputs on the degenerate set); preferable to an eps-floored near-zero whose direction is arbitrary. Mirrors the spirit of `polar_initializer.py:78-97` (`max(norm, eps)`) but upgrades the degenerate output from "tiny arbitrary vector" to "canonical basis vector" per user instruction.
**Anchor-Refs**: `src/dl_techniques/layers/sampling.py:409`

### D-005 | REFLECT (iter 1) | 2026-06-04
**Context**: Both steps executed without a single fix attempt. Independent verifier confirmed all 5 success criteria PASS: ctor validation (radius<=0->ValueError, non-int seed->TypeError); forward shape (200,6) with thin-shell norm mean 2.000326 (delta 3.26e-4 vs radius 2.0); gradient flow to both inputs; get_config/save-load seed=42 array-equal; scoped suite 43 passed (31 TestSampling no-regression + 12 TestHypersphereSampling). Scope = exactly the 2 planned files; no debug artifacts. validate-plan.mjs: 0 ERRORs/WARNs introduced (all repo-wide hits are inherited debt in unrelated files).
**Decision**: Route to CLOSE (pending user confirmation).
**Trade-off**: Closing now **at the cost of** not running the full 1.5h `make test` suite. Acceptable: the scoped suite covers the only changed module and its in-file siblings; purely additive change with zero consumer-facing modification to `Sampling`.
**Reasoning**: 6 Simplification Checks all clear — one new class mirroring the sibling, no speculative abstraction, e_0 branch is user-required (essential not accidental), reused the polar_initializer normalize idiom. No simplification blockers. No Pre-Mortem STOP IF fired.

## plan_2026-06-03_da3a2bbb
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_da3a2bbb/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-04
**Context**: `layers/sequence_pooling.py` (935 LOC, 3 classes + 2 aliases) is a flat single-file module. The user asked to convert it into a self-contained package mirroring `layers/attention/` (factory + README + GUIDE). The top migration risk — bare `@register_keras_serializable()` keys breaking on file move — was FALSIFIED in EXPLORE (verified keys are `Custom>ClassName`, `__module__`-independent). User chose a 3-file split (one class per file, like attention/) and Google-style docstring conversion.
**Decision**: `git mv` the original into `sequence_pooling/sequence_pooling.py` (facade keeps the name + R100 history), extract `AttentionPooling`/`WeightedPooling` into their own files, add a registry-driven `factory.py`, a string-`__all__` `__init__.py` re-exporting all symbols, plus README.md + GUIDE.md — all decorators stay BARE.
**Trade-off**: faithful attention/-style structure + repo-convention Google docstrings **at the cost of** a larger diff (3-file split + ~935 LOC of docstrings touched) and net-additive file count, versus the lower-effort single-file-keep option.
**Reasoning**: The split matches the established template and the user's explicit choice; serialization safety removes the only real risk. Alternative (keep single file) rejected by the user. Adding `package=` to pin keys was REJECTED — it would CHANGE the key and break existing saves (the bare key is already location-independent).
**Anchor-Refs**: none — no `# DECISION` anchors needed (pure template-mirror, no failed approaches or non-obvious workarounds to anchor in source).

### D-002 | REFLECT (iter-1) | 2026-06-04
**Context**: All 8 construction steps committed (e1c664fe → 9723711a). ip-verifier ran all 9 success criteria + regression/scope-drift/diff-review.
**Decision**: Recommend CLOSE. 9/9 criteria PASS; existing suite `53 passed` UNCHANGED; save/load round-trip `roundtrip-ok` (confirms serialization-key stability — finding A2); scope drift clean (all 7 changed paths under `sequence_pooling/`, transformer encoders + test file untouched); no debug artifacts; Google docstring conversion complete (zero leftover RST `:param:`/`:type:` markers); zero `# DECISION` anchors introduced.
**Trade-off**: shipping a net-additive package (+6 files) **at the cost of** more surface to maintain — accepted because the package IS the locked deliverable.
**Reasoning**: Every success criterion has objective evidence. The 53-test suite passing unchanged is the authoritative verbatim-logic check (the extraction preserved all forward-pass math). The round-trip falsification signal did NOT fire (keys stable).
**Simplification checks (6)**: (1) No new abstraction beyond the earned factory (matches attention/ffn/norms idiom; 3 concrete registry entries). (2) No wrapper cascade / config toggle / adapter. (3) Essential complexity only (split + factory + docs = the deliverable). (4) Junior-dev test: README+GUIDE make the package self-explanatory. (5) No copy-paste beyond the verbatim class-body moves (intentional, history-preserving). (6) No dead code; `layers` import dropped from facade because genuinely unused.
**Devil's advocate**: One residual risk — the verifier imports the transformer-encoder MODULES (proves the `from ..sequence_pooling import ...` line resolves) but does not instantiate a full TextEncoder/VisionEncoder end-to-end. Mitigated: the import is the only coupling point the migration touched; the encoders' own tests (separate suite) exercise instantiation and were unaffected (scope-drift clean). Risk judged negligible.
**Validator note**: `validate-plan.mjs` reports 58 repo-wide ERRORs — ALL `anchor-unknown-plan`/`anchor-orphan` in pre-existing files (clifford_block, gpt2, lewm, routing_probabilities, etc.), ZERO in `sequence_pooling/`. Per LESSONS, the CLOSE gate is zero INTRODUCED errors; this plan added no anchors → zero introduced. Inherited debt is out of scope.

### D-003 | REFLECT (iter-1) presentation-contract log | 2026-06-04
**Context**: validate-plan.mjs emitted WARN [presentation-contract-unlogged] — it cannot see chat, only plan files.
**Decision**: Record here that **PC-PLAN was emitted to the user before the PLAN→EXECUTE transition** (full verbatim render of Goal/Problem/Files/Steps/Assumptions/Failure Modes/Pre-Mortem/Success Criteria/Verification Strategy/Complexity Budget), and the user replied "EXECUTE" to approve. PC-EXPLORE (findings digest) and PC-EXECUTE-STEP (per step) were likewise emitted. This entry satisfies the best-effort contract-logging signal.
**Trade-off**: n/a (record-keeping).

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
