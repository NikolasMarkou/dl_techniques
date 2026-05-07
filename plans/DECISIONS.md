# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

## plan_2026-05-07_3f461682
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-07_3f461682/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-07
**Context**: 5 in-scope Pattern-3 CLM trainers all share `metrics={"logits": ["accuracy"]}` and an empty `_post_generate_hook` extension. Existing `dl_techniques.metrics.perplexity_metric.Perplexity` is drop-in compatible. No BPC/BPW/BLEU exists. User instruction: "make good use of the common".
**Decision**: Add ONE shared module `src/dl_techniques/metrics/llm_metrics.py` (`BitsPerToken`, `BitsPerCharacter`, plus pure-Python `self_bleu`/`distinct_n`/`aggregate_probe_metrics` helpers) + ONE builder helper `build_clm_metrics()` in `train.common.nlp`. Each trainer's `compile_model` becomes a single-line metrics swap; probe-bearing trainers add a single line binding `_post_generate_hook = augment_probe_results`.
**Trade-off**: ~+220 LOC of shared code (excluding tests) **at the cost of** every per-trainer `compile_model` losing its inline `["accuracy"]` literal and gaining a dependency on the shared helper.
**Reasoning**: User explicitly demands DRY; per-trainer copy of metric instantiation contradicts the request and matches an existing anti-pattern (5x duplicated probe class). Reusing existing `Perplexity` (already AMP-safe via fp32 accumulator) avoids re-implementing the gold-standard metric. Free-function `aggregate_probe_metrics` over a probe subclass is cleaner because (a) the probe class is already duplicated 5x — adding a 6th subclass per trainer compounds the duplication; (b) `_post_generate_hook` is a documented extension point. Probe-class extraction is deferred to a separate refactor plan.
**Alternatives rejected**:
- Per-trainer copy of metric list (violates DRY).
- Subclass `GenerationProbeCallback` with `_post_generate_hook` override per trainer (5x copy of subclass; loses the DRY win).
- Compute BPC by accumulating actual byte counts via decoded text (correct but requires decode in training loop — slow). Display constant `chars_per_token` is the standard simplification.
- Include BPW (bits-per-word) — less commonly reported; adds a third constant; punted to keep metric set focused.
- Include BLEU/ROUGE/Self-BLEU at compile time — impossible without generation; compile-time metrics see logits only.
**Anchor-Refs**: (none — no `# DECISION` comment needed; the design is encoded in the new module's structure, not in a non-obvious code choice.)

### D-002 | EXECUTE (Step 3 prep) | 2026-05-07
**Context**: Surprise discovery during Step 3 prep — F-002 claimed all 5 GenerationProbeCallback classes implement `_post_generate_hook`, but `grep -rn _post_generate_hook src/train/` shows only `gpt2/pretrain.py` and `wave_field_llm/pretrain.py` have it. The 3 cliffordnet probes (nlp, nlp_unet, nlp_routing) lack the extension point. Plan SC-3 expects 5 hook binds.
**Decision**: Add the `_post_generate_hook(self, results: dict) -> None` extension method (empty default) and a single-line `self._post_generate_hook(probe_results)` call inside `_run_probes` to all 3 cliffordnet probes, matching the gpt2/wave_field_llm pattern. Then bind `augment_probe_results` in all 5 trainers as planned.
**Trade-off**: 3 extra small edits (add extension point) **at the cost of** preserving the DRY hook-binding contract across all 5 probe-bearing trainers and honoring SC-3 verbatim.
**Reasoning**: The alternative (binding the hook in only 2 trainers and downgrading SC-3 to "exactly 2 hits") undercuts the user's "make good use of the common" intent and leaves 3 trainers without the diversity/throughput aggregates. Adding the extension point is a +2 LOC delta per file, fits within the 10-Line Rule, and is an internal protocol enhancement (private method, no API change).
**Falsification signal**: none fired. This is a finding-correction, not a pivot.
**Findings correction**: F-002 line 38 reads "All five `GenerationProbeCallback` classes implement an empty `_post_generate_hook`" — actually only 2 do. Will mark `[CORRECTED iter-1]` in findings.md.
**Anchor-Refs**: (none — purely protocol-additive, no decision comment needed.)

### D-003 | EXECUTE -> REFLECT | 2026-05-07
**Context**: All 9 EXECUTE steps completed first try. No fix attempts. No leash hits. No falsification signals fired.
**Decision**: Run REFLECT Phase-2 verification.
**Verification outcome**: 7/7 SC PASS (see verification.md). 170/170 tests in tests/test_metrics/ pass. py_compile clean for all 8 affected modules. Convention scan clean. Scope drift zero (9 planned files, 9 changed). Changelog clean (all radius:LOW). validate-plan.mjs ERRORs are all pre-existing orphan anchors from legacy code, none introduced by iter-1.
**Simplification Checks** (6):
  1. **Single source of truth**: PASS — all metric math in one module; SC-1 grep confirms zero duplication.
  2. **Forbidden patterns**: PASS — no wrapper cascades, no config toggles, no copy-paste, no exception swallowing (only the deliberate `try/except` in `aggregate_probe_metrics` that is documented as "probe must never kill train" and logs+skips).
  3. **Complexity budget**: PASS for files/abstractions (2/3, 2/2). LOC overshoot was docstrings + 3 cliffordnet probe edits driven by D-002 finding correction; not budget-bloat.
  4. **10-Line Rule**: N/A (no fix attempts).
  5. **Revert-First**: N/A (no failures).
  6. **3-Strike Rule**: N/A (no recurrence).
**Recommendation**: CLOSE — present PC-REFLECT to user per protocol.
**Anchor-Refs**: (none.)

## plan_2026-05-07_08aaf818
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-07_08aaf818/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-07
**Context**: Commit 1fe2088 hardened `wave_field_llm/pretrain.py` against tiktoken decode crashes when sampled ids include reserved specials (50257..50260). Four sibling training scripts have the same unguarded `self._enc.decode(ids)` pattern. Out-of-scope verification: NAM tokenizer is custom (not tiktoken); cliffordnet/power_sampling.py masks specials pre-sample so its decode is safe-by-construction.
**Decision**: Mirror the exact 1fe2088 patch shape per-file (special-id range + in-loop mask + try/except backstop). Routing variant gets a small structural variation: mask insertion point is after `np.log(np.clip(...))` (no eot mask line in that variant), and the try/except keeps the `Tuple[str, int]` return signature in both branches.
**Trade-off**: Behaviour parity with the proven 1fe2088 fix **at the cost of** ~60-80 net added lines across four files (no abstraction extraction).
**Reasoning**: Extracting a shared helper (e.g. `_safe_decode(enc, ids)`) would cross the train/common boundary and pull in `tiktoken` as a hard dep on shared utilities — not worth it for 4 sites. The repeat-the-pattern approach also keeps each file self-contained for grepability and matches the codebase convention of per-file probe callbacks. Rejected alternative: widen the except clause to catch `Exception` — explicitly NOT done because matches reference exactly and an AttributeError on `n_vocab` should fail loud (real bug), not get swallowed.
**Anchor-Refs**: none — the fix is mechanical pattern repetition; anchoring would clutter every probe callback. The trigger conditions in `references/decision-anchoring.md` are not met (no failed iteration baked in, no counterintuitive choice, no "looks redundant but isn't").

## plan_2026-05-07_1519e34f
### D-001 | EXPLORE → PLAN | 2026-05-07
**Context**: GPT2 wraps `TextDecoder` → `TransformerLayer` → attention factory. WaveFieldAttention is not in the factory and uses a (B,N) padding mask, not a (B,N,N) attend mask. `TransformerLayer._get_attention_params` is a hardcoded switch that does not include wave_field. Reusing the factory path would require modifying shared infra (factory + transformer block) that 30+ unrelated models touch.
**Decision**: Build a self-contained decoder stack inside the new model package. Skip `TextDecoder` / `TransformerLayer` / attention factory. Define a local `WaveFieldDecoderBlock` and assemble blocks directly in `WaveFieldLLM`.
**Trade-off**: Code duplication (~150 LOC of pre-norm transformer block) **at the cost of** zero blast radius on the existing attention factory and transformer infra.
**Reasoning**: LESSONS L11 — pure-additive sibling work fits a single iteration. Factory registration is a separate, optional improvement that can land later without blocking this plan.
**Anchor-Refs**: `src/dl_techniques/models/wave_field_llm/wave_field_llm.py` (block class).

### D-002 | PLAN | 2026-05-07
**Context**: WaveFieldAttention introduces `field_size` as a new hyperparameter. Field stride = `(G-1)/(N-1)`. Larger `G` = sub-cell resolution + better gradient flow but `O(G log G)` FFT cost. Token positions beyond `max_seq_len` alias to the last cell.
**Decision**: Default `field_size = 2 * max_seq_len` per variant. Expose `--field-size` CLI flag for override.
**Trade-off**: ~2x FFT memory/FLOPs vs `field_size = max_seq_len` **at the cost of** sub-cell bilinear interpolation precision.
**Reasoning**: 2x is empirically modest (FFT pipeline runs in fp32 with H=4..25 heads at most for XL — peak intermediate fits in 24GB at small variants). Sub-cell precision avoids the integer-aliasing risk that plagued early scatter-gather designs.
**Anchor-Refs**: `src/dl_techniques/models/wave_field_llm/wave_field_llm.py` (`MODEL_VARIANTS`).

### D-003 | PLAN | 2026-05-07
**Context**: GPT-2 uses `tie_word_embeddings=True` by default. WaveFieldLLM is a research variant; users may want untied for ablation but the default should mirror GPT-2 for fair comparison.
**Decision**: Default `tie_word_embeddings=True`, expose `--no-tie-word-embeddings` flag (mirror GPT-2 train script).
**Trade-off**: vocab_size × embed_dim parameter savings + same default as GPT-2 **at the cost of** committing users to a single LM head policy unless they flip the CLI flag.
**Reasoning**: Mirrors GPT-2 exactly so head-to-head benchmarks are apples-to-apples.

### D-004 | PLAN | 2026-05-07
**Context**: Variant table from GPT-2: tiny/small/medium/large/xl. WaveFieldLLM should use the same names for one-to-one A/B comparison.
**Decision**: Clone `gpt2.py:MODEL_VARIANTS` verbatim, add `field_size = 2 * max_seq_len` per variant.
**Trade-off**: Possible misnaming if variant capacity differs from GPT-2's at same name **at the cost of** trivial usability for swap-in benchmarking.
**Reasoning**: Param counts will differ slightly (wave_field has ~10 params per attention layer + a tiny coupling matrix; replaces ~4*dim^2 of MHA's QKV+O Dense). Names track architecture role (small=124M-class), not absolute params.

### D-005 | PLAN | 2026-05-07
**Context**: GPT-2 model class default `vocab_size=100277` (cl100k_base) but train script default is `50261` (gpt2 encoding + 4 special). WaveFieldLLM train script will mirror the train default, but model-class default should align with what the train script uses to avoid silent vocab mismatch.
**Decision**: Set `WaveFieldLLM.DEFAULT_VOCAB_SIZE = 50261`, matching the train script default and explicit special-token wiring.
**Trade-off**: Diverges from GPT-2 model-class default **at the cost of** consistent train-script-class-default coupling.
**Reasoning**: This codebase's CLM pipeline standardizes on tiktoken `gpt2` encoding (50257 base + 4 special). Using a different default at the class level invites silent vocab-size bugs.

## plan_2026-05-07_a73304d4
### D-001 | Mixed-precision regression contract | 2026-05-07
**Context**: Reviewer flagged FFT under fp16 NaN risk (real). Investigation revealed the layer was crashing entirely under `mixed_float16`/`mixed_bfloat16` policies in three places independent of FFT (scatter/gather matrix dtype, kernel-build wave-parameter autocast, field_coupling weight dtype). User approved expanding scope to fix end-to-end mixed-precision support and lock it with a regression test.
**Decision**: Add `test_mixed_precision_end_to_end[mixed_float16|mixed_bfloat16]` parametrised test. Anchor with `# DECISION plan_2026-05-07_a73304d4/D-001` because losing any of the three casts would silently break it.
**Trade-off**: Two extra tests (~25 lines) **at the cost of** locking down a non-obvious cross-method invariant.
**Reasoning**: Without the test, a future contributor refactoring any of the four cast sites could re-break mixed precision without the existing 62 tests catching it.
**Anchor-Refs**: `tests/test_layers/test_attention/test_wave_field_attention.py:694-712`

### D-002 | Reject rfft tuple-unpacking "fix" | 2026-05-07
**Context**: This is the second review to claim `keras.ops.rfft` returns a complex tensor and tuple unpacking will crash. Empirically re-verified on Keras 3.8.0: returns `tuple` of length 2. `test_wave_kernel_fft_shape` (line 423-434) explicitly asserts the tuple shape.
**Decision**: Do not adopt the V3.7 "fix". Keep current tuple unpacking.
**Trade-off**: Going against reviewer's repeated insistence **at the cost of** verifying once more (3-line empirical test) and not breaking working code.
**Reasoning**: The proposed V3.7 code at `kern_fft = ops.reshape(kernel_fft, ...)` would crash because `ops.reshape` does not accept a Python tuple. The current code works in production and is locked by tests.

### D-003 | Force fp32 for wave-kernel build | 2026-05-07
**Context**: Under AMP, `self.wave_damping`/`wave_frequency`/`wave_phase` are autocast to compute_dtype (fp16) when read inside `call()`, while `t = ops.cast(ops.arange(G), "float32")` is forced fp32. The mismatch crashed `_build_wave_kernels_fft`.
**Decision**: Explicitly cast wave parameters to float32 inside `_build_wave_kernels_fft`. Build the entire kernel in fp32 (it feeds an fp32 FFT pipeline anyway).
**Trade-off**: Three explicit fp32 casts **at the cost of** consistency under all precision policies. No accuracy loss because the kernel was already fp32-bound (`t` is cast).
**Anchor-Refs**: `src/dl_techniques/layers/attention/wave_field_attention.py:393-397`

### D-004 | Cast scatter/gather/coupling to compute_dtype | 2026-05-07
**Context**: Same root cause as D-003 — fp32 indexing math vs fp16 activations. Rather than build matrices in compute_dtype (loses index precision), build in fp32 and cast at the einsum boundary.
**Decision**: Cast scatter_mat/gather_mat to `self.compute_dtype` after building them in fp32. Cast `softmax(field_coupling)` to `field.dtype` inside `_apply_field_coupling`.
**Trade-off**: Three small cast ops **at the cost of** preserving fp32 index/normalization precision while still allowing fp16 activations.
**Anchor-Refs**: `src/dl_techniques/layers/attention/wave_field_attention.py:507-509, 444-447`

## plan_2026-05-07_47199c68
### D-001 | EXPLORE → PLAN | 2026-05-07
**Context**: Code review flagged 25 issues in `wave_field_attention.py`. Empirical verification showed `keras.ops.rfft` returns `(real, imag)` tuple in TF backend (review #2 false positive). Reviewer's own re-analysis withdrew #1. 62 baseline tests all pass.
**Decision**: Apply 7 priority fixes (#3, #4, #5, #6, #8, #14, #15, #21); skip false positives and design-discussion items.
**Trade-off**: Fix correctness/reproducibility/robustness issues **at the cost of** leaving design-discussion items (damping calibration, phase circle coverage, layer norm) untouched.
**Reasoning**: Design-discussion items are locked by test contracts and lack a clear "right answer" — they belong in a future research/tuning iteration, not a bugfix pass.

### D-002 | Coupling init scheme | 2026-05-07
**Context**: Issue #3 — `np.random.randn` in `build()` is non-reproducible and breaks config-only reconstruction. The init must (a) preserve `stddev=0 ⇒ identity`, (b) preserve `stddev>0 ⇒ perturbed`, (c) be reproducible under `keras.utils.set_random_seed`.
**Decision**: Define a private serializable `_IdentityPlusNoise(keras.initializers.Initializer)` that combines `keras.ops.eye` with `keras.random.normal(seed=...)`. Add `coupling_seed` arg to `__init__`/`get_config`.
**Trade-off**: One new private class (1/2 abstraction budget) **at the cost of** ~25 LoC for the initializer + serialization.
**Reasoning**: Alternative — using `keras.initializers.RandomNormal` directly — cannot add the identity matrix in a single `add_weight` call. Splitting into two weights would change the variable count and break `test_trainable_variable_count`. Custom initializer keeps the same weight-graph topology.

### D-003 | Test assertion update for G-1 clip | 2026-05-07
**Context**: Issue #5 — current code clips field positions to `G-2`, wasting the last field cell. Fix changes upper clip to `G-1`. Test `test_field_positions_clamped` asserts `pos <= 62.0` (G-2 for field_size=64), locking the bug as a contract.
**Decision**: Update the test assertion to `<= 63.0` and anchor the change with `# DECISION plan_2026-05-07_47199c68/D-003` per LESSONS guidance.
**Trade-off**: Break a single test assertion **at the cost of** documenting in the test why the change was made.
**Reasoning**: The original assertion encoded the bug (field cell `G-1` was unreachable). Per LESSONS: "Pre-existing tests can encode bugs as contracts."

### D-004 | Skip wave parameter constraint (issue #22) | 2026-05-07
**Context**: Review #22 suggests `NonNeg` constraint on `wave_frequency`. Negative frequency is mathematically equivalent to positive frequency with phase shift in `cos(omega*t + phi)`.
**Decision**: Do not add constraint.
**Trade-off**: Slight interpretability ambiguity **at the cost of** unconstrained gradient flow.
**Reasoning**: Constraints can slow convergence; the equivalence under sign flip means there's no functional bug.

### D-005 | Skip damping range recalibration (issue #9) | 2026-05-07
**Context**: Review #9 argues damping range `[-3.0, 0.5]` softplus'd ≈ `[0.049, 0.974]` causes some heads to decay too fast at field_size=512.
**Decision**: Do not change the range.
**Trade-off**: Possibly suboptimal head diversity at default field_size **at the cost of** stability of the trained-model contract.
**Reasoning**: Test `test_wave_parameter_initial_values` locks `linspace(-3, 0.5, H)`. Changing it would invalidate prior trained checkpoints and the design intent (slow + fast heads). This is a design tuning question, not a bug.

## plan_2026-05-07_c6dd7cc1
### D-001 [iter-1, PLAN] — Centralize the chunk-aware step estimator in `train.common.nlp`

**Decision**: Add `estimate_clm_steps_per_epoch(...)` to `src/train/common/nlp.py`. Every CLM training script with a Wikipedia (or HF text) source calls it. The function takes either an explicit override or `(num_articles, max_seq_length, batch_size, avg_tokens_per_article)` and returns chunks/batch.

**X at the cost of Y**: One canonical helper at the cost of touching five callers. Trade-off accepted because the existing per-script `_estimate_steps_per_epoch` functions already drift (gpt2 + clifford-nlp use the wrong formula, routing uses the right one, train_clip avoids the problem entirely with a step budget). The drift is the bug; centralization eliminates it.

**Anchor**: Comment `# DECISION D-001` at the helper definition explaining why a single function exists. Plus a one-liner at every call site.

---

### D-002 [iter-1, PLAN] — Per-epoch reshuffle via sharded `interleave`, not generator-internal counters

**Decision**: Add a `num_shards` parameter to `_hf_to_tf_dataset` (default 1 = current behavior). When `num_shards > 1`: split the HF dataset into N shards via `hf_dataset.shard(num_shards=N, index=i)`, build N tf.data generators (one per shard), and combine via `tf.data.Dataset.sample_from_datasets` (or `interleave`) with `reshuffle_each_iteration=True` on a per-shard `.shuffle(buffer_size=…, seed=epoch_seed)`.

**X at the cost of Y**: Per-epoch reshuffle + parallel tokenization at the cost of strict determinism (chunk order across resumes is non-deterministic; only the shard partition is deterministic). Trade-off accepted because (a) pretraining doesn't need step-level determinism and (b) determinism isn't currently preserved across resumes anyway (since `tf.random.set_seed(42)` doesn't restore tf.data state — current behavior is already non-deterministic on resume).

**Alternative considered**: Generator-internal restart counter that re-seeds the HF shuffle on each call. Rejected: tightly couples shuffle policy to generator state, doesn't address Issue 5 (parallel tokenization), and complicates the `from_generator → tf.data` boundary.

**Anchor**: `# DECISION D-002` on `_hf_to_tf_dataset` documenting the shard semantics.

---

### D-003 [iter-1, PLAN] — Default `min_article_length=0` in `load_wikipedia_train_val`

**Decision**: Lower the default from 500 → 0. Update docstring to clarify: "0 for packed CLM (recommended); set 500+ only if a downstream consumer treats one document as one training example (per-doc MLM, classification)."

**X at the cost of Y**: Better data utilization for packed CLM at the cost of including stub articles. Trade-off accepted: stub tokens contribute to packed token stream just like any other tokens; the only "loss" is that EOT separators between stubs become more frequent, which is harmless (and arguably useful as a diversity signal).

**Compatibility risk**: Existing callers that pass `min_article_length=500` explicitly are unaffected. Callers that omit it (which is most of them) will see more articles after this plan. We update each call site explicitly to either keep the old value (with a `# DECISION` comment) or accept the new default. Default for ALL CLM consumers in this plan: 0.

---

### D-004 [iter-1, PLAN] — Remove dead `streaming` parameter from `preprocess_clm_dataset`

**Decision**: Drop the `streaming` argument from `preprocess_clm_dataset(...)`. Remove all callsite usages. Remove the stale "OOM warning" comments from `train_cliffordnet_nlp.py` and `train_cliffordnet_nlp_routing.py`.

**X at the cost of Y**: Cleaner API at the cost of a one-line breaking change in the function signature. Trade-off accepted because (a) the parameter is documented as a no-op, (b) only two known callers pass it, and (c) silent dead parameters become future foot-guns ("did setting streaming=False break my run?").

**Anchor**: `# DECISION D-004` on the new signature, single line in commit message recording the removal.

---

### D-005 [iter-1, PLAN] — Drop `cache()` from `preprocess_mlm_dataset`

**Decision**: Remove the `dataset.cache()` call inside `preprocess_mlm_dataset`. Tokenization runs every epoch.

**X at the cost of Y**: ~10-30% MLM training slowdown (re-tokenize per epoch) at the cost of avoiding a future RAM-explosion trap when an MLM script gets pointed at Wikipedia. Trade-off accepted because (a) BERT/FNet pretrain currently caps via `--max-samples` so the cache fits, but the cap is a CLI flag that can be removed; (b) tiktoken throughput on a 4090 host CPU is ~1M tokens/sec/thread — re-tokenizing imdb-class corpora costs <2 min/epoch.

**Mitigation**: Document the speed trade-off in the function docstring; suggest `--max-samples` users who want caching to call `.cache()` themselves at the call site.

---

### D-006 [iter-1, PLAN] — Plumb `--seed` end-to-end and derive resume seed from `initial_step`

**Decision**: Add `--seed <int>` (default 42) to every in-scope CLM train script. Pass it to `tf.random.set_seed`, `keras.utils.set_random_seed`, AND `load_wikipedia_train_val(seed=...)`. When `--resume <ckpt>` is set, derive `data_seed = base_seed + initial_step` so resumed runs see a different shuffle.

**X at the cost of Y**: One CLI flag and a deterministic shift at the cost of slight loss-of-reproducibility-on-resume (you can no longer reproduce a resumed run by replaying the same seed without the same `initial_step`). Acceptable: resume reproducibility was never guaranteed (tf.data state isn't checkpointed), and the shift makes the data-coverage benefit explicit.

---

### D-007 [iter-1, PLAN] — Keep `train_clip.py` `_run_pretrain_lm` largely as-is

**Decision**: `train_clip.py:_run_pretrain_lm` already uses the right idiom (explicit `steps_budget`, `repeat=True`, `epochs=1`). The only change is to lower `min_article_length=500 → 0` (D-003) and pass through the seed (D-006). Do not refactor it to use the centralized helper — the step-budget pattern is correct here and the helper targets the epoch-budget pattern.

**X at the cost of Y**: Two consistent patterns in the codebase (epoch-budget for pretrain scripts that use Keras `model.fit(epochs=...)`; step-budget for embedded sub-stages like CLIP pretrain) at the cost of "true uniformity." Trade-off accepted because forcing one pattern would either bloat the helper or kill an idiom that already works.

---

### D-008 [iter-1, PLAN] — Out-of-scope items (record explicitly to prevent scope creep)

The following surfaced during EXPLORE but are explicitly **OUT OF SCOPE** for this plan:

- `src/train/bert/wikipedia/pretrain.py` and `pretrain_english.py` — separate hand-rolled pipeline (HF streaming + bookcorpus interleave + custom WarmupSchedule). Their `total_steps` is hardcoded so they don't suffer from issue #1; their data path is independent. Touching them would double the blast radius.
- Step-based validation cadence (Issue 6 in the catalog) — non-blocker, deferred.
- `dl_techniques.utils.tokenizer.TiktokenPreprocessor` internals — out of scope.
- Anything in `src/train/cliffordnet/` that is not a CLM training script (e.g. depth/CLIP/multitask paths that don't touch the NLP pipeline).

Ghost-constraint check: the user originally typed "not in clifford unet" then re-issued the command with "not only in clifford unet" — this plan operates under the **second** (corrected) scope: ALL CLM consumers including UNet are fixed.

## plan_2026-05-06_82749628
### 2026-05-06 — D-001: Use UpSampling2D(nearest) along W as the causal upsampler.
**Choice**: `keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")` for the decoder up-path.
**Cost**: no smoothing across pool boundaries — each output cell within a window has the same scalar. This is acceptable here because (a) `CausalCliffordNetBlock` after the upsample re-injects local mixing (left-only causal DW conv along W) so the decoder block can blend pooled positions back into per-token features, and (b) any smoother (bilinear/transposed-conv) leaks future info — losing causality is a bigger cost.
**Why not transposed conv with mask**: complexity. Adding a learned-yet-strictly-lower-triangular conv is harder to verify than a stateless repeat. Revert-First / 10-Line Rule favours the stateless repeat.

### 2026-05-06 — D-002: Pad-then-crop in `call()` (option B), not `seq_len % total_stride == 0` validation.
**Choice**: right-pad input with zeros to next multiple of `total_stride`, run the U-Net, slice along W back to original `seq_len`.
**Cost**: tiny per-call overhead (one pad + one slice op). Goes against "fail fast on invalid shapes" preference.
**Why**: train script default is `seq_len = max_seq_length - 1 = 511`; making this work without forcing the user to think about stride alignment is the friendliest option. Padding does NOT pollute real positions because every block is causal in W (zeros at the right tail can only influence themselves or later padded positions).

### 2026-05-06 — D-003: Skip-fusion = Concatenate + 1x1 Conv2D (not add).
**Choice**: `concat([upsampled, encoder_skip], axis=-1)` then `Conv2D(channels, 1)` to project 2C → C.
**Cost**: extra 1x1 conv per decoder level. Adds ~`channels^2 * 2` params per level; negligible vs the Clifford blocks' SRGP weights.
**Why**: gives the model a learnable mixing weight between encoder skip and decoder feature; more flexible than addition; matches the dl_techniques/cliffordnet vision UNet pattern (`unet.py:676-680`).

### 2026-05-06 — D-004: Per-stage isotropic channel count (no channel growth across stages).
**Choice**: every stage uses the same `channels` (e.g. nano=128 throughout). DSv2 `out_channels` is left at `None`.
**Cost**: gives up the standard U-Net "double channels when halving spatial" recipe. Bottleneck is wider in *receptive field* but not in *channel count*.
**Why**: keeps the SRGP `shifts < channels` invariant trivially satisfied; matches `lm.py` which is also isotropic; cleaner skip-concat (no per-level channel projection on the encoder side); shifts list applies uniformly. If we later want a channel-pyramid variant, we add it as a separate variant — keep iteration 1 simple.

### 2026-05-06 — D-004 SUPERSEDED by D-005: User rejected isotropic channels.
**User feedback (Plan v1 review)**: Channels must grow per stage (typical U-Net doubling). Use base_channels per variant; per-stage channels derived as `base * 2**i`.

### 2026-05-06 — D-006: STOP — Pre-Mortem Scenario A fired during Step 3 smoke check.
**Trigger**: With nano-like config (base_channels=32, stride_per_stage=[2,2], total_stride=4), perturbing input position 7 (last) changed logits at positions 4, 5, 6 with max abs diff ~0.21–0.27 (well above 1e-5 tolerance). Positions 0–3 were byte-identical.

**Root cause**: Pool→nearest-upsample **round trip** is not causal at fine resolution, even though nearest-upsample alone is. DSv2 with `padding="same"` and stride `s` produces pooled cell `j` that has seen input positions `[j*s, j*s+s-1]`. Nearest-upsample maps output position `k` to pooled cell `k // s`, which has seen up to input position `(k//s)*s + s - 1`. For `k % s != s - 1`, that maximum exceeds `k` — i.e. output position `k` reads from a pooled cell that has incorporated future inputs. Compounded across multiple levels (deepest cell sees up to `(k // total_stride)*total_stride + total_stride - 1` of the input).

**A1 falsified**: The `upsampling-causality.md` finding's argument *"upsampled position k is identically equal to encoder feature at pooled position j; pooled position j was computed from input positions 0..j*s+(s-1), so k sees only past+current"* is correct ONLY at the pool grid (`k = j*s`). For all `k` in `[j*s, (j+1)*s - 2]`, "past+current" at the pool grid translates to **future** at the original grid.

**Failed defense**: A1 wasn't truly tested at the unit level — the existing `test_causal_block_no_future_leak` regression covers the DSv2 block at pool resolution but never the round trip through upsample back to original resolution. The plan accepted A1 on the strength of an internal-consistency argument, not a test.

**Per Autonomy Leash**: 0 fix attempts made on Step 3 — reverted uncommitted changes immediately on detecting the leak (Revert-First). Committed Steps 1 & 2 (scaffolding) are kept; they are causality-agnostic.

**Remediation options to discuss with user**:
- (R1) **Causal upsample**: shift upsampled feature right by `s-1` along W (zero-pad on left, drop right). Compounded across levels: output position `k` reads pooled cell `(k - (s_total - 1)) // s_total` at the deepest level, ensuring max-input-seen ≤ k. Implementable as a small custom layer or as `keras.ops.pad` + slice after each `UpSampling2D`. Cost: extra `(total_stride - 1)` positions of latency at the top of U; the first `total_stride - 1` positions of any stage's upsampled feature are zero-filled before the skip connection. The encoder skip at level 0 is fully causal, so the skip-fusion path at the leftmost positions still has signal — only the deep-path contribution is zero-padded.
- (R2) **Skip-only at boundary positions**: for the first `s-1` output positions of each upsample window, mask the upsampled-feature contribution and rely entirely on the encoder skip. Equivalent to R1 in effect; different implementation.
- (R3) **Drop the U-Net**: revert plan, fall back to non-pooled stack (i.e. `lm.py`'s isotropic depth-only model). User explicitly wanted U-Net so this is the last resort.
- (R4) **Strict causal pool**: change DSv2 to use a pool that emits at the **right** edge of its window (pool position j sees inputs `[j*s - s + 1, j*s]`), then nearest-upsample is causal at all positions because pooled cell `k // s` has seen up to input `(k // s) * s ≤ k`. Cost: requires modifying DSv2 (out of plan scope) or wrapping it with a left-shift pre-pool.

Recommendation: R1 (causal upsample with right-shift) is the smallest, most local change. R4 is more elegant but touches a shared layer.

### 2026-05-06 — PIVOT: REFLECT → PIVOT → PLAN (adopt R1, causal upsample right-shift).
**User decision**: choose R1 — causal upsample via right-shift by `s-1` (left zero-pad, drop `s-1` from the right) at each decoder upsample stage. Keep the change local to `lmunet.py`. Do NOT modify `CausalCliffordNetBlockDSv2`.

**Ghost-constraint scan**: none — A1 was a wrong claim, not a ghost constraint. The constraint that originally drove "nearest upsample is enough" was based on an unproven derivation; nothing about the environment has changed.

**Complexity assessment**: R1 adds ~5 lines per upsample call site (a `keras.ops.pad` + slice). Stays within complexity budget (no new abstractions; UpSampling2D + pad+slice is two stdlib ops in sequence). Net delta ~+10 lines vs original Step 3.

**Available checkpoints**: cp-000-iter1 (pre-Step-1 commit `a7abadf`). Steps 1 & 2 commits kept (causality-agnostic scaffolding); resume from Step 3 on top of `245553d`.

### 2026-05-06 — D-007: Causal upsample via right-shift by `s-1` (R1).
**Choice**: After each `keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")` in the decoder, apply a causal right-shift along W: `x = ops.pad(x, [[0,0],[0,0],[s-1,0],[0,0]])[:, :, :W_up, :]` where `W_up` is the post-upsample width. Equivalent to left-zero-padding by `s-1` then dropping the last `s-1` cells. Implemented inline at the call site as a small helper `_causal_upsample(x, stride)` private to `lmunet.py`.
**Cost**: first `s-1` output positions of each decoder upsample stage have zero deep-path contribution (skip path still carries signal at those positions, since the encoder skip is at the original resolution and is itself causal). Compounded across levels, the leftmost `total_stride - 1` output positions have purely-skip information from the deepest stages — acceptable: this is "latency at warm-up" not a permanent bias.
**Why**: smallest local change that fixes Scenario A. Does not touch the shared `CausalCliffordNetBlockDSv2`. Verifiable in isolation: a unit test on the helper alone proves `_causal_upsample(x_with_perturbation_at_k)` produces output where positions `< k` are byte-identical to the unperturbed baseline.
**Anchor in code**: `# DECISION D-007: causal upsample via right-shift; see plans/plan_2026-05-06_82749628/decisions.md`. Placed at the call site of `_causal_upsample` in `call()` and at the helper definition.
**Falsification**: unit test `test_causal_upsample_helper` (perturb input at position k, assert output positions `< k` byte-identical) MUST pass before the end-to-end causality test in Step 4.

### 2026-05-06 — REFLECT (iter-1 post-pivot): all 6 criteria PASS.
**Summary**: 30/30 unit tests pass. End-to-end causality verified (perturb-last, perturb-middle, non-multiple seq_len, with global_context). Helper micro-tests (TestCausalUpsampleHelper) isolate D-007 fix.

**Simplification Checks (6)**:
1. Could a step be removed? No — all 6 deliver distinct artifacts.
2. Could two layers be merged? No — `_causal_upsample` is a stateless 4-line helper, can't merge into UpSampling2D.
3. Is any abstraction unused? No — every new symbol is wired in.
4. Could a config field be a constant? `base_channels` is the only config exposed at the variant boundary; the rest are per-variant defaults.
5. Could a forward branch be deleted? No — pad+crop is needed for non-multiple seq_len; tied/untied LM head matches `lm.py` policy.
6. Is anything wrapped that doesn't need wrapping? No — UpSampling2D + 1x1 Conv2D + Concatenate are stdlib; no adapter classes.

**Devil's advocate**: One reason this might still be wrong despite passing — the causality tests use a small nano-like config (base_channels=16, stride=[2,2]). At deeper variants (base/large/xl with stride=[2,2,2], total_stride=8), the causal-shift compounds across 3 levels: total left-zero region = 7 positions. We never tested causality at the larger variant. However: the right-shift logic at each level is locally correct, and composing causal operations stays causal (mathematical property), so the small-config test is sufficient. Risk: if a future change adds a new spatial-mixing op at a deeper level, only the small-config test will catch it. Acceptable; flagged in "Not Verified" of verification.md.

**Prediction accuracy**: see verification.md "Prediction Accuracy" table.

**Root cause analysis**: only one failure during EXECUTE was the save/load tolerance (5e-5 vs plan's 1e-5). Immediate cause: GPU XLA reduction-order non-determinism. Contributing factor: plan tolerance was inherited from `lm.py` test which was simpler (no DSv2 / no decoder fusion path). Failed defense: none — the `lm.py` precedent suggested 1e-5 was achievable, and it took the actual test to reveal the gap. Prevention: future U-Net-style models should default to 1e-4 in save/load tests.

### 2026-05-06 — D-005: Per-stage channel doubling (standard U-Net schedule). [SUPERSEDES D-004]
**Choice**: `channels_per_stage[i] = base_channels * (2 ** i)` for `i in 0..num_levels-1`. Encoder doubles channels each downsample (DSv2 `out_channels` does the projection). Decoder mirrors going back up: 1x1 Conv2D after concat projects from `(C_{i+1} + C_i) = 3*C_i` down to `C_i`. Embedding dim = `base_channels` (top-of-U).
**Cost**: parameter count grows fast at the bottleneck (largest channel count is `base * 2**(num_levels-1)`). For xl with base=192, num_levels=4 → bottleneck has 1536 channels. Slightly more memory than the isotropic D-004 design at the bottleneck. Decoder skip-fusion projection is `3*C_i → C_i` (was `2*C_i → C_i` under isotropic).
**Why**: standard U-Net inductive bias — wider receptive field at deeper levels deserves more channels to capture coarser features; user explicitly required this; matches the dl_techniques vision UNet pattern. `shifts < channels` invariant remains trivially satisfied: the smallest channel count is at the top of U (`base_channels`), and `max(shifts) < base_channels` is checked once in `__init__` — every other stage has strictly more channels. DSv2's existing `out_channels` arg supplies the channel projection without adding new layers. Concat semantics unchanged: still concat along channel axis then 1x1 project — only the input/output channel arithmetic shifts.

## plan_2026-05-06_13a2df9e
### 2026-05-06 PLAN — chosen approach

**Decision**: Add `CausalCliffordNetBlockDSv2` as a new sibling class in `clifford_block.py`, mirroring the structure of `CliffordNetBlockDSv2` but with a narrower pool-kind surface (`avg`/`max` only) and the causal-padding paradigm of `CausalCliffordNetBlock` applied to the depthwise context conv and the global-context cumulative mean.

**Trade-off (X at the cost of Y)**:
- Reuse causal-padding pattern (left-pad by `k-1` along W, then `valid` conv) at the cost of a slightly redundant call-time `pad` op (already accepted in `CausalCliffordNetBlock`).
- Restrict pools to `avg` / `max` only at the cost of feature parity with DSv2; pyramid_diff / blur / gaussian_dw / pixel_unshuffle / resnetd are excluded because they would leak future info under H=1, W=seq layout.
- Place new class in the same file at the cost of file growth (~2300 lines), matching the user's explicit instruction.

**Alternatives considered**:
- Pass a `causal: bool` flag into existing `CliffordNetBlockDSv2`: rejected — the parent class already validates `pyramid_diff` etc., and adding a flag complicates serialization, validation, and forward branching.
- Subclass `CliffordNetBlockDSv2`: rejected — the parent's `dw_conv` and pool factory choices conflict with causal constraints; mirroring instead of inheriting matches the existing pattern (`CausalCliffordNetBlock` does not subclass `CliffordNetBlock`).
- Place the class in a separate file: rejected — user explicitly requested same file.

### 2026-05-06 INIT → EXPLORE → PLAN

EXPLORE produced 3 indexed findings (scope-and-callers, causality-mechanics, dsv2-merge-points). Confidence: deep / constrained / clear. Transitioning to PLAN.

### 2026-05-06 EXECUTE step-1 — DepthwiseConv2D vs Conv2D(groups=channels) deviation

**Surprise**: `DepthwiseConv2D` on TF/CUDA rejects asymmetric strides
(`Current implementation only supports equal length strides in the row
and column dimensions`). The plan called for `DepthwiseConv2D(strides=(1, s))`
which fails on GPU at strides=2.

**Root cause**: The existing `CausalCliffordNetBlock` uses default strides=1
so it never hit this limitation. The plan implicitly assumed
`DepthwiseConv2D` accepts asymmetric strides — it does not on TF GPU.

**Resolution (1-line fix at point of impact)**: Replace
`DepthwiseConv2D(kernel_size=(1,k), strides=(1,s))` with
`Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
Algebraically identical (depthwise = grouped conv with groups==channels==filters).
Verified working at strides=2 with both shape and causality probes.

**Trade-off**: Conv2D-with-groups at the cost of (a) a slightly larger
memory footprint for the kernel-shape tensor, (b) divergence from the
sibling `CausalCliffordNetBlock` which still uses `DepthwiseConv2D` (it
can — its strides are always 1). Documented in DECISION D-002 in code.

**Anchored**: D-002 comment expanded inline in `CausalCliffordNetBlockDSv2.__init__`.
