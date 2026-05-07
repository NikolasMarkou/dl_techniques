# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

## plan_2026-05-07_3f461682
### Index
- [F-001 LLM trainer inventory & current metric setup](plan_2026-05-07_3f461682/findings/01-llm-trainer-inventory.md) — 5 in-scope Pattern-3 CLM trainers (gpt2 pretrain/finetune, cliffordnet_nlp, _unet, _routing, wave_field_llm) all share `metrics={"logits": ["accuracy"]}`. qwen/nano_vlm/blt/bert/fnet are out of scope (custom trainers / different objectives).
- [F-002 Existing metrics infrastructure](plan_2026-05-07_3f461682/findings/02-existing-metrics-infra.md) — `dl_techniques.metrics.perplexity_metric.Perplexity` already exists, drop-in compatible. No BPC/BPW/BLEU exists. `_post_generate_hook` is an empty extension point on every probe.
- [F-003 Integration shape](plan_2026-05-07_3f461682/findings/03-integration-shape.md) — DRY recipe: one `dl_techniques/metrics/llm_metrics.py` module + one `train/common/nlp.py::build_clm_metrics` helper. Per-trainer delta = ~3-5 LOC. Optional Self-BLEU/distinct-N/aggregate-tok/s via `_post_generate_hook` override.

### Key Constraints

### Hard
- **DRY single-source-of-truth**: one shared metric module + one builder helper; no metric math copy-pasted into the 5 trainers (user explicit: "make good use of the common").
- **Reuse `Perplexity` from `perplexity_metric.py`** — do not re-implement.
- **Output dict key MUST stay `"logits"`** (SYSTEM atlas invariant; loss + train wrappers + probe pivot on it).
- **Keras 3 idioms**: `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, full `get_config()` round-trip.
- **`ignore_class`** for PPL must mirror each trainer's loss `ignore_index`. Default for `MaskedCausalLMLoss` is `-1`; packed-CLM pipeline does not actually emit ignore tokens, so this is a no-op in practice but should be wired for symmetry / correctness if labels ever change.
- **Mixed precision**: keep PPL accumulator fp32 (default `add_weight` dtype). AMP isn't currently enabled in CLM training, but write metric to be AMP-safe.
- **No full-training smoke**. Verify by unit tests + (only on user request) a few-step `model.fit` smoke.
- **No emojis, no print, single GPU, MPLBACKEND=Agg, .venv/bin/python** (project conventions).

### Soft
- BPC `chars_per_token` defaults to **4.0** for `gpt2` encoding on EN-Wikipedia (paper-ish constant). Configurable via `build_clm_metrics`.
- BPW `tokens_per_word` ~1.3. Less commonly reported. Recommend including but flagged optional via flag (or simply use BPT + BPC and skip BPW).
- Self-BLEU @ n=4 across probe outputs; distinct-2; aggregate tok/s. NLTK-free pure-Python implementations.
- Defer probe-class extraction (5x duplication) to a separate refactor plan — outside scope here.

### Ghost / out-of-scope
- True BLEU/ROUGE (need references) → offline eval harness only.
- Hallucination, Toxicity, Coherence → LLM-as-judge / classifiers, separate harness.
- Inference Latency / VRAM as live training-loop metrics → recommend deferring or once-per-epoch logger only.
- qwen, nano_vlm, blt, bert/fnet (MLM), CLIP — different shapes, defer.

### Exploration Confidence
- **Problem scope: deep** — every CLM trainer's `compile_model` and probe class read; existing metrics package mapped; tokenizer access confirmed; `ignore_index` semantics traced through `MaskedCausalLMLoss` and the packed-CLM pipeline.
- **Solution space: constrained** — DRY single-module + single-helper is the only design that satisfies the user's "make good use of the common" instruction without per-model duplication.
- **Risk visibility: clear** — main risk is `ignore_class` mismatch (mitigated by reading each trainer's loss config) + AMP-safety of accumulator (existing `Perplexity` is correct).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

- [CORRECTED iter-1] F-002 (`findings/02-existing-metrics-infra.md` line 38): "All five GenerationProbeCallback classes implement an empty `_post_generate_hook`" — actually only `gpt2/pretrain.py` and `wave_field_llm/pretrain.py` do. The 3 cliffordnet probes lack the extension point. See D-002 in decisions.md for resolution (add the extension to the 3 cliffordnet probes during EXECUTE).

## plan_2026-05-07_08aaf818
### Index
- `findings/reference-fix-shape.md` — exact code shape of the 1fe2088 fix; explains both halves.
- `findings/call-sites.md` — inventory of decode sites; confirms scope incl. routing-variant outlier and out-of-scope verification.
- `findings/per-file-patch-shape.md` — per-file insertion line numbers and routing-specific tuple-return shape.

### Key Constraints
- **Hard**: behaviour parity with 1fe2088 (mask + try/except). Each patch mirrors the reference exactly modulo the routing tuple return.
- **Hard**: `dl_techniques.utils.logger` only (no print).
- **Hard**: 4 reserved specials (50257..50260) baked in (`power_sampling.py:86` confirms). `tiktoken.get_encoding("gpt2").n_vocab == 50257`. Reference's `range(n_vocab, max(n_vocab+1, 50261))` covers exactly these four.
- **Hard**: routing returns `Tuple[str, int]` — both branches of try/except must keep this shape.
- **Soft**: one commit per file. Verify each with `python -m py_compile`; no smoke training (too expensive).

### Out of scope (verified)
- `src/dl_techniques/models/nam/tokenizer.py` — custom 21-symbol tokenizer, not tiktoken.
- `src/dl_techniques/models/cliffordnet/power_sampling.py:625` — pre-sample mask at line 207-209 makes bad ids impossible.
- `src/train/wave_field_llm/pretrain.py` — already fixed (1fe2088).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-07_1519e34f
### Index

- [F-001 gpt2.py architecture & attention wiring](plan_2026-05-07_1519e34f/findings/01-gpt2-architecture.md) — `GPT2(keras.Model)` wraps `TextDecoder` which wraps `TransformerLayer` stack with `create_attention_layer` factory. WaveFieldAttention is NOT in the factory; mask shape is (B,N) not (B,N,N); QKV+gate+output projections are internal. Conclusion: build a parallel decoder stack rather than retrofit `TextDecoder`.
- [F-002 src/train/gpt2/pretrain.py conventions](plan_2026-05-07_1519e34f/findings/02-train-gpt2-conventions.md) — Pattern 3 NLP CLM training script, ~95% generic. `TrainingConfig` dataclass + `StepCheckpointCallback` + `GenerationProbeCallback` + AdamW/warmup-cosine + tiktoken/Wikipedia + `MaskedCausalLMLoss`. Mirror file-by-file, swap model class and variant list.
- [F-003 WaveFieldAttention call signature & integration](plan_2026-05-07_1519e34f/findings/03-wave-field-attention-integration.md) — `(B,N,D)` in/out, optional `(B,N)` padding mask, causal-by-construction, internal QKV+gate+output, FFT in fp32 under AMP. New hyperparameter `field_size` (default `2*max_seq_len`). Trainable-var count == 10 is locked by tests.

### Key Constraints

### Hard
- `WaveFieldAttention` API is locked (62 tests). Do NOT modify the layer.
- Mask shape mismatch: WaveField uses `(B,N)`; standard `TransformerLayer` passes `(B,N,N)`. Cannot reuse `TextDecoder`/`TransformerLayer` without invasive changes that would touch unrelated attention types.
- Output dict key MUST be `"logits"` — train wrappers, `MaskedCausalLMLoss`, and `model.compile(loss={"logits": ...})` all key on it.
- Keras 3.8 / TF 2.18 idioms: `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger` (no `print`), full `get_config()` round-trip.
- Causality: WaveFieldAttention is causal by construction. No additional causal mask needed. Padding mask only.
- `dim % num_heads == 0`. `field_size > 1`. `max_seq_len > 0`.
- Tokens beyond `max_seq_len` alias to the last field cell — warn-only. So `max_seq_len` of the layer MUST equal the model's `max_seq_len`.
- AdamW WD only — no `kernel_regularizer=L2`.
- Single GPU jobs. `MPLBACKEND=Agg`. `.venv`.
- Direct import path: `from dl_techniques.models.wave_field_llm.wave_field_llm import WaveFieldLLM`.
- Required CLM CLI flags: `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume` (D-001..D-006 from plan_2026-05-07_c6dd7cc1).
- Resume seed shift: `data_seed = config.seed + initial_step`.
- `custom_objects` on `keras.models.load_model` must include both losses; `WaveFieldAttention`+`_IdentityPlusNoise` are auto-registered via decorator.
- Mirror gpt2.py variant ladder: `tiny / small / medium / large / xl`.
- Test scope: pytest only on the new module. Do NOT run `make test`.

### Soft
- `field_size = 2 * max_seq_len` default. Document trade-off (memory vs accuracy).
- GELU FFN with 4x expansion (matches GPT-2 reference).
- Pre-norm + residual structure.
- Variants: copy gpt2.py MODEL_VARIANTS 1:1, add `field_size` per-variant.
- Tests: mirror `tests/test_models/test_gpt2/test_gpt2.py`. Add `keras.Model`-wrapped save/load round-trip (LESSONS L48-49).

### Ghost (not present)
- "Register WaveFieldAttention in attention factory" — invasive. Defer to a separate plan; build self-contained stack instead.
- "Custom causal mask layer" — false; kernel is causal.
- "3-D attend mask" — false; WaveFieldAttention takes (B,N).

### Exploration Confidence

- **Problem scope**: deep — every reference file read in full (gpt2.py 417 LOC, wave_field_attention.py 591 LOC, pretrain.py 945 LOC, text_decoder.py 475 LOC). Mask shape mismatch confirmed.
- **Solution space**: constrained — single viable composition (sibling stack with WaveFieldAttention block). FFN type, variant ladder, tie/untie are 3 minor knobs.
- **Risk visibility**: clear — main risks: padding-mask under variable seq_len, max_seq_len mismatch, save/load round-trip with wave-kernel weights. All mitigated by prior-plan patterns.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-07_a73304d4
### Index
- F1: rfft tuple return RE-VERIFIED on Keras 3.8.0. Reviewer's V3.7 "fix" would crash because `ops.reshape` cannot accept a tuple. Same false claim as previous review's #2.
- F2: 62/62 tests pass on current commit `67745b5` (HEAD before this iteration).
- F3: 4 real optimizations identified — mixed-precision cast, 4D coupling einsum, single-transpose scatter/gather, `ops.softplus`.
- F4: Test contracts that gate the changes — `test_wave_kernel_fft_shape` (asserts tuple of len 2 — protects against accidentally adopting the false "fix"); `test_field_coupling_*` (output shape + near-identity correlation); `test_numerical_stability` (large/tiny inputs, no NaN/Inf).

### Key Constraints
- **Hard**: `keras.ops.rfft` returns 2-tuple `(real, imag)` on Keras 3.8.0 + TF 2.18 backend. Do not adopt the "complex tensor" fix.
- **Hard**: All 62 existing tests must remain passing.
- **Hard**: Weight topology must remain unchanged (`test_trainable_variable_count == 10`).
- **Soft**: Project convention — `dl_techniques.utils.logger` is imported unconditionally everywhere; do NOT add try/except fallback (user-confirmed scope).

### Corrections
*None.*

## plan_2026-05-07_47199c68
### Index

- [F1: rfft return type empirically verified](plan_2026-05-07_47199c68/findings/rfft-return-type.md) — Keras 3 TF backend returns `(real, imag)` tuple. Review issue #2 is a FALSE POSITIVE.
- [F2: Existing test suite all green (62/62)](plan_2026-05-07_47199c68/findings/test-baseline.md) — `pytest test_wave_field_attention.py` passes; tests encode current behavior as contract.
- [F3: Test contracts locking in current behavior](plan_2026-05-07_47199c68/findings/test-contracts.md) — Several tests will break if naive fixes are applied; identifies which assertions need updates.
- [F4: Issue triage — real vs false positive](plan_2026-05-07_47199c68/findings/issue-triage.md) — Of 25 review issues: ~10 real fixes, ~5 false positives (incl. #1, #2, #12, #18, #20), rest are design choices/minor.

### Key Constraints

- **Hard**: All 62 existing tests must remain passing (or be updated with explicit anchor and justification).
- **Hard**: Wave parameter init values are locked by `test_wave_parameter_initial_values` to `linspace(0, π, H)` for phase, `linspace(-3, 0.5, H)` for damping, `linspace(0.3, 4.0, H)` for frequency. Changes here require test update + decision anchor.
- **Hard**: `coupling_noise_stddev=0.0` must give exact identity (`test_coupling_noise_zero_gives_identity`). Any new init scheme must preserve this.
- **Hard**: `coupling_noise_stddev>0` must perturb away from identity (`test_coupling_noise_applied`).
- **Soft**: Keras 3 idiom — prefer `keras.initializers` over numpy random for reproducibility.
- **Soft**: `dl_techniques.utils.logger` for warnings, not `print`/`warnings`.

### Corrections
*None yet.*

## plan_2026-05-07_c6dd7cc1
### Index

- [01-pipeline-map.md](plan_2026-05-07_c6dd7cc1/findings/01-pipeline-map.md) — Producer/consumer map across the codebase. Lists every train script that touches `load_wikipedia_train_val`, `preprocess_clm_dataset`, `preprocess_clm_packed_dataset`, `preprocess_mlm_dataset`, `load_hf_text_dataset`, `create_warmup_lr_schedule`. **Scope clarification (user re-issued command)**: "fix them everywhere **not only** in clifford unet" — UNet is INCLUDED.
- [02-issue-catalog.md](plan_2026-05-07_c6dd7cc1/findings/02-issue-catalog.md) — Re-verified bug catalog with severity, exact LOC, recommended fix. 9 issues total: 7 confirmed from initial review, 1 downgraded (val cadence), 2 new (`MLM cache()` trap, `seed=42` magic).
- [03-design-precedents.md](plan_2026-05-07_c6dd7cc1/findings/03-design-precedents.md) — Two existing in-repo precedents that already do "the right thing": `train_cliffordnet_nlp_routing.py` (chunk-aware estimator + `--steps-per-epoch` override) and `train_clip.py` `_run_pretrain_lm` (explicit `steps_budget` + `repeat=True`). Centralization should hoist their pattern into `train.common.nlp`.

### Key Constraints

### Hard
- Existing `.keras` checkpoints must continue to load (resume must keep working). No changes to `MaskedCausalLMLoss` / `FocalCausalLMLoss` interfaces.
- `model.fit(epochs=E, steps_per_epoch=S)` is required when the train dataset has `.repeat()` applied (otherwise infinite loop).
- `tf.data.Dataset.from_generator` produces a non-cardinal dataset; `tf.data.experimental.cardinality()` returns `UNKNOWN_CARDINALITY`.
- Tiktoken releases the GIL during `encode_ordinary_batch`, so multi-thread tokenization is feasible without process forks.
- Single GPU jobs only (per LESSONS.md / project memory). No distributed-strategy work.
- `MPLBACKEND=Agg` for any training script invocation in this codebase.

### Soft
- Heuristic ~600 tokens/article post 500-char filter (used by `train_cliffordnet_nlp_routing.py`). With `min_article_length=0` the avg drops; we'll re-derive the constant.
- Shuffle buffer size (4096 today). Bigger is better but eats RAM (~3KB/chunk × 4096 ≈ 12 MB — trivial; could go 64k).
- Number of tokenization shards (4-8 reasonable on the dev box).
- Default `min_article_length` (currently 500 chars). For packed CLM, 0 is correct.

### Ghost
- The "streaming=True is REQUIRED ... otherwise OOM" comments in `train_cliffordnet_nlp.py:637-639` and `train_cliffordnet_nlp_routing.py` (and the `streaming` parameter on `preprocess_clm_dataset`) are vestiges of the pre-packed-pipeline era. They no longer apply since `preprocess_clm_packed_dataset` does not call `.cache()`.
- The hardcoded `4_850_000` article count was empirically correct for `min_article_length=500`. Once we lower the default to 0, the article count rises and the 600-tokens/article heuristic falls — total tokens are roughly invariant (~3-4B for EN Wikipedia 20231101), so the chunk count is what we should be estimating, not articles × tokens.

### Exploration Confidence
- Problem scope: **deep** (every consumer mapped, library code traced, two existing in-repo precedents found and read).
- Solution space: **constrained** (4 viable approaches per issue; recommended approaches map to existing code in this repo).
- Risk visibility: **clear** (resume compatibility is the main risk; sharded interleave changes determinism — explicit user-facing trade-off).

Ready for PLAN.

## plan_2026-05-06_82749628
### Index
- [causal-blocks-api.md](plan_2026-05-06_82749628/findings/causal-blocks-api.md) — CausalCliffordNetBlock (dim-preserving) and CausalCliffordNetBlockDSv2 (causal stride downsampler). Shapes, allowed kwargs, ctx_mode/pool restrictions.
- [upsampling-causality.md](plan_2026-05-06_82749628/findings/upsampling-causality.md) — keras.layers.UpSampling2D(size=(1, s), interpolation="nearest") is causal-safe. Bilinear / transposed-conv leak future. Tail right-pad + crop in call() solves the seq_len % total_stride != 0 case.
- [lm-and-train-mirror.md](plan_2026-05-06_82749628/findings/lm-and-train-mirror.md) — lm.py contract (variants ladder, from_variant, get_config, tie_word_embeddings, {"logits": ...} dict). Train script callbacks are model-agnostic — only model class + custom_objects + dataset/results-dir prefix change.

### Key Constraints

### Hard
- 4D layout (B, 1, seq_len, channels) end-to-end through encoder/decoder.
- Strict causality: changing input position k must leave outputs at all positions < k byte-identical (within fp tolerance).
- DSv2 ctx_mode restricted to {diff, abs} (no pyramid_diff); pool kinds restricted to {avg, max} (LESSONS L33).
- Upsample must be nearest along W only — no bilinear, no transposed conv, no sub-pixel.
- Output shape (B, seq_len, vocab_size) — must preserve full input length even when seq_len % total_stride != 0. Use right-pad + crop inside call().
- Output dict key MUST remain "logits" — MaskedCausalLMLoss and the train script assume this.
- Keras 3 conventions: @keras.saving.register_keras_serializable(), keras.ops, dl_techniques.utils.logger, full get_config() round-trip, no print.
- tie_word_embeddings flag — same default as lm.py (True). Keep output_bias add_weight when tied.
- Skip-connection fusion at SAME resolution as encoder skip — explicit Concatenate(axis=-1) followed by 1x1 Conv2D projection back to channels.
- Test scope: pytest only on the new model file (LESSONS L20: never make test).
- AdamW WD only — no kernel_regularizer=L2.
- Use .venv/bin/python. Commit per step. User pushes themselves.

### Soft
- Mirror lm.py and train_cliffordnet_nlp.py line-by-line where it doesn't conflict with U-Net structure.
- Variant ladder names: nano, mini, base, large, xl (1:1 with lm.py).
- Class name: CliffordNetLMUNet. File path: src/dl_techniques/models/cliffordnet/lmunet.py. Train script: src/train/cliffordnet/train_cliffordnet_nlp_unet.py.
- Default U-Net topology per variant: 3 stages (encoder + bottleneck + decoder), strides [2, 2], mirroring decoder. For deeper variants, allow 4 stages (strides [2, 2, 2]).
- Default upsampler: keras.layers.UpSampling2D(size=(1, s), interpolation="nearest") + Concatenate(axis=-1) + 1x1 Conv2D.

### Ghost constraints (none found)
- "U-Net needs different output dict key" — false; loss + train wrapper + probe pivot on "logits".
- "Need a custom causal upsampler layer" — false; UpSampling2D nearest is sufficient.
- "Need to clone all callbacks" — false; they're model-agnostic.

### Exploration Confidence
- Problem scope: deep — exact line ranges of both causal blocks read; lm.py and train script read in full; vision unet skim confirms encoder/decoder fusion pattern; _make_causal_pool / padding="same" causal status grounded in LESSONS L33.
- Solution space: constrained — block APIs and lm.py/train-script contracts pin the design. Only knobs are stages/strides/blocks-per-stage and skip-fusion (concat-1x1).
- Risk visibility: clear — main risk is a subtle causality leak in upsample/concat/skip path; mitigated by the existing test_causality_* pattern from test_cliffordnet_lm.py (perturb position k, assert all positions < k byte-identical).

Ready for PLAN.

## plan_2026-05-06_13a2df9e
### Index
- [F-001 scope-and-callers.md](plan_2026-05-06_13a2df9e/findings/scope-and-callers.md) — pure-additive sibling class, no callers affected, layers `__init__.py` is empty so no export plumbing.
- [F-002 causality-mechanics.md](plan_2026-05-06_13a2df9e/findings/causality-mechanics.md) — how `CausalCliffordNetBlock` enforces causality + which DSv2 ops are/are not causal under `(H=1, W=seq_len)` layout. Conclusion: avg/max with `padding="same"` are causal; `blur`, `gaussian_dw`, `pyramid_diff`, `pixel_unshuffle`, `resnetd` are not.
- [F-003 dsv2-merge-points.md](plan_2026-05-06_13a2df9e/findings/dsv2-merge-points.md) — exact API delta, build-validation, and call-path changes vs DSv2; padding arithmetic for arbitrary `kernel_size`; tests to mirror.

### Key Constraints

### Hard
- Same file: `src/dl_techniques/layers/geometric/clifford_block.py`. No new modules.
- `H=1, W=seq_len` layout (matches `CausalCliffordNetBlock`).
- Strict causality along W: future positions must not influence past outputs (must be regression-tested).
- Existing 117+ tests in `test_clifford_block.py` must continue to pass — purely additive change.
- Keras 3 / TF 2.18 idioms (`@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, no `print`).
- No public-API breakage to existing classes.

### Soft
- Mirror existing class structure (init/build/call/get_config + the helper functions style).
- Match test layout (per LESSONS.md) — class per layer, fixtures, save/load round-trip, gradient flow, causality regression.

### Ghost constraints (not present)
- Layers `__init__.py` is empty (per `layers/CLAUDE.md`) — no export plumbing needed.

### Exploration Confidence
- Problem scope: **deep** — exact line ranges and semantics of both parents read; constraints classified.
- Solution space: **constrained** — combine two known patterns; only thing genuinely new is the narrower pool-kind surface and reasoning about which pools are causal.
- Risk visibility: **clear** — main risk is a subtle pool-future-leak; mitigated by restricting to avg/max only and writing a perturb-future-position regression test.

Ready for PLAN.
