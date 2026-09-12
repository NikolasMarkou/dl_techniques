# Progress

## Completed
- [x] Iter-1/Step-1: LoRA adapter layer + tests. Built `LoRAAdapter` in
  `src/dl_techniques/models/language/zamba2/layers.py` (additive per-occurrence
  low-rank delta, B zero-init, A initialized slice-by-slice to avoid Keras'
  conv-style 3-D fan-in/out miscalculation). 8 tests in
  `tests/test_models/test_zamba2/test_zamba2.py::TestLoRAAdapter` (init, edge
  cases, shape + zero-init-at-construction, occurrence_idx bounds check,
  anti-vacuity divergence after training, gradient-scoping to exercised
  occurrences only, serialization round-trip, config completeness) — all 8
  PASS. Registered as `dl_techniques.models.zamba2.lora_adapter`. Scaffolded
  `zamba2/__init__.py` (docstring-only placeholder; curated exports land at
  step 6) and the flat `tests/test_models/test_zamba2/` package marker.

- [x] Iter-1/Step-2: Shared attention mem-block layer + tests. Built
  `Zamba2SharedAttentionBlock` in `layers.py`: RMSNorm(hidden_state) ->
  concat([original_embedding, normed]) -> Dense(d_model) -> reshape to heads
  -> `RotaryPositionEmbedding` -> reshape back -> `'multi_head'` attention
  (factory) with a rank-3 causal keep-mask built internally -> residual add
  onto the ORIGINAL `hidden_state` (not the concatenated tensor). No LoRA on
  this block per D-001. 16 tests added to
  `tests/test_models/test_zamba2/test_zamba2.py::TestZamba2SharedAttentionBlock`
  (init, edge cases, shape/finiteness, weight-identity across two calls,
  gradient flow, serialization round-trip, config completeness,
  mixed_float16 x2) plus two dedicated files:
  `test_shared_weights_identical_across_depth.py` (is-identity guard +
  negative twin across independent instances) and
  `test_the_causal_mask_is_honoured.py` (three-armed future-leak probe,
  bit-exact `atol=0`). All 23 tests in `tests/test_models/test_zamba2/`
  PASS. Manually confirmed the causal-mask guard is proven RED (max|delta|
  0.307 before the perturbed position) when `_build_causal_mask` is
  monkeypatched to return `None`, then green again unmodified. Registered
  as `dl_techniques.models.zamba2.shared_attention_block`.

- [x] Iter-1/Step-3: Shared MLP mem-block layer (with per-occurrence LoRA) +
  tests. Built `Zamba2SharedMLPBlock` in `layers.py`: RMSNorm(hidden_state) ->
  parallel `gate_proj`/`up_proj` Dense(hidden_dim) -> the owned `LoRAAdapter`
  (holding all `num_occurrences` A/B pairs) adds its delta onto `up_proj`'s
  output, selected by the `occurrence_idx` argument threaded through `call()`
  -> `SiLU(gate) * up` -> `down_proj` Dense(d_model) -> residual add onto
  `hidden_state`. `hidden_dim` derives via the same 2/3-rule-then-round-up
  arithmetic as `SwiGLUFFN` when not given explicitly. LoRA is scoped to
  `up_proj` only, never `gate_proj`/`down_proj`, per D-001. 11 tests added to
  `tests/test_models/test_zamba2/test_zamba2.py::TestZamba2SharedMLPBlock`
  (init, edge cases, shape/finiteness, weight-identity across calls, gradient
  flow, serialization round-trip, config completeness, derived-hidden-dim
  arithmetic, mixed_float16 x2). New dedicated file
  `test_lora_differs_per_occurrence.py` is the decisive guard for the plan's
  single highest-risk claim: base weights (`norm`/`gate_proj`/`up_proj`/
  `down_proj`) are `is`-identical across two different `occurrence_idx`
  values; the LoRA delta is identically zero at construction (correct,
  non-vacuous B-zero-init) then genuinely diverges after one training step
  per occurrence; the same `occurrence_idx` called twice reproduces the
  identical output bit-for-bit both before and after training; and a
  gradient computed through an `occurrence_idx=0` call carries an exact-zero
  gradient on occurrence 1's and 2's LoRA A/B slices (checked on the
  gradient itself, not on stateful-optimizer post-update weights -- Adam's
  own momentum/velocity accumulators move an already-zero-gradient slice on
  a later `apply_gradients` call, which is correct Adam behaviour and would
  make an optimizer-state assertion fail for a reason unrelated to the
  claim). All 34 tests in `tests/test_models/test_zamba2/` PASS. Manually
  confirmed the decisive guard is proven RED: monkeypatched `call()` to pin
  `occurrence_idx=0` unconditionally (the §12.7 "reads only the last
  occurrence" failure shape) and watched
  `test_lora_differs_per_occurrence.py` fail with the expected
  "LoRA pairs are not independent" message, then restored. Registered as
  `dl_techniques.models.zamba2.shared_mlp_block`. D-001's Anchor-Refs line
  back-links to the `self.lora = LoRAAdapter(...)` construction site.

- [x] Iter-1/Step-4: Mamba2 block wrapper/integration + tests. Built
  `Zamba2MambaBlock` in `layers.py`: a thin adapter owning exactly one
  `Mamba2ResidualBlock` (from `mamba/components_v2.py`), translating Zamba2's
  constructor vocabulary (`d_model`/`d_state`/`d_conv`/`expand`/`headdim`)
  onto it 1:1 plus the remaining pass-through kwargs (`d_ssm`, `ngroups`,
  `norm_epsilon`, `rmsnorm`, `norm_before_gate`, `dt_min`/`dt_max`/
  `dt_init_floor`, `bias`, `conv_bias`). `d_ssm=None` resolves to
  `d_model * expand`, mirroring `Mamba2Layer`'s own default (confirmed
  `Mamba2ResidualBlock` itself has no default for this arg). `call()` invokes
  the owned block with `residual=None` (so `new_residual` is exactly the
  input) and adds `mamba_output + new_residual` itself, giving the same
  closed `hidden_state -> hidden_state` contract as the step 2/3 mem-blocks
  so step 5 can call every `'m'`/`'g'` position uniformly. No SSM/scan
  internals reimplemented -- delegates entirely to `Mamba2ResidualBlock`/
  `Mamba2Layer`. Per `decisions.md` D-005, every instance is built fresh
  (never shared) -- the opposite invariant from the mem-blocks. 10 tests
  added to `tests/test_models/test_zamba2/test_zamba2.py::TestZamba2MambaBlock`
  (init, edge cases incl. an indivisible `d_ssm`/`headdim` pair raising from
  the wrapped `Mamba2Layer`, shape/finiteness, the negative weight-sharing
  twin -- two independently-built instances share no `Variable` object and
  do not coincidentally initialize identically, gradient flow, serialization
  round-trip, config completeness, explicit-`d_ssm` round-trip, mixed_float16
  x2). All 44 tests in `tests/test_models/test_zamba2/` PASS; the full
  `tests/test_models/test_mamba/` suite (181 tests, import-only consumer)
  still PASSes with no regression. Registered as
  `dl_techniques.models.zamba2.mamba_block`.

- [x] Iter-1/Step-5: Zamba2Model decoder stack assembly + tests. Built
  `Zamba2Model(keras.Model)` in `model.py`: token `Embedding` -> walk
  `layer_mapping` (`'m'`/`'g'` list), dispatching each position at
  construction time to either a freshly-built `Zamba2MambaBlock` ('m') or
  one of `num_mem_blocks` physical `Zamba2SharedAttentionBlock`/
  `Zamba2SharedMLPBlock` pairs ('g'), selected round-robin by
  `occurrence_counter % num_mem_blocks`, with the global `occurrence_idx`
  (independent of the physical slot) threaded to the MLP block's LoRA
  selection -> final `RMSNorm` -> tied-embedding LM head
  (`hidden_states @ transpose(embedding.weights[0])`, no separate output
  Dense). The `'m'`/`'g'` dispatch table (`_position_info`, a list of
  `(kind, ref_idx, occurrence_idx)` built once in `__init__`) is the
  single source of truth `call()` and the tests both read, so the
  round-robin/occurrence-counter logic exists in exactly one place.
  `pretrained=True` raises `NotImplementedError` directly from
  `Zamba2Model.__init__` (ahead of step 6's `create_zamba2` factory, since
  the plan's own test list for this step required it). Edge case: zero
  `'g'` entries resolves `num_occurrences = max(count('g'), 1)` so
  `LoRAAdapter`'s positive-occurrence validation never fires on an
  all-Mamba2 stack. `get_config()` round-trips all 33 constructor args.
  Registered as `dl_techniques.models.zamba2.model`. 12 tests added to
  `tests/test_models/test_zamba2/test_zamba2.py::TestZamba2Model` (init,
  edge cases, zero-'g' edge case, forward shape/finiteness, both
  `pretrained` branches, gradient flow to every trainable weight --
  including every exercised LoRA `A`/`B` pair, which needed one Adam
  warmup step first since every `LoRAAdapter.b` is zero-init and
  `dL/dA` is mathematically exactly zero before `B` moves, matching
  `TestLoRAAdapter`'s own documented note -- serialization round-trip,
  config completeness, a three-armed causal-mask probe run through the
  WHOLE model, the round-robin weight-identity guard at the model level
  (3 `'g'` positions, `num_mem_blocks=2`: occurrences 0 and 2 land on the
  same physical slot and are `is`-identical; occurrence 1 lands on a
  different slot), and the decisive per-occurrence LoRA-delta-differs
  guard at the model level). All 56 tests in
  `tests/test_models/test_zamba2/` PASS.

- [x] Iter-1/Step-6: Variant/catalogue registration pass + tests. Added
  `MODEL_VARIANTS` (`zamba2_mini`/`zamba2_small`/`zamba2_base`, repo-scale
  per `decisions.md` D-002, not the paper's 2.7B/7B) to `model.py`, plus a
  private `_build_layer_mapping(num_mamba_blocks, num_shared_occurrences)`
  helper that derives each variant's `layer_mapping` (mostly `'m'` with
  `'g'` at evenly-spaced intervals, scaled down from the original config's
  roughly-every-6th-position density) so the table never stores a literal
  mapping that could drift out of sync with its own `num_mamba_blocks`/
  `num_mem_blocks` counts. Added `Zamba2Model.from_variant(variant,
  **overrides)` (classmethod, mirrors `GPT2.from_variant`/`HNet.from_variant`)
  and the top-level `create_zamba2(variant="zamba2_small", pretrained=False,
  **overrides)` factory, pure delegation with no logic of its own.
  `pretrained=True` already raised `NotImplementedError` from
  `Zamba2Model.__init__` (step 5) -- `from_variant`/`create_zamba2` forward
  it unchanged, so no duplicate check was needed. Curated
  `zamba2/__init__.py` `__all__` (`LoRAAdapter`, `MODEL_VARIANTS`,
  `Zamba2MambaBlock`, `Zamba2Model`, `Zamba2SharedAttentionBlock`,
  `Zamba2SharedMLPBlock`, `create_zamba2`). Wrote `zamba2/README.md`
  (overview, architecture diagram, variant table, pretrained contract,
  serialization, training pointer, testing, citation -- Zamba2 2411.15242
  and Zamba1 2405.16712). Added the `zamba2/` catalogue row to
  `models/README.md`'s `language/` section (count bumped 18->19) and one
  docstring-only sentence to `language/__init__.py` (no imports, confirmed
  the family `__init__.py` stays docstring-only). 12 tests added to
  `tests/test_models/test_zamba2/test_zamba2.py`: `TestBuildLayerMapping`
  (3 tests: counts match inputs across all 3 shipped (num_mamba, num_g)
  pairs, 'g' entries are spread out not clustered, single-occurrence places
  'g' last) and `TestModelVariants` (9 tests: every variant builds and
  forward-passes with shape/finiteness matching its own declared
  hidden_size/vocab_size -- parametrized per-cell per LESSONS.md "one
  defective cell is a family" -- unknown-variant ValueError,
  `pretrained=True` NotImplementedError on both `create_zamba2` and
  `from_variant`, keyword-override precedence, `get_config`/`from_config`
  value-level round trip at 2 variants, full-model save/load bit-exact
  round trip via `tmp_path`, and registry-key resolution through
  `keras.saving.get_registered_name`/`get_registered_object`). All 68 tests
  in `tests/test_models/test_zamba2/` PASS; `tests/test_models/test_mamba/`
  (181 tests, import-only consumer) re-run with no regression.

- [x] Iter-1/Step-7: Training script + dataset wiring. Built
  `src/train/zamba2/common.py` (Pattern-3 shape, mirroring
  `train.hnet.common`'s structure -- config dataclass + argparse +
  `config_from_args` single-wiring-site + `build_datasets`/`build_optimizer`/
  `build_model`/`train` -- rather than `train.common.clm_pretrain
  .ClmPretrainConfig`, whose `load_train_val_datasets` unconditionally wraps
  labels for a dict-output model that `Zamba2Model.call` does not have;
  recorded as D-006). `build_datasets` chains `create_tokenizer` ->
  `load_wikipedia_train_val` -> `preprocess_clm_packed_dataset`
  (`repeat=True` on train, `repeat=False`/`shuffle_buffer=1` on val) ->
  `estimate_clm_steps_per_epoch`, and returns the live tokenizer's
  `vocab_size` (100277 for the default `cl100k_base`, exactly matching
  `DEFAULT_VOCAB_SIZE` in `model.py`) as an explicit override so the model's
  embedding/head always match the encoding actually used to pack the
  corpus. `build_optimizer` reuses `optimizer_builder`/
  `learning_rate_schedule_builder` with the documented
  `decay_steps = total_steps - warmup_steps` trap (D-006 anchor). `build_model`
  calls `create_zamba2(variant=..., vocab_size=..., max_seq_len=...)` and
  compiles with `train.common.clm_pretrain.create_clm_loss_fn` (plain,
  un-dict-keyed -- Zamba2's output is a single tensor) +
  `train.common.nlp.build_clm_metrics`. `train()` calls
  `train.common.create_callbacks` directly (not `create_nlp_callbacks`,
  which has no `output_root` parameter and so could never honour
  `--output-dir`) with `use_lr_schedule=True, include_tensorboard=True`.
  `src/train/zamba2/train_zamba2.py` is the entry point: `main()` parses
  first (`args, config = parse_arguments(argv)`), then `setup_gpu`, then
  delegates to `common.train`; `--gpu` is the sole `NON_CONFIG_DESTS` member.
  Manually verified `python -m train.zamba2.train_zamba2 --help` exits 0
  with a `usage:` line and `--variant`/every declared flag listed, and
  verified the full build->compile->fit wiring end-to-end on a 2-sample
  synthetic batch (CPU, `CUDA_VISIBLE_DEVICES=""`) -- loss finite
  (`11.52`) after one step, no exceptions. 33 tests in
  `tests/test_train/test_zamba2/test_cli_contract.py` PASS: `--help`
  usage-line + zero-allocation guards (sentinels on `setup_gpu`/
  `build_datasets`/`train`, all asserted uncalled), the vacuous-exit-code
  counter-example, no-flags-gives-defaults, one parametrized case per
  config field proving the flag reaches it (each probe value asserted
  different from the default -- the anti-vacuity trap), `--variant`
  choices-vs-validation agreement, unknown-variant rejection, and the
  `--gpu`-is-the-only-non-config-dest guard. Did not run the heavier
  per-field `train.hnet`-style exhaustive CLI-contract sweep (scoped out of
  this step per the plan's own instruction not to attempt real training
  here); Step 8's real smoke run is the end-to-end verification this step
  defers.

- [x] Iter-1/Step-8: Final smoke training run. Ran
  `MPLBACKEND=Agg .venv/bin/python -m train.zamba2.train_zamba2 --variant
  zamba2_mini --gpu 1 --epochs 1 --steps-per-epoch 20 --batch-size 4
  --seq-length 128 --output-dir results/zamba2_smoke_test --max-val-samples
  50 --val-fraction 0.02` against the real Wikipedia corpus
  (`/media/arxwn/data0_4tb/datasets/wikipedia`, 6,279,657 train / 50 val
  articles) on GPU 1 (RTX 4070, confirmed idle via `nvidia-smi` before the
  run; GPU 0 was busy with another process and correctly avoided).
  `--help` sanity check passed first (exits 0, `usage:` line, no GPU alloc).
  No code changes were needed -- the pipeline ran end to end on the first
  attempt. Results: train loss finite and decreasing over the 20-step run
  (step 1 `loss=11.5233` -> step 20 `loss=11.5123`); `val_loss` improved
  from `inf` to `11.50373` (checkpoint saved to `best_model.keras`); every
  logged `accuracy`/`bits_per_character`/`bits_per_token`/`perplexity`
  value finite throughout; no NaN/Inf anywhere in the 20-step log.
  `results/zamba2_smoke_test/zamba2_zamba2_mini_20260912_120857/config.json`
  exists, confirming `prepare_run_dir`'s artifact lands at the repo-root
  `results/` location (not `src/results/`). This satisfies Success
  Criteria #9 and (combined with steps 1-7's automated suites) completes
  all 9 plan success criteria. `results/zamba2_smoke_test/` is left in
  place per the repo's hard `results/`-safety invariant; it is gitignored
  and was not committed.

- [x] Iter-1/Step-1.1 (completion fix): Fixed `LoRAAdapter.build()`'s
  `_a_initializer` to clone a fresh initializer per occurrence slice
  (`clone_initializer(a_initializer_fn)`), not call one resolved instance
  repeatedly — the original implementation produced bit-identical `A`
  slices across every occurrence (`review-iter-1.md` concern 3). Fixed the
  now-false docstring Note claiming independence. Added
  `test_a_slices_are_independently_initialized`, asserting `A[i] != A[j]`
  for every pair at construction, before any training. D-007 anchor placed
  at the fix site. `pytest tests/test_models/test_zamba2/test_zamba2.py -k
  TestLoRAAdapter` — 9 passed.

- [x] Iter-1/Step-6.1 (completion fix): Wrapped the module-level
  `MODEL_VARIANTS` table in `types.MappingProxyType(...)` — the class-level
  alias `Zamba2Model.MODEL_VARIANTS = MODEL_VARIANTS` was the exact
  "class attribute aliasing a module-level mutable" (S3) shape
  `tests/test_models/test_package_api_contract.py::TestNoMutableDefaults::
  test_no_mutable_default_anywhere` forbids, RED at HEAD
  (`review-iter-1.md` concern 1). Same remedy `fastvit/model.py`'s
  `MCI_VARIANTS`/`FastVitImageEncoder.MODEL_VARIANTS` alias already uses.
  D-008 anchor placed. `pytest
  tests/test_models/test_package_api_contract.py -k TestNoMutableDefaults`
  — 7 passed (confirmed GREEN after the fix); zamba2's own suite unaffected.

- [x] Iter-1/Step-6.2 (completion fix): Fixed `from_variant` passing
  `num_mem_blocks` (physical block count) as the occurrence count to
  `_build_layer_mapping`, which made `count('g') == num_mem_blocks` for
  every shipped variant — no physical mem-block was ever invoked more than
  once on the public factory path (`review-iter-1.md` concern 2). Split
  `MODEL_VARIANTS` into `num_mem_blocks` (physical, forwarded to the
  constructor) and a new table-only `num_mem_block_occurrences` (popped by
  `from_variant`, fed to `_build_layer_mapping`); every shipped variant now
  has occurrences = 2x physical blocks (mini 2->4, small 3->6, base 4->8).
  Also rewrote the model-level "decisive" LoRA guard
  (`test_lora_deltas_differ_across_g_occurrences`), which `review-iter-1.md`
  concern 5 found vacuous (compared two different physical blocks with
  zero-init `B`, never trained): now uses `num_mem_blocks=1` with 2 'g'
  occurrences on the SAME physical block, a warm-up optimizer step so `B`
  moves off zero, the decisive divergence assertion, AND a monkeypatch
  RED-proof that forcing `occurrence_idx=0` everywhere (via
  `model._position_info`) makes the same assertion raise
  `AssertionError`, then restores. Added
  `test_occurrences_exceed_physical_blocks` (parametrized over every
  `MODEL_VARIANTS` entry) and
  `test_shipped_variant_reuses_a_physical_block_at_two_depths` (exercises
  the real `create_zamba2("zamba2_mini")` path, confirms object-identity
  reuse at >=2 depths). D-009 anchor placed. Full scoped regression:
  `pytest tests/test_models/test_zamba2/ tests/test_train/test_zamba2/
  tests/test_models/test_package_api_contract.py tests/test_models/test_mamba/`
  — **930 passed, 10 skipped, 0 failed** in 220.12s.

## In Progress
*Nothing currently.*

## Remaining
*Nothing currently -- all 8 plan steps complete; all 3 review-iter-1.md
CRITICAL defects fixed and verified.*

## Blocked
*Nothing currently.*
