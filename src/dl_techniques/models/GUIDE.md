# Models: Lessons From Past Issues

This guide collects the defects found under `src/dl_techniques/models/`, together with the
measurements that exposed them and the rule each one taught. It complements two other files:
`AGENTS.md` in this directory, which is the authoring contract, and
`research/2026_keras_custom_models_instructions_v2.md`, which is the Keras 3 authoring guide.
Read those first; this file explains why several of their rules exist.

Unless a line says otherwise, every behaviour was measured on Keras 3.8 / TF 2.18. Most of these
defects were silent: they raised nothing, gave the right output shape and passed a green suite.
That silence is the reason they are written down here.

## 1. Model shape and factories

- **Follow the ResNet pattern.** The reference is `models/vision/resnet/model.py`:
  - one class with `include_top` in `__init__`;
  - the head is built in `__init__` and applied in `call()`;
  - a `from_variant` classmethod;
  - a single `create_<model>(variant, pretrained=False, **kwargs)` that delegates to
    `from_variant`.

  Do not add `create_<model>_with_head`, per-task factories or a two-input functional wrapper
  around the class. The language package was migrated to this shape in three plans, ending at
  2026-10-13: 15 classes, every `_with_head` factory removed, and old wrapper archives still load.
- **`pretrained=True` must raise `NotImplementedError` when no weights are distributed.** The
  2026-08-14 review found 14 of about 18 factories returning random weights. Some warned through
  `logger.info`, which nobody reads.
- **A functional `keras.Model(inputs, outputs)` prunes any layer that is not on the output path.**
  A subclassed model can build a head and skip it in `call()`, so weights trained with
  `include_top=True` load into `include_top=False`. A functional builder cannot honour that
  contract. Document the difference instead of writing a test that pretends otherwise.
- **A factory that wraps the class in a separate plain `keras.Model` cannot inherit a
  class-level override** such as `compile()`. Before you decide where a fix belongs, check what the
  reproduction actually constructs (AccUNet).
- **A `keras.Model` subclass whose name starts with `_` raises**
  `ValueError: not a valid root scope name`. Keras derives the model name from the class name, so
  test probe classes need a public name.
- **`MODEL_VARIANTS` must be assigned at class level** if the package imports or advertises it.
  The AST sweep in `tests/test_models/test_package_api_contract.py` checks that it is assigned, not
  just imported.
- **Pass the reference's structural literals explicitly.** This repo's constructor defaults are
  not the paper's structure.

## 2. Serialization and checkpoints

- **Verify a round trip by comparing VALUES**: load, run a forward pass, and compare the outputs
  and every weight with `np.array_equal` / `allclose`. These checks are blind and have each passed
  on a broken archive:
  - weight counts;
  - weight paths;
  - `count_params()`;
  - the md5 of a `.keras` file, which is a zip with per-write metadata.
- **Sub-layers build lazily on first `__call__`, but not under `load_model`.** `load_model` builds
  from the saved input shape and restores the weights at once. Two measured cases:
  - `ImageEncoderViT` reloaded 1 of 65 weights;
  - `DPTDecoder` reloaded 0 of 12.

  Fix: `build()` must materialise the whole tree. Where that is impractical, add
  `get_build_config` / `build_from_config`:
  - `get_build_config` returns `None` while the model is unbuilt;
  - `build_from_config` runs a dummy forward pass, so every variable exists before restore.
- **A nested `List[List[Layer]]` owned by a `keras.layers.Layer` is not saved.** The same nesting
  owned by a `keras.Model` is saved. The measurements:
  - a synthetic control saved 2 of 6 weights under a Layer owner and 6 of 6 under a Model owner;
  - `MaskedAutoencoder` lost 98.56% of its parameters this way.

  Store sub-layers in a flat list and expose the grouping through an accessor.
- **In `build()`, build only what `call()` runs.** If you build an unused sub-layer, the
  `.keras` layout changes, which silently breaks existing checkpoints.
- **Keras refuses to substitute a sub-layer after `build()`.** Do structural surgery before
  build, and make `from_config` apply it again.
- **A new validation raise in `__init__` breaks deserialization**, because `from_config` calls
  `__init__`. If the old default produced a value the new check refuses, every checkpoint saved
  before the check fails to load.
- **`from_config` re-raises a constructor exception as `TypeError`**, with the original message
  appended. A test that crosses a `.keras` boundary should accept `(ValueError, TypeError)` and
  assert on the message.
- **Deleting a `from_config` override migrates every field it touched, not only the one you were
  after.** In `vit_siglip` the deleted override had also been resolving the regularizers.
- **Registration**: use `register_dl_technique`, with the family directory stripped from the key
  (see `AGENTS.md`). A save/load round trip cannot validate the key, because writing and reading
  share one in-process registry, so a typo stays self-consistent and loads green. Only a literal
  `==` against the full key string checks it. `custom_objects` dicts are keyed by
  `keras.saving.get_registered_name(cls)`, never by the bare class name.
- **Moving a module changes a `package=`-qualified key.** Before you ship a clean break, scan the
  existing `results/**/*.keras` archives for the old path.
- **A load route without a test is not a feature.** In October 2026, `pretrained=<path>` was dead
  on 6 of 13 language models. The failures:
  - a random int32 dummy forward pass;
  - a dummy forward over `max_position_embeddings`;
  - a load into an unbuilt model;
  - a missing `build()`.

  The rule now: every backbone layer loads, or the call raises. Check everything first and copy
  after, so a raise copies nothing. Only the declared vocabulary tables may stay fresh, identified
  by variable path suffix and shape (`tables=` in `utils/weight_transfer.py`). Test each route in
  both directions: a deeper target, a wider target, and a target with a changed vocabulary.
- **A warm start must assert that the backbone loaded.** A head swap uses
  `load_weights_from_checkpoint(..., skip_prefixes=...)`. Assert that the backbone name is in
  `report.loaded`, rather than trusting that a transfer of zero layers would raise.
- **A refusal raised in `build()` is swallowed when Keras rebuilds from a saved config**, because
  `build_from_config` forces `built=True`. An old archive that recorded an over-long build shape
  loaded with random weights in 12 of 14 guarded language classes. `RecordedBuildShapeMixin` in
  `utils/sequence_length_guard.py` fixes it. Any new build-time refusal needs an archive-load test
  that asserts the weights are equal, not only a test that the refusal raises.
- **Integer lookup tables inside layers must be int64** (`INDEX_WEIGHT_DTYPE` in
  `utils/dtype_policy.py`). TF pins an int32 variable to the host, so a jit-compiled `fit` on GPU
  fails to read it (TopoLM, GaussianReadout, the HRM puzzle embedding).

## 3. Graph mode, XLA and the GPU

- **Keras enables XLA by default when a GPU is visible** (`jit_compile="auto"`), and a CPU-only
  suite cannot see what XLA changes. `Trainer._resolve_auto_jit_compile` returns `False` outright
  on a CPU-only host. When a guard depends on the regime, assert the resolved `jit_compile` inside
  the test, and run it on GPU 1.
- **XLA drops `tf.debugging` assertions and `check_numerics`.** BERT's over-long-sequence
  refusal raised on CPU and returned finite garbage on GPU. The working fix keeps the assertion and
  also adds `reduce_sum(zeros([max_len - seq_len]))` to the input inside its control dependency, so
  an over-long input fails at compile time with a shape error.
- **`keras.ops.take_along_axis` breaks XLA on a GPU predict retrace.** On the TF backend it calls
  `tf.broadcast_dynamic_shape`. A second `predict` at a new sequence length retraces with an
  unknown length, and XLA on GPU refuses it ("BroadcastArgs must be a compile-time constant"). CPU
  XLA accepts the same graph.
  - The fix is a flat-row gather: `take(reshape(x, (-1, H)), row * L + idx)`.
  - A `top_k` with `k = minimum(top_k, seq_len)` fails the same way. Pad to a constant `k` and mark
    the padded picks invalid.
  - Find which op XLA refuses before editing. Replacing `arange(seq_len)` did not help.
  - The guard runs `predict` at two lengths on GPU (`SequencePooling`, 2026-10-13).
- **`keras.ops.tril` and `keras.ops.triu` fail as soon as they are traced**
  (`TypeError: pred must not be a Python bool`). They pass every eager test and break `fit`,
  `predict`, `jit_compile` and `.keras` reload.
- **`.assign()` inside `build()` is discarded on the real path.** Keras runs a symbolic build
  inside a `StatelessScope`. A direct `layer.build()` looks correct, while the same layer reached
  through a parent's `call()` holds zeros. The damage:
  - RoPE tables were zero;
  - a trend-only N-BEATS predicted exactly zero;
  - the Mamba-v1 dt bias was zero (11 sites in 8 files).

  Use an initializer callable instead. A probe for this must go through a parent layer's `call()`
  and compare the table with its closed form.
- **`ops.convert_to_tensor` in `build()` binds to whatever FuncGraph is tracing the build.**
  `fit` on an unbuilt model then dies with `InaccessibleTensorError`. Keep derived constants as
  numpy arrays and convert them inside `call()`.
- **Code reached from `build()` must be traced symbolically, not only called eagerly.**
  `ExtractPatches.compute_output_spec` raised `UnboundLocalError` only on the symbolic path.
- **Variables created inside the first trace** make the second call raise
  ("tf.function only supports singleton tf.Variables created on the first call").
- **`keras.random.uniform(dtype="int32")` raises** every time it is called.
- **NumPy-style fancy indexing (`t[batch_idx, item_idx]`) is not valid on a TF tensor.** Use a
  gather.
- **A `keras.Input` functional build validates neither `Reshape` element counts nor gathers.** An
  impossible geometry constructs cleanly and fails only at the first real forward pass, so a
  construction-only test of it is vacuous.
- **A capture inside `predict` sees a `SymbolicTensor`.** For activation hooks, use one eager
  forward pass.
- **Eager bit-identity does not prove a refactor is inert.** Under `@tf.function` the same code
  can differ by a few ulps through reassociation. Before calling such a difference a regression,
  measure the old code eager against graph.

## 4. Numerics and mixed precision

- **An additive mask bias in `scores.dtype` NaNs every unmasked position under
  `mixed_float16`** (`0 * -inf`). Build the bias with `ops.where(keep, 0, bias)` in a dtype where
  it stays finite. Use `layers/attention/common.apply_attention_mask`; do not write your own.
- **Sub-layers autocast.** A float32 tensor that enters a sub-layer under `mixed_float16` is
  float16 inside that sub-layer's `call()`. A dtype-only fix does not survive a layer boundary.
- **`AutocastScope` casts a variable only when it is read inside a real `Layer.__call__`.** A
  bare-tensor probe can report a different dtype than `model(x)` does.
- **Mixed-precision custom training paths need `scale_loss`.** At 9 sites, the whole update was
  divided by about 2^15 without it.
- **float16 has coarse resolution near 1.0.** The ULP there is 9.77e-4, so a guard margin of
  1e-4 does nothing (shgcn was 100% NaN).
- **TF has no float16 gradient for `ResizeBicubic`.** It silently returns `None`, so a head behind
  it gets no gradient (superpoint).
- **`EnergyLayerNorm` overflows float16 in its backward pass under XLA**: `(var+eps)^(-3/2)` at
  eps below about 6.1e-4, with near-constant tokens. The loss stays finite and the weights move by
  exactly 0. Run the block in the variable dtype rather than raising eps.
- **TF32 is on by default on the GPU.** It changed an identical float64 comparison by up to
  1775x, and `tensor_float_32_execution_enabled()` returns `True` even with no GPU visible.
  - Opt oracle tests into the `tf32_disabled` fixture.
  - Never toggle TF32 at module import time: it changes results by collection order.
  - Measure a tolerance in both regimes.
  - Derive bounds from `tests/numerics.py:reassociation_atol` rather than pasting literals.
  - Use `rtol=0` when you mean a pure `atol`.
- **`x + g - stop_gradient(g)` is not the identity.** It is off by 1 ulp in about 25% of draws.
  Write `x + (g - stop_gradient(g))`.
- **`keras.ops` has no `lgamma`.** A Lanczos version is in `utils/tensors.py`.
- **`keras.ops.fft` exists, but there is no `ifft`.** `ifft2` exists.

## 5. Architecture correctness traps

- **Some blocks are transform-only, with the residual left to the caller.** `CliffordNetBlock`
  has `layer_scale_init=1e-5`, and `x = block(x)` replaced the signal with about 1e-5 of itself
  per block. After 4 blocks, absmax fell from 2.76 to 3e-12. Check whether a block adds its own
  residual before you call it. A magnitude assertion catches this class; LayerScale without a
  residual in fastvlm is the same defect.
- **`RotaryPositionEmbedding` expects `(B, H, S, D)`.** Fed `(B, S, H, D)`, it rotated by head
  index and became an exact no-op on `q·k`: two attention layers never carried position. A
  permutation probe cannot see this. Use the pairing identity.
- **A stock `MultiHeadAttention` key bias can never learn.** `q·b_k` is constant along the
  softmax axis and cancels. Any dead-weight sweep must waive it by name.
- **Assert causality in both directions**: future tokens must not move past outputs, and past
  tokens must move future outputs. Add a negative control with the mask removed. A claim that a
  model has no causal mask was refuted for BLT, CLIP and Qwen only by such a probe, and
  `test_gpt2.py` is the template.
- **`SequencePooling` positional modes:**
  - `cls`/`first` and `middle` never read the mask, and `middle` indexes the padded length;
  - `last` is correct only for contiguous prefix masks;
  - under `cls`, changing a non-first token moved the logits by exactly 0.0.

  Pick the pooling to match the padding side. Decoders pool `last` under
  `head_attention_mask(input_ids, attention_mask, pad_token_id)`.
- **`training=None` is not inference.** `layers/regularization/stochastic_depth.py` only
  short-circuits on `training is False`, so a plain `model(x)` drops paths. Two calls can still
  agree by luck.
- **Norm epsilon differs between frameworks.** Keras BatchNorm, LayerNorm and GroupNorm default
  to 1e-3, PyTorch uses 1e-5, and `create_normalization_layer` sets 1e-6. In one port, 86 of 114
  norms were 100x off. Histogram `layer.epsilon` over `model._flatten_layers()` after a port.
  Exception: MobileCLIP v1 keeps Keras's 1e-3, because its checkpoints were trained that way.
- **Keras `padding='same'` is asymmetric at stride above 1; PyTorch `k//2` is symmetric.** In a
  multi-branch block, the branches then sample different pixels. Use `ZeroPadding2D(k//2)` with
  `padding='valid'`. The Dirac-kernel probe prints the sampling grid.
- **Shape-preserving defects need value probes on non-square inputs.** A DocScanner port found
  four layout defects, and no shape assertion caught any of them:
  - `align_corners`;
  - a channel-order reverse;
  - a sub-pixel interleave;
  - stride-2 padding.

  A fully transposed relative-position bias in BEiT passed 219 tests.
- **Classifier heads emit logits** (ConvNeXt V1/V2 and most others), so compile with
  `from_logits=True`.
- **Initializer instances:**
  - An unseeded Keras initializer instance seeds itself at construction and replays the same draw
    at the same shape. When filling N slices, call `clone_initializer` (`initializers/clone.py`)
    fresh for each slice, inside the loop.
  - A seeded initializer is cloned with its seed on purpose; use `keras.utils.set_random_seed`
    for reproducibility.
  - `np.random.seed` does not reach Keras weight initializers. Neither does `tf.random.set_seed`.
- **When you fix a dead branch, check the new wiring.** Feeding the supervision target in as a
  model input leaks the label, and a presence flag hard-wired to a constant starves the sibling
  branch.

## 6. Language models

- **Task heads:**
  - The head is built in `__init__` through `layers/heads/nlp/model_head.py` and lives under the
    `task_head/` name prefix. Backbone weight paths are therefore identical for every top and
    task, and a head swap is `load_weights_from_checkpoint(..., skip_prefixes=(TASK_HEAD_NAME,))`.
  - The headless output must stay byte-identical when a head is added. Prove it against a pristine
    `git worktree`.
- **`pad_token_id` defaults to `None`.** 0 is a real token in GPT-2 and Qwen vocabularies.
- **Unwrap a functional wrapper that holds one inner model before comparing layer names.**
  Without that, a backbone cannot move to a new head.
- **A generation head's vocabulary is filled from, or checked against, the model's.** A stored
  `task_config` with 32000 on a smaller-vocab model fails `from_config` by design.
- **Sequence-length refusals** live in `utils/sequence_length_guard.py` (14 classes):
  - Reset cached step functions only when `not model.built`. Resetting a built model killed
    `fit()` with a `NoneType` error.
  - Test the refusal mid-fit with a callback.
- **Qwen MoE experts take `moe_intermediate_size` verbatim.** The SwiGLU 2/3 rule had tripled
  it.
- **Publish parameter counts from `trainable_weights`.** `count_params()` includes the
  non-trainable RoPE `cos_cached`/`sin_cached` buffers, about 805M for Qwen3 30b-coder.
- **The bounded sparse arm of the qwen3_next padding test needs a tolerance.** Sparse top-k MoE
  routing adds about 1e-7 of float32 noise to padding-invariance tests; dense routing gives exactly
  0. Remove the confound before you name a leak.
- **Several packages share `BertEmbeddings`**: bert, fnet and distilbert. An edit to it needs all
  three suites plus `test_factory_registry_drift.py`.

## 7. Training-path behaviours that look like model bugs

- **A zero gradient is not a freeze under Adam/AdamW.** Moments drift and decoupled weight decay
  still applies. Freeze by setting `trainable` (toggle through `True`; setting `False` is a no-op
  on layers that are already `False`), or zero the learning rate as well.
- **`optimizer.learning_rate` returns the current value, not the schedule.** The schedule is on
  `optimizer._learning_rate`.
- **`keras.utils.set_random_seed` reseeds `random`, numpy and TF, but not an existing global
  `SeedGenerator`.** Call it before constructing the model.
- **A diagnostic training-mode pass advances dropout seed generators** and moves a same-seed
  result. Restore the weights and every seed-generator state after it.
- **`model.loss_weights = [...]` after `compile()` is dead.** The weights are baked in at compile
  time.
- **A `keras.losses.Loss` given a dict `y` fails inside `fit`.** Pack the labels into one array.
- **`exclude_from_weight_decay` matches leaf variable names**, and a later call replaces the
  earlier one.
- **The `fit()` progress bar shows a smoothed running mean.** Quote the CSV log, never the bar.
- **`EarlyStopping(mode='auto')` raises on unknown metrics.** A `'max' if 'accuracy' in monitor`
  rule made every PSNR or IoU monitor pick the worst epoch.
- **Curriculum runs where `curriculum_epochs == epochs`**: the checkpoint with the best `val_loss`
  is under-exposed to the final noise level, so evaluate the final model too.

## 8. Writing tests that can fail

The 2026-08-14 review of all 73 packages found 21 defects that made a model untrainable,
unconstructible or silently wrong. Every one sat in a package whose tests were build, shape and
round-trip templates. Packages with behavioural tests (SAM, DINO, BEiT, GPT-2, memory_bank) came
back clean. Quality tracks test depth.

- **Prove every guard RED.** Inject the defect, watch the guard fail, then restore it.
  - Mutate the tracked file in place. `pyproject.toml` sets `pythonpath = ["src"]`, which
    overrides `PYTHONPATH`, so a mutant in a scratch copy is never loaded and reads GREEN.
  - A guard for a GPU-only defect can be RED only on a GPU run.
- **Pin identity and value, not shape.** Changing the BEiT MIM slice from `[:, 1:]` to `[:, :-1]`
  passed 91 tests that checked shape only.
- **Avoid degenerate fixtures.** All of these hide defects:
  - constant inputs, because a convex combination of equal vectors ignores attention;
  - square-only inputs, which hide transposes;
  - zero-init branches, which make the untrained model the identity;
  - `training=False` only, which hides dropout.
- **Dead-component probes need positive liveness arms that are budgeted before the run.** Use
  `tests/test_models/test_sam/dead_component_oracle.py` after one real optimizer step, not at
  init.
- **Use the shared oracles** listed in `src/dl_techniques/AGENTS.md`:
  - `smoke_contract_oracle`;
  - `knob_sensitivity_oracle`, where a structural knob is pinned on the weight-shape signature;
  - `dead_component_oracle`;
  - `numerics.reassociation_atol`.
- **A precision-arm exemption** (`dtype_exempt_outputs` in `test_precision_arm_family.py`) is
  verified by a mutation probe, never by "the pin compiles". One backwards index blinded the guard.
- **Do not read the suite through the wrong lens:**
  - `--collect-only` is not a test run.
  - An all-skip module reads as a pass.
  - Quote the passed count together with the collected count.
- **Scoped runs miss the repo-root guards.** After changing a package's public surface or
  README, also run `pytest tests/*.py`. The tree-wide collection gate,
  `CUDA_VISIBLE_DEVICES="" pytest tests/test_models/ -q --collect-only`, takes about 3 seconds.
- **Never run pytest jobs in parallel.** GPU contention produces false failures; the tell is
  `cudaSetDevice` or `ResourceExhaustedError` in the error text.
- **Before calling a failure real:**
  - re-run it alone;
  - compare against a pristine `git worktree` of the base commit;
  - check whether the failure is pinned on the device rather than the commit.
- **Run the real entry point.** 249 green tests coexisted with two broken trainer CLIs.
- **A stale test can defend the bug.** Before "making the tests pass", re-derive the expected
  value from the source formula.
- **Pin a known-open defect with `xfail(strict=True, reason=...)`**, so that it XPASSes loudly
  when someone fixes it.

## 9. Claims and counts

- **A negative probe licenses "my probe did not see it under this protocol"**, never "it cannot
  happen". Every "cannot" or "unreachable" claim recorded in this repo was later refuted by
  execution. Write "exact for X; the exception is Y".
- **Prose claims are checked by running code, not by grep.** grep proves a sentence exists, not
  that it is true.
- **Re-derive every count with the command beside it.** Package counts, test counts, parameter
  counts in READMEs (accunet's "verified 16.8M" was 13.4M) and census pins all drifted.
- **Treat a review finding as a hypothesis.** External reviews were right about the site and
  wrong about the mechanism several times. Re-derive suggested replacement code, wording and test
  inputs before using them.
- **The fix round is the most dangerous commit.** Review a fix as new code: several CRITICALs in
  this repo were introduced by the round that fixed the previous ones.
