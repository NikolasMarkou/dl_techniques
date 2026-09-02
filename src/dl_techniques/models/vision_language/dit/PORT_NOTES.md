# DiT Port Notes — "What Does NOT Fit / What Must Change"

*Port: fast-DiT (PyTorch class-conditional latent Diffusion Transformer, `models.py` +
`diffusion/gaussian_diffusion.py` + `diffusion/respace.py`) → `dl_techniques` (Keras 3 / TF 2.18).*
*Plan: `plan-2026-09-02T170923-1285ed83`. Decisions referenced as D-NNN live in that plan's
`decisions.md`; the in-code anchors carry the same prefix.*

---

## 1. Overview

This package (`src/dl_techniques/models/vision_language/dit/`) is a Keras 3, **channels-last** port
of Peebles & Xie's **DiT** (arXiv:2212.09748) — a transformer that denoises VAE latents, conditioned
on a class label and a diffusion timestep through adaLN-Zero — together with the Gaussian-diffusion
machinery the model is useless without: both named beta schedules, `q_sample`, the true posterior,
the `LEARNED_RANGE` variance interpolation, the ancestral and DDIM reverse steps and their loops,
timestep respacing, and classifier-free guidance.

The port is **architecture-faithful, dependency-free and train-from-scratch**. It carries no
pretrained weights, no VAE and no tokenizer; `pretrained=True` raises `NotImplementedError` naming
the variant. It adds no third-party dependency. Every registered class carries
`@register_dl_technique("dl_techniques.models.dit.<module>")`, creates its sub-layers with an
explicit `name=`, implements `compute_output_shape()`, and round-trips through `.keras` on
**values**. Logging is `dl_techniques.utils.logger`; there is no `print` and no `from keras import`.

**What is faithful.** The forward path (`reference/models.py:232-247`), the six-chunk adaLN-Zero
order, the zero-init identity at initialisation, the frozen 2-D sin-cos table and its w-first grid,
the timestep ladder's `/half` denominator and cos-first concat, the null-label row at index
`num_classes`, the twelve variant rows, the flattened-`Linear` xavier of the patch embedding, the
three-channel CFG split, the `LEARNED_RANGE` interpolation, the respacing index remap, and the
hybrid MSE + frozen-out variational-bound objective.

**What is adapted.** Channels-last everywhere, which forces the unpatchify interleave to be
**re-derived** rather than transcribed (§4.4); a fused `GaussianDiffusion`/`SpacedDiffusion`
(§4.6); `clip_denoised=False` as the default (§4.7); a five-line duplication of the variance
interpolation held in lockstep by a test (§4.8); a `scale, shift` chunk order in the final layer
(§4.1); and the training objective packed into `y_true` so it runs under stock `fit()` (§4.2).

Measured surface at the end of step 8 (commands beside the numbers):

```bash
wc -l src/dl_techniques/models/vision_language/dit/*.py        # 3,303 over 5 modules
wc -l tests/test_models/test_dit/*.py                          # 3,344 over 6 files
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_models/test_dit/ -q
```

| | |
|---|---|
| Package source | 5 modules, **3,303** lines (`blocks.py` 702, `config.py` 328, `diffusion.py` 1,259, `model.py` 928, `__init__.py` 86) |
| New shared modules | 4 — `layers/embedding/sincos_pos_embed_2d.py` (233), `layers/embedding/timestep_embedding.py` (382), `utils/ddpm_schedule.py` (494), `losses/ddpm_hybrid_loss.py` (575) |
| New attention / FFN / norm / patch / label classes | **0** |
| Package tests | **316 collected / 316 passed** over 6 files (3,344 lines) |
| Shared-module tests | `tests/test_utils/test_ddpm_schedule.py` **244 passed** · `tests/test_losses/test_ddpm_hybrid_loss.py` **25 passed** · `tests/test_layers/test_embedding/` **585 collected / 583 passed / 2 pre-existing skips** |
| Parameters (published geometry) | S/2 32,963,488 · B/2 130,512,544 · L/2 458,102,944 · XL/2 675,129,760 — all twelve rows in `README.md`, each measured |
| Trainer | **not in the tree at step 8.** `src/train/dit/` is steps 11-13 of the plan; this document will be wrong about it until then, so it claims nothing |

---

## 2. What Was REUSED (drop-in)

Existing, tested `dl_techniques` components used without modification. The bar for "reuse" is that
the *numerics* match, not merely the name. **No attention, FFN, normalization, patch-embedding or
label-embedding class is defined anywhere under `dit/`** — verify with
`grep -n "^class " src/dl_techniques/models/vision_language/dit/*.py`, which returns exactly
`DiffusionConfig`, `DiTBlock`, `DiTFinalLayer`, `DiT` and `GaussianDiffusion`.

| Upstream component | dl_techniques reuse | Import path | Note |
|---|---|---|---|
| `PatchEmbed` (`x_embedder`) | `PatchEmbedding2D` | `dl_techniques.layers.embedding.patch_embedding` | `kernel = stride = patch_size` by construction. Its **initializer is overridden** — §4.3 |
| `LabelEmbedder` (CFG label dropout, null row) | `ClassLabelEmbedding` | `dl_techniques.layers.embedding.class_label_embedding` | Built by the previous plan for exactly this port. Its **initializer is overridden** — §4.9 |
| `adaLN_modulation` 6-way chunk + zero-init `Dense` + affine-free `norm1` | `AdaLayerNormZero` | `dl_techniques.layers.transformers.sd3_adaln` | Chunk order matches upstream exactly; verified by a per-chunk attribution probe, not by reading |
| `FinalLayer`'s 2-way chunk | `AdaLayerNormContinuous` | same | **Chunk-order divergence**, §4.1 |
| `modulate(x, shift, scale)` | the shared free function | same | Owns its own `expand_dims`; broadcast contract pinned by `tests/test_layers/test_transformers/test_the_modulate_broadcast_contract.py` |
| timm `Attention` (bidirectional, `qkv_bias=True`) | stock `keras.layers.MultiHeadAttention(use_bias=True)` | `keras` | Precedent: `layers/transformers/adaln_zero.py:222-230`. **`use_causal_mask` is never passed** — §4.10 |
| `Mlp(act_layer=GELU(approximate="tanh"))` | `create_ffn_layer("gelu_tanh", ...)` → `GELUMLPFFN` | `dl_techniques.layers.ffn.factory` | The key is `gelu_tanh`, **not** `mlp` — `mlp` is exact-erf GELU, a one-string difference with no shape symptom |
| `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` | bare `keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)` | `keras` | `epsilon` is explicit on purpose: the Keras default is `1e-3`, a silent 1000x error |
| optimizer / callbacks / run-dir wiring | `train.common` | `train.common` | Trainer only, and the trainer is not in the tree yet |

---

## 3. What Was BUILT NEW

Four shared assets, each with a MUST-WRITE-NEW justification. "No existing asset fits" was
established by reading the candidate, not by name-matching.

| Upstream component | New artifact | MUST-WRITE-NEW justification |
|---|---|---|
| `get_1d_sincos_pos_embed_from_grid` / `get_2d_sincos_pos_embed_from_grid` / `get_2d_sincos_pos_embed` | **NEW** `dl_techniques.layers.embedding.sincos_pos_embed_2d` (pure NumPy, unregistered) | The only bit-exact copy in the repo is module-private to `bit_diffusion/blocks.py:724-864`. A `models/` → `models/` import inverts the dependency direction and has no precedent here — D-001 |
| `TimestepEmbedder` | **NEW** `TimestepEmbedding` — `dl_techniques.layers.embedding.timestep_embedding`, registered and added to `EMBEDDING_REGISTRY` under the key `'timestep'` | The existing `ScalarSinusoidalEmbedding` diverges on **three** independent numerics (sin-first concat, `/(half-1)` denominator, an input rescale); `bit_diffusion`'s `DiTXATimestepEmbedder` is correct but model-package-private and moving it would change its registration key — D-001 |
| the beta schedules and every derived constant table | **NEW** `DDPMSchedule` + `get_named_beta_schedule` + `space_timesteps` — `dl_techniques.utils.ddpm_schedule` (frozen dataclass, pure NumPy) | **No DDPM schedule exists anywhere in the repo.** `bit_diffusion` is an SDE bridge; `sd3_mmdit` and `ideogram4` are flow-matching. Three in-plan call sites (loss, sampler, data pipeline) earn the shared home under the rule of three |
| `LossType.MSE` + `ModelVarType.LEARNED_RANGE` training objective | **NEW** `DDPMHybridLoss(keras.losses.Loss)` — `dl_techniques.losses.ddpm_hybrid_loss` | No repo loss implements a DDPM epsilon MSE, and none implements a variational-bound term at all |

Package-local, not shared: `DiTBlock`, `DiTFinalLayer`, `DiT`, `DiffusionConfig` and
`GaussianDiffusion`, all of which are compositions of the assets in §2 rather than new primitives.

---

## 4. What Does NOT Fit / What Had to CHANGE

One subsection per divergence, each anchored to a decision and naming the test that pins it.

### 4.1 The final layer splits `scale, shift`; upstream splits `shift, scale` — **D-011**

`reference/models.py:132` reads

```python
shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
```

while the reused `AdaLayerNormContinuous` (`layers/transformers/sd3_adaln.py:501`) reads
`scale, shift = ops.split(emb, 2)` — the diffusers convention. The port **reuses the existing layer
anyway** rather than hand-rolling a shift-first 2-way split.

Why that is safe, stated so nobody has to re-derive it: the modulation `Dense` is zero-initialised in
**kernel and bias**, so the two orders are the same function class under a permutation of that
Dense's output units, and the zero init is exactly symmetric under that permutation. Gradient descent
is equivariant to it, so the two parameterisations train identically up to relabelling. The one thing
that would differ is loading an upstream checkpoint, which is impossible here.

The shipped order is pinned **by name** (`DIT_FINAL_CHUNK_NAMES = ("scale", "shift")`) and by
`test_dit_blocks.py::TestTheFinalLayerChunkOrderIsScaleFirst`, so it is a stated decision rather
than a silent one.

*Correction to this plan's own exploration:* `findings/reusable-layers-for-dit.md` item 5 claimed
`AdaLayerNormContinuous.call([x, cond])` "is DiT's `FinalLayer.forward` exactly". That "exactly" is
overstated in precisely this respect; everything else in the finding holds.

### 4.2 The training objective is packed into `y_true` — **D-002**

A `keras.losses.Loss` sees only `(y_true, y_pred, sample_weight)`, but upstream's objective
(`reference/gaussian_diffusion.py:463-494`) needs the per-sample timestep `t` and the clean latent
`x_start`, neither of which is in either tensor. A custom `train_step` is forbidden repo-wide, and
`sample_weight` is **not** a free side channel — Keras multiplies the per-sample loss by it, so
carrying `t` there would corrupt the objective.

So `y_true` carries them:

```
y_true = concat([noise, x_start, t_plane], axis=-1)     # [B, H, W, 2C+1]
         [0:C] = noise      [C:2C] = x_start      [2C:2C+1] = t broadcast over (H, W)
y_pred                                                   # [B, H, W, 2C]
```

The loss owns a `DDPMSchedule`, re-derives `x_t` from `(x_start, noise, t)`, computes
`mean_flat((noise - eps_pred)**2)` plus the variational-bound term with `stop_gradient(eps_pred)`,
and returns a per-sample `[B]` vector. `sample_weight` is left unused.

The cost is real and is not hidden: `y_true`'s channel count differs from `y_pred`'s, and the layout
is a hand-maintained contract between whoever builds the batch and the loss. The **premise** that
Keras tolerates it was measured, not assumed —
`tests/test_losses/test_ddpm_hybrid_loss.py::test_stock_fit_accepts_the_ragged_target_with_no_train_step_override`
runs a real `compile()`/`fit()` and asserts
`type(model).train_step is keras.Model.train_step`. The freezing is pinned by
`test_the_vb_term_is_frozen_out_of_the_mean`, and the alternative (MSE-only, which was **not**
taken) is falsified by `test_perturbing_only_the_variance_channels_changes_the_loss`.

### 4.3 The patch embedding uses upstream's FLATTENED xavier — **D-013**

Upstream initialises the patch-embed conv "like `nn.Linear`": it reshapes the kernel to
`(D, p*p*C_in)` and calls `xavier_uniform_` on that view (`reference/models.py`,
`initialize_weights`). Keras computes a convolution's fans over the full kernel shape
`(p, p, C_in, D)`, so its `fan_out` is `p*p*D`, **not** `D`. At `p = 2` that is a 4x difference in
`fan_out` and a real difference in the sampled range — with no shape, no count, no `get_config()`
and no round-trip symptom.

`flattened_linear_xavier(fan_in, fan_out)` writes the exact upstream limit and is passed as
`PatchEmbedding2D(kernel_initializer=...)`. Pinned by
`test_dit_model.py::TestThePatchEmbedInitializer`, which asserts the bound is the flattened one AND
that it is measurably wider than the Keras conv fan would give.

The helper is a **deliberate second copy** of `bit_diffusion/model.py:121` under D-001's rule; both
copies carry an anchor naming the other. It is five lines, pure, stateless and has no Keras layer.

### 4.4 Channels-last forces the unpatchify interleave to be RE-DERIVED

Upstream's `unpatchify` (`reference/models.py:211-224`) ends in `einsum('nhwpqc->nchpwq')`, which
targets **NCHW**. Transcribing it here would be wrong, and wrong in the way that is hardest to see.

Re-derived instead: reshaping the token tensor to `(B, h, w, p, q, c)` gives an element indexed
`[b, i, j, pi, pj, ci]` whose destination pixel is `row = i*p + pi`, `column = j*p + pj`. That
requires the axis order `(b, i, pi, j, pj, c)`, i.e. `transpose(0, 1, 3, 2, 4, 5)`.

**Measured, and the whole reason this has a dedicated guard:** injecting the transposed
`(0, 2, 4, 1, 3, 5)` gives **3 failed / 41 passed** — the two orientation arms plus their meta-arm —
while *every* shape assertion in the file stays green: output shape, `compute_output_shape`, the
`.keras` round trip and the parameter count. `test_dit_model.py::TestTheUnpatchifyOrientation`
therefore places a delta impulse at an asymmetric coordinate on a **non-square** token grid and
computes the destination index by hand, never by re-invoking the model's own arithmetic.

### 4.5 The model outputs exactly `0.0` at initialisation — a property, not a defect

Every block's modulation `Dense` and the whole final layer are zero-initialised
(`reference/models.py:200-209`), so `gate_msa = gate_mlp = 0`, every block is an exact identity, and
the model emits exactly `0.0` before a single gradient step. This is what makes a 28-block stack
trainable.

It is recorded here because it **breaks the repo's stock test oracles**: any "the output changed"
assertion is trivially satisfied or trivially unsatisfiable against a model that emits zeros, and a
gradient-flow oracle run at init against a mean-of-squares loss reports every weight dead. The suite
therefore runs the per-variable oracle **after one real optimizer step against a real loss with a
non-zero target**, and the identity itself is asserted positively by
`test_dit_blocks.py::TestTheIdentityAtInitPremise`.

### 4.6 `SpacedDiffusion` is folded into `GaussianDiffusion`

Upstream splits the two because `SpacedDiffusion` subclasses and overrides four methods
(`reference/respace.py`), but the entire difference is one index remap applied to `t` before the
model sees it: the tables are read at the **respaced** index while the model is handed the
**original** one. That is a two-line private method here, and a subclass for it would be structure
without content.

Pinned by `test_dit_diffusion.py::TestTheRespacingRemapIsReal`, whose arms first assert that the two
candidate index sets **differ** and only then assert which one arrived — so they cannot pass
vacuously. Bypassing the remap gives **3 failed / 56 passed**.

### 4.7 `clip_denoised` defaults to `False`, diverging from upstream's DEFAULT to match its BEHAVIOUR — **D-017**

Upstream's `GaussianDiffusion` defaults `clip_denoised=True`
(`reference/diffusion/gaussian_diffusion.py:188`), inherited from the ADM/IDDPM **pixel** codebase
where data is uint8 rescaled to `[-1, 1]`. DiT diffuses **VAE latents**, which are not in that range,
and upstream's own sampler passes `clip_denoised=False` at every DiT call site
(`reference/train_and_sample_excerpts.py:62`).

The port makes `False` the module-level default (`DEFAULT_CLIP_DENOISED`) shared by all seven entry
points, rather than transcribing `True` and relying on every caller to override it. A
silently-clipping latent sampler destroys the sample while shape, dtype and every finiteness check
stay green — a defect class this repo has repeatedly measured surviving a fully green suite.

Pinned by `test_dit_diffusion.py::TestTheClipDenoisedDefaultIsFalseForLatents`, including an arm that
reads `inspect.signature` for all seven entry points so a new method cannot quietly default the other
way, and an anti-vacuity arm proving the flag is not simply ignored.

### 4.8 The variance interpolation is DUPLICATED, and held in lockstep by an executable guard — **D-016**

The five lines `frac = (v+1)/2; log_var = frac*max_log + (1-frac)*min_log`
(`reference/gaussian_diffusion.py:204-212`) exist **twice**: inline in
`losses/ddpm_hybrid_loss.py`'s variational-bound term, and in `diffusion.py:p_mean_variance`.

Sharing them was rejected on all four available routes: a `models/` → `losses/` import is a new
dependency direction that becomes a cycle the moment the loss wants anything from the sampler; the
inverse would make every `fit()` drag a sampler in; `utils/ddpm_schedule.py` is pure-NumPy by design
and this is Keras-op code over a live model output; and a new shared module would spend the plan's
entire remaining file allowance on five lines. The loss's copy is also not extractable as-is — it
shares `coef1`/`coef2` with the true-posterior branch two lines later.

So the duplication is **converted from a hand-maintained invariant into an executable one**.
`test_dit_diffusion.py::TestTheVarianceInterpolationAgreesWithTheLoss` rebuilds the loss's *entire*
objective out of `GaussianDiffusion.p_mean_variance` plus the loss module's own KL and decoder-NLL
helpers and asserts equality with `DDPMHybridLoss.call` at `atol=1e-6, rtol=0`, over both beta
schedules. Perturbing only the sampler's copy (`/ 2.0` → `/ 2.5`) gives **2 failed**. A comment
asking a future reader to keep two files identical would not have.

### 4.9 The label table's initializer must be overridden

`ClassLabelEmbedding` defaults to Keras' `"uniform"`; upstream is `normal(std=0.02)`
(`reference/models.py`, `initialize_weights`). The override is therefore mandatory at the call site.

Dropping it gives **2 failed / 42 passed**
(`test_dit_model.py::TestTheLabelTableUsesTheUpstreamInitializer`): dispersion `0.020` against the
uniform default's `0.0289`, plus the tail past that distribution's hard bound of `0.05`. Nothing
about the shape, the count or the round trip changes — the model simply trains from a different
distribution.

### 4.10 `use_causal_mask` is never passed to the block's attention — **D-012**

The repo's own adaLN-Zero block (`layers/transformers/adaln_zero.py:114-118`) defaults to
`use_causal_mask=True`, and it is the precedent this block's stock-MHA construction follows in every
other respect. DiT's attention is **bidirectional** over image patches (`reference/models.py:101` —
timm `Attention`, which has no mask at all).

A causal mask changes no shape, no parameter count, no `get_config()` and no `.keras` round trip. It
only makes a later patch invisible to an earlier one, which trains a plausible, wrong model. The call
site therefore carries a written anchor saying the neighbouring precedent's default must **not** be
copied across, and `test_dit_blocks.py::TestTheAttentionIsNonCausal` is proven RED against it: **1
failed / 58 passed**.

A trap found while writing that guard, recorded so nobody re-derives the wrong probe: perturbing one
token by a **channel-uniform** `+5.0` measures a delta of `1.2e-07` at token 0, because
`keras.layers.LayerNormalization` subtracts the mean **regardless of `center=False`** — so the bump
never reaches the attention at all. The probe uses a per-channel bump, and the annihilation itself is
pinned by `test_a_uniform_bump_is_annihilated_by_norm1`.

### 4.11 Classifier-free guidance covers the first THREE channels, not `in_channels` — **D-014**

`forward_with_cfg` splits `model_out[..., :3]` / `model_out[..., 3:]`, reproducing
`reference/models.py:261`. Upstream leaves the `in_channels` form **commented out beside it**, with
the note that "for exact reproducibility reasons, we apply classifier-free guidance on only three
channels by default. The standard approach to cfg applies it to all channels."

Shipped as upstream has it, expressed as the named constant `CFG_GUIDED_CHANNELS = 3`, with an
in-code anchor stating that the obvious "fix" is wrong. At the published `in_channels = 4` this
leaves one epsilon channel unguided. The alternative is not more correct; it is a *different* model
at sampling time, and one that reproduces none of the published `cfg_scale` numbers.

**Honest status of the guard.** As of step 8 this divergence is **NOT YET PINNED**: injecting
`model_out[..., :self.in_channels]` was measured **INERT against the whole package suite — 0 failed /
44 passed** at step 6. It is reported as inert rather than credited. The dedicated guard,
`test_the_cfg_guidance_covers_only_three_channels.py`, is step 10 of the plan and must be proven RED
there. Until it lands, the only thing exercising this path is
`test_dit_diffusion.py::TestClassifierFreeGuidanceEndToEnd`, which proves the guided loop *runs* and
nothing about the split.

### 4.12 `DiffusionConfig` delegates the schedule's legality instead of encoding a threshold — **D-010**

`'linear'` scales its endpoints by `1000 / num_timesteps`, so `beta_end = 20 / T`, which exceeds `1`
below `T = 20`. The obvious `__post_init__` check is a minimum-step constant — and every candidate
constant is measurably false. An executable census over `T` in `1..25` shows the accepted set is
`{1} ∪ [20, 25]`, **not a floor**: `np.linspace(a, b, 1)` returns `[a]` and drops the illegal
endpoint, so `T = 1` is legal while `T = 2..19` are not. No threshold expression states that set.

`DiffusionConfig.__post_init__` therefore calls `self.build_schedule()` and re-raises any
`ValueError` with a message naming `num_timesteps` and `schedule_name`. The cost is that constructing
a config allocates the full `T`-length beta array (~112 KB at the default `T = 1000`) instead of
comparing one integer. Pinned, including the `T = 1` exception and the census arm, by
`test_dit_config.py::TestTheLinearScheduleFloorIsDelegatedNotHardcoded`.

*This corrects a false claim written by this plan's own earlier step:* `utils/ddpm_schedule.py:143`
originally stated that `'linear'` is undefined below `num_timesteps = 50`. It is not; the docstring
was corrected in the same commit.

### 4.13 `keras.backend.standardize_dtype` is forbidden under `models/` — **D-018**

The natural spelling of "what floating dtype is this tensor" is `keras.backend.standardize_dtype`.
Under `models/` it is banned by a whole-tree AST sweep
(`tests/test_models/test_package_api_contract.py::TestNoKeras2Residues::test_no_keras_backend_calls`),
and the first version of `diffusion.py` took that repo-wide contract from **3 failed / 600 passed to
4 failed / 599 passed**.

The port uses `getattr(dtype, "name", None) or str(dtype)` — the same call `bit_diffusion/sde.py:123`
makes — plus `keras.config.floatx()` in place of `keras.backend.floatx()`. `str` alone is
insufficient: a `tf.DType` stringifies as `"<dtype: 'float64'>"`.

**Open, and deliberately not hidden:** the guard is scoped to `models/`, so this plan's own
`losses/ddpm_hybrid_loss.py` still uses the banned call and is *unguarded* rather than approved. Two
spellings of one question now ship in one plan. That is the "a scoped gate cannot see an outside
consumer" shape, and it is flagged for adjudication rather than silently reconciled.

### 4.14 Sampling is seeded explicitly, because `set_random_seed` does not reproduce here

Measured on this Keras: `keras.utils.set_random_seed` does **not** re-seed an already-created global
`SeedGenerator` — two identically-seeded sampling runs disagree. Every sampler entry point therefore
takes an explicit `seed` (an `int` or a `keras.random.SeedGenerator`), and the *measurement itself*
is asserted by `test_dit_diffusion.py::TestSeedingIsExplicit::test_set_random_seed_alone_does_not_reproduce`,
so a future Keras that fixes this reddens loudly instead of drifting silently.

Relatedly, the sampling loops are **eager Python loops** — one model call per step, pinned by a
call-count test (`TestTheLoopIsAnEagerPythonLoop`). The per-step methods themselves are traceable.

### 4.15 The sin-cos trio and the timestep ladder now exist TWICE in the repo — **D-001**, accepted

Both were promoted to `layers/embedding/` so this package does not import from a sibling *model*
package. `bit_diffusion` was deliberately left untouched: moving its registered
`DiTXATimestepEmbedder` would change the registration key and break a package shipped the same day,
and a cross-package `models/` → `models/` import inverts the dependency direction with no precedent
in this repo.

So there are two copies, and they can silently drift. The follow-up — migrate `bit_diffusion` onto
the shared home once a checkpoint-key migration is worth doing — is recorded here and in D-001, and
is **not scheduled**. The bounded mitigation:
`tests/test_layers/test_embedding/test_the_sincos_grid_is_w_first.py::TestItAgreesWithTheBitDiffusionSibling`
asserts the two copies still produce the same table, so a drift reddens.

**A known documentation defect in the sibling, deliberately NOT edited.**
`bit_diffusion/blocks.py:855-857` claims:

> `# Passing (grid_h, grid_w) instead transposes the table with no shape change.`

That is **false**. On a square grid `grid_h` and `grid_w` are the same `np.arange`, so swapping the
two `np.meshgrid` arguments is an **exact no-op** — measured: the injection left the guard suite green
at 31 passed, a false negative that would have credited an unproven guard. The two mutations that do
transpose the table are `indexing="ij"` (12 failed / 23 passed) and swapping `grid[0]`/`grid[1]`
inside `get_2d_sincos_pos_embed_from_grid` (13 failed / 22 passed); those are what the shared module's
guard is proven against, and the inertness of the swap is itself pinned by a test so nobody
re-derives the false claim. The sibling's line is left uncorrected under D-001's "leave
`bit_diffusion` untouched" rule and is recorded here as an upstream-sibling defect this port
knowingly did not fix.

### 4.16 The diagram requirement is a test, not a promise — **D-006**

The user's Sphinx + ASCII-diagram requirement is enforced by
`tests/test_models/test_dit/test_the_package_surface.py`, which asserts every public class under
`dit/` carries a `.. code-block:: text` block with the house box characters, a `⊕`, a `[B, ...]`
shape annotation and Sphinx `:param:` fields, and that every module docstring ends in a
`References:` block.

Its scope is stated in its own docstring and repeated here because it is easy to oversell: this is a
**PRESENCE-and-SHAPE check, not a truth check**. A correct-looking box drawing of the wrong
architecture passes every assertion in it. The truth of each mechanism the diagrams depict is pinned
separately, by §4.4's orientation guard, §4.10's causality probe, the per-chunk attribution probe and
the sampler's oracle comparisons.

### 4.17 What this port cannot do

Not a divergence but a limitation, stated where it cannot be missed:

- **There is no VAE anywhere in this repo**, trained or downloadable. Sampling produces *latents*;
  nothing here decodes them to pixels, so no sample-quality figure, no FID and no published-number
  comparison is derivable from this tree.
- **There are no pretrained weights.** `pretrained=True` raises `NotImplementedError` naming the
  variant.
- **Upstream checkpoints are not loadable** even if you had them — §4.1 alone guarantees a weight
  layout difference.
- **Only `DiT-S/2` at reduced geometry has ever been run.** The other eleven variants have been
  *built* and measured (`README.md`), and nothing more. None has been trained for a single step.
