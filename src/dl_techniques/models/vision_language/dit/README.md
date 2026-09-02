# DiT — the class-conditional latent Diffusion Transformer

[`PORT_NOTES.md`](PORT_NOTES.md) is the authority on *what did and did not survive the port from
PyTorch*, including every deliberate divergence and every honest limitation. This file is the
orientation layer and does not restate it.

## What it is

A Keras 3, **channels-last** port of Peebles & Xie's **DiT** (arXiv:2212.09748, upstream
`chuanyangjin/fast-DiT`): a transformer that denoises VAE latents instead of pixels, conditioned on
a class label and a diffusion timestep through **adaLN-Zero**. The package ships the model, the two
block layers it is built from, the twelve published configurations, and the Gaussian-diffusion
machinery the model is useless without — the forward process, the reverse ancestral and DDIM steps,
their loops, timestep respacing, and classifier-free guidance.

## The architecture, in one paragraph

The latent `x` is cut into non-overlapping `p x p` patches and projected to `hidden_size`, a
**frozen** 2-D sin-cos table is added, and the timestep and label embeddings are **summed** into a
single conditioning vector `c` of shape `[B, D]` that every block sees. Each block is
`x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))` followed by
`x + gate_mlp * mlp(modulate(norm2(x), shift_mlp, scale_mlp))`, where the six modulation chunks come
from one zero-initialised `Dense(6*D)` on `SiLU(c)` — so `gate_msa = gate_mlp = 0` at init and
**every block is an exact identity**. The final layer modulates once more and projects to
`p*p*out_channels` through another zero-initialised `Dense`, so the whole model outputs **exactly
`0.0`** before a single gradient step. That is a property, not a bug; see `PORT_NOTES.md` §4.5.

```
   x [B, H, W, C]        t [B]              y [B]
        |                  |                  |
   PatchEmbedding2D   TimestepEmbedding  ClassLabelEmbedding
        |                  |                  |
   + pos_embed (frozen)    +--------> (+) <----+
        |                                |
        | [B, T, D]                      | c [B, D]
        v                                |
   DiTBlock x depth  <-------------------+     adaLN-Zero, 6-way
        |
   DiTFinalLayer     <-------------------+     2-way + zero-init Dense
        |
   unpatchify -> [B, H, W, out_channels]       out_channels = 2C if learn_sigma
```

With `learn_sigma=True` (the default and the published setting) the **second half of the channel
axis is a variance-interpolation logit, not a second epsilon**: the sampler maps it through
`frac = (v+1)/2; log_var = frac*log(beta_t) + (1-frac)*log(posterior_var_t)`.

## Variants

`DiT.MODEL_VARIANTS` (an alias of `DIT_VARIANTS`, not a copy). Parameter counts are **MEASURED** on
a built model, not estimated, at the **published latent geometry**: `input_size=32`,
`in_channels=4`, `num_classes=1000`, `learn_sigma=True` — i.e. a `32 x 32 x 4` SD-VAE latent of a
256x256 image. The counts move with all four of those, so a count quoted without its geometry is
meaningless.

Re-derived with:

```python
m = DiT.from_variant(v, input_size=32, in_channels=4, num_classes=1000, learn_sigma=True)
m.build([(None, 32, 32, 4), (None,), (None,)])
sum(int(np.prod(w.shape)) for w in m.weights)     # Keras 3 Variables have no `.size`
```

| Variant | `depth` | `hidden_size` | `patch_size` | `num_heads` | Tokens | Parameters (total) | of which frozen `pos_embed` |
|---|---|---|---|---|---|---|---|
| `DiT-S/2` | 12 | 384 | 2 | 6 | 256 | **32,963,488** | 98,432 |
| `DiT-S/4` | 12 | 384 | 4 | 6 | 64 | **32,945,152** | 24,704 |
| `DiT-S/8` | 12 | 384 | 8 | 6 | 16 | **33,148,288** | 6,272 |
| `DiT-B/2` | 12 | 768 | 2 | 12 | 256 | **130,512,544** | 196,736 |
| `DiT-B/4` | 12 | 768 | 4 | 12 | 64 | **130,475,776** | 49,280 |
| `DiT-B/8` | 12 | 768 | 8 | 12 | 16 | **130,881,664** | 12,416 |
| `DiT-L/2` | 24 | 1024 | 2 | 16 | 256 | **458,102,944** | 262,272 |
| `DiT-L/4` | 24 | 1024 | 4 | 16 | 64 | **458,053,888** | 65,664 |
| `DiT-L/8` | 24 | 1024 | 8 | 16 | 16 | **458,594,944** | 16,512 |
| `DiT-XL/2` | 28 | 1152 | 2 | 16 | 256 | **675,129,760** | 295,040 |
| `DiT-XL/4` | 28 | 1152 | 4 | 16 | 64 | **675,074,560** | 73,856 |
| `DiT-XL/8` | 28 | 1152 | 8 | 16 | 16 | **675,683,200** | 18,560 |

All twelve were built and measured on this machine (CPU only, peak RSS 4.23 GB at `DiT-XL/8`), so
**no row is estimated or extrapolated**. Building is not running: `S/2` at reduced geometry is the
only configuration that has been forward-passed, trained for a step or round-tripped in the test
suite. **None of the twelve has been trained** — there are no pretrained weights, and
`pretrained=True` raises `NotImplementedError` naming the variant.

Note the non-monotonicity at `/8`: a bigger patch means fewer tokens and a much smaller `pos_embed`,
but a `p*p*C`-wide input projection and a `p*p*2C`-wide output projection, and at these widths the
second effect wins. The count is not a function of `patch_size` in the direction most readers guess.

`tests/test_models/test_dit/test_the_package_surface.py` re-derives the three `DiT-S/*` rows from
this table and fails if they drift. The `B`/`L`/`XL` rows are **not** in that arm — building all
twelve costs ~17 s and 4 GB, which is not a per-run test cost.

## Usage

Every block below was executed; the output shown is the real output.

### Forward pass

```python
import keras
from dl_techniques.models.vision_language.dit import DiT

model = DiT.from_variant("DiT-S/2", input_size=8, in_channels=4, num_classes=10)

x = keras.random.normal((2, 8, 8, 4), seed=0)               # noised latent, NHWC
t = keras.ops.convert_to_tensor([10, 900], dtype="int32")   # diffusion timestep
y = keras.ops.convert_to_tensor([3, 7], dtype="int32")      # class labels

out = model([x, t, y], training=False)
print(out.shape)                                   # (2, 8, 8, 8)
print(float(keras.ops.max(keras.ops.abs(out))))    # 0.0
```

The `0.0` is the zero-init identity property, measured, not rounded.

### Sampling with classifier-free guidance

The caller stacks a conditional half and an unconditional half into **one** batch, exactly as
upstream's `sample.py` does. The null label is the index `num_classes` — the last row of the table,
which exists only because `class_dropout_rate > 0`.

```python
import keras
import numpy as np
from dl_techniques.models.vision_language.dit import DiT, GaussianDiffusion

model = DiT.from_variant("DiT-S/2", input_size=8, in_channels=4, num_classes=10)

# 1000-step training chain, respaced to 4 steps for sampling.
gd = GaussianDiffusion.from_name("linear", 1000, timestep_respacing=4)
print(gd.num_timesteps)                              # 4

n = 2
labels = np.array([3, 7], dtype="int32")
y_null = np.full((n,), 10, dtype="int32")            # the null row IS index num_classes
y = keras.ops.convert_to_tensor(np.concatenate([labels, y_null]))

z = keras.ops.convert_to_tensor(
    np.tile(np.random.default_rng(0).normal(size=(n, 8, 8, 4)).astype("float32"),
            (2, 1, 1, 1))
)

def guided(x, t, **kwargs):
    return model.forward_with_cfg(x, t, training=False, **kwargs)

samples = gd.p_sample_loop(guided, noise=z,
                           model_kwargs={"y": y, "cfg_scale": 4.0}, seed=3)
latents = keras.ops.convert_to_numpy(samples)[:n]    # sample.py drops the uncond half
print(latents.shape, bool(np.isfinite(latents).all()))   # (2, 8, 8, 4) True
```

Two things a reader of the PyTorch original will not expect: every sampling entry point takes an
**explicit `seed`** (`keras.utils.set_random_seed` does not reproduce an already-created global
`SeedGenerator` here — measured, and pinned by a test), and `clip_denoised` defaults to **`False`**
(`PORT_NOTES.md` §4.7).

### Training objective

`DDPMHybridLoss` is upstream's `MSE + LEARNED_RANGE` objective computed under **stock
`compile()`/`fit()`** — there is no custom `train_step`. Everything a `keras.losses.Loss` cannot
otherwise see (`x_start` and the per-sample `t`) is packed into `y_true`, which is therefore
`2C+1` channels wide against a `2C`-wide `y_pred`:

```python
import keras
import numpy as np
from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit import DiT

T, C = 1000, 4
loss = DDPMHybridLoss(schedule_name="linear", num_timesteps=T, in_channels=C)
sched = loss.schedule

rng = np.random.default_rng(0)
B = 4
x_start = rng.normal(size=(B, 8, 8, C)).astype("float32")
noise = rng.normal(size=(B, 8, 8, C)).astype("float32")
t = rng.integers(0, T, size=(B,)).astype("int32")

# forward process, exactly what the loss re-derives internally
a = sched.sqrt_alphas_cumprod[t][:, None, None, None]
s = sched.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
x_t = (a * x_start + s * noise).astype("float32")

t_plane = np.broadcast_to(t[:, None, None, None].astype("float32"), (B, 8, 8, 1))
y_true = np.concatenate([noise, x_start, t_plane], axis=-1)
print(y_true.shape)                                       # (4, 8, 8, 9)

model = DiT.from_variant("DiT-S/2", input_size=8, in_channels=C, num_classes=10)
model.compile(optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.0), loss=loss)
print(type(model).train_step is keras.Model.train_step)   # True

y = rng.integers(0, 10, size=(B,)).astype("int32")
h = model.fit([x_t, t, y], y_true, epochs=1, batch_size=2, verbose=0)
print(round(h.history["loss"][0], 4))                     # 1.0753
```

The channel layout of `y_true` is `[0:C] = noise`, `[C:2C] = x_start`, `[2C:2C+1] = t` broadcast
over `(H, W)`. It is a hand-maintained contract between whoever builds the batch and the loss;
`PORT_NOTES.md` §4.2 says why it exists and what it costs.

### Running the trainer

`src/train/dit/` trains the model end-to-end on synthetic class-correlated latents, with no
external dataset and no VAE:

```bash
# wiring proof on CPU, ~70 s, writes <repo>/results/dit_<variant>_<timestamp>/
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" python -m train.dit.train_dit --smoke

# a real (still synthetic-latent) run
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.dit.train_dit \
    --variant DiT-S/2 --input-size 32 --num-classes 10 --epochs 100

# real, pre-encoded latents (the contract is in train/dit/synthetic_data.py)
... --train-npz /data/dit/train-00000.npz --val-npz /data/dit/val-00000.npz
```

`--smoke` shrinks the run to a wiring proof and, deliberately, also changes the model *size* and the
latent *geometry* (`DiT-S/2` at an 8x8x4 grid over a 50-step chain) — the carve-out is anchored at
`train_dit.py`'s `SMOKE_PRESET`. Any flag you type explicitly wins over the preset. The defaults
otherwise reproduce upstream's recipe: `AdamW(lr=1e-4, weight_decay=0)` with **no** LR schedule, an
EMA of the trainable weights at decay `0.9999`, and stock `compile()`/`fit()`.

The run directory is TIMESTAMPED by default (`dit_<variant>_<YYYYmmdd_HHMMSS>`), so a second run
never overwrites the first one's `best_model.keras` / `final_model.keras` /
`training_history.json` / `training_log.csv`; pass `--experiment-name` to pin a name.

**The falling `val_loss` has a negative control.** A loss that falls is not by itself evidence of
learning, so the smoke configuration was also run with the optimizer made inert. `--learning-rate 0`
is rejected by `TrainingConfig`, so the control used `--learning-rate 1e-30` — an AdamW update of
order 1e-30 against float32 weights of order 1e-2, with `weight_decay=0`:

| epoch | shipped `lr=1e-4` | control `lr=1e-30` |
|---|---|---|
| 1 | 1.329365 | 1.390627 |
| 2 | 1.270306 | 1.390627 |
| 3 | 1.168718 | 1.390627 |

The control's `val_loss` is bit-identical across all three epochs, so none of the shipped arm's
`-0.160648` comes from the data pipeline. Recorded as D-029.

## Modules

| File | Holds |
|---|---|
| `config.py` | `DIT_VARIANTS` (the twelve rows), `VARIANT_FIELDS`, `normalize_variant_name`, `get_variant_config`, `DiffusionConfig` |
| `blocks.py` | `DiTBlock` (6-way adaLN-Zero), `DiTFinalLayer` (2-way), the two chunk-name tuples |
| `model.py` | `DiT`, `create_dit`, `forward_with_cfg`, `unpatchify_tokens`, `flattened_linear_xavier` |
| `diffusion.py` | `GaussianDiffusion` — `q_sample`, `q_posterior_mean_variance`, `p_mean_variance`, `p_sample`/`ddim_sample` and their loops, timestep respacing |

Shared assets this package uses but does not own: `PatchEmbedding2D`, `ClassLabelEmbedding`,
`TimestepEmbedding` and `get_2d_sincos_pos_embed` (`dl_techniques.layers.embedding`),
`AdaLayerNormZero` / `AdaLayerNormContinuous` / `modulate`
(`dl_techniques.layers.transformers.sd3_adaln`), `create_ffn_layer("gelu_tanh")`
(`dl_techniques.layers.ffn.factory`), `DDPMSchedule` (`dl_techniques.utils.ddpm_schedule`) and
`DDPMHybridLoss` (`dl_techniques.losses.ddpm_hybrid_loss`). **No attention, FFN, normalization,
patch-embedding or label-embedding class is defined anywhere under `dit/`** — see
`PORT_NOTES.md` §2 and §3.

## What this package cannot do

- **There is no VAE anywhere in this repo**, trained or downloadable. Sampling produces *latents*;
  nothing here decodes them to pixels, and no FID or published-number comparison is possible from
  this tree.
- **There are no pretrained weights.** `pretrained=True` raises `NotImplementedError` naming the
  variant; a local path is loaded if you pass one as a string.
- Upstream checkpoints are **not** loadable even if you had them: the final layer's 2-way
  modulation splits `scale, shift` where upstream splits `shift, scale` (`PORT_NOTES.md` §4.1),
  among other weight-layout differences.
- The sampling loops are **eager Python loops** — one model call per step, pinned by a call-count
  test. The per-step methods themselves are traceable.
