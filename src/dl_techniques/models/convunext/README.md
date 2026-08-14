# ConvUNext

A U-Net encoder/decoder built out of ConvNeXt (V1 or V2) blocks, exposed as **one
functional builder**:

```python
create_convunext(input_shape, use_bias=True, ...)  -> keras.Model
```

`use_bias=True` is the ordinary, bias-carrying network. `use_bias=False` is the
**bias-free** network used for image restoration, where the absence of any additive
term makes the model degree-1 homogeneous (`f(a*x) = a*f(x)`) and therefore
scale-invariant across noise levels. Both arms are the same graph.

> **History, because it changes how you read the rest of the repo.** Until 2026-08-14
> this package shipped a subclassed `ConvUNextModel`, and
> `models/bias_free_denoisers/bfconvunext.py` shipped a separate functional builder for
> the bias-free arm. They were two implementations of one architecture. They are now
> merged onto the **functional** graph, here. `ConvUNextModel`, its own `ConvUNextStem`,
> its bespoke `create_inference_model_from_training_model`, `PRETRAINED_WEIGHTS` and
> `_download_weights` **no longer exist**, and neither does `src/train/convunext/`.
> `bfconvunext.py` survives as thin `use_bias=False` wrappers plus the Keras registrar
> (see [§8](#8-relationship-to-bfconvunext)).

Every code block below was executed against this tree before it was written down.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [`use_bias` and the three documented asymmetries](#3-use_bias-and-the-three-documented-asymmetries)
4. [The bias-free guardrails](#4-the-bias-free-guardrails)
5. [`block_normalization` — the default rule](#5-block_normalization--the-default-rule)
6. [Knobs absorbed from the deleted subclass](#6-knobs-absorbed-from-the-deleted-subclass)
7. [Deep supervision, bottleneck exposure, serialization](#7-deep-supervision-bottleneck-exposure-serialization)
8. [Relationship to `bfconvunext`](#8-relationship-to-bfconvunext)
9. [Package surface](#9-package-surface)
10. [Tests](#10-tests)

---

## 1. Quick start

```python
from dl_techniques.models.convunext import create_convunext

model = create_convunext(input_shape=(64, 64, 3), depth=3, initial_filters=16)
print(model.name, model.output_shape, model.count_params())
# convunext (None, 64, 64, 3) 510307
```

Named variants (`tiny`, `small`, `base`, `large`, `xlarge`) fill in
`depth` / `initial_filters` / `blocks_per_level` / `convnext_version` /
`drop_path_rate`:

```python
from dl_techniques.models.convunext import create_convunext_variant, CONVUNEXT_CONFIGS

print(sorted(CONVUNEXT_CONFIGS))
# ['base', 'large', 'small', 'tiny', 'xlarge']
tiny = create_convunext_variant('tiny', input_shape=(64, 64, 1))
print(tiny.count_params())
# 1939521
```

`CONVUNEXT_CONFIGS` is the **single** variant dict for both arms; the bias-free
wrappers in `bfconvunext` reuse it rather than carrying their own copy.

Default output channel count is the INPUT channel count — the
denoiser/autoencoder contract. Override it with `output_channels`
([§6](#6-knobs-absorbed-from-the-deleted-subclass)).

## 2. Architecture

```
input
  |
  stem                    ConvUNextStem  (Conv2D k=7 -> <stem_normalization> -> activation)
  |                       or a frozen Gabor depthwise bank + 1x1 projection (use_gabor_stem)
  |
  encoder level i         blocks_per_level x ConvNeXt block   -> skip_i
  |                       DownsampleAndSkip                   (max / average / strided_conv,
  |                                                            or a Laplacian pyramid split)
  bottleneck              channel adjust
  |                       [bottleneck_attention_blocks x residual SpatialLinearAttention]
  |                       blocks_per_level x ConvNeXt block, drop-path RAMP
  |
  decoder level i         UpSampling2D -> concat(skip_i) -> channel adjust
  |                       blocks_per_level x ConvNeXt block
  |                       [supervision head -> supervision output]   (enable_deep_supervision)
  |
  final_output            Conv2D(output_channels, 1) -> final_activation     (include_top=True)
  or decoder_features     zero-parameter linear tap                          (include_top=False)
```

Channels at encoder level `i` are `int(round(initial_filters * filter_multiplier ** i))`.
The encoder junction is one serializable layer,
`dl_techniques.layers.downsample_and_skip.DownsampleAndSkip`, shared with `bfunet`; it
returns `(skip, downsampled)` and owns the `max` / `average` / `strided_conv` /
Laplacian dispatch in one place.

Optional structure, all off by default and all contributing zero layers when off:
`use_gabor_stem`, `use_laplacian_pyramid` (+ `high_freq_blocks`),
`bottleneck_attention_blocks`, `zero_pad_channels`, `extra_zero_output_channels`,
`final_projection_groups`, `expose_bottleneck`, `enable_deep_supervision`.
Each parameter is documented in full on `create_convunext`'s docstring — that
docstring, not this file, is the reference.

## 3. `use_bias` and the three documented asymmetries

```python
bf = create_convunext(input_shape=(64, 64, 1), use_bias=False, depth=3, initial_filters=16)
on = create_convunext(input_shape=(64, 64, 1), use_bias=True,  depth=3, initial_filters=16)
print('bias adds', on.count_params() - bf.count_params(), 'parameters')
# bias adds 5281 parameters
```

`use_bias` is threaded into the stem conv, every channel-adjust conv, every ConvNeXt
block, every deep-supervision head, the final projection, and the `strided_conv`
junction. **Three sites are deliberately NOT threaded**, and `use_bias=False`
therefore does not mean "provably, strictly bias-free":

| Site | Behaviour | Why |
|---|---|---|
| `GlobalResponseNormalization`'s `beta` | stays trainable (`use_beta` is never passed) | Threading it would change the bias-free arm's parameter count and numerics. Nothing in the repo enforces its absence today; the trainer's `verify_bias_free` logs it and does not raise. Pre-existing, now documented. |
| `SpatialLinearAttention`'s internal attention | hardcoded `use_bias=False` | Held by an earlier decision (`plan_2026-07-11_bb4b38b5/D-001`) that forbids adding knobs there. It is bias-free on BOTH arms. |
| The frozen Gabor bank | hardcoded `use_bias=False`, `trainable=False` | A frozen biased filter bank is a meaningless construct. |

A fourth asymmetry is about activations rather than bias: `block_activation`,
`stem_activation` and `supervision_activation` all default to `'gelu'`, which is NOT
positively homogeneous, and they are **deliberately exempt** from the guardrails below.
Guarding them would make the shipped default configuration raise. Consequence, stated
plainly: **passing the guardrails is not a homogeneity certificate.**

## 4. The bias-free guardrails

Under `use_bias=False` only, `create_convunext` validates three arguments against an
**allowlist** of positively homogeneous activations (an allowlist, not a denylist: a
denylist silently admits every activation nobody thought of).

```python
from dl_techniques.models.convunext import model as convunext_model
print(sorted(a for a in convunext_model.POSITIVELY_HOMOGENEOUS_ACTIVATIONS if a))
# ['leaky_relu', 'linear', 'relu']       (plus None)

for kw in ({'final_activation': 'sigmoid'}, {'supervision_norm_center': True}):
    try:
        create_convunext(input_shape=(32, 32, 1), use_bias=False, depth=2,
                         initial_filters=8, **kw)
    except ValueError as e:
        print(type(e).__name__, str(e).split('.')[0])
# ValueError final_activation='sigmoid' is not positively homogeneous, and use_bias=False requires it to be
# ValueError supervision_norm_center=True is incompatible with use_bias=False: the deep-supervision head LayerNorm's `center` adds a trainable additive offset (beta), which is a bias by another name
```

| Argument | Under `use_bias=False` |
|---|---|
| `final_activation` | must be in the allowlist, else `ValueError` |
| `gabor_activation` | must be in the allowlist **only when `use_gabor_stem=True`** — otherwise it reaches no layer and raising would be a false positive |
| `supervision_norm_center=True` | `ValueError`, **even when `enable_deep_supervision=False`** (the guard's predicate is a pure function of its arguments; the caller declared a contradictory intent) |
| `block_normalization='layernorm'` | `logger.warning`, **never** a raise — see [§5](#5-block_normalization--the-default-rule) |
| a CALLABLE activation | warns; a callable's homogeneity is not statically checkable |
| `downsample_pool_type='max'` | **not guarded**: max pooling is non-linear but IS positively homogeneous |

All of it is inert on the bias-carrying arm:

```python
create_convunext(input_shape=(32, 32, 1), use_bias=True, depth=2, initial_filters=8,
                 final_activation='sigmoid', supervision_norm_center=True)
# builds; all three guards inert

create_convunext(input_shape=(32, 32, 1), use_bias=False, depth=2, initial_filters=8,
                 use_gabor_stem=False, gabor_activation='gelu')
# builds; the gabor guard is SCOPED to use_gabor_stem=True
```

`POSITIVELY_HOMOGENEOUS_ACTIVATIONS` is the single owner of this rule.
`src/train/bfunet/common.py` DERIVES its `GABOR_ACTIVATIONS` argparse choices from it
rather than re-spelling the set.

## 5. `block_normalization` — the default rule

Read this one carefully; the asymmetry is real and it surprises people.

| Entry point | `block_normalization` |
|---|---|
| `convunext.create_convunext(...)` | `'layernorm'` (the builder default, on BOTH arms) |
| `convunext.create_convunext_variant(...)` (bias-ON) | `'layernorm'` |
| `bfconvunext.create_convunext_denoiser(...)` | `'layernorm'` |
| **`bfconvunext.create_convunext_variant(...)`** | **`'batchnorm'`** |

Only the last one flips, and it flips by `kwargs.setdefault(...)` inside
`bfconvunext.create_convunext_variant`, so a caller-supplied value always wins. The
key is deliberately absent from the shared `CONVUNEXT_CONFIGS`: putting it there
would flip the bias-carrying variants too.

Why it matters: `'layernorm'` is per-input `LayerNormalization`, which is scale
INVARIANT (degree 0), not degree-1 homogeneous. `'batchnorm'` is the variance-only
`BiasFreeBatchNorm`, which at `training=False` divides by a frozen constant and so
restores `f(a*x) = a*f(x)`. The named bias-free variants therefore get the
homogeneous choice; the bare bias-free builder keeps the historical default, which a
byte-identity test pins.

## 6. Knobs absorbed from the deleted subclass

**`downsample_pool_type`** gains a third value:

```python
for pt in ('max', 'average', 'strided_conv'):
    m = create_convunext(input_shape=(32, 32, 1), depth=2, initial_filters=8,
                         downsample_pool_type=pt)
    j = m.get_layer('encoder_downsample_0')
    print(f"{pt:13s} {type(j).__name__:18s} trainable weights at junction: {len(j.trainable_weights)}")
# max           DownsampleAndSkip  trainable weights at junction: 0
# average       DownsampleAndSkip  trainable weights at junction: 0
# strided_conv  DownsampleAndSkip  trainable weights at junction: 2
```

`'strided_conv'` is a learned `Conv2D(kernel_size=2, strides=2)` that threads
`use_bias`. It is **channel-preserving** (unlike the deleted subclass's fused
downsample-and-widen), and its skip is the RAW INPUT, exactly like the pooling
branches — both callers already widen channels in a separate, separately-named step
that a channel-changing junction would silently duplicate. With `use_bias=False` a
strided conv is linear and homogeneous, so it is legal on the bias-free arm and is
deliberately not guarded.

**`stem_normalization`** routes the stem's normalization through the norms factory:

```python
for sn in ('global_response_norm', 'layer_norm'):
    m = create_convunext(input_shape=(32, 32, 1), depth=2, initial_filters=8,
                         stem_normalization=sn)
    print(sn, '->', type(m.get_layer('encoder_level_0_stem').norm).__name__)
# global_response_norm -> GlobalResponseNormalization
# layer_norm -> LayerNormalization
```

The default `'global_response_norm'` is the ConvNeXt-V2 / bias-free choice;
`'layer_norm'` reproduces the standard ConvNeXt stem the deleted subclass built. Only
used when `use_gabor_stem=False`.

**`output_channels`** defaults to `input_shape[-1]`; set it for a non-reconstruction
head. It also drives every deep-supervision output and the `extra_zero_output_channels`
tail:

```python
seg = create_convunext(input_shape=(32, 32, 3), depth=2, initial_filters=8,
                       output_channels=1)
print(seg.output_shape)
# (None, 32, 32, 1)
```

**`include_top` — and its DOCUMENTED DIVERGENCE.**

```python
head = create_convunext(input_shape=(32, 32, 3), depth=2, initial_filters=8)
back = create_convunext(input_shape=(32, 32, 3), depth=2, initial_filters=8,
                        include_top=False)
print('with top :', head.output_shape, len(head.weights), 'weights')
print('headless :', back.output_shape, len(back.weights), 'weights')
print('tap      :', back.get_layer('decoder_features').name)
# with top : (None, 32, 32, 3) 124 weights
# headless : (None, 32, 32, 8) 122 weights
# tap      : decoder_features

back.get_layer('final_output')
# ValueError: No such layer: final_output. ...
```

The deleted `ConvUNextModel` CONSTRUCTED its final projection regardless and merely
skipped applying it, so `include_top=False` still carried the head's weights and a
checkpoint could move between the two settings. **A functional graph cannot reproduce
that**, and this was measured rather than argued: `keras.Model(inputs, outputs)` prunes
every layer not on a path to an output, so a constructed-but-unapplied projection owns
no weights and is not reachable through `get_layer`. The weight-compatibility contract
is therefore GONE, not preserved — `include_top=False` yields a strictly smaller
weight list, its primary output is the full-resolution decoder feature map
(`initial_filters` channels), and `set_weights` between the two settings raises.
`include_top=False` combined with `final_projection_groups != 1` raises rather than
silently ignoring the argument.

## 7. Deep supervision, bottleneck exposure, serialization

`enable_deep_supervision=True` returns `[final_output, supervision...]`;
`expose_bottleneck=True` appends a TRAILING `bottleneck` output. Convert a
deep-supervision training model to its single-output inference form with the shared
utility (re-exported from this package):

```python
from dl_techniques.models.convunext import create_inference_model_from_training_model

train_m = create_convunext(input_shape=(32, 32, 1), depth=3, initial_filters=8,
                           enable_deep_supervision=True)
print('training outputs :', len(train_m.outputs))
inf_m = create_inference_model_from_training_model(train_m)
print('inference outputs:', len(inf_m.outputs), inf_m.output_shape)
# training outputs : 3
# inference outputs: 1 (None, 32, 32, 1)
```

This is `dl_techniques.utils.deep_supervision.create_inference_model_from_training_model`
— the ONE implementation. The subclass's bespoke copy is gone.

Full `.keras` round trip:

```python
import os, tempfile, numpy as np, keras

m = create_convunext(input_shape=(32, 32, 1), depth=2, initial_filters=8)
x = np.random.rand(1, 32, 32, 1).astype('float32')
y0 = m(x, training=False)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'convunext.keras')
    m.save(p)
    r = keras.models.load_model(p)
print('max abs delta:', float(np.max(np.abs(np.array(y0) - np.array(r(x, training=False))))))
# max abs delta: 0.0
```

Assert `training=False` explicitly on both sides. `training=None` is not inference and
produces round-trip deltas that look like reinitialized weights.

## 8. Relationship to `bfconvunext`

`src/dl_techniques/models/bias_free_denoisers/bfconvunext.py` is now 232 lines
(`wc -l`) and does exactly two jobs:

1. **Thin `use_bias=False` wrappers** at their historical, frozen signatures:
   `create_convunext_denoiser(...)` forwards here with `use_bias=False`;
   `create_convunext_variant(...)` adds `kwargs.setdefault('block_normalization',
   'batchnorm')` ([§5](#5-block_normalization--the-default-rule)) and defaults
   `enable_deep_supervision=True`. The forward is a `locals()` capture, so a parameter
   can never be silently dropped.
2. **The Keras registrar.** Importing it registers `ConvUNextStem`, `ConvNextV1Block`,
   `GlobalResponseNormalization`, `MatchChannels`, `GaborFiltersInitializer` and
   `SpatialLinearAttention` so `keras.models.load_model` resolves them.
   `applications/bias_free_denoiser/denoiser_prior.py` and the two bfunet eval tools
   import it for that reason alone.

The two builders produce the same network:

```python
from dl_techniques.models.bias_free_denoisers.bfconvunext import create_convunext_denoiser

bf  = create_convunext(input_shape=(64, 64, 1), use_bias=False, depth=3, initial_filters=16)
bf2 = create_convunext_denoiser(input_shape=(64, 64, 1), depth=3, initial_filters=16)
print(bf.count_params(), bf2.count_params(), bf.count_params() == bf2.count_params())
# 503424 503424 True
```

`ConvUNextStem` lives HERE (`models/convunext/model.py`) but keeps the decorator
`@keras.saving.register_keras_serializable(package="dl_techniques.bias_free_denoisers")`.
That package string no longer matches its module path and **that mismatch is
deliberate and load-bearing**: it is the registry key
`dl_techniques.bias_free_denoisers>ConvUNextStem` that existing `.keras` artifacts
carry. Do not "tidy" it. An in-file `# DECISION` anchor says the same thing next to
the code.

## 9. Package surface

```python
from dl_techniques.models.convunext import (
    ConvUNextStem,                            # the merged stem layer
    SpatialLinearAttention,                   # bias-free bottleneck attention block
    CONVUNEXT_CONFIGS,                        # the one variant dict
    create_convunext,                         # the builder
    create_convunext_variant,                 # variant -> builder
    create_inference_model_from_training_model,  # re-export from utils.deep_supervision
)
```

`POSITIVELY_HOMOGENEOUS_ACTIVATIONS` and `_validate_bias_free_arguments` live in
`dl_techniques.models.convunext.model`; the first is public, the second is private.

## 10. Tests

```
tests/test_models/test_convunext/test_model.py                     # both arms, symmetric
tests/test_models/test_bias_free_denoisers/test_bfconvunext_*.py   # the bias-free arm
tests/test_layers/test_downsample_and_skip.py                      # the shared junction layer
```

Run scoped, never the whole suite:

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest tests/test_models/test_convunext/ -q
```
