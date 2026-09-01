# `dl_techniques.layers.ffn`

Nineteen feed-forward layer classes reachable through one factory.
`create_ffn_layer(ffn_type, name=None, **kwargs)` looks the type up in `FFN_REGISTRY` (**21 keys** over those 19
classes — `glu`, `reglu` and `bilinear` are three configurations of `GLUFFN`), rejects any keyword
the target class does not declare, fills in the registry defaults and constructs. A keyword the type
does not accept is a `ValueError`, never a silently dropped argument.

The factory contract and the registry sizes are owned by `src/dl_techniques/layers/CLAUDE.md`.

## Catalogue

| Key | Class | What it is | Pick it when | Required params |
|---|---|---|---|---|
| `mlp` | `MLPBlock` | Dense -> activation -> Dense, with expansion. | The default transformer FFN. | `hidden_dim`, `output_dim` |
| `swiglu` | `SwiGLUFFN` | SwiGLU gating; hidden width derived from `ffn_expansion_factor` and rounded up to `ffn_multiple_of`. | Modern LLM blocks (LLaMA, Qwen). | `output_dim` |
| `geglu` | `GeGLUFFN` | GELU-gated linear unit. | GELU-flavoured gated FFN. | `hidden_dim`, `output_dim` |
| `glu` | `GLUFFN` | Gated linear unit, configurable gate activation (default `swish`). | Gated FFN with your own gate. | `hidden_dim`, `output_dim` |
| `reglu` | `GLUFFN` (alias) | `GLUFFN` with `activation='relu'` (Shazeer 2020). | ReLU-gated variant. | `hidden_dim`, `output_dim` |
| `bilinear` | `GLUFFN` (alias) | `GLUFFN` with `activation='linear'` — no gate nonlinearity (Shazeer 2020). | Bilinear variant. | `hidden_dim`, `output_dim` |
| `gelu_tanh` | `GELUMLPFFN` | Two-layer MLP with the tanh-approximate GELU (SD3-faithful). | Diffusion / transformer blocks that need the tanh GELU exactly. | `hidden_dim` |
| `differential` | `DifferentialFFN` | Two opponent pathways, differenced. | Feature processing with opponent signals. | `hidden_dim`, `output_dim` |
| `residual` | `ResidualBlock` | FFN with an internal skip connection. | Deep stacks needing gradient flow inside the block. | `hidden_dim`, `output_dim` |
| `swin_mlp` | `SwinMLP` | The Swin Transformer MLP. | Swin / windowed-attention vision models. | `hidden_dim` |
| `squared_relu` | `SquaredReLUFFN` | Primer FFN, fixed `relu(x)**2` (So et al. 2021, arXiv:2109.08668). | Squared-ReLU transformer FFN. | `hidden_dim`, `output_dim` |
| `lowrank` | `LowRankFFN` | Each projection factorized as `Dense(rank, no bias) -> Dense(out)`. | Parameter-efficient FFN when `rank << dims`. | `hidden_dim`, `output_dim` |
| `monarch` | `MonarchFFN` | Order-2 Monarch map: a product of two block-diagonal factors (Dao et al. 2022, arXiv:2204.00595). | Structured sub-quadratic replacement for dense projections. | `hidden_dim`, `output_dim` |
| `mixer` | `MixerBlock` | Canonical MLP-Mixer token + channel mixing (Tolstikhin et al. 2021, arXiv:2105.01601). | Attention-free token mixing over a patch sequence. | `tokens_mlp_dim`, `channels_mlp_dim` |
| `orthoglu` | `OrthoGLUFFN` | GLU with orthogonality regularization on the projections. | Deep nets needing stable training dynamics. | `hidden_dim`, `output_dim` |
| `gated_mlp` | `GatedMLP` | Channel-wise GLU from three 1x1 convolutions on rank-4 NHWC/NCHW input. | Position-wise channel gating in vision models. | `filters` |
| `power_mlp` | `PowerMLPLayer` | Dual-branch MLP (power branch + basis branch). | Approximating sharp nonlinear functions. | `units` |
| `kan` | `KANLinear` | Kolmogorov-Arnold layer: B-spline learnable univariate activations per connection. | Expressive per-connection nonlinearities; N-D inputs. | `features` |
| `tversky` | `TverskyProjectionLayer` | Asymmetric Tversky-similarity projection against learned prototypes. | Similarity-based alternative to `Dense`. | `units`, `num_features` |
| `counting` | `CountingFFN` | Learns to count features along the sequence (`counting_scope` is `global`, `local` or `causal`). | Tasks with an explicit counting flavour. | `output_dim`, `count_dim` |
| `logic` | `LogicFFN` | Learnable soft logic operations. | Symbolic-flavoured reasoning. | `output_dim`, `logic_dim` |

Everything else is optional with a registry default. Read the defaults from the registry rather
than from prose:

```python
from dl_techniques.layers.ffn import get_ffn_info

info = get_ffn_info()['mlp']
print(info['required_params'])            # ['hidden_dim', 'output_dim']
print(sorted(info['optional_params']))    # activation, dropout_rate, use_bias, ...
```

## Construction

```python
from dl_techniques.layers.ffn import create_ffn_layer, create_ffn_from_config

mlp = create_ffn_layer('mlp', hidden_dim=512, output_dim=256, activation='relu', dropout_rate=0.1)
swiglu = create_ffn_layer('swiglu', output_dim=768, ffn_expansion_factor=4, dropout_rate=0.1)

ffn = create_ffn_from_config({
    'type': 'differential',
    'hidden_dim': 1024,
    'output_dim': 512,
    'branch_activation': 'relu',
    'dropout_rate': 0.1,
    'name': 'diff_ffn_block',
})
```

Pre-flight validation, for callers that go on to build a class directly:

```python
from dl_techniques.layers.ffn import validate_ffn_config

validate_ffn_config('swiglu', output_dim=768, ffn_expansion_factor=4)  # raises on a bad config
```

Direct import is equally valid when the choice is not config-driven:

```python
from dl_techniques.layers.ffn import MLPBlock, SwiGLUFFN

mlp = MLPBlock(hidden_dim=512, output_dim=256, activation='relu')
swiglu = SwiGLUFFN(output_dim=768, ffn_expansion_factor=4)
```

Inside a custom block, build the FFN from the caller's config so the block stays type-agnostic:

```python
import keras
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# The registration string is the defining module's dotted path. Never a bare
# `@keras.saving.register_keras_serializable()`: its `Custom>ClassName` key carries no module
# path, so two same-named classes claim one slot and the last import wins.
@register_dl_technique("my_project.ffn_block")
class FFNBlock(keras.layers.Layer):
    def __init__(self, ffn_config, **kwargs):
        # ffn_config is the factory's own kwargs, e.g.
        # {'ffn_type': 'mlp', 'hidden_dim': 8, 'output_dim': 6}
        super().__init__(**kwargs)
        self.ffn_config = dict(ffn_config)
        self.ffn = create_ffn_layer(**self.ffn_config)

    def call(self, inputs, training=None):
        return self.ffn(inputs, training=training)

    def get_config(self):
        return {**super().get_config(), 'ffn_config': self.ffn_config}
```

Storing the config dict (not the built layer) is what makes `get_config()` round-trip: the layer is
rebuilt from the same keys on load.

## Gotchas

- **Rank matters.** `mixer` accepts rank-3 `(B, S, C)` only and returns the same shape; `tversky`
  accepts rank-2 `(batch, input_dim)` only and returns `(batch, units)`; `gated_mlp` needs rank-4.
  None of the three is a drop-in for a rank-3 transformer FFN.
- **`gated_mlp` is not gMLP.** Every kernel is 1x1, so nothing mixes across spatial positions. It is
  channel gating, not a stand-in for attention.
- **`monarch`: `nblocks` (default 4) must divide `input_dim`, `hidden_dim` and `output_dim`.**
  Validated in `__init__`/`build`, so pick dimensions that are multiples of it.
- **`swiglu` does not take a hidden width directly by default.** It derives one from
  `ffn_expansion_factor` and rounds it to a multiple of `ffn_multiple_of` (256):
  `output_dim=768, ffn_expansion_factor=4` builds `hidden_dim=2048`, not 3072. Pass `hidden_dim`
  explicitly if you need an exact width.
- **`lowrank`: `rank=None` resolves to `max(1, hidden_dim // 4)`.** `rank <= 0` raises.
- **`squared_relu` has a fixed nonlinearity.** There is no `activation` parameter.
- **`reglu` and `bilinear` return a `GLUFFN`,** so an isinstance check on the class cannot tell the
  three apart; the serialized config carries the `activation`.
- **An undeclared keyword raises.** `create_ffn_layer('mlp', hidden_dim=8, output_dim=8, foo=1)`
  raises a `ValueError` naming the key, the type and the accepted set. It is not a warning.
