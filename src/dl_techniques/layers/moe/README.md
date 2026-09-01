# `dl_techniques.layers.moe`

One layer, `MixtureOfExperts`, that replaces a dense FFN with `num_experts` FFN experts and a router
that sends each token to `top_k` of them. Experts are built by the `ffn/` factory, so any FFN type
key works as an expert. Capacity grows in the parameter dimension while per-token activation stays
sparse.

The package exports five names: `MixtureOfExperts`, `create_ffn_moe`, and the three config
dataclasses `MoEConfig`, `ExpertConfig`, `GatingConfig`. The strict-kwargs factory contract is owned
by `src/dl_techniques/layers/CLAUDE.md`.

## Pick a gating type

| `gating_type` | Class | What it does | Pick it when |
|---|---|---|---|
| `linear` | `LinearGating` | Learned `Dense` router, top-k hard routing, optional exploration noise. | The default. Switch-Transformer-style sparse routing. |
| `cosine` | `CosineGating` | Routes on cosine similarity to per-expert embeddings, with a learnable temperature. | Router logits need to be scale-invariant to the token norm. |
| `softmoe` | `SoftMoEGating` | Soft routing: every token reaches every expert through `num_slots` learned slots. No hard top-k. | You want no discrete routing and no load-balancing loss (Puigcerver et al. 2023). |

`softmoe` ignores `top_k` entirely (the `top_k <= num_experts` check is skipped for it) and receives
neither the auxiliary loss nor the z-loss.

Expert type is whatever `ffn_config['type']` names — `mlp`, `swiglu`, `geglu`, `glu`,
`differential`, `residual`, `swin_mlp`, and any other key in `FFN_REGISTRY`. See `ffn/README.md`.

## Construction

The convenience function covers the common case:

```python
from dl_techniques.layers.moe import create_ffn_moe

moe = create_ffn_moe(
    num_experts=8,
    ffn_config={'type': 'swiglu', 'output_dim': 768, 'ffn_expansion_factor': 4},
    top_k=2,
    gating_type='linear',
    aux_loss_weight=0.01,
    name='moe_ffn',
)
```

The config classes give full control:

```python
from dl_techniques.layers.moe import MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig

config = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        ffn_config={'type': 'mlp', 'hidden_dim': 2048, 'output_dim': 768, 'activation': 'gelu'},
        norm_type='rms_norm', pre_norm=True,           # optional per-expert norm
    ),
    gating_config=GatingConfig(gating_type='linear', top_k=2, aux_loss_weight=0.01),
)
moe = MixtureOfExperts(config)
```

Dropping it into a transformer block is an ordinary FFN swap:

```python
import keras

inputs = keras.Input(shape=(128, 768))
x = keras.layers.MultiHeadAttention(num_heads=12, key_dim=64)(inputs, inputs)
x = keras.layers.LayerNormalization()(x + inputs)
x = keras.layers.LayerNormalization()(moe(x) + x)
model = keras.Model(inputs, keras.layers.Dense(10000)(x))
```

## Configuration reference

`ExpertConfig`: `ffn_config` (the FFN factory's own kwargs, including `type`), plus optional
per-expert normalization — `norm_type` (e.g. `rms_norm`, `band_rms`; `None` disables), `norm_config`,
`pre_norm` (default `True`), `post_norm` (default `False`). Its `use_bias`, `kernel_initializer`,
`bias_initializer`, `kernel_regularizer` and `bias_regularizer` fields are **inert**: the FFN is
configured entirely through `ffn_config`.

`GatingConfig`: `gating_type` (`linear`/`cosine`/`softmoe`), `top_k` (1), `add_noise` (True),
`noise_std` (1.0), `temperature` (1.0), `use_bias` (False, linear), `embedding_dim` (256, cosine),
`learnable_temperature` (True, cosine), `num_slots` (4, softmoe), `aux_loss_weight` (0.01),
`z_loss_weight` (1e-3), and optional pre-gating `norm_type` / `norm_config`.

`MoEConfig`: `num_experts` (8), `expert_config`, `gating_config`, `jitter_noise` (0.01), plus the two
diagnostic-only flags below.

### Validation (all in `__post_init__`, so a bad config raises at construction)

| Rule | Raised by | Notes |
|---|---|---|
| `1 <= num_experts <= 2**31 - 1` | `MoEConfig` | Upper bound is the int32 tensor-dimension ceiling. |
| `top_k <= num_experts` | `MoEConfig` | Skipped for `softmoe`. The cross-field check can only live here: `GatingConfig` owns `top_k` but does not know `num_experts`. |
| `jitter_noise >= 0` | `MoEConfig` | Rejected, not silently disabled. |
| `top_k`, `num_slots`, `embedding_dim` positive ints `<= 2**31 - 1` | `GatingConfig` | |
| `temperature > 0`, `noise_std >= 0` | `GatingConfig` | |
| `gating_type in ('linear', 'cosine', 'softmoe')` | `GatingConfig` | |
| `ffn_config` carries a `'type'` field | `ExpertConfig` | An empty `ffn_config` becomes a default `mlp` rather than an error. |

Every int field rejects `bool` and `np.bool_` **before** the range test, because `isinstance(True,
int)` is `True`. YAML is the live path: `yaml.safe_load` turns an unquoted `true` into `True`, and
`top_k: true` would otherwise become `top_k=1`. Integral numpy scalars (`np.int64(4)`, ...) are
accepted and coerced to Python `int` — storing the numpy scalar verbatim would move the failure to
`model.save()`, since `json.dumps({'n': np.int64(4)})` raises.

## `create_ffn_moe` accepted keywords

An undeclared keyword raises `ValueError`; it is never filtered out.

| Group | Keys |
|---|---|
| Positional / common | `num_experts`, `ffn_config`, `top_k`, `gating_type`, `aux_loss_weight` |
| Expert norm | `norm_type`, `norm_config`, `pre_norm`, `post_norm`, `expert_norm_type`, `expert_norm_config` |
| Gating | `gate_use_bias`, `add_noise`, `noise_std`, `temperature`, `embedding_dim`, `learnable_temperature`, `num_slots`, `z_loss_weight`, `gating_norm_type`, `gating_norm_config` |
| MoE | `jitter_noise`, `drop_tokens`, `use_residual_connection` |
| Keras layer | `name`, `dtype`, `trainable` |

`use_bias=` is **not** accepted: the name exists in two sub-components, so say which one.

```python
create_ffn_moe(..., gate_use_bias=True)                    # the router's Dense bias
create_ffn_moe(..., ffn_config={..., 'use_bias': True})    # the expert FFN's bias
```

## Serialization

MoE layers are registered Keras serializables, so `model.save()` / `keras.models.load_model()` round
trip without custom objects. The config dataclasses also round-trip through plain dicts:

```python
import json
from dl_techniques.layers.moe import MoEConfig, MixtureOfExperts

with open('moe_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)

with open('moe_config.json') as f:
    moe = MixtureOfExperts(MoEConfig.from_dict(json.load(f)))
```

**A config payload from an older version must have `capacity_factor` and `routing_dtype` stripped**
before it loads: both fields are removed outright with no compatibility shim, and passing either to
a constructor or to `MoEConfig.from_dict` raises `TypeError`. The older `train_capacity_factor` /
`eval_capacity_factor` keys are unaffected and stay silently ignored by `from_dict`.

## Gotchas

- **`drop_tokens` and `use_residual_connection` are inert.** Neither kernel drops a token, so
  flipping either leaves the output bit-identical (`max|delta| == 0.0`). They are accepted,
  serialized and echoed by `get_expert_utilization()`, and gate nothing on the forward path. There
  is no capacity knob at all — the capacity-based dispatch they once pointed at is not planned.
- **Sparse in FLOPs is not always faster in wall-clock.** Each expert runs only on its routed tokens
  (gather -> FFN -> scatter-add), an exact `num_experts / top_k` reduction in expert-token pairs, but
  both kernels issue one FFN call per expert and the sparse one adds a gather/scatter to each. At
  small token counts the launch overhead can win. Memory, unlike time, is reduced at every size.
- **The dense kernel is kept on purpose.** `_process_hard_routing_dense` runs every expert on every
  token and masks. It is the numerical oracle the sparse kernel is gated against (`atol=1e-5`,
  `rtol=0`), and it is also what runs whenever `top_k >= num_experts`, where there is no sparsity
  to exploit.
- **`aux_loss_weight` is not `top_k`-invariant.** The Switch formula counts each token once per
  expert it is dispatched to, so with perfectly balanced routing the loss sits exactly on
  `aux_loss_weight * top_k` (0.010 / 0.020 / 0.040 at `top_k` 1 / 2 / 4), independent of
  `num_experts` and token count. The worst case is `aux_loss_weight * num_experts`, so the usable
  dynamic range is `num_experts / top_k`: raising `top_k` lifts the floor and shrinks the headroom.
  **Retune `aux_loss_weight` whenever you change `top_k`.**
- **Both auxiliary losses always return float32.** `compute_z_loss` upcasts because the squared
  log-sum-exp overflows float16 past ~256 logit magnitude. `compute_auxiliary_loss` upcasts for a
  different reason: Keras casts only *non-float* `add_loss` values to `floatx()`, so a float16
  auxiliary loss reaches `ops.sum(self.losses)` beside the float32 compiled loss and every
  mixed-precision `fit()` raised `TypeError: Cannot convert a list containing a tensor of dtype
  float16 to float32`. Both `mixed_float16` and `mixed_bfloat16` were affected. A `training=False`
  forward pass never calls `add_loss` and cannot see this class of defect.
- **`CosineGating.temperature` divides.** Standard softmax-temperature semantics: larger temperature
  means a flatter distribution. Earlier versions multiplied, so an old config routes more diffusely
  under current code.
- **SoftMoE auxiliary keys.** `gating_info` carries `dispatch_weights` (softmax over the sequence
  axis) and `combine_weights` (softmax over experts x slots per token). The old `phi_weights` key is
  gone.
- **`num_experts` costs construction time.** One `FFNExpert` sublayer is materialized per expert in
  a Python loop, ~2.4 ms each (250/500/1000/2000 experts took 0.67/1.19/2.19/4.92 s). The
  `2**31 - 1` bound is representability, not sanity.
- **Expert collapse** shows up as experts that never receive tokens. Raise `aux_loss_weight` (and
  `z_loss_weight`) and check `get_expert_utilization()`.

## References

- Switch Transformer: [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- GLaM: [arXiv:2112.06905](https://arxiv.org/abs/2112.06905)
- Soft MoE: [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)
