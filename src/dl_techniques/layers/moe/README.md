# Mixture of Experts (MoE) Module

A production-ready Mixture of Experts implementation for the dl_techniques framework, providing sparse neural network architectures through conditional computation with FFN expert specialization.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Integration](#integration)
8. [API Reference](#api-reference)
9. [Training Best Practices](#training-best-practices)
10. [Performance Considerations](#performance-considerations)
11. [Model Serialization](#model-serialization)
12. [Troubleshooting](#troubleshooting)
13. [Examples](#examples)
14. [References](#references)

## Overview

The MoE module implements expert-conditioned neural networks where each input is routed to a subset of expert networks, enabling:

- **Model Specialization**: Experts learn to handle specific input patterns.
- **Scalable Architecture**: Add capacity in the *parameter* dimension while keeping per-token activation patterns sparse.
- **Load Balancing**: Auxiliary and z-losses prevent expert collapse.

### Key Components

- **Expert Networks**: FFN-based specialists using the `dl_techniques` FFN factory; optional pre/post normalization via the norms factory.
- **Gating Networks**: Routing mechanisms (linear, cosine similarity, SoftMoE); optional pre-gating normalization via the norms factory.
- **Load Balancing**: Auxiliary and z-losses for uniform expert utilization.

### Known Limitations

- **Hard routing is sparse in FLOPs; wall-clock does not always follow.** Each expert runs only on the tokens routed to it (gather → FFN → scatter-add), an exact `num_experts / top_k` reduction in expert-token pairs (measured). But both kernels issue one FFN call per expert, and the sparse one adds a gather/scatter to each, so at small token counts the launch overhead can outweigh the saving. Measured on an RTX 4070 under `tf.function`, `d_model=512`, `hidden=1024`: 2048 tokens → `0.913x` at `num_experts=64` and `0.871x` at `128` (slightly **slower**); 8192 tokens → `2.108x` at `num_experts=64`; and at the `qwen3_next` preset shape (`num_experts=512`, `top_k=10`, 2048 tokens) sparse completes in 179 ms where the dense kernel exhausts the 12 GB device. Memory, unlike time, is reduced at every size.
- **The dense kernel is retained on purpose.** `MixtureOfExperts._process_hard_routing_dense` runs every expert on every token and masks the result. It is the numerical oracle the sparse kernel is gated against (`atol=1e-5`, `rtol=0`, `tests/test_layers/test_moe/test_the_sparse_kernel_matches_the_dense_oracle.py`), and it is also the kernel that runs whenever `top_k >= num_experts`, where there is no sparsity to exploit.
- **`drop_tokens` and `use_residual_connection` are inert.** Neither kernel drops a token — the sparse kernel computes exactly the routed `(token, expert)` pairs and the dense kernel computes all of them, so no token is ever left without a contribution. These `MoEConfig` flags are reported by `get_expert_utilization()` and gate no forward-path behaviour (measured: flipping both leaves the output bit-identical, `max|delta| == 0.0`). The capacity-based dispatch they were once described as "reserved for" is **not** planned; the two config fields that belonged to that unbuilt scheme, `GatingConfig.capacity_factor` and `MoEConfig.routing_dtype`, have been **removed**.
- **`CosineGating.temperature` semantics changed.** As of the May-2026 review, cosine gating now *divides* logits by `temperature` (standard softmax-temperature semantics: larger `temperature` → flatter distribution). Earlier versions multiplied; checkpoints/configs using the old behavior will route more diffusely under the new code.
- **SoftMoE auxiliary info keys changed.** `phi_weights` is replaced by `dispatch_weights` (softmax over the sequence axis) and `combine_weights` (softmax over experts × slots per token), matching Puigcerver et al. (2023). Callers reading `phi_weights` from `gating_info` must migrate.

## Architecture

### MoE Layer Structure

```
Input → Gating Network → Routing Weights → Per-expert gather → Expert FFN → Scatter-add → Output
         ↓                ↓                  ↓
         Router           Top-K indices      Only the tokens routed to expert e
         Logits                              are gathered, run, and scattered back
```

(See "Known Limitations" above for the measured sparse-vs-dense wall-clock picture, and for why the dense kernel is retained.)

### Expert Types

The module exclusively uses FFN experts, leveraging the `dl_techniques` FFN factory:

- **MLP**: Standard multi-layer perceptron (`type: "mlp"`)
- **SwiGLU**: Gated linear units with SiLU activation (`type: "swiglu"`)
- **GeGLU**: GELU-based gated linear units (`type: "geglu"`)
- **GLU**: Standard gated linear units (`type: "glu"`)
- **Differential**: Dual-pathway processing (`type: "differential"`)
- **Residual**: Skip connections for gradient flow (`type: "residual"`)
- **Swin MLP**: Vision-optimized MLP variant (`type: "swin_mlp"`)

### Gating Mechanisms

- **Linear Gating**: Standard learnable routing with optional noise.
- **Cosine Gating**: Cosine similarity-based routing in a hypersphere space.
- **SoftMoE**: Soft routing with weighted token slots per expert — every token reaches every expert through `num_slots` learned slots.

## Installation

The MoE module is part of the `dl_techniques` framework. Ensure the framework is installed, then import the necessary components:

```python
import keras
from dl_techniques.layers.moe import MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig
```

### Dependencies

- Keras 3
- A compatible backend (e.g., TensorFlow, JAX, PyTorch)
- `dl_techniques` FFN and utility modules
- NumPy, typing

## Configuration

### Basic Configuration

```python
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

# Expert configuration using FFN factory
expert_config = ExpertConfig(
    ffn_config={
        "type": "swiglu",           # FFN type
        "output_dim": 768,             # Model dimension
        "ffn_expansion_factor": 4   # Expansion ratio
    }
)

# Gating configuration
gating_config = GatingConfig(
    gating_type='linear',      # Routing mechanism
    top_k=2,                   # Experts per token
    aux_loss_weight=0.01       # Load balancing weight
)

# Complete MoE configuration
moe_config = MoEConfig(
    num_experts=8,
    expert_config=expert_config,
    gating_config=gating_config
)
```

### Configuration Classes

#### ExpertConfig

```python
@dataclass
class ExpertConfig:
    ffn_config: Dict[str, Any]                                    # FFN configuration dict

    # Reserved for additional (non-FFN) wrapper layers — currently INERT
    # (the FFN itself is configured entirely through ``ffn_config``).
    use_bias: bool = True
    kernel_initializer: Union[str, Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, Initializer] = 'zeros'
    kernel_regularizer: Optional[Regularizer] = None
    bias_regularizer: Optional[Regularizer] = None

    # Optional per-expert normalization via the norms factory.
    norm_type: Optional[str] = None        # e.g. 'rms_norm', 'band_rms'; None disables
    norm_config: Dict[str, Any] = field(default_factory=dict)
    pre_norm: bool = True                  # apply norm before the FFN
    post_norm: bool = False                # apply norm after the FFN
```

#### GatingConfig

```python
@dataclass
class GatingConfig:
    gating_type: Literal['linear', 'cosine', 'softmoe'] = 'linear'
    top_k: int = 1                    # Experts selected per token
    add_noise: bool = True            # Exploration noise (linear gating)
    noise_std: float = 1.0            # Noise standard deviation
    temperature: float = 1.0          # Softmax temperature

    # Linear gating
    use_bias: bool = False

    # Cosine gating
    embedding_dim: int = 256
    learnable_temperature: bool = True

    # SoftMoE
    num_slots: int = 4

    # Load balancing
    aux_loss_weight: float = 0.01     # Auxiliary loss weight (not applied to softmoe)
    z_loss_weight: float = 1e-3       # Router z-loss weight (not applied to softmoe)

    # Optional pre-gating normalization via the norms factory.
    norm_type: Optional[str] = None
    norm_config: Dict[str, Any] = field(default_factory=dict)
```

#### MoEConfig

```python
@dataclass
class MoEConfig:
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)

    # System parameters
    jitter_noise: float = 0.01
    drop_tokens: bool = True               # DIAGNOSTIC ONLY — no kernel drops tokens
    use_residual_connection: bool = True   # DIAGNOSTIC ONLY — pairs with drop_tokens
```

> **Diagnostic-only fields.** `drop_tokens` and `use_residual_connection` are
> accepted, serialized and echoed by `get_expert_utilization()`, but have **no
> runtime effect**: neither the dense nor the sparse kernel drops a token, so
> flipping either leaves the layer's output bit-identical.
>
> **Removed fields.** `GatingConfig.capacity_factor` and `MoEConfig.routing_dtype`
> described a capacity-based dispatch with token dropping that was never built
> and is not planned — the sparse gather/scatter kernel that ships is a compute
> optimization, not a capacity scheme — and `routing_dtype` additionally
> accepted any string, valid dtype or not, without ever being read. Both are
> **removed outright, with no backward-compatibility shim**: passing either to a
> config constructor, or to `MoEConfig.from_dict`, raises `TypeError`. A config
> payload serialized by an earlier version must have these two keys stripped
> before it will load.
> The older `train_capacity_factor` / `eval_capacity_factor` keys are unaffected
> and remain silently ignored by `MoEConfig.from_dict`.

#### Configuration validation

All three dataclasses validate in `__post_init__`, so an invalid configuration
raises at construction rather than deep inside layer assembly:

| Rule | Raised by | Notes |
|---|---|---|
| `1 <= num_experts <= 2**31 - 1` | `MoEConfig` | Upper bound is the int32 tensor-dimension ceiling |
| `top_k <= num_experts` | `MoEConfig` | **Skipped for `gating_type='softmoe'`**, which ignores `top_k` entirely — an unrelated value there is inert |
| `jitter_noise >= 0` | `MoEConfig` | Rejected, not silently disabled |
| `top_k`, `num_slots`, `embedding_dim` are positive ints `<= 2**31 - 1` | `GatingConfig` | |
| `temperature > 0`, `noise_std >= 0` | `GatingConfig` | |
| `gating_type in ('linear', 'cosine', 'softmoe')` | `GatingConfig` | |
| `ffn_config` contains a `'type'` field | `ExpertConfig` | An empty `ffn_config` is replaced by a default `mlp`, not rejected |

`top_k <= num_experts` is a cross-field invariant and `MoEConfig` is the only
place it can be checked: `GatingConfig` owns `top_k` but does not know
`num_experts`.

Every int field rejects `bool` **and `np.bool_`** **before** the range test,
because `isinstance(True, int)` is `True` in Python. YAML is the live path for
this — `yaml.safe_load` turns an unquoted `true` into `True`, and without the
check `top_k: true` would silently become `top_k=1`.

Integral **numpy** scalars are accepted (`np.int64(4)`, `np.int32`, `np.uint8`)
— arriving from `some_array.shape[-1]` is legitimate — and are coerced to a
Python `int`, which is what the field ends up holding. The coercion is not
cosmetic: `json.dumps({"n": np.int64(4)})` raises `TypeError: Object of type
int64 is not JSON serializable`, so storing the numpy scalar verbatim would move
the failure to `model.save()`, after training.

The upper bound is `2**31 - 1`, the largest int32 tensor dimension; all four
fields become tensor dimensions or a one-hot depth. It is a representability
bound, **not** a sanity bound: `MixtureOfExperts.__init__` materializes one
`FFNExpert` sublayer per expert in a Python loop (measured at ~2.4 ms/expert:
250/500/1000/2000 experts took 0.67/1.19/2.19/4.92 s), so a large-but-legal
`num_experts` still costs linear construction time. No arbitrary tighter cap is
imposed — see the plan's decisions.md D-008.

## Basic Usage

### Simple MoE Layer

```python
import keras
from dl_techniques.layers.moe import MixtureOfExperts, create_ffn_moe

# Method 1: Using convenience function (Recommended)
moe_layer = create_ffn_moe(
    num_experts=8,
    ffn_config={
        "type": "swiglu",
        "output_dim": 768,
        "ffn_expansion_factor": 4
    },
    top_k=2,
    gating_type='linear'
)

# Method 2: Using configuration classes
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

config = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "mlp",
            "hidden_dim": 2048,
            "output_dim": 768,
            "activation": "gelu"
        }
    ),
    gating_config=GatingConfig(gating_type='linear', top_k=2)
)
moe_layer = MixtureOfExperts(config)
```

### Integration in Models

```python
import keras
from dl_techniques.layers.moe import create_ffn_moe

# Assuming vocab_size is defined
vocab_size = 10000

def create_transformer_with_moe():
    inputs = keras.Input(shape=(128, 768))

    # Standard transformer layers
    x = keras.layers.MultiHeadAttention(num_heads=12, key_dim=64)(inputs, inputs)
    x = keras.layers.LayerNormalization()(x + inputs)

    # Replace FFN with MoE
    residual = x
    moe_output = create_ffn_moe(
        num_experts=8,
        ffn_config={
            "type": "swiglu",
            "output_dim": 768,
            "ffn_expansion_factor": 4
        },
        top_k=2
    )(x)
    x = keras.layers.LayerNormalization()(moe_output + residual)

    outputs = keras.layers.Dense(vocab_size)(x)
    return keras.Model(inputs, outputs)

model = create_transformer_with_moe()
model.summary()
```

## Advanced Usage

### Custom FFN Expert Types

```python
# GeGLU experts with custom parameters
config = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "geglu",
            "hidden_dim": 3072,
            "output_dim": 768,
            "dropout_rate": 0.1
        }
    ),
    gating_config=GatingConfig(gating_type='cosine', top_k=1)
)

# Differential FFN experts
config = MoEConfig(
    num_experts=12,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "differential",
            "hidden_dim": 1024,
            "output_dim": 768,
            "branch_activation": "relu",
            "combination_activation": "gelu"
        }
    )
)
```

### Advanced Gating Configurations

```python
# Cosine similarity gating
cosine_config = GatingConfig(
    gating_type='cosine',
    top_k=1,
    embedding_dim=256,
    temperature=0.1,
    learnable_temperature=True,
    aux_loss_weight=0.02
)

# SoftMoE gating (no hard routing)
softmoe_config = GatingConfig(
    gating_type='softmoe',
    num_slots=4,
    aux_loss_weight=0.01,
    z_loss_weight=1e-3
)
```

### Training Configuration

This package ships no optimizer helper. Compile an MoE model with a stock Keras
optimizer; the auxiliary losses are added through `add_loss` and are already part of
`model.losses`, so `model.fit()` includes them without any extra wiring.

```python
import keras

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=50_000,
            warmup_target=1e-4,
            warmup_steps=2_000,
        ),
        weight_decay=0.01,
        global_clipnorm=1.0,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

The strength of the load-balancing and router-logit regularizers is set on
`GatingConfig` (`aux_loss_weight`, `z_loss_weight`), not on the optimizer.

### Load Balancing and Regularization

```python
config = MoEConfig(
    num_experts=16,
    expert_config=expert_config, # Assume expert_config is defined
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=2,
        add_noise=True,                # Exploration noise
        noise_std=1.0,
        aux_loss_weight=0.01,          # Load balancing
        z_loss_weight=1e-3             # Entropy regularization
    ),
    # Token management (DIAGNOSTIC ONLY — no effect in either kernel)
    drop_tokens=True,
    use_residual_connection=True,
    jitter_noise=0.01                  # Input noise for regularization
)
```

## Integration

### With `dl_techniques` Optimization

```python
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# Create learning rate schedule
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "learning_rate": 1e-4,
    "decay_steps": 10000
}

# Create optimizer
opt_config = {
    "type": "adamw",
    "gradient_clipping_by_norm": 1.0
}

lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Compile model with MoE layers
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

### With Model Analyzer

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig

# Analyze MoE model performance
# config = AnalysisConfig(
#     analyze_weights=True,
#     analyze_training_dynamics=True,
#     save_plots=True
# )

# analyzer = ModelAnalyzer(
#     models={'moe_model': moe_model},
#     config=config,
#     output_dir='moe_analysis'
# )
# results = analyzer.analyze(data=test_data)

# Get expert utilization statistics from a specific layer
# moe_layer = model.get_layer('mixture_of_experts_layer_name')
# utilization = moe_layer.get_expert_utilization()
```

## API Reference

### `MixtureOfExperts` Layer

```python
class MixtureOfExperts(keras.layers.Layer):
    def __init__(self, config: MoEConfig, **kwargs)
    def call(self, inputs, training=None) -> keras.KerasTensor
    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]
    def get_expert_utilization(self) -> Dict[str, Any]
    def get_config(self) -> Dict[str, Any]
    def from_config(cls, config: Dict[str, Any]) -> 'MixtureOfExperts'
```

### Factory Functions

```python
def create_ffn_moe(
    num_experts: int,
    ffn_config: Dict[str, Any],
    top_k: int = 1,
    gating_type: str = 'linear',
    aux_loss_weight: float = 0.01,
    **kwargs
) -> MixtureOfExperts
```

`create_ffn_moe` **raises `ValueError` on an undeclared keyword** rather than
filtering it out, per the factory contract in `src/dl_techniques/layers/CLAUDE.md`.
The accepted `kwargs` are:

| Group | Keys |
|---|---|
| Expert norm | `norm_type`, `norm_config`, `pre_norm`, `post_norm`, `expert_norm_type`, `expert_norm_config` |
| Gating | `gate_use_bias`, `add_noise`, `noise_std`, `temperature`, `embedding_dim`, `learnable_temperature`, `num_slots`, `z_loss_weight`, `gating_norm_type`, `gating_norm_config` |
| MoE | `jitter_noise`, `drop_tokens`, `use_residual_connection` |
| Keras layer | `name`, `dtype`, `trainable` |

`use_bias=` is **not** accepted. It names a bias that exists in two different
sub-components, so the caller must say which:

```python
create_ffn_moe(..., gate_use_bias=True)                       # the router's Dense bias
create_ffn_moe(..., ffn_config={..., 'use_bias': True})       # the expert FFN's bias
```

### Expert Networks

```python
class FFNExpert(BaseExpert):
    def __init__(self, ffn_config: Dict[str, Any], **kwargs)
    def call(self, inputs, training=None) -> keras.KerasTensor
    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]
```

### Gating Networks

```python
def create_gating(
    gating_type: str,
    num_experts: int,
    **kwargs
) -> BaseGating

class LinearGating(BaseGating):
    def __init__(self, num_experts, top_k=1, use_bias=False, add_noise=True, ...)

class CosineGating(BaseGating):
    def __init__(self, num_experts, embedding_dim=256, top_k=1, ...)

class SoftMoEGating(BaseGating):
    def __init__(self, num_experts, num_slots=4, ...)
```

### Auxiliary Loss Functions

```python
def compute_auxiliary_loss(
    expert_weights: keras.KerasTensor,
    gate_probs: keras.KerasTensor,
    num_experts: int,
    aux_loss_weight: float = 0.01
) -> keras.KerasTensor

def compute_z_loss(
    gate_logits: keras.KerasTensor,
    z_loss_weight: float = 1e-3
) -> keras.KerasTensor
```

> **`aux_loss_weight` is not `top_k`-invariant.** The Switch-Transformer
> formula `N * sum_i(f_i * P_i)` defines `f_i` as the fraction of tokens
> *dispatched to* expert `i`; at `top_k = k` every token is dispatched `k`
> times, so `sum_i f_i = k` and the whole loss scales with `top_k`. MEASURED:
> with perfectly balanced routing the loss sits **exactly** on
> `aux_loss_weight * top_k` — 0.010 / 0.020 / 0.040 at `top_k` 1 / 2 / 4 —
> independent of `num_experts` and of the token count. The worst case (every
> token to the same `k` experts) is `aux_loss_weight * num_experts`, so the
> regularizer's usable dynamic range is `num_experts / top_k`: raising `top_k`
> lifts the floor *and* shrinks the headroom. This is deliberate and is **not**
> normalized away — doing so would rescale the load-balancing term for the
> shipped Qwen3 (`top_k=8`) and Qwen3-Next (`top_k=10`) presets. Retune
> `aux_loss_weight` whenever you change `top_k`.

> **Mixed precision.** `compute_z_loss` upcasts `gate_logits` to float32 before
> `logsumexp`/`square` and **always returns float32**, because under
> `mixed_float16` the squared log-sum-exp overflows past ~256 logit magnitude
> (measured `inf` where the float32 reference is `70.707`). `CosineGating`'s
> temperature floor is likewise derived from the compute dtype (`1e-3` for
> float16/bfloat16) and the learnable `temperature` variable carries a
> min-value constraint, so an optimizer cannot drive it into the overflow zone.
>
> **`compute_auxiliary_loss` also always returns float32, for a different
> reason.** Its value is bounded by `num_experts` and never overflows, so this is
> not an overflow guard — it is a dtype-uniformity requirement of Keras' loss
> aggregation. `_aggregate_additional_loss` (`keras/src/trainers/trainer.py`)
> casts only *non-float* `add_loss` values to `floatx()`, so a float16 auxiliary
> loss survives into the list reduced by `ops.sum(self.losses)` next to the
> float32 compiled loss, and `model.fit()` raises
> `TypeError: Cannot convert a list containing a tensor of dtype
> <dtype: 'float16'> to <dtype: 'float32'>`. Measured: this crashed *every*
> mixed-precision MoE `fit()` at the shipped default `aux_loss_weight=0.01`.
> **Both half-precision policies are affected** — the same failure reproduces
> verbatim under `mixed_bfloat16`. Guarded by
> `tests/test_layers/test_moe/test_the_auxiliary_loss_survives_a_mixed_precision_fit.py`,
> which runs a real optimizer step: a `training=False` forward pass never calls
> `add_loss` and cannot see this class of defect.

## Training Best Practices

### Optimizer Configuration

AdamW with warmup and cosine decay, and a global gradient-norm clip, is a reasonable
starting point. Per-parameter-group learning rates (a lower rate for the experts than
for the router) are **not** supported by this package and are not supported by stock
Keras optimizers either -- `keras.optimizers.AdamW` applies one learning rate to every
variable it is given. Achieving group-specific rates requires either two optimizers
over two disjoint variable lists in a custom training loop, or a `LossScaleOptimizer`-
style wrapper you write yourself.

```python
import keras

optimizer = keras.optimizers.AdamW(
    learning_rate=keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=50_000,
        warmup_target=1e-4,
        warmup_steps=2_000,
    ),
    weight_decay=0.01,
    global_clipnorm=1.0,
)
```

### Loss Monitoring

The auxiliary losses (load balancing, z-loss) are automatically added to the model's loss collection. When using `model.fit()`, they are included in the total loss. In a custom training loop, you can access them via `model.losses`.

```python
# Custom training loop (TensorFlow backend example)
import tensorflow as tf

@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)

        # Main task loss
        task_loss = loss_fn(batch_y, predictions)

        # MoE auxiliary losses are automatically added
        total_loss = task_loss + sum(model.losses)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return {
        'task_loss': task_loss,
        'aux_losses': sum(model.losses),
        'total_loss': total_loss
    }
```

## Performance Considerations

### Memory Usage

- **Expert cost scales with `top_k`**, not `num_experts` — each expert sees only its routed tokens. Peak activation memory is therefore `num_tokens × top_k` expert rows rather than `num_tokens × num_experts`.
- **Top-K Selection**: higher `top_k` raises both compute and activation memory proportionally, and affects routing quality.
- **Capacity / token dropping**: not implemented and not planned. `drop_tokens` and `use_residual_connection` are diagnostic-only flags (see *Diagnostic-only fields* above) and neither kernel drops a token; the `capacity_factor` / `routing_dtype` fields that named the unbuilt scheme have been removed.

### Computational Efficiency

```python
# Efficient configuration for large models
efficient_config = MoEConfig(
    num_experts=32,               # Many experts
    expert_config=ExpertConfig(
        ffn_config={
            "type": "swiglu",     # Efficient gated FFN
            "output_dim": 512,       # Moderate size
            "ffn_expansion_factor": 2  # Lower expansion
        }
    ),
    gating_config=GatingConfig(
        gating_type='linear',     # Fastest gating
        top_k=1                   # Minimal routing
    )
)
```

### Load Balancing Tuning

```python
# Monitor expert utilization during or after training
def monitor_expert_usage(moe_layer):
    stats = moe_layer.get_expert_utilization()
    print(f"Experts: {stats['num_experts']}")
    print(f"Routing: {stats['routing_type']}")
    print(f"Top-K: {stats['top_k']}")
    print(f"Aux loss weight: {stats['aux_loss_weight']}")

# Adjust auxiliary loss weights based on utilization
# High aux_loss_weight (e.g., 0.1) for better load balancing
# Low aux_loss_weight (e.g., 0.001) for minimal interference
#
# NOTE: the balanced-routing floor of the auxiliary loss is
# `aux_loss_weight * top_k`, so these numbers are top_k-relative. Changing
# `top_k` rescales the regularizer even if `aux_loss_weight` is untouched --
# see "Auxiliary Loss Functions" in the API Reference.
```

## Model Serialization

### Save and Load

All MoE layers are registered as Keras serializable objects, enabling seamless saving and loading.

```python
# Save model with MoE layers
model.save('moe_model.keras')

# Load model (automatic registration)
loaded_model = keras.models.load_model('moe_model.keras')

# Verify expert configuration is preserved
moe_layer = loaded_model.get_layer('mixture_of_experts') # Adjust layer name
utilization = moe_layer.get_expert_utilization()
print(f"Loaded model has {utilization['num_experts']} experts")
```

### Configuration Export

```python
# Export configuration for reproducibility
config_dict = moe_config.to_dict()

# Save configuration
import json
with open('moe_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# Load and recreate
with open('moe_config.json', 'r') as f:
    loaded_config_dict = json.load(f)

loaded_config = MoEConfig.from_dict(loaded_config_dict)
recreated_moe_layer = MixtureOfExperts(loaded_config)
```

## Troubleshooting

### Common Issues

#### Expert Collapse
**Symptoms**: Some experts never receive tokens, leading to poor model performance.
```python
# Solution: Increase auxiliary loss weight
gating_config = GatingConfig(
    aux_loss_weight=0.1,      # Increase from 0.01
    z_loss_weight=1e-2        # Increase entropy regularization
)
```

#### Memory Issues
**Symptoms**: Out-of-memory (OOM) errors during training.
```python
# Solution: reduce per-expert cost or the number of active experts.
# NOTE: no kernel drops tokens, so capacity-based token dropping is not
# available — reduce num_experts, top_k, or the FFN size instead.
config = MoEConfig(
    num_experts=8,                                     # fewer experts
    expert_config=ExpertConfig(ffn_config={
        "type": "mlp", "hidden_dim": 1024, "output_dim": 512,  # smaller FFN
    }),
    gating_config=GatingConfig(gating_type='linear', top_k=1),  # minimal routing
)
```

#### Training Instability
**Symptoms**: Loss spikes, gradient explosions.
```python
# Solution: lower the learning rate, clip gradients globally, and reduce input noise.
import keras

optimizer = keras.optimizers.AdamW(learning_rate=5e-5, global_clipnorm=0.5)
config = MoEConfig(
    num_experts=8,
    expert_config=expert_config,
    gating_config=GatingConfig(gating_type='linear', top_k=2),
    jitter_noise=0.001,                    # less input noise
)
```

#### Poor Expert Utilization
**Symptoms**: Experts have very uneven usage despite load balancing loss.
```python
# Solution: Tune the noise and load-balancing parameters
gating_config = GatingConfig(
    add_noise=True,               # Enable exploration
    noise_std=2.0,                # Increase exploration
    aux_loss_weight=0.05          # Stronger load balancing
)
```

### Debugging

#### Expert Utilization Analysis

```python
def analyze_expert_usage(model):
    """Analyze expert usage across all MoE layers in a model."""
    moe_layers = [layer for layer in model.layers if isinstance(layer, MixtureOfExperts)]

    for moe_layer in moe_layers:
        stats = moe_layer.get_expert_utilization()
        print(f"\nMoE Layer: {moe_layer.name}")
        print(f"Configuration: {stats}")

        # Check auxiliary losses (only available during training call)
        if hasattr(moe_layer, '_auxiliary_losses'):
            print(f"Auxiliary losses tracked: {len(moe_layer._auxiliary_losses)}")
```

#### Validation

```python
# TensorFlow backend example -- `GradientTape` is a TF API, not a Keras 3 one,
# so the import is required for this snippet to run.
import tensorflow as tf


def validate_moe_model(model, sample_input):
    """Validate MoE model functionality."""
    # Test forward pass
    output = model(sample_input, training=False)
    print(f"Inference pass successful: {output.shape}")

    # Test training pass
    with tf.GradientTape() as tape:
        training_output = model(sample_input, training=True)
        # Combine task loss and model's internal losses
        loss = keras.ops.mean(keras.ops.square(training_output)) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    print(f"Gradients computed for {len(gradients)} variables")

    # Test serialization
    try:
        model.save('test_moe.keras')
        loaded = keras.models.load_model('test_moe.keras')
        print("Serialization test passed")
    except Exception as e:
        print(f"Serialization failed: {e}")
```

### Error Messages

#### "FFN configuration validation failed"
- Check that `ffn_config` contains a `'type'` field (e.g., `"mlp"`, `"swiglu"`).
- Verify FFN parameters match the selected type's requirements.
- Use `validate_ffn_config()` from `dl_techniques.layers.ffn` for debugging.

#### "Unsupported gating type"
- Supported types: `'linear'`, `'cosine'`, `'softmoe'`.
- Check for spelling errors and case sensitivity.

#### "top_k must be between 1 and num_experts (N)"
- Raised by `MoEConfig.__post_init__` at construction. Lower `top_k` or raise
  `num_experts`; `top_k == num_experts` is legal and dispatches to the dense
  kernel.
- Not raised for `gating_type='softmoe'`, which ignores `top_k`.

#### "... must be an int, got bool"
- An int config field received `True`/`False`/`np.bool_`. Usually a YAML
  `true`/`false` reaching an int field — quote it or write a number. See
  *Configuration validation* above.

#### "... must be <= 2147483647 (the int32 tensor-dimension ceiling)"
- An int config field exceeds what an int32 tensor dimension can hold. This is
  almost always a unit mix-up rather than an intended configuration.

#### Out-of-memory with many experts
- Expert activation memory scales with `top_k` and the per-expert FFN size; the
  per-expert *weights* still scale with `num_experts`.
- Reduce `num_experts`, shrink the FFN (`hidden_dim` / `output_dim`), or lower
  `top_k`. There is no capacity knob: `drop_tokens` is diagnostic-only and
  `capacity_factor` no longer exists.

## Examples

### Complete Training Script

```python
import keras
import numpy as np
from dl_techniques.layers.moe import create_ffn_moe, MixtureOfExperts
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# 1. Dummy Data
batch_size, seq_len, hidden_dim, vocab_size = 4, 128, 768, 1000
train_data = (np.random.rand(batch_size, seq_len, hidden_dim), np.random.randint(0, vocab_size, (batch_size, seq_len)))
val_data = (np.random.rand(batch_size, seq_len, hidden_dim), np.random.randint(0, vocab_size, (batch_size, seq_len)))

# 2. Create model with MoE
inputs = keras.Input(shape=(seq_len, hidden_dim))
moe_layer = create_ffn_moe(
    num_experts=8,
    ffn_config={
        "type": "swiglu",
        "output_dim": hidden_dim,
        "ffn_expansion_factor": 4
    },
    top_k=2,
    aux_loss_weight=0.01,
    name="moe_ffn"
)
x = moe_layer(inputs)
outputs = keras.layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

# 3. Configure optimizer for MoE
lr_config = {"type": "cosine_decay", "learning_rate": 1e-4, "decay_steps": 10000}
opt_config = {"type": "adamw", "weight_decay": 0.01}
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# 4. Compile and train
model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(train_data[0], train_data[1], epochs=3, validation_data=val_data)

# 5. Monitor expert utilization
stats = model.get_layer("moe_ffn").get_expert_utilization()
print("\nExpert utilization stats:")
print(stats)
```

## References

- **Switch Transformer**: [Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- **GLaM**: [Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
- **SoftMoE**: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)
- **`dl_techniques` FFN Module**: See `layers/ffn/README.md` for FFN factory documentation.