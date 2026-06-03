# Sequence Pooling Module

The `dl_techniques.layers.sequence_pooling` module collapses a sequence of token
vectors into a single fixed-size summary representation, with a unified factory
interface for consistent layer creation, configuration management, and parameter
validation.

## Overview

Sequence pooling reduces a tensor of shape `(batch, seq_len, dim)` into a
fixed-size tensor of shape `(batch, dim)` (or a multiple of `dim` for combined
strategies), bridging a sequence encoder (Transformer, LSTM) to a downstream head
that expects a single vector per example. All layers are built with Keras 3 for
backend-agnostic compatibility and support full serialization.

The module exposes **three registered layer types** through the factory:

- `attention` (`AttentionPooling`) — learnable, **content-aware** attention pooling.
- `weighted` (`WeightedPooling`) — learnable, **content-agnostic** per-position pooling.
- `sequence` (`SequencePooling`) — a unified **facade** that dispatches 18 pooling
  strategies and 4 aggregation methods, internally composing the two leaf poolers
  for its learnable strategies.

The `sequence` facade is the broadest entry point: its `strategy` parameter
selects from a catalog of positional, statistical, learnable, top-k, and special
strategies, and multiple strategies can be combined via an aggregation method.

## Available Pooling Types

The following layers are supported by the factory system with automated parameter
validation and defaults. All consume `(batch, seq_len, dim)` and produce
`(batch, dim)` (the `sequence` facade may emit a multiple of `dim` for combined or
concatenated strategies).

| Type | Class | Description | Use Case |
|------|-------|-------------|----------|
| `attention` | `AttentionPooling` | Learnable, content-aware attention pooling. Each token is passed through a `tanh` projection, scored against a learnable context vector, softmax-normalised, and combined as a weighted sum (optionally over multiple averaged heads). | Collapsing a sequence into a summary vector when relevance is input-dependent: sentence/document embeddings, sequence-classification heads. |
| `weighted` | `WeightedPooling` | Learnable, content-agnostic per-position pooling. A scalar learnable weight per position (up to `max_seq_len`) is softmax-normalised and used for a weighted sum, capturing positional importance. | Fixed-length or positionally-structured sequences where token position carries a consistent importance signal. |
| `sequence` | `SequencePooling` | Unified facade exposing 18 strategies (positional, statistical, learnable, top-k, special) and 4 aggregation methods. Composes `AttentionPooling`/`WeightedPooling` for learnable modes. | The default choice for bridging a sequence encoder to a fixed-size head, and for experimenting with or combining multiple pooling strategies. |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.sequence_pooling import create_sequence_pooling_layer

# Content-aware attention pooling
attn_pool = create_sequence_pooling_layer('attention', hidden_dim=128)

# Content-agnostic position-weighted pooling
weighted_pool = create_sequence_pooling_layer('weighted', max_seq_len=256)

# The unified facade (no required params — succeeds with zero kwargs)
seq_pool = create_sequence_pooling_layer('sequence')

# A combined facade: mean + attention, concatenated
combined = create_sequence_pooling_layer(
    'sequence',
    strategy=['mean', 'attention'],
    aggregation_method='concat',
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.sequence_pooling import create_sequence_pooling_from_config

config = {
    'type': 'sequence',
    'strategy': 'mean',
    'aggregation_method': 'concat',
    'name': 'output_pooling',
}

pool = create_sequence_pooling_from_config(config)
```

The `'type'` key selects the layer; all remaining keys (including an optional
`'name'`) are forwarded as constructor parameters.

### Parameter Discovery

```python
from dl_techniques.layers.sequence_pooling import (
    get_sequence_pooling_info,
    list_sequence_pooling_types,
)

# Alphabetically sorted list of registered types
print(list_sequence_pooling_types())   # ['attention', 'sequence', 'weighted']

# Per-type metadata (description, required_params, optional_params, use_case)
info = get_sequence_pooling_info()
seq_info = info['sequence']
print(f"Required: {seq_info['required_params']}")               # []
print(f"Optional: {list(seq_info['optional_params'].keys())}")
```

`get_sequence_pooling_info()` returns shallow copies of each registry entry, so
mutating the result does not corrupt the global `SEQUENCE_POOLING_REGISTRY`.

### Validation

```python
from dl_techniques.layers.sequence_pooling import validate_sequence_pooling_config

# Validate a configuration before creation
try:
    validate_sequence_pooling_config('attention', hidden_dim=128, num_heads=4)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

Validation checks type existence, required-parameter completeness, and light
value-range constraints:

```python
# Raises ValueError: "Unknown sequence pooling type 'softmax_pool'"
validate_sequence_pooling_config('softmax_pool')

# Raises ValueError: "Parameter 'num_heads' must be positive, got -1"
validate_sequence_pooling_config('attention', num_heads=-1)

# Raises ValueError: "Parameter 'dropout_rate' must be between 0.0 and 1.0, got 1.5"
validate_sequence_pooling_config('attention', dropout_rate=1.5)
```

## Layer-Specific Parameters

None of the three layer types have **required** parameters — every constructor
argument has a default, so each type can be created with zero kwargs.

### `attention`
**Required:** None
**Optional:** `hidden_dim` (default: 256), `num_heads` (default: 1),
`dropout_rate` (default: 0.0), `use_bias` (default: True),
`temperature` (default: 1.0), `kernel_initializer` (default: `'glorot_uniform'`),
`kernel_regularizer` (default: None)

```python
attn = create_sequence_pooling_layer(
    'attention',
    hidden_dim=128,
    num_heads=4,
    dropout_rate=0.1,
    temperature=0.5,
)
```

With `num_heads > 1` the per-head weighted sums are averaged; with `num_heads == 1`
the single head's output is returned directly.

### `weighted`
**Required:** None
**Optional:** `max_seq_len` (default: 512), `dropout_rate` (default: 0.0),
`temperature` (default: 1.0), `initializer` (default: `'ones'`),
`regularizer` (default: None)

```python
weighted = create_sequence_pooling_layer(
    'weighted',
    max_seq_len=256,
    temperature=2.0,
)
```

A scalar weight is allocated per position up to `max_seq_len`; only the first
`seq_len` weights are used (softmax-normalised) at call time.

### `sequence`
**Required:** None
**Optional:** `strategy` (default: `'mean'`), `exclude_positions` (default: None),
`aggregation_method` (default: `'concat'`), `attention_hidden_dim` (default: 256),
`attention_num_heads` (default: 1), `attention_dropout` (default: 0.0),
`weighted_max_seq_len` (default: 512), `top_k` (default: 10),
`temperature` (default: 1.0), `use_bias` (default: True),
`kernel_initializer` (default: `'glorot_uniform'`),
`bias_initializer` (default: `'zeros'`), `kernel_regularizer` (default: None),
`bias_regularizer` (default: None)

```python
pool = create_sequence_pooling_layer(
    'sequence',
    strategy=['mean', 'max'],
    aggregation_method='concat',
    exclude_positions=[0],     # e.g. skip a CLS token during statistical pooling
)
```

#### Pooling strategies (`PoolingStrategy`)

`strategy` accepts a single strategy name or a list of names. The full catalog of
**18 strategies** (the `PoolingStrategy` literal) is:

| Category | Strategy | Output dim | Description |
|----------|----------|------------|-------------|
| Positional | `cls` | `dim` | First token (`inputs[:, 0, :]`); alias of `first`. |
| Positional | `first` | `dim` | First token of the sequence. |
| Positional | `last` | `dim` | Last token (mask-aware: last unmasked position). |
| Positional | `middle` | `dim` | Token at `seq_len // 2`. |
| Statistical | `mean` | `dim` | Mask-aware mean over the sequence. |
| Statistical | `max` | `dim` | Mask-aware max over the sequence. |
| Statistical | `min` | `dim` | Mask-aware min over the sequence. |
| Statistical | `sum` | `dim` | Mask-aware sum over the sequence. |
| Advanced statistical | `mean_max` | `2 * dim` | Concatenation of `mean` and `max`. |
| Advanced statistical | `mean_std` | `2 * dim` | Concatenation of `mean` and the (mask-aware) std. |
| Advanced statistical | `mean_max_min` | `3 * dim` | Concatenation of `mean`, `max`, and `min`. |
| Learnable | `attention` | `dim` | Single-head `AttentionPooling`. |
| Learnable | `multi_head_attention` | `dim` | Multi-head `AttentionPooling` (`attention_num_heads`). |
| Learnable | `weighted` | `dim` | `WeightedPooling` (per-position learnable weights). |
| Top-k | `top_k_mean` | `dim` | Mean over the `top_k` tokens by squared L2 norm. |
| Top-k | `top_k_max` | `dim` | Max over the `top_k` tokens by squared L2 norm. |
| Special | `none` | `(seq_len, dim)` | Pass-through; returns the sequence unchanged. |
| Special | `flatten` | `seq_len * dim` | Flattens the sequence to a single vector. |

#### Aggregation methods (`AggregationMethod`)

When `strategy` is a list, the per-strategy outputs are combined with one of the
**4 aggregation methods** (the `AggregationMethod` literal):

| Method | Behaviour |
|--------|-----------|
| `concat` | Concatenate outputs along the feature axis (output dim is the sum of per-strategy dims). Rejects the `none` strategy. |
| `add` | Element-wise sum of all outputs (requires matching dims). |
| `multiply` | Element-wise product of all outputs (requires matching dims). |
| `weighted_sum` | Softmax-normalised learnable scalar per strategy, then a weighted sum (a single `aggregation_weights` vector is created in `build`). |

With a **single** strategy the aggregation method is irrelevant — the lone
strategy output is returned directly.

```python
# Multi-strategy + learnable aggregation
pool = create_sequence_pooling_layer(
    'sequence',
    strategy=['mean', 'attention', 'weighted'],
    aggregation_method='weighted_sum',
    attention_hidden_dim=128,
)
```

## Direct Layer Instantiation

While the factory is recommended, the layer classes can be imported and
instantiated directly.

```python
from dl_techniques.layers.sequence_pooling import (
    SequencePooling,
    AttentionPooling,
    WeightedPooling,
)

attn = AttentionPooling(hidden_dim=128, num_heads=4)
weighted = WeightedPooling(max_seq_len=256)
seq = SequencePooling(strategy=['mean', 'max'], aggregation_method='concat')
```

The two type aliases are also importable for annotating call sites:

```python
from dl_techniques.layers.sequence_pooling import PoolingStrategy, AggregationMethod
```

## Integration Patterns

### In a Transformer encoder (real usage)

Both `TextEncoder` and `VisionEncoder` (in
`dl_techniques.layers.transformers`) delegate their output pooling to
`SequencePooling`, exposing the strategy as an `output_mode` constructor argument.

```python
from ..sequence_pooling import SequencePooling, PoolingStrategy

# Inside the encoder __init__ (text_encoder.py)
self.pooling_layer = SequencePooling(
    strategy=output_mode,          # e.g. 'cls', 'mean', 'max', 'none'
    name='output_pooling',
)
```

`VisionEncoder` additionally uses `exclude_positions` to skip the CLS token when
statistically pooling the patch tokens:

```python
# Inside VisionEncoder __init__ (vision_encoder.py)
exclude_positions = [0] if (use_cls_token and output_mode in ['mean', 'max']) else []

self.pooling_layer = SequencePooling(
    strategy=output_mode,
    exclude_positions=exclude_positions,
    name='output_pooling',
)
```

This is the canonical pattern: pass the pooling mode straight through as
`strategy`, and let `SequencePooling` own the dispatch, masking, and exclusions.

### In a custom classification head

```python
import keras
from dl_techniques.layers.sequence_pooling import create_sequence_pooling_layer


@keras.saving.register_keras_serializable()
class ClassificationHead(keras.layers.Layer):
    def __init__(self, num_classes, pooling_type='attention', **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.pool = create_sequence_pooling_layer(pooling_type, name='pool')
        self.classifier = keras.layers.Dense(num_classes, name='classifier')

    def call(self, inputs, mask=None):
        pooled = self.pool(inputs, mask=mask)   # (batch, seq, dim) -> (batch, dim)
        return self.classifier(pooled)
```

## API Reference

### Functions

- **`create_sequence_pooling_layer(pooling_type, name=None, **kwargs)`** — Factory
  for creating pooling layers with validation, default merging, and parameter
  filtering. Returns a configured `keras.layers.Layer`.
- **`create_sequence_pooling_from_config(config)`** — Creates a layer from a
  configuration dictionary (pops the `'type'` key, forwards the rest).
- **`validate_sequence_pooling_config(pooling_type, **kwargs)`** — Validates type
  existence, required-parameter completeness, and value ranges. Raises `ValueError`.
- **`get_sequence_pooling_info()`** — Returns a dict mapping each pooling type to a
  shallow copy of its metadata (`description`, `required_params`, `optional_params`,
  `use_case`).
- **`list_sequence_pooling_types()`** — Returns an alphabetically sorted list of the
  registered pooling-type strings.

### Types and Registry

- **`SequencePoolingType`** — `Literal['attention', 'weighted', 'sequence']`.
- **`PoolingStrategy`** — `Literal` of the 18 strategy names listed above.
- **`AggregationMethod`** — `Literal['concat', 'add', 'multiply', 'weighted_sum']`.
- **`SEQUENCE_POOLING_REGISTRY`** — the underlying `Dict[str, Dict[str, Any]]`
  registry (prefer `get_sequence_pooling_info()` for read access).

### Classes

- **`AttentionPooling`**, **`WeightedPooling`**, **`SequencePooling`** — the three
  layer classes, importable directly from the package.
