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

The NLP heads' `pooling_type='attention'` (`layers/heads/nlp`) is not
`AttentionPooling` or `SequencePooling('attention')`: it is the head's own inline
`Dense(1, tanh)` scorer, a different mechanism with its own weights.

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

`get_sequence_pooling_info()` returns a deep copy of each registry entry (the
layer class itself is shared), so mutating the result, nested
`optional_params` included, does not change `SEQUENCE_POOLING_REGISTRY`.

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

Validation checks type existence, that every keyword is declared for the type,
required-parameter completeness, light value-range constraints, and, for
`'sequence'`, the `strategy` and `aggregation_method` names. A keyword the type
does not declare raises a `ValueError` carrying `STRICT_DROPPED_KEY_MARKER`
(`"unsupported parameter(s)"`), so a misspelling such as `dropout=` for
`dropout_rate=` fails instead of building a pooler without dropout. The Keras
base keywords (`name`, `dtype`, `trainable`, `activity_regularizer`,
`autocast`) are always accepted.

```python
# Raises ValueError: "Unknown sequence pooling type 'softmax_pool'"
validate_sequence_pooling_config('softmax_pool')

# Raises ValueError: "Parameter 'num_heads' must be positive, got -1"
validate_sequence_pooling_config('attention', num_heads=-1)

# Raises ValueError: "Parameter 'dropout_rate' must be between 0.0 and 1.0, got 1.5"
validate_sequence_pooling_config('attention', dropout_rate=1.5)

# Raises ValueError: "... 1 unsupported parameter(s) ['dropout'] ..."
validate_sequence_pooling_config('attention', dropout=0.5)

# Raises ValueError: "SequencePooling: unknown pooling strategy ['bogus'] ..."
validate_sequence_pooling_config('sequence', strategy='bogus')
```

## Layer-Specific Parameters

None of the three layer types have **required** parameters — every constructor
argument has a default, so each type can be created with zero kwargs.

### `attention`
**Required:** None
**Optional:** `hidden_dim` (default: 256), `num_heads` (default: 1),
`dropout_rate` (default: 0.0), `use_bias` (default: True),
`temperature` (default: 1.0), `kernel_initializer` (default: `'glorot_uniform'`),
`bias_initializer` (default: `'zeros'`), `kernel_regularizer` (default: None),
`bias_regularizer` (default: None)

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

The projection kernel and the context vector each get their own clone of
`kernel_initializer`, so a seedless initializer draws them independently (one
shared instance used to make the context vector a copy of the kernel's leading
values). The bias knobs act on the `tanh` projection's bias when `use_bias=True`.

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
`seq_len` weights are used (softmax-normalised) at call time. A static sequence
longer than `max_seq_len` raises a `ValueError` naming `max_seq_len`; with an
unknown length the backend refuses the longer input at run time
(`Incompatible shapes`). Nothing is truncated.

### `sequence`
**Required:** None
**Optional:** `strategy` (default: `'mean'`), `exclude_positions` (default: None),
`aggregation_method` (default: `'concat'`), `attention_hidden_dim` (default: 256),
`attention_num_heads` (default: 1), `attention_dropout_rate` (default: 0.0),
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

#### Shared knobs

`SequencePooling` builds its learnable strategies from one set of knobs, so one
value can reach more than one sub-layer:

| Knob | Reaches | Note |
|------|---------|------|
| `temperature` | every `AttentionPooling` and the `WeightedPooling` | One value for all learnable strategies; a positive finite number. |
| `attention_dropout_rate` | every `AttentionPooling` and the `WeightedPooling` | Despite the name it is also the `weighted` pooler's dropout. |
| `attention_hidden_dim` | every `AttentionPooling` | |
| `attention_num_heads` | the `multi_head_attention` pooler | `attention` always has one head. |
| `use_bias`, `bias_initializer`, `bias_regularizer` | every `AttentionPooling`'s projection | `weighted` and the `weighted_sum` aggregation weights have no bias. |
| `kernel_initializer` | every `AttentionPooling`'s projection kernel and context vector | One clone per weight. `weighted` and the aggregation weights always start at `'ones'` (a plain mean). |
| `kernel_regularizer` | every `AttentionPooling`'s kernel and context vector, the `weighted` position weights, the `weighted_sum` aggregation weights | The last two have only this one weight each. |
| `weighted_max_seq_len` | the `WeightedPooling`'s `max_seq_len` | |
| `top_k` | `top_k_mean`, `top_k_max` | A positive int. |

`exclude_positions` takes distinct non-negative ints (a negative position, a
repeat, a bool or a non-integer raises; there is no negative indexing, so `[-1]`
cannot silently do nothing). It is honoured by **every** strategy except `none`/`flatten`
(which return the sequence, so there is no index to move and no reduction to
exclude from). It used to be a silent no-op for the four positional strategies;
it is not any more. Unlike a keep-mask it is an EXPLICIT caller instruction, so
it outranks a positional mode's default: `cls`/`first` with position 0 excluded
return the first NON-excluded position, while still ignoring the keep-mask. If
the mask and the exclusions together leave nothing kept, every positional mode
degenerates to index 0 — the same documented, no-rescue degeneration as a
fully-masked row.

#### Pooling strategies (`PoolingStrategy`)

`strategy` accepts a single strategy name or a list of names. The full catalog of
**18 strategies** (the `PoolingStrategy` literal) is:

| Category | Strategy | Output dim | Description |
|----------|----------|------------|-------------|
| Positional | `cls` | `dim` | Token at index 0 (`inputs[:, 0, :]`), BY INTENT — a keep-mask does not redirect it; alias of `first`. |
| Positional | `first` | `dim` | Token at index 0, BY INTENT (see `cls`). |
| Positional | `last` | `dim` | Mask-aware: the LAST KEPT position (not `sum(mask) - 1`, which is only the last kept index for a contiguous-prefix mask). |
| Positional | `middle` | `dim` | Mask-aware: the middle of the KEPT positions (not `seq_len // 2` of the padded length). |
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
| Special | `none` | `(seq_len, dim)` | Pass-through; returns the sequence, with every masked position zeroed when a mask is given. |
| Special | `flatten` | `seq_len * dim` | Flattens the sequence to a single vector; masked positions are zeroed first, so padding content never reaches it. |

#### Aggregation methods (`AggregationMethod`)

When `strategy` is a list, the per-strategy outputs are combined with one of the
**4 aggregation methods** (the `AggregationMethod` literal):

| Method | Behaviour |
|--------|-----------|
| `concat` | Concatenate outputs along the feature axis (output dim is the sum of per-strategy dims). |
| `add` | Element-wise sum of all outputs. |
| `multiply` | Element-wise product of all outputs. |
| `weighted_sum` | Softmax-normalised learnable scalar per strategy, then a weighted sum (a single `aggregation_weights` vector is created in `build`). |

The constructor refuses, with a `ValueError`, every list `call` could not run:
an unknown strategy or aggregation name, a strategy named twice (a learnable one
would share one pooler), `none` in a list of two or more, and `add`, `multiply`
or `weighted_sum` over strategies of different widths (`mean_max`/`mean_std` are
`2 * dim`, `mean_max_min` is `3 * dim`, and `flatten` is `seq_len * dim`, so it
combines only under `concat`). `compute_output_shape` reports exactly the shape
`call` returns, including `flatten` inside a `concat` (`None` when `seq_len` is
unknown).

With a **single** strategy the aggregation method does not change the output —
the lone strategy output is returned directly — but its name is still checked.

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

`VisionEncoder` additionally uses `exclude_positions` to skip the CLS token
whenever it reduces the patch tokens:

```python
# Inside VisionEncoder __init__ (vision_encoder.py)
exclude_positions = (
    [0] if use_cls_token and output_mode not in _READS_THE_WHOLE_SEQUENCE_OR_CLS
    else []
)   # _READS_THE_WHOLE_SEQUENCE_OR_CLS = {'cls', 'first', 'none', 'flatten'}

self.pooling_layer = SequencePooling(
    strategy=output_mode,
    exclude_positions=exclude_positions,
    name='output_pooling',
)
```

#### Which consumers pool the CLS token

Each consumer follows its own reference rather than one package-wide rule.
`cls`/`first` always read the CLS token, and `none`/`flatten` return the whole
sequence.

| Consumer | Reductions over the sequence | Reference |
|----------|-----------------------------|-----------|
| `layers/transformers/VisionEncoder` | exclude the CLS token (every strategy but `cls`, `first`, `none`, `flatten`) | timm `global_pool_nlc` reduces `x[:, num_prefix_tokens:]`; its `map` pool drops prefix tokens by default |
| `models/vision/vit` (`mean`, `max`) | exclude | timm `global_pool_nlc` |
| `models/vision/vit_siglip` (`mean`, `max`) | exclude | big_vision SigLIP tower: `gap` averages patch tokens, no CLS token |
| `models/vision/vit_hmlp` (`mean`, `max`) | exclude | deit hMLP reference reads `x[:, 0]` only; timm convention for reductions |
| `models/vision/beit` (`use_mean_pooling`) | exclude | unilm BEiT `x[:, 1:, :].mean(1)` |
| `layers/transformers/TextEncoder` | include (every strategy) | sentence-transformers `Pooling` averages every token the attention mask keeps, `[CLS]` included |

Each row is pinned by a test that perturbs position 0 of the sequence the
consumer pools: `tests/test_layers/test_transformers/test_the_encoders_pool_the_cls_token_as_their_reference_does.py`
and `tests/test_models/test_the_vit_family_pools_the_cls_token_as_its_reference_does.py`.

This is the canonical pattern: pass the pooling mode straight through as
`strategy`, and let `SequencePooling` own the dispatch, masking, and exclusions.

### In a custom classification head

```python
import keras
from dl_techniques.layers.sequence_pooling import create_sequence_pooling_layer
from dl_techniques.utils.keras_registration import register_dl_technique


# The class is PREFIXED because the bare name `ClassificationHead` is already registered in
# this repo (`dl_techniques.layers.heads.vision.factory>ClassificationHead`), so its legacy
# `Custom>ClassificationHead` alias is taken and claiming it again raises AliasCollisionError.
# Prefixing removes the duplicate at its source and lets BOTH classes keep a legacy alias.
# (`legacy_alias=False` on both sides also silences the error, but it costs each of them its
# alias and leaves the duplicate name in place; it is a last resort for a class that cannot be
# renamed, and this tree has zero users of it.)
@register_dl_technique("my_project.classification_head")
class MyProjectClassificationHead(keras.layers.Layer):
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
  for creating pooling layers with validation and default merging. An undeclared
  keyword raises `ValueError` (it is never dropped). Returns a configured
  `keras.layers.Layer`.
- **`create_sequence_pooling_from_config(config)`** — Creates a layer from a
  configuration dictionary (pops the `'type'` key, forwards the rest).
- **`validate_sequence_pooling_config(pooling_type, **kwargs)`** — Validates type
  existence, declared keywords, required-parameter completeness, value ranges and
  the `'sequence'` strategy and aggregation names. Raises `ValueError`.
- **`get_sequence_pooling_info()`** — Returns a dict mapping each pooling type to a
  deep copy of its metadata (`description`, `required_params`, `optional_params`,
  `use_case`).
- **`list_sequence_pooling_types()`** — Returns an alphabetically sorted list of the
  registered pooling-type strings.

### Types and Registry

- **`SequencePoolingType`** — `Literal['attention', 'weighted', 'sequence']`.
- **`PoolingStrategy`** — `Literal` of the 18 strategy names listed above.
- **`AggregationMethod`** — `Literal['concat', 'add', 'multiply', 'weighted_sum']`.
- **`SEQUENCE_POOLING_REGISTRY`** — the underlying `Dict[str, Dict[str, Any]]`
  registry (prefer `get_sequence_pooling_info()` for read access).
- **`STRICT_DROPPED_KEY_MARKER`** — `"unsupported parameter(s)"`, the substring every
  undeclared-keyword `ValueError` from the factory carries.

### Classes

- **`AttentionPooling`**, **`WeightedPooling`**, **`SequencePooling`** — the three
  layer classes, importable directly from the package.
