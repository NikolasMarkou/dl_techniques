# Authoring Keras 3 Custom Layers and Models

A reference for writing Keras 3 custom layers and models that are correct, serializable, and
verifiably do what they claim.

## Scope

Making a layer construct and serialize is the easy half, and it is not where the defects are. A
library-wide audit of every model package in a large Keras 3 codebase found this pattern repeatedly:

| Defect | Symptom to the author |
|---|---|
| A parameter validated, stored, serialized, documented — and read by no code path | `max\|dy\| = 0.000e+00` across every legal value |
| A rotary embedding fed the head axis instead of the sequence axis | an **exact algebraic no-op**: `(Rq)·(Rk) = q·k` |
| `add_weight(zeros)` + `.assign()` in `build()` | table stays all zeros in every real model |
| A decoder-only LM with no causal mask | bidirectional attention under a next-token objective |
| A reloaded model that restored **zero** weights | round-trip test passed |

Every one shipped behind a green suite: shapes matched, parameter counts matched, gradients existed,
serialization round-tripped, loss curves looked normal.

**Construction correctness and behavioural correctness are different properties, and only the first
is easy to test.** Sections 1–12 are what to write. Section 13 is how to prove it. Section 13 is not
optional polish — a guard that cannot fail is the most likely outcome of writing a new test.

## How to read this

| You are | Read |
|---|---|
| Writing a new layer | §1–§4, §6–§9, then the checklist in §16 |
| Writing a new model package | all of it; §5 is the package shape |
| Fixing a bug | §14 to find the pitfall, §13 to build a guard that can fail |
| Reviewing | §14 as the code checklist, §13.6 as the test checklist |

Conventions used below: **❌ WRONG / ✅ CORRECT** code pairs; **Measured:** lines carry a figure that
was observed, not estimated; **Detect:** lines name the probe that catches the defect.

---

## Table of Contents

1. [Core Design Principles](#1-core-design-principles)
2. [Essential Setup and Registration](#2-essential-setup-and-registration)
3. [Layer Implementation Patterns](#3-layer-implementation-patterns)
4. [Graph-Safe Operations in call()](#4-graph-safe-operations-in-call)
5. [Model Implementation Patterns](#5-model-implementation-patterns)
6. [Configuration Management](#6-configuration-management)
7. [Serialization and Deserialization](#7-serialization-and-deserialization)
8. [Build Materialization and Weight Compatibility](#8-build-materialization-and-weight-compatibility)
9. [Factory Patterns and Layer Reuse](#9-factory-patterns-and-layer-reuse)
10. [Numerics and Precision](#10-numerics-and-precision)
11. [The Training Path](#11-the-training-path)
12. [Causality, Masking and Composition](#12-causality-masking-and-composition)
13. [Testing and Validation](#13-testing-and-validation)
14. [Common Pitfalls and Solutions](#14-common-pitfalls-and-solutions)
15. [Troubleshooting Guide](#15-troubleshooting-guide)
16. [Summary Checklists](#16-summary-checklists)
17. [Appendix: Refuted Claims](#17-appendix-refuted-claims)

---

## 1. Core Design Principles

### 1.1 The Serialization Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAVING (.keras format)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. get_config() is called on each layer                        │
│  2. Config dict is serialized to JSON                           │
│  3. For each BUILT layer, weight VALUES are extracted           │
│  4. Everything is packaged into the .keras archive              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LOADING (.keras format)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Parse JSON config                                           │
│  2. __init__(**config)  →  creates an UNBUILT layer             │
│  3. build()             →  creates the weight variables         │
│  4. Load saved VALUES into those variables                      │
└─────────────────────────────────────────────────────────────────┘
```

Everything in this document follows from step 3 of the load path:

> **If a weight does not exist at the moment values are restored, the value has nowhere to land —
> and nothing raises.**

### 1.2 The Golden Rule: Create vs. Build

| Method | Runs | What belongs there |
|--------|------|--------------------|
| `__init__` | once, at instantiation | CREATE all sub-layers; STORE all configuration |
| `build` | once, when shapes are known | CREATE this layer's weights; MATERIALIZE the sub-layer tree |
| `call` | every batch | symbolic operations only |
| `compute_output_shape` | shape inference | output shape from STORED CONFIG, on an unbuilt layer |
| `get_config` | serialization | every constructor argument |

**NEVER in `__init__`**
- create weights (`self.add_weight`)
- inspect `input_shape` or run any shape-dependent operation

**ALWAYS in `build`**
- create this layer's own weights
- build each sub-layer that `call()` will run — and only those (§8)
- call `super().build(input_shape)` last

**NEVER in `call`**
- construct a layer, mutate a Python container, call `.numpy()` / `convert_to_numpy`
- branch a Python `if` on a tensor *value*
- log

### 1.3 Create Unconditionally, Use Conditionally

**❌ WRONG** — the weight set depends on a flag:

```python
def __init__(self, use_feature_a=True, **kwargs):
    super().__init__(**kwargs)
    if use_feature_a:
        self.feature_a = FeatureLayer()
```

**✅ CORRECT** — create always, gate the usage:

```python
def __init__(self, use_feature_a=True, **kwargs):
    super().__init__(**kwargs)
    self.use_feature_a = use_feature_a
    self.feature_a = FeatureLayer(name="feature_a")

def call(self, inputs, training=None):
    x = inputs
    if self.use_feature_a:
        x = self.feature_a(x, training=training)
    return x
```

**Be precise about what this buys.** It does *not* make the checkpoint configuration-independent:
an unused layer is never built, so it contributes no weights either way — and building it anyway is
the over-building defect of §8.2. What it buys is a **stable object graph and stable names**.
Conditional creation shifts the auto-generated names of every layer after it, so flipping an
unrelated flag silently renames weights and breaks by-name transfer. With an explicit `name=` on
every sub-layer that hazard mostly disappears, and always-create becomes a consistency preference
rather than a correctness requirement. Both are cheap; do both.

**Two consequences that have been mistaken for defects:**

1. A layer created but never called is still parameter-counted, still optimizer-tracked, and still
   appears in `model.weights` and in a gradient walk — while contributing **exactly 0.0**. It will
   emit a missing-gradient `UserWarning`. That warning is sometimes correct and sometimes the
   intended cost of layout stability. Diagnose the `call()` branch before "fixing" it; if the
   inertness is deliberate, set `layer.trainable = False` in `__init__` (pre-build) and say so in a
   comment. Note `model.trainable = True` silently undoes it.
2. The contract does **not** survive a rebuild as a functional graph. Keras prunes any layer with no
   path to a declared output, even one constructed and applied on a dead branch. Do not write a test
   asserting a contract the graph cannot hold; document the divergence and expose a named feature tap.

### 1.4 Configuration as Data

Every architectural decision is a constructor argument with a serializable value. No `add_layer()`
builder methods, no callables stored as configuration, no mutable defaults.

```python
# ❌ WRONG
def __init__(self, layer_sizes=[64, 128]): ...

# ✅ CORRECT
def __init__(self, layer_sizes: Optional[List[int]] = None):
    self.layer_sizes = [64, 128] if layer_sizes is None else list(layer_sizes)
```

---

## 2. Essential Setup and Registration

### 2.1 Core Imports

```python
import keras
from keras import ops, layers, initializers, regularizers, constraints, activations

from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal

import numpy as np
import tensorflow as tf          # tests and graph-mode probes only
```

### 2.2 The Registration Decorator

```python
@keras.saving.register_keras_serializable(package="my_project")
class MyLayer(keras.layers.Layer):
    ...
```

**Rule: always pass an explicit `package=`.**

A bare `register_keras_serializable()` produces a key that is **independent of the defining module**,
so two classes with the same name in different packages claim the same key. Whichever module imports
**last** silently wins; loading a saved model of the other one is broken, and which one breaks
depends on import order.

**Measured:** collisions found between unrelated model packages on the generic names `Downsample`,
`Upsample` and `RepMixerBlock`.

**Corollaries**

| Rule | Reason |
|---|---|
| Prefix a generic class name if one already exists (`FastVitRepMixerBlock`, not a second `RepMixerBlock`) | the registry is keyed by name |
| Key a `custom_objects` dict by `keras.saving.get_registered_name(cls)`, **never** the bare class name | `_retrieve_class_or_fn` never uses a literal class name as a lookup key for classes — a dict keyed that way is decorative and every entry is ignored |
| Never bind a name in a package `__init__.py` matching one of that package's own **subpackages** | re-exporting a class `SAM2` from a package containing a `SAM2/` subpackage shadows it, and `from ...SAM.SAM2.model import ...` stops resolving |

The subpackage-shadowing break happens at **collection** time, so per-package test runs never see it.
Run the tree-wide collection gate after any change to a package's public surface:

```bash
pytest tests/ -q --collect-only
```

### 2.3 Type Hints

```python
def __init__(
    self,
    units: int,
    activation: Optional[Union[str, Callable]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
    use_bias: bool = True,
    **kwargs: Any,
) -> None: ...

def call(
    self,
    inputs: keras.KerasTensor,
    training: Optional[bool] = None,
    mask: Optional[keras.KerasTensor] = None,
) -> keras.KerasTensor: ...

def compute_output_shape(
    self, input_shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]: ...
```

---

## 3. Layer Implementation Patterns

### 3.1 Pattern 1 — A Layer With Its Own Weights

```python
@keras.saving.register_keras_serializable(package="my_project")
class SimpleCustomLayer(keras.layers.Layer):
    """One sentence naming the layer and what distinguishes it.

    Prose on the principle: why this mechanism resolves the problem it addresses,
    with inline math in backticks where it clarifies rather than decorates.

    Args:
        units: Dimensionality of the output space.
        activation: Activation function. Defaults to None.
        use_bias: Whether to add a learnable bias. Defaults to True.

    Input shape:
        N-D tensor `(batch_size, ..., input_dim)`.

    Output shape:
        N-D tensor `(batch_size, ..., units)`.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # 1. VALIDATE, naming the offending value
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        # 2. STORE all configuration
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

        # 3. Declare weight attributes; they are created in build()
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("The last dimension of the input must be defined")

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer="zeros", trainable=True
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        outputs = ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = ops.add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        })
        return config
```

### 3.2 Pattern 2 — A Layer Containing Sub-layers

Sub-layers are **created** in `__init__` and **built** in `build()`:

```python
def __init__(self, hidden_dim, output_dim, dropout_rate=0.1, use_norm=True, **kwargs):
    super().__init__(**kwargs)
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
    self.hidden_dim, self.output_dim = hidden_dim, output_dim
    self.dropout_rate, self.use_norm = dropout_rate, use_norm

    self.dense1  = layers.Dense(hidden_dim, activation="gelu", name="dense1")
    self.dropout = layers.Dropout(dropout_rate, name="dropout")
    self.norm    = layers.LayerNormalization(epsilon=1e-6, name="norm")   # created ALWAYS
    self.dense2  = layers.Dense(output_dim, name="dense2")

def build(self, input_shape):
    self.dense1.build(input_shape)
    hidden_shape = self.dense1.compute_output_shape(input_shape)
    self.dropout.build(hidden_shape)
    if self.use_norm:                     # built ONLY if call() runs it
        self.norm.build(hidden_shape)
    self.dense2.build(hidden_shape)
    super().build(input_shape)

def call(self, inputs, training=None):
    x = self.dense1(inputs)
    x = self.dropout(x, training=training)
    if self.use_norm:
        x = self.norm(x, training=training)
    return self.dense2(x)
```

The asymmetry is deliberate, and it is where §1.3 and §8 meet:

```
 norm is CREATED unconditionally   →  stable object graph and names   (§1.3)
 norm is BUILT   conditionally     →  build() materializes exactly
                                      what call() runs, no more       (§8)
```

Creating it costs nothing. Building a layer that `call()` skips adds weights the lazy path never
makes.

**Always give sub-layers explicit names, including inside loops** (`name=f"block_{i}"`).
Auto-generated names shift when depth changes, and checkpoints stop matching.

### 3.3 Pattern 3 — Constant Tables (the `StatelessScope` trap)

**Rule: never compute a constant table in `build()` and `.assign()` it into a weight.**

Keras 3 runs the symbolic build pass inside a `StatelessScope` whenever a sub-layer is first reached
from a **parent's `call()`**. The scope records the assignment and discards it. The table stays at
its initializer value in every real model.

**❌ WRONG** — all zeros in any model where a parent's `call()` builds this layer:

```python
def build(self, input_shape):
    self.inv_freq = self.add_weight(
        name="inv_freq", shape=(self.dim // 2,), initializer="zeros", trainable=False
    )
    self.inv_freq.assign(1.0 / (self.theta ** (ops.arange(0, self.dim, 2) / self.dim)))
    super().build(input_shape)
```

**✅ CORRECT** — the initializer computes it, so there is nothing to discard:

```python
def build(self, input_shape):
    def _inv_freq_init(shape, dtype=None):
        idx = np.arange(0, self.dim, 2, dtype="float64")[: shape[0]]
        return ops.cast(1.0 / (self.theta ** (idx / self.dim)), dtype or self.compute_dtype)

    self.inv_freq = self.add_weight(
        name="inv_freq", shape=(self.dim // 2,), initializer=_inv_freq_init, trainable=False
    )
    super().build(input_shape)
```

The trap is **path-dependent**, which is why it survives so long:

| How the layer gets built | Table value |
|---|---|
| `layer.build(shape)` called directly | correct |
| eager `layer(x)` on a top-level layer | correct |
| `keras.Model(inp, layer(inp))` | correct |
| first reached from a **parent's `call()`** | **all zeros** |

The last row is every real model.

**Measured:** a trend-only N-BEATS returned exactly `0.0` everywhere and still trained and still
reported a loss.

**Detect:** §13.2.4 — build the layer *through a parent*. A unit test that calls `.build(...)`
directly is structurally blind to this.

**Related, same cause.** Never materialize a constant with `ops.convert_to_tensor` inside `build()`
and close over it: the tensor binds to the tracing `FuncGraph`, and a later `fit()` on an unbuilt
model dies with `InaccessibleTensorError`. Keep the constant as a NumPy array; convert inside
`call()`.

### 3.4 Implementing `compute_output_shape`

Every custom layer implements it. It must work on an **unbuilt** layer, derived from stored
configuration.

```python
# ❌ WRONG - fails before build
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.kernel.shape[-1])

# ✅ CORRECT - uses stored config
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.units)
```

**Rule: shape arithmetic lives in exactly one pure helper**, called by `build()`, `call()` and
`compute_output_shape` alike.

**Measured:** one layer carried three copies of an overlapping-segment formula, one of which used `+`
where the others used `-`, and built its nodes for the wrong length. Another declared a halved
spatial extent unconditionally while the stride lived on a sub-layer a flag could remove.

**Detect:** pin `compute_output_shape` against the layer's own forward output for **every branch of
every mode flag**, including branches no shipped variant reaches.

### 3.5 Validation Placement

| Check | Where | Why |
|---|---|---|
| Argument ranges / types | `__init__` | fail before anything is built |
| Cross-parameter contracts `call()` relies on | `__init__` | a config that builds but cannot forward is a construction-time error |
| Static shape contracts | `build()` **and** `call()` | a contract checked only in `build()` is checked once, against whatever shape arrived first |

**Measured:** a contract on a singleton axis checked only at build silently accepted a non-singleton
axis and convolved the wrong dimension, with no error. Separately, a mismatched dimension pair
constructed, validated and **built** cleanly, then raised on the first forward pass.

`InputSpec` cannot close a dynamic-shape hole: `assert_input_compatibility` tests
`shape[axis] not in {value, None}`, so an unknown dimension is explicitly **accepted**.

When you add a cross-parameter check, **sweep every shipped preset** against it — one preset was
found sitting on a degenerate boundary with nothing saying so.

---

## 4. Graph-Safe Operations in `call()`

Keras traces `call()` once with symbolic inputs. Anything that reads a tensor's *value* at trace time
is either an error or, worse, silently frozen to whatever the trace saw.

### 4.1 The Rules

| ❌ Never in `call()` | ✅ Instead |
|---|---|
| `list(shape)`, `int(x)`, `float(x)` on a tensor | `ops.shape(x)`, index the tensor |
| `.numpy()`, `convert_to_numpy` | stay symbolic |
| Python `if` on a tensor *value* | `ops.where`, `ops.cond` |
| Python `for` over a tensor dimension | vectorize, or `ops.scan` |
| constructing a layer | construct in `__init__` |
| appending to a Python list | a `keras.Variable` |
| `logger.*` | log once in `__init__` |

```python
def call(self, inputs, training=None):
    shape = ops.shape(inputs)         # a tensor, not a list
    batch = shape[0]                  # slicing a tensor is fine
    x = ops.reshape(inputs, ops.stack([batch, -1]))
    return ops.where(ops.mean(x) > 0, x * 2.0, x)
```

Python `if` on **configuration** is correct and expected — `self.use_norm` is known at trace time.
The rule is about tensor *values*.

### 4.2 Operations That Are Traps Under Tracing

**`keras.ops.tril` / `keras.ops.triu`** — both raise the moment they are traced:

```
TypeError: ('pred must not be a Python bool', True)
```

They pass every eager test and simultaneously break `fit`, `predict`, `jit_compile=True`, `.keras`
save/load and every symbolic-shape path. Build triangular masks by comparing `ops.arange`, or reuse a
shared causal-mask helper.

**Symbolic `training` into `BatchNormalization` or `Dropout`** — measured on Keras 3.8:

| Layer | traced `tf.constant(True)` | traced `tf.constant(False)` |
|---|---|---|
| `BatchNormalization` | raises `OperatorNotAllowedInGraphError` | raises |
| `Dropout` | raises | raises |
| `LayerNormalization` | fine | fine |

`tf.get_static_value` on a traced argument returns `None`, so the value cannot be recovered inside
the trace. Route through a gate that keeps the Python-bool path byte-identical and sends only a
tensor flag to `ops.cond` — and gate only the layers that need it.

**`training=` propagation** — Keras 3 propagates `training` through a single mutable `CallContext`
slot that every nested `__call__` overwrites and only the outermost entry restores. A sibling
sub-layer forcing a different `training` poisons the ambient value for every later un-forwarded call.
**Forward `training=` explicitly**, even where omitting it is currently a no-op.

**NumPy fancy indexing** (`t[batch_idx, item_idx]`) is invalid on a backend tensor and raises
eagerly — the layer is dead on every forward pass. Use `ops.take_along_axis`.

### 4.3 A `call()` Crash During Build-Tracing Becomes a Warning

Keras converts an exception raised while tracing `call()` during a build pass into a `UserWarning`.
A completely broken layer therefore sits inside a green suite with exit code 0.

> **Exit code 0 is not evidence.**

Grep the test output for the exception text, or run the gate under `-W error::UserWarning`.

---

## 5. Model Implementation Patterns

A package implementing one architecture with named variants follows the shape below. It is a target,
not a universal law; the exemptions are in §5.6.

### 5.1 Module Skeleton

The module docstring is **substantive prose, not a template**:

| # | Element |
|---|---|
| 1 | **One opening sentence** naming the architecture and its distinguishing options — a sentence, not a title with an `====` underline |
| 2 | **The principle**: what problem the architecture solves and *why its mechanism resolves it*. Inline math in backticks (`` `y = F(x) + x` ``) where an equation carries the idea |
| 3 | **The architecture**: stage/block structure, design trade-offs, and the places where the code does something non-obvious, and why |
| 4 | **Every deliberate behavioural choice**, stated as a choice with its reason (e.g. why `pretrained=True` raises rather than warning) |
| 5 | **`References:`** as `- Author et al., YEAR. Title. (url)`, including the papers the design actually draws on |

This **replaces** terse `Model Variants:` / `Usage Examples:` boilerplate that restates the variant
dict and the factory signature sitting directly below it. Length follows the architecture; do not pad,
and do not move real explanation into a README to hit a line budget.

```python
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from my_project.utils.logger import logger

# ---------------------------------------------------------------------
```

### 5.2 Class API

```python
@keras.saving.register_keras_serializable(package="my_project")
class MyModel(keras.Model):
    MODEL_VARIANTS = {"my_model_base": {...}}

    def __init__(self, ..., **kwargs):
        super().__init__(**kwargs)
        # 1. validate arguments, raising ValueError naming the offending value
        # 2. resolve None-sentinel defaults (never a mutable default argument)
        # 3. store configuration on self
        # 4. call self._build_<part>() helpers
        # 5. ONE logger.info summarizing what was created
```

| Member | Requirement |
|---|---|
| `call(self, inputs, training=None)` | **no logging inside** — it fires on every trace |
| `get_config()` | every constructor argument, complex objects serialized |
| `from_config()` | deserializes them |
| `from_variant(cls, variant, ..., pretrained=False, **kwargs)` | raises `ValueError` **listing available keys** on a miss; accepts the overrides its own docstring advertises; does not splat description metadata into the constructor |

**Measured:** several `from_variant` implementations raised `TypeError` on exactly the override their
docstring advertised.

### 5.3 Two Variant Tables Are Not One

Where a package has both, keep them separate:

```
 architecture table   'tiny'      -> {hidden_size: 192, num_layers: 12, num_heads: 3, ...}
 public-name registry 'beit_tiny' -> {scale: 'tiny'}
```

Merging them collapses a name-to-scale indirection that exists precisely so a variant can pin a patch
size or an input resolution alongside its scale. Where a package genuinely has one table, give it one
canonical name and add an **alias** rather than renaming — trainers and tests reference the old
spelling, and a rename buys nothing an alias does not.

**Rule: variant values are derived from a named reference** — the released checkpoint's own config,
fetched and cited — never from a sibling file in the same codebase, never from a paper table read
once.

**Measured:** variant tables have shipped wrong by roughly half the parameter count in their own
name, with the test suite pinning the wrong values.

### 5.4 Pretrained Weights

```
load_pretrained_weights(path)   →  local path; dummy forward first if needed
_download_weights(variant)      →  raises NotImplementedError, naming the variant
                                    and showing the local-path alternative
```

| Rule | Reason |
|---|---|
| No `by_name` on `Model.load_weights` | Keras 3 removed it. Where it survived as a supposed no-op, every call actually **raised**, and the enclosing `except` turned it into a warning and continued with random weights |
| Never a placeholder URL table + `try/except` that warns and continues | `pretrained=True` then silently returns an untrained model |
| Never swallow a load failure into a warning | a local-path load that restores nothing must raise |

**Measured:** nine packages in one codebase shipped the warn-and-continue pattern simultaneously,
and the documented alternative was broken too.

### 5.5 Factory, Exports and Hygiene

A module-level `create_<name>(variant="<default>", ...)` that delegates to `from_variant` with **no
logic of its own**. The package `__init__.py` exports the class and the factory with a curated
`__all__`, and binds no name matching one of its own subpackages (§2.2).

**Hygiene**

- No comment restating the line below it (`# Store configuration`, `# Squeeze`).
- No `# 1. / # 2. / # 3.` step ladders. A comment earns its place by explaining *why*, or by
  recording a non-obvious constraint — not by narrating *what*.
- No mutable default arguments; `None` sentinels resolved in the body.
- No unused imports — an imported-but-never-called logger is the common case.
- Prefer `keras.ops`; `keras.config.floatx()` / `keras.config.epsilon()` over `keras.backend.*`.
- Route logging through one logger; never `print`.
- **Never convert docstring style wholesale.** Match the file you are editing.
- **Never delete or reword a decision-anchor comment** recording why a non-obvious choice was made.
  Supersede it in place with a dated note. Files with high comment density are often dense *because
  of* these anchors — never target a file by comment density.

### 5.6 When the Shape Does Not Apply

| Case | What to do |
|---|---|
| No genuine named variants | do not invent a variant table; apply §5.1, §5.5 only, and say why in the README |
| Functional builders returning `keras.Model(inputs, outputs)` with no subclass | keep them functional — converting breaks existing checkpoints. §5.1 and §5.5 still apply |
| Multi-model families / nested packages | apply the shape per *inner architecture*, not per directory |

Before classifying a package as functional, verify with `grep -n "^class .*(.*Model)" <pkg>/*.py`. A
grep-based census of this question was wrong about several packages.

---

## 6. Configuration Management

### 6.1 Complete `get_config` / `from_config`

```python
def get_config(self):
    config = super().get_config()
    config.update({
        "depth": self.depth,
        "activation": activations.serialize(self.activation),
        "kernel_initializer": initializers.serialize(self.kernel_initializer),
        "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
    })
    return config

@classmethod
def from_config(cls, config):
    for key in ("kernel_initializer", "kernel_regularizer"):
        if key in config and isinstance(config[key], dict):
            config[key] = getattr(keras, key.split("_")[1] + "s").deserialize(config[key])
    return cls(**config)
```

### 6.2 `**kwargs` Is Not a Channel to Your Sub-layers

A key read out of `**kwargs` that is **also** forwarded to `super().__init__()` is dead on arrival:
Keras rejects unknown base-class keys. Assigning `self.shared_kwargs = kwargs` and forwarding it to
children leaks base arguments the other way. Split base kwargs from pass-through kwargs explicitly
and name them.

> A `from_config` that pops base keys is a **tell that `__init__` has this bug**. Once `__init__` is
> fixed, that pop discards `name` and `trainable` — so a frozen head silently reloads unfrozen, with
> bit-identical outputs.

### 6.3 Pair Every New Validation Raise With a Migration Path

A new `ValueError` on a value the **old default** produced breaks deserialization of every existing
checkpoint.

```
 constructor  →  keep the raise          (fresh code must be correct)
 from_config  →  substitute + warn        (old checkpoints must still load)
                 that numerics changed
```

Record every checkpoint-affecting change in a document that **ships with the code**; a note in an
untracked planning directory does not reach the next reader.

Sometimes the right answer is to refuse the shim: a remapping that would rebuild a *different* weight
tree than the file contains is worse than a hard failure.

### 6.4 Caches Derived From Weights

Value-exact round-tripping is not sufficient for a cache computed **from a weight**.

**Measured:** a cached positional table computed from a stale pre-restore weight was off by `1.999`
while thirteen round-trip tests passed.

**Rule:** cache only pure functions of shape and dtype, or invalidate on the weight.

---

## 7. Serialization and Deserialization

### 7.1 The Round-Trip Test, on VALUES

```python
def test_serialization_cycle_value_identity(self, sample_input):
    inputs = keras.Input(shape=sample_input.shape[1:])
    outputs = MyLayer(**config)(inputs)
    model = keras.Model(inputs, outputs)

    original = ops.convert_to_numpy(model(sample_input, training=False))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layer.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = ops.convert_to_numpy(loaded(sample_input, training=False))

    np.testing.assert_allclose(
        original, restored, atol=1e-6, rtol=0,
        err_msg="MyLayer values differ after a .keras round trip",
    )
```

Three details are load-bearing:

| Detail | Why |
|---|---|
| `rtol=0` | `assert_allclose`'s default `rtol=1e-7` silently contributes to a nominally-`atol` bound. **Measured:** it contributed `1.24e-05` of a `1.53e-05` failure, making the stated `atol` decorative |
| `training=False`, explicit | a bare `model(x)` is not inference; stochastic-depth layers short-circuit only on `training is False` |
| Values, never shapes | a shape-only round trip is satisfied by a model that restored **zero** weights |

### 7.2 Registry Key Collisions

See §2.2. Explicit `package=`, prefixed generic names, `get_registered_name` for `custom_objects`.

### 7.3 Public Methods That Bypass Lazy Build

**Measured:** a model's `encode_image` / `encode_text` did not route through `__call__`, so on a
freshly constructed model a `build()`-created variable was still `None` and the method died inside a
broadcast with a message that never mentioned the model being unbuilt.

**Rule:** every public method reading a `build()`-created variable calls an `_ensure_built()` that
resolves shapes from the constructor config. Every `train_step` / `test_step` / `evaluate` override
has at least one test that actually executes it — one such override used a Keras 2 API that does not
exist in Keras 3, was reachable by any `fit()`, and sat behind a suite that was forward-pass and
save/reload only.

### 7.4 Output Structures That Break `predict`

`predict({"input_ids": ...})` has raised `Structures don't have the same nested structure` because
`call` echoed a bare `None` mask back in its output dict.

**Rule:** fix this at a **single site with one rule for all models**.

**Measured:** resolving the mask earlier is not a no-op for every architecture — exactly `0.0` for
one model, `6.4e-01` for a sibling on an output whose max magnitude was 2.67, because a windowed
attention zero-pads a rank-2 mask up to its synthetic grid. A per-model placement rule is its own
trap.

---

## 8. Build Materialization and Weight Compatibility

### 8.1 The Rule

> **`build()` must materialize precisely the sub-layer tree that `call()` runs — no more, no less.**

Both directions are real defects.

```
 UNDER-BUILD                              OVER-BUILD
 build() creates only own scalars         build() builds a sub-layer call() skips
        ↓                                        ↓
 build_from_config calls self.build()      weights exist that the lazy path
 inside a bare try/except: pass            never makes
        ↓                                        ↓
 load_model returns a model whose          checkpoint layout silently changes
 sub-layers are STILL UNBUILT              — a break dressed as a fix
        ↓
 first forward builds them FRESH + RANDOM
 nothing raises
```

**Measured (under-build):** a reloaded model with `len(model.weights) == 0`; another matching
**0 of 16** weights against its donor.

### 8.2 Two Clarifications That Were Initially Got Wrong

| Claim | Correction |
|---|---|
| "Overriding `build()` is the hazard" | It is not. The discriminating property is whether `build()` **materializes the tree**. A model that overrides `build()` and ends it with a concrete dummy forward pass round-trips cleanly. One that overrides `build()` to create two scalars does not |
| "`Model.build(shape)` builds a subclassed model" | It only marks it built and walks no sub-layers, so `count_params()` returns exactly **0**. Several widely-copied packages do this. It is not a working precedent |

### 8.3 Enforcement

Two tests, because neither alone is sufficient:

```python
def _relative(model):
    """Weight paths with the model-root segment stripped, so two INSTANCES compare equal."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)

def test_explicit_build_matches_lazy_build():
    explicit = build_model(); explicit.build((None, 32))
    lazy = build_model();     lazy(np.zeros((1, 32), "float32"))
    assert _relative(explicit) == _relative(lazy)

def test_no_head_config_builds_no_head_weights():
    """Anti-vacuity sibling: the parity guard above would pass if BOTH built everything."""
    m = build_model(head=None); m(np.zeros((1, 32), "float32"))
    assert not [w for w in m.weights if "head" in w.path]
```

> **Parity requires every sub-layer to carry an explicit `name=`** (§3.2), and both models built by
> the same builder. Keras auto-increments generated names per instance, so two separately-constructed
> models produce `block/w` versus `block_1/w` at *every* unnamed level, and stripping only the root
> does not normalize that away. A parity failure is a naming problem before it is a build problem.

**Parity is blind to over-building** — it passes if *both* paths build everything. That is what the
second test is for.

### 8.4 Weight-Value Comparison, Before the First Call

```python
def test_save_load_restores_the_actual_weights(self, tmp_path):
    model = build_model()
    model(sample_input)                       # build the donor
    saved = [ops.convert_to_numpy(w).copy() for w in model.featurizer.weights]
    assert saved, "the featurizer has no weights to compare"   # anti-vacuity

    path = str(tmp_path / "m.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    # BEFORE any forward pass on `loaded`: after one, a build()-only load path
    # reads the same weight COUNT for the correct and the broken variant, because
    # the gap has been filled with fresh random weights.
    for s, r in zip(saved, loaded.featurizer.weights):
        np.testing.assert_allclose(
            s, ops.convert_to_numpy(r), atol=0.0,
            err_msg="featurizer weights were not restored",
        )
```

`atol=0.0` is correct: restoration is a copy, not a computation.

A weight **count** invariant is blind to an internal-dimension change that reshapes without adding or
removing tensors. Assert the scalar parameter total too.

---

## 9. Factory Patterns and Layer Reuse

### 9.1 Reuse Order

**Writing a bespoke layer is the last resort, not the first move.**

| # | Check |
|---|---|
| 1 | **A config-driven factory for the domain**, if one exists — normalization, attention, feed-forward, activations, embeddings, pooling. Pass a type string plus config |
| 2 | **The existing layer surface.** Search before writing |
| 3 | **Only then a new layer**, placed in the domain package alongside its siblings and registered in that domain's registry. Do not bury a general-purpose layer inside one model's directory |

### 9.2 A Registry-Backed Factory MUST Raise on Undeclared Keys

The tempting implementation filters the caller's keyword arguments against the target type's accepted
parameters and **drops the rest**. A misspelled or undeclared key then produces a valid layer carrying
a default value, and nothing raises, warns or logs.

**Measured consequences of exactly this design:**

| Defect | Effect |
|---|---|
| Four call sites spelled `dropout=` where the registry declared `dropout_rate` | positional dropout **dead across every vision encoder that used them** — `Dropout(rate=0.0)` regardless of what the caller passed |
| One model forwarded `max_seq_len` / `rope_theta` into an attention type declaring no rotary parameter | both keys evaporated; the reasoning stack was exactly permutation-equivariant |
| `qkv_bias=True` where the declared spelling was `use_bias` | layer built with **zero bias weights** |
| Hardening one factory | immediately exposed four more live sites silently discarding a normalization choice on every construction |

**✅ CORRECT — the reference shape:**

```python
REGISTRY: Dict[str, Dict[str, Any]] = {
    "multi_head": {
        "class": MultiHeadAttention,
        "required_params": ["dim", "num_heads"],
        "optional_params": {"dropout_rate": 0.0, "use_bias": True},
    },
}

def create_attention_layer(attention_type: str, **kwargs: Any) -> keras.layers.Layer:
    if attention_type not in REGISTRY:
        raise ValueError(
            f"Unknown attention type {attention_type!r}. Available: {sorted(REGISTRY)}"
        )
    entry = REGISTRY[attention_type]
    declared = set(entry["required_params"]) | set(entry["optional_params"])

    unsupported = sorted(set(kwargs) - declared)
    if unsupported:
        raise ValueError(                       # NEVER filter-and-drop
            f"create_attention_layer({attention_type!r}): "
            f"{len(unsupported)} unsupported parameter(s) {unsupported}. "
            f"Accepted: {sorted(declared)}"
        )
    missing = sorted(set(entry["required_params"]) - set(kwargs))
    if missing:
        raise ValueError(f"{attention_type!r} requires {missing}")

    params = {**entry["optional_params"], **kwargs}
    return entry["class"](**params)
```

**Two design notes**

- **Do not "validate" by relying on the layer constructor to reject the key.** That works only for
  layers whose `__init__` does not accept `**kwargs`, and the error message is about the base class
  rather than the factory.
- A flat allowlist of parameter *names* applied uniformly to every type is weaker than per-type
  schemas: a parameter gets range-checked because of what it is *called*, and a constraint differing
  between two types cannot be expressed at all.

### 9.3 A Registry's Key Set Is Public Surface

Once callers pass type strings from configuration files, the key set, the type aliases and each
entry's declared parameters are **API**. Adding, renaming or removing one is a breaking change, not a
cleanup. Say so in the module docstring and pin it with a drift test.

Some entries may deliberately map to module-level **functions** rather than classes, where the
function pins a mode the class does not encode. A configuration reachable by passing an argument to
the general class should deliberately **not** be registered — adding keys "for consistency" grows the
frozen surface for nothing.

### 9.4 The Inverse Defect — a Hand-Written Kwarg List That Omits a Key

Hardening the factory does not catch the shape where the call site hand-writes its argument list and
simply **omits** a key it holds in `self`.

**Measured:** nine such sites at exactly `0.000000e+00` weight delta — attention projections that
never received `kernel_initializer` or `use_bias`, patch embeddings that never received initializers
or regularizers, a final normalization that never received `epsilon`.

**Rule:** when you audit "who calls factory X", also sweep **"who builds X's argument dict without
calling X directly."** A file assembling an `ffn_args` dict and passing it to a wrapper is invisible
to an AST call inventory, and a suite sweep run at site defaults cannot see the break either, because
it needs a non-empty caller dict to appear. A `**kwargs`-splat site has the identical blind spot.

### 9.5 A Derived "Optional" Parameter Is Not Safe to Pass

Consult the registry entry before deciding.

**Measured:** one feed-forward type derived its hidden dimension from a two-thirds rule plus a
multiple-of constraint while thirteen others required it. Making the parameter conditional turned an
expansion-factor knob into a no-op for eight types — and the change shipped with a guard asserting
that invariance **as correct**, pinning the new defect.

**Rule:** forward the derived parameter registry-driven, and pin the layer's own derivation rather
than pinning invariance.

### 9.6 Normalization Epsilon

| Source | Default |
|---|---|
| Keras `LayerNormalization` / `BatchNormalization` | **1e-3** |
| Most transformer-family reference implementations | 1e-5 or 1e-6 |

A factor of 100 to 1000 in every denominator, with no shape symptom, no warning and no test failure.

**Measured:** direct construction put a normalization layer at `1e-3` inside a stack whose others ran
at `1e-6` — a 1000x spread inside one forward pass. A related port had the reference epsilon reach 1
of `2*num_layers+1` normalization layers; an earlier one had **86 of 114** layers silently wrong with
every test green.

**Rules**

- Route normalization through one factory so there is exactly one source for the value.
- Constructing directly makes `epsilon=` mandatory, with a cited reference.
- Do **not** blanket-fix — some architectures' references genuinely use `1e-3`.
- When you sweep, sweep **every** epsilon-owning sub-layer, not the one that was noticed.

### 9.7 Porting From a Reference Implementation

A port's failure mode is not a missing layer. It is a numerically different layer wearing the right
shape.

| Trap | Rule |
|---|---|
| Constructor defaults are not the reference's structure | audit every implicitly-defaulted numeric hyperparameter of every framework primitive the port touches — epsilon, momentum, activation constants. Fix additively (new kwarg, default unchanged) so existing consumers do not move |
| `padding='same'` is **asymmetric** in Keras, symmetric in PyTorch | at stride > 1 two branches of one block with different kernel sizes sample **different input pixels**. Measured with Dirac kernels: a `k=1` branch read one 2x2 patch while the `k=3` branch of the same block read a different one. Shape assertions cannot see it. Apply a symmetric padding mode uniformly at every port site |
| A hand-transcribed oracle | vendor the reference config or source in the repository, off the import path, and read it with `ast` / `json` (§13.4.3) |
| A class sharing a name with a reference | is not necessarily that architecture. Check the composition rule, not the name |

---

## 10. Numerics and Precision

### 10.1 fp16 Mask Sentinels

`scores + (1 - mask) * (-1e9)` is the most replicated numerical defect in attention code.

```
 np.float16(-1e9)  ==  -inf          ← below float16's finite floor (~-6.55e4)
 0.0 * -inf        ==  nan
```

Under `mixed_float16`:

```
 fully-masked row        →  softmax → NaN
 UNMASKED position       →  0.0 * -inf = NaN     ← the corruption lands on the
                                                   positions the mask KEEPS
```

That second line is why a guard checking "the masked positions are ignored" misses this family
entirely.

**Rules**

- Derive the sentinel from `self.compute_dtype` — `np.float16(-1e4) == -10000.0`, finite.
- Better: express the mask as `ops.where(keep, scores, bias)` rather than an additive product, so
  there is no `0 * -inf` term at all.
- Every layer with a mask or a reduction gets a `mixed_float16` **and** a `float64`
  construction-and-forward test.

**Two follow-on traps**

| Trap | Consequence |
|---|---|
| **Sub-layers autocast** | a float32 tensor entering a sub-layer under `mixed_float16` is float16 inside that sub-layer's `call()`. A fix that only changes a dtype does not survive the sub-layer boundary — the fix must change the **predicate**. Equally, a claim about a sub-layer's dtype cannot stand in for a claim about the model's |
| **"Run this reduction in float32" is relative to the input dtype** | under a float64 policy the identical instruction **narrows** it. Measured: worst-case error `1.31e-15` → `1.99e-08`, every test still green. Use a never-narrow guard (`max(input_dtype, float32)`), not a hard-coded literal |

### 10.2 Degenerate Lengths Return NaN Instead of Raising

A reduction over a band or window whose length can be 0 or 1 fails silently and
**execution-mode-dependently**:

| Op | Eager, zero-length axis | Under `@tf.function` |
|---|---|---|
| `ops.min` / `ops.max` | raises | returns `±inf`, no raise |
| `ops.mean` / `ops.var` | — | returns `NaN` |

**Measured:** a model returned an **all-NaN forward pass at initialization** while its test asserted
only `output.shape == (4, 24, 7)` and was green throughout. A convolutional model whose downsampling
stages all use `padding='valid'` produced an all-NaN output **of the correct shape** whenever a stage
collapsed an axis to length zero — and its own shipped docstring example did exactly that.

**The static-shape guard does not close it:**

```python
# ❌ WRONG - short-circuits and never fires on a [None, None, C] trace
if dim is not None and dim < 2:
    raise ValueError(...)
```

`InputSpec` cannot close it either (§3.5), and `ops.cond` on a traced shape value raises
`OperatorNotAllowedInGraphError`.

**✅ CORRECT:** branch in Python on `tensor.shape[axis] is None` — a trace-time test on a Python
object — and repair at the **value** level in the dynamic branch. Validate the minimum spatial or
sequence extent in `__init__`, computed from the variant, never hard-coded.

Then check what your repair does to real NaNs: one value-level repair was rewriting genuine NaNs to
`0.0`, so a corrupt window looked like a constant one. Keep the repair off the static path where the
length is known good.

> **Every forward test asserts `ops.all(ops.isfinite(y))`, never just `y.shape`.**

### 10.3 Wrong Parameterisation, Sign, Direction or Scaling

All of the following passed shape-only suites:

| Defect | Measured |
|---|---|
| A "reflection" gradient mode returning `x - 2(x·w)w` — a Householder reflection mapping `u → -v` | forward output was **exactly `-q`**; the suite parametrized all four modes and asserted output shape, invariant under a sign flip |
| A codebook lookup rescaling its row by `x_mag / q_mag` | a "discrete" bottleneck leaked a continuous magnitude channel; `decode(encode(x)) != model(x)` in **2048 of 2048** elements |
| An EMA codebook as `ema_embeddings / (ema_cluster_size + eps)`, counts starting at zero, numerator at the initializer | step 1 gave roughly `99000 * init`. Debiasing alone made it **worse**; the fix needed zero-initialization too |
| A forecaster de-normalizing sigma with `sqrt(scale)` while mu used `scale` | the tell: the branch below used the correct form — the `sqrt` had been carried across by copy |
| A score field fed `denoised - noisy` as an epsilon estimate | cosine with the correct direction measured **-1.0** — gradient *descent* on log p. A bare sign flip would still be wrong: the variance-preserving parameterisation carries a factor the variance-exploding form drops |
| A decay schedule counted in batches against a counter in samples | **negative** learning rate for **9 of 10** batches; a negative rate moves every neuron *away* from its input, so the map anti-organised |

**Rule: pin the INVARIANT, not the shape.**

- homogeneity — scale the target by `k`, assert mu **and** sigma scale by `k`;
- sign-discriminating distance comparisons — distance-to-`+q` versus distance-to-`-q`, which cannot
  be satisfied by loosening a tolerance;
- agreement between the forward path and any two-stage public API (§10.4);
- a numerical central-difference check against the closed form.

> A reviewer can be right about the defect and wrong about the mechanism in the same sentence. Run
> the prescribed fix and diff the **number**, not just the shape, before believing it.

### 10.4 Two Producers of the Same Quantity

Where a layer has both a `call()` and a public two-stage API
(`encode_to_indices → quantize_from_indices → decode`), the two compute the same thing by different
code. One gets fixed; the other drifts.

**Measured:** the same invariant was violated twice in sibling branches of one file; in a third case
the disagreement was `1.92e-03` against a suite whose only bound was an `atol=1e-4` — twice as loose
as the defect.

**Rule:** ship a parametrized value-equality test across every mode, with a vacuity arm pinning that
the fixture is in a regime where the two *could* differ.

---

## 11. The Training Path

### 11.1 Custom `train_step`

> **Prefer not to write one at all.** Use stock `fit()` and feed extra signals through `tf.data`
> inputs. The override is what opts you out of the machinery below.

**Mixed-precision loss scaling.** Keras' default TF `train_step` calls `optimizer.scale_loss(loss)`
inside the tape, and `LossScaleOptimizer.apply()` divides every gradient by `dynamic_scale`
**unconditionally**. Overriding `train_step` opts out of the first and keeps the second.

**Measured** over 10 SGD steps, total `|dW|`:

| Configuration | total `\|dW\|` |
|---|---|
| `mixed_float16`, as shipped | 8.740626e-05 |
| `mixed_float16`, with `scale_loss` | 2.436617e+00 |
| float32 control | 2.739021e+00 |

A ratio of `2.79e+04`, which is `2^15`. Nothing raises and nothing warns; training simply does not
move.

**Rule:** any custom `train_step` calls `self.optimizer.scale_loss(loss)` inside the tape, sums
`self.losses` via `self.compute_loss`, and carries a `mixed_float16` A/B on total `|dW|` over N steps
against a float32 control. If the model applies gradients through raw optimizer attributes rather
than `self.optimizer`, `scale_loss` is an inert no-op — say so in a comment, or the next reader
re-adds it.

**Two clarifications that were initially got wrong**

| Claim | Correction |
|---|---|
| "A custom `train_step` drops regularizer terms" | Keras 3.8's default `compute_loss` **already sums `self.losses`**. The AST predicate "does the body mention `self.losses`" measures the wrong thing |
| — | A real instance *did* drop them: four overrides summed `self.quantizer.losses` **only**, so a caller's encoder `kernel_regularizer` reached neither the gradient nor the reported loss — identical loss to six digits with and without an `l2(1e-1)` |

`model.losses` is often non-empty for reasons unrelated to the regularizer under test — a block
hardcoding a layer-scale L2 is enough — so `assert model.losses` can always pass. **Assert a delta**
against a no-regularizer baseline.

### 11.2 Python State That Never Reaches the Traced Graph

```python
# ❌ WRONG - folds to False at trace time; apply_gradients is NEVER EMITTED
if self.accumulation_counter % self.accum_steps == 0:
    self.optimizer.apply_gradients(...)
```

A schedule that sets a Python attribute is never re-read by an already-traced train function.

**Rules**

- Any state that must vary across steps is a `keras.Variable` released by `ops.cond`.
- Any schedule changing a **shape-determining** value cannot be carried by a variable at all and
  requires an explicit retrace.
- Verify by asserting `optimizer.iterations` advances as expected — at accumulation 2 the sequence is
  `0, 1, 1, 2` — not by reading logs.

Related: Keras 3 spells it `reset_state`, not `reset_states`. And an EMA clone made from an
**unbuilt** subclassed model means `set_weights(model.get_weights())` is `set_weights([])`, which
silently succeeds.

### 11.3 A Zero Gradient Is Not a Freeze

| Mechanism | Effect on a "frozen" weight |
|---|---|
| AdamW moment estimates | keep drifting a gradient-masked parameter — **measured `6.8e-3`** over two masked steps at `lr=1e-2` |
| Decoupled weight decay | moves it by exactly `wd * lr` per step **with no gradient at all** |
| `model.trainable = False` | empties `trainable_variables` but leaves each `tf.Variable.trainable` at `True`, so a `GradientTape` still auto-watches them |

The only exact freeze also zeroes that group's learning-rate variable.

**Rules**

- Verify a freeze by **bit-identical weights across steps**, never by a zero gradient.
- Assert `trainable_weights == []` **and** an empty `tape.gradient(...)` — not
  `tape.watched_variables() == ()`.
- Marking previously-trainable weights non-trainable makes a `.keras` optimizer-state resume skip
  optimizer loading entirely, with only a `UserWarning`.

### 11.4 Optimizer and Callback Traps

| Trap | Detail |
|---|---|
| `optimizer.learning_rate` | is the schedule **evaluated at the current `iterations`**, not the schedule object (which lives on `_learning_rate`). Assert liveness by driving `iterations`, not by an `isinstance` check |
| Direction inferred from a metric name | `mode = 'max' if 'accuracy' in monitor else 'min'` silently selects the **worst** epoch for a metric like `val_box_iou`. Audit both the metric and the direction |
| `EarlyStopping(restore_best_weights=True)` + `ModelCheckpoint` on the same metric | make "best" and "final" the same epoch **by construction**, so their bit-identity proves nothing |
| A zero-initialized last projection | back-propagates **exactly zero** gradient into every weight behind it. **Measured: 7 of 9** upstream variables at exactly `0.0`. Normal at init for some architectures, and also how a dead stack looks |

### 11.5 `pretrained=True` Returning Random Weights

See §5.4. **Measured:** nine packages shipped it simultaneously, and the documented alternative was
broken too.

**Guard it by AST shape** — no `if pretrained:` branch may consist solely of logger calls — plus a
behavioural arm over auto-discovered factories. A string-matching guard ("no placeholder URL
appears") sees nothing when the sites have no URL table at all; that exact guard passed on all nine.

---

## 12. Causality, Masking and Composition

### 12.1 The Missing Causal Mask

**Symptom.** A model documented and trained as decoder-only attends bidirectionally.

**Mechanism.** `call` forwards only `attention_mask` — a **padding** mask, `None` by default — and
the attention layers mask only `if attention_mask is not None`. Under a next-token objective the
model has seen the answer.

**Measured:** found in several language models at once; a grep for `causal|triu|tril|j > i` across one
package returned **docstrings only**. A text tower attended bidirectionally and then pooled "the last
non-padding token, because it is the only one to have seen the whole sentence" — a statement true of
*every* position in a bidirectional tower.

**Why the obvious test is blind.** `test_attention_mask_functionality` asserts only that masked and
unmasked outputs differ. Any mask satisfies that. Loss curves look normal.

**Rule — a three-armed future-leak probe:**

```
 1. perturb token t;  assert positions <  t are BIT-IDENTICAL (exactly 0.0)
 2.                   assert positions >= t still move          ← anti-vacuity
 3. negative control: an all-attend mask, proving the isolation
                      is attributable to the mask
```

Arm 1 is exactly `0.0`, not "small": an attention weight of exactly zero on a masked key contributes
exactly nothing. Without arm 2, `0.0` proves only that the model ignores its input.

**Pass the causal mask at rank 3.** A rank-2 causal mask is silently reinterpreted as a padding mask
by grouped-query attention.

### 12.2 Pooling a Causally Isolated Position

Several classifiers pooled token 0 of a causal model — a token that has seen nothing but itself. The
mirror-image error is pooling the last token, justified by a causality that did not exist.

**Rule:** pooling strategy and attention causality are **one decision**. Assert the pooled
representation depends on more than one input token: perturb an interior token, assert the pooled
vector moves.

### 12.3 Masking by Zeroing Both Sides

Masking a metric by zeroing both the prediction and the target makes `zero == zero` always agree.

**Measured:** in one file the sequence-level metric was sound while the per-step metrics understated
error **3x** from this idiom.

**Rule:** mask by **excluding positions from the reduction**, not by zeroing both sides of a
comparison. Check what the reduction does with a masked position.

### 12.4 Repair Granularity

A degenerate-row rescue must operate over the **full axis the softmax reduces over**.

**Measured:** a per-tile rescue inside an online-softmax loop read every strictly-upper causal tile as
degenerate and **un-masked the future** — a 24.14 deviation — while every finiteness test passed.

Softmax is invariant to a constant shift along its reduction axis, so a large delta after a masking
change can be uniform garbage on both sides. Add a control proving the pre-change output was itself
meaningful before calling a delta a regression.

### 12.5 Inert Configuration — the Dead Knob

**Symptom.** A parameter is validated, stored, serialized and documented, and changing it changes
nothing. The constructed layer is valid; it is just not the one requested.

**Mechanism.** One of:

| Shape | What happens |
|---|---|
| The knob is never forwarded to the sub-layer that would consume it | the sub-layer uses its own default |
| It is read only at build time, but the branch is hardcoded | the value is inspected and discarded |
| A sibling consumes it and ignores it | no error anywhere |
| It mutates a Python attribute an already-traced function never re-reads | correct in eager, inert under `fit()` (§11.2) |

**Measured instances:**

| Knob | Result |
|---|---|
| A normalization-position flag on a model whose encoder block had no such parameter at all | `'pre'` built a post-norm stack |
| A KV-head count printed by `summary()` | the attention was plain multi-head |
| An `arch_type` argument accepting three values that no branch consulted | all three gave 612 parameters and `max\|dy\| = 0.000e+00` |
| A weight-sharing flag | identical parameter counts *and* identical object counts both ways |
| A reconstruction-weight documented as a penalty | `model.losses` was `[]` |

**Why the obvious test is blind.** The only assertion was a constructor-attribute echo
(`assert model.d_state == d_state`) or a shape check. Both are invariant under the defect, and
`get_config` round-trips perfectly — because the value *is* stored.

**Rule: every constructor parameter is pinned by a test that varies it and asserts a measured
difference in weights or outputs**, with an anti-vacuity control. Reading the value back off `self`
is not coverage. If a knob is deliberately inert, delete it, or pin the inertness with
`xfail(strict=True)` carrying the measurement and the reason.

Choose the instrument by knob class (§13.3.2) — this is the single most common way a knob test goes
vacuous.

### 12.6 Inert Components

The unifying property: the component exists, has weights, has gradients, serializes — and contributes
nothing. Parameter counts, shapes, finiteness and gradient-existence assertions are all blind by
construction.

**Positional encoding on the wrong axis.** `RotaryPositionEmbedding.call` reads the sequence length
from `ops.shape(inputs)[2]` — it expects `(batch, seq, heads, dim)`. Layers handing it
`(batch, heads, seq, dim)` rotated every token by its **head index**.

```
 R_h is orthogonal and is applied to BOTH q and k of the same head
        →  (R_h q)·(R_h k) = q·k
        →  RoPE was an EXACT NO-OP, not a corruption
```

Shapes, parameter counts, gradients and serialization were all correct. The cancellation only breaks
under grouped-query attention, where a key head is rotated then repeated onto a query head with a
different index — at which point a silent no-op becomes real score corruption.

A second layer in the same family passed a tensor with a singleton head axis, read as sequence length
1, so everything was rotated by position 0 — the identity.

**Related shapes.** A model defaulting to rotary encodings constructed the layer, built it,
serialized it, and never handed it a query. An FFT mixer transformed the **innermost** axis, which is
the feature axis, so the only token-mixing operator in the architecture did no token mixing at all.

**Rule:** any layer claiming positional or token-mixing semantics carries a **non-cyclic
permutation-equivariance probe** — permute the input tokens, assert the output moves by far more than
float32 noise, with an anti-vacuity arm asserting the logits vary across positions in the first place.
Never infer "RoPE is wired" from the existence of a `self.rope` attribute. Assert the axis order
explicitly at the call site.

**Components built and skipped.** Register tokens that were R copies of one vector rather than R
independent ones. A grouped state summed over its group axis, making four groups bit-equal to one.
Deep supervision that supervised the head it already had. Each is a component whose *count* is right
and whose *identity* is wrong.

**Rule:** assert the identity, not the count. Two register tokens must differ from each other; two
groups must produce different outputs.

### 12.7 Composition Failures

Architectures whose entire value is *how blocks compose* are the ones where composition is never
tested, because shape, parameter count, finiteness and gradient existence are all invariant under a
broken composition rule.

| Defect | Measured |
|---|---|
| **Transform-only blocks called without the external residual.** Some blocks compute a *transform* and document that the *caller* supplies the skip. Calling them as `x = block(x)` drops it | signal collapse of roughly `1e-5` per block; a layer-scale init of 1.0 did **not** rescue it |
| **A residual block whose `gamma → 0` limit is zero, not identity.** A learnable multiplier applied to the whole block output *after* the block closed its own residuals gives `x = gamma * f(x)` with no skip | `std(out)/std(in) = 4.97e-05`, restored to `1.0000` by the fix |
| **Sub-blocks sharing an input.** A fractal block applied both depth-`k-1` sub-blocks to the *same* input: `F_k(x) = 0.5*(DP(F_{k-1}(x)) + DP(F_{k-1}(x)))` | every input-to-output path traversed exactly **one** convolution at any depth; `depths=[4,5,5]` was 8/16/16 parallel convolutions instead of a fractal. Twenty tests passed against the broken rule |
| **A stack that reads only the last block.** A graph transformer computed its local tokens once and handed the same tensor to every block | increasing the block count deepened nothing. The code said so, in a comment and in the module docstring, as though it were a permanent property |

**Rules**

- Read the block's docstring for **who owns the residual**.
- Assert a post-ladder magnitude: `std(out) / std(in)` must stay near 1 across the stack.
- For a block with a residual scale, assert the limit directly — as the scale goes to zero the block
  must approach the **identity**, not zero. Two-armed: `gamma=0` is exactly the identity, and `gamma`
  at its shipped init is measurably *not*, because an identity-only assertion is also satisfied by a
  block that returns its input.
- For any architecture whose value is composition, assert composition **directly**: receptive-field
  growth, or a **non-local** probe (perturb an interior input pixel, require the spatially opposite
  corner of the final feature map to move). Use scale-free assertions — a downstream renormalizing
  layer defeats magnitude checks.
- Treat a comment explaining why the architecture does not do what its name says as a **defect
  report**, not as documentation.

---

## 13. Testing and Validation

Every defect in §1–§12 shipped behind a green suite. This section is how to write a suite that would
have caught them.

> **The governing observation, measured repeatedly: a guard that cannot fail is the most likely
> outcome of writing a new test, not an edge case.** One freshly written 140-test suite contained 12
> vacuous tests, including *every* forward-pass test and *both* gradient-flow tests. Another probe
> suite measured **57 of 57** guards blind. Budget the work to prove your guard can fail.

### 13.1 The Five House Rules

| # | Rule |
|---|---|
| 1 | **Every number is measured or derived, and the derivation is written down.** A tolerance carries its measurement — date, device, dtype policy, configuration, the number — and the defect signal it must sit below |
| 2 | **Every guard is proven RED**, by an injected mutation or a recorded bisect, *in the committed record*. A demonstration in a scratch session that is then discarded leaves nothing a later reader can re-check |
| 3 | **Every "nothing changed" assertion has a "something changed" twin.** An identity assertion alone is satisfied by a completely dead component |
| 4 | **Process-global state is owned by exactly one fixture** that captures it, restores it in `finally`, and asserts the restoration — plus a canary that fails the *next* test if it leaked |
| 5 | **A failing guard is never repaired by widening the tolerance**, unless the bound is proven *unattainable in the output dtype*. Unattainable → re-derive. Attainable but flaky → fix the cause. Write the distinction into the test |

### 13.2 The Instruments

#### 13.2.1 `.keras` round trip on values

See §7.1.

#### 13.2.2 Weight-value comparison before the first call

See §8.4.

#### 13.2.3 Build parity by relative weight path

See §8.3.

#### 13.2.4 Build through a parent's `call()`

The only probe that sees the `StatelessScope` trap (§3.3).

```python
class _Parent(keras.layers.Layer):
    """Minimal parent whose call() is the only path that builds the child."""
    def __init__(self, child, **kwargs):
        super().__init__(**kwargs)
        self.child = child
    def call(self, inputs):
        return self.child(inputs)


def _build_through_parent(child, input_shape, dtype="float32"):
    parent = _Parent(child)
    parent(keras.Input(shape=input_shape[1:], dtype=dtype))
    return child


def test_the_table_survives_the_stateless_build_pass():
    child = _build_through_parent(MyEmbedding(dim=16), (None, 8, 16))
    omega = ops.convert_to_numpy(child.omega)
    # Pin a CLOSED FORM, at a DISCRIMINATING entry. omega[0] == 1.0 distinguishes a
    # live table from an all-zero one; a `sin` table is legitimately 0 at position 0
    # and must be pinned at a later position instead.
    np.testing.assert_allclose(omega[0], 1.0, atol=0.0, rtol=0)
```

#### 13.2.5 Gradient flow, per variable

```python
def test_gradients_flow_to_every_trainable_weight(self, sample_input):
    layer = MyLayer(**config)
    with tf.GradientTape() as tape:
        loss = ops.mean(ops.square(layer(sample_input, training=True)))
    grads = tape.gradient(loss, layer.trainable_variables)

    assert len(layer.trainable_variables) > 0            # anti-vacuity
    for var, grad in zip(layer.trainable_variables, grads):
        assert grad is not None, f"no gradient for {var.path}"
        assert np.any(ops.convert_to_numpy(grad) != 0.0), f"all-zero gradient for {var.path}"
```

Non-`None` **and** non-zero, named by `var.path`.

**Measured:** a guard written as `assert all(norm >= 0.0)` reported green while **61 of 61** trainable
weights had identically-zero gradients.

#### 13.2.6 Scoped weight probes for a knob

To prove a knob reached the one subtree it is meant to reach, compare the **weight values of a named
subtree** — not a whole-model output diff, which passes on the broken tree whenever the same knob
reaches other sub-layers by a second route.

```python
def weights_in_scope(model, scope: str):
    return [w for w in model.weights if scope in w.path]
```

#### 13.2.7 Both-ways pairs

Every "nothing moved" assertion is half of a pair.

```python
class TestAttentionMaskIsHonoured:
    def test_masked_tokens_do_not_reach_the_visible_positions(self, perturbation):
        delta = float(np.max(np.abs(hidden(ids)[:, :KEEP] - hidden(bumped)[:, :KEEP])))
        # Bit-identical, not "small": an attention weight of exactly zero on a masked
        # key contributes exactly nothing.
        assert delta == 0.0

    def test_the_perturbation_reached_the_model_at_all(self, perturbation):
        """The other half of the pair: without this, 0.0 above proves nothing."""
        delta = float(np.max(np.abs(hidden(ids)[:, KEEP:] - hidden(bumped)[:, KEEP:])))
        # A model that ignored its input would score 0.0 here and would still pass
        # the assertion above.
        assert delta > 1e-3
```

**Two probe-design rules learned the hard way**

| Rule | Measured |
|---|---|
| **Never perturb with a DC or uniform per-channel signal** when a per-position normalization precedes the guarded reduction | one such probe measured a leak of `1.9e-06` against a real leak of 0.33–1.07. Use fresh non-DC noise |
| **Watch for configurations that make the mechanism structurally unobservable** | a single-layer text tower reads its last position, whose causal row is unmasked, so the pin reads exactly `0.0` with and without the mask. A deep tower at small input resolution can collapse its deepest attention stage to **one token**, where softmax is identically 1.0. Cheap detector: `pytest -W error::UserWarning` turns Keras' size-1-softmax warning into a failure |

#### 13.2.8 Orientation — delta impulses and non-square grids

Orientation and direction are invisible to shape, config and serialization tests.

**Measured:** a single sign error in one `ops.roll` survived 249 tests; a fully transposed
relative-position bias (`bias[h, key, query]`) passed 219; a shifted CLS slice passed **91 of 91**.

**Rule:** use a **delta-impulse probe** — a one-hot input, asserting the destination slot — on a
**non-square grid**. A square-only test cannot see a transposed stride.

#### 13.2.9 Precision arms

```python
DTYPE_POLICIES = ("float32", "mixed_float16", "float64")

@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)
```

For an fp16 arm to mean anything it needs four parts:

| # | Part |
|---|---|
| 1 | **Prove the hazard is real first** — assert `np.float16(-1e9)` really is `-inf` before testing any defence against it |
| 2 | **Realistic sizes.** `N = 512`, not `N = 7`. An `N=7` test hid an fp16 `-inf` that appeared only at `N >= 512` |
| 3 | **A float32 or float64 control on the same input**, so "fp16 is noisy" can never masquerade as "the NaN bug is detected" |
| 4 | **The repair must not weaken what it repairs.** `-1e4` is finite in fp16 but is not `-inf` — assert the masked positions still receive no weight. This arm separates "the sentinel was made survivable" from "the sentinel was made ineffective" |

Build the reference from the **round-tripped bits**, not the float32 original, so there is no fp16
rounding slack:

```python
def _as_compute(x):
    cd = keras.mixed_precision.global_policy().compute_dtype
    x_c = x.astype(cd)
    return ops.convert_to_tensor(x_c), x_c.astype("float32")
```

A **float64** arm needs more than the policy: `keras.Input` still uses `backend.floatx()`, so the
graph rounds to float32 at the boundary. Also call `keras.backend.set_floatx("float64")` and **assert
`inputs[0].dtype`** — otherwise the arm is a fake reading that agrees with float32 to eight digits.
Note `UpSampling2D(interpolation='bilinear')` returns float32 for float64 input.

#### 13.2.10 Graph and XLA equivalence

An eager-only fix is not a fix.

```python
def test_the_rescue_is_graph_safe_under_tf_function_and_jit(self, dtype_policy):
    """The rescue must contain NO data-dependent Python branch."""
    eager = ops.convert_to_numpy(apply_mask(logits, keep))

    @tf.function(jit_compile=True)
    def traced(x, k):
        return apply_mask(x, k)

    compiled = ops.convert_to_numpy(traced(logits, keep)).astype("float32")
    assert np.all(np.isfinite(compiled))
    np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)
```

For exact-integer paths use `np.array_equal` and an `input_signature` with `None` dimensions, so that
a value which must stay a static Python `int` is proven to stay one:

```python
@tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
def traced(x):
    return patcher(x)

assert np.array_equal(eager, graph)
```

Where XLA reassociates, the tolerance is **measured**, not guessed, and recorded with its output
magnitude — for example: *"measured 0.0151 against an output absmax of ~5.9, i.e. 0.25% relative;
0.05 keeps ~3x headroom while still failing loudly on a NaN or a collapsed output."*

#### 13.2.11 Derived tolerances

Where a bound must come from a noise source rather than a measurement, derive it and write the
derivation in the docstring:

```python
_F32_U = np.finfo(np.float32).eps / 2.0    # unit roundoff
_TAIL_FACTOR = 8.0                          # 8-sigma tail on a random-walk model

def reassociation_atol(reduction_lengths, num_steps: int, scale: float) -> float:
    """Bound on the float32 difference between two REASSOCIATED evaluations of one formula."""
    ops_count = 2.0 * num_steps * float(sum(reduction_lengths))
    return _TAIL_FACTOR * np.sqrt(ops_count) * _F32_U * max(1.0, float(scale))
```

Such a helper's docstring carries: the derivation, a calibration table (derived vs measured vs ratio),
a **RED proof** (injecting the real defect puts the diff orders of magnitude above the bound), and the
instruction that callers must pass `rtol=0`.

> A tolerance floor is a claim about a **specific noise source**. Reusing a TF32-derived floor on a
> matmul-free path once dominated the real term by 3 to 12 orders of magnitude and passed a
> projection that was systematically 1% wrong.

#### 13.2.12 Homogeneity and scale invariance

```python
HOMOGENEITY_RTOL = 1e-5
HOMOGENEITY_SCALES = (0.5, 3.0)   # never 1.0 (trivial) and never 2.0

def homogeneity_error(model, x, c) -> float:
    f  = ops.convert_to_numpy(model(x,     training=False))
    fc = ops.convert_to_numpy(model(c * x, training=False))
    denom = float(np.abs(c * f).max())
    if denom == 0.0:
        return float("inf")      # a DEAD model must not read as perfectly homogeneous
    return float(np.abs(fc - c * f).max() / denom)
```

Three things this encodes:

| Element | Why |
|---|---|
| The **dead-model guard** (`denom == 0 → inf`) | without it, a model that outputs zeros scores a perfect 0.0 |
| **Fit the model for one step first** | stock `BatchNormalization`'s `moving_mean` is exactly 0 at init, so an untrained model is *exactly* homogeneous at `training=False` and the test passes for a reason unrelated to what it claims |
| **The tolerance sits between the floor and the defect** | `1e-5` is ~9x above the float32 floor and ~680x below the measured defect signal of `6.8e-03`. Keep models small and fixed: a deeper graph accumulates more round-off, and the headroom is what makes the tolerance meaningful |

Note that layer normalization **breaks** degree-1 homogeneity (81–98% error) while a bias-free batch
norm gives it exactly. A structural "are all the biases zero?" check gives false assurance here —
homogeneity is a property of the normalization too, not only of the biases.

### 13.3 Three Reusable Oracles

Write these once as shared modules and call them from every suite. Name the module **without** a
`test_` prefix so the runner does not collect it, and give each its own RED-proof test module.

#### 13.3.1 The smoke contract, and the meta-test that proves it can reject

A smoke test asserting a model builds and produces a sane output is nearly worthless alone — it
passes on almost any broken model. What makes it real is the **meta-test**: mutate the model's own
forward output and require the contract to reject it.

```python
import contextlib
from unittest import mock

def collapse_to_scalar(output):   return ops.mean(output)
def slice_leading_axis(output):   return output[:1]
def append_trailing_axis(output): return ops.expand_dims(output, -1)

DEFAULT_BREAKERS = (collapse_to_scalar, slice_leading_axis, append_trailing_axis)

@contextlib.contextmanager
def broken_forward(model, breaker):
    original_call = model.call
    def _broken_call(*a, **kw):
        return breaker(original_call(*a, **kw))
    with mock.patch.object(model, "call", _broken_call):
        yield


def assert_contract_rejects_a_broken_forward(model, inputs, contract, breakers=DEFAULT_BREAKERS):
    # Anti-vacuity control FIRST: the contract must pass on the unbroken model,
    # or every rejection below is meaningless.
    contract(model(inputs, training=False))

    for breaker in breakers:
        with broken_forward(model, breaker):
            try:
                contract(model(inputs, training=False))
            except AssertionError:
                continue                      # the contract JUDGED - correct
            except Exception as exc:
                # A TypeError from a contract that indexed a scalar is the contract
                # CRASHING, not judging. Never accept a bare Exception here.
                raise AssertionError(
                    f"{breaker.__name__}: contract raised {type(exc).__name__}, "
                    f"not AssertionError -- it crashed rather than judged"
                ) from exc
            raise AssertionError(f"{breaker.__name__}: contract accepted a broken forward")
```

Used as a mandatory pair:

```python
def test_smoke_build_and_forward(model):
    _assert_contract(model(_inputs(), training=False))

def test_the_contract_rejects_a_broken_forward(model):
    assert_contract_rejects_a_broken_forward(model, _inputs(), _assert_contract)
```

> **A meta-test must break the MODEL, not its argument validation.** One earlier attempt passed an
> invalid variant name, which fails at variant lookup *before a model exists* — the meta-test passed
> while proving nothing.

#### 13.3.2 Knob sensitivity, instrument matched to knob class

Choosing the wrong instrument here is the most common way a knob test goes vacuous.

| Knob class | Example | What must change |
|---|---|---|
| **structural** | depth, heads, filters | the **weight-shape signature** |
| **value** | activation, epsilon | outputs, signature **identical** under the same seed |
| **scoped value** | an initializer honoured in part of the tree | weight **values** of a named subtree |

```python
def weight_signature(model):
    """Shape signature. Capture AFTER a forward pass: before one, a subclassed
    model has no weights and every config yields the same empty signature."""
    sig = tuple(tuple(w.shape) for w in model.weights)
    assert sig, "empty weight signature -- the model was not built before capture"
    return sig


def build_seeded(build_fn, seed=0):
    keras.utils.set_random_seed(seed)
    return build_fn()


def assert_structural_knob_changes_weights(builders, x):
    """Structural knobs are pinned on SHAPES, never on outputs."""
    sigs = {}
    for key, build_fn in builders.items():
        m = build_seeded(build_fn); m(x)
        sigs[key] = weight_signature(m)
    assert len(set(sigs.values())) == len(sigs), \
        f"structural knob did not change the weight shapes: {sigs}"


def assert_value_knob_changes_output(builders, x):
    """Value knobs must move the OUTPUT while leaving the shape signature identical."""
    outs, sigs = {}, {}
    for key, build_fn in builders.items():
        m = build_seeded(build_fn)
        outs[key] = ops.convert_to_numpy(m(x, training=False))
        sigs[key] = weight_signature(m)
    assert len(set(sigs.values())) == 1, "a value knob must not change the weight shapes"
    keys = list(outs)
    for a, b in zip(keys, keys[1:]):
        assert not np.allclose(outs[a], outs[b]), f"value knob {a!r} vs {b!r} changed nothing"
```

> **The trap the shape assertion exists to avoid.** Two models built with different `depth` values
> have different weight shapes, so they consume different draws from the RNG and their outputs differ
> **whether or not the argument was honoured**. An output-difference assertion on a structural knob is
> satisfied by random-init luck alone — a second unfalsifiable test wearing a stronger-looking
> assertion.

The closure gotcha, which silently makes every builder identical:

```python
# ❌ WRONG - a bare closure over the loop variable captures the LAST value for every entry
builders = {a: lambda: create_model(activation=a) for a in ("relu", "gelu")}
# ✅ CORRECT
builders = {a: lambda a=a: create_model(activation=a) for a in ("relu", "gelu")}
```

When a knob measures inert and the fix is deliberately out of scope, pin it with
`@pytest.mark.xfail(strict=True, reason="<measured>: ...")`. It XPASSes loudly when someone fixes it.
A plain `skip` is inert; deleting the test leaves the gap unguarded.

#### 13.3.3 Dead-component detection

```python
NO_GRADIENTS_MESSAGE = "No gradients provided for any variable"


@contextlib.contextmanager
def outputs_stop_gradient(model):
    """Cut every output off the tape. A LIVE training path must then raise.

    `train_function` is reset on BOTH edges. Keras caches the traced train step,
    so a model that has already been fitted keeps running the UNPATCHED graph and
    the injection silently does nothing -- the probe reports green against a model
    it never actually broke. Measured: on a fresh model this raises; on a
    pre-fitted one it does not, until the cache is cleared.
    """
    original_call = model.call
    def _cut(*a, **kw):
        return keras.tree.map_structure(ops.stop_gradient, original_call(*a, **kw))
    with mock.patch.object(model, "call", _cut):
        model.train_function = None
        try:
            yield
        finally:
            model.train_function = None


def fit_one_step_moved_variables(model, x, y):
    """Return the NAME SET of variables that moved -- never a bare count.

    `moved > 0` is not an acceptable assertion: it was once satisfied by a
    118-of-137 result whose 19-variable residual was never identified.
    """
    before = {v.path: ops.convert_to_numpy(v).copy() for v in model.trainable_variables}
    model.fit(x, y, epochs=1, verbose=0)
    return {
        v.path for v in model.trainable_variables
        if not np.array_equal(before[v.path], ops.convert_to_numpy(v))
    }


@contextlib.contextmanager
def layer_returns_its_input(layer):
    """Kill one component by making it the identity."""
    with mock.patch.object(layer, "call", lambda x, *a, **kw: x):
        yield
```

Used together:

```python
def test_every_component_is_live(model, x, y):
    moved = fit_one_step_moved_variables(model, x, y)
    expected = {v.path for v in model.trainable_variables}
    assert moved == expected, f"never moved: {sorted(expected - moved)}"

def test_the_probe_can_detect_a_dead_training_path(model, x, y):
    """RED proof: with the outputs cut off the tape, fit MUST raise."""
    with outputs_stop_gradient(model):
        with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
            model.fit(x, y, epochs=1, verbose=0)
```

Match the message **verbatim**, never `pytest.raises(Exception)`.

> **Ordering hazard in that pair:** `fit_one_step_moved_variables` fits the model, which caches a
> traced train step. Any injection applied afterwards must invalidate that cache or it patches code
> the training loop no longer runs. Same defect class as §11.2 — Python-side state that never reaches
> the traced graph — arriving in the *instrument* rather than the model.

> **The rule all three encode: every probe reports a NUMBER or a NAME SET, never a bare boolean
> verdict alone. A probe with no number is not a probe.**

### 13.4 Why Guards Fail

#### 13.4.1 Budget one mutation per assertion, and check which one fired

Two mutations can fire the **same** assertion, proving one twice and the other zero times. If that
happens, add an isolating mutation. A single guard can need two.

**Judge a RED proof by which assertion fired, by name and `file:line`.** A predicted RED *line* or
exception *type* is wrong more often than the failure *class* is — in one review, **4 of 8** predicted
REDs were right about the class and wrong about the line. Keras' unknown-kwarg check raises
`ValueError`, not `TypeError`. A test can also die at a setup assertion before reaching its point, and
"red for the wrong reason" reads exactly like a pass.

#### 13.4.2 An injection that moves both sides proves nothing

**Measured:** a mixed-precision injection **passed** because the float32 reference was captured at
import from the same source file the injection modified; a dtype-conditional variant of the same
injection fired **44x** over tolerance.

Likewise a default-versus-explicit-default comparison is the same code with the same weights and
cannot be moved by **any** injection.

**Rule:** compare against a **transcribed pre-change oracle** that bypasses the changed `call()`.

#### 13.4.3 An oracle written by the same hand is a second copy

**Measured:** five instances in a single porting round.

**The tell** is a constant, term or sign in the "oracle" that only makes sense if you had read the
implementation.

**Fix:** derive the oracle from the **reference**, and reach the implementation through explicitly
signed, named divergence terms (`_reference_params + _port_only_x - _reference_only_y`). The cheapest
form is to **vendor the reference file** in the repository, off the import path, parsed with `ast` or
`json`, and point the test at it.

Apply the identical suspicion to a **fix round's own new guard**.

#### 13.4.4 An oracle can be wrong before the implementation is

**Measured:** a float64 naive-product oracle disagreed by a relative `6e+261` because the *oracle*
underflowed — `sigmoid(88)` rounds to 1.0, cancelling `1 - p` to zero. `np.longdouble` was still
`2.4e-08` off; only 60-digit `mpmath` settled it. This family recurred three times, each time nearly
escalating a **correct** implementation as refuted.

**Rules**

- Suspect the oracle's precision before suspecting the code.
- An oracle must consume the code's **actual received bits**, never the intended Python literals — a
  gap below the ulp arrives bit-identical. Assert every oracle input is post-cast.
- Prefer exact arithmetic (`fractions.Fraction`) where the quantity permits.

#### 13.4.5 Liveness is not correctness

**Measured:** a raw-layout loss moved under **both** destroy probes while returning `0.765062` where
the correct reshape gives `0.693310`.

"The component responded" is not "the component is right" — **assert the VALUE under the probe.**

Similarly, "the output changed" is not liveness for a **conditioning** input: one head demonstrably
read a prompt (objectness moved `5.59e-01`) while the top-k **selection** it was supposed to condition
moved exactly `0.0000` — a term constant across positions cannot change an argsort.

#### 13.4.6 Budget positive liveness arms before running a dead-component probe

The liveness arm must go through the **same detector** as the absence assertion.

**Measured:** a probe suite whose assertions were all absence, shape or parameter-count checks
measured **57 of 57** guards blind — every one is satisfied *by construction* when the component
emits zero.

#### 13.4.7 A guard goes vacuous when what it watches changes shape

Not wrong — **vacuous**, reporting clean forever.

**Measured:** a listener watching for a dropped-key *warning* reported a clean zero permanently after
a later change converted that warning into a *raise*. A TF32 canary written `if not <flag>: assert ...`
ran zero assertions in exactly the modules where a leak could originate.

**Rules**

- Exception-classify and assert expected values; never skip-on-condition.
- Re-verify a guard whenever a change alters the **signal** it watches, and note the new failure class
  at the guard's docstring.
- **A guard's docstring is not its contract — probe the predicate directly**, even for a guard shipped
  the same day. One dead-config-field guard advertised receiver-scoped checking; its regex counted
  `unrelated_object.field = 5`, a bare `field = 5`, and a `--field-name` substring inside a log message
  as "consumption". It was rewritten as an AST walk.

#### 13.4.8 Anti-vacuity on collection

```python
def test_the_guard_has_subjects():
    """Guard the guard: if the registries fail to import, everything below passes vacuously."""
    assert len(ENTRIES) > 50, (
        f"only {len(ENTRIES)} registry entries collected -- the factories likely failed "
        f"to import, so the drift tests below would pass while testing nothing"
    )
```

The same idea inside an individual test:

```python
assert not np.allclose(by_row, by_col), (
    "anti-vacuity FAILED: rescue_axis=-1 and rescue_axis=-2 gave the same answer, "
    "so this test cannot tell whether the axis is honored at all"
)
```

#### 13.4.9 A green control on the first run is suspicious

A control that comes back GREEN on its first run is more likely a probe defect than a clean pass. So
is a dead-component injection that "passes" cleanly on the first attempt — one anti-causal-mask
injection transposed itself back into a causal mask.

**Re-derive an injection's actual effect before trusting either green or red.**

Two more ways a RED proof can be structurally unable to fail:

| Cause | Detail |
|---|---|
| The effect is achieved by a **different, still-present code path** | deleting a re-export left a class registered, because a sibling import had already run the decorator. Registry **presence** and **name binding** are different contracts; split them |
| **An earlier guard on the same call path masks the defect behind it** | order fixes by masking, not by size. A RED proof written against the reported bug reproduces the wrong one, and the masked defect's "before" value must be measured at HEAD **plus the unmasking fix**, not at HEAD |

#### 13.4.10 Never `git stash` or `git checkout --` mid-proof

**Measured:** destructive five times, including on an **untracked** file where `git checkout --`
silently no-ops so the next injection stacks on already-corrupt source.

Restore from a byte-compared (`diff -q`) `cp` scratch backup.

#### 13.4.11 A guard that cannot distinguish pathological from unusual destroys correct answers

Test a guard's **false-positive** family as hard as its true-positive one. A finiteness guard on a
cumulative sum looked obviously right and poisoned ordinary exact rows.

Before adding a guard, check whether **the framework already raises**: Keras' `Conv2D` already raises
on a groups/filters mismatch, `MultiHeadAttention` on rank and last-dimension errors. And falsify a
mandated guard by measurement before implementing it — one would have crashed nine passing tests on
provably correct geometry.

### 13.5 Test Anti-Patterns

All of these have been found in shipped test suites.

| Anti-pattern | Why it passes |
|---|---|
| `assert True` / "if we reach here, the call was successful" | asserts nothing |
| **Constructor-attribute echo** — `assert model.d_state == d_state` | proves the constructor stored the argument and nothing else |
| **Shape-only knob sweep** — sweep a semantic knob, assert output shape unchanged | invariant under the knob being dead |
| **Shape-only mask test** — build a mask, run the model, assert output shape | replace every attention block with an identity and it still passes |
| Blanket `except Exception: pytest.xfail(...)` around build+forward | a total build break reports green; assertions after the block are unreachable |
| `pytest.raises(Exception)` in a meta-test | any failure counts, including argument validation |
| A meta-test that breaks the **arguments** rather than the model | fails before a model exists |
| Tolerances at `atol=1e-1` | nothing fails |
| Tolerances **below the dtype noise floor** | permanently RED, measuring nothing — three such assertions were never once green for their whole lifetime |
| `assert len(x) > 0`, `assert model.weights` | one scalar created in `__init__` satisfies it; a weight-path guard placed right after `keras.Model()` reads `[]` unconditionally |
| Direct `layer.build(...)` in a unit test | structurally blind to the `StatelessScope` trap |
| Round-tripping only the config that has **no** sub-layers | exercises the path that builds nothing |
| Comparing an **unfitted** model across a round trip | round-trips nothing meaningful |
| A test that pins the **value a defect produces** | goes RED when you fix the bug |

The last one deserves its own note, because it is the most demoralizing. Live examples: an assertion
that a configurable head's activation was `'sigmoid'`; that a model's feature rank was 2, for a model
whose `count_params()` was 0; a gradient-flow test that passed **because of** a dead-table defect,
whose `1e-8` floor cleared only under the broken initialization and would fail the correct one; that a
dead block's output delta was exactly `0.0`; that an echoed attention mask was `None`; and a variant
table test pinning wrong per-variant layer counts.

> **Re-derive what each assertion is pinned to** before trusting it.

Two more, about the shape of the whole gate:

- **A 100%-passing suite is not evidence that entry points work. Run the CLI.** 249 tests were green
  while both trainers were broken — one raised `IndexError` on every real run, the other had no
  argument parser and started a 100-epoch job on `--help`. Defects **cluster at entry points with zero
  tests**, reliably enough to plan around. Weight "has no tests" at least as highly as "review flagged
  it".
- **Collection-only gating hides RED tests.** An all-skip module reads as a pass; a suite whose
  collection errored can "pass" by running almost nothing. Gating on `--collect-only` once hid 12 real
  failures across 8 steps. **Always quote the passed count together with the collected count**, and
  where a pre-existing failure must be preserved, compare the failing **node-id set**, not the count.

### 13.6 Measurement Traps

#### 13.6.1 TF32 is the default false model defect

**Measured:** three confirmed instances. A GPU-only RED with a CPU-green counterpart is a TF32 suspect
**before** it is a bias hunt.

**The diagnostic:**

```
 TF32 artifact          flat across four decades of a scaling constant,
                        exactly 0.0 at powers of two
 additive-bias leak     decays as 1/c
 clipping               grows with c
```

Confirm by toggling `tf.config.experimental.enable_tensor_float_32_execution` in-process on the
**same trained object**, reversibly, and with a direct float64 discriminator.

Own the flag in exactly one fixture:

```python
@pytest.fixture(scope="module")
def tf32_disabled():
    """Opt in per module with `pytestmark = pytest.mark.usefixtures("tf32_disabled")`."""
    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)
        assert tf.config.experimental.tensor_float_32_execution_enabled() == previous, \
            "TF32 setting leaked out of this module"
```

Plus a canary that fails the **next** test after a leak:

```python
@pytest.fixture(autouse=True)
def _tf32_leak_canary():
    expected = False if _TF32_SCOPED_OFF else _TF32_SESSION_BASELINE
    actual = tf.config.experimental.tensor_float_32_execution_enabled()
    assert actual == expected, (
        "TF32 leaked: every float32 tolerance that runs after it now depends on "
        "execution order."
    )
    yield
```

> **Never disable TF32 at module import.** One import-time call is process-global for the whole
> session and made a precision measurement swing by roughly **1000–1500x** depending on what else was
> collected. Verify precision-sensitive tolerances in both regimes.

Gate on `flag AND device_has_the_feature`: `tensor_float_32_execution_enabled()` reads `True` on CPU,
where TF32 does not exist and the numerics are true float32.

#### 13.6.2 Quote near-zero statistics from CPU

`CUDA_VISIBLE_DEVICES=""`.

**Measured:** a GPU disagrees with **itself** run-to-run at about `5e-6`, and across process launches
by exactly `0.228515625` on a fixed, unchanged model, while CPU gives exactly `0.0`.

Pin golden-value probes to CPU, and **do not repair a golden failure by relaxing `atol`.**

#### 13.6.3 Eager-only bit-identity does not license "inert"

**Measured:** under `@tf.function`, reassociation produces nonzero deltas with **zero** source-level
change — `4.77e-07`, `4.23e-04` and exactly `0.0` on three models.

The correct control is the **within-version eager-vs-graph delta on unchanged code**.

Related: a `training=False` equivalence probe is blind to operation-**order** divergence. Moving
dropout from before a norm to after read "bit-identical" at `0.0`, then `0.3953` max delta at
`training=True` under an identical Bernoulli mask. Read the two `call()`s side by side.

#### 13.6.4 Test-order RNG coupling

A test whose statistic reads the process-global Keras RNG is coupled to pytest **collection order**:
the file passes alone and the directory gate goes RED.

**Measured:** merely adding tests to an earlier-sorting file dropped one magnitude probe from `>1.0`
to `0.818`.

**Rules**

- `keras.utils.set_random_seed(N)` immediately before construction; keep the **shipped** initializers;
  record the across-seed spread at the test.
- Prefer `np.random.default_rng(seed)` for input data.
- **Never** tune a synthetic input's sigma until the bar passes.

Note also that Keras 3 hands **one** initializer instance to every same-shaped sibling projection
inside a layer, and the instance materializes its seed once — so identically-shaped weights get
**bit-identical** draws. An "untrained control" built this way had `Q == K` exactly, with entirely
plausible downstream statistics.

#### 13.6.5 Untrained models cannot answer some questions

A zero-initialized gate makes the branch under test inert, so the defect reads as the float32 floor. A
zero-initialized final projection zeroes the gradient of everything behind it.

Use seeded **non-zero** weights and biases — the state a trained model is in.

**Measured:** a layer's default `bias_initializer='zeros'` made two of three masking sites
structurally unobservable, and a sampled path was green with a live defect at a single perturbation
scale, caught only by sweeping four.

#### 13.6.6 Set the tolerance from the defect signal, not the noise floor

A tolerance derived from the noise floor is not a tolerance; it only says the computation ran. The
bound must sit **between** the floor and the smallest defect you intend to catch, and the test should
record both numbers.

#### 13.6.7 Never run GPU jobs in parallel

Contention causes false **failures**, never false passes.

**Measured:** the same suite gave 21 failed / 77 passed under contention and 89 passed alone. Three
parallel explorer agents once manufactured a false "8 pre-existing failures" premise that serial
re-measurement reduced to zero.

A GPU-contention error reads exactly like a regression; the tell is `cudaSetDevice() ... out of
memory` **at import**. Check that exactly one pytest process is running before believing a red run.

Use a pristine `git worktree` at the true base as a control. A **partial revert** is not a substitute
— one "pre-existing RED" claim survived reverting three files while the suspect change lived in a
fourth.

#### 13.6.8 Patch the defining module

A shadow-import or monkeypatch binds the **importing** module only. Patching a re-exported name cannot
reach the defining module's own call site, and a package `__init__.py` re-export can make a
shadow-import exercise the unpatched code. Patch the **defining** module's namespace.

#### 13.6.9 Exhaustiveness by grid size is not exhaustiveness

**Measured:** "0 violations over 281,604 rows" held while a small targeted counterexample broke the
property immediately. A grid can be structurally blind regardless of cell count: sampling one
parameter uniformly made a per-chunk carry about `1.8e-26`, annihilating the very state the test
claimed to check — and pinning that parameter to a constant then made the factors bit-identical,
missing a mis-index.

**Derive an attack; do not sweep and hope.**

Similarly, a fixture can construct a shape the real pipeline can never emit, passing while the shipped
default combination crashes. **Drive guards through the actual factories and data path.** And verify
an assertion actually **executes on every arm** of a parametrized build. Prefer **non-local**
assertions over shape and finiteness ones.

### 13.7 Test Module Layout and Naming

#### 13.7.1 Files

| Kind | Convention |
|---|---|
| Comprehensive suite for one unit | `test_<layer>.py` / `test_<model>.py` |
| Single-claim guard | **sentence-named after the claim**, not the unit: `test_the_attention_mask_is_honoured.py`, `test_tables_survive_stateless_build.py`, `test_the_gates_actually_gate.py` |
| Shared instrument | **no `test_` prefix** so pytest does not collect it: `smoke_contract_oracle.py`, `knob_sensitivity_oracle.py`, `dead_component_oracle.py`. RED proofs live in a mirrored `test_<name>.py` |
| Package-local `conftest.py` | only when there is shared **instrumentation** |

#### 13.7.2 Names

| Kind | Convention |
|---|---|
| Classes | `Test<Unit><Aspect>` — `TestBeitAttentionBiasOrientation`, `TestFloat32IsTheControl`, `TestBuildsExactlyWhatCallRuns`. Guard classes named after the claim |
| Functions | declarative sentences asserting the claim, not `test_X_works` — `test_masked_tokens_do_not_reach_the_visible_positions`, `test_gamma_zero_is_exact_identity` |
| Meta / anti-vacuity | `test_the_guard_…`, `test_the_probe_…`, `test_the_contract_…` |

#### 13.7.3 Module skeleton

```python
"""Test suite for <Layer> (<paper / family>).

Covers initialization and stored config, constructor validation, forward shape over multiple
spatial sizes (including NON-square ones) and channel widths, `compute_output_shape` pre- and
post-build, training-mode behaviour, gradient flow, a `.keras` VALUE round trip, and the N
mandated behavioural pins:
  1. test_gamma_zero_is_exact_identity
  2. test_gamma_nonzero_is_not_identity   (both arms -- an identity-only assertion is also
     satisfied by a completely dead component)
"""

class TestMyLayer:
    # --- fixtures -------------------------------------------------------
    #   basic_config, sample_input (np.random.default_rng(1234), fixed shape)
    # --- initialization / config ----------------------------------------
    #   stores config (incl. `assert not layer.built`), sub-layers created in __init__,
    #   config completeness then from_config, round trip with None options
    # --- validation ------------------------------------------------------
    #   parametrized invalid kwargs -> pytest.raises(ValueError, match=...)
    #   invalid input rank / channels, raised at build()
    # --- forward pass ----------------------------------------------------
    #   parametrized height/width x dim; shape + np.all(np.isfinite(...))
    #   compute_output_shape pre-build (assert `not layer.built` before AND after)
    #   compute_output_shape matches forward
    # --- behavioural pins ------------------------------------------------
    #   one per claim, each with its measured number in the docstring, in both-ways pairs
    # --- training mode ---------------------------------------------------
    #   inference is deterministic; training=True updates BN statistics;
    #   a frozen layer matches inference under training=True
    # --- gradients -------------------------------------------------------
    # --- serialization ---------------------------------------------------
```

#### 13.7.4 Session policy

- Seed with `np.random.default_rng(seed)` for data and `keras.utils.set_random_seed(N)` immediately
  before construction. Avoid statistics that read the global RNG.
- House style for exactness comparisons is `np.testing.assert_allclose(a, b, atol=..., rtol=0)`;
  `atol=0.0` for restoration and bit-identity.
- Any process-global setting (dtype policy, TF32, `floatx`) is owned by one fixture that restores in
  `finally` and asserts the restoration.
- **No test writes into the project's real output directory.** Route every config through `tmp_path`:

```python
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

def _entries():
    return set(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.is_dir() else set()

@pytest.fixture(autouse=True)
def no_writes_to_the_real_output_dir():
    before = _entries()
    yield
    new = sorted(p.name for p in _entries() - before)
    assert not new, (
        f"this test wrote into the project output directory: {new}. "
        f"route the config through tmp_path"
    )
```

> **Assert, never clean up.** A training-output directory is typically untracked and unbacked, so
> deletion there is unrecoverable. A cleanup step written as "remove every output directory this run
> created" once destroyed 62 run directories at once, including a published paper's subject
> checkpoint, because the relative paths in its log resolved against the project root rather than
> against the pytest `tmp_path` the test had actually written to. Delete only absolute paths recorded
> at creation time and verified created, or do not delete at all.

#### 13.7.5 Scoping runs

A full suite over a library of this kind runs for hours, so it is not a routine regression check.
Scope the runner to the modules you changed plus anything that imports what you touched, and run the
tree-wide **collection** gate after any change to a package's public surface. Reserve the full suite
for when it is explicitly asked for.

Do not trust a red run that was not the only pytest on the machine (§13.6.7).

---

## 14. Common Pitfalls and Solutions

### Pitfall 1: Conditional layer creation

```python
# ❌ WRONG
def __init__(self, use_feature_a=True):
    super().__init__()
    if use_feature_a:
        self.feature_a = FeatureLayer()

# ✅ CORRECT - create always, gate usage in call(); build only what call() runs
def __init__(self, use_feature_a=True):
    super().__init__()
    self.use_feature_a = use_feature_a
    self.feature_a = FeatureLayer(name="feature_a")
```

### Pitfall 2: Creating layers in `build()`

```python
# ❌ WRONG
def build(self, input_shape):
    self.dense = layers.Dense(self.units)

# ✅ CORRECT
def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.dense = layers.Dense(units, name="dense")

def build(self, input_shape):
    self.dense.build(input_shape)
    super().build(input_shape)
```

### Pitfall 3: Registration without a package

```python
# ❌ WRONG - module-independent key; last import wins on a name collision
@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer): ...

# ✅ CORRECT
@keras.saving.register_keras_serializable(package="my_project")
class Downsample(keras.layers.Layer): ...
```

### Pitfall 4: Incomplete `get_config`

```python
# ❌ WRONG
def get_config(self):
    return {"units": self.units}

# ✅ CORRECT
def get_config(self):
    config = super().get_config()
    config.update({
        "units": self.units,
        "activation": activations.serialize(self.activation),
        "use_bias": self.use_bias,
        # ... every __init__ parameter
    })
    return config
```

### Pitfall 5: `.assign()` of a constant table in `build()`

```python
# ❌ WRONG - discarded by StatelessScope; all zeros in every real model
self.table = self.add_weight(name="t", shape=(n,), initializer="zeros", trainable=False)
self.table.assign(compute_table())

# ✅ CORRECT
self.table = self.add_weight(name="t", shape=(n,), initializer=_table_init, trainable=False)
```

### Pitfall 6: `build()` that does not materialize the sub-layer tree

```python
# ❌ WRONG - reloaded model restores into nothing; nothing raises
def build(self, input_shape):
    self.threshold = self.add_weight(name="thr", shape=(), initializer="zeros")
    super().build(input_shape)

# ✅ CORRECT - build exactly what call() runs
def build(self, input_shape):
    self.threshold = self.add_weight(name="thr", shape=(), initializer="zeros")
    self.encoder.build(input_shape)
    if self.use_head:
        self.head.build(self.encoder.compute_output_shape(input_shape))
    super().build(input_shape)
```

### Pitfall 7: Graph-breaking shape operations

```python
# ❌ WRONG
shape_list = list(ops.shape(inputs))
batch = int(ops.shape(inputs)[0])

# ✅ CORRECT
shape = ops.shape(inputs)
new_shape = ops.stack([shape[0], self.units])
return ops.reshape(inputs, new_shape)
```

### Pitfall 8: Python conditionals on tensor values

```python
# ❌ WRONG - evaluated once, at trace time
if ops.mean(inputs) > 0:
    return inputs * 2
return inputs

# ✅ CORRECT
return ops.where(ops.mean(inputs) > 0, inputs * 2, inputs)
```

### Pitfall 9: `ops.tril` / `ops.triu` in a traced path

```python
# ❌ WRONG - TypeError: ('pred must not be a Python bool', True) under fit/jit
mask = ops.tril(ops.ones((n, n)))

# ✅ CORRECT
idx = ops.arange(n)
mask = ops.cast(idx[None, :] <= idx[:, None], self.compute_dtype)
```

### Pitfall 10: An fp16-unsafe mask sentinel

```python
# ❌ WRONG - float16(-1e9) is -inf, and 0.0 * -inf = NaN on the KEPT positions
scores = scores + (1.0 - mask) * -1e9

# ✅ CORRECT
scores = ops.where(ops.cast(mask, "bool"), scores, _sentinel_for(self.compute_dtype))
```

### Pitfall 11: Symbolic `training` into BatchNorm / Dropout

```python
# ❌ WRONG - OperatorNotAllowedInGraphError for a traced True AND a traced False
return self.bn(x, training=training_tensor)

# ✅ CORRECT - keep the Python-bool path byte-identical; only a tensor reaches ops.cond
if isinstance(training, bool) or training is None:
    return self.bn(x, training=training)
return ops.cond(training, lambda: self.bn(x, training=True),
                          lambda: self.bn(x, training=False))
```

### Pitfall 12: Mutable default arguments

```python
# ❌ WRONG
def __init__(self, layer_sizes=[64, 128]): ...

# ✅ CORRECT
def __init__(self, layer_sizes: Optional[List[int]] = None):
    self.layer_sizes = [64, 128] if layer_sizes is None else list(layer_sizes)
```

### Pitfall 13: Inconsistent layer names

```python
# ❌ WRONG - auto-generated names shift when depth changes
for i in range(depth):
    self.blocks.append(Block())

# ✅ CORRECT
for i in range(depth):
    self.blocks.append(Block(name=f"block_{i}"))
```

### Pitfall 14: `compute_output_shape` reading weight shapes

```python
# ❌ WRONG - fails on an unbuilt layer
return (input_shape[0], self.kernel.shape[-1])

# ✅ CORRECT
return (input_shape[0], self.units)
```

### Pitfall 15: A factory that filters and drops unknown keys

```python
# ❌ WRONG - a misspelled key silently becomes a default
params = {k: v for k, v in kwargs.items() if k in accepted}

# ✅ CORRECT
unsupported = sorted(set(kwargs) - declared)
if unsupported:
    raise ValueError(f"unsupported parameter(s) {unsupported}; accepted: {sorted(declared)}")
```

### Pitfall 16: Constructing layers or mutating Python state in `call()`

```python
# ❌ WRONG - a new object per trace, untracked; and a list that grows once per TRACE
def call(self, x):
    pool = layers.AveragePooling2D()
    self._accum.append(x)
    return pool(x)

# ✅ CORRECT
def call(self, x):
    return ops.average_pool(x, pool_size=2, strides=2, padding="valid")
```

A layer-tree walk cannot see the untracked object. **Detect by counting constructor invocations
across two forward passes.**

The fix above removes the constructed layer. The list mutation needs its own fix: cross-step state
belongs in a `keras.Variable` updated under `ops.cond`, never a Python container — a list appended to
inside `call()` grows once per **trace**, not once per batch (§11.2).

### Pitfall 17: `pretrained=True` that warns and returns random weights

```python
# ❌ WRONG
if pretrained:
    logger.warning("pretrained weights not yet available")

# ✅ CORRECT
if pretrained:
    raise NotImplementedError(
        f"No pretrained weights are distributed for variant {variant!r}. "
        f"Load a local checkpoint with load_pretrained_weights(path)."
    )
```

### Pitfall 18: A custom `train_step` without `scale_loss`

```python
# ❌ WRONG - under mixed_float16 the gradients are divided but never scaled: 2^15 too small
with tf.GradientTape() as tape:
    loss = self.compute_loss(x, y, self(x, training=True))
grads = tape.gradient(loss, self.trainable_variables)

# ✅ CORRECT - differentiate the SCALED loss; scale_loss is a no-op off mixed precision
with tf.GradientTape() as tape:
    loss = self.compute_loss(x, y, self(x, training=True))
    scaled_loss = self.optimizer.scale_loss(loss)
grads = tape.gradient(scaled_loss, self.trainable_variables)
self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
```

Computing `scale_loss(loss)` and then differentiating `loss` anyway is the same bug with an extra
line — the scaled value must be the one handed to `tape.gradient`.

---

## 15. Troubleshooting Guide

### 15.1 Debug Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ✅ Registration     explicit package= present?               │
│ 2. ✅ Creation         all sub-layers created in __init__?      │
│ 3. ✅ Build            materializes exactly what call() runs?   │
│ 4. ✅ Constant tables  computed in an initializer, not assign?  │
│ 5. ✅ Config           get_config() returns every argument?     │
│ 6. ✅ Output shape     compute_output_shape from stored config? │
│ 7. ✅ Graph safety     call() symbolic only, no tril/triu?      │
│ 8. ✅ Serialization    round trip compares VALUES?              │
│ 9. ✅ Finiteness       every forward test asserts isfinite?     │
└─────────────────────────────────────────────────────────────────┘
```

### 15.2 Errors and Symptoms

| Symptom | Likely cause | Section |
|---|---|---|
| `Unknown layer: MyLayer` | missing registration decorator | §2.2 |
| Two classes load as each other, depending on import order | bare `register_keras_serializable()` key collision | §2.2 |
| `Layer was never built and thus has no variables` | sub-layer not built before weight loading | §8 |
| **Reloaded model has zero weights, or matches 0 of N against its donor** | `build()` does not materialize the sub-layer tree | §8.1 |
| **`count_params()` returns exactly 0** | `Model.build(shape)` on a subclassed model walks no sub-layers | §8.2 |
| A constant table is all zeros in training but correct in a unit test | `.assign()` in `build()` discarded by `StatelessScope` | §3.3 |
| `InaccessibleTensorError` on `fit()` of an unbuilt model | constant materialized with `ops.convert_to_tensor` in `build()` | §3.3 |
| `TypeError: ('pred must not be a Python bool', True)` under `fit`/`jit` but not eagerly | `ops.tril` / `ops.triu` | §4.2 |
| `OperatorNotAllowedInGraphError` | symbolic `training` into BN/Dropout, or `ops.cond` on a traced shape value | §4.2, §10.2 |
| **All-NaN output of the correct shape** | degenerate-length reduction, or an fp16 `-1e9` mask sentinel | §10.2, §10.1 |
| NaN on the positions the mask is meant to KEEP | `0.0 * -inf` from an additive fp16 sentinel | §10.1 |
| `cannot compute AddV2 as input #1 was expected to be a half tensor` | one-sided cast under autocast; cast both sides | §10.1 |
| **Training does not move under `mixed_float16`** | custom `train_step` missing `optimizer.scale_loss` | §11.1 |
| A knob has no measurable effect | dead knob (§12.5), or silently dropped at a factory | §12.5, §9.2, §13.3.2 |
| `Structures don't have the same nested structure` from `predict` | `call` echoing a bare `None` in its output dict | §7.4 |
| A causal model's loss looks fine but generation is poor | no causal mask; or the head pools token 0 | §12.1, §12.2 |
| Green suite, broken trainer | entry point with zero tests; run the CLI | §13.5 |
| Test passes alone, fails in the directory gate | global-RNG coupling, or TF32 leaked from an earlier module | §13.6.4, §13.6.1 |
| GPU red, CPU green, on a precision assertion | TF32 | §13.6.1 |
| `cudaSetDevice() ... out of memory` at import | another GPU job is running; the failure is contention | §13.6.7 |
| `RecursionError` during serialization | circular references in config; store parameters explicitly, not `locals()` | §6.1 |

---

## 16. Summary Checklists

### 16.1 A New Layer

**Construction**
- [ ] `@keras.saving.register_keras_serializable(package=...)` with an explicit package
- [ ] Class name does not collide with an existing registered class; prefixed if generic
- [ ] All sub-layers created in `__init__`, unconditionally, with explicit `name=`
- [ ] All configuration stored on `self` in `__init__`; no mutable defaults
- [ ] Argument validation raises `ValueError` naming the offending value
- [ ] Cross-parameter contracts that `call()` relies on are re-checked in `__init__`

**Build**
- [ ] `build()` materializes **exactly** the tree `call()` runs — no more, no less
- [ ] No `.assign()` of a constant table; tables computed inside an `add_weight` initializer
- [ ] No `ops.convert_to_tensor` on a closed-over constant; NumPy in, convert in `call()`
- [ ] `super().build(input_shape)` last

**Call**
- [ ] Symbolic only: no `.numpy()`, no Python `if` on a tensor value, no Python loop over a tensor
      dimension, no layer construction, no list mutation, no logging
- [ ] No `ops.tril` / `ops.triu`
- [ ] `training=` forwarded explicitly to every sub-layer
- [ ] No possibly-symbolic `training` reaching `BatchNormalization` or `Dropout`
- [ ] Mask sentinel derived from `compute_dtype`, or expressed as `ops.where`
- [ ] Static shape contracts re-asserted here, not only in `build()`

**Shape and config**
- [ ] `compute_output_shape` implemented, from stored config, working unbuilt
- [ ] Shape arithmetic in exactly one pure helper shared by `build` / `call` / `compute_output_shape`
- [ ] `get_config()` returns **every** constructor argument, complex objects serialized
- [ ] `from_config()` deserializes them; no popping of base keys
- [ ] Normalization epsilon from the factory, or passed explicitly with a cited reference

**Reuse**
- [ ] Checked the domain factory, then the existing layer surface, before authoring
- [ ] Registered in the domain factory if one exists; the factory raises on undeclared keys

### 16.2 A New Model Package

Everything above, plus:

- [ ] Module docstring is substantive prose with a `References:` section (§5.1)
- [ ] A variant registry present; a separate architecture table, if any, not merged into it
- [ ] Variant values derived from a **named reference**, cited
- [ ] `from_variant` raises `ValueError` listing available keys, accepts its documented overrides,
      does not splat description metadata
- [ ] `pretrained=True` raises `NotImplementedError` naming the variant; no placeholder URL table; no
      `by_name`; no load failure swallowed into a warning
- [ ] Module-level `create_<name>()` delegating to `from_variant` with no logic of its own
- [ ] Package `__init__.py` exports class and factory with a curated `__all__`, and binds no name
      matching one of its own subpackages
- [ ] One `logger.info` in `__init__`; none in `call`
- [ ] No new custom `train_step`
- [ ] Checkpoint-affecting changes recorded in a shipping document
- [ ] Tree-wide collection gate run: `pytest tests/ -q --collect-only`

### 16.3 The Tests, Before You Call It Done

- [ ] `.keras` round trip on **values**, `rtol=0`, `training=False` explicit
- [ ] Weight-value comparison at `atol=0.0` **before** the loaded model's first call
- [ ] Build parity by relative `w.path`, **plus** a no-sub-layer layout assertion per `None`/`False`
      config
- [ ] Build-through-a-parent probe for every constant table
- [ ] Per-variable gradient flow: non-`None` **and** non-zero, named by `var.path`, with the
      `len(trainable_variables) > 0` anti-vacuity assertion
- [ ] Every constructor knob pinned with the instrument matching its class (§13.3.2)
- [ ] `ops.all(ops.isfinite(y))` in every forward test
- [ ] Degenerate lengths (0, 1) swept on the static path **and** a `TensorSpec([None, ...])` trace
- [ ] `mixed_float16` and `float64` construction-and-forward arms, with a float32 control
- [ ] `@tf.function(jit_compile=True)` versus eager
- [ ] Causality: the three-armed future-leak probe, if the model is causal
- [ ] Composition asserted directly, if the architecture's value is composition
- [ ] Orientation: delta impulse on a **non-square** grid
- [ ] Every "nothing changed" assertion has its twin
- [ ] Every guard proven RED by an injection, **in the committed record**
- [ ] Every parametrized repo-wide guard asserts a non-empty subject set
- [ ] Every tolerance carries its measurement and the defect signal it sits below

---

## 17. Appendix: Refuted Claims

Recorded so they are not re-proposed, and so a rule above is not re-derived from a premise already
falsified by measurement.

| Claim | Status |
|---|---|
| The nested `List[List[Layer]]` weight-loss trap | **Does not reproduce on Keras 3.8.** `_flatten_layers` round-trips every weight regardless of container nesting; a model with a `List[List[Dict[str, Layer]]]` structure restored all 65 weights bit-identically. Check that a code shape still bites before claiming a guard against it is load-bearing |
| "Overrides `build()`" as the discriminating property for round-trip weight loss | **Wrong property.** Whether `build()` *materializes the sub-layer tree* is. See §8.2 |
| A structured-dict `y_pred` cannot be used with stock `compile()` / `fit()` | **False on Keras 3.8.** `CompileLoss.build` breaks in exactly one configuration: a **single `Loss` object** handed a dict `y_pred`, where it broadcasts across every leaf then raises `KeyError`. Supply a dict `loss=` keyed to the same output names, plus a dict `y_true` |
| A custom `train_step` drops regularizer terms | **False.** Keras 3.8's default `compute_loss` already sums `self.losses`. The real instance summed a *sub-layer's* `.losses` explicitly. An AST predicate "does the body mention `self.losses`" measures the wrong thing |
| `assert model.losses` proves a regularizer is live | **Vacuous** when an unrelated block contributes to it. Assert a delta against a no-regularizer baseline |
| A large delta after a masking change is a silent un-masking regression | **Was not one.** Softmax is invariant to a constant shift along its reduction axis, and `x - 1e9` in float32 collapses a row to a single value (the ulp at `1e9` is 64). Add a control proving the pre-change output was itself meaningful |
| A GPU-only homogeneity RED at `5.063e-04` was a bias leak | **It was the TF32 ulp.** See §13.6.1 for the diagnostic |
| `x + g - stop_gradient(g)` is the identity | **False** under left-to-right float association — about 25% of float64 draws differ by up to 1 ulp. Group it as `x + (g - stop_gradient(g))`: exactly `0.0` forward, gradient unchanged |

**The meta-lesson.** A constraint recorded in institutional memory can be a true observation
over-generalized into a false rule — re-execute the one that is blocking you.

And several *prescribed fixes* were themselves regressions, caught only by running them and diffing
the number:

- bias-correcting an EMA codebook without also zero-initializing it made the defect roughly **10x
  worse**;
- forwarding a "dropped" `dropout_rate` stacked a **second** dropout, so a requested 0.25 became an
  effective 0.4375;
- resolving an echoed mask earlier was a no-op for one model and a `6.42e-01` change for its sibling.

> **Run the prescribed fix and diff the number, not just the shape.**
