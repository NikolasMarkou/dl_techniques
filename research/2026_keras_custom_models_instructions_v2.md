# Authoring Keras 3 Custom Layers and Models — v2

The canonical guide for creating layers and models in `dl_techniques`. It supersedes
`research/2026_keras_custom_models_instructions.md` (v1), which is retained only for historical
reference.

## Why there is a v2

v1 taught how to make a layer **construct and serialize**. Those rules were correct and they
held. But a library-wide audit of every model package found that almost none of the real defects
were of that kind. They were defects in code that satisfied every rule v1 states:

- A layer stored, validated, serialized and documented a parameter that no code path read. Several
  models measured `max|dy| = 0.000e+00` across every legal value of a knob.
- A rotary position embedding was constructed, built, parameter-counted and serialized while being
  an **exact algebraic no-op**, because it received the head axis instead of the sequence axis.
  Rotating query-head *h* and key-head *h* by the same matrix leaves `(Rq)·(Rk) = q·k`.
- `add_weight(initializer='zeros')` followed by `.assign(table)` inside `build()` — a pattern v1
  permits — leaves the table all zeros in every real model. A trend-only N-BEATS returned exactly
  `0.0` everywhere and still trained and still reported a loss.
- A decoder-only language model attended bidirectionally under a next-token objective.
- A model reloaded from `.keras` restored **zero** weights and passed its round-trip test.

Each of these shipped behind a green suite. Shapes matched, parameter counts matched, gradients
existed, serialization round-tripped, and the loss curve looked normal.

The lesson is not that v1 was wrong. It is that **construction correctness and behavioural
correctness are different properties, and only the first one is easy to test.** A guide that stops
at construction produces exactly the library the audit found.

v2 therefore has two halves. Parts I–III are what to write. Parts IV–V are how to prove what you
wrote does what you claim. The second half is the one that was missing, and it is not optional:
in this repo, a guard that cannot fail is the *most common* outcome of writing a new test, not an
edge case.

## How to use this document

- Writing a new layer → Parts I, II, III, then the checklist in V.
- Writing a new model package → all of it. Part V carries the house shape.
- Fixing a bug → Part III to find the family, Part IV to build a guard that can actually fail.
- Reviewing → Part III as the checklist, Part IV § anti-patterns as the test-review checklist.

Rules are stated as instructions. Numbers appear only where the number *is* the instruction.
Part VI lists claims that were investigated and **refuted**, so they do not get re-proposed.

---

## Table of Contents

**Part I — Mechanics**
1. [The lifecycle and the Golden Rule](#1-the-lifecycle-and-the-golden-rule)
2. [Registration and serialization identity](#2-registration-and-serialization-identity)
3. [Layer implementation patterns](#3-layer-implementation-patterns)
4. [build(): what to materialize](#4-build-what-to-materialize)
5. [compute_output_shape](#5-compute_output_shape)
6. [Graph-safe call()](#6-graph-safe-call)
7. [Configuration and get_config](#7-configuration-and-get_config)
8. [Model patterns and the house shape](#8-model-patterns-and-the-house-shape)

**Part II — Reuse before you author**
9. [The factory registries](#9-the-factory-registries)
10. [Porting from a reference implementation](#10-porting-from-a-reference-implementation)

**Part III — The failure catalogue**
11. [Inert configuration](#11-inert-configuration)
12. [Inert components](#12-inert-components)
13. [Build and serialization failures](#13-build-and-serialization-failures)
14. [Composition failures](#14-composition-failures)
15. [Numerics](#15-numerics)
16. [The training path](#16-the-training-path)
17. [Causality and masking](#17-causality-and-masking)

**Part IV — Proving it works**
18. [The five house rules](#18-the-five-house-rules)
19. [The instruments](#19-the-instruments)
20. [The shared oracles](#20-the-shared-oracles)
21. [Why guards fail](#21-why-guards-fail)
22. [Test anti-patterns](#22-test-anti-patterns)
23. [Measurement traps](#23-measurement-traps)

**Part V — Shipping**
24. [Checklists](#24-checklists)
25. [Test module layout and naming](#25-test-module-layout-and-naming)
26. [Troubleshooting](#26-troubleshooting)

**Part VI**
27. [Refuted claims](#27-refuted-claims)

---

# Part I — Mechanics

## 1. The lifecycle and the Golden Rule

```
SAVING (.keras)                          LOADING (.keras)
  get_config() on each layer               parse JSON config
  config -> JSON                           __init__(**config)  -> UNBUILT layer
  weights of each BUILT layer              build()             -> creates variables
  package into archive                     load weight VALUES into those variables
```

The load path creates the layer **unbuilt** and then builds it. Every consequence in this section
follows from that one fact: if a weight does not exist at the moment values are restored, the
value has nowhere to land, and **nothing raises**.

| Method | When | What belongs there |
|---|---|---|
| `__init__` | once, at instantiation | create ALL sub-layers; store ALL configuration |
| `build` | once, when shapes are known | create this layer's weights; materialize the sub-layer tree |
| `call` | every batch | symbolic operations only |
| `compute_output_shape` | shape inference | output shape from **stored config**, on an unbuilt layer |
| `get_config` | serialization | every constructor argument |

**Never in `__init__`:** create weights, inspect `input_shape`, run shape-dependent operations.

**Never in `call()`:** construct a layer, mutate a Python container, call `.numpy()` or
`convert_to_numpy`, branch a Python `if` on a tensor value, or log.

### 1.1 Create unconditionally, use conditionally

```python
# WRONG - the weight set depends on a flag
def __init__(self, use_feature_a=True, **kwargs):
    super().__init__(**kwargs)
    if use_feature_a:
        self.feature_a = FeatureLayer()

# RIGHT - create always, gate the USAGE
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

This keeps the checkpoint layout stable across configurations, which is what makes
`include_top=False` transfer and warm-starting work.

**Two things this rule does not give you, both of which have been mistaken for defects:**

- A layer created but never called is still parameter-counted, still optimizer-tracked, and still
  appears in `model.weights` and in a gradient walk — while contributing **exactly 0.0** to the
  output. It will also emit a missing-gradient `UserWarning`. That warning is sometimes correct and
  sometimes the intended cost of layout stability. Diagnose the `call()` branch before "fixing" it;
  if the inertness is deliberate, set `layer.trainable = False` in `__init__` (pre-build) and say
  so in a comment. Note that `model.trainable = True` silently undoes it.
- The contract does **not** survive a rebuild as a functional graph. Keras prunes any layer with no
  path to a declared output, even one that was constructed and applied on a dead branch. Do not
  write a test asserting a contract the graph cannot hold; document the divergence and expose a
  named feature tap instead.

### 1.2 Configuration is data

Every architectural decision is a constructor argument with a serializable value. No `add_layer()`
builder methods, no callables stored as configuration, no mutable default arguments.

```python
# WRONG
def __init__(self, layer_sizes=[64, 128]): ...
# RIGHT
def __init__(self, layer_sizes: Optional[List[int]] = None):
    self.layer_sizes = [64, 128] if layer_sizes is None else list(layer_sizes)
```

## 2. Registration and serialization identity

```python
@keras.saving.register_keras_serializable(package="dl_techniques")
class MyLayer(keras.layers.Layer):
    ...
```

**Always pass an explicit `package=`.** A bare `register_keras_serializable()` produces a key that
is independent of the defining module, so two classes with the same name in different packages
claim the same key. Whichever module imports **last** silently wins; loading a saved model of the
other one is broken, and which one breaks depends on import order. This has happened between
unrelated model packages with generic names (`Downsample`, `Upsample`, `RepMixerBlock`).

Corollaries:

- If a generic class name already exists elsewhere in the tree, **prefix it**
  (`FastVitRepMixerBlock`, not a second `RepMixerBlock`).
- A `custom_objects` dict must be keyed by `keras.saving.get_registered_name(cls)`, **never** the
  bare class name. `_retrieve_class_or_fn` never uses a literal class name as a lookup key for
  classes, so a dict keyed that way is entirely decorative and every entry is ignored.
- Never bind a name in a package `__init__.py` that matches one of that package's own
  **subpackages**. Re-exporting a class `SAM2` from a package that also contains a `SAM2/`
  subpackage shadows the subpackage, and every `from ...SAM.SAM2.model import ...` stops resolving.
  This breaks at **collection** time, so per-package test runs never see it.

Run the tree-wide collection gate after any change to a package's public surface:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_models/ -q --collect-only
```

## 3. Layer implementation patterns

### 3.1 A layer with its own weights

```python
@keras.saving.register_keras_serializable(package="dl_techniques")
class SimpleCustomLayer(keras.layers.Layer):
    """One-line statement of what this layer is and what distinguishes it.

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

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

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

### 3.2 A layer containing sub-layers

Sub-layers are created in `__init__` and **built explicitly** in `build()`:

```python
def __init__(self, hidden_dim, output_dim, dropout_rate=0.1, use_norm=True, **kwargs):
    super().__init__(**kwargs)
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
    self.hidden_dim, self.output_dim = hidden_dim, output_dim
    self.dropout_rate, self.use_norm = dropout_rate, use_norm

    self.dense1  = layers.Dense(hidden_dim, activation="gelu", name="dense1")
    self.dropout = layers.Dropout(dropout_rate, name="dropout")
    self.norm    = layers.LayerNormalization(epsilon=1e-6, name="norm") if use_norm else None
    self.dense2  = layers.Dense(output_dim, name="dense2")

def build(self, input_shape):
    self.dense1.build(input_shape)
    hidden_shape = self.dense1.compute_output_shape(input_shape)
    self.dropout.build(hidden_shape)
    if self.norm is not None:
        self.norm.build(hidden_shape)
    self.dense2.build(hidden_shape)
    super().build(input_shape)
```

Give sub-layers explicit names, including inside loops (`name=f"block_{i}"`). Auto-generated names
shift when depth changes, and checkpoints stop matching.

### 3.3 Constant tables: the `StatelessScope` trap

**Never compute a constant table in `build()` and `.assign()` it into a weight.** Keras 3 runs the
symbolic build pass inside a `StatelessScope` whenever a sub-layer is first reached from a
**parent's** `call()`. The scope records the assignment and discards it. The table stays at its
initializer value in every real model.

```python
# WRONG - all zeros in any model where a parent's call() builds this layer
def build(self, input_shape):
    self.inv_freq = self.add_weight(
        name="inv_freq", shape=(self.dim // 2,), initializer="zeros", trainable=False
    )
    self.inv_freq.assign(1.0 / (self.theta ** (ops.arange(0, self.dim, 2) / self.dim)))
    super().build(input_shape)

# RIGHT - the initializer computes it, so there is nothing to discard
def build(self, input_shape):
    def _inv_freq_init(shape, dtype=None):
        idx = np.arange(0, self.dim, 2, dtype="float64")[: shape[0]]
        return ops.cast(1.0 / (self.theta ** (idx / self.dim)), dtype or self.compute_dtype)

    self.inv_freq = self.add_weight(
        name="inv_freq", shape=(self.dim // 2,), initializer=_inv_freq_init, trainable=False
    )
    super().build(input_shape)
```

This trap is **path-dependent**, which is why it survived for as long as it did:

| how the layer gets built | table value |
|---|---|
| `layer.build(shape)` called directly | correct |
| eager `layer(x)` on a top-level layer | correct |
| `keras.Model(inp, layer(inp))` | correct |
| first reached from a **parent's** `call()` | **all zeros** |

The last row is every real model. A unit test that calls `.build(...)` directly is exactly the test
that cannot see this. See §19.4 for the probe that can.

Related, same cause: **never materialize a constant with `ops.convert_to_tensor` inside `build()`**
and close over it. The tensor binds to the tracing `FuncGraph` and a later `fit()` on an unbuilt
model dies with `InaccessibleTensorError`. Keep the constant as a NumPy array and convert inside
`call()`.

## 4. `build()`: what to materialize

The rule is exact:

> **`build()` must materialize precisely the sub-layer tree that `call()` runs — no more, no less.**

Both directions are real defects.

**Under-building.** If `build()` creates only this layer's own scalars and leaves the sub-layers
unbuilt, a reloaded model has nowhere to put the saved arrays. Keras' `build_from_config` calls
`self.build(input_shape)` inside a bare `try/except: pass`, so nothing raises; `load_model` returns
a model whose sub-layers are still unbuilt, and the first forward pass builds them **fresh and
random**. Observed: a reloaded model with `len(model.weights) == 0`, and a model matching 0 of 16
weights against its donor.

**Over-building.** Building a sub-layer that `call()` skips creates weights the lazy path never
made. The checkpoint layout silently changes — a break dressed as a fix.

Two clarifications that matter, because both were initially got wrong:

- **Overriding `build()` is not itself the hazard.** The discriminating property is whether
  `build()` *materializes the tree*. A model that overrides `build()` and ends it with a concrete
  dummy forward pass round-trips cleanly. A model that overrides `build()` to create two scalars
  does not.
- On a **subclassed** model, `Model.build(batch_shape)` only marks the model built and walks no
  sub-layers, so `count_params()` returns exactly `0`. Several packages in this tree do this. It is
  not a working precedent to copy.

Enforce it with a build-parity test (§19.3) **plus** a direct layout assertion for each
`None`/`False` configuration — parity alone is blind to over-building.

### 4.1 Validate on every call, not only at build

A shape contract checked only in `build()` is checked once, against whatever shape happened to
arrive first. A contract on a singleton axis checked only at build has silently accepted a
non-singleton axis and convolved the wrong dimension, with no error. Re-assert static contracts in
`call()`.

Note that `InputSpec` cannot close a dynamic-shape hole: Keras' `assert_input_compatibility` tests
`shape[axis] not in {value, None}`, so an unknown dimension is explicitly **accepted**.

### 4.2 Validate cross-parameter contracts in `__init__`

A configuration that builds but cannot forward is a construction-time error. Every contract that
`call()` relies on must be re-checked in `__init__`, raising `ValueError` and naming the offending
value. Models have shipped where a mismatched pair constructed, validated and built cleanly, then
raised on the first forward pass — and where a shipped preset sat on a degenerate boundary with
nothing saying so. When you add such a check, **sweep every shipped preset** against it.

## 5. `compute_output_shape`

Every custom layer implements it. It must work on an **unbuilt** layer and must be derived from
stored configuration, never from weight shapes.

```python
# WRONG - fails before build
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.kernel.shape[-1])

# RIGHT
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.units)
```

**Shape arithmetic lives in exactly one pure helper**, called by `build()`, `call()` and
`compute_output_shape` alike. Duplicated formulas drift: one layer carried three copies of an
overlapping-segment formula, one of which used `+` where the others used `-`, and built its nodes
for the wrong length. Another declared a halved spatial extent unconditionally while the stride
lived on a sub-layer that a flag could remove.

Pin `compute_output_shape` against the layer's own forward output for **every branch of every mode
flag**, including branches no shipped variant reaches.

## 6. Graph-safe `call()`

Keras traces `call()` once with symbolic inputs. Anything that reads a tensor's *value* at trace
time is either an error or, worse, silently frozen to whatever the trace saw.

| Never | Instead |
|---|---|
| `list(shape)`, `int(x)`, `float(x)` on a tensor | `ops.shape(x)`, index the tensor |
| `.numpy()`, `convert_to_numpy` | stay symbolic |
| Python `if` on a tensor value | `ops.where`, `ops.cond` |
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

### 6.1 Operations that are traps on this stack

**`keras.ops.tril` and `keras.ops.triu`.** Both raise the moment they are traced:

```
TypeError: ('pred must not be a Python bool', True)
```

They pass every eager test and simultaneously break `fit`, `predict`, `jit_compile=True`, `.keras`
save/load and every symbolic-shape path. Build triangular masks by comparing `ops.arange`, or reuse
`dl_techniques.utils.masking`'s causal-mask helper.

**Symbolic `training` into `BatchNormalization` or `Dropout`.** Measured on this stack: both raise
`OperatorNotAllowedInGraphError` for a traced `tf.constant(True)` **and** a traced
`tf.constant(False)`. `LayerNormalization` does not. `tf.get_static_value` on a traced argument
returns `None`, so the value cannot be recovered inside the trace. Route through an explicit gate
that keeps the Python-bool path byte-identical and sends only a tensor flag to `ops.cond`, and gate
only the layers that need it.

**`training=` propagation.** Keras 3 propagates `training` through a single mutable `CallContext`
slot that every nested `__call__` overwrites and only the outermost entry restores. A sibling
sub-layer that forces a different `training` poisons the ambient value for every later
un-forwarded call. **Forward `training=` explicitly**, even where omitting it is currently a no-op.

**NumPy fancy indexing** (`t[batch_idx, item_idx]`) is invalid on a backend tensor and raises
eagerly — the layer is dead on every forward pass. Use `ops.take_along_axis`.

### 6.2 A `call()` crash during build-tracing is downgraded to a warning

Keras converts an exception raised while tracing `call()` during a build pass into a
`UserWarning`. A completely broken layer therefore sits inside a green suite with exit code 0.
**Exit code 0 is not evidence.** Grep the test output for the exception text, or run the relevant
gate under `-W error::UserWarning`.

## 7. Configuration and `get_config`

`get_config()` returns **every** constructor argument. Complex objects are serialized, and
`from_config` deserializes them.

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

### 7.1 `**kwargs` is not a channel to your sub-layers

A key read out of `**kwargs` that is **also** forwarded to `super().__init__()` is dead on arrival:
Keras rejects unknown base-class keys. Assigning `self.shared_kwargs = kwargs` and forwarding it to
children leaks base arguments in the other direction. Split the base kwargs from pass-through
kwargs explicitly and name them.

A `from_config` that pops base keys is a **tell that `__init__` has this bug**. Once `__init__` is
fixed, that pop discards `name` and `trainable` — so a frozen head silently reloads unfrozen, with
bit-identical outputs.

### 7.2 Pair every new validation raise with a migration path

A new `ValueError` on a value the **old default** produced breaks deserialization of every existing
checkpoint. Soften only the `from_config` path — substitute the value and warn that the numerics
have changed — and leave the constructor raise in place for fresh code. Record every
checkpoint-affecting change in a **shipping** document (`src/dl_techniques/models/CLAUDE.md`
carries the table); notes in gitignored directories do not ship.

Sometimes the right answer is to refuse the shim. A remapping that would rebuild a *different*
weight tree than the file contains is worse than a hard failure.

### 7.3 Caches derived from weights

Value-exact round-tripping is not sufficient for a cache computed **from a weight**. A cached
positional table computed from a stale pre-restore weight was off by 1.999 while thirteen
round-trip tests passed. Cache only pure functions of shape and dtype, or invalidate on the weight.

## 8. Model patterns and the house shape

A model package that implements **one architecture with named variants** follows the shape below.
It is a target, not a universal law; the exemptions are at the end.

### 8.1 Module skeleton

The module docstring is **substantive prose, not a template**:

1. **One opening sentence** naming the architecture and its distinguishing options — a sentence,
   not a title with an `====` underline.
2. **Prose explaining the principle**: what problem the architecture solves and *why its mechanism
   resolves it*, not just what the layers are. Inline math in backticks (`` `y = F(x) + x` ``)
   where an equation carries the idea.
3. **Prose on the architecture itself**: the stage/block structure, the design trade-offs, and —
   importantly — the places where the code does something non-obvious, and why. This is the part a
   reader would otherwise get wrong.
4. **Every deliberate behavioural choice, stated as a choice with its reason** (for example, why
   `pretrained=True` raises rather than warning and returning a random model).
5. **A `References:` section** listing papers as `- Author et al., YEAR. Title. (url)`, including
   the papers the design actually draws on, not only the headline one.

This **replaces** terse `Model Variants:` / `Usage Examples:` boilerplate that restates the
`MODEL_VARIANTS` dict and the factory signature sitting directly below it. The docstring's job is
the reasoning that is *not* in the code. Length follows the architecture; do not pad, and do not
move real explanation into the README to hit a line budget — benchmark tables and usage
walkthroughs are what belongs in a README.

Then imports, a `# local imports` banner, `# -----` separator bars, and the registration decorator:

```python
import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
```

### 8.2 Class API

```python
@keras.saving.register_keras_serializable(package="dl_techniques")
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

- `call(self, inputs, training=None)` — **no logging inside**. It fires on every trace.
- `get_config()` returns every constructor argument, with `keras.regularizers.serialize(...)` for
  regularizers; `from_config()` deserializes them.
- `from_variant(cls, variant, ..., pretrained=False, **kwargs)` looks the name up in
  `MODEL_VARIANTS` and raises `ValueError` **listing the available keys** when it misses. It must
  accept the overrides its own docstring advertises — several packages raised `TypeError` on
  exactly the documented override — and it must not splat description metadata into the
  constructor.

**Two variant tables, deliberately.** `MODEL_VARIANTS` is the canonical name for the registry of
publicly named variants. Where a package predates this and uses another spelling as its *only*
table, add `MODEL_VARIANTS` as a class-level **alias** to the same dict — never rename, because
trainers and tests reference the old spelling and the rename buys nothing the alias does not.

`SCALE_CONFIGS` is **not** a stale spelling and the two must not be merged where both appear. They
answer different questions:

- `SCALE_CONFIGS` is the **architecture table**: `'tiny' -> {hidden_size: 192, num_layers: 12,
  num_heads: 3, ...}`.
- `MODEL_VARIANTS` is the **public-name registry**: `'beit_tiny' -> {scale: 'tiny'}`, one row per
  name a caller may pass, resolving to a scale.

Merging them would collapse a name→scale indirection that exists precisely so a variant can pin a
patch size or an input resolution alongside its scale.

**Variant tables are derived from a named reference** — the released checkpoint's own
`config.json`, fetched and cited — never from a sibling file in this repo and never from a paper
table read once. Variant tables in this tree have been wrong by roughly half the parameter count in
their own name, with the test suite pinning the wrong values.

### 8.3 Pretrained weights

`load_pretrained_weights(weights_path, skip_mismatch)` loads from a **local path**, building the
model with a dummy forward pass first if needed.

There is **no `by_name` parameter** — Keras 3 removed it from `Model.load_weights`. Transfer here
is layer-by-layer and therefore always name-based. The argument survived for a while as a supposed
no-op; in fact every `load_weights(path, by_name=True)` call *raised*, and the enclosing `except`
turned it into a warning and continued with random weights. Do not reintroduce it.

`_download_weights(...)` **raises `NotImplementedError`, naming the variant and showing the
local-path alternative.**

**Never** write a placeholder URL table plus a `try/except` in `from_variant` that logs a warning
and continues with random initialization. That combination means `pretrained=True` silently returns
an untrained model: the caller asked for trained weights, got random ones, and got no error. Nine
packages shipped this simultaneously. Never swallow a load failure into a warning — a local-path
load that restores nothing must raise.

### 8.4 Factory and exports

A module-level `create_<name>(variant="<default>", ...)` that delegates to `from_variant` with **no
logic of its own**. The package `__init__.py` exports the class and the factory with a curated
`__all__`.

### 8.5 Hygiene

- No comment that restates the line below it (`# Store configuration`, `# Squeeze`).
- No `# 1. / # 2. / # 3.` step ladders. A comment earns its place by explaining *why*, or by
  recording a non-obvious constraint — not by narrating *what*.
- No mutable default arguments; `None` sentinels resolved in the body.
- No unused imports — an imported-but-never-called `logger` is the common case.
- Prefer `keras.ops`; `keras.config.floatx()` / `keras.config.epsilon()` over `keras.backend.*`.
- Centralized logging via `dl_techniques.utils.logger`; never `print`.

### 8.6 Things you must not do

- **Never delete or reword a `# DECISION <plan-id>/D-NNN` comment.** Supersede it in place with a
  dated note. Files with a high comment density are often dense *because of* these anchors — never
  target a file by comment density.
- **Never rename a module file to `model.py`** to match the template; it breaks every import.
- **Never convert docstring style wholesale.** Match the file you are editing. `layers/` is
  predominantly Sphinx/reST, and this is effectively mandatory in `layers/attention/`. `models/` is
  measurably mixed; for a **new** package follow `src/dl_techniques/models/bert/bert.py`, which is
  entirely Sphinx/reST.
- **Never delete a deliberate late-import re-export shim** carrying `# noqa: E402` at the bottom of
  a module.

### 8.7 When the shape does not apply

- **No genuine named variants** — do not invent a `MODEL_VARIANTS` table to satisfy the template.
  Apply §8.1, §8.4 and §8.5 only, and say why in the package README.
- **Functional builders** that return `keras.Model(inputs, outputs)` and have no subclass stay
  functional. Converting them breaks existing checkpoints. §8.1, §8.4 and §8.5 still apply.
- **Multi-model families and nested packages** apply the shape per *inner architecture*, not per
  directory.

Before classifying a package as functional, verify with
`grep -n "^class .*(.*Model)" <pkg>/*.py`. A grep-based census of this question was wrong about
several packages.

---

# Part II — Reuse before you author

## 9. The factory registries

**Authoring a bespoke layer is the last resort, not the first move.** Check in this order and only
move to the next step when nothing fits:

1. **The domain factory.** Each exposes a `create_*_layer()` entry point over a registry of named
   types. Pass a type string plus config; do not hand-roll what a factory already builds.

   | Domain | Entry point | Types |
   |---|---|---|
   | Normalization | `create_normalization_layer()` — `layers/norms/factory.py` | 18 |
   | Attention | `create_attention_layer()` — `layers/attention/factory.py` | 32 |
   | FFN / MLP | `create_ffn_layer()` — `layers/ffn/factory.py` | 21 |
   | Activations | `create_activation_layer()` — `layers/activations/factory.py` | 22 |
   | Embeddings | `create_embedding_layer()` — `layers/embedding/factory.py` | 13 |
   | Logic | `create_logic_layer()` — `layers/logic/factory.py` | 4 |
   | Mixtures | `create_mixture_layer()` — `layers/mixtures/factory.py` | 3 |
   | Sequence pooling | `create_sequence_pooling_layer()` — `layers/sequence_pooling/factory.py` | 3 |
   | Heads | `create_head()` / `create_nlp_head()` / `create_vision_head()` — `layers/heads/factory.py` | task-typed |
   | Memory | `create_ntm()` / `create_mann()` / … — `layers/memory/factory.py` | constructor set |
   | Masks | `create_mask()` — `utils/masking/factory.py` | constructor set |

   **Transformer blocks have no `create_*_layer` factory.** Use `TransformerLayer` from
   `layers/transformers/transformer.py` directly — it is highly configurable (selectable attention,
   FFN and normalization types, and normalization position) and composes the factories above
   internally, so it covers most cases without a custom block. Higher-level `create_*_encoder`
   builders exist alongside it.

2. **The broader `layers/` package** — 20+ subpackages of standalone layers.

3. **Only then a new layer**, placed in the appropriate domain subpackage and registered in that
   subpackage's `factory.py` where one exists, so the next author can reuse it too. Do not bury a
   new layer inside a model directory.

### 9.1 A factory's registry is a frozen public surface

Where a factory declares its registry frozen, the key set, the `Literal` type aliases and each
entry's `required_params` / `optional_params` are **public API** consumed by config-driven callers
and asserted by drift tests. Adding, renaming or removing any of them is a breaking change, not a
cleanup. Docstrings and comments may be improved freely; the data may not.

Some registry entries deliberately map to module-level **functions** rather than classes, where the
function pins a mode the class itself does not encode. A configuration reachable by passing an
argument to the general class is deliberately **not** registered — do not add keys "for
consistency", because that grows the frozen surface.

### 9.2 Unknown keys: the factories raise, and it took work

All the main factories now **raise `ValueError` on any keyword the target type does not declare**.
Confirmed by execution on the current tree:

```
create_attention_layer('multi_head', dim=32, num_heads=4, bogus_key=1)
  -> ValueError: 1 unsupported parameter(s) ['bogus_key']
create_ffn_layer('mlp', hidden_dim=32, output_dim=16, bogus_key=1)          -> ValueError
create_normalization_layer('layer_norm', bogus_key=1)                       -> ValueError
create_activation_layer('gelu', bogus_key=1)                                -> ValueError
create_embedding_layer('positional_learned', ..., bogus_key=1)              -> ValueError
```

This is worth stating as history, because it is the single most productive rule in this document.
Before the factories were hardened they **silently filtered** unknown keys against the target type's
allowlist and dropped the rest. A misspelled or undeclared key produced a valid layer carrying a
default value, and nothing raised, warned, or logged. The measured consequences:

- Four call sites spelled a dropout argument `dropout=` where the registry declares
  `dropout_rate`. Positional dropout was **repo-wide dead** across every vision encoder that used
  them, building `Dropout(rate=0.0)` no matter what the caller passed.
- One model forwarded `max_seq_len` and `rope_theta` into an attention type that declares no RoPE
  parameter. Both keys evaporated, and the entire reasoning stack was exactly
  permutation-equivariant.
- `qkv_bias=True` built a layer with **zero bias weights**, because the declared spelling is
  `use_bias`.
- Hardening one factory immediately exposed four more live sites that had been silently discarding
  a normalization choice on every construction.

**Therefore:** when you add a factory or a registry entry, it raises on undeclared keys. When you
add a new registry-backed dispatch anywhere, it raises. And when you *use* a factory, do not assume
a key was accepted because nothing complained — the guard is a scoped weight-value probe on the one
subtree the knob is meant to reach (§19.6), not a whole-model output diff.

### 9.3 The inverse: hand-written kwarg lists that omit a key

Hardening the factory does not catch the other shape, where the call site hand-writes its argument
list and simply **omits** a key it holds in `self`. Nine such sites were measured at exactly
`0.000000e+00` weight delta: attention projections that never received `kernel_initializer` or
`use_bias`, patch embeddings that never received initializers or regularizers, a final norm that
never received `epsilon`.

When you audit "who calls factory X", also sweep **"who builds X's argument dict without calling X
directly"**. Files that assemble an `ffn_args` dict and pass it to a wrapper are invisible to an
AST call inventory, and a suite sweep run at site defaults cannot see the break either, because it
needs a non-empty caller dict to appear. A `**kwargs`-splat site has the identical blind spot.

### 9.4 An "optional" parameter the layer derives is not safe to pass

Consult the registry entry's `required_params` before deciding. One FFN type derives its hidden
dimension from a two-thirds rule plus a multiple-of constraint, while thirteen others require it.
Making the parameter conditional turned an expansion-factor knob into a no-op for eight types — and
the change shipped with a guard that asserted that invariance **as correct**, pinning the new
defect. Forward the derived parameter registry-driven, and pin the layer's own derivation rather
than pinning invariance.

### 9.5 Normalization epsilon

The factory sets `epsilon=1e-6`. Keras' `LayerNormalization` and `BatchNormalization` default to
**`1e-3`**. Both figures confirmed by execution on the current tree. That is a factor of 1000 in
every denominator, with no shape symptom, no warning, and no test failure.

Direct construction has put a norm at `1e-3` inside a stack whose other norms ran at `1e-6` — a
1000x spread inside one forward pass. A related port had the reference epsilon reach 1 of
`2*num_layers+1` norms, and an earlier one had 86 of 114 layers silently wrong with every test
green.

**Route normalization through `create_normalization_layer`.** If you must construct directly,
`epsilon=` is mandatory and must cite a named reference.

Do **not** blanket-fix this. Some architectures' reference implementations genuinely use `1e-3`.
And when you do sweep, sweep **every** epsilon-owning sub-layer rather than naming one.

## 10. Porting from a reference implementation

A port's failure mode is not a missing layer. It is a numerically different layer wearing the right
shape.

- **Constructor defaults are not the reference's structure.** Audit every implicitly-defaulted
  numeric hyperparameter of every framework primitive the port touches — epsilon, momentum, the
  activation's own constants. Fix additively (a new keyword argument, default unchanged) so
  existing consumers do not move.
- **`padding='same'` is asymmetric in Keras and symmetric in PyTorch.** At stride > 1, two branches
  of the same block with different kernel sizes sample **different input pixels**. Measured with
  Dirac kernels, a `k=1` branch read one 2x2 patch of a feature map while the `k=3` branch of the
  same block read a different one. Shape assertions cannot see this. Apply a symmetric padding mode
  uniformly at every port site rather than at the one that was noticed.
- **Vendor the reference.** Put the reference implementation's config or source under `research/`,
  off the import path, and have the test read it with `ast` or `json`. An oracle you transcribed by
  hand in the same session as the port is a second copy of your own understanding, not a
  reference (§21.3).
- **A class that shares a name with a reference is not necessarily that architecture.** This tree
  contains a `RepMixerBlock` that is a different architecture from the FastViT block of the same
  name, and several packages whose names misattribute the architecture they implement. Check the
  composition rule, not the name.

---

# Part III — The failure catalogue

Every family below shipped behind a green test suite. Each is stated as *symptom → mechanism → why
the obvious test is blind → the rule*.

## 11. Inert configuration

### 11.1 The dead knob

**Symptom.** A parameter is validated, stored, serialized and documented, and changing it changes
nothing. The constructed layer is valid; it is just not the one requested.

**Mechanism.** One of: the knob is never forwarded to the sub-layer that would consume it; it is
read only at build time but the branch is hardcoded; a sibling consumes it and ignores it; or it
mutates a Python attribute that an already-traced function never re-reads.

Measured instances include a normalization-position flag on a model whose encoder block had no such
parameter at all (so `'pre'` built a post-norm stack); a KV-head count printed by `summary()` for a
model whose attention was plain multi-head; an architecture-type argument accepting three values
that no branch consulted, with all three giving 612 parameters and `max|dy| = 0.000e+00`; a
weight-sharing flag that produced identical parameter counts and identical object counts both ways;
and a reconstruction-weight documented as a penalty on a model whose `model.losses` was `[]`.

**Why the obvious test is blind.** The only assertion was a constructor-attribute echo
(`assert model.d_state == d_state`) or a shape check. Both are invariant under the defect.
`get_config` round-trips perfectly, because the value *is* stored.

**Rule.** Every constructor parameter is pinned by a test that varies it and asserts a **measured
difference in weights or outputs**, with an anti-vacuity control. Reading the value back off `self`
is not coverage. If a knob is deliberately inert, delete it, or pin the inertness with
`xfail(strict=True)` carrying the measurement and the reason.

Choose the instrument by knob class — this is the single most common way a knob test goes vacuous.
See §20.2.

### 11.2 Variant tables that are wrong, splatted or unreachable

A `from_variant` has built roughly half the blocks its own name advertises, with the test suite
pinning those wrong counts. Another raised `TypeError` for exactly the override its docstring
advertises. Another splatted its description metadata into the constructor. See §8.2 for the rule.

### 11.3 Cross-parameter contracts enforced only at forward time

A configuration where two dimensions must agree has constructed, validated and **built** cleanly,
then raised on the first forward pass. Another left blocks past a threshold built, parameter-counted
and optimizer-tracked while zeroing them moved the output by exactly `0.0`. See §4.2.

### 11.4 `**kwargs` that never arrives

See §7.1. A key read from `**kwargs` that is also forwarded to `super().__init__()` is dead on
arrival.

## 12. Inert components

The unifying property: the component exists, has weights, has gradients, serializes, and
contributes nothing. Parameter counts, shapes, finiteness and gradient-existence assertions are all
blind by construction.

### 12.1 Positional encoding on the wrong axis

**Mechanism.** `RotaryPositionEmbedding.call` reads the sequence length from `ops.shape(inputs)[2]`
— it expects `(batch, seq, heads, dim)`. Layers that handed it `(batch, heads, seq, dim)` rotated
every token by its **head index**.

**Why this is invisible.** A per-head constant rotation `R_h` is orthogonal and is applied to both
the query and the key of the same head, so `(R_h q)·(R_h k) = q·k`. RoPE was an **exact no-op**, not
a corruption. Shapes, parameter counts, gradients and serialization were all correct. The
cancellation only breaks under grouped-query attention, where a key head is rotated and then
repeated onto a query head with a different index — at which point a silent no-op becomes real score
corruption.

A second layer in the same family passed a tensor with a singleton head axis, which was read as
sequence length 1, so everything was rotated by position 0 — the identity.

**Related shapes.** A model defaulting to rotary encodings constructed the layer, built it,
serialized it, and never handed it a query. An FFT mixer transformed the **innermost** axis, which
is the feature axis, so the only token-mixing operator in the architecture did no token mixing at
all.

**Rule.** Any layer claiming positional or token-mixing semantics carries a **non-cyclic
permutation-equivariance probe**: permute the input tokens, assert the output moves by far more than
float32 noise, with an anti-vacuity arm asserting the logits vary across positions in the first
place. Never infer "RoPE is wired" from the existence of a `self.rope` attribute. Assert the axis
order explicitly at the call site.

### 12.2 Components built and skipped

Register tokens that were R copies of one vector rather than R independent ones. A grouped state
summed over its group axis, making four groups bit-equal to one. Deep supervision that supervised
the head it already had. Each is a component whose *count* is right and whose *identity* is wrong.

**Rule.** Assert the identity, not the count. Two register tokens must differ from each other; two
groups must produce different outputs.

## 13. Build and serialization failures

### 13.1 The `StatelessScope` assign trap

See §3.3. The rule again, because it is the highest-yield one in this document: **never `.assign()`
a constant table inside `build()`.**

### 13.2 `build()` that under- or over-materializes

See §4.

### 13.3 Registry key collisions

See §2.

### 13.4 Public methods that bypass lazy build

A model's `encode_image` / `encode_text` did not route through `__call__`, so on a freshly
constructed model a `build()`-created variable was still `None` and the method died inside a
broadcast with a message that never mentioned the model being unbuilt.

**Rule.** Every public method that reads a `build()`-created variable calls an `_ensure_built()`
that resolves shapes from the constructor configuration. Every `train_step`, `test_step` and
`evaluate` override has at least one test that actually executes it — one such override used a
Keras 2 API that does not exist in Keras 3 and was reachable by any `fit()`, while the suite was
forward-pass and save/reload only.

### 13.5 Output structures that break `predict`

`predict({"input_ids": ...})` has raised "Structures don't have the same nested structure" because
`call` echoed a bare `None` mask back in its output dict.

Fix this at a **single site with one rule for all models**. Resolving the mask earlier is not a
no-op for every architecture: for one model it measured exactly `0.0`, and for a sibling it measured
`6.4e-01` on an output whose max magnitude was 2.67, because a windowed attention zero-pads a rank-2
mask up to its synthetic grid. A per-model placement rule is its own trap.

## 14. Composition failures

The architectures whose entire value is *how blocks compose* are the ones where composition is never
tested, because shape, parameter count, finiteness and gradient existence are all invariant under a
broken composition rule.

### 14.1 Transform-only blocks called without the external residual

Some blocks in this tree compute a **transform** and document that the *caller* supplies the skip
connection. Calling them as `x = block(x)` drops the residual. Measured: signal collapse of roughly
`1e-5` per block, and a layer-scale initialization of 1.0 did **not** rescue it.

**Rule.** Read the block's docstring for who owns the residual. Assert a post-ladder magnitude:
`std(out) / std(in)` must stay near 1 across the stack.

### 14.2 A residual block whose `gamma → 0` limit is zero, not identity

A learnable multiplier applied to the whole block output *after* the block had already closed its
own residuals gives `x = gamma * f(x)` with no skip. Measured `std(out)/std(in) = 4.97e-05`, restored
to `1.0000` by the fix.

**Rule.** For any block with a residual scale, assert the limit directly: as the scale goes to zero
the block must approach the **identity**, not zero. This is a two-armed pin — `gamma=0` is exactly
the identity, and `gamma` at its shipped initialization is measurably *not* the identity — because
an identity-only assertion is also satisfied by a block that returns its input.

### 14.3 Sub-blocks that share an input, or stacks that read only the last

A fractal block applied both of its depth-`k-1` sub-blocks to the **same** input, so
`F_k(x) = 0.5*(DP(F_{k-1}(x)) + DP(F_{k-1}(x)))`. Every input-to-output path traversed exactly
**one** convolution at any depth; a `depths=[4,5,5]` configuration was 8/16/16 parallel convolutions
instead of a fractal. Twenty tests passed against the broken rule.

A graph transformer computed its local tokens once and handed the same tensor to every block,
reading only the last — so increasing the block count deepened nothing. The code said so, in a
comment and in the module docstring, as though it were a permanent property.

**Rule.** For any architecture whose value is composition, assert composition **directly**:
receptive-field growth, or a **non-local** probe (perturb an interior input pixel, require the
spatially opposite corner of the final feature map to move). Use scale-free assertions — a
downstream renormalizing layer defeats magnitude checks. And treat a comment explaining why the
architecture does not do what its name says as a defect report, not as documentation.

## 15. Numerics

### 15.1 fp16 mask sentinels

`scores + (1 - mask) * (-1e9)` is the single most replicated numerical defect in this tree.
Confirmed by execution: `np.float16(-1e9)` is `-inf`, and `0.0 * -inf` is `nan`.

So under `mixed_float16`:
- a fully-masked row softmaxes to NaN, and
- **an unmasked position computes `0.0 * -inf = NaN`.**

The corruption lands on the positions the mask is meant to **keep**, which is why a guard that
checks "the masked positions are ignored" misses this family entirely.

**Rule.** Derive the sentinel from `self.compute_dtype` — `-1e4` is finite in float16 (confirmed:
`np.float16(-1e4) == -10000.0`). Better, express the mask as `ops.where(keep, scores, bias)` rather
than as an additive product, so there is no `0 * -inf` term at all. Every layer with a mask or a
reduction gets a `mixed_float16` **and** a `float64` construction-and-forward test.

Two follow-on traps:

- **Sub-layers autocast.** A float32 tensor entering a sub-layer under `mixed_float16` is float16
  inside that sub-layer's `call()`. A fix that only changes a dtype does not survive the sub-layer
  boundary; the fix must change the **predicate**. For the same reason, a claim about a sub-layer's
  dtype cannot stand in for a claim about the model's.
- **"Run this reduction in float32" is relative to the input dtype.** Under a float64 policy the
  identical instruction **narrows** the computation — one measured case went from worst-case error
  `1.31e-15` to `1.99e-08` with every test still green. Use a never-narrow guard
  (`max(input_dtype, float32)`), not a hard-coded literal.

### 15.2 Degenerate lengths return NaN instead of raising

A reduction over a band or window whose length can be 0 or 1 fails silently and
**execution-mode-dependently**: in eager, `ops.min`/`ops.max` over a zero-length axis raise; under
`@tf.function` they return `±inf` without raising, and `ops.mean`/`ops.var` return NaN.

Measured: a model returned an **all-NaN forward pass at initialization** while its test asserted only
`output.shape == (4, 24, 7)` and was green the whole time. A convolutional model whose downsampling
stages all use `padding='valid'` produced an all-NaN output **of the correct shape** whenever a
stage collapsed an axis to length zero — and its own shipped docstring example did exactly that.

**The static-shape guard does not close it.** A guard written as `if dim is not None and dim < 2`
short-circuits and never fires on a `[None, None, C]` trace. `InputSpec` cannot close it either
(§4.1). And `ops.cond` on a traced shape value raises `OperatorNotAllowedInGraphError`.

**Rule.** Branch in Python on `tensor.shape[axis] is None` — a trace-time test on a Python object —
and repair at the **value** level in the dynamic branch. Validate the minimum spatial or sequence
extent in `__init__`, computed from the variant, never hard-coded.

Then check what your repair does to real NaNs: one value-level repair was rewriting genuine NaNs to
`0.0`, so a corrupt window looked like a constant one. Keep the repair off the static path where the
length is known good.

**Every forward test asserts `ops.all(ops.isfinite(y))`, never just `y.shape`.**

### 15.3 Wrong parameterisation, sign, direction or scaling

A representative set, all of which passed shape-only suites:

- A "reflection" gradient mode returned `x - 2(x·w)w`, a Householder reflection that maps `u → -v`.
  The forward output was **exactly `-q`**. The suite parametrized all four modes and asserted output
  shape, which is invariant under a sign flip.
- A codebook lookup rescaled its row by `x_mag / q_mag`, so a "discrete" bottleneck leaked a
  continuous magnitude channel and `decode(encode(x)) != model(x)` in 2048 of 2048 elements.
- An EMA codebook normalized as `ema_embeddings / (ema_cluster_size + eps)` with counts starting at
  zero and the numerator at the initializer, so step 1 gave roughly `99000 * init`. Debiasing alone
  made it **worse**; the fix needed zero-initialization too.
- A probabilistic forecaster de-normalized sigma with `sqrt(scale)` while mu used `scale`. The tell
  was that the branch below it used the correct form — the `sqrt` had been carried across by copy.
- A score field fed `denoised - noisy` as an epsilon estimate; the cosine with the correct direction
  measured **-1.0**, so the navigation routine was doing gradient *descent* on log p. A bare sign
  flip would still have been wrong, because the variance-preserving parameterisation carries a
  factor the familiar variance-exploding form drops.
- A decay schedule counted in batches against a counter in samples, producing a **negative** learning
  rate for 9 of 10 batches — and a negative rate moves every neuron *away* from its input, so the map
  anti-organised.

**Rule.** Pin the **invariant**, not the shape:
- homogeneity (scale the target by `k`, assert mu **and** sigma scale by `k`);
- sign-discriminating distance comparisons (distance-to-`+q` versus distance-to-`-q`, which cannot
  be satisfied by loosening a tolerance);
- agreement between the forward path and any two-stage public API (§15.4);
- a numerical central-difference check against the closed form.

A reviewer can be right about the defect and wrong about the mechanism in the same sentence. Run the
prescribed fix and diff the **number**, not just the shape, before believing it.

### 15.4 Two producers of the same quantity

Where a layer has both a `call()` and a public two-stage API
(`encode_to_indices → quantize_from_indices → decode`), the two compute the same thing by different
code. One gets fixed and the other drifts — the same invariant was violated twice in sibling
branches of one file, and in a third case the disagreement was `1.92e-03` against a suite whose only
bound was an `atol=1e-4` twice as loose as the defect.

**Rule.** Ship a parametrized value-equality test across every mode, with a vacuity arm pinning that
the fixture is in a regime where the two *could* differ.

### 15.5 Build-time and runtime shape arithmetic

See §5.

## 16. The training path

### 16.1 Custom `train_step`

**This repo's standing constraint is: do not write a new custom `train_step`.** Use stock `fit()`
and feed extra signals through `tf.data` inputs. The rules below apply to the ones that already
exist.

**Mixed-precision loss scaling.** Keras' default TF `train_step` calls
`optimizer.scale_loss(loss)` inside the tape, and `LossScaleOptimizer.apply()` divides every
gradient by `dynamic_scale` **unconditionally**. Overriding `train_step` opts out of the first and
keeps the second. Measured over 10 SGD steps, total `|dW|`:

| configuration | total `|dW|` |
|---|---|
| `mixed_float16`, as shipped | 8.740626e-05 |
| `mixed_float16`, with `scale_loss` | 2.436617e+00 |
| float32 control | 2.739021e+00 |

A ratio of 2.79e+04, which is 2^15. Nothing raises and nothing warns; training simply does not
move.

**Rule.** Any custom `train_step` calls `self.optimizer.scale_loss(loss)` inside the tape and sums
`self.losses` via `self.compute_loss`, and carries a `mixed_float16` A/B on total `|dW|` over N
steps against a float32 control. If the model applies gradients through raw optimizer attributes
rather than `self.optimizer`, `scale_loss` is an inert no-op — say so in a comment, or the next
reader re-adds it.

Two clarifications that were initially got wrong:

- Keras 3.8's default `compute_loss` **already sums `self.losses`**, so a custom `train_step` does
  not automatically drop regularizer terms. The AST predicate "does the body mention `self.losses`"
  measures the wrong thing.
- A real instance did drop them: four overrides summed `self.quantizer.losses` **only**, so a
  caller's encoder `kernel_regularizer` reached neither the gradient nor the reported loss —
  identical loss to six digits with and without an `l2(1e-1)`.

`model.losses` is never empty in this tree (several blocks hardcode a layer-scale L2), so
`assert model.losses` always passes. Assert a **delta** against a no-regularizer baseline.

### 16.2 Python state that never reaches the traced graph

An accumulation counter held as a Python `int` and compared with a Python `if` inside a
`@tf.function` folds to `False` at trace time, and `apply_gradients` is **never emitted into the
graph**. A schedule that sets a Python attribute is never re-read by an already-traced train
function.

**Rule.** Any state that must vary across steps is a `keras.Variable` released by `ops.cond`. Any
schedule that changes a **shape-determining** value cannot be carried by a variable at all and
requires an explicit retrace. Verify by asserting `optimizer.iterations` advances as expected — at
accumulation 2 the sequence is 0, 1, 1, 2 — not by reading logs.

Related: Keras 3 spells it `reset_state`, not `reset_states`. And an EMA clone made from an
**unbuilt** subclassed model means `set_weights(model.get_weights())` is `set_weights([])`, which
silently succeeds.

### 16.3 A zero gradient is not a freeze

Under AdamW, moment estimates keep drifting a gradient-masked parameter — measured `6.8e-3` over two
masked steps at `lr=1e-2` — and decoupled weight decay moves it by exactly `wd * lr` per step with
no gradient at all. The only exact freeze also zeroes that group's learning-rate variable.

Also: `model.trainable = False` empties `trainable_variables` but leaves each
`tf.Variable.trainable` at `True`, so a `GradientTape` still auto-watches them.

**Rule.** Verify a freeze by **bit-identical weights across steps**, never by a zero gradient.
Assert `trainable_weights == []` **and** an empty `tape.gradient(...)` — not
`tape.watched_variables() == ()`.

Marking previously-trainable weights non-trainable makes a `.keras` optimizer-state resume skip
optimizer loading entirely, with only a `UserWarning`.

### 16.4 Optimizer and callback traps

- `optimizer.learning_rate` is the schedule **evaluated at the current `iterations`**, not the
  schedule object (which lives on `_learning_rate`). Assert liveness by driving `iterations`, not by
  an `isinstance` check.
- A checkpoint-selection helper that infers direction from the metric name
  (`mode = 'max' if 'accuracy' in monitor else 'min'`) silently selects the **worst** epoch for a
  metric like `val_box_iou`. Audit both the metric and the direction.
- `EarlyStopping(restore_best_weights=True)` plus a `ModelCheckpoint` on the same metric make the
  "best" and "final" checkpoints the same epoch **by construction**, so their bit-identity proves
  nothing.
- A zero-initialized last projection back-propagates **exactly zero** gradient into every weight
  behind it. Measured: 7 of 9 upstream variables at exactly `0.0`. This is normal at
  initialization for some architectures and is also how a dead stack looks.

### 16.5 `pretrained=True` returning random weights

See §8.3. Nine packages shipped it simultaneously, and the documented alternative was also broken.

**Guard it by AST shape** — no `if pretrained:` branch may consist solely of logger calls — plus a
behavioural arm over auto-discovered factories. A string-matching guard ("no placeholder URL
appears") sees nothing when the sites have no URL table at all; that exact guard passed on all nine.

## 17. Causality and masking

### 17.1 The missing causal mask

**Symptom.** A model documented and trained as decoder-only attends bidirectionally.

**Mechanism.** `call` forwards only `attention_mask` — a **padding** mask, `None` by default — and
the attention layers mask only `if attention_mask is not None`. Under a next-token objective the
model has seen the answer.

Found in several language models at once; a grep for `causal|triu|tril|j > i` across one package
returned **docstrings only**. A text tower attended bidirectionally and then pooled "the last
non-padding token, because it is the only one to have seen the whole sentence" — a statement true of
*every* position in a bidirectional tower.

**Why the obvious test is blind.** The existing test was `test_attention_mask_functionality`, which
asserts only that masked and unmasked outputs differ. Any mask satisfies that. Loss curves look
normal.

**Rule.** Every causal model carries a three-armed future-leak probe:

1. perturb token `t`; assert positions `< t` are **bit-identical** (exactly `0.0`, not "small" — an
   attention weight of exactly zero on a masked key contributes exactly nothing);
2. assert positions `>= t` still move (anti-vacuity — without this, `0.0` proves only that the model
   ignores its input);
3. a negative control with an all-attend mask, proving the isolation is attributable to the mask.

Pass the causal mask at **rank 3**. A rank-2 causal mask is silently reinterpreted as a padding mask
by grouped-query attention.

### 17.2 Pooling a causally isolated position

Several classifiers pooled token 0 of a causal model — a token that has seen nothing but itself. The
mirror-image error is pooling the last token, justified by a causality that did not exist.

**Rule.** Pooling strategy and attention causality are **one decision**. Assert that the pooled
representation depends on more than one input token: perturb an interior token, assert the pooled
vector moves.

### 17.3 Masking by zeroing both sides

Masking a metric by zeroing both the prediction and the target makes `zero == zero` always agree. In
one file the sequence-level metric was sound while the per-step metrics understated error 3x from
this idiom.

**Rule.** Mask by **excluding positions from the reduction**, not by zeroing both sides of a
comparison. Check what the reduction does with a masked position.

### 17.4 Repair granularity

A degenerate-row rescue must operate over the **full axis the softmax reduces over**. A per-tile
rescue inside an online-softmax loop read every strictly-upper causal tile as degenerate and
**un-masked the future** — a 24.14 deviation — while every finiteness test passed.

Remember also that softmax is invariant to a constant shift along its reduction axis, so a large
delta after a masking change can be uniform garbage on both sides. Add a control proving the
pre-change output was itself meaningful before calling a delta a regression.

---

# Part IV — Proving it works

Every defect in Part III shipped behind a green suite. This part is how to write a suite that would
have caught them.

The governing observation, measured repeatedly across this library: **a guard that cannot fail is
the most likely outcome of writing a new test, not an edge case.** One freshly written 140-test
suite contained 12 vacuous tests, including *every* forward-pass test and *both* gradient-flow
tests. Another probe suite measured 57 of 57 guards blind. Budget the work to prove your guard can
fail; it is not optional polish.

## 18. The five house rules

1. **Every number is measured or derived, and the derivation is written down.** A tolerance carries
   its measurement — date, device, dtype policy, configuration, the number — and the defect signal
   it must sit below.
2. **Every guard is proven RED**, by an injected mutation or a recorded bisect, *in the committed
   record*. A demonstration in a scratch session that is then discarded is exactly the shape this
   library keeps rediscovering.
3. **Every "nothing changed" assertion has a "something changed" twin.** An identity assertion
   alone is satisfied by a completely dead component.
4. **Process-global state is owned by exactly one fixture** that captures it, restores it in
   `finally`, and asserts the restoration — plus a canary that fails the *next* test if it leaked.
5. **A failing guard is never repaired by widening the tolerance**, unless the bound is proven
   *unattainable in the output dtype*. Unattainable → re-derive. Attainable but flaky → fix the
   cause; widening is forbidden. Write the distinction into the test.

## 19. The instruments

### 19.1 The `.keras` round trip, on values

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

Three details that are load-bearing:

- **`rtol=0`.** `assert_allclose`'s default `rtol=1e-7` silently contributes to a nominally-`atol`
  bound. In one measured case it contributed `1.24e-05` of a `1.53e-05` failure, making the stated
  `atol` decorative.
- **`training=False`, passed explicitly.** A bare `model(x)` is not inference; stochastic-depth
  layers short-circuit only on `training is False`.
- **Values, never shapes.** A round-trip test comparing shapes is satisfied by a model that
  restored **zero** weights.

### 19.2 Weight-value comparison, before the first call

The sharper arm, and the one that catches the whole §13.2 family:

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

`atol=0.0` is correct here: restoration is a copy, not a computation.

A weight **count** invariant is blind to an internal-dimension change that reshapes without adding
or removing tensors. Assert the scalar parameter total too.

### 19.3 Build parity, by relative weight path

Catches under-building. Compare the weight *set* produced by an explicit `build()` against the lazy
path, keyed by path relative to the model root:

```python
def _relative(model):
    """Weight paths with the model-root segment stripped, so two INSTANCES compare equal."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)

def test_explicit_build_matches_lazy_build():
    explicit = build_model(); explicit.build((None, 32))
    lazy = build_model();     lazy(np.zeros((1, 32), "float32"))
    assert _relative(explicit) == _relative(lazy)
```

**This comparison requires every sub-layer to carry an explicit `name=`** (§3.2), and the two
models to be constructed by the same builder. Keras auto-increments generated names per instance,
so two separately-constructed models produce `block/w` versus `block_1/w` at *every* unnamed level,
and stripping only the root segment does not normalize that away. Verified: with explicit names the
parity holds; without them it fails for a reason that has nothing to do with build coverage. If you
see a parity failure, check the names before you go looking for a build defect.

**Parity is blind to over-building** — it passes if *both* paths build everything. Pair it with a
direct layout assertion for each `None`/`False` configuration:

```python
def test_no_head_config_builds_no_head_weights():
    """Anti-vacuity sibling: the parity guard above would pass if BOTH built everything."""
    m = build_model(head=None); m(np.zeros((1, 32), "float32"))
    assert not [w for w in m.weights if "head" in w.path]
```

### 19.4 Build through a parent's `call()`

The only probe that sees the `StatelessScope` trap (§3.3). A test calling `.build()` directly is
exactly the test that missed it.

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

### 19.5 Gradient flow, per variable

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

Non-`None` **and** non-zero, named by `var.path`. A guard written as `assert all(norm >= 0.0)` has
reported green while 61 of 61 trainable weights had identically-zero gradients.

### 19.6 Scoped weight probes for a knob

To prove a knob reached the one subtree it is meant to reach, compare the **weight values of a
named subtree** — not a whole-model output diff, which passes on the broken tree whenever the same
knob reaches other sub-layers by a second route.

```python
def weights_in_scope(model, scope: str):
    return [w for w in model.weights if scope in w.path]
```

### 19.7 Both-ways pairs: causality, permutation, identity

The template. Every "nothing moved" assertion is half of a pair.

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

Two probe-design rules learned the hard way:

- **Never perturb with a DC or uniform per-channel signal** when a per-position normalization
  precedes the guarded reduction. It is vacuous: one such probe measured a leak of `1.9e-06` against
  a real leak of 0.33–1.07. Use fresh non-DC noise.
- **Watch for configurations that make the mechanism structurally unobservable.** A single-layer
  text tower reads its last position, whose causal row is unmasked, so the pin reads exactly `0.0`
  with and without the mask. A deep tower at a small input resolution can collapse its deepest
  attention stage to **one token**, where softmax is identically 1.0. The cheap detector is
  `pytest -W error::UserWarning`, which turns Keras' size-1-softmax warning into a failure.

### 19.8 Orientation: delta impulses and non-square grids

Orientation and direction are invisible to shape, config and serialization tests. A single sign
error in one `ops.roll` survived 249 tests; a fully transposed relative-position bias
(`bias[h, key, query]`) passed 219; a shifted CLS slice passed 91 of 91.

**Rule.** Use a **delta-impulse probe** — a one-hot input, asserting the destination slot — and run
it on a **non-square grid**. A square-only test cannot see a transposed stride.

### 19.9 Precision arms

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

1. **Prove the hazard is real first** — assert that `np.float16(-1e9)` really is `-inf` before
   testing any defence against it.
2. **Realistic sizes.** `N = 512`, not `N = 7`. An `N=7` test hid an fp16 `-inf` that appeared only
   at `N >= 512`.
3. **A float32 or float64 control on the same input**, so "fp16 is noisy" can never masquerade as
   "the NaN bug is detected".
4. **The repair must not weaken what it repairs.** `-1e4` is finite in fp16 but is not `-inf`, so
   assert the masked positions still receive no weight — this arm distinguishes "the sentinel was
   made survivable" from "the sentinel was made ineffective".

Build the reference from the **round-tripped bits**, not the float32 original, so there is no fp16
rounding slack:

```python
def _as_compute(x):
    cd = keras.mixed_precision.global_policy().compute_dtype
    x_c = x.astype(cd)
    return ops.convert_to_tensor(x_c), x_c.astype("float32")
```

A **float64** arm needs more than the policy: `keras.Input` still uses `backend.floatx()`, so the
graph rounds to float32 at the boundary. Also call `keras.backend.set_floatx("float64")` and
**assert `inputs[0].dtype`** — otherwise the arm is a fake reading that agrees with float32 to eight
digits. Note that `UpSampling2D(interpolation='bilinear')` returns float32 for float64 input.

### 19.10 Graph and XLA equivalence

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

For exact-integer paths, use `np.array_equal` and an `input_signature` with `None` dimensions, so
that a value which must stay a static Python `int` is proven to stay one:

```python
@tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
def traced(x):
    return patcher(x)

assert np.array_equal(eager, graph)
```

Where XLA reassociates, the tolerance is **measured**, not guessed, and recorded with its output
magnitude: for example, "measured 0.0151 against an output absmax of ~5.9, i.e. 0.25% relative;
0.05 keeps ~3x headroom while still failing loudly on a NaN or a collapsed output."

### 19.11 Derived tolerances

Where a bound must come from a noise source rather than from a measurement, derive it and write the
derivation in the docstring:

```python
_F32_U = np.finfo(np.float32).eps / 2.0    # unit roundoff
_TAIL_FACTOR = 8.0                          # 8-sigma tail on a random-walk model

def reassociation_atol(reduction_lengths, num_steps: int, scale: float) -> float:
    """Bound on the float32 difference between two REASSOCIATED evaluations of one formula."""
    ops_count = 2.0 * num_steps * float(sum(reduction_lengths))
    return _TAIL_FACTOR * np.sqrt(ops_count) * _F32_U * max(1.0, float(scale))
```

The docstring of such a helper carries: the derivation, a calibration table (derived vs measured vs
ratio), a **RED proof** (injecting the real defect puts the diff orders of magnitude above the
bound), and the instruction that callers must pass `rtol=0`.

A tolerance floor is a claim about a **specific noise source**. Reusing a TF32-derived floor on a
matmul-free path once dominated the real term by 3 to 12 orders of magnitude and passed a
projection that was systematically 1% wrong.

### 19.12 Homogeneity and scale invariance

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

- The **dead-model guard**. Without `denom == 0 → inf`, a model that outputs zeros scores a perfect
  0.0.
- **The model must be fitted for one step first.** Stock `BatchNormalization`'s `moving_mean` is
  exactly 0 at initialization, so an untrained model is *exactly* homogeneous at `training=False`
  and the test passes for a reason unrelated to what it claims.
- **The tolerance sits between the floor and the defect.** `1e-5` is about 9x above the float32
  floor and about 680x below the measured defect signal of `6.8e-03`. Keep the models small and
  fixed: a deeper graph accumulates more round-off, and the headroom is what makes the tolerance
  meaningful rather than decorative.

Note that layer normalization **breaks** degree-1 homogeneity (81–98% error) while a bias-free batch
norm gives it exactly. A generic `verify_bias_free()` structural check gives false assurance here.

## 20. The shared oracles

Three instruments already exist in `tests/`. **Reuse them; do not reinvent them.** All three are
deliberately named **without** a `test_` prefix so pytest does not collect them, and each has its
own RED-proof module.

### 20.1 `tests/test_models/smoke_contract_oracle.py`

Mutation-injects the model's own forward output to prove a smoke contract can reject a broken
model.

```python
DEFAULT_BREAKERS = (collapse_to_scalar, slice_leading_axis, append_trailing_axis)

@contextlib.contextmanager
def broken_forward(model, breaker):
    original_call = model.call
    def _broken_call(*a, **kw):
        return breaker(original_call(*a, **kw))
    with mock.patch.object(model, "call", _broken_call):
        yield
```

Usage contract — the meta-test is mandatory, not optional:

```python
def test_smoke_build_and_forward(model):
    _assert_contract(model(_inputs(), training=False))

def test_the_contract_rejects_a_broken_forward(model):
    assert_contract_rejects_a_broken_forward(model, _inputs(), _assert_contract)
```

Two rules baked in:

- An **anti-vacuity control runs first**: the contract must pass on the unbroken model.
- It requires an **`AssertionError` specifically, never `Exception`**. A `TypeError` from a contract
  that indexed a scalar is the contract *crashing*, not the contract *judging*.

The precedent it replaces: a meta-test that passed an invalid variant name, which fails at variant
lookup *before the model is built*. **A meta-test must break the MODEL, not its argument
validation.**

### 20.2 `tests/test_models/knob_sensitivity_oracle.py`

Three instruments, chosen by knob class. Choosing the wrong one is how a knob test goes vacuous.

| knob class | example | instrument |
|---|---|---|
| **structural** | depth, heads, filters | `assert_structural_knob_changes_weights` — the **weight-shape signature** must change |
| **value** | activation, epsilon | `assert_value_knob_changes_output` — signature identical under the same seed, outputs differ |
| **scoped value** | an initializer honoured in part of the tree | `assert_scoped_value_knob_changes_weights` — weight **values** of a named subtree |

**The trap.** Two models built with different `depth` values have different weight *shapes*, so
they consume different draws from the RNG and their outputs differ **whether or not the argument was
honoured**. An output-difference assertion on a structural knob is satisfied by random-init luck
alone — it is a second unfalsifiable test wearing a stronger-looking assertion.

Capture the weight signature **after** the forward pass. Before it, a subclassed model has
`len(model.weights) == 0`, so two genuinely different configurations both produce the empty
signature `()` and pass. **Raise on an empty signature.**

And the closure gotcha, which silently makes every builder identical:

```python
# WRONG - a bare closure over the loop variable captures the LAST value for every entry
builders = {a: lambda: create_model(activation=a) for a in ("relu", "gelu")}
# RIGHT
builders = {a: lambda a=a: create_model(activation=a) for a in ("relu", "gelu")}
```

When a knob measures inert and the fix is deliberately out of scope, pin it with
`@pytest.mark.xfail(strict=True, reason="<measured>: ...")`. It XPASSes loudly when someone fixes
it. A plain `skip` is inert, and deleting the test leaves the gap unguarded.

### 20.3 `tests/test_models/test_sam/dead_component_oracle.py`

- `fit_one_step_moved_variables()` — returns **name sets**, not a count. `moved > 0` is not an
  acceptable assertion: an earlier iteration shipped "118 of 137 moved" figures whose residual was
  never identified.
- `outputs_stop_gradient(model)` — injects `ops.stop_gradient` on every output. A live training path
  **must** then raise, matched **verbatim** against
  `NO_GRADIENTS_MESSAGE = "No gradients provided for any variable"`, never `raises(Exception)`.
- `component_response()` plus the killers `zeroed_variables`, `destroy_negatives`,
  `destroy_positives`, `layer_returns_its_input`.

The rule the module encodes: **every function reports a NUMBER, never a bare boolean verdict alone.
A probe with no number is not a probe.**

## 21. Why guards fail

### 21.1 Budget one mutation per assertion, and check which one fired

Two mutations can fire the **same** assertion, proving one twice and the other zero times. If that
happens, add an isolating mutation. A single guard can need two.

**Judge a RED proof by which assertion fired, by name and `file:line`.** A plan's predicted RED line
or exception *type* is wrong more often than the failure *class* is — in one review, 4 of 8 predicted
REDs were right about the class and wrong about the line. Note that Keras' unknown-kwarg check
raises `ValueError`, not `TypeError`. A test can also die at a setup assertion before reaching its
point, and "red for the wrong reason" reads exactly like a pass.

### 21.2 An injection that moves both sides proves nothing

A mixed-precision injection **passed** because the float32 reference was captured at import from the
same source file the injection modified; a dtype-conditional variant of the same injection fired 44x
over tolerance. Likewise, a default-versus-explicit-default comparison is the same code with the
same weights and cannot be moved by **any** injection.

Compare against a **transcribed pre-change oracle** that bypasses the changed `call()`.

### 21.3 An oracle written by the same hand is a second copy

Five instances in a single porting round. **The tell** is a constant, term or sign in the "oracle"
that only makes sense if you had read the implementation.

Fix: derive the oracle from the **reference**, and reach the implementation through explicitly
signed, named divergence terms (`_reference_params + _port_only_x - _reference_only_y`). The
cheapest form is to **vendor the reference file** under `research/`, off the import path, parsed
with `ast` or `json`, and point the test at it.

Apply the identical suspicion to a **fix round's own new guard**.

### 21.4 An oracle can be wrong before the implementation is

A float64 naive-product oracle disagreed by a relative `6e+261` because the *oracle* underflowed:
`sigmoid(88)` rounds to 1.0, cancelling `1 - p` to zero. `np.longdouble` was still `2.4e-08` off;
only 60-digit `mpmath` settled it. This family recurred three times, each time nearly escalating a
**correct** implementation as refuted.

**Suspect the oracle's precision before suspecting the code.** And an oracle must consume the code's
**actual received bits**, never the intended Python literals — a gap below the ulp arrives
bit-identical. Assert every oracle input is post-cast. Prefer exact arithmetic
(`fractions.Fraction`) where the quantity permits.

### 21.5 Liveness is not correctness

A raw-layout loss moved under **both** destroy probes while returning `0.765062` where the correct
reshape gives `0.693310`. "The component responded" is not "the component is right" — **assert the
VALUE under the probe.**

Similarly, "the output changed" is not liveness for a **conditioning** input. One head demonstrably
read a prompt (objectness moved `5.59e-01`) while the top-k **selection** it was supposed to
condition moved exactly `0.0000` — a term constant across positions cannot change an argsort.

### 21.6 Budget positive liveness arms before running a dead-component probe

The liveness arm must go through the **same detector** as the absence assertion. A probe suite whose
assertions were all absence, shape or parameter-count checks measured 57 of 57 guards blind: every
one of them is satisfied *by construction* when the component emits zero.

### 21.7 A guard goes vacuous when what it watches changes shape

Not wrong — **vacuous**, reporting clean forever. A listener watching for a dropped-key *warning*
reported a clean zero permanently after a later change converted that warning into a *raise*. A
TF32 canary written `if not <flag>: assert ...` ran zero assertions in exactly the modules where a
leak could originate.

**Rule.** Exception-classify and assert expected values; never skip-on-condition. Re-verify a guard
whenever a change alters the **signal** it watches, and note the new failure class at the guard's
docstring.

Related: **a guard's docstring is not its contract — probe the predicate directly**, even for a
guard shipped the same day. One dead-config-field guard advertised receiver-scoped checking; its
regex counted `unrelated_object.field = 5`, a bare `field = 5`, and a `--field-name` substring
inside a log message as "consumption". It was rewritten as an AST walk.

### 21.8 Anti-vacuity on collection

Every repo-wide parametrized guard asserts its subject set is non-empty:

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

### 21.9 A green control on the first run is suspicious

A control that comes back GREEN on its first run is more likely a probe defect than a clean pass.
So is a dead-component injection that "passes" cleanly on the first attempt — one anti-causal-mask
injection transposed itself back into a causal mask. **Re-derive an injection's actual effect
before trusting either green or red.**

Two more ways a RED proof can be structurally unable to fail:

- The effect is achieved by a **different, still-present code path**. Deleting a re-export left a
  class registered, because a sibling import had already run the decorator. Registry **presence** and
  **name binding** are different contracts; split them.
- **An earlier guard on the same call path masks the defect behind it.** Order fixes by masking, not
  by size: a RED proof written against the reported bug reproduces the wrong one, and the masked
  defect's "before" value must be measured at HEAD **plus the unmasking fix**, not at HEAD.

### 21.10 Never `git stash` or `git checkout --` mid-proof

Confirmed destructive five times, including on an **untracked** file, where `git checkout --`
silently no-ops so the next injection stacks on already-corrupt source. Restore from a
byte-compared (`diff -q`) `cp` scratch backup.

### 21.11 A guard that cannot distinguish pathological from unusual destroys correct answers

Test a guard's **false-positive** family as hard as its true-positive one. A finiteness guard on a
cumulative sum looked obviously right and poisoned ordinary exact rows.

Before adding a guard, check whether **the framework already raises**. Keras' `Conv2D` already
raises on a groups/filters mismatch; `MultiHeadAttention` already raises on rank and last-dimension
errors. And falsify a mandated guard by measurement before implementing it — one would have crashed
nine passing tests on provably correct geometry.

## 22. Test anti-patterns

All of these have live examples in this tree.

| Anti-pattern | Why it passes |
|---|---|
| `assert True` / "if we reach here, the call was successful" | asserts nothing |
| **Constructor-attribute echo** — `assert model.d_state == d_state` | proves the constructor stored the argument and nothing else |
| **Shape-only knob sweep** — sweep a semantic knob, assert the output shape is unchanged | invariant under the knob being dead |
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
that a configurable head's activation was `'sigmoid'`; an assertion that a model's feature rank was
2, for a model whose `count_params()` was 0; a gradient-flow test that passed **because of** a
dead-table defect, whose `1e-8` floor cleared only under the broken initialization and would fail
the correct one; an assertion that a dead block's output delta was exactly `0.0`; an assertion that
an echoed attention mask was `None`; and a variant table test pinning wrong per-variant layer counts.

**Re-derive what each assertion is pinned to** before trusting it.

Two more, about the shape of the whole gate:

- **A 100%-passing suite is not evidence that entry points work. Run the CLI.** 249 tests were green
  while both trainers were broken — one raised `IndexError` on every real run, the other had no
  argument parser and started a 100-epoch job on `--help`. Defects **cluster at entry points with
  zero tests**, reliably enough to plan around. Weight "has no tests" at least as highly as "review
  flagged it".
- **Collection-only gating hides RED tests.** An all-skip module reads as a pass; a suite whose
  collection errored can "pass" by running almost nothing. Gating on `--collect-only` once hid 12
  real failures across 8 steps. **Always quote the passed count together with the collected count**,
  and where a pre-existing failure must be preserved, compare the failing **node-id set**, not the
  count.

## 23. Measurement traps

### 23.1 TF32 is this repo's default false model defect

Three confirmed instances. A GPU-only RED with a CPU-green counterpart is a TF32 suspect **before**
it is a bias hunt.

**The diagnostic.** A TF32 artifact is **flat across four decades** of a scaling constant and
**exactly 0.0 at powers of two**. A genuine additive-bias leak decays as `1/c`; clipping grows.
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

**Never disable TF32 at module import.** One import-time call is process-global for the whole
session and made a precision measurement swing by roughly 1000–1500x depending on what else was
collected. Verify precision-sensitive tolerances in both regimes.

And gate on `flag AND device_has_the_feature`: `tensor_float_32_execution_enabled()` reads `True` on
CPU, where TF32 does not exist and the numerics are true float32.

### 23.2 Quote near-zero statistics from CPU

`CUDA_VISIBLE_DEVICES=""`. A GPU disagrees with **itself** run-to-run at about `5e-6`, and across
process launches by exactly `0.228515625` on a fixed, unchanged model, while CPU gives exactly
`0.0`. Pin golden-value probes to CPU, and **do not repair a golden failure by relaxing `atol`.**

### 23.3 Eager-only bit-identity does not license "inert"

Under `@tf.function`, reassociation produces nonzero deltas with **zero** source-level change —
measured `4.77e-07`, `4.23e-04` and exactly `0.0` on three models. The correct control is the
**within-version eager-vs-graph delta on unchanged code**.

Related: a `training=False` equivalence probe is blind to operation-**order** divergence. Moving
dropout from before a norm to after read "bit-identical" at `0.0`, then `0.3953` max delta at
`training=True` under an identical Bernoulli mask. Read the two `call()`s side by side.

### 23.4 Test-order RNG coupling

A test whose statistic reads the process-global Keras RNG is coupled to pytest **collection order**:
the file passes alone and the directory gate goes RED. Merely adding tests to an earlier-sorting
file dropped one magnitude probe from `>1.0` to `0.818`.

**Rule.** Call `keras.utils.set_random_seed(N)` immediately before construction; keep the **shipped**
initializers; record the across-seed spread at the test. Prefer `np.random.default_rng(seed)` for
input data. **Never** tune a synthetic input's sigma until the bar passes.

Note also that Keras 3 hands **one** initializer instance to every same-shaped sibling projection
inside a layer, and the instance materializes its seed once — so identically-shaped weights get
**bit-identical** draws. An "untrained control" built this way had `Q == K` exactly, with entirely
plausible downstream statistics.

### 23.5 Untrained models cannot answer some questions

A zero-initialized gate makes the branch under test inert, so the defect reads as the float32 floor.
A zero-initialized final projection zeroes the gradient of everything behind it. Use seeded
**non-zero** weights and biases — the state a trained model is in. A layer's default
`bias_initializer='zeros'` made two of three masking sites structurally unobservable, and a sampled
path was green with a live defect at a single perturbation scale, caught only by sweeping four.

### 23.6 Set the tolerance from the defect signal, not the noise floor

A tolerance derived from the noise floor is not a tolerance; it only says the computation ran. The
bound must sit **between** the floor and the smallest defect you intend to catch, and the test should
record both numbers.

### 23.7 Never run GPU jobs in parallel

Contention causes false **failures**, never false passes. The same suite measured 21 failed / 77
passed under contention and 89 passed alone. A GPU-contention error reads exactly like a
regression; the tell is `cudaSetDevice() ... out of memory` **at import**. Three parallel explorer
agents once manufactured a false "8 pre-existing failures" premise that serial re-measurement
reduced to zero.

Check `pgrep -fc "\.venv/bin/python -m pytest"` is 1 before believing a red run.

Use a pristine `git worktree` at the true base as a control. A **partial revert** is not a
substitute — one "pre-existing RED" claim survived reverting three files while the suspect change
lived in a fourth.

### 23.8 Patch the defining module

A shadow-import or monkeypatch binds the **importing** module only. Patching a re-exported name
cannot reach the defining module's own call site, and a package `__init__.py` re-export can make a
shadow-import exercise the unpatched code. Patch the **defining** module's namespace.

### 23.9 Exhaustiveness by grid size is not exhaustiveness

"0 violations over 281,604 rows" held while a small targeted counterexample broke the property
immediately. A grid can be structurally blind regardless of cell count: sampling one parameter
uniformly made a per-chunk carry about `1.8e-26`, annihilating the very state the test claimed to
check — and pinning that parameter to a constant then made the factors bit-identical, missing a
mis-index. Derive an attack; do not sweep and hope.

Similarly, a fixture can construct a shape the real pipeline can never emit, passing while the
shipped default combination crashes. **Drive guards through the actual factories and data path.**

And verify an assertion actually **executes on every arm** of a parametrized build. Prefer
**non-local** assertions over shape and finiteness ones.

---

# Part V — Shipping

## 24. Checklists

### 24.1 A new layer

**Construction**
- [ ] `@keras.saving.register_keras_serializable(package="dl_techniques")` with an explicit package.
- [ ] Class name does not collide with an existing registered class; if a generic name already
      exists in the tree, prefixed.
- [ ] All sub-layers created in `__init__`, unconditionally, with explicit `name=`.
- [ ] All configuration stored on `self` in `__init__`; no mutable defaults.
- [ ] Argument validation raises `ValueError` naming the offending value.
- [ ] Cross-parameter contracts that `call()` relies on are re-checked in `__init__`.

**Build**
- [ ] `build()` materializes **exactly** the tree `call()` runs — no more, no less.
- [ ] No `.assign()` of a constant table; tables computed inside an `add_weight` initializer.
- [ ] No `ops.convert_to_tensor` on a constant that is closed over; NumPy in, convert in `call()`.
- [ ] `super().build(input_shape)` last.

**Call**
- [ ] Symbolic only: no `.numpy()`, no Python `if` on a tensor value, no Python loop over a tensor
      dimension, no layer construction, no list mutation, no logging.
- [ ] No `ops.tril` / `ops.triu`.
- [ ] `training=` forwarded explicitly to every sub-layer.
- [ ] No possibly-symbolic `training` reaching `BatchNormalization` or `Dropout`.
- [ ] Mask sentinel derived from `compute_dtype`, or expressed as `ops.where` rather than an
      additive `(1-mask)*-1e9`.
- [ ] Static shape contracts re-asserted here, not only in `build()`.

**Shape and config**
- [ ] `compute_output_shape` implemented, from stored config, working unbuilt.
- [ ] Shape arithmetic in exactly one pure helper, shared by `build`, `call` and
      `compute_output_shape`.
- [ ] `get_config()` returns **every** constructor argument, complex objects serialized.
- [ ] `from_config()` deserializes them; no popping of base keys.
- [ ] Normalization epsilon comes from the factory, or is passed explicitly with a cited reference.

**Reuse**
- [ ] Checked the domain factory, then `layers/`, before authoring.
- [ ] Registered in the domain `factory.py` if one exists; the factory raises on undeclared keys.

### 24.2 A new model package

Everything above, plus:

- [ ] Module docstring is substantive prose with a `References:` section (§8.1).
- [ ] `MODEL_VARIANTS` present, or an alias to the package's single variant table; `SCALE_CONFIGS`
      not merged into it.
- [ ] Variant values derived from a **named reference**, cited.
- [ ] `from_variant` raises `ValueError` listing available keys, accepts its documented overrides,
      and does not splat description metadata.
- [ ] `pretrained=True` raises `NotImplementedError` naming the variant; no placeholder URL table;
      no `by_name`; no load failure swallowed into a warning.
- [ ] Module-level `create_<name>()` delegating to `from_variant` with no logic of its own.
- [ ] Package `__init__.py` exports class and factory with a curated `__all__`, and binds no name
      matching one of its own subpackages.
- [ ] One `logger.info` in `__init__`; none in `call`.
- [ ] No new custom `train_step`.
- [ ] Checkpoint-affecting changes recorded in a shipping document.
- [ ] Tree-wide collection gate run: `pytest tests/test_models/ -q --collect-only`.

### 24.3 The tests, before you call it done

- [ ] `.keras` round trip on **values**, `rtol=0`, `training=False` explicit.
- [ ] Weight-value comparison at `atol=0.0` **before** the loaded model's first call.
- [ ] Build parity by relative `w.path`, **plus** a no-sub-layer layout assertion for each
      `None`/`False` config.
- [ ] Build-through-a-parent probe for every constant table.
- [ ] Per-variable gradient flow: non-`None` **and** non-zero, named by `var.path`, with the
      `len(trainable_variables) > 0` anti-vacuity assertion.
- [ ] Every constructor knob pinned with the instrument matching its class (§20.2).
- [ ] `ops.all(ops.isfinite(y))` in every forward test.
- [ ] Degenerate lengths (0, 1) swept on the static path **and** a `TensorSpec([None, ...])` trace.
- [ ] `mixed_float16` and `float64` construction-and-forward arms, with a float32 control.
- [ ] `@tf.function(jit_compile=True)` versus eager.
- [ ] Causality: the three-armed future-leak probe, if the model is causal.
- [ ] Composition asserted directly, if the architecture's value is composition.
- [ ] Orientation: delta impulse on a **non-square** grid.
- [ ] Every "nothing changed" assertion has its twin.
- [ ] Every guard proven RED by an injection, **in the committed record**.
- [ ] Every parametrized repo-wide guard asserts a non-empty subject set.
- [ ] Every tolerance carries its measurement and the defect signal it sits below.

## 25. Test module layout and naming

### 25.1 Files

- `test_<layer>.py` / `test_<model>.py` — the comprehensive suite for one unit.
- **Single-claim guard files are sentence-named after the claim**, not the unit:
  `test_the_attention_mask_is_honoured.py`, `test_tables_survive_stateless_build.py`,
  `test_the_gates_actually_gate.py`, `test_evaluate_does_not_see_labels.py`.
- **Shared instruments carry no `test_` prefix** so pytest does not collect them:
  `smoke_contract_oracle.py`, `knob_sensitivity_oracle.py`, `dead_component_oracle.py`,
  `_scaffold.py`. Their RED proofs live in a mirrored `test_<name>.py`.
- A package-local `conftest.py` only when there is shared **instrumentation**.

### 25.2 Names

- **Classes**: `Test<Unit><Aspect>` — `TestBeitAttentionBiasOrientation`, `TestFloat32IsTheControl`,
  `TestBuildsExactlyWhatCallRuns`. Guard classes are named after the claim.
- **Functions**: declarative sentences asserting the claim, not `test_X_works` —
  `test_masked_tokens_do_not_reach_the_visible_positions`, `test_gamma_zero_is_exact_identity`,
  `test_the_perturbation_reached_the_model_at_all`.
- **Meta and anti-vacuity tests**: `test_the_guard_…`, `test_the_probe_…`, `test_the_contract_…`.

### 25.3 Module skeleton

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

### 25.4 Session policy

- Seed with `np.random.default_rng(seed)` for data and `keras.utils.set_random_seed(N)` immediately
  before construction. Avoid statistics that read the global RNG.
- House style for exactness comparisons is `np.testing.assert_allclose(a, b, atol=..., rtol=0)`.
  `atol=0.0` for restoration and bit-identity.
- Any process-global setting (dtype policy, TF32, `floatx`) is owned by one fixture that restores in
  `finally` and asserts the restoration.
- **No test writes into the repo-root `results/`.** Route every config through `tmp_path`. Enforce
  this with an autouse fixture that **asserts** — never one that cleans up:

```python
@pytest.fixture(autouse=True)
def no_repo_root_results_writes():
    before = _results_entries()
    yield
    new = sorted(_results_entries() - before)
    assert not new, (
        f"this test wrote into the repo-root results/ directory: {new}. "
        f"route the config through tmp_path"
    )
```

  `results/` is gitignored and untracked, so deletion there is **unrecoverable**. A cleanup fixture
  once destroyed 62 run directories at once, including a published paper's subject checkpoint,
  because relative paths in its log resolved against the repo root rather than the pytest
  `tmp_path`. Cleanup fixtures that delete artifacts are banned outright.

### 25.5 Scoping runs

The full suite takes about 1.5 hours and is the pre-push hook. Do not run it as a routine
regression check. Scope pytest to the modules you changed plus anything that imports what you
touched, and run the tree-wide **collection** gate after any change to a package's public surface.
Reserve the full suite for when it is explicitly asked for.

Do not trust a red run that was not the only pytest on the machine (§23.7).

## 26. Troubleshooting

| Symptom | Likely cause | Where |
|---|---|---|
| `Unknown layer: MyLayer` | missing registration decorator | §2 |
| Two different classes load as each other, depending on import order | bare `register_keras_serializable()` key collision | §2 |
| `Layer was never built and thus has no variables` | sub-layer not built before weight loading | §4 |
| **Reloaded model has zero weights, or matches 0 of N against its donor** | `build()` does not materialize the sub-layer tree | §4 |
| **`count_params()` returns exactly 0** | `Model.build(shape)` on a subclassed model walks no sub-layers | §4 |
| A constant table is all zeros in training but correct in a unit test | `.assign()` in `build()` discarded by `StatelessScope` | §3.3 |
| `InaccessibleTensorError` on `fit()` of an unbuilt model | constant materialized with `ops.convert_to_tensor` in `build()` | §3.3 |
| `TypeError: ('pred must not be a Python bool', True)` under `fit`/`jit` but not eagerly | `ops.tril` / `ops.triu` | §6.1 |
| `OperatorNotAllowedInGraphError` | symbolic `training` into BN/Dropout, or `ops.cond` on a traced shape value | §6.1, §15.2 |
| **All-NaN output of the correct shape** | degenerate-length reduction, or an fp16 `-1e9` mask sentinel | §15.2, §15.1 |
| NaN on the positions the mask is meant to KEEP | `0.0 * -inf` from an additive fp16 sentinel | §15.1 |
| `cannot compute AddV2 as input #1 was expected to be a half tensor` | one-sided cast under autocast; cast both sides | §15.1 |
| **Training does not move under `mixed_float16`** | custom `train_step` missing `optimizer.scale_loss` | §16.1 |
| A knob has no measurable effect | dead knob, or silently dropped at a factory | §11.1, §9.2 |
| `Structures don't have the same nested structure` from `predict` | `call` echoing a bare `None` in its output dict | §13.5 |
| A causal model's loss looks fine but generation is poor | no causal mask; or the head pools token 0 | §17.1, §17.2 |
| Green suite, broken trainer | entry point with zero tests; run the CLI | §22 |
| Test passes alone, fails in the directory gate | global-RNG coupling, or TF32 leaked from an earlier module | §23.4, §23.1 |
| GPU red, CPU green, on a precision assertion | TF32 | §23.1 |
| `cudaSetDevice() ... out of memory` at import | another GPU job is running; the failure is contention | §23.7 |

---

# Part VI

## 27. Refuted claims

Recorded so they do not get re-proposed, and so a rule below is not re-derived from a premise that
has already been falsified by measurement.

- **The nested `List[List[Layer]]` weight-loss trap does not reproduce on Keras 3.8.**
  `_flatten_layers` round-trips every weight regardless of container nesting. A model with a
  `List[List[Dict[str, Layer]]]` structure restored all 65 weights bit-identically. Check that a
  code shape still bites before claiming a guard against it is load-bearing.
- **"Overrides `build()`" is not the discriminating property for round-trip weight loss.** Whether
  `build()` *materializes the sub-layer tree* is. A model that overrides `build()` and ends it with
  a concrete forward pass round-trips cleanly. See §4.
- **A structured-dict `y_pred` does work under stock `compile()` / `fit()` on Keras 3.8.**
  `CompileLoss.build` breaks in exactly one configuration: a **single `Loss` object** handed a dict
  `y_pred`, where it broadcasts across every leaf and then raises `KeyError`. Supply a dict `loss=`
  keyed to the same output names, plus a dict `y_true`. This was recorded as a hard constraint and
  was a true observation over-generalized into a false rule — which is itself the lesson: **a
  constraint in institutional memory can be a true observation over-generalized; re-execute the one
  that is blocking you.**
- **Keras 3.8's default `compute_loss` already sums `self.losses`**, so a custom `train_step` does
  not automatically drop regularizer terms. The real instance of that defect summed a *sub-layer's*
  `.losses` explicitly. An AST predicate "does the body mention `self.losses`" measures the wrong
  thing.
- **`model.losses` is never empty in this tree**, so `assert model.losses` always passes. Assert a
  delta against a no-regularizer baseline.
- **A "silent un-masking regression" measured as a large delta was not one.** Softmax is invariant
  to a constant shift along its reduction axis, and `x - 1e9` in float32 collapses a row to a single
  value (the ulp at `1e9` is 64). Add a control proving the pre-change output was itself meaningful
  before calling a delta a regression.
- **A GPU-only homogeneity RED at `5.063e-04` was the TF32 ulp**, not a bias leak. See §23.1 for the
  diagnostic that separates them.
- **`x + g - stop_gradient(g)` is not the identity** under left-to-right float association — about
  25% of float64 draws differ by up to 1 ulp. Group it as `x + (g - stop_gradient(g))`, which is
  exactly `0.0` in the forward pass with the gradient unchanged. Do not write a
  forward-bit-identity invariant that the association makes unattainable.
- **Several prescribed fixes were themselves regressions**, caught only by running them and diffing
  the number: bias-correcting an EMA codebook without also zero-initializing it made the defect
  roughly 10x worse; forwarding a "dropped" `dropout_rate` stacked a **second** dropout, so a
  requested 0.25 became an effective 0.4375; resolving an echoed mask earlier was a no-op for one
  model and a `6.42e-01` change for its sibling. **Run the prescribed fix and diff the number, not
  just the shape.**

---

*This document supersedes `research/2026_keras_custom_models_instructions.md`.*
