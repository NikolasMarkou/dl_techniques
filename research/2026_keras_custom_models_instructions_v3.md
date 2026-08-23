# Authoring Keras 3 Custom Layers and Models

A reference for writing Keras 3 custom layers and models that are correct, serializable, and
verifiably do what they claim.

## Scope

Making a layer construct and serialize is the easy half, and it is not where the defects are. Three
successive library-wide audits of a large Keras 3 codebase found the same pattern every time:

| Defect | Symptom to the author |
|---|---|
| A parameter validated, stored, serialized, documented — and read by no code path | `max\|dy\| = 0.000e+00` across every legal value |
| A rotary embedding fed the head axis instead of the sequence axis | an **exact algebraic no-op**: `(Rq)·(Rk) = q·k` |
| `add_weight(zeros)` + `.assign()` in `build()` | table stays all zeros in every real model |
| A decoder-only LM with no causal mask | bidirectional attention under a next-token objective |
| A reloaded model that restored **zero** weights | round-trip test passed |
| A "spline" convolution that was a plain linear convolution | 83 tests green; `\|f(2x) − 2f(x)\| = 0.0` exactly |
| A model reloading its optimizer from **zeroed** moments | training resumed, loss curve looked normal |
| A save that wrote every weight **twice** | round-trip delta exactly `0.0`; the file was 2× |
| A seed guarded by a truthiness test | `seed=0` silently unseeded, at the seed the tests used |
| A shipped variant table half its reference's width | every shape assertion passed |

Every one shipped behind a green suite: shapes matched, parameter counts matched, gradients existed,
serialization round-tripped, loss curves looked normal.

**Construction correctness and behavioural correctness are different properties, and only the first
is easy to test.** Sections 1–13 are what to write. Sections 14–15 are how to prove it. They are not
optional polish — a guard that cannot fail is the most likely outcome of writing a new test, and the
audits measured that outcome repeatedly, including in tests written *during* the audits.

### The four surfaces where the defects actually are

Most Keras guidance covers construction and serialization. These four are where the audits found
shipped, green-suite defects, and they are the reason this document is as long as it is:

- **§7 The Save/Load Path** — archive layout, optimizer state, unbuilt saves, and why a symmetric
  override pair is usually two independent decisions wearing one name.
- **§13 Reachability and Provenance** — a knob nothing can reach, a variant table nobody checked
  against its own reference, a weight nothing reads, a layer that computes something simpler than
  its name.
- **§15 Warnings as a Defect Channel** — the framework is already telling you, the default
  instrument under-reports by 5×, and most teams never turn it on.
- **§14.6 Pin the property, not the sample** — five of nine long-standing RED tests in one audit
  were a single defect class: an exact literal pinned against a seed-dealt or sub-ULP quantity.

## How to read this

| You are | Read |
|---|---|
| Writing a new layer | §1–§4, §6–§10, then the checklist in §18 |
| Writing a new model package | all of it; §5 is the package shape |
| Porting a published architecture | §13.2 first, then §9.7 and §10.4 |
| Fixing a bug | §16 to find the pitfall, §14 to build a guard that can fail |
| Reviewing | §16 as the code checklist, §14.7 as the test checklist |
| Auditing a codebase you did not write | §15 first — it is the cheapest yield per hour |

Conventions used below: **❌ WRONG / ✅ CORRECT** code pairs; **Measured:** lines carry a figure that
was observed, not estimated; **Detect:** lines name the probe that catches the defect; **Refuted:**
lines record a plausible claim that measurement killed.

---

## Table of Contents

1. [Core Design Principles](#1-core-design-principles)
2. [Essential Setup and Registration](#2-essential-setup-and-registration)
3. [Layer Implementation Patterns](#3-layer-implementation-patterns)
4. [Graph-Safe Operations in call()](#4-graph-safe-operations-in-call)
5. [Model Implementation Patterns](#5-model-implementation-patterns)
6. [Configuration Management](#6-configuration-management)
7. [The Save/Load Path](#7-the-saveload-path)
8. [Build Materialization and Weight Compatibility](#8-build-materialization-and-weight-compatibility)
9. [Factory Patterns and Layer Reuse](#9-factory-patterns-and-layer-reuse)
10. [Numerics, Precision and Initialization](#10-numerics-precision-and-initialization)
11. [The Training Path](#11-the-training-path)
12. [Causality, Masking and Composition](#12-causality-masking-and-composition)
13. [Reachability and Provenance](#13-reachability-and-provenance)
14. [Testing and Validation](#14-testing-and-validation)
15. [Warnings as a Defect Channel](#15-warnings-as-a-defect-channel)
16. [Common Pitfalls and Solutions](#16-common-pitfalls-and-solutions)
17. [Troubleshooting Guide](#17-troubleshooting-guide)
18. [Summary Checklists](#18-summary-checklists)
19. [Appendix: Refuted Claims](#19-appendix-refuted-claims)

---

## 1. Core Design Principles

### 1.1 The Serialization Lifecycle

A custom layer has four lifecycle events, and they run in an order most authors never see written
down:

1. `__init__` — store configuration. **Create every sub-layer here.** Touch no shapes.
2. `build(input_shape)` — create own weights; materialize the sub-layer tree.
3. `call(...)` — the forward pass. Graph-safe operations only.
4. `get_config` / `from_config` — reconstruct an equivalent object from data alone.

Deserialization runs `from_config` → `__init__` → `build` (from the saved `input_shape`) → weight
restore. **Any sub-layer that does not exist by the end of `build` receives no weights, and nothing
raises.** That silence is the root of most of §7 and §8.

### 1.2 The Golden Rule: Create vs. Build

**Create in `__init__`. Build in `build`. Never create in `build`.**

A sub-layer created in `build()` exists after a live construction and does *not* exist after a
`from_config` reconstruction that Keras builds from a stored shape — because `from_config` calls
`__init__`, and your creation code lives elsewhere.

```python
# ❌ WRONG - the sub-layer exists only on the live path
def build(self, input_shape):
    self.dense = keras.layers.Dense(self.units)
    self.dense.build(input_shape)

# ✅ CORRECT
def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.dense = keras.layers.Dense(units)      # created unconditionally

def build(self, input_shape):
    self.dense.build(input_shape)               # materialized here
    super().build(input_shape)
```

### 1.3 Create Unconditionally, Use Conditionally

A sub-layer whose *existence* depends on a flag produces two different weight layouts under one class
name, and a checkpoint saved under one is unloadable under the other.

```python
# ❌ WRONG - two weight layouts, one class name
if use_attention:
    self.attn = Attention(dim)

# ✅ CORRECT - one layout; the flag gates USE, not existence
self.attn = Attention(dim)
...
def call(self, x, training=None):
    if self.use_attention:          # a Python flag from config: safe, folds at trace time
        x = self.attn(x, training=training)
    return x
```

The cost is a component that legitimately receives no gradient in some modes. **That is not a
defect** — see §13.4 — but it must be documented *at the creation site*, and waived *by name* in any
dead-weight sweep. Write the comment when you write the layer, not when someone audits it:

```python
# ALWAYS CREATE / CONDITIONALLY USE:
# `mask_token` is built in every configuration so the weight layout is stable across the
# masked-modelling and classification heads. It receives no gradient in classification
# mode. A dead-weight sweep must waive it BY NAME, with a paired liveness control.
self.mask_token = self.add_weight(name="mask_token", shape=(1, 1, dim), ...)
```

**Measured:** in one audit **62 of 74** "gradients do not exist for variables" warnings were exactly
this pattern — deliberate, documented, correct. Treating them as defects would have deleted a prompt
encoder's conditionality, a masked-image-modelling head, a video masking scheme, and a memory bank.
The comment above is what tells the next auditor which of the 74 they are looking at.

### 1.4 Configuration as Data

Store what you were given, not what you derived from it.

```python
# ❌ WRONG - a derived value cannot round-trip, and hard-codes a convention
self.head_dim = dim // num_heads

# ✅ CORRECT - store the input; derive on use, and let it be overridden
self.dim = dim
self.num_heads = num_heads
self.head_dim = head_dim if head_dim is not None else dim // num_heads
```

**Measured:** a released sparse transformer specifies `head_dim: 256` while `dim // num_heads` gives
`128`. The quotient is a *convention*, not a law, and modern architectures decouple it deliberately.
Storing the quotient made a shipped variant half its reference's head width, and every shape
assertion still passed because the model was internally consistent. See §13.2.

---

## 2. Essential Setup and Registration

### 2.1 Core Imports

```python
import keras
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any, List, Literal
```

`keras.ops` is the backend-agnostic surface. Reaching for the backend directly (`tf.*`) in a forward
path ties the layer to one backend and usually breaks XLA. Where a genuinely unmigratable op is
needed (`ifft`, `svd`, `grid_sample`), isolate it and say so in a comment — an accepted exception is
fine; an undocumented one is a trap for the next reader.

### 2.2 The Registration Decorator

```python
@keras.saving.register_keras_serializable(package="MyProject")
class MyLayer(keras.layers.Layer):
    ...
```

The `package=` argument is not decoration. A bare `register_keras_serializable()` produces a
**module-independent key**: two classes with the same name in different modules collide, and the last
import wins. Measured on Keras 3.8 — the key is the bare class name.

**But changing it is a breaking change.** The key is written into every saved file. If a codebase
already ships bare registrations, adding `package=` invalidates every existing checkpoint's
deserialization key.

| Situation | Do |
|---|---|
| New class | `package=` from the start |
| Existing class, checkpoints exist | leave it; migrate only with a deliberate checkpoint migration |
| Existing class, no checkpoints anyone needs | add `package=`, and say so in the commit |

**Measured:** an audit of 212 bare registrations across 761 classes found **0 collisions** — the risk
is real but latent. That is a reason to fix it on new code, not a reason to churn old code.

---

## 3. Layer Implementation Patterns

### 3.1 Pattern 1 — A Layer With Its Own Weights

```python
@keras.saving.register_keras_serializable(package="MyProject")
class ScaledProjection(keras.layers.Layer):
    """Project onto ``units`` and scale by a learned per-feature gain.

    :param units: Output width.
    :type units: int
    :param scale_init: Initial value of the gain.
    :type scale_init: float
    """

    def __init__(
            self,
            units: int,
            scale_init: float = 1.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        self.units = units
        self.scale_init = scale_init
        # declared here so a reader sees the full weight set without reading build()
        self.kernel = None
        self.scale = None

    def build(self, input_shape) -> None:
        if self.built:
            return
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.scale = self.add_weight(
            name="scale",
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.scale_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        return ops.matmul(inputs, self.kernel) * self.scale

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units, "scale_init": self.scale_init})
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

    self.dense1  = keras.layers.Dense(hidden_dim, activation="gelu", name="dense1")
    self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")
    self.norm    = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")  # ALWAYS
    self.dense2  = keras.layers.Dense(output_dim, name="dense2")

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
Auto-generated names shift when depth changes, and checkpoints stop matching. This also matters for
goldens: keying anything by a Keras-generated name makes it process-order dependent (§14.12).

### 3.3 Pattern 3 — Constant Tables (the `StatelessScope` trap)

```python
# ❌ WRONG - discarded by StatelessScope; all zeros in every real model
def build(self, input_shape):
    self.table = self.add_weight(
        name="table", shape=(n, d), initializer="zeros", trainable=False)
    self.table.assign(compute_table(n, d))

# ✅ CORRECT - the initializer IS the value
def build(self, input_shape):
    self.table = self.add_weight(
        name="table",
        shape=(n, d),
        initializer=keras.initializers.Constant(compute_table(n, d)),
        trainable=False,
    )
```

**Measured:** an `.assign()` inside `build()` is silently discarded under `StatelessScope` — the path
every real model build takes. Eleven sites in one codebase. A trend-only forecasting model predicted
**exactly zero**; a rotary table stayed all zeros, making the positional encoding an identity.

**Detect:** assert a statistic of the table itself after construction (`ops.max(ops.abs(table)) > 0`),
not a property of the model's output — a zeroed table often produces plausible-looking outputs.

### 3.4 Implementing `compute_output_shape`

```python
# ❌ WRONG - fails on an unbuilt layer
def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (self.kernel.shape[-1],)

# ✅ CORRECT - uses stored config
def compute_output_shape(self, input_shape):
    return tuple(input_shape[:-1]) + (self.units,)
```

`compute_output_shape` is called during symbolic build, before weights exist. Read config, never
weights.

### 3.5 Validation Placement

| Where | Validate |
|---|---|
| `__init__` | configuration invariants (ranges, enums, mutually exclusive flags) |
| `build` | shape invariants (divisibility, minimum extent, rank) |
| `call` | **nothing** — it runs per trace, and a Python check on a traced value does not do what it looks like |

Raise `ValueError` naming the offending value: `f"units must be positive, got {units}"`. A message
that does not contain the bad value costs the reader a debugging session.

---

## 4. Graph-Safe Operations in `call()`

`call()` is traced, not executed, in every regime that matters — `fit()`, `predict()`, `jit_compile`,
and the symbolic build of §8. Code that works in eager and breaks under tracing is the most common
"works on my machine" failure in this domain.

### 4.1 The Rules

- Use `keras.ops`, not the backend.
- No Python `if` on a tensor **value**. Use `ops.where` / `ops.cond`. A Python `if` on a *config
  flag* is fine — it folds at trace time, which is what you want.
- No `.numpy()`, no `int(tensor)`, no `float(tensor)`.
- No `.shape` arithmetic that assumes a concrete batch or sequence axis.
- Do not construct layers, and do not mutate Python state — `call()` runs once per **trace**, not
  once per step. A list that appends in `call()` grows once per trace and is empty in the graph.

### 4.2 Operations That Are Traps Under Tracing

| Operation | Trap |
|---|---|
| `ops.tril` / `ops.triu` | raise `pred must not be a Python bool` under `fit`/JIT in some versions; build the mask from `ops.arange` comparisons instead |
| `len(tensor.shape)` arithmetic | a `None` axis makes it a `TypeError` at trace time, usually reported far from the cause |
| `if training:` with a traced `training` | `OperatorNotAllowedInGraphError` for a traced `True` **and** a traced `False` |
| Boolean masks / `ops.where` on a `None`-length axis | shape inference silently yields `None` downstream, and the failure appears two layers later |
| `ops.cond` with Python-bool branches | folds at trace time; the untaken branch is never emitted |

```python
# ❌ WRONG - TypeError under fit/jit in some versions
mask = ops.triu(ops.ones((t, t)), k=1)

# ✅ CORRECT - arange comparison, shape-safe and backend-agnostic
idx = ops.arange(t)
mask = ops.cast(idx[None, :] > idx[:, None], dtype)
```

```python
# ❌ WRONG - OperatorNotAllowedInGraphError for a traced True AND a traced False
def call(self, x, training=None):
    if training:
        x = self.dropout(x)

# ✅ CORRECT - keep the Python-bool path byte-identical; only a tensor reaches ops.cond
def call(self, x, training=None):
    if training is None or isinstance(training, bool):
        return self.dropout(x, training=training)
    return ops.cond(training, lambda: self.dropout(x, training=True), lambda: x)
```

### 4.3 A `call()` Crash During Build-Tracing Becomes a Warning

If you materialize a sub-layer tree by tracing `call()` on symbolic inputs (§8.3), an exception
inside `call()` may surface as a build *warning* rather than an error, and the model continues in a
half-built state.

**Re-raise the original exception unchanged.** Wrapping it destroys the message that identifies the
cause.

```python
# ❌ WRONG - opaque
try:
    self.call(symbolic_inputs)
except Exception as e:
    raise RuntimeError(f"could not materialize {self.name}") from e

# ✅ CORRECT - the model's own message is the diagnostic
try:
    self.call(symbolic_inputs)
except Exception:
    raise            # the caller sees "Dictionary input must contain 'input_ids' key"
```

**Measured:** wrapping trace failures in `RuntimeError` turned three precise messages
(`"Dictionary input must contain 'input_ids' key"`, `"exceeds max_seq_len"`, a rank refusal) into
opaque errors, and broke the three tests whose entire subject *was* that message.

---

## 5. Model Implementation Patterns

### 5.1 Module Skeleton

The module docstring is **substantive prose, not a template**:

| # | Element |
|---|---|
| 1 | **One opening sentence** naming the architecture and its distinguishing options — a sentence, not a title with an `====` underline |
| 2 | **The principle**: what problem the architecture solves and *why its mechanism resolves it*. Inline math in backticks (`` `y = F(x) + x` ``) where an equation carries the idea |
| 3 | **The architecture**: stage/block structure, design trade-offs, and the places where the code does something non-obvious, and why |
| 4 | **Every deliberate behavioural choice**, stated as a choice with its reason (e.g. why `pretrained=True` raises rather than warning) |
| 5 | **Provenance**: the reference this is ported from, and *which* reference — see §13.2 |
| 6 | **`References:`** as `- Author et al., YEAR. Title. (url)` |

This **replaces** terse `Model Variants:` / `Usage Examples:` boilerplate that restates the variant
dict and the factory signature sitting directly below it. Length follows the architecture; do not
pad, and do not move real explanation into a README to hit a line budget.

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
| `build(self, input_shape)` | materializes the sub-layer tree (§8) |
| `get_config()` | every constructor argument, complex objects serialized |
| `from_config()` | deserializes them, symmetrically |
| `from_variant(cls, variant, ..., pretrained=False, **kwargs)` | raises `ValueError` **listing available keys** on a miss; accepts the overrides its own docstring advertises; does not splat description metadata into the constructor |

**Measured:** several `from_variant` implementations raised `TypeError` on exactly the override their
docstring advertised.

### 5.3 Two Variant Tables Are Not One

Where a package has both an architecture table and a training-config table, keep them separate and
give each exactly one home:

```
 architecture table   'tiny' -> {hidden_size: 192, num_layers: 12, num_heads: 3, ...}
 training config      'tiny' -> {lr: 1e-3, warmup: 500, batch_size: 256, ...}
```

A single merged table splats learning rates into a constructor, and a constructor kwarg into an
optimizer. Where a *value* has multiple homes, consolidate to one named constant:

```python
# ✅ the rate has exactly one home; three signatures and two variant rows read it
DEFAULT_DROPOUT_RATE = 0.1
```

**Measured:** a hard-wired rate had three homes in one package. Consolidating was the only change
that made the knob provably consistent across the class, the variant table and the factory.

### 5.4 A Variant Table Is a Claim About Someone Else's Model

If your variant is named after a published checkpoint, its numbers are a claim that must be checked
against that checkpoint's own config. This is §13.2, and it is where four shipped defects hid in one
audit.

### 5.5 Pretrained Weights

```python
# ❌ WRONG - produces a plausible model, a plausible loss curve, and an unreproducible result
if pretrained:
    logger.warning("pretrained weights unavailable; using random initialization")

# ✅ CORRECT
if pretrained:
    raise NotImplementedError(
        "Pretrained weights for 'my_model_base' are not published. "
        "Train from scratch with train_my_model.py, or pass pretrained=False."
    )
```

### 5.6 Factory, Exports and Hygiene

Every public factory belongs in the package's `__init__.py` `__all__`. An unexported factory is
invisible to users, to `from package import *`, and to any guard that sweeps the public surface.

**Measured:** one factory sat unexported for two audit cycles because its twin had been fixed and
nobody re-swept the family. Fix the family, not the instance.

Note the inverse, too: if a package deliberately exposes nothing at the top level (forcing
`from package.subpackage import X`), that is a valid convention — but then a guard asserting
importability from the top level is wrong, not the code.

### 5.7 When the Shape Does Not Apply

Not every model has a variant table, and not every model has pretrained weights. A package with a
single configuration should not grow a one-row `MODEL_VARIANTS` to satisfy a checklist. Say in the
docstring that the architecture is original and has no upstream, and the R-045-shaped audit question
("traced to a named external reference?") answers itself as *not applicable* rather than as debt.

**Measured:** of 49 packages carrying that audit question, **12-13 were original to the codebase**,
so the question was N/A. Two had been flagged as vacuous by an earlier audit and never actually
ruled — an unruled N/A looks exactly like unfinished work forever.

---

## 6. Configuration Management

### 6.1 Complete `get_config` / `from_config`

Every `__init__` parameter appears in `get_config`. Objects — initializers, regularizers,
constraints, activations — must be **serialized**, not stored raw:

```python
def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        "units": self.units,
        "activation": keras.activations.serialize(self.activation),
        "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
    })
    return config

@classmethod
def from_config(cls, config: Dict[str, Any]) -> "MyLayer":
    config = dict(config)
    config["activation"] = keras.activations.deserialize(config["activation"])
    config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
    return cls(**config)
```

**The failure is asymmetric — measure all four rows before believing you are safe:**

| Activation kind | `save()` | `load_model()` |
|---|---|---|
| string `"gelu"` | OK | OK — round-trips at exactly `0.0` |
| **registered** callable | OK | OK — Keras's generic encoder resolves it |
| unregistered callable, no `custom_objects` | OK | **raises** `Could not interpret activation function identifier: {...}` |
| unregistered callable + `custom_objects` | OK | loads, but `.activation` is left as a raw `{'module': ..., 'class_name': 'function'}` **dict**, which `get_config` then propagates onward |

The fourth row is the dangerous one: it *works*, and quietly poisons every downstream config and
every derived model built from it.

Two further measured facts:

- **`keras.activations.serialize` rejects a bare string** (`Unknown activation function 'gelu'
  cannot be serialized`). Pass strings through unchanged.
- **Reverting `get_config` and reverting `from_config` fail *different* tests.** A class can have a
  `from_config` that exists and still misses the key — one measured sibling had a `from_config`
  handling only initializers and regularizers, silently dropping the activation. Test both
  directions independently.

> **A guard for this must use an *unregistered* callable.** A registered one already round-trips, so
> a test built on it ships green and proves nothing. Measured: the first draft of exactly this guard
> passed at HEAD.

### 6.2 `**kwargs` Is Not a Channel to Your Sub-layers

```python
# ❌ WRONG - a typo becomes a silent default; an unrelated Keras kwarg becomes a crash
def __init__(self, units, **kwargs):
    self.attn = Attention(**kwargs)
    super().__init__(**kwargs)

# ✅ CORRECT - name what you accept
def __init__(self, units, attn_dropout_rate=0.0, attn_num_heads=8, **kwargs):
    super().__init__(**kwargs)
    self.attn = Attention(dropout_rate=attn_dropout_rate, num_heads=attn_num_heads)
```

### 6.3 A String-Literal Kwarg Table Is Invisible to Static Analysis

```python
# A rename of `classifier_dropout` that misses THIS LIST routes the value into
# model_kwargs -> config -> Model(**config), silently, with no TypeError.
task_specific_keys = ["num_labels", "pooling_strategy", "classifier_dropout"]
```

**Measured:** an AST caller scan — the correct instrument for finding keyword call sites — is
structurally blind to this, because it is a `List` of string constants, not a call keyword. A
repository-wide parameter rename found it only via a second, exact-token pass over string literals.

**Any rename of a public parameter needs two passes**: AST for call sites, and a literal-token scan
for config tables, registry keys, CLI flags, README examples and serialized JSON.

A related trap in the second pass: a naive filter like `grep -v _rate` discards the whole **line**,
so a line mentioning both the old name and an unrelated `drop_path_rate` vanishes from the report.
Match tokens, not lines.

### 6.4 Pair Every New Validation Raise With a Migration Path

A new `ValueError` on a previously-accepted configuration is a breaking change. Ship it with the
message that tells the caller what to write instead, and check whether any in-repo caller currently
passes the newly-rejected value.

### 6.5 Caches Derived From Weights

A cache computed from weight values must be invalidated when weights change — after `load_weights`,
after a `.assign`, after the optimizer's first step. Prefer recomputing to caching unless a
measurement shows the cache matters.

---

## 7. The Save/Load Path

Four separate real defects in one audit lived here, and **not one was visible to a round-trip
value check**. This is the least-tested surface in most Keras codebases, because the
obvious test — save, load, compare outputs — passes through all four.

### 7.1 The Round-Trip Test, on VALUES, and Twice

```python
# ❌ WEAK - blind to a build-side materialization loss, and to default-value coincidences
model.save(p)
reloaded = keras.models.load_model(p)
assert reloaded.count_params() == model.count_params()

# ✅ CORRECT - perturb, then TWO round trips, comparing VALUES
for w in model.weights:
    w.assign(w + 0.137)                       # make every weight distinctive
ref = model(x, training=False)

model.save(p1)
m1 = keras.models.load_model(p1)
d1 = float(ops.max(ops.abs(m1(x, training=False) - ref)))

m1.save(p2)                                    # the SECOND trip is the one that bites
m2 = keras.models.load_model(p2)
d2 = float(ops.max(ops.abs(m2(x, training=False) - ref)))

assert d1 == 0.0 and d2 == 0.0
assert {w.path for w in m2.weights} == {w.path for w in model.weights}
```

**Why twice.** `keras.models.load_model` builds the model from the saved `input_shape` and restores
weights immediately, so a model whose `build()` does not materialize its tree looks *fine* on the
first reload — the loss only appears when that reloaded model is itself saved. Measured instances: an
image encoder that reloaded **1 of 65** weights and a dense decoder that reloaded **0 of 12**, both
invisible to a one-trip check.

**Why perturb.** At initialization many weights are zeros or ones, and comparing default values
cannot distinguish "restored correctly" from "re-initialized to the same defaults".

**Why paths as a set.** A count can match while the identities differ. But note the caveat in §7.8 —
a path set is not always expected to be stable, and you must know which case you are in.

### 7.2 The Archive Layout Is Part of the Contract

A round-trip delta of exactly `0.0` does **not** mean the file is correct.

**Measured:** a depth model wrote **644 HDF5 datasets for 322 live weights — exactly 2.00×** — a
356 MB file of which 178 MB was redundant, with a round-trip delta of exactly `0.0` in both
directions. At the shipped large variant: **1220 datasets for 610 weights, 4.88 GB → 2.44 GB after
the fix.**

The cause: a `save_own_variables` override wrote a flat `vars/N` store *while Keras's default
path-based recursive save still ran*. The override's own comment claimed it "bypasses Keras'
path-walking entirely". It did not, and nobody had ever counted the datasets.

```python
def test_every_weight_is_written_exactly_once(tmp_path):
    model.save(p)
    with zipfile.ZipFile(p) as z:
        z.extract("model.weights.h5", tmp_path)
    names = []
    with h5py.File(tmp_path / "model.weights.h5") as f:
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    assert len(names) == len(model.weights), (
        f"{len(names)} datasets for {len(model.weights)} weights"
    )
    # and the stronger form: no flat-store family alongside the path family
    assert not [n for n in names if n.startswith("vars/")], names[:5]
```

**Check the group breakdown too.** The measured archive had
`{'decoder': 22, 'encoder': 150, 'frozen_encoder': 150, 'vars': 322}` — the first three already
accounted for all 322, and `vars` was the duplicate. That breakdown is what turned "the ratio is 2"
into a diagnosis.

### 7.3 A Symmetric Override Pair Is Usually Two Independent Decisions

`save_own_variables` and `load_own_variables` look like a matched pair. They frequently are not.

**Measured, on the model above:** deleting *both* gave a round-trip error of **4.5575** — the pair
was genuinely load-bearing. But the halves separated cleanly:

- the **load** half was essential. Keras calls the outer model's `load_own_variables` *before*
  recursing into sub-models, whose sub-layers do not exist yet, so a force-build there is what makes
  the restore work at all.
- the **save** half was pure harm. It never replaced the recursive save; it ran alongside it.

The fix was to delete the save half and keep, and correct, the load half.

> **Before deleting an override, delete each half separately and measure each.** And read the
> override's history: the original commit for this pair described a real defect it fixed **and a
> mechanism that was false**. The fix was right; the explanation was wrong; and the explanation is
> what the next author reasons from.

### 7.4 Optimizer State Is Saved, and Silently Not Restored

```python
# ❌ WRONG - reproduces the base method minus its last two lines
def compile_from_config(self, config):
    self.compile(**keras.saving.deserialize_keras_object(config))

# ✅ CORRECT - call super, or reproduce it completely including the optimizer build
def compile_from_config(self, config):
    return super().compile_from_config(config)
```

**Measured:** a VAE fitted for one epoch saved **122** optimizer variables and reloaded **2**. Every
`.keras` resume restarted Adam with zeroed moments and a zeroed step count — so a "resumed" run was
silently a fresh-optimizer run with warm weights. Loss curves looked normal. The only symptom was a
`UserWarning` nobody had turned into an error (§15).

```python
def test_the_reloaded_optimizer_state_survives(tmp_path):
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
    n_before = len(model.optimizer.variables)
    slots_before = [ops.convert_to_numpy(v) for v in model.optimizer.variables]

    model.save(p); reloaded = keras.models.load_model(p)

    assert len(reloaded.optimizer.variables) == n_before          # 2 == 122 catches it
    for a, b in zip(slots_before, reloaded.optimizer.variables):
        np.testing.assert_array_equal(a, ops.convert_to_numpy(b))
```

### 7.5 Saving Before the Optimizer Is Built

The inverse shape: a model compiled but never stepped has an optimizer with **no** variables, so a
later reload finds the counts disagree.

**Measured:** `2` variables at save against `106` at load. Under `-W error::UserWarning` this
escalates to `ValueError: A total of 1 objects could not be loaded`.

Repair according to what the test is actually about:

| The test's subject | Repair |
|---|---|
| weight round-tripping | build the optimizer state first: one `train_on_batch`, or `optimizer.build(model.trainable_variables)` |
| loss/metric config round-tripping | `load_model(compile=False)` is honest, if nothing asserts optimizer state |
| resuming training | build the state **and** assert it survived (§7.4) |
| a bit-identity assertion on outputs | do **not** insert a throwaway train step — it changes the weights you are comparing |

> **`include_optimizer=False` is a silent no-op on the `.keras` path.** Measured: `saving_api` pops
> the kwarg and the `.keras` branch calls `saving_lib.save_model(model, filepath)` without it. A
> repair built on it does nothing, and the warning persists — which is how it was found.

### 7.6 Saving an Unbuilt Model Writes an Empty Archive

**Measured:** a `save_model` convenience method invoked on an unbuilt model wrote a **31,587-byte
archive containing 0 trainable weights** (12 once built), which reloaded as a zero-weight model. A
sibling wrote 9,997 bytes / 0 weights. Both "succeeded".

```python
# ✅ CORRECT - build from the stored shape, or refuse BEFORE writing
def save_model(self, filepath):
    if not self.built:
        if self.input_shape_stored is not None:
            self.build(self.input_shape_stored)
        else:
            raise ValueError(
                "Cannot save an unbuilt model: call it on a batch, or build(input_shape), "
                "before saving. Saving now would write an archive with zero weights."
            )
    self.save(filepath)
```

Note the two cases: a class that *stores* its input shape can build itself; one that does not must
refuse. Refusing is not a lesser fix — it is the correct one when the shape is genuinely unknown.

### 7.7 Public Methods That Bypass Lazy Build

Any public method reachable before the first call — `summary()`, `save_model()`, a custom
`export_*`, a `predict_from_*` helper — must either build or refuse. Silently succeeding on an
unbuilt model is the defect, and it is invisible because the method returns normally.

### 7.8 A Weight-Path Set Is Not Always Expected to Be Stable

**Measured, and it corrected a plausible assertion:** a model holding a frozen teacher produced by
`clone_model` had **172 distinct weight paths before save and 322 after reload** — because the clone
inherits the student's path strings live, and the reload separates them. This was identical before
and after an unrelated fix, and harmless: the archive keys by attribute group (`encoder/` vs
`frozen_encoder/`), not by the live path string.

So: assert path-set equality where the model has no cloned sub-model, and assert **archive
dataset-name set equality across two round trips** where it does. Know which invariant your model
actually has before pinning one.

---

## 8. Build Materialization and Weight Compatibility

### 8.1 The Rule

**`build(input_shape)` must materialize exactly the sub-layer tree that `call()` runs.**

Not "override `build`". Not "call `super().build()`". *Materialize the tree.* A model that overrides
`build` to create two scalars fails this; one that ends `build` with a concrete forward trace passes.

### 8.2 Two Clarifications Authors Get Wrong

| Claim | Correction |
|---|---|
| "Overriding `build()` is the hazard" | It is not. The discriminating property is whether `build()` **materializes the tree** |
| "`Model.build(shape)` builds a subclassed model" | It only marks it built and walks no sub-layers, so `count_params()` returns exactly **0**. Several widely-copied packages do this. It is not a working precedent |

### 8.3 A Trace Beats a Hand-Written Shape Walk

Two ways to materialize a tree:

```python
# Option A - a hand-maintained shape walk, one per model
def build(self, input_shape):
    s = self.stem.compute_output_shape(input_shape)
    self.stem.build(input_shape)
    for blk in self.blocks:
        blk.build(s)
        s = blk.compute_output_shape(s)
    self.head.build(s)
    super().build(input_shape)

# Option B - trace call() on symbolic inputs, via ONE shared helper for the whole codebase
def build(self, input_shape):
    if self.built:
        return
    materialize_sublayers(self, input_shape)     # traces call() on KerasTensors
    super().build(input_shape)
```

**Prefer B.** Option A is a *second, hand-maintained encoding of the forward topology*, and its
failure mode — a sub-layer that silently stops being built when the architecture changes — is exactly
the defect being removed. A trace of `call()` cannot drift from `call()`.

A workable helper, with the properties that matter:

```python
def materialize_sublayers(model, input_shape, batch_size=1):
    """Build every sub-layer by tracing ``call()`` on symbolic inputs.

    Retries once with a concrete batch axis, because several ``call()`` bodies do
    integer arithmetic on the batch dimension. NEVER falls back to an eager pass,
    and re-raises the model's OWN first exception unchanged (see §4.3).
    """
    try:
        model.call(_symbolic(input_shape, batch=None))
        return
    except Exception:
        pass
    model.call(_symbolic(input_shape, batch=batch_size))   # let this one propagate
```

**Measured, applying one shared helper to 22 model classes:** `.build(shape)` alone went from
materializing **0** weights to 267 / 124 / 100 / 49 / 32 / 28 / 23 / 22 / … per model, and in the
worst case **1 → 19**.

**Option B does not always work, and that is fine.** Five of 27 classes could not be traced:

| Cause | Example |
|---|---|
| a sequence axis legitimately `None` | a causal mask needing a concrete `seq_len` |
| dynamic spatial extents by design | a model that owns a dynamic-shape test suite |
| a raw-backend op in `call()` | a `KerasTensor` cannot enter it |
| integer arithmetic on the batch axis, then `add_loss()` | trace-time failure |

**Pin those by name**, with a guard asserting each still inherits the base `build`, so the waiver
cannot outlive its reason:

```python
UNFIXED = {"VideoJEPA", "LeWM", "DynamicUNet", "SparseMoE", "LatentRegistration"}

def test_the_unfixed_subjects_are_pinned_by_name():
    for cls in UNFIXED:
        assert _resolve(cls).build is keras.layers.Layer.build, (
            f"{cls} gained a real build(); remove it from UNFIXED"
        )
```

> **Rejected, with a measurement:** an *eager* fallback trace materializes all five perfectly — and
> executes their `add_loss()` calls and BatchNorm updates for real, leaving accumulated losses and
> moved moving statistics on a model that was merely *built*. A new defect traded for a warning.

### 8.4 The Harm May Be Zero, and the Contract Still Matters

**Measured:** across 24 subjects put through the two-round-trip probe of §7.1 with a live
perturbation, the harm was **exactly `0.0` in every case**. Weight counts, path sets and forward
outputs were identical. The missing `build()` was a contract violation with no current weight loss.

Two readings of that number are wrong:

- "so it does not matter" — the same shape *has* caused real losses elsewhere in the same codebase
  (1 of 65 weights, 0 of 12), and the contract is what stops it recurring;
- "so we fixed a weight loss" — you did not, and saying so mis-prioritizes the next reader.

**Fix it, and state what you measured.** A guard docstring that overstates its own subject is a small
lie that compounds.

Two apparent non-zero readings in that same sweep were **the instrument**: two models with stochastic
outputs read `3.4e-02` and `4.6e+00`, and a self-spread control — same model, same weights, same
input, three calls — read `2.2e-02` and `4.1e+00`. The round-trip delta sat inside the model's own
sampling noise. **Run the self-spread control before calling a delta a defect.**

### 8.5 Enforcement

```python
def test_explicit_build_materializes_everything_a_call_does():
    a = make_model(); a.build(SHAPE)
    b = make_model(); b(sample_input)                  # lazy path
    assert len(a.weights) == len(b.weights)
    assert {w.path for w in a.weights} == {w.path for w in b.weights}
    assert a.count_params() == b.count_params() > 0    # > 0 catches the §8.2 no-op
```

The `> 0` is not padding: without it, a model that builds nothing passes against another model that
builds nothing.

---

## 9. Factory Patterns and Layer Reuse

### 9.1 Reuse Order

Reuse an existing layer → extend it → write a new one.

Grep the **class definition**, not the concept keyword. `grep "class Attention"` answers "is there a
rival implementation?"; `grep -r attention` returns hundreds of docstring hits and answers nothing.
Measured: a de-duplication survey over one codebase found that its *locations* held up while **4 of 5
of its defect claims died on measurement** — treat "where" as a work list and "why it is broken" as a
hypothesis.

### 9.2 A Registry-Backed Factory MUST Raise on Undeclared Keys

```python
_TYPE_TO_CLASS = {
    "layer_norm": keras.layers.LayerNormalization,
    "batch_norm": keras.layers.BatchNormalization,
    "rms_norm": RMSNorm,
    ...
}

def create_normalization_layer(norm_type: str, **kwargs):
    if norm_type not in _TYPE_TO_CLASS:
        raise ValueError(
            f"Unknown normalization type '{norm_type}'. "
            f"Supported types: {sorted(_TYPE_TO_CLASS)}"
        )
    return _TYPE_TO_CLASS[norm_type](**kwargs)
```

This matters more than it looks. In one codebase **149 of 178** call sites passed a *variable* or an
*attribute*, not a string literal, so no static check could ever validate them. Because the factory
raises, all 149 are **self-guarding at runtime**; the residual risk is population *growth*, not
silent fallback. A silent default would have made all 149 permanently unverifiable.

```python
def test_an_unknown_type_raises_and_names_the_valid_ones():
    with pytest.raises(ValueError, match=r"Unknown normalization type"):
        create_normalization_layer("layer_nrom")        # a plausible typo

def test_the_raise_is_not_a_silent_fallback():
    # RED-proof: replacing the raise with a default makes this fail
    with pytest.raises(ValueError):
        create_normalization_layer("definitely_not_a_type")
```

### 9.3 A Registry's Key Set Is Public Surface

Adding a key is additive; renaming or removing one breaks every caller and every saved config that
names it. Treat the key set as API, and pin it:

```python
def test_the_registry_key_set_is_exactly_the_pinned_one():
    assert set(_TYPE_TO_CLASS) == _PINNED_KEYS      # set equality, BOTH directions
```

### 9.4 The Inverse Defect — a Hand-Written Kwarg List That Omits a Key

A factory that *filters* its kwargs against a hand-written allowlist silently drops anything the list
forgot. Derive the list from the signature (`inspect.signature`), or pass through and let the
constructor raise.

### 9.5 A Derived "Optional" Parameter Is Not Safe to Pass

Forwarding a computed value into a constructor that also computes it produces two sources of truth
that agree until they do not. Pass the *input*, let the constructor derive (§1.4).

### 9.6 A Factory Is Not a Drop-In for the Layer It Wraps

**Measured, and this nearly shipped as a routine "consistency cleanup":** a normalization factory
defaulted epsilon to `1e-06`, while `keras.layers.BatchNormalization()` defaults to `1e-3` — a
**1000×** difference. A proposal to route two model families "through the factory like the others"
would have silently changed **189** layers' epsilon.

Across 16 bare-constructible types in that registry, **11 diverged** from their own layer's default
(1000×, 10×, and 0.1× in different families).

```python
# ✅ CORRECT - document it at the factory, in a table, and guard both defaults
def create_normalization_layer(norm_type, **kwargs):
    """Construct a normalization layer by name.

    .. warning::
       This factory's ``epsilon`` default does **not** match every wrapped layer's
       own default. Measured divergences (factory value vs layer value):

       ==================  ==========  ==========
       type                factory     layer
       ==================  ==========  ==========
       batch_norm          1e-06       1e-03
       layer_norm          1e-06       1e-03
       energy_layer_norm   1e-06       1e-05
       ==================  ==========  ==========

       It is therefore **not** a drop-in replacement for the bare constructor.
       Passing ``epsilon`` explicitly is the only portable call.
    """
```

**Rejected, with the reason:** making the factory accept `epsilon=None` meaning "use the layer's own
default" was considered and refused — it makes the resolved epsilon depend on the target class at all
149 dynamic call sites, which is strictly harder to reason about than one documented constant.

### 9.7 Porting From a Reference Implementation

Every implicitly-defaulted numeric in a port is a claim about the reference: epsilon, momentum,
initializer stddev, attention scale, clip bounds, schedule constants, activation coefficients.
Framework defaults differ between ecosystems — Keras norm eps `1e-3` against Torch `1e-5` silently
mis-specified 86 of 114 layers in one measured port, with every test green.

**Refuted, usefully, and it is the general lesson:** a carried claim that a mobile architecture's
BatchNorm momentum was wrong did **not** survive fetching. The reference implementation that hosts
all four shipped generations declares `norm_momentum=0.99, norm_epsilon=0.001` — exactly what
shipped. An *older* reference, covering only the earlier generations, says `0.9997`. **Two references
from the same organization disagreed**, the repo matched the newer one, and the real deviation was
the *epsilon* (183 of 189 layers on a value no reference supports).

> **Fetch the reference that covers the version you ship, name it in the docstring, and record which
> one you used.** "The paper says 0.9997" is not a citation if the paper predates your variant.

Also: `padding='same'` is asymmetric, so at stride 2 two branches can sum *different* pixels than the
reference does. Check the padding arithmetic explicitly when porting a strided block.

---

## 10. Numerics, Precision and Initialization

### 10.1 fp16 Mask Sentinels

`float16(-1e9)` is `-inf`, and `0.0 * -inf = NaN` on the **kept** positions.

```python
# ❌ WRONG - NaN on the kept positions under mixed_float16
scores = scores + (1.0 - mask) * -1e9

# ✅ CORRECT - dtype-aware sentinel
neg_inf = -65504.0 if scores.dtype == "float16" else -1e9
scores = ops.where(ops.cast(mask, "bool"), scores, neg_inf)
```

Rescue fully-masked rows over the **full axis the softmax reduces**, never per-tile: a per-tile
rescue inside an online-softmax block loop reads every strictly-upper causal tile as degenerate and
**un-masks the future** (measured deviation 24.14) while every finiteness test passes.

### 10.2 A Guard Margin Below the Working ULP Fails on Every Input

**Measured:** a model was 100% NaN because a stability margin was smaller than float16's ULP at the
magnitude it guarded — the margin could never separate a valid value from an invalid one. Derive
margins from the dtype's ULP at the operating magnitude, and write the derivation into the code.

The same rule applies to test tolerances (§14.6) and to backward passes: an `eps` safe in the forward
can overflow in the backward — a `(var + eps)^(-3/2)` term overflows fp16 at `eps < 6.1e-4`, giving
silent training death with a finite loss and all-zero weight movement.

### 10.3 A Shared Initializer *Instance* Is the Discriminator — Not `seed=`

```python
# ❌ WRONG - one instance reaching several architectural roles
init = keras.initializers.TruncatedNormal(stddev=0.02)
self.q = keras.layers.Dense(d, kernel_initializer=init)
self.k = keras.layers.Dense(d, kernel_initializer=init)
self.v = keras.layers.Dense(d, kernel_initializer=init)
self.o = keras.layers.Dense(d, kernel_initializer=init)

# ✅ CORRECT - clone per consumer
for name in ("q", "k", "v", "o"):
    setattr(self, name, keras.layers.Dense(d, kernel_initializer=clone_initializer(init)))
```

**Measured, three arms, and it refutes the intuitive rule:** two weights of the same shape built from
**one** initializer instance come out **bit-identical (`max|Δ| = 0.0`) whether or not the initializer
carries an explicit `seed=`.** Instance identity alone is the discriminator; `seed=` is not.

It is a defect exactly when the colliding weights play **different architectural roles**:

| Site | Colliding weights | Why it matters |
|---|---|---|
| attention | `w_q == w_k == w_v == w_o` | Q and K identical makes the initial attention matrix symmetric |
| a difference block | `positive == negative` | the branch difference **is** the architecture; identical init starts at exactly zero |
| a mixture head | `mus == sigmas` | mean and scale heads begin degenerate |
| a residual block | `hidden == output == skip_projection` | the skip and the transform start identical |

This cannot be settled by counting. Four instruments over one codebase gave four different
populations (940/188, 336/67, 178 groups, 175 groups) and **none reproduced the number carried from
the previous audit**. Two AST-flagged candidates were refuted by probing — no bit-identical pairs at
any legal config, because their shapes differ. And apparent collisions between `Ones()`-initialized
norm gammas are constants, not sharing: filter on `std > 0`.

> **Probe per site, fix by role, and pin the population with a ceiling gate (§14.8)** so it cannot
> grow silently. Do not edit hundreds of sites, and do not ignore them.

### 10.4 Residual Initializer Scaling, and the Truncation Factor

Deep residual stacks commonly scale the *residual output* projections' initializer std by
`1/sqrt(2 * n_layer)`, so the residual stream's variance does not grow with depth. It applies to the
attention output projection **and** the MLP output projection — both, not one. (Confirmed against two
independent reference implementations, which agree on the form and on the pair.)

```python
# ✅ CORRECT - one optional parameter, defaulted to the existing initializer, so every
#    other consumer of this shared block is unchanged BY CONSTRUCTION
def __init__(self, ..., output_kernel_initializer=None, **kwargs):
    super().__init__(**kwargs)
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.output_kernel_initializer = (
        keras.initializers.get(output_kernel_initializer)
        if output_kernel_initializer is not None
        else self.kernel_initializer
    )
```

**Measured, and it invalidates the obvious assertion:** `TruncatedNormal(stddev=0.02)` has a realized
std of `0.87964 × 0.02 = 0.017593`, not `0.02`, because truncation removes the tails. The reference
*argument* `0.02 / sqrt(2 · 12) = 0.004082` is therefore **never** the measured value — the real
target is `0.003591`.

**A guard pinning the reference argument goes RED against a correct fix.** Pin the **ratio**, where
the truncation factor cancels:

```python
ratio = residual_proj_std / qkv_std
assert ratio == pytest.approx(1.0 / math.sqrt(2 * n_layer), rel=tol)
```

and add an arm at a **second depth** — otherwise a depth-blind constant `1/sqrt(24)` passes. Derive
`tol` from the sample size (`8 * sqrt(1/(2·Na) + 1/(2·Nb))`), and assert the tolerance actually
discriminates: that an unscaled ratio of 1.0 is >10× the tolerance away, and that the two depths are
separated by >10× the tolerance.

Verify explicitly that **QKV and FC1 stds are unchanged** — that is the entire point of the separate
parameter, and scaling the shared initializer would shrink them too.

### 10.5 A Truthiness Guard on a Numeric Parameter Makes `0` Invisible

```python
# ❌ WRONG - 0 is falsy
if self.random_seed:
    rng = np.random.default_rng(self.random_seed)

# ✅ CORRECT
if self.random_seed is not None:
    rng = np.random.default_rng(self.random_seed)
```

**Measured:** a layer was half-seeded at exactly `random_seed=0` — the value its own test file used in
11 of 14 call sites. Sixty builds at `random_seed=0` gave **60 distinct correlations spanning
0.1244–0.9958**, five of them below the test's own 0.85 threshold, presenting as an occasional
1-in-12 flake. After the fix, 30 builds were bit-identical at exactly `0.0` spread, and an unseeded
control still varied by 1.34 — which is what proves the fix rather than a coincidence.

The first hypothesis — a randomized eigensolver upstream — **died** on 300 refits returning a single
value. Chase the falsy-zero before the exotic explanation.

**This applies to every numeric parameter where 0 is legal**: `seed`, `dropout_rate=0.0`,
`num_layers=0`, `momentum=0.0`, `warmup_steps=0`, `clip_value=0.0`. Use `is not None`.

### 10.6 Wrong Parameterisation, Sign, Direction or Scaling

A single sign or index can survive a whole suite. Measured instances: an `ops.roll` sign that made a
memory head shift the wrong way survived **249 green tests** because nothing tested addressing
*direction*; a transposed attention bias passed **219 tests**; a flipped CLS slice passed 91/91.

**Detect:** assert the **identity** of what is selected, not its shape — a delta impulse at a known
asymmetric position, on a **non-square** grid, so a transpose cannot masquerade as correct.

### 10.7 Two Producers of the Same Quantity

If two code paths compute the same tensor (a mask, a schedule, a normalization constant), they will
diverge. Give the quantity one producer and have the second path call it.

---

## 11. The Training Path

### 11.1 Custom `train_step`

Prefer stock `fit()`. A custom `train_step` is a maintenance liability and the audits found it wrong
more often than right. If you must override:

```python
def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(x, y, y_pred)          # already sums self.losses
        scaled = self.optimizer.scale_loss(loss)        # REQUIRED under mixed_float16
    grads = tape.gradient(scaled, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    ...
```

- **Differentiate the scaled loss.** Omitting `scale_loss` divides the whole update by ~`2^15` under
  `mixed_float16`. Measured at nine sites in one codebase; all nine now measure a ratio of
  `1.00 ± 0.008` against the stock path.
- **Do not hand-roll regularizer summation** — Keras 3's `compute_loss` already sums `self.losses`.
  An AST predicate asking "does the body mention `self.losses`" measures the wrong thing.

### 11.2 Python State That Never Reaches the Traced Graph

```python
# ❌ WRONG - folds to False at trace time; apply_gradients is NEVER EMITTED
if self._step % self.accum_steps == 0:
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

# ✅ CORRECT - a Keras variable and ops.cond
should_apply = ops.equal(ops.mod(self.step_var, self.accum_steps), 0)
ops.cond(should_apply, lambda: self._apply(grads), lambda: None)
```

**Detect:** assert the optimizer's iteration counter advances **exactly once per accumulation
window** over several windows — not that the loss moved.

### 11.3 A Zero Gradient Is Not a Freeze

Assert gradient flow **after one real optimizer step**, never at initialization. A documented
zero-initialized residual gate reads dead at init and is alive after one step. Measured: three
suites showed 30–330 weights "dead" at init; after one `fit()` step two cleared to exactly 0 dead,
and the third fell 30 → 3 after two steps — and *those* 3 were a genuine permanent defect that a
blanket waiver would have hidden.

### 11.4 A Head That Trains Only in the Loop Written For It

**Refuted, and worth the pattern:** a halting head appeared to receive zero gradient under its
factory's `compile(loss={"logits": ...})`. It trains fine — in the custom training loop written for
it, where its kernel moved by `0.9999`. The factory's own docstring said so verbatim, eight lines
above the `compile()` call the finding had read.

**Read the docstring above the line you are indicting.** Then make the advertised behaviour and the
shipped behaviour agree, in whichever direction is true — if the loop is required, the factory should
say so where a user will see it.

### 11.5 Optimizer and Callback Traps

- `optimizer.learning_rate` is the **current value**, not the schedule (the schedule lives on
  `_learning_rate`). A test asserting a schedule by reading `learning_rate` measures one point.
- A `class_weight` effect can be invisible to a total-|ΔW| probe under Adam (~0.9×) and obvious under
  SGD (26×). Prove a weighting reaches the backward pass with SGD, then switch back.
- A "best checkpoint" selected by min-val-loss during a curriculum can pick an epoch under-exposed to
  the hardest regime, costing up to 0.28 dB and inverting model rankings. Prefer the final checkpoint
  when the curriculum and the epoch budget coincide.

### 11.6 `pretrained=True` Returning Random Weights

Raise `NotImplementedError` (§5.5). A warning plus random weights produces a plausible model, a
plausible loss curve, and an unreproducible result.

---

## 12. Causality, Masking and Composition

### 12.1 The Missing Causal Mask

A decoder-only LM without a causal mask trains bidirectionally under a next-token objective and shows
a **better** loss curve, because the task is easier. Measured in three separate packages in one
codebase, each shipped and green.

```python
def test_position_t_cannot_see_the_future():
    y0 = model(x)
    x2 = ops.copy(x); x2 = perturb_at(x2, position=t)
    y1 = model(x2)
    before = ops.max(ops.abs(y1[:, :t] - y0[:, :t]))
    after  = ops.max(ops.abs(y1[:, t:] - y0[:, t:]))
    assert float(before) == 0.0          # positions < t must NOT move
    assert float(after)  > 0.0           # the control: the perturbation reached the model
```

The second assertion is not optional. Without it the test passes on a model that ignores its input.

### 12.2 Pooling a Causally Isolated Position

A causal model pooled at position 0, or with a `middle` pooling mode that indexes the **padded**
length, leaks or starves depending on the padding convention. Measured: a `middle` mode indexed the
padded length and leaked under ordinary prefix padding, while `mean` over a mask was correct.

### 12.3 `supports_masking` Is a Claim, and Setting It May Be a No-Op

**Measured, three arms:**

| arm | `max\|f([5,7,0,0])[:, :2] − f([5,7])\|` |
|---|---|
| shipped, `supports_masking = False` | **1.290977e-02** — attention **does** read padding |
| `supports_masking = True` | **1.290977e-02 — IDENTICAL** |
| explicit `attention_mask=[[1,1,0,0]]` | **2.384186e-07** |

Declaring `supports_masking = True` on a position-wise add is *honest* — the op genuinely preserves
masks — but it fixes nothing on its own: the mask merely propagates one layer further, to attention
layers that drop it too. The auto-mask was consumed by nothing and was numerically inert.

**Set the flag because it is true; fix the leak by passing an explicit mask.** And do not let a
`mask_zero=True` default advertise a masking guarantee the stack does not honour — measured, the
honest repair was to flip that default to `False` and require the explicit mask.

### 12.4 Masking by Zeroing Both Sides

Zeroing both sides of a comparison makes an exact-match metric correct and a per-element metric
silently wrong — `zero == zero` always agrees. Measured in one file: `sequence_accuracy` was sound
while sibling `per_step_accuracy` and `bit_error_rate` understated error 3× from the identical
idiom. **Check what the reduction does with a masked position**, not whether a mask is applied.

### 12.5 Repair Granularity

A rescue applied at the wrong reduction granularity fails silently and passes every finiteness test
(§10.1). The repair must operate over the full axis the reduction reduces over.

### 12.6 Inert Configuration — the Dead Knob

A parameter validated, stored, serialized, documented, and read by nothing.

**Detect:** a knob-sensitivity probe. Set a **non-default** value and assert something measurable
changes — the forward output, or the weight-shape signature for a structural knob.

```python
def test_the_knob_reaches_the_forward_pass():
    a = build(seed=0, knob=DEFAULT)
    b = build(seed=0, knob=NON_DEFAULT)
    assert float(ops.max(ops.abs(a(x) - b(x)))) > 0.0, "knob is a no-op"
```

Match the instrument to the knob class: a knob that adds or removes a weight needs a *structural*
assertion (weight count / shape signature), because its forward delta can legitimately be zero at
some configurations.

### 12.7 Inert Components

Distinguish "dead by design" (§1.3, §13.4) from "dead by defect" by reading the `call()` branch that
produces the symptom, not by reading the warning text.

### 12.8 Composition Failures

A layer correct in isolation can be wrong in a stack:

- a residual expected **externally** but applied **internally** annihilates signal at ~1e-5 per
  block;
- a transform-only block used as `x = block(x)` destroys the residual path entirely;
- a layout/packing contract changed for one consumer silently breaks a second consumer that shares
  it, and validating nothing at the unpack boundary is what keeps it silent.

**Detect:** assert a post-stack **magnitude**, not just a shape, and validate at every unpack
boundary.

---

## 13. Reachability and Provenance

These defects are invisible to every behavioural test, because the code under test is never
reached, or is reached and computes something simpler than its name.

### 13.1 The Unreachable Knob

A layer accepts a parameter and threads it correctly to every sub-layer. The model that constructs
that layer never passes it. The knob is *correct* and *unreachable*.

```python
# ❌ WRONG - the layer accepts dropout_rate; this construction never passes it
self.memory_attention = MemoryAttention(dim=d, num_layers=n)

# ✅ CORRECT - and the knob appears in the variant table, __init__, get_config and the factory
self.memory_attention = MemoryAttention(dim=d, num_layers=n, dropout_rate=dropout_rate)
```

**Measured:** a segmentation model's memory attention accepted `dropout` and threaded it to all 12
(small) / 24 (large) `Dropout` layers; the model's construction call simply never passed it. The
layer default was in force and unreachable from the variant table, `__init__`, `get_config` and the
factory. A sibling package had the identical shape with the opposite consequence — its unreachable
rate was `0.0`, so the knob was a no-op either way, which is why it had never been noticed.

**Detect — and the choice of arm is the whole test:**

```python
# ❌ VACUOUS - the default arm passes whether or not the knob is wired
model = create_model("small")
assert dropout_rates(model) == {0.1}

# ✅ CORRECT - a NON-DEFAULT value, asserted as a SET over every live layer
model = create_model("small", dropout_rate=0.37)
assert dropout_rates(model) == {0.37}      # {0.1, 0.37} if it reaches only some
```

where

```python
def dropout_rates(model):
    return {l.rate for l in model._flatten_layers() if isinstance(l, keras.layers.Dropout)}
```

**Measured:** with the wiring deleted, **every default-arm assertion stayed green** while the
non-default arm failed. The set form is what catches a rate that reaches the attention dropouts but
not the residual ones.

**Prefer a derived read-only property** over a stored duplicate when a sub-layer already owns the
number — a stored outer copy can silently disagree with the sub-layer actually built, especially when
the constructor also accepts an already-constructed sub-layer.

Finally, check the *other* end: a model that hard-wires `training=False` at a sub-layer call makes a
model-level wet-vs-dry probe measure exactly `0.0` — a guard that cannot fail. Site the behavioural
check where the flag is actually threaded.

### 13.2 A Variant Table Is a Claim About a Published Model

**Four shipped defects in one audit, each found only by fetching the upstream config:**

| Variant | Shipped | Reference | Consequence |
|---|---|---|---|
| a tiny BERT | FFN width 512 | **1024** | half the reference's FFN |
| a large CLIP | text depth 12 | **24** | half the released text tower |
| a sparse MoE | `8/64` experts active (12.5%) | `10/512` (**2.0%**) | 6× wrong on the architecture's headline number |
| the same MoE | `head_dim = dim // heads` = 128 | **256**, decoupled | half the reference head width |

The first is the sharpest: a **prior audit had recorded that value as agreeing, from recall.** Recall
shipped the defect; the fetch caught it.

> **Fetch, never recall. Cite the URL in the docstring.**

Classify each variant table honestly:

| Class | Meaning | Terminal state |
|---|---|---|
| **A** | a public released per-variant config exists | fetch it, compare field by field, ship any disagreement |
| **B** | a named upstream exists, but only a paper table | record the paper + table/section; say plainly that no released config exists to fetch |
| **C** | original to this codebase, no upstream | the question is **not applicable** — rule it, do not carry it as debt |

Class C is usually large and is the cheapest honest close. Measured: 12-13 of 49 packages. Two of
them had been noted as vacuous by an earlier audit and **never actually ruled**, so they looked like
open work for two more cycles.

And check before flagging: one citation suspected of being fabricated turned out real, with code
whose constructor arguments agreed with the repository's table field-for-field.

### 13.3 A Weight Nothing Reads, and a Layer Simpler Than Its Name

**Measured:** a spline-based convolution's `control_points` was `add_weight`-ed and read **nowhere**
in the codebase. Worse, the whole layer was decorative:

| probe | result |
|---|---|
| `max\|layer(x) − ops.conv(x, w_a + w_b)\|` | **0.0 EXACTLY** — it is a plain convolution |
| `\|f(2x) − 2f(x)\|` | **0.0 EXACTLY** — degree-1 homogeneous, so **no non-linearity at all**, not even the advertised activation |
| `control_points` movement after one `SGD(lr=1.0)` step | **0.0**, while the other two weights moved by an identical `1.221391` |

**Eighty-three tests passed against this.** Shapes, serialization, gradients-exist, finiteness — all
green, because none of them asked what function the layer computes.

**Detect, cheaply, for any layer claiming a non-linearity:**

```python
def test_the_layer_is_not_secretly_linear():
    y1 = layer(x)
    y2 = layer(2.0 * x)
    delta = float(ops.max(ops.abs(y2 - 2.0 * y1)))
    assert delta > tol, (
        f"degree-1 homogeneous (|f(2x)-2f(x)| = {delta}): this layer computes a "
        f"linear function, whatever its name says"
    )
```

and, for a weight you believe is live:

```python
def test_every_weight_moves_under_one_real_step():
    before = {w.path: ops.convert_to_numpy(w).copy() for w in model.trainable_weights}
    model.compile(optimizer=keras.optimizers.SGD(1.0), loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
    dead = {w.path for w in model.trainable_weights
            if np.array_equal(before[w.path], ops.convert_to_numpy(w))}
    assert dead == DEAD_BY_DESIGN
```

When you repair such a layer, verify the new forward against an **independent transcription of the
documented equation** — a nested-loop reference on a **non-square** kernel — not against itself.

### 13.4 Inert Components, and How to Waive Them

Some components are deliberately inert in some modes (§1.3). Waive them **by name**, with a paired
liveness control, and with **set equality in both directions**:

```python
DEAD_BY_DESIGN = {"backbone/mask_token/mask_token"}     # MIM head inside a classifier

def test_gradients_reach_every_trainable_weight():
    dead = compute_dead_set(model)                       # after ONE optimizer step
    assert dead == DEAD_BY_DESIGN
```

Set equality both ways is what makes the waiver fail when the design changes — an obsolete waiver is
a silent hole, and a one-directional `dead <= DEAD_BY_DESIGN` never notices that a waived weight
started training.

**Never use a two-sided `expect_zero=`** for this: it also fails on any model whose weight *does*
move, which converts a correct change into a false alarm.

A standing example worth knowing: a stock `MultiHeadAttention` **key bias can never learn** —
`q·(k + b_k)ᵀ = q·kᵀ + q·b_kᵀ` adds a term constant along the softmax's reduction axis, and softmax
is shift-invariant. Measured: gradient `3.9e-08` against a query-bias control of `7.8e-01`, a factor
of `2.0e7`. It is a property of the formulation, shared with the reference implementations, and it
must be waived by name in any dead-weight sweep rather than "fixed".

### 13.5 A Library That Blocks on a GUI

**Measured:** seven unconditional `plt.show()` calls in *library* code (not tests). Four
visualization calls emitted 10 backend warnings **and leaked 4 open figures** — the leak is the half
a warning filter hides.

```python
# ✅ CORRECT - return the figure; the caller decides
def visualize_grid(self, ..., show: bool = False):
    fig, ax = plt.subplots()
    ...
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
```

Grep the whole library path, not the two sites a report names — the measured population was **12**,
of which 1 was a docstring example, 4 already conformed, and 7 were live.

---

## 14. Testing and Validation

A guard that cannot fail is the most likely outcome of writing a new test. Across three audits this
was measured repeatedly — including in guards written *by* the audits, caught only because each one
was injected against before being trusted.

### 14.1 The Five House Rules

1. Every test asserts a **value or a relationship**, never only a shape.
2. Every guard is proven **RED by injection** before it is trusted, and you check **which** assertion
   fired.
3. Every tolerance is **derived** — from a dtype's ULP or a measured population — and the derivation
   is written into the test.
4. Every waiver is pinned **by name**, with set equality in **both** directions.
5. Compare failing **node-id SETs**, never counts. Quote `passed` **with** `collected`.

### 14.2 Prove Every Guard RED

Inject the specific defect the guard claims to catch; confirm the guard fails; confirm **which**
assertion fired; restore from a scratch copy verified with `diff -q` / `sha256sum -c`.

```
cp target.py /scratch/target.py.bak
# ...edit target.py to inject the defect...
pytest tests/test_target.py -q          # must FAIL, at the named assertion
cp /scratch/target.py.bak target.py
diff -q /scratch/target.py.bak target.py && echo RESTORED
```

> **Never `git stash` or `git checkout --` mid-proof.** Measured: one such restore destroyed an
> entire uncommitted step, and the resulting mutation then "failed" with an unrelated `SystemExit`
> from a missing CLI flag — a RED for the wrong reason is indistinguishable from a real one unless
> you check which line raised.

### 14.3 A Guard's RED-Proof Must Use the Arm That Can Actually Fail

**Measured twice in one plan, both caught before shipping:**

- an activation-serialization guard drafted with a **registered** callable **passed at HEAD** — the
  registered path already round-trips. Only an *unregistered* callable reproduces the defect (§6.1).
- a structural claim written as `type(model).call is MyModel.call` was **True with the override
  deleted entirely** (it resolves to the parent's `call`). The assertion that can fail is
  `"call" in MyModel.__dict__`.

Before trusting a green guard, ask: **what exact edit would make this red?** If the answer is
"nothing", it is decoration.

### 14.4 An Injection That Fails to Convict Is Evidence, Not a Setback

**Measured:** an injection meant to prove a boundary guard RED did not fire — a constant pad moves no
pre-boundary position. That non-conviction *confirmed* the residual under test was numerical rather
than positional, which was the open question. It was recorded as evidence; a second, correctly
derived injection then fired.

Do not silently swap in an injection that works. And treat a control that comes back green on its
first run as a probe defect until proven otherwise — measured twice in one step, once because a value
grid never crossed the boundary it was meant to exercise, and once because a bytecode fingerprint was
structurally blind to the mutation it was checking.

### 14.5 A Treatment Arm Without a Control Arm Proves Nothing

Running a suite under a new setting and seeing failures does **not** attribute them to the setting.
Run the paired control — same directory, same process shape, setting off — and compare **sets**.

**Measured, three ways in one plan:**

- of ~19 failures under a new warning filter, the control showed **1** pre-existing;
- six directories' worth of failures were *all* genuinely attributable, but only the control could
  establish that;
- one directory failed a **different node id in each arm**, so neither attribution was clean and it
  had to be settled by running both node ids alone.

The same discipline settles "is this pre-existing?": run the suspect at the **base commit in a
detached worktree** — and see §14.7 for the trap that makes that measurement meaningless if you get
it wrong.

### 14.6 Pin the Property, Not the Sample

**Five of nine long-standing RED tests in one audit were the same defect class:** an exact or 7-digit
literal pinned against a seed-dealt or sub-ULP quantity.

| Pinned | Reality |
|---|---|
| `delta == pytest.approx(1.805329, rel=2e-3)` | over 90 draws `delta/amplitude` spans 0.28–1.03; the pin was ~30× tighter than the population |
| `departure == 0.0` | the departure is `7.450581e-09` — **exactly one float32 ULP at 0.1**, so the assertion can never pass |
| `residual == 0.0` (a mean-centring claim) | residual is 0.25–2.50 ULP over 20 blocks, never 0.0 |
| `entropy < 0.9 * log(4)` | at that temperature entropy spans 0.40–1.33 over 20 seeds — the bar sat **inside** its own population |

**The procedure, before writing any numeric assertion:**

1. Sample the quantity over **≥20 seeds** or fresh builds.
2. Report min / max / mean, and whether the distribution is **noisy or bimodal**.
3. Put the bar **outside** the population — or better, assert a **relationship** instead
   (monotonicity, a ratio, an ordering). A ladder strictly monotone in 20 of 20 seeds is a theorem; a
   threshold is a coin flip.
4. Write the derivation into the test as a comment.

```python
# MEASURED over set_random_seed(0..19): entropy at T=0.1 spans 0.4024..1.3254, and the
# old 0.9*log(4)=1.2477 bar sits INSIDE that spread (4 of 20 above it). The property the
# knob actually has is monotonicity, which held in 20 of 20 seeds.
for hi, lo in zip(temps, temps[1:]):
    assert entropy(lo) < entropy(hi)
```

**Bimodal is not noisy, and the distinction changes the repair.** Measured: a separation read
`0.270000` on 19 of 20 seeds and `0.003333` on one. That outlier was not spread — it was a degenerate
draw in which an untrained head coincidentally achieved the pinned target, collapsing the term.
Widening the bar would have made the guard vacuous on exactly the fixture where it stops working.
**Pin the draw; do not widen the bar.**

Conversely, when a bound is provably **unattainable** — 1 of 50 comparisons over it, worst ratio
1.0137, ~7 ULP at the output magnitude — widening *is* correct. Derive the new bound from the
measured population (52 ULP, 7.5× headroom) and write down why this case differs from the seeded one.
Both repairs can be correct in the same plan; the test must say which it is.

### 14.7 Measurement Traps

| Trap | What it does |
|---|---|
| **TF32** | the default false model defect on Ampere+: a ~1e-3-scale "leak" that is the TF32 ULP. Quote near-zero statistics from CPU. Note one module disabling TF32 at import time changes every later measurement in the process |
| **Eager-only bit-identity** | `0.0` eager and `4.2e-04` under `@tf.function` — the regime `fit()` uses. The control is the within-version eager-vs-graph delta |
| **Untrained models** | a fresh classifier emits a uniform distribution, so an output-std probe reads **exactly 0.0** and measures nothing. Measured on four models; only overwriting every weight from a seeded stream produced signal |
| **A detached worktree** | silently imports the **main** repo's package unless `PYTHONPATH` is forced. A "was this pre-existing?" measurement taken without that is meaningless |
| **A trailing slash** | `find results/ ...` hashes differently from `find results ...`. Fix the exact command string in the guard |
| **Parallel GPU jobs** | contention produces flakes that cluster; never run two |
| **Patching a re-exported name** | binds the importing module only; patch the **defining** module |
| **Grid size as exhaustiveness** | a 6-cell grid whose cells never cross the boundary tests the boundary zero times |
| **`get_weights()`** | can hide a defect that `trainable_weights` and an ordinal comparison expose |
| **A save-side-only check** | cannot see a load-side loss (§7.1) |

### 14.8 Ceiling Gates for Populations You Cannot Fix Site-by-Site

When a defect class has hundreds of sites and each needs a per-site judgment, do not edit all of them
and do not ignore them. **Pin the population and ratchet it down:**

```python
_COLLISION_GROUP_CEILING = 159      # measured at <commit>; see the decision record

def test_the_population_has_not_grown():
    assert len(collision_groups()) <= _COLLISION_GROUP_CEILING
```

RED-prove it by lowering the ceiling by one and confirming exactly that test fails. Fix a bounded,
named subset (8 sites, say), rule the remainder explicitly **with the count**, and let the gate stop
it growing. This converts an unbounded sweep into a bounded decision plus a ratchet.

Two cautions:

- **A ceiling is a `src/`-wide or package-wide figure — say which.** The same sweep restricted to a
  sub-package gave a different number, and a ceiling quoted without its scope is unreproducible.
- The pin should key on `(file, symbol)` or `(file, callee, keyword)`, **not on line numbers**, which
  drift with every commit.

### 14.9 Process-Global State Leaks Across Tests

**Measured:** one unrestored `keras.utils.set_random_seed(1)` inside an *earlier* test made a later
test in the same file fail. Deselecting that one test flipped the later one green — causally
sufficient, not merely correlated. Eight directories shared the shape: run individually, **zero**
failures; run in one process, **23**.

```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def _restore_global_rng():
    py_state = random.getstate()
    np_state = np.random.get_state()
    ...capture the framework's global seed state...
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        ...restore...
```

**After the fixture: 481 passed, 3 skipped, 0 failed in one combined process**, and the combined
per-test totals equalled the sum of the eight solo runs exactly — order-neutrality *measured*, not
asserted.

Two facts this surfaced that are not obvious:

- **`keras.utils.set_random_seed` is not the whole leak.** It writes Python `random`, NumPy's legacy
  global, and the backend's global seed — but **not** Keras 3's global `SeedGenerator`, which every
  unseeded `keras.random.*` call advances. That is a second, independent coupling of the same shape.
- **The public getter for the backend's global generator *creates* it as a side effect.** Read the
  module global directly, or the act of observing the state manufactures it.

State what you **cannot** restore — a generator first created inside a test, device-level RNG, a live
dataset iterator — rather than implying full coverage. The same capture/restore discipline applies to
the global dtype policy, `floatx`, and TF32 flags.

### 14.10 An Instrument Can Report a Lower Bound

See §15.1. This is the single highest-yield item in this document: the obvious instrument for a
warning census under-reports by **5×**, and every count derived from it is wrong.

### 14.11 Why Guards Fail

#### 14.11.1 One mutation per assertion, and check which one fired

Two mutations that both break a value before it reaches a later assertion prove the *first*
assertion twice and the second zero times. Verify by **name** which assertion fired for each
mutation; if two land on the same one, add a mutation that isolates the other.

#### 14.11.2 An injection that moves both sides proves nothing

If your injection perturbs the reference as well as the subject, the comparison is unchanged. Name,
for each injection, the exact path it can and cannot see.

#### 14.11.3 An oracle written by the same hand is a second copy of the bug

A float64 "reference" that computes the value the same way the code does reproduces the same
cancellation. **An oracle must consume the code's actual received bits, never the intended Python
literals** — a gap below the dtype's ULP arrives bit-identical to the code, so an oracle fed the
intended values is measuring a different input. Measured three times in one problem family.

#### 14.11.4 Liveness is not correctness

A destroy-negatives / destroy-positives probe can go green on code computing the **wrong quantity**.
Measured: a raw-layout dice loss moved under both destroy probes while returning `0.765062` where the
correct reshape gives `0.693310`. Assert the **value** under the probe, not only that the probe moved
it.

#### 14.11.5 A guard goes vacuous when its subject changes shape

A property test whose subjects no longer have the property passes for a reason the test does not
check. Measured: a "these variants can never window" test became doubly vacuous once a config change
meant those variants built **zero** windowed layers at all — it was true, and true for a reason the
test never examined.

Re-derive a guard's subject whenever the config it depends on changes. And when repairing such a
test, respect any standing ruling about *how* it may not be fixed.

#### 14.11.6 Anti-vacuity on collection

An empty suite reads as a pass. Measured: three test targets collected **zero** tests, one of them
four **0-byte** files, and had done for an unknown period. **Always quote `passed` with
`collected`**, and assert a floor on collection where a suite matters.

#### 14.11.7 A silent skip blinds every sweep in the file

```python
# ❌ WRONG - a broken file silently shrinks the swept population
try:
    tree = ast.parse(src)
except SyntaxError:
    continue

# ✅ CORRECT - fail loudly, naming the file
def _parse_or_fail(path, src):
    try:
        return ast.parse(src)
    except SyntaxError as e:
        raise AssertionError(
            f"{path} does not parse, so every AST guard in this file is blind to it: "
            f"SyntaxError at line {e.lineno}: {e.msg}"
        ) from None
```

**Measured:** injecting a syntax error made an AST sweep drop a file (61 → 60 sites) while the guard
class still read **21 passed**. After the repair the same injection produced **59 failed / 455
passed**. It was not one site — it was **all six** `ast.parse` call sites in that file. Keep the
allow-list empty until a file proves it needs one (measured: 0 of 1,007 source files failed to
parse).

#### 14.11.8 A predicate that filters before it classifies

```python
# ❌ WRONG - drops an item before it can be judged; 5 of 8 became invisible
if "variant" not in params:
    continue

# ✅ CORRECT - carry every item, record a verdict per item
verdict = "no-variant" if "variant" not in params else classify(fn)
sites.append((path, name, verdict))
```

Derive any count from the **verdict**, not from `len(sites)`, so existing floors keep their
calibration when the population widens.

#### 14.11.9 A guard on worktree text is not a behavioural guard

`git diff --stat <path>` being empty asserts nothing about behaviour: it goes RED on a harmless
trailing comment and GREEN on any change that has been committed or staged. Measured, both
directions. Replace it with a behavioural golden (§14.12), and check the **false-positive** family as
hard as the true-positive one — a comment-only edit must leave it green.

#### 14.11.10 A guard that cannot distinguish pathological from unusual destroys correct answers

Test a guard's **false-positive** family as hard as its true-positive one. A cumsum-finiteness guard
that looked obviously right was measured poisoning ordinary rows whose answers were exact.

### 14.12 Deterministic Goldens

A golden pinned from a *seeded* build is a sample (§14.6). Fix every weight by **rule**:

```python
for i, w in enumerate(model.weights):
    rng = np.random.default_rng(1_000_003 * i + zlib.crc32(str(w.shape).encode()))
    w.assign(rng.standard_normal(w.shape).astype("float32"))
```

Then the golden is a property of the architecture's arithmetic and weight layout, not of a seed.
Measured bit-identical across three separate processes.

Two traps measured while building exactly this:

- **`build(None)` materialized only 38 of 120 weights** — the remaining 82 arrive on the first call.
  Assigning before a warm-up made two identical processes disagree. Warm up, then assign.
- **Keying the fill by `w.path` is process-order dependent**, through Keras's name-uniquification
  counter. Key by enumeration index and shape.

Derive the tolerance from the smallest pinned magnitude and the dtype's resolution there, and say so:
`atol=1e-6` is ~3 orders below a smallest pinned magnitude of `2.9e-05`, and far above float32
resolution (~1e-9) at that scale.

### 14.13 Test Anti-Patterns

- A test that asserts only shapes.
- A fixture that constructs a shape the real pipeline can never emit — drive the **actual** factory
  and data path.
- A test whose tolerance is below the output dtype's resolution: it can never pass, so it measures
  nothing.
- A pinned measured **count** that regression-locks a bug (a split count that silently included
  leaked samples). Pin the **property** the count was standing in for.
- A same-seed determinism assertion over a small space — with 2 possible orders, two same-seed runs
  coincide half the time even if the seed is ignored. Assert the **partition** over N seeds.
- A suppression with no paired positive assertion (§15.3).

### 14.14 Scoping Runs

Where a suite cannot run in one process — measured OOM at **36.6 GB RSS** on a 62 GB host — run
per-directory, in separate processes, and compare failing node-id **sets** across runs. Re-run every
failure **alone** before believing it. A directory-level pass/fail count is not comparable across
runs; a node-id set is.

---

## 15. Warnings as a Defect Channel

The cheapest large-scale audit available. The framework is already telling you where the defects
are; most codebases never turn the channel on, and the obvious way to turn it on under-reports by
5×.

### 15.1 The Default Instrument Reports a Lower Bound

```bash
# ❌ WRONG as a census - aborts each test at its FIRST warning
pytest -W error::UserWarning

# ✅ CORRECT as a census - reports every warning, with its node ids
pytest -W always::UserWarning -rw
```

**Measured:** the census went from **58** node ids to **310** on the same tree, purely by changing the
instrument — and from four apparent root causes to **nine**. Every count derived from the first
instrument was wrong, including the estimate that sized the work.

Use `error` as the **gate** (in CI, once the tree is clean) and `always -rw` as the **census** (when
you are finding out what is there).

Validate the instrument before trusting it: run both forms on one directory and check that the node
id set from `always` is a superset of the set from `error`. Measured: same three node ids, plus one
warning text that `error` had masked because an earlier warning aborted the test first.

### 15.2 Triage Into Four Classes, Not Two

| Class | Action |
|---|---|
| **REAL DEFECT** | fix it |
| **EXPECTED-BY-DESIGN** | the component is deliberately inert (§1.3, §13.4) — waive by name, with a liveness control |
| **TEST-ENVIRONMENT** | fix the test or the fixture, not the library |
| **THIRD-PARTY-UNFIXABLE** | ignore, with a one-line justification naming the library and why |

**The second class is the large one, and it is invisible if you only have two buckets.** Measured: 62
of 74 in the dominant family. Treating them as defects would have deleted correct architecture — a
prompt encoder's conditionality, a masked-modelling head, a video masking scheme, a memory bank.

The diagnostic is not the warning text; it is the `call()` branch that produces the symptom, plus
the in-source `ALWAYS CREATE / CONDITIONALLY USE` comment that should be sitting at the creation site
(§1.3). If that comment is missing, write it as part of the triage — the next auditor should not have
to re-derive it.

### 15.3 A Suppression Without a Positive Assertion Is Invisible Debt

An `ignore` filter keeps working forever after the advisory it hid is deleted, reworded, or has its
branch inverted. Nothing goes red. **A suppression with no paired positive assertion is a claim
nobody checks.**

Pair every deliberate advisory with two arms:

```python
def test_the_advisory_still_fires():
    """Positive arm: match= keys on the exact prefix the ignore filter uses, so a
    reword breaks THIS file before it silently widens the filter."""
    init = OrthogonalInitializer()
    with pytest.warns(UserWarning, match=r"Orthogonality constraint violation"):
        out = init(shape=(8, 4))                 # mathematically infeasible request
    assert tuple(out.shape) == (8, 4)            # an advisory must still return a valid tensor

def test_a_feasible_request_does_not_warn():
    """Control: without it the positive arm cannot fail — an unconditional warning
    would satisfy it."""
    init = OrthogonalInitializer()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        init(shape=(4, 8))
```

**Prefer `pytest.warns` at the provoking site over a module-scoped `ignore`.** They are different
strengths: `pytest.warns` **asserts the warning is emitted** and goes RED the day the design silently
changes; an `ignore` mark asserts nothing.

If you cannot cover every family, **name the residue with its count** rather than implying the
suppressions are self-maintaining. Recorded honestly in one closure: 12 of 42 suppressions
rot-guarded, 30 not, with the four unguarded families named.

### 15.4 A Global Ignore Needs a Strictly Stronger Instrument

An entry in a global ignore list is defensible only when something else already proves what the
warning would have told you. **Say which instrument, in the entry:**

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error::UserWarning",

    # Framework warns for N node ids: "Gradients do not exist for variables [...]".
    # MEASURED: M of the N are components inert BY DESIGN under the mode the test runs
    # (see §13.4), each carrying an in-source ALWAYS CREATE / CONDITIONALLY USE comment.
    # The STRONGER instrument is the per-variable gradient oracle, adopted by M packages:
    # it convicts identically-ZERO gradients this warning cannot see, pins its waivers BY
    # NAME, and fails loudly when a waiver goes stale.
    # Read this as "the oracle, not the warning, is what proves they are not dead" --
    # NOT as "dead weights are fine".
    "ignore:Gradients do not exist for variables:UserWarning",
]
```

Keep the list short and justify each entry against that bar. If an entry cannot name a stronger
instrument, it is not an ignore — it is an unfixed defect.

One structural blocker worth knowing: a warning raised with `stacklevel=2` reports its location
inside a **framework internal**, not your module, so no `:module:` filter can target it. Only a
message regex can — and that suppresses the advisory in production runs too. Prefer `pytest.warns` at
the sites in that case.

### 15.5 The Ordering Hazard

**Measured, a self-inflicted regression:** a global `error::UserWarning` landed *last* in a sequence
of fixes converted an earlier step's **deliberate** census advisory into a failure. The earlier step
had verified green — truthfully, at its own commit.

**Land process-global configuration EARLY**, so every later measurement is taken under the final
regime. If it must land late, re-run the suites that predate it before believing their green.

The same hazard applies to an autouse conftest fixture (§14.9), a global dtype policy, and anything
that changes what "a passing run" means.

---

## 16. Common Pitfalls and Solutions

### Pitfall 1: Conditional layer creation
```python
# ❌ WRONG - two weight layouts under one class name
if use_attention:
    self.attn = Attention(dim)

# ✅ CORRECT - create always, gate usage in call(); build only what call() runs
self.attn = Attention(dim)
```

### Pitfall 2: Creating layers in `build()`
```python
# ❌ WRONG - absent after from_config
def build(self, input_shape):
    self.dense = keras.layers.Dense(self.units)

# ✅ CORRECT - create in __init__, build here
def build(self, input_shape):
    self.dense.build(input_shape)
    super().build(input_shape)
```

### Pitfall 3: Registration without a package
```python
# ❌ WRONG - module-independent key; last import wins on a name collision
@keras.saving.register_keras_serializable()

# ✅ CORRECT
@keras.saving.register_keras_serializable(package="MyProject")
```

### Pitfall 4: Incomplete `get_config`
```python
# ❌ WRONG - dropout_rate is lost on reload; the model silently gets the default
def get_config(self):
    return {**super().get_config(), "units": self.units}

# ✅ CORRECT - every constructor argument
def get_config(self):
    return {**super().get_config(), "units": self.units,
            "dropout_rate": self.dropout_rate}
```

### Pitfall 5: `.assign()` of a constant table in `build()`
```python
# ❌ WRONG - discarded by StatelessScope; all zeros in every real model
self.table = self.add_weight(shape=(n, d), initializer="zeros", trainable=False)
self.table.assign(compute_table(n, d))

# ✅ CORRECT
self.table = self.add_weight(
    shape=(n, d), initializer=keras.initializers.Constant(compute_table(n, d)),
    trainable=False)
```

### Pitfall 6: `build()` that does not materialize the sub-layer tree
```python
# ❌ WRONG - reloaded model restores into nothing; nothing raises
def build(self, input_shape):
    super().build(input_shape)

# ✅ CORRECT - build exactly what call() runs
def build(self, input_shape):
    if self.built:
        return
    materialize_sublayers(self, input_shape)
    super().build(input_shape)
```

### Pitfall 7: Python conditionals on tensor values
```python
# ❌ WRONG - evaluated once, at trace time
if ops.max(x) > threshold:
    x = normalize(x)

# ✅ CORRECT
x = ops.where(ops.max(x) > threshold, normalize(x), x)
```

### Pitfall 8: `ops.tril` / `ops.triu` in a traced path
```python
# ❌ WRONG - TypeError: ('pred must not be a Python bool', True) under fit/jit
mask = ops.triu(ops.ones((t, t)), k=1)

# ✅ CORRECT
idx = ops.arange(t)
mask = ops.cast(idx[None, :] > idx[:, None], dtype)
```

### Pitfall 9: An fp16-unsafe mask sentinel
```python
# ❌ WRONG - float16(-1e9) is -inf, and 0.0 * -inf = NaN on the KEPT positions
scores += (1.0 - mask) * -1e9

# ✅ CORRECT
neg = -65504.0 if scores.dtype == "float16" else -1e9
scores = ops.where(ops.cast(mask, "bool"), scores, neg)
```

### Pitfall 10: Symbolic `training` into BatchNorm / Dropout
```python
# ❌ WRONG - OperatorNotAllowedInGraphError for a traced True AND a traced False
if training:
    x = self.dropout(x)

# ✅ CORRECT - keep the Python-bool path byte-identical; only a tensor reaches ops.cond
if training is None or isinstance(training, bool):
    x = self.dropout(x, training=training)
else:
    x = ops.cond(training, lambda: self.dropout(x, training=True), lambda: x)
```

### Pitfall 11: A factory that filters and drops unknown keys
```python
# ❌ WRONG - a misspelled key silently becomes a default
kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED}

# ✅ CORRECT - raise, naming the valid keys
if norm_type not in _TYPE_TO_CLASS:
    raise ValueError(f"Unknown type '{norm_type}'. Supported: {sorted(_TYPE_TO_CLASS)}")
```

### Pitfall 12: Constructing layers or mutating Python state in `call()`
```python
# ❌ WRONG - a new object per trace, untracked; and a list that grows once per TRACE
def call(self, x):
    proj = keras.layers.Dense(self.units)
    self._seen.append(x.shape)
    return proj(x)

# ✅ CORRECT - create in __init__; use a Keras variable for state
```

### Pitfall 13: A custom `train_step` without `scale_loss`
```python
# ❌ WRONG - under mixed_float16 the update is ~2^15 too small
grads = tape.gradient(loss, self.trainable_variables)

# ✅ CORRECT - differentiate the SCALED loss; a no-op off mixed precision
grads = tape.gradient(self.optimizer.scale_loss(loss), self.trainable_variables)
```

### Pitfall 14: `pretrained=True` that warns and returns random weights
```python
# ❌ WRONG
logger.warning("no pretrained weights; using random init")

# ✅ CORRECT
raise NotImplementedError("Pretrained weights are not published for this variant.")
```

### Pitfall 15: A raw callable or object in `get_config`
```python
# ❌ WRONG - raises at load, or loads back as a dict that propagates onward
config.update({"activation": self.activation})

# ✅ CORRECT - symmetric serialize / deserialize
config.update({"activation": keras.activations.serialize(self.activation)})
```

### Pitfall 16: A truthiness test on a numeric parameter
```python
# ❌ WRONG - seed=0, rate=0.0, depth=0 all fall through silently
if self.seed:
    rng = np.random.default_rng(self.seed)

# ✅ CORRECT
if self.seed is not None:
    rng = np.random.default_rng(self.seed)
```

### Pitfall 17: A knob the model never forwards
```python
# ❌ WRONG - the layer accepts dropout_rate; this construction never passes it
self.attn = MemoryAttention(dim=d, num_layers=n)

# ✅ CORRECT - and it appears in the variant table, __init__, get_config and the factory
self.attn = MemoryAttention(dim=d, num_layers=n, dropout_rate=dropout_rate)
```

### Pitfall 18: `save_own_variables` alongside the default save
```python
# ❌ WRONG - runs IN ADDITION to the recursive save; every weight written twice
def save_own_variables(self, store):
    for i, w in enumerate(self.weights):
        store[str(i)] = w.numpy()

# ✅ CORRECT - do not override unless you also suppress the default path, and
#    assert len(archive_datasets) == len(model.weights) in a test
```

### Pitfall 19: `compile_from_config` that drops the optimizer build
```python
# ❌ WRONG - reproduces the base method minus its last two lines; Adam resumes from zero
def compile_from_config(self, config):
    self.compile(**keras.saving.deserialize_keras_object(config))

# ✅ CORRECT
def compile_from_config(self, config):
    return super().compile_from_config(config)
```

### Pitfall 20: Saving an unbuilt model
```python
# ❌ WRONG - writes an archive with zero weights, and returns normally
def save_model(self, path):
    self.save(path)

# ✅ CORRECT - build from the stored shape, or refuse BEFORE writing
def save_model(self, path):
    if not self.built:
        raise ValueError("Cannot save an unbuilt model: ...")
    self.save(path)
```

### Pitfall 21: One initializer instance across several roles
```python
# ❌ WRONG - q, k, v come out BIT-IDENTICAL, with or without an explicit seed
init = keras.initializers.TruncatedNormal(0.02)
self.q = keras.layers.Dense(d, kernel_initializer=init)
self.k = keras.layers.Dense(d, kernel_initializer=init)

# ✅ CORRECT
self.q = keras.layers.Dense(d, kernel_initializer=clone_initializer(init))
self.k = keras.layers.Dense(d, kernel_initializer=clone_initializer(init))
```

### Pitfall 22: A variant table never checked against its reference
```python
# ❌ WRONG - a number someone remembered
MODEL_VARIANTS = {"tiny": {"hidden_size": 256, "intermediate_size": 512}}

# ✅ CORRECT - fetched, and cited where the next reader will look
#   Source: <url to the released config.json>, fetched YYYY-MM-DD
MODEL_VARIANTS = {"tiny": {"hidden_size": 256, "intermediate_size": 1024}}
```

### Pitfall 23: `plt.show()` in library code
```python
# ❌ WRONG - blocks headless runs and leaks the figure
def visualize(self, ...):
    plt.show()

# ✅ CORRECT - return the figure; the caller decides
def visualize(self, ..., show: bool = False):
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
```

### Pitfall 24: Mutable default arguments
```python
# ❌ WRONG - shared across every instance
def __init__(self, dims=[64, 128]):

# ✅ CORRECT
def __init__(self, dims=None):
    self.dims = list(dims) if dims is not None else [64, 128]
```

### Pitfall 25: Inconsistent layer names
```python
# ❌ WRONG - auto-generated names shift when depth changes
self.blocks = [Block() for _ in range(n)]

# ✅ CORRECT
self.blocks = [Block(name=f"block_{i}") for i in range(n)]
```

---

## 17. Troubleshooting Guide

### 17.1 Debug Checklist

Run these in order. Each is cheap, and each has caught a shipped defect.

1. Does the round trip preserve **values**, over **two** trips, after perturbing every weight? (§7.1)
2. Does `len(archive_datasets) == len(model.weights)`? (§7.2)
3. Does `.build(shape)` alone materialize every weight `call()` uses? (§8)
4. Does a **non-default** value of every knob change something measurable? (§13.1)
5. Does every trainable weight move after **one real optimizer step**? (§11.3, §13.3)
6. Does the layer compute a non-linear function — `|f(2x) − 2f(x)| > 0`? (§13.3)
7. Do the variant numbers match a **fetched** reference? (§13.2)
8. Does the optimizer state survive a save/load? (§7.4)
9. What does `pytest -W always::UserWarning -rw` say? (§15.1)
10. Does the suite pass in one process as well as per-file? (§14.9)

### 17.2 Errors and Symptoms

| Symptom | Look at |
|---|---|
| Reloaded model has 0 or too few weights | §8 build materialization; §7.6 unbuilt save |
| Round trip fine, file suspiciously large | §7.2 archive layout; §7.3 override pair |
| Training resumes but converges oddly | §7.4 optimizer state discarded on reload |
| `Skipping variable loading for optimizer` | §7.5 saved before the optimizer was built |
| `A total of N objects could not be loaded` | §7.5, or a deliberately-mismatched load — check which |
| `Could not interpret activation function identifier` | §6.1 raw callable in `get_config` |
| A knob changes nothing | §13.1 unreachable; §12.6 dead knob |
| A weight never moves | §13.3 dead weight; §13.4 inert by design — read the `call()` branch |
| Loss is suspiciously good on a causal task | §12.1 missing causal mask |
| Attention attends to padding | §12.3 mask destroyed; pass an explicit mask |
| A test flakes 1 in 12 | §10.5 falsy seed; §14.6 bar inside its own population; §14.9 RNG leak |
| NaNs only under fp16 | §10.1 sentinel; §10.2 margin below the working ULP |
| Update is ~2^15 too small | §11.1 missing `scale_loss` |
| `OperatorNotAllowedInGraphError` | §4.2 a traced `training`, or a Python branch on a tensor |
| `pred must not be a Python bool` | §4.2 `ops.tril`/`ops.triu` under tracing |
| Passes alone, fails in its directory | §14.9 process-global state |
| Passes in its directory, fails in a combined run | §14.9, and check for import-time global mutation |
| Guard never fails | §14.3 wrong arm; §14.11 vacuity |
| A "pre-existing" failure you cannot reproduce | §14.7 the detached-worktree `PYTHONPATH` trap |

---

## 18. Summary Checklists

### 18.1 A New Layer

- [ ] `@keras.saving.register_keras_serializable(package=...)`
- [ ] Every sub-layer created in `__init__`, **unconditionally**
- [ ] `build()` materializes exactly what `call()` runs; guarded by `if self.built: return`
- [ ] No `.assign()` of a constant table in `build()` — the initializer *is* the value
- [ ] `compute_output_shape` reads config, never weights
- [ ] `get_config` complete; objects **serialized**; `from_config` symmetric; **both directions
      tested separately**
- [ ] `is not None` for every numeric parameter where `0` is legal
- [ ] One initializer **instance** per consumer (clone it)
- [ ] `keras.ops` only in `call()`; no Python branch on a tensor value; no state mutation
- [ ] fp16-safe sentinels; margins derived from the working ULP
- [ ] `supports_masking` set truthfully — and the leak fixed separately if there is one
- [ ] Explicit `name=` on every sub-layer, including inside loops
- [ ] Validation in `__init__` and `build`, never `call`; messages name the offending value

### 18.2 A New Model Package

- [ ] Module docstring is substantive prose: principle, architecture, deliberate choices, references
- [ ] One variant table, one home; architecture and training config kept separate
- [ ] Every variant number traced to a **fetched** reference, with the URL cited — or explicitly
      classed as paper-table-only, or as original-with-no-upstream
- [ ] Every layer knob reachable from `__init__`, the variant table, `get_config` **and** the factory
- [ ] `from_variant` raises `ValueError` listing the available keys, and accepts the overrides its
      docstring advertises
- [ ] Factory exported in `__all__`
- [ ] `pretrained=True` raises rather than returning random weights
- [ ] `compile_from_config` calls `super()` or reproduces it completely
- [ ] No `plt.show()` anywhere in the library path
- [ ] No custom `train_step` unless justified; if present, it uses `scale_loss`

### 18.3 The Tests, Before You Call It Done

- [ ] Two-round-trip **value** equality, after perturbing every weight
- [ ] `len(archive_datasets) == len(model.weights)`, and no unexpected flat store
- [ ] Optimizer state survives a save/load
- [ ] `.build(shape)` materializes everything `call()` runs; un-traceable classes pinned by name
- [ ] Knob sensitivity at a **non-default** value, asserted as a **set** over live layers
- [ ] Gradient flow after **one optimizer step**; waivers pinned by name, set-equal both ways
- [ ] Non-linearity check for any layer claiming one
- [ ] Causal isolation check for any causal model, with the control that proves the probe reached it
- [ ] Every tolerance derived from a dtype or a ≥20-sample population, **derivation in the test**
- [ ] Every guard proven RED by injection, with the firing assertion named, and restored via `cp` +
      `diff -q` (never `git stash`)
- [ ] False-positive family checked for any guard that can trip on innocent edits
- [ ] Every deliberate advisory paired with a `pytest.warns` positive arm **and** a control
- [ ] `passed` quoted **with** `collected`; no suite silently collecting zero
- [ ] An autouse fixture restoring global RNG, dtype policy and `floatx`
- [ ] Failing node-id **sets** compared across runs, never counts

---

## 19. Appendix: Refuted Claims

Recorded so they are not re-proposed, and so a rule above is not re-derived from a premise already
falsified by measurement.

| Claim | Status |
|---|---|
| The nested `List[List[Layer]]` weight-loss trap | **Does not reproduce on Keras 3.8** in general — `_flatten_layers` round-trips regardless of nesting. Where it *does* bite, the discriminating property is the **owner class** (`Layer` vs `Model`), not container depth |
| "Overrides `build()`" is the discriminating property for round-trip loss | **Wrong property.** Whether `build()` *materializes the tree* is (§8.2) |
| `Model.build(shape)` builds a subclassed model | **False.** It marks it built and walks no sub-layers; `count_params()` returns exactly **0** |
| A structured-dict `y_pred` cannot be used with stock `compile()`/`fit()` | **False.** It breaks in exactly one configuration: a single `Loss` object handed a dict `y_pred` |
| A custom `train_step` drops regularizer terms | **False.** Keras 3's `compute_loss` already sums `self.losses` |
| `assert model.losses` proves a regularizer is live | **Vacuous** when an unrelated block contributes. Assert a delta against a no-regularizer baseline |
| `x + g - stop_gradient(g)` is the identity | **False** under left-to-right float association (~25% of float64 draws differ by up to 1 ULP). Group as `x + (g - stop_gradient(g))` |
| A GPU-only homogeneity RED at `5.063e-04` was a bias leak | **It was the TF32 ULP** (§14.7) |
| A hard-coded `training=None` in a functional-graph build breaks training mode | **False.** `Functional.call` injects the caller's `training` into every traced operation at runtime — measured identical **with the `call` override deleted entirely**. The trace-time value is dead thereafter |
| `include_optimizer=False` suppresses optimizer state on the `.keras` path | **False — a silent no-op.** The kwarg is popped and never forwarded |
| Only an *explicitly seeded* shared initializer produces identical weights | **False.** A shared **instance** produces bit-identical weights with or without `seed=`; instance identity is the discriminator (§10.3) |
| A halting head that gets no gradient under its factory `compile()` is untrained | **False.** It trains in the custom loop written for it; the factory docstring said so eight lines above the line the finding read (§11.4) |
| Routing two model families through a shared normalization factory is a safe cleanup | **False.** The factory's epsilon default differs from the layer's by **1000×**; 11 of 16 types diverge (§9.6) |
| A published architecture's BatchNorm momentum was wrong in the port | **False.** The reference covering all shipped generations declares exactly what shipped; an *older* reference disagrees. The **epsilon** was the real deviation (§9.7) |
| The reference initializer argument is the value to assert | **False for truncated normals.** Realized std is `0.87964 ×` the argument; pin the depth **ratio** instead (§10.4) |
| Five "objects could not be loaded" warnings indicated weight loss | **False.** All five were deliberately-mismatched loads; the warning was the instrument working. Two-round-trip probes measured `0.0` on all four real subjects |
| An exact-zero pin on a numerical residual is a strict test | **Broken.** `7.45e-09` is exactly one float32 ULP at 0.1 — the assertion can never pass, so it measures nothing (§14.6) |
| A weight-path set is stable across a round trip | **Not always.** A `clone_model` teacher inherits the student's path strings live and separates on reload — 172 paths before, 322 after, harmlessly (§7.8) |
| A single `-W error::UserWarning` run measures the warning population | **False — it is a lower bound.** It aborts each test at the first warning: 58 node ids against 310 (§15.1) |

**The meta-lesson.** Across every audit that produced this document, the dominant failure mode was
not a wrong fix — it was **a real symptom with a wrong explanation**:

- a doubled archive whose own override comment claimed the opposite;
- a "dead" head that trains in the loop written for it;
- a "flaky" test that was a falsy seed;
- a "momentum" defect that was an epsilon defect;
- a "linear-algebra" bug that was a linear *layer*;
- a "test-order" flake that was one unrestored global seed.

In the most recent audit roughly **one carried premise in five died on measurement**, and **six of the
author's own prescriptions were falsified before they shipped** — one of which would have divided 189
layers' epsilon by 1000, and another of which would have made a correct fix fail its own guard.

Several *prescribed fixes* were themselves regressions, caught only by running them and diffing the
number:

- bias-correcting an EMA codebook without also zero-initializing it made the defect ~**10× worse**;
- forwarding a "dropped" dropout rate stacked a **second** dropout, so a requested 0.25 became an
  effective 0.4375;
- landing a global warning-to-error configuration last converted an earlier step's deliberate
  advisory into a failure (§15.5).

> **Re-derive the premise at the moment you act on it. Then run the prescribed fix and diff the
> number, not the shape.**
