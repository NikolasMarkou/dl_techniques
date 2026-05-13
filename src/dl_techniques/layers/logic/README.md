# `dl_techniques.layers.logic`

Differentiable, learnable logical and arithmetic primitives plus a stackable
"neural circuit" built on top of them. All layers are end-to-end
differentiable, fully serializable, and shape-preserving.

## Overview

The package exposes four layer classes and a factory:

| Class | Purpose | Shape contract |
|---|---|---|
| `LearnableArithmeticOperator` | DARTS-style soft selection over `add / multiply / subtract / divide / power / max / min` | rank-agnostic (rank >= 1); preserves shape |
| `LearnableLogicOperator` | Soft fuzzy logic over `and / or / xor / not / nand / nor` (sigmoid-normalized inputs) | rank-agnostic; preserves shape |
| `CircuitDepthLayer` | One MoE-style stage combining parallel logic + arithmetic experts with learnable routing + fusion | **rank >= 2**; preserves shape |
| `LearnableNeuralCircuit` | Stack of `CircuitDepthLayer` with optional `LayerNormalization` between stages | **rank >= 2**; preserves shape |

The factory `create_logic_layer(layer_type, **kwargs)` mirrors the registry
pattern used in `layers/ffn/factory.py` and `layers/norms/factory.py`.

## Math

### Soft operation selection (DARTS-style)

Given a learnable weight vector `w` (one per operation), a (learnable)
temperature `T`, and candidate operations `f_i`:

```
p_i = exp(w_i / T) / sum_j(exp(w_j / T))
Y   = s * sum_i( p_i * f_i(X) )       # s is an optional learnable scaling factor
```

`T -> 0` sharpens toward one-hot selection; `T -> inf` flattens toward a
uniform combination.

### Soft fuzzy logic

Inputs are first sigmoided into `[0, 1]`, then combined via:

```
NOT(p) = 1 - p
AND(p, q) = p * q
OR(p, q)  = p + q - p*q
XOR(p, q) = p + q - 2*p*q
NAND, NOR = 1 - AND, 1 - OR
```

Combined via the same softmax-weighted mixture used by the arithmetic
operator.

### Circuit depth layer (MoE-style)

For input `X`, routing weights `w_r`, combination weights `w_c`, and `N`
expert operators `f_i`:

```
alpha = softmax(w_r)              # input gating
beta  = softmax(w_c)              # output fusion
Y     = sum_i( beta_i * f_i(alpha_i * X) )  [+ X]   # optional residual
```

`LearnableNeuralCircuit` stacks `D` such depth layers with optional
`LayerNormalization` between stages.

## Classes

All classes are registered with `@keras.saving.register_keras_serializable()`
at their **original module paths**; relocating them would break previously
saved `.keras` archives. The `__init__.py` only re-exports the symbols — it
does not re-register them.

```python
from dl_techniques.layers.logic import (
    LearnableArithmeticOperator,
    LearnableLogicOperator,
    CircuitDepthLayer,
    LearnableNeuralCircuit,
    create_logic_layer,
    LogicLayerType,
)
```

## Factory

```python
from dl_techniques.layers.logic import create_logic_layer, get_logic_info

# Print all available types and their parameters
info = get_logic_info()
for k, v in info.items():
    print(k, v["description"])

# Construct by string id
op   = create_logic_layer("arithmetic", operation_types=["add", "multiply", "max"])
gate = create_logic_layer("logic")
unit = create_logic_layer("circuit_depth", num_logic_ops=4, num_arithmetic_ops=4)
deep = create_logic_layer("neural_circuit", circuit_depth=6, use_layer_norm=True)
```

The factory:

- validates `layer_type` against `LOGIC_REGISTRY`,
- merges registry defaults with user kwargs,
- filters unknown keys (no `Unrecognized keyword arguments` from Keras),
- logs the final parameter set,
- raises a `ValueError` with a contextual message on construction failure.

`create_logic_from_config({"type": "neural_circuit", "circuit_depth": 4})`
is the equivalent dict-driven entry point.

## Integration — when to use, when NOT to use

**Use this package when** you want a learnable, differentiable, shape-preserving
non-linearity that can express simple symbolic / arithmetic combinations
inside a larger network. The MoE-style fusion is mid-network friendly — drop
it in like a residual block.

**Do NOT use this package when**:

- You want an FFN-shaped block (`(B, T, D) -> (B, T, D_out)`). Use
  `dl_techniques.layers.ffn.LogicFFN` instead — it is dimension-changing and
  better integrated with transformer stacks.
- You need a hard (non-differentiable) logical operation. These layers are
  continuous relaxations; output values are real-valued, not Boolean.
- You expect the output to dominate via a single operation early in training.
  Without an entropy regularizer on the gate weights, the soft mixture tends
  to remain diffuse for a long time.

## What's new in `plan_2026-05-13_a2b0f17b` (full-rewrite override)

The deep review's findings were implemented in full — every change is opt-in
or back-compatible by default unless noted as **BREAKING**.

| Change | Flag / API | Default | Notes |
|---|---|---|---|
| **BREAKING**: `CircuitDepthLayer` no longer attenuates input | `circuit_routing='output_only'` | new default | Old math: `Y=Σβ_i·f_i(α_i·X)`. New: `Y=Σβ_i·f_i(X)`. Set `circuit_routing='classic'` to reproduce old. |
| Sigmoid plumbing through circuit | `LearnableNeuralCircuit(apply_sigmoid_per_depth=...)` | `'first_only'` | Was `'all'` (legacy collapse). `'all'` and `'none'` still selectable. |
| Sign-preserving `_safe_power` | (auto) | — | `power(-2, 3) == -8`. Real restriction `cos(πy)·\|x\|^y`. |
| Smooth-clamp divide | `safe_divide_mode='smooth'` | `'hard_clamp'` | Bounded gradient at `x2=0`; opt-in. |
| Softplus-parameterized temperature | `softplus_temperature=True` | `False` | Always-positive, gradient-defined-everywhere. Round-trip via `from_config`. |
| Strict raise on single-input + binary ops | `allow_unary_degenerate=False` | `True` | Default still permits the legacy footgun. |
| Łukasiewicz / Gödel / implication ops | `operation_types=['lukasiewicz_and', ...]` | not in defaults | New tokens: `lukasiewicz_and`, `lukasiewicz_or`, `godel_and`, `godel_or`, `implies`. |
| Gumbel-softmax mode | `gumbel_softmax=True[, gumbel_hard=True]` | `False` | Discrete-at-inference selection. |
| Entropy regularization | `entropy_coefficient=0.1` | `0.0` | Adds `coef · H(probs)` to layer.losses. |
| Shazeer load-balance aux loss | `CircuitDepthLayer(load_balance_coefficient=0.1)` | `0.0` | Penalizes peaky combination distributions. |
| Cross-channel mixing | `CircuitDepthLayer(channel_mix='dense')` | `None` | Appends per-channel `Dense(C, C)` after fusion. |
| `to_symbolic(top_k=k)` | method | — | Returns a string of dominant ops post-training. |
| Vectorized weighted sum | (auto) | — | Replaces Python loop with `ops.stack` + `ops.sum`. |

**Migration**: existing `.keras` archives load unchanged because every new
flag has a back-compat default. Models that depend on the old attenuated
routing should pin `CircuitDepthLayer(circuit_routing='classic')`.

## Limitations

- **Unary inputs to binary operators.** Passing a single tensor (e.g. a plain
  `(B, D)`) to `LearnableArithmeticOperator` or `LearnableLogicOperator`
  causes `x1 == x2`, which degenerates `subtract -> 0`, `divide -> 1`,
  `xor -> 0`, `nand/nor -> constants`, and any binary-logic op into a fixed
  value. The remaining unary-friendly ops (`add`, `multiply`, `power`,
  `max`, `min`, `and`, `or`) still produce meaningful gradients, but the soft
  mixture is biased toward those operations. **If your data is genuinely
  unary, prefer a different layer.**
- **Rank requirement.** Prior to this iteration `CircuitDepthLayer` and
  `LearnableNeuralCircuit` enforced strict 4-D inputs. This has been
  relaxed to **rank >= 2** — the math was always rank-agnostic. Sibling
  arithmetic / logic operators were already rank-agnostic.
- **Bare `@register_keras_serializable()`.** Renaming or relocating any of
  these classes will break `.keras` archives saved with prior versions. New
  callers are encouraged to use the factory; class-direct imports remain
  fully supported.
- **No internal projection.** Output channel count equals input channel
  count. Pair with a `Dense` / `Conv` if you need dimensionality change.
- **Stacking `LearnableLogicOperator` re-squashes via sigmoid every layer.**
  By default each call applies `sigmoid` to both inputs, so the output of one
  logic layer fed into the next collapses toward `0.5` within 2-3 layers
  (empirical: input std `1.76 -> 0.06 -> 0.003 -> 0.0` across three default
  layers). For stacked use, pass `apply_sigmoid=False` to all but the first
  layer (and feed it inputs already in `[0, 1]`).
- **`divide` op has unbounded gradients near zero.** `_safe_divide` clamps the
  forward output (so no NaN/Inf), but `d(x1/x2)/dx2 -> -x1/eps^2` as `|x2|`
  approaches zero. If `divide` is in the softmax pool and inputs can be small,
  gradient explosions are possible. Either avoid `divide` in stacked use or
  guarantee features are bounded away from zero.

## Examples

### 1. Direct class usage

```python
import keras
from dl_techniques.layers.logic import LearnableNeuralCircuit

inputs = keras.Input(shape=(32, 32, 64))                # (B, H, W, C)
x = LearnableNeuralCircuit(
    circuit_depth=4,
    num_logic_ops_per_depth=3,
    num_arithmetic_ops_per_depth=3,
    use_residual=True,
    use_layer_norm=True,
)(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

### 2. Factory + sequence input (rank-3, post-relaxation)

```python
from dl_techniques.layers.logic import create_logic_layer

# (B, T, D) — sequence of token embeddings
circuit = create_logic_layer(
    "circuit_depth",
    num_logic_ops=4,
    num_arithmetic_ops=4,
    use_residual=True,
    name="reasoning_block",
)
```

### 3. Pairwise arithmetic operator on two tensors

```python
import keras
from dl_techniques.layers.logic import LearnableArithmeticOperator

a = keras.Input(shape=(128,))
b = keras.Input(shape=(128,))
fused = LearnableArithmeticOperator(
    operation_types=["add", "multiply", "max"],
)([a, b])
model = keras.Model([a, b], fused)
```

## References

- Liu, H., Simonyan, K., Yang, Y. (2018). *DARTS: Differentiable Architecture
  Search.* arXiv:1806.09055.
- Zadeh, L. A. (1965). *Fuzzy sets.* Information and Control, 8(3): 338-353.
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a
  Neural Network.* arXiv:1503.02531.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538.
- Garcez, A. S., Broda, K., Gabbay, D. M. (2002). *Neural-Symbolic Learning
  Systems: Foundations and Applications.* Springer.
