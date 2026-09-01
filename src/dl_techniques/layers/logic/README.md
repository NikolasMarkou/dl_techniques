# `dl_techniques.layers.logic`

Differentiable, learnable logical and arithmetic primitives plus a stackable
"neural circuit" built on top of them. All layers are end-to-end
differentiable, fully serializable, and shape-preserving.

## Overview

The package exposes four layer classes and a factory:

| Class | Purpose | Shape contract |
|---|---|---|
| `LearnableArithmeticOperator` | DARTS-style soft selection over 7 operations: `add / multiply / subtract / divide / power / max / min`. All 7 are selected by default | rank-agnostic (rank >= 1); preserves shape |
| `LearnableLogicOperator` | Soft fuzzy logic over **18** gates in `VALID_OPS`. The default set is the first 6, `and / or / xor / not / nand / nor`; the other 12 (Lukasiewicz, Godel, Hamacher, Yager, 4 implications) are opt-in. Inputs are sigmoid-normalized unless `apply_sigmoid=False` | rank-agnostic (rank >= 1); preserves shape |
| `CircuitDepthLayer` | One MoE-style stage combining parallel logic + arithmetic experts with learnable routing + fusion | **rank >= 2**; preserves shape |
| `LearnableNeuralCircuit` | Stack of `CircuitDepthLayer` with optional `LayerNormalization` between stages | **rank >= 2**; preserves shape |

The factory `create_logic_layer(layer_type, **kwargs)` follows the registry
pattern of `layers/ffn/factory.py` and `layers/norms/factory.py`, including
its treatment of an unknown keyword: **it raises `ValueError`, naming the key
and the accepted set.** See the Factory section below.

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

Inputs are sigmoided into `[0, 1]` first, unless `apply_sigmoid=False`. The
six default gates are:

```
NOT(p) = 1 - p
AND(p, q) = p * q
OR(p, q)  = p + q - p*q
XOR(p, q) = p + q - 2*p*q
NAND, NOR = 1 - AND, 1 - OR
```

`VALID_OPS` holds 18 gates in total; the class docstring of
`LearnableLogicOperator` prints the formula for every one. The gate results
are combined by the same softmax-weighted mixture the arithmetic operator
uses.

### Circuit depth layer (MoE-style)

For input `X`, routing weights `w_r`, combination weights `w_c`, and `N`
expert operators `f_i`:

```
beta  = softmax(w_c)              # output fusion
Y     = sum_i( beta_i * f_i(X) )            [+ X]   # circuit_routing='output_only', the DEFAULT

alpha = softmax(w_r)              # input gating, read only by 'classic'
Y     = sum_i( beta_i * f_i(alpha_i * X) )  [+ X]   # circuit_routing='classic'
```

`'output_only'` is the default. Under `'classic'` each expert sees `X / N` on
average, so the signal shrinks as experts are added; keep that mode only to
load models trained before the default changed. `w_r` is created in both
modes so a checkpoint from either loads into either.

`LearnableNeuralCircuit` stacks `D` such depth layers with optional
`LayerNormalization` between stages.

## Classes

All four classes carry `@register_dl_technique(...)`, from
`dl_techniques.utils.keras_registration`. **The string each passes is
its own module's dotted path**, so the keys are
`dl_techniques.layers.logic.arithmetic_operators>LearnableArithmeticOperator`,
`dl_techniques.layers.logic.logic_operators>LearnableLogicOperator`, and
`dl_techniques.layers.logic.neural_circuit>` + `CircuitDepthLayer` /
`LearnableNeuralCircuit` — the module path is now **in** the key. So **renaming** a class
breaks a saved `.keras` archive, and so does **moving** one between these modules; what can
no longer happen is a same-named class elsewhere under `layers/` silently taking the same
slot. The helper additionally binds the legacy
`Custom><ClassName>` key as an alias to the same object, and
`keras.saving.get_registered_object("Custom>LearnableNeuralCircuit")` returns this class,
which is what keeps older archives loading. The `__init__.py`
only re-exports the symbols; it does not re-register them.

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
- **raises `ValueError` on any key the target type does not declare**:
  `create_logic_layer("logic", bogus_key=1)` raises, naming the count, the key
  and the accepted set, on all 4 registry keys. This matches every other factory
  in `layers/` (`layers/CLAUDE.md` § The factory contract). The check lives in
  `validate_logic_config`, so calling that directly rejects the same key, and
  every message carries `STRICT_UNSUPPORTED_KEY_MARKER` for tests to match on.
  Read the accepted set off the error or off `get_logic_info()` and correct the
  spelling,
- logs the final parameter set at debug level,
- raises a `ValueError` with a contextual message on construction failure.

`LOGIC_REGISTRY` has 4 entries. Each has an empty `required_params`, so every
key constructs with no arguments, and the `optional_params` counts are:

| Key | Class | `optional_params` | `enum_params` |
|---|---|---:|---:|
| `arithmetic` | `LearnableArithmeticOperator` | 18 | 3 |
| `logic` | `LearnableLogicOperator` | 14 | 1 |
| `circuit_depth` | `CircuitDepthLayer` | 17 | 3 |
| `neural_circuit` | `LearnableNeuralCircuit` | 18 | 4 |

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

## Optional features and defaults

Everything below is available on the current classes; the defaults are the ones the
constructors ship with.

| Parameter | Class(es) | Default | Notes |
|---|---|---|---|
| `circuit_routing` | `CircuitDepthLayer`, `LearnableNeuralCircuit` | `'output_only'` | `'classic'` reproduces the old `Y=Sum beta_i*f_i(alpha_i*X)` attenuated math. |
| `apply_sigmoid_per_depth` | `LearnableNeuralCircuit` | `'first_only'` | `'all'` and `'none'` also selectable. |
| `softplus_temperature` | logic + arithmetic ops | `True` | Always-positive temperature with a gradient defined everywhere. Round-trips through `from_config`. |
| `operation_initializer` / `routing_initializer` / `combination_initializer` | all four | `'zeros'` | Every weight clones the initializer it is given (`initializers/clone.py`), so one `Initializer` INSTANCE shared across parameters still leaves each weight an independent draw. A *seeded* instance keeps its seed and draws identically, which is why a guard for this cannot use one. |
| `allow_unary_degenerate` | `LearnableLogicOperator` | `False` | `True` opts back into the legacy single-tensor rebinding. |
| `safe_divide_mode` | `LearnableArithmeticOperator` | `'hard_clamp'` | `'smooth'` gives a bounded gradient at `x2=0`. |
| `exponent_clip_mode` / `exponent_clip_range` | `LearnableArithmeticOperator` | `'hard'` / `(-2.0, 2.0)` | `'smooth'` is a tanh range squash with non-zero gradient at the boundary. |
| `gumbel_softmax` (+ `gumbel_hard`) | logic + arithmetic ops | `False` | Discrete-at-inference selection. `_operation_probs` computes `softmax((w + g) / T)` per Jang (2017), and injects noise only when `training is True`. |
| `entropy_coefficient` | logic + arithmetic ops | `0.0` | Adds `coef * H(probs)` to `layer.losses`. Because total loss is minimized this **sharpens** selection, the opposite of the NAS convention. Raise it for faster symmetry-breaking; leave `0.0` for a soft mixture. |
| `gate_entropy_coefficient` | `CircuitDepthLayer`, `LearnableNeuralCircuit` | `0.0` | Penalizes peaky combination distributions, per-channel L2 then mean. `load_balance_coefficient` is a deprecated alias that emits a `DeprecationWarning`; the serialized config uses the new name. |
| `diversity_coefficient` | `CircuitDepthLayer`, `LearnableNeuralCircuit` | `0.0` | Pairwise cosine-similarity aux loss between same-arity inner experts, computed from a per-arity Gram matrix. |
| `channel_mix` | `CircuitDepthLayer`, `LearnableNeuralCircuit` | `None` | `'dense'` appends a `Dense(C)` after fusion — C units, bias on, so a `(C, C)` kernel that mixes channels. |
| `selection_mode` | all four | `'global'` | `'per_channel'` stores `(channels, num_operations)` weights so each channel selects its own operator; needs a concrete last-axis dim at build time. |
| `force_clip_when_no_sigmoid` | `LearnableLogicOperator` | `False` | With `apply_sigmoid=False`, defensively clips inputs to `[0, 1]`. Auto-enabled on depths >= 1 inside `LearnableNeuralCircuit` under `apply_sigmoid_per_depth='first_only'` with arithmetic experts or a residual. |
| `inner_logic_kwargs` / `inner_arithmetic_kwargs` | `CircuitDepthLayer`, `LearnableNeuralCircuit` | `None` | Dict forwarded to the inner ops (`temperature_init`, `gumbel_softmax`, `yager_p`, ...). Wrapper-owned keys win; collisions are warned about. |
| `yager_p` | `LearnableLogicOperator` | `2.0` | Sharpness of the Yager t-norm ops. Round-trips. |
| `use_scaling` / `scaling_init` | `LearnableArithmeticOperator` | `True` / `1.0` | The scaling factor's magnitude is clamped to `>= 1e-7` and its sign is learnable. Initialization must be positive (`scaling_init` validation enforces it), but training can drive the sign negative. |

Other behaviour worth knowing:

- **`_safe_power` is sign-preserving.** The real restriction `cos(pi*y)*|x|^y` keeps the sign of a
  negative base. `power(-2, 3)` returns 4.0, not -8, because the default
  `exponent_clip_range=(-2.0, 2.0)` clips the exponent 3 down to 2; with
  `exponent_clip_range=(-4.0, 4.0)` it returns -8.0.
- **`to_symbolic(top_k=k, deterministic=True)`** prints the dominant ops. `deterministic=True`
  (the default) skips Gumbel noise so the printed selection is reproducible during training.
  `LearnableNeuralCircuit.to_symbolic()` walks all depths and includes per-depth combination
  weights; `CircuitDepthLayer` has a standalone one.
- **`compute_output_shape` validates binary-input shape consistency** on both operators and
  raises on a mismatch.
- **Annealing callback.** `dl_techniques.callbacks.temperature_annealing.TemperatureAnnealingCallback`
  anneals `temperature` across epochs with a cosine / linear / exp schedule, honouring
  `softplus_temperature=True` by setting raw = `log(expm1(t))`.
- **Enum pre-validation.** `validate_logic_config` checks `selection_mode`, `circuit_routing`,
  `safe_divide_mode`, `apply_sigmoid_per_depth`, `channel_mix` and `exponent_clip_mode` upfront.

## Limitations

- **Unary inputs to binary operators.** The two classes handle this
  differently now. `LearnableLogicOperator` **raises** `ValueError` on a single
  tensor whenever any selected gate is binary, which is 17 of the 18 (`not` is
  the only unary one); `allow_unary_degenerate=True` opts back into the old
  rebinding. `LearnableArithmeticOperator` still rebinds `x2 = x1` without
  complaint, and that degenerates `subtract -> 0` and `divide -> 1` (measured
  for both signs). Its remaining ops (`add`, `multiply`, `power`, `max`,
  `min`) stay meaningful, but the soft mixture is then biased toward them.
  **If your data is genuinely unary, prefer a different layer.**
- **Rank requirement.** `CircuitDepthLayer` and `LearnableNeuralCircuit` accept
  **rank >= 2**; the sibling arithmetic / logic operators are rank-agnostic.
- **`@register_dl_technique("dl_techniques.layers.logic.<module>")`.** The key is
  `dl_techniques.layers.logic.<module>><ClassName>` and names the defining module, so
  **renaming** one of these classes — or **moving** it to another module in this package —
  breaks every `.keras` archive saved with a prior version. The legacy
  `Custom><ClassName>` alias the helper also binds is keyed on the bare class name only, so
  a rename drops that as well while a move does not. New callers are encouraged to use the
  factory; class-direct imports remain fully supported.
- **No internal projection.** Output channel count equals input channel
  count. Pair with a `Dense` / `Conv` if you need dimensionality change.
- **Stacking `LearnableLogicOperator` re-squashes via sigmoid every layer
  WHEN `apply_sigmoid_per_depth='all'`.** With the default
  `apply_sigmoid_per_depth='first_only'` (set on `LearnableNeuralCircuit`)
  only the first depth applies sigmoid; subsequent depths receive `[0, 1]`
  values from the prior fuzzy output, optionally force-clipped (see
  `force_logic_input_clip`). The empirical collapse `1.76 -> 0.06 -> 0.003`
  was measured under the legacy `'all'` mode. For stacked use, prefer
  `'first_only'` or pass `apply_sigmoid=False` directly to all but the
  first inner op.
- **`divide` op has unbounded gradients near zero.** `_safe_divide` clamps the
  forward output (so no NaN/Inf), but `d(x1/x2)/dx2 -> -x1/eps^2` as `|x2|`
  approaches zero. If `divide` is in the softmax pool and inputs can be small,
  gradient explosions are possible. Either avoid `divide` in stacked use or
  guarantee features are bounded away from zero.
- **Default `arithmetic_op_types` is numerically aggressive in deep stacks.**
  The default set includes `power` (with learned exponents bounded only by
  `exponent_clip_range`) and `divide`. When `LearnableNeuralCircuit` chains
  several `CircuitDepthLayer`s with `use_residual=True`, the residual
  compounds expert outputs across depths and `power`-magnitude amplification
  can drive the forward pass to NaN within a handful of epochs (empirical:
  depth=3, channels=32, K=4 parity, AdamW lr=3e-3 → NaN by epoch ~12). For
  stacks of depth >= 3, **explicitly restrict** to bounded ops, e.g.
  `arithmetic_op_types=['add', 'max', 'min']`, or use
  `exponent_clip_mode='smooth'` and `safe_divide_mode='smooth'` together.
  See `src/train/logic/train_boolean_circuit.py` for a working depth-2 K=4
  parity recipe.

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
