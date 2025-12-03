Here is the `GUIDE.md` for the activations module, structured to match the Attention module guide while addressing the specific requirements of activation functions (stateless math vs. stateful layers, numerical stability, and broadcasting).

```markdown
# Activation Layer Implementation Guide

This guide establishes the architectural standards, coding conventions, and best practices for implementing new activation layers within the `dl_techniques` framework.

**Goal:** Ensure all activation functions—from simple non-linearities to complex learnable gates—share a consistent API, guarantee numerical stability across backends, and integrate seamlessly with the centralized factory system.

---

## 1. Core Design Philosophy

### The "Factory First" Approach
All activation layers must be registered in the central factory. This allows models to define activations via string configuration (e.g., in YAML/JSON configs) without hardcoding class imports.
*   **Do not** create standalone scripts that cannot be imported by the factory.
*   **Do** define strict parameter validation logic in the factory registry.

### Keras 3.0 Compliance
*   **Backend Agnostic:** Use `keras.ops` for all mathematical operations. **Absolutely no** `tf.*`, `torch.*`, or `jax.*` primitives inside the layer logic.
*   **Serialization:** Every layer must be decorated with `@keras.saving.register_keras_serializable()` and implement a complete `get_config()`.
*   **Stateless vs. Stateful:**
    *   If the activation has **no parameters** (e.g., `GELU`), it is stateless.
    *   If the activation **learns** (e.g., `PReLU`, `TrainableThreshMax`), it is stateful. Weights must be created in `build()`.

---

## 2. API Standards & Naming Conventions

To maintain consistency across the library, adhere to these naming standards:

### Class Naming
*   **Standard:** Use CamelCase without suffixes (e.g., `Sparsemax`, `ThreshMax`, `Mish`).
*   **Complex Layers:** Use `Layer` suffix only if the component involves significant internal routing or projection logic (e.g., `HierarchicalRoutingLayer`, `MonotonicityLayer`).
*   **Expanded Variants:** Prefix with `x` for expanded gating ranges (e.g., `xGELU`, `xSiLU`).

### Parameter Standards

| Parameter Concept | Standard Name | Forbidden Names | Default Value |
| :--- | :--- | :--- | :--- |
| **Operation Axis** | `axis` | `dim`, `channel_axis` | `-1` |
| **Stability Constant** | `epsilon` | `eps`, `min_val` | `1e-7` or `1e-12` |
| **Learnable Slope** | `slope` | `alpha` (context dependent) | `1.0` or `10.0` |
| **Gating Scale** | `alpha` | `scale`, `multiplier` | `1.0` |
| **Shape/Dimension** | `output_dim` | `classes`, `units` | `None` (infer if possible) |

### Standard Constructor Signature
```python
def __init__(
    self,
    axis: int = -1,
    epsilon: float = 1e-7,
    # ... layer specific args ...
    **kwargs: Any
) -> None:
    super().__init__(**kwargs)
    # Validation logic here
```

---

## 3. Mathematical Standards & Ops

**CRITICAL:** Activations are the most sensitive components regarding numerical stability.

### 3.1. The `keras.ops` Rule
All math must use Keras 3 operations to ensure cross-backend compatibility.

```python
# ✅ CORRECT
import keras
from keras import ops

x = ops.sigmoid(x)
x = ops.maximum(x, 0.0)

# ❌ INCORRECT
import tensorflow as tf
import numpy as np

x = tf.nn.sigmoid(x)
x = np.maximum(x, 0) # Breaks symbolic tracing
```

### 3.2. Numerical Stability
Always protect divisions and logarithms.

```python
# ✅ CORRECT
safe_norm = ops.sqrt(squared_norm + self.epsilon)
log_prob = ops.log(prob + self.epsilon)

# ❌ INCORRECT
norm = ops.sqrt(squared_norm) # NaN gradient at 0
```

### 3.3. Broadcasting
When working with `axis`, ensure auxiliary tensors broadcast correctly against the input.

```python
# Handling axis logic
ndim = len(input_shape)
# Normalize negative axis
actual_axis = self.axis if self.axis >= 0 else ndim + self.axis

# Reshaping scalars/vectors to broadcast
target_shape = [1] * ndim
target_shape[actual_axis] = -1
param_broadcast = ops.reshape(param, target_shape)
```

---

## 4. Implementation Patterns

### Pattern A: Stateless Activation (Math Function)
For fixed mathematical functions (e.g., `Mish`, `HardSwish`).

```python
@keras.saving.register_keras_serializable()
class MyActivation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # No weights to initialize

    def call(self, inputs):
        # Pure math
        return inputs * ops.tanh(ops.softplus(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape
```

### Pattern B: Learnable Activation (Stateful)
For activations with trainable parameters (e.g., `DifferentiableStep`, `HierarchicalRouting`).

```python
@keras.saving.register_keras_serializable()
class TrainableActivation(keras.layers.Layer):
    def __init__(self, initializer='ones', **kwargs):
        super().__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        
    def build(self, input_shape):
        # Create weights here, NOT in __init__
        self.alpha = self.add_weight(
            name="alpha",
            shape=(), # or (input_shape[-1],) for per-channel
            initializer=self.initializer,
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.alpha
```

---

## 5. Factory Integration Checklist

When adding a new layer, you must update `dl_techniques/layers/activations/factory.py`.

1.  **Define Type Literal:** Add your layer string key to `ActivationType`.
2.  **Update Registry:** Add an entry to `ACTIVATION_REGISTRY`.
3.  **Parameter Definition:**
    *   `required_params`: Params that **must** be passed (raising error if missing).
    *   `optional_params`: Params with default values.

**Example Registry Entry:**
```python
'my_new_act': {
    'class': MyNewActivation,
    'description': 'Description of what it does.',
    'required_params': [],
    'optional_params': {
        'slope': 1.0,
        'axis': -1
    },
    'use_case': 'Brief note on when to use it.'
}
```

4.  **Validation Logic:** Update `validate_activation_config` if your layer has specific constraints (e.g., `slope` must be positive).

---

## 6. Common Pitfalls to Avoid

1.  **Mixing Types in `__init__`:** Do not assume inputs are tensors in `__init__`. Use `build()` for shape-dependent logic.
2.  **Implicit Integer Conversion:** When performing math on shapes, cast explicitly.
    ```python
    # Bad
    uniform_prob = 1.0 / num_classes 
    
    # Good
    uniform_prob = 1.0 / ops.cast(num_classes, x.dtype)
    ```
3.  **Missing `get_config` Updates:** If you add `self.beta` in `__init__`, it **must** appear in `get_config`.
4.  **Forgetting `@register_keras_serializable`:** This causes model loading to fail with "Unknown layer".
5.  **Re-implementing Standard Layers:** If Keras has an optimized kernel (like `ReLU`), use `keras.layers.ReLU` unless you are specifically modifying the logic (e.g., `ReLUK`).

---

## 7. Migration of Legacy Code

If refactoring older activation functions:
1.  **Remove `tf.math`:** Replace all TensorFlow primitives with `keras.ops`.
2.  **Standardize Axes:** Ensure `axis` defaults to -1 and handles negative indexing.
3.  **Clean Configs:** Ensure `get_config` returns serialized initializers/regularizers, not the objects themselves (use `keras.initializers.serialize`).
4.  **Add Type Hints:** Ensure `call` takes `keras.KerasTensor` and returns `keras.KerasTensor`.
```