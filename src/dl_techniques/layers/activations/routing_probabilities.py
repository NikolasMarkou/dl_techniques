"""
Hierarchical routing tree for classification (deterministic or trainable).

This module unifies two routing variants behind a single layer,
``RoutingProbabilitiesLayer``, selectable via the ``mode`` parameter:

1. ``mode="deterministic"`` (default): Non-trainable, parameter-free routing
   using a fixed cosine basis projection. A drop-in alternative to softmax that
   introduces a structured, hierarchical bias without adding any trainable
   parameters.

2. ``mode="trainable"``: Learnable routing using a standard affine projection
   (``W x + b``). A drop-in replacement for ``Dense -> Softmax`` whose output
   projection cost is reduced from ``O(N)`` to ``O(log N)`` decisions.

Both modes share the same hierarchical probability tree:

1. **Padding**: ``output_dim`` is padded to the next power of two,
   ``padded_dim``. The number of routing decisions is
   ``d = log2(padded_dim)``.

2. **Decision Logits**: For each of the ``d`` decisions, a logit ``z_k`` is
   produced. In deterministic mode, ``z_k = <x, w_k>`` with
   ``w_{k,i} = cos(2*pi * (k+1) * i / D)``. In trainable mode,
   ``z = x W + b`` for a learnable ``W`` of shape ``[D, d]``.

3. **Probabilistic Decisions**: ``p_k = sigmoid(z_k)`` is the probability of
   taking the right branch at level ``k``.

4. **Hierarchical Routing**: Probability mass starts at 1.0 and is split at
   each level: ``left = parent * (1 - p_k)``, ``right = parent * p_k``.

5. **Renormalization**: The accumulated mass at each of the ``padded_dim``
   leaves is sliced to ``output_dim`` and renormalized to sum to 1.0.

References:
    - Zhang, Z., et al. (2024). "Softmax-free Large-scale Language Modeling".
      arXiv preprint arXiv:2402.01258.
    - Morin, F., & Bengio, Y. (2005). "Hierarchical Probabilistic Neural
      Network Language Model". AISTATS.
"""

import math
import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


# DECISION D-001: A single class with a `mode` flag is preferred over two
# distinct classes or an inheritance hierarchy. This keeps the shared
# axis-manipulation, tree-build, and slice/renormalize logic in one place.
# Trainable-only kwargs are accepted but ignored in deterministic mode so
# that get_config / from_config round-trip is symmetric.
@keras.saving.register_keras_serializable()
class RoutingProbabilitiesLayer(keras.layers.Layer):
    """
    Hierarchical routing layer for probabilistic classification.

    Supports two modes selected via ``mode``:

    - ``"deterministic"``: parameter-free routing using a fixed cosine basis
      projection. ``output_dim`` may be ``None`` (inferred at build time
      from the input shape at ``axis``).
    - ``"trainable"``: learnable routing via a Dense projection. ``output_dim``
      is required.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │    Input Features [batch, ..., D]       │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Decision Projection                    │
        │  deterministic: z = <x, cos_basis>      │
        │  trainable:     z = x W + b             │
        │  -> [batch, d]                          │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Sigmoid + Clip                         │
        │  p_k = sigma(z_k) in [eps, 1-eps]       │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Hierarchical Probability Tree          │
        │                                         │
        │            Root (p=1.0)                 │
        │           ┌───┴───┐                     │
        │       (1-p0)    (p0)                    │
        │       ┌──┴──┐  ┌──┴──┐                  │
        │      ...   ... ...   ...                │
        │       │     │   │     │                 │
        │      L0    L1  L2    L3  ...            │
        │                                         │
        │  Binary splits at each level k          │
        │  left = parent * (1 - p_k)              │
        │  right = parent * p_k                   │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Slice & Renormalize to output_dim      │
        │  Keep first output_dim leaves           │
        │  Renormalize to sum = 1.0               │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Output Probabilities [batch, ..., N]   │
        └─────────────────────────────────────────┘

    :param output_dim: Dimensionality of the output space (number of classes).
        In ``"deterministic"`` mode this may be ``None`` and is inferred from
        the dimension at ``axis`` of the input shape during build. In
        ``"trainable"`` mode it must be an integer greater than 1.
    :type output_dim: Optional[int]
    :param axis: Axis along which the routing is applied. Defaults to -1.
    :type axis: int
    :param epsilon: Small float added for numerical stability.
    :type epsilon: float
    :param mode: Routing mode. ``"deterministic"`` (default) for the
        parameter-free cosine-basis projection, or ``"trainable"`` for a
        learnable Dense projection.
    :type mode: str
    :param kernel_initializer: Initializer for the trainable kernel
        (``"trainable"`` mode only). Stored unconditionally so config is
        round-trip stable.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the trainable bias.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the trainable kernel.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_bias: Whether to use a bias vector in trainable mode.
    :type use_bias: bool
    :param kwargs: Additional arguments for the Layer base class.
    """

    _VALID_MODES = ("deterministic", "trainable")

    def __init__(
            self,
            output_dim: Optional[int] = None,
            axis: int = -1,
            epsilon: float = 1e-7,
            mode: str = "deterministic",
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if mode not in self._VALID_MODES:
            raise ValueError(
                f"'mode' must be one of {self._VALID_MODES}, got: {mode!r}"
            )

        if not isinstance(axis, int):
            raise ValueError(
                f"The 'axis' must be an integer, but received: {axis}"
            )

        if mode == "trainable":
            if not isinstance(output_dim, int) or output_dim <= 1:
                raise ValueError(
                    f"In 'trainable' mode, 'output_dim' must be an integer "
                    f"greater than 1, but received: {output_dim}"
                )
        else:  # deterministic
            if output_dim is not None:
                if not isinstance(output_dim, int) or output_dim <= 1:
                    raise ValueError(
                        f"The 'output_dim' must be an integer greater than 1, "
                        f"but received: {output_dim}"
                    )

        self.output_dim = output_dim
        self.axis = axis
        self.epsilon = epsilon
        self.mode = mode
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Computed in build()
        self.padded_output_dim: Optional[int] = None
        self.num_decisions: Optional[int] = None
        self._normalized_axis: Optional[int] = None

        # Projection weight (shape: [input_dim, num_decisions]).
        # Non-trainable cosine basis in deterministic mode, learnable in trainable mode.
        self.kernel = None
        self.bias = None  # trainable mode only

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer: compute tree dims and create projection state."""
        # Normalize axis
        input_rank = len(input_shape)
        if self.axis < 0:
            self._normalized_axis = input_rank + self.axis
        else:
            self._normalized_axis = self.axis

        if self._normalized_axis < 0 or self._normalized_axis >= input_rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        input_dim = input_shape[self._normalized_axis]

        # Infer output_dim if needed (deterministic mode only)
        if self.output_dim is None:
            if self.mode != "deterministic":
                # Defensive: __init__ should have rejected this, but check.
                raise ValueError(
                    "output_dim cannot be None in 'trainable' mode."
                )
            if input_dim is None:
                raise ValueError(
                    f"Cannot infer output_dim when the dimension at axis "
                    f"{self.axis} of input_shape is None. Please provide "
                    f"output_dim explicitly."
                )
            self.output_dim = int(input_dim)
            logger.info(
                f"[{self.name}] Inferred output_dim={self.output_dim} "
                f"from input shape: {input_shape} at axis {self.axis}"
            )

        if self.output_dim <= 1:
            raise ValueError(
                f"output_dim must be greater than 1, got {self.output_dim}"
            )

        # Padded power-of-two tree size
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))

        if input_dim is None:
            raise ValueError(
                f"The dimension at axis {self.axis} of input_shape must "
                f"be defined to build the projection kernel, got None."
            )

        logger.info(
            f"[{self.name}] ({self.mode}) Built for {self.output_dim} "
            f"classes along axis {self.axis}. Padded to "
            f"{self.padded_output_dim}, requiring {self.num_decisions} "
            f"routing decisions."
        )

        # DECISION D-003: both modes share the same projection
        # [input_dim, num_decisions] with attribute name `self.kernel`. The
        # only difference is whether it is a trainable Keras weight. In
        # deterministic mode the cosine basis is a plain (non-tracked) tensor
        # so the layer remains parameter-free and lazy-build/save-load works
        # without get_build_config plumbing.
        if self.mode == "deterministic":
            self.kernel = self._cosine_basis(input_dim)
        else:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.num_decisions),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                dtype=self.compute_dtype,
            )
        if self.mode == "trainable" and self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_decisions,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.compute_dtype,
            )

        super().build(input_shape)

    def _cosine_basis(self, input_dim: int):
        """L2-normalized cosine basis, shape (input_dim, num_decisions)."""
        cols = []
        for decision_idx in range(self.num_decisions):
            col = [
                math.cos(
                    2.0 * math.pi * (decision_idx + 1) * feature_idx / input_dim
                )
                for feature_idx in range(input_dim)
            ]
            col_tensor = ops.convert_to_tensor(col, dtype=self.compute_dtype)
            col_norm = ops.sqrt(ops.sum(ops.square(col_tensor)))
            cols.append(col_tensor / (col_norm + self.epsilon))
        return ops.stack(cols, axis=1)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply hierarchical routing to produce class probabilities."""
        # --- Step 0: Move target axis to last, flatten to 2D ---
        input_rank = len(inputs.shape)
        perm = list(range(input_rank))
        perm[self._normalized_axis] = input_rank - 1
        perm[input_rank - 1] = self._normalized_axis

        if self._normalized_axis != input_rank - 1:
            inputs_transposed = ops.transpose(inputs, perm)
        else:
            inputs_transposed = inputs

        input_dim = inputs_transposed.shape[-1]
        inputs_2d = ops.reshape(inputs_transposed, (-1, input_dim))

        # --- Step 1: Decision logits ---
        decision_logits = ops.matmul(inputs_2d, self.kernel)
        if self.bias is not None:
            decision_logits = decision_logits + self.bias

        decision_probs = ops.sigmoid(decision_logits)
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # --- Step 2: Initialize root probability mass = 1.0 ---
        ones_template = inputs_2d[:, 0:1]
        padded_probs = ops.ones_like(ones_template)

        # --- Step 3: Iteratively split tree ---
        for i in range(self.num_decisions):
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right

            probs_for_left = padded_probs * p_go_left
            probs_for_right = padded_probs * p_go_right

            combined = ops.stack(
                [probs_for_left, probs_for_right], axis=2
            )
            padded_probs = ops.reshape(combined, (-1, 2 ** (i + 1)))

        # --- Step 4: Slice and renormalize ---
        # DECISION D-002 (revised): Preserve mode-specific renormalization to
        # avoid regressing the deterministic-mode test, which asserts
        # exact sum=1.0 for arbitrary epsilon. Trainable mode keeps the
        # original `+ epsilon` safety margin (its prior behavior).
        if self.output_dim == self.padded_output_dim:
            final_probs = padded_probs
        else:
            unnormalized_probs = padded_probs[:, :self.output_dim]
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)
            if self.mode == "trainable":
                final_probs = unnormalized_probs / (prob_sum + self.epsilon)
            else:
                final_probs = unnormalized_probs / prob_sum

        # --- Step 5: Reshape back to original rank ---
        input_transposed_shape = ops.shape(inputs_transposed)
        input_transposed_shape_tensor = ops.convert_to_tensor(
            input_transposed_shape, dtype="int32"
        )
        batch_shape_tensor = input_transposed_shape_tensor[:-1]
        target_dim_tensor = ops.convert_to_tensor(
            [self.output_dim], dtype="int32"
        )
        target_shape_tensor = ops.concatenate(
            [batch_shape_tensor, target_dim_tensor], axis=0
        )
        outputs_transposed = ops.reshape(final_probs, target_shape_tensor)

        # --- Step 6: Restore original axis order ---
        if self._normalized_axis != input_rank - 1:
            outputs = ops.transpose(outputs_transposed, perm)
        else:
            outputs = outputs_transposed

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape: input shape with `axis` replaced by `output_dim`."""
        output_shape = list(input_shape)

        if self._normalized_axis is not None:
            axis_to_modify = self._normalized_axis
        else:
            input_rank = len(input_shape)
            axis_to_modify = (input_rank + self.axis if self.axis < 0
                              else self.axis)

        if self.output_dim is not None:
            output_shape[axis_to_modify] = self.output_dim
        # If output_dim is None (deterministic + pre-build), shape inference
        # leaves the axis unchanged — matches old RoutingProbabilitiesLayer.

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Serialize all parameters (both modes) for round-trip stability."""
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "mode": self.mode,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
        })
        return config


# ---------------------------------------------------------------------
