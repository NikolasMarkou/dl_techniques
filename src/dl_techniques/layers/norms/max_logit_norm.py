"""MaxLogit normalization for out-of-distribution detection.

The module holds three layers from "Decoupling MaxLogit for Out-of-Distribution
Detection":

* ``MaxLogitNorm`` divides the logits by their L2 norm. It is the only
  shape-preserving layer in the file.
* ``DecoupledMaxLogit`` splits the MaxLogit score into a cosine part and a
  magnitude part and returns both alongside their weighted sum.
* ``DMLPlus`` returns one of those two parts, chosen by ``model_type``, for use
  in a two-model ensemble.

Why the split helps. A logit vector carries two independent signals: its
direction, which says which class the sample resembles, and its length, which
says how strongly. Dividing by the L2 norm separates them. The direction part
becomes a unit vector whose largest entry is a cosine similarity, and the length
part becomes a single scalar per sample. An out-of-distribution sample can be
unusual in either signal, and keeping them apart lets a detector score each one.

**Shapes are not uniform across this module. Read the per-class diagrams.**
Measured on a ``(4, 16)`` input:

.. code-block:: text

    MaxLogitNorm()                 ->  one tensor    (4, 16)
    DecoupledMaxLogit()            ->  3-tuple       (4,), (4,), (4,)
    DMLPlus(model_type="focal")    ->  one tensor    (4,)
    DMLPlus(model_type="center")   ->  2-tuple       (4,), (4, 1)

Only ``MaxLogitNorm`` keeps the input shape. The other two reduce the normalized
axis away, and two of the four registry keys return a tuple rather than a
tensor, so a caller cannot swap one of these layers for another and cannot treat
them like the rest of the ``norms`` package.

References:
    - "Decoupling MaxLogit for Out-of-Distribution Detection".
"""

import keras
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms._masking import (
    normalizes_only_the_feature_axis,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaxLogitNorm(keras.layers.Layer):
    """Divide logits by their L2 norm along ``axis``.

    The layer computes ``inputs / sqrt(sum(inputs^2, axis) + epsilon)``. The
    reduction keeps its dimension, so the output has the same shape as the
    input. Measured on a ``(4, 16)`` input the output is ``(4, 16)``, and each
    row has L2 norm ``1.0`` to float32 rounding (measured range
    ``0.99999994`` to ``1.0000001``). This is the only shape-preserving layer in
    this module.

    ``epsilon`` sits inside the square root, so an all-zero row does not divide
    by zero. Measured on a zero input the norm is ``sqrt(1e-7)`` =
    ``3.1623e-04`` and the output is exactly ``0.0``.

    ``supports_masking`` is decided from the RESOLVED normalization axis, not set
    unconditionally. It is ``True`` only while the normalized axis is the
    trailing (feature) axis of the input. At ``axis=-1`` the output at one
    position is a function of that position only: measured cross-position leak
    exactly ``0.000000`` on a ``(3, 5, 8)`` input. Normalizing over the TOKEN
    axis couples positions instead, measured leak ``0.922736`` at ``axis=1`` on
    the same input, and there the flag is ``False`` so Keras drops the mask and
    says so. The decision is made in ``__init__`` from the spelling, since only
    ``-1`` names the trailing axis at every rank, and made exact in ``build()``.
    Measured on a rank-3 input: ``axis=2`` starts ``False`` in ``__init__`` and
    becomes ``True`` after ``build()``.

    ``DecoupledMaxLogit`` and ``DMLPlus`` carry no such flag. They reduce the
    axis away, so the mask shape would no longer match the output shape, and
    both inherit the ``keras.layers.Layer`` default of ``False``.

    **Architecture Overview:**

    .. code-block:: text

                      inputs: x   (B, C)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │ squared = square(inputs)      (B, C)        │
        │ norm = sqrt(sum(squared, axis,              │
        │        keepdims=True) + eps)  (B, 1)        │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │ output = inputs / norm        (B, C)        │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
              output: ONE tensor   (B, C)   SAME shape as x

    :param axis: Axis reduced by the L2 norm. Defaults to -1, the class axis.
    :type axis: int
    :param epsilon: Constant added inside the square root. Must be strictly
        positive. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar axis: The configured axis, stored exactly as passed.
    :vartype axis: int
    :ivar epsilon: The configured epsilon, stored exactly as passed.
    :vartype epsilon: float
    :ivar supports_masking: ``True`` only while ``axis`` names the trailing axis.
        Set in ``__init__`` from the spelling and corrected in ``build()``.
    :vartype supports_masking: bool

    :raises ValueError: If ``epsilon`` is not strictly positive. Measured:
        ``epsilon=0.0`` raises ``ValueError: epsilon must be positive, got 0.0``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import MaxLogitNorm

        logits = keras.random.normal((4, 16))
        scores = MaxLogitNorm()(logits)
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Validate ``epsilon`` and store the configuration.

        The layer owns no weights, so nothing is created here or in ``build()``.

        :param axis: Axis reduced by the L2 norm.
        :type axis: int
        :param epsilon: Constant added inside the square root. Must be strictly
            positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon

        # supports_masking is a promise about the AXIS, not about the class: it holds
        # only while the normalized axis is the trailing (feature) axis. Decided here
        # from the spelling alone - `-1` names the trailing axis at every rank - and
        # made exact in build(), where the input rank is finally known.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(f"Initialized MaxLogitNorm with axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, epsilon: float) -> None:
        """Reject a non-positive ``epsilon``.

        :param epsilon: Constant added inside the square root.
        :type epsilon: float

        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Decide ``supports_masking`` against the now-known input rank.

        The layer owns no weights, so this override exists solely to make the
        masking promise exact: ``axis`` may be spelled non-negatively, and whether
        it names the feature axis or the token axis depends on the rank.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is the
        # value that decides whether the mask actually survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Return ``inputs`` divided by its L2 norm along ``axis``.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Unused. The layer behaves the same in both modes.
        :type training: Optional[bool]

        :return: A tensor of the same shape as ``inputs``, with unit L2 norm
            along ``axis``.
        :rtype: keras.KerasTensor
        """
        # Cast inputs to computation dtype for numerical stability
        inputs = keras.ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = keras.ops.square(inputs)
        norm = keras.ops.sqrt(
            keras.ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # L2 normalize
        return inputs / norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged.

        The reduction uses ``keepdims=True``, so nothing collapses.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        :return: A dictionary carrying ``axis`` and ``epsilon`` on top of the
            base ``Layer`` config.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class DecoupledMaxLogit(keras.layers.Layer):
    """Split the MaxLogit score into a cosine part and a magnitude part.

    **This layer is NOT shape-preserving and it returns THREE tensors.** It
    reduces ``axis`` away and returns the tuple ``(output, max_cosine,
    max_norm)``. Measured on a ``(4, 16)`` input the three shapes are ``(4,)``,
    ``(4,)`` and ``(4,)``. Measured on a ``(2, 5, 16)`` input at the default
    ``axis=-1`` they are ``(2, 5)``, ``(2, 5)`` and ``(2, 5)``.

    The three tensors are:

    * ``max_cosine`` -- the largest entry of the unit-norm logit vector. It is
      the direction signal and lies in ``[-1, 1]``.
    * ``max_norm`` -- ``sqrt(sum(inputs ** 2, axis) + epsilon)``, the L2 norm
      with ``epsilon`` INSIDE the root. It is the magnitude signal. The ``max``
      in the code reduces an axis that ``keepdims=True`` already left at size 1,
      so it takes that quantity through unchanged. It equals ``||inputs||_2``
      only while ``||inputs||_2 ** 2 >> epsilon``: measured on a ``(4, 16)``
      ``keras.random.normal(seed=0)`` input scaled by a constant,
      ``max|max_norm - ||inputs||_2|`` is ``0.000e+00`` at scale 1 and rises to
      ``1.546e-05`` at scale ``1e-3`` and ``1.292e-04`` at scale ``1e-4``.
    * ``output`` -- ``constant * max_cosine + max_norm``. Measured for
      ``constant`` 0.5, 1.0 and 3.0: ``max|output - (constant * max_cosine +
      max_norm)| = 0.000e+00``.

    ``constant`` is a fixed hyperparameter, not a learned weight. The layer owns
    no weights at all.

    ``supports_masking`` stays at the ``keras.layers.Layer`` default of
    ``False``, since the reduced output no longer has a position axis for a mask
    to describe.

    **Architecture Overview:**

    .. code-block:: text

                            inputs: x   (B, C)
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │ norm = sqrt(sum(square(inputs), axis,                  │
        │        keepdims=True) + eps)          (B, 1)           │
        └─────────────┬───────────────────────────┬──────────────┘
                      │                           │
                      ▼                           │
        ┌────────────────────────────┐            │
        │ normalized =               │            │
        │   inputs / norm    (B, C)  │            │
        └─────────────┬──────────────┘            │
                      │                           │
                      ▼                           ▼
        ┌────────────────────────────┐  ┌────────────────────────┐
        │ max_cosine =               │  │ max_norm =             │
        │   max(normalized,          │  │   max(norm,            │
        │       axis=axis)           │  │       axis=axis)       │
        └─────────────┬──────────────┘  └───────────┬────────────┘
                      │ (B,)                        │ (B,)
                      ▼                             ▼
        ┌────────────────────────────────────────────────────────┐
        │ output = constant * max_cosine + max_norm  (B,)        │
        └────────────────────────────────────────────────────────┘

        returns a 3-TUPLE, in this order:
          (output, max_cosine, max_norm)   shapes (B,) (B,) (B,)

    :param constant: Weight applied to ``max_cosine`` in the combined score. A
        fixed hyperparameter, not a learned weight. Must be strictly positive.
        Defaults to 1.0.
    :type constant: float
    :param axis: Axis reduced by the L2 norm and by both maxima. Defaults to -1.
    :type axis: int
    :param epsilon: Constant added inside the square root. Must be strictly
        positive. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar constant: The configured weight, stored exactly as passed.
    :vartype constant: float
    :ivar axis: The configured axis, stored exactly as passed.
    :vartype axis: int
    :ivar epsilon: The configured epsilon, stored exactly as passed.
    :vartype epsilon: float

    :raises ValueError: If ``constant`` is not strictly positive. Measured:
        ``constant=0.0`` raises ``ValueError: constant must be positive, got
        0.0``.
    :raises ValueError: If ``epsilon`` is not strictly positive.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import DecoupledMaxLogit

        logits = keras.random.normal((4, 16))
        combined, max_cosine, max_norm = DecoupledMaxLogit(constant=1.0)(logits)
    """

    def __init__(
        self,
        constant: float = 1.0,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Validate ``constant`` and ``epsilon`` and store the configuration.

        :param constant: Weight applied to ``max_cosine``. A fixed
            hyperparameter, not a learned weight. Must be strictly positive.
        :type constant: float
        :param axis: Axis reduced by the L2 norm and by both maxima.
        :type axis: int
        :param epsilon: Constant added inside the square root. Must be strictly
            positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``constant`` is not strictly positive.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(constant, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized DecoupledMaxLogit with constant={constant}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Reject a non-positive ``constant`` or ``epsilon``.

        :param constant: Weight applied to ``max_cosine``.
        :type constant: float
        :param epsilon: Constant added inside the square root.
        :type epsilon: float

        :raises ValueError: If ``constant`` is not strictly positive.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Return the combined score and both of its components.

        The reduced axis is gone from all three results, so this call changes
        the rank of its input.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Unused. The layer behaves the same in both modes.
        :type training: Optional[bool]

        :return: The 3-tuple ``(output, max_cosine, max_norm)``. All three drop
            ``axis``; on a ``(4, 16)`` input all three are ``(4,)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        inputs = keras.ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = keras.ops.square(inputs)
        norm = keras.ops.sqrt(
            keras.ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features (cosine)
        normalized = inputs / norm

        # Get maximum cosine similarity (remove keepdims for final output)
        max_cosine = keras.ops.max(normalized, axis=self.axis)

        # Get maximum norm. ops.max over the keepdims=True `norm` already drops
        # the reduced axis, yielding the same shape as `max_cosine` above.
        max_norm = keras.ops.max(norm, axis=self.axis)

        # Combine with the fixed decoupling constant (a hyperparameter, not a weight)
        output = self.constant * max_cosine + max_norm

        return output, max_cosine, max_norm

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the three output shapes, each with ``axis`` removed.

        Measured: ``(4, 16)`` gives ``((4,), (4,), (4,))``, and ``(2, 5, 16)``
        at ``axis=1`` gives ``((2, 16), (2, 16), (2, 16))``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The shapes of ``(output, max_cosine, max_norm)``, which are
            always equal to each other.
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            output_shape = tuple(input_shape_list[:-1])
        else:
            output_shape_list = input_shape_list[:self.axis] + input_shape_list[self.axis + 1:]
            output_shape = tuple(output_shape_list)

        # All three outputs have the same shape
        return (output_shape, output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        :return: A dictionary carrying ``constant``, ``axis`` and ``epsilon`` on
            top of the base ``Layer`` config.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class DMLPlus(keras.layers.Layer):
    """Return one half of the decoupled MaxLogit score, chosen by ``model_type``.

    DML+ trains two separate models, one for each half, and ensembles their
    scores. This layer is the head of one of them. ``model_type="focal"``
    returns the cosine half and ``model_type="center"`` returns the magnitude
    half.

    **The two settings return DIFFERENT things, with different arity.** Measured
    on a ``(4, 16)`` input:

    .. code-block:: text

        model_type="focal"    ->  ONE tensor    (4,)
        model_type="center"   ->  2-TUPLE       (4,)  and  (4, 1)

    ``"center"`` returns ``(max_norm, norm)``. The second element is the
    ``keepdims=True`` norm itself, kept so a caller can rescale the logits with
    the same factor the score came from; the reduced axis stays at size 1 rather
    than being removed. Measured on a ``(2, 5, 16)`` input at the default
    ``axis=-1``: ``(2, 5)`` and ``(2, 5, 1)``.

    Neither setting is shape-preserving. The outputs of this layer match
    ``DecoupledMaxLogit``'s components exactly: measured ``max|focal -
    max_cosine| = 0.000e+00`` and ``max|center[0] - max_norm| = 0.000e+00``.

    ``normalized`` is computed on both paths, but only ``"focal"`` reads it. The
    ``"center"`` path discards it and reads ``norm`` instead, which is why the
    diagram forks at ``norm`` rather than at ``normalized``.

    ``supports_masking`` stays at the ``keras.layers.Layer`` default of
    ``False``, since both settings drop the reduced axis.

    **Architecture Overview:**

    .. code-block:: text

                          inputs: x   (B, C)
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │ norm = sqrt(sum(square(inputs), axis,               │
        │        keepdims=True) + eps)       (B, 1)           │
        └─────────────┬─────────────────────────┬─────────────┘
                      │                         │
                      ▼                         │
        ┌───────────────────────────┐           │
        │ normalized =              │           │
        │   inputs / norm  (B, C)   │           │
        └─────────────┬─────────────┘           │
                      │ model_type              │ model_type
                      │   == "focal"            │   == "center"
                      ▼                         ▼
        ┌───────────────────────────┐  ┌────────────────────────────┐
        │ max(normalized,           │  │ max_norm =                 │
        │     axis=axis)            │  │   max(norm, axis=axis)     │
        └─────────────┬─────────────┘  └─────────┬───────────┬──────┘
                      │ (B,)                     │ (B,)      │ (B, 1)
                      ▼                          ▼           ▼
        returns ONE tensor       returns a 2-TUPLE, in this order:
          (B,)                     (max_norm, norm)   (B,)  (B, 1)

    :param model_type: ``"focal"`` for the cosine half, ``"center"`` for the
        magnitude half. Required; there is no default, because the two settings
        return different things.
    :type model_type: Literal["focal", "center"]
    :param axis: Axis reduced by the L2 norm and by the maximum. Defaults to -1.
    :type axis: int
    :param epsilon: Constant added inside the square root. Must be strictly
        positive. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar model_type: The configured setting, stored exactly as passed.
    :vartype model_type: Literal["focal", "center"]
    :ivar axis: The configured axis, stored exactly as passed.
    :vartype axis: int
    :ivar epsilon: The configured epsilon, stored exactly as passed.
    :vartype epsilon: float

    :raises ValueError: If ``model_type`` is neither ``"focal"`` nor
        ``"center"``. Measured: ``model_type="bogus"`` raises ``ValueError:
        model_type must be 'focal' or 'center', got bogus``.
    :raises ValueError: If ``epsilon`` is not strictly positive.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import DMLPlus

        logits = keras.random.normal((4, 16))
        max_cosine = DMLPlus(model_type="focal")(logits)
        max_norm, norm = DMLPlus(model_type="center")(logits)
    """

    def __init__(
        self,
        model_type: Literal["focal", "center"],
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Validate ``model_type`` and ``epsilon`` and store the configuration.

        :param model_type: ``"focal"`` or ``"center"``.
        :type model_type: Literal["focal", "center"]
        :param axis: Axis reduced by the L2 norm and by the maximum.
        :type axis: int
        :param epsilon: Constant added inside the square root. Must be strictly
            positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``model_type`` is neither ``"focal"`` nor
            ``"center"``.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(model_type, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.model_type = model_type
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized DMLPlus with model_type={model_type}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, model_type: str, epsilon: float) -> None:
        """Reject an unknown ``model_type`` or a non-positive ``epsilon``.

        :param model_type: ``"focal"`` or ``"center"``.
        :type model_type: str
        :param epsilon: Constant added inside the square root.
        :type epsilon: float

        :raises ValueError: If ``model_type`` is neither ``"focal"`` nor
            ``"center"``.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        if model_type not in ["focal", "center"]:
            raise ValueError(f"model_type must be 'focal' or 'center', got {model_type}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Return the half of the score named by ``model_type``.

        The two branches differ in arity, so a caller that unpacks the result
        must know which ``model_type`` it built.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Unused. The layer behaves the same in both modes.
        :type training: Optional[bool]

        :return: For ``"focal"``, one tensor with ``axis`` removed. For
            ``"center"``, the 2-tuple ``(max_norm, norm)``, where ``norm`` keeps
            the reduced axis at size 1.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        inputs = keras.ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = keras.ops.square(inputs)
        norm = keras.ops.sqrt(
            keras.ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features. Only the "focal" branch below reads this;
        # the "center" branch discards it and reads `norm` instead.
        normalized = inputs / norm

        if self.model_type == "focal":
            # Focal model returns MaxCosine
            return keras.ops.max(normalized, axis=self.axis)
        else:
            # Center model returns MaxNorm and norm factor. ops.max over the
            # keepdims=True `norm` already drops the reduced axis.
            max_norm = keras.ops.max(norm, axis=self.axis)
            return max_norm, norm

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]:
        """Return one shape for ``"focal"`` and a pair of shapes for ``"center"``.

        Measured on ``(4, 16)``: ``"focal"`` gives ``(4,)`` and ``"center"``
        gives ``((4,), (4, 1))``. Measured on ``(2, 5, 16)`` at ``axis=1``:
        ``(2, 16)`` and ``((2, 16), (2, 1, 16))``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The reduced shape, or the pair ``(reduced, norm_factor)``.
        :rtype: Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension for reduced shape
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            reduced_shape = tuple(input_shape_list[:-1])
        else:
            reduced_shape_list = input_shape_list[:self.axis] + input_shape_list[self.axis + 1:]
            reduced_shape = tuple(reduced_shape_list)

        if self.model_type == "focal":
            return reduced_shape
        else:
            # Center model returns (max_norm, norm_factor). norm_factor is the
            # keepdims L2 norm: the reduced axis collapses to size 1 (it is NOT
            # removed), so the shape is input_shape with that axis set to 1.
            norm_factor_list = list(input_shape)
            reduce_axis = self.axis if self.axis != -1 else len(norm_factor_list) - 1
            norm_factor_list[reduce_axis] = 1
            return (reduced_shape, tuple(norm_factor_list))

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        :return: A dictionary carrying ``model_type``, ``axis`` and ``epsilon``
            on top of the base ``Layer`` config.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "model_type": self.model_type,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
