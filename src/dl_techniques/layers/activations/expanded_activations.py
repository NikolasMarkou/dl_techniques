"""
Gated activations, plus expanded-range variants with a trainable gate width.

Every activation here has the shape ``f(x) = x * gate(x)``. The gate is a
squashing function -- ``erf``, ``sigmoid`` or ``arctan`` -- mapped into
``(0, 1)``. GELU and SiLU are the plain versions. The three ``x``-prefixed
variants add one trainable scalar ``alpha`` and widen the gate to
``g * (1 + 2*alpha) - alpha``, which stretches its range from ``(0, 1)`` to
``(-alpha, 1 + alpha)``. Negative ``alpha`` narrows it instead. ``EluPlusOne``
is the odd one out: a strictly positive activation for rate parameters, with
no gate at all.

``alpha`` initializes to zeros, so an untrained ``xGELU`` is bit-for-bit
``GELU`` and an untrained ``xSiLU`` is bit-for-bit ``SiLU`` (measured
elementwise maximum difference 0.0 on both). The variants cost nothing until
the optimizer moves ``alpha``.

The three gates behave differently in the tail, which is what to pick on.
Measured at ``x = 10``, the distance from full saturation ``1 - gate(x)`` is
0.0 for ``erf`` (saturated exactly, in float64), 4.540e-05 for ``sigmoid``
and 3.173e-02 for ``arctan``. The arctan gate decays polynomially rather than
exponentially, so it never really saturates.

All layers subclass ``BaseActivation``, are element-wise, and preserve input
shape. ``get_activation`` builds one by name.

Reference:
    Huang, A. H. (2023). Expanded Gating Ranges Improve Activation Functions.
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class BaseActivation(keras.layers.Layer):
    """Common base for the activation layers in this module.

    Forwards ``trainable``, ``name`` and ``dtype`` to ``keras.layers.Layer``
    and fixes ``compute_output_shape`` to the identity, which is correct for
    every element-wise activation below. It owns no weights and defines no
    ``call``; a subclass supplies that.

    **Architecture Overview:**

    .. code-block:: text

                x  [..., F]
                      ▼
        ┌───────────────────────────┐
        │ f(x), defined by subclass │
        └─────────────┬─────────────┘
                      ▼
                y  [..., F]

    ``F`` stands for the trailing feature dimension, but nothing here reads
    the shape: the transform is element-wise over the whole tensor.

    :param trainable: Whether the layer's variables are trainable. Defaults
        to True. Only the ``ExpandedActivation`` subclasses own a variable.
    :type trainable: bool
    :param name: Name of the layer. ``None`` lets Keras generate one.
    :type name: Optional[str]
    :param dtype: Dtype or dtype policy for the layer.
    :type dtype: Optional[Union[str, keras.ops.dtype]]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[Union[str, keras.ops.dtype]] = None,
        **kwargs: Any
    ) -> None:
        """Forward the standard Layer arguments to the base class.

        :param trainable: Whether the layer's variables are trainable.
        :type trainable: bool
        :param name: Name of the layer.
        :type name: Optional[str]
        :param dtype: Dtype or dtype policy for the layer.
        :type dtype: Optional[Union[str, keras.ops.dtype]]
        :param kwargs: Additional keyword arguments passed to the parent
            class.
        """
        super().__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            **kwargs
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the input shape unchanged.

        Every activation in this module is element-wise, so the shape never
        moves.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        Adds nothing to the base class config; this layer stores no
        configuration of its own.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class GELU(BaseActivation):
    """Exact GELU: ``x * 0.5 * (1 + erf(x / sqrt(2)))``.

    The gate is the standard normal CDF, so the output is ``x`` weighted by
    the probability that a standard normal draw falls below ``x``. It is
    smooth everywhere and non-monotonic: it dips slightly below zero for
    small negative inputs before returning to zero.

    This is the exact form, not the tanh approximation. Measured on
    ``[-3, -1, 0, 1, 3]``, it matches
    ``keras.activations.gelu(x, approximate=False)`` elementwise, output
    ``[-0.00404978, -0.15865526, 0.0, 0.8413447, 2.9959502]``.

    **Architecture Overview:**

    .. code-block:: text

                x  [..., F]
                      ▼
        ┌───────────────────────────┐
        │ GELU(x) = 0.5 * x         │
        │   * (1 + erf(x/sqrt(2)))  │
        └─────────────┬─────────────┘
                      ▼
                y  [..., F]

    The erf gate saturates hard: measured at ``x = 10`` it reaches exactly
    1.0 in float64, so the gradient through the gate is gone there.

    References:
        Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply exact GELU element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-23T203721-009b7ccf/D-017
        # `keras.ops.sqrt(2.0)` returns a float32 TENSOR under every dtype policy, so
        # it met the float16 autocast input here and the divide raised TypeError on
        # ANY mixed-precision forward. Cast the constant, not `inputs`: casting
        # `inputs` to float32 opts every consumer out of mixed precision. See D-017.
        root_two = keras.ops.sqrt(keras.ops.cast(2.0, inputs.dtype))
        return 0.5 * inputs * (1 + keras.ops.erf(inputs / root_two))


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class SiLU(BaseActivation):
    """SiLU, also called Swish: ``x * sigmoid(x)``.

    Smooth everywhere, unlike ReLU, which has no derivative at zero. Like
    GELU it dips below zero for small negative inputs. Measured on
    ``[-3, -1, 0, 1, 3]`` it matches ``keras.activations.silu`` elementwise.

    **Architecture Overview:**

    .. code-block:: text

                x  [..., F]
                      ▼
        ┌───────────────────────────┐
        │ SiLU(x) = x * sigmoid(x)  │
        └─────────────┬─────────────┘
                      ▼
                y  [..., F]

    The sigmoid gate saturates more slowly than GELU's erf gate: measured at
    ``x = 10``, ``1 - gate(x)`` is 4.540e-05 here against 0.0 for erf.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply SiLU element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return inputs * keras.ops.sigmoid(inputs)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class ExpandedActivation(BaseActivation):
    """Base for the gate-widening variants; owns the trainable ``alpha``.

    Adds one scalar weight, ``alpha``, and defines the shared formula
    ``f(x) = x * (gate(x) * (1 + 2*alpha) - alpha)``. A subclass supplies
    ``gate`` and the ``call`` that uses it; this class supplies only the
    weight, the config and the build.

    ``alpha`` widens the gate. Where ``gate(x)`` lives in ``(0, 1)``, the
    expanded gate lives in ``(-alpha, 1 + alpha)``. Measured on
    ``xSiLU(alpha=0.5)`` against ``[-3, -1, 0, 1, 3]``, the expanded gate runs
    ``[-0.405, 0.038, 0.5, 0.962, 1.405]``, so the activation can flip the
    sign of its own input where the plain version could only shrink it. A
    negative ``alpha`` narrows instead: at ``alpha = -0.25`` the gate is
    confined to ``(0.25, 0.75)``.

    **Architecture Overview:**

    .. code-block:: text

                       x  [..., F]
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐        ┌─────────────────────┐
        │ identity  │        │ g = subclass gate(x)│
        │     x     │        │ e = g*(1+2a) - a    │
        └─────┬─────┘        └──────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           ▼  x * e
                       y  [..., F]

    ``a`` is the ``alpha`` weight. The left branch is the input tensor
    itself, not a sub-layer. ``alpha`` is a scalar of shape ``()``, one per
    layer, shared across every element -- it is not per-feature.

    :param alpha_initializer: Initializer for ``alpha``. Defaults to
        ``'zeros'``, which makes each variant start out identical to its
        plain counterpart.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for ``alpha``. Defaults to
        ``None``.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for ``alpha``. Defaults to ``None``,
        so nothing stops training from driving ``alpha`` negative and
        narrowing the gate.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.

    :ivar alpha: The scalar gate-width weight. ``None`` until ``build`` runs.
    :vartype alpha: Optional[keras.Variable]

    Note:
        ``alpha`` is created with ``dtype=self.dtype``, which is the layer's
        variable dtype. Measured under ``dtype="mixed_float16"``:
        ``variable_dtype`` is float32 and ``alpha.dtype`` is float32 while
        ``compute_dtype`` is float16, so the weight itself stays in full
        precision. Passing ``dtype="float16"`` outright does put ``alpha`` in
        float16.
    """

    def __init__(
        self,
        alpha_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        alpha_regularizer: Optional[keras.regularizers.Regularizer] = None,
        alpha_constraint: Optional[keras.constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        """Resolve the ``alpha`` initializer, regularizer and constraint.

        The weight itself is created in ``build``.

        :param alpha_initializer: Initializer for ``alpha``. Defaults to
            ``'zeros'``.
        :type alpha_initializer: Union[str, keras.initializers.Initializer]
        :param alpha_regularizer: Regularizer for ``alpha``.
        :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
        :param alpha_constraint: Constraint for ``alpha``.
        :type alpha_constraint: Optional[keras.constraints.Constraint]
        :param kwargs: Additional keyword arguments passed to the parent
            class.
        """
        super().__init__(**kwargs)

        # Store configuration parameters
        self.alpha_initializer = keras.initializers.get(alpha_initializer)
        self.alpha_regularizer = keras.regularizers.get(alpha_regularizer)
        self.alpha_constraint = keras.constraints.get(alpha_constraint)

        # Initialize weight attribute - created in build()
        self.alpha = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``alpha`` weight.

        The shape is ``()``, so ``alpha`` does not depend on
        ``input_shape``; the argument is only forwarded to the base class.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
            trainable=True,
            dtype=self.dtype
        )
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        Serializes how ``alpha`` is set up, not its value; the value travels
        with the weights. Measured on a saved and reloaded ``xGELU``: an
        ``alpha`` of 0.37 comes back as 0.37.

        :return: Dictionary containing the layer configuration including
            alpha parameter configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': keras.regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': keras.constraints.serialize(self.alpha_constraint),
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class xATLU(ExpandedActivation):
    """Expanded ArcTan Linear Unit: an arctan gate with trainable width.

    Computes ``gate(x) = (arctan(x) + pi/2) / pi`` and then
    ``x * (gate(x) * (1 + 2*alpha) - alpha)``.

    The arctan gate is the one that does not saturate. Measured ``1 -
    gate(x)`` at ``x = 10``: 3.173e-02 here, against 4.540e-05 for the
    sigmoid gate and 0.0 for the erf gate. Its tail decays like ``1/x``, not
    exponentially, so far-out inputs still carry gradient through the gate.
    It is not the gentler gate near the origin -- its slope at zero is
    0.318310, steeper than sigmoid's 0.250000 and shallower than erf's
    0.398942.

    Note that this module ships no plain ``ATLU``; the arctan gate appears
    only in this expanded form.

    **Architecture Overview:**

    .. code-block:: text

                       x  [..., F]
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐        ┌─────────────────────┐
        │ identity  │        │ g = (atan(x) + pi/2)│
        │     x     │        │     / pi            │
        │           │        │ e = g*(1+2a) - a    │
        └─────┬─────┘        └──────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           ▼  x * e
                       y  [..., F]

    ``a`` is the ``alpha`` weight, and ``pi`` is ``numpy.pi`` -- a Python
    float, so it promotes to the input's dtype and needs no cast.

    :param alpha_initializer: Initializer for ``alpha``. Defaults to
        ``'zeros'``.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for ``alpha``. Defaults to
        ``None``.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for ``alpha``. Defaults to ``None``.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply xATLU element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        gate = (keras.ops.arctan(inputs) + np.pi / 2) / np.pi
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class xGELU(ExpandedActivation):
    """Expanded GELU: the erf gate with trainable width.

    Computes ``gate(x) = 0.5 * (1 + erf(x / sqrt(2)))`` and then
    ``x * (gate(x) * (1 + 2*alpha) - alpha)``.

    With the default ``alpha_initializer='zeros'`` this is exactly ``GELU``
    at initialization -- measured elementwise maximum difference 0.0 against
    ``GELU`` on ``[-3, -1, 0, 1, 3]``. Training then moves ``alpha`` and the
    two diverge.

    **Architecture Overview:**

    .. code-block:: text

                       x  [..., F]
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐        ┌─────────────────────┐
        │ identity  │        │ g = 0.5*(1 +        │
        │     x     │        │     erf(x/sqrt(2))) │
        │           │        │ e = g*(1+2a) - a    │
        └─────┬─────┘        └──────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           ▼  x * e
                       y  [..., F]

    ``a`` is the ``alpha`` weight. The ``sqrt(2)`` constant is cast to the
    input's dtype; see the anchor in ``call``.

    :param alpha_initializer: Initializer for ``alpha``. Defaults to
        ``'zeros'``.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for ``alpha``. Defaults to
        ``None``.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for ``alpha``. Defaults to ``None``.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply xGELU element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-23T203721-009b7ccf/D-017
        # Same policy-blind float32 constant as `GELU.call` above, same repair. Do
        # NOT revert to the bare `keras.ops.sqrt(2.0)`. See decisions.md D-017.
        root_two = keras.ops.sqrt(keras.ops.cast(2.0, inputs.dtype))
        gate = 0.5 * (1 + keras.ops.erf(inputs / root_two))
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class xSiLU(ExpandedActivation):
    """Expanded SiLU: the sigmoid gate with trainable width.

    Computes ``gate(x) = sigmoid(x)`` and then
    ``x * (gate(x) * (1 + 2*alpha) - alpha)``.

    With the default ``alpha_initializer='zeros'`` this is exactly ``SiLU``
    at initialization -- measured elementwise maximum difference 0.0 against
    ``SiLU`` on ``[-3, -1, 0, 1, 3]``.

    **Architecture Overview:**

    .. code-block:: text

                       x  [..., F]
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐        ┌─────────────────────┐
        │ identity  │        │ g = sigmoid(x)      │
        │     x     │        │ e = g*(1+2a) - a    │
        └─────┬─────┘        └──────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           ▼  x * e
                       y  [..., F]

    ``a`` is the ``alpha`` weight. No constant is created here, so this
    variant has no dtype-policy hazard of the kind ``GELU`` and ``xGELU``
    carry.

    :param alpha_initializer: Initializer for ``alpha``. Defaults to
        ``'zeros'``.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for ``alpha``. Defaults to
        ``None``.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for ``alpha``. Defaults to ``None``.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply xSiLU element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        gate = keras.ops.sigmoid(inputs)
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


def elu_plus_one_plus_epsilon(x: keras.KerasTensor) -> keras.KerasTensor:
    """Return ``ELU(x) + 1 + epsilon``, which is strictly positive.

    ``ELU`` is bounded below by -1, so adding 1 gives a non-negative result
    and adding ``keras.backend.epsilon()`` lifts it clear of zero. Use it for
    a rate or scale parameter that has to stay positive, such as the lambda
    of an exponential distribution.

    Measured over 20001 points spanning ``x`` in ``[-100, 5]``: the minimum
    output is 1e-07, exactly ``keras.backend.epsilon()``, and no output is
    zero or negative. That floor is the whole guarantee -- it does not stay
    ahead of zero by any margin larger than ``epsilon``.

    :param x: Input tensor.
    :type x: keras.KerasTensor
    :return: Tensor of the same shape as ``x``, every element greater than 0.
    :rtype: keras.KerasTensor
    """
    return keras.ops.elu(x) + 1.0 + keras.backend.epsilon()


@register_dl_technique("dl_techniques.layers.activations.expanded_activations")
class EluPlusOne(BaseActivation):
    """Layer wrapper around :func:`elu_plus_one_plus_epsilon`.

    Computes ``ELU(x) + 1 + epsilon``, so every output is strictly positive.
    Intended for heads that emit a rate or scale parameter. Owns no weights
    and takes no arguments of its own beyond the ``BaseActivation`` ones.

    Measured minimum output over ``x`` in ``[-100, 5]``: 1.0000000116860974e-07.

    **Architecture Overview:**

    .. code-block:: text

                x  [..., F]
                      ▼
        ┌───────────────────────────┐
        │ ELU(x) + 1 + epsilon      │
        └─────────────┬─────────────┘
                      ▼
                y  [..., F],  y > 0

    ``epsilon`` is ``keras.backend.epsilon()``, read at call time, so
    changing it globally changes this layer's floor.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply ``ELU(x) + 1 + epsilon`` element-wise.

        :param inputs: The input tensor, any shape.
        :type inputs: keras.KerasTensor
        :return: Strictly positive tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return elu_plus_one_plus_epsilon(inputs)


# ---------------------------------------------------------------------


def get_activation(activation_name: str) -> BaseActivation:
    """Build one of this module's activation layers by name.

    The name is lowercased and stripped first, so ``'  xAtLu  '`` resolves to
    ``xATLU``. The layer is constructed with its defaults; there is no way to
    pass an initializer or a constraint through this function. This is a
    module-local convenience, separate from the package-level
    ``create_activation_layer`` factory.

    Supported activation names: ``'gelu'``, ``'silu'``, ``'xatlu'``,
    ``'xgelu'``, ``'xsilu'``, ``'elu_plus_one'``.

    :param activation_name: Name of the desired activation function
        (case-insensitive, surrounding whitespace ignored).
    :type activation_name: str
    :return: A new, default-constructed instance of the named class.
    :rtype: BaseActivation
    :raises ValueError: If ``activation_name`` is not one of the six names
        above. The message lists them.
    """
    activations = {
        'gelu': GELU,
        'silu': SiLU,
        'xatlu': xATLU,
        'xgelu': xGELU,
        'xsilu': xSiLU,
        'elu_plus_one': EluPlusOne,
    }

    activation_class = activations.get(activation_name.lower().strip())
    if activation_class is None:
        raise ValueError(
            f"Unknown activation: '{activation_name}'. "
            f"Available activations: {list(activations.keys())}"
        )

    return activation_class()

# ---------------------------------------------------------------------
