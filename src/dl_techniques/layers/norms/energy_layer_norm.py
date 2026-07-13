"""
Energy Layer Normalization for Keras 3.x.

Implements the layer normalization of the Energy Transformer (ET), Hoover, Liang, Pham,
Panda, Strobelt, Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023
(https://arxiv.org/abs/2302.07253), equations (1)-(2).

The distinguishing feature versus ``keras.layers.LayerNormalization`` is the
parameterization: the scale ``gamma`` is a **SCALAR** and the offset ``delta`` is a
**VECTOR** of dimension ``D``. Stock ``LayerNormalization`` has a per-feature (vector)
gamma, which does NOT correspond to the ET Lagrangian and therefore does NOT yield the
positive-semi-definite Hessian that makes the ET energy-descent guarantee provable.
"""

import keras
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EnergyLayerNorm(keras.layers.Layer):
    """Energy Transformer layer normalization (scalar gamma, vector delta).

    **Intent**: provide the ``g = dL/dx`` "activation function" of the Energy Transformer,
    whose Lagrangian ``L`` has a PSD Hessian and therefore turns the block's hand-coded
    closed-form update into a *provable* descent direction on the block's scalar energy.

    **Mathematics** (over the LAST axis ``d``, per token, per sample — no token mixing):

    .. code-block:: text

        xbar = mean_j(x_j)                                  # scalar per token
        var  = mean_j((x_j - xbar)^2)                       # scalar per token
        g_i  = gamma * (x_i - xbar) / sqrt(var + eps) + delta_i

    This ``g`` is exactly the gradient of the Lagrangian

    .. code-block:: text

        L(x) = D * gamma * sqrt(var + eps) + sum_j delta_j * x_j
        g    = dL/dx

    Its Hessian ``dg/dx`` is PSD for ``gamma > 0``, hence for the ET energy ``E``:

    .. code-block:: text

        dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0

    which is the whole reason this parameterization (scalar gamma, vector delta) exists.

    **Shape contract**: gamma is a SCALAR (``shape=()``); delta is a VECTOR
    (``shape=(D,)``). A per-feature gamma would break the Lagrangian identity above.

    :param epsilon: Small positive constant for numerical stability inside the sqrt.
        Defaults to ``1e-5``.
    :type epsilon: float
    :param gamma_initializer: Initializer for the scalar ``gamma``. Defaults to ``'ones'``
        (the paper requires ``gamma > 0`` for the PSD Hessian).
    :type gamma_initializer: Union[str, initializers.Initializer]
    :param delta_initializer: Initializer for the ``(D,)`` offset ``delta``.
        Defaults to ``'zeros'``.
    :type delta_initializer: Union[str, initializers.Initializer]

    :raises ValueError: If ``epsilon <= 0``.

    Input shape:
        Tensor of shape ``(..., D)``; normalization is over the last axis. Typically
        ``(batch, num_tokens, embed_dim)``.

    Output shape:
        Identical to the input shape.

    Example:
        >>> layer = EnergyLayerNorm(epsilon=1e-5)
        >>> g = layer(keras.random.normal((2, 16, 64)))
        >>> g.shape
        (2, 16, 64)

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (1)-(2).
    """

    # DECISION plan_2026-07-13_57c9833e/D-005
    # Do NOT add a `lagrangian()` / `energy()` method to this class. It is tempting
    # (the Lagrangian L is written out above, and the sibling ET layers DO expose an
    # `energy()`/`update()` pair) — but it would have ZERO call sites: the ET block's
    # reported energy is `E_ATT + E_HN` only; the LayerNorm Lagrangian is NOT a term in
    # E. Adding it, and then summing it into the block's reported energy, would make the
    # energy-descent test assert on the WRONG quantity and silently invalidate the
    # headline guarantee. Omitted per the use-before-reuse / earned-abstraction rule.
    # See decisions.md D-005.

    def __init__(
        self,
        epsilon: float = 1e-5,
        gamma_initializer: Union[str, initializers.Initializer] = 'ones',
        delta_initializer: Union[str, initializers.Initializer] = 'zeros',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ----- validation -----
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {epsilon}")

        # ----- store ALL configuration -----
        self.epsilon = float(epsilon)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.delta_initializer = initializers.get(delta_initializer)

        # ----- weights are created in build() -----
        self.gamma: Optional[keras.Variable] = None
        self.delta: Optional[keras.Variable] = None

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyLayerNorm with "
            f"epsilon={self.epsilon}, "
            f"gamma_initializer={gamma_initializer}, "
            f"delta_initializer={delta_initializer}"
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``gamma`` and the ``(D,)`` vector ``delta``.

        :param input_shape: Shape of the input tensor; the last axis is the feature
            axis ``D``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is undefined.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError(
                "The last axis of the input shape must be defined, got "
                f"input_shape={input_shape}"
            )

        # gamma is a SCALAR (paper eq. 1). This is NOT a bug and must NOT be
        # "fixed" to a per-feature vector: a vector gamma breaks g = dL/dx.
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=self.gamma_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        # delta is a VECTOR of dim D (the Lagrangian's linear term).
        self.delta = self.add_weight(
            name="delta",
            shape=(int(feature_dim),),
            initializer=self.delta_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the ET layer norm: ``gamma * (x - xbar) / sqrt(var + eps) + delta``.

        :param inputs: Input tensor of shape ``(..., D)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; present for interface consistency.
        :type training: Optional[bool]

        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # Statistics over the LAST axis only — per token, per sample. No token mixing.
        x_bar = ops.mean(inputs, axis=-1, keepdims=True)
        centered = inputs - x_bar
        variance = ops.mean(ops.square(centered), axis=-1, keepdims=True)

        inv_std = ops.rsqrt(variance + self.epsilon)

        # gamma is a scalar => broadcasts over everything; delta is (D,) => broadcasts
        # over the leading (batch, token) axes.
        return self.gamma * centered * inv_std + self.delta

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identity — the layer is shape-preserving).

        Uses only the passed shape and stored config, never a weight shape, so it is
        valid on an UNBUILT layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'delta_initializer': initializers.serialize(self.delta_initializer),
        })
        return config

    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyLayerNorm":
        """Reconstruct the layer from its serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]

        :return: A new ``EnergyLayerNorm`` instance.
        :rtype: EnergyLayerNorm
        """
        config = dict(config)
        for key in ('gamma_initializer', 'delta_initializer'):
            if key in config:
                config[key] = initializers.deserialize(config[key])
        return cls(**config)

# ---------------------------------------------------------------------
