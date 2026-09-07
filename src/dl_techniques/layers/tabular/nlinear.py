"""``NLinear`` -- ``n`` fully independent linear layers evaluated as one batched einsum.

A batched ensemble normally shares one kernel across its ``k`` members and gives
each member only a rank-1 perturbation (that is ``LinearEfficientEnsemble``).
This layer is the opposite trade: it stores ``n`` genuinely separate kernels of
shape ``(n, input_dim, output_dim)`` and applies them in a single
``einsum('bni,nio->bno', ...)``, so the members share no parameters at all and
still cost one kernel launch rather than ``n``. Use it where true independence
matters more than parameter savings -- the TabM ``'packed'`` ensemble mode and
per-member output heads are the in-repo consumers.

``input_dim`` may be left ``None`` to defer the fan-in to ``build()``. That is
what lets ``TabMMLPBlock`` construct its ``NLinear`` in ``__init__``, before the
input width is known; the concrete value is filled in at build time and is what
``get_config()`` then serializes.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.tabular.nlinear")
class NLinear(keras.layers.Layer):
    """
    N fully independent parallel linear layers using einsum.

    This layer implements ``n`` truly independent linear layers processed in
    parallel via a single weight tensor of shape ``(n, input_dim, output_dim)``
    and ``einsum``. Useful for final output heads of an ensemble where each
    member needs its own independent classifier.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, N, D_in]             │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  einsum('bni,nio->bno',         │
        │         input, kernels)         │
        │  kernels [N, D_in, D_out]       │
        │  + opt. bias [N, D_out]         │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, N, D_out]           │
        └─────────────────────────────────┘

    :param n: Number of parallel linear layers. Must be positive.
    :type n: int
    :param input_dim: Input dimension, or ``None`` to infer it from the input
        shape at build time. Must be positive when given.
    :type input_dim: int or None
    :param output_dim: Output dimension per linear layer. Must be positive.
    :type output_dim: int
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :raises ValueError: If ``n`` or ``output_dim`` is not positive, or if
        ``input_dim`` is given and is not positive.
    """

    def __init__(
            self,
            n: int,
            input_dim: Optional[int],
            output_dim: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if n <= 0:
            raise ValueError(f"n must be positive; got {n!r}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive; got {output_dim!r}")
        if input_dim is not None and input_dim <= 0:
            raise ValueError(f"input_dim must be positive or None; got {input_dim!r}")

        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the parallel linear layer weights.

        :param input_shape: Shape of the input, ``(batch, n, input_dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 3, if
            ``input_shape[1]`` disagrees with ``n``, or if ``input_dim`` was
            passed explicitly and disagrees with ``input_shape[-1]``. Without
            these checks ``build()`` accepts the mismatch,
            ``compute_output_shape`` reports ``(batch, n, output_dim)`` anyway,
            and the failure surfaces later as an opaque backend error from the
            ``einsum`` in :meth:`call`. A ``None`` (symbolic) axis is not
            checked; the RANK always is.
        """
        # DECISION plan-2026-09-07T130829-d709705c/D-006: rank check FIRST. The
        # `bni,nio->bno` einsum in call() is rank-3-only, and at rank 2 axis 1 is
        # the feature axis -- `NLinear(n=3, input_dim=3, output_dim=5)` built on
        # `(None, 3)` and only then failed opaquely inside the einsum, with the
        # axis-1 message blaming `n` for a rank fault. `input_dim=None` (the
        # packed-block deferral) reads `input_shape[-1]` right below, so the rank
        # must be settled before that read. See decisions.md D-006.
        if len(input_shape) != 3:
            raise ValueError(
                f"NLinear requires a rank-3 input (batch, n, input_dim); "
                f"got rank {len(input_shape)}; input_shape={tuple(input_shape)!r}"
            )
        if input_shape[1] is not None and input_shape[1] != self.n:
            raise ValueError(
                f"n was given as {self.n!r} but the input's axis 1 is "
                f"{input_shape[1]!r}; input_shape={tuple(input_shape)!r}"
            )

        # `input_dim=None` defers the fan-in to build time, so a packed
        # TabMMLPBlock can construct its NLinear in __init__, before the input width is known.
        if self.input_dim is None:
            self.input_dim = input_shape[-1]
        elif input_shape[-1] is not None and input_shape[-1] != self.input_dim:
            raise ValueError(
                f"input_dim was given as {self.input_dim!r} but the input's last "
                f"axis is {input_shape[-1]!r}; input_shape={tuple(input_shape)!r}"
            )

        self.kernels = self.add_weight(
            shape=(self.n, self.input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernels'
        )

        if self.use_bias:
            self.biases = self.add_weight(
                shape=(self.n, self.output_dim),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='biases'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through N parallel linear layers.

            :param inputs: Input tensor of shape (batch_size, n, input_dim).
            :type inputs: keras.KerasTensor

            :return: Output tensor of shape (batch_size, n, output_dim).
            :rtype: keras.KerasTensor
        """
        # One batched einsum computes all n independent matmuls at once.
        outputs = keras.ops.einsum('bni,nio->bno', inputs, self.kernels)

        if self.use_bias:
            outputs = keras.ops.add(outputs, keras.ops.expand_dims(self.biases, axis=0))

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.n, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "n": self.n,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
