"""``ScaleEnsemble`` -- one learnable per-feature scaling vector per ensemble member.

This is the smallest per-member perturbation a batched ensemble can carry: a
weight of shape ``(k, input_dim)`` multiplied elementwise into an input of shape
``(batch, k, input_dim)``. Every member sees the same features but learns its own
relative weighting of them, so the ``k`` members specialize without any of them
owning a private copy of the surrounding kernel.

The layer holds no ``kernel_initializer`` knob. Which distribution the scaling
vectors are drawn from is decided entirely by ``init_distribution``, resolved
through :func:`dl_techniques.layers.tabular._ensemble_scaling.ensemble_scaling_initializer`
-- and that choice is what decides whether the members start as different
functions at all: ``'random-signs'`` and ``'normal'`` break symmetry, ``'ones'``
does not.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import (
    EnsembleInitDistribution,
    ensemble_scaling_initializer,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.tabular.scale_ensemble")
class ScaleEnsemble(keras.layers.Layer):
    """
    Learnable per-feature scaling for ensemble members.

    This layer applies a learnable, per-feature scaling factor to each of ``k``
    ensemble members, allowing each member to specialize by learning the
    relative importance of different features. The operation is
    ``output = input * weight`` with broadcasting over the batch dimension.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────┐
        │  Input [B, K, D]            │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │  Element-wise multiply      │
        │  with weight [K, D]         │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │  Output [B, K, D]           │
        └─────────────────────────────┘

    :param k: Number of ensemble members. Must be positive.
    :type k: int
    :param input_dim: Input feature dimension. Must be positive.
    :type input_dim: int
    :param init_distribution: Initialization distribution for the scaling weights
        (``'ones'``, ``'normal'`` or ``'random-signs'``). ``'random-signs'`` draws
        from :math:`\\{-1, +1\\}` and ``'normal'`` from :math:`\\mathcal{N}(1, 0.1)`.
    :type init_distribution: str
    :param kernel_regularizer: Optional regularizer for scaling weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :raises ValueError: If ``k`` or ``input_dim`` is not positive, or if
        ``init_distribution`` is not one of the three supported names.
    """

    def __init__(
            self,
            k: int,
            input_dim: int,
            init_distribution: EnsembleInitDistribution = 'normal',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if k <= 0:
            raise ValueError(f"k must be positive; got {k!r}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim!r}")

        self.k = k
        self.input_dim = input_dim
        self.init_distribution = init_distribution
        # The initializer is fully determined by `init_distribution`; there is
        # no separate `kernel_initializer` knob to contradict it. That is also
        # why `get_config()` below serializes only `init_distribution`.
        self.kernel_initializer = ensemble_scaling_initializer(init_distribution)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the scaling weights with proper initialization.

        :param input_shape: Shape of the input, ``(batch, k, input_dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 3, or if
            ``input_shape[1]`` disagrees with ``k``, or if ``input_shape[-1]``
            disagrees with ``input_dim``. All three are fixed by construction,
            and :meth:`call` is a broadcasting elementwise multiply, so an
            unchecked mismatch is absorbed silently and returns a
            plausible-looking result computed from the wrong weight. A ``None``
            (symbolic) axis is not checked; the RANK always is.
        """
        # DECISION plan-2026-09-07T130829-d709705c/D-006: the rank check comes
        # FIRST and is unconditional. The two axis checks below encode a rank-3
        # assumption they cannot state: on a rank-4 `(2, k, k, D)` input both
        # passed and `call()` then broadcast the `(k, D)` member weight along
        # axis 2 instead of axis 1 -- the wrong computation with no error at all.
        # On a rank-2 input axis 1 is the FEATURE axis, so the k-guard's message
        # blamed `k` for what is really a rank fault. Do NOT move this below the
        # axis checks and do NOT make it `is not None`-conditional: rank is never
        # unknown in a Keras shape tuple. See decisions.md D-006.
        if len(input_shape) != 3:
            raise ValueError(
                f"ScaleEnsemble requires a rank-3 input (batch, k, input_dim); "
                f"got rank {len(input_shape)}; input_shape={tuple(input_shape)!r}"
            )
        if input_shape[1] is not None and input_shape[1] != self.k:
            raise ValueError(
                f"k was given as {self.k!r} but the input's axis 1 is "
                f"{input_shape[1]!r}; input_shape={tuple(input_shape)!r}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.input_dim:
            raise ValueError(
                f"input_dim was given as {self.input_dim!r} but the input's last "
                f"axis is {input_shape[-1]!r}; input_shape={tuple(input_shape)!r}"
            )

        self.weight = self.add_weight(
            shape=(self.k, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='ensemble_weight'
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply ensemble scaling to inputs.

            :param inputs: Input tensor of shape (batch_size, k, input_dim).
            :type inputs: keras.KerasTensor

            :return: Scaled tensor of shape (batch_size, k, input_dim).
            :rtype: keras.KerasTensor
        """
        # Efficient broadcasting: inputs (B, K, D) * weight (K, D) -> (B, K, D)
        return keras.ops.multiply(inputs, keras.ops.expand_dims(self.weight, axis=0))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "k": self.k,
            "input_dim": self.input_dim,
            "init_distribution": self.init_distribution,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
