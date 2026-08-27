"""Global Response Normalization (GRN), generalized to rank 2, 3 and 4 inputs.

GRN comes from ConvNeXt V2. It makes channels compete. Each channel is scored by
its own magnitude. That score is divided by the mean score across channels, and
the input is reweighted by the result. Weak channels are damped and strong ones
are amplified.

Computation
-----------

For an input ``X`` of rank ``N``, with the channel axis last:

1. Per-channel score, reduced over axes ``1 .. N - 2``::

       norm_c = sqrt(sum(X_c ** 2) + eps)

   ``eps`` is added INSIDE the square root, not outside it. So ``norm_c`` is
   never below ``sqrt(eps)``, and it is slightly above the exact ``||X_c||_2``.
2. Mean across channels: ``mu = mean(norm_c)``.
3. Normalized score: ``norm'_c = norm_c / (mu + eps)``.
4. Output: ``Y = X + gamma * (X * norm') + beta``, with ``gamma`` and ``beta``
   broadcast over the reduced axes.

Rank 2 is a degenerate case. The reduction axes are ``range(1, 1)``, an empty
tuple, and summing over no axis is the identity. ``norm_c`` therefore becomes
``sqrt(x ** 2 + eps)``, which is each element's own absolute value. Measured at
float32 on the input ``[[3.0, -4.0]]``: ``norm_c`` is ``[[3.0, 4.0]]``. The
layer OUTPUT on that input is ``[[5.571428, -8.571427]]``, because ``gamma``
initializes to ones and the residual ``X`` is added back.

Weights
-------

``gamma`` and ``beta`` have shape ``(1,) * (rank - 1) + (channels,)``. Measured:
a ``(2, 4, 5, 6)`` input gives ``(1, 1, 1, 6)``. Passing ``use_beta=False``
creates no ``beta`` weight at all, leaving the layer with one weight instead of
two.

References
----------

[1] Woo, S., et al. (2023). "ConvNeXt V2: Co-designing and Scaling ConvNets with
    Masked Autoencoders". arXiv:2301.00808
[2] Liu, Z., et al. (2022). "A ConvNet for the 2020s". arXiv:2201.03545
"""

import keras
from typing import Any, Dict, Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GlobalResponseNormalization(keras.layers.Layer):
    """Global Response Normalization for rank 2, 3 and 4 inputs.

    Scores each channel by its L2 magnitude over the spatial or sequence axes.
    That score is divided by the mean score across channels, and the input is
    reweighted by the result. A learnable ``gamma`` scales the reweighted term,
    an optional ``beta`` offsets it, and the input is added back as a residual::

        Y = X + gamma * (X * (norm_c / (mean(norm) + eps))) + beta

    The output has the same shape as the input.

    .. warning::
        **This layer does not support masking.** ``supports_masking`` is left
        ``False``, because the per-channel score is reduced over the spatial or
        sequence axes. Changing one ``(sample, token)`` slot moves the other
        positions of that sample by up to ``2.971`` (measured on a ``(3, 5, 8)``
        input). A propagated Keras mask would claim the output is independent of
        the padding when it is not, which is worse than having no mask.

    **Architecture Overview:**

    .. code-block:: text

            inputs (X): (B, ..., C)
                            │
                            ├─────────────────────────────────┐
                            │                                 │
                            │                           ┌─────┤
                            │                           │     │
                            ▼                           │     │
           ┌─────────────────────────────────┐          │     │
           │ per-channel L2 over spatial     │          │     │
           │ axes: sqrt(sum(X^2) + eps)      │          │     │
           └────────────────┬────────────────┘          │     │
                            │ norm: (B, 1..., C)        │     │
                            ▼                           │     │
           ┌─────────────────────────────────┐          │     │
           │ mean over channels, then        │          │     │
           │ divide: norm / (mean + eps)     │          │     │
           └────────────────┬────────────────┘          │     │
                            │ norm': (B, 1..., C)       │     │
                            ▼                           ▼     │
           ┌────────────────────────────────────────────────┐ │
           │ gamma * (X * norm')                            │ │
           │ plus beta, when use_beta=True (optional)       │ │
           └───────────────────────┬────────────────────────┘ │
                                   │                          │
                                   ▼                          ▼
           ┌──────────────────────────────────────────────────────┐
           │ residual add: Y = X + transformed                    │
           └──────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
                          output (Y): (B, ..., C)

    :param eps: Constant added inside the square root and to the mean before
        dividing. Must be positive. Defaults to 1e-6.
    :type eps: float
    :param gamma_initializer: Initializer for the ``gamma`` scale weight.
        Defaults to ``'ones'``. The ConvNeXt V2 paper initializes gamma to zero,
        which makes GRN an identity at initialization; pass ``'zeros'`` to
        reproduce that.
    :type gamma_initializer: Union[str, keras.initializers.Initializer]
    :param beta_initializer: Initializer for the ``beta`` offset weight. Defaults
        to ``'zeros'``. Ignored when ``use_beta=False``.
    :type beta_initializer: Union[str, keras.initializers.Initializer]
    :param gamma_regularizer: Optional regularizer for ``gamma``.
    :type gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param beta_regularizer: Optional regularizer for ``beta``.
    :type beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Optional regularizer for the layer output.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_beta: Whether to create the trainable offset ``beta``. Defaults to
        ``True``, which keeps existing ConvNeXt V2 checkpoints loadable. Pass
        ``False`` for a bias-free layer: no ``beta`` weight is created
        (``layer.beta is None``) and the output becomes
        ``Y = X + gamma * (X * norm')``.
    :type use_beta: bool

    :ivar gamma: Scale weight of shape ``(1,) * (rank - 1) + (channels,)``.
        ``None`` until ``build()`` runs.
    :vartype gamma: Optional[keras.Variable]
    :ivar beta: Offset weight of the same shape as ``gamma``, or ``None`` when
        ``use_beta=False``.
    :vartype beta: Optional[keras.Variable]

    :raises ValueError: If eps is not positive.
    :raises ValueError: If the input rank is not 2, 3 or 4.
    :raises ValueError: If the channel dimension is undefined.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import GlobalResponseNormalization

        x = keras.random.normal((2, 8, 8, 16))
        y = GlobalResponseNormalization()(x)
    """

    def __init__(
        self,
        eps: float = 1e-6,
        gamma_initializer: Union[str, keras.initializers.Initializer] = 'ones',
        beta_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_beta: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the layer.

        :param eps: Constant used for numerical stability. Must be positive.
        :type eps: float
        :param gamma_initializer: Initializer for the ``gamma`` scale weight.
        :type gamma_initializer: Union[str, keras.initializers.Initializer]
        :param beta_initializer: Initializer for the ``beta`` offset weight.
            Ignored when ``use_beta=False``.
        :type beta_initializer: Union[str, keras.initializers.Initializer]
        :param gamma_regularizer: Regularizer for ``gamma``.
        :type gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param beta_regularizer: Regularizer for ``beta``.
        :type beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param activity_regularizer: Regularizer for the layer output.
        :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param use_beta: Whether to create the trainable offset ``beta``.
            ``False`` creates no ``beta`` weight and drops the additive term.
        :type use_beta: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If eps is not positive.
        """
        # activity_regularizer is a base-Layer constructor argument, so it is
        # forwarded rather than assigned afterwards. Plain assignment also works,
        # through the inherited property setter, but forwarding is the idiom the
        # rest of this package uses.
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps
        self.use_beta = use_beta
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)

        self.gamma = None
        self.beta = None

        logger.debug(f"Initialized GlobalResponseNormalization with eps={eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``gamma``, and ``beta`` when ``use_beta`` is set.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the input rank is not 2, 3 or 4.
        :raises ValueError: If the channel dimension is undefined.
        """
        if self.built:
            return

        rank = len(input_shape)
        if rank not in [2, 3, 4]:
            raise ValueError(
                f"Input rank must be 2, 3, or 4 (batch, [dims...], channels), "
                f"but got rank {rank}"
            )

        channels = input_shape[-1]
        if channels is None:
            raise ValueError("The channel/feature dimension (last axis) must be defined.")

        logger.debug(f"Building GlobalResponseNormalization for rank {rank} with {channels} channels")

        # Leading axes are 1 so the weights broadcast over the reduced axes.
        param_shape = (1,) * (rank - 1) + (channels,)

        self.gamma = self.add_weight(
            name="gamma",
            shape=param_shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=True,
        )
        # use_beta=False drops the trainable offset entirely, leaving the layer
        # with no additive term. The default True keeps existing ConvNeXt V2
        # checkpoints loadable.
        if self.use_beta:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                trainable=True,
            )
        else:
            self.beta = None

        logger.debug("GlobalResponseNormalization build completed")

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply global response normalization.

        :param inputs: Input tensor of shape ``(batch, ..., channels)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer behaves the same
            in both modes and the argument is kept for API compatibility.
        :type training: Optional[bool]

        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        rank = keras.ops.ndim(inputs)

        # Step 1: L2 norm over the spatial or sequence axes.
        # Rank 3 (batch, seq, features) reduces axis (1,).
        # Rank 4 (batch, h, w, channels) reduces axes (1, 2).
        # Rank 2 (batch, features) reduces an EMPTY tuple. Summing over no axis
        # is the identity, so norm becomes sqrt(x ** 2 + eps), the element's own
        # absolute value.
        axes_to_reduce = tuple(range(1, rank - 1))
        norm = keras.ops.sqrt(
            keras.ops.sum(
                keras.ops.square(inputs),
                axis=axes_to_reduce,
                keepdims=True) + self.eps
        )

        # Step 2: divide by the mean norm across channels.
        mean_norm = keras.ops.mean(norm, axis=-1, keepdims=True)
        normalized_norm = norm / (mean_norm + self.eps)

        # Step 3: reweight, then add the input back. norm, gamma and beta all
        # broadcast against the input shape.
        transformed = self.gamma * (inputs * normalized_norm)
        if self.beta is not None:
            transformed = transformed + self.beta
        output = inputs + transformed

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple that was passed in.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this layer.

        :return: Dictionary holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "eps": float(self.eps),
            "use_beta": self.use_beta,
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
