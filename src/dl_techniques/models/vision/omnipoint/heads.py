"""Output heads for OmniPoint.

Three heads read one shared encoder output. ``RayDistanceHead`` and ``MaskHead``
each wrap a single
:class:`~dl_techniques.models.vision.depth_anything.components.DPTDecoder` over
the spatial feature map, with head-specific ``output_channels`` and
``output_activation``. ``MetricScaleHead`` is a small MLP over one pooled token,
because ``DPTDecoder`` preserves the spatial axes and cannot produce a pooled
scalar.

Two output activations are fixed by ``losses/omnipoint_losses.py`` rather than
chosen here. ``MaskHead`` emits linear logits, because ``MaskLoss`` wraps
``BinaryCrossentropy(from_logits=True)`` and a sigmoid here would be applied
twice. ``MetricScaleHead`` emits a positive scale rather than a log-scale value,
because ``MetricScaleLoss`` takes the log of its ``y_pred`` itself.
``RayDistanceHead``'s decoder is linear for the same kind of reason:
``RayPointCombinator`` owns the normalization and the positivity.

Both dense heads take ``upsample_factor``, which ``DPTDecoder`` requires to be a
power of 2.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.vision.depth_anything.components import DPTDecoder
from dl_techniques.layers.geometric.ray_point_combinator import RayPointCombinator
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.omnipoint.heads.ray_distance_head")
class RayDistanceHead(keras.layers.Layer):
    """Predict a unit ray, a positive distance and their product per pixel.

    One ``DPTDecoder`` emits 4 linear channels: 3 for the unnormalized ray and
    1 for the unnormalized distance. ``RayPointCombinator`` then normalizes the
    ray, makes the distance positive, and multiplies them, so the head's own
    outputs are unit-norm and positive without any caller-side correction.

    Architecture:

    .. code-block:: text

                  features [B, H, W, C]
                            ▼
            ┌───────────────────────────────┐
            │ DPTDecoder                    │
            │  4 channels, linear           │
            └───────────────┬───────────────┘
                     [B, H', W', 4]
                            ▼
            ┌───────────────────────────────┐
            │ split at channel 3            │
            └──────┬─────────────────┬──────┘
                   ▼                 ▼
                raw_ray          raw_distance
                   └────────┬────────┘
                            ▼
            ┌───────────────────────────────┐
            │ RayPointCombinator            │
            │  unit ray, positive distance  │
            └───────────────┬───────────────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           unit_ray      distance       point
          [B,H',W',3]   [B,H',W',1]   [B,H',W',3]

    H' and W' are the input size times ``upsample_factor``.

    :param dims: Channel dimension per ``DPTDecoder`` stage. ``None`` gives
        ``[256, 128, 64, 32]``.
    :type dims: Optional[list]
    :param upsample_factor: Total bilinear upsampling factor forwarded to
        ``DPTDecoder``. Must be a power of 2. Defaults to ``1``.
    :type upsample_factor: int
    :param kernel_initializer: Initializer for every ``DPTDecoder`` conv
        kernel. Defaults to ``"he_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Floor forwarded to ``RayPointCombinator``, guarding both
        the ray norm and the distance activation. Defaults to ``1e-8``.
    :type epsilon: float
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, the encoder's
        spatial feature map.

    Output shape:
        3-tuple ``(unit_ray, positive_distance, point)``, each
        ``(batch_size, height * upsample_factor, width * upsample_factor, C)``
        with ``C=3`` for ``unit_ray`` and ``point`` and ``C=1`` for
        ``positive_distance``.

    Example:
        .. code-block:: python

            head = RayDistanceHead(dims=[256, 128, 64, 32], upsample_factor=1)
            x = keras.random.normal([2, 16, 16, 768])
            ray, distance, point = head(x)
    """

    def __init__(
            self,
            dims: Optional[list] = None,
            upsample_factor: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-8,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dims = list(dims) if dims is not None else [256, 128, 64, 32]
        self.upsample_factor = int(upsample_factor)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        self.decoder = DPTDecoder(
            dims=self.dims,
            output_channels=4,
            output_activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=self.upsample_factor,
            name="ray_distance_decoder",
        )
        self.combinator = RayPointCombinator(
            epsilon=self.epsilon, name="ray_point_combinator"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Decode the raw ray and distance channels, then combine them.

        :param inputs: Encoder spatial feature map, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner ``DPTDecoder``.
        :type training: Optional[bool]
        :return: ``(unit_ray, positive_distance, point)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        raw = self.decoder(inputs, training=training)
        raw_ray, raw_distance = keras.ops.split(raw, [3], axis=-1)
        return self.combinator((raw_ray, raw_distance))

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "upsample_factor": self.upsample_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "epsilon": self.epsilon,
        })
        return config


@register_dl_technique("dl_techniques.models.omnipoint.heads.mask_head")
class MaskHead(keras.layers.Layer):
    """Predict a per-pixel sky and validity mask as linear logits.

    One ``DPTDecoder`` emits a single unbounded channel. The sigmoid lives in
    ``MaskLoss``, which wraps ``BinaryCrossentropy(from_logits=True)``, so a
    caller that wants probabilities applies one itself.

    Architecture:

    .. code-block:: text

                  features [B, H, W, C]
                            ▼
            ┌───────────────────────────────┐
            │ DPTDecoder                    │
            │  1 channel, linear            │
            └───────────────┬───────────────┘
                            ▼
                mask_logit [B, H', W', 1]

    H' and W' are the input size times ``upsample_factor``.

    :param dims: Channel dimension per ``DPTDecoder`` stage. ``None`` gives
        ``[256, 128, 64, 32]``.
    :type dims: Optional[list]
    :param upsample_factor: Total bilinear upsampling factor. Must be a power
        of 2. Defaults to ``1``.
    :type upsample_factor: int
    :param kernel_initializer: Initializer for every conv kernel. Defaults to
        ``"he_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        4D tensor ``(batch_size, height * upsample_factor,
        width * upsample_factor, 1)``, linear mask logits.
    """

    def __init__(
            self,
            dims: Optional[list] = None,
            upsample_factor: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dims = list(dims) if dims is not None else [256, 128, 64, 32]
        self.upsample_factor = int(upsample_factor)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.decoder = DPTDecoder(
            dims=self.dims,
            output_channels=1,
            # DECISION D-010: linear logits, not sigmoid; MaskLoss wraps
            # BinaryCrossentropy(from_logits=True). See decisions.md.
            output_activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=self.upsample_factor,
            name="mask_decoder",
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Decode the mask logits.

        :param inputs: Encoder spatial feature map, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner ``DPTDecoder``.
        :type training: Optional[bool]
        :return: Mask logits, ``(B, H*upsample_factor, W*upsample_factor, 1)``.
        :rtype: keras.KerasTensor
        """
        return self.decoder(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "upsample_factor": self.upsample_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


@register_dl_technique("dl_techniques.models.omnipoint.heads.metric_scale_head")
class MetricScaleHead(keras.layers.Layer):
    """Predict one strictly positive metric scale per sample.

    A two-layer MLP over whichever single ``(B, D)`` pooled token the caller
    supplies. The output is the scale itself, not its logarithm, because
    ``MetricScaleLoss`` takes the log of its ``y_pred``. ``softplus`` gives
    positivity and ``epsilon`` keeps the result away from exact zero.

    Architecture:

    .. code-block:: text

                   metric token [B, D]
                            ▼
            ┌───────────────────────────────┐
            │ Dense(hidden_dim), relu       │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │ Dense(1), linear              │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │ max(softplus(x), epsilon)     │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │ squeeze last axis             │
            └───────────────┬───────────────┘
                            ▼
              scale [B], strictly positive

    :param hidden_dim: Width of the hidden ``Dense`` layer. Defaults to
        ``256``.
    :type hidden_dim: int
    :param kernel_initializer: Initializer for both ``Dense`` kernels. Defaults
        to ``"he_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for both ``Dense`` kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Lower bound applied to the ``softplus`` output, so the
        scale never reaches exact ``0.0`` in float32. Defaults to ``1e-8``.
    :type epsilon: float
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        2D tensor ``(batch_size, embed_dim)``, a pooled or CLS token.

    Output shape:
        1D tensor ``(batch_size,)``, strictly positive.
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-8,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        self.hidden = keras.layers.Dense(
            self.hidden_dim,
            activation="relu",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="metric_scale_hidden",
        )
        # Linear here; softplus and the epsilon floor are applied in call(), so
        # the floor stays visible at the one site that needs it.
        self.projection = keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="metric_scale_projection",
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Compute the positive per-sample metric scale.

        :param inputs: Pooled metric token, ``(B, D)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner ``Dense`` layers.
        :type training: Optional[bool]
        :return: Positive scale, ``(B,)``.
        :rtype: keras.KerasTensor
        """
        x = self.hidden(inputs, training=training)
        raw_scale = self.projection(x, training=training)
        positive_scale = keras.ops.maximum(
            keras.ops.softplus(raw_scale), self.epsilon
        )
        return keras.ops.squeeze(positive_scale, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
