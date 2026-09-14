"""Optional geometric conditioning for OmniPoint.

``ConditioningInputEncoder`` encodes an intrinsics ray map and a sparse depth
pair into a handful of full-resolution channels and concatenates them onto the
RGB image, so conditioning enters before the ViT rather than at the token level.
That leaves ``ViT`` and its patch embedding untouched: the encoder's
``input_shape`` simply carries a wider channel count.
``ConditioningStateEmbedding`` runs after the encoder instead, because the
present/absent indicator it adds operates on a token sequence, which does not
exist until the encoder has run. Every conditioning computation therefore sits
in exactly one of two stages, pixel-space or token-space.

A ``None`` tensor means the modality is absent for the whole batch. Within a
batch that does supply a tensor, a per-sample boolean flag ``(B,)`` marks which
samples carry valid data: those flagged absent have their conditioning channels
zeroed before fusion and receive the absent embedding at the token stage, so a
sample's prediction can depend on a modality only when its own flag marks that
modality present. ``sparse_depth_mask`` is required whenever ``sparse_depth``
is given.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.embedding.state_indicator_embedding import (
    StateIndicatorEmbedding,
)
from dl_techniques.layers.geometric.sparse_depth_splat import SparseDepthSplat
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.omnipoint.conditioning.input_encoder")
class ConditioningInputEncoder(keras.layers.Layer):
    """Fuse an RGB image with intrinsics and sparse-depth channels.

    The intrinsics ray map goes through a small ``Conv2D`` stack. The sparse
    depth pair is densified by
    :class:`~dl_techniques.layers.geometric.sparse_depth_splat.SparseDepthSplat`
    into a depth map and a confidence map, which go through a second stack. Both
    results are concatenated onto the image along the channel axis. Either
    modality may be absent for the whole batch or for individual samples.

    Architecture:

    .. code-block:: text

        image [B,H,W,C]  intrinsics_ray_map   sparse_depth + mask
              │                   │                     │
              │          ┌───────────────────┐ ┌───────────────────┐
              │          │ None -> all zeros │ │ None -> all zeros │
              │          │ flag -> zero rows │ │ flag -> zero rows │
              │          └─────────┬─────────┘ └─────────┬─────────┘
              │                    │                     ▼
              │                    │           ┌───────────────────┐
              │                    │           │ SparseDepthSplat  │
              │                    │           │  dense+confidence │
              │                    │           └─────────┬─────────┘
              │                    ▼                     ▼
              │          ┌───────────────────┐ ┌───────────────────┐
              │          │ Conv2D hidden relu│ │ Conv2D hidden relu│
              │          │ Conv2D out linear │ │ Conv2D out linear │
              │          └─────────┬─────────┘ └─────────┬─────────┘
              │               [B,H,W,ic]            [B,H,W,dc]
              ▼                    ▼                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ concatenate along the channel axis                      │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
                      fused [B, H, W, C+ic+dc]

    Absence handling, per modality:

    .. code-block:: text

        tensor        *_present     effect
        ──────────    ──────────    ─────────────────────────────
        None          any           channels are all zero
        given         None          every sample contributes
        given         given         flagged-absent samples zeroed

    :param intrinsics_channels: Fusion channels emitted by the intrinsics
        encoder. Defaults to ``4``.
    :type intrinsics_channels: int
    :param depth_channels: Fusion channels emitted by the sparse-depth encoder.
        Defaults to ``4``.
    :type depth_channels: int
    :param conv_hidden_channels: Hidden width of both ``Conv2D`` stacks.
        Defaults to ``16``.
    :type conv_hidden_channels: int
    :param splat_kernel_size: Forwarded to ``SparseDepthSplat``. Defaults to
        ``(9, 9)``.
    :type splat_kernel_size: Tuple[int, int]
    :param splat_sigma: Forwarded to ``SparseDepthSplat``. Defaults to ``2.0``.
    :type splat_sigma: float
    :param kernel_initializer: Initializer for every conv kernel. Defaults to
        ``"he_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        ``image``: 4D ``(batch_size, height, width, image_channels)``.
        ``intrinsics_ray_map``: 4D ``(batch_size, height, width, 3)``.
        ``sparse_depth`` and ``sparse_depth_mask``: 4D
        ``(batch_size, height, width, 1)``.

    Output shape:
        4D tensor ``(batch_size, height, width, image_channels +
        intrinsics_channels + depth_channels)``.
    """

    def __init__(
            self,
            intrinsics_channels: int = 4,
            depth_channels: int = 4,
            conv_hidden_channels: int = 16,
            splat_kernel_size: Tuple[int, int] = (9, 9),
            splat_sigma: float = 2.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.intrinsics_channels = int(intrinsics_channels)
        self.depth_channels = int(depth_channels)
        self.conv_hidden_channels = int(conv_hidden_channels)
        self.splat_kernel_size = (int(splat_kernel_size[0]), int(splat_kernel_size[1]))
        self.splat_sigma = float(splat_sigma)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.splat = SparseDepthSplat(
            kernel_size=self.splat_kernel_size,
            sigma=self.splat_sigma,
            name="sparse_depth_splat",
        )
        self.intrinsics_conv1 = keras.layers.Conv2D(
            self.conv_hidden_channels, 3, padding="same", activation="relu",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="intrinsics_conv1",
        )
        self.intrinsics_conv2 = keras.layers.Conv2D(
            self.intrinsics_channels, 3, padding="same", activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="intrinsics_conv2",
        )
        self.depth_conv1 = keras.layers.Conv2D(
            self.conv_hidden_channels, 3, padding="same", activation="relu",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="depth_conv1",
        )
        self.depth_conv2 = keras.layers.Conv2D(
            self.depth_channels, 3, padding="same", activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="depth_conv2",
        )

    def build(self, input_shape: Tuple) -> None:
        """Build every sublayer, whichever modalities a later call supplies.

        The convs need only a channel count, not a spatial size, so each is
        built from a dummy shape here. ``self.splat`` runs only when sparse
        depth is given, so leaving it to auto-build would leave it unbuilt for
        callers that never supply that modality.

        :param input_shape: Shape of the ``image`` argument, which is the first
            positional input this layer is invoked on.
        :type input_shape: tuple
        """
        # DECISION D-015: build every sublayer here, unconditionally; an unbuilt
        # self.splat makes Keras 3 report unbuilt state. See decisions.md.
        self.intrinsics_conv1.build((None, None, None, 3))
        self.intrinsics_conv2.build((None, None, None, self.conv_hidden_channels))
        self.depth_conv1.build((None, None, None, 2))
        self.depth_conv2.build((None, None, None, self.conv_hidden_channels))
        self.splat.build([(None, None, None, 1), (None, None, None, 1)])
        super().build(input_shape)

    def _zero_absent_samples(
            self,
            tensor: keras.KerasTensor,
            present: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Zero the samples whose per-sample flag marks them absent.

        :param tensor: A per-sample tensor, ``(B, ...)``.
        :type tensor: keras.KerasTensor
        :param present: Optional boolean or 0-1 flag, ``(B,)`` or ``(B, 1)``.
            ``None`` leaves the tensor unchanged.
        :type present: Optional[keras.KerasTensor]
        :return: ``tensor``, unchanged where ``present`` is truthy and zeroed
            where it is not.
        :rtype: keras.KerasTensor
        """
        if present is None:
            return tensor
        present = keras.ops.cast(present, tensor.dtype)
        # Reshape to (B, 1, 1, 1) so it broadcasts against (B, H, W, C).
        present = keras.ops.reshape(present, (-1, 1, 1, 1))
        return tensor * present

    def call(
            self,
            image: keras.KerasTensor,
            intrinsics_ray_map: Optional[keras.KerasTensor] = None,
            intrinsics_present: Optional[keras.KerasTensor] = None,
            sparse_depth: Optional[keras.KerasTensor] = None,
            sparse_depth_mask: Optional[keras.KerasTensor] = None,
            sparse_depth_present: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Fuse the RGB image with the optional conditioning signals.

        :param image: RGB image batch, ``(B, H, W, image_channels)``.
        :type image: keras.KerasTensor
        :param intrinsics_ray_map: Optional per-pixel unit ray map,
            ``(B, H, W, 3)``. ``None`` means absent for the whole batch.
        :type intrinsics_ray_map: Optional[keras.KerasTensor]
        :param intrinsics_present: Optional per-sample flag, ``(B,)``. Omitting
            it treats every sample as present.
        :type intrinsics_present: Optional[keras.KerasTensor]
        :param sparse_depth: Optional sparse depth values, ``(B, H, W, 1)``.
            ``None`` means absent for the whole batch.
        :type sparse_depth: Optional[keras.KerasTensor]
        :param sparse_depth_mask: Companion validity mask, required when
            ``sparse_depth`` is given, ``(B, H, W, 1)``.
        :type sparse_depth_mask: Optional[keras.KerasTensor]
        :param sparse_depth_present: Optional per-sample flag, ``(B,)``.
            Omitting it treats every sample as present.
        :type sparse_depth_present: Optional[keras.KerasTensor]
        :param training: Forwarded to every inner sublayer.
        :type training: Optional[bool]
        :return: Fused image, ``(B, H, W, image_channels +
            intrinsics_channels + depth_channels)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(image)[0]
        height, width = keras.ops.shape(image)[1], keras.ops.shape(image)[2]

        if intrinsics_ray_map is None:
            intrinsics_ray_map = keras.ops.zeros(
                (batch_size, height, width, 3), dtype=image.dtype
            )
        elif intrinsics_present is not None:
            intrinsics_ray_map = self._zero_absent_samples(
                intrinsics_ray_map, intrinsics_present
            )

        if sparse_depth is None:
            dense_depth = keras.ops.zeros(
                (batch_size, height, width, 1), dtype=image.dtype
            )
            dense_confidence = keras.ops.zeros(
                (batch_size, height, width, 1), dtype=image.dtype
            )
        else:
            if sparse_depth_present is not None:
                sparse_depth = self._zero_absent_samples(
                    sparse_depth, sparse_depth_present
                )
                sparse_depth_mask = self._zero_absent_samples(
                    sparse_depth_mask, sparse_depth_present
                )
            dense_depth, dense_confidence = self.splat(
                (sparse_depth, sparse_depth_mask), training=training
            )

        intrinsics_features = self.intrinsics_conv1(intrinsics_ray_map, training=training)
        intrinsics_features = self.intrinsics_conv2(intrinsics_features, training=training)

        depth_input = keras.ops.concatenate([dense_depth, dense_confidence], axis=-1)
        depth_features = self.depth_conv1(depth_input, training=training)
        depth_features = self.depth_conv2(depth_features, training=training)

        return keras.ops.concatenate(
            [image, intrinsics_features, depth_features], axis=-1
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus every constructor argument.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "intrinsics_channels": self.intrinsics_channels,
            "depth_channels": self.depth_channels,
            "conv_hidden_channels": self.conv_hidden_channels,
            "splat_kernel_size": self.splat_kernel_size,
            "splat_sigma": self.splat_sigma,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


@register_dl_technique("dl_techniques.models.omnipoint.conditioning.state_embedding")
class ConditioningStateEmbedding(keras.layers.Layer):
    """Add present and absent indicators for both modalities to a token sequence.

    Two :class:`StateIndicatorEmbedding` layers run in turn, one per modality.
    Each selects a learned vector according to the sample's flag and adds it to
    every token of that sample, so the encoder's output carries which modalities
    the sample actually had.

    Architecture:

    .. code-block:: text

                  tokens [B, N, D]
                            ▼
            ┌───────────────────────────┐
            │ StateIndicatorEmbedding   │ ◄── intrinsics_present
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │ StateIndicatorEmbedding   │ ◄── sparse_depth_present
            └─────────────┬─────────────┘
                          ▼
                  tokens [B, N, D]

    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        ``tokens``: 3D ``(batch_size, sequence_length, dim)``.
        ``intrinsics_present`` and ``sparse_depth_present``: 1D
        ``(batch_size,)``.

    Output shape:
        3D tensor, same shape as ``tokens``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.intrinsics_state = StateIndicatorEmbedding(name="intrinsics_state")
        self.depth_state = StateIndicatorEmbedding(name="depth_state")

    def call(
            self,
            tokens: keras.KerasTensor,
            intrinsics_present: keras.KerasTensor,
            sparse_depth_present: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Add both present and absent embeddings to ``tokens``.

        :param tokens: Encoder output sequence, ``(B, N, D)``.
        :type tokens: keras.KerasTensor
        :param intrinsics_present: Per-sample flag, ``(B,)``.
        :type intrinsics_present: keras.KerasTensor
        :param sparse_depth_present: Per-sample flag, ``(B,)``.
        :type sparse_depth_present: keras.KerasTensor
        :param training: Forwarded to the inner embedding layers.
        :type training: Optional[bool]
        :return: ``tokens`` plus both selected embeddings, ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        tokens = self.intrinsics_state((tokens, intrinsics_present), training=training)
        tokens = self.depth_state((tokens, sparse_depth_present), training=training)
        return tokens

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config; this layer adds no constructor
            arguments of its own.
        :rtype: dict
        """
        return super().get_config()

# ---------------------------------------------------------------------
