"""OmniPoint's optional geometric conditioning: intrinsics ray-map + sparse
depth, wired into the ViT input stage.

Fusion-point decision (decisions.md D-013, a genuine architectural fork -- see
that entry for the full trade-off):
    `OmniPoint`'s encoder is a plain `ViT` (D-009) whose `call()` only accepts
    a raw pixel image -- it has no separate entry point for an already
    patchified token sequence, so injecting conditioning "at the token level"
    would mean re-implementing `ViT`'s own patch embedding here, duplicating
    logic that already exists and works. The least invasive path that needs
    NO change to `ViT` itself is: encode each conditioning signal into a
    small number of extra full-resolution channels, then concatenate them
    onto the RGB image BEFORE it enters the encoder. `OmniPoint` widens the
    encoder's own `input_shape` channel count accordingly (`ViT`'s
    `input_shape` is a generic ``(H, W, C)`` -- `PatchEmbedding2D`'s conv
    kernel is built from whatever `C` it is given, so this needs no change to
    `ViT`/`PatchEmbedding2D` either).

    `StateIndicatorEmbedding` (per-sample present/absent) operates on a TOKEN
    SEQUENCE, not a per-pixel map, so it cannot run at the same pre-encoder
    point as the channel concatenation above -- there is no token sequence
    yet. It is instead applied to the encoder's own output sequence
    ``(B, N+1, D)``, immediately after the encoder call and before the
    per-pixel/per-sample heads read it. This keeps every conditioning
    computation in exactly one of two well-defined stages: pre-encoder
    (pixel-space fusion) or post-encoder (token-space state embedding), never
    both mixed, and requires no surgery on `ViT`'s internals in either stage.

Mixed-batch / per-sample-flag contract (Problem Statement edge case):
    A single `OmniPoint.call()` invocation is necessarily a whole-batch
    decision about which OPTIONAL TENSORS are supplied at all (a Python
    ``None`` vs a concrete tensor) -- that is a `call()`-signature-level
    choice, not a per-sample one. Within a batch that DOES supply a
    conditioning tensor, an explicit per-sample boolean flag
    (``intrinsics_present`` / ``sparse_depth_present``, each ``(B,)``) then
    marks which samples in that batch actually carry valid data for that
    modality. Samples flagged absent have their conditioning channels forced
    to exactly zero before pixel-space fusion (`ConditioningInputEncoder`)
    and get the ``absent_embedding`` vector at the token-space stage
    (`ConditioningStateEmbedding`) -- both signals agree, so the ONLY way a
    sample's prediction can depend on this modality is if that sample's own
    flag says it is present.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

from dl_techniques.layers.embedding.state_indicator_embedding import (
    StateIndicatorEmbedding,
)
from dl_techniques.layers.geometric.sparse_depth_splat import SparseDepthSplat
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.omnipoint.conditioning.input_encoder")
class ConditioningInputEncoder(keras.layers.Layer):
    """Pre-encoder pixel-space fusion: RGB + intrinsics + sparse-depth.

    Encodes an intrinsics ray map (already-computed per-pixel unit rays,
    ``(B, H, W, 3)`` -- see ``utils/camera_models.py``) and a sparse depth
    pair (``(sparse_depth, validity_mask)``, each ``(B, H, W, 1)``, densified
    internally via :class:`~dl_techniques.layers.geometric.sparse_depth_splat.SparseDepthSplat`)
    each through a small `Conv2D` stack into a handful of full-resolution
    feature channels, then concatenates ``[image, intrinsics_features,
    depth_features]`` along the channel axis.

    Either conditioning signal may be entirely absent for the whole batch
    (pass ``None``); when present, an explicit per-sample boolean flag zeros
    that sample's contribution before it is concatenated (see this module's
    docstring, "Mixed-batch / per-sample-flag contract").

    :param intrinsics_channels: Number of fusion channels the intrinsics
        ray-map encoder emits.
    :type intrinsics_channels: int
    :param depth_channels: Number of fusion channels the sparse-depth encoder
        emits.
    :type depth_channels: int
    :param conv_hidden_channels: Hidden width of both small `Conv2D` stacks.
    :type conv_hidden_channels: int
    :param splat_kernel_size: Forwarded to `SparseDepthSplat`.
    :type splat_kernel_size: Tuple[int, int]
    :param splat_sigma: Forwarded to `SparseDepthSplat`.
    :type splat_sigma: float
    :param kernel_initializer: Initializer for every conv kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

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
        """Eagerly build every sublayer, independent of the runtime branch.

        `call()` conditionally skips `self.splat` when `sparse_depth` is
        `None` for a whole batch. Left to Keras 3's default auto-build, that
        branch would leave `self.splat` (and transitively its two
        `GaussianFilter` sublayers) unbuilt whenever a test/caller happens to
        never supply sparse depth, which Keras reports as "unbuilt state" on
        THIS layer once it is otherwise marked built. Building every
        sublayer here with a channel-count-only dummy shape (none of these
        convs need a real spatial resolution to build their kernel) avoids
        that regardless of which conditioning signals a given call supplies.

        :param input_shape: Shape of the ``image`` argument (the input this
            layer is invoked on first, positionally).
        :type input_shape: tuple
        """
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-015: build every
        # sublayer HERE, unconditionally -- `self.splat` is only CALLED when
        # sparse_depth is supplied, but its weights must still exist
        # regardless, or Keras 3 reports "unbuilt state" on this layer (an
        # `error::UserWarning`-promoted hard failure) whenever a test/caller
        # happens to never supply that modality. See decisions.md.
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
        """Multiply out samples whose per-sample flag marks them absent.

        :param tensor: A per-sample tensor, ``(B, ...)``.
        :type tensor: keras.KerasTensor
        :param present: Optional boolean/0-1 flag, ``(B,)`` or ``(B, 1)``.
            ``None`` means "all present" (no-op).
        :type present: Optional[keras.KerasTensor]
        :return: ``tensor``, unchanged where ``present`` is truthy and zeroed
            where it is not.
        :rtype: keras.KerasTensor
        """
        if present is None:
            return tensor
        present = keras.ops.cast(present, tensor.dtype)
        # Reshape (B,) or (B, 1) -> (B, 1, 1, 1) to broadcast against (B, H, W, C).
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
        """Fuse the RGB image with the (optional) conditioning signals.

        :param image: RGB image batch, ``(B, H, W, image_channels)``.
        :type image: keras.KerasTensor
        :param intrinsics_ray_map: Optional per-pixel unit ray map,
            ``(B, H, W, 3)``. ``None`` means absent for the whole batch.
        :type intrinsics_ray_map: Optional[keras.KerasTensor]
        :param intrinsics_present: Optional per-sample flag, ``(B,)``.
            Defaults to "all present" when ``intrinsics_ray_map`` is given and
            this is omitted.
        :type intrinsics_present: Optional[keras.KerasTensor]
        :param sparse_depth: Optional sparse depth values, ``(B, H, W, 1)``.
            ``None`` means absent for the whole batch.
        :type sparse_depth: Optional[keras.KerasTensor]
        :param sparse_depth_mask: Required companion validity mask when
            ``sparse_depth`` is given, ``(B, H, W, 1)``.
        :type sparse_depth_mask: Optional[keras.KerasTensor]
        :param sparse_depth_present: Optional per-sample flag, ``(B,)``.
            Defaults to "all present" when ``sparse_depth`` is given and this
            is omitted.
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
    """Post-encoder token-space fusion: two `StateIndicatorEmbedding` calls.

    Adds the intrinsics present/absent embedding, then the sparse-depth
    present/absent embedding, to the encoder's own output token sequence.
    See this module's docstring, "Mixed-batch / per-sample-flag contract".

    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        ``tokens``: 3D ``(batch_size, sequence_length, dim)``.

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
        """Add both present/absent embeddings to ``tokens``.

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

        :return: The base ``Layer`` config (this layer has no extra
            constructor arguments of its own).
        :rtype: dict
        """
        return super().get_config()

# ---------------------------------------------------------------------
