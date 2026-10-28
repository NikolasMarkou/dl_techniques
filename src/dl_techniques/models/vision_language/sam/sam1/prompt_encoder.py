"""
Turns points, boxes and masks into the embeddings the mask decoder takes,
built by :class:`PromptEncoder`.

It produces two kinds of output: sparse embeddings ``(B, N, D)`` for points
and boxes, and a dense embedding grid ``(B, H_emb, W_emb, D)`` for an input
mask. It also owns the image positional encoding -- :meth:`get_dense_pe`
supplies ``image_pe`` to the decoder, using the same random-Fourier
encoding (:class:`PositionEmbeddingRandom`) as the sparse prompts, so
prompts and image live in one coordinate frame.

A mask prompt's spatial size must be exactly ``4 * image_embedding_size`` in
both axes -- the downscaling stack is a fixed 4x reduction -- and any other
size raises. A padding point (label -1) has its positional encoding zeroed
before the padding-type embedding is added, matching reference SAM.

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
"""

import keras
import numpy as np
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import refuse_input_shape
from dl_techniques.utils.validation import positive_int

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam1.prompt_encoder")
class PositionEmbeddingRandom(keras.layers.Layer):
    """
    Random-Fourier-feature positional encoding: coordinates projected
    through a fixed random Gaussian matrix, then sin/cos encoded.

    Architecture:

    .. code-block:: text

        coords [..., 2], in [0, 1]
              │
              ▼
        scale to [-1, 1]
              │
              ▼
        @ positional_encoding_gaussian_matrix [2, num_pos_feats]
              │
              ▼
        * 2*pi
              │
              ▼
        [sin(.), cos(.)]
              │
              ▼
        out [..., 2*num_pos_feats]

    :param num_pos_feats: Number of positional features; the output
        dimension is ``2 * num_pos_feats``. Defaults to 64.
    :type num_pos_feats: int
    :param scale: Standard deviation of the random Gaussian projection
        matrix, controlling the encoding's frequency. Defaults to 1.0.
    :type scale: float
    :param kwargs: Additional arguments for the Layer base class.
    :ivar positional_encoding_gaussian_matrix: Non-trainable weight of shape
        ``(2, num_pos_feats)``.

    :raises ValueError: If ``input_shape`` or ``input_dim`` is passed.

    Input shape:
        - For `call()`: Tuple of two integers (height, width) representing the
          spatial dimensions.
        - For `forward_with_coords()`: Tensor of shape (..., 2) containing
          (x, y) coordinates.

    Output shape:
        - For `call()`: Tensor of shape (2*num_pos_feats, height, width).
        - For `forward_with_coords()`: Tensor of shape (..., 2*num_pos_feats).
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: float = 1.0,
        **kwargs: Any
    ) -> None:
        refuse_input_shape(
            kwargs, "PositionEmbeddingRandom", "its input is a (height, width) pair given to call()")
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        # Will be created in build()
        self.positional_encoding_gaussian_matrix = None

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Create the fixed random projection matrix.

        :param input_shape: Not used; may be None.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        """
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_encoding_gaussian_matrix",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def _pe_encoding(self, coords: keras.KerasTensor) -> keras.KerasTensor:
        """
        Encode normalized coordinates with the sinusoidal random-Fourier
        transform.

        :param coords: Coordinates in ``[0, 1]``, shape ``(..., 2)``.
        :type coords: keras.KerasTensor
        :return: Positional encoding, shape ``(..., 2*num_pos_feats)``.
        :rtype: keras.KerasTensor
        """
        # Scale coords to [-1, 1]
        coords = 2 * coords - 1
        # Project to random features
        coords = coords @ self.positional_encoding_gaussian_matrix
        # Scale by 2π for sinusoidal encoding
        coords = 2 * np.pi * coords
        # Apply sin and cos to get final encoding
        return keras.ops.concatenate([keras.ops.sin(coords), keras.ops.cos(coords)], axis=-1)

    def call(self, size: Tuple[int, int]) -> keras.KerasTensor:
        """
        Generate a grid of positional encodings for a given spatial size.

        :param size: Spatial dimensions ``(height, width)``.
        :type size: Tuple[int, int]
        :return: Positional encoding, shape ``(2*num_pos_feats, height, width)``.
        :rtype: keras.KerasTensor
        """
        h, w = size
        pe = self._pe_encoding(self._coord_grid(h, w))
        # (C, H, W), for compatibility with reference SAM.
        return keras.ops.transpose(pe, (2, 0, 1))

    def _coord_grid(self, h: int, w: int) -> keras.KerasTensor:
        """
        Return the normalized ``(h, w, 2)`` ``(x, y)`` coordinate grid.

        Recomputed on every call rather than cached; see the DECISION comment
        below.

        :param h: Grid height.
        :type h: int
        :param w: Grid width.
        :type w: int
        :return: Pixel-centre coordinates normalized to ``[0, 1]``, ``x``
            first, shape ``(h, w, 2)``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-025: do not memoize this
        # grid on the instance. A tensor built inside one trace's FuncGraph is
        # dead in every later call, raising "out of scope" under fit()/jit_compile. See decisions.md.
        grid = keras.ops.ones((h, w), dtype=self.compute_dtype)
        y_embed = keras.ops.cumsum(grid, axis=0) - 0.5
        x_embed = keras.ops.cumsum(grid, axis=1) - 0.5
        # Normalize to [0, 1]
        y_embed = y_embed / keras.ops.cast(h, dtype=self.compute_dtype)
        x_embed = x_embed / keras.ops.cast(w, dtype=self.compute_dtype)
        return keras.ops.stack([x_embed, y_embed], axis=-1)

    def forward_with_coords(
        self,
        coords_input: keras.KerasTensor,
        image_size: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Encode explicit pixel-space coordinates, e.g. point or box corners.

        :param coords_input: ``(x, y)`` coordinates in pixel space, shape
            ``(..., 2)``.
        :type coords_input: keras.KerasTensor
        :param image_size: Image size ``(height, width)`` for normalization.
        :type image_size: Tuple[int, int]
        :return: Positional encoding, shape ``(..., 2*num_pos_feats)``.
        :rtype: keras.KerasTensor
        """
        coords = keras.ops.copy(coords_input)
        # Normalize coordinates to [0, 1]
        coords_x = coords[..., 0] / keras.ops.cast(image_size[1], dtype=self.compute_dtype)
        coords_y = coords[..., 1] / keras.ops.cast(image_size[0], dtype=self.compute_dtype)
        coords = keras.ops.stack([coords_x, coords_y], axis=-1)
        return self._pe_encoding(keras.ops.cast(coords, dtype=self.compute_dtype))

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of :meth:`call`, which maps a
        ``(height, width)`` spatial size to a positional-encoding grid.

        :param input_shape: The ``(height, width)`` spatial size passed to
            :meth:`call`.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(2 * num_pos_feats, height, width)``.
        :rtype: Tuple[Optional[int], ...]
        """
        height, width = input_shape
        return (2 * self.num_pos_feats, height, width)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_pos_feats": self.num_pos_feats,
            "scale": self.scale,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.sam1.prompt_encoder")
class PromptEncoder(keras.layers.Layer):
    """
    Encode points, boxes and masks into the sparse and dense embeddings the
    mask decoder takes.

    Architecture:

    .. code-block:: text

        points + labels ──► pos. encoding + type embed ──┐
        boxes           ──► corner encoding + type embed ─┤──► sparse [B, N, D]
                                                            │
        mask (present)  ──► mask_downscaling ─────────────►│
        mask (absent)   ──► no_mask_embed, broadcast ──────► dense [B, H, W, D]

    :param embed_dim: Dimension of the output embeddings. Must be positive.
    :type embed_dim: int
    :param image_embedding_size: Spatial size ``(height, width)`` of the
        vision encoder's output, which sets the dense-embedding output size.
    :type image_embedding_size: Tuple[int, int]
    :param input_image_size: Size ``(height, width)`` of the original input
        image, used to normalize point/box coordinates.
    :type input_image_size: Tuple[int, int]
    :param mask_in_chans: Channel count of the mask-downscaling network's
        first layer. Defaults to 16.
    :type mask_in_chans: int
    :param normalization_type: Normalization used in mask downscaling.
        Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param activation: Activation used in mask downscaling: any name Keras
        resolves (``keras.activations.get``). Defaults to ``'gelu'``.
    :type activation: str
    :param kwargs: Additional arguments for the Layer base class.
    :ivar pe_layer: :class:`PositionEmbeddingRandom` positional encoder.
    :ivar point_embeddings: Four Embedding layers, one per point/box-corner
        type.
    :ivar not_a_point_embed: Embedding for padding points.
    :ivar no_mask_embed: Embedding used when no mask is provided.
    :ivar mask_downscaling: Sequential holding one two-convolution
        :class:`~dl_techniques.layers.conv_blocks.stem.ConvStem` named
        ``stem`` (2x2 stride-2 convolutions, normalization and activation after
        each) and the 1x1 ``conv3`` that projects to ``embed_dim``.

    :raises ValueError: If ``embed_dim`` or ``mask_in_chans`` is not a positive
        integer, if ``mask_in_chans`` is below 4, or if ``input_shape`` or
        ``input_dim`` is passed (the sizes are ``image_embedding_size`` and
        ``input_image_size``).

    Input shape (in call):
        - points: Optional tuple of (coords, labels) where:
            - coords: Tensor of shape (batch_size, num_points, 2) with (x, y) coordinates
            - labels: Tensor of shape (batch_size, num_points) with point labels
                (1=foreground, 0=background, -1=padding)
        - boxes: Optional tensor of shape (batch_size, num_boxes, 4) with
            (x1, y1, x2, y2) box coordinates
        - masks: Optional tensor of shape (batch_size, 1, mask_h, mask_w) with
            mask values

    Output shape:
        Tuple of two tensors:
        - sparse_embeddings: Shape (batch_size, num_sparse, embed_dim)
        - dense_embeddings: Shape (batch_size, image_embedding_size[0],
                                   image_embedding_size[1], embed_dim)

    Example:
        ```python
        # Create encoder
        encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16
        )

        # Encode points
        points = keras.ops.convert_to_tensor([[[100.0, 200.0], [300.0, 400.0]]])
        labels = keras.ops.convert_to_tensor([[1, 0]])
        sparse, dense = encoder(points=(points, labels))

        # Encode boxes
        boxes = keras.ops.convert_to_tensor([[[50.0, 50.0, 500.0, 500.0]]])
        sparse, dense = encoder(boxes=boxes)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int = 16,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'gelu',
        **kwargs: Any
    ) -> None:
        refuse_input_shape(
            kwargs, "PromptEncoder",
            "the image sizes are image_embedding_size and input_image_size")
        super().__init__(**kwargs)
        embed_dim = positive_int(embed_dim, "embed_dim", "PromptEncoder")
        mask_in_chans = positive_int(mask_in_chans, "mask_in_chans", "PromptEncoder")
        if mask_in_chans < 4:
            raise ValueError(
                "PromptEncoder: mask_in_chans must be at least 4 (the first mask convolution "
                f"has mask_in_chans // 4 filters), got {mask_in_chans}")
        # Store all configuration parameters
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans
        self.normalization_type = normalization_type
        self.activation = deserialize_activation(activation)

        # CREATE all sub-layers in __init__
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, name="pe_layer")

        # Point embeddings for different types:
        # 0: background point, 1: foreground point, 2: box top-left, 3: box bottom-right
        self.point_embeddings = [
            keras.layers.Embedding(1, embed_dim, name=f"point_embedding_{i}")
            for i in range(4)
        ]
        self.not_a_point_embed = keras.layers.Embedding(1, embed_dim, name="not_a_point_embed")
        self.no_mask_embed = keras.layers.Embedding(1, embed_dim, name="no_mask_embed")

        # Mask downscaling network: the two-convolution stem, then the 1x1 projection.
        # DECISION plan-2026-10-26T125959-7af851e5/D-053: the stem is handed Keras' own activation
        # FUNCTION, not the name. Do NOT pass `activation` as it came: ConvStem resolves a registry
        # key to the registry's layer, and for `sparsemax` that is a different function from
        # `keras.layers.Activation("sparsemax")` (max|delta| above 1 on a unit-scale input), while
        # every other Keras name is bit-equal either way. The original stack accepted any Keras
        # activation name; the function keeps all of them exactly.
        self.mask_downscaling = keras.Sequential(
            [
                ConvStem(
                    mask_in_chans // 4,
                    kernel_size=2,
                    strides=2,
                    padding="valid",
                    num_convs=2,
                    conv2_filters=mask_in_chans,
                    normalization_type=normalization_type,
                    activation_type=keras.activations.get(activation),
                    name="stem"
                ),
                keras.layers.Conv2D(
                    embed_dim,
                    kernel_size=1,
                    name="conv3"
                ),
            ],
            name="mask_downscaling"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Build every sub-layer explicitly so weights exist before
        deserialization restores them.

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        """
        self.pe_layer.build(None)

        for emb in self.point_embeddings:
            emb.build((None,))
        self.not_a_point_embed.build((None,))
        self.no_mask_embed.build((None,))

        # Masks have shape (batch, H, W, 1) after transpose.
        self.mask_downscaling.build((None, None, None, 1))

        super().build(input_shape)

    def get_dense_pe(self) -> keras.KerasTensor:
        """
        Get the dense positional encoding grid for the image embeddings.

        :return: Positional encoding, shape
            ``(1, image_embedding_size[0], image_embedding_size[1], embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # pe is (C, H, W); transpose to (H, W, C) then add the batch axis.
        pe = self.pe_layer(size=self.image_embedding_size)
        pe = keras.ops.transpose(pe, (1, 2, 0))
        return keras.ops.expand_dims(pe, axis=0)

    def _embed_points(
        self,
        points: keras.KerasTensor,
        labels: keras.KerasTensor,
        pad: bool
    ) -> keras.KerasTensor:
        """
        Embed point coordinates and labels.

        :param points: ``(x, y)`` coordinates, shape
            ``(batch_size, num_points, 2)``.
        :type points: keras.KerasTensor
        :param labels: Point labels, shape ``(batch_size, num_points)``.
        :type labels: keras.KerasTensor
        :param pad: Whether to append a padding point.
        :type pad: bool
        :return: Point embeddings, shape ``(batch_size, num_points, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        points = points + 0.5

        if pad:
            # Add a padding point for when no boxes are provided.
            padding_point = keras.ops.zeros((keras.ops.shape(points)[0], 1, 2), dtype=points.dtype)
            padding_label = -keras.ops.ones((keras.ops.shape(labels)[0], 1), dtype=labels.dtype)
            points = keras.ops.concatenate([points, padding_point], axis=1)
            labels = keras.ops.concatenate([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # DECISION plan-2026-08-03T191222-1d751f81/D-013: zero the positional
        # encoding of label == -1 rows before adding any type embedding, matching
        # reference SAM. Skipping this leaves a padding point's embedding coordinate-dependent. See decisions.md.
        point_embedding = keras.ops.where(
            keras.ops.expand_dims(labels, -1) == -1,
            keras.ops.zeros_like(point_embedding),
            point_embedding
        )

        # Add type embeddings based on labels using conditional operations
        # Label -1: not-a-point (padding)
        point_embedding = point_embedding + keras.ops.where(
            keras.ops.expand_dims(labels, -1) == -1,
            self.not_a_point_embed.weights[0],
            keras.ops.zeros_like(point_embedding)
        )
        # Label 0: background point
        point_embedding = point_embedding + keras.ops.where(
            keras.ops.expand_dims(labels, -1) == 0,
            self.point_embeddings[0].weights[0],
            keras.ops.zeros_like(point_embedding)
        )
        # Label 1: foreground point
        point_embedding = point_embedding + keras.ops.where(
            keras.ops.expand_dims(labels, -1) == 1,
            self.point_embeddings[1].weights[0],
            keras.ops.zeros_like(point_embedding)
        )
        return point_embedding

    def _embed_boxes(self, boxes: keras.KerasTensor) -> keras.KerasTensor:
        """
        Embed bounding box coordinates.

        :param boxes: ``(x1, y1, x2, y2)`` box coordinates, shape
            ``(batch_size, num_boxes, 4)``.
        :type boxes: keras.KerasTensor
        :return: Box embeddings, shape ``(batch_size, 2*num_boxes, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        boxes = boxes + 0.5
        # Reshape must preserve the batch axis: split each (x1,y1,x2,y2) box
        # into its two corners as (B, 2N, 2), order [tl0, br0, tl1, br1, ...],
        # not (B*N, 2, D) which crashes the axis-1 concat in call() for N>1.
        batch_size = keras.ops.shape(boxes)[0]
        num_boxes = keras.ops.shape(boxes)[1]
        coords = keras.ops.reshape(boxes, (batch_size, num_boxes * 2, 2))

        # Get positional encoding for corner coordinates -> (B, 2N, D)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)

        # Alternating top-left/bottom-right type embeddings: [emb2, emb3,
        # emb2, emb3, ...] of length 2N, broadcast over the batch.
        type_pair = keras.ops.concatenate([
            self.point_embeddings[2].weights[0],
            self.point_embeddings[3].weights[0],
        ], axis=0)
        type_embeddings = keras.ops.tile(type_pair, [num_boxes, 1])
        corner_embedding = corner_embedding + keras.ops.expand_dims(type_embeddings, 0)
        return corner_embedding

    def _embed_masks(self, masks: keras.KerasTensor) -> keras.KerasTensor:
        """
        Embed mask inputs through convolutional downscaling.

        :param masks: Mask values, shape ``(batch_size, 1, mask_h, mask_w)``.
        :type masks: keras.KerasTensor
        :return: Dense mask embeddings, shape
            ``(batch_size, image_embedding_size[0], image_embedding_size[1], embed_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the mask's spatial size is not exactly
            ``4 * image_embedding_size`` -- the downscaling stack is a fixed
            two-stride-2 convolution chain.
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-016: refuse a mask whose
        # spatial size isn't exactly 4 * image_embedding_size, at this point of
        # violation. Left unchecked it surfaces later as a broadcast error inside the mask decoder instead. See decisions.md.
        expected_h = 4 * self.image_embedding_size[0]
        expected_w = 4 * self.image_embedding_size[1]
        static_shape = tuple(masks.shape)
        if len(static_shape) == 4:
            mask_h, mask_w = static_shape[2], static_shape[3]
            if (
                (mask_h is not None and int(mask_h) != expected_h)
                or (mask_w is not None and int(mask_w) != expected_w)
            ):
                raise ValueError(
                    f"PromptEncoder mask prompt must be exactly "
                    f"4 * image_embedding_size = ({expected_h}, {expected_w}) "
                    f"in its spatial dimensions, because the mask-downscaling "
                    f"stack applies a fixed 4x reduction; got mask spatial size "
                    f"({mask_h}, {mask_w}) from shape {static_shape} with "
                    f"image_embedding_size={tuple(self.image_embedding_size)}. "
                    f"Resize the mask prompt to "
                    f"(batch, 1, {expected_h}, {expected_w}) before calling."
                )

        # Keras Conv2D expects channel-last format
        # Input is (B, 1, H, W), transpose to (B, H, W, 1)
        masks_transposed = keras.ops.transpose(masks, (0, 2, 3, 1))
        return self.mask_downscaling(masks_transposed)

    def call(
        self,
        points: Optional[Tuple[keras.KerasTensor, keras.KerasTensor]] = None,
        boxes: Optional[keras.KerasTensor] = None,
        masks: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Encode prompts into sparse and dense embeddings.

        :param points: Optional ``(coords, labels)`` pair, shapes
            ``(batch_size, num_points, 2)`` and ``(batch_size, num_points)``.
        :type points: Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param boxes: Optional boxes, shape ``(batch_size, num_boxes, 4)``.
        :type boxes: Optional[keras.KerasTensor]
        :param masks: Optional masks, shape ``(batch_size, 1, mask_h, mask_w)``.
        :type masks: Optional[keras.KerasTensor]
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(sparse_embeddings, dense_embeddings)`` -- sparse of shape
            ``(batch_size, num_sparse, embed_dim)``, dense of shape
            ``(batch_size, image_embedding_size[0], image_embedding_size[1], embed_dim)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = keras.ops.zeros((bs, 0, self.embed_dim), dtype=self.compute_dtype)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = keras.ops.concatenate([sparse_embeddings, point_embeddings], axis=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = keras.ops.concatenate([sparse_embeddings, box_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weights[0]
            dense_embeddings = keras.ops.reshape(dense_embeddings, (1, 1, 1, self.embed_dim))
            dense_embeddings = keras.ops.broadcast_to(
                dense_embeddings,
                (bs, self.image_embedding_size[0], self.image_embedding_size[1], self.embed_dim)
            )

        return sparse_embeddings, dense_embeddings

    def _get_batch_size(
        self,
        points: Optional[Tuple[keras.KerasTensor, keras.KerasTensor]],
        boxes: Optional[keras.KerasTensor],
        masks: Optional[keras.KerasTensor]
    ) -> int:
        """
        Determine batch size from whichever prompt input is provided.

        :param points: Optional point inputs.
        :type points: Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param boxes: Optional box inputs.
        :type boxes: Optional[keras.KerasTensor]
        :param masks: Optional mask inputs.
        :type masks: Optional[keras.KerasTensor]
        :return: Batch size, as an integer or tensor.
        :rtype: int
        """
        if points is not None:
            return keras.ops.shape(points[0])[0]
        elif boxes is not None:
            return keras.ops.shape(boxes)[0]
        elif masks is not None:
            return keras.ops.shape(masks)[0]
        else:
            return 1

    def compute_output_shape(
        self,
        input_shape: Optional[Tuple[Optional[int], ...]] = None
    ) -> Tuple[Tuple[Optional[int], Optional[int], int], Tuple[Optional[int], int, int, int]]:
        """
        Compute output shapes for sparse and dense embeddings.

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(sparse_shape, dense_shape)`` -- sparse
            ``(batch_size, num_sparse, embed_dim)``, dense
            ``(batch_size, H, W, embed_dim)``.
        :rtype: Tuple[tuple, tuple]
        """
        sparse_shape = (None, None, self.embed_dim)
        dense_shape = (
            None,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
            self.embed_dim
        )
        return sparse_shape, dense_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "image_embedding_size": self.image_embedding_size,
            "input_image_size": self.input_image_size,
            "mask_in_chans": self.mask_in_chans,
            "normalization_type": self.normalization_type,
            "activation": serialize_activation(self.activation),
        })
        return config

# ---------------------------------------------------------------------
