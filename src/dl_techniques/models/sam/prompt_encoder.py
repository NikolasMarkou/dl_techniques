"""
SAM Prompt Encoder
============================================

Implementation of the prompt encoder from the
Segment Anything Model (SAM). The prompt encoder is responsible for converting
various user inputs (points, boxes, masks) into high-dimensional embeddings that
can be consumed by the mask decoder.

**Intent**: To create a robust, serializable, and flexible prompt encoder that
can handle different combinations of prompts, following modern Keras 3 best
practices for composite layer construction.

**Architecture**: The prompt encoder processes three types of inputs:
1.  **Points and Boxes (Sparse Prompts)**: These are encoded using a learned
    positional encoding (`PositionEmbeddingRandom`) combined with learnable
    embeddings for different prompt types (e.g., foreground point, background
    point, top-left corner, bottom-right corner).
2.  **Masks (Dense Prompts)**: Input masks are processed through a small
    convolutional network to produce a dense embedding grid that matches the
    spatial dimensions of the image embedding.

If no prompt of a certain type is provided, a learned "not-a-prompt" embedding
is used as a placeholder.

**Data Flow**:
```
Points (B, N, 2) --+
Labels (B, N) ----> _embed_points() -> Sparse Embeddings (B, N, D) -----+
                                                                        │
Boxes (B, 1, 4) --> _embed_boxes() --> Sparse Embeddings (B, 2, D) -----+--> Concat
                                                                        │
(No points/boxes)--> not_a_point_embed --> Sparse Embeddings (B, 1, D) -+

Masks (B, 1, H, W) -> _embed_masks (CNN) -> Dense Embeddings (B, H_emb, W_emb, D)
(No mask) ---------> no_mask_embed ------> Dense Embeddings (B, H_emb, W_emb, D)
```

**Usage Example**:
```python
import keras
import numpy as np

# Create prompt encoder
prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(1024, 1024),
    mask_in_chans=16,
)

# Example: Encode points
points = keras.ops.convert_to_tensor(np.array([[[100.0, 200.0], [300.0, 400.0]]]))
labels = keras.ops.convert_to_tensor(np.array([[1, 0]]))  # 1=foreground, 0=background

sparse_emb, dense_emb = prompt_encoder(points=(points, labels))
print(f"Sparse embedding shape: {sparse_emb.shape}")  # (1, 2, 256)
print(f"Dense embedding shape: {dense_emb.shape}")     # (1, 64, 64, 256)

# Example: Encode boxes
boxes = keras.ops.convert_to_tensor(np.array([[[50.0, 50.0, 500.0, 500.0]]]))
sparse_emb, dense_emb = prompt_encoder(boxes=boxes)

# Example: Encode masks
masks = keras.random.normal(shape=(1, 1, 256, 256))
sparse_emb, dense_emb = prompt_encoder(masks=masks)
```

**References**:
- Kirillov, A., et al. (2023). Segment Anything. *arXiv*.
"""

import keras
import numpy as np
from keras import layers, ops, initializers
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PositionEmbeddingRandom(keras.layers.Layer):
    """
    Positional encoding using random spatial frequencies.

    This layer generates positional embeddings using a random Fourier feature
    approach. It maps 2D coordinates to high-dimensional positional encodings
    using random projection followed by sinusoidal encoding.

    **Intent**: To provide learnable positional information for point and box
    prompts, enabling the model to understand spatial relationships in the input.

    **Architecture**:
    ```
    Coordinates (normalized to [0, 1]) -> Scale to [-1, 1]
                                        -> Random projection
                                        -> Scale by 2π
                                        -> [sin(...), cos(...)]
                                        -> Positional encoding
    ```

    Args:
        num_pos_feats: Integer, number of positional features. The output
            dimension will be 2 * num_pos_feats (due to sin and cos).
            Defaults to 64.
        scale: Float, scale for the random Gaussian matrix initialization.
            Controls the frequency of the positional encoding. Defaults to 1.0.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        - For `call()`: Tuple of two integers (height, width) representing the
          spatial dimensions.
        - For `forward_with_coords()`: Tensor of shape (..., 2) containing
          (x, y) coordinates.

    Output shape:
        - For `call()`: Tensor of shape (2*num_pos_feats, height, width).
        - For `forward_with_coords()`: Tensor of shape (..., 2*num_pos_feats).

    Attributes:
        positional_encoding_gaussian_matrix: Non-trainable weight of shape
            (2, num_pos_feats) containing random projection matrix.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: float = 1.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        # Will be created in build()
        self.positional_encoding_gaussian_matrix = None

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Creates the random projection matrix.

        Args:
            input_shape: Optional shape tuple (not used, can be None).
        """
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_encoding_gaussian_matrix",
            shape=(2, self.num_pos_feats),
            initializer=initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def _pe_encoding(self, coords: keras.KerasTensor) -> keras.KerasTensor:
        """
        Encode coordinates using sinusoidal positional encoding.

        Args:
            coords: Tensor of shape (..., 2) with normalized coordinates in [0, 1].

        Returns:
            Positional encoding tensor of shape (..., 2*num_pos_feats).
        """
        # Scale coords to [-1, 1]
        coords = 2 * coords - 1
        # Project to random features
        coords = coords @ self.positional_encoding_gaussian_matrix
        # Scale by 2π for sinusoidal encoding
        coords = 2 * np.pi * coords
        # Apply sin and cos to get final encoding
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size: Tuple[int, int]) -> keras.KerasTensor:
        """
        Generate a grid of positional encodings for a given spatial size.

        Args:
            size: Tuple of (height, width) for the spatial dimensions.

        Returns:
            Positional encoding tensor of shape (2*num_pos_feats, height, width).
        """
        h, w = size
        # Create coordinate grid
        grid = ops.ones((h, w), dtype=self.compute_dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        # Normalize to [0, 1]
        y_embed = y_embed / ops.cast(h, dtype=self.compute_dtype)
        x_embed = x_embed / ops.cast(w, dtype=self.compute_dtype)
        # Stack and encode
        pe = self._pe_encoding(ops.stack([x_embed, y_embed], axis=-1))
        # Return as (C, H, W) for compatibility with original SAM
        return ops.transpose(pe, (2, 0, 1))

    def forward_with_coords(
        self,
        coords_input: keras.KerasTensor,
        image_size: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Encode explicit coordinates (e.g., point or box coordinates).

        Args:
            coords_input: Tensor of shape (..., 2) containing (x, y) coordinates
                in pixel space.
            image_size: Tuple of (height, width) of the image for normalization.

        Returns:
            Positional encoding tensor of shape (..., 2*num_pos_feats).
        """
        coords = ops.copy(coords_input)
        # Normalize coordinates to [0, 1]
        coords_x = coords[..., 0] / ops.cast(image_size[1], dtype=self.compute_dtype)
        coords_y = coords[..., 1] / ops.cast(image_size[0], dtype=self.compute_dtype)
        coords = ops.stack([coords_x, coords_y], axis=-1)
        return self._pe_encoding(ops.cast(coords, dtype=self.compute_dtype))

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "num_pos_feats": self.num_pos_feats,
            "scale": self.scale,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PromptEncoder(layers.Layer):
    """
    Encodes prompts (points, boxes, masks) for the SAM mask decoder.

    This layer is a core component of the Segment Anything Model (SAM), responsible
    for converting various types of user prompts into a format that can be used by
    the mask decoder. It handles three types of prompts:

    1. **Points**: Individual points with labels (foreground/background)
    2. **Boxes**: Bounding boxes defined by two corners
    3. **Masks**: Dense segmentation masks

    **Intent**: To provide a unified interface for encoding different prompt types,
    producing sparse embeddings for points/boxes and dense embeddings for masks.

    **Architecture**:
    ```
    Points + Labels → Positional Encoding + Type Embeddings → Sparse Embeddings
    Boxes → Corner Encoding + Type Embeddings → Sparse Embeddings
    Masks → Conv Network → Dense Embeddings
    No prompt → Learned "no prompt" embeddings
    ```

    Args:
        embed_dim: Integer, the dimension of the output embeddings. Must be positive.
        image_embedding_size: Tuple of two integers, the spatial size (height, width)
            of the image embeddings from the vision encoder. This determines the
            output spatial size for dense embeddings.
        input_image_size: Tuple of two integers, the size (height, width) of the
            original input image. Used for normalizing point/box coordinates.
        mask_in_chans: Integer, number of channels for mask downscaling network's
            first layer. Defaults to 16.
        normalization_type: String, type of normalization to use in mask downscaling.
            Supports 'layer_norm', 'rms_norm', 'batch_norm'. Defaults to 'layer_norm'.
        activation: String, activation function for mask downscaling. Defaults to 'gelu'.
        **kwargs: Additional arguments for the Layer base class.

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

    Attributes:
        pe_layer: PositionEmbeddingRandom layer for positional encoding.
        point_embeddings: List of 4 Embedding layers for different point types.
        not_a_point_embed: Embedding layer for padding points.
        no_mask_embed: Embedding layer for when no mask is provided.
        mask_downscaling: Sequential model for processing mask inputs.

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
        super().__init__(**kwargs)
        # Store all configuration parameters
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans
        self.normalization_type = normalization_type
        self.activation = activation

        # CREATE all sub-layers in __init__
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, name="pe_layer")

        # Point embeddings for different types:
        # 0: background point, 1: foreground point, 2: box top-left, 3: box bottom-right
        self.point_embeddings = [
            layers.Embedding(1, embed_dim, name=f"point_embedding_{i}")
            for i in range(4)
        ]
        self.not_a_point_embed = layers.Embedding(1, embed_dim, name="not_a_point_embed")
        self.no_mask_embed = layers.Embedding(1, embed_dim, name="no_mask_embed")

        # Mask downscaling network using factory for normalization
        self.mask_downscaling = keras.Sequential(
            [
                layers.Conv2D(
                    mask_in_chans // 4,
                    kernel_size=2,
                    strides=2,
                    name="conv1"
                ),
                create_normalization_layer(normalization_type, name="norm1"),
                layers.Activation(activation, name="act1"),
                layers.Conv2D(
                    mask_in_chans,
                    kernel_size=2,
                    strides=2,
                    name="conv2"
                ),
                create_normalization_layer(normalization_type, name="norm2"),
                layers.Activation(activation, name="act2"),
                layers.Conv2D(
                    embed_dim,
                    kernel_size=1,
                    name="conv3"
                ),
            ],
            name="mask_downscaling"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Builds all sub-layers.

        Following the "Create vs. Build" principle, we explicitly build all
        sub-layers to ensure their weights are created before the model attempts
        to load any saved weights during deserialization.

        Args:
            input_shape: Optional shape tuple (not used for this layer).
        """
        # Build positional encoding layer
        self.pe_layer.build(None)

        # Build all embedding layers
        for emb in self.point_embeddings:
            emb.build((None,))
        self.not_a_point_embed.build((None,))
        self.no_mask_embed.build((None,))

        # Build mask downscaling network
        # We know masks will have shape (batch, H, W, 1) after transpose
        self.mask_downscaling.build((None, None, None, 1))

        super().build(input_shape)

    def get_dense_pe(self) -> keras.KerasTensor:
        """
        Get dense positional encoding grid.

        Returns:
            Positional encoding tensor of shape
            (1, image_embedding_size[0], image_embedding_size[1], embed_dim).
        """
        pe = self.pe_layer(size=self.image_embedding_size)  # (C, H, W)
        pe = ops.transpose(pe, (1, 2, 0))  # (H, W, C)
        return ops.expand_dims(pe, axis=0)  # (1, H, W, C)

    def _embed_points(
        self,
        points: keras.KerasTensor,
        labels: keras.KerasTensor,
        pad: bool
    ) -> keras.KerasTensor:
        """
        Embed point coordinates and labels.

        Args:
            points: Tensor of shape (batch_size, num_points, 2) with (x, y) coordinates.
            labels: Tensor of shape (batch_size, num_points) with point labels.
            pad: Boolean, whether to add a padding point.

        Returns:
            Point embeddings of shape (batch_size, num_points, embed_dim).
        """
        # Add 0.5 for pixel center offset
        points = points + 0.5

        if pad:
            # Add padding point when no boxes are provided
            padding_point = ops.zeros((ops.shape(points)[0], 1, 2), dtype=points.dtype)
            padding_label = -ops.ones((ops.shape(labels)[0], 1), dtype=labels.dtype)
            points = ops.concatenate([points, padding_point], axis=1)
            labels = ops.concatenate([labels, padding_label], axis=1)

        # Get positional encoding for coordinates
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Add type embeddings based on labels using conditional operations
        # Label -1: not-a-point (padding)
        point_embedding = point_embedding + ops.where(
            ops.expand_dims(labels, -1) == -1,
            self.not_a_point_embed.weights[0],
            ops.zeros_like(point_embedding)
        )
        # Label 0: background point
        point_embedding = point_embedding + ops.where(
            ops.expand_dims(labels, -1) == 0,
            self.point_embeddings[0].weights[0],
            ops.zeros_like(point_embedding)
        )
        # Label 1: foreground point
        point_embedding = point_embedding + ops.where(
            ops.expand_dims(labels, -1) == 1,
            self.point_embeddings[1].weights[0],
            ops.zeros_like(point_embedding)
        )
        return point_embedding

    def _embed_boxes(self, boxes: keras.KerasTensor) -> keras.KerasTensor:
        """
        Embed bounding box coordinates.

        Args:
            boxes: Tensor of shape (batch_size, num_boxes, 4) with
                (x1, y1, x2, y2) coordinates.

        Returns:
            Box embeddings of shape (batch_size, 2*num_boxes, embed_dim).
        """
        # Add 0.5 for pixel center offset
        boxes = boxes + 0.5
        # Reshape to (batch_size, num_boxes*2, 2) for corner coordinates
        coords = ops.reshape(boxes, (-1, 2, 2))

        # Get positional encoding for corner coordinates
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)

        # Add type embeddings: embedding[2] for top-left, embedding[3] for bottom-right
        corner_embedding = ops.concatenate([
            corner_embedding[:, 0:1, :] + self.point_embeddings[2].weights[0],
            corner_embedding[:, 1:2, :] + self.point_embeddings[3].weights[0]
        ], axis=1)
        return corner_embedding

    def _embed_masks(self, masks: keras.KerasTensor) -> keras.KerasTensor:
        """
        Embed mask inputs through convolutional downscaling.

        Args:
            masks: Tensor of shape (batch_size, 1, mask_h, mask_w) with mask values.

        Returns:
            Dense mask embeddings of shape
            (batch_size, image_embedding_size[0], image_embedding_size[1], embed_dim).
        """
        # Keras Conv2D expects channel-last format
        # Input is (B, 1, H, W), transpose to (B, H, W, 1)
        masks_transposed = ops.transpose(masks, (0, 2, 3, 1))
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

        Args:
            points: Optional tuple of (coords, labels) where:
                - coords: Shape (batch_size, num_points, 2)
                - labels: Shape (batch_size, num_points)
            boxes: Optional tensor of shape (batch_size, num_boxes, 4).
            masks: Optional tensor of shape (batch_size, 1, mask_h, mask_w).
            training: Optional boolean for training mode.

        Returns:
            Tuple of (sparse_embeddings, dense_embeddings):
            - sparse_embeddings: Shape (batch_size, num_sparse, embed_dim)
            - dense_embeddings: Shape (batch_size, image_embedding_size[0],
                                       image_embedding_size[1], embed_dim)
        """
        # Determine batch size from inputs
        bs = self._get_batch_size(points, boxes, masks)

        # Initialize empty sparse embeddings
        sparse_embeddings = ops.zeros((bs, 0, self.embed_dim), dtype=self.compute_dtype)

        # Encode points if provided
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = ops.concatenate([sparse_embeddings, point_embeddings], axis=1)

        # Encode boxes if provided
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = ops.concatenate([sparse_embeddings, box_embeddings], axis=1)

        # Encode masks or use "no mask" embedding
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            # Use learned "no mask" embedding
            dense_embeddings = self.no_mask_embed.weights[0]
            dense_embeddings = ops.reshape(dense_embeddings, (1, 1, 1, self.embed_dim))
            dense_embeddings = ops.broadcast_to(
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
        Determine batch size from provided inputs.

        Args:
            points: Optional point inputs.
            boxes: Optional box inputs.
            masks: Optional mask inputs.

        Returns:
            Batch size as an integer or tensor.
        """
        if points is not None:
            return ops.shape(points[0])[0]
        elif boxes is not None:
            return ops.shape(boxes)[0]
        elif masks is not None:
            return ops.shape(masks)[0]
        else:
            return 1

    def compute_output_shape(
        self,
        input_shape: Optional[Tuple[Optional[int], ...]] = None
    ) -> Tuple[Tuple[Optional[int], Optional[int], int], Tuple[Optional[int], int, int, int]]:
        """
        Compute output shapes for sparse and dense embeddings.

        Args:
            input_shape: Not used for this layer.

        Returns:
            Tuple of (sparse_shape, dense_shape):
            - sparse_shape: (batch_size, num_sparse, embed_dim)
            - dense_shape: (batch_size, H, W, embed_dim)
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
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "image_embedding_size": self.image_embedding_size,
            "input_image_size": self.input_image_size,
            "mask_in_chans": self.mask_in_chans,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
        })
        return config

# ---------------------------------------------------------------------
