"""
SAM Prompt Encoder Implementation in Keras 3
============================================

This file provides a Keras 3 implementation of the prompt encoder from the
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
Labels (B, N) ----> _embed_points() -> Sparse Embeddings (B, N, D) --+
                                                                      |
Boxes (B, 1, 4) --> _embed_boxes() --> Sparse Embeddings (B, 2, D) ---+--> Concat
                                                                      |
(No points/boxes)--> not_a_point_embed --> Sparse Embeddings (B, 1, D) -+

Masks (B, 1, H, W) -> _embed_masks (CNN) -> Dense Embeddings (B, D, H_emb, W_emb)
(No mask) ---------> no_mask_embed ------> Dense Embeddings (B, D, H_emb, W_emb)
```
"""
import keras
import numpy as np
from keras import layers, ops
from typing import Optional, Tuple, Type, List, Any, Dict, Union

@keras.saving.register_keras_serializable()
class PositionEmbeddingRandom(keras.layers.Layer):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.positional_encoding_gaussian_matrix = None  # Created in build

    def build(self, input_shape=None):
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_encoding_gaussian_matrix",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def _pe_encoding(self, coords: keras.KerasTensor) -> keras.KerasTensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size: Tuple[int, int]) -> keras.KerasTensor:
        h, w = size
        grid = ops.ones((h, w), dtype=self.compute_dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(ops.stack([x_embed, y_embed], axis=-1))
        return ops.transpose(pe, (2, 0, 1))

    def forward_with_coords(self, coords_input: keras.KerasTensor, image_size: Tuple[int, int]) -> keras.KerasTensor:
        coords = ops.copy(coords_input)
        coords_x = coords[..., 0] / image_size[1]
        coords_y = coords[..., 1] / image_size[0]
        coords = ops.stack([coords_x, coords_y], axis=-1)
        return self._pe_encoding(ops.cast(coords, dtype=self.compute_dtype))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_pos_feats": self.num_pos_feats, "scale": self.scale})
        return config


@keras.saving.register_keras_serializable()
class PromptEncoder(layers.Layer):
    """
    Encodes prompts (points, boxes, masks) for the SAM mask decoder.
    """

    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            input_image_size: Tuple[int, int],
            mask_in_chans: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, name="pe_layer")

        self.point_embeddings = [
            layers.Embedding(1, embed_dim, name=f"point_embedding_{i}") for i in range(4)
        ]
        self.not_a_point_embed = layers.Embedding(1, embed_dim, name="not_a_point_embed")
        self.no_mask_embed = layers.Embedding(1, embed_dim, name="no_mask_embed")

        self.mask_downscaling = keras.Sequential(
            [
                layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2),
                layers.LayerNormalization(axis=-1),
                layers.Activation("gelu"),
                layers.Conv2D(mask_in_chans, kernel_size=2, strides=2),
                layers.LayerNormalization(axis=-1),
                layers.Activation("gelu"),
                layers.Conv2D(embed_dim, kernel_size=1),
            ], name="mask_downscaling"
        )

    def get_dense_pe(self) -> keras.KerasTensor:
        pe = self.pe_layer(self.image_embedding_size)  # C x H x W
        pe = ops.transpose(pe, (1, 2, 0))  # H x W x C
        return ops.expand_dims(pe, axis=0)

    def _embed_points(self, points: keras.KerasTensor, labels: keras.KerasTensor, pad: bool) -> keras.KerasTensor:
        points = points + 0.5
        if pad:
            padding_point = ops.zeros((ops.shape(points)[0], 1, 2), dtype=points.dtype)
            padding_label = -ops.ones((ops.shape(labels)[0], 1), dtype=labels.dtype)
            points = ops.concatenate([points, padding_point], axis=1)
            labels = ops.concatenate([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Use ops.where for conditional embedding addition
        point_embedding += ops.where(
            ops.expand_dims(labels, -1) == -1,
            self.not_a_point_embed.weights[0],
            ops.zeros_like(point_embedding)
        )
        point_embedding += ops.where(
            ops.expand_dims(labels, -1) == 0,
            self.point_embeddings[0].weights[0],
            ops.zeros_like(point_embedding)
        )
        point_embedding += ops.where(
            ops.expand_dims(labels, -1) == 1,
            self.point_embeddings[1].weights[0],
            ops.zeros_like(point_embedding)
        )
        return point_embedding

    def _embed_boxes(self, boxes: keras.KerasTensor) -> keras.KerasTensor:
        boxes = boxes + 0.5
        coords = ops.reshape(boxes, (-1, 2, 2))
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding = ops.concatenate([
            corner_embedding[:, 0:1, :] + self.point_embeddings[2].weights[0],
            corner_embedding[:, 1:2, :] + self.point_embeddings[3].weights[0]
        ], axis=1)
        return corner_embedding

    def _embed_masks(self, masks: keras.KerasTensor) -> keras.KerasTensor:
        # Keras Conv2D expects channel-last, so we need to transpose
        # Input is Bx1xHxW, needs to be BxHxWx1
        masks_transposed = ops.transpose(masks, (0, 2, 3, 1))
        return self.mask_downscaling(masks_transposed)

    def call(self, points=None, boxes=None, masks=None) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = ops.zeros((bs, 0, self.embed_dim), dtype=self.compute_dtype)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = ops.concatenate([sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = ops.concatenate([sparse_embeddings, box_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weights[0]
            dense_embeddings = ops.reshape(dense_embeddings, (1, 1, 1, self.embed_dim))
            dense_embeddings = ops.broadcast_to(
                dense_embeddings,
                (bs, self.image_embedding_size[0], self.image_embedding_size[1], self.embed_dim)
            )

        return sparse_embeddings, dense_embeddings

    def _get_batch_size(self, points, boxes, masks):
        if points is not None:
            return ops.shape(points[0])[0]
        elif boxes is not None:
            return ops.shape(boxes)[0]
        elif masks is not None:
            return ops.shape(masks)[0]
        else:
            return 1

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "image_embedding_size": self.image_embedding_size,
            "input_image_size": self.input_image_size,
            "mask_in_chans": self.mask_in_chans,
        })
        return config