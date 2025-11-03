"""
SAM Mask Decoder Implementation in Keras 3
==========================================

This file provides a Keras 3 implementation of the mask decoder from the
Segment Anything Model (SAM). The mask decoder takes image and prompt
embeddings as input and predicts segmentation masks.

**Intent**: To create a robust, serializable mask decoder that faithfully
reproduces the SAM architecture while adhering to modern Keras 3 best
practices for composite layers and factory pattern integration.

**Architecture**: The mask decoder is a transformer-based architecture with
several key components:
1.  **Output Tokens**: Learnable embeddings for an IoU prediction token and
    multiple mask tokens. These are concatenated with the sparse prompt
    embeddings to form the initial query for the transformer.
2.  **Two-Way Transformer**: The core of the decoder, which bidirectionally
    updates the query tokens and image embeddings.
3.  **Upscaling Module**: A series of transposed convolutions that upsample
    the final image embeddings to a higher resolution.
4.  **Hypernetwork MLPs**: A set of small MLPs (one for each mask token) that
    transform the final mask token embeddings into parameters for the final
    mask prediction layer.
5.  **Mask Prediction**: The upscaled image embeddings are multiplied by the
    hypernetwork outputs to produce the final low-resolution mask logits.
6.  **IoU Prediction Head**: An MLP that predicts the quality (IoU) of each
    predicted mask from the final IoU token embedding.

**Data Flow**:
```
Image Embeddings (B, H, W, C) ─────┐
Dense Prompt Embeddings (B, H, W, C) ─> Add ─> Source
                                              │
IoU Token (1, C) ──┐                          │
Mask Tokens (N, C) ─┴─> Output Tokens         │
Sparse Prompts (B, M, C) ─> Concat ─> Tokens  │
                                              │
                                              v
                                     Two-Way Transformer
                                              │
                     ┌────────────────────────┴────────────────────┐
                     v                                             v
              Updated Tokens                              Updated Source
              (B, N+M, C)                                  (B, H, W, C)
                     │                                             │
         ┌───────────┴──────────┐                                  v
         v                      v                          Upscale (4x)
    IoU Token           Mask Tokens                                │
         │              (B, N, C)                                  v
         v                      │                          (B, H*4, W*4, C/8)
    IoU Head                    v                                 │
         │              Hypernetwork MLPs                         │
         v              (B, N, C/8)                               │
    IoU Predictions            │                                  │
    (B, N)                     └──────────> Matrix Multiply <─────┘
                                                   |
                                                   v
                                            Mask Logits
                                            (B, N, H*4, W*4)
```

**Usage Example**:
```python
import keras
from .transformer import TwoWayTransformer

# Create transformer
transformer = TwoWayTransformer(
    depth=2,
    embedding_dim=256,
    num_heads=8,
    mlp_dim=2048
)

# Create decoder
decoder = MaskDecoder(
    transformer_dim=256,
    transformer=transformer,
    num_multimask_outputs=3,
    iou_head_hidden_dim=256,
)

# Use decoder
image_embeddings = keras.random.normal(shape=(1, 64, 64, 256))
image_pe = keras.random.normal(shape=(1, 64, 64, 256))
sparse_prompts = keras.random.normal(shape=(1, 2, 256))
dense_prompts = keras.random.normal(shape=(1, 64, 64, 256))

masks, iou_pred = decoder(
    image_embeddings=image_embeddings,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_prompts,
    dense_prompt_embeddings=dense_prompts,
    multimask_output=True
)

print(f"Masks shape: {masks.shape}")      # (1, 3, 256, 256)
print(f"IoU pred shape: {iou_pred.shape}") # (1, 3)
```

**References**:
- Kirillov, A., et al. (2023). Segment Anything. *arXiv*.
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformer import TwoWayTransformer
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaskDecoder(keras.layers.Layer):
    """
    Predicts segmentation masks from image and prompt embeddings using a transformer.

    This layer is the final component of the Segment Anything Model (SAM), responsible
    for generating mask predictions from encoded image features and user prompts. It
    uses a two-way transformer to jointly refine prompt and image embeddings, followed
    by upscaling and dynamic mask prediction.

    **Intent**: To provide a flexible, high-quality mask decoder that can generate
    multiple mask proposals with quality estimates, supporting both single-mask and
    multi-mask prediction modes.

    **Architecture**:
    The decoder processes inputs through several stages:
    1. Combines image embeddings with dense prompt embeddings
    2. Prepends learnable output tokens (IoU + mask tokens) to sparse prompts
    3. Runs two-way transformer to update both tokens and image features
    4. Upscales image features 4x using transposed convolutions
    5. Uses hypernetwork MLPs to generate mask-specific parameters
    6. Produces final masks via dynamic convolution (matrix multiplication)
    7. Predicts mask quality (IoU) from IoU token

    Args:
        transformer_dim: Integer, the embedding dimension used by the transformer.
            Must match the dimension of input embeddings.
        transformer: TwoWayTransformer instance, the core transformer for joint
            refinement of prompts and image features.
        num_multimask_outputs: Integer, number of mask predictions to generate
            beyond the single output mask. Defaults to 3. Total masks =
            num_multimask_outputs + 1.
        iou_head_depth: Integer, number of layers in the IoU prediction head.
            Currently not used (kept for compatibility). Defaults to 3.
        iou_head_hidden_dim: Integer, hidden dimension of the IoU prediction head.
            Defaults to 256.
        normalization_type: String, type of normalization to use in upscaling module.
            Supports 'layer_norm', 'rms_norm', 'batch_norm'. Defaults to 'layer_norm'.
        activation: String, activation function to use in upscaling and MLPs.
            Defaults to 'gelu'.
        **kwargs: Additional arguments for the Layer base class.

    Input shape (in call):
        - image_embeddings: Shape (batch_size, H, W, transformer_dim)
        - image_pe: Shape (batch_size, H, W, transformer_dim), positional encoding
        - sparse_prompt_embeddings: Shape (batch_size, num_sparse, transformer_dim)
        - dense_prompt_embeddings: Shape (batch_size, H, W, transformer_dim)
        - multimask_output: Boolean, whether to return multiple masks or single mask

    Output shape:
        Tuple of two tensors:
        - masks: Shape (batch_size, num_masks, H*4, W*4) where num_masks is either
            num_multimask_outputs (if multimask_output=True) or 1 (if False)
        - iou_predictions: Shape (batch_size, num_masks) with predicted IoU scores

    Attributes:
        iou_token: Embedding layer for the IoU prediction token.
        mask_tokens: Embedding layer for mask prediction tokens.
        output_upscaling: Sequential model for 4x upsampling of image features.
        output_hypernetworks_mlps: List of MLP heads, one per mask token.
        iou_prediction_head: MLP for predicting mask quality (IoU).

    Example:
        ```python
        # Create decoder
        from .transformer import TwoWayTransformer

        transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)
        decoder = MaskDecoder(transformer_dim=256, transformer=transformer)

        # Generate masks
        masks, iou_pred = decoder(
            image_embeddings=image_emb,
            image_pe=pos_encoding,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=True
        )
        ```

    Note:
        The transformer is passed as a parameter to allow flexible transformer
        architectures while maintaining the decoder's structure. The transformer
        must implement a compatible interface returning (tokens, source) tuple.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'gelu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if transformer_dim <= 0:
            raise ValueError(f"transformer_dim must be positive, got {transformer_dim}")
        if num_multimask_outputs <= 0:
            raise ValueError(f"num_multimask_outputs must be positive, got {num_multimask_outputs}")
        if iou_head_hidden_dim <= 0:
            raise ValueError(f"iou_head_hidden_dim must be positive, got {iou_head_hidden_dim}")

        # Store all configuration parameters
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.normalization_type = normalization_type
        self.activation = activation

        # Calculate number of mask tokens (multi-mask outputs + 1 single output)
        self.num_mask_tokens = num_multimask_outputs + 1

        # CREATE all sub-layers in __init__

        # Learnable output tokens
        self.iou_token = layers.Embedding(1, transformer_dim, name="iou_token")
        self.mask_tokens = layers.Embedding(
            self.num_mask_tokens,
            transformer_dim,
            name="mask_tokens"
        )

        # Output upscaling network: 4x upsampling (2x -> 2x)
        # Input: (B, H, W, transformer_dim) -> Output: (B, H*4, W*4, transformer_dim//8)
        self.output_upscaling = keras.Sequential([
            keras.layers.Conv2DTranspose(
                transformer_dim // 4,
                kernel_size=2,
                strides=2,
                name="upsample_conv1"
            ),
            create_normalization_layer(normalization_type, name="upsample_norm1"),
            keras.layers.Activation(activation, name="upsample_act1"),
            keras.layers.Conv2DTranspose(
                transformer_dim // 8,
                kernel_size=2,
                strides=2,
                name="upsample_conv2"
            ),
            keras.layers.Activation(activation, name="upsample_act2"),
        ], name="output_upscaling")

        # Hypernetwork MLPs: one for each mask token
        # Each MLP transforms mask token embedding -> mask prediction parameters
        # Architecture: transformer_dim -> transformer_dim -> transformer_dim//8
        self.output_hypernetworks_mlps = []
        for i in range(self.num_mask_tokens):
            mlp = keras.Sequential([
                keras.layers.Dense(
                    transformer_dim,
                    activation=activation,
                    name=f"hyper_dense1_{i}"
                ),
                keras.layers.Dense(
                    transformer_dim // 8,
                    name=f"hyper_dense2_{i}"
                )
            ], name=f"hypernetwork_mlp_{i}")
            self.output_hypernetworks_mlps.append(mlp)

        # IoU prediction head
        # Predicts mask quality score for each mask token
        self.iou_prediction_head = keras.Sequential([
            keras.layers.Dense(
                self.iou_head_hidden_dim,
                activation=activation,
                name="iou_dense1"
            ),
            keras.layers.Dense(
                self.num_mask_tokens,
                name="iou_dense2"
            )
        ], name="iou_prediction_head")

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Builds all sub-layers.

        Following the "Create vs. Build" principle, we explicitly build all
        sub-layers to ensure their weights are created before the model attempts
        to load any saved weights during deserialization.

        Args:
            input_shape: Optional shape tuple (not used for this layer).
        """
        # Build embedding layers
        self.iou_token.build((None,))
        self.mask_tokens.build((None,))

        # Build transformer (if it has a build method)
        if hasattr(self.transformer, 'build') and callable(self.transformer.build):
            # Transformer will be built on first call with actual shapes
            pass

        # Build upscaling network
        # Input shape: (batch, H, W, transformer_dim)
        self.output_upscaling.build((None, None, None, self.transformer_dim))

        # Build hypernetwork MLPs
        # Input shape: (batch, transformer_dim)
        for mlp in self.output_hypernetworks_mlps:
            mlp.build((None, self.transformer_dim))

        # Build IoU prediction head
        # Input shape: (batch, transformer_dim)
        self.iou_prediction_head.build((None, self.transformer_dim))

        super().build(input_shape)

    def call(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        multimask_output: bool,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass for mask prediction.

        Args:
            image_embeddings: Image features from encoder, shape
                (batch_size, H, W, transformer_dim).
            image_pe: Positional encoding for image features, shape
                (batch_size, H, W, transformer_dim).
            sparse_prompt_embeddings: Encoded sparse prompts (points/boxes), shape
                (batch_size, num_sparse, transformer_dim).
            dense_prompt_embeddings: Encoded dense prompts (masks), shape
                (batch_size, H, W, transformer_dim).
            multimask_output: Boolean, if True returns num_multimask_outputs masks,
                if False returns single best mask.
            training: Optional boolean for training mode.

        Returns:
            Tuple of (masks, iou_predictions):
            - masks: Shape (batch_size, num_masks, H*4, W*4)
            - iou_predictions: Shape (batch_size, num_masks)
        """
        # Predict all masks
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            training=training
        )

        # Select output masks based on mode
        if multimask_output:
            # Return multiple mask predictions (skip the single-mask output)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            # Return only the single best mask
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Generate mask predictions and IoU estimates.

        This is the core prediction logic that:
        1. Prepares output tokens (IoU + mask tokens)
        2. Runs two-way transformer to refine tokens and image features
        3. Upscales image features
        4. Generates masks via hypernetwork dynamic convolution
        5. Predicts IoU scores

        Args:
            image_embeddings: Image features, shape (batch_size, H, W, transformer_dim).
            image_pe: Positional encoding, shape (batch_size, H, W, transformer_dim).
            sparse_prompt_embeddings: Sparse prompts, shape (batch_size, num_sparse, transformer_dim).
            dense_prompt_embeddings: Dense prompts, shape (batch_size, H, W, transformer_dim).
            training: Optional boolean for training mode.

        Returns:
            Tuple of (masks, iou_predictions):
            - masks: All mask logits, shape (batch_size, num_mask_tokens, H*4, W*4)
            - iou_predictions: Quality scores, shape (batch_size, num_mask_tokens)
        """
        # Concatenate IoU token and mask tokens: shape (num_mask_tokens + 1, transformer_dim)
        output_tokens = ops.concatenate(
            [self.iou_token.weights[0], self.mask_tokens.weights[0]],
            axis=0
        )
        # Expand and broadcast to batch size: (batch_size, num_mask_tokens + 1, transformer_dim)
        output_tokens = ops.expand_dims(output_tokens, 0)
        batch_size = ops.shape(sparse_prompt_embeddings)[0]
        output_tokens = ops.broadcast_to(
            output_tokens,
            (batch_size, ops.shape(output_tokens)[1], ops.shape(output_tokens)[2])
        )

        # Concatenate output tokens with sparse prompt embeddings
        # Shape: (batch_size, num_mask_tokens + 1 + num_sparse, transformer_dim)
        tokens = ops.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)

        # Prepare source (image) input for transformer
        # Add dense prompt embeddings to image embeddings
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        B, H, W, C = ops.shape(src)

        # Run two-way transformer
        # Returns refined tokens and refined image features
        hs, src_out = self.transformer(src, pos_src, tokens, training=training)

        # Extract specific tokens
        iou_token_out = hs[:, 0, :]  # IoU token: (batch_size, transformer_dim)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # Mask tokens: (batch_size, num_mask_tokens, transformer_dim)

        # Reshape and upscale image features
        src_out = ops.reshape(src_out, (B, H, W, C))
        upscaled_embedding = self.output_upscaling(src_out, training=training)  # (B, H*4, W*4, C//8)

        # Generate mask-specific parameters using hypernetwork MLPs
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :], training=training)
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)  # (batch_size, num_mask_tokens, C//8)

        # Generate masks via dynamic convolution (matrix multiplication)
        # Flatten upscaled embeddings spatially
        B, H_up, W_up, C_up = ops.shape(upscaled_embedding)
        upscaled_embedding_flat = ops.reshape(
            upscaled_embedding,
            (B, H_up * W_up, C_up)
        )  # (batch_size, H*W, C//8)

        # Matrix multiply: (batch_size, num_mask_tokens, C//8) @ (batch_size, C//8, H*W)
        #                -> (batch_size, num_mask_tokens, H*W)
        masks = hyper_in @ ops.transpose(upscaled_embedding_flat, (0, 2, 1))
        masks = ops.reshape(masks, (B, self.num_mask_tokens, H_up, W_up))

        # Predict IoU scores for each mask
        iou_pred = self.iou_prediction_head(iou_token_out, training=training)

        return masks, iou_pred

    def compute_output_shape(
        self,
        input_shape: Optional[Tuple[Optional[int], ...]] = None
    ) -> Tuple[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
               Tuple[Optional[int], Optional[int]]]:
        """
        Compute output shapes for masks and IoU predictions.

        Args:
            input_shape: Not used for this layer.

        Returns:
            Tuple of (mask_shape, iou_shape):
            - mask_shape: (batch_size, num_mask_tokens, H*4, W*4)
            - iou_shape: (batch_size, num_mask_tokens)
        """
        mask_shape = (None, self.num_mask_tokens, None, None)
        iou_shape = (None, self.num_mask_tokens)
        return mask_shape, iou_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Note: The transformer is handled separately by Keras serialization
        since it's a layer passed as a parameter.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "transformer_dim": self.transformer_dim,
            "num_multimask_outputs": self.num_multimask_outputs,
            "iou_head_depth": self.iou_head_depth,
            "iou_head_hidden_dim": self.iou_head_hidden_dim,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
            "transformer": keras.layers.serialize(self.transformer),
        })
        # Note: self.transformer is passed in __init__ as a layer,
        # so Keras automatically handles its serialization
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskDecoder":
        """Creates a MaskDecoder from its config."""
        config["transformer"] = keras.layers.deserialize(config.pop("transformer"))
        return cls(**config)