"""
SAM Mask Decoder Implementation in Keras 3
==========================================

This file provides a Keras 3 implementation of the mask decoder from the
Segment Anything Model (SAM). The mask decoder takes image and prompt
embeddings as input and predicts segmentation masks.

**Intent**: To create a robust, serializable mask decoder that faithfully
reproduces the SAM architecture while adhering to modern Keras 3 best
practices for composite layers and external module integration (FFN factory).

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
"""
import keras
from keras import layers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformer import TwoWayTransformer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaskDecoder(keras.layers.Layer):
    """
    Predicts masks from image and prompt embeddings using a transformer.
    """

    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: TwoWayTransformer,
            num_multimask_outputs: int = 3,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.iou_token = layers.Embedding(1, transformer_dim, name="iou_token")
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = layers.Embedding(self.num_mask_tokens, transformer_dim, name="mask_tokens")

        self.output_upscaling = keras.Sequential([
            layers.Conv2DTranspose(transformer_dim // 4, kernel_size=2, strides=2),
            layers.LayerNormalization(axis=-1),
            layers.Activation("gelu"),
            layers.Conv2DTranspose(transformer_dim // 8, kernel_size=2, strides=2),
            layers.Activation("gelu"),
        ], name="output_upscaling")

        self.output_hypernetworks_mlps = []
        for i in range(self.num_mask_tokens):
            # The original MLP has 3 layers. The FFN factory's 'mlp' is 2 layers.
            # We build a 3-layer MLP manually here.
            mlp = keras.Sequential([
                layers.Dense(transformer_dim),
                layers.ReLU(),
                layers.Dense(transformer_dim // 8)
            ], name=f"hypernetwork_mlp_{i}")
            self.output_hypernetworks_mlps.append(mlp)

        self.iou_prediction_head = keras.Sequential([
            layers.Dense(iou_head_hidden_dim),
            layers.ReLU(),
            layers.Dense(self.num_mask_tokens)
        ], name="iou_prediction_head")

    def build(self, input_shape):
        # We can pass dummy shapes to build the sub-layers if needed
        # In this case, Keras will handle it upon first call.
        super().build(input_shape)

    def call(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings):
        output_tokens = ops.concatenate([self.iou_token.weights[0], self.mask_tokens.weights[0]], axis=0)
        output_tokens = ops.expand_dims(output_tokens, 0)
        output_tokens = ops.broadcast_to(output_tokens, (ops.shape(sparse_prompt_embeddings)[0], -1, -1))
        tokens = ops.concatenate((output_tokens, sparse_prompt_embeddings), axis=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        B, H, W, C = ops.shape(src)

        hs, src_out = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src_out = ops.reshape(src_out, (B, H, W, C))
        upscaled_embedding = self.output_upscaling(src_out)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = ops.stack(hyper_in_list, axis=1)

        B, H_up, W_up, C_up = ops.shape(upscaled_embedding)
        upscaled_embedding_flat = ops.reshape(upscaled_embedding, (B, H_up * W_up, C_up))

        masks = hyper_in @ ops.transpose(upscaled_embedding_flat, (0, 2, 1))
        masks = ops.reshape(masks, (B, -1, H_up, W_up))

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            "transformer_dim": self.transformer_dim,
            "num_multimask_outputs": self.num_multimask_outputs,
            "iou_head_depth": self.iou_head_depth,
            "iou_head_hidden_dim": self.iou_head_hidden_dim,
        })
        # Note: self.transformer is passed in __init__, so Keras handles its serialization.
        return config

# ---------------------------------------------------------------------
