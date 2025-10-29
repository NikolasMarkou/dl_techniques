"""
Segment Anything Model (SAM) Keras 3 Implementation
===================================================

This file provides the main `Sam` model class, which integrates the image
encoder, prompt encoder, and mask decoder into a single, end-to-end Keras model.
It follows the structure of modern, variant-based models like ConvNeXt, offering
a `from_variant` class method to easily instantiate different model sizes
(e.g., `vit_b`, `vit_l`, `vit_h`).

**Intent**: To provide a user-friendly, high-level interface for the SAM model
that is fully serializable and adheres to modern Keras 3 best practices. This
class handles preprocessing, postprocessing, and the orchestration of the three
main sub-components.
"""

import keras
from keras import ops
from typing import Tuple, List, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .image_encoder import ImageEncoderViT
from .transformer import TwoWayTransformer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SAM(keras.Model):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.pixel_mean = ops.array(pixel_mean, dtype=self.compute_dtype)
        self.pixel_std = ops.array(pixel_std, dtype=self.compute_dtype)

    def call(self, inputs: Dict[str, Any], training=None, multimask_output=True):
        image = inputs['image']  # B, H, W, C

        # Preprocess image
        input_image_shape = ops.shape(image)[1:3]
        image = self.preprocess(image)
        image_embeddings = self.image_encoder(image, training=training)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=inputs.get("points"),
            boxes=inputs.get("boxes"),
            masks=inputs.get("masks"),
            training=training
        )

        # Decode masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            training=training,
        )

        # Postprocess masks
        masks = self.postprocess_masks(low_res_masks, input_image_shape, inputs["original_size"])
        masks = ops.cast(masks > self.mask_threshold, dtype='uint8')

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    def preprocess(self, x: keras.KerasTensor) -> keras.KerasTensor:
        x = (ops.cast(x, self.compute_dtype) - self.pixel_mean) / self.pixel_std
        h, w = ops.shape(x)[1], ops.shape(x)[2]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        return x

    def postprocess_masks(self, masks: keras.KerasTensor, input_size: Tuple[int, int],
                          original_size: Tuple[int, int]) -> keras.KerasTensor:
        # masks are BxNxHxW
        masks = ops.image.resize(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            interpolation="bilinear",
            data_format="channels_first"
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = ops.image.resize(
            masks,
            original_size,
            interpolation="bilinear",
            data_format="channels_first"
        )
        return masks

    @classmethod
    def from_variant(cls, variant: str, **kwargs):
        if variant not in ["vit_b", "vit_l", "vit_h"]:
            raise ValueError(f"Unknown variant: {variant}")

        configs = {
            "vit_h": dict(encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
                          encoder_global_attn_indexes=[7, 15, 23, 31]),
            "vit_l": dict(encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
                          encoder_global_attn_indexes=[5, 11, 17, 23]),
            "vit_b": dict(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                          encoder_global_attn_indexes=[2, 5, 8, 11]),
        }

        config = configs[variant]
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        image_encoder = ImageEncoderViT(
            depth=config["encoder_depth"],
            embed_dim=config["encoder_embed_dim"],
            img_size=image_size,
            mlp_ratio=4,
            num_heads=config["encoder_num_heads"],
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=config["encoder_global_attn_indexes"],
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        return cls(image_encoder, prompt_encoder, mask_decoder, **kwargs)

    def get_config(self):
        # To make this serializable, we need to be able to reconstruct the sub-models.
        # This requires storing their configs.
        config = super().get_config()
        config.update({
            "image_encoder": keras.layers.serialize(self.image_encoder),
            "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
            "mask_decoder": keras.layers.serialize(self.mask_decoder),
            "pixel_mean": self.pixel_mean.numpy().tolist(),
            "pixel_std": self.pixel_std.numpy().tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        image_encoder_config = config.pop("image_encoder")
        prompt_encoder_config = config.pop("prompt_encoder")
        mask_decoder_config = config.pop("mask_decoder")

        config["image_encoder"] = keras.layers.deserialize(image_encoder_config)
        config["prompt_encoder"] = keras.layers.deserialize(prompt_encoder_config)
        config["mask_decoder"] = keras.layers.deserialize(mask_decoder_config)

        return cls(**config)

# ---------------------------------------------------------------------