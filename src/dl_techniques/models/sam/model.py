"""
Segment Anything Model (SAM) Keras 3 Implementation
===================================================

This file provides the main `SAM` model class, which integrates the image
encoder, prompt encoder, and mask decoder into a single, end-to-end Keras model.
It follows the structure of modern, variant-based models like ConvNeXt, offering
a `from_variant` class method to easily instantiate different model sizes
(e.g., `vit_b`, `vit_l`, `vit_h`).

**Intent**: To provide a user-friendly, high-level interface for the SAM model
that is fully serializable and adheres to modern Keras 3 best practices. This
class handles preprocessing, postprocessing, and the orchestration of the three
main sub-components.

**Architecture Overview**:
```
                            SAM Model
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        v                       v                       v
  Image Encoder          Prompt Encoder          Mask Decoder
  (ViT Backbone)         (Multi-modal)           (Transformer)
        │                       │                       │
        v                       v                       v
  Image Features         Sparse + Dense              Masks
  (B, H/16, W/16, C)     Prompt Embeddings          (B, N, H, W)
                                │                       │
                                └───────────────────────┘
                                      IoU Predictions
                                         (B, N)
```

**Data Flow**:
```
Input Image (B, H, W, 3)
    │
    v
Preprocessing (normalize + pad)
    │
    v
Image Encoder (ViT) ──────────────────> Image Embeddings (B, H', W', C)
                                              │
Input Prompts:                                │
- Points (coords, labels)                     │
- Boxes (x1, y1, x2, y2)                      │
- Masks (B, 1, H, W)                          │
    │                                         │
    v                                         │
Prompt Encoder ──> Sparse Embeddings (B, N, C)|
                  Dense Embeddings (B, H', W', C)
                                              │
    ┌─────────────────────────────────────────┘
    │
    v
Mask Decoder (Two-Way Transformer + Hypernetwork)
    │
    v
Low-Res Masks (B, N, H/4, W/4)
    │
    v
Postprocessing (upscale + threshold)
    │
    v
Output Masks (B, N, H, W)
IoU Predictions (B, N)
```

**Usage Example**:
```python
import keras
import numpy as np

# Create SAM model using a predefined variant
model = SAM.from_variant('vit_b')  # Options: 'vit_b', 'vit_l', 'vit_h'

# Prepare input data
image = keras.random.normal(shape=(1, 1024, 1024, 3))
points = (
    keras.ops.convert_to_tensor([[512.0, 512.0]]),  # coordinates
    keras.ops.convert_to_tensor([[1]])               # labels (1=foreground)
)

# Run inference
outputs = model({
    'image': image,
    'points': points,
    'original_size': (1024, 1024)
})

print(f"Masks shape: {outputs['masks'].shape}")                # (1, N, 1024, 1024)
print(f"IoU predictions: {outputs['iou_predictions'].shape}")  # (1, N)
print(f"Low-res logits: {outputs['low_res_logits'].shape}")    # (1, N, 256, 256)

# Example with boxes
boxes = keras.ops.convert_to_tensor([[[100.0, 100.0, 500.0, 500.0]]])
outputs = model({
    'image': image,
    'boxes': boxes,
    'original_size': (1024, 1024)
})

# Example with mask
mask_prompt = keras.random.normal(shape=(1, 1, 256, 256))
outputs = model({
    'image': image,
    'masks': mask_prompt,
    'original_size': (1024, 1024)
})
```

**Model Variants**:
- **vit_b** (Base): 768 dim, 12 layers, 12 heads (~90M parameters)
- **vit_l** (Large): 1024 dim, 24 layers, 16 heads (~300M parameters)
- **vit_h** (Huge): 1280 dim, 32 layers, 16 heads (~630M parameters)

**References**:
- Kirillov, A., et al. (2023). Segment Anything. *arXiv*.
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers
  for Image Recognition at Scale. *ICLR*.
"""

import keras
from keras import ops
from typing import Tuple, List, Any, Dict, Optional, Literal

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
    """
    Segment Anything Model (SAM) - A foundation model for image segmentation.

    SAM is a promptable segmentation system that can generate high-quality object
    masks from various types of prompts (points, boxes, or masks). It consists of
    three main components:

    1. **Image Encoder**: A Vision Transformer (ViT) that processes the input
       image once to produce image embeddings.
    2. **Prompt Encoder**: Encodes various prompt types (points, boxes, masks)
       into embedding space.
    3. **Mask Decoder**: A lightweight transformer decoder that combines image
       and prompt embeddings to predict segmentation masks.

    **Intent**: To provide a unified interface for promptable segmentation that
    can be used for interactive segmentation, automatic mask generation, or as
    a component in larger vision pipelines.

    **Key Features**:
    - Supports multiple prompt types (points, boxes, masks)
    - Can predict single or multiple mask proposals
    - Provides mask quality scores (predicted IoU)
    - Fully serializable with complete state preservation
    - Pre-configured variants for different compute budgets

    Args:
        image_encoder: ImageEncoderViT instance, processes input images into
            feature embeddings.
        prompt_encoder: PromptEncoder instance, encodes user prompts into
            embeddings.
        mask_decoder: MaskDecoder instance, predicts masks from image and
            prompt embeddings.
        pixel_mean: List of floats, mean values for image normalization (RGB order).
            Defaults to ImageNet means [123.675, 116.28, 103.53].
        pixel_std: List of floats, standard deviation for image normalization (RGB order).
            Defaults to ImageNet stds [58.395, 57.12, 57.375].
        mask_threshold: Float, threshold for converting mask logits to binary masks.
            Defaults to 0.0.
        image_format: String, expected color format of input images. Currently
            only 'RGB' is supported. Defaults to 'RGB'.
        **kwargs: Additional arguments for the Model base class.

    Input shape (in call):
        Dictionary with the following keys:
        - 'image': Required tensor of shape (batch_size, H, W, 3)
        - 'points': Optional tuple of (coords, labels) where:
            - coords: Shape (batch_size, num_points, 2) with (x, y) coordinates
            - labels: Shape (batch_size, num_points) with point labels
        - 'boxes': Optional tensor of shape (batch_size, num_boxes, 4) with
            (x1, y1, x2, y2) coordinates
        - 'masks': Optional tensor of shape (batch_size, 1, mask_h, mask_w)
        - 'original_size': Required tuple of (height, width) for the original
            image size before any preprocessing

    Output shape:
        Dictionary with the following keys:
        - 'masks': Binary masks of shape (batch_size, num_masks, H, W)
        - 'iou_predictions': Quality scores of shape (batch_size, num_masks)
        - 'low_res_logits': Low-resolution mask logits of shape
            (batch_size, num_masks, H/4, W/4)

    Attributes:
        image_encoder: The ViT image encoder.
        prompt_encoder: The prompt encoder.
        mask_decoder: The mask decoder.
        pixel_mean: Image normalization mean.
        pixel_std: Image normalization standard deviation.
        mask_threshold: Threshold for binary mask conversion.
        image_format: Expected image format.

    Example:
        ```python
        # Create model from variant
        model = SAM.from_variant('vit_b')

        # Prepare inputs
        image = keras.random.normal(shape=(1, 1024, 1024, 3))
        points = (
            keras.ops.convert_to_tensor([[500.0, 500.0]]),
            keras.ops.convert_to_tensor([[1]])
        )

        # Get predictions
        outputs = model({
            'image': image,
            'points': points,
            'original_size': (1024, 1024)
        })

        # Access results
        masks = outputs['masks']  # Binary masks
        iou_scores = outputs['iou_predictions']  # Quality scores
        ```

    Note:
        The model expects images in RGB format with values in [0, 255] range.
        The image encoder processes the full image once, making subsequent
        predictions with different prompts very efficient.
    """

    # Class attributes (can be overridden per instance)
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        mask_threshold: float = 0.0,
        image_format: str = "RGB",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if len(pixel_mean) != 3:
            raise ValueError(f"pixel_mean must have 3 values (RGB), got {len(pixel_mean)}")
        if len(pixel_std) != 3:
            raise ValueError(f"pixel_std must have 3 values (RGB), got {len(pixel_std)}")
        if image_format != "RGB":
            raise ValueError(f"Only 'RGB' image format is supported, got '{image_format}'")

        # Store all configuration parameters
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.mask_threshold = mask_threshold
        self.image_format = image_format

        # Convert normalization parameters to tensors
        self.pixel_mean = ops.array(pixel_mean, dtype="float32")
        self.pixel_std = ops.array(pixel_std, dtype="float32")

        # Store as Python lists for serialization
        self._pixel_mean_list = pixel_mean
        self._pixel_std_list = pixel_std

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
        multimask_output: bool = True
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through the SAM model.

        Args:
            inputs: Dictionary containing:
                - 'image': Required, shape (batch_size, H, W, 3)
                - 'points': Optional, tuple of (coords, labels)
                - 'boxes': Optional, shape (batch_size, num_boxes, 4)
                - 'masks': Optional, shape (batch_size, 1, mask_h, mask_w)
                - 'original_size': Required, tuple of (height, width)
            training: Optional boolean for training mode.
            multimask_output: Boolean, if True predicts multiple masks
                (usually 3), if False predicts single best mask. Defaults to True.

        Returns:
            Dictionary containing:
            - 'masks': Binary segmentation masks
            - 'iou_predictions': Predicted IoU scores for each mask
            - 'low_res_logits': Low-resolution mask logits
        """
        # Validate inputs
        if 'image' not in inputs:
            raise ValueError("Input dictionary must contain 'image' key")
        if 'original_size' not in inputs:
            raise ValueError("Input dictionary must contain 'original_size' key")

        image = inputs['image']  # (B, H, W, C)

        # Store input image shape for postprocessing
        input_image_shape = ops.shape(image)[1:3]

        # Step 1: Preprocess image (normalize and pad to encoder size)
        image = self.preprocess(image)

        # Step 2: Encode image to get image embeddings
        image_embeddings = self.image_encoder(image, training=training)

        # Step 3: Encode prompts (points, boxes, masks)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=inputs.get("points"),
            boxes=inputs.get("boxes"),
            masks=inputs.get("masks"),
            training=training
        )

        # Step 4: Decode masks from image and prompt embeddings
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            training=training,
        )

        # Step 5: Postprocess masks (upscale and threshold)
        masks = self.postprocess_masks(
            low_res_masks,
            input_image_shape,
            inputs["original_size"]
        )

        # Convert to binary masks
        masks = ops.cast(masks > self.mask_threshold, dtype='uint8')

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    def preprocess(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Preprocess input image for the encoder.

        Performs normalization using ImageNet statistics and pads the image
        to match the encoder's expected input size.

        Args:
            x: Input image tensor of shape (batch_size, H, W, 3) with values
                in [0, 255] range.

        Returns:
            Preprocessed image of shape (batch_size, img_size, img_size, 3)
            where img_size is the encoder's expected size (typically 1024).
        """
        # Normalize using ImageNet statistics
        x = ops.cast(x, self.compute_dtype)
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad to encoder size
        h, w = ops.shape(x)[1], ops.shape(x)[2]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w

        # Add padding to bottom and right
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        return x

    def postprocess_masks(
        self,
        masks: keras.KerasTensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Postprocess predicted masks to match original image size.

        The mask decoder outputs low-resolution masks which need to be:
        1. Upscaled to encoder input size
        2. Cropped to remove padding
        3. Scaled to original image size

        Args:
            masks: Low-resolution mask logits from decoder, shape
                (batch_size, num_masks, H_low, W_low).
            input_size: Size of input image before padding, tuple of (H, W).
            original_size: Original image size before any preprocessing,
                tuple of (H, W).

        Returns:
            Masks at original image resolution, shape
            (batch_size, num_masks, original_H, original_W).
        """
        # Step 1: Upscale to encoder input size
        masks = ops.image.resize(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            interpolation="bilinear",
            data_format="channels_first"
        )

        # Step 2: Remove padding by cropping to input size
        masks = masks[..., :input_size[0], :input_size[1]]

        # Step 3: Scale to original image size
        masks = ops.image.resize(
            masks,
            original_size,
            interpolation="bilinear",
            data_format="channels_first"
        )

        return masks

    @classmethod
    def from_variant(
        cls,
        variant: Literal['vit_b', 'vit_l', 'vit_h'],
        **kwargs: Any
    ) -> 'SAM':
        """
        Create a SAM model from a predefined variant configuration.

        This factory method provides easy access to standard SAM architectures
        with different capacity/compute tradeoffs:

        - **vit_b** (Base): Fastest, lowest memory, good quality
        - **vit_l** (Large): Balanced speed/quality
        - **vit_h** (Huge): Best quality, highest resource requirements

        Args:
            variant: String, model variant name. Options are:
                - 'vit_b': Base model (768 dim, 12 layers, ~90M params)
                - 'vit_l': Large model (1024 dim, 24 layers, ~300M params)
                - 'vit_h': Huge model (1280 dim, 32 layers, ~630M params)
            **kwargs: Additional arguments to pass to SAM constructor
                (e.g., mask_threshold, pixel_mean, pixel_std).

        Returns:
            Configured SAM model instance.

        Raises:
            ValueError: If variant is not one of the supported options.

        Example:
            ```python
            # Create different model sizes
            model_base = SAM.from_variant('vit_b')
            model_large = SAM.from_variant('vit_l')
            model_huge = SAM.from_variant('vit_h')

            # Create with custom settings
            model = SAM.from_variant(
                'vit_b',
                mask_threshold=0.5,
                pixel_mean=[120.0, 115.0, 100.0]
            )
            ```
        """
        if variant not in ["vit_b", "vit_l", "vit_h"]:
            raise ValueError(
                f"Unknown variant: '{variant}'. "
                f"Supported variants are: 'vit_b', 'vit_l', 'vit_h'"
            )

        # Configuration for each variant
        configs = {
            "vit_h": {
                "encoder_embed_dim": 1280,
                "encoder_depth": 32,
                "encoder_num_heads": 16,
                "encoder_global_attn_indexes": [7, 15, 23, 31],
            },
            "vit_l": {
                "encoder_embed_dim": 1024,
                "encoder_depth": 24,
                "encoder_num_heads": 16,
                "encoder_global_attn_indexes": [5, 11, 17, 23],
            },
            "vit_b": {
                "encoder_embed_dim": 768,
                "encoder_depth": 12,
                "encoder_num_heads": 12,
                "encoder_global_attn_indexes": [2, 5, 8, 11],
            },
        }

        config = configs[variant]

        # Common configuration across all variants
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        # Create image encoder (ViT)
        image_encoder = ImageEncoderViT(
            img_size=image_size,
            patch_size=vit_patch_size,
            embed_dim=config["encoder_embed_dim"],
            depth=config["encoder_depth"],
            num_heads=config["encoder_num_heads"],
            mlp_ratio=4.0,
            out_chans=prompt_embed_dim,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=config["encoder_global_attn_indexes"],
        )

        # Create prompt encoder
        prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        # Create two-way transformer for mask decoder
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        # Create mask decoder
        mask_decoder = MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=transformer,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # Create and return SAM model
        return cls(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the model for serialization.

        This method serializes all sub-models and configuration parameters,
        enabling full model reconstruction via `from_config`.

        Returns:
            Configuration dictionary containing all model parameters.
        """
        config = super().get_config()
        config.update({
            "image_encoder": keras.layers.serialize(self.image_encoder),
            "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
            "mask_decoder": keras.layers.serialize(self.mask_decoder),
            "pixel_mean": self._pixel_mean_list,
            "pixel_std": self._pixel_std_list,
            "mask_threshold": self.mask_threshold,
            "image_format": self.image_format,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SAM':
        """
        Creates a SAM model from a configuration dictionary.

        This method deserializes the model from a configuration, reconstructing
        all sub-models and restoring their weights.

        Args:
            config: Configuration dictionary from `get_config()`.

        Returns:
            Reconstructed SAM model instance.
        """
        # Deserialize sub-models
        image_encoder_config = config.pop("image_encoder")
        prompt_encoder_config = config.pop("prompt_encoder")
        mask_decoder_config = config.pop("mask_decoder")

        config["image_encoder"] = keras.layers.deserialize(image_encoder_config)
        config["prompt_encoder"] = keras.layers.deserialize(prompt_encoder_config)
        config["mask_decoder"] = keras.layers.deserialize(mask_decoder_config)

        return cls(**config)

    def compute_output_shape(
        self,
        input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Compute output shapes given input shapes.

        Args:
            input_shape: Dictionary of input shapes.

        Returns:
            Dictionary of output shapes.
        """
        batch_size = input_shape.get('image', (None,))[0]
        original_h, original_w = None, None  # Unknown until runtime

        # Number of masks depends on multimask_output setting
        # Default is 3 for multimask, 1 for single mask
        num_masks = None  # Variable

        return {
            'masks': (batch_size, num_masks, original_h, original_w),
            'iou_predictions': (batch_size, num_masks),
            'low_res_logits': (batch_size, num_masks, None, None),
        }

# ---------------------------------------------------------------------
