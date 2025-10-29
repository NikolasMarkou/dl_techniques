# Segment Anything Model (SAM)

A production-ready, fully-featured implementation of the **Segment Anything Model (SAM)** architecture in **Keras 3**, based on the paper ["Segment Anything"](https://arxiv.org/abs/2304.02643) by Kirillov et al. (2023).

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices, providing a modular, well-documented, and fully serializable codebase that works seamlessly across TensorFlow, PyTorch, and JAX backends.

---

## Table of Contents

1. [What is SAM?](#1-what-is-sam)
2. [Key Components](#2-key-components)
3. [About This Implementation](#3-about-this-implementation)
4. [Quick Start](#4-quick-start)
5. [Architecture Components](#5-architecture-components)
6. [Configuration & Variants](#6-configuration--variants)
7. [Usage Examples](#7-usage-examples)
8. [Serialization](#8-serialization)
9. [Performance Tips](#9-performance-tips)
10. [Testing](#10-testing)
11. [Architecture Details](#11-architecture-details)
12. [Requirements](#12-requirements)
13. [Citation](#13-citation)

---

## 1. What is SAM?

The Segment Anything Model (SAM) is a foundation model for image segmentation. It introduces a new task, model, and dataset for promptable segmentation, enabling it to generalize to new objects and image distributions at test time without additional training—a capability often referred to as zero-shot transfer.

SAM is designed to be **promptable**, meaning it can generate segmentation masks for objects based on various input prompts, such as points, bounding boxes, or even text. Its powerful generalization capabilities stem from being trained on a massive dataset of 11 million images and over 1 billion masks.

### Why SAM?

SAM addresses several key challenges in image segmentation:

-   **Data Scarcity**: Traditional segmentation models require large, manually annotated datasets for specific tasks. SAM's pre-training on a diverse dataset reduces this need.
-   **Generalization**: It can segment objects and images it has never seen before, making it highly versatile.
-   **Interactivity**: The promptable interface allows for human-in-the-loop annotation and real-time interactive segmentation.

By separating the model into a heavy image encoder and lightweight prompt/mask components, SAM achieves impressive efficiency:

-   ✅ **Real-time Performance**: The image is encoded only once, and subsequent prompts are processed in milliseconds.
-   ✅ **Versatile Prompting**: Accepts points, boxes, and masks as input.
-   ✅ **High-Quality Masks**: Produces detailed and accurate segmentation masks.
-   ✅ **Ambiguity Awareness**: Can generate multiple valid masks when a prompt is ambiguous.

---

## 2. Key Components

The SAM architecture is composed of three main parts that work together to achieve promptable segmentation:

### 2.1 Image Encoder

A powerful **Vision Transformer (ViT)** backbone processes a high-resolution input image and generates a high-dimensional image embedding. This is the most computationally intensive part of the model, but it only needs to be run once per image.

```
Input Image (B, 1024, 1024, 3)
      |
      v
Vision Transformer (ViT)
- Patch Embedding
- Transformer Blocks
- Neck Upsampling
      |
      v
Image Embedding (B, 64, 64, 256)
```

### 2.2 Prompt Encoder

This lightweight encoder converts various input prompts into embedding vectors.

-   **Sparse Prompts** (points, boxes): Encoded using positional encodings and learned type embeddings (e.g., "foreground point," "top-left corner").
-   **Dense Prompts** (masks): Processed by a small CNN to produce an embedding grid that is added to the image embedding.

```
Points (x, y) + Labels
      |
      v
Positional + Type Embedding --> Sparse Embeddings
                                      |
Boxes (x1, y1, x2, y2)                |
      |                               |
      v                               v
Corner Encoding -------------> Concatenate
                                      |
Masks (H, W)                          |
      |                               |
      v                               v
CNN --------------------------> Dense Embeddings
```

### 2.3 Mask Decoder

A lightweight transformer decoder that efficiently predicts segmentation masks from the image embedding and prompt embeddings. It uses a **Two-Way Transformer** architecture to bidirectionally update both image and prompt features.

```
Image Embedding + Dense Prompts
Sparse Prompt Embeddings
      |
      v
Two-Way Transformer
      |
      v
Upscaling + Hypernetwork MLP
      |
      v
Output Masks + IoU Scores
```

---

## 3. About This Implementation

This implementation provides a complete, production-ready SAM codebase that integrates seamlessly with the `dl_techniques` framework while following all Keras 3 best practices.

### 3.1 Features

#### **Modular Components**

The code is organized into logical, reusable layers and models:

-   **`ImageEncoderViT`**: The ViT backbone for image feature extraction.
-   **`PromptEncoder`**: Encodes points, boxes, and masks.
-   **`MaskDecoder`**: Predicts masks using a two-way transformer.
-   **`TwoWayTransformer`**: The core attention module of the decoder.
-   **`SAM`**: The final model that integrates all components and handles pre/post-processing.

#### **Framework Integration**

-   ✅ **Factory Integration**: Uses `create_normalization_layer()` and `create_ffn_layer()` from `dl_techniques` for flexible and modern components.
-   ✅ **Standardized Blocks**: Follows project patterns for consistent architecture (e.g., `ViTBlock`).

#### **Keras 3 Best Practices**

-   ✅ **Full Serialization**: Every custom layer and model implements `get_config()` and is registered with `@keras.saving.register_keras_serializable()`.
-   ✅ **Correct Build Logic**: Strictly separates layer creation (`__init__`) from weight creation (`build`) for safe deserialization.
-   ✅ **Backend Agnostic**: Built with `keras.ops`, ensuring it runs on TensorFlow, PyTorch, or JAX.
-   ✅ **Type Hints**: Complete Python 3 type annotations for clarity and correctness.
-   ✅ **Comprehensive Documentation**: Sphinx-style docstrings with architecture diagrams and examples in every file.

#### **Production Quality**

-   ✅ **Input Validation**: Clear error messages for invalid configurations.
-   ✅ **Model Variants**: Includes a `from_variant()` factory to easily create `vit_b`, `vit_l`, and `vit_h` models.
-   ✅ **Pre/Post-processing**: The main `SAM` model handles image normalization, padding, and mask upscaling internally.

---

## 4. Quick Start

### Basic Usage

The easiest way to use the model is through the `from_variant` class method.

```python
import keras
import numpy as np
from sam import SAM  # Assuming the main model is in sam/model.py

# 1. Create SAM model from a predefined variant
# Options: 'vit_b' (Base), 'vit_l' (Large), 'vit_h' (Huge)
model = SAM.from_variant('vit_b')

# 2. Prepare input data
# The model expects a 1024x1024 image
image = keras.random.normal(shape=(1, 1024, 1024, 3))

# Define a point prompt at the center of the image
# Coordinates are (x, y)
points_coords = keras.ops.convert_to_tensor([[[512.0, 512.0]]])
# Labels are (1=foreground, 0=background)
points_labels = keras.ops.convert_to_tensor([[1]])
points = (points_coords, points_labels)

# 3. Run inference
# The 'original_size' argument is used for correct mask upscaling
outputs = model({
    'image': image,
    'points': points,
    'original_size': (1024, 1024)
})

# 4. Inspect outputs
print(f"Masks shape: {outputs['masks'].shape}")
# Expected: (1, 3, 1024, 1024) -> Batch, Num Masks, H, W
print(f"IoU predictions shape: {outputs['iou_predictions'].shape}")
# Expected: (1, 3) -> Batch, Num Masks
print(f"Low-res logits shape: {outputs['low_res_logits'].shape}")
# Expected: (1, 3, 256, 256) -> Batch, Num Masks, H_low, W_low

# 5. Save and load the model
model.save('sam_vit_b.keras')
loaded_model = keras.models.load_model('sam_vit_b.keras')
print("Model saved and loaded successfully!")
```

---

## 5. Architecture Components

### 5.1 ImageEncoderViT

The powerful ViT backbone that generates image embeddings.

```python
from image_encoder import ImageEncoderViT

# Instantiate the "Huge" variant configuration
encoder = ImageEncoderViT(
    img_size=1024,
    patch_size=16,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    out_chans=256,
    use_rel_pos=True,
    window_size=14,
    global_attn_indexes=(7, 15, 23, 31),
)

# Process an image
dummy_image = np.random.rand(1, 1024, 1024, 3).astype("float32")
embedding = encoder(dummy_image)
print(f"Output embedding shape: {embedding.shape}") # (1, 64, 64, 256)
```

**Key Features**:
-   Standard ViT architecture with patch and position embeddings.
-   Uses windowed attention for efficiency in most layers.
-   Supports global attention in specified layers for long-range dependency modeling.
-   Includes a "neck" module to refine and upsample features.

### 5.2 PromptEncoder

Encodes sparse (points, boxes) and dense (masks) prompts.

```python
from prompt_encoder import PromptEncoder

# Create prompt encoder
prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(1024, 1024),
    mask_in_chans=16,
)

# Example: Encode points
points = (
    keras.ops.convert_to_tensor([[[100.0, 200.0]]]),
    keras.ops.convert_to_tensor([[1]])
)
sparse_emb, dense_emb = prompt_encoder(points=points)
print(f"Sparse embedding shape: {sparse_emb.shape}") # (1, 1, 256)
print(f"Dense embedding shape: {dense_emb.shape}")   # (1, 64, 64, 256)
```

**Key Features**:
-   Uses random Fourier features for positional encoding (`PositionEmbeddingRandom`).
-   Learned embeddings for prompt types (foreground/background points, box corners).
-   A small CNN downscales input masks to create dense embeddings.
-   Gracefully handles missing prompts with learned "not-a-prompt" embeddings.

### 5.3 MaskDecoder

The lightweight decoder that predicts masks and IoU scores.

```python
from mask_decoder import MaskDecoder
from transformer import TwoWayTransformer

# Create the transformer needed by the decoder
transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)

# Create decoder
decoder = MaskDecoder(
    transformer_dim=256,
    transformer=transformer,
    num_multimask_outputs=3,
)

# Dummy inputs for demonstration
image_emb = keras.random.normal(shape=(1, 64, 64, 256))
image_pe = keras.random.normal(shape=(1, 64, 64, 256))
sparse_prompts = keras.random.normal(shape=(1, 2, 256)) # 2 points
dense_prompts = keras.random.normal(shape=(1, 64, 64, 256))

# Get predictions
masks, iou = decoder(
    image_embeddings=image_emb,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_prompts,
    dense_prompt_embeddings=dense_prompts,
    multimask_output=True
)
print(f"Masks shape: {masks.shape}") # (1, 3, 256, 256)
print(f"IoU shape: {iou.shape}")     # (1, 3)
```

**Key Features**:
-   Uses a `TwoWayTransformer` to update image and prompt embeddings jointly.
-   A hypernetwork of small MLPs generates parameters for the final mask prediction layer.
-   Upscales image features 4x before final prediction.
-   Includes an IoU prediction head to score mask quality.

---

## 6. Configuration & Variants

The primary way to configure the model is by selecting a variant, which determines the size and capacity of the image encoder.

| Variant | Encoder Dim | Encoder Depth | Encoder Heads | Approx. Params | Use Case |
| :------ | :---------- | :------------ | :------------ | :------------- | :------- |
| `vit_b` | 768         | 12            | 12            | ~90M           | Fastest, good quality |
| `vit_l` | 1024        | 24            | 16            | ~300M          | Balanced speed/quality |
| `vit_h` | 1280        | 32            | 16            | ~630M          | Highest quality, most resources |

You can instantiate these easily:

```python
# Create different model sizes
model_base = SAM.from_variant('vit_b')
model_large = SAM.from_variant('vit_l')
model_huge = SAM.from_variant('vit_h')
```

### Advanced Configuration

You can also customize parameters like the mask threshold when creating a model:

```python
model = SAM.from_variant(
    'vit_b',
    mask_threshold=0.5,
    pixel_mean=[120.0, 115.0, 100.0] # Custom normalization
)
```

The underlying `ViTBlock` also supports factory-based configuration for normalization and FFN layers, which can be modified for experimental purposes.

---

## 7. Usage Examples

### 7.1 Segmentation with Point Prompts

```python
# A single foreground point
points = (
    keras.ops.convert_to_tensor([[[512.0, 384.0]]]), # (x, y) coords
    keras.ops.convert_to_tensor([[1]])               # 1 = foreground
)
outputs = model({'image': image, 'points': points, 'original_size': (1024, 1024)})
```

### 7.2 Segmentation with Box Prompts

```python
# A bounding box
# Format: (x1, y1, x2, y2)
boxes = keras.ops.convert_to_tensor([[[200.0, 150.0, 800.0, 600.0]]])
outputs = model({'image': image, 'boxes': boxes, 'original_size': (1024, 1024)})
```

### 7.3 Segmentation with a Mask Prompt

```python
# A low-resolution mask prompt (256x256)
mask_prompt = keras.random.normal(shape=(1, 1, 256, 256))
outputs = model({'image': image, 'masks': mask_prompt, 'original_size': (1024, 1024)})
```

### 7.4 Combining Prompts

The model can combine multiple sparse prompts (points and boxes).

```python
# Two foreground points and one background point
point_coords = keras.ops.convert_to_tensor([[[512.0, 512.0], [400.0, 400.0], [600.0, 200.0]]])
point_labels = keras.ops.convert_to_tensor([[1, 1, 0]]) # two foreground, one background
points = (point_coords, point_labels)

# A bounding box to guide the segmentation
boxes = keras.ops.convert_to_tensor([[[300.0, 300.0, 700.0, 700.0]]])

outputs = model({
    'image': image,
    'points': points,
    'boxes': boxes,
    'original_size': (1024, 1024)
})
```

---

## 8. Serialization

All components support full Keras 3 serialization, allowing you to save and load the entire model, including its sub-models and custom layers.

### Save and Load Full Model

```python
# Save the entire model to a single file
model.save('sam_model.keras')

# Load the model back, including custom objects
loaded_model = keras.models.load_model('sam_model.keras')

# Verify that the loaded model works
predictions = loaded_model(test_inputs)
```

### Save and Load Weights Only

```python
# Save only the model's weights
model.save_weights('sam_weights.weights.h5')

# Create a new model instance with the same architecture
new_model = SAM.from_variant('vit_b')

# Load the weights into the new model
new_model.load_weights('sam_weights.weights.h5')
```

### Configuration Export

You can also serialize the model's architecture as a JSON-compatible dictionary.

```python
import json

# Get the model's configuration
config = model.get_config()

# Save config to a file
with open('sam_config.json', 'w') as f:
    # Keras serialization may produce non-standard JSON types
    # A proper JSON library is needed for robust serialization
    json.dump(config, f, indent=2, default=str)

# Recreate the model from the configuration
# Note: This requires deserializing nested custom layers
recreated_model = SAM.from_config(config)
```

---

## 9. Performance Tips

### 9.1 Cache Image Embeddings

The most significant optimization is to run the heavy `image_encoder` only once per image.

```python
# 1. Get image embedding
image_embedding = model.image_encoder(model.preprocess(image))

# 2. Use the embedding for multiple prompts (much faster)
for points_prompt in all_prompts:
    # Encode only the prompt
    sparse_emb, dense_emb = model.prompt_encoder(points=points_prompt)

    # Decode with the cached embedding
    masks, iou = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True
    )
    # ... postprocess masks ...
```

### 9.2 Choose the Right Variant

-   Use `vit_b` for applications requiring maximum speed (e.g., interactive tools).
-   Use `vit_h` for offline processing where mask quality is the top priority.
-   `vit_l` provides a good balance for most applications.

### 9.3 Use Mixed Precision

For training or inference on compatible GPUs (NVIDIA Turing architecture or newer), using mixed precision can provide a significant speedup.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Model will now use float16 for computation where possible
model = SAM.from_variant('vit_b')
```

---

## 10. Testing

To ensure correctness and robustness, a comprehensive test suite should be run.

```bash
# Example command to run tests
python test_sam.py
```

### Test Coverage

A good test suite would cover:
1.  ✅ **Layer Tests**: Forward pass and shape correctness for each custom layer.
2.  ✅ **Model Variant Tests**: Instantiation of `vit_b`, `vit_l`, and `vit_h`.
3.  ✅ **Forward Pass Tests**: End-to-end model execution with different prompt types.
4.  ✅ **Serialization Tests**: Save/load for the main `SAM` model and all sub-components.
5.  ✅ **Training Test**: A simple `compile` and `fit` call to check gradient flow.
6.  ✅ **Consistency Test**: Ensure output is identical after saving and loading.

---

## 11. Architecture Details

### 11.1 ViTBlock Data Flow

The core block of the `ImageEncoderViT`.

```
Input
  |
  +------------------------ (Residual Connection 1)
  |
  v
Norm1
  |
  v
Attention (Windowed or Global with Relative Position Bias)
  |
  v
Add (Input + Attention Output)
  |
  +------------------------ (Residual Connection 2)
  |
  v
Norm2
  |
  v
FFN (Feed-Forward Network)
  |
  v
Add (Previous Sum + FFN Output)
  |
  v
Output
```

### 11.2 TwoWayAttentionBlock Data Flow

The core block of the `TwoWayTransformer` in the `MaskDecoder`.

```
Queries_in (tokens), Keys_in (image features)
    |
    v
┌─────────────────────────────────────────┐
│ 1. Self-Attention on Queries            │
│    Queries' = Norm(Queries + Attn(Q,K,V)) │
└─────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────┐
│ 2. Cross-Attention (Token to Image)     │
│    Queries'' = Norm(Queries' + Attn(Q,K,V))│
└─────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────┐
│ 3. MLP/FFN on Queries                   │
│    Queries''' = Norm(Queries'' + FFN(Q))│
└─────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────┐
│ 4. Cross-Attention (Image to Token)     │
│    Keys' = Norm(Keys + Attn(Q,K,V))     │
└─────────────────────────────────────────┘
    |
    v
Queries_out, Keys_out
```

---

## 12. Requirements

### Core Dependencies

```
python>=3.9
keras>=3.0.0
tensorflow>=2.15.0  # or torch, or jax
numpy
```

### Framework Integration

This implementation optionally uses the `dl_techniques` framework for:
-   Normalization factory (`dl_techniques.layers.norms`)
-   FFN factory (`dl_techniques.layers.ffn`)

These can be replaced with standard Keras layers if the framework is not used.

---

## 13. Citation

If you use SAM in your research, please cite the original paper:

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```