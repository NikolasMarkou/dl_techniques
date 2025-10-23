# DETR: End-to-End Object Detection with Transformers

A modern Keras 3 implementation of DETR (DEtection TRansformer), the groundbreaking model that revolutionized object detection by treating it as a direct set prediction problem using transformers.

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
- [Implementation Features](#implementation-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration Options](#configuration-options)
- [Training](#training)
- [Model Components](#model-components)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

DETR (DEtection TRansformer) is a novel approach to object detection that frames the task as a direct set prediction problem. Unlike traditional object detectors that rely on hand-crafted components like anchor boxes, non-maximum suppression (NMS), or region proposals, DETR treats object detection as a direct set-to-set prediction problem using transformers.

### What Makes DETR Special?

1. **End-to-End**: No hand-crafted components needed (no NMS, no anchor generation)
2. **Set Prediction**: Predicts a fixed-size set of detections in parallel
3. **Transformer-Based**: Leverages the power of self-attention for global reasoning
4. **Bipartite Matching**: Uses Hungarian algorithm for unique assignment during training
5. **Simple Architecture**: Remarkably simple compared to traditional detectors

### When to Use DETR?

✅ **Good for:**
- Research and experimentation with transformer-based detection
- Applications where you need the simplicity of end-to-end training
- Scenarios requiring global reasoning over the entire image
- When you want to avoid tuning anchor-based hyperparameters

⚠️ **Consider alternatives for:**
- Real-time applications (DETR is computationally expensive)
- Detecting very small objects (DETR struggles with small instances)
- Limited computational resources

## Key Innovations

### 1. Direct Set Prediction

DETR predicts a fixed set of N object detections in a single pass, where N is larger than the typical number of objects in an image. This eliminates the need for:
- Anchor box generation
- Non-maximum suppression (NMS)
- Region proposal networks

### 2. Bipartite Matching Loss

During training, DETR uses the Hungarian algorithm to find an optimal bipartite matching between predicted and ground-truth objects. This ensures:
- Each ground-truth object is matched to exactly one prediction
- Permutation-invariant loss (order doesn't matter)
- No duplicate predictions for the same object

### 3. Object Queries

DETR uses learned "object queries" that:
- Are learned embeddings that represent potential objects
- Attend to image features via cross-attention
- Each query learns to specialize in detecting objects at specific locations/scales

### 4. Global Reasoning

The transformer encoder allows every position to attend to every other position, enabling:
- Global context understanding
- Long-range dependencies
- Relational reasoning between objects

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DETR Architecture                        │
└─────────────────────────────────────────────────────────────────┘

Input Image (H × W × 3)
         ↓
┌────────────────────┐
│   CNN Backbone     │  (e.g., ResNet-50)
│   (Feature Map)    │  Output: H/32 × W/32 × 2048
└────────────────────┘
         ↓
┌────────────────────┐
│  1×1 Convolution   │  Project to hidden_dim (256)
│  (Input Projection)│  Output: H/32 × W/32 × 256
└────────────────────┘
         ↓
┌────────────────────┐
│   Positional       │  2D sinusoidal encoding
│   Encoding         │  Added to features
└────────────────────┘
         ↓
┌────────────────────┐
│   Flatten          │  Reshape to sequence
│                    │  Output: (H/32 × W/32) × 256
└────────────────────┘
         ↓
┌────────────────────┐
│ Transformer Encoder│  N layers (default: 6)
│   (Self-Attention) │  Global reasoning
└────────────────────┘
         ↓
      Memory
         ↓  ────────────────────────┐
┌────────────────────┐              │
│ Transformer Decoder│              │
│   (Object Queries) │ ←────────────┘
│                    │  Cross-Attention to Memory
│   N Layers (6)     │  Self-Attention among Queries
└────────────────────┘
         ↓
  Decoder Outputs
  (One per layer)
         ↓
    ┌────────┴────────┐
    ↓                 ↓
┌─────────┐    ┌──────────┐
│  Class  │    │   Bbox   │
│  Head   │    │   Head   │
│ (Dense) │    │  (MLP)   │
└─────────┘    └──────────┘
    ↓                 ↓
Predictions    Predictions
(N × C)        (N × 4)

Where:
- N = num_queries (e.g., 100)
- C = num_classes + 1 (includes "no object" class)
```

## Implementation Features

### Modern Keras 3 Design

This implementation leverages the latest Keras 3 features and best practices:

✅ **Factory Pattern Integration**
- Normalization factory: Easy switching between LayerNorm, RMSNorm, BatchNorm, etc.
- FFN factory: Support for MLP, SwiGLU, GeGLU, and other feed-forward variants
- Attention factory ready: Extensible for different attention mechanisms

✅ **Component Reusability**
- Uses `TransformerLayer` from dl_techniques framework
- Leverages battle-tested components
- Reduces code duplication by ~150 lines

✅ **Full Serialization Support**
- Proper `@keras.saving.register_keras_serializable()` decorators
- Complete `get_config()` implementations
- Custom `from_config()` for complex models

✅ **Type Safety**
- Full type hints throughout
- Better IDE support
- Catch errors at development time

✅ **Flexible Configuration**
- Multiple normalization options
- Multiple FFN architectures
- Configurable activation functions

## Installation

```bash
# Install required packages
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy

# Install dl_techniques framework (if using framework components)
# pip install dl_techniques
```

## Quick Start

### Basic Usage

```python
import keras
from detr_refactored import create_detr

# Create a DETR model for COCO dataset
model = create_detr(
    num_classes=80,      # COCO has 80 classes
    num_queries=100,     # Detect up to 100 objects
    backbone_name="resnet50",
    hidden_dim=256
)

# Prepare inputs
image = keras.Input(shape=(None, None, 3), name="image")
mask = keras.Input(shape=(None, None), dtype="bool", name="mask")

# Get predictions
outputs = model([image, mask])

# outputs is a dictionary:
# {
#     'pred_logits': (batch, 100, 81),  # Class predictions
#     'pred_boxes': (batch, 100, 4),     # Bbox predictions (x, y, w, h)
#     'aux_outputs': [...]               # Intermediate predictions
# }
```

### Inference Example

```python
import numpy as np
import keras

# Load trained model
model = keras.models.load_model("detr_model.keras")

# Prepare image (normalized to [0, 1])
image = load_and_preprocess_image("image.jpg")  # Shape: (H, W, 3)
image_batch = np.expand_dims(image, axis=0)     # Shape: (1, H, W, 3)

# Create dummy mask (False = valid pixel, True = padding)
mask = np.zeros((1, image.shape[0], image.shape[1]), dtype=bool)

# Run inference
predictions = model([image_batch, mask], training=False)

# Extract predictions
class_logits = predictions['pred_logits'][0]  # (100, 81)
boxes = predictions['pred_boxes'][0]           # (100, 4)

# Get class probabilities
class_probs = keras.ops.softmax(class_logits, axis=-1)

# Filter predictions by confidence threshold
threshold = 0.7
max_probs = keras.ops.max(class_probs[:, :-1], axis=-1)  # Exclude "no object"
keep = max_probs > threshold

# Get filtered predictions
filtered_boxes = boxes[keep]
filtered_classes = keras.ops.argmax(class_probs[keep, :-1], axis=-1)
filtered_scores = max_probs[keep]

print(f"Detected {len(filtered_boxes)} objects")
```

## Detailed Usage

### 1. Creating a Model

```python
from detr_refactored import create_detr

model = create_detr(
    num_classes=80,           # Number of object classes
    num_queries=100,          # Max detections per image
    backbone_name="resnet50", # CNN backbone
    backbone_trainable=False, # Freeze backbone initially
    hidden_dim=256,           # Transformer dimension
    num_heads=8,              # Attention heads
    num_encoder_layers=6,     # Encoder depth
    num_decoder_layers=6,     # Decoder depth
    ffn_dim=2048,            # FFN hidden dimension
    dropout=0.1,             # Dropout rate
    aux_loss=True,           # Use auxiliary losses
    activation="relu",       # FFN activation
    normalization_type="layer_norm",  # Normalization type
    ffn_type="mlp"           # FFN architecture
)
```

### 2. Custom Backbone

```python
import keras

# Create custom backbone
custom_backbone = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, strides=2, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    # ... more layers
], name="custom_backbone")

# Create transformer
from detr_refactored import DetrTransformer

transformer = DetrTransformer(
    hidden_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Create DETR model manually
from detr_refactored import DETR

model = DETR(
    num_classes=80,
    num_queries=100,
    backbone=custom_backbone,
    transformer=transformer,
    hidden_dim=256
)
```

### 3. Experimenting with Different Architectures

```python
# Try RMSNorm (faster than LayerNorm)
model_rms = create_detr(
    num_classes=80,
    num_queries=100,
    normalization_type="rms_norm"
)

# Try SwiGLU FFN (from LLaMA, GPT-4)
model_swiglu = create_detr(
    num_classes=80,
    num_queries=100,
    ffn_type="swiglu",
    activation="swish"
)

# Try GeGLU FFN (from T5)
model_geglu = create_detr(
    num_classes=80,
    num_queries=100,
    ffn_type="geglu",
    activation="gelu"
)

# Combine different components
model_advanced = create_detr(
    num_classes=80,
    num_queries=100,
    normalization_type="rms_norm",   # Faster normalization
    ffn_type="swiglu",                # Modern FFN
    activation="gelu",                # Better activation
    num_encoder_layers=8,             # Deeper encoder
    num_decoder_layers=8              # Deeper decoder
)
```

## Configuration Options

### Backbone Options

| Option | Description | Default |
|--------|-------------|---------|
| `backbone_name` | CNN backbone identifier | "resnet50" |
| `backbone_trainable` | Whether to fine-tune backbone | False |

Currently supported backbones:
- `"resnet50"`: ResNet-50 with ImageNet weights

### Transformer Options

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `hidden_dim` | Transformer feature dimension | 256 | 128-1024 |
| `num_heads` | Number of attention heads | 8 | 4-16 |
| `num_encoder_layers` | Encoder layer count | 6 | 3-12 |
| `num_decoder_layers` | Decoder layer count | 6 | 3-12 |
| `ffn_dim` | FFN intermediate dimension | 2048 | 512-4096 |
| `dropout` | Dropout rate | 0.1 | 0.0-0.5 |

### Architecture Variants

| Option | Description | Options | Default |
|--------|-------------|---------|---------|
| `activation` | Activation function | "relu", "gelu", "swish", "mish" | "relu" |
| `normalization_type` | Normalization layer | "layer_norm", "rms_norm", "batch_norm" | "layer_norm" |
| `ffn_type` | Feed-forward network type | "mlp", "swiglu", "geglu", "glu" | "mlp" |

### Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_classes` | Number of object classes | Required |
| `num_queries` | Maximum detections per image | 100 |
| `aux_loss` | Use auxiliary decoder losses | True |

## Training

### Loss Components

DETR uses a combination of losses:

1. **Classification Loss**: Focal loss or cross-entropy for class predictions
2. **Bounding Box Loss**: L1 loss for box coordinates
3. **GIoU Loss**: Generalized IoU loss for better box quality
4. **Bipartite Matching**: Hungarian algorithm for optimal assignment

### Typical Training Setup

```python
import keras
from detr_refactored import create_detr

# Create model
model = create_detr(num_classes=80, num_queries=100)

# Define optimizer with learning rate schedule
initial_lr = 1e-4
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=train_steps
)
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4
)

# Compile (note: loss is custom, handled in training loop)
# DETR requires Hungarian matching which is not a standard Keras loss
# Typically implemented in a custom training loop

# Training pseudocode (actual implementation requires Hungarian matching)
for epoch in range(num_epochs):
    for images, masks, targets in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model([images, masks], training=True)
            
            # Compute bipartite matching
            indices = hungarian_matcher(predictions, targets)
            
            # Compute losses based on matched predictions
            loss_class = classification_loss(predictions, targets, indices)
            loss_bbox = bbox_l1_loss(predictions, targets, indices)
            loss_giou = giou_loss(predictions, targets, indices)
            
            # Auxiliary losses from intermediate decoder layers
            aux_losses = compute_aux_losses(predictions['aux_outputs'], targets)
            
            total_loss = loss_class + loss_bbox + loss_giou + aux_losses
        
        # Update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Training Tips

1. **Warmup**: Start with a small learning rate and gradually increase
2. **Backbone**: Initially freeze the backbone, fine-tune later
3. **Batch Size**: DETR benefits from larger batch sizes (8-16)
4. **Epochs**: Typically requires 300-500 epochs to converge
5. **Data Augmentation**: Use strong augmentation (random crop, color jitter, etc.)
6. **Gradient Clipping**: Clip gradients to prevent instability (max_norm=0.1)

### Hyperparameters (from paper)

```python
# Original DETR-DC5 (dilated C5) configuration
config = {
    'hidden_dim': 256,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'ffn_dim': 2048,
    'dropout': 0.1,
    'num_queries': 100,
    
    # Training
    'learning_rate': 1e-4,
    'lr_backbone': 1e-5,  # Lower LR for backbone
    'weight_decay': 1e-4,
    'epochs': 300,
    'batch_size': 2,  # Per GPU
    'gradient_clip': 0.1,
    
    # Loss weights
    'weight_class': 1.0,
    'weight_bbox': 5.0,
    'weight_giou': 2.0,
}
```

## Model Components

### 1. Backbone

```python
# The backbone extracts features from the input image
# Default: ResNet-50 without the final classification layers
# Output: Feature map of size H/32 × W/32 × 2048
```

**Purpose**: Extract hierarchical visual features

**Options**:
- Use pre-trained weights (recommended)
- Freeze initially, fine-tune later
- Can be replaced with any CNN (ResNet, EfficientNet, etc.)

### 2. Input Projection

```python
# Projects backbone features to transformer dimension
# 1×1 convolution: 2048 → hidden_dim (256)
```

**Purpose**: Adapt CNN features to transformer dimension

### 3. Positional Encoding

```python
# 2D sinusoidal positional encoding
# Encodes spatial position information
# Added to the projected features
```

**Purpose**: Provide spatial awareness to the transformer

**Details**:
- Uses sine and cosine functions of different frequencies
- Separate encodings for x and y coordinates
- Concatenated to form full positional encoding

### 4. Transformer Encoder

```python
# Stack of N transformer layers (default: 6)
# Each layer: Self-Attention → FFN
# Enables global reasoning over image features
```

**Purpose**: Process and refine image features with global context

**Key Points**:
- Self-attention allows every position to attend to every other position
- Enables long-range dependencies
- Pre-normalization architecture

### 5. Transformer Decoder

```python
# Stack of N transformer layers (default: 6)
# Each layer: Self-Attention → Cross-Attention → FFN
# Object queries attend to encoder memory
```

**Purpose**: Transform object queries into object detections

**Key Points**:
- Object queries: Learned embeddings (one per potential detection)
- Self-attention: Queries interact with each other
- Cross-attention: Queries attend to image features
- Outputs predictions for each query

### 6. Prediction Heads

```python
# Class head: Linear layer → (num_classes + 1)
# Bbox head: 3-layer MLP → 4 (x, y, w, h)
```

**Purpose**: Convert decoder outputs to class and box predictions

**Details**:
- Class prediction: Includes "no object" class
- Box prediction: Normalized coordinates [0, 1]
- Applied to all decoder layer outputs for auxiliary losses

## Performance Considerations

### Computational Complexity

DETR is computationally expensive:

| Component | Complexity | Notes |
|-----------|------------|-------|
| Backbone | O(HW) | Linear in image size |
| Encoder Self-Attention | O((HW)²) | Quadratic in sequence length |
| Decoder Self-Attention | O(N²) | Quadratic in num_queries |
| Decoder Cross-Attention | O(N × HW) | Queries attend to all positions |

**Memory Requirements**:
- Encoder attention: (H/32 × W/32)² attention map
- For 800×1333 images: ~25×42 = 1050 positions
- Attention matrix: 1050² ≈ 1M elements per head

### Optimization Strategies

1. **Reduce Image Resolution**
```python
# Use smaller input images
target_size = (600, 800)  # Instead of (800, 1333)
```

2. **Reduce Number of Encoder Layers**
```python
model = create_detr(
    num_classes=80,
    num_queries=100,
    num_encoder_layers=3,  # Instead of 6
    num_decoder_layers=6
)
```

3. **Use Efficient Normalization**
```python
model = create_detr(
    num_classes=80,
    num_queries=100,
    normalization_type="rms_norm"  # Faster than layer_norm
)
```

4. **Mixed Precision Training**
```python
keras.mixed_precision.set_global_policy('mixed_float16')
```

5. **Gradient Checkpointing**
- Recompute activations during backward pass
- Trades computation for memory

### Inference Speed

Typical inference times (single image, V100 GPU):
- **DETR-R50**: ~100ms per image
- **DETR-R101**: ~150ms per image
- **Faster R-CNN**: ~50ms per image (for comparison)

DETR is not designed for real-time applications. For faster inference:
- Use smaller backbone
- Reduce number of layers
- Reduce image resolution
- Consider DETR variants (Deformable DETR, Conditional DETR)

## Troubleshooting

### Common Issues

#### 1. Model Not Converging

**Symptoms**: Loss stays high, no detections made

**Solutions**:
```python
# Increase training duration
epochs = 500  # DETR needs many epochs

# Check learning rate
lr = 1e-4  # Original paper value

# Ensure auxiliary losses are enabled
model = create_detr(..., aux_loss=True)

# Use gradient clipping
# (in custom training loop)
```

#### 2. Out of Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:
```python
# Reduce batch size
batch_size = 1  # Start small

# Reduce image resolution
max_size = 800  # Instead of 1333

# Reduce model size
model = create_detr(
    hidden_dim=128,        # Instead of 256
    num_encoder_layers=3,  # Instead of 6
    ffn_dim=1024          # Instead of 2048
)

# Use gradient accumulation
accumulation_steps = 4  # Simulate larger batch
```

#### 3. Slow Training

**Symptoms**: Training takes too long

**Solutions**:
```python
# Use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Reduce encoder layers (most expensive)
model = create_detr(..., num_encoder_layers=3)

# Use RMSNorm (faster than LayerNorm)
model = create_detr(..., normalization_type="rms_norm")

# Freeze backbone initially
model = create_detr(..., backbone_trainable=False)
```

#### 4. Poor Performance on Small Objects

**Symptoms**: Missing small objects in predictions

**Solutions**:
- DETR struggles with small objects inherently
- Use multi-scale features (not in basic DETR)
- Consider Deformable DETR for better small object detection
- Increase input resolution
- Use stronger data augmentation with crops

#### 5. Serialization Errors

**Symptoms**: Cannot save/load model

**Solutions**:
```python
# Ensure all custom layers are registered
@keras.saving.register_keras_serializable()
class MyLayer(keras.layers.Layer):
    ...

# Use .keras format
model.save("model.keras")  # Not .h5

# Check get_config() completeness
config = model.get_config()
print(config.keys())  # Should include all __init__ params
```

### Debugging Tips

```python
# Check model summary
model.summary()

# Inspect intermediate outputs
encoder_output = model.transformer.encoder_layers[0](features)
print(f"Encoder output shape: {encoder_output.shape}")

# Visualize attention weights (requires modification)
# Add return_attention_weights=True to attention layers

# Check gradient flow
with tf.GradientTape() as tape:
    outputs = model([images, masks], training=True)
    loss = compute_loss(outputs, targets)
gradients = tape.gradient(loss, model.trainable_variables)

# Check for None gradients
for var, grad in zip(model.trainable_variables, gradients):
    if grad is None:
        print(f"No gradient for {var.name}")
```

## Advanced Topics

### 1. Panoptic Segmentation

DETR can be extended for panoptic segmentation by adding a mask head:

```python
# Add mask prediction head after decoder
mask_head = keras.Sequential([
    keras.layers.Dense(hidden_dim, activation='relu'),
    keras.layers.Dense(hidden_dim, activation='relu'),
    keras.layers.Dense(num_masks)
])
```

### 2. Multi-Scale Features

Improve detection of objects at different scales:

```python
# Extract features from multiple backbone layers
features_c3 = backbone.get_layer('conv3_block4_out').output
features_c4 = backbone.get_layer('conv4_block6_out').output
features_c5 = backbone.get_layer('conv5_block3_out').output

# Combine multi-scale features (e.g., FPN-style)
```

### 3. Deformable Attention

Replace standard attention with deformable attention for efficiency:

```python
# Use sparse attention to key sampling locations
# Reduces complexity from O(HW) to O(K) where K is small
```

### 4. Conditional DETR

Add conditional spatial query to decoder for faster convergence:

```python
# Decoder cross-attention conditioned on content query
# Helps queries focus on relevant regions earlier in training
```

## Model Variants

### DETR-DC5
- Uses dilated convolutions in C5 stage of ResNet
- Larger feature maps (stride 16 instead of 32)
- Better performance but slower

### DETR-R50
- Standard ResNet-50 backbone
- 41.5 AP on COCO (300 epochs)

### DETR-R101
- ResNet-101 backbone
- 43.5 AP on COCO (300 epochs)
- Larger capacity, better performance

### Deformable DETR
- Uses deformable attention
- 10× faster convergence
- Better performance on small objects

### Conditional DETR
- Adds conditional spatial query
- Faster convergence than vanilla DETR
- Similar final performance

## Best Practices

### 1. Training Strategy

```python
# Phase 1: Train with frozen backbone (50 epochs)
model = create_detr(..., backbone_trainable=False)
train(model, epochs=50, lr=1e-4)

# Phase 2: Fine-tune entire model (250 epochs)
model.backbone.trainable = True
train(model, epochs=250, lr=1e-5)
```

### 2. Data Preprocessing

```python
# Normalize images to [0, 1]
image = image / 255.0

# Apply ImageNet normalization if using pre-trained backbone
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = (image - mean) / std

# Resize while maintaining aspect ratio
# Pad to fixed size or use masking
```

### 3. Data Augmentation

```python
# Recommended augmentations
augmentations = [
    RandomCrop(min_scale=0.3, max_scale=1.0),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    RandomGrayscale(p=0.1),
]
```

### 4. Evaluation

```python
# Use COCO evaluation metrics
from pycocotools.cocoeval import COCOeval

# Post-processing: Filter by confidence threshold
threshold = 0.7
predictions = filter_predictions(outputs, threshold)

# Compute metrics
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Key metrics:
# - AP (Average Precision @ IoU=0.50:0.95)
# - AP50 (Average Precision @ IoU=0.50)
# - AP75 (Average Precision @ IoU=0.75)
# - AP_small, AP_medium, AP_large
```

## Examples

### Complete Training Example

```python
import keras
import tensorflow as tf
from detr_refactored import create_detr

# Hyperparameters
CONFIG = {
    'num_classes': 80,
    'num_queries': 100,
    'hidden_dim': 256,
    'epochs': 300,
    'batch_size': 2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
}

# Create model
model = create_detr(
    num_classes=CONFIG['num_classes'],
    num_queries=CONFIG['num_queries'],
    hidden_dim=CONFIG['hidden_dim'],
    backbone_trainable=False  # Freeze initially
)

# Optimizer with weight decay
optimizer = keras.optimizers.AdamW(
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Custom training loop (pseudo-code)
@tf.function
def train_step(images, masks, targets):
    with tf.GradientTape() as tape:
        predictions = model([images, masks], training=True)
        
        # Compute bipartite matching
        indices = hungarian_matcher(predictions, targets)
        
        # Compute losses
        losses = compute_detr_losses(predictions, targets, indices)
        total_loss = sum(losses.values())
    
    # Update weights
    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 0.1)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return losses

# Training loop
for epoch in range(CONFIG['epochs']):
    for batch in train_dataset:
        losses = train_step(batch['images'], batch['masks'], batch['targets'])
    
    # Validation
    if epoch % 10 == 0:
        val_metrics = evaluate(model, val_dataset)
        print(f"Epoch {epoch}: AP = {val_metrics['AP']:.3f}")
    
    # Unfreeze backbone after 50 epochs
    if epoch == 50:
        model.backbone.trainable = True
        optimizer.learning_rate = 1e-5

# Save final model
model.save("detr_final.keras")
```

### Inference with Post-Processing

```python
import keras
import numpy as np
from PIL import Image

def predict_objects(model, image_path, confidence_threshold=0.7):
    """
    Predict objects in an image.
    
    Args:
        model: Trained DETR model
        image_path: Path to input image
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detections: [(class_id, confidence, bbox), ...]
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image) / 255.0
    
    # Resize to model input size
    h, w = image_array.shape[:2]
    target_h, target_w = 800, 1333
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    image_resized = tf.image.resize(image_array, (new_h, new_w))
    
    # Pad to target size
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    image_padded = tf.pad(
        image_resized,
        [[0, pad_h], [0, pad_w], [0, 0]],
        constant_values=0
    )
    
    # Create mask
    mask = np.zeros((target_h, target_w), dtype=bool)
    mask[new_h:, :] = True
    mask[:, new_w:] = True
    
    # Add batch dimension
    image_batch = np.expand_dims(image_padded, axis=0)
    mask_batch = np.expand_dims(mask, axis=0)
    
    # Predict
    predictions = model([image_batch, mask_batch], training=False)
    
    # Extract predictions
    class_logits = predictions['pred_logits'][0]  # (100, 81)
    boxes = predictions['pred_boxes'][0]           # (100, 4)
    
    # Get probabilities
    probs = tf.nn.softmax(class_logits, axis=-1)
    
    # Filter by confidence (exclude "no object" class)
    max_probs = tf.reduce_max(probs[:, :-1], axis=-1)
    class_ids = tf.argmax(probs[:, :-1], axis=-1)
    
    # Apply threshold
    keep = max_probs > confidence_threshold
    
    # Get detections
    detections = []
    for i in tf.where(keep):
        i = i[0].numpy()
        class_id = class_ids[i].numpy()
        confidence = max_probs[i].numpy()
        box = boxes[i].numpy()
        
        # Convert normalized coords to image coords
        x_center, y_center, width, height = box
        x1 = (x_center - width / 2) * w
        y1 = (y_center - height / 2) * h
        x2 = (x_center + width / 2) * w
        y2 = (y_center + height / 2) * h
        
        detections.append({
            'class_id': int(class_id),
            'confidence': float(confidence),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    
    return detections

# Use the function
model = keras.models.load_model("detr_trained.keras")
detections = predict_objects(model, "test_image.jpg")

for det in detections:
    print(f"Class {det['class_id']}: {det['confidence']:.2f} at {det['bbox']}")
```

## References

### Original Paper

```bibtex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

### Related Work

- **Deformable DETR** (2021): Improves convergence speed and small object detection
- **Conditional DETR** (2021): Faster convergence with conditional cross-attention
- **DINO** (2022): State-of-the-art DETR with contrastive denoising
- **ViT-DETR** (2022): Uses Vision Transformer instead of CNN backbone

### Resources

- **Original Implementation**: https://github.com/facebookresearch/detr
- **Paper**: https://arxiv.org/abs/2005.12872
- **Tutorial**: https://cocodataset.org/#detection-eval
- **Keras Documentation**: https://keras.io/guides/

## License

This implementation is provided for research and educational purposes. Please refer to the original DETR paper and implementation for licensing information.

## Contributing

Contributions are welcome! Please ensure:
- Code follows Keras 3 best practices
- Type hints are included
- Documentation is comprehensive
- Tests are provided

## Acknowledgments

- Original DETR authors at Facebook AI Research
- Keras team for the excellent framework
- dl_techniques framework contributors

---

**Questions or Issues?** Please open an issue on GitHub or refer to the troubleshooting section above.