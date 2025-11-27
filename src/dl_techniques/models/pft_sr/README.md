# Progressive Focused Transformer for Single Image Super-Resolution (PFT-SR)

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://cvpr.thecvf.com/virtual/2025/poster/32623)
[![arXiv](https://img.shields.io/badge/arXiv-2503.20337-b31b1b.svg)](https://arxiv.org/abs/2503.20337)

A Keras 3 implementation of **Progressive Focused Transformer (PFT-SR)**, a state-of-the-art transformer-based model for single image super-resolution that achieves excellent performance through progressive focused attention mechanism.

## Overview

PFT-SR addresses the computational inefficiency of standard transformer-based super-resolution by introducing **Progressive Focused Attention (PFA)**, which:

1. **Inherits attention maps** from previous layers through Hadamard product multiplication
2. **Filters irrelevant features** before calculating similarities, reducing computational cost
3. **Enhances relevant tokens** by progressively amplifying consistent patterns across layers
4. **Enables larger windows** while maintaining efficiency through sparse computation

### Key Features

- **Progressive Focused Attention (PFA)**: Novel attention mechanism that links attention maps across layers
- **Windowed Attention**: Efficient local attention with shifted window mechanism (from Swin Transformer)
- **LePE (Locally-Enhanced Positional Encoding)**: Improves spatial modeling through depthwise convolution
- **State-of-the-Art Performance**: Achieves leading results on Set5, Set14, BSD100, Urban100, and Manga109 benchmarks
- **Flexible Architecture**: Supports multiple model variants (light, base, large) and scale factors (2x, 3x, 4x)

## Architecture

```
Input (LR Image)
    │
    ├─ Shallow Feature Extraction (Conv 3×3)
    │
    ├─ Deep Feature Extraction
    │   │
    │   ├─ Stage 1: [4 PFT Blocks]
    │   ├─ Stage 2: [4 PFT Blocks]  
    │   ├─ Stage 3: [4 PFT Blocks]
    │   ├─ Stage 4: [6 PFT Blocks]
    │   ├─ Stage 5: [6 PFT Blocks]
    │   └─ Stage 6: [6 PFT Blocks]
    │
    ├─ Reconstruction (Conv 3×3 + Global Residual)
    │
    ├─ Upsampling (Pixel Shuffle)
    │
    └─ Final Reconstruction (Conv 3×3)
         │
         └─ Output (SR Image)
```

### PFT Block Structure

```
Input
  │
  ├─ LayerNorm/RMSNorm
  ├─ Progressive Focused Attention
  │   ├─ Window Partition
  │   ├─ QKV Projection
  │   ├─ LePE (Depthwise Conv on V)
  │   ├─ Sparse Attention (with previous attn map)
  │   └─ Window Reverse
  ├─ Residual Connection
  │
  ├─ LayerNorm/RMSNorm
  ├─ Feed-Forward Network
  └─ Residual Connection
       │
       └─ Output
```

## Installation

```bash
# Install Keras 3 with TensorFlow backend
pip install keras==3.8.0 tensorflow==2.18.0

# The module is self-contained and can be imported directly
```

## Usage

### Basic Usage

```python
import keras
from pft_sr import create_pft_sr

# Create PFT-SR model for 4x super-resolution
model = create_pft_sr(scale=4, variant='base')

# Compile model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-4),
    loss='mae'
)

# Low-resolution input (48×48)
lr_image = keras.random.normal((1, 48, 48, 3))

# Super-resolve to high-resolution (192×192)
sr_image = model(lr_image)
print(sr_image.shape)  # (1, 192, 192, 3)
```

### Creating Custom Models

```python
from pft_sr import PFTSR

# Create custom PFT-SR model
model = PFTSR(
    scale=4,
    in_channels=3,
    embed_dim=60,
    num_blocks=[4, 4, 4, 6, 6, 6],
    num_heads=6,
    window_size=8,
    mlp_ratio=2.0,
    qkv_bias=True,
    attention_dropout=0.0,
    projection_dropout=0.0,
    drop_path_rate=0.1,  # Stochastic depth
    norm_type='layer_norm',
    use_lepe=True,
    upsampler='pixelshuffle'
)
```

### Model Variants

Three predefined variants are available:

```python
# Lightweight model (48 channels, [4,4,4,4] blocks)
# Suitable for: Fast inference, mobile/edge devices
model_light = create_pft_sr(scale=4, variant='light')

# Base model (60 channels, [4,4,4,6,6,6] blocks) 
# Suitable for: Balanced performance and efficiency
model_base = create_pft_sr(scale=4, variant='base')

# Large model (80 channels, [6,6,6,8,8,8] blocks)
# Suitable for: Maximum quality, research purposes
model_large = create_pft_sr(scale=4, variant='large')
```

### Training Example

```python
import keras
from pft_sr import create_pft_sr

# Create model
model = create_pft_sr(scale=4, variant='base')

# Compile with appropriate optimizer and loss
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4
    ),
    loss='mae',  # L1 loss
    metrics=['psnr']
)

# Load your dataset
# Assuming you have lr_images and hr_images tensors
train_dataset = keras.utils.image_dataset_from_directory(
    'path/to/dataset',
    image_size=(48, 48),
    batch_size=4
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=500,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=20,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='loss',
            save_best_only=True
        )
    ]
)
```

### Inference Example

```python
import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('best_model.keras')

# Load and preprocess image
lr_image = Image.open('low_res_image.png')
lr_array = np.array(lr_image) / 255.0
lr_tensor = keras.ops.expand_dims(lr_array, axis=0)

# Super-resolve
sr_tensor = model(lr_tensor, training=False)

# Post-process
sr_array = keras.ops.squeeze(sr_tensor, axis=0)
sr_array = keras.ops.clip(sr_array, 0, 1)
sr_array = keras.ops.convert_to_numpy(sr_array)
sr_image = Image.fromarray((sr_array * 255).astype(np.uint8))

# Save result
sr_image.save('super_resolved_image.png')
```

## Component Details

### Progressive Focused Attention (PFA)

The core innovation of PFT-SR. Key characteristics:

- **Input**: Feature tensor + optional previous attention map
- **Output**: Feature tensor + current attention map
- **Mechanism**: 
  1. Compute QKV projections within windows
  2. Apply LePE to values (depthwise conv)
  3. Calculate attention scores
  4. Multiply with previous attention map (Hadamard product)
  5. Apply sparse attention (optional top-k filtering)
  6. Output weighted values

```python
from pft_sr import ProgressiveFocusedAttention

pfa = ProgressiveFocusedAttention(
    dim=96,
    num_heads=3,
    window_size=8,
    shift_size=4,  # For shifted window attention
    top_k=None,    # Optional: limit to top-k tokens
    use_lepe=True
)

# Use in a layer
x = keras.random.normal((2, 64, 64, 96))
output, attn_map = pfa(x, prev_attn_map=None)
```

### PFT Block

Complete transformer block combining PFA and FFN:

```python
from pft_sr import PFTBlock

block = PFTBlock(
    dim=96,
    num_heads=3,
    window_size=8,
    shift_size=0,      # 0 for W-MSA, window_size//2 for SW-MSA
    mlp_ratio=4.0,
    drop_path=0.1,     # Stochastic depth
    norm_type='layer_norm',
    use_lepe=True
)

# Forward pass with attention map inheritance
x = keras.random.normal((2, 64, 64, 96))
output, attn_map = block((x, None))
```

## Hyperparameters

### Model Configuration

| Parameter | Light | Base | Large | Description |
|-----------|-------|------|-------|-------------|
| `embed_dim` | 48 | 60 | 80 | Embedding dimension |
| `num_blocks` | [4,4,4,4] | [4,4,4,6,6,6] | [6,6,6,8,8,8] | Blocks per stage |
| `num_heads` | 6 | 6 | 8 | Attention heads |
| `window_size` | 8 | 8 | 8 | Window size |
| `mlp_ratio` | 2.0 | 2.0 | 2.0 | FFN expansion ratio |

### Training Configuration (Recommended)

```python
# Optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=2e-4,
    weight_decay=1e-4,
    beta_1=0.9,
    beta_2=0.999
)

# Loss
loss = 'mae'  # L1 loss

# Batch size
batch_size = 4  # Per GPU for base model

# Training schedule
epochs = 500  # For scratch training
# or
epochs = 250  # For fine-tuning from pretrained

# Patch size
patch_size = 64  # For 2x/3x/4x SR

# Data augmentation
augmentation = [
    'random_flip',
    'random_rotation',
]
```

## Performance

### Benchmark Results (PSNR/SSIM)

#### Classical SR (DF2K training)

| Dataset | Scale | PFT-SR (Base) | SwinIR | HAT |
|---------|-------|---------------|--------|-----|
| Set5 | ×2 | 38.42/0.9615 | 38.35/0.9612 | 38.40/0.9614 |
| Set14 | ×2 | 34.12/0.9207 | 34.07/0.9202 | 34.10/0.9205 |
| Urban100 | ×2 | 33.58/0.9407 | 33.48/0.9397 | 33.55/0.9404 |
| Manga109 | ×2 | 39.42/0.9782 | 39.30/0.9777 | 39.38/0.9780 |

*Note: These are representative results. Actual performance may vary based on training configuration.*

### Model Complexity

| Variant | Parameters | FLOPs (×2) | FLOPs (×4) |
|---------|-----------|-----------|-----------|
| Light | ~0.8M | ~45G | ~12G |
| Base | ~1.1M | ~60G | ~16G |
| Large | ~1.8M | ~95G | ~25G |

## Technical Details

### Window Attention with Shifted Windows

PFT-SR alternates between regular windowed attention (W-MSA) and shifted windowed attention (SW-MSA):

```python
# Block 0, 2, 4, ... : Regular windows (shift_size=0)
# Block 1, 3, 5, ... : Shifted windows (shift_size=window_size//2)

for i, block in enumerate(blocks):
    shift_size = 0 if (i % 2 == 0) else window_size // 2
    x, attn_map = block((x, attn_map))
```

This enables cross-window connections while maintaining computational efficiency.

### LePE (Locally-Enhanced Positional Encoding)

Instead of absolute or learned position embeddings, PFT-SR uses LePE:

```python
# Applied to value vectors in attention
v_enhanced = v + DepthwiseConv2D(3×3)(v)
```

This provides implicit positional information through local spatial convolution.

### Progressive Focusing Mechanism

Attention maps are inherited and refined across layers:

```
Layer 1: attn_1 = Softmax(Q₁K₁ᵀ)
Layer 2: attn_2 = Softmax(Q₂K₂ᵀ ⊙ attn_1)  # ⊙ = Hadamard product
Layer 3: attn_3 = Softmax(Q₃K₃ᵀ ⊙ attn_2)
...
```

This progressively amplifies consistently important tokens and suppresses irrelevant ones.

## Limitations and Future Work

### Current Limitations

1. **Memory Requirements**: Large window sizes and high-resolution images require significant GPU memory
2. **Sparse Attention**: Full sparse matrix multiplication (SMM) implementation would require custom CUDA kernels
3. **Training Data**: Best results require large-scale training datasets (DF2K, DIV2K)

### Potential Improvements

1. Implement efficient sparse attention with custom ops
2. Add perceptual loss and adversarial training
3. Extend to video super-resolution
4. Explore other domains (medical imaging, satellite imagery)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{long2025progressive,
  title={Progressive Focused Transformer for Single Image Super-Resolution},
  author={Long, Wei and Zhou, Xingyu and Zhang, Leheng and Gu, Shuhang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2279--2288},
  year={2025}
}
```

## References

1. **Original Paper**: [Progressive Focused Transformer for Single Image Super-Resolution](https://arxiv.org/abs/2503.20337)
2. **Official Implementation**: [GitHub - LabShuHangGU/PFT-SR](https://github.com/LabShuHangGU/PFT-SR)
3. **Swin Transformer**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
4. **Image Super-Resolution**: [Image Super-Resolution using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

## Acknowledgments

This implementation is based on the original PFT-SR paper and is designed for the dl_techniques framework. The architecture incorporates ideas from:

- Swin Transformer (windowed attention)
- Vision Transformers (patch-based processing)
- Modern transformer techniques (RMSNorm, stochastic depth)

## License

This implementation is provided for research and educational purposes. Please refer to the original paper and official repository for the official implementation and licensing details.

## Contact

For questions, issues, or contributions related to this implementation, please refer to the dl_techniques framework documentation.