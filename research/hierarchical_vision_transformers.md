# Deep Dive: Hierarchical MLP (hMLP) Stem for Vision Transformers

The hierarchical MLP (hMLP) stem is a novel patch pre-processing technique introduced in the paper "Three things everyone should know about Vision Transformers" by Touvron et al. This document provides a comprehensive explanation of the design, implementation, and benefits of this approach.

## 1. The Problem with Existing Stem Designs

Traditional Vision Transformers (ViTs) use a simple linear projection to transform image patches into embeddings:

```
Image → Split into 16×16 patches → Linear projection → Token embeddings
```

Several researchers have proposed replacing this linear projection with convolutional stems to improve performance. However, these convolutional stems have a critical limitation: **they're incompatible with masked self-supervised learning methods like BeiT**.

**Why?** Convolutional stems cause information leakage between patches. When some patches are masked during self-supervised learning, the convolutions that cross patch boundaries propagate information from unmasked patches to masked ones, undermining the masking mechanism.

## 2. The hMLP Stem Solution

The hierarchical MLP stem solves this problem by processing patches independently while still providing meaningful pre-processing:

1. **Independent processing**: Each patch is processed entirely separately from other patches
2. **Hierarchical structure**: Patches are processed in a progressive, hierarchical manner
3. **Masking compatibility**: Can be used with masked self-supervised learning

## 3. Architecture Details

The hMLP stem processes patches through a hierarchical sequence:

```
2×2 patches → 4×4 patches → 8×8 patches → 16×16 patches
```

At each level, the following operations are applied:
1. Linear projection (implemented as convolution with matching kernel size and stride)
2. Normalization (BatchNorm or LayerNorm)
3. GELU activation

![hMLP Stem Architecture](https://i.imgur.com/Dj5tTQY.png)

### Implementation Trick

Although the conceptual design processes patches independently, the implementation uses convolution operations with matching kernel size and stride to efficiently achieve the same result:

```python
# Stage 1: 4x4 patches
self.stage1_conv = keras.layers.Conv2D(filters=dim1, kernel_size=4, strides=4)

# Stage 2: 8x8 patches (4x4 + 2x2 processing)
self.stage2_conv = keras.layers.Conv2D(filters=dim1, kernel_size=2, strides=2)

# Stage 3: 16x16 patches (8x8 + 2x2 processing)
self.stage3_conv = keras.layers.Conv2D(filters=embed_dim, kernel_size=2, strides=2)
```

When the kernel size equals the stride, there's no overlap between patches, ensuring that each patch is processed independently.

## 4. Key Advantages

### 4.1 Compatible with Masked Self-Supervised Learning

The primary advantage of the hMLP stem is its compatibility with masked self-supervised learning methods:

```
                   ┌───────────────────────┐
                   │                       │
Normal stem:  Image → Split → Stem → Mask → Transformer → ...
                                 ✗
                   ┌───────────────────────┐
                   │                       │
hMLP stem:    Image → Split → Mask → Stem → Transformer → ...
                          OR        
              Image → Split → Stem → Mask → Transformer → ...
```

With hMLP, masking can be applied either before or after the stem with identical results, since patches are processed independently. This makes it compatible with methods like BeiT or MAE.

### 4.2 Performance Benefits

The hMLP stem provides performance improvements with minimal computational overhead:

- **Supervised learning**: On par with the best convolutional stems (~0.3% accuracy improvement)
- **Self-supervised learning (BeiT)**: +0.4% accuracy improvement over linear projection
- **Computational cost**: < 1% increase in FLOPs compared to standard ViT

### 4.3 Implementation Flexibility

- **Normalization**: Works with both BatchNorm and LayerNorm
- **Activation functions**: Compatible with GELU, ReLU, etc.
- **Resolution and patch size**: Can be adapted for different configurations

## 5. Performance Comparison

| Model | Stem Type | Training Method | ImageNet Top-1 |
|-------|-----------|----------------|----------------|
| ViT-B | Linear    | Supervised     | 82.2% |
| ViT-B | Conv Stem | Supervised     | 82.6% |
| ViT-B | hMLP Stem | Supervised     | 82.5% |
| ViT-B | Linear    | BeiT + FT      | 83.1% |
| ViT-B | Conv Stem | BeiT + FT      | 83.0% |
| ViT-B | hMLP Stem | BeiT + FT      | 83.4% |

The table shows that:
1. In supervised learning, hMLP performs comparably to convolutional stems
2. In self-supervised learning (BeiT+FT), hMLP significantly outperforms both linear and convolutional stems

## 6. Integration with Masked Self-Supervised Learning

When using hMLP stem with masked self-supervised learning methods like BeiT:

1. **Tokenization**: The image is first divided into non-overlapping patches
2. **Masking**: Random patches are selected to be masked (typically 40-60%)
3. **Stem Processing**: The hMLP stem processes all patches independently
4. **Transformer**: The transformer tries to predict the visual tokens of masked patches

This maintains the separation between masked and unmasked information that is critical for effective self-supervised learning with masked patch prediction.

## 7. Implementation Considerations

When implementing the hMLP stem, consider:

1. **Normalization type**: