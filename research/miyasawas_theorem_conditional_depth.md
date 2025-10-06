# Generalized Conditional Miyasawa's Theorem: Multi-Modal Conditioning Framework

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Protocol 1: Dense Image Conditioning](#protocol-1-dense-image-conditioning)
4. [Protocol 2: Discrete Class Conditioning](#protocol-2-discrete-class-conditioning)
5. [Unified Multi-Modal Architecture](#unified-multi-modal-architecture)
6. [Implementation Strategy](#implementation-strategy)
7. [Training Procedures](#training-procedures)
8. [Inference Strategies](#inference-strategies)
9. [Architectural Patterns](#architectural-patterns)
10. [Practical Considerations](#practical-considerations)

---

## Executive Summary

This document presents a unified framework for conditional denoising under Miyasawa's theorem, supporting multiple conditioning modalities:

- **Dense Conditioning**: High-dimensional signals (RGB images → depth maps)
- **Discrete Conditioning**: Low-dimensional signals (class labels → images)
- **Hybrid Conditioning**: Multiple modalities simultaneously

**Core Principle:** A single mathematical framework governs all conditioning types:

$$\hat{x}(y, c) = \mathbb{E}[x|y, c] = y + \sigma^2 \nabla_y \log p(y|c)$$

where $c$ can be any conditioning signal—discrete, dense, or composite.

**Applications:**
- Monocular depth estimation with RGB conditioning
- Class-conditional image generation and denoising
- Multi-modal depth estimation (RGB + semantic class)
- Cross-modal translation tasks

---

## Theoretical Foundation

### 1.1 The Generalized Conditional Theorem

**Setup:**
- Let $x \in \mathbb{R}^n$ be the clean target signal (depth map, image, etc.)
- Let $c \in \mathcal{C}$ be the conditioning variable where:
  - $c \in \mathbb{R}^m$ for dense conditioning (e.g., RGB images)
  - $c \in \{1, 2, ..., K\}$ for discrete conditioning (e.g., class labels)
  - $c = (c_1, c_2, ...)$ for multi-modal conditioning
- Let $y = x + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Key assumption**: Noise is independent of conditioning: $p(\epsilon|c) = p(\epsilon)$

**Theorem Statement:**

For the least-squares optimal conditional denoiser that minimizes $\mathbb{E}[\|\hat{x}(y, c) - x\|^2 | c]$:

$$\boxed{\hat{x}(y, c) = \mathbb{E}[x|y, c] = y + \sigma^2 \nabla_y \log p(y|c)}$$

**Proof Sketch:**

1. **Conditional expectation minimizes MSE:**
   $$\hat{x}_{\text{MSE}}(y, c) = \arg\min_{\hat{x}} \mathbb{E}_{x|y,c}[\|x - \hat{x}\|^2] = \mathbb{E}[x|y, c]$$

2. **Apply Bayes' rule:**
   $$p(x|y, c) = \frac{p(y|x, c)p(x|c)}{p(y|c)} = \frac{p(y|x)p(x|c)}{p(y|c)}$$
   (using noise independence)

3. **Compute gradient of marginal:**
   $$\nabla_y p(y|c) = \int \nabla_y p(y|x) p(x|c) \, dx$$
   $$= \int p(y|x) \frac{x-y}{\sigma^2} p(x|c) \, dx$$
   $$= \frac{p(y|c)}{\sigma^2} \mathbb{E}[x-y|y, c]$$

4. **Rearrange:**
   $$\nabla_y \log p(y|c) = \frac{1}{\sigma^2}(\mathbb{E}[x|y,c] - y)$$

### 1.2 Neural Network Interpretation

A neural network $D_\theta(y, c)$ trained with MSE loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{x, c, \epsilon}\left[\|D_\theta(x + \epsilon, c) - x\|^2\right]$$

learns two things simultaneously:

1. **Conditional expectation:**
   $$D_\theta(y, c) \approx \mathbb{E}[x|y, c]$$

2. **Conditional score function:**
   $$\nabla_y \log p(y|c) \approx \frac{D_\theta(y, c) - y}{\sigma^2}$$

**Critical insight:** The same theorem applies regardless of whether $c$ is:
- A discrete class label
- A dense RGB image
- A combination of multiple signals

The architectural differences lie in *how* we process and inject $c$, not in the underlying mathematics.

### 1.3 Conditioning Signal Taxonomy

| Type | Dimensionality | Examples | Processing Method |
|------|----------------|----------|-------------------|
| **Discrete** | Low ($K$ classes) | Class labels, categorical attributes | Embedding → Dense vector |
| **Dense** | High ($\mathbb{R}^m$) | Images, feature maps, spatial data | Encoder → Multi-scale features |
| **Hybrid** | Mixed | Image + class, depth + semantic | Parallel processing → Feature fusion |
| **Sequential** | Temporal | Video, time series | Temporal encoder → Context vector |

This document focuses on discrete, dense, and hybrid conditioning.

---

## Protocol 1: Dense Image Conditioning

### 2.1 Problem Formulation: Monocular Depth Estimation

**Task:** Estimate depth map from RGB image using a denoising formulation.

**Mathematical framing:**
- Target signal: $x = D_{\text{gt}}$ (ground truth depth map)
- Conditioning signal: $c = I_{\text{rgb}}$ (RGB image)
- Noisy observation: $y = D_{\text{gt}} + \epsilon$
- Goal: Learn $p(D|I_{\text{rgb}})$ via denoising

**Why this works:**
Traditional depth estimation learns $f: I_{\text{rgb}} \rightarrow D_{\text{gt}}$ directly. The denoising formulation learns:
$$D_\theta(D_{\text{noisy}}, I_{\text{rgb}}) \rightarrow D_{\text{gt}}$$

This implicitly captures uncertainty and multi-modal distributions (e.g., depth ambiguity).

### 2.2 Training Protocol

**Data requirements:**
- Paired dataset: $\{(I_{\text{rgb}}^{(i)}, D_{\text{gt}}^{(i)})\}_{i=1}^N$
- Examples: NYU Depth V2, KITTI, Matterport3D

**Training procedure:**

```python
for epoch in range(num_epochs):
    for rgb_image, depth_gt in dataloader:
        # 1. Sample noise level
        sigma = sample_noise_level(sigma_min, sigma_max)
        
        # 2. Add noise to ground truth depth
        noise = tf.random.normal(depth_gt.shape) * sigma
        depth_noisy = depth_gt + noise
        
        # 3. Forward pass: denoise conditioned on RGB
        depth_pred = model([depth_noisy, rgb_image])
        
        # 4. Compute MSE loss
        loss = mse_loss(depth_pred, depth_gt)
        
        # 5. Backpropagation
        optimizer.minimize(loss)
```

**Key considerations:**
- Depth maps are typically single-channel: $(H, W, 1)$
- RGB images are three-channel: $(H, W, 3)$
- Spatial dimensions must match or be aligned
- Depth normalization: typically $[0, 1]$ or standardized

### 2.3 Architectural Design

**High-level structure:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Dense Conditioning Path                  │
│                                                             │
│  RGB Image (H, W, 3)                                       │
│         │                                                  │
│         v                                                  │
│  ┌──────────────┐                                         │
│  │ Conditioning │  Features:                              │
│  │   Encoder    │  - Level 0: (H, W, F₀)                 │
│  │  (ResNet/    │  - Level 1: (H/2, W/2, F₁)             │
│  │   EfficientNet)│  - Level 2: (H/4, W/4, F₂)             │
│  └──────────────┘  - Level 3: (H/8, W/8, F₃)             │
│         │          - Level 4: (H/16, W/16, F₄)           │
│         v                                                  │
│  Multi-scale feature pyramid                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│                    Denoising U-Net Path                     │
│                                                             │
│  Noisy Depth (H, W, 1)                                     │
│         │                                                  │
│         v                                                  │
│  ┌─────────────┐   ┌──────────────┐                      │
│  │  Encoder    │<──│ Feature      │                      │
│  │  Level 0    │   │ Injection    │ (from RGB features)  │
│  └─────────────┘   └──────────────┘                      │
│         │                                                  │
│         v (downsample)                                     │
│  ┌─────────────┐   ┌──────────────┐                      │
│  │  Encoder    │<──│ Feature      │                      │
│  │  Level 1    │   │ Injection    │                      │
│  └─────────────┘   └──────────────┘                      │
│         │                                                  │
│        ...                                                 │
│         │                                                  │
│         v                                                  │
│  ┌─────────────┐   ┌──────────────┐                      │
│  │ Bottleneck  │<──│ Feature      │                      │
│  │             │   │ Injection    │                      │
│  └─────────────┘   └──────────────┘                      │
│         │                                                  │
│         v (upsample + skip connections)                    │
│  ┌─────────────┐   ┌──────────────┐                      │
│  │  Decoder    │<──│ Feature      │                      │
│  │  Level N    │   │ Injection    │                      │
│  └─────────────┘   └──────────────┘                      │
│         │                                                  │
│         v                                                  │
│  Predicted Depth (H, W, 1)                                │
└─────────────────────────────────────────────────────────────┘
```

**Feature injection mechanism:**

```python
def inject_dense_conditioning(
    depth_features: tf.Tensor,  # (B, H, W, C_depth)
    rgb_features: tf.Tensor,    # (B, H, W, C_rgb)
    method: str = 'addition'
) -> tf.Tensor:
    """
    Inject RGB features into depth processing path.
    
    Methods:
    - 'addition': Element-wise addition after projection
    - 'concatenation': Channel-wise concatenation
    - 'film': Feature-wise linear modulation
    """
    
    if method == 'addition':
        # Project RGB features to match depth channel count
        projected = BiasFreeConv2D(
            filters=depth_features.shape[-1],
            kernel_size=1,
            use_bias=False
        )(rgb_features)
        
        return layers.Add()([depth_features, projected])
    
    elif method == 'concatenation':
        # Concatenate and process
        combined = layers.Concatenate()([depth_features, rgb_features])
        
        return BiasFreeConv2D(
            filters=depth_features.shape[-1],
            kernel_size=3,
            use_bias=False
        )(combined)
    
    elif method == 'film':
        # FiLM-style modulation (bias-free version)
        scale = BiasFreeConv2D(
            filters=depth_features.shape[-1],
            kernel_size=1,
            use_bias=False
        )(rgb_features)
        
        # Multiplicative modulation only (no shift to maintain bias-free)
        return layers.Multiply()([depth_features, scale + 1.0])
```

### 2.4 Complete Implementation

```python
def create_depth_estimation_denoiser(
    depth_shape: Tuple[int, int, int] = (256, 256, 1),
    rgb_shape: Tuple[int, int, int] = (256, 256, 3),
    depth_unet_depth: int = 4,
    depth_unet_filters: int = 64,
    conditioning_encoder_type: str = 'resnet34',
    injection_method: str = 'addition',
    enable_deep_supervision: bool = False
) -> keras.Model:
    """
    Create depth estimation denoiser with RGB image conditioning.
    
    Args:
        depth_shape: Shape of depth maps (H, W, 1)
        rgb_shape: Shape of RGB images (H, W, 3)
        depth_unet_depth: Depth of denoising U-Net
        depth_unet_filters: Base number of filters
        conditioning_encoder_type: RGB encoder architecture
        injection_method: How to inject RGB features
        enable_deep_supervision: Multi-scale supervision
    
    Returns:
        Keras model with inputs [noisy_depth, rgb_image]
    """
    
    # =====================================================================
    # INPUTS
    # =====================================================================
    
    noisy_depth_input = keras.Input(
        shape=depth_shape,
        name='noisy_depth_input'
    )
    rgb_condition_input = keras.Input(
        shape=rgb_shape,
        name='rgb_condition_input'
    )
    
    # =====================================================================
    # CONDITIONING ENCODER (RGB → Multi-scale features)
    # =====================================================================
    
    rgb_features_per_level = create_conditioning_encoder(
        rgb_input=rgb_condition_input,
        encoder_type=conditioning_encoder_type,
        num_levels=depth_unet_depth + 1,
        bias_free=True
    )
    # Returns: List[Tensor] with shapes:
    #   [(H, W, F₀), (H/2, W/2, F₁), (H/4, W/4, F₂), ...]
    
    # =====================================================================
    # DENOISING U-NET (Depth with RGB injection)
    # =====================================================================
    
    x = noisy_depth_input
    skip_connections = []
    filter_sizes = [
        depth_unet_filters * (2 ** i)
        for i in range(depth_unet_depth + 1)
    ]
    
    # --- Encoder ---
    for level in range(depth_unet_depth):
        current_filters = filter_sizes[level]
        
        # Inject RGB features at this level
        x = inject_dense_conditioning(
            depth_features=x,
            rgb_features=rgb_features_per_level[level],
            method=injection_method
        )
        
        # Bias-free convolution blocks
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        skip_connections.append(x)
        
        # Downsample
        if level < depth_unet_depth - 1:
            x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    # --- Bottleneck ---
    bottleneck_filters = filter_sizes[depth_unet_depth]
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    x = inject_dense_conditioning(
        depth_features=x,
        rgb_features=rgb_features_per_level[depth_unet_depth],
        method=injection_method
    )
    
    for _ in range(2):
        x = BiasFreeResidualBlock(
            filters=bottleneck_filters,
            kernel_size=3
        )(x)
    
    # --- Decoder ---
    deep_supervision_outputs = []
    
    for level in range(depth_unet_depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        
        # Upsample
        x = keras.layers.UpSampling2D(size=2)(x)
        
        # Concatenate skip connection
        skip = skip_connections[level]
        x = keras.layers.Concatenate()([skip, x])
        
        # Inject RGB features
        x = inject_dense_conditioning(
            depth_features=x,
            rgb_features=rgb_features_per_level[level],
            method=injection_method
        )
        
        # Convolution blocks
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        # Deep supervision output
        if enable_deep_supervision and level > 0:
            ds_output = BiasFreeConv2D(
                filters=1,
                kernel_size=1,
                activation='linear',
                use_bias=False
            )(x)
            deep_supervision_outputs.append(ds_output)
    
    # --- Final output ---
    final_output = BiasFreeConv2D(
        filters=1,
        kernel_size=1,
        activation='linear',
        use_bias=False,
        name='final_output'
    )(x)
    
    # =====================================================================
    # MODEL CREATION
    # =====================================================================
    
    if enable_deep_supervision and deep_supervision_outputs:
        all_outputs = [final_output] + list(reversed(deep_supervision_outputs))
        model = keras.Model(
            inputs=[noisy_depth_input, rgb_condition_input],
            outputs=all_outputs,
            name='depth_estimation_denoiser'
        )
    else:
        model = keras.Model(
            inputs=[noisy_depth_input, rgb_condition_input],
            outputs=final_output,
            name='depth_estimation_denoiser'
        )
    
    return model
```

### 2.5 Conditioning Encoder Design

The conditioning encoder extracts multi-scale features from RGB images:

```python
def create_conditioning_encoder(
    rgb_input: keras.layers.Layer,
    encoder_type: str = 'resnet34',
    num_levels: int = 5,
    bias_free: bool = True
) -> List[keras.layers.Layer]:
    """
    Create multi-scale feature extractor for RGB conditioning.
    
    Returns list of feature tensors at different scales.
    """
    
    if encoder_type.startswith('resnet'):
        # Use pre-defined ResNet architecture (modified for bias-free)
        backbone = create_bias_free_resnet(
            input_tensor=rgb_input,
            variant=encoder_type
        )
        
        # Extract features at multiple depths
        feature_layers = [
            'conv1_relu',        # Level 0: original resolution
            'conv2_block3_out',  # Level 1: 1/2 resolution
            'conv3_block4_out',  # Level 2: 1/4 resolution
            'conv4_block6_out',  # Level 3: 1/8 resolution
            'conv5_block3_out',  # Level 4: 1/16 resolution
        ]
        
        features = [
            backbone.get_layer(layer_name).output
            for layer_name in feature_layers[:num_levels]
        ]
        
    elif encoder_type == 'custom':
        # Custom bias-free encoder
        features = []
        x = rgb_input
        
        for level in range(num_levels):
            filters = 64 * (2 ** level)
            
            # Convolution block
            for _ in range(2):
                x = BiasFreeConv2D(
                    filters=filters,
                    kernel_size=3,
                    use_bias=False
                )(x)
                x = keras.layers.Activation('relu')(x)
            
            features.append(x)
            
            # Downsample for next level
            if level < num_levels - 1:
                x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    return features
```

---

## Protocol 2: Discrete Class Conditioning

### 3.1 Problem Formulation: Class-Conditional Image Generation

**Task:** Generate or denoise images conditioned on discrete class labels.

**Mathematical framing:**
- Target signal: $x = I_{\text{gt}}$ (ground truth image)
- Conditioning signal: $c = c_{\text{class}} \in \{1, 2, ..., K\}$
- Noisy observation: $y = I_{\text{gt}} + \epsilon$
- Goal: Learn $p(I|c_{\text{class}})$ via denoising

### 3.2 Training Protocol

**Data requirements:**
- Labeled dataset: $\{(I^{(i)}, c^{(i)})\}_{i=1}^N$
- Examples: ImageNet, CIFAR, custom labeled datasets

**Training procedure:**

```python
for epoch in range(num_epochs):
    for image, class_label in dataloader:
        # 1. Sample noise level
        sigma = sample_noise_level(sigma_min, sigma_max)
        
        # 2. Add noise to image
        noise = tf.random.normal(image.shape) * sigma
        image_noisy = image + noise
        
        # 3. Optionally apply CFG dropout
        if enable_cfg and random.random() < cfg_dropout_prob:
            class_label = unconditional_token
        
        # 4. Forward pass
        image_pred = model([image_noisy, class_label])
        
        # 5. Compute MSE loss
        loss = mse_loss(image_pred, image)
        
        # 6. Backpropagation
        optimizer.minimize(loss)
```

### 3.3 Architectural Design

**Class embedding and injection:**

```python
def create_class_conditional_denoiser(
    image_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 10,
    unet_depth: int = 4,
    unet_filters: int = 64,
    class_embedding_dim: int = 128,
    injection_method: str = 'spatial_broadcast',
    enable_cfg_training: bool = True,
    enable_deep_supervision: bool = False
) -> keras.Model:
    """
    Create class-conditional image denoiser.
    
    Args:
        image_shape: Shape of images (H, W, C)
        num_classes: Number of classes (+ 1 for unconditional if CFG)
        unet_depth: Depth of U-Net
        unet_filters: Base number of filters
        class_embedding_dim: Dimension of class embeddings
        injection_method: 'spatial_broadcast' or 'channel_concat'
        enable_cfg_training: Reserve unconditional token
        enable_deep_supervision: Multi-scale supervision
    
    Returns:
        Keras model with inputs [noisy_image, class_label]
    """
    
    # =====================================================================
    # INPUTS
    # =====================================================================
    
    noisy_image_input = keras.Input(
        shape=image_shape,
        name='noisy_image_input'
    )
    class_label_input = keras.Input(
        shape=(1,),
        dtype='int32',
        name='class_label_input'
    )
    
    # =====================================================================
    # CLASS EMBEDDING
    # =====================================================================
    
    # Convert discrete class to dense vector
    class_embedding = keras.layers.Embedding(
        input_dim=num_classes,
        output_dim=class_embedding_dim,
        embeddings_initializer='he_normal',
        name='class_embedding'
    )(class_label_input)
    
    class_embedding = keras.layers.Flatten()(class_embedding)
    
    # =====================================================================
    # DENOISING U-NET WITH CLASS INJECTION
    # =====================================================================
    
    def inject_class_embedding(
        features: tf.Tensor,
        class_emb: tf.Tensor,
        level: int
    ) -> tf.Tensor:
        """Inject class embedding into feature tensor."""
        
        feature_channels = features.shape[-1]
        
        if injection_method == 'spatial_broadcast':
            # Project embedding to match channels
            projected = keras.layers.Dense(
                feature_channels,
                use_bias=False,
                name=f'class_project_level_{level}'
            )(class_emb)
            
            # Reshape for broadcasting: (B, 1, 1, C)
            projected = keras.layers.Reshape((1, 1, feature_channels))(projected)
            
            # Add to features (bias-free modulation)
            return keras.layers.Add()([features, projected])
        
        elif injection_method == 'channel_concat':
            # Project and tile spatially
            spatial_h, spatial_w = features.shape[1], features.shape[2]
            
            projected = keras.layers.Dense(
                unet_filters // 2,
                use_bias=False,
                name=f'class_project_level_{level}'
            )(class_emb)
            
            # Tile to spatial dimensions
            tiled = keras.layers.RepeatVector(spatial_h * spatial_w)(projected)
            tiled = keras.layers.Reshape(
                (spatial_h, spatial_w, unet_filters // 2)
            )(tiled)
            
            # Concatenate
            return keras.layers.Concatenate()([features, tiled])
    
    x = noisy_image_input
    skip_connections = []
    filter_sizes = [unet_filters * (2 ** i) for i in range(unet_depth + 1)]
    
    # --- Encoder ---
    for level in range(unet_depth):
        current_filters = filter_sizes[level]
        
        # Inject class embedding
        x = inject_class_embedding(x, class_embedding, level)
        
        # Convolution blocks
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        skip_connections.append(x)
        
        if level < unet_depth - 1:
            x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    # --- Bottleneck ---
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = inject_class_embedding(x, class_embedding, unet_depth)
    
    for _ in range(2):
        x = BiasFreeResidualBlock(
            filters=filter_sizes[unet_depth],
            kernel_size=3
        )(x)
    
    # --- Decoder ---
    deep_supervision_outputs = []
    
    for level in range(unet_depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Concatenate()([skip_connections[level], x])
        x = inject_class_embedding(x, class_embedding, level)
        
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        if enable_deep_supervision and level > 0:
            ds_output = BiasFreeConv2D(
                filters=image_shape[-1],
                kernel_size=1,
                activation='linear',
                use_bias=False
            )(x)
            deep_supervision_outputs.append(ds_output)
    
    # --- Final output ---
    final_output = BiasFreeConv2D(
        filters=image_shape[-1],
        kernel_size=1,
        activation='linear',
        use_bias=False,
        name='final_output'
    )(x)
    
    # =====================================================================
    # MODEL CREATION
    # =====================================================================
    
    if enable_deep_supervision and deep_supervision_outputs:
        all_outputs = [final_output] + list(reversed(deep_supervision_outputs))
        model = keras.Model(
            inputs=[noisy_image_input, class_label_input],
            outputs=all_outputs,
            name='class_conditional_denoiser'
        )
    else:
        model = keras.Model(
            inputs=[noisy_image_input, class_label_input],
            outputs=final_output,
            name='class_conditional_denoiser'
        )
    
    return model
```

---

## Unified Multi-Modal Architecture

### 4.1 Problem Formulation: Class-Specific Depth Estimation

**Task:** Estimate depth maps from RGB images with additional class-level priors.

**Use cases:**
- **Object-specific depth**: Cars have different depth profiles than trees
- **Scene-specific depth**: Indoor vs outdoor depth characteristics
- **Multi-task learning**: Joint depth estimation and semantic segmentation

**Mathematical framing:**
- Target: $x = D_{\text{gt}}$ (depth map)
- Conditioning: $c = (I_{\text{rgb}}, c_{\text{class}})$ (image + class)
- Noisy observation: $y = D_{\text{gt}} + \epsilon$
- Goal: Learn $p(D|I_{\text{rgb}}, c_{\text{class}})$

### 4.2 Architectural Synthesis

The unified model combines both conditioning pathways:

```
                    ┌──────────────────────┐
                    │   RGB Image Input    │
                    └──────────────────────┘
                              │
                              v
                    ┌──────────────────────┐
                    │  RGB Feature         │
                    │  Encoder             │
                    │  (Dense Conditioning)│
                    └──────────────────────┘
                              │
                              │ Multi-scale features
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            v                                   v
┌────────────────────┐            ┌────────────────────┐
│ Noisy Depth Input  │            │ Class Label Input  │
└────────────────────┘            └────────────────────┘
            │                                   │
            │                                   v
            │                     ┌────────────────────┐
            │                     │ Class Embedding    │
            │                     │ (Discrete          │
            │                     │  Conditioning)     │
            │                     └────────────────────┘
            │                                   │
            └───────────┬───────────────────────┘
                        │
                        v
            ┌───────────────────────┐
            │   Denoising U-Net     │
            │                       │
            │  Dual Injection:      │
            │  • RGB features       │
            │  • Class embedding    │
            └───────────────────────┘
                        │
                        v
            ┌───────────────────────┐
            │  Predicted Depth Map  │
            └───────────────────────┘
```

### 4.3 Complete Implementation

```python
def create_unified_conditional_denoiser(
    depth_shape: Tuple[int, int, int] = (256, 256, 1),
    rgb_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 10,
    unet_depth: int = 4,
    unet_filters: int = 64,
    class_embedding_dim: int = 128,
    conditioning_encoder_type: str = 'resnet34',
    dense_injection_method: str = 'addition',
    discrete_injection_method: str = 'spatial_broadcast',
    enable_deep_supervision: bool = False
) -> keras.Model:
    """
    Create unified denoiser with both dense and discrete conditioning.
    
    This model learns p(depth | rgb_image, class_label), combining:
    - Dense conditioning: Multi-scale RGB features
    - Discrete conditioning: Class embedding vectors
    
    Args:
        depth_shape: Shape of depth maps
        rgb_shape: Shape of RGB images
        num_classes: Number of semantic classes
        unet_depth: Depth of denoising U-Net
        unet_filters: Base number of filters
        class_embedding_dim: Dimension of class embeddings
        conditioning_encoder_type: RGB encoder architecture
        dense_injection_method: How to inject RGB features
        discrete_injection_method: How to inject class embeddings
        enable_deep_supervision: Multi-scale supervision
    
    Returns:
        Keras model with inputs [noisy_depth, rgb_image, class_label]
    """
    
    # =====================================================================
    # INPUTS
    # =====================================================================
    
    noisy_depth_input = keras.Input(
        shape=depth_shape,
        name='noisy_depth_input'
    )
    rgb_condition_input = keras.Input(
        shape=rgb_shape,
        name='rgb_condition_input'
    )
    class_label_input = keras.Input(
        shape=(1,),
        dtype='int32',
        name='class_label_input'
    )
    
    # =====================================================================
    # CONDITIONING PATHWAYS
    # =====================================================================
    
    # 1. Dense conditioning: RGB → Multi-scale features
    rgb_features_per_level = create_conditioning_encoder(
        rgb_input=rgb_condition_input,
        encoder_type=conditioning_encoder_type,
        num_levels=unet_depth + 1,
        bias_free=True
    )
    
    # 2. Discrete conditioning: Class → Embedding
    class_embedding = keras.layers.Embedding(
        input_dim=num_classes,
        output_dim=class_embedding_dim,
        embeddings_initializer='he_normal',
        name='class_embedding'
    )(class_label_input)
    class_embedding = keras.layers.Flatten()(class_embedding)
    
    # =====================================================================
    # DUAL INJECTION HELPER
    # =====================================================================
    
    def inject_dual_conditioning(
        depth_features: tf.Tensor,
        rgb_features: tf.Tensor,
        class_emb: tf.Tensor,
        level: int
    ) -> tf.Tensor:
        """
        Inject both RGB features and class embedding.
        
        Order matters:
        1. First inject dense RGB features (spatial information)
        2. Then inject discrete class embedding (semantic information)
        """
        
        # Step 1: Dense injection (RGB features)
        x = inject_dense_conditioning(
            depth_features=depth_features,
            rgb_features=rgb_features,
            method=dense_injection_method
        )
        
        # Step 2: Discrete injection (class embedding)
        feature_channels = x.shape[-1]
        
        if discrete_injection_method == 'spatial_broadcast':
            projected = keras.layers.Dense(
                feature_channels,
                use_bias=False,
                name=f'class_project_level_{level}'
            )(class_emb)
            
            projected = keras.layers.Reshape((1, 1, feature_channels))(projected)
            x = keras.layers.Add()([x, projected])
        
        elif discrete_injection_method == 'channel_concat':
            spatial_h, spatial_w = x.shape[1], x.shape[2]
            
            projected = keras.layers.Dense(
                unet_filters // 4,  # Smaller to balance parameters
                use_bias=False,
                name=f'class_project_level_{level}'
            )(class_emb)
            
            tiled = keras.layers.RepeatVector(spatial_h * spatial_w)(projected)
            tiled = keras.layers.Reshape(
                (spatial_h, spatial_w, unet_filters // 4)
            )(tiled)
            
            x = keras.layers.Concatenate()([x, tiled])
        
        return x
    
    # =====================================================================
    # UNIFIED DENOISING U-NET
    # =====================================================================
    
    x = noisy_depth_input
    skip_connections = []
    filter_sizes = [unet_filters * (2 ** i) for i in range(unet_depth + 1)]
    
    # --- Encoder ---
    for level in range(unet_depth):
        current_filters = filter_sizes[level]
        
        # Dual injection: RGB features + class embedding
        x = inject_dual_conditioning(
            depth_features=x,
            rgb_features=rgb_features_per_level[level],
            class_emb=class_embedding,
            level=level
        )
        
        # Convolution blocks
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        skip_connections.append(x)
        
        if level < unet_depth - 1:
            x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    # --- Bottleneck ---
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    x = inject_dual_conditioning(
        depth_features=x,
        rgb_features=rgb_features_per_level[unet_depth],
        class_emb=class_embedding,
        level=unet_depth
    )
    
    for _ in range(2):
        x = BiasFreeResidualBlock(
            filters=filter_sizes[unet_depth],
            kernel_size=3
        )(x)
    
    # --- Decoder ---
    deep_supervision_outputs = []
    
    for level in range(unet_depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Concatenate()([skip_connections[level], x])
        
        x = inject_dual_conditioning(
            depth_features=x,
            rgb_features=rgb_features_per_level[level],
            class_emb=class_embedding,
            level=level
        )
        
        for _ in range(2):
            x = BiasFreeResidualBlock(
                filters=current_filters,
                kernel_size=3
            )(x)
        
        if enable_deep_supervision and level > 0:
            ds_output = BiasFreeConv2D(
                filters=1,
                kernel_size=1,
                activation='linear',
                use_bias=False
            )(x)
            deep_supervision_outputs.append(ds_output)
    
    # --- Final output ---
    final_output = BiasFreeConv2D(
        filters=1,
        kernel_size=1,
        activation='linear',
        use_bias=False,
        name='final_output'
    )(x)
    
    # =====================================================================
    # MODEL CREATION
    # =====================================================================
    
    if enable_deep_supervision and deep_supervision_outputs:
        all_outputs = [final_output] + list(reversed(deep_supervision_outputs))
        model = keras.Model(
            inputs=[noisy_depth_input, rgb_condition_input, class_label_input],
            outputs=all_outputs,
            name='unified_conditional_denoiser'
        )
    else:
        model = keras.Model(
            inputs=[noisy_depth_input, rgb_condition_input, class_label_input],
            outputs=final_output,
            name='unified_conditional_denoiser'
        )
    
    return model
```

### 4.4 Training the Unified Model

```python
# Dataset: (rgb_image, depth_gt, class_label) triples
for rgb, depth_gt, class_label in dataset:
    # Add noise to depth
    sigma = sample_noise_level()
    depth_noisy = depth_gt + tf.random.normal(depth_gt.shape) * sigma
    
    # Forward pass with both conditioning signals
    depth_pred = unified_model([depth_noisy, rgb, class_label])
    
    # MSE loss
    loss = mse_loss(depth_pred, depth_gt)
    
    # Backpropagation
    optimizer.minimize(loss)
```

---

## Implementation Strategy

### 5.1 Modular Design Pattern

**Core principle:** Build reusable components that can be composed.

```python
# Base components
class BiasFreeConv2D(keras.layers.Layer):
    """Bias-free convolution layer."""
    pass

class BiasFreeResidualBlock(keras.layers.Layer):
    """Bias-free residual block."""
    pass

# Conditioning components
class DenseConditioningEncoder(keras.layers.Layer):
    """Extracts multi-scale features from dense signals."""
    pass

class DiscreteConditioningEmbedding(keras.layers.Layer):
    """Converts discrete labels to embeddings."""
    pass

# Injection mechanisms
class SpatialBroadcastInjection(keras.layers.Layer):
    """Injects conditioning via spatial broadcasting."""
    pass

class ChannelConcatenationInjection(keras.layers.Layer):
    """Injects conditioning via channel concatenation."""
    pass

# Composable U-Net
class ConditionalBiasFreeUNet(keras.Model):
    """
    U-Net that accepts arbitrary conditioning signals.
    
    Supports:
    - No conditioning
    - Dense conditioning only
    - Discrete conditioning only
    - Multi-modal conditioning
    """
    
    def __init__(
        self,
        dense_conditioning: bool = False,
        discrete_conditioning: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_conditioning = dense_conditioning
        self.discrete_conditioning = discrete_conditioning
        # Build appropriate injection modules
```

### 5.2 Configuration System

```python
@dataclass
class ConditioningConfig:
    """Configuration for conditioning mechanisms."""
    
    # Dense conditioning (e.g., RGB images)
    enable_dense: bool = False
    dense_encoder_type: str = 'resnet34'
    dense_injection_method: str = 'addition'
    dense_injection_levels: List[int] = field(default_factory=list)
    
    # Discrete conditioning (e.g., class labels)
    enable_discrete: bool = False
    num_classes: int = 10
    embedding_dim: int = 128
    discrete_injection_method: str = 'spatial_broadcast'
    discrete_injection_levels: List[int] = field(default_factory=list)
    
    # CFG for discrete conditioning
    enable_cfg: bool = False
    cfg_dropout_prob: float = 0.1

def create_conditional_denoiser(
    target_shape: Tuple[int, int, int],
    conditioning_config: ConditioningConfig,
    unet_config: UNetConfig,
    **kwargs
) -> keras.Model:
    """
    Factory function that creates appropriate model based on config.
    """
    
    if conditioning_config.enable_dense and conditioning_config.enable_discrete:
        return create_unified_conditional_denoiser(...)
    elif conditioning_config.enable_dense:
        return create_depth_estimation_denoiser(...)
    elif conditioning_config.enable_discrete:
        return create_class_conditional_denoiser(...)
    else:
        return create_unconditional_denoiser(...)
```

---

## Training Procedures

### 6.1 Noise Level Scheduling

**Importance:** Model must handle varying noise levels for robust inference.

**Strategies:**

1. **Uniform sampling:**
   ```python
   sigma = tf.random.uniform([], sigma_min, sigma_max)
   ```

2. **Log-uniform sampling:**
   ```python
   log_sigma = tf.random.uniform(
       [], 
       tf.math.log(sigma_min), 
       tf.math.log(sigma_max)
   )
   sigma = tf.exp(log_sigma)
   ```

3. **Curriculum learning:**
   ```python
   # Start with high noise, gradually decrease
   epoch_progress = epoch / total_epochs
   sigma_max_current = sigma_max * (1 - 0.5 * epoch_progress)
   sigma = tf.random.uniform([], sigma_min, sigma_max_current)
   ```

**Recommendation:** Log-uniform for general purpose, uniform for specific noise ranges.

### 6.2 Multi-Scale Training

For models with deep supervision:

```python
def compute_multi_scale_loss(
    predictions: List[tf.Tensor],
    targets: List[tf.Tensor],
    weights: List[float]
) -> tf.Tensor:
    """
    Compute weighted sum of losses at multiple scales.
    """
    total_loss = 0.0
    
    for pred, target, weight in zip(predictions, targets, weights):
        # Resize target if needed
        if pred.shape != target.shape:
            target = tf.image.resize(target, pred.shape[1:3])
        
        scale_loss = tf.reduce_mean(tf.square(pred - target))
        total_loss += weight * scale_loss
    
    return total_loss

# Weight scheduling
def get_supervision_weights(epoch, total_epochs, num_scales):
    progress = epoch / total_epochs
    
    if progress < 0.5:
        # Early: equal weights
        return [1.0 / num_scales] * num_scales
    else:
        # Late: focus on primary output
        weights = [1.0] + [0.1] * (num_scales - 1)
        return [w / sum(weights) for w in weights]
```

### 6.3 Data Augmentation

**For depth estimation:**
- Horizontal flips (+ flip depth)
- Random crops
- Color jitter on RGB (not on depth)
- **Avoid:** Rotation (breaks depth geometry)

**For class-conditional generation:**
- Horizontal/vertical flips
- Rotations (90° increments)
- Random crops
- Color augmentation

```python
def augment_depth_rgb_pair(rgb, depth):
    # Horizontal flip (synchronized)
    if tf.random.uniform([]) > 0.5:
        rgb = tf.image.flip_left_right(rgb)
        depth = tf.image.flip_left_right(depth)
    
    # Random crop (synchronized)
    combined = tf.concat([rgb, depth], axis=-1)
    combined = tf.image.random_crop(combined, crop_size + [4])
    rgb, depth = tf.split(combined, [3, 1], axis=-1)
    
    # Color jitter (RGB only)
    rgb = tf.image.random_brightness(rgb, 0.2)
    rgb = tf.image.random_contrast(rgb, 0.8, 1.2)
    
    return rgb, depth
```

---

## Inference Strategies

### 7.1 Single-Step Denoising

For estimation tasks (e.g., depth from RGB):

```python
def infer_depth(model, rgb_image, class_label=None):
    """
    Single-step depth inference.
    
    Args:
        model: Trained conditional denoiser
        rgb_image: RGB input (H, W, 3), normalized
        class_label: Optional semantic class
    
    Returns:
        depth_map: Predicted depth (H, W, 1)
    """
    
    # Start from noise or zero
    depth_noisy = tf.random.normal(
        [1, rgb_image.shape[0], rgb_image.shape[1], 1]
    )
    
    # Add batch dimension to RGB
    rgb_batch = tf.expand_dims(rgb_image, axis=0)
    
    # Forward pass
    if class_label is not None:
        class_batch = tf.constant([[class_label]], dtype=tf.int32)
        depth_pred = model([depth_noisy, rgb_batch, class_batch])
    else:
        depth_pred = model([depth_noisy, rgb_batch])
    
    # Remove batch dimension
    depth_map = depth_pred[0]
    
    return depth_map
```

### 7.2 Iterative Refinement

For generation tasks (e.g., class-conditional images):

```python
def generate_with_langevin(
    model,
    class_label,
    image_shape,
    num_steps=200,
    step_size_schedule=None,
    noise_schedule=None
):
    """
    Generate image using Langevin dynamics.
    
    Based on learned score function: ∇log p(y|c)
    """
    
    # Initialize from noise
    y = tf.random.normal([1] + list(image_shape))
    
    # Default schedules
    if step_size_schedule is None:
        step_size_schedule = tf.linspace(0.01, 1.0, num_steps)
    if noise_schedule is None:
        noise_schedule = tf.linspace(0.5, 0.01, num_steps)
    
    class_batch = tf.constant([[class_label]], dtype=tf.int32)
    
    for t in range(num_steps):
        step_size = step_size_schedule[t]
        noise_level = noise_schedule[t]
        
        # Get denoised prediction
        x_hat = model([y, class_batch], training=False)
        
        # Compute score (gradient)
        score = (x_hat - y) / (noise_level ** 2)
        
        # Langevin step
        noise = tf.random.normal(tf.shape(y))
        y = y + step_size * score + tf.sqrt(2 * step_size) * noise
        y = tf.clip_by_value(y, 0.0, 1.0)
    
    return y[0]
```

### 7.3 CFG-Guided Sampling

For discrete conditioning with guidance:

```python
def generate_with_cfg(
    model,
    class_label,
    unconditional_token,
    image_shape,
    guidance_scale=7.5,
    num_steps=200
):
    """
    Generate with Classifier-Free Guidance.
    """
    
    y = tf.random.normal([1] + list(image_shape))
    
    class_cond = tf.constant([[class_label]], dtype=tf.int32)
    class_uncond = tf.constant([[unconditional_token]], dtype=tf.int32)
    
    for t in range(num_steps):
        # Batch conditional and unconditional
        y_double = tf.concat([y, y], axis=0)
        class_double = tf.concat([class_cond, class_uncond], axis=0)
        
        # Single forward pass
        x_hat_double = model([y_double, class_double])
        x_hat_cond, x_hat_uncond = tf.split(x_hat_double, 2, axis=0)
        
        # Apply CFG formula
        x_hat_guided = (
            x_hat_uncond + 
            guidance_scale * (x_hat_cond - x_hat_uncond)
        )
        
        # Gradient step
        score = (x_hat_guided - y) / (sigma ** 2)
        y = y + step_size * score + noise_level * tf.random.normal(tf.shape(y))
        y = tf.clip_by_value(y, 0.0, 1.0)
    
    return y[0]
```

---

## Architectural Patterns

### 8.1 Injection Point Selection

**Question:** Where to inject conditioning signals?

**Options:**

1. **Every level (most information):**
   - Pro: Maximum conditioning influence
   - Con: Most parameters, potential redundancy

2. **Strategic levels only:**
   - Pro: Efficient, targeted conditioning
   - Con: Must select appropriate levels

3. **Adaptive (learned):**
   - Pro: Data-driven selection
   - Con: More complex training

**Recommendation:** Start with every level, then ablate if needed.

### 8.2 Injection Method Comparison

| Method | Parameters | Computation | Flexibility | Bias-Free |
|--------|-----------|-------------|-------------|-----------|
| **Spatial Broadcast** | Low | Low | Medium | ✓ |
| **Channel Concat** | Medium | Medium | High | ✓ |
| **FiLM** | Low | Low | High | ✓ (if no shift) |
| **Cross-Attention** | High | High | Very High | ✗ (has bias) |

**Recommendation:** 
- Dense conditioning: Addition or FiLM
- Discrete conditioning: Spatial broadcast
- Hybrid: Spatial broadcast for discrete, addition for dense

### 8.3 Encoder Architecture Selection

For dense conditioning (RGB images):

| Encoder | Parameters | Speed | Feature Quality | Pre-trained |
|---------|-----------|-------|-----------------|-------------|
| **ResNet-34** | 21M | Fast | Good | ✓ |
| **ResNet-50** | 25M | Medium | Better | ✓ |
| **EfficientNet-B0** | 5M | Fast | Good | ✓ |
| **Custom CNN** | Variable | Fast | Variable | ✗ |

**Considerations:**
- Use pre-trained weights when available (but modify to bias-free)
- Deeper encoders for complex scenes
- Efficient architectures for real-time applications

---

## Practical Considerations

### 9.1 Memory Management

**Challenge:** Multi-modal models have large memory footprint.

**Solutions:**

1. **Gradient checkpointing:**
   ```python
   # Recompute activations during backward pass
   @tf.recompute_grad
   def encoder_block(x):
       return heavy_computation(x)
   ```

2. **Mixed precision training:**
   ```python
   policy = keras.mixed_precision.Policy('mixed_float16')
   keras.mixed_precision.set_global_policy(policy)
   ```

3. **Feature map sharing:**
   ```python
   # Reuse RGB encoder features for multiple tasks
   rgb_features = conditioning_encoder(rgb_input)
   depth_output = depth_decoder(rgb_features)
   semantic_output = semantic_decoder(rgb_features)
   ```

### 9.2 Computational Efficiency

**Bottlenecks:**

1. **Conditioning encoder:** Often the slowest component
   - Solution: Use efficient architectures (EfficientNet, MobileNet)
   - Cache features for multiple forward passes

2. **Multiple injection points:** Repeated projections
   - Solution: Share projection layers across levels
   - Use lower-rank projections

3. **Deep supervision:** Multiple output branches
   - Solution: Use lightweight supervision heads
   - Progressive training (add supervision gradually)

### 9.3 Hyperparameter Tuning

**Key hyperparameters:**

1. **Noise level range** ($\sigma_{\min}$, $\sigma_{\max}$):
   - Depth: [0.01, 0.1] (depth is typically well-behaved)
   - Images: [0.0, 0.5] (more robust to high noise)

2. **Class embedding dimension:**
   - Rule of thumb: $d_{\text{embed}} = 16 \times \sqrt{K}$
   - Minimum: 64, Maximum: 512

3. **Injection method:**
   - Start with spatial broadcast
   - Switch to concatenation if underfitting

4. **Deep supervision weights:**
   - Start uniform, gradually focus on primary
   - Schedule based on validation loss

### 9.4 Common Failure Modes

**1. Mode collapse (class-conditional):**
- **Symptom:** Model ignores class labels, generates similar outputs
- **Solution:** Increase CFG dropout probability, verify embedding gradients

**2. RGB conditioning ignored (depth estimation):**
- **Symptom:** Predictions don't vary with RGB input
- **Solution:** Stronger injection (more levels, larger features), check encoder gradients

**3. Depth bleeding (unified model):**
- **Symptom:** Discrete class affects continuous depth unrealistically
- **Solution:** Separate injection strengths, regularize class embedding influence

**4. Bias-free violations:**
- **Symptom:** Scaling invariance breaks
- **Solution:** Audit all layers for `use_bias=True`, check batch norm configuration

### 9.5 Validation and Testing

**Metrics:**

For depth estimation:
- RMSE (Root Mean Squared Error)
- Absolute Relative Error
- δ₁ threshold accuracy
- Conditional PSNR

For class-conditional generation:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Class-conditional FID
- CFG scaling analysis

**Ablation studies:**
- Injection method comparison
- Conditioning modality importance
- Deep supervision effectiveness
- CFG dropout probability sweep

---

## Conclusion

This generalized framework unifies multiple conditioning paradigms under Miyasawa's theorem:

**Key principles:**
1. **Mathematical consistency:** Same theorem governs all conditioning types
2. **Architectural flexibility:** Modular components adapt to different modalities
3. **Bias-free design:** Critical for theoretical guarantees and practical robustness
4. **Scalability:** Framework extends to arbitrary conditioning signals

**Implementation guidelines:**
- Start simple (single conditioning modality)
- Add complexity gradually (multi-modal)
- Maintain bias-free properties throughout
- Validate against theoretical predictions

**Future extensions:**
- Temporal conditioning (video sequences)
- Hierarchical conditioning (multi-level semantics)
- Continuous control signals (style vectors)
- Multi-task learning (joint estimation and generation)

The conditional denoising framework represents a principled, flexible approach to learning complex conditional distributions, with applications spanning computer vision, graphics, and beyond.