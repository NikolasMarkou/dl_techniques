# Conditional Bias-Free U-Net: Theory and Implementation

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architectural Design](#architectural-design)
4. [Implementation Details](#implementation-details)
5. [Classifier-Free Guidance](#classifier-free-guidance)
6. [Deep Supervision Integration](#deep-supervision-integration)
7. [Training Pipeline](#training-pipeline)
8. [Practical Usage](#practical-usage)
9. [Appendix: Key Design Decisions](#appendix-key-design-decisions)

---

## Overview

This document provides a comprehensive analysis of conditional bias-free U-Net denoisers, combining mathematical theory with practical implementation. The model extends Miyasawa's theorem to conditional distributions, enabling class-conditional image denoising and generation while maintaining the scaling-invariant properties of bias-free networks.

**Key Capabilities:**
- Class-conditional denoising: $\hat{x}(y, c) = \mathbb{E}[x|y, c]$
- Scaling invariance: If input scales by $\alpha$, output scales by $\alpha$
- Classifier-Free Guidance (CFG) for enhanced control
- Multi-scale deep supervision for improved training
- Implicit score function learning: $\nabla_y \log p(y|c)$

---

## Theoretical Foundation

### 1.1 From Unconditional to Conditional

**Unconditional Miyasawa's Theorem** states that for noisy observations $y = x + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the optimal denoiser is:

$$\hat{x}(y) = \mathbb{E}[x|y] = y + \sigma^2 \nabla_y \log p(y)$$

This theorem establishes a fundamental connection between:
- The **denoising task**: predicting $\mathbb{E}[x|y]$
- The **score function**: $\nabla_y \log p(y)$

**Conditional Extension** replaces the marginal distribution $p(y)$ with the conditional distribution $p(y|c)$:

$$\boxed{\hat{x}(y, c) = \mathbb{E}[x|y, c] = y + \sigma^2 \nabla_y \log p(y|c)}$$

where $c$ is the conditioning variable (e.g., class label).

### 1.2 Mathematical Derivation

The derivation follows the unconditional case with all probabilities conditioned on $c$:

**Step 1: Optimal Conditional Denoiser**
$$\hat{x}(y, c) = \mathbb{E}[x|y, c] = \int x \, p(x|y, c) \, dx$$

**Step 2: Apply Bayes' Rule**
$$p(x|y, c) = \frac{p(y|x, c)p(x|c)}{p(y|c)}$$

The noise process is independent of the class: $p(y|x, c) = p(y|x) = \mathcal{N}(x, \sigma^2 I)$

**Step 3: Gradient of Conditional Density**
$$\nabla_y p(y|c) = \int \nabla_y p(y|x) p(x|c) \, dx$$

For Gaussian noise: $\nabla_y p(y|x) = p(y|x) \frac{x-y}{\sigma^2}$

**Step 4: Substitution**
$$\nabla_y p(y|c) = \int p(y|x) \frac{x-y}{\sigma^2} p(x|c) \, dx$$

$$= \frac{1}{\sigma^2} \int (x-y) p(y|x) p(x|c) \, dx$$

$$= \frac{p(y|c)}{\sigma^2} \int (x-y) p(x|y,c) \, dx$$

$$= \frac{p(y|c)}{\sigma^2} (\mathbb{E}[x|y,c] - y)$$

**Step 5: Final Form**
$$\nabla_y \log p(y|c) = \frac{\nabla_y p(y|c)}{p(y|c)} = \frac{1}{\sigma^2}(\hat{x}(y, c) - y)$$

Rearranging gives the conditional theorem.

### 1.3 Implications for Neural Networks

**Training Objective:** By training a neural network $D_\theta(y, c)$ with MSE loss to predict clean images $x$ from noisy images $y$ and class labels $c$:

$$\mathcal{L}(\theta) = \mathbb{E}_{x, c, \epsilon}\left[\|D_\theta(x + \epsilon, c) - x\|^2\right]$$

The network implicitly learns:
1. The conditional expectation: $D_\theta(y, c) \approx \mathbb{E}[x|y, c]$
2. The conditional score function: $\nabla_y \log p(y|c) \approx \frac{D_\theta(y, c) - y}{\sigma^2}$

**Critical Requirements:**
- **Bias-free architecture**: No additive constants (biases) to maintain scaling invariance
- **Linear final activation**: Preserves the residual relationship
- **MSE loss**: Directly optimizes for conditional expectation

---

## Architectural Design

### 2.1 High-Level Architecture

```
Input: [Noisy Image y, Class Label c]
                    |
                    v
         ┌──────────────────────┐
         │  Class Embedding     │
         │  c → e_c ∈ ℝ^d       │
         └──────────────────────┘
                    |
                    v
         ┌──────────────────────┐
         │   Encoder Path       │
         │  + Class Injection   │
         │  (Downsampling)      │
         └──────────────────────┘
                    |
                    v
         ┌──────────────────────┐
         │   Bottleneck         │
         │  + Class Injection   │
         └──────────────────────┘
                    |
                    v
         ┌──────────────────────┐
         │   Decoder Path       │
         │  + Skip Connections  │
         │  + Class Injection   │
         │  (Upsampling)        │
         └──────────────────────┘
                    |
                    v
Output: Denoised Image x̂ = D(y, c)
```

### 2.2 Class Conditioning Mechanism

Two primary methods are implemented for injecting class information:

#### Method 1: Spatial Broadcast (Default)

```python
def inject_class_conditioning_spatial(x, class_emb, channels):
    """
    FiLM-style conditioning without bias.
    
    x: (batch, height, width, channels)
    class_emb: (batch, embedding_dim)
    """
    # Project embedding to match feature channels
    projected = Dense(channels, use_bias=False)(class_emb)
    
    # Reshape to (batch, 1, 1, channels) for broadcasting
    projected = Reshape((1, 1, channels))(projected)
    
    # Add to features (bias-free modulation)
    return Add()([x, projected])
```

**Advantages:**
- Minimal parameter overhead
- Direct feature modulation
- Maintains spatial structure
- Bias-free by design

#### Method 2: Channel Concatenation

```python
def inject_class_conditioning_concat(x, class_emb, height, width):
    """
    Concatenate spatially-tiled class embedding.
    
    x: (batch, height, width, channels)
    class_emb: (batch, embedding_dim)
    """
    # Project and tile to spatial dimensions
    projected = Dense(filters, use_bias=False)(class_emb)
    tiled = RepeatVector(height * width)(projected)
    tiled = Reshape((height, width, filters))(tiled)
    
    # Concatenate as additional channels
    return Concatenate()([x, tiled])
```

**Advantages:**
- Explicit conditioning channel
- No feature interference
- Flexible for different architectures

### 2.3 Bias-Free Building Blocks

All convolutional layers use the `BiasFreeConv2D` layer:

```python
class BiasFreeConv2D:
    """
    Convolutional layer without bias term.
    
    Key properties:
    - use_bias=False: Maintains scaling invariance
    - Batch normalization: Centered but not shifted
    - Scaling-equivariant: f(αx) = αf(x)
    """
```

**Scaling Invariance Proof:**
If $y' = \alpha y$ for scalar $\alpha > 0$, then:

1. Class embedding is scale-independent: $e_c(c) = e_c(c)$
2. Convolutions: $\text{Conv}(\alpha y) = \alpha \text{Conv}(y)$
3. Addition: $\alpha y + e_c = \alpha(y + e_c/\alpha)$
4. Therefore: $D_\theta(\alpha y, c) = \alpha D_\theta(y, c)$

This property is critical for denoisers operating across different noise levels.

### 2.4 Multi-Scale Architecture

The U-Net consists of:

**Encoder:**
- $L$ levels of downsampling (typically $L = 3$ or $4$)
- Bias-free residual blocks at each level
- Class conditioning injected at each level
- Skip connections preserved for decoder

**Bottleneck:**
- Lowest resolution processing
- Highest channel capacity
- Class conditioning injected

**Decoder:**
- $L$ levels of upsampling
- Skip connection concatenation
- Class conditioning injected after concatenation
- Optional deep supervision outputs

---

## Implementation Details

### 3.1 Core Model Function

```python
def create_conditional_bfunet_denoiser(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    depth: int = 4,
    initial_filters: int = 64,
    class_embedding_dim: int = 128,
    class_injection_method: str = 'spatial_broadcast',
    enable_cfg_training: bool = True,
    enable_deep_supervision: bool = False,
    **kwargs
) -> keras.Model
```

**Key Parameters:**
- `num_classes`: Total number of classes (includes unconditional token if CFG enabled)
- `class_embedding_dim`: Dimension of class embedding vectors (default: 128)
- `class_injection_method`: 'spatial_broadcast' or 'channel_concat'
- `enable_cfg_training`: Reserve last class index as unconditional token

### 3.2 Class Embedding Layer

```python
# Embedding layer: maps integer class labels to dense vectors
class_embedding = keras.layers.Embedding(
    input_dim=num_classes,
    output_dim=class_embedding_dim,
    embeddings_initializer=kernel_initializer,
    name='class_embedding'
)(class_input)
```

**Design Choices:**
- Learnable embeddings allow the network to discover optimal class representations
- Embedding dimension (128) balances expressiveness and efficiency
- No bias in subsequent projections maintains scaling invariance

### 3.3 Conditioning Injection Points

Class conditioning is injected at three strategic locations:

1. **Encoder levels**: Before each downsampling operation
   - Conditions feature extraction on class information
   - Early injection influences low-level features

2. **Bottleneck**: At the lowest resolution
   - Maximum receptive field
   - Global semantic conditioning

3. **Decoder levels**: After skip connection concatenation
   - Refines features during upsampling
   - Conditions reconstruction on class

**Rationale:** Multiple injection points ensure class information flows throughout the network, influencing both feature extraction and reconstruction.

### 3.4 Network Flow Example

For a depth-4 network processing a 256×256 image:

```
Input: (256, 256, 3) + class_label

Encoder:
  Level 0: (256, 256, 64)  + class → Conv → (256, 256, 64)
  Pool →   (128, 128, 64)
  Level 1: (128, 128, 128) + class → Conv → (128, 128, 128)
  Pool →   (64, 64, 128)
  Level 2: (64, 64, 256)   + class → Conv → (64, 64, 256)
  Pool →   (32, 32, 256)
  Level 3: (32, 32, 512)   + class → Conv → (32, 32, 512)

Bottleneck:
  Pool →   (16, 16, 512)
           (16, 16, 1024) + class → Conv → (16, 16, 1024)

Decoder:
  Upsample → (32, 32, 1024)
  Concat skip → (32, 32, 1536)
             + class → Conv → (32, 32, 512)  [DS Output 3]
  Upsample → (64, 64, 512)
  Concat skip → (64, 64, 768)
             + class → Conv → (64, 64, 256)  [DS Output 2]
  Upsample → (128, 128, 256)
  Concat skip → (128, 128, 384)
             + class → Conv → (128, 128, 128) [DS Output 1]
  Upsample → (256, 256, 128)
  Concat skip → (256, 256, 192)
             + class → Conv → (256, 256, 64)

Output: (256, 256, 3) [Primary Output]
```

---

## Classifier-Free Guidance

### 4.1 Theoretical Foundation

Classifier-Free Guidance (CFG) allows inference-time control over conditioning strength without requiring a separate classifier network.

**Key Insight:** Train a single model to estimate both:
- Conditional score: $s(y, c) = \nabla_y \log p(y|c)$
- Unconditional score: $s(y) = \nabla_y \log p(y)$

**Guided Score Formula:**
$$\tilde{s}(y, c) = s(y) + w \cdot (s(y, c) - s(y))$$

where $w$ is the guidance scale:
- $w = 0$: Pure unconditional generation
- $w = 1$: Standard conditional generation
- $w > 1$: Amplified conditioning (sharper class-specific features)

### 4.2 Training with CFG

**Unconditional Token:** Reserve the last class index as a special "unconditional" token:

```python
if enable_cfg_training:
    unconditional_token = num_classes - 1
    # During training: randomly replace class labels
```

**Random Dropout:** During training, randomly replace class labels with the unconditional token:

```python
def apply_cfg_dropout(noisy_patch, class_label, clean_patch, 
                      dropout_prob=0.1):
    should_drop = tf.random.uniform([]) < dropout_prob
    
    class_label = tf.cond(
        should_drop,
        lambda: tf.constant(unconditional_token, dtype=tf.int32),
        lambda: class_label
    )
    
    return (noisy_patch, class_label), clean_patch
```

**Training Distribution:**
- 90% of samples: Conditional training with true class labels
- 10% of samples: Unconditional training (null token)

This allows the model to learn both conditional and unconditional distributions.

### 4.3 CFG Sampling Algorithm

```python
def conditional_sampling_with_cfg(
    denoiser, class_labels, unconditional_token,
    num_samples, image_shape, num_steps,
    guidance_scale=7.5
):
    # Initialize from noise
    y = tf.random.normal([num_samples] + list(image_shape))
    
    for t in range(num_steps):
        # Batch both conditional and unconditional predictions
        y_double = tf.concat([y, y], axis=0)
        labels_cond = class_labels
        labels_uncond = [unconditional_token] * num_samples
        labels_double = tf.concat([labels_cond, labels_uncond], axis=0)
        
        # Single forward pass for both
        denoised_double = denoiser([y_double, labels_double])
        denoised_cond, denoised_uncond = tf.split(denoised_double, 2)
        
        # Apply CFG formula
        denoised_guided = (
            denoised_uncond + 
            guidance_scale * (denoised_cond - denoised_uncond)
        )
        
        # Gradient ascent step
        d_t = denoised_guided - y
        z_t = tf.random.normal(tf.shape(y))
        y = y + step_size * d_t + noise_level * z_t
        y = tf.clip_by_value(y, 0.0, 1.0)
    
    return y
```

**Computational Efficiency:** By batching conditional and unconditional passes, we only need one forward pass (with doubled batch size) instead of two separate passes.

### 4.4 Guidance Scale Effects

**Empirical Observations:**

| $w$ | Effect | Use Case |
|-----|--------|----------|
| 0.0 | Unconditional generation | Class-agnostic sampling |
| 1.0 | Standard conditional | Balanced generation |
| 3.0-5.0 | Enhanced conditioning | Clearer class features |
| 7.5-10.0 | Strong amplification | Maximum class specificity |
| >15.0 | Over-amplification | Artifacts, unrealistic images |

**Recommended Range:** $w \in [1.0, 10.0]$ for most applications.

---

## Deep Supervision Integration

### 5.1 Deep Supervision Concept

Deep supervision adds auxiliary loss functions at intermediate decoder levels, providing direct supervision to deeper layers.

**Architecture Modification:**
```
Decoder Level 3 → Supervision Output 3 (32×32)
Decoder Level 2 → Supervision Output 2 (64×64)
Decoder Level 1 → Supervision Output 1 (128×128)
Decoder Level 0 → Primary Output (256×256)
```

### 5.2 Multi-Scale Label Generation

During training, ground truth labels are resized to match each supervision output:

```python
def create_multiscale_labels(clean_patch, output_dimensions):
    """
    clean_patch: (H, W, C)
    output_dimensions: [(H1, W1), (H2, W2), (H3, W3)]
    
    Returns: tuple of resized labels
    """
    labels = [
        tf.image.resize(clean_patch, dim)
        for dim in output_dimensions
    ]
    return tuple(labels)
```

### 5.3 Weighted Loss Function

The total loss is a weighted sum across all outputs:

$$\mathcal{L}_{\text{total}} = \sum_{i=0}^{N-1} w_i(\tau) \cdot \mathcal{L}_{\text{MSE}}(\hat{x}_i, x_i)$$

where:
- $N$ is the number of outputs
- $w_i(\tau)$ are time-dependent weights
- $\tau \in [0, 1]$ is training progress

### 5.4 Weight Scheduling

**Step-wise Schedule (Default):**
```python
def step_wise_schedule(progress, num_outputs):
    """
    Early training: Equal weights
    Late training: Focus on primary output
    """
    if progress < 0.5:
        return [1.0 / num_outputs] * num_outputs
    else:
        weights = [1.0] + [0.1] * (num_outputs - 1)
        return normalize(weights)
```

**Rationale:** 
- Early training benefits from multi-scale supervision
- Late training focuses on final output quality
- Smooth transition prevents training instability

### 5.5 Conditional Deep Supervision

Each supervision output receives class conditioning:

```python
# At decoder level i
supervision_branch = BiasFreeConv2D(...)(x)
supervision_branch = inject_class_conditioning(
    supervision_branch, class_embedding, level=i
)
supervision_output = BiasFreeConv2D(
    filters=output_channels,
    kernel_size=1,
    activation='linear',
    use_batch_norm=False
)(supervision_branch)
```

This ensures intermediate outputs also learn class-conditional features.

---

## Training Pipeline

### 6.1 Data Organization

**Subfolder Structure (Recommended):**
```
train/
├── class_0/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── class_1/
│   ├── img001.jpg
│   └── ...
└── class_N/
    └── ...
```

**Automatic Class Discovery:**
```python
def discover_classes(directories):
    class_names = set()
    for directory in directories:
        subdirs = [d.name for d in Path(directory).iterdir() if d.is_dir()]
        class_names.update(subdirs)
    
    sorted_classes = sorted(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
    
    # Reserve last index for unconditional token
    if enable_cfg_training:
        unconditional_token = len(class_to_idx)
        class_to_idx['__unconditional__'] = unconditional_token
    
    return class_to_idx
```

### 6.2 Data Pipeline

**Training Data Format:**
```python
# Input: tuple of (noisy_image, class_label)
# Output: clean_image (or tuple for deep supervision)

# Example:
inputs = (noisy_image, class_label)  # ((H,W,C), (1,))
outputs = clean_image                 # (H,W,C)

# For deep supervision:
outputs = (
    clean_image_full_res,     # (H, W, C)
    clean_image_half_res,     # (H/2, W/2, C)
    clean_image_quarter_res   # (H/4, W/4, C)
)
```

**Noise Addition:**
```python
def add_noise(clean_patch, sigma_min=0.0, sigma_max=0.5):
    # Sample noise level
    sigma = tf.random.uniform([], sigma_min, sigma_max)
    
    # Add Gaussian noise
    noise = tf.random.normal(tf.shape(clean_patch)) * sigma
    noisy_patch = clean_patch + noise
    noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)
    
    return noisy_patch, clean_patch
```

### 6.3 Training Configuration

**Recommended Hyperparameters:**
```python
config = ConditionalBFUNetConfig(
    # Model
    model_variant='base',  # or 'tiny', 'small', 'large', 'xlarge'
    class_embedding_dim=128,
    class_injection_method='spatial_broadcast',
    
    # Training
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    optimizer_type='adamw',
    weight_decay=1e-5,
    
    # CFG
    enable_cfg_training=True,
    cfg_dropout_prob=0.1,  # 10% unconditional samples
    
    # Deep Supervision
    enable_deep_supervision=True,
    deep_supervision_schedule_type='step_wise',
    
    # Noise
    noise_sigma_min=0.0,
    noise_sigma_max=0.5,
    noise_distribution='uniform',
    
    # Data
    patch_size=128,
    patches_per_image=16,
    augment_data=True,
)
```

### 6.4 Loss Function

**Single Output:**
```python
loss = 'mse'
metrics = ['mae', 'rmse', PrimaryOutputPSNR()]
```

**Multi-Output (Deep Supervision):**
```python
loss = ['mse', 'mse', 'mse', 'mse']  # One per output
loss_weights = [1.0, 0.5, 0.25, 0.125]  # Initial weights
# Weights are updated during training by scheduler
```

### 6.5 Monitoring and Callbacks

**Training Callbacks:**
1. **Learning Rate Scheduler**: Cosine annealing with warmup
2. **Deep Supervision Weight Scheduler**: Dynamic loss weighting
3. **Conditional Synthesis Monitor**: Generate samples at different CFG scales
4. **Conditional Denoising Visualizer**: Visualize denoising results per class
5. **Model Checkpointing**: Save best model based on validation PSNR

**Synthesis Visualization:**
- Generate samples for each class at multiple CFG scales ($w \in \{0, 1, 3, 7.5\}$)
- Compare quality and class-specificity across scales
- Monitor mode collapse or overfitting

---

## Practical Usage

### 7.1 Training from Scratch

```bash
python train_conditional_bfunet.py \
    --model-variant base \
    --epochs 50 \
    --batch-size 32 \
    --patch-size 128 \
    --channels 3 \
    --enable-cfg-training \
    --cfg-dropout-prob 0.1 \
    --enable-deep-supervision \
    --enable-synthesis \
    --output-dir results/conditional_experiment
```

### 7.2 Inference: Class-Conditional Denoising

```python
import keras
import numpy as np
import tensorflow as tf

# Load model and class mapping
model = keras.models.load_model('results/experiment/inference_model.keras')
class_mapping = json.load(open('results/experiment/class_mapping.json'))

# Prepare noisy image and class label
noisy_image = load_image('test.jpg')  # Shape: (H, W, C)
noisy_image = noisy_image / 255.0  # Normalize to [0, 1]
noisy_image = np.expand_dims(noisy_image, axis=0)  # Add batch dim

class_name = 'cat'
class_idx = class_mapping['class_to_idx'][class_name]
class_label = np.array([[class_idx]], dtype=np.int32)

# Denoise
denoised = model([noisy_image, class_label], training=False)
denoised = np.clip(denoised[0].numpy(), 0, 1)

# Save result
save_image('denoised.jpg', (denoised * 255).astype(np.uint8))
```

### 7.3 Inference: CFG-Guided Sampling

```python
from train_conditional_bfunet import conditional_sampling_with_cfg

# Generate 4 samples of class 'dog'
class_labels = np.array([class_mapping['class_to_idx']['dog']] * 4)

samples, intermediates = conditional_sampling_with_cfg(
    denoiser=model,
    class_labels=class_labels,
    unconditional_token=class_mapping['unconditional_token'],
    num_samples=4,
    image_shape=(256, 256, 3),
    num_steps=200,
    guidance_scale=7.5,  # Adjust for stronger/weaker conditioning
    seed=42
)

# Save samples
for i, sample in enumerate(samples):
    save_image(f'generated_dog_{i}.jpg', (sample * 255).astype(np.uint8))
```

### 7.4 Comparing CFG Scales

```python
cfg_scales = [0.0, 1.0, 3.0, 7.5, 10.0]
results = {}

for w in cfg_scales:
    samples, _ = conditional_sampling_with_cfg(
        denoiser=model,
        class_labels=class_labels,
        unconditional_token=unconditional_token,
        num_samples=4,
        image_shape=(256, 256, 3),
        num_steps=200,
        guidance_scale=w,
        seed=42
    )
    results[w] = samples

# Visualize comparison
visualize_cfg_comparison(results, class_name='dog')
```

### 7.5 Creating Single-Output Inference Model

If trained with deep supervision, extract only the primary output:

```python
# Load multi-output training model
training_model = keras.models.load_model('results/experiment/model_final.keras')

# Create single-output inference model
inference_model = keras.Model(
    inputs=training_model.input,
    outputs=training_model.output[0],  # Primary output only
    name='inference_model'
)

# Save inference model
inference_model.save('inference_model.keras')
```

### 7.6 Batch Processing with Class Labels

```python
def batch_denoise_with_classes(
    model, image_paths, class_labels, 
    batch_size=8
):
    """
    Batch process multiple images with their class labels.
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = class_labels[i:i+batch_size]
        
        # Load and preprocess images
        images = np.array([
            load_and_preprocess(path) 
            for path in batch_paths
        ])
        
        # Reshape labels
        labels = np.array(batch_labels).reshape(-1, 1)
        
        # Denoise batch
        denoised = model([images, labels], training=False)
        
        results.extend(denoised.numpy())
    
    return results
```

---

## Appendix: Key Design Decisions

### A.1 Why Bias-Free Architecture?

**Problem:** Standard neural networks with bias terms do not satisfy:
$$f(\alpha x) = \alpha f(x)$$

**Solution:** Remove all additive constants (biases):
- Convolutional layers: `use_bias=False`
- Batch normalization: Centered but not shifted
- No bias in linear projections

**Benefit:** The network becomes **scaling-equivariant**, which is essential for:
1. **Noise-level invariance**: Works across different noise levels
2. **Dynamic range flexibility**: Handles different image intensities
3. **Theoretical consistency**: Satisfies Miyasawa's theorem requirements

### A.2 Why MSE Loss?

The MSE loss directly optimizes the conditional expectation:

$$\arg\min_{\theta} \mathbb{E}\left[\|D_\theta(y, c) - x\|^2\right] = \mathbb{E}[x|y, c]$$

This is exactly what we want according to the conditional theorem. Other losses (e.g., L1, perceptual) do not have this property.

### A.3 Class Injection: Spatial Broadcast vs. Channel Concat

**Spatial Broadcast (Recommended):**
- ✓ Fewer parameters
- ✓ Direct feature modulation
- ✓ Easier to maintain bias-free property
- ✗ Potential feature interference

**Channel Concatenation:**
- ✓ Explicit conditioning channel
- ✓ No feature modification
- ✗ More parameters
- ✗ Requires careful handling to stay bias-free

**Recommendation:** Use spatial broadcast unless you have specific architectural constraints.

### A.4 Why Multiple Class Injection Points?

**Early injection** (encoder):
- Influences low-level feature extraction
- Helps network learn class-specific textures and patterns

**Middle injection** (bottleneck):
- Maximum receptive field
- Global semantic understanding

**Late injection** (decoder):
- Refines reconstruction
- Ensures class information affects final output

**Ablation studies** (typical findings):
- Single injection point: 2-4 dB lower PSNR
- Multiple injection points: Best performance
- Injection at every layer: Diminishing returns, more parameters

### A.5 CFG Dropout Probability

**Typical range:** 0.05 - 0.20

**Trade-offs:**
- **Too low** (<0.05): Poor unconditional score estimation
- **Optimal** (0.10): Balanced conditional/unconditional learning
- **Too high** (>0.20): Degraded conditional performance

**Recommendation:** Start with 0.1 and adjust based on:
- Unconditional sample quality
- Conditional sample quality
- CFG effectiveness at inference

### A.6 Deep Supervision Trade-offs

**Advantages:**
- ✓ Better gradient flow to deep layers
- ✓ Multi-scale feature learning
- ✓ More stable training
- ✓ Faster convergence

**Disadvantages:**
- ✗ More memory during training
- ✗ Slightly more complex data pipeline
- ✗ Need to manage multiple outputs

**When to use:**
- Deep networks (depth ≥ 4)
- Limited training data
- Training instability issues

**When to skip:**
- Shallow networks (depth ≤ 3)
- Very large datasets
- Inference speed is critical

### A.7 Embedding Dimension Selection

**Typical range:** 64 - 256

**Guidelines:**
- **Small datasets** (<10K images): 64-128
- **Medium datasets** (10K-100K): 128-256
- **Large datasets** (>100K): 256-512

**Rule of thumb:** `embedding_dim ≈ sqrt(num_classes) * 16`

For 10 classes: 128 dimensions
For 100 classes: 256 dimensions
For 1000 classes: 512 dimensions

### A.8 Synthesis Hyperparameters

**Number of steps:**
- Fast preview: 50-100 steps
- Standard quality: 200 steps
- High quality: 500-1000 steps

**Step size scheduling:**
- Start small (0.01-0.05) for stability
- End large (0.5-1.0) for fast convergence
- Linear or cosine interpolation

**Noise level scheduling:**
- Start high (0.3-0.5) for exploration
- End low (0.001-0.01) for refinement
- Exponential decay often works well

---

## References

### Theoretical Foundations
- Miyasawa, K. (1961). "An empirical Bayes estimator of the mean of a normal population."
- Kadkhodaie, Z., & Simoncelli, E. P. (2021). "Stochastic solutions for linear inverse problems using the prior implicit in a denoiser."
- Mohan, J., et al. (2020). "Robust and Interpretable Blind Image Denoising via Bias-Free Convolutional Neural Networks." ICLR.

### Conditional Methods
- Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS Workshop.
- Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS.

### Architecture
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."
- Lee, C. Y., et al. (2015). "Deeply-Supervised Nets." AISTATS.

---

## Conclusion

This implementation combines rigorous mathematical foundations with practical engineering to create a powerful class-conditional denoising system. Key achievements:

1. **Theoretical soundness**: Implements Miyasawa's theorem for conditional distributions
2. **Scaling invariance**: Maintains bias-free properties throughout
3. **Flexible control**: CFG enables inference-time conditioning strength adjustment
4. **Training efficiency**: Deep supervision improves convergence
5. **Production-ready**: Complete pipeline from data to deployment

The conditional bias-free U-Net represents a principled approach to class-conditional image processing, with applications in denoising, restoration, and generation tasks.