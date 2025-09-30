# Masked Autoencoder (MAE) Framework - Complete Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [What is Masked Autoencoding?](#what-is-masked-autoencoding)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Training the MAE](#training-the-mae)
6. [Working with ConvNeXt V2](#working-with-convnext-v2)
7. [Fine-tuning for Downstream Tasks](#fine-tuning-for-downstream-tasks)
8. [Visualization and Evaluation](#visualization-and-evaluation)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

The Masked Autoencoder (MAE) framework provides a powerful self-supervised learning approach for training vision models without labeled data. This guide shows you how to use the framework to pretrain encoders that can then be fine-tuned for various computer vision tasks.

### Why Use MAE?

- **No Labels Required**: Train on unlabeled image datasets
- **Better Representations**: Learn robust visual features through reconstruction
- **Transfer Learning**: Pretrained encoders work well on downstream tasks
- **Data Efficiency**: Requires less labeled data for fine-tuning
- **Proven Results**: Used successfully in ConvNeXt V2, ViT, and other architectures

---

## What is Masked Autoencoding?

Masked autoencoding is a self-supervised learning technique inspired by BERT:

1. **Random Masking**: Randomly hide patches of the input image (e.g., 75%)
2. **Encoding**: Process the masked image through an encoder
3. **Decoding**: Reconstruct the original image from the encoded features
4. **Loss**: Compute reconstruction loss only on masked patches

This forces the encoder to learn meaningful representations that capture image structure and semantics.

### Key Concepts

```
Original Image (224x224x3)
        ↓
Random Patch Masking (75% patches hidden)
        ↓
Encoder (ConvNeXt, ResNet, etc.)
        ↓
Encoded Features (7x7x768)
        ↓
Decoder (Lightweight Conv Layers)
        ↓
Reconstructed Image (224x224x3)
        ↓
Loss = MSE(masked_patches_original, masked_patches_reconstructed)
```

---

## Quick Start

Here's a minimal example to get started:

```python
import keras
import numpy as np
from masked_autoencoder import MaskedAutoencoder, visualize_reconstruction
from convnext_v2 import ConvNeXtV2

# 1. Create MAE model
mae = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],      # ConvNeXt-Tiny dimensions
    encoder_output_shape=(7, 7, 768),       # Encoder output after 224x224 input
    patch_size=16,                          # 16x16 patches
    mask_ratio=0.75,                        # Mask 75% of patches
    input_shape=(224, 224, 3)
)

# 2. Compile
mae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

# 3. Train on unlabeled images
# Assuming train_images is array of shape (N, 224, 224, 3)
mae.fit(train_images, epochs=100, batch_size=64)

# 4. Visualize results
sample_image = train_images[0]
original, masked, reconstructed = mae.visualize(sample_image)

# 5. Extract pretrained encoder for downstream tasks
encoder = mae.encoder
```

---

## Detailed Setup

### Step 1: Prepare Your Data

MAE works with unlabeled images. Prepare your dataset:

```python
import numpy as np
from pathlib import Path
from PIL import Image

def load_images_from_directory(image_dir, target_size=(224, 224)):
    """Load and preprocess images from directory."""
    image_paths = list(Path(image_dir).glob("**/*.jpg"))
    images = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
    
    return np.array(images, dtype=np.float32)

# Load unlabeled images
train_images = load_images_from_directory("data/unlabeled_images")
print(f"Loaded {len(train_images)} images with shape {train_images.shape}")
```

### Step 2: Create the MAE Model

The key is specifying your encoder architecture dimensions:

```python
from masked_autoencoder import MaskedAutoencoder

# For ConvNeXt V2 architectures, use these configurations:

# ConvNeXt-Pico (smallest, fastest)
mae_pico = MaskedAutoencoder(
    encoder_dims=[64, 128, 256, 512],
    encoder_output_shape=(7, 7, 512),
    patch_size=16,
    mask_ratio=0.75,
    decoder_dims=[256, 128, 64, 32],
    input_shape=(224, 224, 3)
)

# ConvNeXt-Tiny (balanced)
mae_tiny = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    patch_size=16,
    mask_ratio=0.75,
    decoder_dims=[512, 256, 128, 64],
    input_shape=(224, 224, 3)
)

# ConvNeXt-Base (larger, better representations)
mae_base = MaskedAutoencoder(
    encoder_dims=[128, 256, 512, 1024],
    encoder_output_shape=(7, 7, 1024),
    patch_size=16,
    mask_ratio=0.75,
    decoder_dims=[512, 256, 128, 64],
    input_shape=(224, 224, 3)
)
```

### Step 3: Understanding Key Parameters

**Encoder Parameters:**
- `encoder_dims`: Channel dimensions for each encoder stage
- `encoder_output_shape`: Spatial output after encoding (H, W, C)

**Masking Parameters:**
- `patch_size`: Size of square patches (16 is standard)
- `mask_ratio`: Fraction of patches to mask (0.75 = 75%)
- `mask_value`: How to mask patches ("learnable", "zero", "noise")

**Decoder Parameters:**
- `decoder_dims`: List of channel dimensions for decoder layers
- `decoder_depth`: Number of decoder layers (if decoder_dims=None)

**Loss Parameters:**
- `norm_pix_loss`: Whether to normalize pixels per patch (usually False)

---

## Training the MAE

### Basic Training

```python
# Compile with optimizer
mae.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=0.05
    )
)

# Train
history = mae.fit(
    train_images,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            'mae_checkpoint.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)
```

### Training with tf.data Pipeline

For larger datasets, use tf.data for efficiency:

```python
import tensorflow as tf

def create_mae_dataset(image_paths, batch_size=64, image_size=(224, 224)):
    """Create efficient tf.data pipeline for MAE training."""
    
    def load_and_preprocess(path):
        # Load image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize and normalize
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create dataset
train_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
train_dataset = create_mae_dataset(train_paths, batch_size=64)

# Train
mae.fit(train_dataset, epochs=100)
```

### Training with Data Augmentation

Add augmentation to improve learned representations:

```python
def augment_for_mae(image):
    """Apply augmentation suitable for MAE pretraining."""
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random crop and resize
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, [224, 224])
    
    # Color jittering
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

# Update dataset pipeline
dataset = dataset.map(lambda x: augment_for_mae(x), 
                     num_parallel_calls=tf.data.AUTOTUNE)
```

### Learning Rate Scheduling

Use a learning rate schedule for better convergence:

```python
# Cosine decay with warmup
total_steps = len(train_images) // batch_size * epochs
warmup_steps = total_steps // 10

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=total_steps - warmup_steps,
    alpha=1e-6  # Minimum learning rate
)

# With warmup
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, peak_lr=1e-4, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.cosine_decay = keras.optimizers.schedules.CosineDecay(
            peak_lr, total_steps - warmup_steps, alpha=min_lr/peak_lr
        )
    
    def __call__(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            return self.cosine_decay(step - self.warmup_steps)

lr_schedule = WarmupCosineDecay(
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    peak_lr=1e-4
)

mae.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule))
```

---

## Working with ConvNeXt V2

### Method 1: Training MAE with ConvNeXt Configuration

```python
from masked_autoencoder import MaskedAutoencoder
from convnext_v2 import ConvNeXtV2

# Create MAE with ConvNeXt-Tiny configuration
mae = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    patch_size=16,
    mask_ratio=0.75,
    input_shape=(224, 224, 3)
)

# Train MAE
mae.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4))
mae.fit(train_images, epochs=100, batch_size=64)

# Save MAE model
mae.save('mae_convnext_tiny.keras')
```

### Method 2: Loading Pretrained ConvNeXt Encoder

If you have a pretrained ConvNeXt V2 model, you can initialize the MAE encoder:

```python
# Create pretrained ConvNeXt encoder
pretrained_encoder = ConvNeXtV2.from_variant(
    "tiny",
    include_top=False,
    input_shape=(224, 224, 3),
    pretrained=True  # If you have pretrained weights
)

# Create MAE
mae = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    patch_size=16,
    mask_ratio=0.75,
    input_shape=(224, 224, 3)
)

# Transfer weights from pretrained encoder to MAE encoder
# Note: This requires architecture alignment
mae.encoder.set_weights(pretrained_encoder.get_weights())

# Continue training or use directly
```

### Method 3: Extracting Encoder After MAE Training

After training MAE, extract the encoder for use:

```python
# Train MAE
mae.fit(train_images, epochs=100)

# Save just the encoder
mae.encoder.save('convnext_encoder_pretrained.keras')

# Later, load for downstream tasks
encoder = keras.models.load_model('convnext_encoder_pretrained.keras')

# Use in a new model
inputs = keras.Input(shape=(224, 224, 3))
features = encoder(inputs)
# Add your task-specific head here
```

---

## Fine-tuning for Downstream Tasks

After MAE pretraining, use the encoder for supervised tasks:

### Classification Task

```python
from masked_autoencoder import MaskedAutoencoder
import keras

# Load trained MAE
mae = keras.models.load_model('mae_convnext_tiny.keras')

# Extract encoder
encoder = mae.encoder
encoder.trainable = True  # Make it trainable

# Build classifier
inputs = keras.Input(shape=(224, 224, 3))
features = encoder(inputs)

# Add classification head
x = keras.layers.GlobalAveragePooling2D()(features)
x = keras.layers.LayerNormalization()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)  # 10 classes

# Create model
classifier = keras.Model(inputs, outputs, name='classifier')

# Compile
classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune on labeled data
classifier.fit(
    train_images_labeled,
    train_labels,
    epochs=50,
    validation_split=0.2
)
```

### Two-Stage Fine-tuning (Recommended)

Freeze encoder first, then unfreeze:

```python
# Stage 1: Train only the classifier head
encoder.trainable = False
classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.fit(train_data, epochs=10)

# Stage 2: Fine-tune entire model
encoder.trainable = True
classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Much lower LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.fit(train_data, epochs=40)
```

### Segmentation Task

```python
# Extract encoder
encoder = mae.encoder

# Build U-Net style decoder for segmentation
inputs = keras.Input(shape=(224, 224, 3))
features = encoder(inputs)

# Decoder with skip connections (if you modify encoder to return intermediate features)
x = keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(features)
x = keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
outputs = keras.layers.Conv2D(num_classes, 1, activation='softmax')(x)

segmentation_model = keras.Model(inputs, outputs)
```

---

## Visualization and Evaluation

### Visualizing Reconstructions

```python
from masked_autoencoder import visualize_reconstruction
import matplotlib.pyplot as plt

# Single image visualization
sample_image = test_images[0]
original, masked, reconstructed = mae.visualize(sample_image)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(masked)
axes[1].set_title('Masked Input')
axes[1].axis('off')

axes[2].imshow(reconstructed)
axes[2].set_title('Reconstructed')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('mae_reconstruction.png')
plt.show()

# Multiple images grid
grid = visualize_reconstruction(mae, test_images, num_samples=4)
plt.figure(figsize=(15, 10))
plt.imshow(grid)
plt.axis('off')
plt.title('MAE Reconstructions (Original | Masked | Reconstructed)')
plt.savefig('mae_grid.png')
plt.show()
```

### Monitoring Training Progress

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.legend()
plt.title('Training Progress')

# Visualize reconstructions at different epochs
epochs_to_check = [10, 50, 100]
fig, axes = plt.subplots(len(epochs_to_check), 3, figsize=(15, 15))

for i, epoch in enumerate(epochs_to_check):
    # Load checkpoint from that epoch
    mae_checkpoint = keras.models.load_model(f'checkpoints/mae_epoch_{epoch}.keras')
    original, masked, reconstructed = mae_checkpoint.visualize(test_images[0])
    
    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f'Epoch {epoch}: Original')
    axes[i, 1].imshow(masked)
    axes[i, 1].set_title(f'Epoch {epoch}: Masked')
    axes[i, 2].imshow(reconstructed)
    axes[i, 2].set_title(f'Epoch {epoch}: Reconstructed')

plt.tight_layout()
plt.show()
```

### Evaluating Learned Representations

Use linear probing to evaluate representation quality:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Extract features from MAE encoder
def extract_features(images, encoder):
    """Extract features using the encoder."""
    features = encoder.predict(images, batch_size=32)
    # Global average pooling
    features = features.mean(axis=(1, 2))
    return features

# Extract features
train_features = extract_features(train_images_labeled, mae.encoder)
test_features = extract_features(test_images_labeled, mae.encoder)

# Train linear classifier (no fine-tuning)
clf = LogisticRegression(max_iter=1000)
clf.fit(train_features, train_labels)

# Evaluate
predictions = clf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Linear probe accuracy: {accuracy:.4f}")
```

---

## Advanced Usage

### Custom Encoder Architecture

You can modify the encoder by subclassing:

```python
@keras.saving.register_keras_serializable()
class CustomMAE(MaskedAutoencoder):
    """MAE with custom encoder architecture."""
    
    def _create_encoder_placeholder(self):
        """Override to use custom encoder."""
        # Your custom encoder architecture
        layers_list = [
            keras.layers.Conv2D(64, 7, strides=4, padding='same'),
            keras.layers.LayerNormalization(),
            keras.layers.Activation('gelu'),
            # ... more layers
        ]
        return keras.Sequential(layers_list, name='custom_encoder')

# Use custom MAE
custom_mae = CustomMAE(
    encoder_dims=[64, 128, 256, 512],
    encoder_output_shape=(7, 7, 512),
    patch_size=16,
    mask_ratio=0.75
)
```

### Different Masking Strategies

Experiment with masking parameters:

```python
# Lower mask ratio (easier task, faster convergence)
mae_easy = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    mask_ratio=0.5,  # Only 50% masked
    patch_size=16
)

# Learnable mask tokens
mae_learnable = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    mask_value="learnable",  # Learn what to put in masked patches
    mask_ratio=0.75
)

# Zero masking (harder task)
mae_zero = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    mask_value="zero",  # Replace with zeros
    mask_ratio=0.75
)
```

### Multi-GPU Training

Scale training to multiple GPUs:

```python
import tensorflow as tf

# Create strategy
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create model within strategy scope
with strategy.scope():
    mae = MaskedAutoencoder(
        encoder_dims=[96, 192, 384, 768],
        encoder_output_shape=(7, 7, 768),
        patch_size=16,
        mask_ratio=0.75
    )
    
    mae.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4 * strategy.num_replicas_in_sync)
    )

# Train (batch size will be split across GPUs)
mae.fit(train_dataset, epochs=100)
```

---

## Best Practices

### 1. Dataset Size and Quality

- **Minimum size**: 10k images for meaningful results
- **Ideal size**: 100k+ images for strong representations
- **Image quality**: Use diverse, high-quality images
- **Augmentation**: Apply augmentation during training

### 2. Hyperparameter Selection

**Mask Ratio:**
- Start with 0.75 (standard)
- Lower (0.5-0.6) for easier convergence
- Higher (0.8-0.9) for harder task, potentially better representations

**Learning Rate:**
- Start with 1e-4 to 1e-3
- Use warmup (10% of total steps)
- Apply cosine decay

**Batch Size:**
- Larger is better (64-256)
- Scale learning rate with batch size
- Use gradient accumulation if GPU memory limited

### 3. Training Duration

- **Small datasets (<50k)**: 100-200 epochs
- **Medium datasets (50k-500k)**: 200-400 epochs
- **Large datasets (>500k)**: 400-800 epochs
- Monitor validation loss for early stopping

### 4. Architecture Choices

**Encoder Size:**
- Pico/Nano: Fast experiments, small datasets
- Tiny/Base: Production use, balanced performance
- Large/Huge: Best quality, requires more compute

**Decoder Size:**
- Keep lightweight (4-8 layers)
- Decoder quality doesn't affect encoder learning much
- Faster decoder = faster training

### 5. Fine-tuning Strategy

1. **Linear probe first**: Evaluate representations without fine-tuning
2. **Freeze then unfreeze**: Train head, then full model
3. **Lower learning rate**: Use 10-100x lower LR than pretraining
4. **Regularization**: Add dropout, weight decay during fine-tuning

---

## Troubleshooting

### Problem: Poor Reconstructions

**Symptoms**: Blurry or incoherent reconstructed images

**Solutions:**
- Train longer (more epochs)
- Lower mask ratio (0.5-0.6)
- Check data normalization (should be [0, 1])
- Increase decoder capacity
- Verify patch size divides image dimensions

### Problem: Training Instability

**Symptoms**: Loss spikes, NaN values

**Solutions:**
- Lower learning rate
- Add gradient clipping
- Check for corrupted images in dataset
- Use mixed precision training carefully
- Verify data is normalized properly

```python
# Add gradient clipping
mae.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        clipnorm=1.0  # Clip gradients
    )
)
```

### Problem: Out of Memory

**Symptoms**: GPU OOM errors

**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Lower decoder capacity
- Use mixed precision training

```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Use gradient accumulation
steps_per_epoch = len(train_dataset)
accumulation_steps = 4

for epoch in range(epochs):
    for step, batch in enumerate(train_dataset):
        # Accumulate gradients
        with tf.GradientTape() as tape:
            predictions = mae(batch, training=True)
            loss = mae.compute_loss(x=batch, y_pred=predictions)
            loss = loss / accumulation_steps
        
        grads = tape.gradient(loss, mae.trainable_variables)
        
        if (step + 1) % accumulation_steps == 0:
            mae.optimizer.apply_gradients(zip(grads, mae.trainable_variables))
```

### Problem: Slow Training

**Symptoms**: Very slow iterations per epoch

**Solutions:**
- Use tf.data with prefetching
- Increase num_parallel_calls
- Cache dataset if it fits in memory
- Use SSD for data storage
- Precompute augmentations

```python
dataset = dataset.cache()  # Cache after expensive operations
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### Problem: Poor Transfer Performance

**Symptoms**: Fine-tuned model doesn't perform well

**Solutions:**
- Train MAE longer
- Try higher mask ratio
- Use more pretraining data
- Adjust fine-tuning learning rate
- Check for distribution shift between pretrain and fine-tune data

---

## Complete Example: End-to-End Workflow

Here's a complete example from data loading to fine-tuning:

```python
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from masked_autoencoder import MaskedAutoencoder, visualize_reconstruction

# ============================================
# 1. DATA PREPARATION
# ============================================

def create_dataset(image_paths, batch_size=64):
    """Create tf.data pipeline."""
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Load unlabeled images
unlabeled_paths = [str(p) for p in Path("data/unlabeled").glob("*.jpg")]
train_dataset = create_dataset(unlabeled_paths, batch_size=64)

# ============================================
# 2. MAE PRETRAINING
# ============================================

# Create MAE
mae = MaskedAutoencoder(
    encoder_dims=[96, 192, 384, 768],
    encoder_output_shape=(7, 7, 768),
    patch_size=16,
    mask_ratio=0.75,
    input_shape=(224, 224, 3)
)

# Compile
mae.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4))

# Train
history = mae.fit(
    train_dataset,
    epochs=100,
    callbacks=[
        keras.callbacks.ModelCheckpoint('mae_best.keras', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Save
mae.save('mae_trained.keras')

# ============================================
# 3. VISUALIZATION
# ============================================

# Load test images
test_images = np.load('data/test_images.npy')

# Visualize
grid = visualize_reconstruction(mae, test_images, num_samples=4)
plt.figure(figsize=(15, 10))
plt.imshow(grid)
plt.axis('off')
plt.savefig('results/mae_reconstructions.png')

# ============================================
# 4. FINE-TUNING FOR CLASSIFICATION
# ============================================

# Load labeled data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Resize to match MAE input
train_images = tf.image.resize(train_images, [224, 224]).numpy()
test_images = tf.image.resize(test_images, [224, 224]).numpy()

# Extract encoder
encoder = mae.encoder

# Build classifier
inputs = keras.Input(shape=(224, 224, 3))
features = encoder(inputs)
x = keras.layers.GlobalAveragePooling2D()(features)
x = keras.layers.LayerNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

classifier = keras.Model(inputs, outputs)

# Stage 1: Freeze encoder
encoder.trainable = False
classifier.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Stage 2: Fine-tune encoder
encoder.trainable = True
classifier.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.fit(train_images, train_labels, epochs=40, validation_split=0.1)

# Evaluate
test_loss, test_acc = classifier.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save
classifier.save('classifier_finetuned.keras')
```

---

## Conclusion

The MAE framework provides a powerful approach to self-supervised learning for computer vision. Key takeaways:

1. **Pretrain on unlabeled data** with high mask ratios (0.75)
2. **Train for sufficient epochs** (100-400 depending on dataset size)
3. **Extract the encoder** for downstream tasks
4. **Fine-tune carefully** with lower learning rates
5. **Visualize regularly** to monitor learning progress

For questions or issues, refer to the troubleshooting section or check the framework documentation.