# Vision Training Framework

## Overview

The Vision Training Framework is a comprehensive, production-ready solution for training deep learning vision models with minimal boilerplate code. Built on top of Keras 3 and TensorFlow, it integrates advanced optimization techniques, comprehensive visualization capabilities, and automated model analysis into a unified, configurable pipeline.

### Key Features

**Training Infrastructure**
- **Advanced Optimization**: Built-in support for modern optimizers (AdamW, Adam, SGD, RMSprop) with gradient clipping
- **Learning Rate Schedules**: Cosine decay, exponential decay, cosine restarts with automatic warmup
- **Smart Callbacks**: Model checkpointing, early stopping, TensorBoard logging, CSV export

**Visualization & Monitoring**
- **Real-time Training Curves**: Loss, accuracy, and custom metrics with smoothing
- **Learning Rate Tracking**: Visualize schedule behavior including warmup phase
- **Convergence Analysis**: Monitor training stability, gradient flow, and convergence rates
- **Overfitting Detection**: Automatic analysis of generalization gap and overfitting onset
- **Classification Metrics**: Confusion matrices, ROC/PR curves, per-class performance
- **Neural Network Inspection**: Architecture diagrams, weight distributions, activation patterns

**Analysis & Reporting**
- **Automated Model Analysis**: Weight statistics, calibration metrics, information flow
- **Classification Reports**: Detailed per-class precision, recall, F1-scores
- **Error Analysis**: Misclassification patterns, confidence distributions
- **Comprehensive Dashboards**: Multi-plot summaries combining key visualizations

**Developer Experience**
- **Minimal Boilerplate**: Define config → build dataset → build model → train
- **Reproducible Experiments**: All configurations saved as JSON
- **Extensible Design**: Abstract base classes for custom datasets and models
- **Command-line Interface**: Full CLI support for quick experimentation

### Quick Start

```python
from dl_techniques.optimization.train_vision import (
    TrainingConfig, TrainingPipeline, DatasetBuilder
)
import keras

# 1. Configure training
config = TrainingConfig(
    input_shape=(224, 224, 3),
    num_classes=10,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    lr_schedule_type='cosine_decay',
    warmup_steps=1000,
    optimizer_type='adamw',
    weight_decay=1e-4
)

# 2. Define dataset builder
class MyDataset(DatasetBuilder):
    def build(self):
        # Load and preprocess your data
        train_ds = ...  # tf.data.Dataset
        val_ds = ...    # tf.data.Dataset
        return train_ds, val_ds, steps_per_epoch, val_steps

# 3. Define model builder
def build_model(config):
    model = keras.Sequential([...])
    return model

# 4. Run training pipeline
pipeline = TrainingPipeline(config)
model, history = pipeline.run(build_model, MyDataset(config))
```

That's it! The framework handles:
- Optimizer and schedule configuration with warmup
- Model compilation with appropriate loss and metrics
- Training with callbacks (checkpointing, early stopping, logging)
- Real-time visualization generation (curves, LR schedule, convergence)
- Post-training analysis (classification metrics, model analysis)
- Comprehensive dashboards and reports

---

## Configuration Cheat Sheet

### Basic Configuration
```python
TrainingConfig(
    # Must set these
    input_shape=(H, W, C),
    num_classes=N,
    
    # Common to tune
    epochs=100,
    batch_size=64,
    
    # Learning Rate & Schedule (with warmup support)
    learning_rate=1e-3,
    lr_schedule_type='cosine_decay',  # 'exponential_decay', 'cosine_decay_restarts', 'constant'
    warmup_steps=1000,                # Linear warmup period
    warmup_start_lr=1e-8,             # Starting LR for warmup
    alpha=0.0001,                     # Min LR fraction (cosine schedules)
    
    # Optimizer Configuration
    optimizer_type='adamw',           # 'adam', 'sgd', 'rmsprop', 'adadelta'
    weight_decay=1e-4,                # For AdamW
    beta_1=0.9,                       # Adam/AdamW first moment
    beta_2=0.999,                     # Adam/AdamW second moment
    
    # Gradient Clipping
    gradient_clipping_norm_global=1.0,  # Global norm clipping
    # gradient_clipping_norm_local=1.0, # Per-variable norm clipping
    # gradient_clipping_value=0.5,      # Value clipping
    
    # Visualization & Analysis
    enable_visualization=True,
    enable_analysis=True,
    enable_convergence_analysis=True,      # Convergence dashboard
    enable_overfitting_analysis=True,      # Overfitting detection
    enable_classification_viz=True,        # Confusion matrix, ROC curves, etc.
    create_final_dashboard=True,           # Comprehensive multi-plot dashboard
    enable_gradient_tracking=False,        # Track gradients (may slow training)
    visualization_frequency=10,            # Create plots every N epochs
    
    # Training Control
    early_stopping_patience=25,
    monitor_metric='val_accuracy',  # or 'val_loss'
    monitor_mode='max',  # or 'min' for loss
)
```

### Advanced Schedule Configuration
```python
TrainingConfig(
    # Exponential Decay
    lr_schedule_type='exponential_decay',
    learning_rate=1e-3,
    decay_rate=0.9,
    decay_steps=1000,  # Auto-calculated if None
    warmup_steps=1000,
    
    # OR Cosine Decay with Restarts
    lr_schedule_type='cosine_decay_restarts',
    learning_rate=1e-3,
    t_mul=2.0,         # Period multiplier
    m_mul=0.9,         # LR multiplier
    alpha=0.001,
    warmup_steps=2000,
    
    # OR Constant (no schedule)
    lr_schedule_type='constant',
    learning_rate=1e-3,
    warmup_steps=0,    # No warmup for constant LR
)
```

## Dataset Builder Template

```python
from dl_techniques.optimization.train_vision import DatasetBuilder
from dl_techniques.analyzer import DataInput


class MyDatasetBuilder(DatasetBuilder):
    def build(self):
        # 1. Load data
        (x_train, y_train), (x_val, y_val) = load_my_data()

        # 2. Create tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        # 3. Apply preprocessing and augmentation
        train_ds = (train_ds
                    .shuffle(10000)
                    .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                    .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self.config.batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds
                  .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.config.batch_size)
                  .prefetch(tf.data.AUTOTUNE))

        # 4. Calculate steps
        steps_per_epoch = len(x_train) // self.config.batch_size
        val_steps = len(x_val) // self.config.batch_size

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _preprocess(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def _augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def get_test_data(self):  # Optional - enables classification visualizations
        return DataInput(x_data=x_test, y_data=y_test)
    
    def get_class_names(self):  # Optional - enables labeled visualizations
        return ['cat', 'dog', 'bird', ...]
```

## Model Builder Template

```python
import keras
from dl_techniques.optimization.train_vision import TrainingConfig


def build_my_model(config: TrainingConfig) -> keras.Model:
    inputs = keras.Input(shape=config.input_shape)

    # Your architecture here
    x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    # ... more layers ...

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(config.num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='my_model')
```

## Common Patterns

### Pattern 1: Standard Classification with Warmup

```python
from dl_techniques.optimization.train_vision import TrainingConfig

config = TrainingConfig(
    input_shape=(224, 224, 3),
    num_classes=1000,
    epochs=100,
    batch_size=64,

    # Learning rate with warmup
    learning_rate=1e-3,
    lr_schedule_type='cosine_decay',
    warmup_steps=1000,  # 5% of training typically
    warmup_start_lr=1e-8,

    optimizer_type='adamw',
    weight_decay=1e-4,
    gradient_clipping_norm_global=1.0,
    
    # Full visualization suite
    enable_visualization=True,
    enable_convergence_analysis=True,
    enable_overfitting_analysis=True,
    enable_classification_viz=True,
    create_final_dashboard=True
)
```

### Pattern 2: Small Dataset (CIFAR-10) with Restarts

```python
config = TrainingConfig(
    input_shape=(32, 32, 3),
    num_classes=10,
    epochs=200,
    batch_size=128,
    
    # Cosine decay with restarts for small datasets
    learning_rate=1e-3,
    lr_schedule_type='cosine_decay_restarts',
    warmup_steps=500,
    t_mul=2.0,               # Double period after each restart
    m_mul=0.9,               # Reduce LR by 10% after each restart
    alpha=0.001,
    
    optimizer_type='adam',
    gradient_clipping_norm_global=1.0,
    early_stopping_patience=20
)
```

### Pattern 3: Transformer/Large Model Training

```python
config = TrainingConfig(
    batch_size=32,          # Smaller batch for memory
    
    # Transformer-style schedule
    learning_rate=5e-4,     # Conservative peak LR
    lr_schedule_type='cosine_decay_restarts',
    warmup_steps=4000,      # Longer warmup for transformers
    warmup_start_lr=1e-9,   # Very low starting LR
    t_mul=1.5,
    m_mul=0.95,
    
    # AdamW with higher beta_2
    optimizer_type='adamw',
    beta_1=0.9,
    beta_2=0.98,            # Higher for transformers
    epsilon=1e-9,
    weight_decay=0.01,
    gradient_clipping_norm_global=1.0,
)

# Also use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')
```

### Pattern 4: Large Model, Limited Memory

```python
config = TrainingConfig(
    batch_size=16,                      # Small batch
    learning_rate=5e-4,                 # Lower LR for stability
    
    # Aggressive gradient clipping
    gradient_clipping_norm_global=0.5,  # Prevent exploding gradients
    gradient_clipping_value=0.1,        # Also clip by value
    
    # Longer warmup for stability
    warmup_steps=2000,
)

keras.mixed_precision.set_global_policy('mixed_float16')
```

### Pattern 5: Quick Experiment (No Warmup)

```python
config = TrainingConfig(
    epochs=10,
    lr_schedule_type='constant',  # No schedule
    warmup_steps=0,               # No warmup
    enable_visualization=False,
    enable_analysis=False,
    early_stopping_patience=3
)
```

### Pattern 6: Research with Full Analysis

```python
config = TrainingConfig(
    # Standard hyperparameters
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    lr_schedule_type='cosine_decay',
    warmup_steps=1000,
    
    # Enable all analysis features
    enable_visualization=True,
    enable_analysis=True,
    enable_convergence_analysis=True,
    enable_overfitting_analysis=True,
    enable_classification_viz=True,
    enable_gradient_tracking=True,  # May slow training
    create_final_dashboard=True,
    visualization_frequency=5,  # More frequent updates
)
```

## Command-Line Quick Reference

```bash
# Basic training with warmup
python train.py --model simple_cnn --dataset cifar10 \
    --epochs 50 --warmup-steps 1000

# Custom hyperparameters
python train.py \
    --model convnext_tiny \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --optimizer adamw \
    --lr-schedule cosine_decay \
    --warmup-steps 2000 \
    --alpha 0.0001 \
    --weight-decay 0.0001 \
    --gradient-clip 1.0

# Cosine decay with restarts
python train.py \
    --lr-schedule cosine_decay_restarts \
    --warmup-steps 1000 \
    --learning-rate 0.001

# Quick experiment (minimal features)
python train.py --epochs 10 \
    --lr-schedule constant \
    --warmup-steps 0 \
    --no-analysis \
    --no-visualization \
    --no-convergence-analysis \
    --no-overfitting-analysis

# Full research mode with all features
python train.py \
    --epochs 100 \
    --enable-gradient-tracking \
    --visualization-frequency 5

# Disable specific visualizations
python train.py \
    --no-convergence-analysis \
    --no-classification-viz \
    --no-final-dashboard

# Resume from config
python train.py --config my_experiment/config.json
```

## Output Directory Structure

```
results/
└── experiment_name_timestamp/
    ├── config.json                   # Your configuration
    ├── best_model.keras              # Best checkpoint
    ├── final_model.keras             # Final model
    ├── training_log.csv              # CSV log
    ├── tensorboard_logs/             # TensorBoard
    ├── visualizations/               # Training visualizations
    │   ├── training_curves.png       # Loss and metrics over time
    │   ├── lr_schedule.png           # Learning rate schedule
    │   ├── convergence_analysis.png  # Convergence dashboard
    │   ├── overfitting_analysis.png  # Overfitting detection
    │   ├── network_architecture.png  # Model architecture
    │   ├── class_balance.png         # Dataset class distribution
    │   ├── confusion_matrix.png      # Confusion matrix
    │   ├── roc_pr_curves.png         # ROC and PR curves
    │   ├── classification_report.png # Per-class metrics
    │   ├── per_class_analysis.png    # Detailed per-class dashboard
    │   ├── error_analysis.png        # Misclassification analysis
    │   └── final_dashboard.png       # Comprehensive summary
    └── analysis/                     # Post-training analysis
        ├── summary_dashboard.png
        ├── training_dynamics.png
        ├── weight_distributions.png
        ├── calibration_analysis.png
        └── ...
```

## Visualization Features

### Automatic Real-Time Visualizations

The framework automatically generates visualizations during training:

**Training Curves** (`training_curves.png`)
- Loss and metrics over epochs
- Separate train/val curves
- Smoothing support
- Best epoch highlighting

**Learning Rate Schedule** (`lr_schedule.png`)
- LR progression over epochs
- Warmup phase visualization
- Schedule type indication

**Convergence Analysis** (`convergence_analysis.png`)
- Loss convergence monitoring
- Gradient flow analysis (if enabled)
- Validation gap tracking
- Convergence rate estimation

**Overfitting Analysis** (`overfitting_analysis.png`)
- Generalization gap over time
- Overfitting onset detection
- Train/val metric divergence
- Patience threshold visualization

### Post-Training Classification Visualizations

When test data is provided, the framework generates:

**Confusion Matrix** (`confusion_matrix.png`)
- True vs predicted labels heatmap
- Normalization options
- Percentage annotations

**ROC and PR Curves** (`roc_pr_curves.png`)
- Receiver Operating Characteristic curves
- Precision-Recall curves
- Per-class and micro/macro averages
- AUC scores

**Classification Report** (`classification_report.png`)
- Precision, recall, F1-score per class
- Support counts
- Overall accuracy

**Per-Class Analysis** (`per_class_analysis.png`)
- Class distribution
- Per-class accuracy
- Common confusions
- Hardest examples

**Error Analysis** (`error_analysis.png`)
- Error rates by class
- Confidence distributions
- Misclassification patterns
- Error hotspots

### Network Inspection Visualizations

**Architecture Diagram** (`network_architecture.png`)
- Layer types and connections
- Shape information
- Parameter counts

**Class Balance** (`class_balance.png`)
- Dataset class distribution
- Pie and bar charts
- Imbalance detection

### Final Dashboard

The `final_dashboard.png` combines multiple visualizations into a single comprehensive view for easy presentation and reporting.

## Augmentation Examples

### Basic Augmentation
```python
def _augment(self, image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
```

### Advanced Augmentation
```python
self.augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.2),
])

# In build():
train_ds = train_ds.map(
    lambda x, y: (self.augmentation(x, training=True), y)
)
```

### Cutout / Random Erasing
```python
def random_erasing(image, probability=0.5, sl=0.02, sh=0.4):
    if tf.random.uniform([]) > probability:
        return image
    
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    area = tf.cast(h * w, tf.float32)
    
    erase_area = tf.random.uniform([], sl, sh) * area
    aspect_ratio = tf.random.uniform([], 0.3, 1/0.3)
    
    h_erase = tf.cast(tf.math.sqrt(erase_area * aspect_ratio), tf.int32)
    w_erase = tf.cast(tf.math.sqrt(erase_area / aspect_ratio), tf.int32)
    
    # Apply erasing...
    return image
```

## Model Architecture Snippets

### Residual Block
```python
def residual_block(x, filters):
    shortcut = x
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    
    if keras.ops.shape(shortcut)[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, 1)(shortcut)
    
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x
```

### ConvNeXt Block
```python
def convnext_block(x, filters):
    shortcut = x
    x = keras.layers.DepthwiseConv2D(7, padding='same')(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dense(filters * 4, activation='gelu')(x)
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Add()([x, shortcut])
    return x
```

### Squeeze-and-Excitation Block
```python
def se_block(x, ratio=16):
    filters = keras.ops.shape(x)[-1]
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = keras.layers.Dense(filters, activation='sigmoid')(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.Multiply()([x, se])
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `batch_size`, enable mixed precision, use gradient checkpointing |
| Not converging | Lower `learning_rate`, increase `warmup_steps`, check `gradient_clipping_norm_global` |
| Overfitting | Add dropout, augmentation, increase `weight_decay`, check overfitting analysis |
| Underfitting | Increase model capacity, train longer, reduce regularization |
| Slow training | Check data pipeline, use `AUTOTUNE`, enable prefetching |
| Unstable early training | Increase `warmup_steps`, lower `warmup_start_lr`, check convergence analysis |
| Plateaus mid-training | Try `cosine_decay_restarts` instead of `cosine_decay` |
| Visualization errors | Check matplotlib install, verify output directory permissions |
| Missing classification viz | Ensure `get_test_data()` returns valid data, enable `enable_classification_viz` |

## Hyperparameter Guidelines

### Learning Rate by Architecture
```python
# ResNet family
lr = 1e-3, weight_decay = 1e-4, warmup_steps = 1000

# ConvNeXt family
lr = 4e-3, weight_decay = 0.05, warmup_steps = 2000

# Vision Transformers
lr = 5e-4, weight_decay = 0.05, warmup_steps = 4000, beta_2 = 0.98

# EfficientNet family
lr = 1e-3, weight_decay = 1e-5, warmup_steps = 1000

# Small CNNs (CIFAR-10)
lr = 1e-3, warmup_steps = 500
```

### Warmup Guidelines
```python
# General rule: 5-10% of total training steps
warmup_steps = max(1000, total_steps // 20)

# Small datasets (< 10k samples)
warmup_steps = 500

# Medium datasets (10k-100k samples)
warmup_steps = 1000-2000

# Large datasets (> 100k samples)
warmup_steps = 2000-5000

# Transformers (always use more warmup)
warmup_steps = 4000-10000

# Very low starting point for transformers
warmup_start_lr = 1e-9
```

### Batch Size Guidelines
```python
# Small datasets (CIFAR-10)
batch_size = 128

# Medium datasets (ImageNet)
batch_size = 64-256

# Large models
batch_size = 16-32

# Transformers
batch_size = 32-64

# Rule of thumb: If you double batch_size, multiply LR by sqrt(2)
new_lr = old_lr * sqrt(new_batch_size / old_batch_size)
```

### Schedule Selection
```python
# 'cosine_decay': Smooth decay, best for most cases
#   - Use for: General purpose, standard training
#   - Warmup: 1000 steps recommended

# 'cosine_decay_restarts': Periodic restarts to escape local minima
#   - Use for: When training plateaus, small datasets, fine-tuning
#   - Warmup: 500-1000 steps recommended
#   - t_mul: 1.5-2.0, m_mul: 0.9-0.95

# 'exponential_decay': Stepwise decay, more aggressive
#   - Use for: RNNs, when you need precise control
#   - Warmup: 500-1000 steps recommended

# 'constant': No decay, simple baselines
#   - Use for: Quick experiments, debugging, very small datasets
#   - Warmup: 0 (usually not needed)
```

### Gradient Clipping Guidelines
```python
# Standard training (most cases)
gradient_clipping_norm_global = 1.0

# Transformers / Large models
gradient_clipping_norm_global = 1.0
gradient_clipping_value = None  # Norm clipping usually sufficient

# RNNs / Unstable gradients
gradient_clipping_norm_global = 0.5  # More aggressive
gradient_clipping_value = 0.1       # Also clip by value

# Small models / Stable training
gradient_clipping_norm_global = None  # May not need clipping
```

## Saving & Loading

### Save Configuration
```python
config.save(Path('my_config.json'))
```

### Load Configuration
```python
config = TrainingConfig.load(Path('my_config.json'))
```

### Save Model
```python
# Automatic: best_model.keras and final_model.keras
# Manual:
model.save('my_model.keras')
```

### Load Model
```python
model = keras.models.load_model('my_model.keras')
```

## Accessing Results

### Training Metrics
```python
# From history
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']

# From CSV log
import pandas as pd
log = pd.read_csv('experiment_dir/training_log.csv')
```

### Analysis Results
```python
# Automatically saved to analysis/
# Access programmatically:
summary = analyzer.get_summary_statistics()
ece = summary['calibration_summary']['model_name']['ece']
```

### Visualization Access
```python
# All visualizations saved to visualizations/
# Access them programmatically if needed:
from pathlib import Path
viz_dir = Path('results/experiment_name/visualizations')
confusion_matrix = viz_dir / 'confusion_matrix.png'
```

## Integration Examples

### With WandB
```python
import wandb

# Custom callback
wandb_callback = wandb.keras.WandbCallback()

pipeline.run(
    model_builder=build_model,
    dataset_builder=dataset_builder,
    custom_callbacks=[wandb_callback]
)
```

### With MLflow
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(asdict(config))
    model, history = pipeline.run(...)
    mlflow.log_metrics(history.history)
    mlflow.keras.log_model(model, "model")
```

### Multi-GPU Training

```python
import tensorflow as tf
from dl_techniques.optimization.train_vision import TrainingConfig, TrainingPipeline

strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

# Scale learning rate and warmup with number of GPUs
config = TrainingConfig(
    learning_rate=1e-3 * num_gpus,  # Linear scaling
    warmup_steps=1000 * num_gpus,  # Proportional warmup
    batch_size=64 * num_gpus,  # Effective batch size
)

with strategy.scope():
    pipeline = TrainingPipeline(config)
    model, history = pipeline.run(...)
```

## Learning Resources

- **Understand visualizations**: Check `visualizations/final_dashboard.png` first
- **Debug training**: Use TensorBoard: `tensorboard --logdir results/*/tensorboard_logs`
- **Monitor progress**: Check `training_log.csv` for epoch metrics
- **Check convergence**: Review `visualizations/convergence_analysis.png`
- **Detect overfitting**: See `visualizations/overfitting_analysis.png`
- **Analyze errors**: Inspect `visualizations/error_analysis.png`
- **Verify schedule**: View `visualizations/lr_schedule.png` for warmup behavior
- **Classification metrics**: See `visualizations/confusion_matrix.png` and ROC curves
- **Inspect architecture**: See `visualizations/network_architecture.png`

## Quick Decision Tree

```
Training a new model?
├─ Dataset ready? → Use existing DatasetBuilder
├─ New dataset? → Create DatasetBuilder subclass
├─ Standard architecture? → Use existing model builder
├─ Custom architecture? → Create model builder function
├─ Need warmup? → Set warmup_steps > 0 (recommended for most cases)
├─ Small dataset? → Try cosine_decay_restarts
├─ Large model? → Increase warmup_steps, lower learning_rate
├─ Quick experiment? → Set epochs=10, lr_schedule_type='constant', warmup_steps=0
├─ Research mode? → Enable all visualization and analysis features
└─ Production run? → Enable all features, save config, use warmup

Having issues?
├─ OOM? → Reduce batch_size or use mixed precision
├─ Not learning? → Check LR, increase warmup_steps, review convergence analysis
├─ Unstable early? → Increase warmup_steps, lower warmup_start_lr
├─ Overfitting? → Check overfitting analysis, add regularization
├─ Plateaus? → Try cosine_decay_restarts, check LR schedule visualization
├─ Slow? → Optimize data pipeline, use AUTOTUNE
├─ Classification errors? → Review confusion matrix and per-class analysis
└─ Crashes? → Check logs, reduce complexity
```

## Advanced Topics

### Custom Learning Rate Schedules

If you need a schedule not supported by the framework, you can pass it directly:

```python
# Create custom schedule
custom_schedule = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    end_learning_rate=1e-6,
    power=2.0
)

# Note: This bypasses the optimization module
# For production, consider adding it to dl_techniques.optimization
```

### Hyperparameter Tuning with the Framework

```python
import optuna
from dl_techniques.optimization.train_vision import TrainingConfig, TrainingPipeline


def objective(trial):
    config = TrainingConfig(
        learning_rate=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        warmup_steps=trial.suggest_int('warmup', 500, 5000),
        weight_decay=trial.suggest_float('wd', 1e-6, 1e-3, log=True),
        alpha=trial.suggest_float('alpha', 0.0001, 0.01, log=True),
        enable_visualization=False,
        enable_analysis=False,
    )

    pipeline = TrainingPipeline(config)
    model, history = pipeline.run(build_model, dataset_builder)

    return max(history.history['val_accuracy'])


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Integration with dl_techniques Optimization Module

The framework automatically uses `dl_techniques.optimization.learning_rate_schedule_builder` and 
`dl_techniques.optimization.optimizer_builder`. For advanced use cases, you can 
directly configure these:

```python
from dl_techniques.optimization import learning_rate_schedule_builder, optimizer_builder

# Manual configuration
schedule_config = {
    "type": "cosine_decay_restarts",
    "warmup_steps": 2000,
    "warmup_start_lr": 1e-8,
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "t_mul": 1.5,
    "m_mul": 0.95,
    "alpha": 0.001
}

optimizer_config = {
    "type": "adamw",
    "beta_1": 0.9,
    "beta_2": 0.98,
    "epsilon": 1e-9,
    "gradient_clipping_by_norm": 1.0
}

lr_schedule = learning_rate_schedule_builder(schedule_config)
optimizer = optimizer_builder(optimizer_config, lr_schedule)

# Use in custom training loop
# model.compile(optimizer=optimizer, ...)
```

### Custom Visualization Callbacks

```python
from dl_techniques.optimization.train_vision import EnhancedVisualizationCallback

class CustomVizCallback(EnhancedVisualizationCallback):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        # Add your custom visualization logic
        if (epoch + 1) % self.frequency == 0:
            self._create_custom_plots()
    
    def _create_custom_plots(self):
        # Your custom visualization code
        pass

# Use in training
pipeline.run(
    model_builder=build_model,
    dataset_builder=dataset_builder,
    custom_callbacks=[CustomVizCallback(viz_manager, config)]
)
```

---

**Remember**: 
- Always use warmup for stable training (except quick experiments)
- Start with `warmup_steps=1000` and `learning_rate=1e-3` as defaults
- Use `cosine_decay` for general purpose, `cosine_decay_restarts` for small datasets
- Scale learning rate and warmup proportionally with batch size
- Monitor the LR schedule visualization to verify behavior
- Check convergence and overfitting analysis for training diagnostics
- Enable classification visualizations for detailed model evaluation
- Use the final dashboard for quick comprehensive overviews

**Start simple, iterate quickly, monitor closely with comprehensive visualizations!**