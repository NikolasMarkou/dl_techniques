# Vision Framework Quick Reference

## 🚀 30-Second Start

```python
from vision_training_framework import TrainingConfig, TrainingPipeline
from vision_training_examples import CIFAR10DatasetBuilder, build_simple_cnn

config = TrainingConfig(input_shape=(32, 32, 3), num_classes=10, epochs=50)
dataset_builder = CIFAR10DatasetBuilder(config)
pipeline = TrainingPipeline(config)
model, history = pipeline.run(build_simple_cnn, dataset_builder)
```

## 📋 Configuration Cheat Sheet

```python
TrainingConfig(
    # Must set these
    input_shape=(H, W, C),
    num_classes=N,
    
    # Common to tune
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    optimizer_type='adamw',  # 'adam', 'sgd', 'lion'
    lr_schedule_type='cosine',  # 'exponential', 'reduce_on_plateau', 'constant'
    weight_decay=1e-4,
    
    # Usually leave as default
    gradient_clipping=1.0,
    early_stopping_patience=25,
    monitor_metric='val_accuracy',  # or 'val_loss'
    monitor_mode='max',  # or 'min' for loss
    enable_visualization=True,
    enable_analysis=True,
)
```

## 📦 Dataset Builder Template

```python
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
    
    def get_test_data(self):  # Optional
        return DataInput(x_data=x_test, y_data=y_test)
```

## 🏗️ Model Builder Template

```python
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

## 🎯 Common Patterns

### Pattern 1: Standard Classification

```python
config = TrainingConfig(
    input_shape=(224, 224, 3),
    num_classes=1000,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    optimizer_type='adamw',
    lr_schedule_type='cosine'
)
```

### Pattern 2: Small Dataset (CIFAR-10)

```python
config = TrainingConfig(
    input_shape=(32, 32, 3),
    num_classes=10,
    epochs=200,
    batch_size=128,
    learning_rate=1e-3,
    early_stopping_patience=20
)
```

### Pattern 3: Large Model, Limited Memory

```python
config = TrainingConfig(
    batch_size=16,  # Small batch
    gradient_clipping=0.5,  # Aggressive clipping
    learning_rate=5e-4,  # Lower LR for stability
)

# Also use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')
```

### Pattern 4: Quick Experiment

```python
config = TrainingConfig(
    epochs=10,
    enable_visualization=False,
    enable_analysis=False,
    early_stopping_patience=3
)
```

## 🔧 Command-Line Quick Reference

```bash
# Basic training
python train.py --model simple_cnn --dataset cifar10 --epochs 50

# Custom hyperparameters
python train.py \
    --model convnext_tiny \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --optimizer adamw \
    --lr-schedule cosine

# Quick experiment
python train.py --epochs 10 --no-analysis --no-visualization

# Resume from config
python train.py --config my_experiment/config.json
```

## 📊 Output Directory Structure

```
results/
└── experiment_name_timestamp/
    ├── config.json                   # Your configuration
    ├── best_model.keras              # Best checkpoint
    ├── final_model.keras             # Final model
    ├── training_log.csv              # CSV log
    ├── tensorboard_logs/             # TensorBoard
    ├── visualizations/               # Training plots
    │   ├── training_curves.png
    │   ├── lr_schedule.png
    │   └── network_architecture.png
    └── analysis/                     # Post-training analysis
        ├── summary_dashboard.png
        ├── training_dynamics.png
        └── ...
```

## 🎨 Augmentation Examples

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

## 🏋️ Model Architecture Snippets

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

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `batch_size`, enable mixed precision |
| Not converging | Lower `learning_rate`, increase `gradient_clipping` |
| Overfitting | Add dropout, augmentation, increase `weight_decay` |
| Underfitting | Increase model capacity, train longer |
| Slow training | Check data pipeline, use `AUTOTUNE`, enable prefetching |
| Visualization errors | Check matplotlib install, verify output directory permissions |

## 📈 Hyperparameter Guidelines

### Learning Rate by Architecture
```python
# ResNet family
lr = 1e-3, weight_decay = 1e-4

# ConvNeXt family
lr = 4e-3, weight_decay = 0.05

# Vision Transformers
lr = 5e-4, weight_decay = 0.05

# EfficientNet family
lr = 1e-3, weight_decay = 1e-5
```

### Batch Size Guidelines
```python
# Small datasets (CIFAR-10)
batch_size = 128

# Medium datasets (ImageNet)
batch_size = 64-256

# Large models
batch_size = 16-32

# Rule of thumb: Adjust LR proportionally
# If you double batch_size, multiply LR by sqrt(2)
```

### Schedule Selection
```python
# 'cosine': Smooth decay, best for most cases
# 'exponential': Stepwise decay, aggressive
# 'reduce_on_plateau': Adaptive, good for unstable training
# 'constant': No decay, simple baselines
```

## 💾 Saving & Loading

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

## 🔍 Accessing Results

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

## 🔗 Integration Examples

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
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    pipeline = TrainingPipeline(config)
    model, history = pipeline.run(...)
```

## 🎓 Learning Resources

- **Understand visualizations**: Check `analysis/summary_dashboard.png` first
- **Debug training**: Use TensorBoard: `tensorboard --logdir results/*/tensorboard_logs`
- **Monitor progress**: Check `training_log.csv` for epoch metrics
- **Analyze calibration**: Review `analysis/confidence_calibration_analysis.png`
- **Inspect architecture**: See `visualizations/network_architecture.png`

## 🚦 Quick Decision Tree

```
Training a new model?
├─ Dataset ready? → Use existing DatasetBuilder
├─ New dataset? → Create DatasetBuilder subclass
├─ Standard architecture? → Use existing model builder
├─ Custom architecture? → Create model builder function
├─ Quick experiment? → Set epochs=10, disable analysis
└─ Production run? → Enable all features, save config

Having issues?
├─ OOM? → Reduce batch_size or use mixed precision
├─ Not learning? → Check LR, try lower value
├─ Slow? → Optimize data pipeline, use AUTOTUNE
└─ Crashes? → Check logs, reduce complexity
```

---

**Remember**: Start simple, iterate quickly, monitor closely! 🚀