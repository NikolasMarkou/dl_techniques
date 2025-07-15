# N-BEATS Data Loader Usage Examples

This guide demonstrates how to use the refined N-BEATS data loader with various data sources and configurations.

## Quick Start

### 1. Synthetic Data Generation

```python
from dl_techniques.data.nbeats import create_synthetic_dataset, SyntheticDataGenerator

# Create a complete synthetic dataset
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_synthetic_dataset(
    backcast_length=48,
    forecast_length=12,
    num_samples=1000,
    signal_type='mixed',
    add_noise=True,
    noise_std=0.1,
    random_seed=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Create a data generator for training
generator = SyntheticDataGenerator(
    backcast_length=48,
    forecast_length=12,
    signal_type='seasonality',
    batch_size=32,
    add_noise=True,
    random_seed=42
)

data_gen = generator.create_generator()

# Generate a batch
X_batch, y_batch = next(data_gen)
print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")
```

### 2. Loading Real-World Datasets

```python
from dl_techniques.data.nbeats import load_dataset, get_recommended_config

# Load M4 dataset with recommended configuration
config = get_recommended_config('m4')
print(f"Recommended M4 config: {config}")

# Load M4 Daily data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
    'm4',
    backcast_length=48,
    forecast_length=14,
    frequency='Daily',
    max_series_count=100  # Limit for demo
)

print(f"M4 Dataset loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

### 3. Custom CSV Data Loading

```python
from dl_techniques.data.nbeats import load_csv_data

# Load time series from CSV
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_csv_data(
    filepath='data/my_timeseries.csv',
    backcast_length=48,
    forecast_length=12,
    value_column=1,  # Column index containing values
    header=True,
    delimiter=',',
    validation_split=0.2,
    test_split=0.2,
    normalize=True,
    normalization_method='minmax'
)
```

## Advanced Usage

### 1. Multivariate Time Series Generation

```python
from dl_techniques.data.nbeats import MultivariateSyntheticGenerator

# Create multivariate synthetic data
mv_generator = MultivariateSyntheticGenerator(
    backcast_length=48,
    forecast_length=12,
    num_variables=3,
    batch_size=32,
    correlation_strength=0.7,
    noise_std=0.05,
    random_seed=42
)

# Generate multivariate batch
X_mv, y_mv = mv_generator.generate_batch()
print(f"Multivariate batch shape: X={X_mv.shape}, y={y_mv.shape}")

# Use as generator
mv_data_gen = mv_generator.create_generator()
```

### 2. Custom Normalization

```python
from dl_techniques.data.nbeats import TimeSeriesNormalizer
import numpy as np

# Create sample data
data = np.random.randn(100, 48, 1)

# Initialize normalizer
normalizer = TimeSeriesNormalizer(
    method='standard',  # or 'minmax', 'robust'
    feature_range=(0, 1)
)

# Fit and transform
normalized_data = normalizer.fit_transform(data)
print(f"Normalized data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

# Inverse transform
original_data = normalizer.inverse_transform(normalized_data)
print(f"Reconstruction error: {np.mean(np.abs(data - original_data)):.6f}")
```

### 3. Specific Dataset Loaders

#### M4 Competition Data

```python
from dl_techniques.data.nbeats import M4DataLoader

# Load specific M4 frequency
m4_loader = M4DataLoader(
    backcast_length=48,
    forecast_length=14,
    frequency='Daily',
    data_dir='data/m4',
    min_series_length=100,
    max_series_count=500,
    normalize=True,
    normalization_method='minmax'
)

# Load datasets
(X_train, y_train), (X_val, y_val), (X_test, y_test) = m4_loader.load_datasets()
```

#### ECG Data

```python
from dl_techniques.data.nbeats import ECGDataLoader

# Load ECG data (requires wfdb package)
ecg_loader = ECGDataLoader(
    backcast_length=100,
    forecast_length=25,
    data_dir='data/ecg',
    sampling_rate=360,
    max_samples=50,
    normalize=True,
    normalization_method='standard'
)

# Load datasets
(X_train, y_train), (X_val, y_val), (X_test, y_test) = ecg_loader.load_datasets()
```

#### Energy Data

```python
from dl_techniques.data.nbeats import EnergyDataLoader

# Load energy data with exogenous variables
energy_loader = EnergyDataLoader(
    backcast_length=168,  # 1 week of hourly data
    forecast_length=24,   # 1 day ahead
    data_dir='data/energy',
    include_exogenous=True,
    normalize=True,
    normalization_method='minmax'
)

# Load datasets (returns exogenous variables too)
(X_train, y_train, exog_train), (X_val, y_val, exog_val), (X_test, y_test, exog_test) = energy_loader.load_datasets()
```

## Data Preprocessing Pipeline

### Complete Preprocessing Example

```python
from dl_techniques.data.nbeats import (
    RealDataLoader, 
    TimeSeriesNormalizer,
    create_data_generator
)
import numpy as np

# Step 1: Load raw data
raw_data = np.random.randn(1000)  # Your time series data

# Step 2: Create data loader
loader = RealDataLoader(
    backcast_length=48,
    forecast_length=12,
    validation_split=0.2,
    test_split=0.2,
    normalize=True,
    normalization_method='minmax'
)

# Step 3: Split and process data
train_data, val_data, test_data = loader._split_and_process_data(raw_data)

# Step 4: Create datasets
(X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.create_datasets(
    train_data, val_data, test_data
)

# Step 5: Get normalizer for inverse transformation
normalizer = loader.normalizer

print(f"Processing complete:")
print(f"  Train: {X_train.shape} -> {y_train.shape}")
print(f"  Val: {X_val.shape} -> {y_val.shape}")
print(f"  Test: {X_test.shape} -> {y_test.shape}")
```

## Integration with N-BEATS Model

### Complete Training Pipeline

```python
from dl_techniques.models.nbeats import NBeatsNet
from dl_techniques.losses.smape_loss import SMAPELoss
from dl_techniques.data.nbeats import create_synthetic_dataset
import keras

# Step 1: Create dataset
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_synthetic_dataset(
    backcast_length=48,
    forecast_length=12,
    num_samples=1000,
    signal_type='mixed',
    random_seed=42
)

# Step 2: Create model
model = NBeatsNet(
    backcast_length=48,
    forecast_length=12,
    stack_types=['trend', 'seasonality'],
    nb_blocks_per_stack=3,
    thetas_dim=[4, 8],
    hidden_layer_units=256
)

# Step 3: Compile model
model.compile(
    optimizer='adam',
    loss=SMAPELoss(),
    metrics=['mae']
)

# Step 4: Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Step 5: Evaluate model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss}")

# Step 6: Make predictions
predictions = model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
```

## Data Quality and Monitoring

### Dataset Information

```python
from dl_techniques.data.nbeats import list_available_datasets, check_dataset_availability

# Check available datasets
available = list_available_datasets('data')
print(f"Available datasets: {available}")

# Check specific dataset
is_available = check_dataset_availability('m4', 'data')
print(f"M4 dataset available: {is_available}")
```

### Data Validation

```python
def validate_dataset(X, y, backcast_length, forecast_length):
    """Validate dataset shapes and properties."""
    print(f"Dataset validation:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Expected backcast length: {backcast_length}")
    print(f"  Expected forecast length: {forecast_length}")
    
    # Check shapes
    assert X.shape[1] == backcast_length, f"X sequence length mismatch: {X.shape[1]} != {backcast_length}"
    assert y.shape[1] == forecast_length, f"y sequence length mismatch: {y.shape[1]} != {forecast_length}"
    assert X.shape[0] == y.shape[0], f"Sample count mismatch: {X.shape[0]} != {y.shape[0]}"
    
    # Check for NaN/Inf values
    assert not np.any(np.isnan(X)), "X contains NaN values"
    assert not np.any(np.isnan(y)), "y contains NaN values"
    assert not np.any(np.isinf(X)), "X contains Inf values"
    assert not np.any(np.isinf(y)), "y contains Inf values"
    
    print("  ✓ Dataset validation passed")

# Example usage
X, y = create_synthetic_dataset(48, 12, 1000)[0]
validate_dataset(X, y, 48, 12)
```

## Performance Optimization

### Memory-Efficient Data Loading

```python
from dl_techniques.data.nbeats import SyntheticDataGenerator

# Use generators for large datasets
def create_memory_efficient_generator(config):
    """Create memory-efficient data generator."""
    generator = SyntheticDataGenerator(
        backcast_length=config['backcast_length'],
        forecast_length=config['forecast_length'],
        batch_size=config['batch_size'],
        signal_type=config['signal_type'],
        add_noise=config['add_noise'],
        noise_std=config['noise_std']
    )
    
    return generator.create_generator()

# Configuration
config = {
    'backcast_length': 48,
    'forecast_length': 12,
    'batch_size': 64,
    'signal_type': 'mixed',
    'add_noise': True,
    'noise_std': 0.1
}

# Create generator
data_gen = create_memory_efficient_generator(config)

# Use with model.fit
# model.fit(data_gen, steps_per_epoch=100, epochs=50)
```

### Parallel Data Loading

```python
import keras
from dl_techniques.data.nbeats import create_data_generator

# Create dataset from generator
generator = create_data_generator(
    backcast_length=48,
    forecast_length=12,
    signal_type='mixed',
    batch_size=32
)

# Convert to tf.data.Dataset for better performance
def generator_fn():
    while True:
        X, y = next(generator)
        yield X, y

# Create tf.data.Dataset
dataset = keras.utils.experimental.dataset_from_generator(
    generator_fn,
    output_signature=(
        keras.utils.experimental.TensorSpec(shape=(32, 48, 1), dtype='float32'),
        keras.utils.experimental.TensorSpec(shape=(32, 12, 1), dtype='float32')
    )
)

# Apply optimizations
dataset = dataset.prefetch(keras.utils.experimental.AUTOTUNE)
dataset = dataset.cache()  # Cache if data fits in memory

# Use with model training
# model.fit(dataset, steps_per_epoch=100, epochs=50)
```

## Best Practices

1. **Data Normalization**: Always normalize your data for better training stability
2. **Validation Strategy**: Use temporal splits for time series validation
3. **Batch Size**: Start with 32 and adjust based on memory constraints
4. **Memory Management**: Use generators for large datasets
5. **Data Quality**: Validate your data before training
6. **Reproducibility**: Set random seeds for consistent results
7. **Monitoring**: Track data statistics during training

This refined data loader provides a comprehensive, production-ready solution for N-BEATS time series forecasting with proper error handling, normalization, and integration with the project architecture.