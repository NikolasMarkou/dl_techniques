># Platonic Representation Alignment Module

A Keras 3 implementation of representation alignment metrics from "The Platonic Representation Hypothesis" (Huh et al., ICML 2024).

## Overview

This module provides tools for measuring alignment between neural network representations across different models, architectures, and modalities. It implements various alignment metrics including k-NN based methods, kernel methods, and CCA-based approaches.

## Quick Start

### Basic Usage

```python
from dl_techniques.utils.alignment import Alignment, AlignmentMetrics

# Method 1: Using the high-level Alignment API
scorer = Alignment(metric="mutual_knn", topk=10, normalize=True)

# Compute alignment between two feature sets
score, (layer_a, layer_b) = scorer.compute_pairwise_alignment(
    features_a,  # shape: (N, D) or (N, L, D) or list of (N, D)
    features_b   # shape: (N, D) or (N, L, D) or list of (N, D)
)

print(f"Alignment score: {score:.4f}")
print(f"Best alignment at layers: A={layer_a}, B={layer_b}")

# Method 2: Using metrics directly
score = AlignmentMetrics.mutual_knn(features_a, features_b, topk=10)
```

### With Reference Features

```python
# Set up alignment scorer with pre-computed reference features
scorer = Alignment(
    reference_features=[ref_model_1_feats, ref_model_2_feats],
    metric="mutual_knn",
    topk=10
)

# Score new features against all references
score, (ref_idx, feat_idx) = scorer.score(new_features)
```

### Alignment Matrix for Multiple Models

```python
# Compute pairwise alignment scores for multiple models
features_list = [model1_feats, model2_feats, model3_feats]

results = scorer.compute_alignment_matrix(
    features_list,
    features_list  # or different list for cross-modality
)

# Results contain:
# - scores: (N, M) matrix of alignment scores
# - indices: (N, M, 2) array of best layer indices
print(results['scores'])
```

## Supported Metrics

### k-NN Based Metrics

#### Mutual k-NN
Measures overlap in k-nearest neighbors between representations.

```python
score = AlignmentMetrics.mutual_knn(feats_a, feats_b, topk=10)
```

**Best for**: General purpose alignment, works well across modalities.

#### Cycle k-NN
Measures consistency when cycling through both representations.

```python
score = AlignmentMetrics.cycle_knn(feats_a, feats_b, topk=10)
```

**Best for**: Detecting one-way relationships between representations.

#### LCS k-NN
Uses longest common subsequence of k-NN orderings.

```python
score = AlignmentMetrics.lcs_knn(feats_a, feats_b, topk=10)
```

**Best for**: Order-sensitive alignment measurements.

#### Edit Distance k-NN
Computes edit distance between k-NN orderings.

```python
score = AlignmentMetrics.edit_distance_knn(feats_a, feats_b, topk=10)
```

**Best for**: Quantifying specific differences in neighborhood structure.

### Kernel-Based Metrics

#### CKA (Centered Kernel Alignment)
Classic kernel-based similarity using HSIC.

```python
score = AlignmentMetrics.cka(
    feats_a, feats_b,
    kernel_metric='ip',  # 'ip' or 'rbf'
    unbiased=False
)
```

**Best for**: General purpose, especially for high-dimensional features.

#### Unbiased CKA
Unbiased estimator of CKA using unbiased HSIC.

```python
score = AlignmentMetrics.unbiased_cka(feats_a, feats_b)
```

**Best for**: When sample size is limited or statistical correctness matters.

#### CKNNA (CKA with Nearest Neighbor Attention)
CKA variant that only considers nearest neighbors.

```python
score = AlignmentMetrics.cknna(
    feats_a, feats_b,
    topk=None,  # or specific k
    distance_agnostic=False,
    unbiased=True
)
```

**Best for**: Local structure alignment, combining benefits of k-NN and CKA.

### CCA-Based Metrics

#### SVCCA (Singular Vector CCA)
Combines SVD with canonical correlation analysis.

```python
score = AlignmentMetrics.svcca(feats_a, feats_b, cca_dim=10)
```

**Best for**: Dimensionality-reduced alignment, identifying principal correlations.

## Feature Extraction

### From Keras Models

```python
from dl_techniques.utils.alignment import extract_layer_features

# Extract features from specific layers
layer_features = extract_layer_features(
    model=your_model,
    inputs=data,
    layer_names=['layer1', 'layer2', 'layer3'],
    batch_size=32
)
# Returns: list of tensors, one per layer
```

### Feature Preprocessing

```python
from dl_techniques.utils.alignment import prepare_features, normalize_features

# Remove outliers and normalize
feats_clean = prepare_features(features, q=0.95, exact=True)

# L2 normalization
feats_norm = normalize_features(features, axis=-1)
```

### Saving and Loading

```python
from dl_techniques.utils.alignment import save_features, load_features

# Save features
save_features(
    features,
    save_path="./features/model_features.npz",
    metadata={'model': 'vit_large', 'dataset': 'imagenet'}
)

# Load features
loaded_features = load_features("./features/model_features.npz")
```

## Monitoring Training

Use `AlignmentLogger` as a callback to track representation alignment during training:

```python
from dl_techniques.utils.alignment import AlignmentLogger, Alignment

# Set up alignment scorer with reference features
reference_scorer = Alignment(
    reference_features=reference_model_features,
    metric="mutual_knn",
    topk=10
)

# Create logger
logger = AlignmentLogger(
    alignment_scorer=reference_scorer,
    validation_data=val_data,
    log_freq=5,  # Log every 5 epochs
    log_dir="./alignment_logs"
)

# Use during training (manual integration)
for epoch in range(num_epochs):
    # ... training code ...
    logger.on_epoch_end(epoch, model)

# Plot alignment over time
logger.plot_scores(save_path="alignment_curve.png")
```

## Advanced Usage

### Multi-Layer Alignment

```python
# For models with multiple layers, features can be:
# 1. Single tensor (N, L, D) where L is number of layers
# 2. List of tensors, each (N, D)

# The scorer automatically finds best layer alignment
score, (layer_a_idx, layer_b_idx) = scorer.compute_pairwise_alignment(
    multi_layer_features_a,
    multi_layer_features_b
)
```

### Cross-Modality Alignment

```python
# Measure vision-language alignment
vision_features = extract_layer_features(vision_model, images)
language_features = extract_layer_features(language_model, texts)

scorer = Alignment(metric="mutual_knn", topk=10)
alignment_score, layers = scorer.compute_pairwise_alignment(
    vision_features,
    language_features
)
```

### Batch Processing

```python
from dl_techniques.utils.alignment import batch_generator

# Process large datasets in batches
for batch in batch_generator(large_dataset, batch_size=1000):
    batch_features = model(batch, training=False)
    # ... process batch ...
```

### Custom Metric Parameters

```python
# CKA with RBF kernel
score = AlignmentMetrics.cka(
    feats_a, feats_b,
    kernel_metric='rbf',
    rbf_sigma=2.0,
    unbiased=True
)

# CKNNA with specific parameters
score = AlignmentMetrics.cknna(
    feats_a, feats_b,
    topk=50,
    distance_agnostic=True,
    unbiased=False
)
```

## Best Practices

### Feature Normalization
Always L2-normalize features before computing alignment:
```python
scorer = Alignment(metric="mutual_knn", normalize=True)
```

### Outlier Removal
Remove outliers to improve robustness:
```python
from dl_techniques.utils.alignment import prepare_features
clean_features = prepare_features(features, q=0.95, exact=True)
```

### Choosing Metrics

- **Fast, general purpose**: `mutual_knn` with `topk=10`
- **Statistical rigor**: `unbiased_cka`
- **Local structure**: `cknna`
- **Cross-modality**: `mutual_knn` or `cka`
- **Dimensionality reduction**: `svcca`

### Memory Management

For large-scale experiments:
```python
# Process in batches
from dl_techniques.utils.alignment import batch_generator

all_scores = []
for batch_a, batch_b in zip(
    batch_generator(features_a, 100),
    batch_generator(features_b, 100)
):
    score = scorer.compute_pairwise_alignment(batch_a, batch_b)
    all_scores.append(score)
```

## Examples

### Example 1: Vision Model Alignment

```python
import keras
from dl_techniques.utils.alignment import Alignment, extract_layer_features

# Load models
vit_model = keras.applications.vit.ViT_B16(weights='imagenet')
resnet_model = keras.applications.ResNet50(weights='imagenet')

# Extract features
images = ...  # Your image data
vit_features = extract_layer_features(vit_model, images)
resnet_features = extract_layer_features(resnet_model, images)

# Compute alignment
scorer = Alignment(metric="mutual_knn", topk=10)
score, (vit_layer, resnet_layer) = scorer.compute_pairwise_alignment(
    vit_features, resnet_features
)

print(f"ViT-ResNet alignment: {score:.4f}")
print(f"Best layers: ViT-{vit_layer}, ResNet-{resnet_layer}")
```

### Example 2: Training Dynamics

```python
from dl_techniques.utils.alignment import Alignment, AlignmentLogger

# Initialize with pretrained reference
reference_model = keras.applications.EfficientNetB0(weights='imagenet')
ref_features = extract_layer_features(reference_model, val_images)

scorer = Alignment(reference_features=[ref_features], metric="cka")
logger = AlignmentLogger(scorer, val_images, log_freq=1)

# Train your model
model = build_your_model()

for epoch in range(50):
    model.fit(train_data, epochs=1)
    logger.on_epoch_end(epoch, model)

# Visualize alignment evolution
logger.plot_scores("training_alignment.png")
```

### Example 3: Model Selection

```python
# Compare multiple candidate models to a reference
candidates = [model1, model2, model3, model4]
candidate_names = ['Model_A', 'Model_B', 'Model_C', 'Model_D']

scorer = Alignment(
    reference_features=[reference_features],
    metric="mutual_knn",
    topk=10
)

scores = []
for name, model in zip(candidate_names, candidates):
    feats = extract_layer_features(model, test_data)
    score, _ = scorer.score(feats)
    scores.append(score)
    print(f"{name}: {score:.4f}")

best_model = candidate_names[np.argmax(scores)]
print(f"\nBest model: {best_model}")
```

## API Reference

### AlignmentMetrics

Static class with all metric implementations.

**Methods:**
- `measure(metric, *args, **kwargs)`: Unified interface for all metrics
- `mutual_knn(feats_a, feats_b, topk)`: Mutual k-NN metric
- `cycle_knn(feats_a, feats_b, topk)`: Cycle k-NN metric
- `lcs_knn(feats_a, feats_b, topk)`: LCS k-NN metric
- `cka(feats_a, feats_b, kernel_metric, rbf_sigma, unbiased)`: CKA metric
- `unbiased_cka(*args, **kwargs)`: Unbiased CKA
- `cknna(feats_a, feats_b, topk, distance_agnostic, unbiased)`: CKNNA metric
- `svcca(feats_a, feats_b, cca_dim)`: SVCCA metric
- `edit_distance_knn(feats_a, feats_b, topk)`: Edit distance metric

### Alignment

High-level API for computing alignment.

**Constructor:**
```python
Alignment(
    reference_features=None,
    metric="mutual_knn",
    topk=10,
    normalize=True,
    device="auto",
    dtype="float32"
)
```

**Methods:**
- `score(features, return_layer_indices=True, **kwargs)`: Score against reference
- `compute_pairwise_alignment(features_a, features_b, ...)`: Pairwise alignment
- `compute_alignment_matrix(features_list_a, features_list_b, ...)`: Matrix of scores
- `set_reference_features(reference_features)`: Update reference features
- `get_supported_metrics()`: List supported metrics
- `from_models(reference_models, data, ...)`: Create from Keras models (class method)

### AlignmentLogger

Training callback for monitoring alignment.

**Constructor:**
```python
AlignmentLogger(
    alignment_scorer,
    validation_data,
    log_freq=1,
    log_dir=None
)
```

**Methods:**
- `on_epoch_end(epoch, model, logs=None)`: Compute and log alignment
- `get_scores()`: Retrieve all logged scores
- `plot_scores(save_path=None)`: Visualize alignment over training

## Troubleshooting

### Memory Issues
- Use batch processing for large datasets
- Reduce number of samples
- Use `exact=False` in `prepare_features()`

### Numerical Instability
- Ensure features are normalized: `normalize=True`
- Remove outliers: `prepare_features(feats, q=0.95)`
- Use `unbiased=True` for CKA-based metrics

### Slow Computation
- Use k-NN metrics instead of CKA for speed
- Reduce `topk` parameter
- Use fewer layers/samples
- Set `exact=False` in preprocessing

## References

1. Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. In International Conference on Machine Learning (ICML).

2. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. In International Conference on Machine Learning (ICML).

3. Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. In Advances in Neural Information Processing Systems (NeurIPS).
