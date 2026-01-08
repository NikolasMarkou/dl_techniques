# Integration Guide: Platonic Alignment Module for dl_techniques

## Overview

This guide explains how to integrate the Platonic Representation Alignment module into the dl_techniques project structure.

## File Structure

The module should be placed in: `src/dl_techniques/utils/alignment/`

```
src/dl_techniques/utils/alignment/
├── __init__.py          # Module exports
├── metrics.py           # Alignment metrics implementations
├── alignment.py         # High-level Alignment API
├── utils.py             # Helper utilities
├── README.md            # User documentation
└── test_alignment.py    # Tests and examples
```

## Integration Steps

### Step 1: Copy Files

Copy the `platonic_alignment/` directory contents to:
```bash
cp -r platonic_alignment/* src/dl_techniques/utils/alignment/
```

### Step 2: Update dl_techniques/utils/__init__.py

Add to `src/dl_techniques/utils/__init__.py`:

```python
# Alignment module for representation similarity
from dl_techniques.utils.alignment import (
    AlignmentMetrics,
    Alignment,
    AlignmentLogger,
    prepare_features,
    compute_score,
    normalize_features,
)
```

### Step 3: Dependencies

Ensure these are in your project requirements:
- keras >= 3.8.0
- tensorflow >= 2.18.0
- numpy
- scikit-learn
- tensorflow-probability (optional, for better quantile estimation)

### Step 4: Testing

Run the tests to verify integration:
```bash
python src/dl_techniques/utils/alignment/test_alignment.py
```

## Usage After Integration

### Import

```python
# From anywhere in the project
from dl_techniques.utils.alignment import Alignment, AlignmentMetrics

# Or import specific components
from dl_techniques.utils.alignment import (
    Alignment,
    AlignmentMetrics,
    AlignmentLogger,
    extract_layer_features,
    prepare_features,
    normalize_features
)
```

### Basic Example

```python
import keras
from dl_techniques.utils.alignment import Alignment

# Create alignment scorer
scorer = Alignment(metric="mutual_knn", topk=10, normalize=True)

# Compute alignment between two models
model1 = keras.applications.ResNet50(weights='imagenet')
model2 = keras.applications.VGG16(weights='imagenet')

# Extract features
from dl_techniques.utils.alignment import extract_layer_features
images = ...  # your image data

feats1 = extract_layer_features(model1, images)
feats2 = extract_layer_features(model2, images)

# Compute alignment
score, (layer1, layer2) = scorer.compute_pairwise_alignment(feats1, feats2)
print(f"ResNet-VGG alignment: {score:.4f} (layers: {layer1}, {layer2})")
```

### Training Callback Integration

For use with Keras training:

```python
from dl_techniques.utils.alignment import Alignment, AlignmentLogger

# Set up reference model
reference_model = keras.applications.ResNet50(weights='imagenet')
ref_features = extract_layer_features(reference_model, val_images)

# Create alignment logger
scorer = Alignment(reference_features=[ref_features], metric="cka")
logger = AlignmentLogger(
    alignment_scorer=scorer,
    validation_data=val_images,
    log_freq=5
)

# Training loop
model = build_your_model()
for epoch in range(num_epochs):
    # Train
    history = model.fit(train_data, epochs=1)
    
    # Log alignment
    logger.on_epoch_end(epoch, model)
```

## Integration with Existing dl_techniques Components

### With Model Analyzer

```python
from dl_techniques.analyzer import ModelAnalyzer
from dl_techniques.utils.alignment import Alignment

# Analyze model and compute alignment
analyzer = ModelAnalyzer(model)
analysis = analyzer.analyze(data)

# Extract features from intermediate layers
layer_features = extract_layer_features(
    model,
    data,
    layer_names=analyzer.get_layer_names()
)

# Compare with reference
scorer = Alignment(reference_features=reference_feats, metric="mutual_knn")
alignment_score, _ = scorer.score(layer_features)
```

### With Training Utilities

```python
from dl_techniques.utils.train import train_model
from dl_techniques.utils.alignment import AlignmentLogger

# Add alignment tracking to training
def train_with_alignment(model, train_data, val_data, reference_model):
    # Set up alignment logger
    ref_features = extract_layer_features(reference_model, val_data)
    scorer = Alignment(reference_features=[ref_features], metric="cka")
    logger = AlignmentLogger(scorer, val_data, log_freq=1)
    
    # Training with callbacks
    history = train_model(
        model,
        train_data,
        validation_data=val_data,
        epochs=50,
        # Note: Need to adapt logger for Keras callback interface
    )
    
    return history, logger.get_scores()
```

### With Visualization Manager

```python
from dl_techniques.utils.visualization_manager import VisualizationManager
from dl_techniques.utils.alignment import AlignmentLogger

# Integrate alignment plots with visualization manager
viz_manager = VisualizationManager(output_dir="./visualizations")

# After training with alignment logging
logger.plot_scores(
    save_path=viz_manager.get_path("alignment_training.png")
)
```

## Module Design Philosophy

### Consistency with dl_techniques

1. **Keras 3 Native**: Uses `keras.ops` and `keras.layers` throughout
2. **Type Hints**: Full type annotations for all functions
3. **Sphinx Docstrings**: Google-style docstrings for documentation
4. **Best Practices**: Follows dl_techniques coding standards

### Key Features

1. **Flexible Input**: Accepts tensors, numpy arrays, or lists
2. **Multi-Layer Support**: Automatically finds best layer alignment
3. **Batch Processing**: Memory-efficient for large datasets
4. **Metric Agnostic**: Easy to add new metrics

### Error Handling

The module includes comprehensive error handling:
- Input validation
- Shape checking
- Metric parameter validation
- Graceful degradation (e.g., without tensorflow_probability)

## Advanced Integration Examples

### Custom Metric Integration

Add custom metrics by extending `AlignmentMetrics`:

```python
# In your project code
from dl_techniques.utils.alignment import AlignmentMetrics

class CustomAlignmentMetrics(AlignmentMetrics):
    SUPPORTED_METRICS = AlignmentMetrics.SUPPORTED_METRICS + ['my_metric']
    
    @staticmethod
    def my_metric(feats_a, feats_b, **kwargs):
        # Your metric implementation
        return score
```

### Multi-Modal Alignment

```python
from dl_techniques.models.mobile_clip import MobileCLIPModel
from dl_techniques.utils.alignment import Alignment

# Extract vision and language features
clip_model = MobileCLIPModel(...)

image_features = clip_model.encode_images(images)
text_features = clip_model.encode_texts(texts)

# Measure vision-language alignment
scorer = Alignment(metric="mutual_knn", topk=10)
vl_alignment, _ = scorer.compute_pairwise_alignment(
    image_features,
    text_features
)
print(f"Vision-Language alignment: {vl_alignment:.4f}")
```

### Model Comparison Framework

```python
from dl_techniques.utils.alignment import Alignment

def compare_models(models, data, reference_model, metric="mutual_knn"):
    """Compare multiple models against a reference."""
    # Extract reference features
    ref_features = extract_layer_features(reference_model, data)
    
    # Create scorer
    scorer = Alignment(
        reference_features=[ref_features],
        metric=metric,
        topk=10
    )
    
    # Score each model
    results = {}
    for name, model in models.items():
        features = extract_layer_features(model, data)
        score, layers = scorer.score(features)
        results[name] = {
            'score': score,
            'layers': layers
        }
    
    return results

# Usage
models = {
    'resnet50': keras.applications.ResNet50(weights='imagenet'),
    'vgg16': keras.applications.VGG16(weights='imagenet'),
    'efficientnet': keras.applications.EfficientNetB0(weights='imagenet')
}

reference = keras.applications.InceptionV3(weights='imagenet')
comparison = compare_models(models, test_images, reference)
```

## Testing

### Unit Tests

Basic tests are included in `test_alignment.py`. For full integration testing:

```python
# tests/test_utils_alignment.py
import pytest
import numpy as np
from dl_techniques.utils.alignment import Alignment, AlignmentMetrics

class TestAlignment:
    def test_mutual_knn(self):
        feats_a = np.random.randn(50, 128).astype('float32')
        feats_b = np.random.randn(50, 128).astype('float32')
        score = AlignmentMetrics.mutual_knn(feats_a, feats_b, topk=5)
        assert 0 <= score <= 1
    
    def test_alignment_api(self):
        scorer = Alignment(metric="cka")
        feats_a = np.random.randn(50, 128).astype('float32')
        feats_b = np.random.randn(50, 128).astype('float32')
        score, _ = scorer.compute_pairwise_alignment(feats_a, feats_b)
        assert isinstance(score, float)
```

### Integration Tests

```bash
# Run from project root
pytest tests/test_utils_alignment.py -v
```

## Performance Considerations

### Memory Usage

For large-scale experiments:
- Use batch processing for feature extraction
- Clear GPU memory between computations
- Consider downsampling for initial experiments

### Computation Speed

Metric comparison (approximate, 1000 samples, 512-dim):
- `mutual_knn`: ~0.1s (fastest)
- `cka`: ~0.5s
- `unbiased_cka`: ~0.6s
- `svcca`: ~1.0s (slowest)

### Recommendations

- For real-time monitoring: Use `mutual_knn` with `topk=5`
- For publication results: Use `unbiased_cka` or `cknna`
- For exploration: Start with `mutual_knn`, validate with `cka`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure module is in correct location
2. **Memory Issues**: Reduce batch size or number of samples
3. **Numerical Issues**: Always normalize features before alignment
4. **Slow Performance**: Use approximate quantile (`exact=False`)

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential additions:
1. GPU-accelerated k-NN computation
2. Distributed computing for large-scale alignment matrices
3. Additional metrics (e.g., Procrustes alignment)
4. Integration with attention visualization tools
5. Automatic layer selection heuristics

## Support

For issues specific to this module:
1. Check the README.md for usage examples
2. Run test_alignment.py to verify installation
3. Review the docstrings for API details

For integration with dl_techniques:
1. Refer to existing module patterns in the codebase
2. Follow the project's contribution guidelines
3. Ensure tests pass before committing

## Conclusion

The Platonic Alignment module provides a comprehensive toolkit for measuring representation similarity in neural networks. It integrates seamlessly with dl_techniques while maintaining flexibility for various use cases.

Key takeaways:
- Place in `src/dl_techniques/utils/alignment/`
- Import as `from dl_techniques.utils.alignment import ...`
- Follow existing dl_techniques patterns
- Run tests to verify integration
- Refer to README.md for detailed usage examples
