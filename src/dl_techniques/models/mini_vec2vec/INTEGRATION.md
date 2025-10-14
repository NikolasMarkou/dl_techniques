# Mini-Vec2Vec Framework Integration Guide

This document outlines how the Mini-Vec2Vec model integrates into the dl_techniques framework following modern Keras 3 best practices.

## File Structure

```
src/dl_techniques/models/mini_vec2vec/
├── __init__.py                 # Package exports
├── model.py                    # Main model implementation
└── README.md                   # Documentation

examples/mini_vec2vec/
└── example_alignment.py        # Usage example

tests/models/
└── test_mini_vec2vec.py        # Unit tests
```

## Framework Compliance

### 1. Modern Keras 3 Patterns

The implementation follows the "Golden Rule" from the framework guide:

#### ✓ `__init__`: CREATE sublayers
```python
def __init__(self, embedding_dim: int, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.W = None  # Will be created in build()
```

**Rationale**: Configuration is stored, but weight creation is deferred to `build()`.

#### ✓ `build`: CREATE weights
```python
def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
    self.W = self.add_weight(
        name="transformation_matrix_W",
        shape=(self.embedding_dim, self.embedding_dim),
        initializer=initializers.Identity(),
        trainable=True,
    )
    super().build(input_shape)
```

**Rationale**: Weights are created only when input shape is known.

#### ✓ `call`: FORWARD PASS
```python
def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
    return ops.matmul(inputs, self.W)
```

**Rationale**: Simple, stateless forward pass using Keras ops.

#### ✓ `get_config`: COMPLETE configuration
```python
def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({"embedding_dim": self.embedding_dim})
    return config
```

**Rationale**: All constructor parameters are included for proper serialization.

### 2. Registration and Serialization

```python
@keras.saving.register_keras_serializable(package="dl_techniques.models.mini_vec2vec")
class MiniVec2VecAligner(keras.Model):
    ...
```

**Benefits**:
- Automatic discovery during `keras.models.load_model()`
- Proper namespace management
- No custom objects dict required

### 3. Type Safety and Documentation

#### Type Hints
```python
def align(
    self,
    XA: np.ndarray,
    XB: np.ndarray,
    approx_clusters: int = 20,
    # ... more params
) -> Dict[str, Any]:
```

#### Sphinx-Style Docstrings
```python
"""
Execute the full mini-vec2vec alignment pipeline (Algorithm 1).

Args:
    XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
    XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
    approx_clusters: Number of clusters for anchor alignment.
    
Returns:
    Dictionary containing transformation history.
    
Raises:
    ValueError: If input shapes are incompatible.
"""
```

### 4. Keras Ops Usage

The implementation uses Keras ops for all tensor operations:

```python
# ✓ Correct: Keras ops
return ops.matmul(inputs, self.W)
aligned = ops.convert_to_numpy(aligner(embeddings))

# ✗ Incorrect: Direct backend usage
# return tf.matmul(inputs, self.W)  # Don't do this
```

**Exception**: TensorFlow's `GradientTape` is used for gradient computation as per framework guidelines:

```python
import tensorflow as tf

with tf.GradientTape() as tape:
    output = model(inputs)
    loss = compute_loss(output)
```

## Key Design Decisions

### 1. Custom `align()` Method vs `fit()`

**Decision**: Implement custom `align()` method instead of using Keras `fit()`.

**Rationale**:
- The alignment procedure is not gradient-based
- It follows a specific algorithmic sequence (Algorithms 1-4 from paper)
- Using `fit()` would require artificial loss functions and training loops
- `align()` provides clearer API: `aligner.align(XA, XB, ...)`

### 2. NumPy for Clustering Operations

**Decision**: Use NumPy/scikit-learn for clustering and matching.

**Rationale**:
- K-means and QAP have well-optimized scikit-learn/scipy implementations
- These operations are not part of the forward pass
- No need for gradient computation through clustering
- Maintains compatibility with established libraries

### 3. Internal Preprocessing

**Decision**: Preprocessing (centering, normalization) is done inside `align()`.

**Rationale**:
- Ensures consistent preprocessing
- Reduces user error
- Aligns with the algorithm's requirements
- User-provided data can be in any reasonable format

### 4. Transformation Matrix as Model Weight

**Decision**: W is a trainable Keras weight, not a simple variable.

**Rationale**:
- Enables proper serialization
- Allows potential fine-tuning with gradient-based methods
- Integrates with Keras weight management
- Supports model inspection tools

## Integration with Framework Components

### Logging
```python
from dl_techniques.utils.logger import logger

logger.info("Starting alignment...")
logger.info(f"Final accuracy: {accuracy:.4f}")
```

### Factory Pattern
```python
def create_mini_vec2vec_aligner(
    embedding_dim: int,
    **kwargs: Any
) -> MiniVec2VecAligner:
    """Factory function following framework convention."""
    return MiniVec2VecAligner(embedding_dim=embedding_dim, **kwargs)
```

### Testing Pattern
```python
class TestMiniVec2VecAligner:
    """Test suite following framework conventions."""
    
    @pytest.fixture
    def embedding_dim(self):
        return 32
    
    def test_serialization_cycle(self, ...):
        """Critical test for Keras compatibility."""
        # Save
        aligner.save('model.keras')
        # Load
        loaded = keras.models.load_model('model.keras')
        # Verify
        np.testing.assert_allclose(
            ops.convert_to_numpy(original),
            ops.convert_to_numpy(loaded),
            rtol=1e-6, atol=1e-6
        )
```

## Usage Patterns

### Basic Usage
```python
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

# Create
aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))

# Align
history = aligner.align(XA=source, XB=target)

# Transform
aligned = aligner(new_source)

# Save
aligner.save('aligner.keras')
```

### With Framework Utilities
```python
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner
from dl_techniques.utils.logger import logger

# Initialize with logging
logger.info("Creating aligner...")
aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))

# Align with progress logging
history = aligner.align(XA=source, XB=target)

# Evaluate with framework metrics (if applicable)
# This model doesn't use standard metrics during alignment,
# but you can use them for evaluation
```

### Advanced: With Model Callbacks (Future Enhancement)
```python
# Potential future enhancement: callback support during alignment
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

class AlignmentProgressCallback:
    def on_stage_end(self, stage, metrics):
        logger.info(f"Stage {stage} complete: {metrics}")

aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))

# With callback
history = aligner.align(
    XA=source, 
    XB=target,
    callbacks=[AlignmentProgressCallback()]  # Not yet implemented
)
```

## Testing Strategy

### Unit Tests
- Initialization validation
- Build process
- Forward pass
- Individual method tests (_procrustes, _create_pseudo_pairs, etc.)
- Configuration serialization
- Complete serialization cycle

### Integration Tests
- Full alignment pipeline
- Accuracy improvement verification
- Different hyperparameter combinations
- Edge cases (small datasets, extreme parameters)

### Performance Tests (Recommended)
```python
@pytest.mark.slow
def test_alignment_performance(large_dataset):
    """Test alignment on large dataset."""
    import time
    
    aligner = MiniVec2VecAligner(embedding_dim=256)
    aligner.build(input_shape=(None, 256))
    
    start = time.time()
    history = aligner.align(
        XA=large_dataset['A'],
        XB=large_dataset['B']
    )
    duration = time.time() - start
    
    assert duration < 300  # Should complete in 5 minutes
```

## Common Integration Issues and Solutions

### Issue 1: Serialization Errors
**Symptom**: `ValueError: Layer was never built`

**Solution**: Ensure `build()` is called before `save()`:
```python
aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))  # Required!
aligner.save('model.keras')
```

### Issue 2: Memory Issues with Large Datasets
**Symptom**: `MemoryError` during approximate matching

**Solution**: Reduce `approx_runs` or `approx_neighbors`:
```python
history = aligner.align(
    XA=large_XA,
    XB=large_XB,
    approx_runs=10,        # Reduced from 30
    approx_neighbors=10,   # Reduced from 50
)
```

### Issue 3: Slow Alignment
**Symptom**: Alignment takes too long

**Solution**: Reduce iteration counts:
```python
history = aligner.align(
    XA=XA,
    XB=XB,
    refine1_iterations=25,  # Reduced from 75
    refine2_clusters=200,   # Reduced from 500
)
```

## Maintenance and Extension

### Adding New Refinement Strategies
```python
def _refine_custom(
    self,
    XA: np.ndarray,
    XB: np.ndarray,
    **kwargs
) -> None:
    """
    Custom refinement strategy.
    
    Args:
        XA: Source embeddings.
        XB: Target embeddings.
        **kwargs: Strategy-specific parameters.
    """
    current_W = ops.convert_to_numpy(self.W)
    
    # Implement custom logic
    # ...
    
    # Update W
    self.W.assign(new_W)
```

### Adding Callbacks (Future)
```python
class AlignmentCallback:
    """Base callback for alignment process."""
    
    def on_stage_begin(self, stage: str) -> None:
        pass
    
    def on_stage_end(self, stage: str, metrics: Dict[str, Any]) -> None:
        pass
```

### Adding Metrics
```python
def compute_alignment_metric(
    aligner: MiniVec2VecAligner,
    XA: np.ndarray,
    XB: np.ndarray
) -> float:
    """Compute custom alignment quality metric."""
    aligned = ops.convert_to_numpy(aligner(XA))
    # Compute metric
    return metric_value
```

## Documentation Standards

All public methods must include:
1. **One-line summary**
2. **Detailed description** (if needed)
3. **Args section** with types and descriptions
4. **Returns section** with type and description
5. **Raises section** for exceptions
6. **Example section** for complex methods

See the model.py file for examples.

## Conclusion

The Mini-Vec2Vec implementation fully integrates with the dl_techniques framework by:

1. ✓ Following modern Keras 3 patterns
2. ✓ Using proper serialization
3. ✓ Including comprehensive type hints
4. ✓ Providing Sphinx-style documentation
5. ✓ Using framework utilities (logger)
6. ✓ Including complete test coverage
7. ✓ Following project structure conventions

The model is production-ready and can be used immediately as part of the framework.