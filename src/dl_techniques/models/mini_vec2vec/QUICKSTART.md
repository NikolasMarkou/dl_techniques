# Mini-Vec2Vec Quick Start Integration

This guide provides step-by-step instructions for integrating the Mini-Vec2Vec model into your dl_techniques repository.

## Complete File Structure

After integration, your repository should have these new files:

```
dl_techniques/
├── src/dl_techniques/models/
│   └── mini_vec2vec/
│       ├── __init__.py           # Package exports
│       ├── model.py              # MiniVec2VecAligner implementation
│       └── README.md             # Full documentation
│
├── examples/
│   └── mini_vec2vec/
│       └── example_alignment.py  # Complete working example
│
└── tests/models/
    └── test_mini_vec2vec.py      # Comprehensive unit tests
```

## Step-by-Step Integration

### Step 1: Create Directory Structure

```bash
# Navigate to your dl_techniques repository
cd /path/to/dl_techniques

# Create model directory
mkdir -p src/dl_techniques/models/mini_vec2vec

# Create examples directory
mkdir -p examples/mini_vec2vec

# tests/models should already exist
# If not: mkdir -p tests/models
```

### Step 2: Add Model Files

Copy the provided files to their respective locations:

1. **Model Implementation**: `model.py` → `src/dl_techniques/models/mini_vec2vec/model.py`
2. **Package Init**: `__init__.py` → `src/dl_techniques/models/mini_vec2vec/__init__.py`
3. **Documentation**: `README.md` → `src/dl_techniques/models/mini_vec2vec/README.md`
4. **Example**: `example_alignment.py` → `examples/mini_vec2vec/example_alignment.py`
5. **Tests**: `test_mini_vec2vec.py` → `tests/models/test_mini_vec2vec.py`

### Step 3: Update Main Model Init (Optional)

Add to `src/dl_techniques/models/__init__.py`:

```python
# Mini-Vec2Vec
from dl_techniques.models.mini_vec2vec import (
    MiniVec2VecAligner,
    create_mini_vec2vec_aligner,
)
```

### Step 4: Install Dependencies

The model requires these additional dependencies (most should already be installed):

```bash
pip install scikit-learn scipy tqdm
```

Or add to your `requirements.txt`:
```
scikit-learn>=1.0.0
scipy>=1.7.0
tqdm>=4.60.0
```

### Step 5: Verify Installation

Run the tests to verify everything works:

```bash
# Run specific model tests
pytest tests/models/test_mini_vec2vec.py -v

# Or run all tests
pytest tests/ -v
```

Expected output:
```
tests/models/test_mini_vec2vec.py::TestMiniVec2VecAligner::test_initialization PASSED
tests/models/test_mini_vec2vec.py::TestMiniVec2VecAligner::test_build PASSED
tests/models/test_mini_vec2vec.py::TestMiniVec2VecAligner::test_serialization_cycle PASSED
...
==================== X passed in Y.YYs ====================
```

### Step 6: Run Example

Test the example script:

```bash
python examples/mini_vec2vec/example_alignment.py
```

Expected output:
```
======================================================================
Mini-Vec2Vec Alignment Example
======================================================================
Generating synthetic data...
Data generated: Alignment set size: 25000, Eval set size: 5000

--- Initializing MiniVec2VecAligner ---
Model created with embedding_dim=128

--- Evaluation Results (BEFORE alignment) ---
top1_accuracy: 0.0120
mean_cosine_sim: 0.0015

======================================================================
Starting Alignment Process
======================================================================
Step 1: Preprocessing embeddings...
...
✓ Alignment finished successfully!

--- Evaluation Results (AFTER alignment) ---
top1_accuracy: 0.9840
mean_cosine_sim: 0.9820

--- Testing Model Serialization ---
Model saved to temp_models/mini_vec2vec_aligner.keras
Model loaded successfully
✓ Serialization test PASSED: Predictions match

======================================================================
Alignment Complete - Summary
======================================================================
Final Top-1 Accuracy: 0.9840
Final Mean Cosine Similarity: 0.9820
======================================================================
```

## Quick Usage Examples

### Example 1: Basic Alignment

```python
import numpy as np
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

# Your data (should be normalized embeddings)
source_embeddings = np.load('source_embeddings.npy')  # shape: (N, D)
target_embeddings = np.load('target_embeddings.npy')  # shape: (M, D)

# Create and align
aligner = MiniVec2VecAligner(embedding_dim=128)
aligner.build(input_shape=(None, 128))

history = aligner.align(
    XA=source_embeddings,
    XB=target_embeddings
)

# Transform new embeddings
new_source = np.load('new_source.npy')
aligned = aligner(new_source)

# Save for later use
aligner.save('my_aligner.keras')
```

### Example 2: Cross-Lingual Word Embeddings

```python
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

# Load word embeddings
en_embeddings = load_english_embeddings()  # (50000, 300)
fr_embeddings = load_french_embeddings()   # (50000, 300)

# Align
aligner = MiniVec2VecAligner(embedding_dim=300)
aligner.build(input_shape=(None, 300))

history = aligner.align(
    XA=en_embeddings,
    XB=fr_embeddings,
    approx_clusters=30,
    approx_runs=40,
    refine1_iterations=100
)

# Translate words by finding nearest neighbors
def translate_word(word_embedding):
    aligned = aligner(word_embedding.reshape(1, -1))
    # Find nearest French word...
    return nearest_french_word(aligned)
```

### Example 3: With Custom Hyperparameters

```python
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner

aligner = MiniVec2VecAligner(embedding_dim=512)
aligner.build(input_shape=(None, 512))

# Fine-tune for quality
history = aligner.align(
    XA=source,
    XB=target,
    # More clusters for finer matching
    approx_clusters=40,
    approx_runs=50,
    approx_neighbors=20,
    # More iterations for refinement
    refine1_iterations=100,
    refine1_sample_size=10000,
    refine1_neighbors=20,
    # Many clusters for fine adjustments
    refine2_clusters=1000,
    # Balanced smoothing
    smoothing_alpha=0.5
)
```

## Troubleshooting

### Issue: Import Error

**Symptom**:
```python
ImportError: cannot import name 'MiniVec2VecAligner'
```

**Solution**:
Ensure you've installed the package:
```bash
cd /path/to/dl_techniques
pip install -e .
```

### Issue: Missing Dependencies

**Symptom**:
```python
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**:
Install missing dependencies:
```bash
pip install scikit-learn scipy tqdm
```

### Issue: Tests Fail

**Symptom**:
```
tests/models/test_mini_vec2vec.py FAILED
```

**Solution**:
1. Check Python version (requires 3.8+)
2. Update dependencies: `pip install -U keras tensorflow numpy`
3. Clear cache: `pytest --cache-clear`
4. Check backend: `export KERAS_BACKEND=tensorflow`

### Issue: Slow Performance

**Symptom**: Alignment takes too long

**Solution**: Reduce hyperparameters:
```python
history = aligner.align(
    XA=source,
    XB=target,
    approx_runs=10,         # Reduce from 30
    refine1_iterations=25,  # Reduce from 75
    refine2_clusters=200    # Reduce from 500
)
```

## Verification Checklist

Before committing, verify:

- [ ] All files are in correct locations
- [ ] Tests pass: `pytest tests/models/test_mini_vec2vec.py -v`
- [ ] Example runs: `python examples/mini_vec2vec/example_alignment.py`
- [ ] Import works: `from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner`
- [ ] Documentation is complete
- [ ] Type hints are present
- [ ] Docstrings follow Sphinx format
- [ ] Code follows framework style guide

## Next Steps

1. **Add to Documentation**: Update main framework docs to include Mini-Vec2Vec
2. **Create Notebook**: Add Jupyter notebook tutorial
3. **Performance Benchmarks**: Add timing benchmarks for different sizes
4. **Add Visualizations**: Integrate with framework's visualization module
5. **Real-World Examples**: Add examples with actual word embeddings

## Support and Contributing

### Getting Help

- Check `README.md` for detailed documentation
- Review `example_alignment.py` for usage patterns
- Examine tests for edge cases and patterns
- Open an issue on GitHub

### Contributing

To extend or improve the model:

1. **Add Features**: Follow existing patterns
2. **Write Tests**: Add to `test_mini_vec2vec.py`
3. **Update Docs**: Keep README.md current
4. **Follow Style**: Match framework conventions

### Code Style Guidelines

- Type hints for all function arguments and returns
- Sphinx docstrings for all public methods
- Use `keras.ops` for tensor operations
- Use `tf.GradientTape` for gradients (per framework)
- Follow PEP 8 naming conventions
- Add comprehensive tests

## Complete Example Command Sequence

Here's the complete sequence to integrate and verify:

```bash
# 1. Navigate to repository
cd /path/to/dl_techniques

# 2. Create directories
mkdir -p src/dl_techniques/models/mini_vec2vec
mkdir -p examples/mini_vec2vec

# 3. Copy files (assumes files are in current directory)
cp model.py src/dl_techniques/models/mini_vec2vec/
cp __init__.py src/dl_techniques/models/mini_vec2vec/
cp README.md src/dl_techniques/models/mini_vec2vec/
cp example_alignment.py examples/mini_vec2vec/
cp test_mini_vec2vec.py tests/models/

# 4. Install dependencies
pip install -e .
pip install scikit-learn scipy tqdm

# 5. Run tests
pytest tests/models/test_mini_vec2vec.py -v

# 6. Run example
python examples/mini_vec2vec/example_alignment.py

# 7. Verify import
python -c "from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner; print('✓ Import successful')"
```

## Success Criteria

Integration is successful when:

1. ✓ All tests pass
2. ✓ Example runs without errors
3. ✓ Model can be imported
4. ✓ Model can be saved and loaded
5. ✓ Alignment achieves >95% accuracy on synthetic data
6. ✓ Documentation is complete and accurate

Congratulations! The Mini-Vec2Vec model is now fully integrated into your dl_techniques framework.