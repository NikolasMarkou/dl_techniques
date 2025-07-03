# Multi-Task MDN Time Series Forecasting

This implementation demonstrates a novel approach to time series forecasting using a single Mixture Density Network (MDN) trained simultaneously on multiple tasks. Instead of training separate models for different time series patterns, this system learns shared representations while maintaining task-specific expertise through task embeddings.

## 🚀 Key Features

- **Single Model, Multiple Tasks**: One MDN model handles sine waves, noisy data, and stock prices simultaneously
- **Task Embeddings**: Learnable embeddings help the model distinguish between different data patterns
- **Comprehensive Uncertainty Quantification**: Separates aleatoric and epistemic uncertainty for each task
- **Robust Evaluation**: Individual task evaluation with detailed metrics and visualizations
- **Advanced Analysis Tools**: Complete analysis suite for understanding multi-task performance

## 📁 Project Structure

```
experiments/mdn/
├── multitask_mdn_training.py      # Main training script
├── test_multitask_mdn.py          # Comprehensive test suite
├── multitask_mdn_analyzer.py      # Results analysis tools
└── README.md                      # This file
```

## 🏗️ Architecture Overview

### Multi-Task MDN Model

```
Input: [Time Series Window, Task ID]
    ↓
Task Embedding Layer (8-dim embedding)
    ↓
Concatenation: [Flattened Window + Task Embedding]
    ↓
Feature Extraction Network:
    - Dense layers with configurable sizes
    - Batch normalization (optional)
    - Dropout for regularization
    - Activation functions
    ↓
MDN Layer:
    - Mixture parameters (π, μ, σ)
    - Handles uncertainty quantification
    ↓
Output: Probabilistic predictions
```

### Task-Specific Processing

1. **Sine Wave**: Clean periodic patterns
2. **Noisy Sine Wave**: Periodic patterns with heteroscedastic noise
3. **Stock Price (GBM)**: Financial time series with drift and volatility

## 🚀 Quick Start

### 1. Basic Training

```bash
cd experiments/mdn/
python multitask_mdn_training.py
```

This will:
- Generate synthetic time series data for all tasks
- Train a single MDN model on all tasks simultaneously
- Evaluate performance on each task individually
- Save results and visualizations

### 2. Custom Configuration

```python
from multitask_mdn_training import MultiTaskMDNConfig, MultiTaskTrainer

# Customize configuration
config = MultiTaskMDNConfig(
    n_samples=2000,           # More data samples
    window_size=50,           # Longer input windows
    num_mixtures=7,           # More mixture components
    hidden_units=[256, 128, 64],  # Larger network
    epochs=200,               # More training epochs
    batch_size=64,            # Larger batches
    task_embedding_dim=16     # Richer task embeddings
)

# Run experiment
trainer = MultiTaskTrainer(config)
# ... rest of training code
```

### 3. Results Analysis

```bash
# Basic analysis
python multitask_mdn_analyzer.py /path/to/results

# Comprehensive report
python multitask_mdn_analyzer.py /path/to/results --report --output-dir analysis_output

# Specific analyses
python multitask_mdn_analyzer.py /path/to/results --performance --uncertainty --heatmap
```

## 📊 Understanding the Results

### Key Metrics

1. **RMSE/MAE**: Standard regression metrics for point predictions
2. **Coverage**: Percentage of true values within prediction intervals (target: 95%)
3. **Interval Width**: Average width of prediction intervals
4. **Aleatoric Uncertainty**: Data-inherent uncertainty (noise, ambiguity)
5. **Epistemic Uncertainty**: Model uncertainty (knowledge gaps)

### Expected Performance Patterns

- **Sine Wave**: Lowest uncertainty, best coverage (predictable pattern)
- **Noisy Sine**: Higher aleatoric uncertainty (inherent noise)
- **Stock Price**: Balanced uncertainties (complex but learnable patterns)

### Multi-Task Benefits

1. **Shared Representations**: Common time series patterns learned across tasks
2. **Regularization Effect**: Training on multiple tasks prevents overfitting
3. **Efficiency**: Single model deployment instead of multiple specialized models
4. **Transfer Learning**: Knowledge transfer between related tasks

## 🔧 Configuration Options

### Model Architecture

```python
config = MultiTaskMDNConfig(
    # Task embedding
    task_embedding_dim=8,         # Embedding dimension for task IDs
    
    # Network architecture
    hidden_units=[128, 64, 32],   # Hidden layer sizes
    num_mixtures=5,               # Number of Gaussian mixtures
    dropout_rate=0.2,             # Dropout rate for regularization
    use_batch_norm=True,          # Use batch normalization
    
    # Regularization
    l2_regularization=1e-5        # L2 weight regularization
)
```

### Training Parameters

```python
config = MultiTaskMDNConfig(
    # Training
    epochs=150,                   # Maximum training epochs
    batch_size=128,               # Training batch size
    learning_rate=0.001,          # Initial learning rate
    optimizer='adamw',            # Optimizer choice
    early_stopping_patience=20,   # Early stopping patience
    
    # Multi-task specific
    task_balance_sampling=True    # Balance samples across tasks
)
```

### Data Generation

```python
config = MultiTaskMDNConfig(
    # Data parameters
    n_samples=1000,               # Samples per task
    window_size=30,               # Input sequence length
    pred_horizon=1,               # Prediction horizon
    
    # Time series specific
    sine_freq=0.1,                # Sine wave frequency
    noisy_sine_noise_level=0.15,  # Noise level for noisy sine
    stock_volatility=0.2          # Stock price volatility
)
```

## 📈 Analysis Tools

### Performance Comparison

```python
analyzer = MultiTaskMDNAnalyzer('results/multitask_mdn_20241203_143022')

# Individual analyses
analyzer.create_performance_comparison()
analyzer.create_uncertainty_analysis()
analyzer.create_task_comparison_heatmap()

# Comprehensive report
analyzer.create_comprehensive_report('analysis_output')
```

### Baseline Comparison

```python
# Compare with single-task baselines
baseline_results = {
    "Sine Wave": 0.15,      # Baseline RMSE
    "Noisy Sine": 0.25,
    "Stock Price": 0.35,
    "coverage": 0.90        # Baseline coverage
}

analyzer.compare_with_baseline(baseline_results)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest test_multitask_mdn.py -v
```

Test categories:
- Configuration validation
- Data generation and processing
- Model architecture
- Training pipeline
- Evaluation metrics
- Edge cases and error handling
- Integration tests

## 🎯 Best Practices

### Model Design

1. **Task Embedding Size**: Start with 8-16 dimensions, increase for more tasks
2. **Network Depth**: 2-4 hidden layers work well for most time series
3. **Mixture Components**: 3-7 mixtures balance complexity and expressiveness
4. **Regularization**: Use dropout (0.1-0.3) and L2 regularization (1e-5 to 1e-4)

### Training Tips

1. **Data Balance**: Ensure similar amounts of data across tasks
2. **Learning Rate**: Start with 1e-3, use learning rate scheduling
3. **Early Stopping**: Monitor validation loss with patience 15-25 epochs
4. **Batch Size**: Use 64-128 for stable gradient estimates

### Evaluation Guidelines

1. **Coverage Calibration**: Target 95% coverage for prediction intervals
2. **Uncertainty Decomposition**: High aleatoric = noisy data, high epistemic = model uncertainty
3. **Task Comparison**: Similar performance across tasks indicates good multi-task learning

## 🔍 Troubleshooting

### Common Issues

1. **Poor Convergence**
   - Reduce learning rate
   - Increase batch size
   - Add more regularization

2. **Low Coverage**
   - Increase number of mixtures
   - Check data scaling
   - Verify uncertainty calculations

3. **High Epistemic Uncertainty**
   - Increase model capacity
   - More training data
   - Better architecture design

4. **Task Imbalance**
   - Enable balanced sampling
   - Adjust task weights
   - Check data preprocessing

### Performance Optimization

1. **Memory Usage**: Reduce batch size or sequence length for large models
2. **Training Speed**: Use mixed precision training, optimize data loading
3. **Inference Speed**: Consider model quantization for deployment

## 📚 References

1. Bishop, C. M. (1994). "Mixture Density Networks"
2. Graves, A. (2013). "Generating Sequences With Recurrent Neural Networks"
3. Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation"
4. Ruder, S. (2017). "An Overview of Multi-Task Learning in Deep Neural Networks"

## 🤝 Contributing

When contributing to this multi-task MDN implementation:

1. Follow the project's coding standards and documentation style
2. Add comprehensive tests for new features
3. Update this README for significant changes
4. Ensure backward compatibility with existing configurations

## 📄 License

This implementation is part of the dl_techniques library and follows the same license terms.