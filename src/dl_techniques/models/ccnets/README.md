# CCNets Framework for dl_techniques

## Causal Cooperative Networks: A Revolutionary Three-Network Architecture for Explainable AI

CCNets implements a groundbreaking approach to machine learning inspired by the Three Kingdoms political philosophy. Unlike traditional single-network or adversarial two-network systems, CCNets uses three cooperative networks that work together to achieve explanation, reasoning, and faithful reproduction.

### 🏛️ Philosophical Foundation

CCNets draws inspiration from three historical periods:

- **Supervised Learning Era**: Single-network centralized approach (efficient but rigid)
- **Generative Adversarial Era**: Two-network competitive struggle (innovative but unstable)  
- **Causal Cooperative Era**: Three-network balanced cooperation (sustainable and explainable)

### 🔧 Architecture Overview

CCNets consists of three specialized networks:

1. **Explainer Network**: Creates compressed explanation vectors from input data
2. **Reasoner Network**: Makes predictions using input data and explanations
3. **Producer Network**: Generates and reconstructs data from labels and explanations

This cooperative architecture enables:
- ✨ Built-in explainability through explanation vectors
- 🔄 Cross-verification through multiple validation paths
- 🎯 Bidirectional inference (prediction and data generation)
- 🤝 Cooperative rather than adversarial training
- 📊 Data simulation for practical applications

## 🚀 Quick Start

### Installation

```python
# CCNets is integrated into dl_techniques
from dl_techniques.models.ccnets import quick_start_example

# Run a quick demonstration
results = quick_start_example("classification")
```

### Basic Usage

```python
from dl_techniques.models.ccnets import (
    create_ccnets_model,
    prepare_ccnets_data,
    train_ccnets_model
)
import numpy as np

# 1. Prepare your data
observations = np.random.randn(1000, 20)  # Input features
labels = np.random.randint(0, 5, (1000, 5))  # One-hot labels

# 2. Create data generators
train_gen, val_gen = prepare_ccnets_data(observations, labels)

# 3. Create CCNets model
model = create_ccnets_model(
    input_dim=20,
    explanation_dim=10,
    output_dim=5
)

# 4. Train the model
history = train_ccnets_model(
    model=model,
    train_data=train_gen,
    val_data=val_gen,
    epochs=50
)
```

## 📋 Complete Examples

### Classification Example

```python
from dl_techniques.models.ccnets import train_ccnets_classifier_example

# Run complete classification example
model, history, metrics = train_ccnets_classifier_example()

print(f"Cooperation Score: {metrics['cooperation_score']:.6f}")
print(f"Cross-verification Accuracy: {metrics['cross_verification_accuracy']:.4f}")
```

### Loan Approval with Data Simulation

```python
from dl_techniques.models.ccnets import train_ccnets_loan_approval_example

# Run loan approval example with bidirectional inference
model, history, metrics = train_ccnets_loan_approval_example()

# This example demonstrates:
# - Predicting loan approval/rejection
# - Generating approvable profiles from rejected applications
# - Providing explanations for decisions
```

### Custom Configuration

```python
from dl_techniques.models.ccnets import (
    create_classification_config,
    CCNetsConfig
)

# Use pre-built configuration
config = create_classification_config(
    input_dim=20,
    n_classes=5,
    explanation_dim=8
)

model = config.create_model()

# Or create custom configuration
custom_config = CCNetsConfig(
    input_dim=20,
    explanation_dim=10,
    output_dim=5,
    explainer_config={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
    reasoner_config={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
    producer_config={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
    loss_weights=[1.2, 1.0, 1.0]  # Emphasize inference consistency
)
```

## 🔬 Advanced Features

### Bidirectional Inference

CCNets uniquely supports both prediction and data generation:

```python
from dl_techniques.models.ccnets import ccnets_inference, simulate_approval_scenario

# Standard prediction
results = ccnets_inference(model, test_observations)
predictions = results['predictions']
explanations = results['explanations']

# Data simulation (e.g., for loan approval)
simulation = simulate_approval_scenario(
    model=model,
    rejected_application=rejected_data,
    explanation_vector=explanation,
    approval_label=approval_label
)

approvable_profile = simulation['approvable_application']
validation_error = simulation['validation_error']
```

### Monitoring Cooperation Quality

```python
from dl_techniques.models.ccnets import CCNetsCallback, CCNetsMetrics

# Custom callback for monitoring
callback = CCNetsCallback(
    validation_data=val_data,
    log_frequency=10,
    save_explanations=True
)

# Evaluate cooperation quality
cooperation_score = CCNetsMetrics.compute_cooperation_score(
    inference_loss, generation_loss, reconstruction_loss
)

cross_verification_acc = CCNetsMetrics.compute_cross_verification_accuracy(
    generated_observations, reconstructed_observations
)
```

## 📊 Mathematical Framework

### Core Loss Functions

CCNets implements three fundamental losses:

```python
# Inference Loss: Consistency between reconstruction and generation
inference_loss = |reconstructed_observation - generated_observation|

# Generation Loss: Fidelity of generation to original input  
generation_loss = |generated_observation - input_observation|

# Reconstruction Loss: Fidelity of reconstruction to original input
reconstruction_loss = |reconstructed_observation - input_observation|
```

### Network-Specific Errors

Each network optimizes its own cooperative objective:

```python
# Explainer improves inference and generation, reduces reconstruction dependency
explainer_error = inference_loss + generation_loss - reconstruction_loss

# Reasoner improves reconstruction and inference, reduces generation dependency
reasoner_error = reconstruction_loss + inference_loss - generation_loss

# Producer improves generation and reconstruction, reduces inference dependency  
producer_error = generation_loss + reconstruction_loss - inference_loss
```

## 🎯 Use Cases

### 1. Financial Services: Loan Approval

```python
from dl_techniques.models.ccnets import create_loan_approval_config

config = create_loan_approval_config()
model = config.create_model()

# After training, simulate approvable profiles
approvable_profile = simulate_approval_scenario(
    model, rejected_application, explanation, approval_label
)
```

**Benefits:**
- Transparent loan decisions with explanations
- Actionable feedback showing path to approval
- Regulatory compliance through auditable decisions
- Risk management via cross-verification

### 2. Healthcare: Diagnostic Systems

```python
# Configure for medical diagnosis
config = create_classification_config(
    input_dim=50,  # Symptoms and biomarkers
    n_classes=10,  # Diagnostic categories
    explanation_dim=15
)

# Provides explainable diagnoses with treatment recommendations
```

### 3. Autonomous Systems: Decision Verification

```python
# Configure for autonomous vehicle decisions
config = create_regression_config(
    input_dim=100,  # Sensor data
    output_dim=5    # Control outputs
)

# Enables decision verification through cross-validation paths
```

## 📈 Evaluation Metrics

CCNets provides specialized metrics for evaluating cooperative learning:

```python
from dl_techniques.models.ccnets import evaluate_ccnets_model

metrics = evaluate_ccnets_model(model, test_data, return_outputs=True)

# Key metrics
print(f"Cooperation Score: {metrics['cooperation_score']}")
print(f"Cross-verification Accuracy: {metrics['cross_verification_accuracy']}")
print(f"Explanation Consistency: {metrics['explanation_consistency']}")

# Individual loss components
print(f"Inference Loss: {metrics['mean_inference_loss']}")
print(f"Generation Loss: {metrics['mean_generation_loss']}")  
print(f"Reconstruction Loss: {metrics['mean_reconstruction_loss']}")
```

## 🎨 Visualization

```python
from dl_techniques.models.ccnets import visualize_ccnets_cooperation

# Visualize training dynamics
visualize_ccnets_cooperation(history)

# Shows:
# - Individual loss components over time
# - Network-specific errors
# - Cooperation quality metrics
# - Training stability indicators
```

## ⚙️ Configuration Options

### Pre-built Configurations

```python
from dl_techniques.models.ccnets import (
    create_classification_config,
    create_regression_config,
    create_loan_approval_config,
    create_default_config
)

# Classification (optimized for multi-class problems)
classification_config = create_classification_config(input_dim=20, n_classes=5)

# Regression (optimized for continuous outputs)
regression_config = create_regression_config(input_dim=15, output_dim=3)

# Loan approval (optimized for financial applications)
loan_config = create_loan_approval_config()

# Default (general purpose)
default_config = create_default_config(input_dim=20, explanation_dim=10, output_dim=5)
```

### Custom Network Architectures

```python
# Customize individual networks
explainer_kwargs = {
    'hidden_dims': [128, 64, 32],
    'activation': 'gelu',
    'dropout_rate': 0.1,
    'kernel_initializer': 'he_normal'
}

reasoner_kwargs = {
    'hidden_dims': [256, 128],
    'activation': 'swish', 
    'output_activation': 'softmax',
    'dropout_rate': 0.2
}

producer_kwargs = {
    'hidden_dims': [128, 64],
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'dropout_rate': 0.15
}

model = create_ccnets_model(
    input_dim=50,
    explanation_dim=20,
    output_dim=10,
    explainer_kwargs=explainer_kwargs,
    reasoner_kwargs=reasoner_kwargs,
    producer_kwargs=producer_kwargs,
    loss_weights=[1.5, 1.0, 1.0]  # Custom loss weighting
)
```

## 🔧 Integration with dl_techniques

CCNets is fully integrated with the dl_techniques framework:

```python
# Use with other dl_techniques components
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.utils.logger import logger

# Custom optimizer and scheduler
lr_config = {
    "type": "cosine_decay",
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "warmup_steps": 1000
}

opt_config = {
    "type": "adamw",
    "gradient_clipping_by_norm": 1.0
}

lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Use with CCNets
model.compile(optimizer=optimizer)
```

---

**Note**: CCNets represents a paradigm shift from competitive to cooperative AI. The three-network architecture provides built-in explainability, cross-verification, and bidirectional inference capabilities that make it particularly suitable for applications requiring transparency and trust.