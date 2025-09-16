# FNet: Fourier Transform-based Language Model

A production-ready implementation of the FNet architecture, a highly efficient transformer model that replaces self-attention with parameter-free Fourier transforms. This implementation is enhanced with modern features from the `dl-techniques` library, including a robust multi-task learning framework.

Based on: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)

## Overview

FNet offers a compelling alternative to traditional transformer models like BERT by replacing the computationally intensive self-attention mechanism with simple, unparameterized Fourier transforms. This design choice leads to dramatic speedups in training and inference, especially for long sequences, while maintaining a high percentage of the accuracy of attention-based models.

The architecture is ideal for applications where latency and computational cost are critical factors.

### Key Architecture Features

- **Fourier Transform Token Mixing**: The core innovation. A 2D Fast Fourier Transform (FFT) is applied across the sequence and hidden dimensions to mix token information, completely replacing the need for attention.
- **Parameter-Free Mixing**: The FFT requires no learnable weights, significantly reducing the model's parameter count and memory footprint compared to BERT.
- **O(N log N) Complexity**: The FFT scales with `O(N log N)` complexity for sequence length `N`, a major improvement over the `O(N^2)` complexity of self-attention.
- **Architectural Simplicity**: FNet retains the standard transformer block structure (mixing layer -> FFN) but with a much simpler mixing component, making it easier to implement and optimize.
- **Multi-Task Capability**: This implementation includes a powerful multi-task framework, allowing a single FNet backbone to be fine-tuned on multiple NLP tasks simultaneously.

## Architecture Diagram

The basic structure of a single FNet Encoder block:

```
Input Processing:
    Token IDs + Position IDs → Word & Position Embeddings
                                        ↓
FNet Block (repeated N times):
    ┌─ LayerNorm → Fourier Transform → ⊕ (Residual)
    └─ LayerNorm → Feed-Forward Network → ⊕ (Residual)
                                        ↓
Output Processing:
    Sequence Output → [CLS] Token → Pooler (Dense + Tanh) → Pooled Output
```

### Detailed Layer Structure

Each FNet block applies two main transformations with residual connections:
1. **Fourier Mixing Layer**: The input is normalized, then processed by a 2D FFT, and the result is added back to the original input.
2. **Feed-Forward Layer**: The output of the mixing layer is normalized, passed through a standard feed-forward network (e.g., MLP or SwiGLU), and the result is added back.

## Model Variants

| Variant | Layers | Hidden Size | FFN Size | Parameters (BERT-like) | Description |
|---------|--------|-------------|----------|------------------------|-------------|
| `base` | 12 | 768 | 3072 | ~110M | Matches BERT-Base architecture |
| `large` | 24 | 1024 | 4096 | ~340M | Matches BERT-Large architecture |
| `small` | 6 | 512 | 2048 | ~60M | Lightweight variant for faster inference |
| `tiny` | 4 | 256 | 1024 | ~20M | Ultra-lightweight for edge deployment |

*Note: Parameter counts are approximate and vary based on FFN type and vocabulary size.*

## Installation

```bash
pip install dl-techniques keras>=3.8.0 tensorflow>=2.18.0
```

## Quick Start

### Basic Usage

```python
from dl_techniques.models.fnet.model import FNet

# Create a base FNet model with a pooling layer
model = FNet.from_variant("base", add_pooling_layer=True)

# Get sequence and pooled outputs
input_ids = keras.random.uniform((2, 128), 0, 30522, dtype="int32")
sequence_output, pooled_output = model(input_ids)

print(f"Sequence Output Shape: {sequence_output.shape}") # (2, 128, 768)
print(f"Pooled Output Shape: {pooled_output.shape}")   # (2, 768)

# Print detailed model information
model.summary()
```

### Classification Fine-tuning

Use the factory function to create a complete model with a classification head.

```python
from dl_techniques.models.fnet.model import create_fnet

# Create an FNet model for a 3-class classification task
classifier = create_fnet(
    variant="base",
    task_type="classification",
    num_classes=3
)

# Compile for classification
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# classifier.fit(train_data, epochs=3, validation_data=val_data)
```

### Multi-Task Learning with FNet

Leverage a single, efficient FNet backbone for multiple tasks simultaneously.

```python
from dl_techniques.models.fnet.multitask import MultiTaskFNet, TaskConfig, TaskType

# 1. Define your tasks
task_configs = [
    TaskConfig(name="sentiment", task_type=TaskType.CLASSIFICATION, num_classes=2),
    TaskConfig(name="ner", task_type=TaskType.TOKEN_CLASSIFICATION, num_classes=9),
    TaskConfig(name="sts", task_type=TaskType.REGRESSION, loss_weight=0.5)
]

# 2. Create the Multi-Task model from an FNet variant
multi_task_model = MultiTaskFNet.from_variant("base", task_configs=task_configs)

# 3. Compile the model with custom settings
multi_task_model.compile_multitask(optimizer="adamw", learning_rate=3e-5)

# 4. Train the model (requires a data pipeline yielding dicts of labels)
# multi_task_model.fit(train_dataset, epochs=3)
```

## Advanced Configuration

Create a custom FNet model with modern components like RMSNorm and SwiGLU FFNs.

```python
from dl_techniques.models.fnet.model import FNet

# Define a custom FNet configuration
custom_fnet = FNet(
    vocab_size=50265,
    hidden_size=512,
    num_layers=8,
    intermediate_size=1536,
    max_position_embeddings=1024,
    normalization_type="rms_norm",  # Use RMSNorm instead of LayerNorm
    ffn_type="swiglu",             # Use SwiGLU instead of standard MLP
    add_pooling_layer=True,
    name="custom_fnet"
)

custom_fnet.summary()
```

## Performance Characteristics

### Computational Efficiency

FNet's primary advantage is its speed, stemming directly from replacing attention.

- **Speed**: Up to 7x faster than BERT-Base on TPUs and 2x faster on GPUs for training.
- **Linearithmic Complexity**: The Fourier Transform scales at `O(L log L)` with sequence length `L`, making it highly effective for long documents.
- **Reduced Memory**: The absence of large attention matrices (`N x N`) reduces the memory required during the forward pass.

### FNet vs. BERT: A Comparison

| Metric | BERT | FNet | Improvement |
|-----------------|---------------------|-----------------------|-----------------------|
| **Complexity** | `O(L² * D)` | `O(L * log(L) * D)` | Quadratic to Linearithmic |
| **Parameters** | Baseline (e.g., 110M) | ~5-10% fewer | Reduced model size |
| **Training Speed**| Baseline | 2-7x faster | Significant speedup |
| **GLUE Accuracy** | Baseline (100%) | ~92-97% | Minor accuracy trade-off |

## Training and Fine-tuning

### Fine-tuning Best Practices

- **Learning Rate**: Use a small learning rate, typically between 1e-5 and 5e-5, with a linear decay schedule.
- **Batch Size**: Larger batch sizes (e.g., 32, 64) often improve stability and performance.
- **Epochs**: Fine-tuning usually requires only a few epochs (2-4).

### Standard Fine-tuning Loop

Since FNet has no special loss terms (unlike MoE models), it can be trained with a standard Keras `fit` loop.

```python
# Assuming 'classifier' is the model from the Quick Start section
# and 'train_dataset' yields (inputs, labels) tuples
# where inputs is a dict: {"input_ids": ..., "attention_mask": ..., "position_ids": ...}

classifier.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=3
)
```

## Technical Specifications

### Architecture Details

- **Block Structure**: FNet Encoder (Fourier Mixing + FFN).
- **Normalization**: Configurable (default: `layer_norm`, supports `rms_norm`).
- **Position Encoding**: Absolute, learned position embeddings.
- **FFN**: Configurable (default: `mlp` with GELU, supports `swiglu`).
- **Token Mixing**: 2D Discrete Fourier Transform over sequence and hidden dimensions.

### Mathematical Foundation

**Fourier Transform Token Mixing:**
The core mixing operation is a two-step, parameter-free process:
1. Apply DFT along the sequence dimension: $X' = \mathcal{F}_{\text{seq}}(X)$
2. Apply DFT along the hidden dimension: $X'' = \mathcal{F}_{\text{hidden}}(X')$
3. The output is the real part of the result: $Y = \Re(X'')$

This operation effectively mixes information across all token positions in a computationally efficient manner.

## Model Serialization

The FNet model and its multi-task variants are fully serializable using the Keras 3 format.

```python
import keras
from dl_techniques.models.fnet.model import FNet

# Save a trained model
model = FNet.from_variant("base")
# ... train the model ...
model.save("my_fnet_model.keras")

# Load the model with custom objects automatically registered
loaded_model = keras.models.load_model("my_fnet_model.keras")

loaded_model.summary()
```