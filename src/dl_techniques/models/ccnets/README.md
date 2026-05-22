# Causal Cooperative Networks (CCNets) Framework

A model-agnostic meta-framework for implementing Causal Cooperative Networks (CCNets) - neural architectures that learn true causal relationships rather than mere associations.

## Overview

Modern deep learning excels at learning associations but struggles with causation. CCNets bridge this gap by employing three cooperative neural networks that continuously verify each other's reasoning, forcing the system to learn the true data-generating process.

### Key Innovation

Unlike traditional neural networks that learn `P(Y|X)` (correlation), CCNets decompose the problem into three causal components:

- **Explainer**: Models `P(E|X)` - Extracts latent causes/context
- **Reasoner**: Models `P(Y|X,E)` - Performs context-aware inference  
- **Producer**: Models `P(X|Y,E)` - Verifies by reconstruction/generation

This tripartite architecture enables **counterfactual reasoning** - the ability to answer "what if" questions impossible for standard classifiers.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Framework Architecture](#framework-architecture)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```python
import keras
from dl_techniques.models.ccnets import CCNetOrchestrator, CCNetConfig, wrap_keras_model

# 1. Define your three models (see CLAUDE.md for the required call signatures:
#    explainer(x) -> (mu, log_var); reasoner(x, e) -> y_probs; producer(y, e) -> x_hat)
explainer = ...  # Your model for P(E|X)
reasoner = ...   # Your model for P(Y|X,E)
producer = ...   # Your model for P(X|Y,E)

# 2. Create orchestrator
orchestrator = CCNetOrchestrator(
    explainer=wrap_keras_model(explainer),
    reasoner=wrap_keras_model(reasoner),
    producer=wrap_keras_model(producer),
    config=CCNetConfig(explanation_dim=128)
)

# 3. Train with automatic orchestration
trainer = CCNetTrainer(orchestrator)
trainer.train(train_dataset, epochs=50)

# 4. Generate counterfactuals
x_counterfactual = orchestrator.counterfactual_generation(x_reference, y_target)
```

## Core Concepts

### Mathematical Foundation

CCNets enforce three fundamental conditions for causal learning:

1. **Independence**: `P(Y, E) = P(Y) * P(E)`
   - Explicit and latent causes are independent
   
2. **Conditional Dependence**: `P(Y | X, E) ≠ P(Y | X)`
   - Context is essential for accurate inference
   
3. **Necessity & Sufficiency**: Modeled by `P(X | Y, E)`
   - Both causes together fully explain the effect

### The Three Losses

The framework computes three fundamental losses that measure different aspects of causal consistency:

| Loss | Formula | Meaning |
|------|---------|---------|
| **Generation Loss** | `\|\|X_generated - X_input\|\|` | Quality of generation from true causes |
| **Reconstruction Loss** | `\|\|X_reconstructed - X_input\|\|` | Total inference + reconstruction error |
| **Inference Loss** | `\|\|X_reconstructed - X_generated\|\|` | Error attributable to incorrect inference |

### Evolved Causal Credit Assignment

Each network receives a specialized error signal computed from the three fundamental losses. The formulation has evolved from early conceptual models to ensure stable and robust training dynamics.

- **Explainer Error** = `w_inf * Inference Loss + w_gen * Generation Loss + w_kl * KL Divergence`
- **Reasoner Error** = `w_inf * Classification Loss (cross-entropy) + w_rec * Reconstruction Loss`
- **Producer Error** = `w_gen * Generation Loss + w_rec * Reconstruction Loss`

This credit assignment ensures each network learns its specific causal role:
- The **Explainer** is penalized for producing a latent code `E` that is ambiguous for the Reasoner (high `Inference Loss`) or insufficient for the Producer (high `Generation Loss`). The KL term regularizes the latent space for efficiency and smoothness.
- The **Reasoner** is penalized by a direct categorical cross-entropy **anchor** against the ground-truth label (stable, well-conditioned classification gradients) plus the cooperative `Reconstruction Loss` (it is penalized for inferences the Producer cannot turn back into the observation). The cooperative term only carries gradient if the Producer consumes the label *differentiably* — see `CLAUDE.md`, Invariant 1.
- The **Producer** is penalized for any failure in its ability to manifest an observation, whether from the system's own inference (`Reconstruction Loss`) or from ground-truth causes (`Generation Loss`).

#### Note on the Evolution of the Error Formulation
Previous iterations of this doctrine utilized a subtractive component in the error calculation to model a conceptual "reward." This was found to be operationally unsound, creating the potential for negative losses and unstable gradient dynamics. The current additive formulation is a more robust implementation that guarantees a positive definite error landscape, ensuring stable convergence while implicitly optimizing the same systemic objectives.

## Framework Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CCNetOrchestrator                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Forward Pass Flow                  │    │
│  │                                                 │    │
│  │    X ──► Explainer ──► E ──┐                    │    │
│  │    │                       ▼                    │    │
│  │    └──────────────► Reasoner ──► Y_inf          │    │
│  │                            │                    │    │
│  │    Y_truth ──┐             ▼                    │    │
│  │              ├──► Producer ──► X_gen            │    │
│  │    Y_inf ────┤      │                           │    │
│  │              └──────┴──────► X_recon            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Loss Computation Engine               │    │
│  │                                                 │    │
│  │  • Generation Loss    = ||X_gen - X||           │    │
│  │  • Reconstruction Loss = ||X_recon - X||        │    │
│  │  • Inference Loss     = ||X_recon - X_gen||     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Gradient Isolation & Routing            │    │
│  │                                                 │    │
│  │  • Explainer ← Explainer Error (freeze others)  │    │
│  │  • Reasoner  ← Reasoner Error (freeze others)   │    │
│  │  • Producer  ← Producer Error (freeze others)   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Description | Input | Output |
|-----------|-------------|-------|--------|
| **CCNetModule** | Protocol defining model interface | - | - |
| **CCNetOrchestrator** | Main orchestration engine | 3 models + config | Managed training |
| **CCNetConfig** | Configuration container | Parameters | Settings |
| **CCNetTrainer** | High-level training manager | Orchestrator | Training loop |
| **CCNetLosses** | Loss value container | Tensors | 3 losses |
| **CCNetModelErrors** | Error signal container | Losses | 3 errors |

## Usage Guide

### Model Requirements

Any neural network can be used with CCNet as long as it implements the correct interface:

#### Explainer Requirements
```python
def explainer(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        x: Input observation [batch, ...]
    Returns:
        (mu, log_var): parameters of the latent distribution, each
                       [batch, explanation_dim]. The orchestrator samples
                       E = mu + eps * exp(0.5 * log_var) and applies a KL term.
    """
```

#### Reasoner Requirements
```python
def reasoner(x: Tensor, e: Tensor) -> Tensor:
    """
    Args:
        x: Input observation [batch, ...]
        e: Latent explanation [batch, explanation_dim]
    Returns:
        y: Predicted labels [batch, num_classes]
    """
```

#### Producer Requirements
```python
def producer(y: Tensor, e: Tensor) -> Tensor:
    """
    Args:
        y: Class labels [batch, num_classes]
        e: Latent explanation [batch, explanation_dim]
    Returns:
        x: Generated observation [batch, ...]
    """
```

### Configuration Options

```python
config = CCNetConfig(
    explanation_dim=128,          # Dimension of latent vector E
    loss_fn='l2',                 # 'l1', 'l2', 'huber', or 'polynomial'
    loss_fn_params={},            # kwargs for the loss (e.g. {'delta': 1.0} for huber)
    learning_rates={              # Per-module learning rates
        'explainer': 1e-4,
        'reasoner': 1e-4,
        'producer': 1e-4
    },
    gradient_clip_norm=1.0,       # Max gradient norm (None to disable)
    use_mixed_precision=False,    # Enable mixed precision training
    sequential_data=False,        # Use SequentialCCNetOrchestrator for sequences
    # The KL-divergence weight lives inside explainer_weights, not as a top-level field:
    explainer_weights={'inference': 1.0, 'generation': 1.0, 'kl_divergence': 0.01},
    reasoner_weights={'inference': 1.0, 'reconstruction': 1.0},
    producer_weights={'generation': 1.0, 'reconstruction': 1.0},
    dynamic_weighting=False       # DEPRECATED — leave False (no effect; see CLAUDE.md)
)
```

> **Note:** loss weights are read inside the compiled `train_step`, so they are fixed at
> graph-trace time. The KL weight is the exception — the orchestrator exposes it as a live
> `tf.Variable` (`orchestrator.kl_weight`), which is what makes `CCNetTrainer`'s KL
> annealing work.

### Training Workflow

```python
# 1. Prepare your data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(32).shuffle(1000)

# 2. Create trainer with callbacks
trainer = CCNetTrainer(orchestrator)

# 3. Define custom callbacks
early_stopping = EarlyStoppingCallback(patience=10, error_threshold=1e-4)

def metrics_callback(epoch, metrics):
    print(f"Epoch {epoch}: Generation Loss = {metrics['generation_loss']:.4f}")

# 4. Train with automatic orchestration
trainer.train(
    train_dataset=train_dataset,
    epochs=100,
    validation_dataset=val_dataset,
    callbacks=[early_stopping, metrics_callback]
)

# 5. Access training history
history = trainer.history
```

## Advanced Features

### Counterfactual Generation

Generate "what if" scenarios by mixing causes:

```python
# "What would this '3' look like if it were an '8' in the same style?"
x_reference = load_image_of_3()
y_target = one_hot_encode(8)

x_counterfactual = orchestrator.counterfactual_generation(x_reference, y_target)
```

### Style Transfer

Combine content from one observation with style from another:

```python
# "Draw this '7' in the style of that '4'"
x_content = load_image_of_7()
x_style = load_image_of_4()

x_transferred = orchestrator.style_transfer(x_content, x_style)
```

### Causal Disentanglement

Extract the independent causal factors:

```python
y_explicit, e_latent = orchestrator.disentangle_causes(x_input)

print(f"Explicit cause (label): {y_explicit}")
print(f"Latent cause (style): {e_latent}")
```

### Consistency Verification

Check if the model's reasoning is internally consistent:

```python
is_consistent = orchestrator.verify_consistency(x_input, threshold=0.01)

if is_consistent:
    print("Model reasoning is causally consistent")
```

### Sequential Data Support

For time series and text, use the specialized orchestrator:

```python
from dl_techniques.models.ccnets import SequentialCCNetOrchestrator

# Producer will use reverse causality via sequence reversal
seq_orchestrator = SequentialCCNetOrchestrator(
    explainer=transformer_explainer,
    reasoner=transformer_reasoner,
    producer=reverse_transformer_producer,  # Implements reverse-causal mask
    config=CCNetConfig(sequential_data=True)
)
```

## API Reference

### Core Classes

#### CCNetOrchestrator

```python
class CCNetOrchestrator:
    def __init__(self, explainer, reasoner, producer, config=None, control_strategy=None)
    kl_weight: tf.Variable                       # live KL weight (annealable)
    def forward_pass(self, x_input, y_truth, training=True) -> Dict
    def compute_losses(self, tensors) -> CCNetLosses
    def compute_model_errors(self, losses, tensors) -> CCNetModelErrors
    def train_step(self, x_input, y_truth) -> Dict[str, float]
    def evaluate(self, x_input, y_truth) -> Dict[str, float]
    def counterfactual_generation(self, x_reference, y_target) -> Tensor
    def style_transfer(self, x_content, x_style) -> Tensor
    def disentangle_causes(self, x_input) -> Tuple[Tensor, Tensor]
    def verify_consistency(self, x_input, threshold=0.01) -> bool
    def save_models(self, base_path: str)
    def load_models(self, base_path: str)
```

#### CCNetConfig

```python
@dataclass
class CCNetConfig:
    explanation_dim: int = 128
    loss_fn: str = 'l2'                  # 'l1' | 'l2' | 'huber' | 'polynomial'
    loss_fn_params: Dict[str, Any] = {}
    learning_rates: Dict[str, float] = {'explainer': 1e-4, 'reasoner': 1e-4, 'producer': 1e-4}
    gradient_clip_norm: Optional[float] = 1.0
    use_mixed_precision: bool = False
    sequential_data: bool = False
    explainer_weights: Dict[str, float] = {'inference': 1.0, 'generation': 1.0, 'kl_divergence': 0.01}
    reasoner_weights: Dict[str, float] = {'reconstruction': 1.0, 'inference': 1.0}
    producer_weights: Dict[str, float] = {'generation': 1.0, 'reconstruction': 1.0}
    dynamic_weighting: bool = False      # DEPRECATED — no effect
```

#### CCNetTrainer

```python
class CCNetTrainer:
    def __init__(self, orchestrator, metrics_callback=None, kl_annealing_epochs=None)
    def train(self, train_dataset, epochs, validation_dataset=None, callbacks=None)
    history: Dict[str, List[float]]      # populated during train()
```

### Utility Functions

```python
def wrap_keras_model(model: keras.Model) -> CCNetModule
"""Wrap a Keras model to comply with CCNetModule protocol."""
```

## Examples

### MNIST Digit Generation (reference task)

The implemented, runnable reference task is `src/train/ccnets/mnist.py`. It defines the
three networks (`MNISTExplainer`, `MNISTReasoner`, `MNISTProducer`), builds and wraps them
via `create_mnist_ccnet`, trains with `CCNetTrainer`, and emits reconstruction /
counterfactual / latent-space visualizations.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.ccnets.mnist
```

```python
from train.ccnets.mnist import create_mnist_ccnet, prepare_mnist_data, ExperimentConfig
from dl_techniques.models.ccnets import CCNetTrainer

config = ExperimentConfig()
orchestrator = create_mnist_ccnet(config)
train_ds, val_ds = prepare_mnist_data(config.data)
CCNetTrainer(orchestrator, kl_annealing_epochs=10).train(train_ds, epochs=50,
                                                         validation_dataset=val_ds)

# Counterfactual: "what would this digit look like as a different digit, same style?"
x_counterfactual = orchestrator.counterfactual_generation(x_reference, y_target_one_hot)
```

See `src/train/ccnets/CLAUDE.md` for how to derive a new training script from this template.

### Other modalities

The framework is modality-agnostic — text and time-series CCNets are possible by supplying
appropriate Explainer/Reasoner/Producer networks (use `SequentialCCNetOrchestrator` for
sequential data). No text or time-series task is implemented in the repository yet; the
MNIST task above is the only worked example. See `CLAUDE.md` for the task-adaptation table.