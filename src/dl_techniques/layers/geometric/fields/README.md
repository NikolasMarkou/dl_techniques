# Holonomic Field Layers

A Keras 3 implementation of field-based neural network layers inspired by gauge theory and differential geometry, implementing the core concepts of **Holonomic AI**.

## Overview

Traditional neural networks represent data as **point vectors** in a high-dimensional space. Holonomic AI represents data as **fields** on a manifold, complete with:
- **Curvature**: How meaning varies locally
- **Connection**: How to transport information between points
- **Holonomy**: Global geometric invariants

This provides several key advantages:

| Traditional Approach | Holonomic Approach |
|---------------------|-------------------|
| Point embeddings | Field embeddings with curvature |
| Euclidean distance | Geodesic distance on manifold |
| Raw attention | Gauge-invariant attention |
| Post-hoc anomaly detection | Built-in manifold stress detection |
| Vulnerable to adversarial attacks | Geometric constraints reject inconsistent inputs |

## Mathematical Foundation

### Gauge Theory Basics

In physics, **gauge theory** describes how fields transform under local symmetries. A **gauge transformation** is a local change that doesn't affect observable physics. For neural networks:

- **Field**: The representation at each position (not just a vector, but includes curvature)
- **Connection (Γ)**: Describes how to parallel transport vectors between positions
- **Curvature (R)**: Measures how the space is curved
- **Holonomy (H)**: The result of transporting a vector around a closed loop

### Key Equations

**Parallel Transport**: Moving a vector V along a path while keeping it "parallel"
```
dV^k/dt + Γ^k_{ij} (dγ^i/dt) V^j = 0
```

**Holonomy**: The path-ordered exponential around a loop γ
```
H[γ] = P exp(-∮_γ Γ)
```

**Curvature**: Measures non-commutativity of transport
```
R^k_{lij} = ∂_i Γ^k_{jl} - ∂_j Γ^k_{il} + Γ^k_{im} Γ^m_{jl} - Γ^k_{jm} Γ^m_{il}
```

## Installation

The field layers are part of the `dl_techniques` package. Import them as:

```python
from dl_techniques.layers.geometric.fields import (
    create_field_layer,
    FieldEmbedding,
    ConnectionLayer,
    ParallelTransportLayer,
    HolonomyLayer,
    GaugeInvariantAttention,
    ManifoldStressLayer,
    HolonomicTransformerLayer,
)
```

## Quick Start

### Using the Factory

The recommended way to create field layers:

```python
from dl_techniques.layers.geometric.fields import create_field_layer

# Create a complete holonomic transformer layer
layer = create_field_layer(
    'holonomic_transformer',
    hidden_dim=256,
    num_heads=8,
    use_holonomy_features=True,
    use_anomaly_detection=True
)

# Process input
x = keras.ops.random.normal((batch_size, seq_len, 256))
output, anomaly_scores = layer(x)
```

### Direct Instantiation

```python
from dl_techniques.layers.geometric.fields import (
    FieldEmbedding,
    HolonomicTransformerLayer
)

# Field embedding for vocabulary
embedding = FieldEmbedding(
    vocab_size=10000,
    embed_dim=256,
    curvature_type='ricci',
    curvature_regularization=0.01
)

# Get embeddings and curvature
tokens = keras.ops.convert_to_tensor([[1, 2, 3, 4]])
embeddings, curvature = embedding(tokens)
```

## Layer Reference

### FieldEmbedding

Embeds tokens as fields with curvature information.

```python
embedding = FieldEmbedding(
    vocab_size=10000,           # Vocabulary size
    embed_dim=256,              # Embedding dimension
    curvature_type='ricci',     # 'metric', 'riemann', 'ricci', 'scalar'
    curvature_scale=0.1,        # Scale of curvature values
    curvature_regularization=0.01  # Smoothness regularization
)

# Returns tuple: (embeddings, curvature)
embeddings, curvature = embedding(token_ids)
```

**Curvature Types:**
- `'metric'`: Full metric tensor (d × d matrix per position)
- `'riemann'`: Riemann-like curvature tensor (antisymmetric)
- `'ricci'`: Ricci curvature (diagonal, d values per position)
- `'scalar'`: Single scalar curvature per position

### ConnectionLayer

Computes the gauge connection from field representations.

```python
connection = ConnectionLayer(
    hidden_dim=256,
    connection_type='yang_mills',  # 'yang_mills', 'levi_civita', 'affine'
    num_generators=8,              # Number of Lie algebra generators
    use_metric=True,               # Metric-compatible connection
    antisymmetric=True             # Enforce antisymmetry
)

# Compute connection from embeddings and curvature
conn = connection([embeddings, curvature])
# conn shape: (batch, seq_len, dim, dim)
```

**Connection Types:**
- `'yang_mills'`: Non-abelian gauge connection (most general)
- `'levi_civita'`: Metric-compatible, torsion-free
- `'affine'`: General affine connection

### ParallelTransportLayer

Transports vectors along paths using the connection.

```python
transport = ParallelTransportLayer(
    transport_dim=256,
    num_steps=10,                 # Integration steps
    transport_method='iterative',  # 'direct', 'iterative', 'path_ordered'
    step_size=0.1
)

# Transport vectors
transported = transport([vectors, connection])
```

**Transport Methods:**
- `'direct'`: Single-step (fast but less accurate)
- `'iterative'`: Multi-step Euler integration
- `'path_ordered'`: Full path-ordered exponential (most accurate)

### HolonomyLayer

Computes holonomy around loops for gauge-invariant features.

```python
holonomy = HolonomyLayer(
    hidden_dim=256,
    loop_sizes=[2, 4, 8],         # Sizes of loops to compute
    loop_type='rectangular',       # 'rectangular', 'triangular', 'circular'
    num_loops=4,                   # Number of loop orientations
    use_trace=True                 # Return Wilson loop (trace)
)

# Compute holonomy features
holonomy_features = holonomy([embeddings, connection])
```

**Loop Types:**
- `'rectangular'`: Axis-aligned rectangular loops
- `'triangular'`: Triangular paths
- `'circular'`: Approximate circular loops (for 2D data)

### GaugeInvariantAttention

Attention mechanism that respects gauge structure.

```python
attention = GaugeInvariantAttention(
    hidden_dim=256,
    num_heads=8,
    attention_metric='hybrid',     # 'holonomy', 'geodesic', 'curvature', 'hybrid'
    use_curvature_gating=True,     # Gate attention by curvature
    use_parallel_transport=True    # Transport values before aggregation
)

# Compute attention
output = attention([embeddings, curvature, connection])
```

**Attention Metrics:**
- `'holonomy'`: Based on holonomy between positions
- `'geodesic'`: Uses curvature-weighted distance
- `'curvature'`: Attends more to similar curvature
- `'hybrid'`: Combines all metrics

### ManifoldStressLayer

Detects anomalies through geometric inconsistency.

```python
stress = ManifoldStressLayer(
    hidden_dim=256,
    stress_types=['curvature', 'connection', 'combined'],
    stress_threshold=0.5,
    use_learnable_baseline=True
)

# Compute stress and anomaly mask
stress_values, anomaly_mask = stress([embeddings, curvature, connection])
```

**Use Cases:**
- Detect adversarial inputs
- Identify poisoned training data
- Flag out-of-distribution samples
- Confidence scoring

### HolonomicTransformerLayer

Complete transformer layer with all holonomic components.

```python
layer = HolonomicTransformerLayer(
    hidden_dim=256,
    num_heads=8,
    ffn_dim=1024,
    curvature_type='ricci',
    connection_type='yang_mills',
    attention_metric='hybrid',
    use_holonomy_features=True,
    use_anomaly_detection=True,
    dropout_rate=0.1,
    normalization_type='field_norm',
    activation='gelu'
)

# Forward pass
output, anomaly_scores = layer(x, training=True)
```

## Architecture Patterns

### Building a Holonomic Transformer

```python
class HolonomicTransformer(keras.Model):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Field embedding with curvature
        self.embedding = FieldEmbedding(
            vocab_size=vocab_size,
            embed_dim=hidden_dim,
            curvature_type='ricci'
        )
        
        # Stack of holonomic transformer layers
        self.layers_list = [
            HolonomicTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                use_holonomy_features=True,
                use_anomaly_detection=True
            )
            for _ in range(num_layers)
        ]
        
        # Final projection
        self.output_projection = keras.layers.Dense(vocab_size)
    
    def call(self, tokens, training=None):
        # Embed tokens as fields
        embeddings, curvature = self.embedding(tokens, training=training)
        
        # Accumulate anomaly scores
        total_anomaly = None
        x = embeddings
        
        # Process through layers
        for layer in self.layers_list:
            x, anomaly = layer(x, training=training)
            if total_anomaly is None:
                total_anomaly = anomaly
            else:
                total_anomaly = total_anomaly + anomaly
        
        # Output
        logits = self.output_projection(x)
        
        return logits, total_anomaly / len(self.layers_list)
```

### Anomaly-Aware Training

```python
# Custom training step that uses anomaly scores
class AnomalyAwareModel(keras.Model):
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits, anomaly_scores = self(x, training=True)
            
            # Standard loss
            task_loss = self.compute_loss(y=y, y_pred=logits)
            
            # Weight samples by inverse anomaly (downweight anomalous)
            sample_weights = 1.0 / (1.0 + anomaly_scores)
            weighted_loss = task_loss * keras.ops.squeeze(sample_weights)
            
            # Mean loss
            loss = keras.ops.mean(weighted_loss) + sum(self.losses)
        
        # Update
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss, 'anomaly_mean': keras.ops.mean(anomaly_scores)}
```

## Security Benefits

### Adversarial Robustness

The holonomic approach provides natural adversarial robustness:

1. **Perturbations must respect geometry**: Random noise increases manifold stress and is detected
2. **Gauge invariance**: Transformations that don't affect holonomy are recognized as equivalent
3. **Curvature consistency**: Adversarial examples often have inconsistent local curvature

```python
# Example: Filtering adversarial inputs
stress_layer = ManifoldStressLayer(hidden_dim=256, stress_threshold=0.3)

def filter_adversarial(model, inputs):
    embeddings, curvature, connection = model.compute_field_structure(inputs)
    stress, anomaly_mask = stress_layer([embeddings, curvature, connection])
    
    # Reject high-stress inputs
    clean_mask = ~anomaly_mask
    return inputs[clean_mask]
```

### Poison Detection

Poisoned training data increases manifold stress:

```python
# During data loading
def detect_poison(dataset, model, threshold=0.5):
    stress_scores = []
    
    for batch in dataset:
        _, anomaly = model(batch, training=False)
        stress_scores.extend(anomaly.numpy())
    
    # Flag potential poison
    poison_candidates = [
        i for i, score in enumerate(stress_scores) 
        if score > threshold
    ]
    
    return poison_candidates
```

## Performance Considerations

### Computational Complexity

| Layer | Complexity | Notes |
|-------|-----------|-------|
| FieldEmbedding | O(n·d) | Same as standard embedding |
| ConnectionLayer | O(n·d²) | Quadratic in dimension |
| ParallelTransport | O(n·k·d²) | k = num_steps |
| HolonomyLayer | O(n·L·s·d²) | L = loops, s = loop size |
| GaugeInvariantAttention | O(n²·d/h) | h = num_heads |
| ManifoldStressLayer | O(n·d²) | One-time computation |
| HolonomicTransformer | O(n²·d + n·d²) | Dominated by attention |

### Memory Usage

The field representation requires additional memory for curvature and connection:
- Curvature (ricci): O(n·d) additional
- Connection: O(n·d²) additional

For large models, consider:
- Using `'scalar'` curvature type
- Reducing `num_generators` in connection
- Using `'direct'` transport method

### Optimization Tips

1. **Start with defaults**: The factory provides sensible defaults
2. **Adjust curvature type**: Use `'scalar'` for memory efficiency
3. **Tune regularization**: Higher `curvature_regularization` for stability
4. **Use anomaly detection wisely**: Can be disabled for inference speed

## Integration with Existing Models

### Adding Holonomic Layers to Existing Architectures

```python
# Replace standard attention with gauge-invariant
class EnhancedTransformerBlock(keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        
        # Replace standard attention
        self.attention = GaugeInvariantAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attention_metric='curvature'
        )
        
        # Add curvature computation
        self.curvature_proj = keras.layers.Dense(hidden_dim, activation='tanh')
        
        # Keep standard FFN
        self.ffn = keras.layers.Dense(hidden_dim * 4, activation='gelu')
        self.ffn_out = keras.layers.Dense(hidden_dim)
        
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
    
    def call(self, x, training=None):
        # Compute curvature
        curvature = self.curvature_proj(x) * 0.1
        
        # Gauge-invariant attention (no connection for simplicity)
        attn = self.attention([self.norm1(x), curvature, None], training=training)
        x = x + attn
        
        # Standard FFN
        ffn = self.ffn_out(self.ffn(self.norm2(x)))
        return x + ffn
```

## API Reference

### Factory Functions

#### `create_field_layer(layer_type, name=None, **kwargs)`

Main factory function for creating field layers.

**Parameters:**
- `layer_type`: One of `'field_embedding'`, `'connection'`, `'parallel_transport'`, `'holonomy'`, `'gauge_attention'`, `'manifold_stress'`, `'holonomic_transformer'`, `'field_norm'`
- `name`: Optional layer name
- `**kwargs`: Layer-specific parameters

**Returns:** Configured `keras.layers.Layer` instance

#### `create_field_layer_from_config(config)`

Create layer from configuration dictionary.

**Parameters:**
- `config`: Dictionary with `'type'` key and layer parameters

**Returns:** Configured `keras.layers.Layer` instance

#### `get_field_layer_info()`

Get information about available layer types.

**Returns:** Dictionary mapping types to info including `'class'`, `'required_params'`, `'default_params'`, `'description'`

#### `validate_field_config(layer_type, **kwargs)`

Validate configuration before layer creation.

**Raises:** `ValueError` if invalid

## References

- Holonomic AI concepts
- Differential geometry and gauge theory
- Fiber bundles in machine learning
- Non-Abelian geometric deep learning