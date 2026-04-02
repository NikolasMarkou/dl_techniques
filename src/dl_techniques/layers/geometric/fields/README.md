# Holonomic Field Layers

Keras 3 layers that replace flat vector representations with **gauge fields** ---
vectors enriched with curvature, connection, and holonomy --- bringing the
mathematics of differential geometry and gauge theory into neural network
processing.

## Introduction

Standard neural network layers treat representations as point vectors in flat
Euclidean space.  Every position gets a single vector, and information flows
between positions through bilinear operations (dot-product attention, linear
projections) that know nothing about the *geometry* of the representation
space.

This package replaces that flat picture with one borrowed from **gauge theory**
and **differential geometry** --- the same mathematics that describes
electromagnetism, general relativity, and the Standard Model of particle
physics.  The central idea is:

> **Represent each token not as a point in R^d, but as a local section of a
> fibre bundle** --- a vector *plus* the curvature and connection that describe
> how the space bends and twists around it.

### Why geometry matters for neural networks

When a network carries geometric structure, three things change:

1. **Information transport becomes structure-aware.**
   Moving a value from position *i* to position *j* is no longer a simple copy;
   it passes through a **parallel transport** operator derived from the learned
   connection.  The transport preserves inner products and respects curvature,
   so the network learns *how* meaning transforms as it moves --- not just
   *that* it moves.

2. **Attention becomes gauge-invariant.**
   Ordinary dot-product attention depends on the raw coordinate values of Q and
   K.  A local change of basis (a *gauge transformation*) at one position can
   arbitrarily change the attention pattern.  Gauge-invariant attention scores
   depend only on geometric quantities --- curvature agreement, geodesic
   distance, holonomy along the path between positions --- which are
   independent of the choice of local frame.

3. **Anomaly detection falls out of the geometry.**
   Inputs that are consistent with the learned manifold structure have low
   *manifold stress* (smooth curvature, small connection variation,
   near-identity holonomy).  Adversarial perturbations, poisoned data, or
   out-of-distribution samples break these geometric invariants and can be
   detected without a separate classifier.

### How the layers compose

The layers form a pipeline that mirrors the mathematical structure of a
gauge field theory:

```
tokens
  |
  v
FieldEmbedding             token -> (embedding, curvature)
  |
  v
ConnectionLayer            (embedding, curvature) -> connection Gamma
  |
  +---> ParallelTransport        transport vectors using Gamma
  |
  +---> HolonomyLayer            gauge-invariant loop features from Gamma
  |
  +---> GaugeInvariantAttention   attention scored by geometric quantities
  |
  +---> ManifoldStressLayer       anomaly detection via geometric stress
  |
  v
HolonomicTransformerLayer  wraps all of the above into a single
                           drop-in transformer block
```

Each layer can be used independently (e.g. plug `GaugeInvariantAttention`
into an existing transformer) or composed through
`HolonomicTransformerLayer` which orchestrates the full pipeline.

### Key mathematical objects

| Object | Symbol | Shape (per position) | Role |
|---|---|---|---|
| Embedding | *e* | `(D,)` | Representation vector |
| Curvature | *R* | `(D,)` ricci / `(D,D)` metric | Local geometry |
| Connection | *Gamma* | `(D, D)` | Parallel transport rule |
| Holonomy | *H[gamma]* | scalar (trace) | Gauge-invariant loop feature |
| Stress | *sigma* | scalar | Anomaly / inconsistency score |

---

## Mathematical Foundation

### Gauge Theory for Neural Networks

In physics, **gauge theory** describes how fields transform under local
symmetries.  A **gauge transformation** is a local change of basis that does
not affect observable (gauge-invariant) quantities.  Translating to neural
networks:

- **Field** -- the representation at each position: a vector *plus*
  curvature that describes how meaning varies locally.
- **Connection (Gamma)** -- a matrix at each position prescribing how to
  parallel-transport vectors between neighbouring positions.
- **Curvature (R)** -- measures how much the space is curved; computed from
  the connection or learned directly.
- **Holonomy (H)** -- the net rotation accumulated by transporting a vector
  around a closed loop.  Non-trivial holonomy signals curvature.

### Core Equations

**Parallel transport** -- move a vector *V* along a path keeping it "parallel":

```
dV^k / dt  +  Gamma^k_j  V^j  =  0
```

In the discrete layer this becomes an Euler step
`V' = V - h * Gamma @ V`, iterated `num_steps` times for higher accuracy.

**Holonomy** -- the path-ordered exponential around a closed loop gamma:

```
H[gamma]  =  P exp( -oint_gamma  Gamma )
```

Approximated in the layer via the commutator `[Gamma_s, Gamma_{s+k}]` at
different offsets *k*, whose trace `Tr([A,B]^2)` is a gauge-invariant
scalar feature.

**Curvature** -- measures non-commutativity of transport:

```
F_ij  =  d_i Gamma_j  -  d_j Gamma_i  +  [Gamma_i, Gamma_j]
```

The learned curvature tensor serves as a soft metric on the representation
space; positions with similar curvature attend to each other more strongly.

---

## Usage

### Imports

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
    FieldNormalization,
)
```

### Factory construction (recommended)

```python
layer = create_field_layer(
    'holonomic_transformer',
    hidden_dim=256,
    num_heads=8,
    use_holonomy_features=True,
    use_anomaly_detection=True,
)

output, anomaly_scores = layer(x)   # x: (batch, seq_len, 256)
```

### Direct instantiation

```python
embedding = FieldEmbedding(
    vocab_size=10000,
    embed_dim=256,
    curvature_type='ricci',
)
embeddings, curvature = embedding(token_ids)  # token_ids: (batch, seq_len)
```

---

## Layer Reference

### FieldEmbedding

Maps integer token ids to `(embedding, curvature)` pairs.

```python
FieldEmbedding(
    vocab_size,                       # vocabulary size
    embed_dim,                        # embedding dimension D
    curvature_type='ricci',           # 'metric' | 'riemann' | 'ricci' | 'scalar'
    curvature_scale=0.1,              # initial curvature magnitude
    curvature_regularization=0.01,    # smoothness penalty
)
# call(token_ids) -> (embeddings [B,S,D], curvature [B,S,...])
```

| `curvature_type` | Output shape | Description |
|---|---|---|
| `'metric'` | `(B, S, D, D)` | Full symmetric positive-definite metric tensor |
| `'riemann'` | `(B, S, D, D)` | Antisymmetric Riemann-like curvature tensor |
| `'ricci'` | `(B, S, D)` | Diagonal Ricci curvature (one value per dimension) |
| `'scalar'` | `(B, S, 1)` | Single scalar curvature per position |

### ConnectionLayer

Computes a gauge connection matrix from embeddings and curvature.

```python
ConnectionLayer(
    hidden_dim,                       # representation dimension
    connection_type='yang_mills',     # 'yang_mills' | 'levi_civita' | 'affine'
    num_generators=8,                 # Lie algebra generators (yang_mills only)
    use_metric=True,                  # learn a metric tensor
    antisymmetric=True,               # enforce Lie algebra antisymmetry
)
# call([embeddings, curvature]) -> connection [B, S, D, D]
```

| `connection_type` | Description |
|---|---|
| `'yang_mills'` | Non-abelian gauge connection: `A = sum_g c_g T_g` where `T_g` are learnable antisymmetric generators |
| `'levi_civita'` | Metric-compatible, torsion-free (symmetrised affine) |
| `'affine'` | General affine connection (optionally antisymmetrised) |

### ParallelTransportLayer

Transports vectors along the sequence using the connection.

```python
ParallelTransportLayer(
    transport_dim,                    # vector dimension (must match connection)
    num_steps=10,                     # Euler integration steps
    transport_method='iterative',     # 'direct' | 'iterative' | 'path_ordered'
    step_size=0.1,                    # integration step size
)
# call([vectors, connection]) -> transported [B, S, D]
```

| `transport_method` | Cost | Description |
|---|---|---|
| `'direct'` | 1 matmul | Single Euler step `V' = V - h * A @ V` |
| `'iterative'` | `num_steps` matmuls | Multi-step Euler integration |
| `'path_ordered'` | `num_steps` matmuls + exp approx | Second-order exponential: `exp(A) ~ I + A + A^2/2` |

### HolonomyLayer

Computes gauge-invariant holonomy features at every sequence position.

```python
HolonomyLayer(
    hidden_dim,                       # output projection dimension
    loop_sizes=[2, 4, 8],             # offsets for commutator computation
    num_loops=4,                      # orientations per loop size
    use_trace=True,                   # True: Tr([A,B]^2), False: Frobenius norm
)
# call([embeddings, connection]) -> features [B, S, hidden_dim]
```

Internally computes the commutator `[Gamma_s, Gamma_{s+offset}]` at each
position for each `(loop_size, orientation)` pair, extracts a scalar feature
(trace of squared commutator or Frobenius norm), and projects the stacked
features to `hidden_dim`.  Fully vectorised --- no Python loops over the
sequence dimension.

### GaugeInvariantAttention

Multi-head attention whose scores incorporate geometric information.

```python
GaugeInvariantAttention(
    hidden_dim,                       # model dimension
    num_heads=8,                      # attention heads
    attention_metric='hybrid',        # 'holonomy' | 'geodesic' | 'curvature' | 'hybrid'
    use_curvature_gating=True,        # gate scores by local curvature
    use_parallel_transport=True,      # transport values before aggregation
    dropout_rate=0.0,
)
# call([embeddings, curvature, connection]) -> output [B, S, hidden_dim]
```

| `attention_metric` | What it adds to standard QK scores |
|---|---|
| `'holonomy'` | Penalises pairs with large connection difference (high holonomy) |
| `'geodesic'` | Penalises pairs separated by high-curvature regions |
| `'curvature'` | Boosts pairs with similar local curvature (cosine similarity) |
| `'hybrid'` | Weighted combination of all available metrics |

### ManifoldStressLayer

Measures geometric inconsistency to flag anomalous inputs.

```python
ManifoldStressLayer(
    hidden_dim,
    stress_types=['curvature', 'connection', 'combined'],
    stress_threshold=0.5,             # initial anomaly threshold
    use_learnable_baseline=True,      # learn expected curvature/connection values
    return_components=False,          # True: return per-type stress
)
# call([embeddings, curvature, connection]) -> (stress [B,S,1], anomaly_mask [B,S,1])
```

| `stress_type` | What it measures |
|---|---|
| `'curvature'` | `\|\|R - R_baseline\|\|` -- deviation from learned curvature baseline |
| `'connection'` | Local variation `\|\|Gamma_{s+1} - Gamma_s\|\|` plus magnitude |
| `'holonomy'` | Commutator Frobenius norm (curvature proxy) |
| `'metric'` | Variation in embedding finite-difference norms |
| `'combined'` | Mean of all available stress types |

### FieldNormalization

Layer normalisation that scales by inverse curvature magnitude: high-curvature
regions receive less aggressive normalisation.

```python
FieldNormalization(
    epsilon=1e-6,
    use_curvature_scaling=True,       # scale by 1/(1 + alpha * ||curv||)
    center=True,
    scale=True,
)
# call(embeddings) or call([embeddings, curvature]) -> normalised [B, S, D]
```

### HolonomicTransformerLayer

Drop-in transformer block that orchestrates all field components.

```python
HolonomicTransformerLayer(
    hidden_dim,
    num_heads=8,
    ffn_dim=None,                     # defaults to 4 * hidden_dim
    curvature_type='ricci',
    connection_type='yang_mills',
    attention_metric='hybrid',
    use_holonomy_features=True,       # add holonomy features to the residual
    use_anomaly_detection=True,       # compute manifold stress
    dropout_rate=0.1,
    normalization_type='field_norm',  # 'field_norm' | 'layer_norm' | 'rms_norm'
    activation='gelu',
)
# call(x)           -> output [B,S,D]                     (anomaly off)
# call(x)           -> (output [B,S,D], stress [B,S,1])   (anomaly on)
```

**Internal pipeline:**

```
x
|-> curvature = tanh(Dense(x)) * 0.1
|-> connection = ConnectionLayer([x, curvature])
|-> attn_out = GaugeInvariantAttention([Norm(x), curvature, connection])
|-> x = ParallelTransport([x, connection]) + Dropout(attn_out)
|-> x += 0.1 * HolonomyProj(HolonomyLayer([x, connection]))   (optional)
|-> x += Dropout(FFN(Norm(x)))
|-> stress = ManifoldStressLayer([x, curvature, connection])    (optional)
```

---

## Architecture Patterns

### Full holonomic transformer

```python
class HolonomicTransformer(keras.Model):
    def __init__(self, vocab_size, num_layers=6, hidden_dim=256,
                 num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embedding = FieldEmbedding(
            vocab_size=vocab_size, embed_dim=hidden_dim,
            curvature_type='ricci',
        )
        self.blocks = [
            HolonomicTransformerLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                use_holonomy_features=True, use_anomaly_detection=True,
            )
            for _ in range(num_layers)
        ]
        self.head = keras.layers.Dense(vocab_size)

    def call(self, tokens, training=None):
        x, _ = self.embedding(tokens, training=training)
        total_stress = 0.0
        for block in self.blocks:
            x, stress = block(x, training=training)
            total_stress = total_stress + stress
        return self.head(x), total_stress / len(self.blocks)
```

### Retrofitting an existing transformer

Swap standard attention for `GaugeInvariantAttention` without touching
the rest of the architecture:

```python
class GeometricAttentionBlock(keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = GaugeInvariantAttention(
            hidden_dim=hidden_dim, num_heads=num_heads,
            attention_metric='curvature',
        )
        self.curvature_proj = keras.layers.Dense(
            hidden_dim, activation='tanh',
        )
        self.ffn1 = keras.layers.Dense(hidden_dim * 4, activation='gelu')
        self.ffn2 = keras.layers.Dense(hidden_dim)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()

    def call(self, x, training=None):
        curvature = self.curvature_proj(x) * 0.1
        attn = self.attention(
            [self.norm1(x), curvature], training=training,
        )
        x = x + attn
        x = x + self.ffn2(self.ffn1(self.norm2(x)))
        return x
```

### Anomaly-aware training

Down-weight high-stress samples during training:

```python
class AnomalyAwareModel(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, stress = self(x, training=True)
            task_loss = self.compute_loss(y=y, y_pred=logits)
            sample_weights = 1.0 / (1.0 + stress)
            loss = keras.ops.mean(
                task_loss * keras.ops.squeeze(sample_weights)
            ) + sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            'loss': loss,
            'anomaly_mean': keras.ops.mean(stress),
        }
```

---

## Performance Considerations

### Computational cost

| Layer | Time complexity | Dominant operation |
|---|---|---|
| FieldEmbedding | O(n d) | Embedding lookup + curvature projection |
| ConnectionLayer | O(n d^2) | Two-layer MLP + generator combination |
| ParallelTransport | O(n k d^2) | k Euler steps, each a batched matmul |
| HolonomyLayer | O(n L d^2) | L commutator matmuls (fully vectorised) |
| GaugeInvariantAttention | O(n^2 d / h) | Standard QKV attention + geometric corrections |
| ManifoldStressLayer | O(n d^2) | Commutator + norm computation |

*n* = sequence length, *d* = hidden dimension, *h* = number of heads,
*k* = transport steps, *L* = `len(loop_sizes) * num_loops`.

### Memory overhead

The field representation adds curvature `O(n d)` and connection `O(n d^2)`
tensors on top of the standard `O(n d)` embeddings.  To reduce memory:

- Use `curvature_type='scalar'` (adds only `O(n)`)
- Reduce `num_generators` in the connection layer
- Use `transport_method='direct'` (single step, no intermediate tensors)
- Disable `use_holonomy_features` and `use_anomaly_detection` at inference time

---

## Factory API

### `create_field_layer(layer_type, name=None, **kwargs)`

Create a field layer with validated parameters and sensible defaults.

`layer_type` is one of: `'field_embedding'`, `'connection'`,
`'parallel_transport'`, `'holonomy'`, `'gauge_attention'`,
`'manifold_stress'`, `'holonomic_transformer'`, `'field_norm'`.

### `create_field_layer_from_config(config)`

Create a field layer from a dictionary containing a `'type'` key and
layer-specific parameters.

### `validate_field_config(layer_type, **kwargs)`

Check that required parameters are present and `layer_type` is valid.
Raises `ValueError` on failure.

### `get_field_layer_info()`

Returns a dictionary mapping each layer type to its class, required
parameters, default parameters, and a short description.
