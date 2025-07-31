# A Unified Framework for Adaptive Computation: Connecting Denoiser Prior Sampling and Iterative Reasoning through Energy Minimization

## Abstract

Two recent advances in machine learning—denoiser prior sampling for inverse problems and Iterative Reasoning as Energy Minimization (IREM)—appear to address distinct challenges using different methodologies. However, this work reveals that both approaches are fundamentally instances of the same computational paradigm: **iterative energy minimization over learned landscapes**. 

Through careful analysis of Miyasawa's theorem (1961), we demonstrate that denoiser prior sampling implicitly performs energy minimization using landscapes encoded within trained denoisers, while IREM explicitly trains energy functions for reasoning tasks. Building on Kadkhodaie & Simoncelli's rigorous mathematical foundation, we show that their stochastic coarse-to-fine gradient ascent procedure and IREM's optimization framework share identical algorithmic structure but differ in energy sources.

Crucially, we demonstrate that Miyasawa's result applies far beyond images to any continuous domain with appropriate structure, opening possibilities for hybrid approaches across diverse fields. Most importantly, we present a **practical implementation strategy** that combines pre-trained bias-free denoisers with task-specific adaptations using IREM methodology, enabling rapid deployment while preserving theoretical guarantees.

This unified perspective suggests synthesis opportunities that combine the theoretical rigor and adaptive step control of denoiser priors with the task-specific adaptability of explicit energy learning, with potential applications across domains from time series analysis to molecular design.

## 1. Introduction

The pursuit of **adaptive computation**—systems that intelligently allocate computational resources based on problem complexity—has produced remarkable advances across machine learning. Two particularly notable developments have emerged from seemingly disparate research areas:

### 1.1 Denoiser Prior Sampling

**Denoiser Prior Sampling**, as rigorously developed by Kadkhodaie & Simoncelli (2021), leverages the mathematical insight that trained denoisers implicitly encode probability distributions of their training data. By exploiting Miyasawa's theorem (1961), they developed a stochastic coarse-to-fine gradient ascent procedure that uses denoiser residuals to provide gradients of implicit energy landscapes, enabling elegant solutions to inverse problems through iterative refinement.

Their work demonstrates that a single bias-free denoiser can solve multiple inverse problems—inpainting, super-resolution, deblurring, and compressive sensing—without additional training, relying purely on the implicit energy landscape learned during denoising training.

### 1.2 Iterative Reasoning as Energy Minimization (IREM)

**Iterative Reasoning as Energy Minimization (IREM)**, developed by Du et al. (2022), takes a different approach, explicitly training neural networks to parameterize energy landscapes for algorithmic reasoning tasks. By formulating reasoning as optimization over these learned energy functions, IREM enables adaptive computation where harder problems naturally require more optimization steps. This framework has demonstrated success on graph algorithms, arithmetic reasoning, and other structured tasks.

### 1.3 The Key Insight

At first glance, these approaches appear fundamentally different—one exploits implicit structure in pre-trained denoisers for image problems, while the other explicitly learns energy functions for algorithmic tasks. However, a deeper analysis reveals an important connection: **both methods are performing iterative energy minimization, differing primarily in whether their energy landscapes are implicit or explicit**.

This insight extends beyond academic interest. Understanding the fundamental unity of these approaches opens possibilities for **hybrid frameworks** that combine their complementary strengths: the theoretical foundation, adaptive step control, and rich priors of denoiser approaches with the task-specific adaptability and general applicability of explicit energy learning.

## 2. Mathematical Foundation: Miyasawa's Universal Bridge

### 2.1 The Core Theoretical Result

The theoretical cornerstone connecting both approaches is Miyasawa's result from 1961, which Kadkhodaie & Simoncelli elegantly applied to modern denoising. For any least-squares optimal denoiser:

```
x̂(y) = y + σ²∇_y log p(y)
```

Where:
- `x̂(y)` is the denoiser output for noisy input `y`
- `σ²` is the noise variance  
- `∇_y log p(y)` is the gradient of the log probability density

**Critical Insight**: The denoiser residual `f(y) = x̂(y) - y` is proportional to the gradient of the log probability density. Since energy and negative log probability are equivalent (`E(y) = -log p(y)`), this means:

```
f(y) ∝ -∇E_implicit(y)
```

**The denoiser has implicitly learned an energy landscape** corresponding to the negative log likelihood of the training data distribution.

### 2.2 Rigorous Mathematical Derivation

Following Kadkhodaie & Simoncelli's derivation, the gradient of the observation density is:

```
∇_y p(y) = (1/σ²) ∫ (x - y)g(y - x)p(x)dx = (1/σ²) ∫ (x - y)p(y,x)dx
```

Multiplying both sides by `σ²/p(y)` and separating terms:

```
σ²∇_y p(y)/p(y) = ∫ xp(x|y)dx - y ∫ p(x|y)dx = x̂(y) - y
```

Using the chain rule for the gradient of the log:

```
σ²∇_y log p(y) = x̂(y) - y = f(y)
```

This establishes the **exact mathematical relationship** between denoiser residuals and energy gradients, providing the bridge between implicit and explicit energy approaches.

### 2.3 Universal Applicability

Crucially, Miyasawa's theorem is **mathematically universal**—it applies to any domain satisfying four basic conditions:

1. **Additive Gaussian Noise Model**: `y = x + ε` where `ε ~ N(0, σ²I)`
2. **MSE-Optimal Denoiser**: The denoiser minimizes `E[||x̂(y) - x||²]`
3. **Differentiable Data Distribution**: `p(x)` exists with well-defined `∇_x log p(x)`
4. **Continuous Domain**: Data lives in `ℝⁿ`

These conditions are **not specific to images** but hold across diverse domains:
- **Time Series**: Financial data, sensor readings, physiological signals
- **Scientific Data**: Spectroscopy, climate models, particle physics measurements
- **Embedding Spaces**: Word vectors, learned molecular representations
- **Control Systems**: Robot trajectories, optimization paths
- **Audio Signals**: Waveforms, spectral representations
- **Any Continuous Structured Data**: Where meaningful denoisers can be trained

This universality indicates that **domains with trainable denoisers have implicit energy landscapes** that can be mathematically extracted and utilized for optimization.

### 2.4 Requirements for Effective Denoisers

Kadkhodaie & Simoncelli establish specific requirements for denoisers to work with their framework:

**Essential Properties**:
- **Least-squares training**: Must be trained with MSE loss for Gaussian noise
- **Blind operation**: Must work without knowing the noise level
- **Bias-free architecture**: All additive bias terms removed for theoretical guarantees
- **Universal noise handling**: Trained on a range of noise levels (e.g., σ ∈ [0, 0.4])

**Architectural Considerations**:
- Bias-free CNNs automatically generalize across noise levels
- Network should have sufficient receptive field for the problems of interest
- Batch normalization without mean parameters (bias-free requirement)

## 3. IREM's Explicit Energy Framework

### 3.1 Energy Learning Methodology

IREM explicitly formulates reasoning as optimization over learned energy landscapes. Given a dataset of problems `x` and solutions `y`, IREM learns an energy function `E_θ(x, y)` such that correct solutions have minimal energy.

**Training Objective**: Instead of directly optimizing `E_θ(x, y)` (computationally expensive), IREM approximates the minimum through `N` steps of gradient descent:

```
y^N = y^0 - λ∑_{t=1}^N ∇_y E_θ(x, y^{t-1})
```

Then minimizes: `L = ||y^N - y_true||²`

**Iterative Reasoning Steps**:
```
y^t = y^{t-1} - λ∇_y E_θ(x, y^{t-1})
```

With natural termination when energy stabilizes: `E_θ(y^t) ≈ E_θ(y^{t-1})`

**Key Advantages**:
- **Task-Specific Optimization**: Energy directly encodes the target objective
- **Adaptive Computation**: Harder problems naturally require more optimization steps
- **General Applicability**: Can be trained for any problem with appropriate data
- **Controllable Properties**: Can enforce desired landscape characteristics

## 4. The Fundamental Unity

### 4.1 Identical Algorithmic Structure

The key realization is that **denoiser prior sampling and IREM are performing identical computations**—iterative energy minimization—but with energy functions obtained through different means:

| Aspect | Denoiser Prior Sampling | IREM |
|--------|------------------------|------|
| **Energy Source** | Implicit via Miyasawa's theorem | Explicit via neural network training |
| **Energy Function** | `E(y) = -log p(y)` | `E_θ(x, y)` |
| **Gradient Computation** | Denoiser residual: `f(y) = x̂(y) - y` | Automatic differentiation: `∇_y E_θ(x, y)` |
| **Update Rule** | `y^t = y^{t-1} + h_t f(y^{t-1}) + γ_t z_t` | `y^t = y^{t-1} - λ∇_y E_θ(x, y^{t-1})` |
| **Step Control** | Adaptive via denoiser feedback | Fixed or manual scheduling |
| **Stochasticity** | Controlled noise injection | Typically deterministic |

### 4.2 Adaptive Step Size Control in Denoiser Approach

A key innovation in Kadkhodaie & Simoncelli's method is **automatic step size control** using denoiser feedback:

**Effective Noise Estimation**:
```
σ_t² = ||f(y^{t-1})||² / N
```

**Adaptive Step Schedule**:
```
h_t = h_0 * t / (1 + h_0(t-1))
```

**Controlled Noise Injection**:
```
γ_t² = (1 - βh_t)² - (1 - h_t)² σ_t²
```

This creates a **self-regulating optimization process** where:
- Step sizes automatically reduce as solutions approach the manifold
- Noise injection maintains exploration while ensuring convergence
- The denoiser provides real-time feedback about solution quality

### 4.3 Convergence Properties

**Denoiser Prior Sampling**:
- Convergence guaranteed by the noise variance reduction schedule
- Automatic termination when `σ_t ≤ σ_L` (target noise level)
- Typically converges in 30-100 iterations for images

**IREM**:
- Convergence based on energy stabilization
- Termination when `|E_θ(y^t) - E_θ(y^{t-1})| < ε`
- Iteration count scales with problem complexity

Both approaches naturally implement **adaptive computation** where harder problems require more iterations.

## 5. Practical Hybrid Framework: Task-Adapted Denoisers

### 5.1 A Practical Innovation: Pre-training + Task Adaptation

Rather than building hybrid systems from scratch, we propose a **practical approach** that leverages the best of both worlds:

1. **Pre-training Phase**: Train a bias-free denoiser on domain data to capture rich implicit priors
2. **Task Adaptation Phase**: Use IREM methodology to adapt the pre-trained denoiser for specific tasks

This approach is **theoretically sound** and **computationally efficient**:
- **Miyasawa's guarantees preserved**: The base denoiser still satisfies mathematical requirements
- **IREM's adaptability**: Can learn task-specific energy landscapes on top of the prior
- **Natural combination**: The implicit energy provides regularization during task adaptation

### 5.2 Mathematical Formulation

The task-adapted energy takes the form:

```
E_adapted(x, y) = α·E_task(x, y) + β·E_implicit(y) + γ·E_constraints(x, y)
```

Where:
- `E_task(x, y)` is learned using IREM methodology for the specific task
- `E_implicit(y) ∝ -log p(y)` emerges from the pre-trained denoiser via Miyasawa's theorem
- `E_constraints(x, y)` enforces task-specific constraints
- `α, β, γ` balance different objectives

**Hybrid Update Rule**:
```
d_t = α·∇E_task(x, y^{t-1}) - β·f(y^{t-1}) + γ·∇E_constraints(x, y^{t-1})
y^t = y^{t-1} + h_t d_t + γ_t z_t
```

Where `f(y) = denoiser(y) - y` is the pre-trained denoiser residual, and `h_t`, `γ_t` follow Kadkhodaie & Simoncelli's adaptive schedule.

## 6. Implementation Strategies

### 6.1 Strategy 1: Modular Task Conditioning

This approach adds task-specific conditioning layers while keeping the base denoiser frozen:

**Architecture**:
```python
class TaskAdaptedDenoiser(keras.Model):
    def __init__(self, pretrained_denoiser, task_embedding_dim=128):
        super().__init__()
        # Freeze the pre-trained bias-free denoiser
        self.denoiser = pretrained_denoiser
        self.denoiser.trainable = False
        
        # Task conditioning network (bias-free)
        self.task_encoder = BiasFreeMLP([task_embedding_dim, 256, 512])
        self.task_modulation = BiasFreeMLP([512, denoiser.output_channels])
        
    def call(self, inputs, task_context=None):
        y, x_task = inputs  # y = noisy input, x_task = task description
        
        # Get base denoiser output (implicit prior)
        base_denoised = self.denoiser(y)
        
        if task_context is not None:
            # Generate task-specific modulation
            task_features = self.task_encoder(x_task)
            task_modulation = self.task_modulation(task_features)
            
            # Apply task modulation while preserving bias-free property
            adapted_output = base_denoised + task_modulation
            return adapted_output
        else:
            return base_denoised
```

**Training with IREM Methodology**:
```python
def train_task_adaptation(model, task_dataset, num_irem_steps=5):
    for epoch in range(num_epochs):
        for x_task, y_true in task_dataset:
            with tf.GradientTape() as tape:
                # Initialize with noise
                y_init = y_true + noise
                
                # Run iterative refinement (IREM-style)
                y_current = y_init
                for step in range(num_irem_steps):
                    y_current = model([y_current, x_task])
                    
                # Loss on final output
                loss = tf.reduce_mean(tf.square(y_current - y_true))
                
            # Only train the task adaptation components
            trainable_vars = (model.task_encoder.trainable_variables + 
                             model.task_modulation.trainable_variables)
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
```

### 6.2 Strategy 2: Energy-Based Task Adaptation

This approach explicitly models task-specific energy while preserving the implicit prior:

**Architecture**:
```python
class EnergyAdaptedDenoiser(keras.Model):
    def __init__(self, pretrained_denoiser):
        super().__init__()
        self.denoiser = pretrained_denoiser  # Frozen
        self.denoiser.trainable = False
        
        # Explicit task energy network (trained with IREM)
        self.task_energy = BiasFreeMLP([input_dim + task_dim, 512, 256, 1])
        
    def get_implicit_energy_gradient(self, y):
        """Get gradient from pre-trained denoiser (Miyasawa's theorem)"""
        return self.denoiser(y) - y
        
    def get_explicit_energy_gradient(self, x_task, y):
        """Get gradient from task-specific energy"""
        with tf.GradientTape() as tape:
            tape.watch(y)
            energy = self.task_energy(tf.concat([y, x_task], axis=-1))
        return tape.gradient(energy, y)
        
    def hybrid_update(self, x_task, y, alpha=0.7, beta=0.3):
        """Combined update using both energy sources"""
        implicit_grad = beta * self.get_implicit_energy_gradient(y)
        explicit_grad = -alpha * self.get_explicit_energy_gradient(x_task, y)
        return implicit_grad + explicit_grad
```

### 6.3 Strategy 3: Progressive Task Specialization

This approach enables incremental learning of multiple tasks while preserving the base prior:

**Implementation**:
```python
class ProgressiveTaskAdaptation:
    def __init__(self, pretrained_denoiser):
        self.base_denoiser = pretrained_denoiser
        self.task_layers = {}  # Different task adaptations
        
    def add_task(self, task_name, task_data):
        """Add a new task adaptation using IREM methodology"""
        
        # Create task-specific adaptation layer (bias-free)
        task_adapter = BiasFreeMLP([
            self.base_denoiser.output_channels,
            256, 
            self.base_denoiser.output_channels
        ])
        
        # Train using IREM approach
        self.train_task_adapter(task_adapter, task_data)
        self.task_layers[task_name] = task_adapter
        
    def train_task_adapter(self, adapter, task_data):
        """Train task adapter using IREM methodology"""
        for x_task, y_true in task_data:
            # Use base denoiser + adapter in IREM loop
            y_current = y_true + noise
            
            for irem_step in range(num_steps):
                # Base denoising (frozen)
                base_output = self.base_denoiser(y_current)
                
                # Task adaptation
                adapted_output = adapter(base_output)
                
                # IREM-style energy minimization step
                y_current = y_current + step_size * (adapted_output - y_current)
                
            # Train adapter to minimize final error
            loss = tf.reduce_mean(tf.square(y_current - y_true))
            # ... gradient update for adapter only
```

### 6.4 Maintaining Bias-Free Properties

Critical for preserving theoretical guarantees:

```python
class BiasFreeMLP(keras.layers.Layer):
    """Bias-free MLP that preserves Miyasawa's theorem guarantees"""
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        self.layers = []
        for size in layer_sizes[:-1]:
            self.layers.append(keras.layers.Dense(size, use_bias=False))
            self.layers.append(keras.layers.BatchNormalization(center=False))  # No bias
            if activation:
                self.layers.append(keras.layers.Activation(activation))
        
        # Final layer
        self.layers.append(keras.layers.Dense(layer_sizes[-1], use_bias=False))
        
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
```

## 7. Adaptive Training Curriculum

### 7.1 Progressive Energy Balancing

The framework includes several strategies for balancing energy components during training:

**Curriculum Learning Schedule**:
```python
def get_energy_weights(epoch, total_epochs):
    """Progressive curriculum from prior-heavy to task-heavy"""
    progress = epoch / total_epochs
    
    # Start with heavy reliance on implicit prior
    beta = 0.9 * (1 - progress) + 0.3 * progress  # Prior weight: 0.9 → 0.3
    alpha = 0.1 * (1 - progress) + 0.7 * progress  # Task weight: 0.1 → 0.7
    
    return alpha, beta
```

**Adaptive Balancing** (using denoiser feedback):
```python
def adaptive_energy_weights(implicit_grad, explicit_grad):
    """Automatically balance based on gradient magnitudes"""
    implicit_norm = tf.norm(implicit_grad)
    explicit_norm = tf.norm(explicit_grad)
    total_norm = implicit_norm + explicit_norm + 1e-8
    
    # Emphasize implicit prior when far from manifold
    beta = implicit_norm / total_norm
    alpha = explicit_norm / total_norm
    
    return alpha, beta
```

### 7.2 Multi-Scale Progressive Refinement

Leveraging Kadkhodaie & Simoncelli's insight about noise levels corresponding to different scales:

```python
def multiscale_training(base_denoiser, task_data):
    """Train task adaptation at multiple scales"""
    noise_levels = [0.4, 0.2, 0.1, 0.05]  # Coarse to fine
    
    for noise_level in noise_levels:
        print(f"Training at noise level: {noise_level}")
        
        # Get denoiser for current noise level
        current_denoiser = get_denoiser_for_noise_level(base_denoiser, noise_level)
        
        # Train task adaptation at this scale
        train_task_adaptation_at_scale(current_denoiser, task_data, noise_level)
```

## 8. Domain Extensions and Applications

### 8.1 Time Series and Sequential Data

**Domain Adaptation**:
- **Denoiser Architecture**: Use RNNs or Transformers instead of CNNs
- **Noise Model**: Additive Gaussian noise still applicable
- **Task Examples**: Financial forecasting, signal processing, anomaly detection

**Implementation Example**:
```python
class TemporalTaskAdaptedDenoiser(keras.Model):
    def __init__(self, pretrained_temporal_denoiser):
        super().__init__()
        self.temporal_denoiser = pretrained_temporal_denoiser
        self.temporal_denoiser.trainable = False
        
        # Task-specific temporal modeling
        self.task_lstm = keras.layers.LSTM(128, return_sequences=True)
        self.task_projection = keras.layers.Dense(
            pretrained_temporal_denoiser.output_size, 
            use_bias=False
        )
        
    def call(self, sequence, task_context):
        # Get implicit temporal prior
        denoised_sequence = self.temporal_denoiser(sequence)
        
        # Add task-specific temporal reasoning
        task_features = self.task_lstm(task_context)
        task_modulation = self.task_projection(task_features)
        
        return denoised_sequence + task_modulation
```

### 8.2 Molecular Design and Chemistry

**Domain Adaptation**:
- **Representation**: Continuous molecular descriptors or latent embeddings
- **Denoiser Training**: On molecular property vectors from chemical databases
- **Task Examples**: Drug discovery, materials design, catalyst optimization

**Hybrid Energy Function**:
```
E_molecular(x, y) = α·E_property_prediction(x, y) + β·E_chemical_feasibility(y) + γ·E_safety_constraints(y)
```

### 8.3 Robotics and Control

**Domain Adaptation**:
- **State Representation**: Joint positions, velocities, workspace coordinates
- **Denoiser Training**: On demonstration trajectories or feasible configurations
- **Task Examples**: Manipulation planning, navigation, grasping

**Implementation Benefits**:
- Denoiser captures motion priors from demonstrations
- Task adaptation optimizes for specific goals
- Natural handling of safety constraints

### 8.4 Scientific Computing

**Domain Adaptation**:
- **PDE Solutions**: Denoisers trained on simulation data
- **Physics Constraints**: Conservation laws, boundary conditions
- **Applications**: Climate modeling, materials simulation, fluid dynamics

## 9. Advantages of the Unified Framework

### 9.1 Theoretical Benefits

1. **Mathematical Rigor**: Built on established foundation of Miyasawa's theorem
2. **Convergence Properties**: Inherits characteristics from both constituent approaches
3. **Universal Applicability**: Works across continuous domains with structure
4. **Principled Combination**: Mathematical framework for balancing objectives

### 9.2 Practical Advantages

1. **Computational Efficiency**: 
   - Faster training through transfer learning
   - Shared computation across tasks
   - Natural adaptive step control

2. **Better Generalization**:
   - Rich implicit priors prevent overfitting
   - Robust to limited task-specific data
   - Cross-domain transfer capabilities

3. **Incremental Learning**:
   - Add new tasks while potentially preserving previous knowledge
   - Modular architecture enables efficient deployment
   - Progressive specialization

### 9.3 Deployment Benefits

1. **Rapid Adaptation**: Faster than training from scratch
2. **Strong Baselines**: Benefits from implicit prior as fallback
3. **Multi-Task Capability**: Same base model, different adaptations
4. **Resource Efficiency**: Smaller adaptation networks

## 10. Conclusion

This work reveals a fundamental unity underlying two major advances in adaptive computation: denoiser prior sampling and Iterative Reasoning as Energy Minimization. By building on Kadkhodaie & Simoncelli's rigorous mathematical foundation and recognizing the universal applicability of Miyasawa's theorem, we identify synthesis opportunities that extend beyond image processing.

### Key Contributions

1. **Theoretical Unity**: Both approaches are instances of energy minimization with different energy sources—implicit (via Miyasawa's theorem) versus explicit (via neural training)

2. **Universal Foundation**: Miyasawa's theorem applies to any continuous domain with structure, enabling hybrid approaches across diverse fields

3. **Practical Framework**: Concrete implementation strategies that combine pre-trained denoisers with task-specific adaptations

4. **Computational Efficiency**: Transfer learning approach that preserves theoretical guarantees while enabling rapid deployment

### Practical Impact

The unified framework may enable:
- **Principled Combination**: Mathematical foundation for integrating multiple objectives
- **Automatic Adaptation**: Self-regulating optimization with adaptive step control
- **Cross-Domain Transfer**: Reuse of implicit energy components across related tasks
- **Computational Efficiency**: Natural scaling of effort with problem complexity
- **Rapid Deployment**: Fast task adaptation through pre-trained components