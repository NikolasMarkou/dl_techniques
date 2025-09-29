"""
Implements a differentiable, n-dimensional memory with probabilistic addressing.

This layer provides a structured, spatially organized memory system that is fully
differentiable, enabling end-to-end training within modern deep learning
frameworks. It bridges the gap between the topology-preserving properties of
classical Self-Organizing Maps (SOMs) and the requirements of gradient-based
optimization, offering a powerful mechanism for learning structured
representations.

Architectural and Mathematical Underpinnings:

The core architecture consists of two main components: a probabilistic
addressing mechanism and an n-dimensional grid of learnable latent vectors
(the memory lattice). The goal is to perform a "soft lookup" where an input
vector is mapped not to a single memory location, but to a weighted average
of all memory locations, with weights determined by learned similarity.

1.  **Factorized, Probabilistic Addressing**: For an n-dimensional grid of
    shape (d₁, d₂, ..., dₙ), the addressing is factorized across dimensions.
    An input vector `x` is fed into `n` independent projection networks,
    `f₁, f₂, ..., fₙ`. Each network `fᵢ` outputs a logit vector of size `dᵢ`.
    These logits are then transformed into a probability distribution `Pᵢ`
    over the `i`-th dimension using a temperature-controlled softmax:

        Pᵢ = softmax(fᵢ(x) / T)

    The temperature parameter `T` controls the sharpness of the distribution.
    A low temperature results in a sparse, focused distribution (akin to a
    hard lookup), while a high temperature yields a smooth, diffuse
    distribution. This continuous, differentiable transformation is the key
    departure from the discrete Best Matching Unit (BMU) search in a
    traditional SOM.

2.  **Joint Probability via Outer Product**: The individual dimensional
    probabilities `P₁, P₂, ..., Pₙ` are combined to form a joint probability
    distribution over the entire n-dimensional grid. This is achieved through
    an outer product (tensor product):

        P_joint = P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ

    The resulting tensor `P_joint` has the shape (d₁, d₂, ..., dₙ), where each
    element `P_joint[i₁, i₂, ..., iₙ]` represents the probability of addressing
    the memory cell at that specific coordinate. This compositional approach
    is inspired by Product Key Memories, allowing for a vast addressable
    memory space with a modest number of parameters.

3.  **Differentiable Soft Lookup**: The final output is the expectation of the
    latent vectors in the memory grid `G`, weighted by the joint probability
    distribution. This is a weighted sum over all grid positions:

        Output = Σ_{i₁, i₂, ..., iₙ} P_joint[i₁, i₂, ..., iₙ] * G[i₁, i₂, ..., iₙ]

    This operation is fully differentiable with respect to both the grid
    vectors `G` and the parameters of the projection networks `fᵢ`. During
    training, gradients flow through this weighted sum, simultaneously
    updating the memory content (`G`) and the addressing logic (`fᵢ`) to
    minimize the task-specific loss. This process encourages semantically
    similar inputs to activate neighboring regions of the grid, leading to an
    emergent, self-organized topological representation of the input space.

This layer serves as a modern reinterpretation of classic unsupervised
learning algorithms, reformulated for deep learning. By replacing discrete,
non-differentiable operations with their probabilistic, soft counterparts, it
integrates the powerful concept of structured, topologically-aware memory
directly into end-to-end trainable models.

References:
    - Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*.
      (Conceptual foundation for topological data representation).
    - Graves, A., et al. (2014). Neural Turing Machines. *arXiv preprint*.
      (Pioneered differentiable addressing for external memory).
    - Lample, G., et al. (2019). Large Memory Layers with Product Keys. *NeurIPS*.
      (Inspiration for compositional, factorized addressing).
    - Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
      *arXiv preprint*. (Popularized the use of temperature in softmax).
"""

import keras
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from ..initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NeuroGrid(keras.layers.Layer):
    """
    NeuroGrid: Differentiable N-Dimensional Memory Lattice with Probabilistic Addressing for Transformers.

    This layer implements a differentiable n-dimensional lookup table that combines
    probabilistic addressing with self-organizing map-like behavior, designed to work
    seamlessly with both traditional dense layers and modern transformer architectures.
    The input is mapped through N separate projection networks (one per grid dimension),
    each producing a probability distribution over that dimension via temperature-controlled
    softmax. These distributions are combined via outer product to create joint
    probabilities for soft addressing into an n-dimensional grid of learnable latent vectors.

    **Transformer Integration**: The layer automatically handles both 2D (batch_size, input_dim)
    and 3D (batch_size, seq_len, embed_dim) inputs, making it compatible with transformer
    architectures. For 3D inputs, each token position is processed independently through
    the same shared grid, preserving sequence structure while enabling token-level
    structured memory access.

    **Intent**: Create a structured memory mechanism that can learn to organize
    representations in an n-dimensional grid topology while providing smooth,
    differentiable lookup operations based on learned similarity metrics. The layer
    exhibits emergent clustering behavior similar to Self-Organizing Maps but remains
    fully differentiable for end-to-end training in both traditional and transformer-based
    architectures.

    **Architecture for 2D Inputs**:
    ```
    Input(shape=[batch, input_dim])
           ↓
    Dense₁(d1) + Softmax(T) → P₁(shape=[batch, d1])
    Dense₂(d2) + Softmax(T) → P₂(shape=[batch, d2])
           ...
    DenseN(dn) + Softmax(T) → PN(shape=[batch, dn])
           ↓
    Joint Probability: P₁ ⊗ P₂ ⊗ ... ⊗ PN → (batch, d1, d2, ..., dn)
           ↓
    Soft Grid Lookup: Σ(joint_prob * grid_weights) → (batch, latent_dim)
    ```

    **Architecture for 3D Inputs (Transformer Mode)**:
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    Reshape → (batch * seq_len, embed_dim)  [flatten for processing]
           ↓
    Dense₁(d1) + Softmax(T) → P₁(shape=[batch * seq_len, d1])
    Dense₂(d2) + Softmax(T) → P₂(shape=[batch * seq_len, d2])
           ...
    DenseN(dn) + Softmax(T) → PN(shape=[batch * seq_len, dn])
           ↓
    Joint Probability: P₁ ⊗ P₂ ⊗ ... ⊗ PN → (batch * seq_len, d1, d2, ..., dn)
           ↓
    Soft Grid Lookup: Σ(joint_prob * grid_weights) → (batch * seq_len, latent_dim)
           ↓
    Reshape → (batch, seq_len, latent_dim)  [restore sequence structure]
    ```

    **Mathematical Operations**:
    1. **Temperature-Controlled Projections**:
       P_i = softmax(Dense_i(input) / temperature) for i=1..N
    2. **Joint Probability Computation**:
       P_joint = P₁ ⊗ P₂ ⊗ ... ⊗ PN (outer product)
    3. **Soft Lookup with Weighted Aggregation**:
       output = Σ_{i₁,i₂,...,iN} P_joint[b,i₁,i₂,...,iN] * grid[i₁,i₂,...,iN,:]

    **Key Properties**:
    - **Differentiable**: All operations support gradient flow for end-to-end training
    - **Structured**: Grid topology encourages spatial organization of representations
    - **Adaptive**: Temperature control allows adjustment of addressing sharpness
    - **Scalable**: Supports arbitrary dimensional grids with efficient einsum operations
    - **Interpretable**: Addressing probabilities provide insight into memory access patterns
    - **Quality-Aware**: Comprehensive quality assessment based on addressing behavior
    - **Filterable**: Built-in data filtering capabilities for robust processing pipelines

    Args:
        grid_shape: List or tuple of integers defining the n-dimensional grid shape.
            Example: [10, 8, 6] creates a 3D grid of size 10×8×6. All values must be positive.
            Larger grids provide more representational capacity but increase computational cost.
        latent_dim: Integer, dimensionality of the latent vector stored at each grid position.
            Must be positive. This determines the output feature dimensionality.
        temperature: Float, initial temperature parameter for softmax sharpening. Lower values
            create sharper, more focused addressing. Higher values create smoother, more
            distributed addressing. Must be positive. This becomes a learnable parameter
            during training. Defaults to 1.0.
        learnable_temperature: Boolean, whether temperature should be learnable during training.
            When True, temperature is updated via backpropagation. When False, temperature
            remains fixed. Defaults to True.
        entropy_regularizer_strength: Float, strength of entropy regularization to encourage
            sharper probability distributions. Higher values enforce sharper addressing.
            Set to 0.0 to disable entropy regularization. Must be non-negative. Defaults to 0.0.
        kernel_initializer: Initializer for Dense layer kernels. Controls how projection
            weights are initialized. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for Dense layer biases. Defaults to 'zeros'.
        grid_initializer: Initializer for the grid latent vectors. Use 'random_normal'
            for diverse initial representations or 'zeros' for uniform start.
            Defaults to 'random_normal'.
        kernel_regularizer: Optional regularizer for Dense layer kernels. Useful for
            preventing overfitting in projection layers.
        bias_regularizer: Optional regularizer for Dense layer biases.
        grid_regularizer: Optional regularizer for the grid weights. Can help with
            learning smoother grid organization.
        epsilon: Float, small constant for numerical stability. Added to prevent
            numerical issues in computations. Defaults to 1e-7.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        - **2D tensor**: `(batch_size, input_dim)` for traditional dense layer usage
        - **3D tensor**: `(batch_size, seq_len, embed_dim)` for transformer token sequences

    Output shape:
        - **2D input**: `(batch_size, latent_dim)`
        - **3D input**: `(batch_size, seq_len, latent_dim)` preserving sequence structure

    Attributes:
        projection_layers: List of Dense layers for dimension projections.
        grid_weights: N-dimensional tensor storing learnable latent vectors.

    Methods:
        get_grid_weights(): Access current grid weight values.
        get_addressing_probabilities(): Analyze addressing patterns.
        get_grid_utilization(): Compute grid usage statistics.
        find_best_matching_units(): Find BMUs for given inputs.
        compute_input_quality(): Compute multiple quality measures for inputs.
        get_quality_statistics(): Get batch-level quality statistics.
        filter_by_quality_threshold(): Filter inputs based on quality thresholds.

    Example:
        ```python
        # Standard 3D grid lookup with learnable temperature
        layer = NeuroGrid(
            grid_shape=[10, 8, 6],
            latent_dim=128,
            temperature=0.5,  # Initial temperature
            learnable_temperature=True,  # Will adapt during training
            entropy_regularizer_strength=0.1  # Encourage sharp addressing
        )
        inputs = keras.Input(shape=(256,))
        outputs = layer(inputs)  # Shape: (batch, 128)

        # 2D grid with fixed temperature and strong sharpening regularization
        layer = NeuroGrid(
            grid_shape=[16, 16],
            latent_dim=64,
            temperature=2.0,
            learnable_temperature=False,  # Fixed temperature
            entropy_regularizer_strength=0.5,  # Strong sharpening
            grid_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(0.01),
            grid_regularizer=keras.regularizers.l1(0.001)
        )

        # High-dimensional structured memory with adaptive temperature
        layer = NeuroGrid(
            grid_shape=[8, 8, 4, 4],
            latent_dim=256,
            temperature=1.5,
            learnable_temperature=True,
            entropy_regularizer_strength=0.2
        )

        # ========== TRANSFORMER INTEGRATION EXAMPLES ==========

        # Token-level structured memory in transformer
        embed_dim = 768
        seq_len = 512
        layer = NeuroGrid(
            grid_shape=[16, 12, 8],
            latent_dim=embed_dim,  # Keep same dimensionality for residual connections
            temperature=0.8,
            learnable_temperature=True,
            entropy_regularizer_strength=0.05  # Lower for token sequences
        )

        # Use in transformer block
        token_embeddings = keras.Input(shape=(seq_len, embed_dim))
        structured_memories = layer(token_embeddings)  # (batch, seq_len, embed_dim)

        # Residual connection preserves sequence structure
        enhanced_tokens = token_embeddings + structured_memories

        # Multi-head attention on enhanced tokens
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=12, key_dim=64
        )(enhanced_tokens, enhanced_tokens)

        # Complete transformer-style model with NeuroGrid
        def create_transformer_with_neurogrid():
            inputs = keras.Input(shape=(seq_len, embed_dim))

            # Token embedding layer
            x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)

            # Transformer blocks with NeuroGrid enhancement
            for i in range(6):  # 6 transformer layers
                # Layer normalization
                norm1 = keras.layers.LayerNormalization()(x)

                # NeuroGrid structured memory
                memory_enhanced = NeuroGrid(
                    grid_shape=[16, 12, 8],
                    latent_dim=embed_dim,
                    temperature=1.0 / (i + 1),  # Sharper addressing in deeper layers
                    learnable_temperature=True,
                    name=f'neurogrid_{i}'
                )(norm1)

                # Residual connection
                x = x + memory_enhanced

                # Standard multi-head attention
                norm2 = keras.layers.LayerNormalization()(x)
                attention = keras.layers.MultiHeadAttention(
                    num_heads=12, key_dim=64
                )(norm2, norm2)
                x = x + attention

                # Feed-forward network
                norm3 = keras.layers.LayerNormalization()(x)
                ffn = keras.Sequential([
                    keras.layers.Dense(embed_dim * 4, activation='gelu'),
                    keras.layers.Dense(embed_dim)
                ])(norm3)
                x = x + ffn

            # Output projection
            outputs = keras.layers.Dense(vocab_size, activation='softmax')(x)
            return keras.Model(inputs, outputs)

        # Token-level quality analysis for transformers
        transformer_inputs = keras.random.normal((batch_size, seq_len, embed_dim))
        token_quality = layer.compute_input_quality(transformer_inputs)  # (batch, seq_len)

        # Attention weighting based on token quality
        attention_weights = token_quality['overall_quality']  # (batch, seq_len)
        attention_mask = attention_weights > 0.5  # Focus on high-quality tokens

        # Quality-aware sequence processing
        def process_sequence_with_quality(sequence_tokens):
            # Get token-level quality scores
            quality_info = layer.compute_input_quality(sequence_tokens)

            # Filter low-quality tokens
            high_quality_tokens = layer.filter_by_quality_threshold(
                sequence_tokens,
                quality_threshold=0.6,
                quality_measure='overall_quality'
            )

            # Process only reliable tokens
            reliable_features = layer(high_quality_tokens['high_quality_inputs'])
            return reliable_features, quality_info

        # Integration with existing transformer models
        base_transformer = keras.applications.BertModel(...)

        # Add NeuroGrid as memory enhancement layer
        enhanced_transformer = keras.Sequential([
            base_transformer,
            NeuroGrid(
                grid_shape=[20, 15, 10],
                latent_dim=base_transformer.config.hidden_size,
                temperature=0.5,
                learnable_temperature=True
            ),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Usage in a model with analysis and temperature control
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            layer,
            keras.layers.Dense(10, activation='softmax')
        ])

        # Monitor temperature evolution during training
        print(f"Initial temperature: {layer.get_current_temperature()}")
        model.fit(x_train, y_train, epochs=10)
        print(f"Final temperature: {layer.get_current_temperature()}")

        # Analyze grid utilization after training
        utilization = layer.get_grid_utilization(test_inputs)
        best_matches = layer.find_best_matching_units(test_inputs)

        # Manually adjust temperature for different phases
        layer.set_temperature(0.1)  # Very sharp addressing for inference

        # Analyze input quality
        quality_measures = layer.compute_input_quality(test_inputs)
        print(f"Average quality: {keras.ops.mean(quality_measures['overall_quality'])}")
        print(f"Average confidence: {keras.ops.mean(quality_measures['addressing_confidence'])}")

        # Filter inputs by quality
        filtered = layer.filter_by_quality_threshold(
            test_inputs,
            quality_threshold=0.7,
            quality_measure='overall_quality'
        )
        high_quality_count = keras.ops.shape(filtered['high_quality_inputs'])[0]
        print(f"High quality inputs: {high_quality_count}")

        # Get batch statistics
        stats = layer.get_quality_statistics(test_inputs)
        print(f"Quality stats: {stats}")

        # Quality-aware model deployment
        def quality_aware_inference(inputs, quality_threshold=0.7):
            quality_measures = layer.compute_input_quality(inputs)
            high_quality_mask = quality_measures['overall_quality'] >= quality_threshold

            # Process high-quality inputs with full model
            high_quality_results = full_model(inputs[high_quality_mask])

            # Flag low-quality inputs for alternative processing
            low_quality_count = keras.ops.sum(keras.ops.cast(~high_quality_mask, 'int32'))
            if low_quality_count > 0:
                print(f"Warning: {low_quality_count} low-quality samples detected")

            return high_quality_results, high_quality_mask

        # Continuous quality monitoring
        class QualityMonitor:
            def __init__(self, neurogrid_layer):
                self.layer = neurogrid_layer
                self.quality_history = []

            def monitor_batch(self, inputs):
                stats = self.layer.get_quality_statistics(inputs)
                self.quality_history.append(stats['overall_quality_mean'])

                # Detect quality drift
                if len(self.quality_history) > 10:
                    recent_avg = np.mean(self.quality_history[-5:])
                    baseline_avg = np.mean(self.quality_history[:5])
                    if abs(recent_avg - baseline_avg) > 0.1:
                        print("Quality drift detected!")

        monitor = QualityMonitor(layer)
        for batch in data_stream:
            monitor.monitor_batch(batch)
        ```

    References:
        - Self-Organizing Maps: Kohonen, T. (1990)
        - Product Key Memories: Lample et al. (2019)
        - Differentiable Neural Computers: Graves et al. (2016)
        - Temperature-based Softmax: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
        - Probabilistic Addressing: Graves et al. (2014) - "Neural Turing Machines"

    **Applications & Use Cases**:

    **Core Applications**:
    1. **Structured Memory Systems**: Learnable lookup tables with spatial organization
    2. **Feature Quantization**: Differentiable vector quantization with grid topology
    3. **Representation Learning**: Learning structured embeddings with neighborhood preservation
    4. **Content-Addressable Memory**: Soft addressing based on learned similarity metrics

    **Transformer & Token-Based Applications**:
    5. **Token-Level Structured Memory**: Enhanced transformer blocks with structured memory
    6. **Sequence Quality Assessment**: Token-level quality evaluation in language models
    7. **Attention Enhancement**: Quality-weighted attention mechanisms
    8. **Dynamic Token Filtering**: Runtime filtering of low-quality tokens
    9. **Structured Token Representations**: Learning organized token embeddings
    10. **Memory-Augmented Transformers**: Adding structured external memory to transformers

    **Quality-Aware Applications**:
    11. **Data Quality Assessment**: Real-time evaluation of input data reliability
    12. **Out-of-Distribution Detection**: Identifying samples that don't fit learned patterns
    13. **Confidence Estimation**: Providing uncertainty measures for model predictions
    14. **Robust Inference Pipelines**: Quality-based routing and processing strategies
    15. **Active Learning**: Selecting informative samples based on addressing behavior
    16. **Production Monitoring**: Tracking data quality in deployed systems

    Note:
        The computational complexity scales with the product of all grid dimensions
        (d1 × d2 × ... × dn). For very large grids (>10^6 total positions), consider
        hierarchical approaches or dimensionality reduction. Memory usage scales
        linearly with grid size and latent dimension.

        During training, the layer naturally develops organized representations where
        similar inputs activate nearby grid regions, exhibiting SOM-like topology
        preservation without explicit neighborhood functions.
    """

    def __init__(
            self,
            grid_shape: Union[List[int], Tuple[int, ...]],
            latent_dim: int,
            use_bias: bool = False,
            temperature: float = 1.0,
            learnable_temperature: bool = False,
            entropy_regularizer_strength: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            grid_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            grid_regularizer: Optional[keras.regularizers.Regularizer] = SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.001),
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not grid_shape or len(grid_shape) == 0:
            raise ValueError("grid_shape cannot be empty")
        if any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"All grid dimensions must be positive, got {grid_shape}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if entropy_regularizer_strength < 0:
            raise ValueError(f"entropy_regularizer_strength must be non-negative, got {entropy_regularizer_strength}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.grid_shape = tuple(grid_shape)
        self.latent_dim = latent_dim
        self.initial_temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.entropy_regularizer_strength = entropy_regularizer_strength
        self.epsilon = epsilon
        self.n_dims = len(self.grid_shape)
        self.use_bias = use_bias
        self.total_grid_size = int(np.prod(self.grid_shape))

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.grid_initializer = keras.initializers.get(grid_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.grid_regularizer = keras.regularizers.get(grid_regularizer)

        # Create projection layers in __init__
        self.projection_layers = []
        for i, dim_size in enumerate(self.grid_shape):
            # Create Dense layer without activation (we'll apply softmax manually with temperature)
            layer = keras.layers.Dense(
                units=dim_size,
                use_bias=self.use_bias,
                activation=None,  # No activation, we'll apply temperature-controlled softmax
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'projection_{i}'
            )
            self.projection_layers.append(layer)

        # Grid weights and temperature created in build()
        self.grid_weights = None
        self.temperature = None
        self.input_is_3d = None  # Set during build based on input shape

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the grid weights, temperature parameter, and build projection layers."""
        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {input_shape}")

        # For 3D inputs (transformer mode), we use the last dimension (embed_dim)
        # For 2D inputs (traditional mode), we use the last dimension (input_dim)
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension (input/embedding dimension) must be defined")

        # Store input shape info for call method
        self.input_is_3d = len(input_shape) == 3

        # Build projection layers - they work on the last dimension regardless of 2D/3D
        projection_input_shape = (None, input_dim)  # Generic shape for Dense layers
        for layer in self.projection_layers:
            layer.build(projection_input_shape)

        # Create learnable temperature parameter
        if self.learnable_temperature:
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                constraint=keras.constraints.NonNeg(),  # Ensure temperature stays positive
                trainable=True
            )
        else:
            # Fixed temperature as non-trainable weight
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                trainable=False
            )

        # Create grid weights: (d1, d2, ..., dn, latent_dim)
        grid_weight_shape = self.grid_shape + (self.latent_dim,)
        self.grid_weights = self.add_weight(
            name='grid_weights',
            shape=grid_weight_shape,
            initializer=self.grid_initializer,
            regularizer=self.grid_regularizer,
            trainable=True
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: project input to probabilities and perform soft lookup.

        Handles both 2D and 3D inputs automatically:
        - 2D inputs: (batch_size, input_dim) → (batch_size, latent_dim)
        - 3D inputs: (batch_size, seq_len, embed_dim) → (batch_size, seq_len, latent_dim)

        For 3D inputs, each token position is processed independently through the
        shared grid, enabling token-level structured memory access while preserving
        sequence structure for transformer compatibility.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, embed_dim).
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same rank as input:
            - 2D input → (batch_size, latent_dim)
            - 3D input → (batch_size, seq_len, latent_dim)
        """
        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            # Reshape to (batch_size * seq_len, embed_dim) for processing
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get temperature-controlled probability distributions for each dimension
        probabilities = []
        total_entropy_loss = 0.0

        for layer in self.projection_layers:
            logits = layer(inputs_2d, training=training)  # (batch_size [* seq_len], dim_i)
            # Apply learnable temperature-controlled softmax for sharper/smoother addressing
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Add entropy regularization to encourage sharper probabilities
            if self.entropy_regularizer_strength > 0.0 and training:
                # Compute entropy: -sum(p * log(p))
                entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
                # Average entropy across batch and add as regularization loss
                avg_entropy = keras.ops.mean(entropy)
                entropy_loss = self.entropy_regularizer_strength * avg_entropy
                total_entropy_loss += entropy_loss

        # Add entropy regularization loss if enabled
        if self.entropy_regularizer_strength > 0.0 and training:
            self.add_loss(total_entropy_loss)

        # Compute joint probability using efficient outer product
        joint_prob = self._compute_joint_probability(probabilities)

        # Perform soft lookup with numerical stability
        output_2d = self._soft_lookup(joint_prob)

        # Restore original shape for 3D inputs (transformer mode)
        if input_rank == 3:
            output = keras.ops.reshape(output_2d, (batch_size, seq_len, self.latent_dim))
        else:
            output = output_2d

        return output

    def _compute_joint_probability(self, probabilities: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Compute joint probability via efficient outer product of individual probabilities.

        Args:
            probabilities: List of probability tensors, each of shape (batch, dim_i).

        Returns:
            Joint probability tensor of shape (batch, d1, d2, ..., dn).
        """
        # Start with first probability: (batch, d1)
        joint_prob = probabilities[0]

        # Sequentially compute outer products with numerical stability
        for i, prob in enumerate(probabilities[1:], 1):
            # Add new axis for broadcasting: joint_prob becomes (batch, d1, ..., di-1, 1)
            joint_prob = keras.ops.expand_dims(joint_prob, axis=-1)

            # Add axes to prob for proper broadcasting: (batch, 1, ..., 1, di)
            for _ in range(i):
                if len(prob.shape) == 2:  # First iteration
                    prob = keras.ops.expand_dims(prob, axis=1)
                else:
                    prob = keras.ops.expand_dims(prob, axis=-2)

            # Element-wise multiplication gives outer product
            joint_prob = joint_prob * prob

        # Add small epsilon for numerical stability
        joint_prob = joint_prob + self.epsilon

        # Renormalize to ensure probabilities sum to 1
        prob_sum = keras.ops.sum(
            joint_prob,
            axis=tuple(range(1, len(joint_prob.shape))),
            keepdims=True
        )
        joint_prob = joint_prob / (prob_sum + self.epsilon)

        return joint_prob

    def _soft_lookup(self, joint_prob: keras.KerasTensor) -> keras.KerasTensor:
        """
        Perform soft lookup using joint probabilities and grid weights.

        Args:
            joint_prob: Joint probability tensor of shape (batch, d1, d2, ..., dn).

        Returns:
            Weighted sum tensor of shape (batch, latent_dim).
        """
        # Create einsum equation for the weighted sum
        # joint_prob: (batch, d1, d2, ..., dn)
        # grid_weights: (d1, d2, ..., dn, latent_dim)
        # output: (batch, latent_dim)

        # Build einsum equation dynamically based on grid dimensions
        batch_idx = 'b'
        # Fix: Use alphabet letters that don't conflict and handle higher dimensions
        grid_indices = ''.join([chr(ord('i') + j) for j in range(min(self.n_dims, 23))])  # Limit to available letters
        latent_idx = 'z'  # Use 'z' to avoid conflicts

        # joint_prob indices: batch + grid dimensions
        joint_indices = batch_idx + grid_indices

        # grid_weights indices: grid dimensions + latent
        grid_indices_with_latent = grid_indices + latent_idx

        # output indices: batch + latent
        output_indices = batch_idx + latent_idx

        # Einsum equation: e.g., 'bijk,ijkz->bz' for 3D grid
        equation = f"{joint_indices},{grid_indices_with_latent}->{output_indices}"

        # For very high dimensional grids, fall back to manual computation
        if self.n_dims > 6:  # Einsum becomes inefficient/problematic for very high dims
            # Reshape for manual computation
            batch_size = keras.ops.shape(joint_prob)[0]
            joint_flat = keras.ops.reshape(joint_prob, (batch_size, self.total_grid_size))
            grid_flat = keras.ops.reshape(self.grid_weights, (self.total_grid_size, self.latent_dim))
            output = keras.ops.matmul(joint_flat, grid_flat)
        else:
            # Use einsum for lower dimensional grids
            output = keras.ops.einsum(equation, joint_prob, self.grid_weights)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for both 2D and 3D inputs.

        Args:
            input_shape: Input shape tuple.

        Returns:
            Output shape tuple:
            - 2D input (batch, input_dim) → (batch, latent_dim)
            - 3D input (batch, seq_len, embed_dim) → (batch, seq_len, latent_dim)
        """
        if len(input_shape) == 2:
            # 2D input: (batch_size, input_dim) → (batch_size, latent_dim)
            return (input_shape[0], self.latent_dim)
        elif len(input_shape) == 3:
            # 3D input: (batch_size, seq_len, embed_dim) → (batch_size, seq_len, latent_dim)
            return (input_shape[0], input_shape[1], self.latent_dim)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'use_bias': self.use_bias,
            'grid_shape': list(self.grid_shape),
            'latent_dim': self.latent_dim,
            'temperature': self.initial_temperature,
            'learnable_temperature': self.learnable_temperature,
            'entropy_regularizer_strength': self.entropy_regularizer_strength,
            'epsilon': self.epsilon,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'grid_initializer': keras.initializers.serialize(self.grid_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'grid_regularizer': keras.regularizers.serialize(self.grid_regularizer),
        })
        # Note: input_is_3d is not serialized as it's determined during build() from input shape
        return config

    def get_grid_weights(self) -> keras.KerasTensor:
        """
        Get the current grid weights for analysis or visualization.

        Returns:
            Grid weights tensor of shape (d1, d2, ..., dn, latent_dim).

        Raises:
            ValueError: If layer has not been built yet.
        """
        if self.grid_weights is None:
            raise ValueError("Layer must be built before accessing grid weights")
        return self.grid_weights

    def get_addressing_probabilities(
            self,
            inputs: keras.KerasTensor
    ) -> Dict[str, Union[List[keras.KerasTensor], keras.KerasTensor]]:
        """
        Get the addressing probabilities for analysis and interpretation.

        Works with both 2D and 3D inputs, automatically handling transformer token sequences
        by processing each token position independently through the shared grid structure.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, embed_dim).

        Returns:
            Dictionary containing:
            - 'individual': List of probability tensors for each dimension
            - 'joint': Joint probability tensor
            - 'entropy': Entropy of each individual distribution (uncertainty measure)

            **Tensor shapes depend on input format**:
            - **2D inputs**: individual probs shape (batch_size, dim_i), joint shape (batch_size, d1, d2, ...)
            - **3D inputs**: individual probs shape (batch * seq_len, dim_i), joint shape (batch * seq_len, d1, d2, ...)

        Raises:
            ValueError: If layer has not been built yet.

        Note:
            For 3D inputs, the method processes tokens in flattened format internally.
            Use the quality methods for properly reshaped token-level analysis.
        """
        if not self.built:
            raise ValueError("Layer must be built before getting probabilities")

        # Handle input reshaping for 3D inputs
        input_rank = len(inputs.shape)
        if input_rank == 3:
            original_shape = keras.ops.shape(inputs)
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get individual probabilities (recompute to match forward pass)
        probabilities = []
        entropies = []

        for layer in self.projection_layers:
            logits = layer(inputs_2d)
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Compute entropy as uncertainty measure
            entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
            entropies.append(entropy)

        # Compute joint probability
        joint_prob = self._compute_joint_probability(probabilities)

        return {
            'individual': probabilities,
            'joint': joint_prob,
            'entropy': entropies
        }

    def get_grid_utilization(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Compute grid utilization statistics for understanding memory usage patterns.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).

        Returns:
            Dictionary containing:
            - 'activation_counts': How often each grid position is maximally activated
            - 'total_activation': Sum of all activations per grid position
            - 'utilization_rate': Fraction of total activations per position

        Raises:
            ValueError: If layer has not been built yet.
        """
        if not self.built:
            raise ValueError("Layer must be built before computing utilization")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find most activated position per input
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        max_positions = keras.ops.argmax(joint_prob_flat, axis=-1)

        # Fix: Replace scatter_update with proper counting using one-hot
        activation_counts = keras.ops.zeros((self.total_grid_size,))

        # Convert positions to one-hot and sum
        one_hot_positions = keras.ops.one_hot(max_positions, self.total_grid_size)
        activation_counts = keras.ops.sum(one_hot_positions, axis=0)

        # Total activations per position (sum of all probabilities)
        total_activation = keras.ops.sum(joint_prob_flat, axis=0)

        # Utilization rate (normalized)
        total_inputs = keras.ops.cast(keras.ops.shape(inputs)[0], 'float32')
        utilization_rate = activation_counts / (total_inputs + self.epsilon)

        return {
            'activation_counts': activation_counts,
            'total_activation': total_activation,
            'utilization_rate': utilization_rate
        }

    def find_best_matching_units(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Find Best Matching Units (BMUs) for given inputs in SOM terminology.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).

        Returns:
            Dictionary containing:
            - 'bmu_indices': Grid indices of BMUs, shape (batch_size, n_dims)
            - 'bmu_probabilities': Probability values at BMU positions
            - 'bmu_coordinates': Flat indices of BMUs for easier indexing

        Raises:
            ValueError: If layer has not been built yet.
        """
        if not self.built:
            raise ValueError("Layer must be built before finding BMUs")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find maximum probability positions
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        bmu_coordinates = keras.ops.argmax(joint_prob_flat, axis=-1)
        bmu_probabilities = keras.ops.max(joint_prob_flat, axis=-1)

        # Convert flat indices back to n-dimensional indices
        bmu_indices = []
        remaining = bmu_coordinates

        for dim_size in reversed(self.grid_shape):
            dim_indices = remaining % dim_size
            bmu_indices.append(dim_indices)
            remaining = remaining // dim_size

        # Reverse to get correct order and stack
        bmu_indices = keras.ops.stack(list(reversed(bmu_indices)), axis=-1)

        return {
            'bmu_indices': bmu_indices,
            'bmu_probabilities': bmu_probabilities,
            'bmu_coordinates': bmu_coordinates
        }

    def set_temperature(self, new_temperature: float) -> None:
        """
        Update the temperature parameter for dynamic control during training.

        Args:
            new_temperature: New temperature value, must be positive.

        Raises:
            ValueError: If temperature is not positive or layer not built.
        """
        if new_temperature <= 0:
            raise ValueError(f"temperature must be positive, got {new_temperature}")
        if self.temperature is None:
            raise ValueError("Layer must be built before setting temperature")

        # Update the temperature weight value
        self.temperature.assign(new_temperature)

    def get_current_temperature(self) -> float:
        """
        Get the current temperature value.

        Returns:
            Current temperature as a float value.

        Raises:
            ValueError: If layer has not been built yet.
        """
        if self.temperature is None:
            raise ValueError("Layer must be built before getting temperature")

        return float(keras.ops.convert_to_numpy(self.temperature))

    def compute_input_quality(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Compute comprehensive quality measures for input data based on probabilistic addressing behavior.

        This method analyzes how effectively inputs interact with the learned grid structure,
        providing multiple complementary metrics that assess input quality from different perspectives.
        These measures leverage the NeuroGrid's probabilistic addressing mechanism to evaluate
        data distinctiveness, mapping reliability, and representational coherence.

        **Mathematical Foundations**:

        Given input x producing individual probabilities P₁(x), P₂(x), ..., Pₙ(x) and
        joint probability P_joint(x) = P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ:

        1. **Addressing Confidence**: max(P_joint(x))
           - Measures peak probability mass concentration
           - Range: [1/|Grid|, 1] where |Grid| = ∏dᵢ
           - Higher values indicate clearer, more decisive addressing

        2. **Addressing Entropy**: H(P_joint) = -Σ P_joint(x) log P_joint(x)
           - Measures uncertainty in joint probability distribution
           - Range: [0, log(|Grid|)]
           - Lower values indicate more focused, predictable addressing

        3. **Dimension Consistency**: (1/N) Σᵢ max(Pᵢ(x))
           - Average peak probability across individual dimensions
           - Range: [1/max(dᵢ), 1]
           - Higher values indicate consistent, sharp dimension-wise addressing

        4. **Grid Coherence**: 1/(1 + Var(P_joint_flat))
           - Inverse relationship with probability variance
           - Range: (0, 1]
           - Higher values indicate concentrated, structured addressing patterns

        5. **Uncertainty**: (1/N) Σᵢ H(Pᵢ) = (1/N) Σᵢ [-Σⱼ Pᵢⱼ log Pᵢⱼ]
           - Average entropy across individual dimensions
           - Range: [0, max(log(dᵢ))]
           - Lower values indicate more certain, focused addressing

        6. **Overall Quality**: Weighted composite of normalized measures
           - Formula: 0.25×conf + 0.25×(1-H_norm) + 0.20×cons + 0.15×coh + 0.15×(1-unc_norm)
           - Range: [0, 1]
           - Balanced assessment combining multiple quality aspects

        **Quality Interpretations**:

        **High-Quality Inputs (scores > 0.7)**:
        - Sharp, focused addressing with clear grid position preferences
        - Low uncertainty across all dimensions
        - Consistent mapping behavior suggesting good representational fit
        - Likely in-distribution samples that align well with learned structure
        - Suitable for confident inference and reliable feature extraction

        **Medium-Quality Inputs (scores 0.3-0.7)**:
        - Moderate addressing focus with some uncertainty
        - Partial consistency across dimensions
        - May represent boundary cases or transitional patterns
        - Useful but may benefit from additional context or preprocessing

        **Low-Quality Inputs (scores < 0.3)**:
        - Diffuse, uncertain addressing patterns
        - High entropy and low consistency
        - Likely out-of-distribution, noisy, or ambiguous samples
        - May require filtering, cleaning, or alternative processing

        **Use Cases**:

        1. **Data Quality Assessment**: Identify problematic samples in datasets
        2. **Out-of-Distribution Detection**: Flag samples that don't fit learned patterns
        3. **Confidence Estimation**: Provide uncertainty estimates for predictions
        4. **Data Filtering**: Remove low-quality samples before downstream processing
        5. **Active Learning**: Select informative samples based on addressing patterns
        6. **Model Monitoring**: Track input quality distribution over time
        7. **Anomaly Detection**: Identify unusual patterns in addressing behavior

        **Practical Thresholds** (empirically derived guidelines):
        - **addressing_confidence**: >0.1 (good), >0.3 (excellent) for typical grids
        - **addressing_entropy**: <log(|Grid|)/2 (good), <log(|Grid|)/4 (excellent)
        - **dimension_consistency**: >0.3 (good), >0.6 (excellent)
        - **overall_quality**: >0.5 (acceptable), >0.7 (good), >0.85 (excellent)

        Args:
            inputs: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, embed_dim).
                For 3D inputs, quality measures are computed independently for each token position.

        Returns:
            Dictionary containing quality measures as tensors:
            - **2D inputs**: All measures have shape (batch_size,)
            - **3D inputs**: All measures have shape (batch_size, seq_len) for token-level quality

            The returned measures are:

            - **'addressing_confidence'**: Peak probability in joint distribution.
              Higher values (→1) indicate very focused addressing to specific grid regions.
              Lower values (→0) suggest diffuse, unfocused addressing patterns.

            - **'addressing_entropy'**: Entropy of joint probability distribution.
              Lower values (→0) indicate concentrated, predictable addressing behavior.
              Higher values (→log|Grid|) suggest uniform, uncertain addressing.

            - **'dimension_consistency'**: Average sharpness across individual dimensions.
              Higher values (→1) indicate consistent, decisive addressing per dimension.
              Lower values suggest inconsistent or uncertain dimension-wise addressing.

            - **'grid_coherence'**: Structural coherence of addressing patterns.
              Higher values (→1) indicate well-structured interaction with grid topology.
              Lower values suggest poor fit with learned representational structure.

            - **'uncertainty'**: Combined uncertainty across all addressing dimensions.
              Lower values (→0) indicate confident, focused addressing behavior.
              Higher values suggest high uncertainty and ambiguous addressing.

            - **'overall_quality'**: Composite quality score normalized to [0,1] range.
              Values >0.7 typically indicate high-quality, reliable inputs.
              Values <0.3 suggest problematic samples requiring attention.

            **For 3D inputs (transformer mode)**: Each quality measure provides token-level
            assessment, enabling identification of problematic tokens within sequences,
            attention-based quality weighting, and token-specific processing decisions.

        Raises:
            ValueError: If layer has not been built yet (no grid weights available).

        Example:
            ```python
            # Analyze batch quality
            quality_measures = layer.compute_input_quality(batch_inputs)

            # Get overall quality scores
            overall_scores = quality_measures['overall_quality']

            # Find high-confidence samples
            confident_mask = quality_measures['addressing_confidence'] > 0.3
            confident_inputs = batch_inputs[confident_mask]

            # Analyze quality distribution
            mean_quality = keras.ops.mean(overall_scores)
            quality_std = keras.ops.std(overall_scores)
            print(f"Quality: {mean_quality:.3f} ± {quality_std:.3f}")

            # Identify potential outliers
            low_quality_mask = overall_scores < 0.3
            outlier_count = keras.ops.sum(keras.ops.cast(low_quality_mask, 'int32'))
            print(f"Potential outliers: {outlier_count}/{len(batch_inputs)}")
            ```

        Note:
            Quality measures are computed based on the current state of learned grid weights
            and will evolve during training as the representation space develops. Early in
            training, quality scores may be less meaningful until the grid structure stabilizes.
        """
        if not self.built:
            raise ValueError("Layer must be built before computing quality")

        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
            effective_batch_size = batch_size * seq_len
        else:
            inputs_2d = inputs
            effective_batch_size = keras.ops.shape(inputs)[0]

        prob_info = self.get_addressing_probabilities(inputs_2d)
        individual_probs = prob_info['individual']
        joint_prob = prob_info['joint']
        entropies = prob_info['entropy']

        # 1. Addressing Confidence: Maximum probability in joint distribution
        joint_prob_flat = keras.ops.reshape(joint_prob, (effective_batch_size, -1))
        addressing_confidence = keras.ops.max(joint_prob_flat, axis=-1)

        # 2. Addressing Entropy: Entropy of joint probability distribution (lower = better quality)
        joint_entropy = -keras.ops.sum(
            joint_prob_flat * keras.ops.log(joint_prob_flat + self.epsilon),
            axis=-1
        )

        # 3. Dimension Consistency: How sharp/consistent are individual dimensions
        dimension_sharpness = []
        for prob in individual_probs:
            # Higher max probability = more consistent/sharp addressing
            max_prob = keras.ops.max(prob, axis=-1)
            dimension_sharpness.append(max_prob)

        # Average sharpness across all dimensions
        dimension_consistency = keras.ops.mean(keras.ops.stack(dimension_sharpness, axis=-1), axis=-1)

        # 4. Grid Coherence: Measure based on probability distribution spread
        # Lower spread (higher concentration) indicates better mapping to grid structure
        prob_variance = keras.ops.var(joint_prob_flat, axis=-1)
        # Invert so higher values mean better coherence
        grid_coherence = 1.0 / (1.0 + prob_variance)

        # 5. Combined Uncertainty: Average of individual dimension entropies
        avg_dimension_entropy = keras.ops.mean(keras.ops.stack(entropies, axis=-1), axis=-1)
        uncertainty = avg_dimension_entropy

        # 6. Overall Quality Score: Composite measure (0-1 scale)
        # Normalize and combine multiple factors
        confidence_norm = addressing_confidence  # Already 0-1
        consistency_norm = dimension_consistency  # Already 0-1
        coherence_norm = grid_coherence  # Already 0-1

        # Entropy-based terms (invert and normalize to 0-1)
        max_joint_entropy = keras.ops.log(keras.ops.cast(self.total_grid_size, 'float32'))
        entropy_quality = 1.0 - (joint_entropy / (max_joint_entropy + self.epsilon))
        entropy_quality = keras.ops.clip(entropy_quality, 0.0, 1.0)

        max_dim_entropy = keras.ops.log(keras.ops.cast(keras.ops.max(keras.ops.array(self.grid_shape)), 'float32'))
        uncertainty_quality = 1.0 - (uncertainty / (max_dim_entropy + self.epsilon))
        uncertainty_quality = keras.ops.clip(uncertainty_quality, 0.0, 1.0)

        # Weighted combination of quality factors
        overall_quality = (
                0.25 * confidence_norm +  # Addressing confidence
                0.25 * entropy_quality +  # Joint entropy quality
                0.20 * consistency_norm +  # Dimension consistency
                0.15 * coherence_norm +  # Grid coherence
                0.15 * uncertainty_quality  # Individual dimension quality
        )

        # Reshape results back to match input format for 3D inputs
        if input_rank == 3:
            # Reshape from (batch * seq_len,) to (batch, seq_len)
            addressing_confidence = keras.ops.reshape(addressing_confidence, (batch_size, seq_len))
            joint_entropy = keras.ops.reshape(joint_entropy, (batch_size, seq_len))
            dimension_consistency = keras.ops.reshape(dimension_consistency, (batch_size, seq_len))
            grid_coherence = keras.ops.reshape(grid_coherence, (batch_size, seq_len))
            uncertainty = keras.ops.reshape(uncertainty, (batch_size, seq_len))
            overall_quality = keras.ops.reshape(overall_quality, (batch_size, seq_len))

        return {
            'addressing_confidence': addressing_confidence,
            'addressing_entropy': joint_entropy,
            'dimension_consistency': dimension_consistency,
            'grid_coherence': grid_coherence,
            'uncertainty': uncertainty,
            'overall_quality': overall_quality
        }

    def get_quality_statistics(self, inputs: keras.KerasTensor) -> Dict[str, float]:
        """
        Compute comprehensive batch-level statistical summaries for all input quality measures.

        This method provides detailed statistical analysis of quality measure distributions
        across a batch of inputs, enabling comprehensive assessment of data quality patterns,
        detection of distribution shifts, and identification of statistical anomalies.

        **Statistical Measures Computed**:

        For each quality measure (addressing_confidence, addressing_entropy, dimension_consistency,
        grid_coherence, uncertainty, overall_quality), the following statistics are calculated:

        - **Mean**: Central tendency of the quality distribution
        - **Standard Deviation**: Variability and spread in quality scores
        - **Minimum**: Worst-case quality in the batch
        - **Maximum**: Best-case quality in the batch
        - **Median**: Robust central tendency measure (50th percentile)

        **Interpretation Guidelines**:

        **Mean Quality Patterns**:
        - High mean (>0.7): Batch contains predominantly high-quality samples
        - Medium mean (0.3-0.7): Mixed quality batch with diverse sample types
        - Low mean (<0.3): Batch dominated by problematic or out-of-distribution samples

        **Standard Deviation Patterns**:
        - Low std (<0.1): Homogeneous batch with consistent quality
        - Medium std (0.1-0.3): Natural variation in sample quality
        - High std (>0.3): Heterogeneous batch with wide quality range

        **Min-Max Range Analysis**:
        - Narrow range: Consistent data quality across batch
        - Wide range: Batch contains both excellent and poor samples
        - Low minimum: Presence of outliers or problematic samples

        **Mean vs Median Comparison**:
        - Similar values: Symmetric quality distribution
        - Mean < Median: Distribution skewed toward lower quality (outliers present)
        - Mean > Median: Distribution skewed toward higher quality

        **Applications**:

        1. **Data Quality Monitoring**: Track quality distributions over time
        2. **Batch Assessment**: Evaluate incoming data quality before processing
        3. **Distribution Analysis**: Understand quality patterns in datasets
        4. **Outlier Detection**: Identify batches with unusual quality characteristics
        5. **Model Performance Correlation**: Relate quality statistics to model performance
        6. **Data Filtering Thresholds**: Set appropriate quality filtering thresholds
        7. **Quality Control**: Monitor data pipeline quality in production systems

        Args:
            inputs: Input tensor of shape (batch_size, input_dim). Statistical analysis
                will be performed across the batch dimension for all quality measures.

        Returns:
            Dictionary containing statistical summaries with keys formatted as
            "{measure_name}_{statistic}". Each value is a Python float for easy
            logging and analysis:

            **Addressing Confidence Statistics**:
            - 'addressing_confidence_mean': Average confidence across batch
            - 'addressing_confidence_std': Standard deviation of confidence scores
            - 'addressing_confidence_min': Minimum confidence (worst addressing)
            - 'addressing_confidence_max': Maximum confidence (best addressing)
            - 'addressing_confidence_median': Median confidence (robust central measure)

            **Addressing Entropy Statistics**:
            - 'addressing_entropy_mean': Average entropy (higher = more uncertain)
            - 'addressing_entropy_std': Entropy variability across batch
            - 'addressing_entropy_min': Minimum entropy (most focused addressing)
            - 'addressing_entropy_max': Maximum entropy (most diffuse addressing)
            - 'addressing_entropy_median': Median entropy value

            **Dimension Consistency Statistics**:
            - 'dimension_consistency_mean': Average consistency across dimensions
            - 'dimension_consistency_std': Consistency variability
            - 'dimension_consistency_min': Poorest dimensional consistency
            - 'dimension_consistency_max': Best dimensional consistency
            - 'dimension_consistency_median': Median consistency score

            **Grid Coherence Statistics**:
            - 'grid_coherence_mean': Average structural coherence
            - 'grid_coherence_std': Coherence variability
            - 'grid_coherence_min': Poorest grid structure alignment
            - 'grid_coherence_max': Best grid structure alignment
            - 'grid_coherence_median': Median coherence score

            **Uncertainty Statistics**:
            - 'uncertainty_mean': Average uncertainty level
            - 'uncertainty_std': Uncertainty variability
            - 'uncertainty_min': Minimum uncertainty (most certain)
            - 'uncertainty_max': Maximum uncertainty (most uncertain)
            - 'uncertainty_median': Median uncertainty level

            **Overall Quality Statistics**:
            - 'overall_quality_mean': Average composite quality score
            - 'overall_quality_std': Quality score variability
            - 'overall_quality_min': Worst overall quality in batch
            - 'overall_quality_max': Best overall quality in batch
            - 'overall_quality_median': Median quality score

        Example:
            ```python
            # Get comprehensive quality statistics
            stats = layer.get_quality_statistics(validation_batch)

            # Analyze overall batch quality
            mean_quality = stats['overall_quality_mean']
            quality_std = stats['overall_quality_std']
            quality_range = stats['overall_quality_max'] - stats['overall_quality_min']

            print(f"Batch Quality Summary:")
            print(f"  Mean: {mean_quality:.3f} ± {quality_std:.3f}")
            print(f"  Range: [{stats['overall_quality_min']:.3f}, {stats['overall_quality_max']:.3f}]")
            print(f"  Median: {stats['overall_quality_median']:.3f}")

            # Check for quality issues
            if mean_quality < 0.3:
                print("WARNING: Low average batch quality detected")
            if quality_std > 0.4:
                print("WARNING: High quality variance - heterogeneous batch")
            if stats['overall_quality_min'] < 0.1:
                print("WARNING: Extreme outliers present")

            # Confidence analysis
            conf_mean = stats['addressing_confidence_mean']
            conf_median = stats['addressing_confidence_median']
            if abs(conf_mean - conf_median) > 0.1:
                print("INFO: Skewed confidence distribution detected")

            # Log for monitoring
            logger.info(f"Quality stats: mean={mean_quality:.3f}, std={quality_std:.3f}")
            ```

        Note:
            Statistics reflect the current state of the learned grid representation.
            During training, these statistics will evolve as the model learns better
            representations. For reliable quality assessment, ensure the model has
            been trained sufficiently for the grid structure to stabilize.
        """
        quality_measures = self.compute_input_quality(inputs)

        statistics = {}
        for measure_name, measure_values in quality_measures.items():
            measure_np = keras.ops.convert_to_numpy(measure_values)
            statistics[f"{measure_name}_mean"] = float(np.mean(measure_np))
            statistics[f"{measure_name}_std"] = float(np.std(measure_np))
            statistics[f"{measure_name}_min"] = float(np.min(measure_np))
            statistics[f"{measure_name}_max"] = float(np.max(measure_np))
            statistics[f"{measure_name}_median"] = float(np.median(measure_np))

        return statistics

    def filter_by_quality_threshold(
            self,
            inputs: keras.KerasTensor,
            quality_threshold: float = 0.5,
            quality_measure: str = 'overall_quality'
    ) -> Dict[str, keras.KerasTensor]:
        """
        Intelligently filter and partition input data based on configurable quality thresholds.

        This method provides sophisticated data filtering capabilities by leveraging quality
        measures derived from the NeuroGrid's probabilistic addressing behavior. It enables
        separation of high-quality from potentially problematic samples, supporting robust
        data processing pipelines and quality-aware model deployment strategies.

        **Filtering Strategy**:

        The method computes the specified quality measure for all inputs, then partitions
        the data based on the threshold, creating separate high-quality and low-quality
        subsets along with corresponding masks for flexible downstream processing.

        **Quality Measure Selection Guidelines**:

        - **'overall_quality'** (recommended default): Balanced composite measure
          - Best for general-purpose filtering
          - Combines multiple quality aspects
          - Threshold 0.5-0.7 typically works well

        - **'addressing_confidence'**: Focus on addressing certainty
          - Best for applications requiring confident predictions
          - Use when you need samples that map clearly to specific grid regions
          - Threshold 0.1-0.3 for typical grids

        - **'addressing_entropy'**: Focus on uncertainty (inverted logic)
          - Lower values are better (more focused)
          - Use threshold based on log(grid_size): threshold < log(|Grid|)/2
          - Best for uncertainty-sensitive applications

        - **'dimension_consistency'**: Focus on dimensional coherence
          - Best when individual dimension behavior is critical
          - Threshold 0.3-0.6 typically effective

        - **'grid_coherence'**: Focus on structural alignment
          - Best when grid topology preservation is important
          - Threshold 0.2-0.5 typically effective

        **Threshold Selection Strategies**:

        1. **Conservative Filtering** (high thresholds):
           - overall_quality > 0.8, addressing_confidence > 0.5
           - Keeps only highest-quality samples
           - Reduces data volume but maximizes reliability

        2. **Moderate Filtering** (medium thresholds):
           - overall_quality > 0.5, addressing_confidence > 0.2
           - Balanced approach between quality and quantity
           - Suitable for most production applications

        3. **Permissive Filtering** (low thresholds):
           - overall_quality > 0.3, addressing_confidence > 0.1
           - Removes only clearly problematic samples
           - Preserves maximum data while filtering obvious outliers

        **Applications**:

        1. **Data Preprocessing**: Remove low-quality samples before training
        2. **Inference Filtering**: Flag uncertain predictions for manual review
        3. **Active Learning**: Select high-quality samples for labeling
        4. **Anomaly Detection**: Identify and separate outliers
        5. **Quality Control**: Monitor and filter production data streams
        6. **A/B Testing**: Compare model performance on different quality subsets
        7. **Confidence-Based Routing**: Route samples based on quality assessment

        **Performance Considerations**:

        - Filtering preserves tensor structure but may significantly reduce batch size
        - Consider batch size implications for downstream processing
        - High-quality subset may be empty if threshold is too stringent
        - Low-quality subset provides valuable insights for data improvement

        Args:
            inputs: Input tensor of shape (batch_size, input_dim). All samples will be
                evaluated using the specified quality measure and partitioned based on
                the provided threshold.

            quality_threshold: Float threshold value for quality-based filtering.
                - Range depends on selected quality measure
                - Higher thresholds = more stringent filtering
                - Lower thresholds = more permissive filtering
                - Default 0.5 works well for 'overall_quality' measure

            quality_measure: String specifying which quality measure to use for filtering.
                Must be one of: 'overall_quality', 'addressing_confidence',
                'addressing_entropy', 'dimension_consistency', 'grid_coherence', 'uncertainty'.
                Default 'overall_quality' provides balanced filtering behavior.

        Returns:
            Dictionary containing filtered data and metadata:

            - **'high_quality_inputs'**: Tensor of inputs with quality >= threshold.
              Shape: (n_high_quality, input_dim) where n_high_quality <= batch_size.
              Contains samples that meet the quality criteria for confident processing.

            - **'low_quality_inputs'**: Tensor of inputs with quality < threshold.
              Shape: (n_low_quality, input_dim) where n_low_quality <= batch_size.
              Contains samples that may require special handling or filtering.

            - **'high_quality_mask'**: Boolean tensor indicating high-quality samples.
              Shape: (batch_size,). True for samples meeting quality threshold.
              Useful for custom filtering operations or index-based selection.

            - **'quality_scores'**: Tensor of quality scores for all input samples.
              Shape: (batch_size,). Contains the computed quality measure values
              used for filtering. Useful for analysis and threshold tuning.

        Raises:
            ValueError: If quality_measure is not recognized or layer not built.

        Example:
            ```python
            # Basic quality filtering
            filtered = layer.filter_by_quality_threshold(
                batch_inputs,
                quality_threshold=0.7,
                quality_measure='overall_quality'
            )

            high_quality_data = filtered['high_quality_inputs']
            low_quality_data = filtered['low_quality_inputs']
            quality_mask = filtered['high_quality_mask']

            print(f"Original batch: {batch_inputs.shape[0]} samples")
            print(f"High quality: {high_quality_data.shape[0]} samples")
            print(f"Low quality: {low_quality_data.shape[0]} samples")

            # Confidence-based filtering for critical applications
            confident_filtered = layer.filter_by_quality_threshold(
                batch_inputs,
                quality_threshold=0.3,
                quality_measure='addressing_confidence'
            )

            # Process high-confidence samples with full model
            confident_predictions = model(confident_filtered['high_quality_inputs'])

            # Flag low-confidence samples for review
            uncertain_samples = confident_filtered['low_quality_inputs']
            if uncertain_samples.shape[0] > 0:
                print(f"Flagging {uncertain_samples.shape[0]} samples for manual review")

            # Adaptive threshold based on batch statistics
            quality_scores = filtered['quality_scores']
            median_quality = keras.ops.median(quality_scores)
            adaptive_threshold = float(median_quality) * 0.8  # 80% of median

            adaptive_filtered = layer.filter_by_quality_threshold(
                batch_inputs,
                quality_threshold=adaptive_threshold,
                quality_measure='overall_quality'
            )

            # Quality-aware batch processing
            if keras.ops.mean(quality_scores) > 0.8:
                print("High-quality batch: processing with full precision")
                results = high_precision_model(batch_inputs)
            else:
                print("Mixed-quality batch: using filtered processing")
                results = standard_model(filtered['high_quality_inputs'])
            ```

        Note:
            Filtering effectiveness depends on the quality of the learned grid representation.
            During early training stages, quality measures may be less discriminative.
            Consider using validation data to calibrate appropriate thresholds for your
            specific use case and data distribution.
        """
        quality_measures = self.compute_input_quality(inputs)

        if quality_measure not in quality_measures:
            raise ValueError(f"Unknown quality measure: {quality_measure}")

        quality_scores = quality_measures[quality_measure]
        high_quality_mask = quality_scores >= quality_threshold
        low_quality_mask = quality_scores < quality_threshold

        # Fix: Replace keras.ops.boolean_mask with proper tensor indexing
        high_quality_indices = keras.ops.where(high_quality_mask)[:, 0]
        low_quality_indices = keras.ops.where(low_quality_mask)[:, 0]

        high_quality_inputs = keras.ops.take(inputs, high_quality_indices, axis=0)
        low_quality_inputs = keras.ops.take(inputs, low_quality_indices, axis=0)

        return {
            'high_quality_inputs': high_quality_inputs,
            'low_quality_inputs': low_quality_inputs,
            'high_quality_mask': high_quality_mask,
            'quality_scores': quality_scores
        }