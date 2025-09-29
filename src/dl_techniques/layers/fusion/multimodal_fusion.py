"""Implements a unified framework for multi-modal information fusion.

This layer provides a modular and extensible framework for combining feature
representations from multiple, heterogeneous data modalities (e.g., vision,
language, audio). It serves as a factory that implements several distinct
fusion strategies, each grounded in different theoretical assumptions about
how cross-modal interactions should be modeled.

Architecture Overview:
    The layer uses the Strategy pattern where a high-level interface delegates
    fusion logic to specific implementations. Core strategies include:

    1. Early Fusion (Concatenation): Y = W[X₁; X₂; ...; Xₙ] + b
    2. Element-wise Fusion: Addition/Multiplication with alignment
    3. Interaction-based: Bilinear pooling and tensor fusion
    4. Attention-based: Cross-modal attention mechanisms

References:
    - Baltrusaitis et al. (2018): Multimodal Machine Learning: A Survey
    - Vaswani et al. (2017): Attention Is All You Need
    - Zadeh et al. (2017): Tensor Fusion Network
    - Lu et al. (2019): ViLBERT
"""

import keras
from typing import Optional, Union, List, Dict, Any, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# Local imports - Assumed to exist in the project structure
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.factory import create_ffn_layer, FFNType
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.norms.factory import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------
# Type definitions for fusion strategies
# ---------------------------------------------------------------------

FusionStrategy = Literal[
    'cross_attention',    # Bidirectional cross-attention between modalities
    'concatenation',      # Concatenate and project
    'addition',           # Element-wise addition with optional projection
    'multiplication',     # Element-wise multiplication with optional projection
    'gated',              # Learned gating mechanism
    'attention_pooling',  # Attention-based pooling and fusion
    'bilinear',           # Bilinear pooling
    'tensor_fusion'       # Multi-dimensional tensor fusion
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='MultiModalFusion')
class MultiModalFusion(keras.layers.Layer):
    """General-purpose configurable multi-modal fusion layer.

    This layer enables flexible fusion between multiple modalities using various
    strategies including cross-attention, concatenation, gating, and tensor fusion.

    Args:
        dim: Feature dimension for the fused representation.
        fusion_strategy: Strategy for combining modalities.
        num_fusion_layers: Number of fusion blocks (only for iterative strategies).
        attention_config: Configuration dict for attention layers.
        ffn_type: Type of feed-forward network to use.
        ffn_config: Configuration dict for FFN layers.
        norm_type: Type of normalization to apply.
        norm_config: Configuration dict for normalization layers.
        num_tensor_projections: Number of projections for tensor fusion.
        dropout_rate: Dropout probability for regularization.
        use_residual: Whether to use residual connections.
        activation: Activation function for projections.
        kernel_initializer: Initializer for weight matrices.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for weight matrices.
        bias_regularizer: Regularizer for bias vectors.

    Raises:
        ValueError: If configuration parameters are invalid.

    Example:
        >>> # Create a cross-attention fusion layer
        >>> fusion = MultiModalFusion(
        ...     dim=768,
        ...     fusion_strategy='cross_attention',
        ...     num_fusion_layers=2
        ... )
        >>> # Apply to vision and text features
        >>> vision_features = keras.random.normal((2, 196, 768))
        >>> text_features = keras.random.normal((2, 50, 768))
        >>> fused = fusion([vision_features, text_features])
    """

    def __init__(
        self,
        dim: int = 768,
        fusion_strategy: FusionStrategy = 'cross_attention',
        num_fusion_layers: int = 1,
        attention_config: Optional[Dict[str, Any]] = None,
        ffn_type: FFNType = 'mlp',
        ffn_config: Optional[Dict[str, Any]] = None,
        norm_type: NormalizationType = 'layer_norm',
        norm_config: Optional[Dict[str, Any]] = None,
        num_tensor_projections: int = 8,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        activation: Union[str, Callable] = 'gelu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ) -> None:
        """Initialize the MultiModalFusion layer with configuration."""
        super().__init__(**kwargs)

        # Validate input parameters early to fail fast
        self._validate_init_params(
            dim, fusion_strategy, num_fusion_layers,
            num_tensor_projections, dropout_rate
        )

        # Store configuration - these will be used in build()
        self.dim = dim
        self.fusion_strategy = fusion_strategy
        self.num_fusion_layers = num_fusion_layers
        self.attention_config = attention_config or {}
        self.ffn_type = ffn_type
        self.ffn_config = ffn_config or {}
        self.norm_type = norm_type
        self.norm_config = norm_config or {}
        self.num_tensor_projections = num_tensor_projections
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        # Convert string activations/initializers/regularizers to objects
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Initialize layer containers - will be populated in build()
        self._init_layer_containers()

    def _validate_init_params(
        self,
        dim: int,
        fusion_strategy: str,
        num_fusion_layers: int,
        num_tensor_projections: int,
        dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            dim: Feature dimension.
            fusion_strategy: Fusion strategy name.
            num_fusion_layers: Number of fusion layers.
            num_tensor_projections: Number of tensor projections.
            dropout_rate: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if num_fusion_layers <= 0:
            raise ValueError(f"num_fusion_layers must be positive, got {num_fusion_layers}")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        if fusion_strategy == 'tensor_fusion' and num_tensor_projections <= 0:
            raise ValueError(
                f"num_tensor_projections must be positive for tensor_fusion, "
                f"got {num_tensor_projections}"
            )

        # Only iterative strategies support multiple fusion layers
        iterative_strategies = {'cross_attention'}
        if fusion_strategy not in iterative_strategies and num_fusion_layers > 1:
            raise ValueError(
                f"num_fusion_layers > 1 is only supported for iterative strategies "
                f"{iterative_strategies}, but got strategy '{fusion_strategy}'"
            )

    def _init_layer_containers(self) -> None:
        """Initialize containers for sublayers.

        Note:
            We use lists to store sublayers instead of creating them here.
            Sublayers are created in build() when we know the input shapes.
        """
        self.fusion_layers = []      # Main fusion layers
        self.projection_layers = []  # Dense projection layers
        self.norm_layers = []        # Normalization layers
        self.ffn_layers = []         # Feed-forward network layers
        self.gate_layers = []        # Gating mechanism layers
        self.dropout_layers = []     # Dropout layers

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Create and build all sublayers based on input shapes.

        This method is called automatically the first time the layer is called.
        We create all sublayers here because we need to know the input shapes.

        Args:
            input_shape: List of shapes for each modality input.
                        Each shape should be (batch_size, sequence_length, dim).

        Raises:
            ValueError: If input shapes are invalid.
        """
        if self.built:
            return

        # Validate input shapes
        self._validate_input_shapes(input_shape)

        # Build strategy-specific sublayers
        strategy_builders = {
            'cross_attention': self._build_cross_attention,
            'concatenation': self._build_concatenation,
            'addition': self._build_elementwise,
            'multiplication': self._build_elementwise,
            'gated': self._build_gated,
            'attention_pooling': self._build_attention_pooling,
            'bilinear': self._build_bilinear,
            'tensor_fusion': self._build_tensor_fusion
        }

        builder = strategy_builders.get(self.fusion_strategy)
        if builder is None:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Call the appropriate builder
        builder(input_shape)

        # Mark as built
        super().build(input_shape)

    def _validate_input_shapes(self, input_shape: Union[Tuple, List]) -> None:
        """Validate that input shapes are compatible.

        Args:
            input_shape: List of input shapes.

        Raises:
            ValueError: If shapes are invalid or incompatible.
        """
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("Expected list or tuple of input shapes")

        if not input_shape or not hasattr(input_shape[0], '__len__'):
            raise ValueError("Expected non-empty list of tensor shapes")

        if len(input_shape) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(input_shape)}")

        # Check each modality shape
        for i, shape in enumerate(input_shape):
            if len(shape) != 3:
                raise ValueError(
                    f"Expected 3D shape (batch, seq, dim) for modality {i}, "
                    f"got shape {shape}"
                )
            if shape[-1] != self.dim:
                raise ValueError(
                    f"Modality {i} dimension {shape[-1]} doesn't match "
                    f"expected dim {self.dim}"
                )

    def _build_cross_attention(self, input_shape: List[Tuple]) -> None:
        """Build layers for cross-attention fusion strategy.

        Creates multiple fusion blocks, each containing cross-attention
        between all modality pairs, normalization, and FFN layers.

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        for layer_idx in range(self.num_fusion_layers):
            # Create a container to group layers for this fusion block
            # This helps with organization and serialization
            block = keras.layers.Layer(name=f'fusion_block_{layer_idx}')

            for i in range(num_modalities):
                # Create cross-attention layers for modality i attending to others
                attn_layers = []
                for j in range(num_modalities):
                    if i != j:  # Skip self-attention
                        attn_config = self.attention_config.copy()
                        attn_config.setdefault('dim', self.dim)

                        attn_layer = create_attention_layer(
                            'multi_head_cross',
                            name=f'cross_attn_{layer_idx}_{i}_to_{j}',
                            **attn_config
                        )
                        # Build with query from i and key/value from j
                        attn_layer.build([input_shape[i], input_shape[j]])
                        attn_layers.append(attn_layer)

                # Store attention layers as attribute on the block
                setattr(block, f'attention_{i}', attn_layers)

                # Create normalization layer for this modality
                norm_layer = create_normalization_layer(
                    self.norm_type,
                    name=f'norm_{layer_idx}_{i}',
                    **self.norm_config
                )
                norm_layer.build(input_shape[i])
                setattr(block, f'norm_{i}', norm_layer)

                # Create FFN layer for this modality
                ffn_config = self.ffn_config.copy()
                ffn_config.setdefault('hidden_dim', self.dim * 4)
                ffn_config['output_dim'] = self.dim

                ffn_layer = create_ffn_layer(
                    self.ffn_type,
                    name=f'ffn_{layer_idx}_{i}',
                    **ffn_config
                )
                ffn_layer.build(input_shape[i])
                setattr(block, f'ffn_{i}', ffn_layer)

            self.fusion_layers.append(block)

    def _build_concatenation(self, input_shape: List[Tuple]) -> None:
        """Build layers for concatenation fusion strategy.

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        # Create projection layer for concatenated features
        proj_layer = keras.layers.Dense(
            self.dim,
            activation=self.activation,
            name='concat_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # Calculate concatenated shape
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj_layer.build(tuple(concat_shape))
        self.projection_layers.append(proj_layer)

        # Create normalization layer
        norm_layer = create_normalization_layer(
            self.norm_type,
            name='concat_norm',
            **self.norm_config
        )
        output_shape = proj_layer.compute_output_shape(tuple(concat_shape))
        norm_layer.build(output_shape)
        self.norm_layers.append(norm_layer)

        # Create dropout layer
        self.dropout_layers.append(
            keras.layers.Dropout(self.dropout_rate, name='concat_dropout')
        )

    def _build_elementwise(self, input_shape: List[Tuple]) -> None:
        """Build layers for element-wise fusion strategies (add/multiply).

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        # Create alignment projections if more than 2 modalities
        # This ensures all modalities are in the same semantic space
        if num_modalities > 2:
            for i in range(num_modalities):
                proj = keras.layers.Dense(
                    self.dim,
                    name=f'align_projection_{i}',
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer
                )
                proj.build(input_shape[i])
                self.projection_layers.append(proj)

        # Create normalization layer
        norm_layer = create_normalization_layer(
            self.norm_type,
            name='elementwise_norm',
            **self.norm_config
        )
        norm_layer.build(input_shape[0])
        self.norm_layers.append(norm_layer)

        # Create FFN layer for final processing
        ffn_config = self.ffn_config.copy()
        ffn_config.setdefault('hidden_dim', self.dim * 4)
        ffn_config['output_dim'] = self.dim

        ffn_layer = create_ffn_layer(
            self.ffn_type,
            name='elementwise_ffn',
            **ffn_config
        )
        ffn_layer.build(input_shape[0])
        self.ffn_layers.append(ffn_layer)

    def _build_gated(self, input_shape: List[Tuple]) -> None:
        """Build layers for gated fusion strategy.

        Creates gates that learn to weight each modality's contribution.

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        # Create gating layers for each modality
        for i in range(num_modalities):
            gate = keras.layers.Dense(
                self.dim,
                activation='sigmoid',  # Sigmoid for gating [0, 1]
                name=f'gate_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
            gate.build(input_shape[i])
            self.gate_layers.append(gate)

        # Create projection for gated concatenated features
        proj = keras.layers.Dense(
            self.dim,
            activation=self.activation,
            name='gated_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj.build(tuple(concat_shape))
        self.projection_layers.append(proj)

        # Create normalization layer
        norm = create_normalization_layer(
            self.norm_type,
            name='gated_norm',
            **self.norm_config
        )
        output_shape = proj.compute_output_shape(tuple(concat_shape))
        norm.build(output_shape)
        self.norm_layers.append(norm)

    def _build_attention_pooling(self, input_shape: List[Tuple]) -> None:
        """Build layers for attention pooling fusion strategy.

        Uses self-attention to pool features before fusion.

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        # Create self-attention layers for pooling each modality
        for i in range(num_modalities):
            attn_config = self.attention_config.copy()
            attn_config.setdefault('dim', self.dim)

            attn_layer = create_attention_layer(
                'multi_head_cross',
                name=f'pool_attention_{i}',
                **attn_config
            )
            # Self-attention: same input for query and key/value
            attn_layer.build([input_shape[i], input_shape[i]])
            self.fusion_layers.append(attn_layer)

        # Create projection for pooled features
        proj = keras.layers.Dense(
            self.dim,
            activation=self.activation,
            name='pool_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        # After pooling, sequence dimension is removed
        pooled_shape = (input_shape[0][0], self.dim * num_modalities)
        proj.build(pooled_shape)
        self.projection_layers.append(proj)

    def _build_bilinear(self, input_shape: List[Tuple]) -> None:
        """Build layers for bilinear pooling fusion strategy.

        Computes outer product between modalities.

        Args:
            input_shape: List of input shapes for each modality.

        Raises:
            ValueError: If not exactly 2 modalities provided.
        """
        num_modalities = len(input_shape)
        if num_modalities != 2:
            raise ValueError(
                f"Bilinear fusion requires exactly 2 modalities, got {num_modalities}"
            )

        # Create projection for flattened bilinear features
        proj = keras.layers.Dense(
            self.dim,
            activation=self.activation,
            name='bilinear_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # Shape after outer product and flattening
        batch_size, seq_len, _ = input_shape[0]
        bilinear_flat_shape = (batch_size, seq_len, self.dim * self.dim)
        proj.build(bilinear_flat_shape)
        self.projection_layers.append(proj)

        # Create normalization layer
        norm = create_normalization_layer(
            self.norm_type,
            name='bilinear_norm',
            **self.norm_config
        )
        output_shape = proj.compute_output_shape(bilinear_flat_shape)
        norm.build(output_shape)
        self.norm_layers.append(norm)

    def _build_tensor_fusion(self, input_shape: List[Tuple]) -> None:
        """Build layers for tensor fusion strategy.

        Creates multiple projections to model higher-order interactions.

        Args:
            input_shape: List of input shapes for each modality.
        """
        num_modalities = len(input_shape)

        # Calculate concatenated shape
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities

        # Create multiple projection layers for tensor decomposition
        for i in range(self.num_tensor_projections):
            proj = keras.layers.Dense(
                self.dim,
                activation=self.activation,
                name=f'tensor_proj_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
            proj.build(tuple(concat_shape))
            self.projection_layers.append(proj)

        # Create final projection layer
        final_proj = keras.layers.Dense(
            self.dim,
            name='tensor_final_proj',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        final_concat_shape = list(input_shape[0])
        final_concat_shape[-1] = self.dim * self.num_tensor_projections
        final_proj.build(tuple(final_concat_shape))
        self.projection_layers.append(final_proj)

    def call(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """Apply the fusion strategy to combine multiple modalities.

        Args:
            inputs: List or tuple of tensors, one for each modality.
                   Each tensor shape: (batch_size, sequence_length, dim)
            training: Boolean flag for training mode (affects dropout).

        Returns:
            Fused representation(s). Shape depends on fusion strategy:
            - cross_attention: Tuple of tensors, one per modality
            - attention_pooling: (batch_size, dim)
            - Others: (batch_size, sequence_length, dim)

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Expected list or tuple of input tensors")
        if len(inputs) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(inputs)}")

        # Dispatch to strategy-specific implementation
        strategy_handlers = {
            'cross_attention': self._call_cross_attention,
            'concatenation': self._call_concatenation,
            'addition': self._call_elementwise,
            'multiplication': self._call_elementwise,
            'gated': self._call_gated,
            'attention_pooling': self._call_attention_pooling,
            'bilinear': self._call_bilinear,
            'tensor_fusion': self._call_tensor_fusion
        }

        handler = strategy_handlers.get(self.fusion_strategy)
        if handler is None:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return handler(inputs, training)

    def _call_cross_attention(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, ...]:
        """Apply cross-attention fusion.

        Each modality attends to all others, results are averaged,
        then passed through normalization and FFN.

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Tuple of refined features for each modality.
        """
        outputs = list(inputs)
        num_modalities = len(inputs)

        # Apply multiple fusion blocks iteratively
        for layer_idx in range(self.num_fusion_layers):
            block = self.fusion_layers[layer_idx]
            new_outputs = []

            for i in range(num_modalities):
                # Collect attention results from all other modalities
                attended_features = []
                attention_layers = getattr(block, f'attention_{i}')
                attention_idx = 0

                for j in range(num_modalities):
                    if i != j:
                        # Modality i attends to modality j
                        attention_layer = attention_layers[attention_idx]
                        attended = attention_layer(
                            query_input=outputs[i],
                            kv_input=outputs[j],
                            training=training
                        )
                        attended_features.append(attended)
                        attention_idx += 1

                # Average attention results from all other modalities
                # Stack along new axis then average
                combined = keras.ops.mean(
                    keras.ops.stack(attended_features, axis=0),
                    axis=0
                )

                # Add residual connection if enabled
                if self.use_residual:
                    combined = outputs[i] + combined

                # Apply normalization
                norm_layer = getattr(block, f'norm_{i}')
                normalized = norm_layer(combined, training=training)

                # Apply FFN with optional residual
                ffn_layer = getattr(block, f'ffn_{i}')
                ffn_out = ffn_layer(normalized, training=training)

                if self.use_residual:
                    output = normalized + ffn_out
                else:
                    output = ffn_out

                new_outputs.append(output)

            outputs = new_outputs

        return tuple(outputs)

    def _call_concatenation(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply concatenation fusion.

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after concatenation and projection.
        """
        # Concatenate along feature dimension
        concatenated = keras.ops.concatenate(inputs, axis=-1)

        # Project to target dimension
        output = self.projection_layers[0](concatenated, training=training)

        # Apply normalization
        output = self.norm_layers[0](output, training=training)

        # Apply dropout for regularization
        output = self.dropout_layers[0](output, training=training)

        return output

    def _call_elementwise(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply element-wise fusion (addition or multiplication).

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after element-wise operation.
        """
        # Apply alignment projections if available
        if self.projection_layers:
            aligned = [
                proj(inp, training=training)
                for proj, inp in zip(self.projection_layers, inputs)
            ]
        else:
            aligned = list(inputs)

        # Apply element-wise operation
        if self.fusion_strategy == 'addition':
            # Sum all aligned modalities
            output = keras.ops.sum(
                keras.ops.stack(aligned, axis=0),
                axis=0
            )
        else:  # multiplication
            # Element-wise product of all modalities
            output = aligned[0]
            for inp in aligned[1:]:
                output = keras.ops.multiply(output, inp)

        # Apply normalization
        output = self.norm_layers[0](output, training=training)

        # Apply FFN for final processing
        output = self.ffn_layers[0](output, training=training)

        return output

    def _call_gated(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply gated fusion with learned importance weights.

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after gating and projection.
        """
        # Apply learned gates to each modality
        gated_features = []
        for i, inp in enumerate(inputs):
            # Compute gate values (sigmoid ensures [0, 1] range)
            gate_values = self.gate_layers[i](inp, training=training)
            # Apply gate element-wise
            gated = keras.ops.multiply(inp, gate_values)
            gated_features.append(gated)

        # Concatenate gated features
        concatenated = keras.ops.concatenate(gated_features, axis=-1)

        # Project to target dimension
        output = self.projection_layers[0](concatenated, training=training)

        # Apply normalization
        output = self.norm_layers[0](output, training=training)

        return output

    def _call_attention_pooling(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply attention-based pooling fusion.

        Uses self-attention to create pooled representations.

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after attention pooling.
        """
        pooled_features = []

        for i, inp in enumerate(inputs):
            # Apply self-attention for context-aware pooling
            attention_layer = self.fusion_layers[i]
            attended_output = attention_layer(
                query_input=inp,
                kv_input=inp,
                training=training
            )

            # Global average pooling over sequence dimension
            pooled = keras.ops.mean(attended_output, axis=1)
            pooled_features.append(pooled)

        # Concatenate pooled features from all modalities
        concatenated = keras.ops.concatenate(pooled_features, axis=-1)

        # Project to target dimension
        output = self.projection_layers[0](concatenated, training=training)

        return output

    def _call_bilinear(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply bilinear pooling fusion.

        Computes outer product between two modalities.

        Args:
            inputs: List of exactly 2 modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after bilinear pooling.
        """
        x1, x2 = inputs

        # Compute outer product efficiently
        # Expand dims for broadcasting: x1 -> [..., d1, 1], x2 -> [..., 1, d2]
        x1_expanded = keras.ops.expand_dims(x1, axis=-1)
        x2_expanded = keras.ops.expand_dims(x2, axis=-2)

        # Outer product via broadcasting
        bilinear = keras.ops.multiply(x1_expanded, x2_expanded)

        # Flatten the outer product matrix
        batch_size = keras.ops.shape(bilinear)[0]
        seq_len = keras.ops.shape(bilinear)[1]
        bilinear_flat = keras.ops.reshape(
            bilinear,
            [batch_size, seq_len, -1]
        )

        # Project to target dimension
        output = self.projection_layers[0](bilinear_flat, training=training)

        # Apply normalization
        output = self.norm_layers[0](output, training=training)

        return output

    def _call_tensor_fusion(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply tensor fusion with multiple projections.

        Models higher-order interactions through tensor decomposition.

        Args:
            inputs: List of modality tensors.
            training: Training mode flag.

        Returns:
            Fused tensor after tensor fusion.
        """
        # Concatenate all modalities
        concatenated = keras.ops.concatenate(inputs, axis=-1)

        # Apply multiple projections (tensor decomposition)
        projections = []
        for i in range(self.num_tensor_projections):
            proj_layer = self.projection_layers[i]
            projection = proj_layer(concatenated, training=training)
            projections.append(projection)

        # Concatenate all projections
        if projections:
            combined = keras.ops.concatenate(projections, axis=-1)
        else:
            combined = concatenated

        # Final projection to target dimension
        output = self.projection_layers[-1](combined, training=training)

        return output

    def compute_output_shape(
        self,
        input_shape: Union[Tuple, List[Tuple]]
    ) -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape for given input shapes.

        Args:
            input_shape: List of input shapes for each modality.

        Returns:
            Output shape(s) depending on fusion strategy.
        """
        if self.fusion_strategy == 'cross_attention':
            # Returns same shape for each modality
            return input_shape
        elif self.fusion_strategy == 'attention_pooling':
            # Sequence dimension is pooled
            return (input_shape[0][0], self.dim)
        else:
            # Most strategies preserve batch and sequence dimensions
            return (input_shape[0][0], input_shape[0][1], self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'fusion_strategy': self.fusion_strategy,
            'num_fusion_layers': self.num_fusion_layers,
            'attention_config': self.attention_config,
            'ffn_type': self.ffn_type,
            'ffn_config': self.ffn_config,
            'norm_type': self.norm_type,
            'norm_config': self.norm_config,
            'num_tensor_projections': self.num_tensor_projections,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiModalFusion':
        """Create layer from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            New MultiModalFusion instance.
        """
        # Deserialize activation, initializers, and regularizers
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(config['kernel_initializer'])
        config['bias_initializer'] = keras.initializers.deserialize(config['bias_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(config.get('kernel_regularizer'))
        config['bias_regularizer'] = keras.regularizers.deserialize(config.get('bias_regularizer'))

        return cls(**config)

# ---------------------------------------------------------------------
