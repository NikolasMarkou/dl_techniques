"""Implements a unified framework for multi-modal information fusion.

This layer provides a modular and extensible framework for combining feature
representations from multiple, heterogeneous data modalities (e.g., vision_heads,
language, audio). It serves as a factory that implements several distinct
fusion strategies, each grounded in different theoretical assumptions about
how cross-modal interactions should be modeled. The primary goal is to learn a
joint representation that is more informative than the sum of its parts.

Architectural and Conceptual Underpinnings:

The layer is designed around a strategy pattern, where a high-level interface
delegates the core fusion logic to one of several underlying implementations.
This allows for easy experimentation with different approaches to modeling the
complex relationships between modalities. The core strategies fall into several
conceptual categories:

1.  **Early Fusion (Concatenation)**: This is the simplest strategy, where
    feature vectors from all modalities are concatenated and then passed
    through a learned linear projection.
        `Y = W[X₁; X₂; ...; Xₙ] + b`
    The intuition is to preserve all information from all modalities and allow a
    subsequent feed-forward network to learn the complex cross-modal
    interactions implicitly. It is a robust and often surprisingly effective
    baseline.

2.  **Element-wise Fusion (Addition & Multiplication)**: These strategies assume
    that the input modalities have been projected into a semantically aligned
    space.
    -   **Addition**: `Y = W(X₁ + X₂ + ... + Xₙ) + b`. This assumes modalities
        provide complementary, independent evidence.
    -   **Multiplication**: `Y = W(X₁ ⊙ X₂ ⊙ ... ⊙ Xₙ) + b`. This captures
        synergistic interactions, where the presence of a feature in one
        modality amplifies orgates the effect of a feature in another.

3.  **Interaction-based Fusion (Bilinear & Tensor)**: These methods explicitly
    model higher-order, multiplicative interactions between all pairs of
    features across modalities.
    -   **Bilinear Pooling**: For two modalities, this is conceptually
        equivalent to computing an outer product and projecting it:
        `Y = W(X₁ ⊗ X₂) + b`. It captures a rich set of pairwise feature
        interactions but is computationally expensive, scaling quadratically
        with the feature dimension (`O(d²)`).
    -   **Tensor Fusion**: An extension of bilinear pooling that creates a
        multi-dimensional tensor to model higher-order interactions among
        multiple modalities, often using low-rank approximations to remain
        computationally tractable.

4.  **Attention-based Fusion (Cross-Attention)**: This is the most powerful and
    flexible strategy, forming the basis of modern multi-modal Transformers.
    Each modality acts as a query to attend to the others, which serve as keys
    and values.
        `X'₁ = X₁ + Attention(Q=X₁, K=X₂, V=X₂)`
        `X'₂ = X₂ + Attention(Q=X₂, K=X₁, V=X₁)`
    This allows each modality to dynamically select and integrate the most
    relevant information from the others in a context-dependent manner. It
    excels at modeling asymmetric relationships where, for example, a textual
    query needs to ground itself in specific regions of an image.

By providing these varied strategies within a single interface, this layer
enables systematic exploration of the optimal way to integrate multi-modal
information for a given task.

References:
    - Baltrusaitis, T., et al. (2018). Multimodal Machine Learning: A Survey
      and Taxonomy. *IEEE Transactions on Pattern Analysis and Machine
      Intelligence*.
    - Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
    - Zadeh, A., et al. (2017). Tensor Fusion Network for Multimodal Sentiment
      Analysis. *EMNLP*.
    - Lu, J., et al. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic
      Representations for Vision-and-Language Tasks. *NeurIPS*.
"""

import keras
from keras import layers, initializers, regularizers, activations
from typing import Optional, Union, List, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ffn.factory import create_ffn_layer, FFNType
from .attention.factory import create_attention_layer
from .norms.factory import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------

FusionStrategy = Literal[
    'cross_attention',   # Bidirectional cross-attention between modalities
    'concatenation',     # Concatenate and project
    'addition',          # Element-wise addition with optional projection
    'multiplication',    # Element-wise multiplication with optional projection
    'gated',             # Learned gating mechanism
    'attention_pooling', # Attention-based pooling and fusion
    'bilinear',          # Bilinear pooling
    'tensor_fusion'      # Multi-dimensional tensor fusion
]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiModalFusion(keras.layers.Layer):
    """
    General-purpose configurable multi-modal fusion layer.

    This layer enables flexible fusion between multiple modalities using various strategies
    including cross-attention, concatenation, gating, and tensor fusion. It uses factory
    patterns for creating configurable FFN and normalization sublayers, and is hardcoded to
    use the 'multi_head_cross' attention layer for all attention-based strategies.
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
            activation: Union[str, keras.KerasTensor] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # --- Store configuration ONLY in __init__ ---
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_fusion_layers <= 0:
            raise ValueError(f"num_fusion_layers must be positive, got {num_fusion_layers}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if fusion_strategy == 'tensor_fusion' and num_tensor_projections <= 0:
            raise ValueError("num_tensor_projections must be positive for tensor_fusion strategy.")

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
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        iterative_strategies = {'cross_attention'}
        if self.fusion_strategy not in iterative_strategies and self.num_fusion_layers > 1:
            raise ValueError(
                f"num_fusion_layers > 1 is only supported for iterative strategies "
                f"like {iterative_strategies}, but strategy is '{self.fusion_strategy}'."
            )

        self.fusion_layers = []
        self.projection_layers = []
        self.norm_layers = []
        self.ffn_layers = []
        self.gate_layers = []
        self.dropout_layers = []

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """Create and explicitly build all sub-layers based on the input shape."""
        if self.built:
            return

        if not isinstance(input_shape, (list, tuple)) or not input_shape or not hasattr(input_shape[0], '__len__'):
             raise ValueError("Expected list or tuple of input tensors")
        if len(input_shape) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(input_shape)}")
        for i, shape in enumerate(input_shape):
            if len(shape) != 3:
                raise ValueError(f"Expected 3D shapes, got modality {i}: {shape}")
            if shape[-1] != self.dim:
                raise ValueError(f"Modality {i} dimension {shape[-1]} != dim {self.dim}")

        num_modalities = len(input_shape)

        # --- Create and Build sub-layers here ---
        strategy = self.fusion_strategy
        if strategy == 'cross_attention':
            self._build_and_create_cross_attention(input_shape)
        elif strategy == 'concatenation':
            self._build_and_create_concatenation(input_shape)
        elif strategy in ['addition', 'multiplication']:
            self._build_and_create_elementwise(input_shape)
        elif strategy == 'gated':
            self._build_and_create_gated(input_shape)
        elif strategy == 'attention_pooling':
            self._build_and_create_attention_pooling(input_shape)
        elif strategy == 'bilinear':
            self._build_and_create_bilinear(input_shape)
        elif strategy == 'tensor_fusion':
            self._build_and_create_tensor_fusion(input_shape)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        super().build(input_shape)

    def _build_and_create_cross_attention(self, input_shape):
        num_modalities = len(input_shape)
        for layer_idx in range(self.num_fusion_layers):
            # Use a Layer as a container for better tracking during serialization
            block_container = keras.layers.Layer(name=f'fusion_block_{layer_idx}')
            for i in range(num_modalities):
                # Attention Layers
                attn_layers_for_modality_i = []
                for j in range(num_modalities):
                    if i != j:
                        attn_config = self.attention_config.copy()
                        attn_config.setdefault('dim', self.dim)
                        attn_layer = create_attention_layer('multi_head_cross', name=f'cross_attention_{layer_idx}_{i}_{j}', **attn_config)
                        attn_layer.build([input_shape[i], input_shape[j]])
                        attn_layers_for_modality_i.append(attn_layer)
                # Set the list of layers as an attribute on the container
                setattr(block_container, f'attention_{i}', attn_layers_for_modality_i)

                # Norm Layer
                norm_layer = create_normalization_layer(self.norm_type, name=f'norm_{layer_idx}_{i}', **self.norm_config.copy())
                norm_layer.build(input_shape[i])
                setattr(block_container, f'norm_{i}', norm_layer)

                # FFN Layer
                ffn_config = self.ffn_config.copy()
                ffn_config.setdefault('hidden_dim', self.dim * 4)
                ffn_config['output_dim'] = self.dim
                ffn_layer = create_ffn_layer(self.ffn_type, name=f'ffn_{layer_idx}_{i}', **ffn_config)
                ffn_layer.build(input_shape[i])
                setattr(block_container, f'ffn_{i}', ffn_layer)

            # Append the container, which now holds and tracks all sub-layers for this block
            self.fusion_layers.append(block_container)

    def _build_and_create_concatenation(self, input_shape):
        num_modalities = len(input_shape)
        proj_layer = layers.Dense(self.dim, activation=self.activation, name='concat_projection', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj_layer.build(tuple(concat_shape))
        self.projection_layers.append(proj_layer)

        norm_layer = create_normalization_layer(self.norm_type, name='concat_norm', **self.norm_config.copy())
        norm_layer.build(proj_layer.compute_output_shape(tuple(concat_shape)))
        self.norm_layers.append(norm_layer)

        self.dropout_layers.append(layers.Dropout(self.dropout_rate, name='concat_dropout'))

    def _build_and_create_elementwise(self, input_shape):
        num_modalities = len(input_shape)
        if num_modalities > 2:
            for i in range(num_modalities):
                proj = layers.Dense(self.dim, name=f'align_projection_{i}', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
                proj.build(input_shape[i])
                self.projection_layers.append(proj)

        norm_layer = create_normalization_layer(self.norm_type, name='elementwise_norm', **self.norm_config.copy())
        norm_layer.build(input_shape[0])
        self.norm_layers.append(norm_layer)

        ffn_config = self.ffn_config.copy()
        ffn_config.setdefault('hidden_dim', self.dim * 4)
        ffn_config['output_dim'] = self.dim
        ffn_layer = create_ffn_layer(self.ffn_type, name='elementwise_ffn', **ffn_config)
        ffn_layer.build(input_shape[0])
        self.ffn_layers.append(ffn_layer)

    def _build_and_create_gated(self, input_shape):
        num_modalities = len(input_shape)
        for i in range(num_modalities):
            gate = layers.Dense(self.dim, activation='sigmoid', name=f'gate_{i}', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
            gate.build(input_shape[i])
            self.gate_layers.append(gate)

        proj = layers.Dense(self.dim, activation=self.activation, name='gated_projection', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj.build(tuple(concat_shape))
        self.projection_layers.append(proj)

        norm = create_normalization_layer(self.norm_type, name='gated_norm', **self.norm_config.copy())
        norm.build(proj.compute_output_shape(tuple(concat_shape)))
        self.norm_layers.append(norm)

    def _build_and_create_attention_pooling(self, input_shape):
        num_modalities = len(input_shape)
        for i in range(num_modalities):
            attn_config = self.attention_config.copy()
            attn_config.setdefault('dim', self.dim)
            attn_layer = create_attention_layer('multi_head_cross', name=f'pool_attention_{i}', **attn_config)
            attn_layer.build([input_shape[i], input_shape[i]])
            self.fusion_layers.append(attn_layer)

        proj = layers.Dense(self.dim, activation=self.activation, name='pool_projection', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        proj.build((input_shape[0][0], self.dim * num_modalities))
        self.projection_layers.append(proj)

    def _build_and_create_bilinear(self, input_shape):
        num_modalities = len(input_shape)
        if num_modalities != 2:
            raise ValueError("Bilinear fusion currently supports only 2 modalities")

        proj = layers.Dense(self.dim, activation=self.activation, name='bilinear_projection', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        bilinear_flat_shape = (input_shape[0][0], input_shape[0][1], self.dim * self.dim)
        proj.build(bilinear_flat_shape)
        self.projection_layers.append(proj)

        norm = create_normalization_layer(self.norm_type, name='bilinear_norm', **self.norm_config.copy())
        norm.build(proj.compute_output_shape(bilinear_flat_shape))
        self.norm_layers.append(norm)

    def _build_and_create_tensor_fusion(self, input_shape):
        num_modalities = len(input_shape)
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities

        for i in range(self.num_tensor_projections):
            proj = layers.Dense(self.dim, activation=self.activation, name=f'tensor_proj_{i}', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
            proj.build(tuple(concat_shape))
            self.projection_layers.append(proj)

        final_proj = layers.Dense(self.dim, name='tensor_final_proj', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
        final_concat_shape = list(input_shape[0])
        final_concat_shape[-1] = self.dim * self.num_tensor_projections
        final_proj.build(tuple(final_concat_shape))
        self.projection_layers.append(final_proj)

    def call(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Expected list or tuple of input tensors")
        if len(inputs) < 2:
            raise ValueError("Expected at least 2 input modalities")

        strategy = self.fusion_strategy
        if strategy == 'cross_attention':
            return self._call_cross_attention(inputs, training)
        elif strategy == 'concatenation':
            return self._call_concatenation(inputs, training)
        elif strategy in ['addition', 'multiplication']:
            return self._call_elementwise(inputs, training)
        elif strategy == 'gated':
            return self._call_gated(inputs, training)
        elif strategy == 'attention_pooling':
            return self._call_attention_pooling(inputs, training)
        elif strategy == 'bilinear':
            return self._call_bilinear(inputs, training)
        elif strategy == 'tensor_fusion':
            return self._call_tensor_fusion(inputs, training)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _call_cross_attention(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, ...]:
        outputs = list(inputs)
        num_modalities = len(inputs)
        for layer_idx in range(self.num_fusion_layers):
            block_container = self.fusion_layers[layer_idx]
            new_outputs = []
            for i in range(num_modalities):
                attended_features = []
                attention_layers_for_i = getattr(block_container, f'attention_{i}')
                attention_idx = 0
                for j in range(num_modalities):
                    if i != j:
                        attention_layer = attention_layers_for_i[attention_idx]
                        attended = attention_layer(
                            query_input=outputs[i], kv_input=outputs[j], training=training
                        )
                        attended_features.append(attended)
                        attention_idx += 1
                combined = keras.ops.mean(keras.ops.stack(attended_features, axis=0), axis=0)
                if self.use_residual:
                    combined = outputs[i] + combined

                norm_layer = getattr(block_container, f'norm_{i}')
                ffn_layer = getattr(block_container, f'ffn_{i}')

                norm_out = norm_layer(combined, training=training)
                ffn_out = ffn_layer(norm_out, training=training)
                if self.use_residual:
                    combined = norm_out + ffn_out
                else:
                    combined = ffn_out
                new_outputs.append(combined)
            outputs = new_outputs
        return tuple(outputs)

    def _call_concatenation(self, inputs, training=None):
        concatenated = keras.ops.concatenate(inputs, axis=-1)
        output = self.projection_layers[0](concatenated)
        output = self.norm_layers[0](output, training=training)
        output = self.dropout_layers[0](output, training=training)
        return output

    def _call_elementwise(self, inputs, training=None):
        aligned = [
            proj(inp) for proj, inp in zip(self.projection_layers, inputs)
        ] if self.projection_layers else list(inputs)
        if self.fusion_strategy == 'addition':
            output = keras.ops.sum(keras.ops.stack(aligned, axis=0), axis=0)
        else:
            output = aligned[0]
            for inp in aligned[1:]:
                output = keras.ops.multiply(output, inp)
        output = self.norm_layers[0](output, training=training)
        output = self.ffn_layers[0](output, training=training)
        return output

    def _call_gated(self, inputs, training=None):
        gated_features = [
            keras.ops.multiply(inp, self.gate_layers[i](inp, training=training))
            for i, inp in enumerate(inputs)
        ]
        concatenated = keras.ops.concatenate(gated_features, axis=-1)
        output = self.projection_layers[0](concatenated)
        output = self.norm_layers[0](output, training=training)
        return output

    def _call_attention_pooling(self, inputs, training=None):
        pooled_features = []
        for i, inp in enumerate(inputs):
            attended_output = self.fusion_layers[i](
                query_input=inp, kv_input=inp, training=training
            )
            pooled = keras.ops.mean(attended_output, axis=1)
            pooled_features.append(pooled)
        concatenated = keras.ops.concatenate(pooled_features, axis=-1)
        output = self.projection_layers[0](concatenated)
        return output

    def _call_bilinear(self, inputs, training=None):
        x1, x2 = inputs
        x1_expanded = keras.ops.expand_dims(x1, axis=-1)
        x2_expanded = keras.ops.expand_dims(x2, axis=-2)
        bilinear = keras.ops.multiply(x1_expanded, x2_expanded)
        batch_size, seq_len = keras.ops.shape(bilinear)[:2]
        bilinear_flat = keras.ops.reshape(bilinear, [batch_size, seq_len, -1])
        output = self.projection_layers[0](bilinear_flat)
        output = self.norm_layers[0](output, training=training)
        return output

    def _call_tensor_fusion(self, inputs, training=None):
        concatenated = keras.ops.concatenate(inputs, axis=-1)
        projections = [
            layer(concatenated) for layer in self.projection_layers[:-1]
        ]
        combined = keras.ops.concatenate(projections, axis=-1) if projections else concatenated
        output = self.projection_layers[-1](combined)
        return output


    def compute_output_shape(self, input_shape):
        if self.fusion_strategy == 'cross_attention':
            return input_shape
        if self.fusion_strategy == 'attention_pooling':
            # Output is (batch_size, dim)
            return input_shape[0][0], self.dim
        # All other strategies output (batch_size, sequence_length, dim)
        return input_shape[0][0], input_shape[0][1], self.dim

    def get_config(self):
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
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config