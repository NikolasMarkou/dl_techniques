"""
A configurable multi-paradigm Graph Neural Network.

This layer provides a unified and flexible framework for graph representation
learning by encapsulating several foundational Graph Neural Network (GNN)
architectures. It is designed as a composite layer that stacks multiple message-
passing blocks, allowing for the construction of deep GNNs for complex graph-
structured data.

The core principle of a GNN is to learn node representations by iteratively
aggregating information from local neighborhoods. This layer abstracts this
process into a configurable pipeline, enabling seamless switching between
different message-passing, normalization, and aggregation strategies.

Architectural Overview:
The layer operates as a sequence of `num_layers` GNN blocks. Each block
performs a transformation on the node features, informed by the graph's
topology as defined by the adjacency matrix. The data flow within a single
block is as follows:

1.  **Message Passing**: The central mechanism where each node gathers
    information from its neighbors. This layer supports four distinct paradigms
    (GCN, GraphSAGE, GAT, GIN), each with a unique inductive bias.
2.  **Non-linear Activation**: An activation function is applied to the aggregated
    messages to introduce non-linearity, enabling the model to learn complex
    functions.
3.  **Regularization**: Dropout is applied to prevent overfitting on both the
    node features and the graph structure.
4.  **Residual Connection**: A skip connection adds the input of the block to
    its output, facilitating gradient flow and enabling the training of deeper
    GNNs.
5.  **Normalization**: A normalization layer (e.g., LayerNorm, RMSNorm) is
    applied to stabilize training and improve convergence.

After the final GNN block, an optional aggregation step can be applied to
pool node-level representations into a single graph-level embedding.

Foundational Mathematics:
The layer's behavior is determined by the chosen `message_passing` scheme. Let
`h_i` be the feature vector for node `i` and `N(i)` be its neighbors.

-   **GCN (Graph Convolutional Network)**: Treats message passing as a form of
    spectral convolution on the graph. The update rule is a simplified,
    spatially-localized version of this:
    `h'_i = σ(Σ_{j ∈ N(i) ∪ {i}} (1/√(deg(i)deg(j))) * W @ h_j)`
    This is a weighted average of a node's features and its neighbors', where
    the weights are determined by node degrees to ensure stability.

-   **GraphSAGE (Graph Sample and AGgregate)**: An inductive framework that
    explicitly separates the aggregation of neighbor information from the
    update step.
    `h'_i = σ(W_self @ h_i + W_neighbor @ AGG({h_j | j ∈ N(i)}))`
    This separation allows it to generalize to unseen nodes, as the aggregation
    function `AGG` (e.g., mean, max) is independent of the global graph
    structure.

-   **GAT (Graph Attention Network)**: Assigns learnable, data-dependent
    importance weights (attention) to neighbors, rather than using fixed
    coefficients.
    `h'_i = σ(Σ_{j ∈ N(i)} α_{ij} * W @ h_j)`
    The attention coefficients `α_{ij}` are computed based on the features of
    nodes `i` and `j`, allowing the model to focus on more relevant neighbors.

-   **GIN (Graph Isomorphism Network)**: Designed to be a maximally powerful GNN
    for distinguishing graph structures. Its update rule is proven to be as
    expressive as the Weisfeiler-Lehman graph isomorphism test.
    `h'_i = MLP((1 + ε) * h_i + Σ_{j ∈ N(i)} h_j)`
    It uses a Multi-Layer Perceptron (MLP) as a universal function approximator
    on the aggregated neighborhood information, with a learnable parameter `ε`
    to balance self-features versus neighbor features.

References:
Each message passing scheme is based on a seminal paper in the field:
-   **GCN**: Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
    with Graph Convolutional Networks. ICLR.
-   **GraphSAGE**: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive
    Representation Learning on Large Graphs. NeurIPS.
-   **GAT**: Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
-   **GIN**: Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful
    are Graph Neural Networks? ICLR.

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn.mlp import MLPBlock
from ..norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GraphNeuralNetworkLayer(keras.layers.Layer):
    """
    Complete configurable Graph Neural Network for concept relationship modeling.

    This layer implements various GNN architectures including GCN (Graph Convolutional Network),
    GraphSAGE (Graph Sample and Aggregate), GAT (Graph Attention Network), and GIN (Graph
    Isomorphism Network) with configurable message passing, aggregation, and normalization.
    It demonstrates proper sub-layer management and serialization patterns for complex
    composite layers in Keras 3.

    **Intent**: Provide a flexible, production-ready GNN implementation that can be easily
    configured for different graph learning tasks while maintaining proper Keras patterns
    for serialization and weight management.

    **Architecture**:
    ```
    Input: (node_features, adjacency_matrix)
           ↓
    [For each GNN layer i=1..num_layers]:
        Message Passing (GCN/GraphSAGE/GAT/GIN)
               ↓
        Activation Function
               ↓
        Dropout(dropout_rate)
               ↓
        Residual Addition (if use_residual=True)
               ↓
        Normalization (Layer/RMS/Batch)
           ↓
    Final Aggregation (mean/max/sum/attention)
           ↓
    Output: Updated node embeddings
    ```

    **Message Passing Schemes**:
    - **GCN**: H' = σ(ÂHW) where Â is normalized adjacency
    - **GraphSAGE**: H' = σ(W_self·H + W_neighbor·AGG(A·H))
    - **GAT**: H' = AttentionHeads(H, H) with graph structure
    - **GIN**: H' = MLP((1+ε)·H + Σ_neighbors)

    Args:
        concept_dim: Integer, dimension of concept/node embeddings. Must be positive.
            This determines the feature dimension throughout the network.
        num_layers: Integer, number of GNN layers to stack. Must be positive.
            More layers allow learning higher-order graph structures. Defaults to 3.
        message_passing: String, type of message passing mechanism to use.
            Options: 'gcn', 'graphsage', 'gat', 'gin'. Each has different
            inductive biases and computational costs. Defaults to 'gcn'.
        aggregation: String, type of final aggregation over nodes.
            Options: 'mean', 'max', 'attention', 'sum', 'none'.
            Determines how node features are pooled. Defaults to 'attention'.
        normalization: String, type of normalization to apply after each layer.
            Options: 'none', 'batch', 'layer', 'rms'. Helps with training
            stability and convergence. Defaults to 'layer'.
        activation: String or callable, activation function to use.
            Standard Keras activation names or custom functions. Defaults to 'relu'.
        dropout_rate: Float between 0 and 1, dropout probability during training.
            Helps prevent overfitting on graph structure. Defaults to 0.1.
        use_residual: Boolean, whether to use residual connections between layers.
            Helps with gradient flow in deep networks. Defaults to True.
        num_attention_heads: Integer, number of attention heads for GAT and
            attention aggregation. More heads capture different relationships.
            Must be positive and divide concept_dim. Defaults to 4.
        epsilon: Float, small constant for numerical stability in GIN.
            Controls the weight of self-loops. Defaults to 0.0.
        kernel_initializer: Initializer for weight matrices. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for weight matrices.
        bias_regularizer: Optional regularizer for bias vectors.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Tuple of (node_features, adjacency_matrix):
        - node_features: Tensor with shape `(batch_size, num_nodes, concept_dim)`
        - adjacency_matrix: Tensor with shape `(batch_size, num_nodes, num_nodes)`
          Should contain normalized adjacency values (typically 0-1 range).

    Output shape:
        Updated node embeddings with shape:
        - If aggregation='none': `(batch_size, num_nodes, concept_dim)`
        - If aggregation in ['mean','max','sum']: `(batch_size, 1, concept_dim)`
        - If aggregation='attention': `(batch_size, num_nodes, concept_dim)`

    Attributes:
        gnn_layers: List of message passing layers for each GNN layer.
        dropout_layers: List of dropout layers for regularization.
        norm_layers: List of normalization layers (or None if normalization='none').
        aggregation_attention: Attention layer for final aggregation (if aggregation='attention').
        gin_epsilon: Learnable epsilon parameter for GIN (if message_passing='gin').

    Example:
        ```python
        # Create a 3-layer Graph Attention Network with attention aggregation
        gnn = GraphNeuralNetworkLayer(
            concept_dim=256,
            num_layers=3,
            message_passing='gat',
            aggregation='attention',
            normalization='layer',
            dropout_rate=0.2,
            num_attention_heads=8
        )

        # Use in a model
        node_features = keras.Input(shape=(100, 256))  # 100 nodes, 256 features
        adjacency = keras.Input(shape=(100, 100))      # Adjacency matrix
        outputs = gnn((node_features, adjacency))
        model = keras.Model(inputs=[node_features, adjacency], outputs=outputs)

        # Switch to GraphSAGE for inductive learning
        gnn_sage = GraphNeuralNetworkLayer(
            concept_dim=256,
            num_layers=2,
            message_passing='graphsage',
            aggregation='mean'
        )
        ```

    Note:
        The adjacency matrix should be pre-normalized for stable training.
        For GCN, use symmetric normalization: D^(-1/2) * A * D^(-1/2)
        For other methods, row normalization (D^(-1) * A) often works well.
        The implementation handles the normalization internally for consistency.
    """

    def __init__(
            self,
            concept_dim: int,
            num_layers: int = 3,
            message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn',
            aggregation: Literal['mean', 'max', 'attention', 'sum', 'none'] = 'attention',
            normalization: Literal['none', 'batch', 'layer', 'rms'] = 'layer',
            activation: Union[str, Callable] = 'relu',
            dropout_rate: float = 0.1,
            use_residual: bool = True,
            num_attention_heads: int = 4,
            epsilon: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if concept_dim <= 0:
            raise ValueError(f"concept_dim must be positive, got {concept_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if message_passing not in ['gcn', 'graphsage', 'gat', 'gin']:
            raise ValueError(f"Invalid message_passing: {message_passing}")
        if aggregation not in ['mean', 'max', 'attention', 'sum', 'none']:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        # Store ALL configuration
        self.concept_dim = concept_dim
        self.num_layers = num_layers
        self.message_passing = message_passing
        self.aggregation = aggregation
        self.normalization = normalization
        self.activation = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.num_attention_heads = num_attention_heads
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.gnn_layers = []
        self.dropout_layers = []
        self.norm_layers = []

        for i in range(self.num_layers):
            # Message passing layers based on type
            if self.message_passing == 'gcn':
                # GCN uses a simple linear transformation
                self.gnn_layers.append(
                    keras.layers.Dense(
                        self.concept_dim,
                        activation=None,  # Apply activation separately for flexibility
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'gcn_dense_{i}'
                    )
                )
            elif self.message_passing == 'graphsage':
                # GraphSAGE uses separate transformations for self and neighbors
                self.gnn_layers.append({
                    'self': keras.layers.Dense(
                        self.concept_dim,
                        activation=None,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'sage_self_{i}'
                    ),
                    'neighbor': keras.layers.Dense(
                        self.concept_dim,
                        activation=None,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'sage_neighbor_{i}'
                    )
                })
            elif self.message_passing == 'gat':
                # GAT uses multi-head attention
                self.gnn_layers.append(
                    keras.layers.MultiHeadAttention(
                        num_heads=self.num_attention_heads,
                        key_dim=self.concept_dim // self.num_attention_heads,
                        dropout=self.dropout_rate,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'gat_attention_{i}'
                    )
                )
            elif self.message_passing == 'gin':
                # GIN uses an MLP for expressive power
                self.gnn_layers.append(
                    MLPBlock(
                        hidden_dim=self.concept_dim * 2,
                        output_dim=self.concept_dim,
                        activation=keras.activations.serialize(self.activation),
                        dropout_rate=self.dropout_rate,
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'gin_mlp_{i}'
                    )
                )

            # Dropout for regularization
            self.dropout_layers.append(
                keras.layers.Dropout(self.dropout_rate, name=f'gnn_dropout_{i}')
            )

            # Normalization layers
            if self.normalization == 'layer':
                self.norm_layers.append(
                    keras.layers.LayerNormalization(name=f'gnn_layer_norm_{i}')
                )
            elif self.normalization == 'rms':
                # Re-use RMSNorm from dl_techniques
                self.norm_layers.append(
                    RMSNorm(name=f'gnn_rms_norm_{i}')
                )
            elif self.normalization == 'batch':
                self.norm_layers.append(
                    keras.layers.BatchNormalization(name=f'gnn_batch_norm_{i}')
                )
            else:  # 'none'
                self.norm_layers.append(None)

        # Final aggregation layer
        if self.aggregation == 'attention':
            self.aggregation_attention = keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.concept_dim // 4,
                dropout=self.dropout_rate,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='aggregation_attention'
            )
        else:
            self.aggregation_attention = None

        # Learnable epsilon for GIN (created in build)
        self.gin_epsilon = None

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        node_shape, adjacency_shape = input_shape

        # Create GIN epsilon if needed
        if self.message_passing == 'gin':
            self.gin_epsilon = self.add_weight(
                name='gin_epsilon',
                shape=(self.num_layers,),
                initializer=keras.initializers.Constant(self.epsilon),
                trainable=True
            )

        # Build all sub-layers explicitly
        for i in range(self.num_layers):
            if self.message_passing == 'gcn':
                # GCN layer expects node features
                self.gnn_layers[i].build(node_shape)

            elif self.message_passing == 'graphsage':
                # Build both self and neighbor transformations
                self.gnn_layers[i]['self'].build(node_shape)
                self.gnn_layers[i]['neighbor'].build(node_shape)

            elif self.message_passing == 'gat':
                # GAT attention expects query and key inputs
                self.gnn_layers[i].build(node_shape, node_shape)

            elif self.message_passing == 'gin':
                # GIN MLP expects aggregated features
                self.gnn_layers[i].build(node_shape)

            # Build dropout
            self.dropout_layers[i].build(node_shape)

            # Build normalization if present
            if self.norm_layers[i] is not None:
                self.norm_layers[i].build(node_shape)

        # Build final aggregation attention if needed
        if self.aggregation_attention is not None:
            self.aggregation_attention.build(node_shape, node_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Process concept graph through GNN layers.

        Args:
            inputs: Tuple of (node_features, adjacency_matrix)
                - node_features: Shape (batch_size, num_nodes, concept_dim)
                - adjacency_matrix: Shape (batch_size, num_nodes, num_nodes)
            training: Whether in training mode

        Returns:
            Updated node embeddings based on aggregation type
        """
        node_features, adjacency_matrix = inputs

        # Normalize adjacency matrix inline for stability
        # Compute degree matrix
        degree = keras.ops.sum(adjacency_matrix, axis=-1, keepdims=False)  # (batch, num_nodes)
        degree = keras.ops.maximum(degree, 1e-12)  # Avoid division by zero

        # Row normalization: D^(-1) * A
        degree_inv = 1.0 / degree  # (batch, num_nodes)
        degree_inv_matrix = keras.ops.expand_dims(degree_inv, axis=1)  # (batch, 1, num_nodes)
        normalized_adj = adjacency_matrix * degree_inv_matrix  # Broadcasting

        # Process through GNN layers
        h = node_features

        for i in range(self.num_layers):
            h_input = h  # Store for residual connection

            if self.message_passing == 'gcn':
                # GCN: H' = σ(A_norm * H * W)
                messages = keras.ops.matmul(normalized_adj, h)
                h_new = self.gnn_layers[i](messages)

            elif self.message_passing == 'graphsage':
                # GraphSAGE: H' = σ(W_self * H + W_neighbor * AGG(A * H))
                self_features = self.gnn_layers[i]['self'](h)
                neighbor_messages = keras.ops.matmul(normalized_adj, h)
                neighbor_features = self.gnn_layers[i]['neighbor'](neighbor_messages)
                h_new = self_features + neighbor_features

            elif self.message_passing == 'gat':
                # GAT: Use attention mechanism with masking based on adjacency
                # Create attention mask from adjacency matrix
                attention_mask = keras.ops.cast(adjacency_matrix > 0, dtype='float32')
                h_new = self.gnn_layers[i](
                    query=h,
                    value=h,
                    attention_mask=attention_mask,
                    training=training
                )

            elif self.message_passing == 'gin':
                # GIN: H' = MLP((1 + ε) * H + Σ_neighbors)
                neighbor_sum = keras.ops.matmul(adjacency_matrix, h)
                if self.gin_epsilon is not None:
                    eps = self.gin_epsilon[i]
                    combined = (1 + eps) * h + neighbor_sum
                else:
                    combined = h + neighbor_sum
                h_new = self.gnn_layers[i](combined, training=training)

            # Apply activation (except for GIN which has it built-in)
            if self.message_passing != 'gin':
                h_new = self.activation(h_new)

            # Apply dropout
            h_new = self.dropout_layers[i](h_new, training=training)

            # Residual connection if dimensions match
            if self.use_residual and h_input.shape[-1] == h_new.shape[-1]:
                h = h_input + h_new
            else:
                h = h_new

            # Apply normalization
            if self.norm_layers[i] is not None:
                h = self.norm_layers[i](h, training=training)

        # Final aggregation
        if self.aggregation == 'attention' and self.aggregation_attention is not None:
            # Use self-attention for aggregation
            h = self.aggregation_attention(h, h, training=training)
        elif self.aggregation == 'mean':
            # Global mean pooling
            h = keras.ops.mean(h, axis=1, keepdims=True)
        elif self.aggregation == 'max':
            # Global max pooling
            h = keras.ops.max(h, axis=1, keepdims=True)
        elif self.aggregation == 'sum':
            # Global sum pooling
            h = keras.ops.sum(h, axis=1, keepdims=True)
        # If aggregation == 'none', return as is

        return h

    def compute_output_shape(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> Tuple[
        Optional[int], ...]:
        """Compute output shape based on aggregation type."""
        node_shape, _ = input_shape
        batch_size = node_shape[0]
        num_nodes = node_shape[1]

        if self.aggregation in ['mean', 'max', 'sum']:
            # Global pooling reduces to single node
            return (batch_size, 1, self.concept_dim)
        else:
            # Keep all nodes
            return (batch_size, num_nodes, self.concept_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'concept_dim': self.concept_dim,
            'num_layers': self.num_layers,
            'message_passing': self.message_passing,
            'aggregation': self.aggregation,
            'normalization': self.normalization,
            'activation': keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'num_attention_heads': self.num_attention_heads,
            'epsilon': self.epsilon,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
