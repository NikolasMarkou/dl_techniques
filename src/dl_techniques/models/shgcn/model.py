"""
Complete Simplified Hyperbolic Graph Convolutional Neural Network Model.

This module provides a flexible model wrapper for sHGCN that can be configured
for different graph learning tasks:
- Node classification (generative: node embeddings)
- Link prediction (predictive: edge probabilities)

The model stacks multiple SHGCNLayer instances and provides appropriate output
layers based on the task.
"""

import keras
from keras import layers, ops
from typing import List, Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.graphs.shgcn_layer import SHGCNLayer
from dl_techniques.layers.graphs.fermi_diract_decoder import FermiDiracDecoder

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SHGCNModel(keras.Model):
    """
    Multi-layer Simplified Hyperbolic Graph Convolutional Neural Network.

    This model stacks multiple sHGCN layers to create a deep graph neural network
    that operates efficiently by leveraging both Euclidean and hyperbolic geometries.
    The model can be configured for different downstream tasks through the output
    layer configuration.

    **Architecture**:
    ```
    Input: [Features [N, D_in], Adjacency [N, N] sparse]
            ↓
    sHGCN Layer 1: [N, D_in] → [N, hidden_dims[0]]
            ↓
    sHGCN Layer 2: [N, hidden_dims[0]] → [N, hidden_dims[1]]
            ↓
    ...
            ↓
    sHGCN Layer L: [N, hidden_dims[-1]] → [N, output_dim]
            ↓
    Output: [N, output_dim]
    ```

    **Task Configurations**:

    1. **Node Classification** (output_activation='linear' or 'softmax'):
       - Produces node embeddings or class logits
       - Typically followed by a task-specific head
       - Use 'linear' for embeddings, 'softmax' for direct classification

    2. **Link Prediction** (output_activation='linear'):
       - Produces node embeddings in Euclidean space
       - Pair with FermiDiracDecoder for edge probability prediction
       - Embeddings should be unit-normalized for best results

    Args:
        hidden_dims: List of hidden layer dimensions, e.g., [64, 32]. Must contain
            at least one value. Each value must be positive.
        output_dim: Output dimensionality. For node classification, this is the
            number of classes or embedding size. For link prediction, this is
            the embedding size. Must be positive.
        output_activation: Activation for output layer. Use 'linear' for embeddings,
            'softmax' for classification, or None. Defaults to 'linear'.
        dropout_rate: Dropout probability applied within each layer. Range [0, 1).
            Higher values increase regularization. Defaults to 0.5.
        use_bias: Whether to use hyperbolic bias in all layers. Defaults to True.
        use_curvature: Whether curvature is learnable in all layers. When True,
            each layer learns its own curvature. Defaults to True.
        **kwargs: Additional keyword arguments for Model base class.

    Input:
        List of two tensors:
        - features: Dense tensor of shape (num_nodes, input_dim)
        - adjacency: Sparse tensor of shape (num_nodes, num_nodes), normalized

    Output:
        Dense tensor of shape (num_nodes, output_dim).
        - For embeddings: Euclidean vectors in tangent space
        - For classification: Class logits or probabilities

    Attributes:
        hidden_layers: List of SHGCNLayer instances for hidden representations.
        output_layer: Final SHGCNLayer for task-specific output.

    Example:
        ```python
        # Node classification model (3 classes)
        model = SHGCNModel(
            hidden_dims=[64, 32],
            output_dim=3,
            output_activation='softmax',
            dropout_rate=0.5
        )

        # Compile for classification
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(
            x=[features, adj_sparse],
            y=labels,
            epochs=100,
            validation_split=0.2
        )

        # Link prediction model (embedding dimension 16)
        model = SHGCNModel(
            hidden_dims=[32, 16],
            output_dim=16,
            output_activation='linear',  # Embeddings
            dropout_rate=0.3
        )

        # Get embeddings
        embeddings = model([features, adj_sparse], training=False)

        # Use with decoder
        from fermi_dirac_decoder import FermiDiracDecoder
        decoder = FermiDiracDecoder()

        u_embed = tf.gather(embeddings, u_indices)
        v_embed = tf.gather(embeddings, v_indices)
        edge_probs = decoder([u_embed, v_embed])
        ```

    Note:
        - All hidden layers use 'relu' activation by default
        - Output layer activation is configurable for task flexibility
        - For link prediction, embeddings are in Euclidean space
        - Model automatically handles sparse adjacency matrices
        - Each layer can learn its own curvature if use_curvature=True

    References:
        Arevalo et al. "Simplified Hyperbolic Graph Convolutional Neural Networks"
    """

    def __init__(
            self,
            hidden_dims: List[int],
            output_dim: int,
            output_activation: Optional[Union[str, callable]] = 'linear',
            dropout_rate: float = 0.5,
            use_bias: bool = True,
            use_curvature: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize multi-layer sHGCN model."""
        super().__init__(**kwargs)

        # Validate inputs
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden_dims must be positive")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.use_curvature = use_curvature

        # Create hidden layers
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            layer = SHGCNLayer(
                units=dim,
                activation='relu',
                use_bias=use_bias,
                use_curvature=use_curvature,
                dropout_rate=dropout_rate,
                name=f'shgcn_hidden_{i}'
            )
            self.hidden_layers.append(layer)

        # Create output layer
        self.output_layer = SHGCNLayer(
            units=output_dim,
            activation=output_activation,
            use_bias=use_bias,
            use_curvature=use_curvature,
            dropout_rate=dropout_rate,
            name='shgcn_output'
        )

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through all sHGCN layers.

        Args:
            inputs: List of [features, adjacency].
                - features: [num_nodes, input_dim]
                - adjacency: [num_nodes, num_nodes] sparse
            training: Whether in training mode (affects dropout).

        Returns:
            Node embeddings or logits of shape [num_nodes, output_dim].
        """
        x, adj = inputs

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer([x, adj], training=training)

        # Pass through output layer
        x = self.output_layer([x, adj], training=training)

        return x

    def get_config(self) -> dict:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'output_activation': (
                keras.activations.serialize(
                    keras.activations.get(self.output_activation)
                )
                if self.output_activation is not None
                else None
            ),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_curvature': self.use_curvature,
        })
        return config


@keras.saving.register_keras_serializable()
class SHGCNNodeClassifier(keras.Model):
    """
    Complete node classification model with sHGCN backbone and classification head.

    This is a convenience wrapper that combines the sHGCN feature extractor with
    a final classification layer, providing a ready-to-use model for supervised
    node classification tasks.

    **Architecture**:
    ```
    Input: [Features, Adjacency]
            ↓
    sHGCN Backbone: Multi-layer feature extraction
            ↓
    Classification Head: Dense(num_classes, activation='softmax')
            ↓
    Output: Class probabilities [N, num_classes]
    ```

    Args:
        num_classes: Number of output classes. Must be >= 2.
        hidden_dims: List of hidden layer dimensions for sHGCN backbone.
        embedding_dim: Dimension of node embeddings before classification.
            Defaults to 16.
        dropout_rate: Dropout rate for regularization. Defaults to 0.5.
        use_bias: Whether to use hyperbolic bias in sHGCN layers. Defaults to True.
        use_curvature: Whether to learn curvature in sHGCN layers. Defaults to True.
        **kwargs: Additional keyword arguments for Model base class.

    Input:
        List of [features, adjacency] as for SHGCNModel.

    Output:
        Class probabilities of shape (num_nodes, num_classes), values sum to 1.

    Example:
        ```python
        # Create classifier
        model = SHGCNNodeClassifier(
            num_classes=7,  # Citation network classes
            hidden_dims=[64, 32],
            embedding_dim=16,
            dropout_rate=0.5
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            x=[features, adj_sparse],
            y=train_labels,
            epochs=200,
            validation_data=([features, adj_sparse], val_labels),
            verbose=1
        )

        # Predict
        predictions = model.predict([features, adj_sparse])
        predicted_classes = ops.argmax(predictions, axis=-1)
        ```
    """

    def __init__(
            self,
            num_classes: int,
            hidden_dims: List[int],
            embedding_dim: int = 16,
            dropout_rate: float = 0.5,
            use_bias: bool = True,
            use_curvature: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize node classification model."""
        super().__init__(**kwargs)

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.use_curvature = use_curvature

        # sHGCN backbone for feature extraction
        self.backbone = SHGCNModel(
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            output_activation='relu',  # Embeddings with non-linearity
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_curvature=use_curvature,
            name='shgcn_backbone'
        )

        # Classification head
        self.classifier = layers.Dense(
            num_classes,
            activation='softmax',
            name='classifier'
        )

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for node classification.

        Args:
            inputs: List of [features, adjacency].
            training: Whether in training mode.

        Returns:
            Class probabilities of shape [num_nodes, num_classes].
        """
        # Extract features through sHGCN
        embeddings = self.backbone(inputs, training=training)

        # Classify
        logits = self.classifier(embeddings)

        return logits

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'hidden_dims': self.hidden_dims,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_curvature': self.use_curvature,
        })
        return config


@keras.saving.register_keras_serializable()
class SHGCNLinkPredictor(keras.Model):
    """
    Complete link prediction model with sHGCN backbone and Fermi-Dirac decoder.

    This model combines node embedding generation via sHGCN with edge probability
    prediction using the Fermi-Dirac decoder. It provides an end-to-end solution
    for link prediction tasks on graphs.

    **Architecture**:
    ```
    Input: [Features, Adjacency, Edge_Pairs]
            ↓
    sHGCN Backbone: Generate node embeddings
            ↓
    Gather: Select embeddings for edge pairs
            ↓
    Fermi-Dirac Decoder: Compute edge probabilities
            ↓
    Output: Edge probabilities [num_edges,]
    ```

    Args:
        hidden_dims: List of hidden layer dimensions for sHGCN backbone.
        embedding_dim: Dimension of node embeddings. Should be large enough to
            capture graph structure (typically 16-64). Defaults to 16.
        dropout_rate: Dropout rate for regularization. Defaults to 0.3.
        use_bias: Whether to use hyperbolic bias. Defaults to True.
        use_curvature: Whether to learn curvature. Defaults to True.
        **kwargs: Additional keyword arguments for Model base class.

    Input:
        List of three tensors:
        - features: [num_nodes, input_dim]
        - adjacency: [num_nodes, num_nodes] sparse
        - edge_pairs: [num_edges, 2] with [source_idx, target_idx] per row

    Output:
        Edge probabilities of shape (num_edges,) with values in [0, 1].

    Example:
        ```python
        # Create link predictor
        model = SHGCNLinkPredictor(
            hidden_dims=[64, 32],
            embedding_dim=16,
            dropout_rate=0.3
        )

        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )

        # Prepare edge pairs
        # Positive edges: actual edges in graph
        pos_edges = np.array([[0, 1], [1, 2], [2, 3]])
        # Negative edges: non-existent edges (sampled)
        neg_edges = np.array([[0, 5], [1, 7], [3, 9]])

        edge_pairs = np.vstack([pos_edges, neg_edges])
        labels = np.array([1, 1, 1, 0, 0, 0])  # 1=exists, 0=doesn't exist

        # Train
        model.fit(
            x=[features, adj_sparse, edge_pairs],
            y=labels,
            epochs=100,
            batch_size=32
        )

        # Predict on new edge pairs
        test_pairs = np.array([[0, 2], [4, 6]])
        probs = model.predict([features, adj_sparse, test_pairs])
        print(probs)  # e.g., [0.85, 0.12] - first edge likely exists
        ```

    Note:
        - Edge pairs should be [source, target] integer indices
        - Model outputs probabilities, use threshold (e.g., 0.5) for binary prediction
        - Training requires both positive and negative edge samples
        - Embeddings are in Euclidean space, decoder uses Euclidean distance
    """

    def __init__(
            self,
            hidden_dims: List[int],
            embedding_dim: int = 16,
            dropout_rate: float = 0.3,
            use_bias: bool = True,
            use_curvature: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize link prediction model."""
        super().__init__(**kwargs)

        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.use_curvature = use_curvature

        # sHGCN backbone for node embeddings
        self.backbone = SHGCNModel(
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            output_activation='linear',  # Raw embeddings
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_curvature=use_curvature,
            name='shgcn_backbone'
        )

        # Fermi-Dirac decoder for edge probabilities
        self.decoder = FermiDiracDecoder(name='fermi_dirac_decoder')

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for link prediction.

        Args:
            inputs: List of [features, adjacency, edge_pairs].
                - features: [num_nodes, input_dim]
                - adjacency: [num_nodes, num_nodes] sparse
                - edge_pairs: [num_edges, 2] integer indices
            training: Whether in training mode.

        Returns:
            Edge probabilities of shape [num_edges,].
        """
        features, adjacency, edge_pairs = inputs

        # Generate node embeddings
        embeddings = self.backbone([features, adjacency], training=training)

        # Extract embeddings for edge pairs
        # edge_pairs: [num_edges, 2] with [src, tgt] indices
        src_indices = edge_pairs[:, 0]
        tgt_indices = edge_pairs[:, 1]

        src_embeddings = ops.take(embeddings, src_indices, axis=0)
        tgt_embeddings = ops.take(embeddings, tgt_indices, axis=0)

        # Compute edge probabilities
        probabilities = self.decoder([src_embeddings, tgt_embeddings])

        return probabilities

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_curvature': self.use_curvature,
        })
        return config

# ---------------------------------------------------------------------
