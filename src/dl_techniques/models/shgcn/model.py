"""
Complete Simplified Hyperbolic Graph Convolutional Neural Network Model.

This module provides a flexible model wrapper for sHGCN that can be configured
for different graph learning tasks:
- Node classification (generative: node embeddings)
- Link prediction (predictive: edge probabilities)

The model stacks multiple SHGCNLayer instances and provides appropriate output
layers based on the task.

References:
    - Arevalo et al., 2024. Simplified Hyperbolic Graph Convolutional Neural
      Networks. (https://arxiv.org/abs/2411.15266) -- the sHGCN formulation
      this model stacks; the package README section 16 records the citation as
      it was taken.
    - Chami et al., 2019. Hyperbolic Graph Convolutional Neural Networks.
      NeurIPS 2019. (https://arxiv.org/abs/1910.12933) -- HGCN, the model sHGCN
      simplifies: hyperbolic feature transform, aggregation and activation.
    - Wu et al., 2019. Simplifying Graph Convolutional Networks (SGC). ICML
      2019. (https://arxiv.org/abs/1902.07153) -- the "collapse the nonlinear
      layers" argument, transplanted to the hyperbolic setting.
    - Nickel and Kiela, 2017. Poincare Embeddings for Learning Hierarchical
      Representations. NeurIPS 2017. (https://arxiv.org/abs/1705.08039) -- the
      Poincare-ball geometry the exp/log maps in
      ``dl_techniques.utils.geometry.poincare_math`` implement.
    - Kipf and Welling, 2017. Semi-Supervised Classification with Graph
      Convolutional Networks. ICLR 2017. (https://arxiv.org/abs/1609.02907)
"""

import keras
from typing import List, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.graphs.fermi_diract_decoder import FermiDiracDecoder
from dl_techniques.layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer import SHGCNLayer

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
    Input: [Features [N, D_in], Adjacency [N, N] dense]
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
        - adjacency: Dense tensor of shape (num_nodes, num_nodes), normalized.
          A leading batch axis is supported throughout: (B, N, D_in) with
          (B, N, N). A tf.sparse.SparseTensor is also accepted, but only on the
          TensorFlow backend -- see SHGCNLayer's aggregation step.

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

        # Train -- FULL GRAPH, one step per epoch. Do NOT call
        # model.fit(x=[features, adj], ...): Keras batches axis 0 of EVERY
        # input, so it slices the [N, N] adjacency alongside the features and
        # the run dies in the data pipeline. One graph is one sample.
        for _ in range(100):
            model.train_on_batch([features, adj], labels)

        # Link prediction model (embedding dimension 16)
        model = SHGCNModel(
            hidden_dims=[32, 16],
            output_dim=16,
            output_activation='linear',  # Embeddings
            dropout_rate=0.3
        )

        # Get embeddings
        embeddings = model([features, adj], training=False)

        # Use with decoder
        from dl_techniques.layers.graphs.fermi_diract_decoder import (
            FermiDiracDecoder,
        )
        decoder = FermiDiracDecoder()

        u_embed = tf.gather(embeddings, u_indices)
        v_embed = tf.gather(embeddings, v_indices)
        edge_probs = decoder([u_embed, v_embed])
        ```

    Note:
        - All hidden layers use 'relu' activation by default
        - Output layer activation is configurable for task flexibility
        - For link prediction, embeddings are in Euclidean space
        - Aggregation is a dense keras.ops.matmul; a tf.sparse.SparseTensor
          adjacency also works on the TensorFlow backend only
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

        self.output_layer = SHGCNLayer(
            units=output_dim,
            activation=output_activation,
            use_bias=use_bias,
            use_curvature=use_curvature,
            dropout_rate=dropout_rate,
            name='shgcn_output'
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Materialise every sHGCN layer from ``[features, adjacency]`` shapes.

        Args:
            input_shape: List of ``[features_shape, adjacency_shape]``.
        """
        feat_shape, adj_shape = input_shape

        current_shape = tuple(feat_shape)
        for layer in self.hidden_layers:
            layer.build([current_shape, adj_shape])
            current_shape = layer.compute_output_shape([current_shape, adj_shape])

        self.output_layer.build([current_shape, adj_shape])

        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(..., num_nodes, output_dim)``."""
        feat_shape = input_shape[0]
        return tuple(feat_shape[:-1]) + (self.output_dim,)

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
                - adjacency: [num_nodes, num_nodes] dense
            training: Whether in training mode (affects dropout).

        Returns:
            Node embeddings or logits of shape [num_nodes, output_dim].
        """
        x, adj = inputs

        for layer in self.hidden_layers:
            x = layer([x, adj], training=training)

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

        # Train -- FULL GRAPH, one step per epoch. model.fit(x=[features, adj])
        # does NOT work: Keras batches axis 0 of every input, so it slices the
        # [N, N] adjacency alongside the features.
        for _ in range(200):
            model.train_on_batch([features, adj], train_labels)
            val_metrics = model.test_on_batch([features, adj], val_labels)

        # Predict -- predict_on_batch, NOT predict: predict batches axis 0 too,
        # and it does so silently while N <= 32 (the default batch size), which
        # is exactly how this example read as working.
        predictions = model.predict_on_batch([features, adj])
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

        self.classifier = keras.layers.Dense(
            num_classes,
            activation='softmax',
            name='classifier'
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Materialise backbone and classification head.

        Args:
            input_shape: List of ``[features_shape, adjacency_shape]``.
        """
        self.backbone.build(input_shape)
        self.classifier.build(self.backbone.compute_output_shape(input_shape))
        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(..., num_nodes, num_classes)``."""
        feat_shape = input_shape[0]
        return tuple(feat_shape[:-1]) + (self.num_classes,)

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
        embeddings = self.backbone(inputs, training=training)

        # NOT `logits`: `self.classifier` is
        # `Dense(num_classes, activation='softmax')`, so these rows already sum
        # to 1. The name mattered -- a caller who compiled
        # `from_logits=True` on the strength of it got a silently wrong loss.
        probabilities = self.classifier(embeddings)

        return probabilities

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

        # Train -- FULL GRAPH, one step per epoch. model.fit(...) does NOT
        # work here either: `features` and `adj` are whole-graph inputs, so
        # Keras' axis-0 batching would slice them along with `edge_pairs`.
        for _ in range(100):
            model.train_on_batch([features, adj, edge_pairs], labels)

        # Predict on new edge pairs
        test_pairs = np.array([[0, 2], [4, 6]])
        probs = model.predict_on_batch([features, adj, test_pairs])
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

        self.decoder = FermiDiracDecoder(name='fermi_dirac_decoder')

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Materialise the backbone AND the Fermi-Dirac decoder.

        # DECISION plan-2026-08-19T163559-499b6f0e/D-029
        This method exists for SERIALIZATION, not for eager convenience -- do not
        delete it on the grounds that the model already runs without it. Keras
        restores a sub-layer's variables only if that sub-layer is BUILT at load
        time. With no `build()` here, `FermiDiracDecoder` was unbuilt when the
        archive was read, its `load_own_variables` was skipped, and the reloaded
        model silently fell back to the class defaults r=2.0 / t=1.0. MEASURED:
        the archive was COMPLETE (8 of 8 datasets, both scalars stored at the
        perturbed 3.75) and the RELOAD was lossy -- 6 of 8 tensors identical, a
        forward delta of 1.497385e-01 against an output range of 7.310586e-01,
        and NO warning of any kind. An archive-content check alone cannot see
        this; the guard must compare reloaded VALUES. See decisions.md D-029.

        Args:
            input_shape: List of ``[features_shape, adjacency_shape,
                edge_pairs_shape]``.
        """
        feat_shape, adj_shape, edge_shape = input_shape

        self.backbone.build([feat_shape, adj_shape])
        embedding_shape = self.backbone.compute_output_shape(
            [feat_shape, adj_shape])

        # `take` along axis 0 replaces the node axis with the edge axis.
        gathered_shape = (edge_shape[0],) + tuple(embedding_shape[1:])
        self.decoder.build([gathered_shape, gathered_shape])

        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(num_edges,)``."""
        return (input_shape[2][0],)

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
                - adjacency: [num_nodes, num_nodes] dense
                - edge_pairs: [num_edges, 2] integer indices
            training: Whether in training mode.

        Returns:
            Edge probabilities of shape [num_edges,].
        """
        features, adjacency, edge_pairs = inputs

        embeddings = self.backbone([features, adjacency], training=training)

        # edge_pairs: [num_edges, 2] with [src, tgt] indices
        src_indices = edge_pairs[:, 0]
        tgt_indices = edge_pairs[:, 1]

        src_embeddings = keras.ops.take(embeddings, src_indices, axis=0)
        tgt_embeddings = keras.ops.take(embeddings, tgt_indices, axis=0)

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
# Factories
# ---------------------------------------------------------------------


def create_shgcn(
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        output_activation: Optional[Union[str, callable]] = 'linear',
        dropout_rate: float = 0.5,
        use_bias: bool = True,
        use_curvature: bool = True,
        **kwargs: Any
) -> SHGCNModel:
    """Create a base sHGCN feature-extraction model.

    There is no ``MODEL_VARIANTS`` table and none was invented: sHGCN is
    specified by a per-layer hidden-dimension list chosen for the dataset at
    hand, and the paper publishes no named scale family. This factory therefore
    constructs the class with a sensible two-layer default.

    Args:
        hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
            ``[64, 32]``.
        output_dim: Dimension of the output layer.
        output_activation: Activation applied to the output layer.
        dropout_rate: Dropout applied between layers.
        use_bias: Whether the sHGCN layers use a bias term.
        use_curvature: Whether learnable curvature is enabled.
        **kwargs: Additional arguments forwarded to the model constructor.

    Returns:
        A configured SHGCNModel. It is called with ``[features, adjacency]``.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    return SHGCNModel(
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        output_activation=output_activation,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        use_curvature=use_curvature,
        **kwargs
    )


def create_shgcn_node_classifier(
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 16,
        dropout_rate: float = 0.5,
        use_bias: bool = True,
        use_curvature: bool = True,
        **kwargs: Any
) -> SHGCNNodeClassifier:
    """Create an sHGCN node-classification model.

    Args:
        num_classes: Number of target classes. Required -- it is a property of
            the dataset, not something a default can guess.
        hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
            ``[64, 32]``.
        embedding_dim: Dimension of the node embedding fed to the classifier
            head.
        dropout_rate: Dropout applied between layers.
        use_bias: Whether the sHGCN layers use a bias term.
        use_curvature: Whether learnable curvature is enabled.
        **kwargs: Additional arguments forwarded to the model constructor.

    Returns:
        A configured SHGCNNodeClassifier. It is called with
        ``[features, adjacency]`` and returns class PROBABILITIES -- its head
        is ``Dense(num_classes, activation='softmax')``, so compile with
        ``from_logits=False`` (the Keras default).

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    return SHGCNNodeClassifier(
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        use_curvature=use_curvature,
        **kwargs
    )


def create_shgcn_link_predictor(
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 16,
        dropout_rate: float = 0.3,
        use_bias: bool = True,
        use_curvature: bool = True,
        **kwargs: Any
) -> SHGCNLinkPredictor:
    """Create an sHGCN link-prediction model.

    Args:
        hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
            ``[64, 32]``.
        embedding_dim: Dimension of the node embeddings scored by the
            Fermi-Dirac decoder.
        dropout_rate: Dropout applied between layers.
        use_bias: Whether the sHGCN layers use a bias term.
        use_curvature: Whether learnable curvature is enabled.
        **kwargs: Additional arguments forwarded to the model constructor.

    Returns:
        A configured SHGCNLinkPredictor. It is called with
        ``[features, adjacency, edge_pairs]`` and returns edge probabilities.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    return SHGCNLinkPredictor(
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        use_curvature=use_curvature,
        **kwargs
    )

# ---------------------------------------------------------------------
