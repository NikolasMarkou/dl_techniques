"""
SHGCNModel stacks SHGCNLayer instances into node-embedding, node-classification and
link-prediction models for graphs in hyperbolic space. Each layer transforms features in
the Poincare ball instead of Euclidean space, and the stack collapses the per-layer
nonlinearity into one aggregation step, extending Wu et al.'s SGC simplification to
hyperbolic geometry. A model call takes a dense feature tensor and a dense adjacency
tensor (or, on the TensorFlow backend only, a sparse one) as a two-item list. A whole
graph is one training sample, so training loops call ``train_on_batch``, not ``fit``,
because ``fit`` batches axis 0 of every input and would slice the adjacency tensor along
with the features.

References:
    - Arevalo et al., 2025. sHGCN: Simplified hyperbolic graph convolutional neural
      networks. (https://arxiv.org/abs/2506.14438)
    - Chami et al., 2019. Hyperbolic Graph Convolutional Neural Networks. NeurIPS 2019.
      (https://arxiv.org/abs/1910.12933)
    - Wu et al., 2019. Simplifying Graph Convolutional Networks (SGC). ICML 2019.
      (https://arxiv.org/abs/1902.07153)
    - Nickel and Kiela, 2017. Poincare Embeddings for Learning Hierarchical
      Representations. NeurIPS 2017. (https://arxiv.org/abs/1705.08039)
    - Kipf and Welling, 2017. Semi-Supervised Classification with Graph Convolutional
      Networks. ICLR 2017. (https://arxiv.org/abs/1609.02907)
"""

import keras
from typing import List, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.graphs.fermi_diract_decoder import FermiDiracDecoder
from dl_techniques.layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer import SHGCNLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.shgcn.model")
class SHGCNModel(keras.Model):
    """Stack of sHGCN layers producing node embeddings or logits.

    Each hidden layer applies ``relu``; the output layer's activation is
    configurable so the same class serves node classification (``softmax``
    or ``linear`` for raw embeddings) and link prediction (``linear``
    embeddings, paired with a decoder such as :class:`FermiDiracDecoder`).

    Architecture:

        .. code-block:: text

            features [N, D_in]      adjacency [N, N]
                  │                        │
                  ▼                        │
            ┌──────────────┐               │
            │ sHGCN layer 1├───────────────┤
            └──────┬───────┘               │
                    │ [N, hidden[0]]        │
                   ...                      │
                    │ [N, hidden[-1]]       │
                    ▼                        │
            ┌──────────────┐               │
            │ sHGCN layer L├───────────────┘
            │  (output)    │
            └──────┬───────┘
                    ▼
             output [N, output_dim]

    :param hidden_dims: Per-layer hidden dimensions, e.g. ``[64, 32]``. At
        least one positive value.
    :type hidden_dims: List[int]
    :param output_dim: Output dimensionality: class count for
        classification, embedding size for link prediction. Must be
        positive.
    :type output_dim: int
    :param output_activation: Activation of the output layer. ``'linear'``
        for embeddings, ``'softmax'`` for classification, or ``None``.
        Defaults to ``'linear'``.
    :type output_activation: Optional[Union[str, callable]]
    :param dropout_rate: Dropout probability inside each layer, in
        ``[0, 1)``. Defaults to ``0.5``.
    :type dropout_rate: float
    :param use_bias: Whether every layer uses a hyperbolic bias. Defaults
        to ``True``.
    :type use_bias: bool
    :param use_curvature: Whether every layer learns its own curvature.
        Defaults to ``True``.
    :type use_curvature: bool
    :param kwargs: Forwarded to ``keras.Model``.
    :raises ValueError: If ``hidden_dims`` is empty or has a non-positive
        entry, if ``output_dim`` is not positive, or if ``dropout_rate``
        is outside ``[0, 1)``.

    Input shape:
        Two dense tensors ``[features, adjacency]``: features
        ``(num_nodes, input_dim)``, adjacency ``(num_nodes, num_nodes)``,
        normalized. Both accept a leading batch axis. A
        ``tf.sparse.SparseTensor`` adjacency works only on the TensorFlow
        backend.

    Output shape:
        ``(num_nodes, output_dim)`` -- embeddings in tangent space, or
        class logits/probabilities depending on ``output_activation``.

    :ivar hidden_layers: List of :class:`SHGCNLayer` instances.
    :ivar output_layer: Final :class:`SHGCNLayer` producing the model output.

    Example:
        .. code-block:: python

            model = SHGCNModel(
                hidden_dims=[64, 32],
                output_dim=3,
                output_activation='softmax',
                dropout_rate=0.5,
            )
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            for _ in range(100):
                model.train_on_batch([features, adj], labels)

    Note:
        Aggregation is a dense ``keras.ops.matmul``; a sparse adjacency
        works only on the TensorFlow backend.
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
        """Build every sHGCN layer from the ``[features, adjacency]`` shapes.

        :param input_shape: ``[features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
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
        """Return ``(..., num_nodes, output_dim)``.

        :param input_shape: ``[features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        feat_shape = input_shape[0]
        return tuple(feat_shape[:-1]) + (self.output_dim,)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run every sHGCN layer in sequence.

        :param inputs: ``[features, adjacency]`` with shapes
            ``[num_nodes, input_dim]`` and ``[num_nodes, num_nodes]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether dropout is active.
        :type training: Optional[bool]
        :return: Node embeddings or logits, shape ``[num_nodes, output_dim]``.
        :rtype: keras.KerasTensor
        """
        x, adj = inputs

        for layer in self.hidden_layers:
            x = layer([x, adj], training=training)

        x = self.output_layer([x, adj], training=training)

        return x

    def get_config(self) -> dict:
        """Return the config needed to reconstruct this model.

        :return: Constructor arguments.
        :rtype: dict
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


@register_dl_technique("dl_techniques.models.shgcn.model")
class SHGCNNodeClassifier(keras.Model):
    """sHGCN feature extractor with a softmax classification head.

    Architecture:

        .. code-block:: text

            features, adjacency
                  │
                  ▼
            ┌───────────────┐
            │ sHGCN backbone│
            └──────┬────────┘
                    │ [N, embedding_dim]
                    ▼
            ┌───────────────┐
            │ Dense+softmax │
            └──────┬────────┘
                    ▼
            probabilities [N, num_classes]

    :param num_classes: Number of output classes. Must be at least 2.
    :type num_classes: int
    :param hidden_dims: Per-layer hidden dimensions for the sHGCN backbone.
    :type hidden_dims: List[int]
    :param embedding_dim: Dimension of node embeddings before
        classification. Defaults to ``16``.
    :type embedding_dim: int
    :param dropout_rate: Dropout rate for regularization. Defaults to
        ``0.5``.
    :type dropout_rate: float
    :param use_bias: Whether sHGCN layers use a hyperbolic bias. Defaults
        to ``True``.
    :type use_bias: bool
    :param use_curvature: Whether sHGCN layers learn curvature. Defaults
        to ``True``.
    :type use_curvature: bool
    :param kwargs: Forwarded to ``keras.Model``.
    :raises ValueError: If ``num_classes`` is less than 2.

    Input shape:
        ``[features, adjacency]`` as for :class:`SHGCNModel`.

    Output shape:
        ``(num_nodes, num_classes)``, rows summing to 1.

    Example:
        .. code-block:: python

            model = SHGCNNodeClassifier(
                num_classes=7,
                hidden_dims=[64, 32],
                embedding_dim=16,
                dropout_rate=0.5,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            for _ in range(200):
                model.train_on_batch([features, adj], train_labels)
                val_metrics = model.test_on_batch([features, adj], val_labels)
            predictions = model.predict_on_batch([features, adj])
            predicted_classes = ops.argmax(predictions, axis=-1)

    Note:
        Use ``train_on_batch`` / ``test_on_batch`` / ``predict_on_batch``,
        not ``fit`` / ``predict``: a whole graph is one sample, and the
        batching APIs slice axis 0 of every input -- including the
        adjacency matrix -- which silently corrupts the graph on any
        node count that happens to fit inside the default batch size.
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

        self.backbone = SHGCNModel(
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            output_activation='relu',
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
        """Build the backbone and the classification head.

        :param input_shape: ``[features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        """
        self.backbone.build(input_shape)
        self.classifier.build(self.backbone.compute_output_shape(input_shape))
        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(..., num_nodes, num_classes)``.

        :param input_shape: ``[features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        feat_shape = input_shape[0]
        return tuple(feat_shape[:-1]) + (self.num_classes,)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the backbone then the classification head.

        :param inputs: ``[features, adjacency]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether dropout is active.
        :type training: Optional[bool]
        :return: Class probabilities, shape ``[num_nodes, num_classes]``.
        :rtype: keras.KerasTensor
        """
        embeddings = self.backbone(inputs, training=training)

        # These rows already sum to 1 (softmax head): treat as
        # probabilities, not logits, when compiling a loss.
        probabilities = self.classifier(embeddings)

        return probabilities

    def get_config(self) -> dict:
        """Return the config needed to reconstruct this model.

        :return: Constructor arguments.
        :rtype: dict
        """
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


@register_dl_technique("dl_techniques.models.shgcn.model")
class SHGCNLinkPredictor(keras.Model):
    """sHGCN backbone paired with a Fermi-Dirac edge-probability decoder.

    Architecture:

        .. code-block:: text

            features, adjacency        edge_pairs [E, 2]
                  │                          │
                  ▼                          │
            ┌───────────────┐                │
            │ sHGCN backbone│                │
            └──────┬────────┘                │
                    │ embeddings [N, D]        │
                    ▼                          │
            ┌────────────────────────────┐     │
            │ gather src/tgt embeddings  │◄────┘
            └──────┬─────────────────────┘
                    ▼
            ┌───────────────────┐
            │ Fermi-Dirac decoder│
            └──────┬─────────────┘
                    ▼
            edge probabilities [E]

    :param hidden_dims: Per-layer hidden dimensions for the sHGCN backbone.
    :type hidden_dims: List[int]
    :param embedding_dim: Node embedding dimension, typically 16-64 to
        capture graph structure. Defaults to ``16``.
    :type embedding_dim: int
    :param dropout_rate: Dropout rate for regularization. Defaults to
        ``0.3``.
    :type dropout_rate: float
    :param use_bias: Whether sHGCN layers use a hyperbolic bias. Defaults
        to ``True``.
    :type use_bias: bool
    :param use_curvature: Whether sHGCN layers learn curvature. Defaults
        to ``True``.
    :type use_curvature: bool
    :param kwargs: Forwarded to ``keras.Model``.

    Input shape:
        ``[features, adjacency, edge_pairs]``: features
        ``[num_nodes, input_dim]``, adjacency ``[num_nodes, num_nodes]``,
        edge_pairs ``[num_edges, 2]`` holding ``[source_idx, target_idx]``
        per row.

    Output shape:
        ``(num_edges,)``, values in ``[0, 1]``.

    Example:
        .. code-block:: python

            model = SHGCNLinkPredictor(
                hidden_dims=[64, 32], embedding_dim=16, dropout_rate=0.3,
            )
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()],
            )
            pos_edges = np.array([[0, 1], [1, 2], [2, 3]])
            neg_edges = np.array([[0, 5], [1, 7], [3, 9]])
            edge_pairs = np.vstack([pos_edges, neg_edges])
            labels = np.array([1, 1, 1, 0, 0, 0])
            for _ in range(100):
                model.train_on_batch([features, adj, edge_pairs], labels)
            probs = model.predict_on_batch([features, adj, test_pairs])

    Note:
        Training needs both positive and negative edge samples. Use
        ``train_on_batch`` / ``predict_on_batch``, never ``fit`` /
        ``predict``: ``features`` and ``adjacency`` are whole-graph
        inputs, and axis-0 batching would slice them along with
        ``edge_pairs``. Embeddings live in Euclidean space; the decoder
        scores them by Euclidean distance.
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

        self.backbone = SHGCNModel(
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            output_activation='linear',
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_curvature=use_curvature,
            name='shgcn_backbone'
        )

        self.decoder = FermiDiracDecoder(name='fermi_dirac_decoder')

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the backbone and the Fermi-Dirac decoder.

        # DECISION plan-2026-08-19T163559-499b6f0e/D-029: build() must stay; without
        # it FermiDiracDecoder loads unbuilt and silently reverts to r=2.0/t=1.0
        # on reload (measured delta 1.497e-01). See decisions.md D-029.

        :param input_shape: ``[features_shape, adjacency_shape,
            edge_pairs_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
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
        """Return ``(num_edges,)``.

        :param input_shape: ``[features_shape, adjacency_shape,
            edge_pairs_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[2][0],)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Embed nodes, gather edge endpoints, and decode edge probability.

        :param inputs: ``[features, adjacency, edge_pairs]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether dropout is active.
        :type training: Optional[bool]
        :return: Edge probabilities, shape ``[num_edges]``.
        :rtype: keras.KerasTensor
        """
        features, adjacency, edge_pairs = inputs

        embeddings = self.backbone([features, adjacency], training=training)

        src_indices = edge_pairs[:, 0]
        tgt_indices = edge_pairs[:, 1]

        src_embeddings = keras.ops.take(embeddings, src_indices, axis=0)
        tgt_embeddings = keras.ops.take(embeddings, tgt_indices, axis=0)

        probabilities = self.decoder([src_embeddings, tgt_embeddings])

        return probabilities

    def get_config(self) -> dict:
        """Return the config needed to reconstruct this model.

        :return: Constructor arguments.
        :rtype: dict
        """
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
    """Build a base sHGCN feature-extraction model.

    sHGCN has no named scale family in the paper -- it is specified by a
    per-layer hidden-dimension list chosen for the dataset at hand -- so
    this factory has no ``MODEL_VARIANTS`` table and applies a plain
    two-layer default instead.

    :param hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
        ``[64, 32]``.
    :type hidden_dims: Optional[List[int]]
    :param output_dim: Dimension of the output layer.
    :type output_dim: int
    :param output_activation: Activation applied to the output layer.
    :type output_activation: Optional[Union[str, callable]]
    :param dropout_rate: Dropout applied between layers.
    :type dropout_rate: float
    :param use_bias: Whether the sHGCN layers use a bias term.
    :type use_bias: bool
    :param use_curvature: Whether learnable curvature is enabled.
    :type use_curvature: bool
    :param kwargs: Forwarded to the model constructor.
    :return: A configured model, called with ``[features, adjacency]``.
    :rtype: SHGCNModel
    :raises ValueError: If any argument is outside its valid range.
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
    """Build an sHGCN node-classification model.

    :param num_classes: Number of target classes. Required -- it is a
        property of the dataset, not something a default can guess.
    :type num_classes: int
    :param hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
        ``[64, 32]``.
    :type hidden_dims: Optional[List[int]]
    :param embedding_dim: Dimension of the node embedding fed to the
        classifier head.
    :type embedding_dim: int
    :param dropout_rate: Dropout applied between layers.
    :type dropout_rate: float
    :param use_bias: Whether the sHGCN layers use a bias term.
    :type use_bias: bool
    :param use_curvature: Whether learnable curvature is enabled.
    :type use_curvature: bool
    :param kwargs: Forwarded to the model constructor.
    :return: A configured model. It is called with
        ``[features, adjacency]`` and returns class probabilities -- its
        head is ``Dense(num_classes, activation='softmax')``, so compile
        with ``from_logits=False`` (the Keras default).
    :rtype: SHGCNNodeClassifier
    :raises ValueError: If any argument is outside its valid range.
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
    """Build an sHGCN link-prediction model.

    :param hidden_dims: Per-layer hidden dimensions. ``None`` resolves to
        ``[64, 32]``.
    :type hidden_dims: Optional[List[int]]
    :param embedding_dim: Node embedding dimension scored by the
        Fermi-Dirac decoder.
    :type embedding_dim: int
    :param dropout_rate: Dropout applied between layers.
    :type dropout_rate: float
    :param use_bias: Whether the sHGCN layers use a bias term.
    :type use_bias: bool
    :param use_curvature: Whether learnable curvature is enabled.
    :type use_curvature: bool
    :param kwargs: Forwarded to the model constructor.
    :return: A configured model, called with
        ``[features, adjacency, edge_pairs]``; returns edge probabilities.
    :rtype: SHGCNLinkPredictor
    :raises ValueError: If any argument is outside its valid range.
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
