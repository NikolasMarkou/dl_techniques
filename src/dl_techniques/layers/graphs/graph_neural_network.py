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
from ...initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.graphs.graph_neural_network")
class GraphNeuralNetworkLayer(keras.layers.Layer):
    """Configurable multi-paradigm Graph Neural Network layer.

    Provides a unified framework for graph representation learning by stacking
    multiple message-passing blocks with configurable paradigms (GCN, GraphSAGE,
    GAT, GIN). Each block performs neighborhood aggregation H' = f(A, H, W),
    followed by activation, dropout, optional residual connection, and
    normalization. The layer supports four message-passing schemes:
    GCN uses spectral convolution H' = sigma(D^{-1/2} A D^{-1/2} H W),
    GraphSAGE separates self and neighbor transforms H' = sigma(W_self H + W_neigh AGG(AH)),
    GAT applies learnable attention H' = sigma(sum_j alpha_{ij} W h_j),
    and GIN maximises expressiveness H' = MLP((1+eps) h_i + sum_j h_j).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │  Input: (node_features, adjacency_matrix)        │
        │         [B, N, D]        [B, N, N]               │
        └─────────────────────┬────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────┐
        │  Adjacency Normalisation  D⁻¹ A                  │
        └─────────────────────┬────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────┐
        │  For i = 1 .. num_layers:                        │
        │  ┌────────────────────────────────────────────┐  │
        │  │ Message Passing (GCN/GraphSAGE/GAT/GIN)    │  │
        │  └──────────────────┬─────────────────────────┘  │
        │                     ▼                            │
        │  ┌────────────────────────────────────────────┐  │
        │  │ Activation  σ(·)                           │  │
        │  └──────────────────┬─────────────────────────┘  │
        │                     ▼                            │
        │  ┌────────────────────────────────────────────┐  │
        │  │ Dropout                                    │  │
        │  └──────────────────┬─────────────────────────┘  │
        │                     ▼                            │
        │  ┌────────────────────────────────────────────┐  │
        │  │ Residual  h = h_in + h_new                 │  │
        │  └──────────────────┬─────────────────────────┘  │
        │                     ▼                            │
        │  ┌────────────────────────────────────────────┐  │
        │  │ Normalisation (Layer / RMS / Batch)        │  │
        │  └──────────────────┬─────────────────────────┘  │
        └─────────────────────┼────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────┐
        │  Final Aggregation (mean/max/sum/attention/none) │
        └─────────────────────┬────────────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────┐
        │  Output  [B, N or 1, D]                          │
        └──────────────────────────────────────────────────┘

    :param concept_dim: Dimension of concept/node embeddings. Must be positive.
    :type concept_dim: int
    :param num_layers: Number of GNN layers to stack. Defaults to 3.
    :type num_layers: int
    :param message_passing: Message-passing paradigm
        (``'gcn'``, ``'graphsage'``, ``'gat'``, ``'gin'``). Defaults to ``'gcn'``.
    :type message_passing: str
    :param aggregation: Final node aggregation
        (``'mean'``, ``'max'``, ``'attention'``, ``'sum'``, ``'none'``).
        Defaults to ``'attention'``.
    :type aggregation: str
    :param normalization: Per-block normalisation
        (``'none'``, ``'batch'``, ``'layer'``, ``'rms'``). Defaults to ``'layer'``.
    :type normalization: str
    :param activation: Activation function name or callable. Defaults to ``'relu'``.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout probability in ``[0, 1]``. Defaults to 0.1.
    :type dropout_rate: float
    :param use_residual: Whether to add residual connections. Defaults to ``True``.
    :type use_residual: bool
    :param num_attention_heads: Attention heads for GAT / attention aggregation.
        Must divide ``concept_dim``. Defaults to 4.
    :type num_attention_heads: int
    :param epsilon: Learnable self-loop weight for GIN. Defaults to 0.0.
    :type epsilon: float
    :param kernel_initializer: Initializer for weight matrices.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weight matrices.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.gnn_layers = []
        # DECISION plan-2026-08-19T163559-499b6f0e/D-091
        # GraphSAGE's two per-block transforms live in TWO FLAT, PARALLEL
        # LISTS -- one per role, indexed by block -- and NOT in the obvious
        # `self.gnn_layers.append({'self': Dense(...), 'neighbor': Dense(...)})`.
        # Do NOT "tidy" them back into a list of dicts: Keras 3.8 does not
        # write a layer container nested two or more levels deep to
        # `model.weights.h5` when its owner is a `keras.layers.Layer`, and this
        # class IS a `Layer`. MEASURED on this exact class at
        # `message_passing='graphsage'`, num_layers=2: the list-of-dicts form
        # archived 12 of 20 tensors -- ALL EIGHT GraphSAGE Dense weights absent
        # -- and a perturb / save / reload comparison came back with
        # max|dW| = 1.407037e+00 and max|dOut| = 1.659781e+00, i.e. a different
        # model. The other three modes hold a layer ONE level deep and were
        # clean in both arms (gcn 16/16, gat 28/28, gin 21/21, all at exactly
        # 0.0). See decisions.md D-091 and D-026 (the same mechanism in
        # `models/vision/masked_autoencoder/conv_decoder.py`).
        self.sage_self_layers = []
        self.sage_neighbor_layers = []
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
                        kernel_initializer=clone_initializer(self.kernel_initializer),
                        bias_initializer=clone_initializer(self.bias_initializer),
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'gcn_dense_{i}'
                    )
                )
            elif self.message_passing == 'graphsage':
                # GraphSAGE uses separate transformations for self and neighbors
                # DECISION plan-2026-08-19T163559-499b6f0e/D-091 (see __init__)
                self.sage_self_layers.append(
                    keras.layers.Dense(
                        self.concept_dim,
                        activation=None,
                        kernel_initializer=clone_initializer(self.kernel_initializer),
                        bias_initializer=clone_initializer(self.bias_initializer),
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'sage_self_{i}'
                    )
                )
                self.sage_neighbor_layers.append(
                    keras.layers.Dense(
                        self.concept_dim,
                        activation=None,
                        kernel_initializer=clone_initializer(self.kernel_initializer),
                        bias_initializer=clone_initializer(self.bias_initializer),
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f'sage_neighbor_{i}'
                    )
                )
                # index alignment with `gnn_layers` is preserved so every
                # `for i in range(self.num_layers)` loop below stays uniform.
                self.gnn_layers.append(None)
            elif self.message_passing == 'gat':
                # GAT uses multi-head attention
                # DECISION plan-2026-09-07T183458-be1c267e/D-005
                # `output_shape=self.concept_dim` is LOAD-BEARING, not
                # decorative. `keras.layers.MultiHeadAttention` defaults its
                # output width to the QUERY tensor's last axis -- NOT to
                # `num_heads * key_dim` -- so without it the GAT branch emits
                # the raw input width `D` at every block and the stack never
                # reaches `concept_dim`. MEASURED at `concept_dim=32`, D=16:
                # the forward pass returned `(2, 5, 16)` while
                # `compute_output_shape` promised `(2, 5, 32)`. Do NOT remove
                # it on the grounds that `key_dim` already mentions
                # `concept_dim`: `key_dim` sizes the per-head projection, not
                # the output projection. At `concept_dim == D` it is a
                # measured no-op (identical weight shapes, identical weight
                # bytes, identical output). See decisions.md D-005.
                # DECISION plan-2026-09-07T183458-be1c267e/D-007
                # `self.kernel_initializer` / `self.bias_initializer` are passed
                # here WITHOUT `clone_initializer(...)`, deliberately, unlike the
                # `gcn_dense_{i}` / `sage_self_{i}` / `sage_neighbor_{i}` Dense
                # layers above. Do NOT "fix" this by adding the wrapper: it would
                # add a guard that is green before AND after its own revert, i.e.
                # a test that can never fail, which is this repo's most-repeated
                # test defect.
                #
                # Why the site is already independent: stock
                # `keras.layers.MultiHeadAttention._get_common_kwargs_for_sublayer`
                # runs `initializer.__class__.from_config(initializer.get_config())`
                # for EVERY sub-layer (query / key / value / attention_output), so
                # the callee re-clones for us. MEASURED on this class: all 24
                # `gat_attention_{i}` + `aggregation_attention` weights come out
                # independent of a replay from the shared instance, while the four
                # `sage_*` Dense weights come out bit-identical to it.
                #
                # The mechanism, stated exactly: `GlorotUniform().get_config()`
                # reports `{'seed': None}` even when the LIVE instance already has
                # a resolved `.seed`, so `from_config` self-assigns a fresh seed and
                # the tie breaks. This is the OPPOSITE of `clone.py` exemption 3,
                # where a `get_config()` round trip that RAISES falls back to
                # `copy.deepcopy`, which COPIES the resolved seed and leaves the site
                # tied. A round trip that succeeds unties; a round trip that raises
                # stays tied. Do not "correct" one into the other.
                #
                # The dependency is MONITORED, not assumed:
                # `TestTheCalleeReClonesTheSharedInitializer` in
                # `tests/test_layers/test_graphs/test_graph_neural_network.py` pins
                # the stock-`MultiHeadAttention` and `MLPBlock` behaviour and goes
                # RED, naming this site, if a future version drops the re-clone.
                # Independence here holds for a RANDOM SEEDLESS initializer; the
                # exceptions are a SEEDED instance (replays by contract, across
                # differing shapes too), a DETERMINISTIC one (`'zeros'`/`'ones'`/
                # `Constant`, `Identity` at 2-D -- identical and correctly so), and
                # a CUSTOM one failing the `get_config()` round trip (falls back to
                # `copy.deepcopy`, keeping the resolved seed). See decisions.md
                # D-007 and `src/dl_techniques/initializers/clone.py`.
                self.gnn_layers.append(
                    keras.layers.MultiHeadAttention(
                        num_heads=self.num_attention_heads,
                        key_dim=self.concept_dim // self.num_attention_heads,
                        output_shape=self.concept_dim,
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
                # DECISION plan-2026-09-07T183458-be1c267e/D-007
                # Not wrapped in `clone_initializer(...)`, deliberately: `MLPBlock`
                # already clones per sub-layer for both `fc1` and `fc2`
                # (`layers/ffn/mlp.py:261,271`), so wrapping here would buy nothing
                # and would ship a guard that cannot be reddened by reverting it.
                # MEASURED: all 8 `gin_mlp_{i}` weights are independent of a replay
                # from the shared instance. Same reasoning, same exemptions and the
                # same monitoring test as the `gat_attention_{i}` anchor above --
                # see decisions.md D-007.
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
            # DECISION plan-2026-09-07T183458-be1c267e/D-007
            # Not wrapped in `clone_initializer(...)`, deliberately: this is a stock
            # `keras.layers.MultiHeadAttention`, which re-clones its initializer for
            # every sub-layer. MEASURED independent. See the full anchor at the
            # `gat_attention_{i}` construction above and decisions.md D-007.
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

    def _block_input_shape(
            self,
            node_shape: Tuple[Optional[int], ...],
            block_index: int
    ) -> Tuple[Optional[int], ...]:
        """Return the node-feature shape entering GNN block ``block_index``.

        Every block widens (or narrows) the running node features to
        ``concept_dim``, so only block 0 sees the layer's raw input width.
        ``block_index == self.num_layers`` therefore names the shape leaving
        the whole stack, which is what :meth:`compute_output_shape` needs.

        This is the single home for the stack's shape arithmetic: both
        :meth:`build` and :meth:`compute_output_shape` derive from it, so they
        cannot drift apart. It is pure -- it reads only ``self.concept_dim``
        and its arguments -- and is therefore valid on an UNBUILT layer.

        :param node_shape: Shape of the node-feature tensor handed to the layer.
        :type node_shape: Tuple[Optional[int], ...]
        :param block_index: Index of the block, in ``[0, self.num_layers]``.
        :type block_index: int
        :return: Node-feature shape entering that block.
        :rtype: Tuple[Optional[int], ...]
        """
        if block_index == 0:
            return tuple(node_shape)
        return tuple(node_shape[:-1]) + (self.concept_dim,)

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """Build the layer and all its sub-layers.

        :param input_shape: Tuple of (node_features_shape, adjacency_shape).
        :type input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
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

        # Build all sub-layers explicitly, each at the shape it actually sees.
        # DECISION plan-2026-09-07T183458-be1c267e/D-006
        # Two distinct shapes per block, and they are NOT interchangeable:
        # the message-passing sub-layer consumes the block's INPUT width
        # (`node_shape` at block 0, `concept_dim` after that), while dropout
        # and normalization run on the block's OUTPUT, which is `concept_dim`
        # wide at EVERY block including block 0. Do NOT collapse them back to
        # one `node_shape` for the whole loop -- that was the shipped defect:
        # MEASURED crashes at `concept_dim=8, num_layers=2` (`ValueError` in
        # `gcn_dense_1`, expected axis -1 == 16, got (2, 5, 8)) and at
        # `concept_dim=32, num_layers=1` (`InvalidArgumentError` inside
        # `LayerNormalization.call()`, reshape of 16 values into 32). It
        # survived because every test pinned `concept_dim == D`, which makes
        # the two shapes coincide. See decisions.md D-006.
        for i in range(self.num_layers):
            block_input_shape = self._block_input_shape(node_shape, i)
            block_output_shape = self._block_input_shape(node_shape, i + 1)

            if self.message_passing == 'gcn':
                # GCN layer expects node features
                self.gnn_layers[i].build(block_input_shape)

            elif self.message_passing == 'graphsage':
                # Build both self and neighbor transformations
                self.sage_self_layers[i].build(block_input_shape)
                self.sage_neighbor_layers[i].build(block_input_shape)

            elif self.message_passing == 'gat':
                # GAT attention expects query and key inputs
                self.gnn_layers[i].build(block_input_shape, block_input_shape)

            elif self.message_passing == 'gin':
                # GIN MLP expects aggregated features
                self.gnn_layers[i].build(block_input_shape)

            # Build dropout on the block's OUTPUT
            self.dropout_layers[i].build(block_output_shape)

            # Build normalization if present, also on the block's OUTPUT
            if self.norm_layers[i] is not None:
                self.norm_layers[i].build(block_output_shape)

        # Build final aggregation attention if needed -- it runs on the
        # tensor leaving the last block, not on the layer's raw input.
        if self.aggregation_attention is not None:
            stack_output_shape = self._block_input_shape(node_shape, self.num_layers)
            self.aggregation_attention.build(stack_output_shape, stack_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process concept graph through GNN layers.

        :param inputs: Tuple of ``(node_features, adjacency_matrix)``.
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Updated node embeddings based on aggregation type.
        :rtype: keras.KerasTensor
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
                self_features = self.sage_self_layers[i](h)
                neighbor_messages = keras.ops.matmul(normalized_adj, h)
                neighbor_features = self.sage_neighbor_layers[i](neighbor_messages)
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
        """Compute output shape based on aggregation type.

        :param input_shape: Tuple of (node_features_shape, adjacency_shape).
        :type input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        node_shape, _ = input_shape
        # Same pure helper `build()` uses, so the two cannot disagree.
        stack_output_shape = self._block_input_shape(node_shape, self.num_layers)
        batch_size = stack_output_shape[0]
        num_nodes = stack_output_shape[1]
        feature_dim = stack_output_shape[-1]

        if self.aggregation in ['mean', 'max', 'sum']:
            # Global pooling reduces to single node
            return (batch_size, 1, feature_dim)
        else:
            # Keep all nodes
            return (batch_size, num_nodes, feature_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
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
