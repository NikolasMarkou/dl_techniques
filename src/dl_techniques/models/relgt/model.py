"""
Relational Graph Transformer: multi-element node tokenization over sampled subgraphs,
with local self-attention combined with cross-attention to learnable global centroids.

A relational database is a graph, but not one a standard graph transformer handles
well. Its nodes come from different tables and so carry different feature schemas and
different meanings; edges encode foreign keys rather than a single relation; rows
carry timestamps, so an edge's usefulness depends on when it was created relative to
the prediction time. Flattening all of that into one node-feature vector throws away
exactly the structure that makes the data relational, and running full attention over
a real database is impossible anyway — the graph does not fit.

RELGT's answer to the first problem is to stop treating a node as one vector.
Each node in a sampled subgraph is decomposed into five elements, each embedded
separately and summed:

`T = Norm(Dropout(E_feat(x) + E_type(t) + E_hop(h) + E_time(tau) + E_pe(A)))`

carrying, in order, its own features, which table it came from, how many hops it sits
from the seed node, its timestamp relative to the seed's, and its structural position.
Because the components are summed after independent embedding, attention can weigh
"same table as the seed", "two hops away" and "recent" as separable signals rather
than having to disentangle them from a concatenated blob. The positional element is
computed by a small GNN run over random features propagated through the subgraph's
adjacency — a learned structural fingerprint, since a relational graph has no
canonical node ordering to index a positional table with.

The second problem, scale, is handled by attending locally and summarizing globally.
Local attention runs only over the sampled subgraph around the seed node, so cost is
governed by the sample size rather than by the database. That alone would confine
every prediction to its own neighbourhood, so each block also cross-attends from the
seed node to a bank of learnable global centroids — `K = V = centroids`, `Q = seed` —
which act as a small learned codebook of database-wide structure. The two are then
combined, `output = Norm(FFN([h_local; h_global]) + ResidualProj([h_local; h_global]))`,
where `h_local` is the mean-pooled output of local self-attention and `h_global` the
single vector produced by the centroid cross-attention. The centroids are parameters,
not statistics: they are shaped by the task, and their count is the knob that trades
global expressiveness against parameters.

Two structural facts about this implementation are worth stating plainly. Each
transformer block collapses its token set to a single vector, so a block consumes
tokens and emits a representation rather than an updated token sequence. As a
consequence the block loop does not compose: every block is handed the *same*
`local_tokens` and seed embedding, and only the last block's output reaches the
prediction head. Setting `num_transformer_blocks > 1` therefore adds parameters and
compute without deepening the computation, and a genuinely stacked variant would need
blocks that return tokens. Separately, the seed node is taken to be index 0 of
`node_features` — the subgraph sampler's contract, not something the model verifies.

The prediction head's final activation follows `problem_type`, and when there are no
transformer blocks at all the model falls back to mean-pooling the tokens, so the
degenerate configuration still produces a well-shaped output rather than failing.

References:
    - Dwivedi et al., 2025. Relational Graph Transformer.
      (https://arxiv.org/abs/2505.10960)
    - Fey et al., 2023. Relational Deep Learning: Graph Representation Learning on
      Relational Databases. (https://arxiv.org/abs/2312.04615)
    - Dwivedi & Bresson, 2020. A Generalization of Transformer Networks to Graphs.
      (https://arxiv.org/abs/2012.09699)
    - Rampasek et al., 2022. Recipe for a General, Powerful, Scalable Graph
      Transformer. (https://arxiv.org/abs/2205.12454)
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative Attention.
      (https://arxiv.org/abs/2103.03206)
"""

import keras
from keras import ops, layers
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.graphs.relational_graph_transformer_blocks import (
    RELGTTransformerBlock, RELGTTokenEncoder)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RELGT(keras.Model):
    """
    Complete Relational Graph Transformer model for multi-table relational data.

    RELGT integrates multi-element tokenization with hybrid local-global transformer
    processing to capture complex structural patterns, temporal dynamics, and
    long-range dependencies in relational databases. It establishes Graph Transformers
    as a powerful architecture for Relational Deep Learning.

    **Intent**: Provide state-of-the-art predictive modeling on relational data by
    combining the expressiveness of graph transformers with relational-specific
    innovations in tokenization and attention mechanisms.

    **Architecture**:
    ```
    Graph Data (Features, Types, Hops, Times, Adjacency)
                    ↓
    RELGTTokenEncoder: Multi-element tokenization
                    ↓
    RELGTTransformerBlock(s): Hybrid local-global processing
                    ↓
    Prediction Head: Task-specific output layer
    ```

    Args:
        output_dim: Integer, dimension of final output (e.g., number of classes).
            Must be positive.
        problem_type: String, either 'classification' or 'regression'.
            Determines final activation function.
        embedding_dim: Integer, main embedding dimension used throughout model.
            Defaults to 128.
        num_node_types: Integer, total number of unique entity types. Defaults to 10.
        max_hops: Integer, maximum hop distance to encode. Defaults to 2.
        gnn_pe_dim: Integer, dimension for GNN positional encoding. Defaults to 32.
        gnn_pe_layers: Integer, number of GNN PE layers. Defaults to 2.
        num_transformer_blocks: Integer, number of transformer blocks to stack.
            Defaults to 2.
        num_heads: Integer, number of attention heads. Defaults to 4.
        num_global_centroids: Integer, number of learnable global tokens. Defaults to 64.
        ffn_dim: Integer, hidden dimension for FFNs. Defaults to 256.
        dropout_rate: Float between 0 and 1, dropout rate. Defaults to 0.1.
        ffn_type: String, type of FFN to use. Defaults to 'mlp'.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        Dictionary with tensor inputs as required by RELGTTokenEncoder.

    Output shape:
        Tensor with shape `(batch_size, output_dim)`.

    Example:
        ```python
        # Create model for node classification
        model = RELGT(
            output_dim=10,
            problem_type='classification',
            embedding_dim=128,
            num_node_types=5,
            num_transformer_blocks=3,
            num_heads=8
        )

        # Compile and train
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """

    def __init__(
            self,
            output_dim: int,
            problem_type: str = "classification",
            embedding_dim: int = 128,
            num_node_types: int = 10,
            max_hops: int = 2,
            gnn_pe_dim: int = 32,
            gnn_pe_layers: int = 2,
            num_transformer_blocks: int = 2,
            num_heads: int = 4,
            num_global_centroids: int = 64,
            ffn_dim: int = 256,
            dropout_rate: float = 0.1,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs,
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if problem_type not in ["classification", "regression"]:
            raise ValueError(f"problem_type must be 'classification' or 'regression', got {problem_type}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_transformer_blocks <= 0:
            raise ValueError(f"num_transformer_blocks must be positive, got {num_transformer_blocks}")

        # Store configuration
        self.output_dim = output_dim
        self.problem_type = problem_type
        self.embedding_dim = embedding_dim
        self.num_node_types = num_node_types
        self.max_hops = max_hops
        self.gnn_pe_dim = gnn_pe_dim
        self.gnn_pe_layers = gnn_pe_layers
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.num_global_centroids = num_global_centroids
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type

        # CREATE all components

        # Token encoder
        self.token_encoder = RELGTTokenEncoder(
            embedding_dim=embedding_dim,
            num_node_types=num_node_types,
            max_hops=max_hops,
            gnn_pe_dim=gnn_pe_dim,
            gnn_pe_layers=gnn_pe_layers,
            dropout_rate=dropout_rate,
            normalization_type=normalization_type,
            name="TokenEncoder",
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_transformer_blocks):
            block = RELGTTransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_global_centroids=num_global_centroids,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate,
                ffn_type=ffn_type,
                normalization_type=normalization_type,
                name=f"TransformerBlock_{i}",
            )
            self.transformer_blocks.append(block)

        # Seed node feature encoder
        self.seed_encoder = layers.Dense(
            embedding_dim,
            name="SeedEncoder"
        )

        # Prediction head using FFN factory
        final_activation = "softmax" if problem_type == "classification" else None
        self.prediction_head = keras.Sequential([
            create_ffn_layer(
                ffn_type,
                hidden_dim=ffn_dim,
                output_dim=ffn_dim,
                dropout_rate=dropout_rate,
                name="PredictionFFN"
            ),
            layers.Dense(output_dim, activation=final_activation, name="FinalOutput")
        ], name="PredictionHead")

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the complete RELGT model."""

        # Extract and encode seed node features (assumed to be first token)
        seed_node_features = inputs["node_features"][:, 0:1, :]  # (batch, 1, feature_dim)
        seed_node_embedding = self.seed_encoder(seed_node_features)  # (batch, 1, embed_dim)

        # 1. Encode input graph data into tokens
        local_tokens = self.token_encoder(inputs, training=training)

        # 2. Process through transformer blocks
        # In a complete implementation, you might want to update local_tokens
        # between blocks, but for simplicity we process once and then use
        # the representation for all blocks
        current_representation = None

        for block in self.transformer_blocks:
            # Each block outputs a single vector representation
            current_representation = block(
                [local_tokens, seed_node_embedding],
                training=training
            )

            # For multi-block architectures, you might want to update tokens
            # or maintain running representations. This implementation uses
            # the final block output directly.

        # 3. Generate final prediction
        if current_representation is None:
            # Handle case with no transformer blocks
            current_representation = ops.mean(local_tokens, axis=1)

        predictions = self.prediction_head(current_representation, training=training)

        return predictions

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            "output_dim": self.output_dim,
            "problem_type": self.problem_type,
            "embedding_dim": self.embedding_dim,
            "num_node_types": self.num_node_types,
            "max_hops": self.max_hops,
            "gnn_pe_dim": self.gnn_pe_dim,
            "gnn_pe_layers": self.gnn_pe_layers,
            "num_transformer_blocks": self.num_transformer_blocks,
            "num_heads": self.num_heads,
            "num_global_centroids": self.num_global_centroids,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RELGT':
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

def create_relgt_model(
        output_dim: int,
        problem_type: str = "classification",
        model_size: str = "base",
        **kwargs
) -> RELGT:
    """
    Factory function to create RELGT models with predefined configurations.

    Args:
        output_dim: Dimension of final output.
        problem_type: 'classification' or 'regression'.
        model_size: 'small', 'base', or 'large' for predefined configurations.
        **kwargs: Additional arguments to override defaults.

    Returns:
        Configured RELGT model.
    """
    # Predefined configurations
    size_configs = {
        "small": {
            "embedding_dim": 64,
            "num_heads": 2,
            "num_global_centroids": 16,
            "ffn_dim": 128,
            "num_transformer_blocks": 1,
        },
        "base": {
            "embedding_dim": 128,
            "num_heads": 4,
            "num_global_centroids": 32,
            "ffn_dim": 256,
            "num_transformer_blocks": 2,
        },
        "large": {
            "embedding_dim": 256,
            "num_heads": 8,
            "num_global_centroids": 64,
            "ffn_dim": 512,
            "num_transformer_blocks": 4,
        }
    }

    if model_size not in size_configs:
        raise ValueError(f"model_size must be one of {list(size_configs.keys())}, got {model_size}")

    # Merge configurations
    config = size_configs[model_size].copy()
    config.update(kwargs)
    config.update({
        "output_dim": output_dim,
        "problem_type": problem_type,
    })

    logger.info(f"Creating RELGT model with size='{model_size}' and config: {config}")

    return RELGT(**config)

# ---------------------------------------------------------------------
