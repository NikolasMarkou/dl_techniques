import keras
from keras import ops
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PointCloudAutoencoder(keras.layers.Layer):
    """DGCNN-based autoencoder for point cloud feature extraction.

    Uses shared EdgeConv-style blocks to extract multi-scale local and global
    features from two point clouds and reconstructs them via a shared MLP
    decoder. Three EdgeConv blocks capture progressively abstract geometric
    structure; their outputs are concatenated and projected to 1024-D local
    features, then max- and average-pooled to form a 2048-D global descriptor.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────┐
        │  Input: (Source PC, Target PC)  [B,N,3]   │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │  Shared Encoder (per PC)                  │
        │  ┌─────────────────────────────────────┐  │
        │  │ EdgeConv 1  KNN→MLP[64,64]→Max→f1  │  │
        │  │ EdgeConv 2  KNN→MLP[64,64]→Max→f2  │  │
        │  │ EdgeConv 3  KNN→MLP[64]→Max→f3     │  │
        │  └──────────────────┬──────────────────┘  │
        │                     ▼                     │
        │  Concat[f1,f2,f3]→MLP[1024]→local [B,N,1024]│
        │  MaxPool + AvgPool → global [B, 2048]     │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │  Shared Decoder (per PC)                  │
        │  Dense[2048]→Dense[1024]→Dense[N*3]       │
        │  → Reshape → reconstruction [B, N, 3]     │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │  Output: (reconstructions, locals, globals)│
        └───────────────────────────────────────────┘

    :param k_neighbors: Number of neighbours for k-NN graphs. Defaults to 20.
    :type k_neighbors: int
    :param kwargs: Additional arguments for the ``Layer`` base class.
    """

    def __init__(self, k_neighbors: int = 20, **kwargs):
        """Initialise PointCloudAutoencoder."""
        super().__init__(**kwargs)

        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")

        self.k_neighbors = k_neighbors

        # CREATE sub-layers in __init__
        # Encoder MLPs for EdgeConv blocks (progressive feature learning)
        self.mlp1 = self._make_mlp([64, 64], "encoder_mlp1")
        self.mlp2 = self._make_mlp([64, 64], "encoder_mlp2")
        self.mlp3 = self._make_mlp([64], "encoder_mlp3")

        # Final shared MLP before pooling (feature aggregation)
        self.mlp4 = self._make_mlp([1024], "encoder_mlp4")

        # Decoder MLPs (progressive reconstruction)
        self.decoder_mlp1 = keras.layers.Dense(2048, activation='leaky_relu', name="decoder_mlp1")
        self.decoder_mlp2 = keras.layers.Dense(1024, activation='leaky_relu', name="decoder_mlp2")
        # decoder_mlp3 will be created in build() as its size depends on input num_points

    def _make_mlp(self, units: List[int], name: str) -> keras.Sequential:
        """Create a Sequential MLP with ReLU activations.

        :param units: List of hidden unit sizes.
        :type units: List[int]
        :param name: Name for the Sequential model.
        :type name: str
        :return: Sequential MLP.
        :rtype: keras.Sequential
        """
        return keras.Sequential([
            keras.layers.Dense(u, activation='relu') for u in units
        ], name=name)

    def build(self, input_shape: Tuple[Tuple, Tuple]):
        """Build shape-dependent sub-layers.

        :param input_shape: Tuple of ``(source_shape, target_shape)``.
        :type input_shape: Tuple[Tuple, Tuple]
        """
        source_shape, target_shape = input_shape
        # Assuming both clouds are padded/sampled to the same number of points
        num_points = source_shape[1]

        if num_points is None:
            raise ValueError("The number of points (dimension 1) must be defined.")

        # Create the final decoder layer which depends on input shape
        # Output size = num_points × 3 (flattened point cloud)
        self.decoder_mlp3 = keras.layers.Dense(num_points * 3, name="decoder_mlp3")

        # Explicitly build sub-layers for robust serialization
        # This ensures weight variables exist before weight restoration during loading

        # Encoder MLPs - build with expected input shapes from graph features
        self.mlp1.build((None, None, None, 6))  # Edge features: [pt_feat; neighbor_feat - pt_feat]
        self.mlp2.build((None, None, None, 128))  # Combined features from previous EdgeConv
        self.mlp3.build((None, None, None, 128))  # Combined features from previous EdgeConv
        self.mlp4.build((None, None, 192))  # Concatenated multi-scale features [f1, f2, f3]

        # Decoder MLPs - build with progressive dimensionality reduction
        self.decoder_mlp1.build((None, 2048))  # Input: global feature (max + avg pooling)
        self.decoder_mlp2.build((None, 2048))  # Input: decoder_mlp1 output
        self.decoder_mlp3.build((None, 1024))  # Input: decoder_mlp2 output

        super().build(input_shape)

    def _encode(self, pc: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Shared encoder using DGCNN-style EdgeConv blocks.

        :param pc: Point cloud ``(B, N, 3)``.
        :type pc: keras.KerasTensor
        :return: Tuple of (local_features ``(B, N, 1024)``, global_features ``(B, 2048)``).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # EdgeConv Block 1: Extract first-level geometric features
        # Constructs k-NN graph and computes edge features for each point-neighbor pair
        # Edge feature = [point_features; neighbor_features - point_features]
        graph1 = keras.ops.get_graph_feature(pc, k=self.k_neighbors)
        f1 = self.mlp1(graph1)  # Apply MLP to edge features
        f1 = ops.max(f1, axis=2)  # Max pool over neighbors → (B, N, 64)

        # EdgeConv Block 2: Extract second-level features from f1
        # Now working in feature space rather than coordinate space
        graph2 = keras.ops.get_graph_feature(f1, k=self.k_neighbors)
        f2 = self.mlp2(graph2)
        f2 = ops.max(f2, axis=2)  # → (B, N, 64)

        # EdgeConv Block 3: Extract third-level features from f2
        # Captures higher-order geometric relationships
        graph3 = keras.ops.get_graph_feature(f2, k=self.k_neighbors)
        f3 = self.mlp3(graph3)
        f3 = ops.max(f3, axis=2)  # → (B, N, 64)

        # Multi-scale feature aggregation
        # Concatenate features from all levels to capture both fine and coarse details
        combined_features = ops.concatenate([f1, f2, f3], axis=-1)  # → (B, N, 192)
        local_features = self.mlp4(combined_features)  # → (B, N, 1024)

        # Global feature extraction via symmetric pooling functions
        # Max pooling captures most prominent features, avg pooling captures overall distribution
        global_max = ops.max(local_features, axis=1)  # → (B, 1024)
        global_avg = ops.mean(local_features, axis=1)  # → (B, 1024)
        global_features = ops.concatenate([global_max, global_avg], axis=-1)  # → (B, 2048)

        return local_features, global_features

    def _decode(self, global_feature: keras.KerasTensor, num_points: int) -> keras.KerasTensor:
        """Shared decoder reconstructing point cloud from global features.

        :param global_feature: Global descriptor ``(B, 2048)``.
        :type global_feature: keras.KerasTensor
        :param num_points: Number of points to reconstruct.
        :type num_points: int
        :return: Reconstructed point cloud ``(B, N, 3)``.
        :rtype: keras.KerasTensor
        """
        # Progressive expansion from global feature to full point cloud
        x = self.decoder_mlp1(global_feature)  # → (B, 2048)
        x = self.decoder_mlp2(x)  # → (B, 1024)
        x = self.decoder_mlp3(x)  # → (B, num_points × 3)

        # Reshape flat vector to point cloud format
        return ops.reshape(x, (-1, num_points, 3))  # → (B, num_points, 3)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> Tuple[Tuple, Tuple, Tuple]:
        """Forward pass through the autoencoder for both point clouds.

        :param inputs: Tuple of ``(source_pc, target_pc)``.
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :return: Tuple of (reconstructions, local_features, global_features).
        :rtype: Tuple[Tuple, Tuple, Tuple]
        """
        source_pc, target_pc = inputs
        num_points_source = ops.shape(source_pc)[1]
        num_points_target = ops.shape(target_pc)[1]

        # Process source point cloud through shared encoder-decoder
        # Extract hierarchical features and reconstruct
        local_x, global_x = self._encode(source_pc)
        x_rec = self._decode(global_x, num_points_source)

        # Process target point cloud through same shared encoder-decoder
        # Weight sharing ensures consistent feature space for both clouds
        local_y, global_y = self._encode(target_pc)
        y_rec = self._decode(global_y, num_points_target)

        # Return all intermediate representations for downstream tasks
        return (x_rec, y_rec), (local_x, local_y), (global_x, global_y)

    def compute_output_shape(
        self, input_shape: Tuple[Tuple, Tuple]
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """Compute output shape.

        :param input_shape: Tuple of (source_shape, target_shape) where each is
            (batch_size, num_points, 3).
        :type input_shape: Tuple[Tuple, Tuple]
        :return: Tuple of three tuples:
            1. Reconstructions: ((B, N_src, 3), (B, N_tgt, 3))
            2. Local features: ((B, N_src, 1024), (B, N_tgt, 1024))
            3. Global features: ((B, 2048), (B, 2048))
        :rtype: Tuple[Tuple, Tuple, Tuple]
        """
        source_shape, target_shape = input_shape
        batch = source_shape[0]
        n_src = source_shape[1]
        n_tgt = target_shape[1]

        reconstructions = ((batch, n_src, 3), (batch, n_tgt, 3))
        local_features = ((batch, n_src, 1024), (batch, n_tgt, 1024))
        global_features = ((batch, 2048), (batch, 2048))

        return reconstructions, local_features, global_features

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'k_neighbors': self.k_neighbors})
        return config


@keras.saving.register_keras_serializable()
class CorrespondenceNetwork(keras.layers.Layer):
    """Correspondence network estimating point-to-GMM soft assignments.

    Replaces the iterative E-step of a traditional GMM with a learned feed-forward
    network. Local per-point features are concatenated with tiled global features
    and passed through a 4-layer MLP followed by softmax, yielding per-point
    probability distributions over K Gaussian components:
    gamma[b,i,k] = P(point i belongs to component k | features).

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────┐  ┌──────────────────┐
        │ Local Features    │  │ Global Features   │
        │ [B, N, F_local]   │  │ [B, F_global]     │
        └────────┬──────────┘  └────────┬─────────┘
                 │                      ▼
                 │            ┌──────────────────┐
                 │            │ Tile → [B,N,F_g] │
                 │            └────────┬─────────┘
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────────────┐
                 │ Concat → [B, N, F_l + F_g]  │
                 └────────────┬────────────────┘
                              ▼
                 ┌─────────────────────────────┐
                 │ MLP: 1024→256→128→K         │
                 └────────────┬────────────────┘
                              ▼
                 ┌─────────────────────────────┐
                 │ Softmax → gamma [B, N, K]   │
                 └─────────────────────────────┘

    :param num_gaussians: Number of latent GMM components K. Must be positive.
    :type num_gaussians: int
    :param kwargs: Additional arguments for the ``Layer`` base class.
    """

    def __init__(self, num_gaussians: int, **kwargs):
        """Initialise CorrespondenceNetwork."""
        super().__init__(**kwargs)

        if num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")

        self.num_gaussians = num_gaussians

        # CREATE sub-layers in __init__
        # Deep MLP for learning point-to-GMM component correspondences
        # Architecture: progressively reduces dimensionality while increasing abstraction
        self.mlp = keras.Sequential([
            keras.layers.Dense(1024, activation='relu'),  # High-capacity initial layer
            keras.layers.Dense(256, activation='relu'),  # Intermediate abstraction
            keras.layers.Dense(128, activation='relu'),  # Refined features
            keras.layers.Dense(num_gaussians)  # Output logits (no activation, softmax applied in call)
        ], name="correspondence_mlp")

    def build(self, input_shape: Tuple[Tuple, Tuple]):
        """Build the MLP with the correct input dimension.

        :param input_shape: Tuple of ``(local_shape, global_shape)``.
        :type input_shape: Tuple[Tuple, Tuple]
        """
        local_shape, global_shape = input_shape

        # Input to MLP is concatenated features: [local_features; tiled_global_features]
        combined_dim = local_shape[-1] + global_shape[-1]

        # Build the correspondence estimation network with combined feature dimension
        # Use 3D shape (batch, num_points, features) to match actual call input
        self.mlp.build((None, None, combined_dim))

        super().build(input_shape)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        """Compute soft point-to-GMM component assignments.

        :param inputs: Tuple of ``(local_features, global_features)``.
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :return: Soft assignment matrix gamma ``(B, N, K)``.
        :rtype: keras.KerasTensor
        """
        local_features, global_features = inputs
        num_points = ops.shape(local_features)[1]

        # Broadcast global features to each point
        # This provides context about the overall point cloud structure to each point
        global_features_tiled = ops.tile(
            ops.expand_dims(global_features, axis=1),  # (B, 1, F_global)
            [1, num_points, 1]  # → (B, N, F_global)
        )

        # Combine local and global context for each point
        # Local features capture neighborhood geometry, global features provide overall structure
        combined_features = ops.concatenate(
            [local_features, global_features_tiled],
            axis=-1
        )  # → (B, N, F_local + F_global)

        # Pass through deep MLP to compute correspondence logits
        # Network learns to map features to GMM component affinities
        logits = self.mlp(combined_features)  # → (B, N, K)

        # Apply softmax to get probability distribution over components per point
        # This ensures each row (point's assignments) sums to 1
        gamma = keras.activations.softmax(logits, axis=-1)  # → (B, N, K)

        return gamma

    def compute_output_shape(
        self, input_shape: Tuple[Tuple, Tuple]
    ) -> Tuple:
        """Compute output shape.

        :param input_shape: Tuple of (local_shape, global_shape) where:
            - local_shape: (batch_size, num_points, F_local)
            - global_shape: (batch_size, F_global)
        :type input_shape: Tuple[Tuple, Tuple]
        :return: Output shape (batch_size, num_points, num_gaussians).
        :rtype: Tuple
        """
        local_shape, global_shape = input_shape
        return (local_shape[0], local_shape[1], self.num_gaussians)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'num_gaussians': self.num_gaussians})
        return config

# ---------------------------------------------------------------------
