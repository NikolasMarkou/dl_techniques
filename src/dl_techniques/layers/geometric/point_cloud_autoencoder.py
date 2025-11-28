import keras
from keras import ops
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PointCloudAutoencoder(keras.layers.Layer):
    """
    Modified DGCNN-based autoencoder for point cloud feature extraction.

    This layer implements the Feature Extraction Module from the paper. It uses
    shared EdgeConv-style blocks to extract local and global features from two
    point clouds and then reconstructs them using a decoder.

    **Intent**: To learn distinctive and pose-attentive features from point clouds
    in a semi-supervised manner, forming the foundation for the registration task.

    **Architecture**:
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │  Input: (Source PC, Target PC)                               │
    │         Shape: (B, N, 3) each                                │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Shared Encoder (DGCNN-style, processes each PC separately)  │
    │                                                              │
    │  EdgeConv Block 1:                                           │
    │  ├─ KNN Graph Construction (k neighbors)                     │
    │  ├─ MLP [64, 64]                                             │
    │  └─ Max Pooling → f1  (B, N, 64)                             │
    │                                                              │
    │  EdgeConv Block 2:                                           │
    │  ├─ KNN Graph from f1                                        │
    │  ├─ MLP [64, 64]                                             │
    │  └─ Max Pooling → f2  (B, N, 64)                             │
    │                                                              │
    │  EdgeConv Block 3:                                           │
    │  ├─ KNN Graph from f2                                        │
    │  ├─ MLP [64]                                                 │
    │  └─ Max Pooling → f3  (B, N, 64)                             │
    │                                                              │
    │  Feature Aggregation:                                        │
    │  ├─ Concatenate [f1, f2, f3] → (B, N, 192)                   │
    │  └─ MLP [1024] → local_features  (B, N, 1024)                │
    │                                                              │
    │  Global Feature Extraction:                                  │
    │  ├─ Max Pooling(local_features) → (B, 1024)                  │
    │  ├─ Avg Pooling(local_features) → (B, 1024)                  │
    │  └─ Concatenate → global_features  (B, 2048)                 │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Shared Decoder (processes global features separately)       │
    │                                                              │
    │  ├─ Dense [2048] + LeakyReLU                                 │
    │  ├─ Dense [1024] + LeakyReLU                                 │
    │  ├─ Dense [N×3] (linear)                                     │
    │  └─ Reshape → reconstruction  (B, N, 3)                      │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Output Tuple                                                │
    │  ├─ Reconstructions: (x_rec, y_rec)  [(B,N,3), (B,N,3)]      │
    │  ├─ Local Features: (local_x, local_y)  [(B,N,1024), ...]    │
    │  └─ Global Features: (global_x, global_y)  [(B,2048), ...]   │
    └──────────────────────────────────────────────────────────────┘

    Legend: B=batch_size, N=num_points
    Note: Encoder and Decoder are shared (same weights) for both point clouds
    ```

    Args:
        k_neighbors (int): Number of neighbors for the k-NN graph in EdgeConv blocks.
        **kwargs: Additional arguments for Layer base class.

    Call arguments:
        inputs: A tuple of two Tensors (source_pc, target_pc).
            - source_pc: Source point cloud. Shape: (batch, num_points_x, 3).
            - target_pc: Target point cloud. Shape: (batch, num_points_y, 3).

    Output:
        A tuple containing:
        - reconstructions: Tuple of (X_rec, Y_rec).
        - local_features: Tuple of (local_X, local_Y).
        - global_features: Tuple of (global_X, global_Y).

    Examples:
        >>> autoencoder = PointCloudAutoencoder(k_neighbors=20)
        >>> source = keras.random.normal((8, 1024, 3))
        >>> target = keras.random.normal((8, 1024, 3))
        >>> (x_rec, y_rec), (local_x, local_y), (global_x, global_y) = autoencoder((source, target))
        >>> print(x_rec.shape, local_x.shape, global_x.shape)
        (8, 1024, 3) (8, 1024, 1024) (8, 2048)
    """

    def __init__(self, k_neighbors: int = 20, **kwargs):
        """Initialize PointCloudAutoencoder.

        Args:
            k_neighbors: Number of neighbors for the k-NN graph in EdgeConv blocks.
                Must be positive. Default is 20.
            **kwargs: Additional arguments for Layer base class.

        Raises:
            ValueError: If k_neighbors is not positive.
        """
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

        Args:
            units: List of hidden unit sizes for each Dense layer.
            name: Name for the Sequential model.

        Returns:
            Sequential model with Dense layers and ReLU activations.
        """
        return keras.Sequential([
            keras.layers.Dense(u, activation='relu') for u in units
        ], name=name)

    def build(self, input_shape: Tuple[Tuple, Tuple]):
        """Build the layer by creating shape-dependent sub-layers.

        The decoder's final layer depends on the number of points, so it must
        be created here rather than in __init__.

        Args:
            input_shape: Tuple of (source_shape, target_shape) where each is
                (batch_size, num_points, 3).

        Raises:
            ValueError: If num_points dimension is not defined.
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
        """Shared encoder logic using DGCNN-style EdgeConv blocks.

        EdgeConv captures local geometric structures by constructing a k-NN graph
        and learning edge features between each point and its neighbors. Multiple
        EdgeConv blocks capture features at different scales.

        Args:
            pc: Input point cloud of shape (batch_size, num_points, 3).

        Returns:
            Tuple of (local_features, global_features) where:
                - local_features: Per-point features (batch_size, num_points, 1024)
                - global_features: Point cloud descriptor (batch_size, 2048)
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
        """Shared decoder logic to reconstruct point cloud from global features.

        The decoder is a simple MLP that expands the global feature vector
        back to the original point cloud dimensionality. This reconstruction
        loss encourages the encoder to preserve geometric information.

        Args:
            global_feature: Global descriptor of shape (batch_size, 2048).
            num_points: Number of points to reconstruct.

        Returns:
            Reconstructed point cloud of shape (batch_size, num_points, 3).
        """
        # Progressive expansion from global feature to full point cloud
        x = self.decoder_mlp1(global_feature)  # → (B, 2048)
        x = self.decoder_mlp2(x)  # → (B, 1024)
        x = self.decoder_mlp3(x)  # → (B, num_points × 3)

        # Reshape flat vector to point cloud format
        return ops.reshape(x, (-1, num_points, 3))  # → (B, num_points, 3)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> Tuple[Tuple, Tuple, Tuple]:
        """Forward pass through the autoencoder for both point clouds.

        Processes both source and target point clouds through shared encoder-decoder,
        extracting features at multiple levels and reconstructing the inputs.

        Args:
            inputs: Tuple of (source_pc, target_pc) point clouds.

        Returns:
            Tuple of three tuples:
                1. Reconstructions: (x_rec, y_rec)
                2. Local features: (local_x, local_y) - per-point features
                3. Global features: (global_x, global_y) - point cloud descriptors
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

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'k_neighbors': self.k_neighbors})
        return config


@keras.saving.register_keras_serializable()
class CorrespondenceNetwork(keras.layers.Layer):
    """
    Augmented regression network to estimate point-to-GMM correspondences.

    This layer takes local and global features from a point cloud and outputs
    a probability distribution over the latent GMM components for each point.

    **Intent**: To replace the iterative E-step of a traditional GMM with a
    fast, learned correspondence estimation.

    **Architecture**:
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │  Input: (Local Features, Global Features)                    │
    │         (B, N, F_local), (B, F_global)                       │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Feature Combination                                         │
    │  ├─ Tile global_features to (B, N, F_global)                 │
    │  └─ Concatenate with local_features                          │
    │      → combined  (B, N, F_local + F_global)                  │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  4-Layer MLP (Correspondence Estimation Network)             │
    │                                                              │
    │  ├─ Dense(1024) + ReLU                                       │
    │  ├─ Dense(256) + ReLU                                        │
    │  ├─ Dense(128) + ReLU                                        │
    │  └─ Dense(K) [no activation]                                 │
    │      → logits  (B, N, K)                                     │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Softmax (per point, over K components)                      │
    │  → gamma  (B, N, K)                                          │
    │                                                              │
    │  gamma[b,i,k] = probability that point i belongs to          │
    │                 Gaussian component k in batch b              │
    └──────────────────────────────────────────────────────────────┘

    Legend: B=batch_size, N=num_points, K=num_gaussians
           F_local=local_feature_dim, F_global=global_feature_dim
    ```

    Args:
        num_gaussians (int): The number of latent GMM components (J).
        **kwargs: Additional arguments for Layer base class.

    Call arguments:
        inputs: A tuple of (local_features, global_features).
            - local_features: Shape (batch, num_points, feat_dim).
            - global_features: Shape (batch, global_feat_dim).

    Output:
        Correspondence matrix gamma. Shape: (batch, num_points, num_gaussians).

    Examples:
        >>> corr_net = CorrespondenceNetwork(num_gaussians=32)
        >>> local_feat = keras.random.normal((8, 1024, 1024))
        >>> global_feat = keras.random.normal((8, 2048))
        >>> gamma = corr_net((local_feat, global_feat))
        >>> print(gamma.shape)
        (8, 1024, 32)
        >>> # Verify each row sums to 1 (probability distribution)
        >>> print(keras.ops.sum(gamma, axis=-1)[0, 0])  # Should be ~1.0
    """

    def __init__(self, num_gaussians: int, **kwargs):
        """Initialize CorrespondenceNetwork.

        Args:
            num_gaussians: The number of latent GMM components (K). Must be positive.
                Determines the expressiveness of the correspondence space.
            **kwargs: Additional arguments for Layer base class.

        Raises:
            ValueError: If num_gaussians is not positive.
        """
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

        The MLP input size depends on the concatenation of local and global features,
        so it must be built here after shapes are known.

        Args:
            input_shape: Tuple of (local_shape, global_shape) where:
                - local_shape: (batch_size, num_points, F_local)
                - global_shape: (batch_size, F_global)
        """
        local_shape, global_shape = input_shape

        # Input to MLP is concatenated features: [local_features; tiled_global_features]
        combined_dim = local_shape[-1] + global_shape[-1]

        # Build the correspondence estimation network with combined feature dimension
        self.mlp.build((None, combined_dim))

        super().build(input_shape)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        """Compute soft point-to-GMM component assignments.

        For each point, computes a probability distribution over K Gaussian components,
        effectively replacing the E-step of traditional GMM with a learned function.

        Args:
            inputs: Tuple of (local_features, global_features) where:
                - local_features: Per-point features (B, N, F_local)
                - global_features: Point cloud descriptor (B, F_global)

        Returns:
            Soft assignment matrix gamma of shape (B, N, K) where:
                gamma[b,i,k] represents the probability that point i in batch b
                belongs to Gaussian component k. Each row sums to 1.
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

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'num_gaussians': self.num_gaussians})
        return config

# ---------------------------------------------------------------------
