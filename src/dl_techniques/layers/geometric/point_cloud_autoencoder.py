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
    Input (X, Y)
        |
    Shared Encoder (DGCNN-style)
        |
    (Local Features, Global Features) for X and Y
        |
    Shared Decoder (MLP)
        |
    Output (Reconstructed X, Reconstructed Y)
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
    """

    def __init__(self, k_neighbors: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors

        # CREATE sub-layers in __init__
        # Encoder MLPs for EdgeConv blocks
        self.mlp1 = self._make_mlp([64, 64], "encoder_mlp1")
        self.mlp2 = self._make_mlp([64, 64], "encoder_mlp2")
        self.mlp3 = self._make_mlp([64], "encoder_mlp3")

        # Final shared MLP before pooling
        self.mlp4 = self._make_mlp([1024], "encoder_mlp4")

        # Decoder MLPs
        self.decoder_mlp1 = keras.layers.Dense(2048, activation='leaky_relu', name="decoder_mlp1")
        self.decoder_mlp2 = keras.layers.Dense(1024, activation='leaky_relu', name="decoder_mlp2")
        # The final layer will be created in build() as its size depends on input num_points

    def _make_mlp(self, units: List[int], name: str) -> keras.Sequential:
        return keras.Sequential([
            keras.layers.Dense(u, activation='relu') for u in units
        ], name=name)

    def build(self, input_shape: Tuple[Tuple, Tuple]):
        source_shape, target_shape = input_shape
        # Assuming both clouds are padded/sampled to the same number of points
        num_points = source_shape[1]

        if num_points is None:
            raise ValueError("The number of points (dimension 1) must be defined.")

        # Create the final decoder layer which depends on input shape
        self.decoder_mlp3 = keras.layers.Dense(num_points * 3, name="decoder_mlp3")

        # Explicitly build sub-layers for robust serialization
        # Encoder MLPs
        self.mlp1.build((None, None, None, 6))  # Shape of edge features
        self.mlp2.build((None, None, None, 128))
        self.mlp3.build((None, None, None, 128))
        self.mlp4.build((None, None, 192))  # Shape after feature concatenation

        # Decoder MLPs
        self.decoder_mlp1.build((None, 2048))  # Shape of global feature
        self.decoder_mlp2.build((None, 2048))
        self.decoder_mlp3.build((None, 1024))

        super().build(input_shape)

    def _encode(self, pc: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Shared encoder logic."""
        # EdgeConv 1
        graph1 = keras.ops.get_graph_feature(pc, k=self.k_neighbors)
        f1 = self.mlp1(graph1)
        f1 = ops.max(f1, axis=2)

        # EdgeConv 2
        graph2 = keras.ops.get_graph_feature(f1, k=self.k_neighbors)
        f2 = self.mlp2(graph2)
        f2 = ops.max(f2, axis=2)

        # EdgeConv 3
        graph3 = keras.ops.get_graph_feature(f2, k=self.k_neighbors)
        f3 = self.mlp3(graph3)
        f3 = ops.max(f3, axis=2)

        # Concatenate features and pass through final MLP
        combined_features = ops.concatenate([f1, f2, f3], axis=-1)
        local_features = self.mlp4(combined_features)

        # Global features via max and average pooling
        global_max = ops.max(local_features, axis=1)
        global_avg = ops.mean(local_features, axis=1)
        global_features = ops.concatenate([global_max, global_avg], axis=-1)

        return local_features, global_features

    def _decode(self, global_feature: keras.KerasTensor, num_points: int) -> keras.KerasTensor:
        """Shared decoder logic."""
        x = self.decoder_mlp1(global_feature)
        x = self.decoder_mlp2(x)
        x = self.decoder_mlp3(x)
        return ops.reshape(x, (-1, num_points, 3))

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> Tuple[Tuple, Tuple, Tuple]:
        source_pc, target_pc = inputs
        num_points_source = ops.shape(source_pc)[1]
        num_points_target = ops.shape(target_pc)[1]

        # Process source point cloud
        local_x, global_x = self._encode(source_pc)
        x_rec = self._decode(global_x, num_points_source)

        # Process target point cloud
        local_y, global_y = self._encode(target_pc)
        y_rec = self._decode(global_y, num_points_target)

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
    Input (Local Features, Global Features)
        |
    Combine Features (Tile global and concatenate with local)
        |
    4-Layer MLP (1024 -> 256 -> 128 -> J)
        |
    Softmax
        |
    Output (Correspondence Matrix gamma)
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
    """

    def __init__(self, num_gaussians: int, **kwargs):
        super().__init__(**kwargs)
        self.num_gaussians = num_gaussians

        # CREATE sub-layers in __init__
        self.mlp = keras.Sequential([
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_gaussians)  # No activation, softmax is applied in call
        ], name="correspondence_mlp")

    def build(self, input_shape: Tuple[Tuple, Tuple]):
        local_shape, global_shape = input_shape
        # Input to MLP is concatenated features
        combined_dim = local_shape[-1] + global_shape[-1]
        self.mlp.build((None, combined_dim))
        super().build(input_shape)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> keras.KerasTensor:
        local_features, global_features = inputs
        num_points = ops.shape(local_features)[1]

        # Tile global features to match each point
        global_features_tiled = ops.tile(
            ops.expand_dims(global_features, axis=1),
            [1, num_points, 1]
        )

        # Combine features
        combined_features = ops.concatenate([local_features, global_features_tiled], axis=-1)

        # Pass through MLP and apply softmax
        logits = self.mlp(combined_features)
        gamma = keras.activations.softmax(logits, axis=-1)

        return gamma

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'num_gaussians': self.num_gaussians})
        return config

# ---------------------------------------------------------------------
