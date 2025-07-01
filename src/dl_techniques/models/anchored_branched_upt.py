import keras
from keras import ops

from dl_techniques.layers.geometric import (
    ContinuousSincosEmbed,
    ContinuousRoPE,
    PerceiverBlock,
    SupernodePooling,
    TransformerBlock
)

@keras.saving.register_keras_serializable()
class AnchoredBranchedUPT(keras.Model):
    """Anchored Branched Universal Physics Transformer for CFD.

    This model processes multi-modal CFD data:
    - Geometry: 3D point cloud processed through supernode pooling
    - Surface: Positions with pressure and wall shear stress
    - Volume: Positions with pressure, velocity, and vorticity

    Args:
        ndim: Number of coordinate dimensions (typically 3 for 3D).
        input_dim: Input coordinate dimension (typically 3).
        output_dim_surface: Surface output dimension (pressure + wall shear stress).
        output_dim_volume: Volume output dimension (pressure + velocity + vorticity).
        dim: Model hidden dimension.
        geometry_depth: Number of transformer blocks for geometry processing.
        num_heads: Number of attention heads.
        blocks: String specifying shared attention pattern (e.g., "pscscs").
        num_volume_blocks: Number of volume-specific transformer blocks.
        num_surface_blocks: Number of surface-specific transformer blocks.
        radius: Radius for supernode pooling.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            ndim: int = 3,
            input_dim: int = 3,
            output_dim_surface: int = 4,  # pressure (1) + wall shear stress (3)
            output_dim_volume: int = 7,  # pressure (1) + velocity (3) + vorticity (3)
            dim: int = 192,
            geometry_depth: int = 1,
            num_heads: int = 3,
            blocks: str = "pscscs",
            num_volume_blocks: int = 6,
            num_surface_blocks: int = 6,
            radius: float = 0.25,
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.ndim = ndim
        self.input_dim = input_dim
        self.output_dim_surface = output_dim_surface
        self.output_dim_volume = output_dim_volume
        self.dim = dim
        self.geometry_depth = geometry_depth
        self.num_heads = num_heads
        self.blocks_config = blocks
        self.num_volume_blocks = num_volume_blocks
        self.num_surface_blocks = num_surface_blocks
        self.radius = radius
        self.dropout_rate = dropout

        # Build layers
        self._build_layers()

    def _build_layers(self):
        """Build all model layers."""
        # RoPE for positional encoding
        self.rope = ContinuousRoPE(dim=self.dim // self.num_heads, ndim=self.input_dim, name="rope")

        # Geometry encoder
        self.geometry_encoder = SupernodePooling(
            hidden_dim=self.dim,
            ndim=self.ndim,
            radius=self.radius,
            mode="relpos",
            name="geometry_encoder"
        )

        # Geometry transformer blocks
        self.geometry_blocks = [
            TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                name=f"geometry_block_{i}"
            )
            for i in range(self.geometry_depth)
        ]

        # Position embedding layers
        self.pos_embed = ContinuousSincosEmbed(dim=self.dim, ndim=self.ndim, name="pos_embed")

        # Surface and volume bias networks
        self.surface_bias = keras.Sequential([
            keras.layers.Dense(self.dim, activation="gelu", name="surface_bias_1"),
            keras.layers.Dense(self.dim, name="surface_bias_2")
        ], name="surface_bias")

        self.volume_bias = keras.Sequential([
            keras.layers.Dense(self.dim, activation="gelu", name="volume_bias_1"),
            keras.layers.Dense(self.dim, name="volume_bias_2")
        ], name="volume_bias")

        # Shared weight blocks
        self.shared_blocks = []
        for i, block_type in enumerate(self.blocks_config):
            if block_type == "s":
                # Shared split attention (within modality)
                self.shared_blocks.append(
                    TransformerBlock(
                        dim=self.dim,
                        num_heads=self.num_heads,
                        attention_class="standard",
                        dropout=self.dropout_rate,
                        name=f"shared_block_{i}_split"
                    )
                )
            elif block_type == "c":
                # Shared cross attention (between modalities)
                self.shared_blocks.append(
                    TransformerBlock(
                        dim=self.dim,
                        num_heads=self.num_heads,
                        attention_class="shared_cross",
                        dropout=self.dropout_rate,
                        name=f"shared_block_{i}_cross"
                    )
                )
            elif block_type == "p":
                # Perceiver block (cross attention to geometry)
                self.shared_blocks.append(
                    PerceiverBlock(
                        dim=self.dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout_rate,
                        name=f"shared_block_{i}_perceiver"
                    )
                )

        # Surface-specific blocks
        self.surface_blocks = [
            TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                attention_class="anchor",
                dropout=self.dropout_rate,
                name=f"surface_block_{i}"
            )
            for i in range(self.num_surface_blocks)
        ]

        # Volume-specific blocks
        self.volume_blocks = [
            TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                attention_class="anchor",
                dropout=self.dropout_rate,
                name=f"volume_block_{i}"
            )
            for i in range(self.num_volume_blocks)
        ]

        # Output decoders
        self.surface_decoder = keras.layers.Dense(
            self.output_dim_surface,
            name="surface_decoder"
        )
        self.volume_decoder = keras.layers.Dense(
            self.output_dim_volume,
            name="volume_decoder"
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights following the PyTorch implementation."""

        def init_weights(layer):
            if isinstance(layer, keras.layers.Dense):
                # Truncated normal initialization
                keras.utils.set_random_seed(42)  # For reproducibility
                layer.kernel.assign(
                    keras.random.truncated_normal(layer.kernel.shape, stddev=0.02)
                )
                if layer.bias is not None:
                    layer.bias.assign(keras.ops.zeros_like(layer.bias))

        # Apply to all layers
        for layer in self.layers:
            init_weights(layer)

    def call(self, inputs, training=None):
        """Forward pass of the AB-UPT model.

        Args:
            inputs: Dictionary containing:
                - geometry_position: (num_geometry_points, 3)
                - geometry_supernode_idx: (num_supernodes,)
                - geometry_batch_idx: Optional batch indices
                - surface_anchor_position: (batch_size, num_surface_anchors, 3)
                - volume_anchor_position: (batch_size, num_volume_anchors, 3)
                - surface_query_position: Optional (batch_size, num_surface_queries, 3)
                - volume_query_position: Optional (batch_size, num_volume_queries, 3)

        Returns:
            Dictionary with predictions for surface and volume quantities.
        """
        # Extract inputs
        geometry_position = inputs["geometry_position"]
        geometry_supernode_idx = inputs["geometry_supernode_idx"]
        geometry_batch_idx = inputs.get("geometry_batch_idx")

        surface_anchor_position = inputs["surface_anchor_position"]
        volume_anchor_position = inputs["volume_anchor_position"]

        surface_query_position = inputs.get("surface_query_position")
        volume_query_position = inputs.get("volume_query_position")

        # Determine configuration
        has_queries = (surface_query_position is not None and volume_query_position is not None)

        # Prepare position data
        if has_queries:
            surface_position_all = ops.concatenate([surface_anchor_position, surface_query_position], axis=1)
            volume_position_all = ops.concatenate([volume_anchor_position, volume_query_position], axis=1)
            num_surface_anchors = ops.shape(surface_anchor_position)[1]
            num_volume_anchors = ops.shape(volume_anchor_position)[1]
        else:
            surface_position_all = surface_anchor_position
            volume_position_all = volume_anchor_position
            num_surface_anchors = ops.shape(surface_anchor_position)[1]
            num_volume_anchors = ops.shape(volume_anchor_position)[1]

        # Generate RoPE frequencies (simplified - using first batch element)
        geometry_rope_input = ops.take(geometry_position, geometry_supernode_idx, axis=0)
        # Add batch dimension for RoPE
        geometry_rope_input = ops.expand_dims(geometry_rope_input, 0)

        # Process geometry
        geometry_features = self.geometry_encoder({
            "positions": geometry_position,
            "supernode_indices": geometry_supernode_idx
        }, training=training)

        # Apply geometry transformer blocks
        for block in self.geometry_blocks:
            geometry_features = block(geometry_features, training=training)

        # Process surface and volume positions
        surface_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        volume_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))

        # Combine features
        combined_features = ops.concatenate([surface_pos_embed, volume_pos_embed], axis=1)

        # Shared weight blocks
        x = combined_features
        for i, block in enumerate(self.shared_blocks):
            if isinstance(block, PerceiverBlock):
                # Cross attention to geometry
                x = block(x, geometry_features, training=training)
            elif isinstance(block, TransformerBlock) and block.attention_class == "shared_cross":
                # Cross attention between modalities
                surface_len = ops.shape(surface_pos_embed)[1]
                volume_len = ops.shape(volume_pos_embed)[1]
                x = block(x, split_sizes=[surface_len, volume_len], training=training)
            else:
                # Standard attention
                x = block(x, training=training)

        # Split back into surface and volume
        surface_len = ops.shape(surface_pos_embed)[1]
        volume_len = ops.shape(volume_pos_embed)[1]

        x_surface = x[:, :surface_len, :]
        x_volume = x[:, surface_len:, :]

        # Apply modality-specific blocks
        for block in self.surface_blocks:
            if has_queries:
                x_surface = block(x_surface, num_anchor_tokens=num_surface_anchors, training=training)
            else:
                x_surface = block(x_surface, training=training)

        for block in self.volume_blocks:
            if has_queries:
                x_volume = block(x_volume, num_anchor_tokens=num_volume_anchors, training=training)
            else:
                x_volume = block(x_volume, training=training)

        # Generate outputs
        surface_output = self.surface_decoder(x_surface)
        volume_output = self.volume_decoder(x_volume)

        # Prepare output dictionary
        outputs = {}

        if has_queries:
            # Split anchor and query outputs
            surface_anchor_out = surface_output[:, :num_surface_anchors, :]
            surface_query_out = surface_output[:, num_surface_anchors:, :]
            volume_anchor_out = volume_output[:, :num_volume_anchors, :]
            volume_query_out = volume_output[:, num_volume_anchors:, :]

            # Surface outputs (flatten for sparse format)
            outputs["surface_anchor_pressure"] = ops.reshape(
                surface_anchor_out[:, :, :1], (-1, 1)
            )
            outputs["surface_anchor_wallshearstress"] = ops.reshape(
                surface_anchor_out[:, :, 1:], (-1, 3)
            )
            outputs["surface_query_pressure"] = ops.reshape(
                surface_query_out[:, :, :1], (-1, 1)
            )
            outputs["surface_query_wallshearstress"] = ops.reshape(
                surface_query_out[:, :, 1:], (-1, 3)
            )

            # Volume outputs (flatten for sparse format)
            outputs["volume_anchor_totalpcoeff"] = ops.reshape(
                volume_anchor_out[:, :, :1], (-1, 1)
            )
            outputs["volume_anchor_velocity"] = ops.reshape(
                volume_anchor_out[:, :, 1:4], (-1, 3)
            )
            outputs["volume_anchor_vorticity"] = ops.reshape(
                volume_anchor_out[:, :, 4:], (-1, 3)
            )
            outputs["volume_query_totalpcoeff"] = ops.reshape(
                volume_query_out[:, :, :1], (-1, 1)
            )
            outputs["volume_query_velocity"] = ops.reshape(
                volume_query_out[:, :, 1:4], (-1, 3)
            )
            outputs["volume_query_vorticity"] = ops.reshape(
                volume_query_out[:, :, 4:], (-1, 3)
            )
        else:
            # Only anchor outputs
            outputs["surface_anchor_pressure"] = ops.reshape(
                surface_output[:, :, :1], (-1, 1)
            )
            outputs["surface_anchor_wallshearstress"] = ops.reshape(
                surface_output[:, :, 1:], (-1, 3)
            )
            outputs["volume_anchor_totalpcoeff"] = ops.reshape(
                volume_output[:, :, :1], (-1, 1)
            )
            outputs["volume_anchor_velocity"] = ops.reshape(
                volume_output[:, :, 1:4], (-1, 3)
            )
            outputs["volume_anchor_vorticity"] = ops.reshape(
                volume_output[:, :, 4:], (-1, 3)
            )

        return outputs

    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "ndim": self.ndim,
            "input_dim": self.input_dim,
            "output_dim_surface": self.output_dim_surface,
            "output_dim_volume": self.output_dim_volume,
            "dim": self.dim,
            "geometry_depth": self.geometry_depth,
            "num_heads": self.num_heads,
            "blocks": self.blocks_config,
            "num_volume_blocks": self.num_volume_blocks,
            "num_surface_blocks": self.num_surface_blocks,
            "radius": self.radius,
            "dropout": self.dropout_rate,
        })
        return config


def create_abupt_model(
        dim: int = 192,
        num_heads: int = 3,
        geometry_depth: int = 1,
        blocks: str = "pscscs",
        num_surface_blocks: int = 6,
        num_volume_blocks: int = 6,
        radius: float = 0.25,
        dropout: float = 0.1
) -> AnchoredBranchedUPT:
    """Factory function to create AB-UPT model with standard configuration.

    Args:
        dim: Model hidden dimension.
        num_heads: Number of attention heads.
        geometry_depth: Number of geometry processing blocks.
        blocks: Shared attention pattern string.
        num_surface_blocks: Number of surface-specific blocks.
        num_volume_blocks: Number of volume-specific blocks.
        radius: Supernode pooling radius.
        dropout: Dropout rate.

    Returns:
        Configured AB-UPT model.
    """
    model = AnchoredBranchedUPT(
        ndim=3,
        input_dim=3,
        output_dim_surface=4,  # pressure (1) + wall shear stress (3)
        output_dim_volume=7,  # pressure (1) + velocity (3) + vorticity (3)
        dim=dim,
        geometry_depth=geometry_depth,
        num_heads=num_heads,
        blocks=blocks,
        num_volume_blocks=num_volume_blocks,
        num_surface_blocks=num_surface_blocks,
        radius=radius,
        dropout=dropout
    )

    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_abupt_model()

    # Create test data
    batch_size = 2
    num_geometry_points = 1000
    num_supernodes = 100
    num_surface_anchors = 200
    num_volume_anchors = 300
    num_surface_queries = 50
    num_volume_queries = 75

    test_inputs = {
        "geometry_position": keras.random.uniform((num_geometry_points, 3)) * 10,
        "geometry_supernode_idx": keras.random.uniform((num_supernodes,), 0, num_geometry_points, dtype="int32"),
        "geometry_batch_idx": None,
        "surface_anchor_position": keras.random.uniform((batch_size, num_surface_anchors, 3)) * 10,
        "volume_anchor_position": keras.random.uniform((batch_size, num_volume_anchors, 3)) * 10,
        "surface_query_position": keras.random.uniform((batch_size, num_surface_queries, 3)) * 10,
        "volume_query_position": keras.random.uniform((batch_size, num_volume_queries, 3)) * 10,
    }

    # Test forward pass
    try:
        outputs = model(test_inputs)
        print("✅ Model forward pass successful!")
        print("\nOutput shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")

        # Test without queries
        test_inputs_no_queries = {k: v for k, v in test_inputs.items()
                                  if not k.endswith("query_position")}
        outputs_no_queries = model(test_inputs_no_queries)
        print(f"\n✅ Model without queries successful!")
        print(f"   Surface pressure: {outputs_no_queries['surface_anchor_pressure'].shape}")
        print(f"   Volume velocity: {outputs_no_queries['volume_anchor_velocity'].shape}")

    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback

        traceback.print_exc()

