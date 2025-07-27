"""
Encoder-only Mask Transformer (EoMT) Model Implementation

Based on: "Your ViT is Secretly an Image Segmentation Model" by Kerssies et al.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional,  Any, Dict, Tuple


from dl_techniques.utils.logger import logger
from dl_techniques.layers.eomt import MaskModule, EoMTLayer
from dl_techniques.layers.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.positional_embedding import PositionalEmbedding
from dl_techniques.layers.vision_transformer import VisionTransformerLayer as VisionTransformer
from dl_techniques.layers.upsample import Upsample


@keras.saving.register_keras_serializable()
class EoMT(keras.Model):
    """Encoder-only Mask Transformer for Image Segmentation.

    This model repurposes a Vision Transformer for image segmentation by:
    1. Using a pre-trained ViT backbone
    2. Adding learnable queries after the first L1 blocks
    3. Processing queries and patches together in the final L2 blocks
    4. Using a simple mask module for predictions

    Args:
        num_classes: Integer, number of classes to predict.
        num_queries: Integer, number of learnable object queries.
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads.
        num_layers: Integer, total number of transformer layers.
        l1_layers: Integer, number of layers to process only patches.
        patch_size: Integer, size of image patches.
        mlp_ratio: Float, ratio of MLP hidden dim to embedding dim.
        dropout: Float, dropout rate.
        attention_dropout: Float, attention dropout rate.
        use_masked_attention: Boolean, whether to use masked attention.
        mask_hidden_dim: Integer, hidden dimension for mask MLP.
        activation: String, activation function.
        use_layer_norm: Boolean, whether to use layer normalization.
        pretrained_backbone: Optional pretrained ViT backbone.
        **kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
            self,
            num_classes: int,
            num_queries: int = 100,
            embed_dim: int = 1024,
            num_heads: int = 16,
            num_layers: int = 24,
            l1_layers: int = 20,
            patch_size: int = 16,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            use_masked_attention: bool = True,
            mask_hidden_dim: int = 256,
            activation: str = "gelu",
            use_layer_norm: bool = True,
            pretrained_backbone: Optional[keras.Model] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.l1_layers = l1_layers
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_masked_attention = use_masked_attention
        self.mask_hidden_dim = mask_hidden_dim
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.pretrained_backbone = pretrained_backbone

        # Validation
        if l1_layers >= num_layers:
            raise ValueError(f"l1_layers ({l1_layers}) must be less than num_layers ({num_layers})")

        self.l2_layers = num_layers - l1_layers

        # Initialize components
        self.patch_embedding = None
        self.positional_embedding = None
        self.query_tokens = None
        self.l1_blocks = None
        self.l2_blocks = None
        self.mask_module = None
        self.upsampler = None

        # For mask annealing
        self.current_epoch = 0
        self.total_epochs = 12  # Default, can be updated

        logger.info(f"Initializing EoMT with {num_layers} layers ({l1_layers} L1 + {self.l2_layers} L2)")

    def build(self, input_shape):
        """Build the model."""
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size, height, width, channels = input_shape

        # Calculate number of patches
        num_patches = (height // self.patch_size) * (width // self.patch_size)

        logger.info(f"Building EoMT for input shape {input_shape}")
        logger.info(f"Number of patches: {num_patches}, Patch size: {self.patch_size}")

        # Use pretrained backbone if provided
        if self.pretrained_backbone is not None:
            logger.info("Using pretrained backbone")
            self._build_from_pretrained(input_shape)
        else:
            logger.info("Building from scratch")
            self._build_from_scratch(input_shape, num_patches)

        # Learnable query tokens
        self.query_tokens = self.add_weight(
            shape=(1, self.num_queries, self.embed_dim),
            initializer="truncated_normal",
            trainable=True,
            name="query_tokens"
        )

        # L2 blocks (process patches + queries together)
        self.l2_blocks = []
        for i in range(self.l2_layers):
            # Calculate mask probability for annealing
            # Earlier blocks should have higher probability initially
            base_prob = 1.0 - (i / max(self.l2_layers - 1, 1)) * 0.1

            block = EoMTLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
                use_masked_attention=self.use_masked_attention,
                mask_probability=base_prob,
                name=f"l2_block_{i}"
            )
            self.l2_blocks.append(block)

        # Mask prediction module
        self.mask_module = MaskModule(
            num_classes=self.num_classes,
            hidden_dim=self.mask_hidden_dim,
            mask_dim=self.embed_dim,
            name="mask_module"
        )

        # Upsampler for high-resolution masks
        self.upsampler = Upsample(
            scale_factor=4,  # Upsample to 4x resolution
            name="upsampler"
        )

        super().build(input_shape)

    def _build_from_pretrained(self, input_shape):
        """Build using a pretrained backbone."""
        # Extract layers from pretrained model
        # This is a simplified approach - in practice, you'd need to carefully
        # extract the patch embedding, positional embedding, and transformer blocks

        # For now, we'll build from scratch but this is where you'd integrate
        # pretrained weights like DINOv2
        logger.warning("Pretrained backbone integration not fully implemented")
        batch_size, height, width, channels = input_shape
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        self._build_from_scratch(input_shape, num_patches)

    def _build_from_scratch(self, input_shape, num_patches):
        """Build model from scratch."""
        # Patch embedding
        self.patch_embedding = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embedding"
        )

        # Positional embedding
        self.positional_embedding = PositionalEmbedding(
            sequence_length=num_patches,
            embed_dim=self.embed_dim,
            name="positional_embedding"
        )

        # L1 blocks (process only patches)
        self.l1_blocks = []
        for i in range(self.l1_layers):
            block = VisionTransformer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout,
                attention_dropout_rate=self.attention_dropout,
                activation=self.activation,
                name=f"l1_block_{i}"
            )
            self.l1_blocks.append(block)

    def update_mask_annealing(self, epoch: int, total_epochs: int):
        """Update mask annealing probabilities based on training progress.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of training epochs
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        # Update mask probabilities in L2 blocks
        for i, block in enumerate(self.l2_blocks):
            # Calculate progress for this block
            # Earlier blocks should start annealing earlier
            block_start_epoch = (i / self.l2_layers) * total_epochs * 0.3
            block_end_epoch = total_epochs * 0.8

            if epoch < block_start_epoch:
                # Full masking
                prob = 1.0
            elif epoch >= block_end_epoch:
                # No masking
                prob = 0.0
            else:
                # Linear annealing
                progress = (epoch - block_start_epoch) / (block_end_epoch - block_start_epoch)
                prob = 1.0 - progress

            block.mask_probability = prob

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        Args:
            inputs: Input images [batch, height, width, channels]
            training: Boolean indicating training mode

        Returns:
            Dictionary containing:
            - class_logits: [batch, num_queries, num_classes]
            - mask_logits: [batch, num_queries, H, W]
            - intermediate_masks: List of intermediate mask predictions (if training)
        """
        batch_size = ops.shape(inputs)[0]

        # Patch embedding
        x = self.patch_embedding(inputs)  # [batch, num_patches, embed_dim]

        # Add positional embedding
        x = self.positional_embedding(x)

        # Process through L1 blocks (patches only)
        for block in self.l1_blocks:
            x = block(x, training=training)

        # Add query tokens
        query_tokens = ops.tile(self.query_tokens, [batch_size, 1, 1])
        x = ops.concatenate([x, query_tokens], axis=1)  # [batch, num_patches + num_queries, embed_dim]

        # Process through L2 blocks (patches + queries)
        intermediate_masks = []
        for i, block in enumerate(self.l2_blocks):
            # Get current mask prediction for masked attention
            if self.use_masked_attention and training:
                current_queries = x[:, -self.num_queries:, :]  # Extract query tokens

                # Create dummy pixel features for mask prediction
                patch_tokens = x[:, :-self.num_queries, :]
                height = width = int(ops.sqrt(ops.cast(ops.shape(patch_tokens)[1], ops.float32)))
                pixel_features = ops.reshape(
                    patch_tokens, [batch_size, height, width, self.embed_dim]
                )

                # Predict intermediate mask
                _, intermediate_mask = self.mask_module(current_queries, pixel_features, training=training)
                intermediate_masks.append(intermediate_mask)

                # Apply block with mask
                x = block(x, mask=intermediate_mask, training=training)
            else:
                x = block(x, training=training)

        # Extract final patch and query tokens
        patch_tokens = x[:, :-self.num_queries, :]  # [batch, num_patches, embed_dim]
        query_tokens = x[:, -self.num_queries:, :]  # [batch, num_queries, embed_dim]

        # Reshape patch tokens to spatial format
        num_patches = ops.shape(patch_tokens)[1]
        height = width = int(ops.sqrt(ops.cast(num_patches, ops.float32)))
        pixel_features = ops.reshape(
            patch_tokens, [batch_size, height, width, self.embed_dim]
        )

        # Predict final masks and classes
        class_logits, mask_logits = self.mask_module(query_tokens, pixel_features, training=training)

        # Upsample mask logits for higher resolution
        mask_logits = self.upsampler(mask_logits)

        result = {
            "class_logits": class_logits,
            "mask_logits": mask_logits
        }

        if training and intermediate_masks:
            result["intermediate_masks"] = intermediate_masks

        return result

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "l1_layers": self.l1_layers,
            "patch_size": self.patch_size,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_masked_attention": self.use_masked_attention,
            "mask_hidden_dim": self.mask_hidden_dim,
            "activation": self.activation,
            "use_layer_norm": self.use_layer_norm,
        })
        return config


@keras.saving.register_keras_serializable()
class MaskAnnealingCallback(keras.callbacks.Callback):
    """Callback for mask annealing during training.

    This callback gradually reduces the mask probability in EoMT layers
    during training, allowing the model to learn without masked attention
    for efficient inference.

    Args:
        annealing_schedule: String, annealing schedule type ("linear" or "polynomial").
        annealing_factor: Float, factor for polynomial annealing.
        start_epoch: Integer, epoch to start annealing.
        end_epoch: Integer, epoch to end annealing.
    """

    def __init__(
            self,
            annealing_schedule: str = "polynomial",
            annealing_factor: float = 0.9,
            start_epoch: int = 0,
            end_epoch: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.annealing_schedule = annealing_schedule
        self.annealing_factor = annealing_factor
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        if self.end_epoch is None:
            self.end_epoch = self.params.get('epochs', 12)

        logger.info(f"Starting mask annealing from epoch {self.start_epoch} to {self.end_epoch}")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        if hasattr(self.model, 'update_mask_annealing'):
            self.model.update_mask_annealing(epoch, self.end_epoch)

        # Log current mask probabilities
        if hasattr(self.model, 'l2_blocks'):
            probs = [block.mask_probability for block in self.model.l2_blocks]
            logger.info(f"Epoch {epoch}: Mask probabilities = {probs}")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        # Log mask annealing progress
        if hasattr(self.model, 'l2_blocks') and self.model.l2_blocks:
            avg_prob = np.mean([block.mask_probability for block in self.model.l2_blocks])
            logs = logs or {}
            logs['avg_mask_probability'] = avg_prob


def create_eomt_model(
        num_classes: int,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        num_queries: int = 100,
        model_size: str = "large",
        use_masked_attention: bool = True,
        pretrained_backbone: Optional[keras.Model] = None
) -> EoMT:
    """Create an EoMT model with predefined configurations.

    Args:
        num_classes: Number of classes to predict
        input_shape: Input image shape (height, width, channels)
        num_queries: Number of learnable object queries
        model_size: Model size ("small", "base", "large", "giant")
        use_masked_attention: Whether to use masked attention
        pretrained_backbone: Optional pretrained backbone

    Returns:
        EoMT model instance
    """
    # Model configurations
    configs = {
        "small": {
            "embed_dim": 384,
            "num_heads": 6,
            "num_layers": 12,
            "l1_layers": 9,
        },
        "base": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "l1_layers": 9,
        },
        "large": {
            "embed_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "l1_layers": 20,
        },
        "giant": {
            "embed_dim": 1536,
            "num_heads": 24,
            "num_layers": 40,
            "l1_layers": 35,
        }
    }

    if model_size not in configs:
        raise ValueError(f"Invalid model_size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]

    # Adjust num_queries based on task
    if num_classes <= 20:  # Semantic segmentation
        num_queries = 100
    else:  # Panoptic/instance segmentation
        num_queries = 200

    logger.info(f"Creating EoMT-{model_size} model with {num_queries} queries for {num_classes} classes")

    model = EoMT(
        num_classes=num_classes,
        num_queries=num_queries,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        l1_layers=config["l1_layers"],
        use_masked_attention=use_masked_attention,
        pretrained_backbone=pretrained_backbone,
        name=f"eomt_{model_size}"
    )

    # Build model with input shape
    model.build((None,) + input_shape)

    return model


def create_mask_annealing_callback(
        total_epochs: int = 12,
        annealing_schedule: str = "polynomial",
        annealing_factor: float = 0.9
) -> MaskAnnealingCallback:
    """Create a mask annealing callback.

    Args:
        total_epochs: Total number of training epochs
        annealing_schedule: Annealing schedule type
        annealing_factor: Factor for polynomial annealing

    Returns:
        MaskAnnealingCallback instance
    """
    return MaskAnnealingCallback(
        annealing_schedule=annealing_schedule,
        annealing_factor=annealing_factor,
        end_epoch=total_epochs
    )