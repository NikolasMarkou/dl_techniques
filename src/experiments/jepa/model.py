"""
Matryoshka-JEPA: Joint Embedding Predictive Architecture with Nested Representations.

This module combines JEPA's embedding prediction objective with Matryoshka
representation learning, enabling variable-dimension embeddings at inference
while maintaining semantic structure at all scales.

Architecture:
    - Context Encoder: Processes visible patches, learns Matryoshka embeddings
    - Target Encoder: EMA copy, provides stable targets
    - Predictor: Predicts embeddings at multiple truncated dimensions
    - Loss: Multi-scale JEPA loss + per-scale VICReg regularization
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Literal

import keras
from keras import ops

# Project imports - these are assumed to exist per user instructions
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer


# =============================================================================
# Utility Functions
# =============================================================================


def off_diagonal(x: keras.KerasTensor) -> keras.KerasTensor:
    """
    Extract off-diagonal elements from a square matrix.

    :param x: Square matrix of shape [D, D]
    :type x: keras.KerasTensor
    :return: Flattened off-diagonal elements
    :rtype: keras.KerasTensor
    """
    n = ops.shape(x)[0]
    mask = 1.0 - ops.eye(n, dtype=x.dtype)
    return x * mask


def gather_along_batch(
    tensor: keras.KerasTensor,
    indices: keras.KerasTensor
) -> keras.KerasTensor:
    """
    Gather elements along axis 1 for each batch element.

    :param tensor: Input tensor [B, N, D]
    :type tensor: keras.KerasTensor
    :param indices: Indices to gather [B, K]
    :type indices: keras.KerasTensor
    :return: Gathered tensor [B, K, D]
    :rtype: keras.KerasTensor
    """
    # Get dimensions
    batch_size = ops.shape(tensor)[0]
    num_positions = ops.shape(tensor)[1]
    embed_dim = ops.shape(tensor)[2]
    num_indices = ops.shape(indices)[1]

    # Flatten tensor: [B, N, D] -> [B*N, D]
    tensor_flat = ops.reshape(tensor, [-1, embed_dim])

    # Create batch offsets: [0, N, 2N, ..., (B-1)*N]
    batch_offsets = ops.arange(batch_size) * num_positions
    batch_offsets = ops.expand_dims(batch_offsets, 1)  # [B, 1]

    # Add offsets to indices: [B, K] + [B, 1] -> [B, K]
    flat_indices = indices + batch_offsets

    # Flatten indices: [B, K] -> [B*K]
    flat_indices = ops.reshape(flat_indices, [-1])

    # Gather: [B*N, D] -> [B*K, D]
    gathered = ops.take(tensor_flat, flat_indices, axis=0)

    # Reshape: [B*K, D] -> [B, K, D]
    return ops.reshape(gathered, [batch_size, num_indices, embed_dim])


def cosine_ema_schedule(
    step: int,
    total_steps: int,
    base_tau: float = 0.996,
    final_tau: float = 1.0
) -> float:
    """
    Compute EMA decay using cosine schedule.

    :param step: Current training step
    :type step: int
    :param total_steps: Total number of training steps
    :type total_steps: int
    :param base_tau: Initial EMA decay value
    :type base_tau: float
    :param final_tau: Final EMA decay value
    :type final_tau: float
    :return: Current EMA decay rate
    :rtype: float
    """
    progress = step / max(total_steps, 1)
    tau = final_tau - (final_tau - base_tau) * (1 + math.cos(math.pi * progress)) / 2
    return tau


# =============================================================================
# Loss Functions
# =============================================================================


def variance_loss(
    z: keras.KerasTensor,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> keras.KerasTensor:
    """
    Compute variance loss to prevent collapse.

    Encourages each embedding dimension to have standard deviation >= gamma.

    :param z: Embeddings of shape [B, D]
    :type z: keras.KerasTensor
    :param gamma: Target standard deviation
    :type gamma: float
    :param eps: Numerical stability epsilon
    :type eps: float
    :return: Variance loss scalar
    :rtype: keras.KerasTensor
    """
    std_z = ops.sqrt(ops.var(z, axis=0) + eps)
    var_loss = ops.mean(ops.relu(gamma - std_z))
    return var_loss


def covariance_loss(z: keras.KerasTensor) -> keras.KerasTensor:
    """
    Compute covariance loss to decorrelate embedding dimensions.

    Penalizes off-diagonal elements of the covariance matrix.

    :param z: Embeddings of shape [B, D]
    :type z: keras.KerasTensor
    :return: Covariance loss scalar
    :rtype: keras.KerasTensor
    """
    batch_size = ops.cast(ops.shape(z)[0], z.dtype)
    d = ops.cast(ops.shape(z)[1], z.dtype)

    z_centered = z - ops.mean(z, axis=0, keepdims=True)
    cov = ops.matmul(ops.transpose(z_centered), z_centered) / (batch_size - 1.0)

    off_diag = off_diagonal(cov)
    cov_loss = ops.sum(ops.square(off_diag)) / d
    return cov_loss


def vicreg_loss(
    z1: keras.KerasTensor,
    z2: keras.KerasTensor,
    lambda_var: float = 25.0,
    lambda_cov: float = 25.0,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """
    Compute VICReg regularization loss components.

    :param z1: First embedding batch [B, D]
    :type z1: keras.KerasTensor
    :param z2: Second embedding batch [B, D]
    :type z2: keras.KerasTensor
    :param lambda_var: Weight for variance loss
    :type lambda_var: float
    :param lambda_cov: Weight for covariance loss
    :type lambda_cov: float
    :param gamma: Target standard deviation
    :type gamma: float
    :param eps: Numerical stability epsilon
    :type eps: float
    :return: Tuple of (total_vicreg_loss, variance_loss, covariance_loss)
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
    """
    var_loss = variance_loss(z1, gamma, eps) + variance_loss(z2, gamma, eps)
    cov_loss = covariance_loss(z1) + covariance_loss(z2)

    total = lambda_var * var_loss + lambda_cov * cov_loss
    return total, var_loss, cov_loss


def matryoshka_jepa_loss(
    predicted_emb: keras.KerasTensor,
    target_emb: keras.KerasTensor,
    matryoshka_dims: List[int],
    dim_weights: Optional[List[float]] = None,
    use_vicreg: bool = True,
    lambda_var: float = 10.0,
    lambda_cov: float = 10.0,
) -> Dict[str, keras.KerasTensor]:
    """
    Compute multi-scale Matryoshka-JEPA loss.

    Applies JEPA prediction loss at multiple embedding dimension truncations,
    with optional VICReg regularization at each scale.

    :param predicted_emb: Predicted embeddings [B, N, D]
    :type predicted_emb: keras.KerasTensor
    :param target_emb: Target embeddings (stop_gradient applied) [B, N, D]
    :type target_emb: keras.KerasTensor
    :param matryoshka_dims: List of dimensions to apply loss at
    :type matryoshka_dims: List[int]
    :param dim_weights: Optional weights per dimension (default: equal)
    :type dim_weights: Optional[List[float]]
    :param use_vicreg: Whether to apply VICReg regularization
    :type use_vicreg: bool
    :param lambda_var: VICReg variance weight
    :type lambda_var: float
    :param lambda_cov: VICReg covariance weight
    :type lambda_cov: float
    :return: Dictionary of loss components
    :rtype: Dict[str, keras.KerasTensor]
    """
    if dim_weights is None:
        dim_weights = [1.0] * len(matryoshka_dims)

    total_weight = sum(dim_weights)
    dim_weights = [w / total_weight for w in dim_weights]

    pred_losses = []
    var_losses = []
    cov_losses = []

    for dim, weight in zip(matryoshka_dims, dim_weights):
        pred_d = predicted_emb[..., :dim]
        tgt_d = target_emb[..., :dim]

        # Flatten spatial dims for loss computation: [B, N, D] -> [B*N, D]
        pred_flat = ops.reshape(pred_d, (-1, dim))
        tgt_flat = ops.reshape(tgt_d, (-1, dim))

        # L2 prediction loss
        pred_loss = ops.mean(ops.square(pred_flat - tgt_flat))
        pred_losses.append(weight * pred_loss)

        if use_vicreg:
            _, v_loss, c_loss = vicreg_loss(
                pred_flat, tgt_flat,
                lambda_var=1.0,
                lambda_cov=1.0,
                gamma=1.0
            )
            # Scale VICReg by dimension ratio (smaller dims need more regularization)
            scale = dim / matryoshka_dims[-1]
            var_losses.append(weight * v_loss / scale)
            cov_losses.append(weight * c_loss / scale)

    total_pred = ops.sum(ops.stack(pred_losses), axis=0)

    losses = {
        "prediction_loss": total_pred,
        "total_loss": total_pred,
    }

    if use_vicreg:
        total_var = lambda_var * ops.sum(ops.stack(var_losses), axis=0)
        total_cov = lambda_cov * ops.sum(ops.stack(cov_losses), axis=0)
        losses["variance_loss"] = total_var
        losses["covariance_loss"] = total_cov
        losses["total_loss"] = total_pred + total_var + total_cov

    return losses


# =============================================================================
# Transformer Components
# =============================================================================


@keras.saving.register_keras_serializable(package="MatryoshkaJEPA")
class TransformerBlock(keras.layers.Layer):
    """
    Pre-norm transformer block.

    :param embed_dim: Embedding dimension
    :type embed_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio
    :type mlp_ratio: float
    :param dropout_rate: Dropout rate
    :type dropout_rate: float
    :param norm_type: Normalization type
    :type norm_type: str
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        norm_type: str = "rms_norm",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        # Attention
        self.norm1 = create_normalization_layer(norm_type, name="attn_norm")
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name="attention"
        )

        # FFN
        self.norm2 = create_normalization_layer(norm_type, name="ffn_norm")
        self.ffn = create_ffn_layer(
            "swiglu",
            output_dim=embed_dim,
            ffn_expansion_factor=int(mlp_ratio),
            dropout_rate=dropout_rate,
            name="ffn"
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape, input_shape)

        attn_output_shape = input_shape
        self.norm2.build(attn_output_shape)
        self.ffn.build(attn_output_shape)

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with pre-norm residual connections.

        :param x: Input tensor [B, N, D]
        :type x: keras.KerasTensor
        :param training: Training mode flag
        :type training: Optional[bool]
        :return: Output tensor [B, N, D]
        :rtype: keras.KerasTensor
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, training=training)
        x = x + attn_out

        # Pre-norm FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, training=training)
        x = x + ffn_out

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "norm_type": self.norm_type,
        })
        return config


# =============================================================================
# JEPA Encoder
# =============================================================================


@keras.saving.register_keras_serializable(package="MatryoshkaJEPA")
class MatryoshkaJEPAEncoder(keras.layers.Layer):
    """
    JEPA Encoder that learns Matryoshka-compatible embeddings.

    The encoder produces embeddings where any prefix z[:d'] is a valid
    d'-dimensional embedding, enabling variable-dimension inference.

    :param embed_dim: Full embedding dimension
    :type embed_dim: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio
    :type mlp_ratio: float
    :param patch_size: Patch size for image tokenization
    :type patch_size: int
    :param dropout_rate: Dropout rate
    :type dropout_rate: float
    :param norm_type: Normalization type
    :type norm_type: str
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        dropout_rate: float = 0.0,
        norm_type: str = "rms_norm",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        # Patch embedding
        self.patch_embed = create_embedding_layer(
            "patch_2d",
            patch_size=patch_size,
            embed_dim=embed_dim,
            name="patch_embed"
        )

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                name=f"block_{i}"
            )
            for i in range(num_layers)
        ]

        # Final normalization
        self.norm = create_normalization_layer(norm_type, name="final_norm")

    def build(self, input_shape):
        # Build patch embedding
        self.patch_embed.build(input_shape)

        # Infer sequence shape after patching
        # Input: [B, H, W, C] -> [B, num_patches, embed_dim]
        h = input_shape[1] // self.patch_size if input_shape[1] else None
        w = input_shape[2] // self.patch_size if input_shape[2] else None
        num_patches = (h * w) if (h and w) else None

        seq_shape = (input_shape[0], num_patches, self.embed_dim)

        # Learnable position embeddings
        max_patches = 196  # 14x14 for 224 images with patch_size=16
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, max_patches, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Build transformer blocks
        for block in self.blocks:
            block.build(seq_shape)

        self.norm.build(seq_shape)

        super().build(input_shape)

    def call(
        self,
        images: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode images to embeddings.

        :param images: Input images [B, H, W, C]
        :type images: keras.KerasTensor
        :param mask: Optional boolean mask for visible patches [B, N]
        :type mask: Optional[keras.KerasTensor]
        :param training: Training mode flag
        :type training: Optional[bool]
        :return: Patch embeddings [B, N, D]
        :rtype: keras.KerasTensor
        """
        # Patchify
        x = self.patch_embed(images)
        num_patches = ops.shape(x)[1]

        # Add position embeddings
        x = x + self.pos_embed[:, :num_patches, :]

        # Apply mask if provided by zeroing out masked positions
        # Note: For JEPA, we typically pass pre-gathered patches instead
        # This is a soft mask that zeros masked positions
        if mask is not None:
            # mask: [B, N] boolean, True = visible
            mask_expanded = ops.cast(
                ops.expand_dims(mask, -1),
                x.dtype
            )
            x = x * mask_expanded

        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final norm
        x = self.norm(x)

        return x

    def compute_output_shape(self, input_shape):
        h = input_shape[1] // self.patch_size if input_shape[1] else None
        w = input_shape[2] // self.patch_size if input_shape[2] else None
        num_patches = (h * w) if (h and w) else None
        return (input_shape[0], num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "patch_size": self.patch_size,
            "dropout_rate": self.dropout_rate,
            "norm_type": self.norm_type,
        })
        return config


# =============================================================================
# JEPA Predictor
# =============================================================================


@keras.saving.register_keras_serializable(package="MatryoshkaJEPA")
class MatryoshkaJEPAPredictor(keras.layers.Layer):
    """
    JEPA Predictor that outputs Matryoshka-compatible predictions.

    Predicts target embeddings from context embeddings and target positions.
    Designed to be narrower than the encoder to prevent shortcuts.

    :param encoder_dim: Encoder output dimension
    :type encoder_dim: int
    :param predictor_dim: Predictor hidden dimension (should be < encoder_dim)
    :type predictor_dim: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param max_targets: Maximum number of target positions
    :type max_targets: int
    :param dropout_rate: Dropout rate
    :type dropout_rate: float
    :param norm_type: Normalization type
    :type norm_type: str
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        max_targets: int = 196,
        dropout_rate: float = 0.0,
        norm_type: str = "rms_norm",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_targets = max_targets
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        # Project to predictor dimension
        self.input_proj = keras.layers.Dense(
            predictor_dim,
            name="input_proj"
        )

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                embed_dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                name=f"pred_block_{i}"
            )
            for i in range(num_layers)
        ]

        # Final norm
        self.norm = create_normalization_layer(norm_type, name="pred_norm")

        # Project back to encoder dimension
        self.output_proj = keras.layers.Dense(
            encoder_dim,
            name="output_proj"
        )

    def build(self, input_shape):
        # input_shape is context_emb shape: [B, N_ctx, encoder_dim]
        ctx_shape = input_shape

        # Mask token for target positions
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, 1, self.predictor_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Position embeddings for all possible positions
        self.pred_pos_embed = self.add_weight(
            name="pred_pos_embed",
            shape=(1, self.max_targets, self.predictor_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Build input projection
        self.input_proj.build(ctx_shape)

        # After projection shape
        proj_shape = (ctx_shape[0], ctx_shape[1], self.predictor_dim)

        # Build blocks with combined context + target shape
        # Conservative estimate - actual varies based on num targets
        combined_shape = (ctx_shape[0], self.max_targets, self.predictor_dim)
        for block in self.blocks:
            block.build(combined_shape)

        self.norm.build(combined_shape)
        self.output_proj.build((None, None, self.predictor_dim))

        super().build(input_shape)

    def call(
        self,
        context_emb: keras.KerasTensor,
        target_positions: keras.KerasTensor,
        context_positions: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict target embeddings.

        :param context_emb: Context embeddings [B, N_ctx, D]
        :type context_emb: keras.KerasTensor
        :param target_positions: Target position indices [B, N_tgt]
        :type target_positions: keras.KerasTensor
        :param context_positions: Optional context position indices [B, N_ctx]
        :type context_positions: Optional[keras.KerasTensor]
        :param training: Training mode flag
        :type training: Optional[bool]
        :return: Predicted embeddings [B, N_tgt, D]
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(context_emb)[0]
        num_ctx = ops.shape(context_emb)[1]
        num_tgt = ops.shape(target_positions)[1]

        # Project context to predictor dimension
        context = self.input_proj(context_emb)

        # Add position embeddings to context if provided
        if context_positions is not None:
            # Broadcast pos_embed to batch: [1, max, D] -> [B, max, D]
            pos_embed_batch = ops.broadcast_to(
                self.pred_pos_embed,
                [batch_size, self.max_targets, self.predictor_dim]
            )
            ctx_pos_emb = gather_along_batch(pos_embed_batch, context_positions)
            context = context + ctx_pos_emb

        # Create mask tokens for targets
        mask_tokens = ops.broadcast_to(
            self.mask_token,
            [batch_size, num_tgt, self.predictor_dim]
        )

        # Gather position embeddings for target positions
        pos_embed_batch = ops.broadcast_to(
            self.pred_pos_embed,
            [batch_size, self.max_targets, self.predictor_dim]
        )
        tgt_pos_emb = gather_along_batch(pos_embed_batch, target_positions)
        mask_tokens = mask_tokens + tgt_pos_emb

        # Concatenate context and mask tokens
        x = ops.concatenate([context, mask_tokens], axis=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)

        # Extract target predictions (last num_tgt tokens)
        predictions = x[:, num_ctx:, :]

        # Project to encoder dimension
        predictions = self.output_proj(predictions)

        return predictions

    def compute_output_shape(self, input_shape):
        # Returns shape based on context; actual depends on target_positions
        return (input_shape[0], None, self.encoder_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "encoder_dim": self.encoder_dim,
            "predictor_dim": self.predictor_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_targets": self.max_targets,
            "dropout_rate": self.dropout_rate,
            "norm_type": self.norm_type,
        })
        return config


# =============================================================================
# Full Matryoshka-JEPA Model
# =============================================================================


@keras.saving.register_keras_serializable(package="MatryoshkaJEPA")
class MatryoshkaJEPA(keras.Model):
    """
    Matryoshka-JEPA: JEPA with nested representation learning.

    Combines JEPA's embedding prediction objective with Matryoshka
    representation learning for variable-dimension embeddings.

    :param embed_dim: Embedding dimension
    :type embed_dim: int
    :param encoder_layers: Number of encoder transformer layers
    :type encoder_layers: int
    :param predictor_layers: Number of predictor transformer layers
    :type predictor_layers: int
    :param encoder_heads: Number of encoder attention heads
    :type encoder_heads: int
    :param predictor_heads: Number of predictor attention heads
    :type predictor_heads: int
    :param predictor_dim: Predictor hidden dimension
    :type predictor_dim: int
    :param patch_size: Image patch size
    :type patch_size: int
    :param matryoshka_dims: Dimensions for multi-scale loss
    :type matryoshka_dims: List[int]
    :param ema_decay_base: Base EMA decay for target encoder
    :type ema_decay_base: float
    :param ema_decay_final: Final EMA decay for target encoder
    :type ema_decay_final: float
    :param use_vicreg: Whether to use VICReg regularization
    :type use_vicreg: bool
    :param lambda_var: VICReg variance weight
    :type lambda_var: float
    :param lambda_cov: VICReg covariance weight
    :type lambda_cov: float
    """

    def __init__(
        self,
        embed_dim: int = 384,
        encoder_layers: int = 6,
        predictor_layers: int = 4,
        encoder_heads: int = 6,
        predictor_heads: int = 4,
        predictor_dim: int = 192,
        patch_size: int = 16,
        matryoshka_dims: Optional[List[int]] = None,
        ema_decay_base: float = 0.996,
        ema_decay_final: float = 1.0,
        use_vicreg: bool = True,
        lambda_var: float = 10.0,
        lambda_cov: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.encoder_layers = encoder_layers
        self.predictor_layers = predictor_layers
        self.encoder_heads = encoder_heads
        self.predictor_heads = predictor_heads
        self.predictor_dim = predictor_dim
        self.patch_size = patch_size
        self.ema_decay_base = ema_decay_base
        self.ema_decay_final = ema_decay_final
        self.use_vicreg = use_vicreg
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        # Default Matryoshka dimensions
        if matryoshka_dims is None:
            matryoshka_dims = [64, 128, 256, embed_dim] if embed_dim >= 256 else [embed_dim]
        self.matryoshka_dims = [d for d in matryoshka_dims if d <= embed_dim]

        # Context encoder (trainable)
        self.context_encoder = MatryoshkaJEPAEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            patch_size=patch_size,
            name="context_encoder"
        )

        # Target encoder (EMA, non-trainable)
        self.target_encoder = MatryoshkaJEPAEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            patch_size=patch_size,
            name="target_encoder"
        )
        self.target_encoder.trainable = False

        # Predictor
        self.predictor = MatryoshkaJEPAPredictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            name="predictor"
        )

        # Track training state
        self._ema_initialized = False

    def build(self, input_shape):
        # Build all components
        self.context_encoder.build(input_shape)
        self.target_encoder.build(input_shape)

        # Predictor build with encoder output shape
        enc_output_shape = self.context_encoder.compute_output_shape(input_shape)
        self.predictor.build(enc_output_shape)

        super().build(input_shape)

    def initialize_target_encoder(self):
        """Initialize target encoder weights from context encoder."""
        if not self._ema_initialized:
            for ctx_var, tgt_var in zip(
                self.context_encoder.trainable_variables,
                self.target_encoder.variables
            ):
                tgt_var.assign(ctx_var)
            self._ema_initialized = True

    def update_target_encoder(self, tau: float):
        """
        Update target encoder via EMA.

        :param tau: EMA decay rate
        :type tau: float
        """
        for ctx_var, tgt_var in zip(
            self.context_encoder.trainable_variables,
            self.target_encoder.variables
        ):
            tgt_var.assign(tau * tgt_var + (1.0 - tau) * ctx_var)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for inference (feature extraction).

        :param inputs: Input images [B, H, W, C]
        :type inputs: keras.KerasTensor
        :param training: Training mode flag
        :type training: Optional[bool]
        :return: Encoded features [B, N, D]
        :rtype: keras.KerasTensor
        """
        return self.context_encoder(inputs, training=training)

    def compute_jepa_loss(
        self,
        images: keras.KerasTensor,
        context_positions: keras.KerasTensor,
        target_positions: keras.KerasTensor,
        training: bool = True
    ) -> Dict[str, keras.KerasTensor]:
        """
        Compute JEPA training loss with Matryoshka multi-scale objective.

        :param images: Input images [B, H, W, C]
        :type images: keras.KerasTensor
        :param context_positions: Position indices for context [B, N_ctx]
        :type context_positions: keras.KerasTensor
        :param target_positions: Position indices for targets [B, N_tgt]
        :type target_positions: keras.KerasTensor
        :param training: Training mode flag
        :type training: bool
        :return: Dictionary of loss components
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Encode full image with context encoder (trainable)
        full_emb_ctx = self.context_encoder(images, training=training)

        # Encode full image with target encoder (EMA, no gradient)
        full_emb_tgt = self.target_encoder(images, training=False)
        full_emb_tgt = ops.stop_gradient(full_emb_tgt)

        # Gather context embeddings using batch gather
        # context_positions: [B, N_ctx], full_emb_ctx: [B, N, D]
        context_emb = gather_along_batch(full_emb_ctx, context_positions)

        # Gather target embeddings
        target_emb = gather_along_batch(full_emb_tgt, target_positions)

        # Predict target embeddings from context
        pred_emb = self.predictor(
            context_emb,
            target_positions,
            context_positions=context_positions,
            training=training
        )

        # Compute multi-scale loss
        losses = matryoshka_jepa_loss(
            predicted_emb=pred_emb,
            target_emb=target_emb,
            matryoshka_dims=self.matryoshka_dims,
            use_vicreg=self.use_vicreg,
            lambda_var=self.lambda_var,
            lambda_cov=self.lambda_cov,
        )

        return losses

    def get_features(
        self,
        images: keras.KerasTensor,
        dim: Optional[int] = None
    ) -> keras.KerasTensor:
        """
        Extract features at specified dimension (Matryoshka inference).

        :param images: Input images [B, H, W, C]
        :type images: keras.KerasTensor
        :param dim: Embedding dimension to use (default: full)
        :type dim: Optional[int]
        :return: Features [B, N, dim]
        :rtype: keras.KerasTensor
        """
        features = self.context_encoder(images, training=False)

        if dim is not None and dim < self.embed_dim:
            features = features[..., :dim]

        return features

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "encoder_layers": self.encoder_layers,
            "predictor_layers": self.predictor_layers,
            "encoder_heads": self.encoder_heads,
            "predictor_heads": self.predictor_heads,
            "predictor_dim": self.predictor_dim,
            "patch_size": self.patch_size,
            "matryoshka_dims": self.matryoshka_dims,
            "ema_decay_base": self.ema_decay_base,
            "ema_decay_final": self.ema_decay_final,
            "use_vicreg": self.use_vicreg,
            "lambda_var": self.lambda_var,
            "lambda_cov": self.lambda_cov,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================================================================
# Mask Generation
# =============================================================================


def generate_block_mask(
    grid_size: int,
    num_targets: int = 4,
    target_scale: Tuple[float, float] = (0.15, 0.2),
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    seed: Optional[int] = None
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """
    Generate I-JEPA style multi-block masks.

    :param grid_size: Spatial grid size (e.g., 14 for 14x14 patches)
    :type grid_size: int
    :param num_targets: Number of target blocks
    :type num_targets: int
    :param target_scale: Min/max fraction of image per target
    :type target_scale: Tuple[float, float]
    :param aspect_ratio: Min/max aspect ratio for target blocks
    :type aspect_ratio: Tuple[float, float]
    :param seed: Random seed
    :type seed: Optional[int]
    :return: (context_mask, target_mask, context_positions, target_positions)
    :rtype: Tuple of tensors
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    num_patches = grid_size * grid_size
    all_target_indices = set()

    for _ in range(num_targets):
        # Sample block size
        scale = np.random.uniform(*target_scale)
        aspect = np.random.uniform(*aspect_ratio)

        block_area = int(num_patches * scale)
        block_h = int(np.sqrt(block_area / aspect))
        block_w = int(block_area / max(block_h, 1))

        block_h = min(max(block_h, 1), grid_size)
        block_w = min(max(block_w, 1), grid_size)

        # Random position
        start_h = np.random.randint(0, grid_size - block_h + 1)
        start_w = np.random.randint(0, grid_size - block_w + 1)

        # Collect target indices
        for h in range(start_h, start_h + block_h):
            for w in range(start_w, start_w + block_w):
                all_target_indices.add(h * grid_size + w)

    # Create masks
    target_mask = np.zeros(num_patches, dtype=bool)
    for idx in all_target_indices:
        target_mask[idx] = True

    context_mask = ~target_mask

    # Get position indices
    context_positions = np.where(context_mask)[0]
    target_positions = np.where(target_mask)[0]

    return (
        ops.convert_to_tensor(context_mask),
        ops.convert_to_tensor(target_mask),
        ops.convert_to_tensor(context_positions, dtype="int32"),
        ops.convert_to_tensor(target_positions, dtype="int32"),
    )


def generate_batch_masks(
    batch_size: int,
    grid_size: int,
    num_targets: int = 4,
    target_scale: Tuple[float, float] = (0.15, 0.2),
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """
    Generate masks for a batch of samples.

    For simplicity, uses the same mask for all samples in batch.
    In practice, you may want per-sample masks.

    :param batch_size: Batch size
    :type batch_size: int
    :param grid_size: Spatial grid size
    :type grid_size: int
    :param num_targets: Number of target blocks
    :type num_targets: int
    :param target_scale: Min/max target scale
    :type target_scale: Tuple[float, float]
    :return: Batch masks
    :rtype: Tuple of tensors
    """
    ctx_mask, tgt_mask, ctx_pos, tgt_pos = generate_block_mask(
        grid_size=grid_size,
        num_targets=num_targets,
        target_scale=target_scale,
    )

    # Tile for batch
    ctx_mask = ops.tile(ops.expand_dims(ctx_mask, 0), [batch_size, 1])
    tgt_mask = ops.tile(ops.expand_dims(tgt_mask, 0), [batch_size, 1])
    ctx_pos = ops.tile(ops.expand_dims(ctx_pos, 0), [batch_size, 1])
    tgt_pos = ops.tile(ops.expand_dims(tgt_pos, 0), [batch_size, 1])

    return ctx_mask, tgt_mask, ctx_pos, tgt_pos