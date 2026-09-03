"""
DINOv1 vision transformer trunk, its projection head, and a factory that
builds a correctly-initialized teacher/student pair.

DINO learns visual features with no labels by training a student network to
predict what a teacher, a slowly moving copy of the student, says about a
different crop of the same image. Both networks end in a projection to
`dino_out_dim` prototypes, matched by cross-entropy between the two softmax
distributions. The centering and sharpening that keep this from collapsing
to a trivial constant output, and the teacher's EMA weight update, are not
in this file: they live in `dl_techniques.losses.dino_loss.DINOLoss` and
`DINOTrainingModel.update_teacher_ema`. This module supplies only the ViT
trunk, the projection head, and the pair factory.

The trunk is a plain pre-normalized ViT: patch embedding, a prepended [CLS]
token, a learned absolute positional table, `depth` transformer blocks with
linearly increasing stochastic depth, and a final normalization. Attention,
FFN and normalization are each selected by string through the shared
factories. Features are the [CLS] row when `use_cls_token=True`, otherwise
the mean over patch tokens. `include_projection_head` takes precedence over
`include_top`: the DINO head, else a classifier `Dense` when
`num_classes > 0`, else raw features.

`norm_last_layer` is enforced as an invariant, not the reference's weight-norm
reparameterization: a `UnitNorm(axis=0)` kernel constraint plus a one-off
build-time normalization gives the same `||kernel[:, j]|| == 1` invariant
without adding forward-path arithmetic to a 256 x 65536 kernel at paper
scale. `get_last_selfattention` raises `NotImplementedError` rather than
returning zeros, since the `multi_head` attention path exposes no attention
probabilities and a zero-tensor stub previously rendered attention-map
visualizations as an indistinguishable all-black image. The `giant` variant
is not in the DINOv1 paper (which stops at ViT-B/8); it exists so
`DINOv1`, `DINOv2VisionTransformer` and `DINOv3` share one `MODEL_VARIANTS`
key set, carrying the shared ViT-g/14 dimensions with no version-specific
extras.

The three `create_dino_v*` factories share one parameter scheme: `input_shape`
is not among them and raises `TypeError`, since the input shape is always
derived from `image_size` to prevent the two spellings from disagreeing.
`patch_size=None` defers to the variant's own entry; `DINOv1.MODEL_VARIANTS`
defines none, so it resolves to 16 for every variant.

References:
    - Caron et al., 2021. Emerging Properties in Self-Supervised Vision
      Transformers. (https://arxiv.org/abs/2104.14294)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Caron et al., 2020. Unsupervised Learning of Visual Features by Contrasting
      Cluster Assignments. (https://arxiv.org/abs/2006.09882)
    - Grill et al., 2020. Bootstrap Your Own Latent: A New Approach to
      Self-Supervised Learning. (https://arxiv.org/abs/2006.07733)
    - Salimans and Kingma, 2016. Weight Normalization: A Simple Reparameterization
      to Accelerate Training of Deep Neural Networks.
      (https://arxiv.org/abs/1602.07868)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vision.dino.reference_init import DINO_KERNEL_INITIALIZER
from dl_techniques.models.vision.dino.common import (
    reject_input_shape,
    sync_teacher_to_student,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

ModelVariant = Literal["tiny", "small", "base", "large", "giant"]

# DINOv1's patch size when the caller does not specify one. `MODEL_VARIANTS`
# here carries no per-variant `patch_size` (unlike `dino_v3.py`'s, whose `giant`
# entry sets 14), so `create_dino_v1(patch_size=None)` resolves to this for
# EVERY variant. Caron et al. 2021 use ViT-S/16 and ViT-B/16.
_DEFAULT_PATCH_SIZE = 16


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.dino.dino_v1")
class DINOHead(keras.layers.Layer):
    """DINO projection head: MLP, L2 normalize, unit-norm-constrained projection.

    Architecture:

    .. code-block:: text

        Input [B, in_dim]
              │
              ▼
        (nlayers - 1) x [Dense(hidden_dim) -> norm? -> activation -> dropout?]
              │
              ▼
        Dense(bottleneck_dim)
              │
              ▼
        L2 normalize (variable_dtype)
              │
              ▼
        Dense(out_dim, use_bias=False, kernel_constraint=UnitNorm if norm_last_layer)
              │
              ▼
        Output [B, out_dim]

    `norm_last_layer` reproduces the reference DINO implementation's frozen
    weight-norm reparameterization (`w = g * v / ||v||` with `g` pinned to 1)
    as a `keras.constraints.UnitNorm(axis=0)` constraint plus a one-off
    build-time normalization, giving the same invariant
    (`||kernel[:, j]||_2 == 1` for every output unit `j`) through a different
    optimization path: the constraint projects after each optimizer step,
    where the reference reparameterizes before it.

    :param in_dim: Input dimension (backbone output dimension).
    :type in_dim: int
    :param out_dim: Output dimension for contrastive learning.
    :type out_dim: int
    :param use_bn: Whether to use batch normalization in intermediate layers.
    :type use_bn: bool
    :param norm_last_layer: Whether to constrain the final projection's
        weights to unit L2 norm per output unit.
    :type norm_last_layer: bool
    :param nlayers: Number of layers in the projection head, minimum 1.
    :type nlayers: int
    :param hidden_dim: Hidden dimension in intermediate layers.
    :type hidden_dim: int
    :param bottleneck_dim: Dimension before the final projection layer.
    :type bottleneck_dim: int
    :param normalization_type: Normalization type used in intermediate layers.
    :type normalization_type: str
    :param activation: Activation function.
    :type activation: Union[str, callable]
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param kernel_initializer: Weight initialization scheme. Defaults to
        `TruncatedNormal(stddev=0.02)`, DINO's published `trunc_normal_(std=.02)`.
    :type kernel_initializer: Union[str, Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        2D tensor ``(batch_size, in_dim)``.

    Output shape:
        2D tensor ``(batch_size, out_dim)``.

    Example:
        .. code-block:: python

            dino_head = DINOHead(
                in_dim=384, out_dim=65536, use_bn=False,
                norm_last_layer=True, nlayers=3,
                hidden_dim=2048, bottleneck_dim=256
            )
            cls_token = keras.Input(shape=(384,))
            projection = dino_head(cls_token)
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bn: bool = False,
            norm_last_layer: bool = True,
            nlayers: int = 3,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            normalization_type: str = "batch_norm",
            activation: str = "gelu",
            dropout_rate: float = 0.0,
            # DECISION plan-2026-08-23T091307-9a110062/D-504: default to DINO_KERNEL_INITIALIZER, never the bare string "truncated_normal".
            # That string resolves to Keras' TruncatedNormal(stddev=0.05), 2.5x wider than DINO's published std=.02. See decisions.md.
            kernel_initializer: Union[str, Dict[str, Any]] = DINO_KERNEL_INITIALIZER,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if nlayers < 1:
            raise ValueError(f"nlayers must be at least 1, got {nlayers}")
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim and out_dim must be positive, got {in_dim}, {out_dim}")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        # Store configuration
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.normalization_type = normalization_type
        self.activation = deserialize_activation(activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Initialize layer lists
        self.mlp_layers = []
        self.last_layer = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the DINO head layers."""
        # DECISION plan_2026-06-14_8c7365d0/D-006: reset the sublayer accumulators before appending, every build.
        # build() must be idempotent; a second build without this duplicates every sublayer/weight. See decisions.md.
        self.mlp_layers = []
        self.last_layer = None

        if self.nlayers == 1:
            # Single layer: direct projection to bottleneck dimension
            layer = keras.layers.Dense(
                units=self.bottleneck_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name="mlp_single"
            )
            self.mlp_layers.append(layer)

        else:
            # Multi-layer MLP
            # First layer: in_dim -> hidden_dim
            layer = keras.layers.Dense(
                units=self.hidden_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name="mlp_0"
            )
            self.mlp_layers.append(layer)

            # Batch norm after first layer if requested
            if self.use_bn:
                norm_layer = create_normalization_layer(
                    self.normalization_type,
                    name="mlp_norm_0"
                )
                self.mlp_layers.append(norm_layer)

            # Activation after first layer
            if isinstance(self.activation, str):
                activation_layer = keras.layers.Activation(
                    self.activation, name="mlp_activation_0"
                )
            else:
                activation_layer = self.activation
            self.mlp_layers.append(activation_layer)

            # Dropout if specified
            if self.dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=self.dropout_rate, name="mlp_dropout_0"
                )
                self.mlp_layers.append(dropout_layer)

            # Intermediate layers: hidden_dim -> hidden_dim
            for i in range(1, self.nlayers - 1):
                layer = keras.layers.Dense(
                    units=self.hidden_dim,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    name=f"mlp_{i}"
                )
                self.mlp_layers.append(layer)

                if self.use_bn:
                    norm_layer = create_normalization_layer(
                        self.normalization_type,
                        name=f"mlp_norm_{i}"
                    )
                    self.mlp_layers.append(norm_layer)

                if isinstance(self.activation, str):
                    activation_layer = keras.layers.Activation(
                        self.activation, name=f"mlp_activation_{i}"
                    )
                else:
                    activation_layer = self.activation
                self.mlp_layers.append(activation_layer)

                if self.dropout_rate > 0.0:
                    dropout_layer = keras.layers.Dropout(
                        rate=self.dropout_rate, name=f"mlp_dropout_{i}"
                    )
                    self.mlp_layers.append(dropout_layer)

            # Final layer before bottleneck: hidden_dim -> bottleneck_dim
            final_mlp_layer = keras.layers.Dense(
                units=self.bottleneck_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name=f"mlp_{self.nlayers - 1}"
            )
            self.mlp_layers.append(final_mlp_layer)

        # Final projection layer (bottleneck_dim -> out_dim)
        # DECISION plan-2026-08-01T105809-dc0c402e/D-011: enforce `norm_last_layer` with a UnitNorm(axis=0) constraint, not weight-norm reparameterization.
        # A constraint avoids extra forward-path arithmetic on a 256x65536 kernel; keep the build-time normalize below too, since a constraint only applies after an optimizer step. See decisions.md.
        self.last_layer = keras.layers.Dense(
            units=self.out_dim,
            use_bias=False,  # DINO typically doesn't use bias in the last layer
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=(
                keras.constraints.UnitNorm(axis=0) if self.norm_last_layer else None
            ),
            name="last_layer"
        )

        # Explicitly build every sublayer in forward order so their weights
        # materialize on .keras reload (lazy first-call build leaves the
        # sublayers unbuilt -> their weights are silently dropped on load).
        # Guards tolerate a non-layer activation callable.
        current_shape = input_shape
        for layer in self.mlp_layers:
            if hasattr(layer, "build") and not getattr(layer, "built", False):
                layer.build(current_shape)
            if hasattr(layer, "compute_output_shape"):
                current_shape = layer.compute_output_shape(current_shape)
        self.last_layer.build(current_shape)

        # Apply the constraint once at build time so a never-trained head also satisfies the unit-norm invariant (see D-011).
        # On a .keras reload this runs before the saved weights are restored, and they already satisfy it.
        if self.norm_last_layer:
            self.last_layer.kernel.assign(
                self.last_layer.kernel_constraint(self.last_layer.kernel)
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the MLP stack, L2-normalize, then apply the final projection.

        :param inputs: Input tensor, shape ``(batch_size, in_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer runs in training or inference mode.
        :type training: Optional[bool]
        :return: Projected tensor, shape ``(batch_size, out_dim)``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Apply MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # L2 normalize before final projection (as in DINO paper).
        # DECISION plan-2026-08-01T105809-dc0c402e/D-020: normalize in `variable_dtype`, never bare `compute_dtype`.
        # Under mixed_float16, `sum(x**2)` over bottleneck_dim overflows fp16 and `x / inf == 0` silently zeros the head. See decisions.md.
        x = keras.ops.cast(x, self.variable_dtype)
        x = keras.utils.normalize(x, axis=-1, order=2)
        x = keras.ops.cast(x, self.compute_dtype)

        # Final projection
        x = self.last_layer(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Project (batch_size, in_dim) -> (batch_size, out_dim)."""
        return (input_shape[0], self.out_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "use_bn": self.use_bn,
            "norm_last_layer": self.norm_last_layer,
            "nlayers": self.nlayers,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "normalization_type": self.normalization_type,
            "activation": serialize_activation(self.activation),
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer,
        })
        return config


@register_dl_technique("dl_techniques.models.dino.dino_v1")
class DINOv1(keras.Model):
    """DINOv1 Vision Transformer trunk, for feature extraction or SSL pretraining.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C]
              │
              ▼
        ┌──────────────────────┐
        │ PatchEmbedding2D     │
        └──────────┬────────────┘
                   ▼
        ┌──────────────────────┐
        │ ClassTokenPrepend    │  (if use_cls_token)
        └──────────┬────────────┘
                   ▼
        ┌──────────────────────┐
        │ PositionalEmbedding  │
        └──────────┬────────────┘
                   ▼
        depth x TransformerLayer (linear stochastic depth)
                   ▼
        ┌──────────────────────┐
        │ final normalization  │
        └──────────┬────────────┘
                   ▼
        [CLS] row, or mean over patch tokens
                   │
        ┌──────────┼──────────────────┐
        ▼          ▼                  ▼
     DINOHead   Dense(num_classes)   raw features
   (if include_  (if include_top     (otherwise)
   projection_    and num_classes>0)
   head)

    :param embed_dim: Embedding dimension of the model.
    :type embed_dim: int
    :param depth: Number of transformer layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param patch_size: Size of image patches.
    :type patch_size: Union[int, Tuple[int, int]]
    :param image_size: Input image size.
    :type image_size: Union[int, Tuple[int, int]]
    :param in_channels: Number of input channels.
    :type in_channels: int
    :param num_classes: Number of output classes for classification; 0 for
        feature extraction only.
    :type num_classes: int
    :param mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
    :type mlp_ratio: float
    :param qkv_bias: Whether to use bias in the attention QKV/output
        projections. Forwarded to the attention factory as `use_bias`.
    :type qkv_bias: bool
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention dropout rate.
    :type attention_dropout_rate: float
    :param stochastic_depth_rate: Stochastic depth rate.
    :type stochastic_depth_rate: float
    :param normalization_type: Normalization layer type.
    :type normalization_type: str
    :param attention_type: Attention mechanism type.
    :type attention_type: str
    :param ffn_type: Feed-forward network type.
    :type ffn_type: str
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param include_projection_head: Whether to include the DINO projection head.
    :type include_projection_head: bool
    :param dino_out_dim: Output dimension for the DINO head.
    :type dino_out_dim: int
    :param dino_hidden_dim: Hidden dimension for the DINO head.
    :type dino_hidden_dim: int
    :param dino_bottleneck_dim: Bottleneck dimension for the DINO head.
    :type dino_bottleneck_dim: int
    :param dino_nlayers: Number of layers in the DINO head.
    :type dino_nlayers: int
    :param use_cls_token: Whether to use a [CLS] token.
    :type use_cls_token: bool
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_projection_head=True``: 2D tensor ``(batch_size, dino_out_dim)``.
        - ``include_top=True`` and ``num_classes>0``: 2D tensor ``(batch_size, num_classes)``.
        - Otherwise: 2D tensor ``(batch_size, embed_dim)``.

    Example:
        .. code-block:: python

            # Feature extraction model
            model = DINOv1(
                embed_dim=384, depth=12, num_heads=6, patch_size=16,
                num_classes=0, include_top=False, image_size=224
            )

            # Self-supervised learning model with DINO head
            model = DINOv1(
                embed_dim=384, depth=12, num_heads=6, patch_size=16,
                num_classes=0, include_projection_head=True,
                dino_out_dim=65536, image_size=224
            )
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
        },
        "small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        },
        # Present for key-set parity with dino_v2/dino_v3 only — see the module
        # docstring. Not a DINOv1-paper variant.
        "giant": {
            "embed_dim": 1536,
            "depth": 40,
            "num_heads": 24,
            "mlp_ratio": 4.0,
        }
    }

    def __init__(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            patch_size: Union[int, Tuple[int, int]] = 16,
            image_size: Union[int, Tuple[int, int]] = 224,
            in_channels: int = 3,
            num_classes: int = 1000,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            stochastic_depth_rate: float = 0.0,
            normalization_type: str = "layer_norm",
            attention_type: str = "multi_head",
            ffn_type: str = "mlp",
            include_top: bool = True,
            include_projection_head: bool = False,
            dino_out_dim: int = 65536,
            dino_hidden_dim: int = 2048,
            dino_bottleneck_dim: int = 256,
            dino_nlayers: int = 3,
            use_cls_token: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ):
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Store configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        # Normalize to a 2-tuple. Accept list/tuple (e.g. from .keras
        # deserialization, where tuples come back as TrackedList) as well as int.
        self.patch_size = (
            tuple(patch_size) if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )
        self.image_size = (
            tuple(image_size) if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.include_top = include_top
        self.include_projection_head = include_projection_head
        self.dino_out_dim = dino_out_dim
        self.dino_hidden_dim = dino_hidden_dim
        self.dino_bottleneck_dim = dino_bottleneck_dim
        self.dino_nlayers = dino_nlayers
        self.use_cls_token = use_cls_token

        # Validate patch size alignment. Without this the floor-division below
        # silently truncates patches; v2/v3 both raise here (same message).
        if (self.image_size[0] % self.patch_size[0] != 0
                or self.image_size[1] % self.patch_size[1] != 0):
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"patch_size {self.patch_size}"
            )

        # Calculate derived parameters
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # Set input shape
        if input_shape is None:
            input_shape = (*self.image_size, self.in_channels)
        self._input_shape = input_shape

        # Build the model
        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(f"Created DINO Vision Transformer with {depth} layers, "
                    f"{num_heads} heads, {embed_dim} embed_dim")

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete DINO Vision Transformer model."""
        x = inputs

        # Patch embedding
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embed"
        )
        x = self.patch_embed(x)

        # Add CLS token if requested
        if self.use_cls_token:
            # DECISION plan_2026-06-15_39a31d4a/D-001: own the CLS token via ClassTokenPrepend, never an inline `self.add_weight` here.
            # An inline add_weight fires before `super().__init__` in this functional model and crashes. See decisions.md.
            self.cls_token_layer = ClassTokenPrepend(name="cls_token")
            x = self.cls_token_layer(x)

        # Positional embedding
        max_seq_len = self.num_patches + (1 if self.use_cls_token else 0)
        self.pos_embed = PositionalEmbedding(
            max_seq_len=max_seq_len,
            dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="pos_embed"
        )
        x = self.pos_embed(x)

        # Transformer blocks
        self.transformer_blocks = []
        dpr = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        for i in range(self.depth):
            # DECISION plan-2026-08-01T105809-dc0c402e/D-010: forward qkv_bias unconditionally, spelled `use_bias` (the registry's actual key).
            # A gate on `attention_type == "multi_head_attention"` was dead code; that string was never the registry key. See decisions.md.
            block = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                attention_args={"use_bias": self.qkv_bias},
                normalization_type=self.normalization_type,
                normalization_position="pre",  # Pre-normalization as in DINO
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=dpr[i],
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
            x = block(x)

        # Final layer normalization
        self.norm = create_normalization_layer(
            self.normalization_type,
            name="norm"
        )
        x = self.norm(x)

        # Extract features based on configuration
        if self.use_cls_token:
            # Use CLS token representation
            cls_output = x[:, 0]  # Shape: (batch_size, embed_dim)
            features = cls_output
        else:
            # Global average pooling over patch tokens
            features = keras.ops.mean(x, axis=1)  # Shape: (batch_size, embed_dim)

        # Output head
        if self.include_projection_head:
            # DINO projection head for self-supervised learning
            self.head = DINOHead(
                in_dim=self.embed_dim,
                out_dim=self.dino_out_dim,
                hidden_dim=self.dino_hidden_dim,
                bottleneck_dim=self.dino_bottleneck_dim,
                nlayers=self.dino_nlayers,
                use_bn=False,
                norm_last_layer=True,
                dropout_rate=self.dropout_rate,
                name="dino_projection_head"
            )
            outputs = self.head(features)
        elif self.include_top and self.num_classes > 0:
            # Standard classification head
            # DECISION plan-2026-08-23T091307-9a110062/D-504: use DINO_KERNEL_INITIALIZER here too, never the bare string.
            # Same trap as the DINOHead default above: the bare string is Keras' stddev=0.05, not DINO's 0.02. See decisions.md.
            self.head = keras.layers.Dense(
                units=self.num_classes,
                kernel_initializer=keras.initializers.get(DINO_KERNEL_INITIALIZER),
                name="classifier"
            )
            outputs = self.head(features)
        else:
            # Feature extraction only
            self.head = None
            outputs = features

        return outputs

    def get_last_selfattention(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Get attention probabilities from the last transformer layer.

        Not implemented; see :raises:. This method exists so the DINO
        attention-map visualization API fails loudly rather than silently
        returning something useless.

        :param inputs: Input tensor, shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :raises NotImplementedError: Always; see the message for the missing
            capability.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-012: raise, never restore the old zero-tensor fallback.
        # That fallback rendered an attention-map visualization as an indistinguishable all-black map. See decisions.md.
        raise NotImplementedError(
            "get_last_selfattention() is not implemented for DINOv1. The "
            "'multi_head' attention path does not expose its attention "
            "probabilities: TransformerLayer.call, MultiHeadAttention.call and "
            "MultiHeadCrossAttention.call all lack a return_attention_scores "
            "argument and cache no attention tensor. Implementing DINO-style "
            "attention-map visualization requires adding that capability to "
            "dl_techniques.layers.attention.multi_head_attention and "
            "dl_techniques.layers.transformers.transformer first."
        )

    @classmethod
    def from_variant(
            cls,
            variant: ModelVariant,
            num_classes: int = 0,
            patch_size: Union[int, Tuple[int, int]] = 16,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "DINOv1":
        """Create a DINO model from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``, ``"giant"``.
        :type variant: ModelVariant
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param patch_size: Size of image patches.
        :type patch_size: Union[int, Tuple[int, int]]
        :param input_shape: An explicit input shape. Prefer ``image_size``
            (passed through ``**kwargs``); if this is ``None`` the
            constructor derives ``(*image_size, in_channels)``. The two
            spellings can disagree, which is why `create_dino_v1` refuses
            this one outright.
        :type input_shape: Optional[Tuple[int, ...]]
        :param kwargs: Additional arguments passed to the constructor.

        :return: DINOv1 model instance.
        :rtype: DINOv1

        Example:
            .. code-block:: python

                # DINO-Small for feature extraction
                model = DINOv1.from_variant("small", num_classes=0, patch_size=16)

                # DINO-Base with projection head
                model = DINOv1.from_variant(
                    "base", num_classes=0,
                    include_projection_head=True, dino_out_dim=65536
                )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-01T105809-dc0c402e/D-033: merge `**kwargs` over the variant table's copy, never spell out table keys beside `**kwargs`.
        # The latter raised a duplicate-keyword TypeError on any override, e.g. `from_variant("tiny", embed_dim=32)`. See decisions.md.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINO-{variant.upper()} model")

        return cls(
            num_classes=num_classes,
            patch_size=patch_size,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: call `super().get_config()` first, never a literal dict.
        # Without it, `name` and `trainable` reload at their defaults, so a frozen model comes back unfrozen. See decisions.md.
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "normalization_type": self.normalization_type,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "include_top": self.include_top,
            "include_projection_head": self.include_projection_head,
            "dino_out_dim": self.dino_out_dim,
            "dino_hidden_dim": self.dino_hidden_dim,
            "dino_bottleneck_dim": self.dino_bottleneck_dim,
            "dino_nlayers": self.dino_nlayers,
            "use_cls_token": self.use_cls_token,
            "input_shape": self._input_shape,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOv1":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINO Vision Transformer configuration:")
        logger.info(f"  - Embedding dimension: {self.embed_dim}")
        logger.info(f"  - Number of layers: {self.depth}")
        logger.info(f"  - Number of heads: {self.num_heads}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Image size: {self.image_size}")
        logger.info(f"  - Number of patches: {self.num_patches}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Use CLS token: {self.use_cls_token}")
        logger.info(f"  - Include top: {self.include_top}")
        logger.info(f"  - Include DINO head: {self.include_projection_head}")
        if self.include_projection_head:
            logger.info(f"  - DINO output dim: {self.dino_out_dim}")
        if self.num_classes > 0:
            logger.info(f"  - Number of classes: {self.num_classes}")


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def create_dino_v1(
        variant: ModelVariant = "small",
        *,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 0,
        include_top: bool = True,
        include_projection_head: bool = False,
        dino_out_dim: int = 65536,
        **kwargs
) -> DINOv1:
    """Build a DINOv1 model from a named variant.

    Shares its parameter scheme with `create_dino_v2` and `create_dino_v3`:
    `input_shape` is not among them, since the input shape is always derived
    as `(*image_size, in_channels)` and passing `input_shape=` raises
    `TypeError` rather than letting the two spellings disagree. `patch_size`
    follows the same precedence in all three: an explicit value always wins
    over the variant's; `None` defers to the variant's own entry if its
    `MODEL_VARIANTS` defines one, else to this version's default. v1 defines
    no per-variant `patch_size`, so `None` resolves to 16.

    :param variant: Model variant, one of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``, ``"giant"``.
    :type variant: ModelVariant
    :param image_size: Input image size.
    :type image_size: Union[int, Tuple[int, int]]
    :param patch_size: Size of image patches. ``None`` defers to the variant.
    :type patch_size: Optional[Union[int, Tuple[int, int]]]
    :param num_classes: Number of output classes; 0 for feature extraction.
    :type num_classes: int
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param include_projection_head: Whether to include the DINO projection head.
    :type include_projection_head: bool
    :param dino_out_dim: Output dimension for the DINO head.
    :type dino_out_dim: int
    :param kwargs: Additional arguments passed to the model constructor.
    :return: DINOv1 model instance.
    :rtype: DINOv1
    :raises TypeError: If ``input_shape`` is passed; use ``image_size`` instead.

    Example:
        .. code-block:: python

            # DINO-Small for self-supervised learning
            model = create_dino_v1(
                variant="small", include_projection_head=True, dino_out_dim=65536,
            )

            # DINO-Base for fine-tuning
            model = create_dino_v1(variant="base", image_size=224, num_classes=1000)
    """
    reject_input_shape(kwargs, "create_dino_v1")

    if patch_size is None:
        patch_size = _DEFAULT_PATCH_SIZE

    return DINOv1.from_variant(
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        patch_size=patch_size,
        include_top=include_top,
        include_projection_head=include_projection_head,
        dino_out_dim=dino_out_dim,
        **kwargs
    )


def create_dino_teacher_student_pair(
        variant: ModelVariant = "small",
        *,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        dino_out_dim: int = 65536,
        **kwargs
) -> Tuple[DINOv1, DINOv1]:
    """
    Create teacher-student pair for DINO self-supervised learning.

    The returned teacher is a weight-for-weight copy of the returned student:
    this factory synchronizes them before returning
    (`common.sync_teacher_to_student`), because DINO's teacher is defined as
    an exponential moving average of the student's own trajectory starting
    from the student's own initialization. The two remain distinct objects
    with distinct variables; only their values agree at construction, and
    `update_teacher_ema` moves them apart from there. See plan decision D-034.
    `DINOTrainingModel.__init__` performs the same synchronization for pairs
    that did not come from this factory, so calling both is harmless.

    Follows the same converged parameter scheme as `create_dino_v1`:
    `image_size` (int or tuple), `patch_size` with the `None`-defers-to-variant
    rule, and no `input_shape`.

    :param variant: Model variant for both teacher and student.
    :type variant: ModelVariant
    :param teacher_temp: Teacher temperature; not used in model creation,
        provided for API compatibility with loss computation.
    :type teacher_temp: float
    :param student_temp: Student temperature; not used in model creation.
    :type student_temp: float
    :param image_size: Input image size.
    :type image_size: Union[int, Tuple[int, int]]
    :param patch_size: Size of image patches. ``None`` resolves to 16 (v1 has
        no per-variant patch size).
    :type patch_size: Optional[Union[int, Tuple[int, int]]]
    :param dino_out_dim: Output dimension for the DINO heads.
    :type dino_out_dim: int
    :param kwargs: Additional arguments passed to both model constructors.
    :return: Tuple of ``(teacher_model, student_model)``. The teacher's
        weight values are identical to the student's; the objects and
        variables are distinct.
    :rtype: Tuple[DINOv1, DINOv1]
    :raises TypeError: If ``input_shape`` is passed; use ``image_size`` instead.

    Example:
        .. code-block:: python

            teacher, student = create_dino_teacher_student_pair(
                variant="small", teacher_temp=0.04, student_temp=0.1, dino_out_dim=65536
            )

        # The teacher starts equal to the student and is thereafter updated by
        # EMA from the student during training.
        ```
    """
    reject_input_shape(kwargs, "create_dino_teacher_student_pair")

    if patch_size is None:
        patch_size = _DEFAULT_PATCH_SIZE

    # Create teacher model
    teacher = DINOv1.from_variant(
        variant=variant,
        num_classes=0,
        image_size=image_size,
        patch_size=patch_size,
        include_projection_head=True,
        dino_out_dim=dino_out_dim,
        name="dino_teacher",
        **kwargs
    )

    # Create student model (identical architecture)
    student = DINOv1.from_variant(
        variant=variant,
        num_classes=0,
        image_size=image_size,
        patch_size=patch_size,
        include_projection_head=True,
        dino_out_dim=dino_out_dim,
        name="dino_student",
        **kwargs
    )

    # DECISION plan-2026-08-01T105809-dc0c402e/D-034
    # The teacher STARTS as the student. Do not remove this and do not make it
    # optional -- without it the "EMA teacher" is an EMA between two unrelated
    # random networks for the first several hundred steps. The full measurement
    # and the reasoning live at `common.sync_teacher_to_student`.
    sync_teacher_to_student(teacher, student)

    logger.info(f"Created DINO teacher-student pair with variant '{variant}'")
    logger.info(f"Teacher temp: {teacher_temp}, Student temp: {student_temp}")
    logger.info("Teacher initialized from the student (D-034)")

    return teacher, student