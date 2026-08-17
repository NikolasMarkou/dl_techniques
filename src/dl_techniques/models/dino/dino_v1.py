"""
DINOv1 vision transformer trunk, its projection head, and the teacher/student pair.

DINO learns visual features with no labels by asking one network to predict what
another network — a slowly moving copy of itself — says about a different crop of
the same image. Both networks end in a projection to `dino_out_dim` prototypes,
and the objective is the cross-entropy `-sum_k p_t(k) log p_s(k)` between the two
softmax distributions. Self-distillation of this kind has one failure mode that
dominates everything else: collapse, where the network maps every image to the
same output and the loss is trivially minimized. DINO resolves it with two
opposing forces applied to the teacher only. Centering subtracts an
exponential-moving-average of the teacher's own logits, which penalizes any single
prototype from dominating and pushes the output toward uniform. Sharpening divides
by a low teacher temperature, which pushes it toward one-hot. Neither alone is
stable — centering alone collapses to uniform, sharpening alone to a single
dimension — and the balance between them is what keeps training alive.

**None of that machinery is in this file.** Centering, sharpening and the teacher
temperature schedule live in `dl_techniques.losses.dino_loss.DINOLoss`; the EMA
update of the teacher's weights lives in
`dl_techniques.models.dino.training.DINOTrainingModel.update_teacher_ema`. This
module supplies only the pieces those need: the ViT trunk, the projection head,
and a factory that returns a correctly-initialized teacher/student pair.

Architecturally the trunk is a plain pre-normalized ViT: `PatchEmbedding2D`
tokenizes the image, `ClassTokenPrepend` prepends a learnable [CLS], a learned
absolute positional table is added, `depth` `TransformerLayer` blocks follow with
a linearly increasing stochastic-depth rate, and a final normalization precedes
feature extraction. Attention, FFN and normalization are all selected by string
through the shared factories, so the same class covers several block variants.
Features are the [CLS] row when `use_cls_token=True` and the mean over patch
tokens otherwise. Exactly one head is attached, and `include_projection_head`
takes precedence over `include_top`: the DINO head, else a classifier `Dense` when
`num_classes > 0`, else the raw features.

Two places in the code look wrong until you know why they are not. First,
`qkv_bias` is forwarded to the attention factory spelled `use_bias`, not
`qkv_bias`, and unconditionally. The attention registry USED TO silently drop an
unrecognized keyword rather than raising, so the natural spelling built a bias-free
attention while the caller believed otherwise; since 2026-08-17
(plan-2026-08-17T183311-79c63e38/D-011) `create_attention_layer` raises on such a
key, so the same mistake would now fail loudly at construction — the spelling here
is still `use_bias` because that is the name the registry accepts. Second, the projection head's
pre-projection L2 normalization is computed in `variable_dtype` rather than
`compute_dtype`. Under `mixed_float16` the sum of squares over `bottleneck_dim`
overflows fp16 long before any individual activation does, and `x / inf` is zero —
the head would return exactly zero for every sample, with no NaN and no error.

`norm_last_layer` is honoured as an invariant rather than as a
reparameterization. The reference wraps the final projection in weight norm
`w = g * v / ||v||` with `g` pinned to 1; here a `UnitNorm(axis=0)` kernel
constraint plus a one-off normalization at build time gives the identical
invariant (`||kernel[:, j]|| == 1` for every prototype `j`) without adding
forward-path arithmetic to a kernel that is 256 x 65536 at paper scale. The
optimization path differs — a constraint projects after each step where weight
norm reparameterizes before it — and the build-time normalization exists because
a Keras constraint is applied by the optimizer, so a never-trained head would
otherwise violate the invariant the flag promises.

`create_dino_teacher_student_pair` synchronizes the teacher to the student before
returning. This is deliberate and not an optimization: DINO defines the teacher as
an EMA of the student's own trajectory from the student's own initialization, so
without the copy the "EMA teacher" is an average of two unrelated random networks
for the first several hundred steps.

`get_last_selfattention` raises `NotImplementedError` instead of returning zeros.
The `multi_head` attention path exposes no attention probabilities, and the
previous zero-tensor stub made an attention-map visualization render an all-black
image that a caller could not distinguish from uniform attention.

The `giant` variant is not from the DINOv1 paper, which stops at ViT-B/8. It
exists so the `MODEL_VARIANTS` key sets of `DINOv1`, `DINOv2VisionTransformer` and
`DINOv3` match, and it carries the shared ViT-g/14 dimensions with no
version-specific extras — v2's giant adds `ffn_type='swiglu'` and v3's adds
`patch_size=(14, 14)` and `stochastic_depth_rate=0.4`, both of which are v2/v3
mechanisms rather than v1's.

The three `create_dino_v*` factories share one parameter scheme. `input_shape` is
not among them and raises `TypeError`; the input shape is always derived from
`image_size`, because the two spellings could disagree and silently build a model
with the wrong patch count. `patch_size=None` defers to the variant's own entry,
and `DINOv1.MODEL_VARIANTS` defines none, so it resolves to 16 for every variant.

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
from dl_techniques.models.dino.common import (
    reject_input_shape,
    sync_teacher_to_student,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates

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


@keras.saving.register_keras_serializable()
class DINOHead(keras.layers.Layer):
    """
    DINO projection head for self-supervised learning.

    This head projects the [CLS] token representation to a higher-dimensional
    space and applies normalization and a final linear layer for contrastive learning.

    Args:
        in_dim: Integer, input dimension (backbone output dimension).
        out_dim: Integer, output dimension for contrastive learning.
        use_bn: Boolean, whether to use batch normalization in intermediate layers.
        norm_last_layer: Boolean, whether to constrain the final projection's
            weights to unit L2 norm per output unit. See the "Last-layer weight
            normalization" note below for exactly what is and is not implemented.
        nlayers: Integer, number of layers in the projection head (minimum 1).
        hidden_dim: Integer, hidden dimension in intermediate layers.
        bottleneck_dim: Integer, dimension before the final projection layer.
        normalization_type: String, type of normalization to use.
        activation: String or callable, activation function to use.
        dropout_rate: Float, dropout rate for regularization.
        kernel_initializer: String or initializer, weight initialization scheme.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, in_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, out_dim)`

    Last-layer weight normalization (``norm_last_layer``):
        The reference DINO implementation wraps the final projection in PyTorch's
        weight-norm reparameterization ``w = g * v / ||v||`` and, when
        ``norm_last_layer=True``, pins ``g = 1`` and freezes it — so every output
        prototype has unit L2 norm throughout training.

        Here that INVARIANT is reproduced with a
        ``keras.constraints.UnitNorm(axis=0)`` on the final ``Dense`` kernel plus
        a one-off normalization at ``build()`` time, NOT with a ``(g, v)``
        reparameterization. The invariant (``||kernel[:, j]||_2 == 1`` for every
        output unit ``j``) is identical; the optimization path is not — the
        constraint PROJECTS after each optimizer step where the reference
        reparameterizes before it. ``norm_last_layer=False`` leaves the kernel
        unconstrained, matching the reference's trainable-``g`` branch only in
        that the norms are then free.

    Example:
        ```python
        # Create DINO head for 384-dim input to 65536-dim output
        dino_head = DINOHead(
            in_dim=384,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256
        )

        # Forward pass
        cls_token = keras.Input(shape=(384,))
        projection = dino_head(cls_token)
        ```
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
            kernel_initializer: str = "truncated_normal",
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
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Initialize layer lists
        self.mlp_layers = []
        self.last_layer = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the DINO head layers."""
        # DECISION plan_2026-06-14_8c7365d0/D-006
        # Reset the sublayer accumulators to their __init__ empty state BEFORE
        # appending. build() must be idempotent: a second build (functional-API
        # reuse or from_config reconstruction) would otherwise duplicate every
        # sublayer/weight. Do NOT append without clearing first.
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
        # DECISION plan-2026-08-01T105809-dc0c402e/D-011
        # `norm_last_layer` is honoured by a UnitNorm(axis=0) CONSTRAINT on the
        # Dense kernel, not by a (g, v) weight-norm reparameterization. Do NOT
        # "upgrade" this to PolarWeightNorm or a hand-rolled g/v split: both add
        # forward-path arithmetic on a kernel that is (256 x 65536) at paper
        # scale, and a post-build `radius.trainable = False` does NOT survive a
        # .keras reload (build() recreates the variable trainable). Also do NOT
        # drop the build-time normalize below: a Keras constraint is applied by
        # the OPTIMIZER, so without it a freshly built or never-trained head
        # violates the invariant this flag promises (MEASURED: column norms
        # 0.089-0.136 before the first step, 1.0 +/- 2.4e-07 after).
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

        # Apply the constraint once at build time so the unit-norm invariant
        # holds for a never-trained head too (see D-011 above). On a .keras
        # reload this runs BEFORE the saved weights are restored, and those
        # weights already satisfy the constraint, so it does not perturb them.
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
        """
        Forward pass of the DINO head.

        Args:
            inputs: Input tensor of shape (batch_size, in_dim).
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Projected tensor of shape (batch_size, out_dim).
        """
        x = inputs

        # Apply MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # L2 normalize before final projection (as in DINO paper).
        #
        # DECISION plan-2026-08-01T105809-dc0c402e/D-020
        # The normalization runs in `variable_dtype`, NOT in `compute_dtype`. Do
        # NOT "simplify" this back to a bare
        # `keras.utils.normalize(x, axis=-1, order=2)`: under `mixed_float16`
        # that reduces `sum(x**2)` over `bottleneck_dim` in fp16, and the sum
        # overflows 65504 long before any individual value does. Overflow gives
        # `x / inf == 0`, so the head returns EXACTLY ZERO for every sample --
        # no NaN, no Inf, no error, a silently dead projection head.
        # MEASURED at the ordinary DINO head scale (in_dim=384, hidden=2048,
        # bottleneck=256, weights ~N(0, 0.5)): pre-normalize `sum(x**2)` reaches
        # 1.649e+09, fp16 output absmax 0.0 (100% of entries exactly zero) vs
        # float32 absmax 0.2536 on bit-identical weights.
        # `variable_dtype` is float32 under `mixed_float16` and equals
        # `compute_dtype` under float32/float64, so this is a no-op outside
        # mixed precision (float32 outputs are unchanged).
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
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer,
        })
        return config


@keras.saving.register_keras_serializable()
class DINOv1(keras.Model):
    """
    DINO Vision Transformer model for self-supervised learning.

    This model implements the Vision Transformer backbone used in DINO
    (DIstillation with NO labels) self-supervised learning framework.
    It can be used both for feature extraction and with the DINO head
    for contrastive self-supervised training.

    Args:
        embed_dim: Integer, embedding dimension of the model.
        depth: Integer, number of transformer layers.
        num_heads: Integer, number of attention heads.
        patch_size: Integer or tuple, size of image patches.
        image_size: Integer or tuple, input image size. Default is 224.
        in_channels: Integer, number of input channels. Default is 3.
        num_classes: Integer, number of output classes for classification.
            Set to 0 for feature extraction only.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
        qkv_bias: Boolean, whether to use bias in the attention QKV/output
            projections. Forwarded to the attention factory as `use_bias`.
        dropout_rate: Float, dropout rate.
        attention_dropout_rate: Float, attention dropout rate.
        stochastic_depth_rate: Float, stochastic depth rate.
        normalization_type: String, normalization layer type.
        attention_type: String, type of attention mechanism to use.
        ffn_type: String, type of feed-forward network to use.
        include_top: Boolean, whether to include classification head.
        include_projection_head: Boolean, whether to include DINO projection head.
        dino_out_dim: Integer, output dimension for DINO head.
        dino_hidden_dim: Integer, hidden dimension for DINO head.
        dino_bottleneck_dim: Integer, bottleneck dimension for DINO head.
        dino_nlayers: Integer, number of layers in DINO head.
        use_cls_token: Boolean, whether to use [CLS] token.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        - If include_projection_head=True: 2D tensor `(batch_size, dino_out_dim)`
        - If include_top=True and num_classes>0: 2D tensor `(batch_size, num_classes)`
        - Otherwise: 2D tensor `(batch_size, embed_dim)`

    Example:
        ```python
        # Feature extraction model
        model = DINOv1(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            num_classes=0,
            include_top=False,
            image_size=224
        )

        # Self-supervised learning model with DINO head
        model = DINOv1(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            num_classes=0,
            include_projection_head=True,
            dino_out_dim=65536,
            image_size=224
        )
        ```
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
            # DECISION plan_2026-06-15_39a31d4a/D-001: add_weight (cls_token) must
            # not fire before super().__init__ in a Functional model. The CLS token
            # is owned by the ClassTokenPrepend sub-layer (its build() runs add_weight),
            # called inside this functional graph, so DINOv1 itself creates no weight
            # before super().__init__(inputs=, outputs=). Do NOT inline self.add_weight
            # here — that re-introduces the pre-super crash.
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
            # DECISION plan-2026-08-01T105809-dc0c402e/D-010
            # Forward qkv_bias UNCONDITIONALLY and spell it `use_bias` — the
            # name the attention registry actually accepts. Do NOT reinstate
            # either half of the old form
            # (`{"qkv_bias": ...} if attention_type == "multi_head_attention"`):
            # the gate string was never a registry key (the key is
            # `multi_head`), and `create_attention_layer` USED TO SILENTLY DROP
            # an unrecognized kwarg rather than raising (MEASURED at the time:
            # passing qkv_bias=True yielded a layer with use_bias=False and zero
            # bias weights) — so both halves were independently dead. HISTORICAL
            # as of 2026-08-17 (plan-2026-08-17T183311-79c63e38/D-011): the
            # factory now RAISES on an undeclared key, in line with
            # `create_ffn_layer`, so the old form would fail at construction
            # rather than quietly. The instruction stands either way: never
            # infer a factory's drop-or-raise behaviour, execute it.
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
            self.head = keras.layers.Dense(
                units=self.num_classes,
                kernel_initializer="truncated_normal",
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
        """
        Get attention probabilities from the last transformer layer.

        NOT IMPLEMENTED — see `Raises`. This method exists so that the DINO
        attention-map visualization API of the reference implementation fails
        loudly rather than silently returning something useless.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Raises:
            NotImplementedError: Always. See the message for the missing
                capability.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-012
        # RAISE. Do NOT restore the previous body (log a warning, then
        # `return keras.ops.zeros((batch, heads, seq, seq))`): it made an
        # attention-map visualization silently render an all-black map, and a
        # caller could not tell "uniform attention" from "no implementation".
        # Implementing it truthfully is blocked at the layer level, MEASURED on
        # keras 3.8.0 by signature inspection: neither `TransformerLayer.call`
        # (inputs, attention_mask, layer_idx, training), nor
        # `MultiHeadAttention.call` (inputs, attention_mask, training), nor the
        # `MultiHeadCrossAttention.call` it delegates to (query_input, kv_input,
        # attention_mask, training) accepts a `return_attention_scores` flag or
        # caches the probabilities on an attribute. Other registry types DO
        # (`performer`, `rpc`, `hopfield`) — so the fix is to add that flag to
        # `multi_head`/`TransformerLayer`, in layers/, not to fake it here.
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
        """
        Create a DINO model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "giant".
            num_classes: Integer, number of output classes.
            patch_size: Integer or tuple, size of image patches.
            input_shape: Tuple, an explicit input shape. Prefer `image_size`
                (passed through **kwargs); if this is None the constructor
                derives `(*image_size, in_channels)`. The two spellings can
                DISAGREE, which is why the `create_dino_v1` factory refuses
                this one outright.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            DINOv1 model instance.

        Example:
            ```python
            # Create DINO-Small for feature extraction
            model = DINOv1.from_variant(
                "small",
                num_classes=0,
                patch_size=16
            )

            # Create DINO-Base with projection head
            model = DINOv1.from_variant(
                "base",
                num_classes=0,
                include_projection_head=True,
                dino_out_dim=65536
            )
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-01T105809-dc0c402e/D-033
        # An explicit architecture override in **kwargs WINS over the variant
        # table; it does not collide with it. Do NOT go back to spelling the
        # four table keys out as `embed_dim=config["embed_dim"], ...` next to a
        # bare `**kwargs` -- that form raised
        #   TypeError: DINOv1() got multiple values for keyword argument
        #   'embed_dim'
        # for `from_variant("tiny", embed_dim=32)` and for every caller that
        # reaches it, including `create_dino_teacher_student_pair`. `DINOv2`
        # and `DINOv3` already used the merge form, so this is a convergence
        # on the majority spelling, not a new convention. See decisions.md
        # D-033 (and D-017, the same explicit-value-must-be-honoured rule).
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
        config = {
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
        }
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
    """
    Convenience function to create DINO Vision Transformer models.

    Signature note (converged surface): ``create_dino_v1``, ``create_dino_v2`` and
    ``create_dino_v3`` share ``(variant, *, image_size, patch_size, num_classes,
    include_top, **kwargs)``. The redundant ``input_shape`` spelling was removed —
    the input shape is always derived as ``(*image_size, in_channels)``. Passing
    ``input_shape=`` raises ``TypeError`` rather than silently disagreeing with
    ``image_size`` (a mismatch used to build a model with the wrong patch count).

    ``patch_size`` precedence rule (shared by all three factories):
    ``None`` means "use the variant's own ``patch_size`` if its ``MODEL_VARIANTS``
    entry defines one, otherwise this version's default". An explicitly passed
    ``patch_size`` ALWAYS wins over the variant's. ``DINOv1.MODEL_VARIANTS`` defines
    no per-variant ``patch_size``, so ``None`` resolves to 16 for every v1 variant.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "giant").
        image_size: Integer or ``(height, width)``, input image size.
        patch_size: Integer or tuple, size of image patches. ``None`` defers to the
            variant (v1 has no per-variant patch size, so ``None`` -> 16).
        num_classes: Integer, number of output classes. Set to 0 for feature extraction.
        include_top: Boolean, whether to include classification head.
        include_projection_head: Boolean, whether to include DINO projection head.
        dino_out_dim: Integer, output dimension for DINO head.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        DINOv1 model instance.

    Raises:
        TypeError: If ``input_shape`` is passed — use ``image_size`` instead.

    Example:
        ```python
        # Create DINO-Small for self-supervised learning
        model = create_dino_v1(
            variant="small",
            include_projection_head=True,
            dino_out_dim=65536,
        )

        # Create DINO-Base for fine-tuning
        model = create_dino_v1(
            variant="base",
            image_size=224,
            num_classes=1000,
        )
        ```
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

    **The returned teacher is a WEIGHT-FOR-WEIGHT COPY of the returned student.**
    This factory synchronizes them before returning (`common.sync_teacher_to_student`),
    because DINO's teacher is defined as an exponential moving average of the
    STUDENT'S OWN trajectory starting from the student's own initialization — the
    reference implementation runs `teacher.load_state_dict(student.state_dict())`
    before the first step. The two are still DISTINCT objects with DISTINCT
    variables; only their VALUES agree at construction, and `update_teacher_ema`
    moves them apart from there. See plan decision D-034.

    `DINOTrainingModel.__init__` performs the same synchronization for pairs that
    did not come from this factory, so calling both is harmless (the second copy
    is a no-op).

    Follows the same converged parameter scheme as ``create_dino_v1``:
    ``image_size`` (int or tuple), ``patch_size`` with the ``None``-defers-to-variant
    rule, and no ``input_shape``.

    Args:
        variant: String, model variant for both teacher and student.
        teacher_temp: Float, temperature for teacher model (not used in model creation).
        student_temp: Float, temperature for student model (not used in model creation).
        image_size: Integer or ``(height, width)``, input image size.
        patch_size: Integer or tuple, size of image patches. ``None`` -> 16 (v1 has no
            per-variant patch size).
        dino_out_dim: Integer, output dimension for DINO heads.
        **kwargs: Additional arguments passed to both model constructors.

    Raises:
        TypeError: If ``input_shape`` is passed — use ``image_size`` instead.

    Returns:
        Tuple of (teacher_model, student_model). The teacher's weight VALUES are
        identical to the student's; the objects and variables are distinct.

    Note:
        The temperature parameters are provided for API compatibility but are
        typically applied during loss computation, not in the model architecture.

    Example:
        ```python
        teacher, student = create_dino_teacher_student_pair(
            variant="small",
            teacher_temp=0.04,
            student_temp=0.1,
            dino_out_dim=65536
        )

        # The teacher starts EQUAL to the student and is thereafter updated by
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