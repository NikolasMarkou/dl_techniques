"""
Pre-norm vision transformer in the DINO family, with optional rotary position
embeddings.

The DINOv3 paper's central problem is that self-distillation, run long enough and
large enough, degrades *dense* features even while global ones keep improving:
patch representations drift toward a few high-magnitude directions, and
segmentation or depth probes get worse as linear-probe classification gets better.
The paper's answer is Gram anchoring — an extra loss pulling the student's
patch-feature Gram matrix back toward that of an earlier, frozen copy of itself —
plus a positional scheme that does not bake in one resolution.

That second half is what this file implements. `positional_embedding_type='rope'`
replaces the learned absolute table with rotary embeddings applied to Q and K
inside the attention operator. Rotating the queries and keys by an angle
proportional to position makes the resulting score depend on the *difference* of
two positions rather than on their absolute values, so the model generalizes
across sequence lengths that a fixed learned table cannot represent at all. The
first half is not implemented: **Gram anchoring is an explicit non-goal here**, and
none of its machinery — a third frozen Gram teacher, a Gram-matrix loss term —
exists in this repository.

RoPE is reached through the registered `group_query` attention with
`num_kv_heads == num_heads`, which reduces grouped-query attention to ordinary
multi-head attention that happens to rotate Q and K. Two alternatives were
rejected for concrete reasons. Applying `RotaryPositionEmbedding` to the token
stream before the projections destroys the relative-position property the
mechanism is defined by, because the rotation must act on Q and K *after* they are
projected. Passing rope arguments to the `multi_head` type does not work either:
until 2026-08-17 it did not even complain — the attention factory silently dropped
unknown keys, so such a model built, forward-passed and round-tripped with RoPE
entirely absent. `create_attention_layer` now raises on those keys
(plan-2026-08-17T183311-79c63e38/D-011), so the mistake is loud rather than silent;
the reason `'group_query'` is used here is unchanged.

Under `'rope'` the learned absolute table is omitted, never stacked on top of the
rotation. Two independent position signals would be redundant, and the learned
table alone breaks permutation equivariance, which would make it impossible to
tell a live rotation from a dead one. A consequence worth knowing: `rope_percentage=0.0`
is legal and leaves the model with *no* positional information whatsoever, a
permutation-equivariant bag of patches. It exists as the control arm of the
RoPE-liveness test, not as a training configuration. A checkpoint is not portable
between the two positional modes — they instantiate different attention classes.

Everything else is a conventional pre-norm ViT: patch embedding, a learnable
[CLS] token, `depth` `TransformerLayer` encoder blocks with a linearly increasing
stochastic-depth rate, a final normalization, and the [CLS] row as the feature
vector, optionally followed by a classifier. Pre-norm is not incidental — it is
what keeps DINO-style self-distillation stable at depth.

Several other DINOv3 mechanisms are absent, named here so the file does not claim
by its title what it does not do. The RoPE here is **1-D over the flattened token
sequence** (position = token index); the paper uses a 2-D axial formulation over
patch (row, column) coordinates with random coordinate jittering during training.
The rotation implemented here is real and live, but it is not the paper's.
**Sinkhorn-Knopp centering is not implemented** — `dl_techniques.losses.dino_loss`
offers EMA centering only. **Register tokens are not implemented in this model**;
`dino_v2.DINOv2VisionTransformer` has them. **High-resolution adaptation and
distillation from a large pretrained teacher are not implemented**, and no
pretrained weights are shipped — `pretrained=True` raises `NotImplementedError`
rather than handing back a randomly initialized model.

The variant table is the only place in the DINO trio where `patch_size=None`
resolves to something other than a constant: `giant` carries `(14, 14)` and
`stochastic_depth_rate=0.4` while every other variant carries `(16, 16)`, so
`create_dino_v3('giant')` gives a /14 model and passing `patch_size=16`
explicitly gives a /16 one. Giving that parameter a concrete default would
silently break the first of those.

`get_last_selfattention` raises `NotImplementedError` rather than returning a
zero tensor a caller cannot distinguish from a broken model. The gap differs by
path and both are real: under `'learned'` the multi-head attention classes accept
no `return_attention_scores` at all, while under `'rope'` `GroupedQueryAttention`
does return a correct probability map but `TransformerLayer.call` never forwards
the flag.

References:
    - Siméoni et al., 2025. DINOv3. (arXiv preprint; Gram anchoring and 2-D axial
      RoPE, neither implemented here)
    - Caron et al., 2021. Emerging Properties in Self-Supervised Vision
      Transformers. (https://arxiv.org/abs/2104.14294)
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.dino.common import reject_input_shape

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DINOv3(keras.Model):
    """
    DINOv3 Vision Transformer Model Implementation.

    A pre-normalization Vision Transformer backbone, usable for classification and
    as the trunk of DINO-style self-supervised training. Pre-norm is what the DINO
    line relies on for self-distillation stability.

    The model consists of:
    - A patch embedding layer to convert images into sequences of tokens.
    - A learnable [CLS] token for global image representation.
    - Positional information, selected by ``positional_embedding_type``:
      either a learned absolute embedding table added to the token stream
      (``"learned"``), or 1-D rotary position embeddings applied to Q and K
      inside every attention operator (``"rope"``). The two are mutually
      exclusive — under ``"rope"`` no learned table is created at all.
    - A stack of pre-normalized Transformer encoder layers.
    - A final normalization layer.
    - An optional classification head.

    Implemented vs. not implemented (DINOv3 mechanisms):
        RoPE is implemented (1-D, over the flattened token sequence).
        **Gram anchoring is NOT implemented** — it requires a third frozen "Gram
        teacher" network and is an explicit non-goal here. **Sinkhorn-Knopp
        centering is NOT implemented.** **2-D axial RoPE with coordinate jittering
        is NOT implemented** — the rotation here is 1-D over token index.
        **Register tokens are NOT implemented** in this model. See the module
        docstring for the full list.

    Args:
        image_size: Integer, or tuple of integers (height, width), for the input
            image. An integer ``s`` is normalized to ``(s, s)``.
            Defaults to (224, 224).
        patch_size: Integer, or tuple of integers (height, width), for the image
            patches. An integer ``p`` is normalized to ``(p, p)``.
            Defaults to (16, 16).
        num_classes: Number of output classes for the classification head. If 0,
            no head is added. Defaults to 1000.
        embed_dim: The dimensionality of the token embeddings. Defaults to 768.
        depth: The number of transformer encoder layers. Defaults to 12.
        num_heads: The number of attention heads in each transformer layer.
            Defaults to 12.
        mlp_ratio: Ratio to determine the hidden dimension of the FFN in
            transformer layers (hidden_dim = embed_dim * mlp_ratio). Defaults to 4.0.
        qkv_bias: If True, add a learnable bias to the query, key, and value
            projections. Defaults to True.
        dropout_rate: Dropout rate for the embedding and FFN layers.
            Defaults to 0.0.
        attention_dropout_rate: Dropout rate for the attention weights.
            Defaults to 0.0.
        stochastic_depth_rate: Maximum drop rate for stochastic depth, which
            linearly increases across layers. Defaults to 0.0.
        normalization_type: The type of normalization to use ('layer_norm',
            'rms_norm'). Defaults to 'layer_norm'.
        positional_embedding_type: How positional information enters the model.
            ``'learned'`` (default) adds a learned absolute embedding table to the
            token stream. ``'rope'`` instead applies rotary position embeddings to
            Q and K inside each attention operator and creates **no** learned
            table. Defaults to 'learned'.
        rope_theta: Base frequency for the rotary embeddings. Only read when
            ``positional_embedding_type='rope'``. Defaults to 10000.0.
        rope_percentage: Fraction of each head's dimensions the rotation is
            applied to. Only read when ``positional_embedding_type='rope'``.
            **0.0 disables the rotation entirely**, which — since no learned table
            is created under ``'rope'`` — leaves the model with NO positional
            information at all (a permutation-equivariant bag of patches). That
            configuration is legal because it is the control arm the RoPE-liveness
            test needs; it is not a useful training configuration.
            Defaults to 1.0.
        activation: Activation function for the FFN layers. Defaults to 'gelu'.
        kernel_initializer: Initializer for kernel weights. Defaults to
            'glorot_uniform'.
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        include_top: If True, include the final classification head. If False,
            the model outputs features from the transformer. Defaults to True.
        **kwargs: Additional arguments for the `keras.Model` base class.

    Input shape:
        A 4D tensor of shape `(batch_size, height, width, channels)`, where
        height and width must match `image_size`.

    Output shape:
        - If `include_top=True`: A 2D tensor of shape `(batch_size, num_classes)`.
        - If `include_top=False`: A 2D tensor of shape `(batch_size, embed_dim)`
          representing the [CLS] token features.

    Raises:
        ValueError: If model parameters are invalid or incompatible.
    """

    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "small": {
            "embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "base": {
            "embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "large": {
            "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "giant": {
            "embed_dim": 1536, "depth": 40, "num_heads": 24, "mlp_ratio": 4.0,
            "patch_size": (14, 14), "stochastic_depth_rate": 0.4
        }
    }

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        normalization_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        positional_embedding_type: Literal['learned', 'rope'] = 'learned',
        rope_theta: float = 10000.0,
        rope_percentage: float = 1.0,
        activation: Union[str, Callable] = 'gelu',
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        # Normalize int-or-tuple spellings BEFORE any subscript. Passing
        # `image_size=224` used to crash on `image_size[0]` with a bare TypeError
        # instead of building a 224x224 model, unlike v1/v2 which both accept an int.
        image_size = (
            tuple(image_size) if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        patch_size = (
            tuple(patch_size) if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )

        # Input validation
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if positional_embedding_type not in ('learned', 'rope'):
            raise ValueError(
                f"positional_embedding_type must be 'learned' or 'rope', "
                f"got '{positional_embedding_type}'"
            )
        if rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

        # DECISION plan_2026-06-15_39a31d4a/D-001: a Functional Model calls
        # super().__init__(inputs=, outputs=) EXACTLY once. The previous bare
        # super().__init__(**kwargs) here was a double-init (the functional call
        # below at the end of __init__ is the real one). Do NOT re-add a bare
        # super().__init__ before the functional graph is built.

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.normalization_type = normalization_type
        self.positional_embedding_type = positional_embedding_type
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.include_top = include_top

        # Compute derived values
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.sequence_length = self.num_patches + 1

        # Build the model using the functional API pattern
        inputs = keras.Input(shape=(*image_size, 3), name="input_image")
        outputs = self._build_model(inputs)

        # Finalize the Model
        super().__init__(inputs=inputs, outputs=outputs, name="DINOv3", **kwargs)

        logger.info(
            f"Created DINOv3 model with {depth} layers, {embed_dim} embedding dim for "
            f"input shape {image_size}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Constructs the model architecture."""
        # 1. Patch Embedding
        x = self._build_patch_embedding(inputs)

        # 2. Add CLS Token and Positional Embedding
        x = self._build_token_processing(x)

        # 3. Transformer Encoder Layers
        x = self._build_encoder(x)

        # 4. Final Processing and Head
        x = self._build_head(x)

        return x

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the patch embedding layer."""
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='patch_embedding'
        )
        return self.patch_embed(inputs)

    def _build_token_processing(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Adds the [CLS] token and, for the 'learned' mode, positional embeddings."""
        # DECISION plan_2026-06-15_39a31d4a/D-001: this runs inside _build_model,
        # which executes BEFORE the functional super().__init__ — so an inline
        # self.add_weight(cls_token) here would fire before super().__init__ and
        # crash. The CLS token is owned by the ClassTokenPrepend sub-layer (build()
        # owns add_weight) called in the functional graph. Do NOT inline add_weight.
        self.cls_token_layer = ClassTokenPrepend(name="cls_token")
        x = self.cls_token_layer(x)

        # DECISION plan-2026-08-01T105809-dc0c402e/D-015: under 'rope' the learned
        # absolute table is OMITTED, never stacked on top of the rotation. Adding
        # both would give the model two independent, redundant position signals and
        # would make the RoPE-liveness test unable to tell a live rotation from a
        # dead one (the learned table alone breaks permutation equivariance).
        # Do NOT "keep the table for compatibility" under 'rope'.
        if self.positional_embedding_type == 'rope':
            self.pos_embed = None
            # PositionalEmbedding also owned the post-embedding dropout; preserve it.
            if self.dropout_rate > 0.0:
                self.embed_dropout = layers.Dropout(
                    self.dropout_rate, name="embedding_dropout"
                )
                x = self.embed_dropout(x)
            else:
                self.embed_dropout = None
            return x

        # Add positional embedding using the shared layer
        self.embed_dropout = None
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.sequence_length,
            dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="positional_embedding"
        )
        return self.pos_embed(x)

    def _attention_spec(self) -> Tuple[str, Dict[str, Any]]:
        """The attention type + factory args every encoder block is built with.

        Returns ``(attention_type, attention_args)``. Under
        ``positional_embedding_type='rope'`` this selects the rope-capable
        ``group_query`` attention with ``num_kv_heads == num_heads``; otherwise the
        plain ``multi_head`` attention, byte-identically to the pre-RoPE behaviour.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-015: RoPE is reached through the
        # registered `group_query` attention (Route A), NOT by applying
        # RotaryPositionEmbedding to the token stream (Route B). RoPE must rotate Q
        # and K AFTER their projections; rotating the token stream BEFORE them
        # destroys the relative-position property the mechanism is defined by
        # (MEASURED: Toeplitz defect 5.078e+01 vs 5.722e-06 at score magnitude ~35).
        # `num_kv_heads == num_heads` makes GQA plain MHA (num_groups == 1, no K/V
        # repeat) — verified against an independent hand-computed MHA oracle to
        # 2.22e-06 at magnitude 10.88.
        #
        # WHAT NOT TO DO: do NOT "simplify" this to `multi_head` plus a rope kwarg.
        # `create_attention_layer` USED TO SILENTLY DROP an unknown key (D-010,
        # MEASURED at the time — the opposite of `create_ffn_layer`, which raises),
        # so `attention_type='multi_head', attention_args={'rope_theta': ...}`
        # built, forward-passed and round-tripped with RoPE entirely absent.
        # HISTORICAL as of 2026-08-17 (plan-2026-08-17T183311-79c63e38/D-011):
        # that factory now RAISES on the undeclared key, so the shortcut fails at
        # construction instead of shipping a position-blind model. The guidance is
        # unchanged — `'group_query'` with `num_kv_heads == num_heads` is the
        # RoPE-carrying plain-MHA path, and it is what this branch returns.
        if self.positional_embedding_type == 'rope':
            return 'group_query', {
                'num_kv_heads': self.num_heads,
                'rope_theta': self.rope_theta,
                'rope_percentage': self.rope_percentage,
                'max_seq_len': self.sequence_length,
                'use_bias': self.qkv_bias,
            }
        return 'multi_head', {'use_bias': self.qkv_bias}

    def _build_encoder(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the stack of transformer encoder layers."""
        self.encoder_layers = []
        # DECISION plan-2026-08-11T165740-53dac34a/D-004: use the shared
        # `linear_drop_path_rates` helper, NOT a hand-rolled `ops.linspace` ramp.
        # It returns plain Python floats, so this also retires the older
        # plan_2026-06-15_2a23a001/D-004 hazard (keras tensors have no `.item()`;
        # that fix spelled the conversion `float(r)`). Do not reintroduce either
        # a `linspace` call or a `.item()` conversion here.
        dpr = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        attention_type, attention_args = self._attention_spec()

        for i in range(self.depth):
            encoder_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * self.mlp_ratio),
                attention_type=attention_type,
                attention_args=dict(attention_args),
                normalization_type=self.normalization_type,
                normalization_position='pre',  # DINO uses pre-norm
                ffn_type='mlp',
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=dpr[i] > 0.0,
                stochastic_depth_rate=dpr[i],
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'encoder_layer_{i}'
            )
            x = encoder_layer(x)
            self.encoder_layers.append(encoder_layer)
        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the final normalization and classification head."""
        # Final normalization using the shared factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='final_norm'
        )
        x = self.norm(x)

        # Extract [CLS] token for classification
        features = x[:, 0]

        # Add classification head if requested
        if self.include_top:
            if self.num_classes > 0:
                self.classifier = layers.Dense(
                    units=self.num_classes,
                    kernel_initializer="truncated_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='classifier'
                )
                outputs = self.classifier(features)
            else:
                # If include_top is True but num_classes is 0, return features
                outputs = features
        else:
            # If not including top, return features
            outputs = features

        return outputs

    def get_last_selfattention(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Not implemented. Raises ``NotImplementedError``.

        DINO's attention-map visualization needs the last block's attention
        PROBABILITIES. This model composes ``TransformerLayer``, whose ``call()``
        signature is ``(inputs, attention_mask, layer_idx, training)`` and which
        returns only the block output — it has no way to surface its attention
        sub-layer's probabilities, under EITHER
        ``positional_embedding_type``. MEASURED on keras 3.8.0.

        The gap differs per path, and both are named here rather than collapsed:

        - ``'learned'`` -> ``multi_head`` attention: ``MultiHeadAttention.call`` and
          the ``MultiHeadCrossAttention.call`` it delegates to accept no
          ``return_attention_scores`` at all, and cache no probabilities.
        - ``'rope'`` -> ``group_query`` attention: ``GroupedQueryAttention.call``
          DOES accept ``return_attention_weights`` and returns a correct
          ``(batch, heads, seq, seq)`` map whose rows sum to 1 — but
          ``TransformerLayer.call`` does not forward that flag, so the capability
          is unreachable without reaching into block internals and re-implementing
          the block's pre-norm ordering in this model.

        Implementing this truthfully means adding the flag through
        ``TransformerLayer`` (and, for the learned path, through
        ``MultiHeadAttention`` / ``MultiHeadCrossAttention``) — a change to shared
        layers used across the repository, out of scope for this model file.

        Args:
            inputs: A batch of images. Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DINOv3.get_last_selfattention() is not implemented. It needs the last "
            "block's attention probabilities, and TransformerLayer.call does not "
            "forward a return_attention_scores / return_attention_weights flag. "
            "For positional_embedding_type='learned' the multi_head attention path "
            "(MultiHeadAttention / MultiHeadCrossAttention) does not accept such a "
            "flag either; for 'rope' the group_query path (GroupedQueryAttention) "
            "does accept return_attention_weights, but TransformerLayer does not "
            "pass it through. Previously this method returned an all-zero tensor, "
            "which is indistinguishable from a broken model."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        num_classes: int = 1000,
        include_top: bool = True,
        **kwargs: Any
    ) -> "DINOv3":
        """
        Creates a DINOv3 model from a predefined variant.

        Args:
            variant: The model variant, one of "tiny", "small", "base", "large", "giant".
            image_size: The input image size; an int, or (height, width).
            num_classes: Number of output classes.
            include_top: Whether to include the classification head.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            A DINOv3 model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINOv3-{variant.upper()} model with config: {config}")

        return cls(
            image_size=image_size,
            num_classes=num_classes,
            include_top=include_top,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        config = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'normalization_type': self.normalization_type,
            'positional_embedding_type': self.positional_embedding_type,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'include_top': self.include_top,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOv3":
        """Creates a model from its configuration."""
        return cls(**config)


def create_dino_v3(
    variant: str = "base",
    *,
    image_size: Union[int, Tuple[int, int]] = (224, 224),
    patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_classes: int = 1000,
    include_top: bool = True,
    positional_embedding_type: Literal['learned', 'rope'] = 'learned',
    rope_theta: float = 10000.0,
    rope_percentage: float = 1.0,
    pretrained: bool = False,
    **kwargs: Any
) -> DINOv3:
    """
    A factory function to create DINOv3 models.

    Signature note (converged surface): ``create_dino_v1``, ``create_dino_v2`` and
    ``create_dino_v3`` share ``(variant, *, image_size, patch_size, num_classes,
    include_top, **kwargs)``. ``patch_size`` used to be reachable only through
    ``**kwargs`` here; it is now a named parameter on all three. There is no
    ``input_shape`` spelling on any of them — the input shape is derived from
    ``image_size``.

    **Variant-defers precedence rule** (shared by all three factories):
    ``patch_size=None`` defers to the variant's own ``MODEL_VARIANTS`` entry; an
    EXPLICIT non-``None`` value ALWAYS wins over it. This matters here and only
    here: ``DINOv3.MODEL_VARIANTS['giant']`` carries ``patch_size=(14, 14)`` while
    every other variant carries ``(16, 16)``, so ``create_dino_v3('giant')`` gives
    a /14 model and ``create_dino_v3('giant', patch_size=16)`` gives a /16 one.
    ``giant`` likewise carries ``stochastic_depth_rate=0.4``, overridable the same
    way by passing it through ``**kwargs``.

    Args:
        variant: Model variant ("tiny", "small", "base", "large", "giant").
        image_size: Input image size; an int, or (height, width).
        patch_size: Patch size; an int, or (height, width). ``None`` defers to the
            variant ((14, 14) for 'giant', (16, 16) otherwise).
        num_classes: Number of output classes.
        include_top: Whether to include the final classification layer.
        positional_embedding_type: ``'learned'`` (absolute table) or ``'rope'``
            (1-D rotary, applied inside a ``group_query`` attention). A checkpoint
            is NOT portable between the two — they run different attention classes.
        rope_theta: RoPE base frequency. Ignored unless
            ``positional_embedding_type='rope'``.
        rope_percentage: Fraction of each head's dimensions that are rotated.
            Ignored unless ``positional_embedding_type='rope'``. ``0.0`` is legal
            but leaves the model with NO positional information at all.
        pretrained: Must be False. `True` raises `NotImplementedError` — NO
            pretrained DINOv3 weights are shipped with this repository.
        **kwargs: Additional arguments for the model constructor, e.g.
            ``stochastic_depth_rate``, ``normalization_type``.

    Returns:
        A DINOv3 model instance.

    Raises:
        TypeError: If ``input_shape`` is passed — use ``image_size`` instead.
        NotImplementedError: If ``pretrained=True`` (no checkpoints shipped).
    """
    reject_input_shape(kwargs, "create_dino_v3")

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained DINOv3 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_dino_v3('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    # DECISION plan-2026-08-01T105809-dc0c402e/D-017: `patch_size=None` defers to
    # the variant. Do NOT give this parameter a concrete default (e.g. `= 16`):
    # `DINOv3.from_variant` does `config = MODEL_VARIANTS[variant].copy();
    # config.update(kwargs)`, so a non-None value ALWAYS overrides the variant's —
    # and a concrete default would silently turn `create_dino_v3('giant')` from
    # the /14 model its variant entry specifies into a /16 one. `None` is the only
    # value that means "the caller said nothing". See decisions.md D-017.
    if patch_size is not None:
        kwargs['patch_size'] = patch_size

    return DINOv3.from_variant(
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        include_top=include_top,
        positional_embedding_type=positional_embedding_type,
        rope_theta=rope_theta,
        rope_percentage=rope_percentage,
        **kwargs
    )
