"""Energy Transformer image models: one shared backbone with a completion head
and a classification head.

Defines :class:`EnergyTransformerBackbone` (patch embedding, mask token, learned
positional embedding, one :class:`EnergyTransformer` block) and the two models
that compose it: :class:`EnergyTransformerMIM` for masked image completion and
:class:`EnergyTransformerClassifier` for logits. Depth here does not come from
stacking distinct layers. The block defines a scalar energy over the token
states and the forward pass is ``T`` steps of gradient descent on it,

    x <- x - alpha * dE/dg   with   g = EnergyLayerNorm(x)

reusing one block's weights at every step, so ``T`` is an inference-time knob.
The energy sums an attention term, whose gradient lets an occluded token pull
information from its neighbours, and a Hopfield term over a tied memory matrix
that pulls each token toward a stored pattern. Four things a caller needs: the
MIM model trains through stock ``compile(loss='mse')`` and ``fit``, with the
occlusion mask arriving as a ``sample_weight`` in the batch rather than through
any training step here; the classifier mean-pools, because this backbone has no
CLS token; the mask token is created and built in both models, which keeps the
trunks weight-identical for
``load_weights_from_checkpoint(..., skip_prefixes=("decoder_",))``; and both
heads reject a ``return_energy=True`` backbone, since the energy trace is
float32 and a float16 head would overflow it.

References:
    - Hoover et al., 2023. Energy Transformer. NeurIPS 2023 (§3, Table 4).
      (https://arxiv.org/abs/2302.07253)
    - Ramsauer et al., 2020. Hopfield Networks is All You Need.
      (https://arxiv.org/abs/2008.02217)
    - Krotov & Hopfield, 2016. Dense Associative Memory for Pattern Recognition.
      (https://arxiv.org/abs/1606.01164)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for Image
      Recognition at Scale. (https://arxiv.org/abs/2010.11929)
"""

import keras
from keras import layers
from keras.saving import serialize_keras_object, deserialize_keras_object
from typing import Any, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.embedding.mask_token import MaskTokenApply

# The ET block has no factory home; direct import is the sanctioned path here
# (D-004 of plan_2026-07-13_57c9833e). See decisions.md.
from dl_techniques.layers.transformers.energy_transformer import EnergyTransformer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

ETScale = Literal['tiny', 'small', 'base']
HopfieldActivation = Literal['relu', 'softmax']

# `load_weights_from_checkpoint` matches layers by name, so the MIM model and the
# classifier have to name their backbone identically for a warm start to transfer.
BACKBONE_NAME = "et_backbone"

# Every variant shares the paper's Table-4 image defaults: num_steps 12, step_size 0.1,
# beta None (1/sqrt(head_dim)), attn_self False (ET-Full), hopfield_activation 'relu'.
# head_dim stays 64 at every scale; ET attention has no value matrix, so it is free.
SCALE_CONFIGS: Dict[str, Dict[str, int]] = {
    'tiny':  {'embed_dim': 192, 'num_heads': 3,  'head_dim': 64, 'hopfield_dim': 768},
    'small': {'embed_dim': 384, 'num_heads': 6,  'head_dim': 64, 'hopfield_dim': 1536},
    'base':  {'embed_dim': 768, 'num_heads': 12, 'head_dim': 64, 'hopfield_dim': 3072},
}

# Variant registry (house convention, mirrors ViT.MODEL_VARIANTS).
MODEL_VARIANTS: Dict[str, Dict[str, str]] = {
    'et_tiny':  {'scale': 'tiny'},
    'et_small': {'scale': 'small'},
    'et_base':  {'scale': 'base'},
}


def _resolve_scale(variant: str) -> str:
    """Accept either a scale key (``'tiny'``) or a variant key (``'et_tiny'``).

    :param variant: A key of :data:`SCALE_CONFIGS` or of :data:`MODEL_VARIANTS`.
    :type variant: str
    :return: The scale key.
    :rtype: str
    :raises ValueError: If ``variant`` is in neither table; the message lists both.
    """
    if variant in SCALE_CONFIGS:
        return variant
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]['scale']
    raise ValueError(
        f"Unknown variant '{variant}'. Available: "
        f"{sorted(SCALE_CONFIGS)} or {sorted(MODEL_VARIANTS)}"
    )


# DECISION plan-2026-07-14T163315-29a4fef4/D-009: walk the sub-layer tree to set a policy;
# the factory takes no `dtype` and the `dtype_policy` setter does not recurse. See decisions.md.
def _apply_dtype_policy(layer: keras.layers.Layer, policy: Any) -> keras.layers.Layer:
    """Set ``policy`` on ``layer`` and on every sub-layer, before anything is built.

    :param layer: The layer to retag.
    :type layer: keras.layers.Layer
    :param policy: A dtype policy or policy name.
    :type policy: Any
    :return: The same layer.
    :rtype: keras.layers.Layer
    """
    if hasattr(layer, "_flatten_layers"):
        for sub in layer._flatten_layers(include_self=True):
            sub.dtype_policy = policy
    else:  # pragma: no cover
        layer.dtype_policy = policy
    return layer


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.energy_transformer.model")
class EnergyTransformerBackbone(keras.Model):
    """Embed patches, optionally replace masked tokens, and run the energy descent.

    One separately-checkpointable image trunk that both the masked-completion model
    and the classifier compose under the same name, so a pretrained encoder transfers
    into the classifier layer for layer. Call it as ``backbone(image)`` or
    ``backbone((image, input_mask))``; whether a mask is passed is a trace-time
    structural fact, a Python ``if`` on the presence of a second tensor, not a runtime
    choice on tensor values. ``MaskTokenApply`` is created and built either way, which
    is what keeps the two trunks weight-identical. The descent sign lives in the block:
    its ``update()`` returns ``-dE/dg`` and the block adds it.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]
                 │
                 ▼
        ┌─────────────────────┐
        │ patch_embed         │
        └─────────────────────┘
                 │  [B, N, D]
                 ▼
        ┌─────────────────────┐
        │ mask_token          │ ◄── input_mask [B, N] bool (input)
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ pos_embed           │  dropout pos_dropout_rate
        └─────────────────────┘
                 │  cast to the block's variable dtype
                 ▼
        ┌─────────────────────┐
        │ et_block            │  T steps, weights reused
        └─────────────────────┘
                 │
           ┌─────┴─────┐
           ▼           ▼
         tokens      tokens, energies
         [B, N, D]   [B, T+1] float32
         (default)   (return_energy)

    mask_token is built in both models and called only when a mask arrives.

    Dtype path:

    .. code-block:: text

        input at compute_dtype
                      │
                      ▼
        cast to et_block.compute_dtype
                      │
                      ▼
        et_block: float32 descent and energy
                      │
              ┌───────┴───────┐
              ▼               ▼
           tokens            energies
           cast back         stay float32

    Scales:

    .. code-block:: text

        scale   embed_dim  num_heads  head_dim  hopfield_dim
        tiny    192        3          64        768
        small   384        6          64        1536
        base    768        12         64        3072

    head_dim is 64 at every scale, not embed_dim // num_heads.

    :param input_shape: Image shape ``(height, width, channels)``. Defaults to ``(224,224,3)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: Patch size; ``int`` for square patches or ``(h, w)``. Defaults to ``16``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param scale: One of ``'tiny'``, ``'small'``, ``'base'`` (see :data:`SCALE_CONFIGS`), or
        the matching variant key.
    :type scale: ETScale
    :param embed_dim: Override the scale's token dimension ``D``. ``None`` -> from ``scale``.
    :type embed_dim: Optional[int]
    :param num_heads: Override the scale's head count ``H``. ``None`` -> from ``scale``.
    :type num_heads: Optional[int]
    :param head_dim: Override the scale's per-head dim ``Y``. ``None`` -> from ``scale``.
    :type head_dim: Optional[int]
    :param hopfield_dim: Override the scale's memory count ``K``. ``None`` -> from ``scale``.
    :type hopfield_dim: Optional[int]
    :param num_steps: Descent steps ``T``. Backward memory is linear in ``T``. Defaults to 12.
    :type num_steps: int
    :param step_size: Descent step ``alpha``. Defaults to ``0.1``.
    :type step_size: float
    :param beta: Attention inverse temperature; ``None`` -> ``1/sqrt(head_dim)``, resolved by
        ``EnergyAttention``.
    :type beta: Optional[float]
    :param attn_self: ``False`` (default) is the paper's ET-Full: a token does not attend to
        itself.
    :type attn_self: bool
    :param hopfield_activation: ``'relu'`` (default) or ``'softmax'``.
    :type hopfield_activation: HopfieldActivation
    :param hopfield_beta: Temperature of the ``'softmax'`` Hopfield branch, separate from
        ``beta``.
    :type hopfield_beta: float
    :param noise_std: eq.-27 Langevin noise std (training only). ``0.0`` (default) keeps the
        descent guarantee.
    :type noise_std: float
    :param norm_epsilon: ``epsilon`` of the block's inner ``EnergyLayerNorm``.
    :type norm_epsilon: float
    :param pos_dropout_rate: Dropout after the positional embedding, in ``[0, 1]``. Defaults
        to ``0.0``.
    :type pos_dropout_rate: float
    :param return_energy: If ``True``, :meth:`call` returns ``(tokens, energies)`` with
        ``energies`` of shape ``(B, num_steps + 1)`` and dtype float32 even under
        mixed_float16. Used by the out-of-graph energy-trace probe; the training models are
        always built with ``False``.
    :type return_energy: bool
    :param seed: Seed for the ``noise_std`` RNG.
    :type seed: Optional[int]
    :param name: Model name. Defaults to :data:`BACKBONE_NAME`, which the warm-start matches
        on.
    :type name: Optional[str]
    :param **kwargs: Forwarded to :class:`keras.Model`.

    :raises ValueError: If ``input_shape`` or ``patch_size`` has the wrong length, if any
        image or patch dimension is non-positive, if the image dims are not divisible by the
        patch dims, if ``pos_dropout_rate`` is outside ``[0, 1]``, or if ``scale`` is unknown.

    Input shape:
        ``(batch, H, W, C)``; or a 2-tuple ``[(batch, H, W, C), (batch, N)]`` where the second
        entry is the boolean occlusion mask.

    Output shape:
        ``(batch, N, embed_dim)``; or, with ``return_energy=True``, the pair
        ``((batch, N, embed_dim), (batch, num_steps + 1))``.
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            patch_size: Union[int, Tuple[int, int]] = 16,
            scale: ETScale = 'tiny',
            embed_dim: Optional[int] = None,
            num_heads: Optional[int] = None,
            head_dim: Optional[int] = None,
            hopfield_dim: Optional[int] = None,
            num_steps: int = 12,
            step_size: float = 0.1,
            beta: Optional[float] = None,
            attn_self: bool = False,
            hopfield_activation: HopfieldActivation = 'relu',
            hopfield_beta: float = 1.0,
            noise_std: float = 0.0,
            norm_epsilon: float = 1e-5,
            pos_dropout_rate: float = 0.0,
            return_energy: bool = False,
            seed: Optional[int] = None,
            name: Optional[str] = BACKBONE_NAME,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a 3-tuple (height, width, channels), got {input_shape}"
            )
        img_h, img_w, img_c = (int(v) for v in input_shape)
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dims must be positive, got {input_shape}")

        if isinstance(patch_size, int):
            patch_h = patch_w = int(patch_size)
        else:
            if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
                raise ValueError(f"patch_size must be int or a 2-tuple, got {patch_size}")
            patch_h, patch_w = (int(p) for p in patch_size)
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size dims must be positive, got {patch_size}")
        if img_h % patch_h != 0:
            raise ValueError(
                f"Image height ({img_h}) must be divisible by patch height ({patch_h})"
            )
        if img_w % patch_w != 0:
            raise ValueError(
                f"Image width ({img_w}) must be divisible by patch width ({patch_w})"
            )

        scale = _resolve_scale(str(scale))
        cfg = SCALE_CONFIGS[scale]

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(
                f"pos_dropout_rate must be in [0, 1], got {pos_dropout_rate}"
            )

        self.input_shape_config = (img_h, img_w, img_c)
        self.patch_size = (patch_h, patch_w)
        self.scale = scale
        # Resolved here, never None, so get_config round-trips an explicit architecture.
        self.embed_dim = int(embed_dim) if embed_dim is not None else cfg['embed_dim']
        self.num_heads = int(num_heads) if num_heads is not None else cfg['num_heads']
        self.head_dim = int(head_dim) if head_dim is not None else cfg['head_dim']
        self.hopfield_dim = (
            int(hopfield_dim) if hopfield_dim is not None else cfg['hopfield_dim']
        )
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.beta = beta
        self.attn_self = bool(attn_self)
        self.hopfield_activation = str(hopfield_activation)
        self.hopfield_beta = float(hopfield_beta)
        self.noise_std = float(noise_std)
        self.norm_epsilon = float(norm_epsilon)
        self.pos_dropout_rate = float(pos_dropout_rate)
        self.return_energy = bool(return_energy)
        self.seed = seed

        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.patch_dim = patch_h * patch_w * img_c

        self.patch_embed = _apply_dtype_policy(
            create_embedding_layer(
                'patch_2d',
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                name="patch_embed",
            ),
            self.dtype_policy,
        )

        # Created even for the classifier, which never calls it: owning the weight is
        # what keeps the classifier trunk matching the MIM trunk.
        self.mask_token = MaskTokenApply(name="mask_token", dtype=self.dtype_policy)

        self.pos_embed = _apply_dtype_policy(
            create_embedding_layer(
                'positional_learned',
                max_seq_len=self.num_patches,
                dim=self.embed_dim,
                # The registry key is `dropout_rate`; `dropout=` is silently dropped.
                dropout_rate=self.pos_dropout_rate,
                name="pos_embed",
            ),
            self.dtype_policy,
        )

        # DECISION plan-2026-07-14T163315-29a4fef4/D-011: pass the variable dtype, not the
        # policy; EnergyLayerNorm's backward zeroes gradients at fp16. See decisions.md.
        self.et_block = EnergyTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hopfield_dim=self.hopfield_dim,
            num_steps=self.num_steps,
            step_size=self.step_size,
            beta=self.beta,
            attn_self=self.attn_self,
            hopfield_activation=self.hopfield_activation,
            hopfield_beta=self.hopfield_beta,
            noise_std=self.noise_std,
            return_energy=self.return_energy,
            norm_epsilon=self.norm_epsilon,
            seed=self.seed,
            name="et_block",
            dtype=self.dtype_policy.variable_dtype,
        )

        logger.info(
            f"Created EnergyTransformerBackbone-{scale}: {self.embed_dim}d, "
            f"{self.num_heads}h x {self.head_dim}, K={self.hopfield_dim}, "
            f"T={self.num_steps}, alpha={self.step_size}, N={self.num_patches}"
        )

    # -----------------------------------------------------------------

    @staticmethod
    def _split_inputs(
            inputs: Any
    ) -> Tuple[Any, Optional[Any]]:
        """Split ``image`` or ``(image, input_mask)`` into its two parts.

        :param inputs: A tensor, or a 2-sequence of image and mask.
        :type inputs: Any
        :return: ``(image, input_mask)``, with ``None`` for a missing mask.
        :rtype: Tuple[Any, Optional[Any]]
        :raises ValueError: If ``inputs`` is a sequence of length other than 2.
        """
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError(
                    "EnergyTransformerBackbone accepts either `image` or "
                    f"`(image, input_mask)`; got a sequence of length {len(inputs)}"
                )
            return inputs[0], inputs[1]
        return inputs, None

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer from the stored config.

        The shapes come from the config, not from the optional mask entry of
        ``input_shape``, so ``mask_token`` is built the same way whether or not a caller
        ever passes a mask. A sub-layer left to build lazily loses its weights on a
        ``.keras`` round-trip.

        :param input_shape: Ignored for shape purposes; recorded by Keras.
        :type input_shape: Any
        """
        if self.built:
            return

        image_shape = (None,) + self.input_shape_config
        token_shape = (None, self.num_patches, self.embed_dim)
        mask_shape = (None, self.num_patches)

        self.patch_embed.build(image_shape)
        # Built even in the classifier, which never calls it.
        self.mask_token.build([token_shape, mask_shape])
        self.pos_embed.build(token_shape)
        self.et_block.build(token_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Embed the image, apply the mask token if given, and run the descent.

        :param inputs: ``image (B, H, W, C)``, or ``(image, input_mask (B, N) bool)``.
        :param training: Keras training flag.
        :return: ``(B, N, D)`` tokens; or ``((B, N, D), (B, T+1))`` if ``return_energy``.
        """
        image, input_mask = self._split_inputs(inputs)

        x = self.patch_embed(image, training=training)

        # Structural test on whether the caller passed a mask, not on tensor values;
        # the layer stays built either way.
        if input_mask is not None:
            x = self.mask_token([x, input_mask])

        x = self.pos_embed(x, training=training)

        # DECISION plan-2026-07-14T163315-29a4fef4/D-011: keep both casts; the block runs at
        # its variable dtype because its norm backward overflows fp16. See decisions.md.
        x = keras.ops.cast(x, self.et_block.compute_dtype)
        outputs = self.et_block(x, training=training)
        if self.return_energy:
            # Energies stay float32; casting them down overflows to nan.
            tokens, energies = outputs
            return keras.ops.cast(tokens, self.compute_dtype), energies
        return keras.ops.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the output shape from the stored config, valid before the model is built.

        :param input_shape: An image shape, or a 2-sequence of image and mask shapes.
        :type input_shape: Any
        :return: The token shape, or the token and energy shapes with ``return_energy``.
        :rtype: Any
        """
        image_shape = (
            input_shape[0]
            if (isinstance(input_shape, (tuple, list))
                and len(input_shape) > 0
                and isinstance(input_shape[0], (tuple, list)))
            else input_shape
        )
        batch = image_shape[0] if len(image_shape) == 4 else None

        token_shape = (batch, self.num_patches, self.embed_dim)
        if self.return_energy:
            return token_shape, (batch, self.num_steps + 1)
        return token_shape

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, with the scale overrides resolved.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "patch_size": self.patch_size,
            "scale": self.scale,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "hopfield_dim": self.hopfield_dim,
            "num_steps": self.num_steps,
            "step_size": self.step_size,
            "beta": self.beta,
            "attn_self": self.attn_self,
            "hopfield_activation": self.hopfield_activation,
            "hopfield_beta": self.hopfield_beta,
            "noise_std": self.noise_std,
            "norm_epsilon": self.norm_epsilon,
            "pos_dropout_rate": self.pos_dropout_rate,
            "return_energy": self.return_energy,
            "seed": self.seed,
        })
        return config


# ---------------------------------------------------------------------


def _coerce_backbone(backbone: Any) -> EnergyTransformerBackbone:
    """Accept a live backbone or its serialized config dict.

    :param backbone: A backbone instance or the dict ``from_config`` receives.
    :type backbone: Any
    :return: The backbone.
    :rtype: EnergyTransformerBackbone
    :raises TypeError: If ``backbone`` is neither, or deserializes to another type.
    """
    if isinstance(backbone, EnergyTransformerBackbone):
        return backbone
    if isinstance(backbone, dict):
        obj = deserialize_keras_object(backbone)
        if not isinstance(obj, EnergyTransformerBackbone):
            raise TypeError(
                f"Deserialized backbone is a {type(obj).__name__}, expected "
                "EnergyTransformerBackbone"
            )
        return obj
    raise TypeError(
        "backbone must be an EnergyTransformerBackbone (or its serialized config dict), "
        f"got {type(backbone).__name__}"
    )


# DECISION plan-2026-07-14T163315-29a4fef4/D-010: both heads refuse a return_energy=True
# backbone; the float32 trace autocast down in a fp16 head goes to nan. See decisions.md.
def _reject_energy_backbone(backbone: EnergyTransformerBackbone, owner: str) -> None:
    if backbone.return_energy:
        raise ValueError(
            f"{owner} requires a backbone with return_energy=False (got True). The energy "
            "trace is float32 by design and must never reach a (possibly fp16) head; read it "
            "out-of-graph with a probe backbone instead."
        )


@register_dl_technique("dl_techniques.models.energy_transformer.model")
class EnergyTransformerMIM(keras.Model):
    """Reconstruct patch pixels from the energy-descended tokens.

    The paper's §3 image model. Every token is decoded back to raw normalized patch
    pixels; the loss is narrowed to the occluded set by the ``sample_weight`` carried in
    the ``tf.data`` batch, so nothing in this class touches the loss. The decoder is a
    single affine projection, which leaves the reconstruction quality to the energy
    descent in the trunk. Head sub-layers all carry a ``decoder_`` prefix, so the
    classifier's warm-start skips exactly them and transfers the rest.

    Architecture:

    .. code-block:: text

        image [B, H, W, C] (+ mask [B, N])
                 │
                 ▼
        ┌─────────────────────┐
        │ et_backbone         │
        └─────────────────────┘
                 │  [B, N, D]
                 ▼
        ┌─────────────────────┐
        │ decoder_norm        │  layernorm, eps 1e-6
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ decoder_proj        │  dense, no activation
        └─────────────────────┘
                 │
                 ▼
           [B, N, P*P*C]

    :param backbone: An :class:`EnergyTransformerBackbone` (must be named ``"et_backbone"``
        for the warm-start to match by name), or its serialized config dict.
    :type backbone: EnergyTransformerBackbone
    :param name: Model name.
    :type name: Optional[str]
    :param **kwargs: Standard ``keras.Model`` kwargs.

    :raises ValueError: If ``backbone.return_energy`` is ``True`` (see D-010).
    :raises TypeError: If ``backbone`` is neither a backbone nor a config dict for one.

    Input shape:
        ``[(batch, H, W, C), (batch, N) bool]`` — image + occlusion mask. A bare
        ``(batch, H, W, C)`` image is also accepted (no tokens are replaced).

    Output shape:
        ``(batch, N, patch_size_h * patch_size_w * channels)``.
    """

    def __init__(
            self,
            backbone: EnergyTransformerBackbone,
            name: Optional[str] = "energy_transformer_mim",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        backbone = _coerce_backbone(backbone)
        _reject_energy_backbone(backbone, "EnergyTransformerMIM")

        self.backbone = backbone
        self.patch_dim = backbone.patch_dim
        self.num_patches = backbone.num_patches
        self.embed_dim = backbone.embed_dim

        self.decoder_norm = layers.LayerNormalization(
            epsilon=1e-6, name="decoder_norm", dtype=self.dtype_policy
        )
        self.decoder_proj = layers.Dense(
            self.patch_dim, name="decoder_proj", dtype=self.dtype_policy
        )

    def build(self, input_shape: Any) -> None:
        """Build the backbone and both decoder layers.

        :param input_shape: Forwarded to the backbone; head shapes come from its config.
        :type input_shape: Any
        """
        if self.built:
            return
        self.backbone.build(input_shape)
        token_shape = (None, self.num_patches, self.embed_dim)
        self.decoder_norm.build(token_shape)
        self.decoder_proj.build(token_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the trunk, then decode every token to patch pixels.

        :param inputs: ``image (B, H, W, C)``, or ``(image, input_mask (B, N) bool)``.
        :param training: Keras training flag.
        :return: ``(B, N, patch_dim)`` reconstructed patches.
        """
        tokens = self.backbone(inputs, training=training)
        x = self.decoder_norm(tokens, training=training)
        return self.decoder_proj(x, training=training)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(batch, num_patches, patch_dim)``.

        :param input_shape: An image shape, or image and mask shapes.
        :type input_shape: Any
        :return: The reconstruction shape.
        :rtype: Tuple[Optional[int], ...]
        """
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_patches, self.patch_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the config, with the backbone serialized inline.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"backbone": serialize_keras_object(self.backbone)})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyTransformerMIM":
        """Rebuild from :meth:`get_config`, deserializing the backbone first.

        :param config: The dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: The rebuilt model.
        :rtype: EnergyTransformerMIM
        """
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.energy_transformer.model")
class EnergyTransformerClassifier(keras.Model):
    """Classify an image from the mean of its energy-descended tokens.

    Shows that the MIM-pretrained trunk transfers: the backbone is composed under the
    same name and the same config path, so
    ``load_weights_from_checkpoint(model, mim_ckpt, skip_prefixes=("decoder_",))`` moves
    the whole trunk and nothing else. Pooling is a mean over tokens because this
    backbone has no CLS token; adding one would make ``N = 197`` here against ``196`` in
    the MIM model, change the positional-embedding table's shape and break that transfer
    (D-004). The head emits logits, so compile with
    ``SparseCategoricalCrossentropy(from_logits=True)``.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]
                 │
                 ▼
        ┌─────────────────────┐
        │ et_backbone         │
        └─────────────────────┘
                 │  [B, N, D]
                 ▼
        ┌─────────────────────┐
        │ head_norm           │  layernorm, eps 1e-6
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ head_pool           │  mean over tokens
        └─────────────────────┘
                 │  [B, D]
                 ▼
        ┌─────────────────────┐
        │ head_dropout        │  (rate may be 0.0)
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ head_dense          │
        └─────────────────────┘
                 │
                 ▼
           [B, num_classes] logits

    :param backbone: An :class:`EnergyTransformerBackbone` (named ``"et_backbone"``), or its
        serialized config dict.
    :type backbone: EnergyTransformerBackbone
    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param dropout_rate: Dropout before the final Dense, in ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param name: Model name.
    :type name: Optional[str]
    :param **kwargs: Standard ``keras.Model`` kwargs.

    :raises ValueError: If ``num_classes <= 0``, ``dropout_rate`` is outside ``[0, 1]``, or
        ``backbone.return_energy`` is ``True``.
    :raises TypeError: If ``backbone`` is neither a backbone nor a config dict for one.

    Input shape:
        ``(batch, H, W, C)``. (A ``(image, mask)`` pair is accepted but the classifier is not
        meant to be fed one.)

    Output shape:
        ``(batch, num_classes)`` — logits.
    """

    def __init__(
            self,
            backbone: EnergyTransformerBackbone,
            num_classes: int,
            dropout_rate: float = 0.0,
            name: Optional[str] = "energy_transformer_classifier",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        backbone = _coerce_backbone(backbone)
        _reject_energy_backbone(backbone, "EnergyTransformerClassifier")

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)
        self.num_patches = backbone.num_patches
        self.embed_dim = backbone.embed_dim

        # `head_` prefix, distinct from `decoder_`, so these are never transferred.
        self.head_norm = layers.LayerNormalization(
            epsilon=1e-6, name="head_norm", dtype=self.dtype_policy
        )
        self.head_pool = layers.GlobalAveragePooling1D(
            name="head_pool", dtype=self.dtype_policy
        )
        # Created at every rate, so the layer structure does not depend on a number.
        self.head_dropout = layers.Dropout(
            self.dropout_rate, name="head_dropout", dtype=self.dtype_policy
        )
        self.head_dense = layers.Dense(
            self.num_classes, name="head_dense", dtype=self.dtype_policy
        )

    def build(self, input_shape: Any) -> None:
        """Build the backbone and the four head layers.

        :param input_shape: Forwarded to the backbone; head shapes come from its config.
        :type input_shape: Any
        """
        if self.built:
            return
        self.backbone.build(input_shape)
        token_shape = (None, self.num_patches, self.embed_dim)
        pooled_shape = (None, self.embed_dim)
        self.head_norm.build(token_shape)
        self.head_pool.build(token_shape)
        self.head_dropout.build(pooled_shape)
        self.head_dense.build(pooled_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the trunk, pool the tokens, and return class logits.

        :param inputs: ``image (B, H, W, C)``.
        :param training: Keras training flag.
        :return: ``(B, num_classes)`` logits, with no softmax applied.
        """
        tokens = self.backbone(inputs, training=training)
        x = self.head_norm(tokens, training=training)
        x = self.head_pool(x)
        x = self.head_dropout(x, training=training)
        return self.head_dense(x)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(batch, num_classes)``.

        :param input_shape: An image shape, or image and mask shapes.
        :type input_shape: Any
        :return: The logits shape.
        :rtype: Tuple[Optional[int], ...]
        """
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """Return the config, with the backbone serialized inline.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyTransformerClassifier":
        """Rebuild from :meth:`get_config`, deserializing the backbone first.

        :param config: The dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: The rebuilt model.
        :rtype: EnergyTransformerClassifier
        """
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------


def create_energy_transformer_backbone(
        variant: str = 'tiny',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        **overrides: Any
) -> EnergyTransformerBackbone:
    """Create a standalone :class:`EnergyTransformerBackbone`.

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` (or ``'et_tiny'`` ...).
    :param input_shape: ``(H, W, C)``.
    :param patch_size: ``int`` or ``(h, w)``.
    :param **overrides: Any :class:`EnergyTransformerBackbone` ctor kwarg (e.g. ``num_steps``,
        ``return_energy``, ``noise_std``).
    :return: The backbone, named ``"et_backbone"``.
    :raises ValueError: If ``variant`` is unknown, or the geometry does not divide.
    """
    return EnergyTransformerBackbone(
        input_shape=input_shape,
        patch_size=patch_size,
        scale=_resolve_scale(variant),
        name=BACKBONE_NAME,
        **overrides,
    )


def create_energy_transformer_mim(
        variant: str = 'tiny',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        **overrides: Any
) -> EnergyTransformerMIM:
    """Create the masked-image-completion model.

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` (or ``'et_tiny'`` ...).
    :param input_shape: ``(H, W, C)``.
    :param patch_size: ``int`` or ``(h, w)``.
    :param **overrides: Backbone ctor kwargs (``num_steps``, ``step_size``, ``noise_std``, ...).
    :return: An :class:`EnergyTransformerMIM` whose trunk is named ``"et_backbone"``.
    :raises ValueError: If ``variant`` is unknown, the geometry does not divide, or
        ``return_energy=True`` is passed through ``overrides``.

    Example:
        >>> model = create_energy_transformer_mim('tiny', (224, 224, 3), 16)
        >>> model.compile(optimizer='adamw', loss='mse')   # sample_weight does the masking
    """
    backbone = create_energy_transformer_backbone(
        variant=variant,
        input_shape=input_shape,
        patch_size=patch_size,
        **overrides,
    )
    return EnergyTransformerMIM(backbone=backbone)


def create_energy_transformer_classifier(
        variant: str = 'tiny',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        **overrides: Any
) -> EnergyTransformerClassifier:
    """Create the classifier (logits head; warm-startable from an MIM checkpoint).

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` (or ``'et_tiny'`` ...).
    :param input_shape: ``(H, W, C)``.
    :param patch_size: ``int`` or ``(h, w)``.
    :param num_classes: Number of classes.
    :param dropout_rate: Dropout before the final Dense.
    :param **overrides: Backbone ctor kwargs.
    :return: An :class:`EnergyTransformerClassifier` whose trunk is named ``"et_backbone"``
        and is weight-identical to :func:`create_energy_transformer_mim`'s at the same config.
    :raises ValueError: If ``variant`` is unknown, the geometry does not divide,
        ``num_classes`` is not positive, or ``return_energy=True`` is passed through
        ``overrides``.

    Example:
        >>> model = create_energy_transformer_classifier('tiny', (224, 224, 3), 16, 10)
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )
    """
    backbone = create_energy_transformer_backbone(
        variant=variant,
        input_shape=input_shape,
        patch_size=patch_size,
        **overrides,
    )
    return EnergyTransformerClassifier(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


# ---------------------------------------------------------------------