"""
Energy Transformer image models — backbone, masked-image-completion (MIM) and classifier.

This module gives the ``EnergyTransformer`` block (``layers/transformers/energy_transformer.py``,
arXiv:2302.07253) its two image-domain consumers:

* :class:`EnergyTransformerBackbone` — patch-embed -> (optional learnable MASK token) ->
  learnable positional embedding -> ONE ``EnergyTransformer`` block running ``T`` internal
  descent steps -> ``(B, N, D)`` token states.
* :class:`EnergyTransformerMIM` — backbone -> LayerNorm -> affine ``Dense(P*P*C)`` decoder ->
  ``(B, N, P*P*C)``, the paper's §3 masked-image-completion model.
* :class:`EnergyTransformerClassifier` — the SAME backbone -> LayerNorm -> mean-pool ->
  ``Dense(num_classes)`` **logits**, warm-startable from an MIM checkpoint.

**Architecture**::

    image (B, H, W, C)              input_mask (B, N) bool   [MIM only]
          |                                |
    PatchEmbedding2D  ---------------------+
          |                                |
          v                                v
    (B, N, D)  ----------------->  MaskTokenApply   (skipped when no mask is passed,
          |                                |         but ALWAYS created AND built)
          +--------------------------------+
                          |
                  PositionalEmbedding (learned)
                          |
                  EnergyTransformer  (T descent steps on ONE scalar energy)
                          |
                    (B, N, D) tokens
                    /              \\
        decoder_norm                head_norm
        decoder_proj -> (B,N,P*P*C) head_pool -> head_dense -> (B, num_classes) logits

**THE DATA CONTRACT (do not "simplify" it).** The MIM model is trained through STOCK
``model.compile(loss='mse')`` + ``model.fit(ds)``. There is no ``train_step``, no
``test_step`` and no ``compute_loss`` anywhere in this file, and none may be added. The
occlusion mask reaches the LOSS as a Keras ``sample_weight``, supplied as the third element
of each ``tf.data`` batch (built by ``datasets/vision/masked_patches.py``)::

    ((image (B,H,W,C), input_mask (B,N) bool), target_patches (B,N,P*P*C), loss_weight (B,N))

with ``loss_weight = 1{i in S} * (N / n_loss)``, which makes Keras' ``sum_over_batch_size``
reduction equal ``mean_{i in S} MSE`` exactly. ``input_mask`` is a STRICT SUBSET of the loss
set ``S`` (the paper's 90/10 rule: ~10% of the loss tokens keep their true patch embedding),
so the two masks are NOT interchangeable.

**Weight-compatibility invariant.** ``MaskTokenApply`` is created AND built by EVERY backbone,
including the classifier's, which never calls it. This is the authoring guide's §9 "ALWAYS
CREATE / CONDITIONALLY USE" rule and it is what keeps the MIM trunk and the classifier trunk
weight-identical, so ``load_weights_from_checkpoint(..., skip_prefixes=("decoder_",))``
transfers the trunk 1:1. Removing the "dead" mask token from the classifier would silently
break the warm-start.

References:
    - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253 (§3, Table 4).
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

# The ET block has NO factory home — direct import is the sanctioned path for this
# feature (D-004 of plan_2026-07-13_57c9833e; G2). Do not route it through a factory.
from dl_techniques.layers.transformers.energy_transformer import EnergyTransformer

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

ETScale = Literal['tiny', 'small', 'base']
HopfieldActivation = Literal['relu', 'softmax']

# The stable sub-model name. `load_weights_from_checkpoint` matches layer BY NAME, so the
# MIM model and the classifier MUST name their backbone identically or the warm-start
# transfers zero layers.
BACKBONE_NAME = "et_backbone"

# Scale configurations. All variants share the paper's Table-4 image defaults:
# num_steps (T) = 12, step_size (alpha) = 0.1, beta = None (-> 1/sqrt(head_dim)),
# attn_self = False (ET-Full), hopfield_activation = 'relu'. Note head_dim is FIXED at 64
# across scales (it is NOT embed_dim // num_heads) — the ET attention has no value matrix,
# so the head dimension is a free parameter.
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
    """Accept either a scale key (``'tiny'``) or a variant key (``'et_tiny'``)."""
    if variant in SCALE_CONFIGS:
        return variant
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]['scale']
    raise ValueError(
        f"Unknown variant '{variant}'. Available: "
        f"{sorted(SCALE_CONFIGS)} or {sorted(MODEL_VARIANTS)}"
    )


# DECISION plan-2026-07-14T163315-29a4fef4/D-009
# `create_embedding_layer()` SILENTLY DROPS a `dtype=` kwarg. Its registry filters the call
# down to `required_params | optional_params` (embedding/factory.py:350-351) and `dtype` is
# in NEITHER, so the obvious-looking
#     create_embedding_layer('patch_2d', ..., dtype=self.dtype_policy)
# is a NO-OP: the layer silently keeps the GLOBAL policy. Executed control (keras 3.8):
# passing `dtype='float64'` yields a layer whose `dtype_policy` is still `float32`. That
# breaks H4 — under `EnergyTransformerBackbone(dtype='float64')` (or any explicit non-global
# policy) the patch/positional embeddings would compute in float32 while the ET block computes
# in float64, and the add would die with an InvalidArgumentError; the mirror-image failure
# under `mixed_float16` is the one that already bit this feature once.
# Keras' `Layer.dtype_policy` SETTER does not recurse either, so setting it on the returned
# layer alone leaves the inner Conv2D / Dropout at the global policy. Hence this helper walks
# the layer tree. `layers/` is frozen for this plan (I8), so the factory cannot be fixed here.
# WHAT NOT TO DO: do NOT pass `dtype=` to `create_embedding_layer` and assume it landed, and
# do NOT "simplify" this to `layer.dtype_policy = policy` — see decisions.md D-009.
def _apply_dtype_policy(layer: keras.layers.Layer, policy: Any) -> keras.layers.Layer:
    """Force ``policy`` onto ``layer`` AND every sub-layer, before anything is built."""
    if hasattr(layer, "_flatten_layers"):
        for sub in layer._flatten_layers(include_self=True):
            sub.dtype_policy = policy
    else:  # pragma: no cover - defensive, keras always provides _flatten_layers
        layer.dtype_policy = policy
    return layer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EnergyTransformerBackbone(keras.Model):
    """Shared Energy Transformer trunk: patch-embed -> [mask token] -> pos-embed -> ET block.

    **Intent**: give the ``EnergyTransformer`` block a single, separately-checkpointable image
    trunk that BOTH the masked-completion model and the classifier compose under the same name,
    so a pretrained encoder transfers into the classifier layer-for-layer.

    **Call signature** — ``backbone(image)`` OR ``backbone((image, input_mask))``. Passing the
    mask is a TRACE-TIME structural choice (a Python ``if`` on whether the caller supplied a
    second tensor), not a runtime ``ops.where`` on tensor values: the MIM model always passes
    one, the classifier never does.

    **``MaskTokenApply`` is ALWAYS created and ALWAYS built**, even for the classifier that
    never calls it. That is deliberate (authoring guide §9): it is what makes the two trunks
    weight-identical so the warm-start is complete. Do not "optimize" it away.

    **Descent sign**: the ET block's ``update()`` returns ``-dE/dg`` and the block ADDS it.
    Nothing here re-derives or re-signs the descent — see the block's class docstring.

    :param input_shape: Image shape ``(height, width, channels)``. Defaults to ``(224,224,3)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: Patch size; ``int`` for square patches or ``(h, w)``. Defaults to ``16``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param scale: One of ``'tiny'``, ``'small'``, ``'base'`` (see :data:`SCALE_CONFIGS`).
    :type scale: ETScale
    :param embed_dim: Override the scale's token dimension ``D``. ``None`` -> from ``scale``.
    :type embed_dim: Optional[int]
    :param num_heads: Override the scale's head count ``H``. ``None`` -> from ``scale``.
    :type num_heads: Optional[int]
    :param head_dim: Override the scale's per-head dim ``Y``. ``None`` -> from ``scale``.
    :type head_dim: Optional[int]
    :param hopfield_dim: Override the scale's memory count ``K``. ``None`` -> from ``scale``.
    :type hopfield_dim: Optional[int]
    :param num_steps: Descent steps ``T``. Backward memory is LINEAR in ``T``. Defaults to 12.
    :type num_steps: int
    :param step_size: Descent step ``alpha``. Defaults to ``0.1``.
    :type step_size: float
    :param beta: Attention inverse temperature; ``None`` -> ``1/sqrt(head_dim)`` (resolved by
        ``EnergyAttention``, not duplicated here).
    :type beta: Optional[float]
    :param attn_self: ``False`` (default) is the paper's ET-Full: a token does not attend to
        itself.
    :type attn_self: bool
    :param hopfield_activation: ``'relu'`` (default) or ``'softmax'``.
    :type hopfield_activation: HopfieldActivation
    :param hopfield_beta: Temperature of the ``'softmax'`` Hopfield branch. NOT ``beta``.
    :type hopfield_beta: float
    :param noise_std: eq.-27 Langevin noise std (training only). ``0.0`` (default) keeps the
        descent guarantee.
    :type noise_std: float
    :param norm_epsilon: ``epsilon`` of the block's inner ``EnergyLayerNorm``.
    :type norm_epsilon: float
    :param pos_dropout_rate: Dropout after the positional embedding. Defaults to ``0.0``.
    :type pos_dropout_rate: float
    :param return_energy: If ``True``, :meth:`call` returns ``(tokens, energies)`` with
        ``energies`` of shape ``(B, num_steps + 1)`` and dtype **float32 even under
        mixed_float16**. Used by the out-of-graph energy-trace probe; the TRAINING models are
        always built with ``False``.
    :type return_energy: bool
    :param seed: Seed for the ``noise_std`` RNG.
    :type seed: Optional[int]

    :raises ValueError: If the image dims are not divisible by the patch dims, or any
        dimension is non-positive, or ``scale`` is unknown.

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

        # ----- validate the image / patch geometry -----
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

        # ----- store ALL configuration (serialization contract) -----
        self.input_shape_config = (img_h, img_w, img_c)
        self.patch_size = (patch_h, patch_w)
        self.scale = scale
        # Resolved (never None) so get_config round-trips an explicit architecture.
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

        # ----- derived -----
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.patch_dim = patch_h * patch_w * img_c

        # ----- CREATE all sub-layers in __init__ (unbuilt) -----
        # `dtype=` is applied via _apply_dtype_policy, NOT through the factory (D-009).
        self.patch_embed = _apply_dtype_policy(
            create_embedding_layer(
                'patch_2d',
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                name="patch_embed",
            ),
            self.dtype_policy,
        )

        # I6 / guide §9: ALWAYS CREATE, CONDITIONALLY USE. The classifier never calls this,
        # but it MUST own the weight or its trunk stops matching the MIM trunk.
        self.mask_token = MaskTokenApply(name="mask_token", dtype=self.dtype_policy)

        self.pos_embed = _apply_dtype_policy(
            create_embedding_layer(
                'positional_learned',
                max_seq_len=self.num_patches,
                dim=self.embed_dim,
                # NOTE: the registry key is `dropout_rate`. `dropout=` is silently dropped.
                dropout_rate=self.pos_dropout_rate,
                name="pos_embed",
            ),
            self.dtype_policy,
        )

        # DECISION plan-2026-07-14T163315-29a4fef4/D-011
        # `dtype=self.dtype_policy.variable_dtype`, NOT `self.dtype_policy`. Under
        # `mixed_float16` this runs the ET block in float32 (its variable dtype) rather than
        # float16; `call()` casts the tokens in and back out. Under float32/float64 the two
        # spellings are IDENTICAL (compute == variable), so nothing outside a mixed policy
        # changes, and the block's variables were float32 under either spelling — the weight
        # count, the weight dtypes and every checkpoint are untouched.
        #
        # WHY (executed control, N=196, tiny/small/base, XLA): `EnergyLayerNorm`'s BACKWARD
        # forms `(var + eps)^(-3/2)`. In fp16 that intermediate OVERFLOWS `65504` whenever
        # `eps < 65504^(-2/3) ~ 6.1e-4` — at the default `eps = 1e-5` it is 3.2e7 -> `inf`,
        # and `0 * inf` -> `NaN`. The occlusion mask is what supplies the near-constant
        # (`var ~ 0`) tokens that reach the cliff, and XLA is what keeps the intermediate in
        # fp16 (it is finite eagerly and at `jit_compile=False`) — so the bug needs the TRIPLE
        # fp16 x mask x XLA, and `fit` turns XLA on BY DEFAULT.
        # The failure is SILENT: the loss stays FINITE, `TerminateOnNaN` never fires, the
        # energy trace still descends, `LossScaleOptimizer` just rejects 100% of steps (dynamic
        # scale 2^15 -> 2.98e-08), every weight moves by EXACTLY 0.0, and the user ships a
        # random-init checkpoint on top of a plausible flat loss curve.
        #
        # WHAT NOT TO DO: (1) do NOT "simplify" this back to `dtype=self.dtype_policy`, and do
        # not drop the casts in `call()`. (2) Do NOT "fix" it instead by raising `norm_epsilon`
        # to 1e-3 (which does clear the overflow): that silently makes the fp16 model train a
        # DIFFERENT network than the fp32 one (the norm's Jacobian ceiling is `gamma/sqrt(eps)`,
        # so 1e-5 -> 1e-3 cuts it 10x) and it sits 2x from the cliff. (3) Casting around
        # `MaskTokenApply` does NOT work — `where` is a select, so an fp32 round-trip there is
        # forward-identical and the NaN is manufactured DOWNSTREAM, in the block's backward.
        # Guarded by `test_model.py::TestMixedPrecisionBackwardPass` (a BACKWARD-pass test — the
        # forward-only fp16 test passed throughout). See decisions.md D-011.
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
        """Split ``image`` / ``(image, input_mask)``. Trace-time structural, not value-based."""
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError(
                    "EnergyTransformerBackbone accepts either `image` or "
                    f"`(image, input_mask)`; got a sequence of length {len(inputs)}"
                )
            return inputs[0], inputs[1]
        return inputs, None

    def build(self, input_shape: Any) -> None:
        """Explicitly build EVERY sub-layer from stored config.

        The shapes come from the CONFIG, never from ``input_shape``'s optional mask entry, so
        ``mask_token`` is built identically whether or not the caller ever passes a mask
        (I6). A lazily-built sub-layer silently drops its weights on a ``.keras`` round-trip.
        """
        if self.built:
            return

        image_shape = (None,) + self.input_shape_config
        token_shape = (None, self.num_patches, self.embed_dim)
        mask_shape = (None, self.num_patches)

        self.patch_embed.build(image_shape)
        # ALWAYS built — even in the classifier, which never calls it.
        self.mask_token.build([token_shape, mask_shape])
        self.pos_embed.build(token_shape)
        self.et_block.build(token_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Forward pass.

        :param inputs: ``image (B, H, W, C)``, or ``(image, input_mask (B, N) bool)``.
        :param training: Keras training flag.
        :return: ``(B, N, D)`` tokens; or ``((B, N, D), (B, T+1))`` if ``return_energy``.
        """
        image, input_mask = self._split_inputs(inputs)

        x = self.patch_embed(image, training=training)

        # Python `if` on a TRACE-TIME structural fact (did the caller pass a mask?), which is
        # the sanctioned "ALWAYS CREATE / CONDITIONALLY USE" pattern — NOT a Python `if` on a
        # tensor VALUE. The layer stays built either way.
        if input_mask is not None:
            x = self.mask_token([x, input_mask])

        x = self.pos_embed(x, training=training)

        # D-011: the block computes in its VARIABLE dtype (float32 under mixed_float16), never
        # in fp16 — its EnergyLayerNorm backward overflows fp16 under XLA and silently kills
        # training. Both casts are no-ops under float32/float64. DO NOT REMOVE THEM.
        x = keras.ops.cast(x, self.et_block.compute_dtype)
        outputs = self.et_block(x, training=training)
        if self.return_energy:
            tokens, energies = outputs      # energies stay float32 (I5) — never cast down
            return keras.ops.cast(tokens, self.compute_dtype), energies
        return keras.ops.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Output shape from stored config — valid UNBUILT."""
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
    """Accept a live backbone or its serialized config dict (the ``from_config`` path)."""
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


# DECISION plan-2026-07-14T163315-29a4fef4/D-010
# Both heads REFUSE a `return_energy=True` backbone instead of passing the trace through.
# This enforces I5 STRUCTURALLY rather than by convention. `EnergyTransformer.energy()` is
# >= float32 ALWAYS (an O(-1e5) trace is `-inf` in fp16 at a realistic N), so under
# `mixed_float16` a default-policy head that ingests the trace autocasts it DOWN to fp16 and
# overflows to nan/inf. That mechanism has already been falsified-and-re-derived once on this
# feature, and it is NOT fixable inside the block — the fix is consumer-side.
# WHAT NOT TO DO: do not "helpfully" forward or ignore the energies here, and do not make the
# model dict-output so the trace can ride along. The trace is read OUT OF GRAPH by
# `EnergyTraceCallback`, which rebuilds a probe BACKBONE (not a head) with
# `return_energy=True`. See decisions.md D-010.
def _reject_energy_backbone(backbone: EnergyTransformerBackbone, owner: str) -> None:
    if backbone.return_energy:
        raise ValueError(
            f"{owner} requires a backbone with return_energy=False (got True). The energy "
            "trace is float32 by design and must never reach a (possibly fp16) head; read it "
            "out-of-graph with a probe backbone instead."
        )


@keras.saving.register_keras_serializable()
class EnergyTransformerMIM(keras.Model):
    """Masked-image-completion model: ET backbone -> LayerNorm -> affine ``Dense(P*P*C)``.

    **Intent**: the paper's §3 image model. Reconstructs raw (normalized) patch pixels for
    EVERY token; the loss is restricted to the occluded set by the ``sample_weight`` carried
    in the ``tf.data`` batch, NOT by anything in this class (H6: no ``train_step``).

    The decoder is a SINGLE affine projection on purpose — the reconstruction quality is
    supposed to come from the energy descent in the trunk, not from a deep decoder.

    All head sub-layers are named with a ``decoder_`` prefix so the classifier's warm-start
    (``skip_prefixes=("decoder_",)``) skips exactly them and transfers the rest.

    :param backbone: An :class:`EnergyTransformerBackbone` (must be named ``"et_backbone"``
        for the warm-start to match by name), or its serialized config dict.
    :type backbone: EnergyTransformerBackbone
    :param kwargs: Standard ``keras.Model`` kwargs.

    :raises ValueError: If ``backbone.return_energy`` is ``True`` (see D-010).

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
        tokens = self.backbone(inputs, training=training)
        x = self.decoder_norm(tokens, training=training)
        return self.decoder_proj(x, training=training)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_patches, self.patch_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"backbone": serialize_keras_object(self.backbone)})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyTransformerMIM":
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EnergyTransformerClassifier(keras.Model):
    """Classifier: the SAME ET backbone -> LayerNorm -> mean-pool -> ``Dense(num_classes)``.

    **Intent**: demonstrate that the MIM-pretrained trunk transfers. The backbone is composed
    under the identical name (``"et_backbone"``) and identical config path, so
    ``load_weights_from_checkpoint(model, mim_ckpt, skip_prefixes=("decoder_",))`` moves the
    whole trunk and nothing else.

    **Mean-pool, not a CLS token** — the ET block has no CLS concept, and a CLS token would
    make ``N = 197`` here versus ``196`` in the MIM model, changing the positional-embedding
    table's shape and BREAKING the very transfer this model exists to show (D-004).

    **The head emits LOGITS** (no softmax). Compile with
    ``SparseCategoricalCrossentropy(from_logits=True)`` — the house convention.

    :param backbone: An :class:`EnergyTransformerBackbone` (named ``"et_backbone"``), or its
        serialized config dict.
    :type backbone: EnergyTransformerBackbone
    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param dropout_rate: Dropout before the final Dense. Defaults to ``0.0``.
    :type dropout_rate: float

    :raises ValueError: If ``num_classes <= 0`` or ``backbone.return_energy`` is ``True``.

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

        # `head_` prefix: distinct from `decoder_`, so it is never transferred.
        self.head_norm = layers.LayerNormalization(
            epsilon=1e-6, name="head_norm", dtype=self.dtype_policy
        )
        self.head_pool = layers.GlobalAveragePooling1D(
            name="head_pool", dtype=self.dtype_policy
        )
        # ALWAYS CREATE / CONDITIONALLY USE (guide §9): the Dropout exists at every rate so
        # the layer structure does not depend on a numeric value.
        self.head_dropout = layers.Dropout(
            self.dropout_rate, name="head_dropout", dtype=self.dtype_policy
        )
        self.head_dense = layers.Dense(
            self.num_classes, name="head_dense", dtype=self.dtype_policy
        )

    def build(self, input_shape: Any) -> None:
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
        tokens = self.backbone(inputs, training=training)
        x = self.head_norm(tokens, training=training)
        x = self.head_pool(x)
        x = self.head_dropout(x, training=training)
        return self.head_dense(x)  # logits — no softmax

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyTransformerClassifier":
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
    :param overrides: Any :class:`EnergyTransformerBackbone` ctor kwarg (e.g. ``num_steps``,
        ``return_energy``, ``noise_std``).
    :return: The backbone, named ``"et_backbone"``.
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
    :param overrides: Backbone ctor kwargs (``num_steps``, ``step_size``, ``noise_std``, ...).
    :return: An :class:`EnergyTransformerMIM` whose trunk is named ``"et_backbone"``.

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
    :param overrides: Backbone ctor kwargs.
    :return: An :class:`EnergyTransformerClassifier` whose trunk is named ``"et_backbone"``
        and is weight-identical to :func:`create_energy_transformer_mim`'s at the same config.

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
