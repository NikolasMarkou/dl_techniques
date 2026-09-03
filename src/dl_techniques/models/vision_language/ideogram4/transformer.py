"""Ideogram4 flow-matching diffusion transformer.

Defines :class:`Ideogram4Transformer`, which packs text-conditioning tokens
and image latent tokens into one self-attention stream and regresses a
velocity field for Euler integration, using 3D multi-axis rotary positions
and a tanh-gated AdaLN block.

There is no cross-attention. Text and image tokens share one sequence; an
integer `indicator` marks each token's role, and float masks derived from it
gate each stream both before and after its projection, so the two streams'
biases stay separated when they are added into one hidden state. Attention
is block-diagonal over `segment_ids` rather than causal, since a diffusion
transformer must see the whole latent and caption at once. Position is 3D:
`position_ids` carries `(t, h, w)`, and mRoPE splits the rotary frequency
budget into three bands sized by `mrope_section`.

Conditioning is precomputed `llm_features`, taken as a call input, since
Qwen3-VL has no Keras equivalent; this is not an end-to-end text-to-image
model on its own. The architecture is faithful in math to the reference but
not weight-compatible with the released checkpoint: weights here are always
randomly initialized. `t = 0` is clean data and `t = 1` is pure noise; see
`pipeline.py` for how a sampler must walk that convention. The velocity head
always returns float32, even under a mixed-precision policy.

References:
    - Lipman et al., 2023. Flow Matching for Generative Modeling.
      (https://arxiv.org/abs/2210.02747)
    - Liu et al., 2022. Flow Straight and Fast: Learning to Generate and Transfer
      Data with Rectified Flow. (https://arxiv.org/abs/2209.03003)
    - Peebles & Xie, 2023. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748)
    - Esser et al., 2024. Scaling Rectified Flow Transformers for High-Resolution
      Image Synthesis. (https://arxiv.org/abs/2403.03206)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Wang et al., 2024. Qwen2-VL: Enhancing Vision-Language Model's Perception of
      the World at Any Resolution. (introduces the multi-axis / M-RoPE scheme)
    - Zhang & Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Ho & Salimans, 2022. Classifier-Free Diffusion Guidance.
      (https://arxiv.org/abs/2207.12598)

    Ideogram 4 has no published architecture paper; this port follows the released
    reference implementation.
"""

import keras
from typing import Any, Dict, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import Ideogram4MRoPE
from dl_techniques.layers.embedding.scalar_sinusoidal_embedding import (
    ScalarSinusoidalEmbedding,
)
from dl_techniques.layers.transformers.ideogram4_block import (
    Ideogram4TransformerBlock,
    Ideogram4FinalLayer,
)
from dl_techniques.models.vision_language.ideogram4.config import (
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.vision_language.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.ideogram4.transformer")
class Ideogram4Transformer(keras.Model):
    """Ideogram4 flow-matching DiT: packed-stream masked-add velocity predictor.

    The model takes a DICT of packed-sequence inputs and returns a float32
    velocity prediction. Conditioning is a precomputed ``llm_features`` tensor
    (decision D1 -- no Qwen3-VL in Keras).

    Call inputs (a single ``dict`` -- keeps the multi-input model serializable):

    - ``"llm_features"``: ``(B, L, llm_features_dim)`` precomputed conditioning.
    - ``"x"``:            ``(B, L, in_channels)`` patchified noise latents.
    - ``"t"``:            ``(B,)`` or ``(B, L)`` diffusion time in ``[0, 1]``.
    - ``"position_ids"``: ``(B, L, 3)`` integer ``(t, h, w)`` mRoPE coordinates.
    - ``"segment_ids"``:  ``(B, L)`` integer block-diagonal attention segments.
    - ``"indicator"``:    ``(B, L)`` integer per-token role marker
      (``LLM_TOKEN_INDICATOR`` for text, ``OUTPUT_IMAGE_INDICATOR`` for image).

    Output: ``(B, L, in_channels)`` velocity, always float32.

    Forward pass:

    .. code-block:: text

        x [B,L,C]         llm_features [B,L,D]        t [B] or [B,L]
          │ mask,proj,mask   │ mask,norm,proj,mask       │
          ▼                  ▼                           ▼
        ┌──────────────────────────┐              ┌─────────────┐
        │ h = image + text (masked │              │ t_embedding │
        │ add) + indicator embed   │              │ + adaln proj│
        └────────────┬─────────────┘              └──────┬──────┘
                      │                                   │
                      │  cos, sin ◄── mRoPE(position_ids)  │
                      ▼                                   │
              ┌───────────────────┐                       │
              │ block_0 .. block_N│◄── segment_ids  ◄──────┘
              └────────┬──────────┘   (attention mask)
                       ▼
                ┌─────────────┐
                │ final_layer │
                └──────┬──────┘
                       ▼
              velocity [B,L,in_channels] (float32)

    :param config: The :class:`Ideogram4Config` describing the model.
    :type config: Ideogram4Config
    :param kwargs: Additional ``keras.Model`` arguments.

    :raises TypeError: If ``config`` is not an :class:`Ideogram4Config`.
    """

    # PyTorch ``Ideogram4RMSNorm(llm_features_dim, eps=1e-6)``.
    _LLM_COND_NORM_EPS: float = 1e-6

    def __init__(
        self,
        config: Ideogram4Config,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(config, Ideogram4Config):
            raise TypeError(
                f"config must be an Ideogram4Config, got {type(config)}"
            )

        self.config = config
        emb_dim = config.emb_dim
        head_dim = config.head_dim

        # Noise (image) projection: in_channels -> emb_dim.
        self.input_proj = keras.layers.Dense(
            emb_dim, use_bias=True, name="input_proj"
        )

        # Conditioning normalization + projection: llm_features_dim -> emb_dim.
        # RMSNorm over the last (feature) axis with the PyTorch eps=1e-6.
        self.llm_cond_norm = RMSNorm(
            axis=-1, epsilon=self._LLM_COND_NORM_EPS, name="llm_cond_norm"
        )
        self.llm_cond_proj = keras.layers.Dense(
            emb_dim, use_bias=True, name="llm_cond_proj"
        )

        # Time embedding: scalar t in [0, 1] -> emb_dim, then AdaLN projection.
        self.t_embedding = ScalarSinusoidalEmbedding(
            dim=emb_dim, input_range=(0.0, 1.0), name="t_embedding"
        )
        self.adaln_proj = keras.layers.Dense(
            config.adanln_dim, use_bias=True, name="adaln_proj"
        )

        # Image-indicator embedding: index 0 (text) / 1 (image) -> emb_dim.
        self.embed_image_indicator = keras.layers.Embedding(
            input_dim=2, output_dim=emb_dim, name="embed_image_indicator"
        )

        # 3D multi-axis rotary embedding (non-trainable cos/sin tables).
        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            name="rotary_emb",
        )

        # The DiT block stack (flat list of sub-layers -- NOT List[List]).
        self.blocks = [
            Ideogram4TransformerBlock(
                hidden_size=emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                adaln_dim=config.adanln_dim,
                norm_eps=config.norm_eps,
                name=f"block_{i}",
            )
            for i in range(config.num_layers)
        ]

        # Final layer: emb_dim -> in_channels velocity head.
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=emb_dim,
            out_channels=config.in_channels,
            adaln_dim=config.adanln_dim,
            name="final_layer",
        )

        logger.debug(
            f"Initialized Ideogram4Transformer(emb_dim={emb_dim}, "
            f"head_dim={head_dim}, num_layers={config.num_layers}, "
            f"in_channels={config.in_channels}, "
            f"llm_features_dim={config.llm_features_dim})"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        The default `Layer.build` would mark the model built while its
        sub-layers stay unbuilt. This traces `call()` on symbolic inputs
        instead, so what gets built cannot drift from what gets called.

        :param input_shape: Shape, or nest of shapes, of the input to `call`.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the packed-stream masked-add DiT forward.

        :param inputs: Dict with keys ``"llm_features"``, ``"x"``, ``"t"``,
            ``"position_ids"``, ``"segment_ids"``, ``"indicator"`` (see class
            docstring for shapes).
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: Velocity prediction ``(B, L, in_channels)`` in float32.
        :rtype: keras.KerasTensor
        """
        llm_features = inputs["llm_features"]
        x = inputs["x"]
        t = inputs["t"]
        position_ids = inputs["position_ids"]
        segment_ids = inputs["segment_ids"]
        indicator = inputs["indicator"]

        compute_dtype = self.input_proj.compute_dtype

        is_text = keras.ops.equal(indicator, LLM_TOKEN_INDICATOR)
        is_image = keras.ops.equal(indicator, OUTPUT_IMAGE_INDICATOR)
        llm_token_mask = keras.ops.cast(
            keras.ops.expand_dims(is_text, axis=-1), compute_dtype
        )
        output_image_mask = keras.ops.cast(
            keras.ops.expand_dims(is_image, axis=-1), compute_dtype
        )

        # Mask both before and after the projection, since the Dense bias
        # would otherwise leak across the text/image boundary after the add.
        x = keras.ops.cast(x, compute_dtype) * output_image_mask
        x = self.input_proj(x, training=training) * output_image_mask

        t_cond = self.t_embedding(t, training=training)
        # Rank is known at trace time, so this stays a plain python branch.
        if keras.ops.ndim(t_cond) == 2:
            # t was (B,): add a length-1 token axis so it broadcasts over L.
            t_cond = keras.ops.expand_dims(t_cond, axis=1)
        adaln_input = keras.ops.silu(
            self.adaln_proj(t_cond, training=training)
        )

        llm_features = keras.ops.cast(llm_features, compute_dtype) * llm_token_mask
        llm_features = self.llm_cond_norm(llm_features, training=training)
        llm_features = (
            self.llm_cond_proj(llm_features, training=training) * llm_token_mask
        )

        h = x + llm_features

        indicator_index = keras.ops.cast(is_image, "int32")
        h = h + self.embed_image_indicator(indicator_index)

        cos, sin = self.rotary_emb(position_ids)

        for block in self.blocks:
            h = block(h, segment_ids, cos, sin, adaln_input, training=training)

        # Always float32, even under a mixed-precision policy.
        out = self.final_layer(h, c=adaln_input, training=training)
        return keras.ops.cast(out, "float32")

    def compute_output_shape(
        self, input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return ``x``'s shape with the last dim set to ``in_channels``.

        :param input_shape: Dict of per-key input shapes (uses ``"x"``).
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: ``(B, L, in_channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        x_shape = input_shape["x"]
        return tuple(x_shape[:-1]) + (self.config.in_channels,)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config (the ``Ideogram4Config`` as a dict).

        :return: Dictionary carrying the config under ``"config"``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Ideogram4Transformer":
        """Reconstruct from :meth:`get_config` output.

        :param config: The serialized config dict.
        :type config: Dict[str, Any]
        :return: A reconstructed :class:`Ideogram4Transformer`.
        :rtype: Ideogram4Transformer
        """
        config = dict(config)
        config["config"] = Ideogram4Config.from_dict(config["config"])
        return cls(**config)


def create_ideogram4_transformer(
    variant: str = "tiny",
    **overrides: Any,
) -> Ideogram4Transformer:
    """Build an :class:`Ideogram4Transformer` from a named preset.

    Retrieves the ``(config, ae)`` pair for ``variant`` via
    :func:`get_ideogram4_config` (which runs all config invariants), applies any
    field ``overrides`` (re-validated by ``Ideogram4Config.__post_init__``), and
    returns the constructed model. The paired ``AutoEncoderParams`` is not needed
    by the transformer and is discarded here.

    :param variant: One of the config presets (``"tiny"`` or ``"full"``).
    :type variant: str
    :param overrides: Field overrides applied to the preset ``Ideogram4Config``
        (e.g. ``num_layers=4``). Re-validated on construction.
    :type overrides: Any
    :return: The constructed (un-built) transformer model.
    :rtype: Ideogram4Transformer
    """
    config, _ = get_ideogram4_config(variant)
    if overrides:
        merged = {**config.to_dict(), **overrides}
        config = Ideogram4Config.from_dict(merged)

    logger.info(
        "Creating Ideogram4Transformer variant='%s' (emb_dim=%d, num_layers=%d)",
        variant,
        config.emb_dim,
        config.num_layers,
    )
    return Ideogram4Transformer(config=config)
