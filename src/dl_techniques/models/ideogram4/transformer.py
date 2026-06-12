"""
Ideogram4 flow-matching DiT transformer model (Keras 3 port).

This module assembles the Ideogram4 ``Ideogram4Transformer`` from the
already-built, separately-tested layers (mRoPE, scalar/time embed, the
4-stream tanh-gated AdaLN block, RMSNorm) into a single ``keras.Model``. It is
a faithful port of the PyTorch ``Ideogram4Transformer.__init__`` / ``forward``,
with the single deliberate structural change locked by decision D1: conditioning
arrives as a precomputed ``llm_features`` tensor (no Qwen3-VL in Keras).

Packed-stream forward (the structurally-novel bit, ported exactly)
------------------------------------------------------------------
The transformer consumes a SINGLE packed self-attention stream that interleaves
text tokens (carrying projected ``llm_features``) and image tokens (carrying
projected noise ``x``). There is NO cross-attention. Per-token roles are marked
by an integer ``indicator``:

- ``LLM_TOKEN_INDICATOR`` (text)   -> contributes ``llm_features``.
- ``OUTPUT_IMAGE_INDICATOR`` (img) -> contributes the projected noise ``x``.

Two float masks gate the two contributions so each role only sees its own
content:

    llm_token_mask    = (indicator == LLM_TOKEN_INDICATOR)[..., None]   # float
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR)[..., None]

    llm_features = llm_features * llm_token_mask
    x            = x * output_image_mask
    x            = input_proj(x) * output_image_mask
    llm_features = llm_cond_proj(llm_cond_norm(llm_features)) * llm_token_mask
    h            = x + llm_features                              # masked add
    h            = h + embed_image_indicator(indicator == OUTPUT_IMAGE_INDICATOR)

Conditioning (time) goes through the scalar sinusoidal embed + a SiLU AdaLN
projection to ``adaln_dim``; the per-block 4-stream modulation lives inside the
``Ideogram4TransformerBlock``. The final velocity is cast to float32 even under
mixed precision (PyTorch returns ``.float()``).

PyTorch reference (faithfully ported)::

    head_dim = emb_dim // num_heads
    self.input_proj    = nn.Linear(in_channels, emb_dim, bias=True)
    self.llm_cond_norm = Ideogram4RMSNorm(llm_features_dim, eps=1e-6)
    self.llm_cond_proj = nn.Linear(llm_features_dim, emb_dim, bias=True)
    self.t_embedding   = Ideogram4EmbedScalar(emb_dim, input_range=(0, 1))
    self.adaln_proj    = nn.Linear(emb_dim, adanln_dim, bias=True)
    self.embed_image_indicator = nn.Embedding(2, emb_dim)
    self.rotary_emb    = Ideogram4MRoPE(head_dim, base=rope_theta, mrope_section=...)
    self.layers        = [Ideogram4TransformerBlock(...) for _ in range(num_layers)]
    self.final_layer   = Ideogram4FinalLayer(emb_dim, in_channels, adanln_dim)

    def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
        llm_token_mask    = (indicator == LLM_TOKEN_INDICATOR)[..., None]
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR)[..., None]
        llm_features = llm_features * llm_token_mask
        x = self.input_proj(x * output_image_mask) * output_image_mask
        t_cond = self.t_embedding(t)
        if t.ndim == 1: t_cond = t_cond[:, None, :]
        adaln_input = silu(self.adaln_proj(t_cond))
        llm_features = self.llm_cond_proj(self.llm_cond_norm(llm_features)) * llm_token_mask
        h = x + llm_features
        h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).long())
        cos, sin = self.rotary_emb(position_ids)
        for layer in self.layers: h = layer(h, segment_ids, cos, sin, adaln_input)
        return self.final_layer(h, c=adaln_input).float()
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

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
from dl_techniques.models.ideogram4.config import (
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
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

        # --- sub-layers (created in __init__; functional weights build on first
        #     call). A keras.Model may own sub-layers as plain attributes. ---

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

        # --- role masks (float, gate each contribution to its own tokens) ---
        is_text = keras.ops.equal(indicator, LLM_TOKEN_INDICATOR)
        is_image = keras.ops.equal(indicator, OUTPUT_IMAGE_INDICATOR)
        llm_token_mask = keras.ops.cast(
            keras.ops.expand_dims(is_text, axis=-1), compute_dtype
        )  # (B, L, 1)
        output_image_mask = keras.ops.cast(
            keras.ops.expand_dims(is_image, axis=-1), compute_dtype
        )  # (B, L, 1)

        # --- image (noise) stream: mask -> project -> mask ---
        x = keras.ops.cast(x, compute_dtype) * output_image_mask
        x = self.input_proj(x, training=training) * output_image_mask

        # --- conditioning (time) -> AdaLN input ---
        t_cond = self.t_embedding(t, training=training)  # (B, emb) or (B, L, emb)
        if len(t_cond.shape) == 2:
            # t was (B,): add a length-1 token axis to broadcast over L.
            t_cond = keras.ops.expand_dims(t_cond, axis=1)  # (B, 1, emb)
        adaln_input = keras.ops.silu(
            self.adaln_proj(t_cond, training=training)
        )  # (B, 1, adaln) or (B, L, adaln)

        # --- text (conditioning) stream: mask -> norm -> project -> mask ---
        llm_features = keras.ops.cast(llm_features, compute_dtype) * llm_token_mask
        llm_features = self.llm_cond_norm(llm_features, training=training)
        llm_features = (
            self.llm_cond_proj(llm_features, training=training) * llm_token_mask
        )

        # --- masked add into the single packed stream ---
        h = x + llm_features

        # Image-indicator embedding: index 1 for image tokens, 0 otherwise.
        indicator_index = keras.ops.cast(is_image, "int32")  # (B, L) in {0, 1}
        h = h + self.embed_image_indicator(indicator_index)

        # --- mRoPE tables (shared across blocks) ---
        cos, sin = self.rotary_emb(position_ids)

        # --- DiT block stack ---
        for block in self.blocks:
            h = block(h, segment_ids, cos, sin, adaln_input, training=training)

        # --- velocity head; always float32 (PyTorch returns .float()) ---
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


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


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

# ---------------------------------------------------------------------
