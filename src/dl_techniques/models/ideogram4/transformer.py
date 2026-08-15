"""
Ideogram4 flow-matching diffusion transformer: a single packed self-attention
stream that interleaves text-conditioning tokens with image latent tokens and
regresses a velocity field, with 3D multi-axis rotary positions and a tanh-gated
4-stream AdaLN block.

The generative principle is flow matching. Training pairs a clean latent with a
noise sample, interpolates them along a straight line, and asks the network to
predict the path's velocity -- a target that is constant along the path, which is
why a handful of Euler steps can integrate it. What the transformer itself pins is
narrow: `t` is a scalar in `[0, 1]` (`ScalarSinusoidalEmbedding(input_range=(0.0,
1.0))`), it may be per-sample `(B,)` or per-token `(B, L)`, and the output is a
velocity in the latent's own channel space. Everything about *which* endpoint `t = 0`
denotes lives outside this module; the convention the package settles on is stated in
the note below.

The structurally distinctive choice is that there is no cross-attention anywhere.
Rather than keeping conditioning in a separate tower and injecting it through
cross-attention layers, Ideogram4 packs text and image tokens into ONE sequence and
lets ordinary self-attention do the mixing. Roles are carried as data, not as
structure: an integer `indicator` marks each position as text or image, and two
float masks derived from it gate each stream's contribution before a plain addition
merges them into the single hidden state `h`. The masking is applied twice on each
branch, before *and* after its projection, and the second application is the
load-bearing one: `input_proj` and `llm_cond_proj` both carry a bias, so without the
post-projection mask every text position would pick up the image projection's bias
vector (and vice versa) and the merge would no longer be role-pure. A learned 2-entry
`embed_image_indicator` is then added on top, so the stack can distinguish the two
roles by more than the content that happens to be there. The payoff of packing is
uniformity: one block stack, one rotary table and one mask mechanism serve any
text/image split without reshaping, and the split may vary per batch.

Attention structure comes from `segment_ids`, not from a triangular mask. Positions
sharing a segment attend to each other and nothing else, which makes attention
block-diagonal over independently packed rows. This is deliberately NOT causal, and
should not be: a diffusion transformer denoises the whole latent at once, so an
image token must see the entire caption and the entire latent. The block-diagonal
keep-mask is realized as an ADDITIVE finite mask, `where(same_segment, 0.0, -1e9)`
added to the scaled scores (D-004), rather than a boolean mask that could produce an
all-`-inf` row and hence a NaN softmax; the finite form is also XLA-safe.

Positional information is 3D. `position_ids` carries `(t, h, w)` coordinates per
token, and mRoPE splits the head's rotary frequency budget into three bands sized by
`mrope_section`, interleaving them across `head_dim / 2` so each axis rotates its own
subset of feature pairs. The cos/sin tables are computed once per forward and shared
by every block. The PyTorch reference builds the band interleave with a dynamic
in-place scatter; this port precomputes a static per-slot one-hot selector at
`build()` and applies it with an einsum (D-003), which is XLA-safe and was verified
element-wise against a NumPy reproduction of the reference forward.

Conditioning enters only through modulation. The scalar time is sinusoidally
embedded, projected to `adanln_dim` and passed through SiLU -- note the activation
comes *after* the projection -- yielding `adaln_input`, which every block and the
output head consume. When `t` arrives per-sample the resulting `(B, adaln)` vector is
expanded to `(B, 1, adaln)` so it broadcasts over the sequence; the branch is a
Python `if` on static rank, not a tensor-valued condition, so it is trace-safe. The
per-block modulation itself lives in `Ideogram4TransformerBlock` and departs from
this repo's usual AdaLN-zero block in three ways worth knowing before comparing them:
it emits 4 streams rather than 6 (scale and gate per sublayer, NO shift), the gates
are `tanh` rather than raw, and each sublayer sits in a 4-RMSNorm sandwich with a
post-norm applied INSIDE the residual branch, i.e.
`x = x + tanh(gate) * norm2(sublayer(norm1(x) * (1 + scale)))`. That post-norm
placement is unusual but is replicated exactly rather than normalized away.

Deliberate choices and divergences:

- **The time convention, settled: `t = 0` is clean data, `t = 1` is pure noise, and
  the velocity points data -> noise.** `src/train/ideogram4/train_ideogram4.py` trains
  with `x_t = (1 - tau) * x0 + tau * x1` for `x1 ~ N(0, I)` and target `v = x1 - x0`,
  which fixes that assignment; `pipeline.py`'s sampler follows it, seeding pure noise
  at the top of the time grid and integrating DOWN, so its `z += v * (s - t)` runs with
  `s < t` (a negative step) and the `t` it evaluates the transformer at descends
  monotonically toward the data end. Because `LogitNormalSchedule` is strictly
  DECREASING in its uniform argument, the loop reads its step grid in reverse to obtain
  that descent -- see the D-002 derivation at the loop itself. This is the same
  convention as `models/sd3_mmdit/`, which implements the identical rectified flow, so
  the two packages' schedulers and samplers can be read against each other. The module's
  `[0, 1]` range alone still implies nothing; the endpoints are fixed by the trainer.
- Conditioning is a PRECOMPUTED `llm_features` call input (D1). The reference
  conditions on 13 stacked hidden-state taps of Qwen3-VL-8B; that model has no Keras
  equivalent, so the text tower is out of scope and this is not an end-to-end
  text-to-image model as it stands.
- The architecture is faithful in math but NOT weight-compatible with the released
  `ideogram-4-nf4` checkpoint: there is no nf4 dequantization path and no parameter
  name map. Models built here are randomly initialized.
- The velocity head casts to float32 unconditionally, even under a mixed-precision
  policy, mirroring the reference's `.float()` return.
- `Ideogram4Config` is a plain frozen dataclass rather than a Keras-serializable
  object, so `from_config` rebuilds it via `Ideogram4Config.from_dict`.

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
        # Static-rank branch (guide-preferred ndim over len(.shape)); rank is
        # known at trace, so this stays a python `if`, not a tensor-value branch.
        if keras.ops.ndim(t_cond) == 2:
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
