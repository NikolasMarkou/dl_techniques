"""Ideogram4 image-generation inference pipeline (Keras 3 port).

Integrates the three already-built, separately-tested pieces -- the
flow-matching DiT :class:`Ideogram4Transformer`, the Flux2 KL-VAE
:class:`AutoEncoder`, and the logit-normal / Euler :class:`LogitNormalSchedule`
-- into a single end-to-end denoise-and-decode pipeline.

Decision D1 (conditioning as input)
-----------------------------------
There is NO Qwen3-VL text encoder and NO tokenizer in this Keras port. The
pipeline takes the precomputed ``llm_features`` conditioning tensor
``(B, T, llm_features_dim)`` DIRECTLY as a call input. The PyTorch
``_build_inputs(num_text_tokens_per_sample, ...)`` (which packs variable-length
text from a tokenizer) is therefore simplified: ``T`` is fixed per batch as
``llm_features.shape[1]`` and there is no left-padding (one segment per row).

Faithful bits ported from PyTorch
---------------------------------
- ``_build_inputs``: packed ``position_ids`` with ``IMAGE_POSITION_OFFSET`` on
  the image grid, ``segment_ids``, and the per-token ``indicator``
  (``LLM_TOKEN_INDICATOR`` on text, ``OUTPUT_IMAGE_INDICATOR`` on image).
- ``__call__``: the Euler denoise loop with ASYMMETRIC CFG
  ``v = gw * pos_v + (1 - gw) * neg_v`` and the update ``z += v * (s - t)``.
- ``_decode``: unpatchify ``(B, gh, gw, p, p, c)`` -> ``(B, gh*p, gw*p, c)``
  [NHWC], optional latent denorm (shift/scale), VAE decode, clamp to ``[-1, 1]``.

Deliberate Keras-port simplifications (documented surprises)
-----------------------------------------------------------
1. **Single shared transformer for both CFG branches** (PyTorch uses two
   separate transformer modules -- a conditional and an unconditional one).
   Here the SAME ``transformer`` runs both branches by default: the negative
   (unconditional) branch is image-only with ``llm_features`` zeroed, so the
   asymmetric CFG is still well-defined. An optional
   ``unconditional_transformer`` may be supplied to recover the two-model form.
2. **Latent denorm guarded on ``in_channels == 128``.** ``get_latent_norm``
   returns 128-element shift/scale vectors keyed to the FULL config's
   ``in_channels``. For the TINY preset (``in_channels=32 != 128``) those
   vectors do not apply, so latent denorm is skipped (identity pass-through)
   with a logged note. Only the full-scale latent (128) is denormed.
3. **Image size is derived from the ACTUAL VAE upsample factor**
   (``2 ** (len(ch_mult) - 1)``), not the nominal ``config.ae_scale_factor``.
   The tiny VAE (``ch_mult=(1, 2)``) upsamples by 2, while the full VAE
   (``ch_mult=(1, 2, 4, 4)``) upsamples by 8 -- matching ``ae_scale_factor``
   only for the full preset. Deriving from ``ch_mult`` keeps the pipeline
   self-consistent for any preset (the latent the decoder receives always has
   ``z_channels`` channels at the correct spatial size).

The denoise loop runs at inference (no ``GradientTape``); the schedule eval is
scalar / NumPy on the CPU (built in step 8). Random noise is drawn via
``keras.random.normal`` with an integer ``seed`` for determinism.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.ideogram4.config import (
    AutoEncoderParams,
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.models.ideogram4.latent_norm import get_latent_norm
from dl_techniques.models.ideogram4.scheduler import (
    LogitNormalSchedule,
    get_schedule_for_resolution,
    make_step_intervals,
)
from dl_techniques.models.ideogram4.transformer import (
    Ideogram4Transformer,
    create_ideogram4_transformer,
)
from dl_techniques.models.ideogram4.vae import (
    AutoEncoder,
    create_ideogram4_autoencoder,
)

# ---------------------------------------------------------------------
# Latent-norm guard: get_latent_norm() returns 128-element vectors keyed to the
# full config's in_channels. Denorm only applies when in_channels matches.
# ---------------------------------------------------------------------
_LATENT_NORM_CHANNELS = 128


def apply_cfg_blend(
    pos_v: keras.KerasTensor,
    neg_v: keras.KerasTensor,
    gw: float,
) -> keras.KerasTensor:
    """Asymmetric classifier-free-guidance blend ``gw*pos + (1-gw)*neg``.

    Factored out so the blend math is unit-testable in isolation. With
    ``gw == 1`` the result is the conditional branch alone; with ``gw == 0`` it
    is the unconditional branch alone.

    Args:
        pos_v: Conditional (positive) velocity ``(B, L_img, C)``.
        neg_v: Unconditional (negative) velocity ``(B, L_img, C)``.
        gw: Scalar guidance weight for this step.

    Returns:
        The blended velocity, same shape as the inputs.
    """
    return gw * pos_v + (1.0 - gw) * neg_v


class Ideogram4Pipeline:
    """End-to-end Ideogram4 image generation: denoise (Euler + CFG) then decode.

    A plain orchestration class (NOT a ``keras.Layer`` / ``keras.Model``) that
    holds the trained sub-models and the structural config. Conditioning is the
    precomputed ``llm_features`` tensor passed to :meth:`__call__` (decision D1).

    Args:
        transformer: The flow-matching DiT velocity predictor. Used for BOTH the
            conditional and (by default) the unconditional CFG branches.
        autoencoder: The Flux2 KL-VAE; only :meth:`AutoEncoder.decode` is used.
        config: The transformer / pipeline :class:`Ideogram4Config`.
        ae_params: The VAE :class:`AutoEncoderParams` (drives unpatchify /
            spatial-factor math).
        unconditional_transformer: Optional separate model for the negative CFG
            branch (PyTorch's two-model form). Defaults to ``None`` -> the shared
            ``transformer`` runs both branches.

    Raises:
        TypeError: If ``transformer`` / ``autoencoder`` / ``config`` /
            ``ae_params`` are not of the expected types.
    """

    def __init__(
        self,
        transformer: Ideogram4Transformer,
        autoencoder: AutoEncoder,
        config: Ideogram4Config,
        ae_params: AutoEncoderParams,
        unconditional_transformer: Optional[Ideogram4Transformer] = None,
    ) -> None:
        if not isinstance(transformer, Ideogram4Transformer):
            raise TypeError(
                f"transformer must be an Ideogram4Transformer, "
                f"got {type(transformer)}"
            )
        if not isinstance(autoencoder, AutoEncoder):
            raise TypeError(
                f"autoencoder must be an AutoEncoder, got {type(autoencoder)}"
            )
        if not isinstance(config, Ideogram4Config):
            raise TypeError(
                f"config must be an Ideogram4Config, got {type(config)}"
            )
        if not isinstance(ae_params, AutoEncoderParams):
            raise TypeError(
                f"ae_params must be an AutoEncoderParams, got {type(ae_params)}"
            )

        self.transformer = transformer
        self.autoencoder = autoencoder
        self.config = config
        self.ae_params = ae_params
        self.unconditional_transformer = unconditional_transformer

        logger.debug(
            "Initialized Ideogram4Pipeline(in_channels=%d, patch_size=%d, "
            "vae_factor=%d, shared_transformer=%s)",
            config.in_channels,
            config.patch_size,
            self.vae_upsample_factor,
            unconditional_transformer is None,
        )

    # -----------------------------------------------------------------
    # Derived geometry
    # -----------------------------------------------------------------

    @property
    def vae_upsample_factor(self) -> int:
        """Actual VAE decode spatial-upsample factor ``2**(len(ch_mult)-1)``.

        The decoder upsamples once between every pair of resolution stages, so
        the latent->pixel ratio is ``2 ** (num_stages - 1)``. This is the
        ground-truth factor (the nominal ``config.ae_scale_factor`` matches it
        only for the full preset).
        """
        return 2 ** (len(self.ae_params.ch_mult) - 1)

    @property
    def pixels_per_token_edge(self) -> int:
        """Pixel edge length covered by one image token: ``patch * vae_factor``."""
        return self.config.patch_size * self.vae_upsample_factor

    # -----------------------------------------------------------------
    # Packed-input construction (simplified _build_inputs)
    # -----------------------------------------------------------------

    def _build_inputs(
        self,
        batch_size: int,
        num_text_tokens: int,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        """Build packed ``position_ids`` / ``segment_ids`` / ``indicator``.

        Simplified port of the PyTorch ``_build_inputs``: one sample per row, no
        left-padding (``T`` is fixed per batch), a single attention segment per
        row. The sequence is ``[T text tokens][grid_h*grid_w image tokens]``.

        Args:
            batch_size: Number of samples ``B``.
            num_text_tokens: Conditioning length ``T`` (= ``llm_features.shape[1]``).
            height: Target image height in pixels (divisible by
                :attr:`pixels_per_token_edge`).
            width: Target image width in pixels.

        Returns:
            ``(position_ids, segment_ids, indicator, num_image_tokens, grid_h,
            grid_w)`` -- the three packed arrays plus the image-grid sizes.

        Raises:
            ValueError: If ``height`` / ``width`` are not divisible by the
                per-token pixel edge.
        """
        patch = self.pixels_per_token_edge
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"height ({height}) and width ({width}) must be divisible by "
                f"patch_size * vae_factor = {patch}."
            )
        grid_h = height // patch
        grid_w = width // patch
        num_image = grid_h * grid_w
        total_len = num_text_tokens + num_image

        # --- image position ids (t=0, h in [0,grid_h), w in [0,grid_w)) each
        #     offset by IMAGE_POSITION_OFFSET so they never collide with text. ---
        hh, ww = np.meshgrid(
            np.arange(grid_h), np.arange(grid_w), indexing="ij"
        )
        image_pos = np.stack(
            [
                np.zeros(num_image, dtype=np.int32),            # t axis
                hh.reshape(-1).astype(np.int32),                # h axis
                ww.reshape(-1).astype(np.int32),                # w axis
            ],
            axis=-1,
        )
        image_pos = image_pos + IMAGE_POSITION_OFFSET           # (num_image, 3)

        # --- text position ids: arange(T) replicated across the 3 axes. ---
        text_arange = np.arange(num_text_tokens, dtype=np.int32)
        text_pos = np.stack([text_arange] * 3, axis=-1)         # (T, 3)

        # --- pack: text block then image block, broadcast over the batch. ---
        pos_single = np.concatenate([text_pos, image_pos], axis=0)  # (L, 3)
        position_ids = np.broadcast_to(
            pos_single[None], (batch_size, total_len, 3)
        ).astype(np.int32).copy()

        # --- segment_ids: one segment per row (all ones). ---
        segment_ids = np.ones((batch_size, total_len), dtype=np.int32)

        # --- indicator: LLM on text positions, OUTPUT_IMAGE on image positions. ---
        indicator = np.empty((batch_size, total_len), dtype=np.int32)
        indicator[:, :num_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

        return position_ids, segment_ids, indicator, num_image, grid_h, grid_w

    # -----------------------------------------------------------------
    # Decode (unpatchify + optional latent denorm + VAE decode)
    # -----------------------------------------------------------------

    def _decode(
        self,
        z: keras.KerasTensor,
        grid_h: int,
        grid_w: int,
    ) -> keras.KerasTensor:
        """Unpatchify the latent, (optionally) denorm, and VAE-decode.

        Args:
            z: Final denoised latent ``(B, num_image, in_channels)``.
            grid_h: Image-grid height in tokens.
            grid_w: Image-grid width in tokens.

        Returns:
            The decoded image ``(B, H, W, out_ch)`` in ``[0, 1]``.
        """
        patch = self.config.patch_size
        in_channels = self.config.in_channels
        ae_channels = in_channels // (patch * patch)  # = z_channels

        # --- latent denorm (guarded on in_channels == 128). ---
        if in_channels == _LATENT_NORM_CHANNELS:
            shift, scale = get_latent_norm()  # (128,), (128,)
            z = z * scale + shift
        else:
            logger.debug(
                "Skipping latent denorm: in_channels=%d != %d (latent_norm "
                "vectors apply to the full config only); pass-through.",
                in_channels,
                _LATENT_NORM_CHANNELS,
            )

        # --- unpatchify: (B, gh*gw, p*p*c) -> (B, gh*p, gw*p, c) [NHWC]. ---
        batch = keras.ops.shape(z)[0]
        z = keras.ops.reshape(
            z, (batch, grid_h, grid_w, patch, patch, ae_channels)
        )
        # (B, gh, p, gw, p, c) so reshape merges (gh,p) -> H and (gw,p) -> W.
        z = keras.ops.transpose(z, (0, 1, 3, 2, 4, 5))
        z_img = keras.ops.reshape(
            z, (batch, grid_h * patch, grid_w * patch, ae_channels)
        )

        # --- VAE decode -> image, clamp to [-1, 1], map to [0, 1]. ---
        image = self.autoencoder.decode(z_img)
        image = keras.ops.clip(image, -1.0, 1.0)
        image = (image + 1.0) * 0.5
        return image

    # -----------------------------------------------------------------
    # Denoise + decode
    # -----------------------------------------------------------------

    def __call__(
        self,
        llm_features: keras.KerasTensor,
        height: int,
        width: int,
        num_steps: int = 4,
        guidance_scale: float = 7.0,
        guidance_schedule: Optional[Sequence[float]] = None,
        mu: float = 0.0,
        std: float = 1.0,
        seed: int = 0,
        schedule: Optional[LogitNormalSchedule] = None,
    ) -> keras.KerasTensor:
        """Run the full Euler + asymmetric-CFG denoise loop, then VAE-decode.

        Args:
            llm_features: Precomputed conditioning ``(B, T, llm_features_dim)``.
            height: Target image height in pixels.
            width: Target image width in pixels.
            num_steps: Number of Euler integration steps.
            guidance_scale: Constant CFG weight (used when ``guidance_schedule``
                is ``None``).
            guidance_schedule: Optional per-step CFG weights in LOOP-INDEX order
                (length must equal ``num_steps``); index 0 is the LAST step.
            mu: Logit-normal schedule mean (``known_mean``). Used when
                ``schedule`` is ``None``.
            std: Logit-normal schedule stddev. Used when ``schedule`` is ``None``.
            seed: Integer seed for the initial noise (determinism).
            schedule: Optional prebuilt :class:`LogitNormalSchedule`; if ``None``
                one is built via :func:`get_schedule_for_resolution`.

        Returns:
            The generated image ``(B, height, width, out_ch)`` in ``[0, 1]``.

        Raises:
            ValueError: If ``guidance_schedule`` length disagrees with
                ``num_steps`` or ``height``/``width`` are not patch-divisible.
        """
        batch_size = int(keras.ops.shape(llm_features)[0])
        num_text_tokens = int(keras.ops.shape(llm_features)[1])
        in_channels = self.config.in_channels
        llm_dim = self.config.llm_features_dim

        # --- schedule + step grid + per-step guidance weights. ---
        if schedule is None:
            schedule = get_schedule_for_resolution(
                (height, width), known_mean=mu, std=std
            )
        step_intervals = make_step_intervals(num_steps)  # (num_steps+1,) numpy

        if guidance_schedule is not None:
            if len(guidance_schedule) != num_steps:
                raise ValueError(
                    f"guidance_schedule has length {len(guidance_schedule)}, "
                    f"expected num_steps={num_steps}."
                )
            gw_per_step: List[float] = [float(g) for g in guidance_schedule]
        else:
            gw_per_step = [float(guidance_scale)] * num_steps

        # --- packed inputs. ---
        (
            position_ids,
            segment_ids,
            indicator,
            num_image,
            grid_h,
            grid_w,
        ) = self._build_inputs(batch_size, num_text_tokens, height, width)

        position_ids = keras.ops.convert_to_tensor(position_ids)
        segment_ids = keras.ops.convert_to_tensor(segment_ids)
        indicator = keras.ops.convert_to_tensor(indicator)

        # --- conditional llm_features placed at text positions, zero at image. ---
        text_feats = keras.ops.cast(llm_features, "float32")  # (B, T, llm_dim)
        image_feat_pad = keras.ops.zeros(
            (batch_size, num_image, llm_dim), dtype="float32"
        )
        llm_features_full = keras.ops.concatenate(
            [text_feats, image_feat_pad], axis=1
        )  # (B, L, llm_dim)

        # --- initial latent noise (image tokens only). ---
        z = keras.random.normal(
            (batch_size, num_image, in_channels), seed=seed, dtype="float32"
        )
        text_z_padding = keras.ops.zeros(
            (batch_size, num_text_tokens, in_channels), dtype="float32"
        )
        pos_z = keras.ops.concatenate([text_z_padding, z], axis=1)  # (B, L, C)

        # --- negative (unconditional) branch: IMAGE-ONLY, conditioning zeroed. ---
        neg_position_ids = position_ids[:, num_text_tokens:]
        neg_segment_ids = segment_ids[:, num_text_tokens:]
        neg_indicator = indicator[:, num_text_tokens:]
        neg_llm_features = keras.ops.zeros(
            (batch_size, num_image, llm_dim), dtype="float32"
        )

        neg_model = self.unconditional_transformer or self.transformer

        # --- Euler loop: i from num_steps-1 down to 0. ---
        for i in range(num_steps - 1, -1, -1):
            t_val = float(schedule(float(step_intervals[i + 1])))
            s_val = float(schedule(float(step_intervals[i])))
            gw_i = gw_per_step[i]

            # Shape (B, 1) -- NOT (B,). ScalarSinusoidalEmbedding squeezes a
            # trailing singleton, so a rank-1 (B,) tensor with B == 1 would
            # collapse to a scalar and break its Dense. (B, 1) squeezes to (B,)
            # safely for any B, then the transformer expands it back over L.
            t = keras.ops.full((batch_size, 1), t_val, dtype="float32")

            # conditional (full-seq) velocity, image slice only.
            pos_out = self.transformer(
                dict(
                    llm_features=llm_features_full,
                    x=pos_z,
                    t=t,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                )
            )
            pos_v = pos_out[:, num_text_tokens:]  # (B, num_image, C)

            # unconditional (image-only) velocity.
            neg_v = neg_model(
                dict(
                    llm_features=neg_llm_features,
                    x=z,
                    t=t,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                )
            )

            v = apply_cfg_blend(pos_v, neg_v, gw_i)
            z = z + v * (s_val - t_val)
            pos_z = keras.ops.concatenate([text_z_padding, z], axis=1)

        # --- decode the final latent. ---
        return self._decode(z, grid_h, grid_w)

    # -----------------------------------------------------------------
    # Convenience constructor
    # -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        variant: str = "tiny",
        seed: Optional[int] = None,
    ) -> "Ideogram4Pipeline":
        """Build a fresh (UNTRAINED) pipeline from a named preset for smoke tests.

        Constructs a new transformer + autoencoder from the ``variant`` preset.
        The resulting pipeline runs end-to-end but produces noise (no weights are
        trained) -- useful only for shape / finiteness / determinism smoke tests.

        Args:
            variant: One of the config presets (``"tiny"`` or ``"full"``).
            seed: Optional sampling seed forwarded to the VAE ``Sampling`` layer
                (unused at decode, which is deterministic).

        Returns:
            A constructed :class:`Ideogram4Pipeline`.
        """
        config, ae_params = get_ideogram4_config(variant)
        transformer = create_ideogram4_transformer(variant)
        autoencoder = create_ideogram4_autoencoder(variant, sampling_seed=seed)
        return cls(
            transformer=transformer,
            autoencoder=autoencoder,
            config=config,
            ae_params=ae_params,
        )
