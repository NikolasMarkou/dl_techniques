"""End-to-end Ideogram4 image generation: denoise, then decode.

Defines :class:`Ideogram4Pipeline`, which wires together three already-built
pieces: the flow-matching DiT `Ideogram4Transformer`, the VAE `AutoEncoder`,
and the `LogitNormalSchedule` time warp. It runs an Euler integration loop
with asymmetric classifier-free guidance to denoise a latent, then decodes
that latent into an image.

There is no text encoder or tokenizer here. The pipeline takes a precomputed
`llm_features` conditioning tensor directly as a call input, one segment per
row with no padding, instead of building it from raw text. The negative CFG
branch reuses the same transformer as the positive branch by default, with
`llm_features` zeroed, rather than requiring two separate trained models.

Callers must respect the time convention: `t = 0` is clean data and `t = 1`
is pure noise, so the Euler loop walks from the noise end to the data end,
and the per-step delta `s - t` is negative throughout. `height` and `width`
must be divisible by `patch_size * vae_upsample_factor`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision_language.ideogram4.config import (
    AutoEncoderParams,
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.vision_language.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.models.vision_language.ideogram4.latent_norm import get_latent_norm
from dl_techniques.models.vision_language.ideogram4.scheduler import (
    LogitNormalSchedule,
    get_schedule_for_resolution,
    make_step_intervals,
)
from dl_techniques.models.vision_language.ideogram4.transformer import (
    Ideogram4Transformer,
    create_ideogram4_transformer,
)
from dl_techniques.models.vision_language.ideogram4.vae import (
    AutoEncoder,
    create_ideogram4_autoencoder,
)

# get_latent_norm() returns 128-element vectors keyed to the full config's
# in_channels; denorm applies only when in_channels matches.
_LATENT_NORM_CHANNELS = 128


def apply_cfg_blend(
    pos_v: keras.KerasTensor,
    neg_v: keras.KerasTensor,
    gw: float,
) -> keras.KerasTensor:
    """Blend positive and negative velocities: ``gw * pos + (1 - gw) * neg``.

    With ``gw == 1`` the result is the conditional branch alone; with
    ``gw == 0`` it is the unconditional branch alone.

    :param pos_v: Conditional (positive) velocity, shape ``(B, L_img, C)``.
    :param neg_v: Unconditional (negative) velocity, shape ``(B, L_img, C)``.
    :param gw: Scalar guidance weight for this step.
    :return: The blended velocity, same shape as the inputs.
    """
    return gw * pos_v + (1.0 - gw) * neg_v


class Ideogram4Pipeline:
    """End-to-end Ideogram4 image generation: Euler + CFG denoise, then decode.

    A plain orchestration class, not a `keras.Layer` or `keras.Model`, that
    holds the trained sub-models and the structural config. Conditioning is
    the precomputed `llm_features` tensor passed to :meth:`__call__`.

    Denoise and decode:

    .. code-block:: text

        llm_features [B,T,D]      noise z [B,N,C]
              │                         │
              ▼                         ▼
        ┌──────────────┐       Euler loop, num_steps
        │ _build_inputs │       ┌────────────────────┐
        └──────┬───────┘        │ transformer (pos)   │◄── llm_features
               │                │ transformer (neg)   │◄── zeroed feats
               ▼                │ cfg blend -> v       │
        position/segment/       │ z += v * (s - t)     │
        indicator ids           └──────────┬───────────┘
                                            ▼
                                   final latent z [B,N,C]
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ _decode        │
                                    │ unpatchify     │
                                    │ (optional)     │
                                    │ latent denorm  │
                                    │ VAE decode     │
                                    └───────┬───────┘
                                            ▼
                                  image [B,H,W,out_ch] in [0,1]

    :param transformer: The flow-matching DiT velocity predictor. Runs both
        the conditional and, by default, the unconditional CFG branches.
    :param autoencoder: The VAE; only :meth:`AutoEncoder.decode` is used.
    :param config: The transformer and pipeline :class:`Ideogram4Config`.
    :param ae_params: The VAE :class:`AutoEncoderParams`, driving the
        unpatchify and spatial-factor math.
    :param unconditional_transformer: Optional separate model for the
        negative CFG branch. Defaults to `None`, so the shared `transformer`
        runs both branches.
    :raises TypeError: If `transformer`, `autoencoder`, `config`, or
        `ae_params` are not of the expected types.
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

    @property
    def vae_upsample_factor(self) -> int:
        """VAE decode spatial-upsample factor, ``2 ** (len(ch_mult) - 1)``.

        The decoder upsamples once between every pair of resolution stages,
        so the latent-to-pixel ratio is ``2 ** (num_stages - 1)``. This is
        the actual factor; `config.ae_scale_factor` matches it only for the
        full preset.
        """
        return 2 ** (len(self.ae_params.ch_mult) - 1)

    @property
    def pixels_per_token_edge(self) -> int:
        """Pixel edge length covered by one image token: ``patch * vae_factor``."""
        return self.config.patch_size * self.vae_upsample_factor

    def _build_inputs(
        self,
        batch_size: int,
        num_text_tokens: int,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        """Build packed ``position_ids``, ``segment_ids``, and ``indicator``.

        One sample per row, no padding: `T` is fixed per batch, and each row
        holds a single attention segment. The packed sequence is
        ``[T text tokens][grid_h * grid_w image tokens]``.

        :param batch_size: Number of samples `B`.
        :param num_text_tokens: Conditioning length `T`, equal to `llm_features.shape[1]`.
        :param height: Target image height in pixels, divisible by :attr:`pixels_per_token_edge`.
        :param width: Target image width in pixels.
        :return: ``(position_ids, segment_ids, indicator, num_image_tokens, grid_h, grid_w)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]
        :raises ValueError: If `height` or `width` are not divisible by the
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

        # Image position ids are (t=0, h, w) offset so they never collide with text.
        hh, ww = np.meshgrid(
            np.arange(grid_h), np.arange(grid_w), indexing="ij"
        )
        image_pos = np.stack(
            [
                np.zeros(num_image, dtype=np.int32),
                hh.reshape(-1).astype(np.int32),
                ww.reshape(-1).astype(np.int32),
            ],
            axis=-1,
        )
        image_pos = image_pos + IMAGE_POSITION_OFFSET

        text_arange = np.arange(num_text_tokens, dtype=np.int32)
        text_pos = np.stack([text_arange] * 3, axis=-1)

        pos_single = np.concatenate([text_pos, image_pos], axis=0)
        position_ids = np.broadcast_to(
            pos_single[None], (batch_size, total_len, 3)
        ).astype(np.int32).copy()

        segment_ids = np.ones((batch_size, total_len), dtype=np.int32)

        indicator = np.empty((batch_size, total_len), dtype=np.int32)
        indicator[:, :num_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

        return position_ids, segment_ids, indicator, num_image, grid_h, grid_w

    def _decode(
        self,
        z: keras.KerasTensor,
        grid_h: int,
        grid_w: int,
    ) -> keras.KerasTensor:
        """Unpatchify the latent, denorm it if applicable, and VAE-decode.

        :param z: Final denoised latent, shape ``(B, num_image, in_channels)``.
        :param grid_h: Image-grid height in tokens.
        :param grid_w: Image-grid width in tokens.
        :return: The decoded image, shape ``(B, H, W, out_ch)``, values in ``[0, 1]``.
        """
        patch = self.config.patch_size
        in_channels = self.config.in_channels
        ae_channels = in_channels // (patch * patch)  # = z_channels

        # Latent-norm vectors are keyed to the full config's in_channels.
        if in_channels == _LATENT_NORM_CHANNELS:
            shift, scale = get_latent_norm()
            z = z * scale + shift
        else:
            logger.debug(
                "Skipping latent denorm: in_channels=%d != %d (latent_norm "
                "vectors apply to the full config only); pass-through.",
                in_channels,
                _LATENT_NORM_CHANNELS,
            )

        batch = keras.ops.shape(z)[0]
        z = keras.ops.reshape(
            z, (batch, grid_h, grid_w, patch, patch, ae_channels)
        )
        # Transpose so the reshape below merges (gh,p) into H and (gw,p) into W.
        z = keras.ops.transpose(z, (0, 1, 3, 2, 4, 5))
        z_img = keras.ops.reshape(
            z, (batch, grid_h * patch, grid_w * patch, ae_channels)
        )

        image = self.autoencoder.decode(z_img)
        image = keras.ops.clip(image, -1.0, 1.0)
        image = (image + 1.0) * 0.5
        return image

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
        """Run the Euler plus asymmetric-CFG denoise loop, then VAE-decode.

        :param llm_features: Precomputed conditioning, shape ``(B, T, llm_features_dim)``.
        :param height: Target image height in pixels.
        :param width: Target image width in pixels.
        :param num_steps: Number of Euler integration steps.
        :param guidance_scale: Constant CFG weight, used when `guidance_schedule` is `None`.
        :param guidance_schedule: Optional per-step CFG weights in loop-index
            order (length must equal `num_steps`); index 0 is the last step.
        :param mu: Logit-normal schedule mean (`known_mean`). Used when `schedule` is `None`.
        :param std: Logit-normal schedule standard deviation. Used when `schedule` is `None`.
        :param seed: Integer seed for the initial noise, for determinism.
        :param schedule: Optional prebuilt :class:`LogitNormalSchedule`; if
            `None`, one is built via :func:`get_schedule_for_resolution`.
        :return: The generated image, shape ``(B, height, width, out_ch)``, values in ``[0, 1]``.
        :raises ValueError: If `guidance_schedule`'s length disagrees with
            `num_steps`, or `height`/`width` are not patch-divisible.
        """
        batch_size = int(keras.ops.shape(llm_features)[0])
        num_text_tokens = int(keras.ops.shape(llm_features)[1])
        in_channels = self.config.in_channels
        llm_dim = self.config.llm_features_dim

        if schedule is None:
            schedule = get_schedule_for_resolution(
                (height, width), known_mean=mu, std=std
            )
        step_intervals = make_step_intervals(num_steps)

        if guidance_schedule is not None:
            if len(guidance_schedule) != num_steps:
                raise ValueError(
                    f"guidance_schedule has length {len(guidance_schedule)}, "
                    f"expected num_steps={num_steps}."
                )
            gw_per_step: List[float] = [float(g) for g in guidance_schedule]
        else:
            gw_per_step = [float(guidance_scale)] * num_steps

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

        # Conditional llm_features sit at text positions; image positions get zeros.
        text_feats = keras.ops.cast(llm_features, "float32")
        image_feat_pad = keras.ops.zeros(
            (batch_size, num_image, llm_dim), dtype="float32"
        )
        llm_features_full = keras.ops.concatenate(
            [text_feats, image_feat_pad], axis=1
        )

        z = keras.random.normal(
            (batch_size, num_image, in_channels), seed=seed, dtype="float32"
        )
        text_z_padding = keras.ops.zeros(
            (batch_size, num_text_tokens, in_channels), dtype="float32"
        )
        pos_z = keras.ops.concatenate([text_z_padding, z], axis=1)

        # Negative branch is image-only; conditioning is zeroed rather than dropped.
        neg_position_ids = position_ids[:, num_text_tokens:]
        neg_segment_ids = segment_ids[:, num_text_tokens:]
        neg_indicator = indicator[:, num_text_tokens:]
        neg_llm_features = keras.ops.zeros(
            (batch_size, num_image, llm_dim), dtype="float32"
        )

        neg_model = self.unconditional_transformer or self.transformer

        for i in range(num_steps - 1, -1, -1):
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-002: index step_intervals
            # by j = num_steps-1-i (not i) so t descends noise -> data as i falls.
            # guidance_schedule stays in loop-index order. See decisions.md.
            j = num_steps - 1 - i
            t_val = float(schedule(float(step_intervals[j])))
            s_val = float(schedule(float(step_intervals[j + 1])))
            gw_i = gw_per_step[i]

            # Shape (B, 1), not (B,): a B==1 rank-1 tensor would collapse to a
            # scalar and break ScalarSinusoidalEmbedding's trailing squeeze.
            t = keras.ops.full((batch_size, 1), t_val, dtype="float32")

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
            pos_v = pos_out[:, num_text_tokens:]

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

        return self._decode(z, grid_h, grid_w)

    @classmethod
    def from_config(
        cls,
        variant: str = "tiny",
        seed: Optional[int] = None,
    ) -> "Ideogram4Pipeline":
        """Build a fresh, untrained pipeline from a named preset.

        Constructs a new transformer and autoencoder from the `variant`
        preset. The resulting pipeline runs end to end but produces noise,
        since no weights are trained; useful for shape, finiteness, and
        determinism smoke tests.

        :param variant: One of the config presets (``"tiny"`` or ``"full"``).
        :param seed: Optional sampling seed forwarded to the VAE `Sampling`
            layer. Unused at decode, which is deterministic.
        :return: A constructed :class:`Ideogram4Pipeline`.
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
