"""SD3 16-channel KL-VAE, built by :func:`create_sd3_vae`, as a thin wrapper
over the ideogram4 AutoEncoder.

This module reuses
:class:`~dl_techniques.models.vision_language.ideogram4.vae.AutoEncoder` with
``z_channels=16`` rather than re-implementing an encoder/decoder, and adds
only the SD3-specific latent-normalization convention: a scalar
``scaling_factor``/``shift_factor`` pair, unlike ideogram4's per-channel
``LATENT_SHIFT``/``LATENT_SCALE`` vectors (see :data:`SD3_SCALING_FACTOR`).
:class:`SD3VAE` bundles the reused autoencoder with the normalize/denormalize
helpers for the inference pipeline. For the SD3 tiny preset
(``ch=32, ch_mult=(1, 2)``) the VAE downsamples by a factor of 2: a
``(B, 32, 32, 3)`` image encodes to a ``(B, 16, 16, 16)`` latent.

References:
    - Esser et al., 2024. Scaling Rectified Flow Transformers for
      High-Resolution Image Synthesis (SD3). (https://arxiv.org/abs/2403.03206)
    - Rombach et al., 2022. High-Resolution Image Synthesis with Latent
      Diffusion Models. (https://arxiv.org/abs/2112.10752)
"""

from __future__ import annotations

import keras
from typing import Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from dl_techniques.models.vision_language.ideogram4.vae import AutoEncoder
from dl_techniques.models.vision_language.ideogram4.config import AutoEncoderParams
from dl_techniques.models.vision_language.sd3_mmdit.config import get_sd3_config

# ---------------------------------------------------------------------
# SD3 scalar latent-normalization constants
# ---------------------------------------------------------------------

# DECISION plan_2026-06-12_dfce0712/D-008: SD3 normalizes latents with a scalar
# scale/shift pair, never ideogram4's per-channel LATENT_SHIFT/LATENT_SCALE -- those vectors target a different (patchified, 32-channel) latent shape. See decisions.md.
# DECISION plan-2026-08-18T140459-7991552f/D-058: these are SD3's own constants,
# not SDXL's 0.13025/0.0 pair -- that shipped here until 2026-08-19 and left latents ~12x under-scaled. Verified against SD3/SD3.5 vae/config.json. See decisions.md.
SD3_SCALING_FACTOR: float = 1.5305
SD3_SHIFT_FACTOR: float = 0.0609


def normalize_latent(z: keras.KerasTensor) -> keras.KerasTensor:
    """Map a raw VAE latent into SD3 diffusion space: ``(z - shift) * scale``.

    Matches diffusers' SD3 encode-time normalization
    (``latents = (latents - shift_factor) * scaling_factor``).

    :param z: Raw VAE-space latent (e.g. from ``AutoEncoder.sample``).
    :return: Diffusion-space latent the MMDiT operates on.
    """
    return (z - SD3_SHIFT_FACTOR) * SD3_SCALING_FACTOR


def denormalize_latent(z_norm: keras.KerasTensor) -> keras.KerasTensor:
    """Map an SD3 diffusion-space latent back to VAE space: ``z / scale + shift``.

    Inverse of :func:`normalize_latent`; matches diffusers' SD3 decode-time
    denormalization (``latents = latents / scaling_factor + shift_factor``).

    :param z_norm: Diffusion-space latent (MMDiT output / sampler state).
    :return: VAE-space latent ready for ``AutoEncoder.decode``.
    """
    return z_norm / SD3_SCALING_FACTOR + SD3_SHIFT_FACTOR


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_sd3_vae(
    params: Optional[AutoEncoderParams | str] = None,
    variant: str = "tiny",
    sampling_seed: Optional[int] = None,
) -> AutoEncoder:
    """Build the SD3 16-channel KL-VAE as a reused ideogram4 ``AutoEncoder``.

    The first positional argument is ergonomically overloaded: pass an explicit
    :class:`AutoEncoderParams`, OR pass a preset-name string (e.g.
    ``create_sd3_vae("tiny")``) which is treated as ``variant``, OR omit it and
    use the ``variant`` keyword.

    :param params: Either an explicit ``AutoEncoderParams``, a preset-name
        string (used as ``variant``), or ``None``. When ``None`` (or a string),
        the parameters are derived from :func:`get_sd3_config` (which already
        sets ``z_channels=16`` and validates GroupNorm divisibility).
    :param variant: Preset name used when no explicit ``AutoEncoderParams`` is
        given (``"tiny"`` or ``"full"``).
    :param sampling_seed: Optional seed forwarded to the KL ``Sampling`` layer.
    :return: The constructed (un-built) ``AutoEncoder`` (``z_channels=16``).
    """
    # A first positional string is treated as the variant name.
    if isinstance(params, str):
        variant = params
        params = None

    source = "explicit-params" if params is not None else variant
    if params is None:
        _, params = get_sd3_config(variant)

    logger.info(
        "Creating SD3 VAE (reused ideogram4 AutoEncoder) variant='%s': "
        "z_channels=%d, ch=%d, ch_mult=%s, num_res_blocks=%d, resolution=%d "
        "(SD3 scalar latent-norm: scale=%.5f, shift=%.5f)",
        source,
        params.z_channels,
        params.ch,
        params.ch_mult,
        params.num_res_blocks,
        params.resolution,
        SD3_SCALING_FACTOR,
        SD3_SHIFT_FACTOR,
    )
    return AutoEncoder(params=params, sampling_seed=sampling_seed)


class SD3VAE:
    """Plain Python bundle of an ``AutoEncoder`` and the SD3 latent-norm helpers.

    Not a ``keras.Model``: the reused :class:`AutoEncoder` is already a
    serializable Keras model, saved and loaded directly via
    ``self.autoencoder``. This class only pairs it with the SD3 encode/decode
    latent-norm convention for the inference pipeline.

    :param autoencoder: The reused 16-channel ``AutoEncoder``.
    """

    def __init__(self, autoencoder: AutoEncoder) -> None:
        self.autoencoder = autoencoder

    @classmethod
    def from_variant(
        cls,
        variant: str = "tiny",
        sampling_seed: Optional[int] = None,
    ) -> "SD3VAE":
        """Build from a preset name via :func:`create_sd3_vae`."""
        return cls(create_sd3_vae(variant=variant, sampling_seed=sampling_seed))

    def encode_to_latent(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Image -> SD3 diffusion-space latent: encode -> sample -> normalize.

        :param x: Input image ``(B, H, W, in_channels)``.
        :param training: Forwarded to the encoder.
        :return: Normalized (diffusion-space) latent ``(B, H', W', 16)``.
        """
        z_mean, z_log_var = self.autoencoder.encode(x, training=training)
        z = self.autoencoder.sample(z_mean, z_log_var)
        return normalize_latent(z)

    def decode_from_latent(
        self,
        z_norm: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """SD3 diffusion-space latent -> image: denormalize -> decode.

        :param z_norm: Diffusion-space latent ``(B, H', W', 16)``.
        :param training: Forwarded to the decoder.
        :return: Reconstructed image ``(B, H, W, out_ch)``.
        """
        z = denormalize_latent(z_norm)
        return self.autoencoder.decode(z, training=training)
