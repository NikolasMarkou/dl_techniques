"""SD3 16-channel KL-VAE: a thin REUSE wrapper over the ideogram4 AutoEncoder.

This module does NOT re-implement an encoder/decoder. It reuses
:class:`~dl_techniques.models.ideogram4.vae.AutoEncoder` (a fully-tested Flux2
KL-VAE) with ``z_channels=16`` and supplies only the two pieces that are
SD3-specific:

1. The SD3 latent-normalization convention -- a SCALAR ``scaling_factor`` /
   ``shift_factor`` (NOT ideogram4's per-channel ``LATENT_SHIFT`` /
   ``LATENT_SCALE`` vectors; see :data:`SD3_SCALING_FACTOR` and D-008).
2. A factory :func:`create_sd3_vae` returning the reused ``AutoEncoder`` built
   from the SD3 ``(z_channels=16)`` preset, plus a thin convenience class
   :class:`SD3VAE` bundling the AutoEncoder with the normalize/denormalize
   helpers.

Reuse map (this package's VAE story): architecture = ideogram4 ``AutoEncoder``
(drop-in, ``z_channels=16``); latent-norm = SD3 scalar (overridden here).

API of the reused AutoEncoder (verified against ``ideogram4/vae.py``):

- ``encode(x) -> (z_mean, z_log_var)``     -- deterministic; each
  ``(B, H', W', 16)``.
- ``sample(z_mean, z_log_var) -> z``       -- stochastic KL reparameterization.
- ``decode(z) -> image``                   -- ``(B, H', W', out_ch)``.

For the SD3 tiny preset (``ch=32, ch_mult=(1, 2)``) the VAE downsamples by
``2 ** (len(ch_mult) - 1) = 2``: a ``(B, 32, 32, 3)`` image encodes to a
``(B, 16, 16, 16)`` latent.
"""

from __future__ import annotations

import keras
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# Reuse the ideogram4 AutoEncoder + its parameter dataclass -- do NOT
# re-implement (D-002). The SD3 config pairs an AutoEncoderParams with
# z_channels=16 already (see config.PRESETS / get_sd3_config).
from dl_techniques.models.ideogram4.vae import AutoEncoder
from dl_techniques.models.ideogram4.config import AutoEncoderParams
from dl_techniques.models.sd3_mmdit.config import get_sd3_config

# ---------------------------------------------------------------------
# SD3 scalar latent-normalization constants
# ---------------------------------------------------------------------

# DECISION plan_2026-06-12_dfce0712/D-008: SD3 normalizes VAE latents with a
# SCALAR scaling_factor / shift_factor, NOT ideogram4's 128-element per-channel
# LATENT_SHIFT / LATENT_SCALE vectors. Do NOT import or reuse ideogram4's
# latent_norm.py here: those vectors were derived for z_channels=32 *patchified*
# latents (128 = 32 * 2**2) and are dimensionally and semantically wrong for
# SD3's z_channels=16 spatial latent. SD3/SDXL use a single scalar (0.13025)
# that diffusers applies as `(latents - shift) * scale` on encode and
# `latents / scale + shift` on decode. We reuse the AutoEncoder *architecture*
# and replace the latent-norm *convention*. See decisions.md D-008.
SD3_SCALING_FACTOR: float = 0.13025
SD3_SHIFT_FACTOR: float = 0.0


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
    # Allow create_sd3_vae("tiny") -- first positional str is the variant.
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


# ---------------------------------------------------------------------
# Thin convenience wrapper (NOT a keras.Model -- avoids a redundant
# serializable wrapper over an already-serializable AutoEncoder)
# ---------------------------------------------------------------------


class SD3VAE:
    """Thin, non-Keras bundle of an ``AutoEncoder`` + SD3 latent-norm helpers.

    This is a plain Python convenience holder, NOT a ``keras.Model``: the reused
    :class:`AutoEncoder` is already a serializable Keras model (save/load it
    directly via ``self.autoencoder``). Wrapping it in a second serializable
    Model would be a redundant abstraction (earned-abstraction); this class
    exists only to pair the autoencoder with the SD3 encode/decode latent-norm
    convention for the inference pipeline.

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
