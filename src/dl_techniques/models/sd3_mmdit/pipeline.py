"""SD3 end-to-end text-to-image inference pipeline (Keras 3 port).

Integrates the four already-built, separately-tested pieces -- the dual-stream
:class:`~dl_techniques.models.sd3_mmdit.transformer.SD3MMDiT`, the rectified-flow
:class:`~dl_techniques.models.sd3_mmdit.scheduler.FlowMatchEulerScheduler`, the
16-channel :class:`~dl_techniques.models.ideogram4.vae.AutoEncoder` (SD3 wrapper),
and the three from-scratch text encoders (CLIP / OpenCLIP / T5) -- into a single
plain-Python (NOT ``keras.Model``) inference object that runs prompt-conditioned
generation.

This file adds NO new Keras weights: it only orchestrates the existing
components. The riskiest part is the prompt-feature assembly (faithful to SD3's
``HandlePrompt`` / the diffusers ``SD3 encode_prompt``), whose dim contract must
line up exactly with the ``SD3MMDiTConfig``.

Prompt feature assembly (faithful to SD3 HandlePrompt / encode_prompt)
----------------------------------------------------------------------
SD3 conditions on three text towers:

- ``encoder_hidden_states`` (the SEQUENCE stream the joint attention attends to):

    1. concat the CLIP **penultimate** hidden states with the OpenCLIP
       **penultimate** along the FEATURE axis
       -> ``(B, L_clip, clip_dim + openclip_dim)``;
    2. zero-pad that on the feature axis up to the T5 feature dim
       -> ``(B, L_clip, t5_dim)``;
    3. concat with the T5 sequence along the SEQUENCE axis
       -> ``(B, L_clip + L_t5, t5_dim)``.

  ``t5_dim`` MUST equal ``config.joint_attention_dim`` (the MMDiT
  ``context_embedder`` consumes exactly this width).

- ``pooled_projections`` (the pooled vector summed into every block's
  conditioning): concat the CLIP **pooled** and OpenCLIP **pooled** along the
  feature axis -> ``(B, clip_pooled_dim + openclip_pooled_dim)``. This MUST
  equal ``config.pooled_projection_dim``.

Dim contract (fail-loud)
------------------------
For a runnable pipeline the encoder dims MUST satisfy::

    t5.embed_dim                      == config.joint_attention_dim
    clip.embed_dim + openclip.embed_dim == config.pooled_projection_dim

(The CLIP/OpenCLIP pooled width equals their ``embed_dim`` -- the
``text_projection`` is a square ``Dense(embed_dim)``.) :func:`assemble_prompt_features`
and :class:`SD3Pipeline.__init__` raise a clear ``ValueError`` when these do not
hold.

Timestep scaling (documented)
-----------------------------
The transformer's ``ScalarSinusoidalEmbedding`` uses ``input_range=(0, 1000)``
(SD3 convention), but the scheduler's continuous ``t`` lives in ``[0, 1]``. The
denoise loop therefore passes ``timestep = t * 1000`` to the transformer. See
:meth:`SD3Pipeline.generate`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from dl_techniques.utils.logger import logger
from dl_techniques.models.sd3_mmdit.config import SD3MMDiTConfig, get_sd3_config
from dl_techniques.models.sd3_mmdit.scheduler import FlowMatchEulerScheduler
from dl_techniques.models.sd3_mmdit.vae import denormalize_latent, create_sd3_vae
from dl_techniques.models.sd3_mmdit.transformer import SD3MMDiT, create_sd3_mmdit
from dl_techniques.models.sd3_mmdit.text_encoders import (
    CLIPTextEncoder,
    OpenCLIPTextEncoder,
    T5Encoder,
)

# Transformer timestep embedding range is (0, 1000); the scheduler's t is in
# [0, 1]. The denoise loop scales t -> t * TIMESTEP_SCALE before the call.
TIMESTEP_SCALE: float = 1000.0


# ---------------------------------------------------------------------
# Prompt feature assembly
# ---------------------------------------------------------------------


def assemble_prompt_features(
    clip_out: Dict[str, keras.KerasTensor],
    openclip_out: Dict[str, keras.KerasTensor],
    t5_out: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Build SD3 ``(encoder_hidden_states, pooled_projections)`` from the towers.

    Faithful to SD3's ``HandlePrompt`` / diffusers ``encode_prompt``:

    1. ``clip_penult`` ``(B, L_clip, clip_dim)`` and ``openclip_penult``
       ``(B, L_clip, openclip_dim)`` are concatenated on the FEATURE axis to
       ``(B, L_clip, clip_dim + openclip_dim)``.
    2. That is zero-padded on the FEATURE axis up to the T5 feature dim ``t5_dim``
       -> ``(B, L_clip, t5_dim)``.
    3. It is concatenated with the T5 sequence ``(B, L_t5, t5_dim)`` on the
       SEQUENCE axis -> ``encoder_hidden_states (B, L_clip + L_t5, t5_dim)``.
    4. ``pooled_projections`` = concat of the CLIP pooled ``(B, clip_dim)`` and
       OpenCLIP pooled ``(B, openclip_dim)`` on the feature axis.

    The CLIP/OpenCLIP sequence lengths must match (both are the same prompt
    token sequence); SD3 uses a shared 77-token CLIP context. The combined
    CLIP+OpenCLIP feature width must not exceed the T5 width (so the pad in step
    2 is well-defined).

    :param clip_out: ``CLIPTextEncoder`` output dict
        (``{"pooled", "last_hidden", "penultimate"}``).
    :param openclip_out: ``OpenCLIPTextEncoder`` output dict (same keys).
    :param t5_out: ``T5Encoder`` sequence output ``(B, L_t5, t5_dim)``.
    :return: ``(encoder_hidden_states, pooled_projections)``.
    :raises ValueError: If the combined CLIP+OpenCLIP feature width exceeds the
        T5 feature width (the pad-to-T5 step would be ill-defined).
    """
    clip_penult = clip_out["penultimate"]  # (B, L_clip, clip_dim)
    openclip_penult = openclip_out["penultimate"]  # (B, L_clip, openclip_dim)
    clip_pooled = clip_out["pooled"]  # (B, clip_dim)
    openclip_pooled = openclip_out["pooled"]  # (B, openclip_dim)

    clip_dim = clip_penult.shape[-1]
    openclip_dim = openclip_penult.shape[-1]
    t5_dim = t5_out.shape[-1]
    combined_clip_dim = clip_dim + openclip_dim

    if combined_clip_dim > t5_dim:
        raise ValueError(
            f"CLIP+OpenCLIP penultimate feature width "
            f"({clip_dim} + {openclip_dim} = {combined_clip_dim}) must be <= the "
            f"T5 feature width ({t5_dim}); SD3 zero-pads the CLIP context up to "
            f"the T5 width before concatenating along the sequence axis."
        )

    # 1. concat CLIP + OpenCLIP penultimate along the feature axis.
    clip_context = ops.concatenate(
        [clip_penult, openclip_penult], axis=-1
    )  # (B, L_clip, combined_clip_dim)

    # 2. zero-pad on the feature axis up to the T5 width.
    pad = t5_dim - combined_clip_dim
    if pad > 0:
        clip_context = ops.pad(
            clip_context, [[0, 0], [0, 0], [0, pad]]
        )  # (B, L_clip, t5_dim)

    # 3. concat with the T5 sequence along the SEQUENCE axis.
    encoder_hidden_states = ops.concatenate(
        [clip_context, t5_out], axis=1
    )  # (B, L_clip + L_t5, t5_dim)

    # 4. pooled = concat CLIP pooled + OpenCLIP pooled along the feature axis.
    pooled_projections = ops.concatenate(
        [clip_pooled, openclip_pooled], axis=-1
    )  # (B, clip_dim + openclip_dim)

    return encoder_hidden_states, pooled_projections


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


class SD3Pipeline:
    """SD3 text-to-image inference pipeline (plain Python, no new weights).

    Holds the five components and runs text-conditioned generation: encode the
    three text towers, assemble the SD3 prompt features, sample an initial
    latent, run the rectified-flow Euler denoise loop with the
    :class:`SD3MMDiT` velocity predictor, then VAE-decode the final latent.

    :param transformer: The :class:`SD3MMDiT` velocity predictor.
    :param vae: The 16-channel ``AutoEncoder`` (decode-only is used here).
    :param clip: The :class:`CLIPTextEncoder` tower.
    :param openclip: The :class:`OpenCLIPTextEncoder` tower.
    :param t5: The :class:`T5Encoder` tower.
    :param scheduler: The :class:`FlowMatchEulerScheduler`.
    :raises ValueError: If the encoder dims violate the SD3 dim contract against
        ``transformer.config`` (see module docstring).
    """

    def __init__(
        self,
        transformer: SD3MMDiT,
        vae: Any,
        clip: CLIPTextEncoder,
        openclip: OpenCLIPTextEncoder,
        t5: T5Encoder,
        scheduler: FlowMatchEulerScheduler,
    ) -> None:
        self.transformer = transformer
        self.vae = vae
        self.clip = clip
        self.openclip = openclip
        self.t5 = t5
        self.scheduler = scheduler
        self.config: SD3MMDiTConfig = transformer.config

        self._validate_dim_contract()

    def _validate_dim_contract(self) -> None:
        """Fail loud if the encoder dims do not satisfy the SD3 dim contract.

        Requires::

            t5.embed_dim                        == config.joint_attention_dim
            clip.embed_dim + openclip.embed_dim == config.pooled_projection_dim

        :raises ValueError: If either equality fails.
        """
        cfg = self.config
        if self.t5.embed_dim != cfg.joint_attention_dim:
            raise ValueError(
                f"T5 embed_dim ({self.t5.embed_dim}) must equal "
                f"config.joint_attention_dim ({cfg.joint_attention_dim}): the "
                f"encoder_hidden_states feature width feeds the MMDiT "
                f"context_embedder."
            )
        pooled = self.clip.embed_dim + self.openclip.embed_dim
        if pooled != cfg.pooled_projection_dim:
            raise ValueError(
                f"CLIP embed_dim + OpenCLIP embed_dim "
                f"({self.clip.embed_dim} + {self.openclip.embed_dim} = {pooled}) "
                f"must equal config.pooled_projection_dim "
                f"({cfg.pooled_projection_dim}): the concatenated pooled vector "
                f"feeds the MMDiT combined timestep-text embedding."
            )

    def encode_prompt(
        self,
        clip_token_ids: keras.KerasTensor,
        openclip_token_ids: keras.KerasTensor,
        t5_token_ids: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Run the three towers and assemble the SD3 prompt features.

        :param clip_token_ids: ``(B, L_clip)`` integer ids for the CLIP tower.
        :param openclip_token_ids: ``(B, L_clip)`` integer ids for OpenCLIP.
        :param t5_token_ids: ``(B, L_t5)`` integer ids for the T5 tower.
        :param attention_mask: Optional ``(B, L_t5)`` 1/0 padding mask applied to
            the T5 tower (CLIP/OpenCLIP use their causal mask only).
        :param training: Forwarded to the encoders.
        :return: ``(encoder_hidden_states, pooled_projections)``.
        """
        clip_out = self.clip(clip_token_ids, training=training)
        openclip_out = self.openclip(openclip_token_ids, training=training)
        t5_out = self.t5(
            t5_token_ids, attention_mask=attention_mask, training=training
        )
        return assemble_prompt_features(clip_out, openclip_out, t5_out)

    def generate(
        self,
        clip_token_ids: keras.KerasTensor,
        openclip_token_ids: keras.KerasTensor,
        t5_token_ids: keras.KerasTensor,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Text-conditioned image generation (encode -> denoise -> decode).

        Steps:

        1. Encode the three text towers and assemble
           ``encoder_hidden_states`` + ``pooled_projections``.
        2. Sample an initial latent ``z ~ N(0, 1)`` of shape
           ``(B, sample_size, sample_size, in_channels)`` (``B`` from the token
           batch) via ``keras.random.normal`` with ``seed``.
        3. Build the descending time grid ``ts`` of length
           ``num_inference_steps + 1`` via the scheduler.
        4. Euler denoise loop: for each ``(t, t_next)`` pair, predict the
           velocity with the transformer (passing ``timestep = t * 1000`` to
           match its ``input_range=(0, 1000)`` embedding) and take one
           ``scheduler.euler_step``.
        5. Denormalize the final latent and VAE-decode to an image.

        :param clip_token_ids: ``(B, L_clip)`` CLIP token ids.
        :param openclip_token_ids: ``(B, L_clip)`` OpenCLIP token ids.
        :param t5_token_ids: ``(B, L_t5)`` T5 token ids.
        :param num_inference_steps: Number of Euler integration steps.
        :param seed: Optional RNG seed for the initial latent (determinism).
        :param attention_mask: Optional ``(B, L_t5)`` T5 padding mask.
        :return: Decoded image ``(B, H, W, 3)``.
        """
        cfg = self.config

        # --- 1. encode + assemble prompt features ----------------------
        encoder_hidden_states, pooled_projections = self.encode_prompt(
            clip_token_ids,
            openclip_token_ids,
            t5_token_ids,
            attention_mask=attention_mask,
            training=False,
        )

        # Batch size from the token ids (static where known, else dynamic).
        batch = clip_token_ids.shape[0]
        if batch is None:
            batch = int(ops.shape(clip_token_ids)[0])

        # --- 2. sample initial latent z ~ N(0, 1) ----------------------
        latent_shape = (batch, cfg.sample_size, cfg.sample_size, cfg.in_channels)
        rng = keras.random.SeedGenerator(seed) if seed is not None else None
        z = keras.random.normal(latent_shape, seed=rng)

        # --- 3. descending time grid -----------------------------------
        ts = self.scheduler.timesteps(num_inference_steps)  # (N+1,) float32

        logger.info(
            "SD3Pipeline.generate: B=%d, steps=%d, latent=%s, "
            "encoder_hidden_states=%s, pooled=%s",
            batch,
            num_inference_steps,
            latent_shape,
            tuple(encoder_hidden_states.shape),
            tuple(pooled_projections.shape),
        )

        # --- 4. Euler denoise loop -------------------------------------
        for i in range(num_inference_steps):
            t = float(ts[i])
            t_next = float(ts[i + 1])
            # Transformer expects timestep in [0, 1000] (ScalarSinusoidalEmbedding
            # input_range=(0, 1000)); the scheduler t is in [0, 1]. Scale here.
            timestep = ops.full((batch,), t * TIMESTEP_SCALE, dtype="float32")
            v = self.transformer(
                {
                    "latent": z,
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                    "timestep": timestep,
                },
                training=False,
            )
            z = self.scheduler.euler_step(z, v, t, t_next)

        # --- 5. denormalize + VAE decode -------------------------------
        z = denormalize_latent(z)
        image = self.vae.decode(z, training=False)  # (B, H, W, 3)
        return image


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_sd3_pipeline(
    variant: str = "tiny",
    seed: Optional[int] = None,
) -> SD3Pipeline:
    """Build a full :class:`SD3Pipeline` from matching presets (no weights).

    Constructs all five components so the dim contract holds for ``variant``:

    - transformer + VAE come from :func:`get_sd3_config` / the SD3 factories.
    - The three text encoders are built TINY with dims picked to satisfy the
      contract. For the ``"tiny"`` config (``joint_attention_dim=512``,
      ``pooled_projection_dim=256``): T5 ``embed_dim=512``, CLIP ``embed_dim=128``,
      OpenCLIP ``embed_dim=128`` (so ``128 + 128 = 256``). For ``"full"`` the
      encoders are built at the SD3 reference widths (T5 4096, CLIP 768,
      OpenCLIP 1280 -> ``768 + 1280 = 2048``).

    The encoders are intentionally shallow (few layers / small vocab) here -- the
    pipeline adds NO trained weights and exists to exercise the integration; a
    real deployment would supply the full-depth, weight-loaded towers.

    :param variant: Config preset (``"tiny"`` or ``"full"``).
    :param seed: Optional seed forwarded to the VAE ``Sampling`` layer.
    :return: A constructed :class:`SD3Pipeline`.
    :raises ValueError: If ``variant`` has no matching encoder-dim recipe.
    """
    config, _ = get_sd3_config(variant)

    transformer = create_sd3_mmdit(variant)
    vae = create_sd3_vae(variant=variant, sampling_seed=seed)
    scheduler = FlowMatchEulerScheduler()

    # Encoder dim recipes chosen so the dim contract holds for each preset.
    if variant == "tiny":
        clip_dim, openclip_dim, t5_dim = 128, 128, 512
        clip_layers = openclip_layers = t5_layers = 2
        clip_heads = openclip_heads = 4
        t5_heads = 8
        t5_ff = 512
        vocab = 512
        max_seq = 32
    elif variant == "full":
        clip_dim, openclip_dim, t5_dim = 768, 1280, 4096
        clip_layers, openclip_layers, t5_layers = 12, 32, 24
        clip_heads, openclip_heads = 12, 16
        t5_heads = 64
        t5_ff = 10240
        vocab = 49408
        max_seq = 77
    else:
        raise ValueError(
            f"create_sd3_pipeline has no encoder-dim recipe for variant "
            f"'{variant}'. Supported: 'tiny', 'full'."
        )

    # Sanity: the recipe must satisfy the contract before SD3Pipeline checks it.
    if t5_dim != config.joint_attention_dim or \
            clip_dim + openclip_dim != config.pooled_projection_dim:
        raise ValueError(
            f"Internal recipe error for variant '{variant}': "
            f"t5_dim={t5_dim} vs joint_attention_dim={config.joint_attention_dim}, "
            f"clip_dim+openclip_dim={clip_dim + openclip_dim} vs "
            f"pooled_projection_dim={config.pooled_projection_dim}."
        )

    clip = CLIPTextEncoder(
        vocab_size=vocab,
        embed_dim=clip_dim,
        num_layers=clip_layers,
        num_heads=clip_heads,
        max_seq_len=max_seq,
        act_fn="quick_gelu",
        name="clip_text_encoder",
    )
    openclip = OpenCLIPTextEncoder(
        vocab_size=vocab,
        embed_dim=openclip_dim,
        num_layers=openclip_layers,
        num_heads=openclip_heads,
        max_seq_len=max_seq,
        name="openclip_text_encoder",
    )
    t5 = T5Encoder(
        vocab_size=vocab,
        embed_dim=t5_dim,
        num_layers=t5_layers,
        num_heads=t5_heads,
        ff_dim=t5_ff,
        name="t5_encoder",
    )

    logger.info(
        "Created SD3Pipeline variant='%s': T5 embed_dim=%d (==joint_attention_dim"
        "=%d), CLIP=%d + OpenCLIP=%d (==pooled_projection_dim=%d)",
        variant,
        t5_dim,
        config.joint_attention_dim,
        clip_dim,
        openclip_dim,
        config.pooled_projection_dim,
    )

    return SD3Pipeline(
        transformer=transformer,
        vae=vae,
        clip=clip,
        openclip=openclip,
        t5=t5,
        scheduler=scheduler,
    )
