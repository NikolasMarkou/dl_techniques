"""Lossless channels-last packing of a token sequence into the bridge tensor.

A diffusion process can only carry one tensor. To diffuse text and image into
each other, the caption's token embeddings are re-tiled, not projected or
compressed, into a latent-shaped tensor, so both endpoints of the bridge
live in one space. The map is a pure permutation, so it is exactly
invertible: whatever the sampler produces reads back as tokens with no
reconstruction error of its own, and `SharedTokenDecoder` sees only the
sampler's error.

The permutation, for the ``sd`` preset::

    token_flat                      (B, 64 tokens x 64 dims = 4096)
      |
      |  split each token into patches_per_token = 4 payloads of 16 numbers
      v
    token payloads                  (B, 64, 4, 16)
      |
      |  positions = arange(64 * 4) reshaped (64, 4)   [row_major = identity]
      v
    patch payloads                  (B, 256 patches, 16)
      |
      |  reshape (B, h=16, w=16, p=2, q=2, c=4)
      |  transpose (0, 1, 3, 2, 4, 5)      -> (B, h, p, w, q, c)
      |  reshape (B, h*p = 32, w*p = 32, c = 4)
      v
    bridge                          (B, 32, 32, 4)

    one patch, payload index -> pixel offset (channel varies fastest)

        payload   0  1  2  3 | 4  5  6  7 | 8  9 10 11 |12 13 14 15
        (p, q)    (0,0)      | (0,1)      | (1,0)      | (1,1)
        channel   0  1  2  3 | 0  1  2  3 | 0  1  2  3 | 0  1  2  3

Upstream is PyTorch and spells the same permutation as
``einsum("nhwpqc->nchpwq")``; here the channel axis is already last, so the
whole thing is one axis swap, ``transpose(0, 1, 3, 2, 4, 5)``, matching the
unpatchify in ``sd3_mmdit/transformer.py``. The width dimension is written
``w * p`` rather than ``h * p`` even though every shipped preset is square,
so a future non-square preset does not silently transpose.

References:
    - Upstream ``token_bridge.py``.
    - ``src/dl_techniques/models/vision_language/sd3_mmdit/transformer.py``,
      the channels-last unpatchify this chain matches.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger

from .config import TOKEN_LAYOUTS, BridgeConfig


def _patch_positions(config: BridgeConfig, layout: str = "row_major") -> np.ndarray:
    """Patch index occupied by each ``(token, sub-patch)`` slot.

    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param layout: Registered layout name; ``row_major`` is the identity.
    :type layout: str
    :return: ``(token_seq_len * patches_per_token,)`` int64 permutation.
    :rtype: np.ndarray
    :raises ValueError: If ``layout`` is not in :data:`TOKEN_LAYOUTS`.
    """
    if layout not in TOKEN_LAYOUTS:
        raise ValueError(
            f"Unknown token layout '{layout}'. Available: {list(TOKEN_LAYOUTS)}"
        )
    positions = np.arange(
        config.token_seq_len * config.patches_per_token, dtype="int64"
    )
    return positions.reshape(config.token_seq_len, config.patches_per_token).reshape(-1)


def _unpatchify_payloads(payloads: Any, config: BridgeConfig) -> Any:
    """``(B, num_patches, patch_payload_dim)`` -> ``(B, H, W, C)``.

    :param payloads: Per-patch payload tensor.
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :return: The channels-last bridge tensor.
    """
    batch = keras.ops.shape(payloads)[0]
    p = config.patch_size
    h, w, c = config.patch_h, config.patch_w, config.channels
    x = keras.ops.reshape(payloads, (batch, h, w, p, p, c))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (batch, h * p, w * p, c))


def _patchify_bridge(x: Any, config: BridgeConfig) -> Any:
    """``(B, H, W, C)`` -> ``(B, num_patches, patch_payload_dim)``.

    Exact inverse of :func:`_unpatchify_payloads` -- the transpose only swaps
    axes 2 and 3, so it is its own inverse.

    :param x: Channels-last bridge tensor.
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :return: Per-patch payload tensor.
    """
    batch = keras.ops.shape(x)[0]
    p = config.patch_size
    h, w, c = config.patch_h, config.patch_w, config.channels
    x = keras.ops.reshape(x, (batch, h, p, w, p, c))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (batch, h * w, p * p * c))


def token_flat_to_bridge(
    token_flat: Any, config: BridgeConfig, layout: str = "row_major"
) -> Any:
    """Pack flattened token embeddings into the channels-last bridge tensor.

    :param token_flat: ``(B, token_flat_dim)`` token embeddings.
    :param config: Bridge geometry; validated on entry.
    :type config: BridgeConfig
    :param layout: Registered token->patch layout.
    :type layout: str
    :return: ``(B, H, W, C)`` bridge tensor.
    """
    config.validate()
    batch = keras.ops.shape(token_flat)[0]
    payloads = keras.ops.reshape(
        token_flat,
        (
            batch,
            config.token_seq_len * config.patches_per_token,
            config.patch_payload_dim,
        ),
    )
    positions = _patch_positions(config, layout)
    # Upstream scatters (`patch_payloads[:, positions] = payloads`); the gather
    # form of the same scatter is `argsort(positions)`, and gathering is what
    # keras.ops offers backend-agnostically.
    patch_payloads = keras.ops.take(payloads, np.argsort(positions), axis=1)
    return _unpatchify_payloads(patch_payloads, config)


def bridge_to_token_flat(
    x: Any, config: BridgeConfig, layout: str = "row_major"
) -> Any:
    """Unpack a bridge tensor back into flattened token embeddings.

    Exact inverse of :func:`token_flat_to_bridge`.

    :param x: ``(B, H, W, C)`` bridge tensor.
    :param config: Bridge geometry; validated on entry.
    :type config: BridgeConfig
    :param layout: Registered token->patch layout.
    :type layout: str
    :return: ``(B, token_flat_dim)`` token embeddings.
    """
    config.validate()
    batch = keras.ops.shape(x)[0]
    patch_payloads = _patchify_bridge(x, config)
    positions = _patch_positions(config, layout)
    token_payloads = keras.ops.take(patch_payloads, positions, axis=1)
    return keras.ops.reshape(token_payloads, (batch, config.token_flat_dim))


def compute_token_norms(
    x_bridge: Any,
    config: BridgeConfig,
    token_scale: Optional[float] = None,
    layout: str = "row_major",
) -> Any:
    """Per-token L2 norm of the *unscaled* embeddings carried by a bridge tensor.

    Real token embeddings are unit-norm once divided by ``token_scale``;
    padding tokens are exactly zero.

    :param x_bridge: ``(B, H, W, C)`` bridge tensor.
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param token_scale: Divisor applied before the norm; defaults to
        ``config.token_scale``.
    :type token_scale: Optional[float]
    :param layout: Registered token->patch layout.
    :type layout: str
    :return: ``(B, token_seq_len)`` norms.
    """
    scale = config.token_scale if token_scale is None else float(token_scale)
    token_flat = bridge_to_token_flat(x_bridge, config, layout=layout)
    batch = keras.ops.shape(token_flat)[0]
    tokens = keras.ops.reshape(
        token_flat, (batch, config.token_seq_len, config.token_emb_dim)
    )
    tokens = keras.ops.divide(tokens, keras.ops.cast(scale, tokens.dtype))
    return keras.ops.sqrt(keras.ops.sum(keras.ops.square(tokens), axis=-1))


def _first_true_index(mask: Any) -> Any:
    """First ``True`` column per row of a ``(B, T)`` boolean mask, else ``T``.

    :param mask: ``(B, T)`` boolean tensor.
    :return: ``(B,)`` int32 indices; ``T`` for an all-``False`` row.
    """
    length = keras.ops.shape(mask)[1]
    idx = keras.ops.arange(length, dtype="int32")
    idx = keras.ops.broadcast_to(
        keras.ops.expand_dims(idx, 0), keras.ops.shape(mask)
    )
    filled = keras.ops.where(mask, idx, keras.ops.cast(length, "int32"))
    return keras.ops.min(filled, axis=1)


def norm_based_token_stops(
    x_bridge: Any,
    config: BridgeConfig,
    token_scale: Optional[float] = None,
    zero_thresh: float = 0.1,
    layout: str = "row_major",
) -> Tuple[Any, Any]:
    """Locate the first padding token of every row by embedding norm.

    :param x_bridge: ``(B, H, W, C)`` bridge tensor.
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param token_scale: See :func:`compute_token_norms`.
    :type token_scale: Optional[float]
    :param zero_thresh: Norms below this count as padding.
    :type zero_thresh: float
    :param layout: Registered token->patch layout.
    :type layout: str
    :return: ``(stops, norms)`` -- ``(B,)`` stop indices (``token_seq_len`` when a
        row has no padding at all) and the ``(B, token_seq_len)`` norms.
    :rtype: Tuple[Any, Any]
    """
    norms = compute_token_norms(
        x_bridge, config, token_scale=token_scale, layout=layout
    )
    stops = _first_true_index(keras.ops.less(norms, zero_thresh))
    return stops, norms


def pad_id_token_stops(pred_ids: Any, pad_id: int) -> Any:
    """Locate the first padding token of every row by token id.

    :param pred_ids: ``(B, T)`` integer token ids.
    :param pad_id: The padding id.
    :type pad_id: int
    :return: ``(B,)`` stop indices; ``T`` when the row holds no padding.
    """
    return _first_true_index(keras.ops.equal(pred_ids, int(pad_id)))


def prepare_bridge_batch(
    batch: Dict[str, Any],
    config: BridgeConfig,
    layout: str = "row_major",
    seed: Optional[int] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Map one dataset batch to the five tensors the bridge process consumes.

    :param batch: Dict with ``"latent"`` ``(B, H, W, C)``, ``"text_token_emb"``
        ``(B, token_flat_dim)`` and ``"prompt_kind_label"`` ``(B,)``.
    :type batch: Dict[str, Any]
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param layout: Registered token->patch layout.
    :type layout: str
    :param seed: Seed for the ablation noise draws, when either ``*_as_noise``
        flag is set.
    :type seed: Optional[int]
    :return: ``(x_0_process, x_1_process, y, x_0, x_1)``. The trailing pair is
        always the *real* endpoints, so a caller can still condition on the
        genuine text/image even when the process runs on noise.
    :rtype: Tuple[Any, Any, Any, Any, Any]
    """
    x_1 = keras.ops.cast(batch["latent"], "float32")
    x_0 = token_flat_to_bridge(
        keras.ops.cast(batch["text_token_emb"], "float32"), config, layout=layout
    )
    y = keras.ops.cast(batch["prompt_kind_label"], "int32")

    x_0_process, x_1_process = x_0, x_1
    if config.text_as_noise:
        logger.debug("bit_diffusion: text endpoint replaced by noise (text_as_noise)")
        x_0_process = keras.random.normal(keras.ops.shape(x_0), dtype=x_0.dtype, seed=seed)
    if config.image_as_noise:
        logger.debug("bit_diffusion: image endpoint replaced by noise (image_as_noise)")
        x_1_process = keras.random.normal(keras.ops.shape(x_1), dtype=x_1.dtype, seed=seed)
    return x_0_process, x_1_process, y, x_0, x_1
