"""
Lossless channels-last packing of a token sequence into the bridge tensor.

A diffusion process carries one tensor, so the caption's token embeddings are
re-tiled, not projected or compressed, into a latent-shaped tensor and both
endpoints of the bridge live in one space. The map is a permutation, so
:func:`bridge_to_token_flat` inverts :func:`token_flat_to_bridge` exactly and
``SharedTokenDecoder`` sees only the sampler's error. Upstream spells it
``einsum("nhwpqc->nchpwq")``; channels-last, it is one axis swap matching the
unpatchify in ``sd3_mmdit/transformer.py``, with the width written ``w * p`` so a
non-square preset would not transpose. The module also finds where padding
starts, by norm or by id, and turns a dataset batch into bridge endpoints.
Packing for the ``sd`` preset:

.. code-block:: text

    token_flat [B, 4096]                64 tokens x 64 dims
             │
             ▼
    token payloads [B, 64, 4, 16]       4 payloads of 16 per token
             │
             ▼
    patch payloads [B, 256, 16]         layout gather; identity
             │
             ▼
    [B, h=16, w=16, p=2, q=2, c=4]
             │  transpose (0, 1, 3, 2, 4, 5)
             ▼
    [B, h, p, w, q, c]
             │
             ▼
    bridge [B, 32, 32, 4]               h * p by w * p

One patch, payload index to pixel offset (channel varies fastest):

.. code-block:: text

    payload   0  1  2  3 | 4  5  6  7 | 8  9 10 11 |12 13 14 15
    (p, q)    (0,0)      | (0,1)      | (1,0)      | (1,1)
    channel   0  1  2  3 | 0  1  2  3 | 0  1  2  3 | 0  1  2  3

References:
    - Upstream ``token_bridge.py``.
    - ``src/dl_techniques/models/vision_language/sd3_mmdit/transformer.py``,
      the channels-last unpatchify this chain matches.
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .config import TOKEN_LAYOUTS, BridgeConfig

# ---------------------------------------------------------------------------

def _patch_positions(config: BridgeConfig, layout: str = "row_major") -> np.ndarray:
    """Return the patch index occupied by each ``(token, sub-patch)`` slot.

    The result is the identity permutation for every registered layout: the
    ``layout`` name is validated but does not change the order.

    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param layout: Registered layout name.
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

# ---------------------------------------------------------------------------

def _unpatchify_payloads(payloads: Any, config: BridgeConfig) -> Any:
    """Reshape ``(B, num_patches, patch_payload_dim)`` into ``(B, H, W, C)``.

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

# ---------------------------------------------------------------------------

def _patchify_bridge(x: Any, config: BridgeConfig) -> Any:
    """Reshape ``(B, H, W, C)`` into ``(B, num_patches, patch_payload_dim)``.

    Exact inverse of :func:`_unpatchify_payloads`: the transpose swaps axes 2
    and 3, so it is its own inverse.

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

# ---------------------------------------------------------------------------

def token_flat_to_bridge(
    token_flat: Any, config: BridgeConfig, layout: str = "row_major"
) -> Any:
    """Pack flattened token embeddings into the channels-last bridge tensor.

    :param token_flat: ``(B, token_flat_dim)`` token embeddings.
    :param config: Bridge geometry; ``config.validate()`` runs on entry.
    :type config: BridgeConfig
    :param layout: Registered token-to-patch layout.
    :type layout: str
    :return: ``(B, H, W, C)`` bridge tensor, in the dtype of ``token_flat``.
    :raises ValueError: If ``layout`` is unknown; ``config.validate()`` may
        raise too.
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
    # Upstream scatters by `positions`; gathering by `argsort(positions)` is the
    # same permutation, and gather is what keras.ops offers on every backend.
    patch_payloads = keras.ops.take(payloads, np.argsort(positions), axis=1)
    return _unpatchify_payloads(patch_payloads, config)

# ---------------------------------------------------------------------------

def bridge_to_token_flat(
    x: Any, config: BridgeConfig, layout: str = "row_major"
) -> Any:
    """Unpack a bridge tensor back into flattened token embeddings.

    Exact inverse of :func:`token_flat_to_bridge`.

    :param x: ``(B, H, W, C)`` bridge tensor.
    :param config: Bridge geometry; ``config.validate()`` runs on entry.
    :type config: BridgeConfig
    :param layout: Registered token-to-patch layout.
    :type layout: str
    :return: ``(B, token_flat_dim)`` token embeddings, in the dtype of ``x``.
    :raises ValueError: If ``layout`` is unknown; ``config.validate()`` may
        raise too.
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
    """Return the per-token L2 norm of the unscaled embeddings in a bridge tensor.

    Real token embeddings have unit norm once divided by ``token_scale``;
    padding tokens are exactly zero.

    :param x_bridge: ``(B, H, W, C)`` bridge tensor.
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param token_scale: Divisor applied before the norm; defaults to
        ``config.token_scale``. Not validated; 0 gives non-finite norms.
    :type token_scale: Optional[float]
    :param layout: Registered token-to-patch layout.
    :type layout: str
    :return: ``(B, token_seq_len)`` norms.
    :raises ValueError: As :func:`bridge_to_token_flat`.
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
    """Return the first ``True`` column per row of a ``(B, T)`` mask, else ``T``.

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
    :param zero_thresh: Norms strictly below this count as padding.
    :type zero_thresh: float
    :param layout: Registered token-to-patch layout.
    :type layout: str
    :return: ``(stops, norms)``: ``(B,)`` int32 stop indices (``token_seq_len``
        when a row has no padding) and the ``(B, token_seq_len)`` norms.
    :rtype: Tuple[Any, Any]
    :raises ValueError: As :func:`bridge_to_token_flat`.
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
    :return: ``(B,)`` int32 stop indices; ``T`` when the row holds no padding.
    """
    return _first_true_index(keras.ops.equal(pred_ids, int(pad_id)))


def prepare_bridge_batch(
    batch: Dict[str, Any],
    config: BridgeConfig,
    layout: str = "row_major",
    seed: Optional[int] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Map one dataset batch to the five tensors the bridge process consumes.

    Both endpoints are cast to float32, so a float64 batch is narrowed. With
    ``text_as_noise`` or ``image_as_noise`` set, that endpoint is replaced by
    standard normal noise. The same ``seed`` feeds both draws: with both flags
    and an integer seed, the two noise tensors are identical.

    :param batch: Dict with ``"latent"`` ``(B, H, W, C)``, ``"text_token_emb"``
        ``(B, token_flat_dim)`` and ``"prompt_kind_label"`` ``(B,)``.
    :type batch: Dict[str, Any]
    :param config: Bridge geometry.
    :type config: BridgeConfig
    :param layout: Registered token->patch layout.
    :type layout: str
    :param seed: Seed for the noise draws, used only when a ``*_as_noise``
        flag is set; ``None`` uses the global RNG.
    :type seed: Optional[int]
    :return: ``(x_0_process, x_1_process, y, x_0, x_1)``, with ``y`` as int32.
        The trailing pair is always the real endpoints, so a caller can still
        condition on the real text and image when the process runs on noise.
    :rtype: Tuple[Any, Any, Any, Any, Any]
    :raises KeyError: If ``batch`` lacks one of the three keys.
    :raises ValueError: As :func:`token_flat_to_bridge`.
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