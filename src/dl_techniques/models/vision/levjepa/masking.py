"""Block-causal attention mask + random token-dropping utilities for LeVJEPA.

Model-internal helpers, not registered Keras layers/models — plain functions
operating on ``keras.KerasTensor``/backend tensors, consumed by
:mod:`dl_techniques.models.vision.levjepa.blocks` and
:mod:`dl_techniques.models.vision.levjepa.encoder` (Step 4). Following the
package's registration convention (``keras_registration.CLAUDE.md`` /
``dl_techniques/CLAUDE.md`` § Registration), only serializable
``Layer``/``Model``/``Loss``/etc. classes are decorated with
``@register_dl_technique`` — a bare function has no config to round-trip and
registering it would be a no-op, so neither function here carries the
decorator.

Ported from the pasted PyTorch reference's ``build_block_causal_mask`` and the
random-token-drop block inside ``VisionTransformer.forward``. The mask
semantics below are re-derived line-by-line from that source (not a
paraphrase — see the plan's Step 3 execution prompt, which corrects
``plan.md``'s Assumption A2 hedge):

- ``mask[..., i, j] = True`` means query position ``i`` MAY attend to key
  position ``j`` (row = query, column = key — the standard attention-mask
  convention).
- Patch-to-patch: ``frame_ids[query] >= frame_ids[key]`` — causal across
  frames, bidirectional *within* a frame (the comparison is satisfied both
  ways when the two frame ids are equal).
- CLS row (CLS as query): all-True — CLS attends to every key, including
  itself and every patch in every frame.
- CLS column for patch queries: all-False — a patch can never attend to CLS
  as a key. This is the branch the PyTorch source leaves at its tensor's
  ``zeros()`` initial value (never explicitly set), so it is easy to
  transcribe backwards; the delta-impulse probe in
  ``tests/test_models/test_levjepa/test_masking.py`` pins it explicitly.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import keras


def build_block_causal_mask(
    num_frames: int,
    tokens_per_frame: int,
    token_ids: Optional[Any] = None,
    num_prefix_tokens: int = 1,
    batch_size: Optional[int] = None,
) -> Any:
    """Build a block-causal attention mask (bidirectional within a frame,
    causal across frames), with an optional all-seeing / never-seen prefix
    (CLS) carve-out.

    :param num_frames: Number of temporal frames ``T``. Only used to build
        the default ``arange`` grid when ``token_ids`` is ``None``.
    :param tokens_per_frame: Number of patch tokens per frame,
        ``H_patches * W_patches``.
    :param token_ids: Optional integer tensor of shape ``(B, num_patches)``
        giving the TRUE flat grid index of each patch position in the
        (possibly token-dropped) sequence. When given, frame membership is
        computed from these true positions rather than from the sequence's
        own (post-drop) ordering — required so the mask stays consistent
        with :func:`random_token_drop`'s ``token_ids`` output. When
        ``None``, the default identity grid ``arange(num_frames *
        tokens_per_frame)`` is used, matching the PyTorch reference's
        ``ids.unsqueeze(0)`` no-batch-dim default (shape ``(1, N, N)``,
        broadcastable over any batch dimension downstream).
    :param num_prefix_tokens: Number of prefix (CLS) tokens prepended to the
        patch sequence. ``0`` skips the carve-out entirely — the mask is
        then exactly the patch-only causal grid, needed for a possible
        ``attn_mode="full"`` bypass upstream (that bypass itself just means
        "do not call this function," so no ``"full"`` branch lives here).
    :param batch_size: Optional explicit batch size used only when
        ``token_ids`` is ``None`` and a batch dimension wider than the
        default broadcastable ``1`` is wanted. Ignored when ``token_ids``
        is given (its own leading dimension is the batch size).
    :return: Boolean tensor of shape ``(B, 1, N, N)`` where ``N =
        num_frames * tokens_per_frame + num_prefix_tokens`` (or ``B == 1``
        when neither ``token_ids`` nor ``batch_size`` supply a batch
        dimension) and ``True`` means "attend."
    """
    p = int(num_prefix_tokens)

    if token_ids is None:
        num_patches = int(num_frames * tokens_per_frame)
        ids = keras.ops.arange(num_patches, dtype="int32")
        frame_ids = ids // int(tokens_per_frame)
        # (N, N): patch-query x patch-key causal-within-frame grid.
        patch_mask = keras.ops.greater_equal(
            keras.ops.expand_dims(frame_ids, axis=-1),
            keras.ops.expand_dims(frame_ids, axis=-2),
        )

        if p > 0:
            # CLS rows: all-True over the full (p + N) key axis.
            top = keras.ops.ones((p, num_patches + p), dtype="bool")
            # Patch rows: CLS column False, then the causal patch block.
            bottom_left = keras.ops.zeros((num_patches, p), dtype="bool")
            bottom = keras.ops.concatenate([bottom_left, patch_mask], axis=1)
            full = keras.ops.concatenate([top, bottom], axis=0)
        else:
            full = patch_mask

        # (N', N') -> (1, N', N') -> (1, 1, N', N').
        full = keras.ops.expand_dims(full, axis=0)
        if batch_size is not None:
            full = keras.ops.tile(full, (int(batch_size), 1, 1))
        full = keras.ops.expand_dims(full, axis=1)
        return full

    # `token_ids`-driven path: (B, K) true grid positions.
    frame_ids = token_ids // int(tokens_per_frame)
    # (B, K, K)
    patch_mask = keras.ops.greater_equal(
        keras.ops.expand_dims(frame_ids, axis=-1),
        keras.ops.expand_dims(frame_ids, axis=-2),
    )

    if p > 0:
        batch = keras.ops.shape(token_ids)[0]
        keep_len = keras.ops.shape(token_ids)[1]
        top = keras.ops.ones((batch, p, keep_len + p), dtype="bool")
        bottom_left = keras.ops.zeros((batch, keep_len, p), dtype="bool")
        bottom = keras.ops.concatenate([bottom_left, patch_mask], axis=2)
        full = keras.ops.concatenate([top, bottom], axis=1)
    else:
        full = patch_mask

    # (B, N', N') -> (B, 1, N', N').
    full = keras.ops.expand_dims(full, axis=1)
    return full


def random_token_drop(
    x: Any,
    drop_rate: float,
    training: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[Any, Optional[Any]]:
    """Randomly drop a fraction of patch tokens (train-time only).

    Ports the PyTorch reference's ``token_drop_rate`` block: for each
    sample independently, draws uniform noise over the ``N`` sequence
    positions, keeps the ``keep_len`` lowest-noise positions (an
    ``argsort``-of-noise permutation-sample, matching
    ``masked_autoencoder/patch_masking.py``'s idiom), and gathers ``x``
    down to that shortened sequence.

    :param x: Input tensor of shape ``(B, N, C)``.
    :param drop_rate: Fraction of tokens to drop, in ``[0, 1)``. Values
        ``<= 0`` degenerate to the identity branch below.
    :param training: Standard Keras training flag. ``False``/``None``
        degenerates to the identity branch below, matching the reference's
        ``self.training and self.token_drop_rate > 0`` guard.
    :param seed: Optional seed forwarded to ``keras.random.uniform`` for
        deterministic/testable dropping.
    :return: ``(dropped_x, token_ids)``. When the identity branch is taken,
        ``dropped_x is x`` (no argsort/gather machinery runs at all — a
        true no-op, not a degenerate keep-everything gather) and
        ``token_ids`` is ``None``. Otherwise ``dropped_x`` has shape
        ``(B, keep_len, C)`` and ``token_ids`` has shape ``(B, keep_len)``,
        holding the TRUE (pre-drop) grid index of each kept position — feed
        this straight into :func:`build_block_causal_mask`'s ``token_ids``
        argument so the mask and the dropped sequence stay consistent.
    """
    if not training or drop_rate <= 0:
        return x, None

    # `num_patches` must be a concrete Python int at trace time (it drives
    # `keep_len`'s rounding and the static slice width below); the patch
    # grid is architecture-fixed, so the static shape is always known —
    # unlike `batch_size`, which stays dynamic via `ops.shape`.
    num_patches = int(x.shape[1])
    batch_size = keras.ops.shape(x)[0]
    keep_len = max(1, int(round(num_patches * (1.0 - drop_rate))))

    noise = keras.random.uniform((batch_size, num_patches), seed=seed)
    token_ids = keras.ops.argsort(noise, axis=1)[:, :keep_len]

    gathered = keras.ops.take_along_axis(
        x,
        keras.ops.expand_dims(token_ids, axis=-1),
        axis=1,
    )
    return gathered, token_ids
