"""Attention masks and time positions for Decoupled Horizon Forecasting (DHF) decoding.

The sequence is [Enc; Dec]: encoder tokens embed observed context patches;
decoder tokens are forecast queries for horizon patches. All masks are boolean,
True meaning "may attend". Pure Keras ops — no framework-specific calls.
"""

import keras
from typing import Optional, Tuple


def compute_enc_key_validity(obs_per_patch, attend_missing: bool):
    """(R, N) encoder key validity.

    A patch is a valid key if it is observed, or if attend_missing is set and
    the row's series has started at or before it. Leading all-unobserved patches
    are left-padding and are never valid keys.
    """
    if not attend_missing:
        return obs_per_patch
    started = keras.ops.cumsum(keras.ops.cast(obs_per_patch, "float32"), axis=-1) > 0
    return keras.ops.logical_or(obs_per_patch, started)


def enc_causal_mask(key_valid):
    """(R, L, L) causal encoder mask over valid keys, with diagonal always open."""
    n = int(keras.ops.shape(key_valid)[1])
    rows = int(keras.ops.shape(key_valid)[0])
    causal = keras.ops.tril(keras.ops.ones((n, n), dtype="bool"))
    causal = keras.ops.broadcast_to(causal[None], (rows, n, n))
    eye = keras.ops.eye(n, dtype="bool")
    eye = keras.ops.broadcast_to(eye[None], (rows, n, n))
    key_col = keras.ops.broadcast_to(
        keras.ops.expand_dims(key_valid, axis=1), (rows, n, n)
    )
    return keras.ops.logical_or(keras.ops.logical_and(causal, key_col), eye)


def build_forecast_time_mask(context_len: int, horizon: int, key_valid):
    """(R, 1, L+h, L+h) time mask for one block without covariates.

    Encoder is causal; every decoder token reads whole valid context and its
    own causal decoder chain; context never reads horizon.
    """
    rows = int(keras.ops.shape(key_valid)[0])
    enc = enc_causal_mask(key_valid)  # (R, L, L)
    enc_dec = keras.ops.zeros((rows, context_len, horizon), dtype="bool")
    dec_enc = keras.ops.broadcast_to(
        keras.ops.expand_dims(key_valid, axis=1), (rows, horizon, context_len)
    )
    dec_dec = keras.ops.tril(keras.ops.ones((horizon, horizon), dtype="bool"))
    dec_dec = keras.ops.broadcast_to(dec_dec[None], (rows, horizon, horizon))
    top = keras.ops.concatenate([enc, enc_dec], axis=2)
    bottom = keras.ops.concatenate([dec_enc, dec_dec], axis=2)
    mask = keras.ops.concatenate([top, bottom], axis=1)
    return keras.ops.expand_dims(mask, axis=1)


def build_forecast_positions(context_len: int, horizon: int, first_obs):
    """(R, L+h) time positions 0..L+h-1, anchored at each row's first observed patch."""
    p = keras.ops.arange(context_len + horizon)
    p = keras.ops.expand_dims(p, axis=0)
    first_obs = keras.ops.cast(keras.ops.expand_dims(first_obs, axis=-1), p.dtype)
    return p - first_obs


def compute_block_layout(min_context, n_patches: int, block_size: int):
    """Per decoder position: anchor of its block and whether active (p >= s)."""
    p = keras.ops.expand_dims(keras.ops.arange(n_patches), axis=0)
    s = keras.ops.cast(keras.ops.expand_dims(min_context, axis=-1), p.dtype)
    active = p >= s
    offset = p - s
    anchor = s + block_size * (offset // block_size)
    anchor = keras.ops.where(active, anchor, keras.ops.zeros_like(anchor))
    return anchor, active


def build_two_stream_time_mask(
    min_context,
    n_patches: int,
    block_size: int,
    key_valid,
    bidirectional_rows=None,
):
    """(R, 1, 2N, 2N) time mask over full [Enc; Dec] window (covariate path).

    - Enc->Enc: causal over valid keys; full (non-causal) for bidirectional_rows.
    - Enc->Dec: never.
    - Dec->Enc: decoder token in block anchored at b reads valid Enc[j], j < b.
    - Dec->Dec: causal within same block; every decoder token reads itself.
    """
    anchor, active = compute_block_layout(min_context, n_patches, block_size)
    rows, n = int(keras.ops.shape(active)[0]), int(keras.ops.shape(active)[1])
    causal = keras.ops.tril(keras.ops.ones((n, n), dtype="bool"))
    causal = keras.ops.broadcast_to(causal[None], (rows, n, n))
    key_col = keras.ops.broadcast_to(
        keras.ops.expand_dims(key_valid, axis=1), (rows, n, n)
    )
    j = keras.ops.reshape(keras.ops.arange(n), (1, 1, n))
    anchor_col = keras.ops.expand_dims(anchor, axis=-1)  # (R, N, 1)
    eye = keras.ops.eye(n, dtype="bool")
    eye = keras.ops.broadcast_to(eye[None], (rows, n, n))

    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        bi = keras.ops.cast(bidirectional_rows, "float32")
        bi = keras.ops.expand_dims(keras.ops.expand_dims(bi, axis=-1), axis=-1)  # (R,1,1)
        enc_bi = keras.ops.logical_or(key_col, eye)
        enc_enc = keras.ops.where(keras.ops.broadcast_to(bi, (rows, n, n)), enc_bi, enc_enc)

    enc_dec = keras.ops.zeros((rows, n, n), dtype="bool")

    # Dec->Enc: active & (j < anchor) & key_valid
    dec_enc = keras.ops.logical_and(
        keras.ops.logical_and(
            keras.ops.expand_dims(active, axis=-1),          # (R,N,1)
            keras.ops.broadcast_to(j, (rows, n, n)) < anchor_col,
        ),
        key_col,
    )

    # Dec->Dec: same block & causal & active
    same_block = keras.ops.equal(anchor_col, keras.ops.expand_dims(anchor, axis=-2))
    active_col = keras.ops.expand_dims(active, axis=-1)   # (R,N,1)
    active_row = keras.ops.broadcast_to(
        keras.ops.expand_dims(active, axis=-2), (rows, 1, n)
    )   # (R,1,N)
    active_both = keras.ops.logical_and(active_col, active_row)
    dec_dec = keras.ops.logical_and(keras.ops.logical_and(same_block, causal), active_both)
    dec_dec = keras.ops.logical_or(dec_dec, eye)

    top = keras.ops.concatenate([enc_enc, enc_dec], axis=2)
    bottom = keras.ops.concatenate([dec_enc, dec_dec], axis=2)
    mask = keras.ops.concatenate([top, bottom], axis=1)
    return keras.ops.expand_dims(mask, axis=1)


def variate_firewall(
    r: int,
    target_rows=None,
    bidirectional_rows=None,
):
    """(R, R) forbidden query->key row pairs, or None.

    - Covariate row may not read target row (exogenous).
    - Known-future (bidirectional) row may not read causal row.
    """
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = keras.ops.zeros((r, r), dtype="bool")
    if target_rows is not None:
        # Covariate (non-target) queries cannot read target keys
        not_target = keras.ops.logical_not(target_rows)
        target_col = keras.ops.broadcast_to(
            keras.ops.expand_dims(target_rows, axis=0), (r, r)
        )
        not_target_row = keras.ops.broadcast_to(
            keras.ops.expand_dims(not_target, axis=1), (r, r)
        )
        forbid = keras.ops.logical_or(
            forbid, keras.ops.logical_and(not_target_row, target_col)
        )
    if bidirectional_rows is not None:
        # Bidirectional queries (rows) cannot read causal (non-bidirectional) keys
        causal_rows = keras.ops.logical_not(bidirectional_rows)
        bi_row = keras.ops.broadcast_to(
            keras.ops.expand_dims(bidirectional_rows, axis=1), (r, r)
        )  # [i, j] = bidirectional[i]
        causal_col = keras.ops.broadcast_to(
            keras.ops.expand_dims(causal_rows, axis=0), (r, r)
        )  # [i, j] = causal[j]
        forbid = keras.ops.logical_or(forbid, keras.ops.logical_and(bi_row, causal_col))
    return forbid


def build_two_stream_variate_mask(
    group_ids,
    enc_key_valid,
    n_dec: int,
    target_rows=None,
    bidirectional_rows=None,
):
    """(N_enc + N_dec, 1, R, R) variate mask, or None for single row.

    A row reads another row of the same series; on encoder slots the key row
    must also be a valid key. Firewalls apply on every slot; diagonal always open.
    """
    rows = int(keras.ops.shape(enc_key_valid)[0])
    if rows == 1:
        return None
    n_enc = int(keras.ops.shape(enc_key_valid)[1])

    row_i = keras.ops.cast(keras.ops.expand_dims(group_ids, axis=1), "int32")
    row_j = keras.ops.cast(keras.ops.expand_dims(group_ids, axis=0), "int32")
    row_eq = keras.ops.equal(row_i, row_j)  # (R, R)

    dec_valid = keras.ops.ones((rows, n_dec), dtype="bool")
    slot_valid = keras.ops.concatenate([enc_key_valid, dec_valid], axis=1)
    key_valid = keras.ops.transpose(slot_valid, (1, 0))  # (n_enc+n_dec, R)
    key_valid = keras.ops.broadcast_to(
        keras.ops.expand_dims(keras.ops.expand_dims(key_valid, axis=1), axis=2),
        (n_enc + n_dec, 1, rows, rows),
    )
    row_eq = keras.ops.broadcast_to(
        keras.ops.expand_dims(keras.ops.expand_dims(row_eq, axis=0), axis=1),
        (n_enc + n_dec, 1, rows, rows),
    )
    mask = keras.ops.logical_and(row_eq, key_valid)

    forbid = variate_firewall(rows, target_rows, bidirectional_rows)
    if forbid is not None:
        forbid = keras.ops.broadcast_to(
            keras.ops.expand_dims(keras.ops.expand_dims(forbid, axis=0), axis=1),
            (n_enc + n_dec, 1, rows, rows),
        )
        mask = keras.ops.logical_and(mask, keras.ops.logical_not(forbid))

    eye = keras.ops.eye(rows, dtype="bool")
    eye = keras.ops.broadcast_to(
        keras.ops.reshape(eye, (1, 1, rows, rows)),
        (n_enc + n_dec, 1, rows, rows),
    )
    mask = keras.ops.logical_or(mask, eye)
    return mask


def build_two_stream_positions(n_patches: int, first_obs):
    """(R, 2N) positions: encoder and decoder token of same patch share a time."""
    p = keras.ops.arange(n_patches)
    p = keras.ops.expand_dims(p, axis=0)
    first_obs = keras.ops.cast(keras.ops.expand_dims(first_obs, axis=-1), p.dtype)
    row = p - first_obs
    return keras.ops.concatenate([row, row], axis=-1)


def first_observed_patch(obs_per_patch, group_ids):
    """(R,) index of the first patch this row's series group observes (0 if none).

    group_ids must be non-negative integers sorted so that segment ops see whole
    groups (forecaster produces np.repeat(np.arange(b), v) which satisfies this).
    """
    group_obs = keras.ops.segment_max(
        keras.ops.cast(obs_per_patch, "float32"), keras.ops.cast(group_ids, "int32")
    )  # (G, N)
    first = keras.ops.argmax(group_obs, axis=-1)  # (G,)
    return keras.ops.cast(keras.ops.take(first, keras.ops.cast(group_ids, "int32")), "int32")
