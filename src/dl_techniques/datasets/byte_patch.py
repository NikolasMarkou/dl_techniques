"""Fixed-size byte-patch transforms for tokenizer-free sequence models.

``datasets/byte_lm.py`` rebuilds the packed-CLM pipeline at byte resolution;
this module is the one further step a patch-level model needs: grouping the
byte stream into fixed patches of ``k`` (CANINE-style blocking). Everything
here is static — patch boundaries fall every ``k`` bytes regardless of content
— so shapes are known ahead of time, unlike entropy- or learned-chunking
alternatives (``layers/blt/``, ``layers/dynamic_chunking/``), which produce
per-row dynamic lengths and need their own stacks.

Four transforms, all pure integer plumbing:

* :func:`bytes_to_patches` — flat byte ids → ``(P, k)`` patch rows,
* :func:`patches_to_bytes` — the exact inverse (flatten only),
* :func:`pack_byte_patch_windows` — concat-and-chunk at patch resolution,
* :func:`make_lm_labels` — label masking for prompt/pad positions.

Conventions (shared with :mod:`dl_techniques.datasets.byte_lm`, never
retyped): ``BYTE_VOCAB_SIZE == 256``, ``EOS_ID == 255`` terminates a document,
``BOS_ID == 254`` is available. Padding inside a patch row is ``0x00``, which
never occurs in valid UTF-8, so a ``0x00`` input byte unambiguously means pad;
labels use ``ignore_index`` (default ``-1``, the
:mod:`dl_techniques.losses.masked_causal_lm_loss` convention) for positions the
loss must not score — prompt bytes as well as pads.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

from dl_techniques.datasets.byte_lm import pack_byte_windows

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Pad byte inside a patch row. ``0x00`` never occurs in valid UTF-8, so it is
#: unambiguous with content. Labels never carry this value directly — see
#: :func:`make_lm_labels`, which maps ignored positions to ``ignore_index``.
PAD_BYTE_ID = 0

#: Default patch width in bytes. English averages ~4 bytes per BPE token, so
#: k=4 keeps patch counts comparable to token counts at the same content.
DEFAULT_PATCH_SIZE = 4


# ---------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------


def bytes_to_patches(
    ids: Sequence[int],
    patch_size: int = DEFAULT_PATCH_SIZE,
    pad_id: int = PAD_BYTE_ID,
) -> np.ndarray:
    """Group flat byte ids into fixed ``(P, patch_size)`` patch rows.

    The input is padded with ``pad_id`` up to a multiple of ``patch_size``
    and reshaped row-major, so patch ``i`` always holds bytes
    ``[i * patch_size, (i + 1) * patch_size)``. An input already a multiple
    gains no pad row. Mid-codepoint splits are preserved untouched — this
    function only reshapes, so :func:`patches_to_bytes` restores the input
    byte-exactly including any split codepoint.

    :param ids: Flat byte ids, each in ``[0, 255]``.
    :param patch_size: Patch width in bytes. Must be >= 1.
    :param pad_id: Fill value for the tail row. Must be in ``[0, 255]``.
    :return: ``int32`` array of shape ``(ceil(len(ids) / patch_size),
        patch_size)``. Empty input yields shape ``(0, patch_size)``.
    :raises ValueError: If ``patch_size < 1`` or ``pad_id`` is outside
        ``[0, 255]``.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if not 0 <= pad_id <= 255:
        raise ValueError(f"pad_id must be in [0, 255], got {pad_id}")

    flat = np.asarray(list(ids), dtype=np.int32)
    n_pad = (-len(flat)) % patch_size
    if n_pad:
        flat = np.concatenate(
            [flat, np.full(n_pad, pad_id, dtype=np.int32)]
        )
    return flat.reshape(-1, patch_size)


def patches_to_bytes(patches: Sequence[Sequence[int]]) -> np.ndarray:
    """Flatten ``(P, k)`` patch rows back to a 1-D byte-id array.

    The exact inverse of :func:`bytes_to_patches` — flatten only, no pad
    stripping (a trailing pad row is caller data; the loss ignores it via
    :func:`make_lm_labels` and the decoder tolerates it via ``errors``).

    :param patches: Patch rows of any uniform width.
    :return: 1-D ``int32`` array of length ``P * k``.
    """
    return np.asarray(patches, dtype=np.int32).reshape(-1)


def pack_byte_patch_windows(
    stream: Iterable[Sequence[int]],
    patch_size: int = DEFAULT_PATCH_SIZE,
    patches_per_window: int = 128,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    """Concat-and-chunk a stream of byte-id documents into patch windows.

    Composition of :func:`pack_byte_windows` (byte resolution) with
    :func:`bytes_to_patches`: documents are concatenated and sliced into
    ``patches_per_window * patch_size``-byte windows, each reshaped to
    ``(patches_per_window, patch_size)``. Every source byte is trained on
    exactly once per epoch; a window may span a document boundary. As with
    :func:`pack_byte_windows`, document separators are the caller's job
    (append :data:`EOS_ID` per document upstream).

    :param stream: Iterable of per-document byte-id sequences. Consumed in a
        single pass (it may be a generator).
    :param patch_size: Patch width in bytes. Must be >= 1.
    :param patches_per_window: Patches per yielded window. Must be >= 1.
    :param drop_remainder: When ``True`` a trailing short buffer is discarded
        (the token pipeline's behaviour); when ``False`` the tail is yielded
        as a short patch window — still rectangular, padded to full patches
        by :func:`bytes_to_patches`.
    :return: Iterator of ``int32`` arrays of shape ``(patches_per_window,
        patch_size)``, except possibly the last when ``drop_remainder`` is
        ``False`` and the tail holds fewer patches.
    :raises ValueError: If ``patch_size < 1`` or ``patches_per_window < 1``.
    """
    if patches_per_window < 1:
        raise ValueError(
            f"patches_per_window must be >= 1, got {patches_per_window}"
        )
    byte_len = patches_per_window * patch_size
    for window in pack_byte_windows(
        stream, byte_len, drop_remainder=False
    ):
        if len(window) == byte_len:
            yield bytes_to_patches(window, patch_size)
        elif not drop_remainder:
            yield bytes_to_patches(window, patch_size)


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------


def make_lm_labels(
    byte_ids: Sequence[int],
    num_prompt_bytes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Build next-token labels with prompt and pad positions masked.

    The data-side half of the instruction-tuning contract (GAP-3): the model
    scores only answer bytes. Positions ``[0, num_prompt_bytes)`` and any
    ``0x00`` pad byte map to ``ignore_index``; every other position carries
    its own byte id (the loss shifts internally, so labels align with inputs).

    :param byte_ids: Flat byte ids of the full ``[prompt ; answer]`` sequence.
    :param num_prompt_bytes: Leading positions to mask. Must satisfy
        ``0 <= num_prompt_bytes <= len(byte_ids)``.
    :param ignore_index: Mask value. Must be negative (byte ids are
        non-negative, so a non-negative mask would collide with content).
    :return: ``int32`` array of the same length as ``byte_ids``.
    :raises ValueError: If ``num_prompt_bytes`` is out of range or
        ``ignore_index`` is non-negative.
    """
    if ignore_index >= 0:
        raise ValueError(
            f"ignore_index must be negative, got {ignore_index}"
        )
    flat = np.asarray(list(byte_ids), dtype=np.int32)
    if not 0 <= num_prompt_bytes <= len(flat):
        raise ValueError(
            f"num_prompt_bytes must be in [0, {len(flat)}], "
            f"got {num_prompt_bytes}"
        )
    labels = flat.copy()
    labels[:num_prompt_bytes] = ignore_index
    labels[flat == PAD_BYTE_ID] = ignore_index
    return labels
