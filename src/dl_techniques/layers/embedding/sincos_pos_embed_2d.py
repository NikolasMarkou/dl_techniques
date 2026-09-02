"""Fixed 2-D sin-cos positional tables for square patch grids (pure NumPy).

This module holds the three MAE ``util/pos_embed.py`` helpers that every DiT
descendant uses to build its **frozen** positional table:
:func:`get_1d_sincos_pos_embed_from_grid`,
:func:`get_2d_sincos_pos_embed_from_grid` and :func:`get_2d_sincos_pos_embed`.
They are pure NumPy -- no Keras op runs here -- so they are safe to call inside
``build()`` and feed straight to
``add_weight(trainable=False, initializer=keras.initializers.Constant(table))``.

Architecture:

.. code-block:: text

    grid_size (int)
      |
      v
    ┌──────────────────────────────────────────────────────────────┐
    │ np.meshgrid(grid_w, grid_h)      "here w goes first"         │
    │   grid[0][row, col] == col   (the COLUMN / w index)          │
    │   grid[1][row, col] == row   (the ROW / h index)             │
    └──────────────────────────────────────────────────────────────┘
      |                                   |
      v grid[0]                           v grid[1]
    get_1d(D/2)                         get_1d(D/2)
      | [G*G, D/2]                        | [G*G, D/2]
      └───────────────┐   ⊕ concat  ┌─────┘
                      ▼             ▼
             ┌──────────────────────────────┐
             │  pos_embed  [G*G, D]         │
             │  cols 0..D/2-1  <- COLUMN    │
             │  cols D/2..D-1  <- ROW       │
             └──────────────────────────────┘
                            │
                            ▼
              added to the patch tokens  [B, G*G, D]

Why the layout matters:
    Transposing the grid -- by passing ``indexing="ij"`` to ``np.meshgrid``, or
    by swapping ``grid[0]``/``grid[1]`` in
    :func:`get_2d_sincos_pos_embed_from_grid`, or by swapping the two output
    halves -- is a pure permutation on a square grid. Shape, dtype, every norm
    and every per-row statistic are IDENTICAL, so only an elementwise comparison
    against an independently computed destination index can see it. A model
    trained on the transposed table trains fine and is incompatible with every
    published checkpoint. Do not "clean up" the ``w``-first meshgrid, and do not
    rename the halves into agreement with upstream's ``emb_h``/``emb_w`` (which
    are themselves inverted relative to what they encode).

    MEASURED, and worth stating because it is the obvious mutation and it is
    the WRONG one: swapping the two *arguments* to
    ``np.meshgrid(grid_w, grid_h)`` is an exact **no-op** here, not a
    transposition. ``grid_h`` and ``grid_w`` are the same ``np.arange(grid_size)``
    for a square grid, so ``meshgrid(a, a)`` is argument-order invariant
    (``np.array_equal`` is ``True`` at every size tried). Anyone reaching for
    that injection to prove the guard works will get a false GREEN. The two
    mutations named in the paragraph above are the real ones, and
    ``tests/test_layers/test_embedding/test_the_sincos_grid_is_w_first.py``
    was proven RED against both.

Concat order, and why it disagrees with the timestep ladder:
    :func:`get_1d_sincos_pos_embed_from_grid` concatenates ``[sin, cos]``,
    while
    :class:`~dl_techniques.layers.embedding.timestep_embedding.TimestepEmbedding`
    concatenates ``[cos, sin]``. Both are correct: they are independently
    specified upstream (MAE vs GLIDE) and must not be unified.

Relationship to ``bit_diffusion``:
    An identical bit-exact copy of these three functions lives, module-private,
    in
    ``src/dl_techniques/models/vision_language/bit_diffusion/blocks.py:724-864``.
    This module is the shared promotion of it (plan
    ``plan-2026-09-02T170923-1285ed83`` D-001); ``bit_diffusion`` was left
    untouched because moving its registered classes would change their
    serialization keys. The duplication is deliberate and recorded -- if you
    change the numerics here, the two copies have drifted and
    ``tests/test_layers/test_embedding/test_the_sincos_grid_is_w_first.py``
    (which cross-checks the sibling copy) will say so.

References:
    - Peebles & Xie, 2022. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377) -- the origin of these three
      functions (``facebookresearch/mae``, ``util/pos_embed.py``).
"""

from typing import Any

import numpy as np

# DECISION plan-2026-09-02T170923-1285ed83/D-001
# This module is a deliberate promotion, NOT a cleanup target. Do NOT delete it
# in favour of importing from `bit_diffusion/blocks.py` (a models -> models
# import inverts the dependency direction), and do NOT "de-duplicate" by MOVING
# the sibling: that changes its registered keys. decisions.md D-001.
__all__ = [
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
]


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: Any
) -> np.ndarray:
    """Sinusoidally embed a flat array of positions, **sin first**.

    :param embed_dim: Output width per position. Must be a positive even
        integer.
    :type embed_dim: int
    :param pos: Positions of any shape; flattened to ``(M,)`` first.
    :type pos: Any
    :return: ``(M, embed_dim)`` float64 array, ``[sin | cos]``.
    :rtype: np.ndarray
    :raises ValueError: If ``embed_dim`` is not a positive even integer.

    Example:
        >>> get_1d_sincos_pos_embed_from_grid(4, np.arange(3)).shape
        (3, 4)
    """
    if embed_dim <= 0 or embed_dim % 2 != 0:
        raise ValueError(
            f"embed_dim must be a positive even integer, got {embed_dim}"
        )

    # float64 throughout, exactly as upstream. Do NOT narrow this to float32:
    # the table is a constant computed once, and the extra precision is free.
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = np.asarray(pos).reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # SIN FIRST here -- the opposite of `TimestepEmbedding`'s cos-first basis,
    # and correct: MAE and GLIDE specify the two independently.
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: Any
) -> np.ndarray:
    """Concatenate two 1-D embeddings, one per meshgrid output.

    ``grid[0]`` (the COLUMN index, because the grid is built ``w``-first)
    becomes the FIRST ``embed_dim // 2`` columns of the result; ``grid[1]``
    (the ROW index) becomes the last ``embed_dim // 2``. Upstream names those
    halves ``emb_h`` and ``emb_w``, which is backwards relative to what they
    encode; the ORDER is what a port must match, not the names.

    :param embed_dim: Total output width. Must be even (each half is halved
        again by the 1-D helper, so a multiple of 4 is the practical
        constraint).
    :type embed_dim: int
    :param grid: ``(2, ...)`` array of positions.
    :type grid: Any
    :return: ``(H*W, embed_dim)`` float64 array.
    :rtype: np.ndarray
    :raises ValueError: If ``embed_dim`` is not even.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    emb_col = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_row = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    return np.concatenate([emb_col, emb_row], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """Build the fixed 2-D sin-cos positional table for a square patch grid.

    **How to install the result, and how not to.** The returned array is a
    constant, but it must still become a non-trainable WEIGHT:

    * NEVER a plain tensor attribute
      (``self.pos_embed = keras.ops.convert_to_tensor(...)``) -- that does not
      round-trip through ``.keras`` save/load.
    * NEVER ``add_weight(...)`` followed by ``.assign(...)`` inside
      ``build()`` -- ``StatelessScope`` DISCARDS the assign and the table stays
      all zeros in every real model, with no shape symptom.

    The one correct form is a ``Constant`` initializer passed to
    ``add_weight``.

    :param embed_dim: Width of the embedding per grid position.
    :type embed_dim: int
    :param grid_size: Side length of the square grid; the table has
        ``grid_size ** 2`` rows.
    :type grid_size: int
    :param cls_token: Whether to prepend ``extra_tokens`` zero rows.
    :type cls_token: bool
    :param extra_tokens: Number of zero rows prepended when ``cls_token`` is
        true. Upstream prepends only when BOTH are set; reproduced exactly.
    :type extra_tokens: int
    :return: ``(grid_size**2, embed_dim)``, or
        ``(extra_tokens + grid_size**2, embed_dim)`` with a cls token. float64.
    :rtype: np.ndarray
    :raises ValueError: If ``grid_size`` is not positive or ``embed_dim`` is
        not even.

    Example:
        >>> table = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)
        >>> table.shape
        (16, 8)
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # "here w goes first" -- upstream's own annotation, and NumPy's default
    # indexing='xy'. So grid[0] holds the COLUMN index and grid[1] the ROW
    # index. Adding indexing='ij' transposes the table with no shape change and
    # no statistic moving. Swapping the two ARGUMENTS does not: `grid_h` and
    # `grid_w` are the same arange for a square grid, so that mutation is an
    # exact no-op (see the module docstring).
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed
