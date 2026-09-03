"""Fixed 3-D sin-cos positional table for square-spatial video grids (pure NumPy).

This module holds :func:`get_3d_sincos_pos_embed`, a thin NumPy assembly
wrapper that ports the LeVJEPA PyTorch reference's ``get_3d_sincos_pos_embed``
(``module.py``). It reuses this package's existing
:func:`~dl_techniques.layers.embedding.sincos_pos_embed_2d.get_1d_sincos_pos_embed_from_grid`
three times, once per axis (depth/time, height, width), and assembles the
three 1-D tables into one ``(N, embed_dim)`` table. No new sincos math is
introduced here.

Like ``get_2d_sincos_pos_embed``, this is pure NumPy -- no Keras op runs here
-- so it is safe to call inside ``build()`` and feed straight to
``add_weight(trainable=False, initializer=keras.initializers.Constant(table))``.

Axis-order fidelity (read before touching this file):
    The reference builds its grid with a **3-argument**
    ``np.meshgrid(grid_h, grid_d, grid_w)`` call, not the 2-argument
    ``indexing='xy'`` convention ``sincos_pos_embed_2d.py`` uses. NumPy's
    default ``indexing='xy'`` swaps the output SHAPE of only the first two
    positional arguments (``grid_h``, ``grid_d`` here); the third
    (``grid_w``) keeps its own axis position. MEASURED: for
    ``grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)`` with
    ``grid_h`` length ``grid_size``, ``grid_d`` length ``grid_depth`` and
    ``grid_w`` length ``grid_size``, every one of the three returned arrays
    has shape ``(grid_depth, grid_size, grid_size)`` -- i.e. ``(D, H, W)``,
    NOT ``(H, D, W)`` as the argument order alone would suggest. ``grid_d``
    varies along axis 0 (depth-major), ``grid_h`` along axis 1, ``grid_w``
    along axis 2 (width-minor). This happens to be exactly the flatten order
    ``PatchEmbed3D`` produces (``Conv3D`` output ``(T', H', W', C)`` reshaped
    row-major to ``(T'*H'*W', C)``), which is WHY the reference calls
    meshgrid this way rather than the more "obvious" ``indexing='ij'`` or a
    2-argument call per axis-pair. Port this 3-argument call FAITHFULLY. Do
    not "simplify" it into three separate ``np.meshgrid`` calls or add
    ``indexing='ij'`` -- either would silently permute which token index maps
    to which (t, h, w) coordinate, with no shape or dtype symptom.

Concatenation order:
    The reference concatenates ``[emb_d, emb_h, emb_w]`` (depth first), then
    truncates to ``embed_dim`` columns with ``[:, :embed_dim]``. Ported
    exactly; do not reorder to ``[emb_h, emb_w, emb_d]`` even though that
    would read more naturally next to ``get_2d_sincos_pos_embed_from_grid``'s
    ``[emb_col, emb_row]`` order -- they are independently specified.

References:
    - LeVJEPA PyTorch reference, ``module.py::get_3d_sincos_pos_embed``
      (pasted transcript; no public arXiv id in this plan's context).
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      arXiv:2111.06377 (origin of the 1-D sincos primitive this builds on).
"""

from typing import Any

import numpy as np

from dl_techniques.layers.embedding.sincos_pos_embed_2d import (
    get_1d_sincos_pos_embed_from_grid,
)

__all__ = ["get_3d_sincos_pos_embed"]


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """Build the fixed 3-D sin-cos positional table for a video patch grid.

    Splits ``embed_dim`` into a depth (time) band and two spatial bands (h,
    w), computes a 1-D sincos table per axis via
    :func:`~dl_techniques.layers.embedding.sincos_pos_embed_2d.get_1d_sincos_pos_embed_from_grid`,
    and concatenates ``[depth | height | width]`` before truncating to
    ``embed_dim`` columns.

    Band widths:
        * ``uniform_power=False`` (default): the depth band gets half of
          ``embed_dim``, and each spatial band gets a quarter --
          ``d_embed_dim = embed_dim // 2``,
          ``h_embed_dim = w_embed_dim = embed_dim // 4``.
        * ``uniform_power=True``: all three bands get an equal, rounded-up
          share -- ``int(ceil(embed_dim / 6) * 2)`` each -- so the
          concatenated width may exceed ``embed_dim`` before the final
          ``[:, :embed_dim]`` truncation.

    **How to install the result, and how not to** (same rule as
    ``get_2d_sincos_pos_embed``): the returned array is a constant, but it
    must become a non-trainable WEIGHT via a ``Constant`` initializer passed
    to ``add_weight`` -- never a plain tensor attribute, and never
    ``add_weight`` followed by a post-``build()`` ``.assign()`` (discarded by
    Keras 3's ``StatelessScope``).

    :param embed_dim: Width of the embedding per grid position.
    :type embed_dim: int
    :param grid_size: Side length of the square spatial (height/width) grid.
    :type grid_size: int
    :param grid_depth: Number of positions along the depth (time) axis.
    :type grid_depth: int
    :param cls_token: Whether to prepend one all-zero row for a class token.
    :type cls_token: bool
    :param uniform_power: Whether to give all three axes an equal (rounded)
        band width instead of the default half/quarter/quarter split.
    :type uniform_power: bool
    :return: ``(grid_depth * grid_size**2, embed_dim)``, or with one extra
        leading zero row when ``cls_token`` is ``True``. float64.
    :rtype: np.ndarray

    Example:
        >>> table = get_3d_sincos_pos_embed(embed_dim=16, grid_size=4, grid_depth=2)
        >>> table.shape
        (32, 16)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)

    # DECISION plan-2026-09-03-2a714a91/D-009
    # 3-argument np.meshgrid, NOT the 2-argument indexing='xy' convention
    # sincos_pos_embed_2d.py uses. With default indexing='xy', only the first
    # two positional arguments (grid_h, grid_d) have their output-shape axes
    # swapped; the third (grid_w) keeps its own axis. MEASURED: every one of
    # the three returned arrays has shape (grid_depth, grid_size, grid_size)
    # -- depth-major, height-mid, width-minor -- which is exactly the flatten
    # order PatchEmbed3D produces (Conv3D output (T',H',W',C) reshaped
    # row-major). This is why the reference calls meshgrid this way instead
    # of indexing='ij' or three separate calls. Do NOT "clean this up": doing
    # so silently permutes which token index maps to which (t,h,w)
    # coordinate, with no shape/dtype symptom. See decisions.md D-009.
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)

    # Depth first, ported exactly -- see the module docstring's
    # "Concatenation order" note.
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed
