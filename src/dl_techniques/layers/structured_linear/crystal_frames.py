"""Geometric frame constructors shared by every crystal-structured layer.

A *frame* here is a matrix whose **columns** are unit vectors, so that
``x @ frame`` reads as the set of inner products ``<x, v_j>``. Every layer in
this family needs the same three objects and must agree on them exactly, so
they are defined once here rather than restated per layer:

- :func:`hyperspherical_simplex` -- the regular simplex, ``dimensions + 1``
  unit vectors in ``dimensions`` dimensions, every pair at inner product
  ``-1/dimensions``. This is the Equiangular Tight Frame that Neural Collapse
  drives the final layer toward.
- :func:`haar_orthogonal` -- a Haar-distributed rotation, used to place
  several simplexes on the sphere without letting them coincide.
- :func:`crystal_frame` -- an arbitrary-width frame packed from rotated
  simplexes by a greedy minimax-angular-separation heuristic.

**Why the packing matters.** A single simplex holds exactly ``dimensions + 1``
distinct directions. Any frame wider than that cannot come from one simplex,
and the obvious extension -- tiling the same vertices -- produces a frame
whose columns are exact duplicates of each other. Measured on the tiled
construction at ``dimensions=16, width=100``: **17** distinct columns out of
100, and **490** column pairs at inner product exactly ``1.0``. An 83%-duplicated
matrix is not a codebook; every duplicated column receives an identical logit.
:func:`crystal_frame` fills the same width with rotated simplexes instead.

**The ceiling, stated honestly.** Packing yields ``width`` distinct columns
*up to a limit set by the sphere itself*: ``S^(N-1)`` contains only ``2N``
points, so no frame can have more than ``2 * dimensions`` distinct columns, and
below that the simplex count binds first at ``dimensions + 1``. Past the
ceiling the duplicates are unavoidable rather than a defect -- at
``dimensions=1`` the sphere is ``S^0 = {-1, +1}``, so a width-3 frame of 3
distinct directions does not exist and the caller gets 2 distinct plus one
repeat. Every configuration the layers actually use sits well inside the limit;
the degenerate corner is documented rather than silently produced.

Pure NumPy by design. These run once per layer, inside a weight initializer,
where a ``keras.ops`` call would be tracing-time work for no benefit. The
layers consume the result as a NumPy array and hand it to
``keras.initializers.Constant``.

References:
    - Papyan, Han and Donoho, 2020. Prevalence of Neural Collapse during the
      Terminal Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Strohmer and Heath, 2003. Grassmannian Frames with Applications to
      Coding and Communication. (Applied and Computational Harmonic Analysis
      14(3))
    - Martin, 2006. The Orthogonal Group as a Special Case of Elementary
      Geometry. (https://arxiv.org/abs/math/0611249) -- Haar measure on O(N).
"""

import math

import numpy as np

# ---------------------------------------------------------------------

__all__ = [
    'crystal_frame',
    'haar_orthogonal',
    'hyperspherical_simplex',
    'tiled_simplex_frame',
]

# ---------------------------------------------------------------------


def hyperspherical_simplex(dimensions: int) -> np.ndarray:
    """
    Build the regular ``dimensions``-simplex inscribed on the unit sphere.

    Three steps, in float64 throughout:

    1. The ``N`` rows of the identity are pairwise a distance ``sqrt(2)`` apart.
       To find a further point ``P = (c, ..., c)`` at the same distance from
       every one of them, solve
       ``(c - 1)^2 + (N - 1) c^2 = 2``, whose positive root is
       ``c = (1 + sqrt(N + 1)) / N``.
    2. Subtract the centroid of the ``N + 1`` points. This is what makes them
       sum to zero, and a zero-sum equiangular set of unit vectors is forced to
       have mutual inner product ``-1 / N``.
    3. Normalize each row onto the unit sphere.

    Steps 1-3 are used rather than a per-vertex closed form because the
    centering has to happen over all ``N + 1`` vertices at once; centering only
    the ``N`` basis rows and closing with their negated sum does **not** give
    an equiangular set (measured at N=16: row norms spanned 0.24 to 0.998 and
    inner products spanned -0.066 to -0.0037, against a target of -0.0625 for
    every pair).

    :param dimensions: Ambient dimensionality ``N``. Must be at least 1.
    :type dimensions: int
    :return: Array of shape ``(N + 1, N)``, one unit vertex per row, with
        every pairwise inner product equal to ``-1 / N``.
    :rtype: np.ndarray
    :raises ValueError: If ``dimensions`` is less than 1.
    """

    if dimensions < 1:
        raise ValueError(f"dimensions must be >= 1, got {dimensions}")

    frame = np.identity(dimensions, dtype=np.float64)

    coordinate = (1.0 + math.sqrt(dimensions + 1.0)) / dimensions
    frame = np.vstack([frame, np.full((1, dimensions), coordinate, dtype=np.float64)])

    frame -= frame.mean(axis=0, keepdims=True)

    norms = np.linalg.norm(frame, axis=1, keepdims=True)

    return frame / norms


def haar_orthogonal(dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an orthogonal matrix uniformly from the Haar measure on O(N).

    QR decomposition of a standard Gaussian matrix gives a Haar-distributed
    matrix up to a per-column sign, because ``QR`` pins ``R`` to a fixed
    sign convention. Scaling the columns of ``Q`` by ``sign(diag(R))`` undoes
    that convention, leaving the Haar distribution intact.

    :param dimensions: Ambient dimensionality ``N``. Must be at least 1.
    :type dimensions: int
    :param rng: Random generator used for the Gaussian draw.
    :type rng: np.random.Generator
    :return: Array of shape ``(N, N)`` satisfying ``Q.T @ Q == I``.
    :rtype: np.ndarray
    :raises ValueError: If ``dimensions`` is less than 1.
    """

    if dimensions < 1:
        raise ValueError(f"dimensions must be >= 1, got {dimensions}")

    gaussian = rng.normal(size=(dimensions, dimensions))
    q, r = np.linalg.qr(gaussian)

    # A zero diagonal entry of R has an undefined sign; np.where picks +1 there
    # rather than propagating the ambiguity into the phase correction.
    phase = np.where(np.diagonal(r) >= 0.0, 1.0, -1.0)

    return q * phase


def tiled_simplex_frame(dimensions: int, width: int) -> np.ndarray:
    """
    Build a frame by repeating one simplex's vertices until ``width`` is met.

    This is the **degenerate** construction, retained so that configurations
    written against it keep loading and keep producing their original values.
    A width at or below ``dimensions + 1`` is exact. Beyond it the frame holds
    only ``dimensions + 1`` distinct columns, each repeated, and every repeated
    pair sits at inner product exactly ``1.0``. Prefer
    :func:`crystal_frame` for any new configuration.

    :param dimensions: Ambient dimensionality ``N``. Must be at least 1.
    :type dimensions: int
    :param width: Number of columns. Must be at least 1.
    :type width: int
    :return: Array of shape ``(N, width)``, columns unit-norm, drawn from the
        ``N + 1`` simplex vertices.
    :rtype: np.ndarray
    :raises ValueError: If ``dimensions`` or ``width`` is less than 1.
    """

    if dimensions < 1:
        raise ValueError(f"dimensions must be >= 1, got {dimensions}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    frame = hyperspherical_simplex(dimensions).T  # (N, N + 1)
    repeats = int(np.ceil(width / frame.shape[1]))

    return np.tile(frame, (1, repeats))[:, :width].astype(np.float32)


def crystal_frame(
        dimensions: int,
        width: int,
        candidate_trials: int = 5,
        seed=None
) -> np.ndarray:
    """
    Pack a frame of arbitrary width from rotated simplexes on the sphere.

    A single simplex supplies exactly ``dimensions + 1`` directions, so a wider
    frame needs several. Packing them is a spherical code problem: placing each
    new simplex greedily, choosing the rotation that minimizes its largest
    cosine similarity to everything already placed,

        R* = argmin_R  max_{i, j}  <v_i(R), w_j>

    maximizes the minimum angular separation between any two columns, which is
    the Grassmannian-line packing objective. ``candidate_trials`` rotations are
    sampled per simplex and the best is kept, so the search is stochastic --
    ``seed`` makes it reproducible, and ``seed=None`` (the default) makes each
    frame independent.

    The last simplex is sliced to fit, so a width that is not a whole multiple
    of ``dimensions + 1`` still produces exactly ``width`` columns.

    :param dimensions: Ambient dimensionality ``N``. Must be at least 1.
    :type dimensions: int
    :param width: Number of columns to produce. Must be at least 1.
    :type width: int
    :param candidate_trials: Rotations sampled per simplex, from which the
        least-conflicting one is kept. Values below 2 fall back to a single
        random rotation, i.e. no search. Defaults to 5.
    :type candidate_trials: int
    :param seed: Anything ``np.random.default_rng`` accepts -- an integer, a
        sequence of integers, or a ``SeedSequence``. ``None`` (the default)
        draws a fresh, unreproducible frame on every construction. An integer
        gives a frame that is stable across builds and across a ``.keras`` round
        trip of the *config* (the finished frame is a saved weight, so it is
        restored bit-exactly regardless of this value). Pass a
        ``SeedSequence`` when the frame must be decorrelated from another drawn
        in the same run: two calls given the same integer reproduce the same
        first simplex, which matters wherever a second frame must not coincide
        with the first.
    :type seed: Any
    :return: Array of shape ``(N, width)`` with unit-norm columns.
    :rtype: np.ndarray
    :raises ValueError: If ``dimensions`` or ``width`` is less than 1.
    :raises ValueError: If ``candidate_trials`` is less than 1.
    """

    if dimensions < 1:
        raise ValueError(f"dimensions must be >= 1, got {dimensions}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if candidate_trials < 1:
        raise ValueError(f"candidate_trials must be >= 1, got {candidate_trials}")

    rng = np.random.default_rng(seed)
    simplex = hyperspherical_simplex(dimensions)
    num_simplexes = int(math.ceil(width / (dimensions + 1)))

    stacked = None

    for index in range(num_simplexes):
        if index == 0 or candidate_trials < 2 or stacked is None:
            # Nothing to repel from yet, or the caller declined a search.
            rotation = haar_orthogonal(dimensions, rng)
            best = simplex @ rotation
        else:
            best = None
            best_conflict = math.inf
            # `stacked` is the vstack of every vertex placed so far, built once
            # per simplex. Re-deriving it inside this loop re-copied the whole
            # placed set on every trial.
            for _ in range(candidate_trials):
                rotation = haar_orthogonal(dimensions, rng)
                candidate = simplex @ rotation
                conflict = float(np.max(candidate @ stacked.T))

                if conflict < best_conflict:
                    best_conflict = conflict
                    best = candidate

        stacked = best if stacked is None else np.vstack([stacked, best])

    return stacked[:width].T.astype(np.float32)

# ---------------------------------------------------------------------
