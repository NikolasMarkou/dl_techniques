"""A K-Means centroid that owns no data mass must keep its distinctiveness.

Single-claim guard (plan-2026-08-26T061816-c515641a, E3 / F-3).

**The claim.** ``KMeansLayer._update_centroids`` divides the weighted point sum
by ``sum_weights + epsilon``. When a centroid owns essentially zero
responsibility the quotient is ``0/epsilon -> 0``, so its EMA target becomes the
ORIGIN rather than "leave it alone". Every training call therefore drags a dead
centroid toward the data manifold even though not one point voted for it.

**Why this guard does NOT assert on ``||c||``.** Two measured reasons, both from
``findings/kmeans-rbf-numerics.md`` § 3.3 and re-measured on the fixture below:

1. *The transient is non-monotonic.* A centroid at ``||c|| = 56.6`` (D=8) shrinks
   every one of the first 23 steps (56.6 -> 0.72), then **overshoots back out** to
   7.24 by step 30 and 7.77 by step 40 -- from momentum inertia, not repulsion
   (the effect survives ``repulsion_strength=0.0``, where the norm reaches 8.66 at
   step 34). A norm assertion sampled anywhere in 25..50 passes for entirely the
   wrong reason.
2. *The steady state is a MERGER, not a collapse to zero.* Over 150 steps all four
   centroids converge to near-identical norms (~0.58-0.63). "``||c||`` is small"
   is not the failure; "the dead centroid is no longer distinguishable from the
   live ones" is.

**What is asserted instead.** The dead centroid's *minimum pairwise distance to
the live centroids* -- a separation measure that is blind to where the pack
happens to sit. Two thresholds, both anchored to numbers measured on THIS
fixture against the unmodified source:

* **Absolute floor.** It must retain at least half its initial separation.
  Measured pre-fix at step 60: ``0.3393`` against an initial ``56.4632`` -- it
  has kept **0.6%**. Post-fix the dead centroid receives an exactly-zero update
  (zero mass *and* zero repulsion, being far beyond ``min_distance``), so the
  separation is unchanged. The half-way threshold sits in a gap spanning two
  orders of magnitude; it is not a tuned number.
* **Distinctiveness.** It must stay further from every live centroid than the
  live centroids are from each other -- i.e. still an outlier, not a member of
  the pack. Measured pre-fix at step 60: min-separation ``0.3393`` versus a
  largest live-live distance of ``0.5917``. The dead centroid has moved *inside*
  the pack. This is the assertion that states the actual failure mode.

**Why 60 steps.** Fewer than ~25 and the centroid is still in free fall with
large separation left; 25..50 is the momentum overshoot window. Step 60 is the
first point past the overshoot where the merger is stable.

The second test pins the other half of the fix (SC-6): for a configuration where
every cluster owns real mass, the masked update must be a **strict no-op**. Its
expected values were captured from the UNMODIFIED module before the fix was
written and are transcribed here verbatim -- the post-fix code is never compared
against itself.

The third and fourth tests guard the two ways the mask itself can go wrong, both
found by adversarial review of the first shipped threshold (``1e-3``):

* the threshold is high enough to call an ordinary VQ codebook dead wholesale
  (mass is ``N/K`` under near-uniform assignments), and
* "the data-driven pull is zeroed" is read as "the centroid is held still", which
  is false the moment the momentum buffer is non-zero at the instant of death.
"""

from typing import Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.mixtures.kmeans import KMeansLayer

# ---------------------------------------------------------------------
# fixtures (plain functions -- these are deterministic constructors, not state)
# ---------------------------------------------------------------------

_DEAD_STEPS = 60
"""Training calls: past the momentum-overshoot window measured at steps 25..50."""


def _dead_cluster_case() -> Tuple[np.ndarray, np.ndarray]:
    """Build the dead-centroid fixture.

    Three live centroids sit on a tight ring of radius 0.3 around the origin with
    the data drawn around them, so a centroid dragged to the data mean lands
    *inside* the pack. The fourth centroid sits at ``||c|| = 56.57`` -- far enough
    that its softmax responsibility underflows to exactly 0.0 and far beyond the
    default ``min_distance=1.0``, so repulsion on it is exactly zero too.

    :return: ``(initial_centroids, data)`` of shapes ``(4, 8)`` and ``(96, 8)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(0)
    live = np.array(
        [
            [0.30, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.15, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.15, -0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    dead = np.full((1, 8), 20.0, dtype="float32")
    data = np.concatenate(
        [live[i] + 0.02 * rng.standard_normal((32, 8)) for i in range(3)], axis=0
    ).astype("float32")
    return np.concatenate([live, dead], axis=0), data


def _min_separation(centroids: np.ndarray) -> float:
    """Smallest distance from the dead centroid (row 3) to any live centroid.

    :param centroids: Centroid matrix of shape ``(4, D)``.
    :type centroids: np.ndarray
    :return: ``min_j ||c_3 - c_j||`` over the three live rows.
    :rtype: float
    """
    return float(np.min(np.linalg.norm(centroids[3] - centroids[:3], axis=-1)))


def _widest_live_pair(centroids: np.ndarray) -> float:
    """Largest pairwise distance among the three live centroids.

    :param centroids: Centroid matrix of shape ``(4, D)``.
    :type centroids: np.ndarray
    :return: ``max_{i<j<3} ||c_i - c_j||``.
    :rtype: float
    """
    return float(
        max(
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
    )


# ---------------------------------------------------------------------
# E3 -- the guard
# ---------------------------------------------------------------------


def test_the_dead_centroid_keeps_its_distinctiveness() -> None:
    """A zero-mass centroid must not be dragged into the live centroid pack."""
    initial, data = _dead_cluster_case()

    layer = KMeansLayer(n_clusters=4, cluster_axis=-1)
    layer.build((None, 8))
    layer.centroids.assign(initial)

    x = keras.ops.convert_to_tensor(data)

    # Precondition: the dead centroid really does own no mass. Without this the
    # test could go green because the fixture never produced a dead cluster.
    responsibilities = np.array(layer(x, training=False))
    dead_mass = float(responsibilities[:, 3].sum())
    # Measured exactly 0.0 (softmax underflow). The bound is deliberately three
    # orders below `_MIN_CLUSTER_MASS` so the precondition does not merely restate
    # the threshold under test.
    assert dead_mass < 1e-9, (
        f"fixture is not a dead-cluster fixture: centroid 3 owns mass {dead_mass}"
    )

    initial_separation = _min_separation(np.array(layer.centroids))

    for _ in range(_DEAD_STEPS):
        layer(x, training=True)

    final = np.array(layer.centroids)
    separation = _min_separation(final)
    live_spread = _widest_live_pair(final)

    # Absolute floor. Measured pre-fix: 0.3393 of an initial 56.4632 (0.6%).
    assert separation >= 0.5 * initial_separation, (
        f"the dead centroid lost {100.0 * (1.0 - separation / initial_separation):.1f}% "
        f"of its separation from the live centroids in {_DEAD_STEPS} steps "
        f"({initial_separation:.4f} -> {separation:.4f}); it owns no data mass and "
        f"must not be pulled by the data-driven EMA target"
    )

    # Distinctiveness. Measured pre-fix: 0.3393 vs a live-live spread of 0.5917 --
    # the dead centroid ends up closer to a live centroid than the live centroids
    # are to each other.
    assert separation > live_spread, (
        f"the dead centroid has merged into the live pack: its nearest-live distance "
        f"is {separation:.4f} while the live centroids span {live_spread:.4f}"
    )


# ---------------------------------------------------------------------
# SC-6 -- the fix must be a strict no-op wherever every cluster owns mass
# ---------------------------------------------------------------------

_ALIVE_EXPECTED = np.array(
    [
        [
            1.00731360912323,
            0.011745413765311241,
            -0.5044094324111938,
            0.25025972723960876,
            1.9992262125015259,
        ],
        [
            -1.0094702243804932,
            0.498892217874527,
            -0.0005124262534081936,
            -0.25641539692878723,
            0.9934678673744202,
        ],
        [
            -0.00208844942972064,
            -1.0019891262054443,
            1.5016597509384155,
            0.7402184009552002,
            -0.9940881133079529,
        ],
    ],
    dtype="float32",
)
"""Post-step centroids captured from the PRE-FIX module (commit 4c043ef6b's
``kmeans.py``) before the mass mask was written, on GPU 1, and reproduced
bit-identically across two independent processes. Transcribed rather than
recomputed so the fix can never be graded against itself."""


def _all_alive_case() -> Tuple[np.ndarray, np.ndarray]:
    """Build the all-alive fixture: 3 clusters, D=5, 8 points each, no dead cluster.

    :return: ``(initial_centroids, data)`` of shapes ``(3, 5)`` and ``(24, 5)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(20260826)
    centroids = np.array(
        [
            [1.0, 0.0, -0.5, 0.25, 2.0],
            [-1.0, 0.5, 0.0, -0.25, 1.0],
            [0.0, -1.0, 1.5, 0.75, -1.0],
        ],
        dtype="float32",
    )
    data = np.concatenate(
        [centroids[i] + 0.3 * rng.standard_normal((8, 5)) for i in range(3)], axis=0
    ).astype("float32")
    return centroids, data


def test_the_mass_mask_is_a_strict_no_op_when_every_cluster_is_alive() -> None:
    """With every cluster carrying real mass, centroids must be bit-identical to pre-fix."""
    initial, data = _all_alive_case()

    layer = KMeansLayer(n_clusters=3, cluster_axis=-1, random_seed=0)
    layer.build((None, 5))
    layer.centroids.assign(initial)

    x = keras.ops.convert_to_tensor(data)

    # Precondition: every cluster is alive by a wide margin (measured 8.0 each,
    # against a threshold of 1e-3). If this ever stops holding, the bit-identity
    # assertion below would be exercising the masked branch instead.
    masses = np.array(layer(x, training=False)).sum(axis=0)
    assert float(masses.min()) > 1.0, f"fixture has a near-dead cluster: masses={masses}"

    for _ in range(3):
        layer(x, training=True)

    max_abs_diff = float(np.max(np.abs(np.array(layer.centroids) - _ALIVE_EXPECTED)))
    assert max_abs_diff == 0.0, (
        f"the mass mask changed the all-alive update: max |c_new - c_prefix| = "
        f"{max_abs_diff!r} (must be exactly 0.0)"
    )


# ---------------------------------------------------------------------
# The threshold must not be so high that an ordinary VQ codebook is all "dead"
# ---------------------------------------------------------------------


def test_a_large_codebook_with_one_point_is_not_frozen_wholesale() -> None:
    """At a VQ-realistic ``K`` with a small ``N``, the data term must still move centroids.

    ``_MIN_CLUSTER_MASS`` is an ABSOLUTE floor, so the regime it must clear is the
    one where every cluster's mass is small for a structural reason rather than a
    dead-cluster reason: near-uniform assignments give every cluster exactly
    ``N / K``. Measured here (``K=1024``, ``N=1``, ``temperature=1e6``): every mass
    is ``9.766e-04``. At the originally-shipped ``1e-3`` that is **1024 of 1024
    clusters judged dead**, the EMA data term is switched off for the whole
    codebook, and with ``repulsion_strength=0.0`` the update is exactly zero -- a
    regression against the pre-mask code, which trained fine here. Measured at
    ``K=2048/8192/65536`` the freeze is likewise total.

    RED-proof: restoring ``_MIN_CLUSTER_MASS = 1e-3`` makes ``max |dc| == 0.0`` and
    the mean-distance assertion below fails with no movement at all.
    """
    n_clusters, features = 1024, 16
    layer = KMeansLayer(
        n_clusters=n_clusters,
        cluster_axis=-1,
        temperature=1e6,          # near-uniform responsibilities
        repulsion_strength=0.0,   # so the ONLY remaining force is the data term
        random_seed=0,
    )
    layer.build((None, features))

    point = np.full((1, features), 2.0, dtype="float32")
    x = keras.ops.convert_to_tensor(point)

    # Precondition: this really is the small-mass regime the threshold must clear.
    masses = np.array(layer(x, training=False)).sum(axis=0)
    assert float(masses.max()) < 1e-3, (
        f"fixture is not in the N/K regime: max mass {float(masses.max())}"
    )

    before = np.array(layer.centroids)
    distance_before = float(np.mean(np.linalg.norm(before - point, axis=-1)))

    for _ in range(5):
        layer(x, training=True)

    after = np.array(layer.centroids)
    moved = float(np.max(np.abs(after - before)))
    distance_after = float(np.mean(np.linalg.norm(after - point, axis=-1)))

    assert moved > 0.0, (
        f"the EMA data term is switched off for the entire {n_clusters}-entry codebook: "
        f"max |dc| = {moved} after 5 training steps at mass {float(masses.max()):.4e} "
        f"per cluster. _MIN_CLUSTER_MASS must stay below the N/K floor of a real codebook"
    )
    # Measured post-fix: 8.0017 -> 6.9746 in 5 steps (max |dc| = 0.2695).
    assert distance_after < distance_before, (
        f"the codebook did not move toward the data: mean distance "
        f"{distance_before:.6f} -> {distance_after:.6f}"
    )


# ---------------------------------------------------------------------
# A masked centroid is NOT "frozen" -- pin what it actually does
# ---------------------------------------------------------------------


def test_a_centroid_that_dies_after_moving_coasts_on_its_momentum() -> None:
    """A centroid that goes dead with a non-zero momentum buffer keeps coasting.

    The E3 fixture above starts dead, so its momentum buffer is zero and "the data
    term is masked" and "the centroid does not move" coincide. They are not the
    same statement. Here the centroid is alive for 20 steps (momentum buffer norm
    reaches ``0.118``), then the data moves far away and its mass collapses to
    ``2.05e-21``. Measured displacement over the next 10 steps:
    ``0.01062, 0.02018, 0.02878, 0.03652, 0.04349, 0.04976, 0.05541, 0.06049,
    0.06506, 0.06917`` -- strictly increasing, with per-step increments decaying
    geometrically by the ``momentum`` factor (``0.01062, 0.00956, 0.00860, ...``).

    What the mask DOES buy is direction: the centroid does not set off toward the
    data cloud it owns no mass in. Measured distance to that cloud over the same
    10 steps: ``119.745 -> 119.711``, i.e. 0.03 out of 119.7.
    """
    features = 4
    rng = np.random.default_rng(3)
    near_data = np.concatenate(
        [
            np.array([1.0, 0.0, 0.0, 0.0]) + 0.05 * rng.standard_normal((32, features)),
            np.array([-1.0, 0.0, 0.0, 0.0]) + 0.05 * rng.standard_normal((32, features)),
        ],
        axis=0,
    ).astype("float32")

    layer = KMeansLayer(
        n_clusters=3, cluster_axis=-1, temperature=1.0, repulsion_strength=0.0
    )
    layer.build((None, features))
    layer.centroids.assign(
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0]],
            dtype="float32",
        )
    )

    x_near = keras.ops.convert_to_tensor(near_data)
    for _ in range(20):
        layer(x_near, training=True)

    momentum_norm = float(np.linalg.norm(np.array(layer.centroid_momentum)[2]))
    assert momentum_norm > 1e-3, (
        f"fixture did not accumulate momentum on centroid 2: {momentum_norm}"
    )

    far_point = np.full((1, features), 60.0, dtype="float32")
    x_far = keras.ops.convert_to_tensor(np.repeat(far_point, 64, axis=0))

    mass = float(np.array(layer(x_far, training=False))[:, 2].sum())
    assert mass < 1e-6, f"centroid 2 did not go dead under the far data: mass {mass}"

    start = np.array(layer.centroids)[2].copy()
    distance_before = float(np.linalg.norm(start - far_point[0]))
    displacements = []
    for _ in range(10):
        layer(x_far, training=True)
        displacements.append(
            float(np.linalg.norm(np.array(layer.centroids)[2] - start))
        )
    distance_after = float(np.linalg.norm(np.array(layer.centroids)[2] - far_point[0]))

    # The module docstring must NOT claim a dead centroid is held still: it is not.
    assert displacements[0] > 0.0, (
        "a dead centroid with a non-zero momentum buffer did not move at all; if this "
        "is now true, the momentum term changed and the docstring needs revisiting"
    )
    assert all(
        displacements[i] > displacements[i - 1] for i in range(1, len(displacements))
    ), f"residual momentum motion was expected to be monotone: {displacements}"

    increments = [
        displacements[i] - (displacements[i - 1] if i else 0.0)
        for i in range(len(displacements))
    ]
    assert all(
        increments[i] < increments[i - 1] for i in range(1, len(increments))
    ), f"the momentum tail was expected to decay: {increments}"

    # What the mask buys: the centroid is not pulled toward data it owns no mass in.
    assert abs(distance_after - distance_before) < 0.01 * distance_before, (
        f"the masked centroid travelled toward the cloud it owns no mass in: distance "
        f"{distance_before:.5f} -> {distance_after:.5f}"
    )
