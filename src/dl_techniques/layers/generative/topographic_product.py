"""The Topographic Product of Student's-t construction, as a differentiable
deterministic layer.

Why this exists
---------------
A standard VAE's latent variables are *independent*: the prior factorizes, and
nothing in the objective encourages any relation between coordinate ``i`` and
coordinate ``j``. Topographic generative models drop exactly that assumption
while keeping the marginal a clean Student-t. The trick (Welling, Osindero &
Hinton, 2003; Osindero, 2004) is that a univariate Student-t is a scale mixture
of normals: with `Z ~ N(0,1)` and `U_1..U_nu ~ N(0,1)` independent,

    T = Z * sqrt(nu / sum_i U_i^2)                                          (Eq. 2)

and sharing the `U`s between *neighbouring* `T`s makes neighbours correlated
while `T`s that are far apart stay independent. The sharing pattern is the
**adjacency structure W**: `T` is a deterministic function of two Gaussian
vectors `z` and `u`,

    T = sqrt(2) * (Z - mu) / sqrt(nu * (W u^2))                              (Eq. 6)

so the whole nonlinear, heavy-tailed prior is the *first layer of the
generative decoder*. That is what makes an energy-based TPoT model trainable by
variational inference — every latent variable is Gaussian, and the topography is
ordinary arithmetic.

What this layer computes
------------------------
The energy fed to the `1/sqrt(.)` in Eq. 6, for three ways of correlating `u`:

- ``temporal_coherence="none"`` — a single timestep, ``W u^2``. With
  ``neighborhood_size = 1`` this is exactly the identity and the model is a
  plain VAE with a Student's-t latent; with a larger `K` it is the local
  topographic neighbourhood of Osindero (2004) and the Figure 3 experiment.
- ``temporal_coherence="stationary"`` — Eq. 8, "Bubbles"
  (Hyvarinen, Hurri & Varrynen, 2004): `u` from neighbouring timesteps is summed
  at the *same* spatial location, giving `cov(T_{l,i}^2, T_{l-1,i}^2) > 0`.
- ``temporal_coherence="shifting"`` — Eq. 9, the Topographic VAE's own mode: the
  contribution of timestep ``l + delta`` is first *cyclically rolled by delta
  within its capsule*, giving `cov(T_{l,i}^2, T_{l-1,i-1}^2) > 0`. This is the
  inductive bias that makes a transformation of the input show up as a roll
  inside a capsule rather than as a change of capsule.

The bake-in that makes this cheap
---------------------------------
Each term of the sum is a *linear* map. Writing the roll as a permutation matrix
`R_delta` with `(R_delta v)[i] = v[(i - delta) mod D]` (the same index table as
:class:`dl_techniques.layers.capsules.CapsuleRoll`),
the whole window is

    energy_l = sum_{delta=-L..L} W R_delta u_{l+delta}

so all `2L+1` terms are precomputed once into a single constant tensor

    G[delta + L] = W R_delta          shape (2L+1, D, D)

and one einsum replaces the loop. `G` is stored as a **non-trainable weight
computed inside an `add_weight` initializer**, never assigned in `build()`:
Keras runs a parent's first `build()` inside a `StatelessScope` that discards
assignments, so a table assigned in `build()` reads back as all zeros in every
real model while a unit test that calls `.build()` directly still passes.

Why `torus_2d` cannot be shifted
--------------------------------
A `shifting` window needs a *cyclic* direction to roll along. A one-dimensional
capsule has one; a two-dimensional torus lattice has no single cyclic axis
without choosing one, and choosing one silently makes the model's capability
depend on an arbitrary axis ordering. Section C.3 of the paper proposes an
extension with a per-axis roll and a product denominator; that is a different
layer with a different neighbourhood structure, so `topography="torus_2d"`
combined with `temporal_coherence != "none"` raises in `__init__` instead.

References:
    - Welling, Osindero & Hinton, 2003. Learning Sparse Topographic
      Representations with Products of Student-t Distributions. NeurIPS.
      (https://papers.nips.cc/paper/2003/hash/fa46f3c4e812052a6f1b7f19bdae41f-Abstract.html)
    - Osindero, 2004. Contrastive Topographic Models. PhD thesis, U. London.
    - Osindero et al., 2006. Topographic Product Models Applied to Natural
      Scene Statistics. Neural Computation 18(2). (https://doi.org/10.1162/neco.2006.18.2.381)
    - Hyvarinen, Hurri & Varrynen, 2004. A Unifying Framework for Natural Image
      Statistics: Spatiotemporal Activity Bubbles. Neurocomputing 58-60.
      (https://doi.org/10.1016/j.neucom.2004.09.007)
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: The three correlation structures this layer implements. ``"none"`` is a
#: single timestep, ``"stationary"`` is Eq. 8 (Bubbles) and ``"shifting"`` is
#: Eq. 9 (the Topographic VAE).
TEMPORAL_COHERENCE_TYPES = ("none", "stationary", "shifting")

#: Latent layouts. ``"capsule_1d"`` is a bank of disjoint circular capsules —
#: one topography per capsule, the setting of Figure 1 and Figure 4.
#: ``"torus_2d"`` is a single 2-D lattice, the Figure 3 experiment.
TOPOGRAPHY_TYPES = ("capsule_1d", "torus_2d")

TemporalCoherenceType = Literal["none", "stationary", "shifting"]
TopographyType = Literal["capsule_1d", "torus_2d"]

#: Added under the ``1/sqrt(.)`` of Eq. 6. Its floor is 1e-12, three orders
#: below the smallest positive energy a window of ``neighborhood_size`` standard
#: normals can produce at this layer's operating precision, so it never
#: materially resizes a healthy energy; it exists to make an all-zero energy
#: (every ``u`` exactly 0) produce a finite value instead of ``inf``.
ENERGY_EPSILON = 1e-12


# ---------------------------------------------------------------------
# pure helpers — one definition per formula, shared by build/call/tests
# ---------------------------------------------------------------------


def build_window_matrix_1d(latent_dim: int, neighborhood_size: int) -> np.ndarray:
    """Circulant neighbourhood-sum matrix for one circular capsule.

    ``W[i, j] = 1`` when ``(j - i) mod latent_dim < neighborhood_size``, so row
    ``i`` sums the ``neighborhood_size`` coordinates starting at ``i`` and
    wrapping around the end. ``neighborhood_size == latent_dim`` gives the
    all-ones matrix, which is the degenerate "every variable shares everything"
    case the paper reports as the invariant-representation failure (Table 5,
    ``K = 18`` on MNIST).

    :param latent_dim: Capsule dimensionality ``D``.
    :type latent_dim: int
    :param neighborhood_size: Window width ``K``.
    :type neighborhood_size: int
    :return: A ``(latent_dim, latent_dim)`` float64 matrix.
    :rtype: np.ndarray
    """
    rows = np.arange(latent_dim, dtype=np.int64)[:, None]
    cols = np.arange(latent_dim, dtype=np.int64)[None, :]
    offsets = (cols - rows) % latent_dim
    return (offsets < neighborhood_size).astype(np.float64)


def build_window_matrix_2d(
    grid_height: int, grid_width: int, kernel_height: int, kernel_width: int
) -> np.ndarray:
    """Block-circulant neighbourhood-sum matrix for a 2-D torus.

    The lattice is flattened row-major to ``grid_height * grid_width`` and the
    returned matrix sums the ``kernel_height x kernel_width`` neighbourhood with
    wrap-around in both axes — the "5x5 kernel of 1's, stride 1, cyclic padding"
    of the paper's Section 6.2.

    :param grid_height: Lattice rows.
    :type grid_height: int
    :param grid_width: Lattice columns.
    :type grid_width: int
    :param kernel_height: Neighbourhood extent along the row axis.
    :type kernel_height: int
    :param kernel_width: Neighbourhood extent along the column axis.
    :type kernel_width: int
    :return: A ``(H*W, H*W)`` float64 matrix.
    :rtype: np.ndarray
    """
    latent_dim = grid_height * grid_width
    window = np.zeros((latent_dim, latent_dim), dtype=np.float64)
    for i in range(grid_height):
        for j in range(grid_width):
            source = i * grid_width + j
            for di in range(kernel_height):
                for dj in range(kernel_width):
                    target = ((i + di) % grid_height) * grid_width + (
                        (j + dj) % grid_width
                    )
                    window[source, target] = 1.0
    return window


def build_roll_matrix_1d(latent_dim: int, shift: int) -> np.ndarray:
    """Permutation matrix for a cyclic roll within one capsule.

    ``(R_shift v)[i] = v[(i - shift) % D]`` — **exactly** ``CapsuleRoll``'s
    index table, so the two operators are interchangeable. MEASURED: at
    ``latent_dim=4, shift=1`` this sends the source vector ``[0,1,2,3]`` to
    ``[3,0,1,2]``, the paper's written ``Roll_1`` example.

    The sign of ``shift`` is the whole content of this function and it is easy to
    get backwards: a ``(i - 1 + shift)`` here produces ``R_1 = identity``, which
    makes ``shifting`` coherence *identical* to ``stationary`` while every
    shape test and every layer-versus-oracle comparison still passes, because both
    sides would read the same wrong helper. ``tests/.../test_topographic_product.py``
    therefore pins this matrix against the paper's literal example rather than
    against a second transcription.

    :param latent_dim: Capsule dimensionality ``D``.
    :type latent_dim: int
    :param shift: Cyclic steps; ``0`` gives the identity.
    :type shift: int
    :return: A ``(latent_dim, latent_dim)`` float64 permutation matrix.
    :rtype: np.ndarray
    """
    roll = np.zeros((latent_dim, latent_dim), dtype=np.float64)
    for i in range(latent_dim):
        roll[i, (i - int(shift)) % latent_dim] = 1.0
    return roll


def build_window_stack(
    window: np.ndarray,
    latent_dim: int,
    coherence_window: int,
    temporal_coherence: str,
) -> np.ndarray:
    """Assemble the ``(2L+1, D, D)`` stack ``G`` with ``G[delta+L] = W R_delta``.

    :param window: The single-timestep neighbourhood matrix ``W``.
    :type window: np.ndarray
    :param latent_dim: Latent dimensionality ``D``.
    :type latent_dim: int
    :param coherence_window: Half-width ``L`` of the temporal window.
    :type coherence_window: int
    :param temporal_coherence: One of :data:`TEMPORAL_COHERENCE_TYPES`.
    :type temporal_coherence: str
    :return: A ``(2L+1, D, D)`` float64 stack.
    :rtype: np.ndarray
    """
    stack = np.empty(
        (2 * coherence_window + 1, latent_dim, latent_dim), dtype=np.float64
    )
    for index, delta in enumerate(range(-coherence_window, coherence_window + 1)):
        if temporal_coherence == "shifting" and delta != 0:
            roll = build_roll_matrix_1d(latent_dim, delta)
            stack[index] = window @ roll
        else:
            # "none" and "stationary" both use W unpermuted: "none" only ever
            # has L == 0, and "stationary" is Eq. 8's deliberate absence of a
            # roll.
            stack[index] = window
    return stack


def windowed_stack_energy(
    window_stack: np.ndarray, squared_u: np.ndarray
) -> np.ndarray:
    """Reference NumPy evaluation of the layer's windowed energy.

    Transcribes the einsum and the edge-replicated gather of
    :meth:`TopographicProduct.call` step by step, for use as a test oracle. It
    consumes the *same* arguments the layer does, and is deliberately written
    as an explicit Python loop over the window offsets so that a mistake in the
    layer's vectorized gather cannot cancel out against a matching mistake here.

    :param window_stack: The ``(2L+1, D, D)`` stack ``G``.
    :type window_stack: np.ndarray
    :param squared_u: ``(B, S, D)`` squared inputs.
    :type squared_u: np.ndarray
    :return: ``(B, S, D)`` energies.
    :rtype: np.ndarray
    """
    num_offsets = window_stack.shape[0]
    coherence_window = (num_offsets - 1) // 2
    sequence = squared_u.shape[1]

    energy = np.zeros(squared_u.shape, dtype=np.float64)
    for index in range(num_offsets):
        delta = index - coherence_window
        # Edge replication via clipped indices, matching call()'s gather.
        positions = np.clip(
            np.arange(sequence) + delta, 0, sequence - 1
        )
        gathered = squared_u[:, positions, :]
        energy += np.einsum("de,bse->bsd", window_stack[index], gathered)
    return energy


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.generative.topographic_product")
class TopographicProduct(keras.layers.Layer):
    """Build topographic Student-t latent variables from Gaussian `z` and `u`.

    Implements Eq. 6 of Keller & Welling (2022):

    .. code-block:: text

        T = sqrt(2) * (Z - mu) / sqrt(nu * energy + eps)
        energy = sum_{delta=-L..L} W R_delta u_{l+delta}^2

    The three correlation structures of :data:`TEMPORAL_COHERENCE_TYPES` differ
    only in how the window stack `G` is assembled; see the module docstring for
    the three cases and for why `torus_2d` cannot be shifted.

    The `z` and `u` arguments must both be shaped ``(batch, sequence,
    latent_dim)``. The layer has **no trainable weights**: the entire
    neighbourhood structure is a fixed constant, because a learned `W` would no
    longer be the a-priori topographic prior the model is defined by.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────┐
        │  z  (B, S, D)                 │  shape only; no roll
        │  u  (B, S, D)                 │
        └───────────┬───────────────────┘
                    │  gather u[l+delta] for each delta in [-L, L],
                    │  edge-replicated at the sequence boundaries
                    ▼
        ┌───────────────────────────────────────────┐
        │  energy = einsum("sde,bse->bsd", G, u^2)  │  G is (2L+1, D, D),
        │                                           │  non-trainable
        └───────────┬───────────────────────────────┘
                    ▼
        ┌───────────────────────────────────────────┐
        │  T = sqrt(2) * (z - mu) /                  │
        │            sqrt(nu * energy + eps)         │
        └───────────┬───────────────────────────────┘
                    ▼
              output  (B, S, D)

    Input shape:
        Two tensors of ``(batch, sequence, latent_dim)``. In the nested
        ``capsule_1d`` use they arrive pre-reshaped to
        ``(batch, sequence, num_capsules, capsule_dim)`` and flattened on the
        trailing two axes.

    Output shape:
        ``(batch, sequence, latent_dim)``.

    :param num_capsules: Number of capsules ``C``. Each capsule gets its own
        neighbourhood; capsules are statistically independent by construction.
        For ``topography="torus_2d"`` this is ``1``.
    :type num_capsules: int
    :param capsule_dim: Dimensions per capsule ``D``, i.e. the latent width of
        one capsule. For ``topography="torus_2d"`` this is ``grid_height *
        grid_width``.
    :type capsule_dim: int
    :param coherence_window: Half-width ``L`` of the temporal window; ``2L`` is
        the time extent of the induced coherence and ``L = 0`` means single
        inputs. Must be non-negative. Defaults to 0.
    :type coherence_window: int
    :param neighborhood_size: Window width ``K`` within one capsule. Must be at
        least 1 and at most ``capsule_dim``; ``K = capsule_dim`` shares
        everything and is the invariant-representation degenerate case.
    :type neighborhood_size: int
    :param temporal_coherence: One of :data:`TEMPORAL_COHERENCE_TYPES`.
        Defaults to ``"shifting"``, except under ``topography="torus_2d"``, where
        it defaults to ``"none"`` because that layout has no cyclic axis.
        Passing ``"shifting"`` explicitly with a torus raises.
    :type temporal_coherence: str
    :param topography: One of :data:`TOPOGRAPHY_TYPES`. Defaults to
        ``"capsule_1d"``.
    :type topography: str
    :param grid_shape: ``(grid_height, grid_width)`` for
        ``topography="torus_2d"``. Must be ``None`` otherwise, and must satisfy
        ``grid_height * grid_width == capsule_dim``.
    :type grid_shape: Optional[Tuple[int, int]]
    :param kernel_shape: ``(kernel_height, kernel_width)`` for
        ``topography="torus_2d"``, defaulting to ``(neighborhood_size,
        neighborhood_size)``. Must be ``None`` otherwise.
    :type kernel_shape: Optional[Tuple[int, int]]
    :param degrees_of_freedom: The Student's-t degrees of freedom ``nu``
        (Eq. 2). Must be positive. Defaults to 1.
    :type degrees_of_freedom: float
    :param prior_mean: The scalar ``mu`` subtracted from ``z`` in Eq. 6.
        Defaults to 30.0, which is the value the paper initializes topographic
        models to (Section A.2) because it materially speeds convergence and is
        sometimes necessary for topography to form at all in deep models. The
        Figure 3 two-dimensional model used 10.0.
    :type prior_mean: float
    :param epsilon: Floor under ``nu * energy``. Defaults to
        :data:`ENERGY_EPSILON`.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: On a non-positive ``num_capsules`` / ``capsule_dim`` /
        ``degrees_of_freedom``, a negative ``coherence_window``, a
        ``neighborhood_size`` outside ``[1, capsule_dim]``, an unknown
        ``temporal_coherence`` or ``topography``, or a
        ``topography``/``temporal_coherence``/``grid_shape`` combination that
        ``call()`` could not honour.

    :Example:

    >>> import numpy as np
    >>> from keras import ops
    >>> topo = TopographicProduct(
    ...     num_capsules=2, capsule_dim=4, coherence_window=0, neighborhood_size=1
    ... )
    >>> z = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(1, 3, 8)))
    >>> u = ops.convert_to_tensor(np.random.default_rng(1).normal(size=(1, 3, 8)))
    >>> t = topo([z, u])
    >>> tuple(t.shape)
    (1, 3, 8)
    """

    def __init__(
        self,
        num_capsules: int,
        capsule_dim: int,
        coherence_window: int = 0,
        neighborhood_size: int = 1,
        # `None` is the sentinel, NOT "shifting". "shifting" would be resolved to
        # "none" for a torus below, which would silently swallow an explicit
        # request for the one mode the torus cannot honour -- a validation branch
        # that can never fire because its own default defeated it.
        temporal_coherence: Optional[TemporalCoherenceType] = None,
        topography: TopographyType = "capsule_1d",
        grid_shape: Optional[Tuple[int, int]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        degrees_of_freedom: float = 1.0,
        prior_mean: float = 30.0,
        epsilon: float = ENERGY_EPSILON,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # A 2-D torus has no single cyclic axis, so "shifting" is the wrong
        # default there. Resolving the UNSPECIFIED case to "none" lets
        # `topography` alone decide the default, while an EXPLICIT
        # temporal_coherence="shifting" still raises below -- which it must, or
        # the combination would be silently accepted and computed as "none".
        if temporal_coherence is None:
            temporal_coherence = "none" if topography == "torus_2d" else "shifting"

        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if capsule_dim <= 0:
            raise ValueError(f"capsule_dim must be positive, got {capsule_dim}")
        if coherence_window < 0:
            raise ValueError(
                f"coherence_window must be non-negative, got {coherence_window}"
            )
        if not 1 <= neighborhood_size <= capsule_dim:
            raise ValueError(
                f"neighborhood_size must be in [1, capsule_dim] = "
                f"[1, {capsule_dim}], got {neighborhood_size}"
            )
        if temporal_coherence not in TEMPORAL_COHERENCE_TYPES:
            raise ValueError(
                f"temporal_coherence must be one of "
                f"{list(TEMPORAL_COHERENCE_TYPES)}, got {temporal_coherence!r}"
            )
        if topography not in TOPOGRAPHY_TYPES:
            raise ValueError(
                f"topography must be one of {list(TOPOGRAPHY_TYPES)}, "
                f"got {topography!r}"
            )
        if degrees_of_freedom <= 0:
            raise ValueError(
                f"degrees_of_freedom must be positive, got {degrees_of_freedom}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # A 2-D lattice has no single cyclic axis, so a *shifting* window over it
        # would silently make the model depend on an arbitrary axis ordering.
        # Section C.3 of the paper proposes a per-axis variant; until that exists
        # this combination is a construction-time error, not a runtime surprise.
        if topography == "torus_2d" and temporal_coherence != "none":
            raise ValueError(
                f"topography='torus_2d' supports only "
                f"temporal_coherence='none', got {temporal_coherence!r}: a 2-D "
                f"torus has no single cyclic axis for Roll_delta to permute "
                f"along, and choosing one would make the model depend on an "
                f"arbitrary axis ordering."
            )
        if topography == "capsule_1d" and grid_shape is not None:
            raise ValueError(
                f"grid_shape is only meaningful for topography='torus_2d', "
                f"got {grid_shape} with topography='capsule_1d'"
            )
        if topography == "capsule_1d" and kernel_shape is not None:
            raise ValueError(
                f"kernel_shape is only meaningful for topography='torus_2d', "
                f"got {kernel_shape} with topography='capsule_1d'"
            )
        if topography == "torus_2d":
            if grid_shape is None:
                raise ValueError(
                    f"topography='torus_2d' requires grid_shape="
                    f"(grid_height, grid_width)"
                )
            grid_height, grid_width = (int(grid_shape[0]), int(grid_shape[1]))
            if grid_height <= 0 or grid_width <= 0:
                raise ValueError(
                    f"grid_shape entries must be positive, got {grid_shape}"
                )
            if grid_height * grid_width != capsule_dim:
                raise ValueError(
                    f"grid_shape={grid_shape} gives latent_dim "
                    f"{grid_height * grid_width}, which does not equal "
                    f"capsule_dim={capsule_dim}"
                )
            self.grid_shape = (grid_height, grid_width)
            if kernel_shape is None:
                kernel_shape = (neighborhood_size, neighborhood_size)
            else:
                kernel_height, kernel_width = (
                    int(kernel_shape[0]),
                    int(kernel_shape[1]),
                )
                if not (
                    1 <= kernel_height <= grid_height
                    and 1 <= kernel_width <= grid_width
                ):
                    raise ValueError(
                        f"kernel_shape={kernel_shape} must lie in "
                        f"[1, {grid_height}] x [1, {grid_width}]"
                    )
                if kernel_height * kernel_width != neighborhood_size**2:
                    raise ValueError(
                        f"neighborhood_size={neighborhood_size} does not match "
                        f"kernel_shape={kernel_shape}: a 2-D KxK neighbourhood "
                        f"has area K^2, so the two must agree"
                    )
            self.kernel_shape = (int(kernel_shape[0]), int(kernel_shape[1]))
        else:
            self.grid_shape = None
            self.kernel_shape = None

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.coherence_window = coherence_window
        self.neighborhood_size = neighborhood_size
        self.temporal_coherence = temporal_coherence
        self.topography = topography
        self.degrees_of_freedom = degrees_of_freedom
        self.prior_mean = prior_mean
        self.epsilon = epsilon

        self.latent_dim = num_capsules * capsule_dim

        # Declared here, CREATED in build(). Never assigned in build(): a
        # parent's first build() runs inside a StatelessScope that discards the
        # assignment, leaving the table at its initializer value (all zeros for
        # "zeros") in every real model.
        self.window_stack = None

    def build(self, input_shape) -> None:
        """Create the fixed neighbourhood stack from stored configuration.

        :param input_shape: A list of two shapes, each ``(batch, sequence,
            latent_dim)``.
        :type input_shape: Any
        :raises ValueError: If the inputs are not a pair of matching rank-3
            tensors whose last axis is ``latent_dim``.
        """
        shapes = self._resolve_input_shapes(input_shape)
        latent_dim = shapes[0][-1]
        if latent_dim is not None and latent_dim != self.latent_dim:
            raise ValueError(
                f"inputs have last axis {latent_dim}, expected latent_dim="
                f"{self.latent_dim} = num_capsules {self.num_capsules} * "
                f"capsule_dim {self.capsule_dim}"
            )

        window = self._build_window_matrix()
        stack = build_window_stack(
            window,
            self.latent_dim,
            self.coherence_window,
            self.temporal_coherence,
        )

        def _window_stack_initializer(shape, dtype=None, stack=stack):
            """Emit the precomputed stack; nothing to assign later."""
            return ops.cast(ops.convert_to_tensor(stack), dtype or self.compute_dtype)

        self.window_stack = self.add_weight(
            name="window_stack",
            shape=stack.shape,
            initializer=_window_stack_initializer,
            trainable=False,
        )

        logger.info(
            f"TopographicProduct: latent_dim={self.latent_dim}, "
            f"K={self.neighborhood_size}, L={self.coherence_window}, "
            f"temporal_coherence={self.temporal_coherence}, "
            f"topography={self.topography}, "
            f"window_stack={tuple(stack.shape)}"
        )

        super().build(input_shape)

    def _resolve_input_shapes(self, input_shape) -> Tuple[Tuple, Tuple]:
        """Normalise ``input_shape`` into a pair of tuples and validate rank.

        :param input_shape: Keras's ``input_shape`` for a list input.
        :type input_shape: Any
        :return: The two shapes as tuples.
        :rtype: Tuple[Tuple, Tuple]
        """
        shapes = input_shape
        if isinstance(shapes, tuple) and len(shapes) == 1 and isinstance(
            shapes[0], (list, tuple)
        ):
            # (([B, S, D], [B, S, D]),) — the nested form Keras passes for a
            # list input from a parent's call().
            shapes = shapes[0]
        if not isinstance(shapes, (list, tuple)) or len(shapes) != 2:
            raise ValueError(
                f"TopographicProduct expects a pair of inputs (z, u), got "
                f"input_shape={input_shape}"
            )
        resolved = []
        for name, shape in zip(("z", "u"), shapes):
            shape = tuple(shape)
            if len(shape) != 3:
                raise ValueError(
                    f"{name} must be rank 3 (batch, sequence, latent_dim), got "
                    f"shape {shape}"
                )
            resolved.append(shape)
        if resolved[0][-1] != resolved[1][-1]:
            raise ValueError(
                f"z and u must share latent_dim, got {resolved[0][-1]} and "
                f"{resolved[1][-1]}"
            )
        return resolved[0], resolved[1]

    def _build_window_matrix(self) -> np.ndarray:
        """The single-timestep neighbourhood matrix ``W`` for this config.

        :return: A ``(latent_dim, latent_dim)`` float64 matrix.
        :rtype: np.ndarray
        """
        if self.topography == "torus_2d":
            grid_height, grid_width = self.grid_shape
            return build_window_matrix_2d(
                grid_height, grid_width, *self.kernel_shape
            )
        # capsule_1d: one independent circulant block per capsule, so W is
        # block-diagonal and capsules cannot influence each other's energy.
        block = build_window_matrix_1d(self.capsule_dim, self.neighborhood_size)
        window = np.zeros((self.latent_dim, self.latent_dim), dtype=np.float64)
        for capsule in range(self.num_capsules):
            start = capsule * self.capsule_dim
            stop = start + self.capsule_dim
            window[start:stop, start:stop] = block
        return window

    def call(self, inputs, training=None):
        """Construct the topographic Student-t variables ``T``.

        :param inputs: ``[z, u]``, each ``(batch, sequence, latent_dim)``.
        :type inputs: Any
        :param training: Unused; accepted so the layer composes with a parent's
            ``training`` propagation. The operator holds no stochastic or
            normalization behaviour for the flag to switch.
        :type training: Optional[bool]
        :return: ``(batch, sequence, latent_dim)`` topographic variables.
        :rtype: keras.KerasTensor
        """
        z, u = inputs
        z = ops.cast(z, self.compute_dtype)
        u = ops.cast(u, self.compute_dtype)

        if self.coherence_window == 0:
            # (B, S, 1, D) so the einsum below is identical in every mode.
            windowed = ops.expand_dims(u, axis=2)
        else:
            # Gather u at offsets delta = -L..L, one (B, S, 1, D) slice per
            # offset, stacked on a new axis 2 -> (B, S, 2L+1, D).
            #
            # The sequence boundaries are EDGE-REPLICATED, not zero-padded (the
            # padding convention of the paper's Section C.2): a zero would
            # inject an artificial absence of variance at the first and last
            # timesteps, which is exactly the quantity the topographic prior is
            # about. Replicating is done by CLAMPING the gather index, so it
            # stays a single vectorized gather rather than a Python loop that
            # branches on sequence length.
            positions = ops.arange(ops.shape(u)[1])
            offsets = [
                ops.expand_dims(
                    ops.take(
                        u,
                        ops.clip(positions + delta, 0, ops.shape(u)[1] - 1),
                        axis=1,
                    ),
                    axis=2,
                )
                for delta in range(
                    -self.coherence_window, self.coherence_window + 1
                )
            ]
            windowed = ops.concatenate(offsets, axis=2)

        # energy[b, l, d] = sum_{s,e} G[s, d, e] * windowed[b, l, s, e]^2
        squared = ops.square(windowed)
        energy = ops.einsum("sde,blse->bld", self.window_stack, squared)
        # The energy is a NON-NEGATIVE QUANTITY in exact arithmetic -- `G`'s
        # entries are 0 or 1 -- but it is computed as a signed sum of products,
        # so under a perturbed or hand-set `G` it can come out negative and take
        # the `sqrt` below to NaN. `epsilon` does NOT rescue that: adding a
        # positive constant to a negative energy leaves it negative. MEASURED
        # before this clamp: `G.assign(G - 2.0)` made every `t` NaN.
        #
        # The clamp is at ZERO, not at `epsilon`, so it does not silently
        # rescale a healthy energy the way flooring the product would.
        energy = ops.maximum(energy, 0.0)

        # Eq. 6. prior_mean and the sqrt(2) are Python floats, so this broadcast is
        # static and no variable is materialized for either of them. The
        # constant is cast so it cannot silently promote under a float64 or
        # mixed_float16 policy (an fp16 1.414 is exact enough here; the cast
        # only guarantees the operand dtype matches z's).
        numerator = ops.cast(
            np.sqrt(2.0), self.compute_dtype
        ) * (z - ops.cast(self.prior_mean, self.compute_dtype))
        denominator = ops.sqrt(
            self.degrees_of_freedom * energy + self.epsilon
        )
        return numerator / denominator

    def compute_output_shape(self, input_shape):
        """Output shape — the topography is elementwise in the latent width.

        :param input_shape: A pair of ``(batch, sequence, latent_dim)`` shapes.
        :type input_shape: Any
        :return: The same shape as ``z``.
        :rtype: Tuple[Optional[int], ...]
        """
        shapes = self._resolve_input_shapes(input_shape)
        return shapes[0]

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "capsule_dim": self.capsule_dim,
            "coherence_window": self.coherence_window,
            "neighborhood_size": self.neighborhood_size,
            "temporal_coherence": self.temporal_coherence,
            "topography": self.topography,
            "grid_shape": self.grid_shape,
            "kernel_shape": self.kernel_shape,
            "degrees_of_freedom": self.degrees_of_freedom,
            "prior_mean": self.prior_mean,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------