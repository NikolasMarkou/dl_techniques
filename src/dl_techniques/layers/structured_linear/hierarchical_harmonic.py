"""HierarchicalHarmonicHead: tree-structured distance-based classifier head.

Scales the harmonic-loss idea (Baek, Liu, Tyagi, Tegmark, "Harmonic Loss
Trains Interpretable AI Models", TMLR 2025) to large output spaces by
factorizing the flat ``HarMax`` distribution over a fixed balanced tree
with a **learned** class-to-leaf assignment.

A flat :class:`dl_techniques.layers.structured_linear.harmonic_dense.HarmonicDense`
head scores every class against its prototype (``O(N)`` distances) and
normalizes once. This head instead owns one prototype set per tree level
plus a member prototype per leaf slot, and routes top-down: each internal
node holds one prototype per child, and the conditional distribution over
its children is ``HarMax`` over the child distances. The leaf probability
is the product of conditionals along its path::

    p(leaf) = prod_l softmax_l(-n_l * log d_l)[child_l | parent_l]

Every level self-normalizes, so the leaves sum to 1 by construction. With
``branching=(100, 100)`` and ``N = 10000`` classes, a forward pass scores
``100 + 10000`` prototypes in two batched matmuls -- the same FLOPs as flat
but with a hierarchical loss structure and a learned taxonomy. Depth is
arbitrary (``len(branching)``); the default is 2 with an automatic balanced
split.

The tree itself is fixed and balanced; what is learned is the assignment of
classes to leaf slots, held in the non-trainable ``slot_of_class`` table
and updated by the explicit :meth:`reassign` E-step (nearest routing-center
assignment with exact per-node capacities, so dead clusters are
structurally impossible). Member prototypes are slot-indexed, so
:meth:`reassign` permutes the member kernel rows to follow their classes.

``output_mode`` mirrors ``HarmonicDense``:

- ``"probs"`` -- full ``(..., N)`` distribution (exact). Train with
  ``SparseCategoricalCrossentropy(from_logits=False)``.
- ``"logits"`` -- full ``(..., N)`` *normalized* log-probabilities, for
  NLL-style losses. These are already normalized; do NOT pass them to a
  ``from_logits=True`` loss (that would normalize twice).
- ``"distances"`` -- finest-level distances ``(..., N)`` in class order.

For retrieval-scale inference the exact path can be replaced by
:meth:`beam_call`, which evaluates only the top-k level-1 subtrees and
returns retrieval-style ``(probs, class_ids)`` pairs renormalized over the
evaluated set.

Measured scope -- what this buys and what it does not (all figures GPU /
float32 / keras 3.x, pinned by tests where stated):

- Exactness: leaf path-sums reproduce the flat heads (depth-1 equals
  ``HarmonicDense``, single-cluster equals ``HarMax``), level-1 marginals
  equal summed leaves to < 1e-7, and full-width beam equals exact (TV = 0).
  Narrow beam degrades gracefully (top-2 of 4 clusters: recall@1 = 1.00
  at mean TV 0.22 on random weights).
- Taxonomy: on 4-supercluster x 5-class synthetic data, train + ``reassign``
  recovers the true superclasses at purity 1.00, and per-level exponent
  annealing (1 -> 2 coarse, 2 -> 8 fine over 12 epochs) beats fixed
  exponents (acc 0.703 vs 0.590, chance 0.05).
- Cost (NOT a win at N = 1e4): exact forward runs ~21ms vs ~12-15ms flat
  per (256, 64) batch -- two matmuls plus gathers lose to one fused
  matmul -- and holds ~4% more parameters (internal prototypes). Beam
  scores 33x fewer distances at top-2 but its top-k + dynamic-gather
  overhead makes it slower wall-clock at this scale too. The compute win
  appears only where the flat ``(B, D, N)`` matmul dominates (N >> 1e4,
  or latency-bound serving). Accuracy is at parity with flat (0.986 vs
  0.988 on the same task), not above it.

Buy this head for the learned taxonomy, the exact partial-credit
structure (coarse accuracy, cluster aux-losses), and the retrieval path
to 1e5+ vocabularies -- not for speed or accuracy at 1e4.

References:
    - Baek et al., 2025. "Harmonic Loss Trains Interpretable AI Models".
    - Grave et al., 2017. "Efficient softmax approximation for GPUs"
      (adaptive softmax; frequency-based fixed clustering, the ancestor of
      the learned assignment used here).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _balanced_branching(num_classes: int) -> Tuple[int, int]:
    """Return the default depth-2 balanced split covering ``num_classes``.

    :param num_classes: Number of classes.
    :type num_classes: int
    :return: ``(K, M)`` with ``K * M >= num_classes``.
    :rtype: Tuple[int, int]
    """
    k = int(math.ceil(math.sqrt(num_classes)))
    m = int(math.ceil(num_classes / k))
    return (k, m)


def _greedy_capacity_assignment(
        costs: np.ndarray,
        capacities: np.ndarray
) -> np.ndarray:
    """Assign each row to a column, minimizing total cost under capacities.

    Greedy by regret (second-best minus best cost, descending), so the
    most constrained rows choose first. Ties break by row index (stable
    sort), making the result fully deterministic.

    :param costs: ``(n_items, n_nodes)`` assignment costs (float).
    :type costs: np.ndarray
    :param capacities: ``(n_nodes,)`` integer capacities summing to
        at least ``n_items``.
    :type capacities: np.ndarray
    :return: ``(n_items,)`` integer node index per row.
    :rtype: np.ndarray
    """
    costs = np.asarray(costs, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    n_items, n_nodes = costs.shape
    order = np.argsort(costs, axis=1, kind="stable")
    best = costs[np.arange(n_items)[:, None], order[:, :2]]
    regret = best[:, 1] - best[:, 0] if n_nodes > 1 else np.zeros(n_items)
    # Stable descending: most regret first, ties by row index.
    sequence = np.argsort(-regret, kind="stable")
    remaining = capacities.copy()
    assignment = np.full(n_items, -1, dtype=np.int64)
    for i in sequence:
        for node in order[i]:
            if remaining[node] > 0:
                assignment[i] = node
                remaining[node] -= 1
                break
        if assignment[i] < 0:  # pragma: no cover - capacities sum >= n_items
            raise ValueError("Capacities exhausted during assignment.")
    return assignment


@register_dl_technique("dl_techniques.layers.structured_linear.hierarchical_harmonic")
class HierarchicalHarmonicHead(keras.layers.Layer):
    """Tree-factored harmonic classifier head for large output spaces.

    Owns per-level node prototypes plus one member prototype per leaf slot,
    and a non-trainable ``slot_of_class`` assignment table (persisted in
    ``.keras``). The forward pass is exact: every level is scored and
    sibling-group-normalized, leaf log-probabilities are path sums, and the
    output is gathered into class order.

    **Architecture Overview (depth 2):**

    .. code-block:: text

                        x  [..., D]
                                │
                 ┌────────────────┼───────────────┐
                 │                │               │
                 ▼                ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ coarse dists  │ │ member dists  │ │ slot_of_class │
        │ (..., K1)     │ │ (..., L)      │ │ (N,) lookup   │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                ▼                 ▼                 │
        ┌───────────────┐ ┌───────────────┐        │
        │ log-softmax   │ │ log-softmax   │        │
        │ over K1       │ │ within each   │        │
        │               │ │ sibling group │        │
        └───────┬───────┘ └───────┬───────┘        │
                └────────┬────────┘                │
                         ▼                         ▼
              leaf logprob (..., L) ──gather──► y (..., N)

    Padding (when ``prod(branching) > num_classes``) is a suffix of the
    slot range, so empty nodes are known statically at build: their logits
    read ``-inf`` inside their sibling group, padded leaves land on exactly
    0, and real leaves sum to exactly 1.

    :param num_classes: Positive integer number of classes.
    :type num_classes: int
    :param branching: Per-level branching factors, e.g. ``(100, 100)``.
        Depth is ``len(branching)``. ``None`` (default) means the balanced
        depth-2 split covering ``num_classes``. The product must cover
        ``num_classes``.
    :type branching: Optional[Tuple[int, ...]]
    :param n: Harmonic exponent: a single positive float broadcast to all
        levels, a per-level tuple, or ``None`` (default) for the
        ``sqrt(input_dim)`` heuristic at every level. Stored as given;
        resolved per level in :meth:`build`.
    :type n: Optional[Union[float, Tuple[float, ...]]]
    :param output_mode: One of ``"probs"``, ``"logits"``, ``"distances"``.
        Defaults to ``"probs"``. See the module docstring: ``"logits"``
        here are *normalized* log-probabilities.
    :type output_mode: str
    :param epsilon: Floor for squared distances. Must be positive.
        Defaults to 1e-8.
    :type epsilon: float
    :param kernel_initializer: Initializer for the member (leaf-slot)
        prototypes.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the member prototypes.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param node_initializer: Initializer for the internal-node prototypes.
    :type node_initializer: Union[str, keras.initializers.Initializer]
    :param node_regularizer: Regularizer for the internal-node prototypes.
    :type node_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``num_classes`` is not a positive integer, if
        ``branching`` has a non-positive entry or its product does not
        cover ``num_classes``, if ``output_mode`` is unknown, if ``n`` is
        not ``None``, positive, or a matching positive per-level tuple, or
        if ``epsilon`` is not positive.

    :ivar slot_of_class: ``(N,)`` assignment table (float32 storage, int
        values; see :meth:`build`), ``None`` until :meth:`build` runs.
    :vartype slot_of_class: Optional[keras.Variable]
    """

    #: Every accepted ``output_mode`` spelling, checked in ``__init__``.
    _SUPPORTED_MODES: Tuple[str, ...] = ("probs", "logits", "distances")

    def __init__(
            self,
            num_classes: int,
            branching: Optional[Tuple[int, ...]] = None,
            n: Optional[Union[float, Tuple[float, ...]]] = None,
            output_mode: str = "probs",
            epsilon: float = 1e-8,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            node_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            node_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and store it.

        No weight is created here; :meth:`build` creates the prototype
        kernels once the input dimension is known. The static tree tables
        (paths, masks, capacities) derive from ``num_classes`` and
        ``branching`` alone and are computed here.

        :param num_classes: Positive integer number of classes.
        :type num_classes: int
        :param branching: Per-level branching factors or ``None`` for the
            balanced depth-2 default.
        :type branching: Optional[Tuple[int, ...]]
        :param n: Harmonic exponent, per-level tuple, or ``None``.
        :type n: Optional[Union[float, Tuple[float, ...]]]
        :param output_mode: One of ``"probs"``, ``"logits"``,
            ``"distances"``. Defaults to ``"probs"``.
        :type output_mode: str
        :param epsilon: Floor for squared distances. Must be positive.
        :type epsilon: float
        :param kernel_initializer: Initializer for member prototypes.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for member prototypes.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param node_initializer: Initializer for internal-node prototypes.
        :type node_initializer: Union[str, keras.initializers.Initializer]
        :param node_regularizer: Regularizer for internal-node prototypes.
        :type node_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If any argument violates its contract above.
        """
        super().__init__(**kwargs)
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) \
                or num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes}"
            )
        if branching is None:
            branching = _balanced_branching(num_classes)
        branching = tuple(int(b) for b in branching)
        if len(branching) == 0 or any(b <= 0 for b in branching):
            raise ValueError(
                f"branching must be a non-empty tuple of positive integers, "
                f"got {branching}"
            )
        total_leaves = int(np.prod(branching))
        if total_leaves < num_classes:
            raise ValueError(
                f"branching product {total_leaves} does not cover "
                f"num_classes={num_classes}"
            )
        if output_mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Invalid output_mode '{output_mode}'. "
                f"Supported modes: {list(self._SUPPORTED_MODES)}"
            )
        if n is not None:
            if isinstance(n, (tuple, list)):
                if len(n) != len(branching):
                    raise ValueError(
                        f"Per-level n has length {len(n)} but branching has "
                        f"depth {len(branching)}"
                    )
                n = tuple(float(v) for v in n)
                if any(v <= 0 for v in n):
                    raise ValueError(f"Every per-level n must be positive, got {n}")
            else:
                n = float(n)
                if n <= 0:
                    raise ValueError(f"n must be positive or None, got {n}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.num_classes = num_classes
        self.branching = branching
        self.depth = len(branching)
        self.n = n
        self.output_mode = output_mode
        self.epsilon = float(epsilon)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.node_initializer = keras.initializers.get(node_initializer)
        self.node_regularizer = keras.regularizers.get(node_regularizer)

        # Static tree tables: padding is a slot suffix, so emptiness is a
        # function of (num_classes, branching) only -- assignment never
        # affects which nodes are empty.
        self.total_leaves = total_leaves
        # Slots owned per level-l node (0-based l): prod(branching[l + 1:]).
        # The last level owns 1 slot per node, i.e. leaf slots themselves.
        self._slots_per_node: List[int] = []
        running = 1
        for b in reversed(branching):
            self._slots_per_node.append(running)
            running *= b
        self._slots_per_node.reverse()
        # Nodes per level: prod(branching[:l]).
        self._nodes_per_level: List[int] = []
        running = 1
        for b in branching:
            running *= b
            self._nodes_per_level.append(running)
        # Path of each slot: node index at every level.
        paths = np.zeros((total_leaves, self.depth), dtype=np.int64)
        for s in range(total_leaves):
            for l in range(self.depth):
                paths[s, l] = s // self._slots_per_node[l]
        self._paths = paths
        # Empty-node mask per level (True == owns zero real slots).
        self._empty_masks: List[np.ndarray] = []
        for l in range(self.depth):
            first_slot = np.arange(self._nodes_per_level[l]) * self._slots_per_node[l]
            self._empty_masks.append(first_slot >= num_classes)
        # Real-slot capacity of each level-1 node (for reassign).
        self._level1_capacities = np.array([
            max(0, min(self._slots_per_node[0], num_classes - j * self._slots_per_node[0]))
            for j in range(self._nodes_per_level[0])
        ], dtype=np.int64)

        self.member_kernel = None
        self.node_kernels: List[Any] = []
        self.slot_of_class = None
        self.class_of_slot = None
        self.effective_n: List[float] = []

    @property
    def num_levels(self) -> int:
        """Return the tree depth (number of harmonic exponents).

        :return: ``len(branching)``.
        :rtype: int
        """
        return self.depth

    def set_n_per_level(self, values: Union[float, List[float], Tuple[float, ...]]) -> None:
        """Replace the resolved per-level exponents (annealing hook).

        Takes effect on the next call (floats bake into the graph as
        constants, so changing them retraces -- the same cost class as any
        per-epoch schedule). Values must be positive.

        :param values: A single positive float broadcast to all levels, or
            one positive float per level.
        :type values: Union[float, List[float], Tuple[float, ...]]
        :raises ValueError: If a value is not positive, or a sequence does
            not match the depth.
        """
        if isinstance(values, (list, tuple)):
            if len(values) != self.depth:
                raise ValueError(
                    f"Expected {self.depth} per-level values, got {len(values)}"
                )
            resolved = [float(v) for v in values]
        else:
            resolved = [float(values)] * self.depth
        if any(v <= 0 for v in resolved):
            raise ValueError(f"Every per-level n must be positive, got {resolved}")
        self.effective_n = resolved

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create prototype kernels and the assignment tables.

        Member kernel ``(dim, L)`` is slot-indexed; internal level ``l``
        owns ``(dim, P_l)``. ``effective_n`` resolves ``None`` to
        ``sqrt(dim)`` per level. The assignment tables start as identity
        (class ``c`` in slot ``c``); :meth:`reassign` permutes them.

        :param input_shape: Shape of the input tensor. The last dimension
            must be defined.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is
            ``None``.
        """
        if self.built:
            return
        dim = input_shape[-1]
        if dim is None:
            raise ValueError(
                "The last dimension of the input shape must be defined. "
                f"Received input_shape={input_shape}"
            )
        heuristic = float(dim) ** 0.5
        if self.n is None:
            self.effective_n = [heuristic] * self.depth
        elif isinstance(self.n, tuple):
            self.effective_n = [float(v) for v in self.n]
        else:
            self.effective_n = [float(self.n)] * self.depth

        self.node_kernels = []
        for l in range(self.depth - 1):
            self.node_kernels.append(self.add_weight(
                name=f"nodes_l{l + 1}",
                shape=(dim, self._nodes_per_level[l]),
                initializer=self.node_initializer,
                regularizer=self.node_regularizer,
                trainable=True,
            ))
        self.member_kernel = self.add_weight(
            name="members",
            shape=(dim, self.total_leaves),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.slot_of_class = self.add_weight(
            name="slot_of_class",
            shape=(self.num_classes,),
            initializer="zeros",
            trainable=False,
        )
        # NOTE: stored as float32, not int32. TensorFlow pins integer
        # variables to CPU, and a GPU (XLA) graph then refuses to read
        # them ("Trying to access resource ... located in CPU:0 from
        # GPU:0"). Float variables mirror to the device; every use site
        # casts back to int32. Integers below 2**24 round-trip exactly,
        # far above any sane class count -- asserted in the scale test.
        self.slot_of_class.assign(
            np.arange(self.num_classes, dtype=np.float32)
        )
        self.class_of_slot = self.add_weight(
            name="class_of_slot",
            shape=(self.total_leaves,),
            initializer="zeros",
            trainable=False,
        )
        full = np.full(self.total_leaves, -1, dtype=np.float32)
        full[:self.num_classes] = np.arange(self.num_classes, dtype=np.float32)
        self.class_of_slot.assign(full)
        super().build(input_shape)

    def _level_logprob(
            self,
            inputs: keras.KerasTensor,
            kernel: keras.KerasTensor,
            exponent: float,
            level: int
    ) -> keras.KerasTensor:
        """Return sibling-group log-softmax of ``-n log d`` at one level.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :param kernel: Level prototypes ``(dim, P_l)``.
        :type kernel: keras.KerasTensor
        :param exponent: Harmonic exponent for this level.
        :type exponent: float
        :param level: 1-based level index (selects the sibling grouping).
        :type level: int
        :return: Conditional log-probabilities shaped ``(..., P_l)``.
        :rtype: keras.KerasTensor
        """
        x2 = keras.ops.sum(keras.ops.square(inputs), axis=-1, keepdims=True)
        w2 = keras.ops.sum(keras.ops.square(kernel), axis=0, keepdims=True)
        d2 = keras.ops.maximum(
            x2 - 2.0 * keras.ops.matmul(inputs, kernel) + w2, self.epsilon
        )
        logits = -0.5 * exponent * keras.ops.log(d2)
        neg_inf = keras.ops.cast(
            keras.ops.convert_to_tensor(float("-inf")), logits.dtype
        )
        if level == 1:
            grouped = keras.ops.reshape(logits, (-1, self._nodes_per_level[0]))
            if np.any(self._empty_masks[0]):
                mask = keras.ops.cast(
                    keras.ops.convert_to_tensor(
                        np.where(self._empty_masks[0], -np.inf, 0.0)
                    ),
                    grouped.dtype,
                )
                grouped = grouped + mask
            out = keras.ops.log_softmax(grouped, axis=-1)
            out = keras.ops.where(keras.ops.isfinite(out), out, neg_inf)
            return keras.ops.reshape(out, keras.ops.shape(logits))
        # Deeper levels: group children under each parent.
        parents = self._nodes_per_level[level - 2]
        branch = self.branching[level - 1]
        grouped = keras.ops.reshape(logits, (-1, parents, branch))
        empty = self._empty_masks[level - 1].reshape(parents, branch)
        if np.any(empty):
            mask = keras.ops.cast(
                keras.ops.convert_to_tensor(np.where(empty, -np.inf, 0.0)),
                grouped.dtype,
            )
            grouped = grouped + mask
            # A fully-empty group would read all -inf, and log_softmax of
            # that is NaN -- whose gradient (NaN * 0 through the sanitize
            # below) poisons the kernels. Those entries never reach a real
            # leaf, so evaluate them on finite zeros and overwrite with
            # -inf afterwards: identical forward, finite backward.
            group_empty = keras.ops.convert_to_tensor(empty.all(axis=-1))
            grouped = keras.ops.where(
                keras.ops.reshape(group_empty, (1, parents, 1)),
                keras.ops.zeros_like(grouped),
                grouped,
            )
        out = keras.ops.log_softmax(grouped, axis=-1)
        if np.any(empty):
            out = keras.ops.where(
                keras.ops.reshape(group_empty, (1, parents, 1)), neg_inf, out
            )
        out = keras.ops.where(keras.ops.isfinite(out), out, neg_inf)
        return keras.ops.reshape(out, keras.ops.shape(logits))

    def _leaf_logprobs(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Return unnormalized-path leaf log-probabilities ``(..., L)``.

        Path sums of per-level conditionals; padded leaves land on ``-inf``.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :return: Leaf log-probabilities shaped ``(..., total_leaves)``.
        :rtype: keras.KerasTensor
        """
        kernels = list(self.node_kernels) + [self.member_kernel]
        total = None
        for l in range(self.depth):
            cond = self._level_logprob(
                inputs, kernels[l], self.effective_n[l], l + 1
            )
            path_idx = keras.ops.convert_to_tensor(
                self._paths[:, l].astype("int32")
            )
            gathered = keras.ops.take(cond, path_idx, axis=-1)
            total = gathered if total is None else total + gathered
        return total

    def cluster_probs(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Return the level-1 (coarse cluster) distribution.

        Useful as an auxiliary target (cluster-level loss) or for
        inspecting the learned taxonomy.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :return: Cluster probabilities shaped ``(..., K1)``.
        :rtype: keras.KerasTensor
        """
        cond = self._level_logprob(
            inputs, self.node_kernels[0] if self.depth > 1 else self.member_kernel,
            self.effective_n[0], 1,
        )
        return keras.ops.exp(cond)

    def _slot_indices(self) -> keras.KerasTensor:
        """Return ``slot_of_class`` cast to int32 for ``take`` indexing.

        The table is stored float (see :meth:`build`) so GPU graphs can
        read it; the cast is exact below 2**24 classes.

        :return: ``(N,)`` int32 slot index per class.
        :rtype: keras.KerasTensor
        """
        return keras.ops.cast(self.slot_of_class, "int32")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Score ``inputs`` against the tree and emit class-ordered output.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor shaped ``(..., num_classes)``: probabilities,
            normalized log-probabilities, or finest-level distances
            according to ``output_mode``.
        :rtype: keras.KerasTensor
        """
        if self.output_mode == "distances":
            kernels = self.member_kernel
            x2 = keras.ops.sum(keras.ops.square(inputs), axis=-1, keepdims=True)
            w2 = keras.ops.sum(keras.ops.square(kernels), axis=0, keepdims=True)
            d2 = keras.ops.maximum(
                x2 - 2.0 * keras.ops.matmul(inputs, kernels) + w2, self.epsilon
            )
            return keras.ops.take(
                keras.ops.sqrt(d2), self._slot_indices(), axis=-1
            )
        leaf = self._leaf_logprobs(inputs)
        gathered = keras.ops.take(leaf, self._slot_indices(), axis=-1)
        if self.output_mode == "logits":
            return gathered
        return keras.ops.exp(gathered)

    def beam_call(
            self,
            inputs: keras.KerasTensor,
            top_k: int = 2
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Approximate retrieval: evaluate only the top-k level-1 subtrees.

        Computes level-1 cluster probabilities, keeps the ``top_k``
        clusters per sample, scores member distances only inside those
        subtrees with per-cluster renormalization, and multiplies by the
        selecting cluster's probability. Returns retrieval-style pairs
        rather than a dense ``N``-vector (a dense scatter would cost
        ``O(B * k * S1 * N)`` memory).

        At ``top_k == K1`` and depth 2 this reproduces the exact
        :meth:`call` distribution over the evaluated slots; at deeper
        depths the intermediate levels are folded into the fine exponent
        and it is approximate even at full beam. Use the exact
        :meth:`call` for training.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :param top_k: Number of level-1 clusters to expand per sample.
            Clipped to ``K1``.
        :type top_k: int
        :return: ``(probs, class_ids)`` with shapes ``(..., k*S1)`` and
            ``(..., k*S1)`` (int32, ``-1`` for padded slots). Rows of
            ``probs`` sum to 1 over the valid (non-negative-id) entries.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        k1 = self._nodes_per_level[0]
        top_k = max(1, min(int(top_k), k1))
        s1 = self._slots_per_node[0]
        neg_inf = keras.ops.cast(
            keras.ops.convert_to_tensor(float("-inf")), inputs.dtype
        )
        cluster_p = self.cluster_probs(inputs)  # (..., K1)
        top_v, top_i = keras.ops.top_k(cluster_p, k=top_k)  # (..., k)
        # Slots of the selected clusters: (..., k, S1) -> (B, k*S1).
        slots = (
            keras.ops.expand_dims(top_i, -1) * s1
            + keras.ops.arange(s1, dtype="int32")
        )
        lead = tuple(inputs.shape[:-1])
        flat_slots = keras.ops.reshape(slots, (-1, top_k * s1))
        members_t = keras.ops.transpose(self.member_kernel)  # (L, dim)
        cand = keras.ops.take(members_t, keras.ops.reshape(flat_slots, (-1,)), axis=0)
        cand = keras.ops.reshape(cand, (-1, top_k, s1, keras.ops.shape(inputs)[-1]))
        flat_x = keras.ops.reshape(inputs, (-1, keras.ops.shape(inputs)[-1]))
        x2 = keras.ops.sum(keras.ops.square(flat_x), axis=-1)
        x2 = keras.ops.reshape(x2, (-1, 1, 1))
        w2 = keras.ops.sum(keras.ops.square(cand), axis=-1)
        xw = keras.ops.sum(
            keras.ops.reshape(flat_x, (-1, 1, 1, keras.ops.shape(inputs)[-1])) * cand,
            axis=-1,
        )
        d2 = keras.ops.maximum(x2 - 2.0 * xw + w2, self.epsilon)
        n_fine = self.effective_n[-1]
        cand_logits = -0.5 * n_fine * keras.ops.log(d2)
        # Mask padded slots (slot index >= N) inside each selected cluster.
        valid = keras.ops.reshape(flat_slots, (-1, top_k, s1)) < self.num_classes
        cand_logits = keras.ops.where(valid, cand_logits, neg_inf)
        # A fully-padded selected cluster would read all -inf (NaN after
        # log_softmax, poisoning gradients); evaluate it on zeros and mask
        # back to -inf: identical forward, finite backward.
        grp_empty = keras.ops.logical_not(keras.ops.any(valid, axis=-1))
        cand_logits = keras.ops.where(
            keras.ops.expand_dims(grp_empty, -1),
            keras.ops.zeros_like(cand_logits),
            cand_logits,
        )
        # Per-cluster member conditionals, weighted by the cluster prob.
        cond = keras.ops.log_softmax(cand_logits, axis=-1)
        cond = keras.ops.where(
            keras.ops.expand_dims(grp_empty, -1), neg_inf, cond
        )
        cond = keras.ops.where(keras.ops.isfinite(cond), cond, neg_inf)
        joint = cond + keras.ops.log(
            keras.ops.maximum(keras.ops.expand_dims(top_v, -1), self.epsilon)
        )
        joint = keras.ops.where(valid, joint, neg_inf)
        # Renormalize over the evaluated set (no-op at full beam, depth 2).
        norm = keras.ops.logsumexp(keras.ops.reshape(joint, (-1, top_k * s1)), axis=-1)
        joint = joint - keras.ops.reshape(norm, (-1, 1, 1))
        # Numerical hygiene: a fully-padded cluster contributes -inf, never NaN.
        joint = keras.ops.where(keras.ops.isfinite(joint), joint, neg_inf)
        joint = keras.ops.where(valid, joint, neg_inf)
        probs = keras.ops.exp(keras.ops.reshape(joint, (-1, top_k * s1)))
        gathered = keras.ops.reshape(flat_slots, (-1, top_k * s1))
        class_ids = keras.ops.where(
            keras.ops.reshape(valid, (-1, top_k * s1)),
            keras.ops.reshape(
                keras.ops.take(
                    keras.ops.cast(self.class_of_slot, "int32"),
                    keras.ops.reshape(gathered, (-1,)),
                    axis=0,
                ),
                (-1, top_k * s1),
            ),
            keras.ops.convert_to_tensor(-1, dtype="int32"),
        )
        out_shape = lead + (top_k * s1,)
        return keras.ops.reshape(probs, out_shape), keras.ops.reshape(
            keras.ops.cast(class_ids, "int32"), out_shape
        )

    def reassign(
            self,
            representatives: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Re-estimate the class-to-slot assignment (E-step).

        Top-down greedy capacity-constrained assignment: at each level,
        classes are ordered by regret (second-best minus best routing
        cost, descending, ties by class index) and placed at their nearest
        node with remaining capacity. Capacities are the static real-slot
        counts, so every class lands exactly once and no cluster is left
        empty unless it owns zero real slots (padding). Member kernel rows
        permute to follow their classes; internal node prototypes are
        untouched.

        Runs eagerly on numpy; call it between epochs (or from a
        callback), never inside the graph.

        :param representatives: ``(N, D)`` per-class vectors the routing
            cost is measured against. ``None`` (default) reuses the
            current member prototypes in class order, making the step
            self-contained.
        :type representatives: Optional[np.ndarray]
        :return: ``{"moved": <classes whose slot changed>}``.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``representatives`` has the wrong shape.
        """
        reps = self._representatives_or_raise(representatives)
        new_slot = np.full(self.num_classes, -1, dtype=np.int64)
        # Per-level node centers as (P_l, D).
        centers = [
            np.asarray(keras.ops.convert_to_numpy(k)).T
            for k in list(self.node_kernels) + [self.member_kernel]
        ]
        if self.depth == 1:
            # Flat tree: the only assignment is identity.
            new_slot = np.arange(self.num_classes, dtype=np.int64)
        else:
            # The virtual root (level -1) owns every class; distribute
            # them over the level-0 nodes, then recurse top-down.
            top_children = np.arange(self.branching[0])
            top_costs = (
                np.sum(reps ** 2, axis=1, keepdims=True)
                - 2.0 * reps @ centers[0][top_children].T
            )
            top_caps = np.array([
                self._subtree_classes(int(ch), 0) for ch in top_children
            ])
            top_assignment = _greedy_capacity_assignment(top_costs, top_caps)
            all_classes = np.arange(self.num_classes)
            for pos, ch in enumerate(top_children):
                group = all_classes[top_assignment == pos]
                if group.size:
                    self._assign_level(
                        group, reps, 0, int(ch), centers, new_slot
                    )
        old_slot = np.asarray(
            keras.ops.convert_to_numpy(self.slot_of_class), dtype=np.int64
        )
        moved = int(np.count_nonzero(new_slot != old_slot))
        # Permute member rows to follow their classes; pads keep old values.
        kernel = np.asarray(keras.ops.convert_to_numpy(self.member_kernel))
        # new_slot is a permutation of 0..N-1:
        # kernel[:, new_slot[c]] = old kernel[:, old_slot[c]].
        new_kernel = kernel.copy()
        new_kernel[:, new_slot] = kernel[:, old_slot]
        self.member_kernel.assign(new_kernel)
        self.slot_of_class.assign(new_slot.astype(np.float32))
        classes = np.full(self.total_leaves, -1, dtype=np.float32)
        classes[new_slot] = np.arange(self.num_classes, dtype=np.float32)
        self.class_of_slot.assign(classes)
        return {"moved": moved}

    def _representatives_or_raise(
            self,
            representatives: Optional[np.ndarray]
    ) -> np.ndarray:
        """Return ``(N, D)`` float64 representatives or raise.

        :param representatives: User value or ``None``.
        :type representatives: Optional[np.ndarray]
        :return: ``(N, D)`` float64 array.
        :rtype: np.ndarray
        :raises ValueError: On shape mismatch.
        """
        if representatives is None:
            kernel = np.asarray(keras.ops.convert_to_numpy(self.member_kernel))
            slots = np.asarray(
                keras.ops.convert_to_numpy(self.slot_of_class), dtype=np.int64
            )
            return np.ascontiguousarray(kernel[:, slots].T, dtype=np.float64)
        reps = np.asarray(representatives, dtype=np.float64)
        dim = self.member_kernel.shape[0]
        if reps.shape != (self.num_classes, dim):
            raise ValueError(
                f"representatives must have shape ({self.num_classes}, {dim}), "
                f"got {reps.shape}"
            )
        return reps

    def _assign_level(
            self,
            class_ids: np.ndarray,
            reps: np.ndarray,
            level: int,
            node: int,
            centers: List[np.ndarray],
            new_slot: np.ndarray,
    ) -> None:
        """Distribute ``class_ids`` among the children of ``node``.

        ``node`` is a level-``level`` node (0-based); its children live at
        level ``level + 1`` and are indexed ``node * branching[level + 1] +
        i``. Level ``depth - 1`` nodes each own exactly one slot, so when
        the children are at the last level they *are* leaf slots and
        classes land directly. Sibling capacities are the static real-slot
        counts, hence exact.

        :param class_ids: Class indices owned by this subtree.
        :type class_ids: np.ndarray
        :param reps: ``(N, D)`` representatives.
        :type reps: np.ndarray
        :param level: 0-based level index of ``node``.
        :type level: int
        :param node: Flat node index at ``level``.
        :type node: int
        :param centers: Per-level ``(P_l, D)`` routing centers.
        :type centers: List[np.ndarray]
        :param new_slot: Output ``(N,)`` slot table, written in place.
        :type new_slot: np.ndarray
        """
        branch = self.branching[level + 1]
        children = node * branch + np.arange(branch)
        sub = reps[class_ids]
        costs = (
            np.sum(sub ** 2, axis=1, keepdims=True)
            - 2.0 * sub @ centers[level + 1][children].T
        )
        if level + 1 == self.depth - 1:
            # Children are leaf slots: capacity 1 for real slots, else 0.
            capacities = np.array([
                0 if int(ch) >= self.num_classes else 1 for ch in children
            ])
            order = _greedy_capacity_assignment(costs, capacities)
            for cls, pos in zip(class_ids, order):
                new_slot[cls] = int(children[pos])
            return
        capacities = np.array([
            self._subtree_classes(int(ch), level + 1) for ch in children
        ])
        assignment = _greedy_capacity_assignment(costs, capacities)
        for pos, ch in enumerate(children):
            group = class_ids[assignment == pos]
            if group.size:
                self._assign_level(group, reps, level + 1, int(ch), centers, new_slot)

    def _subtree_classes(self, node: int, level: int) -> int:
        """Return the real-slot count owned by ``node`` at 0-based ``level``.

        :param node: Flat node index at ``level``.
        :type node: int
        :param level: 0-based level index.
        :type level: int
        :return: Number of non-padding slots under the node.
        :rtype: int
        """
        first = node * self._slots_per_node[level]
        return max(0, min(self._slots_per_node[level], self.num_classes - first))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``tuple(input_shape[:-1]) + (num_classes,)``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.num_classes,)

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``branching`` is written as a list (JSON-safe); ``n`` is written
        back as passed to the constructor. The assignment tables ride
        along as non-trainable weights, not config.

        :return: The base Layer config plus the constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "branching": list(self.branching),
            "n": list(self.n) if isinstance(self.n, tuple) else self.n,
            "output_mode": self.output_mode,
            "epsilon": self.epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "node_initializer": keras.initializers.serialize(self.node_initializer),
            "node_regularizer": keras.regularizers.serialize(self.node_regularizer),
        })
        return config

# ---------------------------------------------------------------------
