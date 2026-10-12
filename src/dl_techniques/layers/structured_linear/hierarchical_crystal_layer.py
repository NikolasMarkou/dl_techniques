"""HierarchicalCrystalLayer, a two-tier crystal classifier over a class tree.

:class:`~dl_techniques.layers.structured_linear.crystal_layer.CrystalLayer`
flattens classes onto one sphere. When the label space is itself hierarchical --
coarse groups of fine classes -- that flattening throws the group structure
away. This layer keeps it, in two tiers of the same geometric construction:

- **Macro tier.** ``num_macro_classes`` centroids spread over the sphere by the
  same packed-simplex procedure, so the coarse groups are as far apart as the
  dimension allows.
- **Micro tier.** For each macro centroid ``C_m``, ``num_micro_per_macro`` leaf
  vertices sit in an isotropic bundle around it::

      V_{m,k} = normalize(C_m + cluster_spread * S_{m,k})

  where ``S_{m,k}`` is itself a packed frame. The displacement is drawn over
  the whole sphere, not tangential to ``C_m``, so the bundle is isotropic
  rather than a cone. With ``|C| = |S| = 1`` and ``c = <C, S>`` in ``[-1, 1]``,
  the leaf's angle to its own centroid is exactly

      angle(C, V) = arccos((1 + s*c) / sqrt(1 + 2*s*c + s*s))

  so ``cluster_spread`` is a perturbation magnitude, not an angular aperture:
  past ``s ~= 0.3`` the leaves leave any cap around their centroid and can pass
  its antipode. The exact maximum over all ``c`` is ``arcsin(s)``, attained at
  ``<C, S> = -s``; the realized maxima below sit just under it. Measured
  leaf-to-own-centroid aperture ranges (``input_dim=32``, 4 groups of 6,
  ``seed=0``):

  ============  =======================  ==============
  ``s``         measured aperture        ``arcsin(s)``
  ============  =======================  ==============
  0.05          [2.55, 2.87] degrees     2.87
  0.25          [12.58, 14.48] degrees   14.48
  1.0           [36.60, 59.83] degrees   90.00
  3.0           [56.98, 100.53] degrees  unbounded
  10.0          [67.89, 114.44] degrees  unbounded
  ============  =======================  ==============

  As ``s`` shrinks toward 0 the leaves collapse onto their centroid exactly.

**``cluster_spread`` near 1 can zero a leaf.** The bundle norm
``sqrt(1 + 2*s*c + s*s)`` vanishes at ``s = 1, c = -1`` -- a leaf displaced
exactly opposite its centroid -- and near that point it is small enough that
the leaf direction is numerical garbage rather than a direction. The
initializer floors the norm at ``1e-12``, so the degenerate leaf is an
all-zero column, and the cosine logits downstream normalize it to a
permanently-zero score. This is a measure-zero configuration, not a working
aperture; it stays legal for the same reason ``spread=0`` does, and is pinned
by the test suite so it cannot be mistaken for one.

**``cluster_spread=0`` is degenerate, and is allowed deliberately.** At exactly
zero the bundle collapses: all ``num_micro_per_macro`` leaves of a group become
the same vector as their centroid, so the frame holds ``num_macro_classes``
distinct columns out of ``num_macro_classes * num_micro_per_macro`` -- the same
duplication ``tiled_simplex_frame`` produces, arriving through a different door
(measured at 3 distinct of 15 at ``num_macro_classes=3``). Every leaf of a group
then receives an identical logit, and the conditional softmax within that group
is **uniform by construction** (measured within-group logit spread: exactly
``0.0`` at ``spread=0``, against ``1.19e-05`` at ``1e-06`` and ``2.71`` at
``0.25``). The macro tier still discriminates; the micro tier does not.

This is a real limit rather than a rounding artifact, so the constructor does
not reject it -- a caller may legitimately want a single-leaf-per-group model,
and that is exactly the configuration ``spread=0`` describes. It is documented
here and pinned by tests so that it cannot be mistaken for a working leaf
discriminator. **Any caller wanting per-leaf discrimination must use a
strictly positive ``cluster_spread``**; there is no value below which the
collapse is avoided, because the collapse is exact at zero rather than
approximate near it.

**Joint probabilities.** With ``return_hierarchy=True`` the layer also emits the
Bayesian decomposition of a leaf probability, and the identity it is built to
satisfy is exact by construction rather than approximately::

    P(macro = m | x)             = softmax(scale * <x, C_m>)
    P(micro = k | macro = m, x)   = softmax_k(scale * <x, V_{m,k}>)
    P(class = (m, k) | x)        = P(macro = m | x) * P(micro = k | macro = m, x)

Both factors are probability distributions over their own axis, so the product
sums to 1.0 over all ``(m, k)`` without any renormalization step. That is the
property the test suite pins, and it holds to within float32 accumulation.

References:
    - Papyan, Han and Donoho, 2020. Prevalence of Neural Collapse during the
      Terminal Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Hinton, Srivastava and Teh, 2006. Reducing the Dimensionality of Data
      with Neural Networks. (https://www.cs.toronto.edu/~fritz/absps/nips06_unsup.pdf)
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.structured_linear.crystal_frames import crystal_frame
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import (
    canonical_binary_input_shape,
    l2_normalize
)

# ---------------------------------------------------------------------


def _split_leading_dims(
        tensor: keras.KerasTensor,
        macro: int,
        micro: int
) -> keras.KerasTensor:
    """
    Reshape ``(..., macro * micro)`` to ``(..., macro, micro)`` symbolically.

    ``ops.shape(x)`` returns a *tuple* whose elements are Python ``int``s in
    eager mode but symbolic scalars under ``tf.function``, so it cannot be fed
    to ``ops.concatenate`` or ``ops.stack`` mixed with Python ints -- both raise
    on this backend (measured: ``ops.concatenate`` gives ``'tuple' object has no
    attribute 'dtype'``, and ``ops.stack`` gives a ``DataType`` mismatch because
    the traced element is ``int32`` while the literals are ``int``). Tuple
    concatenation sidesteps both: the elements stay whatever ``ops.shape``
    produced and the trailing pair stays Python ``int``.

    :param tensor: Tensor whose last axis is ``macro * micro``.
    :type tensor: keras.KerasTensor
    :param macro: Size of the macro axis to split out.
    :type macro: int
    :param micro: Size of the micro axis.
    :type micro: int
    :return: Tensor of shape ``(..., macro, micro)``.
    :rtype: keras.KerasTensor
    """

    return keras.ops.reshape(tensor, keras.ops.shape(tensor)[:-1] + (macro, micro))


def _merge_trailing_pair(
        tensor: keras.KerasTensor,
        width: int
) -> keras.KerasTensor:
    """
    Reshape ``(..., width_a, width_b)`` back to ``(..., width_a * width_b)``.

    The inverse of :func:`_split_leading_dims`, and subject to the same
    constraint on why the shape tuple is built by concatenation. The two
    trailing axes are dropped -- ``shape[:-2]``, not ``shape[:-1]`` -- because
    both of them are absorbed into ``width``; keeping either would leave the
    requested element count a multiple of ``width_a * width_b`` too large.

    :param tensor: Tensor whose final two axes are to be merged.
    :type tensor: keras.KerasTensor
    :param width: The product the two trailing axes collapse to.
    :type width: int
    :return: Tensor of shape ``(..., width)``.
    :rtype: keras.KerasTensor
    """

    return keras.ops.reshape(tensor, keras.ops.shape(tensor)[:-2] + (width,))


@register_dl_technique("dl_techniques.layers.structured_linear.hierarchical_crystal_layer")
class HierarchicalCrystalLayer(keras.layers.Layer):
    """
    Two-tier spherical-code classifier emitting macro, leaf and joint outputs.

    .. code-block:: text

        input  [..., input_dim]
               │  l2 normalize
               ▼
        ┌────────────────────────┐  macro_centroids (input_dim, num_macro)
        │ macro matmul           │ ──► macro_logits  [..., num_macro]
        └────────────────────────┘
        ┌────────────────────────┐  micro_leaves    (input_dim, num_macro * num_micro)
        │ micro matmul           │ ──► micro_logits  [..., num_macro * num_micro]
        └────────────────────────┘
               │
               └─► reshape (..., num_macro, num_micro)
                   softmax over micro, × broadcast macro softmax
                   ──► joint_probabilities [..., num_macro * num_micro]

    Input shape:
        N-D tensor ``(..., input_dim)``. The last dimension must be static.

    Output shape:
        With ``return_hierarchy=False`` (the default), same rank as the input
        with the last dimension set to ``num_macro_classes * num_micro_per_macro``.
        With ``return_hierarchy=True``, a dict of three tensors at those shapes.

    :param num_macro_classes: Number of coarse groups. Must be positive.
    :type num_macro_classes: int
    :param num_micro_per_macro: Number of leaf classes within each group. Must
        be positive, and may exceed ``input_dim + 1`` -- the micro frame is
        packed from several simplexes rather than being capped at one.
    :type num_micro_per_macro: int
    :param cluster_spread: Perturbation magnitude of each leaf bundle around
        its macro centroid. Must be non-negative. Defaults to 0.25. See the
        module docstring for the exact aperture law: this is NOT an angular
        aperture past ``s ~= 0.3``.

        .. warning::
           ``0.0`` is degenerate, though legal. Every leaf of a group collapses
           onto its centroid, giving that group identical logits and a uniform
           conditional softmax. Use a strictly positive value whenever the leaf
           tier must discriminate. See the module docstring for measurements.

        .. warning::
           A bundle norm of ``sqrt(1 + 2*s*c + s*s)`` vanishes at ``s = 1``
           with ``c = -1`` (a leaf displaced exactly opposite its centroid),
           leaving an all-zero column that scores a permanent zero. The
           configuration stays legal -- the collapse is measure-zero, not a
           rounding artifact near some threshold -- but ``s = 1.0`` is not a
           working aperture.
    :type cluster_spread: float
    :param scale: Inverse-temperature multiplier on the cosine logits.
        Defaults to 10.0.
    :type scale: float
    :param return_hierarchy: Whether to return the dict of macro logits, leaf
        logits and joint probabilities instead of a bare leaf-logit tensor.
        Defaults to ``False``.
    :type return_hierarchy: bool
    :param trainable_crystal: Whether the frames may train. Defaults to ``False``.
    :type trainable_crystal: bool
    :param seed: Seed for the frame rotation sampling. ``None`` (the default)
        draws fresh frames per build; pass an integer for reproducible ones.
        Note ``seed`` is a no-op on the MACRO frame whenever
        ``num_macro_classes <= input_dim + 1``: one sliced simplex has pairwise
        inner product ``-1/input_dim`` under ANY rotation, so the centroids
        are the same equiangular set whatever the seed draws (measured maximum
        off-diagonal macro inner product ``-0.0312 = -1/32`` at
        ``input_dim=32``, identical at ``seed=None/0/1/2``). The seed still
        governs the leaf-bundle displacements, which is where distinct frames
        come from at these widths.
    :type seed: Optional[int]
    :param eps: Added inside the square root of each norm computation.
    :type eps: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar macro_weights: Macro centroids of shape ``(input_dim, num_macro_classes)``.
    :vartype macro_weights: keras.Variable
    :ivar micro_weights: Leaf vertices of shape
        ``(input_dim, num_macro_classes * num_micro_per_macro)``.
    :vartype micro_weights: keras.Variable

    :raises ValueError: If either class count is not positive.
    :raises ValueError: If ``cluster_spread`` is negative.
    :raises ValueError: If ``scale`` is not positive.
    :raises ValueError: If ``eps`` is not positive.
    """

    def __init__(
            self,
            num_macro_classes: int,
            num_micro_per_macro: int,
            cluster_spread: float = 0.25,
            scale: float = 10.0,
            return_hierarchy: bool = False,
            trainable_crystal: bool = False,
            seed: Optional[int] = None,
            eps: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if num_macro_classes <= 0:
            raise ValueError(
                f"num_macro_classes must be positive, got {num_macro_classes}"
            )
        if num_micro_per_macro <= 0:
            raise ValueError(
                f"num_micro_per_macro must be positive, got {num_micro_per_macro}"
            )
        if cluster_spread < 0:
            raise ValueError(
                f"cluster_spread must be non-negative, got {cluster_spread}"
            )
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.num_macro = num_macro_classes
        self.num_micro = num_micro_per_macro
        self.total_classes = num_macro_classes * num_micro_per_macro
        self.cluster_spread = cluster_spread
        self.scale = scale
        self.return_hierarchy = return_hierarchy
        self.trainable_crystal = trainable_crystal
        self.seed = seed
        self.eps = eps

        self.macro_weights = None
        self.micro_weights = None
        self._input_dim = None
        self._root_sequence = None
        self._child_seed_cache = None

    def _child_seeds(self) -> list:
        """
        Return the child seed sequences both frames are derived from.

        Child 0 is the macro centroid frame; children 1..num_macro are the leaf
        bundles. Both initializers call this and get the SAME list, so the leaf
        bundles are guaranteed to be centred on the centroids that actually
        landed in ``macro_weights``.

        Two things had to be right for that, and both are measurable:

        - The root is built ONCE and cached. ``SeedSequence(None)`` draws fresh
          entropy per construction, so a new root per initializer call hands the
          two frames different roots (measured leaf-to-stored-centroid cosine of
          0.30 at ``seed=None``).
        - The spawn is memoized too. ``SeedSequence.spawn`` is STATEFUL -- each
          call advances the internal counter and returns the next children -- so
          ``spawn(1)[0]`` and ``spawn(num_macro + 1)[0]`` on the same root are
          *different* sequences (measured 0.65 at ``seed=0`` before this was
          fixed, against 0.98 after).

        :return: ``num_macro + 1`` child seed sequences.
        :rtype: list
        """

        if self._child_seed_cache is None:
            self._root_sequence = np.random.SeedSequence(self.seed)
            self._child_seed_cache = self._root_sequence.spawn(self.num_macro + 1)

        return self._child_seed_cache

    def _macro_frame(self, input_dim: int) -> np.ndarray:
        """
        Build the macro centroid frame.

        Drawn from the root seed's FIRST child stream, so that
        :meth:`_micro_initializer` can build the remaining frames from the SAME
        root without re-drawing the macro frame.

        Re-drawing it there instead would be silently wrong whenever
        ``seed is None``: the two draws are independent, so the leaf bundles
        would be centred on centroids that are not the ones stored in
        ``macro_weights`` -- the leaf geometry would describe a different
        classifier from the one the layer actually runs (measured: leaf-to-stored
        centroid cosine of 0.30 at ``seed=None`` against 0.99 at ``seed=0``,
        where the two draws happen to agree).

        :param input_dim: Ambient dimensionality.
        :type input_dim: int
        :return: Centroid frame of shape ``(input_dim, num_macro)``, float64.
        :rtype: np.ndarray
        """

        return crystal_frame(
            dimensions=input_dim,
            width=self.num_macro,
            seed=self._child_seeds()[0]
        ).astype(np.float64)

    def _macro_initializer(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """
        Build the macro centroid frame for ``add_weight``.

        :param shape: Target weight shape ``(input_dim, num_macro)``.
        :type shape: Tuple[int, ...]
        :param dtype: Dtype requested by the caller; the result is cast to it.
        :type dtype: Optional[Any]
        :return: Centroid frame of shape ``shape``.
        :rtype: np.ndarray
        """

        return self._macro_frame(shape[0]).astype(dtype or self.dtype)

    def _micro_initializer(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """
        Build the leaf vertices as isotropic bundles around the macro centroids.

        Each group draws its own frame from its own spawned stream, so two
        groups never receive the same displacement directions and collapse onto
        identical bundles.

        The centroids are re-derived here with a second packing pass rather
        than read from ``macro_weights``: this initializer runs inside
        ``add_weight``, where ``macro_weights`` may not exist yet, and sharing
        one frame object between two weights would alias them. The doubling is
        deliberate -- do not "simplify" it into a read of the sibling weight.

        :param shape: Target weight shape ``(input_dim, total_classes)``.
        :type shape: Tuple[int, ...]
        :param dtype: Dtype requested by the caller; the result is cast to it.
        :type dtype: Optional[Any]
        :return: Leaf vertex frame of shape ``shape``.
        :rtype: np.ndarray
        """

        input_dim = shape[0]

        # Child 0 is the macro frame's stream and the bundles take 1..num_macro,
        # so the two sets of draws cannot overlap. Decorrelating by
        # `seed + group` instead does NOT work: crystal_frame draws its first
        # simplex from the generator, so a group whose offset lands on the macro
        # seed reproduces the macro simplex exactly, and that leaf then sits at
        # cosine 1.0 with its own centroid whatever cluster_spread is (measured).
        children = self._child_seeds()
        centroids = crystal_frame(
            dimensions=input_dim,
            width=self.num_macro,
            seed=children[0]
        ).astype(np.float64)

        bundles = []
        for group in range(self.num_macro):
            frame = crystal_frame(
                dimensions=input_dim,
                width=self.num_micro,
                seed=children[group + 1]
            ).astype(np.float64)  # (input_dim, num_micro)

            centroid = centroids[:, group:group + 1]  # (input_dim, 1)

            # Isotropic bundle: offset from the centroid, then renormalized back
            # onto the sphere. The renormalization is what keeps the leaves unit
            # vectors, which the cosine logits downstream assume. The 1e-12
            # floor is load-bearing, not decoration: at cluster_spread=1 a leaf
            # displaced exactly opposite its centroid has a bundle norm of
            # exactly 0, and without the floor that leaf is NaN rather than an
            # all-zero column (see the module docstring).
            bundle = centroid + self.cluster_spread * frame
            norms = np.linalg.norm(bundle, axis=0, keepdims=True)
            bundles.append(bundle / np.maximum(norms, 1e-12))

        return np.concatenate(bundles, axis=1).astype(dtype or self.dtype)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the macro centroid and leaf vertex weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is ``None``.
        """

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input must be defined; "
                f"HierarchicalCrystalLayer needs a static input_dim, "
                f"got input_shape={input_shape}"
            )

        self._input_dim = input_dim

        self.macro_weights = self.add_weight(
            name='macro_centroids',
            shape=(input_dim, self.num_macro),
            initializer=self._macro_initializer,
            trainable=self.trainable_crystal,
            dtype=self.dtype
        )

        self.micro_weights = self.add_weight(
            name='micro_leaves',
            shape=(input_dim, self.total_classes),
            initializer=self._micro_initializer,
            trainable=self.trainable_crystal,
            dtype=self.dtype
        )

        super().build(input_shape)

    def _normalize(self, tensor: keras.KerasTensor, axis: int) -> keras.KerasTensor:
        """
        L2-normalize ``tensor`` along ``axis``.

        Delegates to the shared helper, which rescales before squaring. The
        naive form returns exact zeros above ~1.8e19 and loses the norm entirely
        near 1e-8, where ``eps`` dominates it -- both silently.

        :param tensor: Tensor to normalize.
        :type tensor: keras.KerasTensor
        :param axis: Axis to reduce over.
        :type axis: int
        :return: Tensor of the same shape, normalized along ``axis``.
        :rtype: keras.KerasTensor
        """

        return l2_normalize(tensor, axis=axis, eps=self.eps)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Compute macro and leaf logits, and optionally their Bayesian product.

        :param inputs: Input tensor ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; accepted for API symmetry.
        :type training: Optional[bool]
        :return: Leaf logits ``(..., num_macro * num_micro)``, or with
            ``return_hierarchy=True`` a dict of ``macro_logits``,
            ``micro_logits`` and ``joint_probabilities``. The joint
            probabilities sum to 1.0 over the last axis.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :raises ValueError: If the last dimension of ``inputs`` disagrees with
            the width this layer was built for.
        """

        if inputs.shape[-1] != self._input_dim:
            raise ValueError(
                f"HierarchicalCrystalLayer was built for input_dim={self._input_dim} "
                f"but received input_dim={inputs.shape[-1]}"
            )

        normalized_inputs = self._normalize(inputs, axis=-1)

        macro_logits = keras.ops.matmul(
            normalized_inputs, self._normalize(self.macro_weights, axis=0)
        ) * self.scale

        micro_logits = keras.ops.matmul(
            normalized_inputs, self._normalize(self.micro_weights, axis=0)
        ) * self.scale

        if not self.return_hierarchy:
            return micro_logits

        macro_probs = keras.ops.softmax(macro_logits, axis=-1)

        # (..., num_macro * num_micro) -> (..., num_macro, num_micro) so the
        # conditional softmax reduces within a group rather than across groups.
        grouped = _split_leading_dims(micro_logits, self.num_macro, self.num_micro)
        conditional_probs = keras.ops.softmax(grouped, axis=-1)

        # Broadcast the macro distribution down the micro axis and multiply.
        # Both factors are normalized distributions on their own axis, which is
        # what makes the product sum to 1.0 without a renormalization pass.
        expanded_macro = keras.ops.expand_dims(macro_probs, axis=-1)
        joint = expanded_macro * conditional_probs

        return {
            'macro_logits': macro_logits,
            'micro_logits': micro_logits,
            'joint_probabilities': _merge_trailing_pair(joint, self.total_classes),
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]:
        """
        Compute output shape from stored config, on an unbuilt layer.

        Both branches are derived from ``self.return_hierarchy`` alone, so the
        flag cannot leave one of them unstated.

        :param input_shape: Shape of the input. A shape arriving as a list is
            canonicalized first; ``input_shape[:-1]`` over ``[[None, 8]]``
            slices the outer list and would drop the batch axis entirely.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape, or the per-key dict of output shapes when
            ``return_hierarchy`` is set.
        :rtype: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        """

        base = tuple(canonical_binary_input_shape(input_shape)[:-1])

        if not self.return_hierarchy:
            return base + (self.total_classes,)

        return {
            'macro_logits': base + (self.num_macro,),
            'micro_logits': base + (self.total_classes,),
            'joint_probabilities': base + (self.total_classes,),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        ``total_classes`` is deliberately absent: it is the product of two
        constructor arguments, and serializing a derived value invites the two
        copies to disagree.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """

        config = super().get_config()
        config.update({
            'num_macro_classes': self.num_macro,
            'num_micro_per_macro': self.num_micro,
            'cluster_spread': self.cluster_spread,
            'scale': self.scale,
            'return_hierarchy': self.return_hierarchy,
            'trainable_crystal': self.trainable_crystal,
            'seed': self.seed,
            'eps': self.eps,
        })

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HierarchicalCrystalLayer":
        """Create a layer instance from configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New HierarchicalCrystalLayer instance.
        :rtype: HierarchicalCrystalLayer
        """
        return cls(**config)

# ---------------------------------------------------------------------
