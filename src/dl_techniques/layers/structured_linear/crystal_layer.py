"""CrystalLayer, a classification head over a packed equiangular spherical code.

Neural Collapse (Papyan et al., 2020) describes what a classifier's final
layer converges to during the terminal phase of training: the feature
representations and the class prototypes collapse onto a regular simplex --
an Equiangular Tight Frame -- inscribed on the unit hypersphere. This layer
takes that endpoint as its *initialization* rather than waiting for training to
discover it.

The head computes cosine-similarity logits between an L2-normalized input `x`
and a frame `W` of `num_classes` unit vectors::

    logits = scale * (x / ||x||) @ (W / ||W||)

**The frame is overcomplete.** A single simplex in `N` dimensions holds exactly
`N + 1` directions, so `num_classes > N + 1` cannot come from one. Rather than
repeat vertices -- which would make every repeated column an exact duplicate
receiving an identical logit -- the frame is packed from several *rotated*
simplexes by a greedy minimax-angular-separation search. See
:mod:`dl_techniques.layers.structured_linear.crystal_frames`.

The packing gives `num_classes` distinct columns at every width measured
(up to 50,000 in 2 dimensions). Only `input_dim=1` forces a repeat, because
`S^0 = {-1, +1}` is two points -- at `input_dim=1` a 3-class head over a 1-D
feature has only two distinct directions available. What degrades with width is
the angular separation between columns, not their distinctness: the frame is a
dense spherical code past the simplex count, and at low dimension a wide frame
holds near-identical columns (measured at `input_dim=2, num_classes=64`: a
pair at inner product 0.9992, i.e. near-identical logits). A frame is a set of
*lines*, but the packing search prices only the signed inner product, so an
antipodal pair (``cos -> -1``, the same line in the opposite sense) is never
penalised -- see :mod:`dl_techniques.layers.structured_linear.crystal_frames`
for the measured antipodal degeneracy, which is documented there and left
unchanged here by design.

Setting ``trainable_crystal=True`` releases the frame so the network may leave
the collapsed configuration it was handed. Weights are re-normalized inside
``call()`` in that mode, because a trained frame drifts off the unit sphere and
the logits would otherwise stop being cosines. The normalization is applied
unconditionally rather than only in that mode: the frozen frames are already
unit-norm, so the two paths agree numerically while the code has one branch
instead of two.

References:
    - Papyan, Han and Donoho, 2020. Prevalence of Neural Collapse during the
      Terminal Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Strohmer and Heath, 2003. Grassmannian Frames with Applications to
      Coding and Communication. (Applied and Computational Harmonic Analysis
      14(3))
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple

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


@register_dl_technique("dl_techniques.layers.structured_linear.crystal_layer")
class CrystalLayer(keras.layers.Layer):
    """
    Cosine logits against a frozen (or releasable) spherical-code frame.

    .. code-block:: text

        input  [..., input_dim]
               │  l2 normalize  (eps inside the sqrt: x = 0 must not divide by 0)
               ▼
        ┌────────────────────────────┐
        │ crystal_vertices matmul    │  (input_dim, num_classes), frozen
        └────────────────────────────┘
               │ [..., num_classes]
               ▼  × scale
        output [..., num_classes]

    Input shape:
        N-D tensor ``(..., input_dim)``. The last dimension must be static.

    Output shape:
        Same rank as the input, with the last dimension set to ``num_classes``.

    :param num_classes: Number of output logits, i.e. of frame columns.
    :type num_classes: int
    :param scale: Inverse-temperature multiplier on the cosine logits.
        Defaults to 10.0.
    :type scale: float
    :param trainable_crystal: Whether the frame may train. Defaults to ``False``,
        keeping the Neural Collapse prior fixed at initialization.
    :type trainable_crystal: bool
    :param candidate_trials: Rotations sampled per simplex while packing the
        frame. Values below 2 skip the search. Defaults to 5.
    :type candidate_trials: int
    :param seed: Seed for the frame's rotation sampling. ``None`` (the default)
        draws a fresh frame per build; pass an integer for a reproducible one.
    :type seed: Optional[int]
    :param eps: Added inside the square root of the norm computation, so a
        zero input yields a zero output instead of a division by zero.
    :type eps: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar crystal_weights: Frame of shape ``(input_dim, num_classes)``.
    :vartype crystal_weights: keras.Variable

    :raises ValueError: If ``num_classes`` is not positive.
    :raises ValueError: If ``scale`` is not positive.
    :raises ValueError: If ``candidate_trials`` is not positive.
    :raises ValueError: If ``eps`` is not positive.
    """

    def __init__(
            self,
            num_classes: int,
            scale: float = 10.0,
            trainable_crystal: bool = False,
            candidate_trials: int = 5,
            seed: Optional[int] = None,
            eps: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if candidate_trials < 1:
            raise ValueError(f"candidate_trials must be >= 1, got {candidate_trials}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.num_classes = num_classes
        self.scale = scale
        self.trainable_crystal = trainable_crystal
        self.candidate_trials = candidate_trials
        self.seed = seed
        self.eps = eps

        self.crystal_weights = None
        self._input_dim = None

    def _frame_initializer(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """
        Build the packed frame for this layer's input width.

        The frame is computed here rather than assigned after ``add_weight``:
        Keras runs a parent's symbolic build inside a ``StatelessScope`` that
        records assignments and discards them, so a table assigned outside an
        initializer comes back all zeros in every real model. Computing it
        inside the initializer leaves nothing to discard.

        :param shape: Target weight shape ``(input_dim, num_classes)``.
        :type shape: Tuple[int, ...]
        :param dtype: Dtype requested by the caller; the result is cast to it.
        :type dtype: Optional[Any]
        :return: Frame of shape ``shape`` as a NumPy array.
        :rtype: np.ndarray
        """

        input_dim = shape[0]

        frame = crystal_frame(
            dimensions=input_dim,
            width=self.num_classes,
            candidate_trials=self.candidate_trials,
            seed=self.seed
        )

        return frame.astype(dtype or self.compute_dtype)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the frame weight.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is ``None``.
        """

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input must be defined; "
                f"CrystalLayer needs a static input_dim, got input_shape={input_shape}"
            )

        self._input_dim = input_dim

        self.crystal_weights = self.add_weight(
            name='crystal_vertices',
            shape=(input_dim, self.num_classes),
            initializer=self._frame_initializer,
            trainable=self.trainable_crystal,
            dtype=self.dtype
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute scaled cosine logits against the frame.

        :param inputs: Input tensor ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; accepted for API symmetry so the layer drops
            into a training call site unchanged.
        :type training: Optional[bool]
        :return: Logits of shape ``(..., num_classes)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the last dimension of ``inputs`` disagrees with
            the width this layer was built for.
        """

        if inputs.shape[-1] != self._input_dim:
            raise ValueError(
                f"CrystalLayer was built for input_dim={self._input_dim} "
                f"but received input_dim={inputs.shape[-1]}"
            )

        # keras.ops has no l2_normalize in keras 3.8, so this routes through the
        # shared utils helper, which rescales before squaring. The naive form
        # returns exact zeros for inputs above ~1.8e19 (x/inf == 0) and loses
        # the norm entirely for inputs around 1e-8, where eps dominates it.
        normalized_inputs = l2_normalize(inputs, axis=-1, eps=self.eps)

        # Applied in both modes. The frozen frame is already unit-norm, so the
        # frozen path is numerically unchanged by this; running it in both
        # keeps one branch and makes a drifted trainable frame still yield
        # true cosines.
        normalized_weights = l2_normalize(
            self.crystal_weights, axis=0, eps=self.eps
        )

        return keras.ops.matmul(normalized_inputs, normalized_weights) * self.scale

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape from stored config, on an unbuilt layer.

        :param input_shape: Shape of the input. A shape arriving as a list is
            canonicalized first; ``list(input_shape)`` over ``[[None, 8]]``
            iterates the outer list and would rewrite the batch axis, returning
            ``(num_classes,)`` instead of ``(None, num_classes)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension replaced by ``num_classes``.
        :rtype: Tuple[Optional[int], ...]
        """

        output_shape = list(canonical_binary_input_shape(input_shape))
        output_shape[-1] = self.num_classes

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """

        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'scale': self.scale,
            'trainable_crystal': self.trainable_crystal,
            'candidate_trials': self.candidate_trials,
            'seed': self.seed,
            'eps': self.eps,
        })

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrystalLayer":
        """Create a layer instance from configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New CrystalLayer instance.
        :rtype: CrystalLayer
        """
        return cls(**config)

# ---------------------------------------------------------------------
