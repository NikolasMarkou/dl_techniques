"""CrystalMultiHeadProjection, a geometric multi-head filter bank for encoders.

:class:`~dl_techniques.layers.structured_linear.crystal_layer.CrystalLayer` uses
one frame on the whole feature vector. This layer splits the vector into ``H``
independent sub-spaces and gives each its own frame, then reads out ``D_h + 1``
directional similarities per head. A head therefore sees a genuinely separate
coordinate block, and the ``H`` heads' readouts are combined by a learned
projection rather than by sharing one projection's weights.

Mechanism, for input ``[..., D]`` split into ``H`` heads of width ``D_h = D // H``:

1. Reshape to ``[..., H, D_h`` and L2-normalize each head separately.
2. Contract against ``H`` independent prisms of shape ``(H, D_h, V)``, where each
   prism is a rotated simplex of ``V = D_h + 1`` vertices, via Einstein
   summation ``"...hd,hdv->...hv"``. The ``h`` index appearing in both operands
   is what keeps the heads from mixing: this is a batched contraction, not a
   single wide matmul.
3. Scale, apply GELU, flatten the ``H * V`` features, and project to
   ``output_dim``.

**The sub-layer is created in ``__init__`` and built in ``build()``,** not
constructed inside ``build()``. A layer built inside ``build()`` is registered
late and named by a counter, so its auto-generated name shifts whenever anything
ahead of it changes and by-name weight transfer stops matching. The Dense is
given an explicit name and its ``kernel_initializer`` / ``use_bias`` are
constructor arguments that reach ``get_config()`` -- reaching Keras' defaults by
omission while being unable to record them is the shape a knob silently takes.

References:
    - Papyan, Han and Donoho, 2020. Prevalence of Neural Collapse during the
      Terminal Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.structured_linear.crystal_frames import (
    haar_orthogonal,
    hyperspherical_simplex
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import canonical_binary_input_shape

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.structured_linear.crystal_multihead_projection")
class CrystalMultiHeadProjection(keras.layers.Layer):
    """
    Split features into heads, read each through a rotated simplex prism.

    .. code-block:: text

        input  (..., D),  D % num_heads == 0
               │  reshape (..., H, D_h)
               │  l2 normalize per head
               ▼
        ┌────────────────────────────────────┐
        │ einsum "...hd,hdv->...hv"           │  crystal_prisms (H, D_h, V)
        │ × scale, then GELU                 │  V = D_h + 1 vertices per head
        └────────────────────────────────────┘
               │ (..., H, V)
               │ reshape (..., H * V)
               ▼
        ┌────────────────────────────────────┐
        │ out_projection Dense               │  (H * V, output_dim)
        └────────────────────────────────────┘
               │
               ▼
        output (..., output_dim)

    Input shape:
        N-D tensor ``(..., D)``. The last dimension must be static and divisible
        by ``num_heads``.

    Output shape:
        Same rank as the input, with the last dimension set to ``output_dim``,
        or to ``D`` when ``output_dim`` is ``None``.

    :param num_heads: Number of sub-spaces. Must be positive and divide the
        input width.
    :type num_heads: int
    :param output_dim: Width of the output. ``None`` (the default) keeps the
        input width.
    :type output_dim: Optional[int]
    :param scale: Inverse-temperature multiplier on the prism similarities.
        Defaults to 5.0.
    :type scale: float
    :param trainable_crystal: Whether the prisms may train. Defaults to ``False``.
    :type trainable_crystal: bool
    :param seed: Seed for the prism rotations. ``None`` (the default) draws
        fresh prisms per build; pass an integer for reproducible ones.
    :type seed: Optional[int]
    :param eps: Added inside the square root of each normalization.
    :type eps: float
    :param kernel_initializer: Initializer for the output projection's kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param use_bias: Whether the output projection carries a bias.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar crystal_prisms: Prisms of shape ``(num_heads, head_dim, head_dim + 1)``.
    :vartype crystal_prisms: keras.Variable
    :ivar out_projection: Dense projection back to ``output_dim``.
    :vartype out_projection: keras.layers.Dense

    :raises ValueError: If ``num_heads`` is not positive.
    :raises ValueError: If ``output_dim`` is given and not positive.
    :raises ValueError: If ``scale`` or ``eps`` is not positive.
    """

    def __init__(
            self,
            num_heads: int,
            output_dim: Optional[int] = None,
            scale: float = 5.0,
            trainable_crystal: bool = False,
            seed: Optional[int] = None,
            eps: float = 1e-7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive or None, got {output_dim}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.num_heads = num_heads
        self.output_dim = output_dim
        self.scale = scale
        self.trainable_crystal = trainable_crystal
        self.seed = seed
        self.eps = eps
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias

        # Created ONCE, here, and given an explicit name. Constructing it inside
        # build() instead is what produces the shifting auto-name. The width is
        # a placeholder when output_dim is None: the real one is known only in
        # build(), where `units` is set before the Dense builds its kernel. The
        # placeholder is never built and so contributes no weights on that path.
        self.out_projection = keras.layers.Dense(
            output_dim if output_dim is not None else 1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='out_projection'
        )

        self.crystal_prisms = None
        self._input_dim = None
        self._head_dim = None
        self._vertices_per_head = None
        self._resolved_output_dim = None

    def _prism_initializer(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        """
        Build ``num_heads`` independently rotated simplex prisms.

        :param shape: Target weight shape ``(H, D_h, D_h + 1)``.
        :type shape: Tuple[int, ...]
        :param dtype: Dtype requested by the caller; the result is cast to it.
        :type dtype: Optional[Any]
        :return: Prisms of shape ``shape``.
        :rtype: np.ndarray
        """

        num_heads, head_dim, vertices = shape
        rng = np.random.default_rng(self.seed)
        simplex = hyperspherical_simplex(head_dim)

        prisms = []
        for _ in range(num_heads):
            rotation = haar_orthogonal(head_dim, rng)
            # (D_h + 1, D_h) rotated on the right is (D_h + 1, D_h); transposing
            # gives the (D_h, D_h + 1) contraction the einsum consumes.
            prisms.append((simplex @ rotation).T)

        return np.stack(prisms, axis=0).astype(dtype or self.compute_dtype)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the prisms and materialize the output projection.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is ``None``.
        :raises ValueError: If the input width is not divisible by ``num_heads``.
        """

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input must be defined; "
                f"CrystalMultiHeadProjection needs a static input_dim, "
                f"got input_shape={input_shape}"
            )

        if input_dim % self.num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads "
                f"({self.num_heads})"
            )

        self._input_dim = input_dim
        self._head_dim = input_dim // self.num_heads
        self._vertices_per_head = self._head_dim + 1
        self._resolved_output_dim = self.output_dim if self.output_dim is not None \
            else input_dim

        self.crystal_prisms = self.add_weight(
            name='crystal_prisms',
            shape=(self.num_heads, self._head_dim, self._vertices_per_head),
            initializer=self._prism_initializer,
            trainable=self.trainable_crystal,
            dtype=self.dtype
        )

        # Materialize exactly what call() runs. The feature width here is
        # num_heads * vertices_per_head, not the input width -- the prisms
        # expand each head from D_h features to D_h + 1 readouts. `units` is
        # assigned before build() because that is when Dense sizes its kernel.
        self.out_projection.units = self._resolved_output_dim
        self.out_projection.build(
            tuple(input_shape[:-1]) + (self.num_heads * self._vertices_per_head,)
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Split, contract against the prisms, and project back.

        :param inputs: Input tensor ``(..., D)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded explicitly to the output projection, which
            takes no training-dependent path today.
        :type training: Optional[bool]
        :return: Tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the last dimension of ``inputs`` disagrees with
            the width this layer was built for.
        """

        # The width contract is re-asserted here, not only in build(): build()
        # only ever sees the shape that arrived first, so a contract checked
        # there alone is checked exactly once.
        #
        # Divisibility by num_heads is deliberately NOT re-checked. It is a
        # property of `self._input_dim`, which build() has already validated and
        # which never changes afterwards -- so a second check on it can never
        # fire, and a guard that cannot fail is worse than no guard.
        if inputs.shape[-1] != self._input_dim:
            raise ValueError(
                f"CrystalMultiHeadProjection was built for input_dim={self._input_dim} "
                f"but received input_dim={inputs.shape[-1]}"
            )

        # ops.shape returns a tuple whose elements are Python ints eagerly but
        # symbolic scalars when traced, so the target shape is built by tuple
        # concatenation -- mixing those elements into ops.stack or
        # ops.concatenate with Python ints raises on this backend.
        split_shape = keras.ops.shape(inputs)[:-1] + (self.num_heads, self._head_dim)
        heads = keras.ops.reshape(inputs, split_shape)

        head_norms = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(heads), axis=-1, keepdims=True) + self.eps
        )
        normalized_heads = heads / head_norms

        prism_norms = keras.ops.sqrt(
            keras.ops.sum(
                keras.ops.square(self.crystal_prisms), axis=1, keepdims=True
            ) + self.eps
        )
        normalized_prisms = self.crystal_prisms / prism_norms

        # "h" in both operands makes this a per-head batched contraction, not a
        # wide matmul across heads.
        features = keras.ops.einsum(
            '...hd,hdv->...hv', normalized_heads, normalized_prisms
        ) * self.scale
        features = keras.ops.gelu(features)

        total_features = self.num_heads * self._vertices_per_head
        flat_shape = keras.ops.shape(inputs)[:-1] + (total_features,)

        return self.out_projection(
            keras.ops.reshape(features, flat_shape), training=training
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape from stored config, on an unbuilt layer.

        Reads ``self.output_dim`` rather than the resolved value, so it works
        before ``build()`` has seen an input width.

        :param input_shape: Shape of the input. A shape arriving as a list is
            canonicalized first; ``list(input_shape)`` over ``[[None, 8]]``
            iterates the outer list and would rewrite the batch axis.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension set to ``output_dim``, or to
            the input's own width when ``output_dim`` is ``None``.
        :rtype: Tuple[Optional[int], ...]
        """

        shape = canonical_binary_input_shape(input_shape)

        output_shape = list(shape)
        output_shape[-1] = self.output_dim if self.output_dim is not None \
            else shape[-1]

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        The output projection's own config is **not** nested under a sub-key;
        this layer's constructor reproduces those settings directly, and
        ``from_config`` rebuilds the Dense from them.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """

        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'scale': self.scale,
            'trainable_crystal': self.trainable_crystal,
            'seed': self.seed,
            'eps': self.eps,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'use_bias': self.use_bias,
        })

        return config

# ---------------------------------------------------------------------
