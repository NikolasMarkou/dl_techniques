"""SD3 MMDiT adaptive layer normalization (AdaLN) modulation trio.

Three AdaLN variants that condition a hidden-state stream `x` of shape
`(B, N, dim)` on a single per-sample vector `cond` of shape `(B, dim)`:
:class:`AdaLayerNormZero` (6-way, attention + MLP shift/scale/gate),
:class:`AdaLayerNormZeroX` (9-way, adds a second attention path reusing
the same normalized stream), and :class:`AdaLayerNormContinuous` (2-way,
scale + shift with no gate, for the final blocks). Each runs
`SiLU(cond) -> Dense(K*dim) -> split` to produce `K` modulation chunks,
then modulates an affine-free `LayerNormalization` of `x`.

The modulation Dense is zero-initialized (kernel and bias), so at
initialization every chunk is zero: the normalized stream equals a plain
no-affine LayerNorm of `x` and every gate is `0`, the identity-at-init
property that lets the optimizer turn conditioning on gradually. The
gate and MLP-shift/scale chunks are returned at shape `(B, dim)` for the
surrounding block to expand as needed; only the in-layer `x_norm`
modulation is broadcast here, to `(B, 1, dim)`.

This is a Keras 3 port of diffusers' `AdaLayerNormZero` /
`AdaLayerNormZeroX` / `AdaLayerNormContinuous`.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


def _unpack_pair_shape(
    input_shape: Any,
) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
    """Split a ``[x_shape, cond_shape]`` build/compute input into its parts.

    :param input_shape: a list/tuple of exactly two shapes ``[x_shape, cond_shape]``
        where ``x_shape = (B, N, dim)`` and ``cond_shape = (B, dim)``.
    :type input_shape: Any
    :return: ``(x_shape, cond_shape)``.
    :rtype: Tuple[Tuple, Tuple]
    :raises ValueError: If ``input_shape`` is not a pair of shapes.
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError(
            "Expected input_shape to be a pair [x_shape, cond_shape], got "
            f"{input_shape}"
        )
    x_shape, cond_shape = input_shape[0], input_shape[1]
    if not isinstance(x_shape, (list, tuple)) or not isinstance(
        cond_shape, (list, tuple)
    ):
        raise ValueError(
            "Each element of input_shape must itself be a shape tuple; got "
            f"x_shape={x_shape}, cond_shape={cond_shape}"
        )
    return tuple(x_shape), tuple(cond_shape)


# DECISION plan-2026-08-31T175140-a4e0c303/D-015: this helper's name stays
# public since models/vision_language/sd3_mmdit/blocks.py imports it. Do not unify with AdaLNZeroConditionalBlock._modulate. See decisions.md.
def modulate(
    h: keras.KerasTensor,
    shift: keras.KerasTensor,
    scale: keras.KerasTensor,
) -> keras.KerasTensor:
    """AdaLN modulation ``h * (1 + scale) + shift`` with ``(B, dim)`` chunks.

    ``h`` is ``(B, N, dim)`` and ``shift`` / ``scale`` are ``(B, dim)``; this
    function broadcasts them to ``(B, 1, dim)`` via ``expand_dims`` so the
    modulation is graph-safe.

    .. note::
       A same-named ``@staticmethod`` exists on
       :class:`~dl_techniques.layers.transformers.adaln_zero.AdaLNZeroConditionalBlock`
       with a different contract: it has no ``expand_dims``, so its caller owns
       the broadcast and passes already-aligned operands. The two stay separate
       because merging them would move a ``(B, dim)`` operand from the sample
       axis to the sequence axis.
    """
    scale = keras.ops.expand_dims(scale, axis=1)
    shift = keras.ops.expand_dims(shift, axis=1)
    return h * (1.0 + scale) + shift


# =====================================================================
# AdaLayerNormZero (6-way)
# =====================================================================


@register_dl_technique("dl_techniques.layers.transformers.sd3_adaln")
class AdaLayerNormZero(keras.layers.Layer):
    """SD3 6-way AdaLN-Zero modulation (attention + MLP shift/scale/gate).

    Produces the pre-attention normalized + modulated stream and the five
    remaining modulation chunks the surrounding MMDiT block applies (the
    attention gate, and the MLP shift/scale/gate).

    The modulation ``Dense(6*dim)`` is zero-initialized so at init the
    returned ``x_norm`` equals a plain no-affine LayerNorm of ``x`` and all
    gates are ``0`` (identity-at-init AdaLN-Zero property).

    :param dim: Model / embedding dimensionality.
    :type dim: int
    :param eps: Epsilon for the affine-free LayerNorm. Defaults to ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` is not a positive integer or ``eps <= 0``.

    Input/Output:
        ``call([x, cond])`` with ``x: (B, N, dim)`` and ``cond: (B, dim)``.
        Returns the tuple ``(x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)``
        where ``x_norm`` is ``(B, N, dim)`` and the four chunks are ``(B, dim)``.

    Example:
        >>> ln = AdaLayerNormZero(dim=64)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = ln([x, c])
        >>> x_norm.shape, gate_msa.shape
        ((2, 16, 64), (2, 64))
    """

    _N_CHUNKS = 6

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = float(eps)

        self.silu = keras.layers.Activation("silu", name="silu")
        # Zero-init: identity-at-init AdaLN-Zero property.
        self.linear = keras.layers.Dense(
            self._N_CHUNKS * dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="linear",
        )
        self.norm = keras.layers.LayerNormalization(
            epsilon=self.eps, center=False, scale=False, name="norm"
        )

        logger.debug(
            f"Initialized AdaLayerNormZero(dim={dim}, eps={self.eps})"
        )

    def build(self, input_shape: Any) -> None:
        if self.built:
            return

        x_shape, cond_shape = _unpack_pair_shape(input_shape)
        if x_shape[-1] != self.dim:
            raise ValueError(
                f"x last dim must be dim={self.dim}, got x_shape={x_shape}"
            )
        if cond_shape[-1] != self.dim:
            raise ValueError(
                f"cond last dim must be dim={self.dim}, got cond_shape={cond_shape}"
            )
        self.silu.build(cond_shape)
        self.linear.build(cond_shape)
        self.norm.build(x_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, ...]:
        x, cond = inputs[0], inputs[1]
        emb = self.linear(self.silu(cond))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = keras.ops.split(emb, self._N_CHUNKS, axis=-1)
        x_norm = modulate(self.norm(x), shift_msa, scale_msa)
        return x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def compute_output_shape(
        self, input_shape: Any
    ) -> List[Tuple[Optional[int], ...]]:
        x_shape, cond_shape = _unpack_pair_shape(input_shape)
        chunk_shape = (cond_shape[0], self.dim)
        return [
            tuple(x_shape),  # x_norm
            chunk_shape,     # gate_msa
            chunk_shape,     # shift_mlp
            chunk_shape,     # scale_mlp
            chunk_shape,     # gate_mlp
        ]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"dim": self.dim, "eps": self.eps})
        return config


# =====================================================================
# AdaLayerNormZeroX (9-way, dual attention)
# =====================================================================


@register_dl_technique("dl_techniques.layers.transformers.sd3_adaln")
class AdaLayerNormZeroX(keras.layers.Layer):
    """SD3 9-way AdaLN-Zero modulation with a dual (second) attention path.

    Extends :class:`AdaLayerNormZero` with a second attention modulation
    triple (``shift_msa2``, ``scale_msa2``, ``gate_msa2``). ``norm(x)`` is
    computed once and reused to build both the primary ``x_norm`` and the
    secondary ``x_norm2`` (matches PyTorch ``AdaLayerNormZeroX``).

    The modulation ``Dense(9*dim)`` is zero-initialized so at init both
    ``x_norm`` and ``x_norm2`` equal a plain no-affine LayerNorm of ``x`` and
    all three gates are ``0``.

    :param dim: Model / embedding dimensionality.
    :type dim: int
    :param eps: Epsilon for the affine-free LayerNorm. Defaults to ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` is not a positive integer or ``eps <= 0``.

    Input/Output:
        ``call([x, cond])`` with ``x: (B, N, dim)`` and ``cond: (B, dim)``.
        Returns ``(x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp, x_norm2,
        gate_msa2)`` where ``x_norm`` / ``x_norm2`` are ``(B, N, dim)`` and the
        remaining chunks are ``(B, dim)``.
    """

    _N_CHUNKS = 9

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = float(eps)

        self.silu = keras.layers.Activation("silu", name="silu")
        self.linear = keras.layers.Dense(
            self._N_CHUNKS * dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="linear",
        )
        self.norm = keras.layers.LayerNormalization(
            epsilon=self.eps, center=False, scale=False, name="norm"
        )

        logger.debug(
            f"Initialized AdaLayerNormZeroX(dim={dim}, eps={self.eps})"
        )

    def build(self, input_shape: Any) -> None:
        if self.built:
            return

        x_shape, cond_shape = _unpack_pair_shape(input_shape)
        if x_shape[-1] != self.dim:
            raise ValueError(
                f"x last dim must be dim={self.dim}, got x_shape={x_shape}"
            )
        if cond_shape[-1] != self.dim:
            raise ValueError(
                f"cond last dim must be dim={self.dim}, got cond_shape={cond_shape}"
            )
        self.silu.build(cond_shape)
        self.linear.build(cond_shape)
        self.norm.build(x_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, ...]:
        x, cond = inputs[0], inputs[1]
        emb = self.linear(self.silu(cond))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = keras.ops.split(emb, self._N_CHUNKS, axis=-1)
        # norm(x) computed once, reused for both modulations (PyTorch parity).
        n = self.norm(x)
        x_norm = modulate(n, shift_msa, scale_msa)
        x_norm2 = modulate(n, shift_msa2, scale_msa2)
        return (
            x_norm,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            x_norm2,
            gate_msa2,
        )

    def compute_output_shape(
        self, input_shape: Any
    ) -> List[Tuple[Optional[int], ...]]:
        x_shape, cond_shape = _unpack_pair_shape(input_shape)
        x_shape = tuple(x_shape)
        chunk_shape = (cond_shape[0], self.dim)
        return [
            x_shape,         # x_norm
            chunk_shape,     # gate_msa
            chunk_shape,     # shift_mlp
            chunk_shape,     # scale_mlp
            chunk_shape,     # gate_mlp
            x_shape,         # x_norm2
            chunk_shape,     # gate_msa2
        ]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"dim": self.dim, "eps": self.eps})
        return config


# =====================================================================
# AdaLayerNormContinuous (2-way, scale + shift, no gate)
# =====================================================================


@register_dl_technique("dl_techniques.layers.transformers.sd3_adaln")
class AdaLayerNormContinuous(keras.layers.Layer):
    """SD3 2-way AdaLN modulation: scale and shift, no gate.

    Used by the SD3 final / context-final blocks: normalizes ``x`` (affine
    free) and applies a single conditioned scale + shift. Returns one fully
    modulated tensor (no gate, no residual decomposition).

    The modulation ``Dense(2*dim)`` is zero-initialized so at init the output
    equals a plain no-affine LayerNorm of ``x``.

    :param dim: Model / embedding dimensionality.
    :type dim: int
    :param eps: Epsilon for the affine-free LayerNorm. Defaults to ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` is not a positive integer or ``eps <= 0``.

    Input/Output:
        ``call([x, cond])`` with ``x: (B, N, dim)`` and ``cond: (B, dim)``.
        Returns a single tensor ``(B, N, dim)``.
    """

    _N_CHUNKS = 2

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = float(eps)

        self.silu = keras.layers.Activation("silu", name="silu")
        self.linear = keras.layers.Dense(
            self._N_CHUNKS * dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="linear",
        )
        self.norm = keras.layers.LayerNormalization(
            epsilon=self.eps, center=False, scale=False, name="norm"
        )

        logger.debug(
            f"Initialized AdaLayerNormContinuous(dim={dim}, eps={self.eps})"
        )

    def build(self, input_shape: Any) -> None:
        if self.built:
            return

        x_shape, cond_shape = _unpack_pair_shape(input_shape)
        if x_shape[-1] != self.dim:
            raise ValueError(
                f"x last dim must be dim={self.dim}, got x_shape={x_shape}"
            )
        if cond_shape[-1] != self.dim:
            raise ValueError(
                f"cond last dim must be dim={self.dim}, got cond_shape={cond_shape}"
            )
        self.silu.build(cond_shape)
        self.linear.build(cond_shape)
        self.norm.build(x_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x, cond = inputs[0], inputs[1]
        emb = self.linear(self.silu(cond))
        scale, shift = keras.ops.split(emb, self._N_CHUNKS, axis=-1)
        return modulate(self.norm(x), shift, scale)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        x_shape, _ = _unpack_pair_shape(input_shape)
        return tuple(x_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"dim": self.dim, "eps": self.eps})
        return config

# ---------------------------------------------------------------------
