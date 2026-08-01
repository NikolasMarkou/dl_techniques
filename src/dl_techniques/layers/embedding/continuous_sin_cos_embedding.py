"""
Generates continuous, multi-dimensional positional embeddings using sinusoids.

This layer implements a technique for encoding continuous spatial coordinates
(e.g., 2D or 3D points) into a high-dimensional vector space. It serves as
a continuous and multi-dimensional generalization of the fixed sinusoidal
positional encodings used in the original Transformer architecture, making it
suitable for tasks involving geometric data like point clouds, images, or
physical simulations.

Architecture:
    The core principle is to map each scalar coordinate value into a vector
    of sine and cosine values at different frequencies. This creates a rich,
    smooth, and periodic representation that allows a neural network to
    easily reason about relative positions and distances.

    The architecture partitions the total embedding dimension (`dim`) among
    the number of input coordinate dimensions (`ndim`). For each coordinate
    `p_k` (e.g., the x, y, or z value of a point), the layer performs the
    following steps:
    1.  It scales the coordinate by a set of fixed, non-learnable
        frequencies that form a geometric progression.
    2.  It applies both the `sin` and `cos` functions to these scaled values.
    3.  The resulting sine and cosine values for all frequencies are
        concatenated to form an embedding for that single coordinate.

    The final output is the concatenation of the embeddings from all input
    coordinate dimensions, resulting in a single vector that encodes the
    full multi-dimensional position.

Foundational Mathematics:
    This method is a direct extension of the positional encoding formula
    from "Attention Is All You Need". For a single continuous coordinate
    `p`, its embedding `E(p)` is a vector where each pair of elements is
    defined by:

        E(p)_{2i}   = sin(p * omega_i)
        E(p)_{2i+1} = cos(p * omega_i)

    The frequencies `omega_i` are fixed and decrease exponentially, forming
    a geometric progression:

        omega_i = 1 / (max_wavelength^(2i / d'))

    where `d'` is the embedding dimension allocated per coordinate. This
    formulation has several key properties:
    -   **Continuity:** The embedding function is smooth, so nearby points in
        coordinate space are mapped to nearby points in the embedding space.
    -   **Relative Positioning:** For any displacement `delta`, the embedding
        `E(p + delta)` can be represented as a linear transformation of
        `E(p)`, making it easy for models like transformers to learn relative
        positional relationships.
    -   **Multi-Frequency Representation:** The use of a spectrum of
        frequencies, from low (`max_wavelength`) to high, allows the model
        to capture both coarse, global positional information and
        fine-grained, local details simultaneously.

References:
    - The core technique is inspired by the original Transformer positional
      encodings:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - This style of continuous coordinate embedding is a key component in
      Neural Radiance Fields (NeRF) for representing 3D coordinates:
      Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural
      Radiance Fields for View Synthesis".
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContinuousSinCosEmbed(keras.layers.Layer):
    """Continuous sinusoidal coordinate embedding for multi-dimensional positions.

    Embeds continuous coordinates (e.g. 2D/3D positions) into a high-dimensional
    vector space using alternating sine and cosine functions at geometrically
    spaced frequencies. For each coordinate ``p_k``, the embedding computes
    ``E(p_k)_{2i} = sin(p_k * omega_i)`` and
    ``E(p_k)_{2i+1} = cos(p_k * omega_i)`` where
    ``omega_i = 1 / (max_wavelength^(2i/d'))``. The per-coordinate embeddings
    are concatenated to produce the final output of dimension ``dim``. This
    creates smooth, continuous representations that preserve spatial
    relationships and enable learning of relative positions.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input coords (..., ndim)        │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  For each coord dimension k:     │
        │    freqs = p_k * omega           │
        │    emb_k = [sin(freqs),          │
        │             cos(freqs)]          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Concatenate [emb_1,...,emb_ndim]│
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Pad if dim % ndim != 0          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output (..., dim)               │
        └──────────────────────────────────┘

    :param dim: Dimensionality of the embedded output coordinates. Must be
        positive and should be at least ``2 * ndim``.
    :type dim: int
    :param ndim: Number of dimensions of the input coordinate space (e.g., 2
        for 2D, 3 for 3D). Must be positive.
    :type ndim: int
    :param max_wavelength: Maximum wavelength for the sinusoidal embedding.
        Controls the frequency range. Higher values create more gradual spatial
        variations. Defaults to ``10000.0``.
    :type max_wavelength: float
    :param assert_positive: Whether to check that all input coordinates are
        positive. Useful for normalized coordinate systems. Defaults to
        ``True``.
    :type assert_positive: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``dim`` is too small for the given ``ndim``.
    :raises ValueError: If input parameters are invalid.
    :raises ValueError: If input shape is invalid.

    .. note::
       **Accuracy limitation at reduced-precision policies, and at float64.**
       The layer RUNS at ``float32``, ``mixed_float16``, ``float64`` and
       ``mixed_bfloat16``, but running is not the same as being correct.

       *Large positions under fp16/bf16.* Measured on CPU against a float64 oracle
       with ``dim=64, ndim=3, max_wavelength=10000`` and coordinates drawn from
       ``[0, 2000)`` -- the output, a ``[-1, 1]``-ranged quantity, is wrong by up to
       **0.47** at ``mixed_float16`` and up to **1.99** at ``mixed_bfloat16``
       (8-seed sweep: 0.45-0.48 and 1.96-2.00). At bfloat16 that is nearly the full
       output range, i.e. the embedding carries essentially no usable position
       information there. This is NOT fixable inside ``call()``: Keras narrows
       ``coords`` at the AUTOCAST BOUNDARY, before ``call()`` runs (measured: a
       coordinate of ``1934.448`` arrives as ``1934.0``), and the ``omega`` weight
       READ is autocast to the compute dtype for the same reason. The widened working
       dtype in ``call()`` therefore starts from operands that have already lost the
       precision. The error is driven by the MAGNITUDE of the coordinates, not by
       the layer: the same configuration over ``[0, 64)`` is roughly 40x more
       accurate at both policies (those per-policy figures live with the tolerance
       table of the test that uses them, not restated here).

       *Remedies, as measured -- do not assume, they do not all work.* Passing
       ``coords`` as float32 does NOT help (identical 0.47 at fp16: Keras narrows
       the input at the layer boundary regardless of what the caller hands it).
       Constructing the layer with ``autocast=False`` and float32 coordinates helps
       but does not fix it -- 0.47 -> 0.16 (fp16) and 1.99 -> 0.56 (bf16) at the
       regime above -- because the ``omega`` read is still narrowed. A real fp16
       long-sequence consumer should keep this layer OUT of the reduced-precision
       region (run it under a float32 policy and cast the result), which is
       UNTESTED here.

       *float64 frequency ceiling.* ``build()`` computes the frequency table at
       float32 at every policy, so a ``float64``-policy caller gets float32-accurate
       frequencies: max relative deviation from the float64 frequencies is
       ``2.79e-07`` (at ``dim=64, ndim=3, max_wavelength=10000``; ``3.02e-07`` at
       ``max_wavelength=1e5``). Widening it was measured and DELIBERATELY REJECTED:
       it moves the float32 output at 10 of 10 configurations swept, and float32
       bit-identity is a hard invariant for this layer.
    """

    def __init__(
            self,
            dim: int,
            ndim: int,
            max_wavelength: float = 10000.0,
            assert_positive: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")
        if max_wavelength <= 0:
            raise ValueError(f"max_wavelength must be positive, got {max_wavelength}")

        # Store ALL configuration parameters
        self.dim = dim
        self.ndim = ndim
        self.max_wavelength = max_wavelength
        self.assert_positive = assert_positive

        # Calculate padding needed if dim is not cleanly divisible by ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim

        # Calculate effective dimensions for wave generation
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        if effective_dim_per_wave <= 0:
            raise ValueError(f"dim ({dim}) too small for ndim ({ndim}). "
                             f"Need at least {ndim * 2} dimensions.")

        # Store effective dimension for weight creation
        self.effective_dim_per_wave = effective_dim_per_wave

        # Initialize weight attributes - created in build()
        self.omega = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's fixed frequency weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        if input_shape[-1] != self.ndim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match ndim ({self.ndim})")

        # Create frequency weights
        arange_vals = np.arange(0, self.effective_dim_per_wave, 2, dtype=np.float32)
        omega_vals = 1.0 / (self.max_wavelength ** (arange_vals / self.effective_dim_per_wave))

        # Create layer's own weights
        self.omega = self.add_weight(
            name="omega",
            shape=omega_vals.shape,
            initializer="zeros",
            trainable=False,
        )

        # Set the omega values
        self.omega.assign(omega_vals)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            coords: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Embed continuous coordinates using sinusoidal functions.

        :param coords: Input tensor of coordinates with shape ``(..., ndim)``.
        :type coords: keras.KerasTensor
        :param training: Whether in training mode (unused).
        :type training: Optional[bool]
        :return: Embedded coordinates with shape ``(..., dim)`` using
            alternating sine and cosine functions at different frequencies.
        :rtype: keras.KerasTensor
        """
        # NOTE: `assert_positive` is retained as a config-compatible flag but no
        # longer triggers a runtime check. The previous implementation called
        # `ops.convert_to_numpy(ops.min(coords))` here, which is an eager host
        # materialization that breaks `@tf.function` / graph tracing. Removed for
        # graph compatibility (DECISION plan_2026-06-15_9dbb87c1/D-001).

        # DECISION plan-2026-07-31T210633-b63a35aa/D-002
        # Compute the sinusoids in a NEVER-NARROWING working dtype, and cast the
        # frequency table to that same dtype.
        #
        # Do NOT write `ops.cast(coords, "float32")` here. A hard dtype literal is
        # measured against `self.omega`, which Keras AUTOCASTS to the active compute
        # dtype on read -- so the multiply below had mismatched operands and raised
        # `InvalidArgumentError ... cannot compute Mul as input #1 was expected to be
        # a float tensor but is a half/double/bfloat16 tensor [Op:Mul]` under
        # mixed_float16, float64 AND mixed_bfloat16, i.e. this whole layer (and every
        # consumer: the embedding factory, TextEncoder, TextDecoder, SupernodePooling)
        # was dead at 3 of the 4 standard policies.
        #
        # Do NOT "simplify" this to `ops.cast(coords, self.compute_dtype)` either:
        # that NARROWS the sinusoid math to fp16/bf16 and measured ~2.4x worse against
        # a float64 reference at large positions. (Those two prototype figures used to
        # be quoted here; they were re-derived against the SHIPPED path in step 7 and
        # did NOT reproduce -- the prototype never narrowed `coords` at the autocast
        # boundary. The measured shipped numbers now have exactly one home, the class
        # docstring's `.. note::`, and are pinned by
        # `test_fp16_large_position_error_bound_is_pinned`.) The conditional below IS
        # the never-narrow rule, stated rather than inherited from `variable_dtype`.
        #
        # The final `ops.cast(..., self.compute_dtype)` at the return is load-bearing:
        # without it this returns float32 under mixed_float16, a silently wrong output
        # dtype that every downstream `add`/`concatenate` would have to absorb.
        #
        # NOTE this cannot recover fp16/bf16 accuracy at large positions: Keras has
        # already narrowed `coords` at the autocast boundary BEFORE `call()` runs.
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"
        coords = ops.cast(coords, work_dtype)

        # Expand coordinates for frequency multiplication
        # coords: (..., ndim) -> coords_expanded: (..., ndim, 1)
        coords_expanded = ops.expand_dims(coords, axis=-1)

        # omega: (freq_dim,) -> omega_expanded: (1, ..., 1, freq_dim)
        omega_shape = [1] * (len(coords.shape) - 1) + [self.omega.shape[0]]
        omega_expanded = ops.reshape(ops.cast(self.omega, work_dtype), omega_shape)

        # Compute frequencies: (..., ndim, freq_dim)
        freqs = coords_expanded * omega_expanded

        # Apply sin and cos
        sin_vals = ops.sin(freqs)
        cos_vals = ops.cos(freqs)

        # Concatenate sin and cos: (..., ndim, 2*freq_dim)
        emb = ops.concatenate([sin_vals, cos_vals], axis=-1)

        # Flatten the (ndim, 2*freq_dim) trailing axes into one. Use the dynamic
        # leading dims from ops.shape (runtime values, never None) plus a single
        # inferred -1 so this is graph-safe for any rank / dynamic batch dim.
        emb = ops.reshape(emb, (*ops.shape(coords)[:-1], -1))

        # Add padding if necessary. Build the pad shape from dynamic leading
        # dims (ops.shape -> runtime values, never None) + a static last dim,
        # so ops.zeros is graph-safe under symbolic tracing.
        if self.padding > 0:
            padding = ops.zeros(
                (*ops.shape(emb)[:-1], self.padding), dtype=emb.dtype
            )
            emb = ops.concatenate([emb, padding], axis=-1)

        # Back to the caller's compute dtype (see D-002 above -- not optional).
        return ops.cast(emb, self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension changed to ``dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.dim])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ndim": self.ndim,
            "max_wavelength": self.max_wavelength,
            "assert_positive": self.assert_positive,
        })
        return config

# ---------------------------------------------------------------------
