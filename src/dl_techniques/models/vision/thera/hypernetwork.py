"""
THERA hypernetwork and per-pixel implicit-field decoder, as a Keras layer.

:class:`TheraHypernetwork` turns a frozen backbone encoding into a per-pixel,
spatially-varying SIREN-style heat field and evaluates it at arbitrary query
coordinates. This is what makes THERA arbitrary-scale and aliasing-free: a 1x1
convolution emits per-pixel field parameters from a nearest-sampled encoding,
and the field is evaluated at each query's coordinate relative to its source
pixel, rather than at a fixed output grid. The layer owns the 1x1 ``out_conv``
and a :class:`HeatField`; the backbone and tail live in the ``Thera`` model.

The inference path (`decode`, `get_phi_at_coords`) is pure `keras.ops`. Only
:meth:`decode_with_jac`, the training-only TV-loss path, uses a raw
`tf.GradientTape.batch_jacobian` call, since `keras.ops` has no backend-agnostic
batched-Jacobian primitive; it is not used on the inference forward path.

References:
    - Becker et al. Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
      Neural Heat Fields.
"""

import keras
import tensorflow as tf
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.spatial_layer import coordinate_grid, interpolate_grid
from dl_techniques.layers.thera_heat_field import HeatField, DEFAULT_K_INIT
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.hypernetwork")
class TheraHypernetwork(keras.layers.Layer):
    """THERA hypernetwork and per-pixel implicit-field decoder.

    Given a backbone ``encoding`` ``(B, Hs, Ws, C)``, query coordinates
    ``coords`` ``(B, Hq, Wq, 2)`` (channel order ``[h, w]``, pixel-center
    convention), and a heat time ``t`` (broadcastable to ``(B, 1)``), this
    layer samples the encoding at the query coordinates, turns the sample into
    per-pixel heat-field parameters, and evaluates the field at the query's
    coordinate relative to its nearest source pixel. Output shape tracks the
    query coordinate grid: ``(B, Hq, Wq, out_dim)``.

    Architecture:

    .. code-block:: text

        encoding [B, Hs, Ws, C]        coords [B, Hq, Wq, 2]
              │                              │
        ┌─────▼─────┐                        │
        │ sample     │  nearest at coords     │
        │ (order=0)  │                        │
        └─────┬─────┘                        │
              ▼                              │
        ┌─────────────┐                      │
        │ out_conv 1x1 │                      │
        └─────┬───────┘                      │
              ▼                              │
        phi [B, Hq, Wq, output_size]          │
              │ split                        │
              ▼                              ▼
        phi_phase, phi_kernel      rel = coords - nearest(coords), * [Hs, Ws]
              │                              │
              └──────────────┬───────────────┘
                              ▼
                    ┌───────────────────┐
                    │ HeatField          │  (rel, phi_phase, phi_kernel, t)
                    └─────────┬─────────┘
                               ▼
                    out [B, Hq, Wq, out_dim]

    :param hidden_dim: Heat-field hidden width, the number of frequency
        components. Must be positive.
    :type hidden_dim: int
    :param out_dim: Output channel count, e.g. 3 for an RGB residual.
    :type out_dim: int
    :param w0: SIREN frequency multiplier forwarded to :class:`HeatField`.
    :type w0: float
    :param c: Forwarded to :class:`HeatField`, which only stores it. `out_conv`
        uses Keras' default `glorot_uniform` init rather than a `c`-derived one.
    :type c: float
    :param k_init: Initial value of the heat-field scalar `k`.
    :type k_init: float
    :param components_init_scale: Frequency-disk scale forwarded to
        :class:`HeatField`'s `components` init.
    :type components_init_scale: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :Example:

    >>> hyper = TheraHypernetwork(hidden_dim=32, out_dim=3)
    >>> encoding = keras.random.normal((2, 8, 8, 16))
    >>> coords = keras.ops.broadcast_to(coordinate_grid(12)[None], (2, 12, 12, 2))
    >>> t = keras.ops.ones((2, 1))
    >>> out = hyper.decode(encoding, coords, t)   # (2, 12, 12, 3)
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 3,
        w0: float = 1.0,
        c: float = 6.0,
        k_init: float = DEFAULT_K_INIT,
        components_init_scale: float = 16.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.w0 = float(w0)
        self.c = float(c)
        self.k_init = float(k_init)
        self.components_init_scale = float(components_init_scale)

        # phi layout (D-008): phase (hidden,) then kernel (hidden, out), flattened.
        self.output_size = self.hidden_dim + self.hidden_dim * self.out_dim

        # Owned sublayers (built explicitly in ``build``). Stored as plain
        # attributes -- NO nested lists (LESSONS iter-1: nested layer lists break
        # `.keras` reload).
        self.out_conv = keras.layers.Conv2D(
            filters=self.output_size,
            kernel_size=1,
            use_bias=True,
            name="out_conv",
        )
        self.heat_field = HeatField(
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            w0=self.w0,
            c=self.c,
            k_init=self.k_init,
            components_init_scale=self.components_init_scale,
            name="heat_field",
        )

    def build(self, input_shape: Any) -> None:
        """Build both sublayers explicitly before `super().build()`.

        `input_shape` is the bare encoding shape `(B, Hs, Ws, C)` when `decode`
        builds the layer, or a `[encoding, coords, t]` shape list when invoked
        functionally via `call((encoding, coords, t))`.

        :param input_shape: Encoding shape, or a 3-element list of input shapes.
        """
        enc_shape = self._encoding_shape(input_shape)
        batch = enc_shape[0]
        channels = enc_shape[-1]

        # The out_conv sees the NEAREST-sampled encoding, which has the SAME
        # channel count C but the QUERY spatial size (unknown at build -> None).
        enc_at_shape = (batch, None, None, channels)
        if not self.out_conv.built:
            self.out_conv.build(enc_at_shape)

        # The HeatField consumes (rel_coords, phi_phase, phi_kernel, t). Build it
        # with the rel_coords shape (..., 2) at the query spatial size (None).
        rel_shape = (batch, None, None, 2)
        phase_shape = (batch, None, None, self.hidden_dim)
        kernel_shape = (batch, None, None, self.hidden_dim, self.out_dim)
        t_shape = (batch, 1)
        if not self.heat_field.built:
            self.heat_field.build([rel_shape, phase_shape, kernel_shape, t_shape])

        super().build(input_shape)

    @staticmethod
    def _encoding_shape(input_shape: Any) -> Tuple[Optional[int], ...]:
        """Extract the encoding feature-map shape from a single- or multi-input arg.

        When built via :meth:`decode`, ``input_shape`` is the bare encoding shape
        ``(B, Hs, Ws, C)``. When built functionally via ``call((encoding, coords,
        t))``, Keras passes a list/tuple of three per-input shapes; the first is
        the encoding's. Detect the latter by a nested first element.
        """
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (list, tuple))
        ):
            return tuple(input_shape[0])
        return tuple(input_shape)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Infer the decode output shape.

        The query spatial dims track the `coords` grid, not the source
        `encoding`, and only stored config is used, so this works before the
        layer is built.

        :param input_shape: Either the multi-input `[enc_shape, coords_shape,
            t_shape]` list, where `coords_shape` supplies the query dims, or a
            bare encoding shape `(B, Hs, Ws, C)` where the query dims are
            unknown.
        :return: `(B, Hq, Wq, out_dim)` for the multi-input case, else
            `(B, None, None, out_dim)`.
        :rtype: Tuple[Optional[int], ...]
        """
        # Multi-input: [enc_shape, coords_shape, t_shape]; coords are input #1.
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) >= 2
            and isinstance(input_shape[1], (list, tuple))
        ):
            coords_shape = input_shape[1]  # (B, Hq, Wq, 2)
            return (coords_shape[0], coords_shape[1], coords_shape[2], self.out_dim)
        # Bare encoding shape: query spatial dims unknown without coords.
        batch = input_shape[0] if isinstance(input_shape, (list, tuple)) else None
        return (batch, None, None, self.out_dim)

    # -----------------------------------------------------------------

    def get_phi_at_coords(
        self,
        encoding: Any,
        coords: Any,
    ) -> Tuple[Any, Any]:
        """Sample the encoding at the query coords and emit per-pixel field params.

        :param encoding: Backbone feature map, shape `(B, Hs, Ws, C)`.
        :param coords: Query coordinates, shape `(B, Hq, Wq, 2)`, channel order
            `[h, w]`, pixel-center convention.
        :return: `(phi_phase, phi_kernel)`, shapes `(B, Hq, Wq, hidden_dim)` and
            `(B, Hq, Wq, hidden_dim, out_dim)`.
        :rtype: Tuple[Any, Any]
        """
        # NEAREST (order=0) sample of the encoding at the target coords.
        enc_at = interpolate_grid(coords, encoding, order=0)  # (B, Hq, Wq, C)
        phi = self.out_conv(enc_at)  # (B, Hq, Wq, output_size)

        # Phase first, then the flattened kernel slab reshaped.
        phi_phase = phi[..., : self.hidden_dim]
        phi_kernel_flat = phi[..., self.hidden_dim:]

        # ops.shape returns a mix of static ints and dynamic scalar tensors, so
        # the reshape target is a plain Python tuple, not an ops.concatenate call.
        # Leading dims are (B, Hq, Wq).
        lead = ops.shape(phi_kernel_flat)[:-1]
        phi_kernel = ops.reshape(
            phi_kernel_flat, (*lead, self.hidden_dim, self.out_dim)
        )
        return phi_phase, phi_kernel

    # -----------------------------------------------------------------

    def _source_grid(self, encoding: Any) -> Any:
        """Build a pixel-center source coordinate grid at the encoding resolution.

        Matches `coordinate_grid`'s centers/`ij` convention: channel order
        `[h, w]`, `linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)`, `indexing='ij'`.
        Uses static Python ints when the spatial shape is known, otherwise a
        dynamic `keras.ops.linspace` build.

        :param encoding: `(B, Hs, Ws, C)` feature map.
        :return: A `(Hs, Ws, 2)` float32 grid, un-batched.
        :rtype: Any
        """
        hs_static = encoding.shape[1]
        ws_static = encoding.shape[2]

        if hs_static is not None and ws_static is not None:
            # Static fast path: reuse the verified numpy coordinate_grid.
            grid = coordinate_grid((int(hs_static), int(ws_static)))
            return ops.convert_to_tensor(grid, dtype="float32")

        # Dynamic path: build with keras.ops using the pixel-center formula,
        # indexing='ij' so mesh_h varies along axis 0 and mesh_w along axis 1.
        dyn = ops.shape(encoding)
        hs = ops.cast(dyn[1], "float32")
        ws = ops.cast(dyn[2], "float32")
        off_h = 1.0 / (2.0 * hs)
        off_w = 1.0 / (2.0 * ws)
        space_h = ops.linspace(-0.5 + off_h, 0.5 - off_h, dyn[1])
        space_w = ops.linspace(-0.5 + off_w, 0.5 - off_w, dyn[2])
        mesh_h, mesh_w = ops.meshgrid(space_h, space_w, indexing="ij")
        # Stack [h, w] on the last axis to match coordinate_grid.
        return ops.stack([mesh_h, mesh_w], axis=-1)

    # -----------------------------------------------------------------

    def _compute_rel_and_phi(
        self,
        encoding: Any,
        coords: Any,
    ) -> Tuple[Any, Any, Any]:
        """Compute the reusable `(rel_coords, phi_phase, phi_kernel)` triple.

        Shared by :meth:`decode` and :meth:`decode_with_jac` so both use the
        same rel-coordinate and per-pixel-parameter computation. `rel` carries
        the coordinate gradient through its direct `coords` term; the nearest
        term is piecewise-constant with zero gradient almost everywhere.

        :param encoding: Backbone feature map, shape `(B, Hs, Ws, C)`.
        :param coords: Query coordinates, shape `(B, Hq, Wq, 2)`.
        :return: `(rel, phi_phase, phi_kernel)`: `rel` `(B, Hq, Wq, 2)` scaled
            to source pixel units, `phi_phase` `(B, Hq, Wq, hidden)`, `phi_kernel`
            `(B, Hq, Wq, hidden, out)`.
        :rtype: Tuple[Any, Any, Any]
        """
        phi_phase, phi_kernel = self.get_phi_at_coords(encoding, coords)

        # Source pixel-center grid at the encoding resolution.
        source_grid = self._source_grid(encoding)

        # Broadcast target mixes static ints with a dynamic batch scalar, so
        # it is a plain Python tuple, not an ops.concatenate call.
        batch = ops.shape(encoding)[0]
        grid_shape = ops.shape(source_grid)
        target = (batch, *grid_shape)
        source_coords = ops.broadcast_to(source_grid[None, ...], target)

        # Nearest source-pixel coordinate for each query coordinate.
        interp_coords = interpolate_grid(coords, source_coords, order=0)

        # this arithmetic stays in float32 and only the result casts down to
        # compute_dtype -- under mixed_float16, subtracting float16 coords here
        # raised. See decisions.md D-046 (plan-2026-08-19T163559-499b6f0e).
        rel = ops.cast(coords, "float32") - ops.cast(interp_coords, "float32")
        hs_f = ops.cast(ops.shape(encoding)[1], "float32")
        ws_f = ops.cast(ops.shape(encoding)[2], "float32")
        rel_h = rel[..., 0] * hs_f
        rel_w = rel[..., 1] * ws_f
        rel = ops.stack([rel_h, rel_w], axis=-1)
        rel = ops.cast(rel, self.compute_dtype)
        return rel, phi_phase, phi_kernel

    def decode(
        self,
        encoding: Any,
        coords: Any,
        t: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Decode the heat field at the query coordinates.

        :param encoding: Backbone feature map, shape `(B, Hs, Ws, C)`.
        :param coords: Query coordinates, shape `(B, Hq, Wq, 2)`, channel order
            `[h, w]`, pixel-center convention.
        :param t: Heat-diffusion time, broadcastable to `(B, 1)`.
        :param training: Forwarded to the heat field.
        :return: Super-resolved field values, shape `(B, Hq, Wq, out_dim)`.
        :rtype: Any
        """
        rel, phi_phase, phi_kernel = self._compute_rel_and_phi(encoding, coords)
        return self.heat_field(rel, phi_phase, phi_kernel, t, training=training)

    # -----------------------------------------------------------------

    # DECISION plan_2026-06-11_f662207d/D-010: exact analytic Jacobian via a
    # flattened per-pixel batch_jacobian, not finite differences; the heat field
    # is pointwise so off-diagonal pixel-cross terms are exactly zero. See decisions.md.
    def decode_with_jac(
        self,
        encoding: Any,
        coords: Any,
        t: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Decode the field and its exact spatial Jacobian `d(field)/d(rel)` at t=0.

        The forward output is at the real `t`, while the Jacobian is evaluated
        at `t=0` (envelope 1) and consumed by the TV penalty, differentiating
        through to the weights via the trainer's outer tape.

        :param encoding: Backbone feature map, shape `(B, Hs, Ws, C)`.
        :param coords: Query coordinates, shape `(B, Hq, Wq, 2)`.
        :param t: Heat-diffusion time, broadcastable to `(B, 1)`.
        :param training: Forwarded to the heat field.
        :return: `(out, jac)`: `out` `(B, Hq, Wq, out_dim)` at the real `t`,
            `jac` `(B, Hq, Wq, out_dim, 2)` the per-pixel spatial Jacobian at `t=0`.
        :rtype: Tuple[Any, Any]
        """
        rel, phi_phase, phi_kernel = self._compute_rel_and_phi(encoding, coords)

        # Forward output at the real t.
        out = self.heat_field(rel, phi_phase, phi_kernel, t, training=training)

        # Flatten leading (B, Hq, Wq) dims to one pixel axis N: the field is
        # pointwise, so a per-pixel batch_jacobian gives the exact block-diagonal Jacobian.
        out_dim = self.out_dim
        hidden = self.hidden_dim
        # DECISION plan-2026-08-19T163559-499b6f0e/D-083: keras.ops for the shape
        # arithmetic, not raw tf; only GradientTape stays raw since keras.ops has
        # no batch_jacobian. See decisions.md.
        flat = ops.shape(rel)
        n = flat[0] * flat[1] * flat[2]

        rel_flat = ops.reshape(rel, (n, 2))
        phase_flat = ops.reshape(phi_phase, (n, hidden))
        kernel_flat = ops.reshape(phi_kernel, (n, hidden, out_dim))
        # t=0 for every pixel, so the heat envelope is 1.
        t_zero = ops.zeros((n, 1), dtype=rel_flat.dtype)

        # persistent=True: batch_jacobian with experimental_use_pfor=False in
        # eager mode unrolls a loop that re-reads the tape across calls.
        with tf.GradientTape(persistent=True) as jac_tape:
            jac_tape.watch(rel_flat)
            out0_flat = self.heat_field(
                rel_flat, phase_flat, kernel_flat, t_zero, training=training
            )
        # experimental_use_pfor=False composes with the trainer's outer tape; see D-010.
        jac_flat = jac_tape.batch_jacobian(
            out0_flat, rel_flat, experimental_use_pfor=False
        )
        del jac_tape

        out_shape = tuple(flat[:3]) + (out_dim, 2)
        jac = ops.reshape(jac_flat, out_shape)
        return out, jac

    # -----------------------------------------------------------------

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Functional entry point: `inputs = (encoding, coords, t)`.

        Keras layers prefer a single `inputs` argument for functional and
        serialization use, so the three decode tensors are packed into a tuple
        and unpacked here. `Thera` may call :meth:`decode` directly instead.

        :param inputs: A 3-tuple `(encoding, coords, t)`.
        :param training: Forwarded to :meth:`decode`.
        :return: Field values, shape `(B, Hq, Wq, out_dim)`.
        :rtype: Any
        """
        encoding, coords, t = inputs
        return self.decode(encoding, coords, t, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "w0": self.w0,
            "c": self.c,
            "k_init": self.k_init,
            "components_init_scale": self.components_init_scale,
        })
        return config

# ---------------------------------------------------------------------
