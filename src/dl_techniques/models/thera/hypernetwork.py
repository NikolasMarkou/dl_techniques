# DECISION plan_2026-06-11_f662207d/D-008
# This layer ports THERA's hypernetwork + apply_decoder (reference `model/hyper.py`
# + `model/thera.py`). Several choices are deliberate and must NOT be "simplified":
#
#  1. phi SPLIT CONVENTION (phase first, kernel reshaped second) is OUR choice. THERA
#     trains the hypernetwork from a JAX param-tree whose flat layout we are NOT
#     porting (INV-6: no param-tree, no .pkl weight port). Under train-from-scratch
#     the only requirement is that the split is INTERNALLY CONSISTENT between
#     `out_conv` output ordering and the `HeatField` inputs. Do not "match the JAX
#     layout" -- there is no weight port to match.
#  2. PER-PIXEL field via the step-3 HeatField BATCHED EINSUM, NOT JAX vmap over query
#     pixels / a nested param tree (INV-5). Do not reintroduce vmap.
#  3. SOURCE GRID is built pixel-CENTER to match step-2 `make_grid` EXACTLY (channel
#     order [h, w], `linspace(-0.5+1/(2n), 0.5-1/(2n), n)`, indexing='ij'). A naive
#     `linspace(-0.5, 0.5, n)` endpoint grid would silently mis-register rel_coords.
#  4. SAMPLING is order=0 (NEAREST), matching THERA. Do NOT "upgrade" to bilinear:
#     the step-9 TV-loss coordinate Jacobian flows through the DIRECT `coords` term of
#     `rel = coords - nearest(coords)` (the nearest term is piecewise-constant,
#     zero-grad a.e.), into the heat field -- NOT through the sampler.
#
# See decisions.md D-008.
"""THERA hypernetwork + implicit-field decoder as a Keras layer.

This module ports THERA's per-pixel implicit-field decoder. A frozen encoding
(the backbone feature map) is turned, at each *query* coordinate, into a
spatially-varying SIREN-style heat field that is then evaluated to produce the
super-resolved pixel value.

Pipeline (reference ``model/hyper.py`` ``get_params_at_coords`` +
``model/thera.py`` ``apply_decoder``)::

    # hypernetwork: 1x1 conv emits a flat per-pixel parameter vector phi
    enc_at = interpolate_grid(coords, encoding, order=0)   # NEAREST sample
    phi    = out_conv(enc_at)                              # (B, Hq, Wq, output_size)
    phi_phase, phi_kernel = split_and_reshape(phi)         # per-pixel field params

    # decoder: local relative coordinate of the query w.r.t. its source pixel
    source_grid   = make_grid(encoding H, W)               # pixel-center grid
    interp_coords = interpolate_grid(coords, source_grid, order=0)  # NEAREST
    rel           = coords - interp_coords
    rel[..., 0]  *= Hs ; rel[..., 1] *= Ws                 # scale to pixel units
    out           = heat_field(rel, phi_phase, phi_kernel, t)

The :class:`TheraHypernetwork` OWNS the 1x1 ``out_conv`` and a :class:`HeatField`
(which owns ``components`` + ``k``). It does NOT own the backbone or the tail --
the step-8 ``Thera`` model holds those and passes ``encoding`` in.

phi layout (train-from-scratch -- OUR consistent convention, see D-008)::

    output_size = hidden_dim + hidden_dim * out_dim
    phi_phase   = phi[..., :hidden_dim]                              # (B,Hq,Wq,hidden)
    phi_kernel  = reshape(phi[..., hidden_dim:], (..., hidden, out)) # (B,Hq,Wq,hidden,out)

Differentiability (step-9 TV loss)
----------------------------------
``interpolate_grid`` is ``order=0`` (NEAREST), matching THERA. The coordinate
Jacobian for the aliasing TV loss flows through ``rel = coords - nearest(coords)``:
the ``nearest`` term is piecewise-constant (zero gradient almost everywhere) while
the direct ``coords`` term carries gradient 1, propagating into the heat field --
NOT through the sampler. Do not upgrade the sampler to bilinear here.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/hyper.py`` + ``model/thera.py``).
"""

import keras
import tensorflow as tf
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.grid_sample import make_grid, interpolate_grid
from dl_techniques.layers.thera_heat_field import HeatField, DEFAULT_K_INIT
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TheraHypernetwork(keras.layers.Layer):
    """THERA hypernetwork + per-pixel implicit-field decoder.

    Given a backbone ``encoding`` ``(B, Hs, Ws, C)``, query coordinates
    ``coords`` ``(B, Hq, Wq, 2)`` (THERA pixel-center convention, channel order
    ``[h, w]``), and a heat time ``t`` (broadcastable to ``(B, 1)``), this layer:

    1. samples ``encoding`` at the query coords (NEAREST) and applies a 1x1
       ``out_conv`` to emit a flat per-pixel parameter vector ``phi``;
    2. splits/reshapes ``phi`` into the per-pixel heat-field params
       (``phi_phase``, ``phi_kernel``);
    3. computes the query's relative coordinate w.r.t. its nearest source pixel
       (scaled to source pixel units);
    4. evaluates the :class:`HeatField` at those relative coords.

    Output shape tracks the query coordinate grid: ``(B, Hq, Wq, out_dim)``.

    **Intent**: Turn a frozen backbone ``encoding`` into a per-pixel,
    spatially-varying SIREN-style heat field and evaluate it at arbitrary query
    coordinates -- the implicit-field decoder that makes THERA arbitrary-scale
    and aliasing-free. This layer owns the 1x1 ``out_conv`` hypernetwork and the
    :class:`HeatField`; the backbone/tail live in the step-8 ``Thera`` model.

    **Architecture**::

        encoding (B, Hs, Ws, C)
              |  sample at coords (NEAREST, order=0)
              v
        enc_at (B, Hq, Wq, C)
              |  1x1 out_conv
              v
        phi (B, Hq, Wq, output_size)
              |  split -> phi_phase (.., hidden), phi_kernel (.., hidden, out)
              |
        coords (B, Hq, Wq, 2) --+--> rel = coords - nearest(coords)   (* Hs, Ws)
                                |        (source-pixel relative coord)
                                v
        HeatField einsum(rel, phi_phase, phi_kernel, t)
              |
              v
        out (B, Hq, Wq, out_dim)

    Args:
        hidden_dim: Heat-field hidden width ``N`` (number of frequency
            components). Must be positive.
        out_dim: Output channel count (e.g. 3 for an RGB residual). Defaults to 3.
        w0: SIREN frequency multiplier forwarded to the :class:`HeatField`.
            Defaults to 1.0.
        c: SIREN variance constant forwarded to the :class:`HeatField` (stored for
            config fidelity). Defaults to 6.0.
        k_init: Initial value of the heat-field scalar ``k``. Defaults to the
            THERA reference ``sqrt(log 4) / (2*pi^2)`` (same default as
            :class:`HeatField`).
        components_init_scale: Frequency-disk scale forwarded to the
            :class:`HeatField` ``components`` init. Defaults to 16.0.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

    Example:
        >>> hyper = TheraHypernetwork(hidden_dim=32, out_dim=3)
        >>> encoding = keras.random.normal((2, 8, 8, 16))
        >>> coords = keras.ops.broadcast_to(make_grid(12)[None], (2, 12, 12, 2))
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
        # Keras-3 four-strike build-ordering: explicitly build BOTH sublayers with
        # the correct propagated shapes BEFORE super().build(), so a `.keras`
        # reload restores their weights and no unbuilt-sublayer warning fires
        # (LESSONS.md build-order discipline).
        #
        # ``input_shape`` may arrive as the bare encoding feature-map shape
        # (B, Hs, Ws, C) when ``decode`` builds the layer, OR as a list/tuple of
        # three per-input shapes [encoding, coords, t] when invoked functionally
        # via ``call((encoding, coords, t))``. Normalize to the encoding shape.
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
        """Infer the decode output shape ``(B, Hq, Wq, out_dim)`` (guide Pitfall 11).

        The query spatial dims ``Hq, Wq`` track the ``coords`` grid, NOT the
        source ``encoding``. Uses only stored config (``self.out_dim``), never a
        weight shape (Pitfall 12), so it works before the layer is built.

        Args:
            input_shape: Either the multi-input ``[enc_shape, coords_shape,
                t_shape]`` list (functional ``call`` path -- ``coords_shape`` =
                ``(B, Hq, Wq, 2)`` supplies the query dims) or a bare encoding
                shape ``(B, Hs, Ws, C)`` (direct ``decode`` build path, where the
                query dims are unknown -> ``None``).

        Returns:
            ``(B, Hq, Wq, out_dim)`` for the multi-input case, else
            ``(B, None, None, out_dim)``.
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

        Args:
            encoding: Backbone feature map, shape ``(B, Hs, Ws, C)``.
            coords: Query coordinates, shape ``(B, Hq, Wq, 2)`` (channel order
                ``[h, w]``, THERA pixel-center convention).

        Returns:
            A ``(phi_phase, phi_kernel)`` tuple where ``phi_phase`` has shape
            ``(B, Hq, Wq, hidden_dim)`` and ``phi_kernel`` has shape
            ``(B, Hq, Wq, hidden_dim, out_dim)``.
        """
        # NEAREST (order=0) sample of the encoding at the target coords.
        enc_at = interpolate_grid(coords, encoding, order=0)  # (B, Hq, Wq, C)
        phi = self.out_conv(enc_at)  # (B, Hq, Wq, output_size)

        # Split (D-008): phase first, then the flattened kernel slab reshaped.
        phi_phase = phi[..., : self.hidden_dim]
        phi_kernel_flat = phi[..., self.hidden_dim:]

        # Reshape the kernel slab to (..., hidden, out). Use dynamic leading dims
        # so arbitrary query grids work; only the trailing two axes are static.
        # NOTE: keras.ops.shape returns a TUPLE of scalar tensors on the TF
        # backend, which keras.ops.concatenate cannot consume -- use tf.shape
        # (a 1-D tensor) sliced and concatenated with a static int32 tail.
        dyn = tf.shape(phi_kernel_flat)  # 1-D int32 tensor: (B, Hq, Wq, hidden*out)
        tail = tf.constant([self.hidden_dim, self.out_dim], dtype=dyn.dtype)
        kernel_shape = tf.concat([dyn[:-1], tail], axis=0)
        phi_kernel = ops.reshape(phi_kernel_flat, kernel_shape)
        return phi_phase, phi_kernel

    # -----------------------------------------------------------------

    def _source_grid(self, encoding: Any) -> Any:
        """Build a pixel-center source coordinate grid at the encoding resolution.

        Matches the step-2 :func:`make_grid` convention EXACTLY (channel order
        ``[h, w]``, ``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)``, ``indexing='ij'``).
        Prefers static Python ints (training uses a fixed patch size); falls back
        to a dynamic ``keras.ops.linspace`` build when either spatial dim is None.

        Args:
            encoding: ``(B, Hs, Ws, C)`` feature map.

        Returns:
            A ``(Hs, Ws, 2)`` ``float32`` grid (un-batched).
        """
        hs_static = encoding.shape[1]
        ws_static = encoding.shape[2]

        if hs_static is not None and ws_static is not None:
            # Static fast path: reuse the verified numpy make_grid verbatim.
            grid = make_grid((int(hs_static), int(ws_static)))  # (Hs, Ws, 2)
            return ops.convert_to_tensor(grid, dtype="float32")

        # Dynamic path: build with keras.ops using THERA's pixel-center formula.
        dyn = ops.shape(encoding)
        hs = ops.cast(dyn[1], "float32")
        ws = ops.cast(dyn[2], "float32")
        # linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n) per axis.
        off_h = 1.0 / (2.0 * hs)
        off_w = 1.0 / (2.0 * ws)
        space_h = ops.linspace(-0.5 + off_h, 0.5 - off_h, dyn[1])  # (Hs,)
        space_w = ops.linspace(-0.5 + off_w, 0.5 - off_w, dyn[2])  # (Ws,)
        # indexing='ij': mesh_h varies along axis 0, mesh_w along axis 1.
        mesh_h, mesh_w = ops.meshgrid(space_h, space_w, indexing="ij")  # (Hs, Ws)
        # Stack [h, w] on the last axis to match make_grid (index 0 = h, 1 = w).
        return ops.stack([mesh_h, mesh_w], axis=-1)  # (Hs, Ws, 2)

    # -----------------------------------------------------------------

    def _compute_rel_and_phi(
        self,
        encoding: Any,
        coords: Any,
    ) -> Tuple[Any, Any, Any]:
        """Compute the reusable ``(rel_coords, phi_phase, phi_kernel)`` triple.

        Factored out of :meth:`decode` so both the plain forward path and the
        :meth:`decode_with_jac` (step-9 Jacobian-TV) path share EXACTLY the same
        rel-coordinate and per-pixel-parameter computation. ``rel`` here carries
        the coordinate gradient (the TV-loss path flows through the direct
        ``coords`` term of ``rel = coords - nearest(coords)``; the nearest term
        is piecewise-constant, zero-grad a.e. -- D-008).

        Args:
            encoding: Backbone feature map, shape ``(B, Hs, Ws, C)``.
            coords: Query coordinates, shape ``(B, Hq, Wq, 2)``.

        Returns:
            ``(rel, phi_phase, phi_kernel)`` where ``rel`` has shape
            ``(B, Hq, Wq, 2)`` (scaled to source pixel units), ``phi_phase`` has
            shape ``(B, Hq, Wq, hidden)`` and ``phi_kernel`` has shape
            ``(B, Hq, Wq, hidden, out)``.
        """
        phi_phase, phi_kernel = self.get_phi_at_coords(encoding, coords)

        # Source pixel-center grid at the ENCODING resolution -> (Hs, Ws, 2).
        source_grid = self._source_grid(encoding)

        # Tile to (B, Hs, Ws, 2) so interpolate_grid can sample it per batch.
        # tf.shape gives 1-D int32 tensors; assemble the broadcast target as one.
        batch = tf.shape(encoding)[0:1]  # (1,) int32
        grid_shape = tf.shape(source_grid)  # (3,) -> (Hs, Ws, 2)
        target = tf.concat([batch, grid_shape], axis=0)  # (B, Hs, Ws, 2)
        source_coords = ops.broadcast_to(source_grid[None, ...], target)

        # NEAREST: coordinate of the nearest source pixel for each query coord.
        interp_coords = interpolate_grid(coords, source_coords, order=0)

        # Relative coord (query - nearest-source), scaled to source pixel units.
        rel = coords - interp_coords  # (B, Hq, Wq, 2)
        hs_f = ops.cast(ops.shape(encoding)[1], "float32")
        ws_f = ops.cast(ops.shape(encoding)[2], "float32")
        rel_h = rel[..., 0] * hs_f  # (B, Hq, Wq)
        rel_w = rel[..., 1] * ws_f
        rel = ops.stack([rel_h, rel_w], axis=-1)  # (B, Hq, Wq, 2)
        return rel, phi_phase, phi_kernel

    def decode(
        self,
        encoding: Any,
        coords: Any,
        t: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Decode the heat field at the query coordinates (THERA ``apply_decoder``).

        Args:
            encoding: Backbone feature map, shape ``(B, Hs, Ws, C)``.
            coords: Query coordinates, shape ``(B, Hq, Wq, 2)`` (channel order
                ``[h, w]``, THERA pixel-center convention).
            t: Heat-diffusion time, broadcastable to ``(B, 1)``.
            training: Forwarded to the heat field.

        Returns:
            Super-resolved field values, shape ``(B, Hq, Wq, out_dim)``.
        """
        rel, phi_phase, phi_kernel = self._compute_rel_and_phi(encoding, coords)
        return self.heat_field(rel, phi_phase, phi_kernel, t, training=training)

    # -----------------------------------------------------------------

    # DECISION plan_2026-06-11_f662207d/D-010
    # EXACT analytic spatial Jacobian d(field)/d(rel_coords) at t=0, NOT a
    # finite-difference approximation (Q3 forbids finite-difference; see
    # decisions.md D-010). THERA's reference (`model/thera.py apply_decoder`
    # return_jac branch) takes `jacrev(field.apply, argnums=rel_coords)` at t=0;
    # we reproduce it with a nested `tf.GradientTape.batch_jacobian` because:
    #   * the heat field is POINTWISE in rel_coords (query pixel n's output
    #     depends ONLY on rel_coords[n]), so a flattened per-pixel
    #     `batch_jacobian` over the leading (B*Hq*Wq) axis is mathematically
    #     EXACT -- the off-diagonal pixel-cross terms are identically zero, never
    #     computed.
    #   * the Jacobian is evaluated at t=0 (heat envelope == 1, the un-smoothed
    #     "clean" field), matching the reference `zeros_like(t)`.
    # SECOND-ORDER / pfor NOTE: this `jac` is itself differentiated by the
    # trainer's OUTER tape (TV loss -> weight grads, WGAN-GP style). TF's default
    # pfor vectorization of `batch_jacobian` does NOT compose with a second
    # outer tape (the vectorized while-loop yields None weight-grads). We pass
    # `experimental_use_pfor=False` so the inner Jacobian is built with an
    # unrolled per-output loop that the outer tape CAN differentiate -- slower,
    # but second-order-safe and STILL EXACT (NOT finite-difference). Do NOT
    # re-enable pfor here: the STOP-IF #1 nested-tape weight-grad oracle
    # (test_thera_jacobian_tv.py) goes None if you do.
    # This raw-`tf` GradientTape usage is the explicit INV-3 exemption (H3/plan).
    def decode_with_jac(
        self,
        encoding: Any,
        coords: Any,
        t: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Decode the field AND its exact spatial Jacobian ``d(field)/d(rel)`` at t=0.

        Reproduces THERA's ``apply_decoder(..., return_jac=True)`` branch: the
        forward output is taken at the REAL ``t`` while the aliasing Jacobian is
        the per-pixel ``d(field)/d(rel_coords)`` evaluated at ``t=0`` (envelope
        == 1). The Jacobian is consumed by the step-9 TV penalty
        (``mean(abs(jac))``) and differentiates through to the weights via the
        trainer's outer tape (STOP-IF #1).

        Args:
            encoding: Backbone feature map, shape ``(B, Hs, Ws, C)``.
            coords: Query coordinates, shape ``(B, Hq, Wq, 2)``.
            t: Heat-diffusion time, broadcastable to ``(B, 1)``.
            training: Forwarded to the heat field.

        Returns:
            ``(out, jac)`` where ``out`` has shape ``(B, Hq, Wq, out_dim)`` (at
            the real ``t``) and ``jac`` has shape ``(B, Hq, Wq, out_dim, 2)``
            (the per-pixel spatial Jacobian at ``t=0``).
        """
        rel, phi_phase, phi_kernel = self._compute_rel_and_phi(encoding, coords)

        # Forward output at the REAL t.
        out = self.heat_field(rel, phi_phase, phi_kernel, t, training=training)

        # Flatten the leading (B, Hq, Wq) dims to a single pixel axis N so a
        # per-pixel batch_jacobian is exact (pointwise field => block-diagonal
        # full Jacobian; batch_jacobian computes only the per-pixel blocks).
        out_dim = self.out_dim
        hidden = self.hidden_dim
        flat = tf.shape(rel)  # (B, Hq, Wq, 2)
        n = flat[0] * flat[1] * flat[2]

        rel_flat = ops.reshape(rel, (n, 2))  # (N, 2)
        phase_flat = ops.reshape(phi_phase, (n, hidden))  # (N, hidden)
        kernel_flat = ops.reshape(phi_kernel, (n, hidden, out_dim))  # (N, hidden, out)
        # t=0 broadcast to every pixel: (N, 1) zeros -> envelope == 1.
        t_zero = tf.zeros((n, 1), dtype=rel_flat.dtype)

        # persistent=True is REQUIRED by tf: batch_jacobian with
        # experimental_use_pfor=False in eager mode unrolls a per-output loop that
        # re-reads the tape, so the tape must outlive a single gradient call.
        with tf.GradientTape(persistent=True) as jac_tape:
            jac_tape.watch(rel_flat)
            out0_flat = self.heat_field(
                rel_flat, phase_flat, kernel_flat, t_zero, training=training
            )  # (N, out_dim)
        # experimental_use_pfor=False -> unrolled per-output loop; composes with
        # the trainer's outer tape (D-010). (N, out_dim, 2).
        jac_flat = jac_tape.batch_jacobian(
            out0_flat, rel_flat, experimental_use_pfor=False
        )
        del jac_tape

        # Reshape back to (B, Hq, Wq, out_dim, 2).
        out_shape = tf.concat(
            [flat[:3], tf.constant([out_dim, 2], dtype=flat.dtype)], axis=0
        )
        jac = ops.reshape(jac_flat, out_shape)
        return out, jac

    # -----------------------------------------------------------------

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Functional entry point: ``inputs = (encoding, coords, t)``.

        Keras layers prefer a single ``inputs`` argument for functional /
        serialization use, so the three decode tensors are passed as a tuple and
        unpacked here. The step-8 ``Thera`` model may call :meth:`decode`
        directly instead.

        Args:
            inputs: A 3-tuple ``(encoding, coords, t)``.
            training: Forwarded to :meth:`decode`.

        Returns:
            Field values, shape ``(B, Hq, Wq, out_dim)``.
        """
        encoding, coords, t = inputs
        return self.decode(encoding, coords, t, training=training)

    def get_config(self) -> Dict[str, Any]:
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
