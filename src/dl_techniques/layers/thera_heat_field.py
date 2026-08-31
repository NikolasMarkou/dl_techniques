"""
THERA neural heat field, as two Keras layers: ``ThermalActivation`` and ``HeatField``.

What problem this solves
------------------------
Classic super-resolution networks are trained for one fixed upscale factor (2x, 4x, ...).
Arbitrary-scale methods instead learn a *continuous* image: a small neural network that
maps a coordinate to a color, so you can sample it at any output resolution you like.
The catch is that a pixel is not a point. A pixel is a small area, and sampling a
continuous function at single points ignores that area, which produces aliasing
(jaggies, moire, shimmering) whenever the output grid is coarser than the detail the
field contains.

THERA (Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural
Heat Fields", TMLR 2025, arXiv:2311.17643) fixes this by construction instead of by
post-hoc blurring or supersampling. The continuous image is written as a sum of sine
waves (a SIREN-style field), and each sine wave is multiplied by the closed-form solution
of the heat equation. Blurring an image is mathematically the same as letting heat
diffuse for some time ``t``, and in the frequency domain that diffusion is just a
Gaussian decay factor per frequency. So the "correct amount of blur for this output
resolution" becomes a single scalar knob, applied analytically, at zero extra cost.

The mental model
----------------
    high frequency component  ->  decays fast as t grows
    low  frequency component  ->  survives

    amplitude = exp(-(w0 * norm)^2 * k * t)

``norm`` is how fast a given sine wave oscillates, ``k`` is a learned heat conductivity,
and ``t`` is the diffusion time. Decay is *quadratic* in frequency, which is exactly the
Gaussian low-pass filter you would want as an anti-aliasing pre-filter. Because the
target scale enters only through ``t``, one trained model is correctly anti-aliased at
every scale, with no retraining and no per-scale filter bank.

    t small  ->  fine output grid, little smoothing, keep the detail
    t large  ->  coarse output grid, more smoothing, drop the detail that would alias

What is in this file
--------------------
``ThermalActivation``
    The pointwise nonlinearity: ``sin(w0 * x + phase) * exp(-(w0 * norm)^2 * k * t)``.
    A SIREN sine times the heat envelope. It is stateless and owns no weights.

``HeatField``
    The field itself, evaluated on a grid of query coordinates. Forward pass:

        rel_coords (..., 2)
          -> project through shared frequency ``components`` (2, hidden)   [einsum]
          -> ThermalActivation with the per-pixel ``phase`` and time ``t``
          -> project through the per-pixel output ``kernel`` (hidden, out) [einsum]
          -> out (..., out_dim)

The one thing to understand before editing
------------------------------------------
THERA's field is *spatially varying*. A hypernetwork looks at the low-resolution input
and predicts a different field for every pixel. Concretely, the ``phase`` vector and the
final output ``kernel`` are per-pixel tensors produced by that hypernetwork (the ``phi``
tree in the reference code), not parameters of the field. Only the frequency
``components`` and the conductivity ``k`` are global and shared.

That split is mirrored here:

    components      shared, global   ->  OWNED WEIGHT, shape (2, hidden_dim)
    k               shared, global   ->  OWNED WEIGHT, scalar
    phi_phase       per pixel        ->  INPUT to call(), shape (..., hidden_dim)
    phi_kernel      per pixel        ->  INPUT to call(), shape (..., hidden_dim, out_dim)

Adding ``phase`` or the output kernel as weights of ``HeatField`` would duplicate the
hypernetwork's outputs and collapse the model to a single shared field for the whole
image, which destroys the property the architecture is named for. See decision D-004.

Difference from the reference implementation
--------------------------------------------
The original JAX/Flax code ``vmap``s one field instance over every pixel and threads a
nested parameter tree through it. Here that is replaced by ordinary batched einsums whose
leading dimensions are ``(B, Hq, Wq)``: the per-pixel kernel slab simply rides along the
batch axes. No ``vmap``, no parameter tree (invariant INV-5). The math is identical.

Defaults
--------
``k_init = sqrt(log 4) / (2 * pi^2)`` is the reference value, chosen so the Gaussian heat
kernel at unit time matches THERA's reference anti-alias filter. ``components_init_scale
= 16.0`` sets the radius of the frequency disk the initial components are drawn from.
``c`` is kept for config compatibility only and is read by nothing in this port; see the
``HeatField`` docstring.

Reference:
    Becker, Caye Daudt, Narnhofer, Peters, Metzger, Wegner, Schindler.
    "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields."
    TMLR 2025. arXiv:2311.17643. https://github.com/prs-eth/thera (``model/thera.py``)
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import LinearUpInitializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# THERA argument defaults (from the reference ``args/train.py``).
# ---------------------------------------------------------------------

# Heat-conductivity initial value: sqrt(log 4) / (2 * pi^2). Chosen so the
# Gaussian heat kernel at unit time matches THERA's reference anti-alias filter.
DEFAULT_K_INIT: float = float(np.sqrt(np.log(4.0)) / (np.pi ** 2 * 2.0))

# Frequency-disk scale for the ``components`` (first-layer frequencies) init.
DEFAULT_COMPONENTS_INIT_SCALE: float = 16.0

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.thera_heat_field")
class ThermalActivation(keras.layers.Layer):
    """THERA thermal activation: a phase-shifted sine with a heat-decay envelope.

    Provides THERA's aliasing-free nonlinearity: a SIREN-style sinusoid whose
    amplitude is attenuated by the closed-form solution of the heat equation. As
    the diffusion time ``t`` increases, high-frequency hidden units (large
    ``norm``) are damped FASTER, which is what yields an analytically smooth
    (anti-aliased) field response at any target scale. This layer is the
    pointwise activation core shared by every query pixel of :class:`HeatField`,
    and it is **STATELESS**: it owns no weights, because ``phase`` is a
    per-pixel hypernetwork output and ``norm`` / ``k`` are derived from
    :class:`HeatField`'s shared weights.

    **Operation:**

    .. code-block:: text

        x     [..., hidden] ──┐
                              ▼
        phase [..., hidden] ──► sin(w0·x + phase) ─────┐   SIREN oscillation
                                                       │
                                                      (×)──► out [..., hidden]
                                                       │
        norm  [hidden]     ──┐                         │
        k     scalar       ──┼─► exp(−(w0·norm)²·k·t) ─┘   heat envelope
        t     [..., 1]     ──┘

        out = sin(w0·x + phase) · exp(−(w0·‖norm‖)² · k · t)

    **Why the envelope is the whole point:**

    .. code-block:: text

        amplitude
            1 ┤████████░░░░░░░░              t = 0   nothing damped
              │
              ┤██████░░░░░░░░░░              t small  high freqs fade
              │
              ┤███░░░░░░░░░░░░░              t large  only low freqs survive
            0 ┴──────────────────► norm (per-unit frequency magnitude)
              low            high

        damping goes as exp(−norm²·k·t), so a unit's decay rate is
        QUADRATIC in its frequency. The target downscale factor enters
        only through t, which is why the field is aliasing-free at any
        scale WITHOUT re-training or a per-scale filter.

    :param w0: Frequency multiplier (``w0_scale`` in the reference). Multiplies
        both the oscillation argument and the envelope's frequency term. Must be
        positive. Defaults to 1.0.
    :type w0: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :raises ValueError: If ``w0`` is not positive.

    Input shape:
        See :meth:`call`. The primary input ``x`` is ``(..., hidden)``; the
        remaining arguments broadcast into it.

    Output shape:
        Same as ``x``, i.e. ``(..., hidden)``.

    Example:
        >>> act = ThermalActivation(w0=1.0)
        >>> # x, phase: (..., H); norm: (H,); k, t: scalars / broadcastable
        >>> y = act(x, t, norm, k, phase)

    Note:
        Stateless by design. Do not give it a ``phase`` weight: in THERA the
        phase is produced per pixel by the hypernetwork.
    """

    def __init__(self, w0: float = 1.0, **kwargs: Any) -> None:
        """Initialize the activation.

        :param w0: Frequency multiplier.
        :type w0: float
        :param kwargs: Forwarded to :class:`keras.layers.Layer`.
        :raises ValueError: If ``w0`` is not positive.
        """
        super().__init__(**kwargs)
        if w0 <= 0:
            raise ValueError(f"w0 must be positive, got {w0}")
        self.w0 = float(w0)

    def build(self, input_shape: Any) -> None:
        """Mark the layer built; there are no weights to create.

        An explicit ``build`` is kept rather than relying on the default so a
        parent layer calling ``child.build(...)`` does not trigger Keras'
        "build() was called but layer does not have a build() method" warning,
        which this repo treats as an unbuilt-sublayer serialization hazard
        (``LESSONS.md`` build-order).

        :param input_shape: Shape of the primary input; unused.
        :type input_shape: Any
        """
        super().build(input_shape)

    def call(
        self,
        x: Any,
        t: Any,
        norm: Any,
        k: Any,
        phase: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Apply the thermal activation.

        :param x: Pre-activation of shape ``(..., hidden)``.
        :type x: keras tensor
        :param t: Heat-diffusion time, broadcastable to ``(..., 1)`` or scalar.
            A rank gap against ``x`` is closed by inserting singleton spatial
            axes, so a ``(B, 1)`` tensor lines its batch dim up with ``x``'s.
        :type t: keras tensor
        :param norm: Per-hidden-unit frequency norms, shape ``(hidden,)``.
        :type norm: keras tensor
        :param k: Scalar heat conductivity (broadcastable).
        :type k: keras tensor
        :param phase: Per-pixel phase offsets, shape ``(..., hidden)``.
        :type phase: keras tensor
        :param training: Unused; present for the standard Keras signature.
        :type training: Optional[bool]
        :return: ``sin(w0 * x + phase) * exp(-(w0 * norm)^2 * k * t)``, shape
            ``(..., hidden)``.
        :rtype: keras tensor
        """
        oscillation = ops.sin(self.w0 * x + phase)
        # Envelope: norm is (hidden,) on the trailing axis; k scalar; t carries
        # only a leading batch dim (e.g. (B, 1)) so it must be rank-aligned to
        # x = (..., hidden) before broadcasting. We insert singleton spatial
        # axes between t's batch dim and its trailing 1 so the batch dim lines
        # up with x's leading batch dim and the trailing 1 broadcasts over the
        # hidden axis. (Scalar / already-aligned t passes through unchanged.)
        t = ops.convert_to_tensor(t)
        rank_gap = len(x.shape) - len(t.shape)
        if len(t.shape) >= 1 and rank_gap > 0:
            new_shape = (-1,) + (1,) * rank_gap + tuple(t.shape[1:])
            t = ops.reshape(t, new_shape)
        envelope = ops.exp(-ops.square(self.w0 * norm) * k * t)
        return oscillation * envelope

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which matches the primary input ``x``.

        The envelope only broadcasts in, so it never alters ``x``'s shape.

        :param input_shape: Shape of ``x``, ``(..., hidden)``.
        :type input_shape: Any
        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"w0": self.w0})
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.thera_heat_field")
class HeatField(keras.layers.Layer):
    """THERA spatially-varying neural heat field evaluated at query coordinates.

    Realizes THERA's spatially-varying neural heat field as a single Keras
    layer: a per-pixel SIREN field whose decay follows the heat equation,
    producing aliasing-free arbitrary-scale super-resolution. The field is
    shared across pixels ONLY through its frequency ``components`` and its
    conductivity ``k``; the per-pixel ``phase`` and output ``kernel`` are
    injected as INPUTS from the hypernetwork, so every query pixel evaluates its
    own field while still vectorizing over the ``(B, Hq, Wq)`` grid via a
    batched einsum (INV-5: no ``vmap``, no parameter tree).

    **Forward pass:**

    .. code-block:: text

        rel_coords [..., 2]
              │
              │  einsum('...c,ck->...k')
              ▼
        ┌──────────────────────────────────────┐
        │  components [2, hidden]   ◄── WEIGHT │
        │  shared by every query pixel         │
        └───────────────┬──────────────────────┘
                        │  x [..., hidden]
                        ▼
        ┌──────────────────────────────────────┐
        │  ThermalActivation(w0)               │
        │    sin(w0·x + phi_phase)             │
        │      × exp(−(w0·norm)²·k·t)          │
        │                                      │
        │  norm = ‖components‖₂ over axis −2   │
        │       → [hidden]                     │
        │  k    ◄── WEIGHT (scalar)            │
        │  t    ◄── input, the diffusion time  │
        │  phi_phase [..., hidden] ◄── INPUT   │
        └───────────────┬──────────────────────┘
                        │  thermal [..., hidden]
                        │
                        │  einsum('...k,...ko->...o')
                        ▼
        ┌──────────────────────────────────────┐
        │ phi_kernel [..., hidden, out] ◄─INPUT│
        │  a PER-PIXEL slab, no bias           │
        └───────────────┬──────────────────────┘
                        ▼
                  out [..., out_dim]

    **Weights versus per-pixel inputs (the load-bearing split):**

    .. code-block:: text

                       THERA reference        this port
                       ──────────────        ─────────
        components     global param     →    OWNED WEIGHT (2, hidden)
        k              global param     →    OWNED WEIGHT scalar
        phase          hypernetwork φ   →    INPUT phi_phase
        Dense kernel   hypernetwork φ   →    INPUT phi_kernel

        `field.init` in the reference exists only for SHAPE inference;
        phase and the output kernel are never the field's own weights.

        Adding them as weights here would double-create the
        hypernetwork's per-pixel params and collapse the field to a
        single SHARED phase/kernel -- destroying the "spatially
        varying" property the architecture is named for. See D-004.

    **Vectorization (what replaced the JAX vmap):**

    .. code-block:: text

        JAX:    vmap(field)(per-pixel param tree)
                one field instance conceptually per pixel

        here:   ONE einsum over leading dims (B, Hq, Wq)
                '...k,...ko->...o'
                the per-pixel kernel slab rides the batch axes

    :param hidden_dim: Field hidden width ``N``, the number of frequency
        components. Must be positive.
    :type hidden_dim: int
    :param out_dim: Output channel count, e.g. 3 for an RGB residual. Must be
        positive.
    :type out_dim: int
    :param w0: SIREN frequency multiplier for the field and its thermal
        activation. Must be positive. Defaults to 1.0.
    :type w0: float
    :param c: **DEAD KNOB.** In the THERA reference this is the SIREN variance
        constant behind the last Dense layer's init,
        ``w_std = sqrt(c / dim_hidden) / w0``. That initialization has **no
        counterpart anywhere in this port** -- and in particular NOT in the
        hypernetwork, which this docstring claimed until 2026-08-18:
        ``ThéraHypernetwork.out_conv`` (the layer that produces ``phi_kernel``)
        is a plain 1x1 ``keras.layers.Conv2D`` at Keras' default
        ``glorot_uniform``. Nothing reads ``c``; it is stored and serialized
        only. Must still be positive. Defaults to 6.0.
    :type c: float
    :param k_init: Initial value of the scalar ``k`` weight. Defaults to
        ``sqrt(log 4) / (2*pi^2)``, the THERA reference value, chosen so the
        Gaussian heat kernel at unit time matches the reference anti-alias
        filter.
    :type k_init: float
    :param components_init_scale: Frequency-disk scale passed to
        :class:`LinearUpInitializer` for ``components``. Defaults to 16.0.
    :type components_init_scale: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :raises ValueError: If ``hidden_dim``, ``out_dim``, ``w0`` or ``c`` is not
        positive.

    Input shape:
        Four tensors, in :meth:`call` order:

        - ``rel_coords``: ``(..., 2)``
        - ``phi_phase``: ``(..., hidden_dim)``
        - ``phi_kernel``: ``(..., hidden_dim, out_dim)``
        - ``t``: broadcastable to ``(..., 1)``, or scalar

    Output shape:
        ``(..., out_dim)``, where the leading dims are ``rel_coords``' own.

    :ivar components: Shared frequency components, shape ``(2, hidden_dim)``,
        initialized with :class:`LinearUpInitializer`.
    :vartype components: keras.Variable
    :ivar k: Scalar heat conductivity.
    :vartype k: keras.Variable
    :ivar thermal: The stateless :class:`ThermalActivation` sub-layer.
    :vartype thermal: ThermalActivation

    Example:
        >>> hf = HeatField(hidden_dim=32, out_dim=3)
        >>> out = hf(rel_coords, phi_phase, phi_kernel, t)  # (..., 3)
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        w0: float = 1.0,
        c: float = 6.0,
        k_init: float = DEFAULT_K_INIT,
        components_init_scale: float = DEFAULT_COMPONENTS_INIT_SCALE,
        **kwargs: Any,
    ) -> None:
        """Initialize the field and create its stateless activation sub-layer.

        The two owned weights are created in :meth:`build`.

        :param hidden_dim: Field hidden width.
        :type hidden_dim: int
        :param out_dim: Output channel count.
        :type out_dim: int
        :param w0: SIREN frequency multiplier.
        :type w0: float
        :param c: Dead knob; see the class docstring.
        :type c: float
        :param k_init: Initial value of the scalar ``k``.
        :type k_init: float
        :param components_init_scale: Frequency-disk scale for ``components``.
        :type components_init_scale: float
        :param kwargs: Forwarded to :class:`keras.layers.Layer`.
        :raises ValueError: If any positivity constraint is violated.
        """
        super().__init__(**kwargs)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if w0 <= 0:
            raise ValueError(f"w0 must be positive, got {w0}")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.w0 = float(w0)
        self.c = float(c)
        self.k_init = float(k_init)
        self.components_init_scale = float(components_init_scale)

        # Stateless thermal-activation sublayer (built explicitly in ``build``).
        self.thermal = ThermalActivation(w0=self.w0, name="thermal")

        # Owned weights (created in ``build``).
        self.components = None
        self.k = None

    def build(self, input_shape: Any) -> None:
        """Create ``components`` and ``k``, and build the thermal sub-layer.

        The owned weights depend only on ``hidden_dim`` / ``out_dim``, so
        ``input_shape`` is not unpacked for them; it is normalized robustly for
        both single- and multi-input invocation styles.

        :param input_shape: The list/tuple of per-input shapes in :meth:`call`
            order, ``[rel_coords, phi_phase, phi_kernel, t]``, or a single shape
            tuple treated as ``rel_coords``.
        :type input_shape: Any
        """
        # DECISION plan_2026-06-11_f662207d/D-004 (see decisions.md):
        # The per-pixel phi (phi_phase, phi_kernel) are INPUTS produced by the
        # hypernetwork, NOT weights of this field; only ``components`` and ``k``
        # are owned (global, shared) weights. The batched einsum below replaces
        # JAX's vmap-over-pixels + nested param-tree (INV-5). Do NOT add
        # ``phase`` or the output ``kernel`` as weights here -- that would
        # double-create the hypernetwork's per-pixel params and break THERA's
        # spatially-varying field (every query pixel must get its OWN phase /
        # kernel slab from phi, not a single shared one).
        #
        # ``HeatField`` is multi-input: ``input_shape`` is the list/tuple of
        # shapes [rel_coords, phi_phase, phi_kernel, t] (in call order). The
        # owned weights depend only on hidden_dim / out_dim, so we do not need
        # to unpack it, but we accept it robustly for both single- and
        # multi-input invocation styles.
        self.components = self.add_weight(
            name="components",
            shape=(2, self.hidden_dim),
            initializer=LinearUpInitializer(scale=self.components_init_scale),
            trainable=True,
            dtype="float32",
        )
        self.k = self.add_weight(
            name="k",
            shape=(),
            initializer=keras.initializers.Constant(self.k_init),
            trainable=True,
            dtype="float32",
        )

        # Explicitly build the (stateless) thermal sublayer so a ``.keras``
        # reload restores cleanly and no unbuilt-sublayer warning is emitted
        # (LESSONS.md Keras-3 build-order discipline).
        thermal_in_shape = self._normalize_input_shapes(input_shape)
        # x fed to ThermalActivation has shape (..., hidden_dim).
        x_shape = thermal_in_shape[0][:-1] + (self.hidden_dim,)
        if not self.thermal.built:
            self.thermal.build(x_shape)

        super().build(input_shape)

    @staticmethod
    def _normalize_input_shapes(
        input_shape: Any,
    ) -> List[Tuple[Optional[int], ...]]:
        """Coerce a single- or multi-input ``build`` shape argument to a list.

        ``HeatField`` is invoked with four positional tensors. Depending on how
        Keras routes ``build`` (functional versus subclass or explicit call),
        ``input_shape`` may arrive as a list/tuple of per-input shapes or as a
        single shape tuple.

        :param input_shape: Per-input shapes, or one shape tuple.
        :type input_shape: Any
        :return: A list of shape tuples whose FIRST element is always treated
            as ``rel_coords``' shape.
        :rtype: List[Tuple[Optional[int], ...]]
        """
        # A list/tuple whose first element is itself a shape (list/tuple) =>
        # already a collection of per-input shapes.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            return [tuple(s) for s in input_shape]
        # Otherwise treat the whole thing as a single shape (rel_coords).
        return [tuple(input_shape)]

    def call(
        self,
        rel_coords: Any,
        phi_phase: Any,
        phi_kernel: Any,
        t: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Evaluate the heat field at the query coordinates.

        :param rel_coords: Relative query coordinates, shape ``(..., 2)``.
        :type rel_coords: keras tensor
        :param phi_phase: Per-pixel phase offsets from the hypernetwork, shape
            ``(..., hidden_dim)``.
        :type phi_phase: keras tensor
        :param phi_kernel: Per-pixel output kernel slabs from the hypernetwork,
            shape ``(..., hidden_dim, out_dim)``.
        :type phi_kernel: keras tensor
        :param t: Heat-diffusion time, broadcastable to ``(..., 1)`` or scalar.
        :type t: keras tensor
        :param training: Unused; present for the standard Keras signature.
        :type training: Optional[bool]
        :return: Field values, shape ``(..., out_dim)``.
        :rtype: keras tensor
        """
        # Project coords through shared frequency components: (...,2),(2,k)->(...,k)
        x = ops.einsum("...c,ck->...k", rel_coords, self.components)

        # Per-component frequency magnitude over the x/y axis -> (hidden,).
        norm = ops.norm(self.components, axis=-2)

        # Thermal activation (sin + heat envelope) -> (..., hidden).
        thermal = self.thermal(x, t, norm, self.k, phi_phase, training=training)

        # Per-pixel output projection (no bias): (...,k),(...,k,o)->(...,o).
        out = ops.einsum("...k,...ko->...o", thermal, phi_kernel)
        return out

    def compute_output_shape(
        self,
        input_shape: Any,
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from ``rel_coords``' leading dimensions.

        :param input_shape: As accepted by :meth:`build`.
        :type input_shape: Any
        :return: ``rel_coords``' shape with the size-2 coordinate axis replaced
            by ``out_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        shapes = self._normalize_input_shapes(input_shape)
        leading = shapes[0][:-1]  # drop the coordinate (size-2) axis
        return tuple(leading) + (self.out_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

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