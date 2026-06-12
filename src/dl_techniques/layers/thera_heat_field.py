"""THERA neural heat field as Keras layers: ``ThermalActivation`` + ``HeatField``.

THERA (Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields)
represents the high-resolution signal as a **spatially-varying** SIREN-style
field whose activations decay according to a closed-form solution of the heat
equation. The field is evaluated at query coordinates ``x`` and a query "time"
``t`` (the heat-diffusion time, which encodes the target downscale / smoothing
level), giving an analytically aliasing-free response.

The reference JAX/Flax field (``model/thera.py``) is::

    class Thermal(nn.Module):
        w0_scale: float = 1.
        def __call__(self, x, t, norm, k):
            phase = self.param('phase', uniform(.5), x.shape[-1:])
            return jnp.sin(self.w0_scale * x + phase) \
                 * jnp.exp(-(self.w0_scale * norm)**2 * k * t)

    class HeatField(nn.Module):
        dim_hidden: int; dim_out: int; w0: float = 1.; c: float = 6.
        def __call__(self, x, t, k, components):     # x: coord; components: (2, hidden)
            x = x @ components                                # (..., hidden)
            norm = jnp.linalg.norm(components, axis=-2)       # (hidden,)
            x = Thermal(self.w0)(x, t, norm, k)               # (..., hidden)
            w_std = math.sqrt(self.c / self.dim_hidden) / self.w0
            x = nn.Dense(self.dim_out, kernel_init=uniform_between(-w_std, w_std),
                         use_bias=False)(x)
            return x                                          # (..., dim_out)

Critical architecture fact (this shapes the Keras interface)
------------------------------------------------------------
In THERA the field's ``phase`` (shape ``(hidden,)``) and the final ``Dense``
``kernel`` (shape ``(hidden, dim_out)``) are **NOT** trainable weights of the
field -- they are **per-pixel parameters produced by the hypernetwork** (the
``phi`` parameter tree; ``field.init`` is used only for *shape* inference). Only
``components`` (shape ``(2, hidden)``) and ``k`` (scalar) are global shared
params.

So in this Keras port:

* ``HeatField`` OWNS as trainable weights only ``components`` (shape
  ``(2, hidden_dim)``, init :class:`LinearUpInitializer`) and ``k`` (scalar).
* ``HeatField.call`` RECEIVES the per-pixel ``phi`` as **inputs**: ``phi_phase``
  (``(..., hidden)``) and ``phi_kernel`` (``(..., hidden, dim_out)``), alongside
  the relative query coordinates ``rel_coords`` (``(..., 2)``) and the heat time
  ``t`` (``(..., 1)`` or scalar / broadcastable).

The original JAX implementation ``vmap``s the field over pixels and threads a
nested parameter tree. Here that is replaced by a single **batched einsum** over
the leading ``(B, Hq, Wq)`` dims (no ``vmap``, no param-tree -- invariant INV-5).

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/thera.py``).
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import LinearUpInitializer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# THERA argument defaults (from the reference ``args/train.py``).
# ---------------------------------------------------------------------

# Heat-conductivity initial value: sqrt(log 4) / (2 * pi^2). Chosen so the
# Gaussian heat kernel at unit time matches THERA's reference anti-alias filter.
DEFAULT_K_INIT: float = float(np.sqrt(np.log(4.0)) / (np.pi ** 2 * 2.0))

# Frequency-disk scale for the ``components`` (first-layer frequencies) init.
DEFAULT_COMPONENTS_INIT_SCALE: float = 16.0

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ThermalActivation(keras.layers.Layer):
    """THERA thermal activation: a phase-shifted sine with a heat-decay envelope.

    Computes, elementwise over the hidden axis::

        sin(w0 * x + phase) * exp(-(w0 * norm)^2 * k * t)

    The sine term is the SIREN-style oscillation; the exponential is the
    closed-form heat-equation envelope that smoothly attenuates high-frequency
    components as the diffusion time ``t`` grows. This layer is **stateless**
    (it owns no weights): ``phase`` and ``norm`` / ``k`` / ``t`` are all passed
    in by the caller (:class:`HeatField`), because in THERA ``phase`` is a
    per-pixel hypernetwork output and ``norm`` / ``k`` are derived from
    :class:`HeatField`'s shared weights.

    Args:
        w0: The frequency multiplier (``w0_scale`` in the reference). Multiplies
            both the oscillation argument and the envelope's frequency term.
            Defaults to 1.0.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input/Output:
        See :meth:`call`. Output has the same shape as ``x`` (broadcast with the
        envelope), i.e. ``(..., hidden)``.

    Example:
        >>> act = ThermalActivation(w0=1.0)
        >>> # x, phase: (..., H); norm: (H,); k, t: scalars / broadcastable
        >>> y = act(x, t, norm, k, phase)
    """

    def __init__(self, w0: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.w0 = float(w0)

    def build(self, input_shape: Any) -> None:
        # Stateless layer: no weights to create. An explicit ``build`` is kept
        # (rather than relying on the default) so parent layers calling
        # ``child.build(...)`` do not trigger Keras' "build() was called but
        # layer does not have a build() method" warning, which the repo treats
        # as an unbuilt-sublayer serialization hazard (LESSONS.md build-order).
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

        Args:
            x: Pre-activation, shape ``(..., hidden)``.
            t: Heat-diffusion time, broadcastable to ``(..., 1)`` (or scalar).
            norm: Per-hidden-unit frequency norms, shape ``(hidden,)``.
            k: Scalar heat conductivity (broadcastable).
            phase: Per-pixel phase offsets, shape ``(..., hidden)``.
            training: Unused; present for the standard Keras signature.

        Returns:
            ``sin(w0 * x + phase) * exp(-(w0 * norm)^2 * k * t)``, shape
            ``(..., hidden)``.
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

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"w0": self.w0})
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HeatField(keras.layers.Layer):
    """THERA spatially-varying neural heat field evaluated at query coordinates.

    A per-pixel SIREN-style field with a heat-equation decay envelope. Given
    relative query coordinates ``rel_coords`` and a heat time ``t``, it projects
    the coordinates through shared frequency ``components``, applies the
    :class:`ThermalActivation`, and contracts the result with a **per-pixel**
    output kernel to produce the field value.

    The forward pass implements (with leading dims ``B, Hq, Wq`` flowing through
    a single batched einsum -- INV-5, no ``vmap`` / param-tree)::

        x       = einsum('...c,ck->...k', rel_coords, components)   # (..., hidden)
        norm    = ||components||_2  over axis -2                      # (hidden,)
        thermal = sin(w0 * x + phi_phase) * exp(-(w0*norm)^2 * k * t) # (..., hidden)
        out     = einsum('...k,...ko->...o', thermal, phi_kernel)     # (..., out)

    Owned (global, shared) trainable weights:
        * ``components``: shape ``(2, hidden_dim)``, init :class:`LinearUpInitializer`.
        * ``k``: scalar heat conductivity, init constant ``k_init``.

    Per-pixel inputs (produced by the hypernetwork, NOT weights):
        * ``phi_phase``: shape ``(..., hidden_dim)``.
        * ``phi_kernel``: shape ``(..., hidden_dim, out_dim)``.

    Args:
        hidden_dim: Field hidden width ``N`` (number of frequency components).
        out_dim: Output channel count (e.g. 3 for RGB residual).
        w0: SIREN frequency multiplier for the field / thermal activation.
            Defaults to 1.0.
        c: SIREN variance constant; only documents the THERA Dense init scale
            (``sqrt(c / hidden) / w0``), which here lives in the hypernetwork
            that produces ``phi_kernel``. Stored for config fidelity.
            Defaults to 6.0.
        k_init: Initial value of the scalar ``k`` weight. Defaults to
            ``sqrt(log 4) / (2*pi^2)`` (THERA reference).
        components_init_scale: Frequency-disk scale passed to
            :class:`LinearUpInitializer` for ``components``. Defaults to 16.0.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

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

        # Stateless thermal-activation sublayer (built explicitly in ``build``).
        self.thermal = ThermalActivation(w0=self.w0, name="thermal")

        # Owned weights (created in ``build``).
        self.components = None
        self.k = None

    def build(self, input_shape: Any) -> None:
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
        """Coerce a single- or multi-input ``build`` shape arg into a shape list.

        ``HeatField`` is invoked with four positional tensors. Depending on how
        Keras routes ``build`` (functional vs. subclass / explicit call), the
        ``input_shape`` may arrive as a list/tuple of per-input shapes, or as a
        single shape tuple. This normalizes to a list of shape tuples; the first
        element is always treated as ``rel_coords``' shape.
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

        Args:
            rel_coords: Relative query coordinates, shape ``(..., 2)``.
            phi_phase: Per-pixel phase offsets, shape ``(..., hidden_dim)``.
            phi_kernel: Per-pixel output kernel slabs, shape
                ``(..., hidden_dim, out_dim)``.
            t: Heat-diffusion time, broadcastable to ``(..., 1)`` (or scalar).
            training: Unused; present for the standard Keras signature.

        Returns:
            Field values, shape ``(..., out_dim)``.
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
        shapes = self._normalize_input_shapes(input_shape)
        leading = shapes[0][:-1]  # drop the coordinate (size-2) axis
        return tuple(leading) + (self.out_dim,)

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
