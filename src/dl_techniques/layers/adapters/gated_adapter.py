"""**What it does:**

`GatedAdapter` composes a low-rank adapter with a
:class:`~dl_techniques.layers.statistics.local_support_gate.LocalSupportGate`,
which together implement *Local Support Learning* (LSL): an additive weight
update whose effect is confined to the input region that produced it.

The update to a single weight matrix `W` is `ΔW = W_adapter`. Applied
everywhere, `ΔW` alters the output of *every* input, which is why conventional
adaptation forgets. LSL interposes a gate so the update reaches only inputs
drawn from the phase's own distribution:

.. code-block:: text

    x  (..., input_dim)          ← the gate scores the base projection's INPUT,
          │                        not its output
          ├──► LocalSupportGate ──► g  (...,)        1[log Φ_pos(x) > log Φ_neg(x)]
          │
          └──► LoRAAdapter ──────► delta  (..., output_dim)

    output = x @ W.T + Σ_p  g_p ⊙ LSL_p(x)          Σ over phases

    W               the caller's own base projection, untouched
    g_p             phase p's gate decision on THIS token
    LSL_p(x)        phase p's low-rank delta, via LoRAAdapter(num_adapters=P)
                    indexed by adapter_idx=p

The per-phase sum is what makes the composition a continual learner rather than
an ordinary adapter: every phase keeps its own delta and its own support
estimate, so a later phase cannot overwrite an earlier one's behaviour. Cost
grows linearly in the number of phases at inference, which is the price of not
being able to merge the deltas into the base weights.

**The gate is held open during a phase's own training.** Its decision carries no
gradient, and during training every token comes from the distribution being
learned — the one case the gate is guaranteed to open for. Routing on a gate
that is still being fitted would bootstrap against its own initialisation.

**It does not own the base projection.** Like :class:`LoRAAdapter`, this layer
returns a **delta**, and the caller adds it onto its own output. That is the
seam that lets the same composition serve a `Dense` inside a SwiGLU block and a
`GroupedQueryAttention` projection inside a decoder, without either knowing the
other exists.

**References:**
    - Ben-Kish, A., Kumar, A., Glass, J., & Giryes, R., 2026. Local Support
      Learning. (https://arxiv.org/abs/2610.02126)
    - Hu, E. J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language
      Models. (https://arxiv.org/abs/2106.09685)

Examples:
    ```python
    # One phase: a plain gated adapter.
    adapter = GatedAdapter(input_dim=2048, output_dim=2048, rank=16, alpha=32.0)
    adapter.fit_gate(phase_activations=phase_x, generic_activations=generic_x)
    delta = adapter(hidden, phase_idx=0)

    # Several phases: one delta and one gate per phase, summed with each
    # phase's own decision.
    adapter = GatedAdapter(input_dim=2048, output_dim=2048, rank=16,
                           alpha=32.0, num_phases=3)
    for p, (in_phase, generic) in enumerate(corpora):
        adapter.fit_gate(phase_activations=in_phase, generic_activations=generic,
                         phase_idx=p)
    delta = adapter(hidden)   # sums every phase's gated contribution
    ```
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

from .factory import create_adapter_layer
from .lora import LoRAAdapter
from ..statistics.local_support_gate import LocalSupportGate

# ---------------------------------------------------------------------

#: Which gates the layer composes.
GatedAdapterType = Literal['local_support']


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.adapters.gated_adapter")
class GatedAdapter(keras.layers.Layer):
    """
    A per-phase low-rank delta, applied only where that phase's support gate opens.

    Owns one :class:`LoRAAdapter` carrying ``num_phases`` independent ``A``/``B``
    pairs and one :class:`LocalSupportGate` per phase, and sums each phase's
    delta scaled by its own gate decision. Returns the **summed delta**, not an
    output — the caller adds it onto its own base projection.

    The sub-layers are created **unconditionally** in ``__init__`` and indexed by
    a Python ``int`` per call, rather than built lazily per phase. That keeps the
    object graph and the weight layout identical no matter which phases have been
    fitted, which is what lets a checkpoint round-trip: a phase added later lands
    in a layer whose shape was already fixed.

    Architecture:

    .. code-block:: text

        inputs  (..., input_dim)
              │
              ├─► gate[0]  ─► g_0  (...,)      1[log Φ_pos⁰(x) > log Φ_neg⁰(x)]
              ├─► gate[1]  ─► g_1  (...,)      ...
              │      ⋮
              ├─► lora(adapter_idx=p) ─► LSL_p(x)   (..., output_dim)
              │
              ▼  Σ_p  g_p ⊙ LSL_p(x)
        delta  (..., output_dim)

    :param input_dim: Width of the activations the gate scores and the delta is
        computed from. Must be positive and statically known.
    :type input_dim: int
    :param output_dim: Width of the delta this adapter produces. Must be positive.
    :type output_dim: int
    :param rank: Bottleneck width of every phase's low-rank pair. Must be positive.
    :type rank: int
    :param alpha: LoRA scaling numerator; the applied scale is ``alpha / rank``.
        Must be positive.
    :type alpha: float
    :param num_phases: Number of independent phases to allocate — one delta and
        one gate each. Must be positive. At `1` this is a single gated adapter,
        which is the ordinary LSL setting.
    :type num_phases: int
    :param adapter_type: Which adapter the per-phase delta uses. `'local_support'`
        resolves to :class:`~dl_techniques.layers.adapters.lora.LoRAAdapter`;
        the argument exists so the dispatcher is the only construction path.
    :type adapter_type: GatedAdapterType
    :param gate_args: Constructor arguments forwarded to every phase's
        :class:`LocalSupportGate`. Must not contain ``input_dim`` (this layer
        supplies it) nor ``name`` (each phase needs a distinct one).
    :type gate_args: Optional[Dict[str, Any]]
    :param kernel_initializer: Initializer for the low-rank ``A`` matrices. Each
        phase's slice receives its own
        :func:`~dl_techniques.initializers.clone.clone_initializer` clone, per
        the D-007 note on :class:`LoRAAdapter`.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param active_phases: Phases that contribute to the output. Defaults to every
        phase. Restricting it is how a caller freezes earlier phases — their
        deltas stay in the checkpoint and stay fitted, but stop being applied.
        Must be a subset of ``range(num_phases)``.
    :type active_phases: Optional[Sequence[int]]
    :param name: Optional Keras layer name.
    :type name: Optional[str]
    :param kwargs: Extra arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar lora: The owned adapter, carrying ``num_phases`` ``A``/``B`` pairs.
    :vartype lora: LoRAAdapter
    :ivar gates: One gate per phase, indexed to match ``lora``'s slots.
    :vartype gates: List[LocalSupportGate]
    :ivar active_phases: The phases currently contributing.
    :vartype active_phases: List[int]

    :raises ValueError: If ``input_dim``, ``output_dim``, ``rank`` or
        ``num_phases`` is not positive; if ``alpha`` is not positive; if
        ``gate_args`` carries ``input_dim`` or ``name``; or if
        ``active_phases`` names a phase outside ``[0, num_phases)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, last axis ``output_dim``.

    Example:
        .. code-block:: python

            adapter = GatedAdapter(input_dim=512, output_dim=512, rank=8,
                                   alpha=16.0, num_phases=2, seed=3)
            adapter.fit_gate(phase_activations=x0, generic_activations=g,
                             phase_idx=0)
            adapter.fit_gate(phase_activations=x1, generic_activations=g,
                             phase_idx=1)
            delta = adapter(hidden)            # both phases, each gated
            delta = adapter(hidden, phase_idx=0)   # phase 0 only

    Note:
        A gate that has not been fitted routes every token closed, so an unfitted
        phase contributes **nothing** rather than its ungated delta. That is the
        safe direction and it is silent: the phase's parameters are present and
        trained-looking while having no effect. Check
        :meth:`LocalSupportGate.is_fitted` when a phase appears to be ignored.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        alpha: float,
        num_phases: int = 1,
        adapter_type: GatedAdapterType = 'local_support',
        gate_args: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        active_phases: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        No weights are created here beyond what the sub-layers own; the adapter
        builds them in its own ``build``, against the input width this
        constructor was given.
        """
        super().__init__(name=name, **kwargs)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if num_phases <= 0:
            raise ValueError(f"num_phases must be positive, got {num_phases}")

        gate_args = dict(gate_args or {})
        for reserved in ('input_dim', 'name'):
            if reserved in gate_args:
                raise ValueError(
                    f"gate_args must not carry '{reserved}': this layer "
                    f"supplies input_dim and a distinct name per phase. Pass "
                    f"it as a GatedAdapter constructor argument instead."
                )

        if active_phases is None:
            resolved_active = list(range(num_phases))
        else:
            resolved_active = list(active_phases)
            for phase in resolved_active:
                if not (0 <= phase < num_phases):
                    raise ValueError(
                        f"active_phases entry {phase} is outside "
                        f"[0, {num_phases})"
                    )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.num_phases = num_phases
        self.adapter_type = adapter_type
        self.gate_args = gate_args
        self.active_phases = resolved_active
        self.seed = seed
        # Resolved here, not read through `**kwargs` in build(): an unstored
        # constructor argument is dead on arrival (guide §6.2), and get_config
        # has to serialize the resolved value so the round trip is exact.
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.lora: Optional[LoRAAdapter] = None
        self.gates: List[LocalSupportGate] = []

        logger.info(
            f"Initialized GatedAdapter with input_dim={input_dim}, "
            f"output_dim={output_dim}, rank={rank}, alpha={alpha}, "
            f"num_phases={num_phases}, active_phases={resolved_active}, "
            f"adapter_type={adapter_type}"
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the sub-layer tree exactly as ``call`` runs it.

        Both sub-layers are created here rather than in ``__init__`` because the
        adapter's weights depend on the input width, and because creating them
        here keeps a single place that owns the tree ``call`` walks — which is
        what build-parity by relative ``w.path`` is checked against.

        :param input_shape: Shape of the input activations; the last axis must
            match ``input_dim``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input's last axis does not match
            ``input_dim``.
        """
        if self.built:
            return

        input_width = input_shape[-1]
        if input_width is not None and input_width != self.input_dim:
            raise ValueError(
                f"input width {input_width} does not match the adapter's "
                f"input_dim={self.input_dim}"
            )

        self.lora = create_adapter_layer(
            'lora',
            name='lora',
            output_dim=self.output_dim,
            rank=self.rank,
            alpha=self.alpha,
            num_adapters=self.num_phases,
            kernel_initializer=clone_initializer(self.kernel_initializer),
        )

        # One gate per phase, created unconditionally so the layer's layout does
        # not depend on which phases have been fitted. `seed` is offset per phase
        # so two phases do not start from the same sketch — with one shared seed
        # their gates would be correlated at initialisation.
        self.gates = []
        for phase in range(self.num_phases):
            gate_seed = None if self.seed is None else self.seed + phase
            self.gates.append(
                LocalSupportGate(
                    input_dim=self.input_dim,
                    seed=gate_seed,
                    name=f'gate_{phase}',
                    **self.gate_args,
                )
            )

        self.lora.build(input_shape)
        for gate in self.gates:
            gate.build(input_shape)

        super().build(input_shape)

    # -----------------------------------------------------------------
    # inference
    # -----------------------------------------------------------------

    def call(self, inputs, training=None, phase_idx: Optional[int] = None):
        """Compute the gated delta, summing over the active phases.

        :param inputs: Tensor of shape ``(..., input_dim)``.
        :type inputs: Any
        :param training: Forwarded to every gate. Unused by them (the gate is
            non-differentiable), but the adapter's own ``A``/``B`` are ordinary
            trainable weights reached through the caller's optimizer.
        :type training: Optional[bool]
        :param phase_idx: Apply only this phase, instead of every active phase.
            A plain Python ``int`` — which phases exist is fixed at
            construction. Used for per-phase evaluation and for training one
            phase at a time.
        :type phase_idx: Optional[int]
        :return: Delta tensor of shape ``(..., output_dim)``.
        :rtype: Any
        :raises ValueError: If ``phase_idx`` is outside ``[0, num_phases)``.
        """
        phases = self._resolve_phases(phase_idx)
        if not phases:
            # Every phase inactive: a zero delta of the right shape, so the
            # caller's arithmetic still works and the layer stays callable.
            return ops.zeros(
                ops.shape(inputs)[:-1] + (self.output_dim,), dtype='float32'
            )

        x = ops.cast(inputs, 'float32')
        deltas = ops.zeros(
            ops.shape(x)[:-1] + (self.output_dim,), dtype='float32'
        )
        for phase in phases:
            # `broadcast_to` rather than `*`: the gate's decision has one entry
            # per token while the delta has one per token per feature, so the
            # score has to be expanded along the feature axis before it can
            # scale anything. A plain multiply would broadcast the score along
            # the WRONG axis and silently scale the wrong dimension.
            score = ops.expand_dims(self.gates[phase](x, training=training), -1)
            delta = self.lora(x, adapter_idx=phase)
            deltas = deltas + ops.broadcast_to(score, ops.shape(delta)) * delta
        return deltas

    def _resolve_phases(self, phase_idx: Optional[int]) -> List[int]:
        """Reconcile ``phase_idx`` and ``active_phases`` into the phase list to apply.

        :param phase_idx: An explicit single phase, or ``None``.
        :type phase_idx: Optional[int]
        :return: The phases whose contribution should be summed.
        :rtype: List[int]
        :raises ValueError: If ``phase_idx`` is outside ``[0, num_phases)``, or
            names a phase that ``active_phases`` has switched off.
        """
        if phase_idx is None:
            return list(self.active_phases)
        if not (0 <= phase_idx < self.num_phases):
            raise ValueError(
                f"phase_idx must be in [0, {self.num_phases}), got {phase_idx}"
            )
        if phase_idx not in self.active_phases:
            raise ValueError(
                f"phase {phase_idx} is not active (active_phases="
                f"{self.active_phases}); it cannot be applied on its own. "
                f"Its parameters are still present."
            )
        return [phase_idx]

    def set_active_phases(self, phases: Optional[Sequence[int]]) -> None:
        """Restrict which phases contribute to the output.

        The mechanism for freezing an earlier phase: its delta stays in the
        checkpoint and its gate stays fitted, but it stops being applied, so the
        optimizer cannot reach it through this layer.

        :param phases: The phases to keep active, or ``None`` for all of them.
        :type phases: Optional[Sequence[int]]
        :raises ValueError: If any entry is outside ``[0, num_phases)``.
        """
        if phases is None:
            self.active_phases = list(range(self.num_phases))
            return
        resolved = list(phases)
        for phase in resolved:
            if not (0 <= phase < self.num_phases):
                raise ValueError(
                    f"active_phases entry {phase} is outside "
                    f"[0, {self.num_phases})"
                )
        self.active_phases = resolved

    # -----------------------------------------------------------------
    # gate fitting
    # -----------------------------------------------------------------

    def fit_gate(
        self,
        phase_activations,
        generic_activations,
        phase_idx: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> Dict[str, int]:
        """Fit one phase's positive and negative mixtures by exact batch EM.

        Convenience wrapper over the two gates' own fitting entry points. For the
        streaming path, call :meth:`LocalSupportGate.observe` on each phase's gate
        directly — there is no streaming form here on purpose, because the
        streaming tradeoff (a ~20% likelihood floor and a materially worse
        out-of-distribution hit rate) is worth choosing per gate rather than
        inheriting silently from this layer.

        :param phase_activations: Tensor of shape ``(n, input_dim)`` — this
            phase's own activations.
        :type phase_activations: Any
        :param generic_activations: Tensor of shape ``(m, input_dim)`` — the
            generic reference sample.
        :type generic_activations: Any
        :param phase_idx: Which phase to fit. ``None`` fits every phase, which is
            what a single-phase caller wants and what a multi-phase caller
            almost certainly does not.
        :type phase_idx: Optional[int]
        :param max_iter: EM iteration cap, per mixture.
        :type max_iter: int
        :param tol: Relative improvement below which to stop, per mixture.
        :type tol: float
        :return: ``{'pos': iterations, 'neg': iterations}``.
        :rtype: Dict[str, int]
        :raises ValueError: If ``phase_idx`` is outside ``[0, num_phases)``.
        """
        phases = (
            list(range(self.num_phases)) if phase_idx is None
            else self._resolve_phases(phase_idx)
        )
        result: Dict[str, int] = {}
        for phase in phases:
            result[f'phase_{phase}'] = self.gates[phase].fit(
                pos_samples=phase_activations,
                neg_samples=generic_activations,
                max_iter=max_iter,
                tol=tol,
            )
        return result

    def is_phase_fitted(self, phase_idx: int) -> bool:
        """Whether ``phase_idx`` has both of its mixtures fitted.

        :param phase_idx: The phase to query.
        :type phase_idx: int
        :return: ``True`` only if the phase's gate is fully fitted.
        :rtype: bool
        :raises ValueError: If ``phase_idx`` is outside ``[0, num_phases)``.
        """
        if not (0 <= phase_idx < self.num_phases):
            raise ValueError(
                f"phase_idx must be in [0, {self.num_phases}), got {phase_idx}"
            )
        gate = self.gates[phase_idx]
        return gate.is_fitted('pos') and gate.is_fitted('neg')

    # -----------------------------------------------------------------
    # shape
    # -----------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape, from stored config, while unbuilt.

        :param input_shape: Shape tuple of the input activations.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape`` with the last axis set to ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = list(input_shape)
        shape[-1] = self.output_dim
        return tuple(shape)

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dict suitable for ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'rank': self.rank,
            'alpha': self.alpha,
            'num_phases': self.num_phases,
            'adapter_type': self.adapter_type,
            'gate_args': self.gate_args,
            'kernel_initializer': keras.initializers.serialize(
                self.kernel_initializer
            ),
            'active_phases': list(self.active_phases),
            'seed': self.seed,
        })
        return config
