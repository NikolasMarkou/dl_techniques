"""``FlowMatchingODE`` can actually be sampled, and ``force_unconditional`` bites.

This file exists because of a **measured dead branch** (decisions.md D-027):
``force_unconditional`` was stored, serialized and read by nothing, and the
variant inherited ``BridgeSDE.dX_t``, whose first act after the network call is
``self.sigma(t)`` -- which ``FlowMatchingODE`` deliberately raises on. The
flow-matching baseline could be TRAINED and could not be SAMPLED at all, under a
green 497-arm suite, because no arm had ever called ``simulate`` on it.

**What each arm pins, and why the obvious version of it is blind.**

``TestItSamples``
    End-to-end finiteness in both directions. This is the arm the old suite was
    missing entirely; it is necessary and nowhere near sufficient.

``TestForceUnconditionalRejectsCFG``
    Upstream raises rather than silently ignoring ``cfg_scale`` under
    ``force_unconditional``. A "returns the same value" arm would be VACUOUS
    here: this port's non-standard CFG formula is ``cond + s*(cond - uncond)``,
    which at ``s = 0`` returns ``cond`` unchanged -- so the two branches agree
    exactly at the only scale that is legal anyway.

``TestForceUnconditionalForcesTheUnconditionalBranch``
    The load-bearing arm. It records what the NETWORK actually received, not
    what the trajectory looks like, because both the forced and the unforced
    call produce finite, correctly shaped, plausible trajectories. Two separate
    claims: the ``cond_mask`` is all-false, and ``direction`` is FORWARD even
    when simulating in reverse -- upstream's "one shared field (theoretically
    guaranteed); outer reverse only flips dt".

``TestTheSignFlipIsObservable``
    ``signed_dt = -dt if reverse else dt``. Dropping the flip leaves every
    shape, dtype and finiteness arm green; what changes is which way the
    trajectory moves from a fixed start. Both arms here hold ``t`` and the
    state FIXED and call ``dX_t`` directly. There is deliberately **no**
    ``simulate``-level version: through ``simulate`` the two directions do not
    share a time (step 0 forward is at ``t = 0``, step 0 reverse at ``t = 1``),
    so the increments are only approximately opposed and any statistic over
    them is marginal. One was tried, and was FLAKY -- ">0.9 of entries opposed
    in sign" measured, over 24 seeds, ``min 0.8555 / mean 0.9204 / max 0.9609``
    unforced, i.e. 20.8% of draws at or below its own threshold. Do not
    re-add it, and in particular do not re-add it with a lower bound: the exact
    identity ``dX_t == velocity * signed_dt`` is already pinned below, at both
    ``force_unconditional`` settings and in both directions. See
    decisions.md D-030.

``TestTheThreeUndefinedQuantitiesStayUndefined``
    The anti-regression partner: the whole point of the override is that the
    sampling path touches neither ``sigma`` nor ``phi`` nor ``C``, so a "fix"
    that made them return ``0.0`` would also make sampling work -- and would be
    wrong. Pinned here as well as in ``test_the_sde_closed_forms.py`` because
    this file is where the temptation now lives.
"""

from typing import Any, Dict, List, Optional

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    FlowMatchingODE,
    create_bridge_sde,
)

from ._ditxa_helpers import activate, batch, np_

NUM_STEPS = 4
BATCH = 2


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built, non-degenerate ``tiny`` model.

    ``activate`` is not optional: a freshly built ``DiTXA`` emits the EXACT zero
    tensor, so every velocity would be zero, every trajectory would equal its
    start, and the sign-flip arm below would compare ``0`` with ``0``.
    """
    m = DiTXA.from_variant("tiny", label_seed=17)
    m(batch(m, batch_size=BATCH))
    return activate(m, seed=5)


@pytest.fixture(scope="module")
def anchor(model) -> Dict[str, Any]:
    """The anchored endpoint and labels every simulation here starts from."""
    rng = np.random.default_rng(20260902)
    shape = (BATCH, model.input_size, model.input_size, model.in_channels)
    return {
        "x_start": rng.normal(size=shape).astype("float32"),
        "y": np.zeros((BATCH,), dtype="int32"),
    }


class RecordingNetwork:
    """A pass-through wrapper that records every input dict the SDE builds.

    Wraps a real :class:`DiTXA` rather than faking one, so the recorded dict is
    the dict the model would actually have consumed -- a stub returning a
    constant would also pass a shape arm while proving nothing about whether the
    keys are the ones ``DiTXA.call`` reads.

    Interface contract: callable as ``self(inputs, training=...)`` and
    ``self.forward_with_cfg(inputs, cfg_scale=..., training=...)``, i.e. exactly
    the surface :meth:`BridgeSDE._evaluate_score` uses. Returns whatever the
    wrapped model returns. ``calls`` accumulates the input dicts in order;
    ``cfg_calls`` counts the guided route separately.
    """

    def __init__(self, wrapped: keras.Model) -> None:
        self.wrapped = wrapped
        self.calls: List[Dict[str, Any]] = []
        self.cfg_calls = 0

    def __call__(self, inputs: Dict[str, Any], training: Optional[bool] = None):
        self.calls.append(dict(inputs))
        return self.wrapped(inputs, training=training)

    def forward_with_cfg(
        self,
        inputs: Dict[str, Any],
        cfg_scale: float,
        training: Optional[bool] = None,
    ):
        self.cfg_calls += 1
        self.calls.append(dict(inputs))
        return self.wrapped.forward_with_cfg(
            inputs, cfg_scale=cfg_scale, training=training
        )


def simulate(sde, network, anchor, **kwargs) -> np.ndarray:
    """Run ``sde.simulate`` on the shared anchor and return NumPy."""
    return np_(
        sde.simulate(
            x_start=anchor["x_start"],
            num_steps=kwargs.pop("num_steps", NUM_STEPS),
            score_network=network,
            y=anchor["y"],
            **kwargs,
        )
    )


# ---------------------------------------------------------------------
# It samples at all
# ---------------------------------------------------------------------


class TestItSamples:
    """The claim the old suite could not make: ``simulate`` runs and is finite."""

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("force_unconditional", [False, True])
    def test_simulate_runs_end_to_end(
        self, model, anchor, reverse, force_unconditional
    ):
        sde = FlowMatchingODE(force_unconditional=force_unconditional)
        out = simulate(sde, model, anchor, reverse=reverse)
        assert out.shape == anchor["x_start"].shape
        assert np.all(np.isfinite(out)), "the flow produced non-finite states"
        # Anti-vacuity: a trajectory that never left its start would satisfy
        # every assertion above while integrating nothing.
        moved = float(np.max(np.abs(out - anchor["x_start"])))
        assert moved > 0.0, (
            "the trajectory is bit-identical to its start; nothing was "
            f"integrated (max|delta| = {moved})"
        )

    def test_return_all_gives_one_state_per_step(self, model, anchor):
        sde = FlowMatchingODE()
        states = sde.simulate(
            x_start=anchor["x_start"],
            num_steps=NUM_STEPS,
            score_network=model,
            y=anchor["y"],
            return_all=True,
        )
        assert len(states) == NUM_STEPS
        assert all(np.all(np.isfinite(np_(s))) for s in states)

    @pytest.mark.parametrize("ode", [False, True])
    def test_the_ode_flag_is_accepted_and_ignored(self, model, anchor, ode):
        """Upstream's override takes ``ode``/``x_start`` and references neither.

        A deterministic flow has no separate probability-flow branch: the
        transport already IS the ODE, and the base class's ``ode=True`` branch
        exists only to subtract an analytic base score that divides by ``C`` --
        which this class raises on. So ``ode=True`` must not change the result
        and must not raise. Matched to ``reference/sde_utils_sde.py:69-82``.
        """
        sde = FlowMatchingODE()
        out = simulate(sde, model, anchor, ode=ode)
        reference = simulate(FlowMatchingODE(), model, anchor, ode=False)
        assert np.array_equal(out, reference), (
            "ode=True changed the flow-matching trajectory; upstream's "
            "FlowMatchingODE.dX_t ignores the flag entirely"
        )

    def test_the_factory_builds_a_sampleable_one(self, model, anchor):
        """``create_bridge_sde('flow_matching')`` is a live route, not a stub."""
        sde = create_bridge_sde("flow_matching", force_unconditional=True)
        assert isinstance(sde, FlowMatchingODE)
        assert np.all(np.isfinite(simulate(sde, model, anchor)))


# ---------------------------------------------------------------------
# CFG rejection
# ---------------------------------------------------------------------


class TestForceUnconditionalRejectsCFG:
    """``cfg_scale != 0`` under ``force_unconditional`` is an error, not a no-op."""

    @pytest.mark.parametrize("cfg_scale", [1.5, -1.5, 1e-6])
    def test_it_raises(self, model, anchor, cfg_scale):
        sde = FlowMatchingODE(force_unconditional=True)
        with pytest.raises(ValueError, match="force_unconditional"):
            simulate(sde, model, anchor, cfg_scale=cfg_scale)

    def test_zero_is_allowed(self, model, anchor):
        """The boundary: the rejection is ``!= 0``, not ``> 0`` and not ``>= 0``."""
        sde = FlowMatchingODE(force_unconditional=True)
        assert np.all(np.isfinite(simulate(sde, model, anchor, cfg_scale=0.0)))

    def test_unforced_still_accepts_cfg(self, model, anchor):
        """Anti-vacuity: the raise is the KNOB's doing, not a blanket ban on CFG."""
        network = RecordingNetwork(model)
        sde = FlowMatchingODE(force_unconditional=False)
        out = simulate(sde, network, anchor, cfg_scale=1.5)
        assert np.all(np.isfinite(out))
        assert network.cfg_calls == NUM_STEPS, (
            "cfg_scale > 0 did not route through forward_with_cfg on every step; "
            f"got {network.cfg_calls} guided calls out of {NUM_STEPS}"
        )


# ---------------------------------------------------------------------
# The knob actually forces the branch
# ---------------------------------------------------------------------


def mask_of(call: Dict[str, Any]) -> Optional[np.ndarray]:
    """The ``cond_mask`` a recorded call carried, or ``None`` if absent."""
    return None if "cond_mask" not in call else np_(call["cond_mask"])


class TestForceUnconditionalForcesTheUnconditionalBranch:
    """Recorded at the network boundary, because the trajectory cannot tell."""

    @pytest.mark.parametrize("reverse", [False, True])
    def test_the_network_receives_an_all_false_cond_mask(
        self, model, anchor, reverse
    ):
        network = RecordingNetwork(model)
        sde = FlowMatchingODE(force_unconditional=True)
        simulate(sde, network, anchor, reverse=reverse)

        assert len(network.calls) == NUM_STEPS
        for i, call in enumerate(network.calls):
            mask = mask_of(call)
            assert mask is not None, (
                f"step {i} carried no cond_mask key at all, so DiTXA read it as "
                "all-ones and the conditioning stream was never masked off"
            )
            assert mask.shape == (BATCH,), mask.shape
            assert np.all(mask == 0.0), (
                f"step {i} passed a cond_mask that is not all-false: {mask}"
            )

    def test_direction_is_forward_even_when_simulating_in_reverse(
        self, model, anchor
    ):
        """One shared velocity field; the outer ``reverse`` only flips ``dt``.

        The subtle half of the port. Threading the outer ``reverse`` into the
        network instead selects the reverse conditioning embedder and the
        reverse ``t_cond`` -- a DIFFERENT field, with identical shapes, finite
        values and perfectly plausible trajectories.
        """
        network = RecordingNetwork(model)
        sde = FlowMatchingODE(force_unconditional=True)
        simulate(sde, network, anchor, reverse=True)

        for i, call in enumerate(network.calls):
            direction = np_(call["direction"])
            assert np.all(direction == 0.0), (
                f"step {i} of a REVERSE forced-unconditional run passed "
                f"direction = {direction}; upstream hard-codes reverse=False "
                "here and lets the sign of dt carry the direction"
            )

    def test_unforced_does_not_force_it(self, model, anchor):
        """The anti-vacuity partner. Without the knob, nothing is masked."""
        network = RecordingNetwork(model)
        sde = FlowMatchingODE(force_unconditional=False)
        simulate(sde, network, anchor, reverse=True)

        assert len(network.calls) == NUM_STEPS
        for i, call in enumerate(network.calls):
            assert mask_of(call) is None, (
                f"step {i} of an UNFORCED run still passed a cond_mask; the "
                "knob is not what produces the masking"
            )
            assert np.all(np_(call["direction"]) == 1.0), (
                f"step {i} of an UNFORCED reverse run did not pass the reverse "
                "direction to the network"
            )

    def test_the_two_settings_produce_different_trajectories(self, model, anchor):
        """The consequence, downstream of the recorded cause.

        Kept BESIDE the recording arms, not instead of them: this difference is
        also what a ``cond_mask`` of the wrong shape, or a mask applied to the
        wrong stream, would produce.
        """
        forced = simulate(
            FlowMatchingODE(force_unconditional=True), model, anchor
        )
        free = simulate(
            FlowMatchingODE(force_unconditional=False), model, anchor
        )
        assert float(np.max(np.abs(forced - free))) > 1e-6, (
            "forcing the unconditional branch changed nothing about the "
            "trajectory"
        )


# ---------------------------------------------------------------------
# The sign flip
# ---------------------------------------------------------------------


class TestTheSignFlipIsObservable:
    """``signed_dt = -dt if reverse else dt``, pinned by direction of travel."""

    @pytest.mark.parametrize("force_unconditional", [True, False])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_the_increment_is_the_velocity_times_a_SIGNED_dt(
        self, model, anchor, force_unconditional, reverse
    ):
        """``dX_t == velocity * (-dt if reverse else dt)``, exactly, at both settings.

        The velocity is not re-derived here: it is obtained by replaying the
        input dict the SDE *itself* handed the network (recorded at the
        boundary) back through the same model. So the only thing this arm can
        fail on is the sign and magnitude ``dX_t`` applied to that network
        output -- which is ``signed_dt``, the whole claim.

        **Why not the obvious ``simulate``-level version.** This arm replaces a
        "more than 0.9 of the entries of a forward and a reverse increment are
        opposed in sign" arm that was **flaky** (found at step 12; fixed at step
        8.2). Two separate reasons, and neither is fixable by moving the
        threshold:

        * *Structural*: through :meth:`simulate` the two directions do not share
          a time -- step 0 forward sits at ``t = 0`` and step 0 reverse at
          ``t = 1`` -- so the two velocity fields are evaluated at different
          ``t`` (and, unforced, on different conditioning), and the increments
          are only APPROXIMATELY opposed. The statistic's true mean is nowhere
          near 1.
        * *RNG*: nothing seeds the Glorot draws of the model fixture (``activate``
          only replaces the all-zero adaLN weights), so the statistic is a fresh
          draw per process and shifts with test order. Measured over 24
          independent global seeds: forced ``min 0.9766 / mean 0.9893 / max
          0.9980``, unforced ``min 0.8555 / mean 0.9204 / max 0.9609``, with
          **20.8% of draws at or below the 0.9 threshold**. The bound sat inside
          its own noise band.

        A bound that survives that distribution would have to sit near 0.8, i.e.
        assert something much weaker than the exact identity below already
        proves -- so the marginal statistic is gone rather than re-tuned.
        """
        network = RecordingNetwork(model)
        sde = FlowMatchingODE(force_unconditional=force_unconditional)
        dt = 0.25
        increment = np_(
            sde.dX_t(
                x_t=anchor["x_start"],
                t=np.full((BATCH,), 0.35, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=dt,
                score_network=network,
                reverse=reverse,
            )
        )

        assert len(network.calls) == 1, (
            f"expected exactly one network call per dX_t, got "
            f"{len(network.calls)}"
        )
        velocity = np_(model(network.calls[0], training=False))
        assert float(np.max(np.abs(velocity))) > 0.0, (
            "the recorded velocity is the exact zero tensor, so every sign "
            "below would be vacuous"
        )

        signed_dt = -dt if reverse else dt
        assert np.allclose(increment, signed_dt * velocity, atol=1e-6, rtol=0.0), (
            f"reverse={reverse} did not scale the network's velocity by "
            f"{signed_dt}: max|delta| = "
            f"{float(np.max(np.abs(increment - signed_dt * velocity)))}"
        )
        # Anti-vacuity: the flip is what makes the two branches differ at all.
        # Without it the SAME assertion would hold with the opposite sign.
        assert not np.allclose(
            increment, -signed_dt * velocity, atol=1e-6, rtol=0.0
        ), (
            "the increment satisfies BOTH signs of dt, so this arm cannot see "
            "signed_dt at all (velocity is degenerate or dt is 0)"
        )

    def test_dX_t_at_a_FIXED_time_negates_exactly(self, model, anchor):
        """The strongest form of the claim -- and why ``simulate`` cannot make it.

        MEASURED, and the reason this arm calls ``dX_t`` directly: through
        ``simulate`` the two directions do NOT share a time. Step 0 of a forward
        run sits at ``t = 0`` and step 0 of a reverse run at ``t = 1``, so the
        network is evaluated on two different inputs and the increments are only
        approximately opposed (``max|fwd + rev| = 0.241`` against entries of
        magnitude ~2 -- enough for the sign arm above, nowhere near a negation).
        Holding ``t`` fixed isolates ``signed_dt`` as the single difference, and
        then anything but an exact negation is a different mechanism.
        """
        start = anchor["x_start"]
        common = dict(
            x_t=start,
            t=np.full((BATCH,), 0.35, dtype="float32"),
            x_cond=start,
            y=anchor["y"],
            dt=0.25,
            score_network=model,
        )
        forced = FlowMatchingODE(force_unconditional=True)
        fwd = np_(forced.dX_t(reverse=False, **common))
        rev = np_(forced.dX_t(reverse=True, **common))
        assert float(np.max(np.abs(fwd))) > 0.0
        assert np.allclose(fwd, -rev, atol=1e-6, rtol=0.0), (
            f"max|fwd + rev| = {float(np.max(np.abs(fwd + rev)))}"
        )

        # And the partner claim, which is the reason the exact form is only
        # available when forced: WITHOUT forcing, `reverse` also selects the
        # reverse conditioning embedder and the reverse `t_cond`, so the two
        # increments are genuinely two different velocity fields and must NOT
        # negate. Asserting that keeps the arm above from being read as "flow
        # matching is direction-symmetric".
        free = FlowMatchingODE(force_unconditional=False)
        fwd_free = np_(free.dX_t(reverse=False, **common))
        rev_free = np_(free.dX_t(reverse=True, **common))
        assert not np.allclose(fwd_free, -rev_free, atol=1e-6, rtol=0.0), (
            "an UNFORCED reverse step negated the forward step exactly, so "
            "`reverse` never reached the network's `direction` input"
        )


# ---------------------------------------------------------------------
# What the override must NOT have quietly fixed
# ---------------------------------------------------------------------


class TestTheThreeUndefinedQuantitiesStayUndefined:
    """Making sampling work by defining ``sigma`` would be the wrong fix.

    The override earns its existence by touching none of the three. If a future
    change makes ``sigma`` return ``0.0`` so the inherited ``dX_t`` "works", the
    sampler silently integrates a zero-diffusion bridge and the bridge score
    targets divide by a zero ``C`` -- plausible numbers, no exception.
    """

    @pytest.mark.parametrize("name", ["sigma", "phi", "C"])
    def test_they_still_raise(self, name):
        sde = FlowMatchingODE()
        args = {
            "sigma": (np.array([0.5], dtype="float32"),),
            "phi": (0.0, np.array([0.5], dtype="float32")),
            "C": (0.0, np.array([0.5], dtype="float32"), np.array([0.5], dtype="float32")),
        }[name]
        with pytest.raises(NotImplementedError):
            getattr(sde, name)(*args)

    def test_sampling_never_calls_them(self, model, anchor):
        """Belt and braces: the raise is the proof, so make it observable.

        Replacing the three with recorders rather than trusting that the raise
        would have surfaced -- an exception swallowed by a broad ``except`` in
        some future refactor would leave this file green otherwise.
        """
        touched = []

        class Watched(FlowMatchingODE):
            def sigma(self, t):
                touched.append("sigma")
                return keras.ops.zeros_like(t)

            def phi(self, start, end):
                touched.append("phi")
                return keras.ops.ones_like(end)

            def C(self, start, t_a, t_b):
                touched.append("C")
                return keras.ops.ones_like(t_a)

        simulate(Watched(force_unconditional=True), model, anchor)
        simulate(Watched(force_unconditional=False), model, anchor, reverse=True)
        assert touched == [], (
            f"the flow-matching sampling path called {sorted(set(touched))}; it "
            "must not depend on any base-process quantity"
        )
