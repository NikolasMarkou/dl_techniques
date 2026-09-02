"""The sampler's first step is always stochastic, even when ``ode=True``.

Invariant "the ``ode and i > 0`` first-step skip" from the plan's boundary
cases, plus the three other properties of ``BridgeSDE.dX_t`` / ``simulate`` that
have no shape symptom.

**Why the skip exists.** The probability-flow branch subtracts an *analytic*
base-process score from the network's prediction, and that analytic score
divides by ``C(start, t, t)`` for the endpoint the trajectory is anchored to.
``C`` is exactly zero there -- ``C(1,1,1) = 0`` for a reverse run starting at
``t = 1``, ``C(0,0,0) = 0`` for a forward run starting at ``t = 0``. Step ``0``
therefore divides by zero, and because the state is fed forward the ``nan``
poisons every later step. Upstream (``sde_utils_sde.py:50``) passes
``ode=ode and i > 0``, which takes the plain SDE branch exactly once, at the
start. This port reproduces the skip; it does not "fix" it into a pure-ODE
start.

**Why finiteness alone is not enough.** A pure-ODE start with a clamp or an
``eps`` would also be finite everywhere, so :class:`TestTheFirstStepIsStochastic`
pins the skip's *observable consequence* instead: with ``ode=True`` the result
still depends on the RNG seed, because one real Brownian increment was injected.
:class:`TestTheSingularityIsReal` then pins the reason, by taking the ODE branch
at the anchored endpoint directly and asserting it is NOT finite -- an arm that
also reddens if the reverse/forward analytic-score assignment is un-swapped,
since ``score_target_reverse`` is perfectly finite at ``t = 1``.
"""

from typing import Any, Dict, Optional

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.bridge_process import (
    score_target_forward,
    score_target_reverse,
)
from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    CosineDecayingVolatilitySDE,
    UniformVolatilitySDE,
    _expand_like,
)

from ._ditxa_helpers import activate, batch, np_

NUM_STEPS = 6


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built, non-degenerate ``tiny`` model.

    ``activate`` matters here: a freshly initialised ``DiTXA`` predicts the exact
    zero tensor, which makes the learned score identically zero and every
    "the ODE branch differs from the SDE branch" arm a statement about noise
    alone.
    """
    m = DiTXA.from_variant("tiny", label_seed=17)
    m(batch(m, batch_size=2))
    return activate(m, seed=9)


@pytest.fixture(scope="module")
def anchor(model) -> Dict[str, Any]:
    """The anchored endpoint and the labels a simulation needs."""
    rng = np.random.default_rng(4242)
    shape = (2, model.input_size, model.input_size, model.in_channels)
    return {
        "x_start": rng.normal(size=shape).astype("float32"),
        "y": np.zeros((2,), dtype="int32"),
    }


@pytest.fixture(scope="module")
def sde() -> CosineDecayingVolatilitySDE:
    """A driftless variant -- the ODE branch requires ``A == 0``."""
    return CosineDecayingVolatilitySDE()


def _simulate(sde, model, anchor, **kwargs) -> np.ndarray:
    return np_(
        sde.simulate(
            x_start=anchor["x_start"],
            num_steps=kwargs.pop("num_steps", NUM_STEPS),
            score_network=model,
            y=anchor["y"],
            **kwargs,
        )
    )


class TestAFullOdeRunIsFinite:
    """SC-7: the anchored-endpoint run completes with no non-finite value."""

    @pytest.mark.parametrize("reverse", [True, False])
    def test_the_whole_trajectory_is_finite(self, sde, model, anchor, reverse):
        out = _simulate(sde, model, anchor, reverse=reverse, ode=True, seed=3)
        non_finite = int(np.sum(~np.isfinite(out)))
        assert non_finite == 0, f"{non_finite} non-finite entries"
        assert np.abs(out).max() > 0.0

    def test_every_intermediate_state_is_finite(self, sde, model, anchor):
        """``return_all`` -- so a mid-trajectory blow-up cannot hide in the tail."""
        states = sde.simulate(
            x_start=anchor["x_start"],
            num_steps=NUM_STEPS,
            score_network=model,
            y=anchor["y"],
            reverse=True,
            ode=True,
            return_all=True,
            seed=3,
        )
        assert len(states) == NUM_STEPS
        for i, state in enumerate(states):
            assert np.all(np.isfinite(np_(state))), f"step {i} went non-finite"


class TestTheFirstStepIsStochastic:
    """The skip's observable consequence, not merely its absence of ``nan``."""

    def test_an_ode_run_still_depends_on_the_seed(self, sde, model, anchor):
        """One real Brownian increment is injected, at step 0 only.

        A "repaired" pure-ODE start would make this deterministic, so this arm
        reddens under the repair the docstring forbids -- which a finiteness
        arm alone would not.
        """
        first = _simulate(sde, model, anchor, reverse=True, ode=True, seed=1)
        second = _simulate(sde, model, anchor, reverse=True, ode=True, seed=2)
        assert not np.allclose(first, second)

    def test_an_ode_run_is_reproducible_at_a_fixed_seed(
        self, sde, model, anchor
    ):
        first = _simulate(sde, model, anchor, reverse=True, ode=True, seed=5)
        second = _simulate(sde, model, anchor, reverse=True, ode=True, seed=5)
        np.testing.assert_array_equal(first, second)


class TestTheSingularityIsReal:
    """Taking the ODE branch AT the anchored endpoint really does divide by 0."""

    @pytest.mark.parametrize(
        "reverse,t_value", [(True, 1.0), (False, 0.0)]
    )
    def test_the_ode_branch_at_the_anchor_is_not_finite(
        self, sde, model, anchor, reverse, t_value
    ):
        """This is what ``ode=ode and i > 0`` prevents.

        It is also the discriminator for the SWAPPED analytic-score assignment:
        at ``t = 1`` the reverse branch's ``score_target_forward`` divides by
        ``C(1,1,1) = 0``, while ``score_target_reverse`` divides by the perfectly
        finite ``C(0,1,1)``. Un-swapping the assignment makes this arm finite.
        """
        step = np_(
            sde.dX_t(
                x_t=anchor["x_start"],
                t=np.full((2,), t_value, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=1.0 / NUM_STEPS,
                score_network=model,
                reverse=reverse,
                ode=True,
                x_start=anchor["x_start"],
            )
        )
        assert not np.all(np.isfinite(step))

    @pytest.mark.parametrize("reverse", [True, False])
    def test_the_sde_branch_at_the_same_point_is_finite(
        self, sde, model, anchor, reverse
    ):
        """Anti-vacuity: the endpoint itself is not the problem, the branch is."""
        step = np_(
            sde.dX_t(
                x_t=anchor["x_start"],
                t=np.full((2,), 1.0 if reverse else 0.0, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=1.0 / NUM_STEPS,
                score_network=model,
                reverse=reverse,
                ode=False,
                seed=0,
            )
        )
        assert np.all(np.isfinite(step))


class TestTheAnchoringIsTheSwappedOne:
    """``reverse`` uses the FORWARD target and vice versa (``dX_t`` docstring).

    At sampling time ``x_start`` is whichever endpoint the trajectory is
    anchored to, not the training-time role, so the pairing is the opposite of
    ``dsm_loss``'s. Verified against ``reference/sde_utils_sde.py:22-29``.
    """

    @staticmethod
    def _increment(sde, x_t, t, dt, score, analytic):
        sigma_t = _expand_like(keras.ops.cast(sde.sigma(t), "float32"), x_t)
        return np_(0.5 * sigma_t ** 2 * (score - analytic) * dt)

    @pytest.mark.parametrize("reverse", [True, False])
    def test_the_increment_uses_the_swapped_analytic_target(
        self, sde, model, anchor, reverse
    ):
        """Pinned at an interior ``t`` where BOTH candidates are finite.

        The endpoint arms above cannot separate the two by finiteness alone in
        the forward direction, so this arm separates them by value in both.

        ``x_t`` is drawn INDEPENDENTLY of ``x_start`` on purpose. Both analytic
        targets are proportional to ``x_start - x_t``, so evaluating them at
        ``x_t == x_start`` makes both of them the exact zero tensor and the arm
        vacuous -- which is what the anti-vacuity assertion below caught on its
        first run.
        """
        rng = np.random.default_rng(31337)
        x_t = keras.ops.convert_to_tensor(
            rng.normal(size=anchor["x_start"].shape).astype("float32")
        )
        t = np.full((2,), 0.5, dtype="float32")
        dt = 1.0 / NUM_STEPS
        direction = np.full((2,), 1.0 if reverse else 0.0, dtype="float32")
        score = model(
            {
                "x_t": x_t,
                "t": t,
                "y": anchor["y"],
                "x_cond": anchor["x_start"],
                "direction": direction,
            },
            training=False,
        )
        swapped = self._increment(
            sde,
            x_t,
            t,
            dt,
            score,
            keras.ops.cast(
                (score_target_forward if reverse else score_target_reverse)(
                    sde, x_t, t, anchor["x_start"]
                ),
                "float32",
            ),
        )
        unswapped = self._increment(
            sde,
            x_t,
            t,
            dt,
            score,
            keras.ops.cast(
                (score_target_reverse if reverse else score_target_forward)(
                    sde, x_t, t, anchor["x_start"]
                ),
                "float32",
            ),
        )
        assert not np.allclose(swapped, unswapped), (
            "the two candidate targets agree here; the arm cannot discriminate"
        )

        actual = np_(
            sde.dX_t(
                x_t=x_t,
                t=t,
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=dt,
                score_network=model,
                reverse=reverse,
                ode=True,
                x_start=anchor["x_start"],
            )
        )
        np.testing.assert_allclose(actual, swapped, rtol=0, atol=1e-6)


class TestTheStochasticBranchIsLive:
    """``ode=True`` is a different code path, not dead configuration."""

    def test_two_stochastic_runs_at_one_seed_are_identical(
        self, sde, model, anchor
    ):
        first = _simulate(sde, model, anchor, reverse=True, ode=False, seed=8)
        second = _simulate(sde, model, anchor, reverse=True, ode=False, seed=8)
        np.testing.assert_array_equal(first, second)

    def test_the_ode_trajectory_differs_from_the_stochastic_one(
        self, sde, model, anchor
    ):
        stochastic = _simulate(
            sde, model, anchor, reverse=True, ode=False, seed=8
        )
        deterministic = _simulate(
            sde, model, anchor, reverse=True, ode=True, seed=8
        )
        assert not np.allclose(stochastic, deterministic)

    def test_consecutive_steps_draw_independent_noise(self, model, anchor):
        """D-019: a bare integer seed would make every step's noise IDENTICAL.

        ``keras.random.*`` is stateless given an int, so passing the caller's
        seed straight down to the per-step draw repeats one increment
        ``num_steps`` times -- finite, seed-dependent and reproducible, i.e.
        invisible to every other arm in this file. ``simulate`` promotes the int
        to one ``SeedGenerator`` outside the loop; this arm is what says so.

        Run with a zero-returning network and a constant-volatility process, so
        each increment is exactly ``sigma * sqrt(dt) * noise_i``.
        """
        constant = UniformVolatilitySDE(A=0.0, K=1.0)
        states = constant.simulate(
            x_start=anchor["x_start"],
            num_steps=3,
            score_network=_ZeroScoreStub(),
            y=anchor["y"],
            reverse=True,
            return_all=True,
            seed=99,
        )
        increments = [np_(states[0]) - anchor["x_start"]]
        for previous, current in zip(states, states[1:]):
            increments.append(np_(current) - np_(previous))
        for i in range(len(increments) - 1):
            assert not np.allclose(increments[i], increments[i + 1]), (
                f"steps {i} and {i + 1} drew the same Brownian increment"
            )


class _ZeroScoreStub:
    """A network that predicts an exactly zero score, and counts its calls."""

    def __init__(self) -> None:
        self.plain_calls = 0
        self.cfg_calls = 0
        self.directions = []

    def __call__(
        self, inputs: Dict[str, Any], training: Optional[bool] = None
    ) -> Any:
        self.plain_calls += 1
        self.directions.append(np_(inputs["direction"]))
        return keras.ops.zeros_like(inputs["x_t"])

    def forward_with_cfg(
        self,
        inputs: Dict[str, Any],
        cfg_scale: float,
        training: Optional[bool] = None,
    ) -> Any:
        self.cfg_calls += 1
        return keras.ops.zeros_like(inputs["x_t"])


class TestTheCfgGateRoutes:
    """``cfg_scale > 0`` is a routing decision, and it costs a forward pass."""

    @pytest.mark.parametrize(
        "cfg_scale,expected_plain,expected_cfg", [(0.0, 4, 0), (2.5, 0, 4)]
    )
    def test_the_gate_picks_the_branch(
        self, sde, anchor, cfg_scale, expected_plain, expected_cfg
    ):
        """Counted, not inferred: at ``s = 0`` the two formulas agree in VALUE.

        So a gate deleted in favour of always calling ``forward_with_cfg`` is
        numerically invisible and only doubles the cost. Only a call count sees
        it.
        """
        stub = _ZeroScoreStub()
        sde.simulate(
            x_start=anchor["x_start"],
            num_steps=4,
            score_network=stub,
            y=anchor["y"],
            reverse=True,
            cfg_scale=cfg_scale,
            seed=2,
        )
        assert stub.plain_calls == expected_plain
        assert stub.cfg_calls == expected_cfg

    def test_the_direction_flag_matches_the_simulation_direction(
        self, sde, anchor
    ):
        """``reverse`` is a Python bool here and a ``(B,)`` tensor at the model.

        D-005 turned upstream's branch selector into per-sample data; the
        sampler is the one place that conversion happens, and getting it
        backwards would silently sample the wrong direction's model.
        """
        for reverse, expected in ((True, 1.0), (False, 0.0)):
            stub = _ZeroScoreStub()
            sde.simulate(
                x_start=anchor["x_start"],
                num_steps=2,
                score_network=stub,
                y=anchor["y"],
                reverse=reverse,
                seed=2,
            )
            for seen in stub.directions:
                np.testing.assert_array_equal(
                    seen, np.full((2,), expected, dtype=seen.dtype)
                )


class TestTheSamplerContract:
    """The loud failures ``simulate`` / ``dX_t`` owe their callers."""

    def test_a_missing_score_network_raises(self, sde, anchor):
        with pytest.raises(ValueError, match="score_network"):
            sde.dX_t(
                x_t=anchor["x_start"],
                t=np.full((2,), 0.5, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=0.1,
            )

    def test_a_missing_label_raises(self, sde, model, anchor):
        with pytest.raises(ValueError, match="y is required"):
            sde.simulate(
                x_start=anchor["x_start"], num_steps=2, score_network=model
            )

    def test_a_non_positive_step_count_raises(self, sde, model, anchor):
        with pytest.raises(ValueError, match="num_steps"):
            sde.simulate(
                x_start=anchor["x_start"],
                num_steps=0,
                score_network=model,
                y=anchor["y"],
            )

    def test_the_ode_branch_refuses_a_drifting_process(self, model, anchor):
        """Upstream asserts ``A == 0``; a drifting OU process has no such branch."""
        drifting = UniformVolatilitySDE(A=1.5, K=1.0)
        with pytest.raises(ValueError, match="driftless"):
            drifting.dX_t(
                x_t=anchor["x_start"],
                t=np.full((2,), 0.5, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=0.1,
                score_network=model,
                ode=True,
                x_start=anchor["x_start"],
            )

    def test_the_ode_branch_refuses_a_missing_anchor(self, sde, model, anchor):
        with pytest.raises(ValueError, match="x_start"):
            sde.dX_t(
                x_t=anchor["x_start"],
                t=np.full((2,), 0.5, dtype="float32"),
                x_cond=anchor["x_start"],
                y=anchor["y"],
                dt=0.1,
                score_network=model,
                ode=True,
            )

    def test_x_cond_defaults_to_the_anchor(self, sde, model, anchor):
        """Upstream's ``if x_cond is None: x_cond = x_start``."""
        explicit = _simulate(
            sde,
            model,
            anchor,
            reverse=True,
            seed=6,
            num_steps=2,
            x_cond=anchor["x_start"],
        )
        implicit = _simulate(
            sde, model, anchor, reverse=True, seed=6, num_steps=2
        )
        np.testing.assert_array_equal(explicit, implicit)

    def test_the_time_grid_runs_the_right_way(self, sde, anchor):
        """Reverse walks ``1 -> 0``; forward walks ``0 -> 1``."""
        for reverse, first, last in ((True, 1.0, 0.25), (False, 0.0, 0.75)):
            stub = _ZeroScoreStub()

            seen = []
            original = sde.__class__.sigma

            def recording(self, t, _seen=seen):
                _seen.append(float(np_(t)[0]))
                return original(self, t)

            sde.__class__.sigma = recording
            try:
                sde.simulate(
                    x_start=anchor["x_start"],
                    num_steps=4,
                    score_network=stub,
                    y=anchor["y"],
                    reverse=reverse,
                    seed=1,
                )
            finally:
                sde.__class__.sigma = original
            assert seen[0] == pytest.approx(first)
            assert seen[-1] == pytest.approx(last)
