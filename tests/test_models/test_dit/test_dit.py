"""The comprehensive ``DiT`` suite (v2 §16.3): degenerate configs, precision,
XLA, serialization across the config space, and knob liveness.

**What is deliberately NOT here.** Three §16.3 items were already discharged by
earlier steps of this port, and a second copy of a guard is the hand-maintained
lockstep smell this package has already been bitten by once (decisions.md
D-019). Each is named with the file that owns it, so a reader can check rather
than assume:

===================================================  =======================================
§16.3 item                                           owned by
===================================================  =======================================
``.keras`` round trip on VALUES, ``training=False``  ``test_dit_model.py::TestSerialization``
weight values at ``atol=0.0`` before the first call  ``test_dit_model.py::TestSerialization``
``pos_embed`` survives a build through a parent      ``test_dit_model.py::TestThePosEmbed…``
public exports are all constructed                   ``test_the_package_surface.py::TestTheExportsAreConstructible``
explicit ``build()`` parity + anti-vacuity           ``test_build_materialization_parity.py``
per-variable gradient flow after a real step         ``test_gradient_flow.py``
ASCII diagrams in the class docstrings               ``test_the_package_surface.py`` (D-019)
===================================================  =======================================

What this file adds is the part those cannot make: the same claims across the
**config space** rather than at one point in it, and the four regimes a single
tiny float32 eager forward pass never enters -- ``mixed_float16``, ``float64``,
XLA, and the degenerate geometries (``depth=1``, a one-token grid).

**The premise every arm here is written against.** A fresh ``DiT`` emits the
EXACT zero tensor, so "the output changed" is unsatisfiable and "the outputs
agree" is vacuously true at initialisation. Every output-level arm below calls
:func:`_dit_helpers.activate` first and asserts non-degeneracy before it asserts
anything else. See ``_dit_helpers.py`` for the full statement of the trap.

**Measured, and stated as findings rather than left implicit:**

* The three precision arms are **genuinely distinct, not inert**: they compute in
  ``float32`` / ``float16`` / ``float64`` respectively, and the first two hold
  BIT-IDENTICAL variables (``mixed_float16`` keeps float32 masters) while their
  outputs differ by ``2.2e-03``. That difference is the evidence the policy
  reaches the arithmetic; without it "finite under three policies" would be one
  claim stated three times.
* ``pos_embed`` is created with an explicit ``dtype="float32"`` and therefore
  stays float32 under EVERY policy, including ``float64``. ``DiT.call`` casts it
  at the point of use. This is recorded as a positive assertion below, not
  fixed: under a float64 policy the frozen table carries float32 precision, and
  changing that is a behaviour change to shipped code, out of scope for a test
  step.
* XLA (``jit_compile=True``) and eager agree on ``predict`` to ``1.43e-06`` at
  an output scale of ``6.50``, i.e. 1.8 ``eps_f32`` -- float32 re-association
  noise and nothing more. The bound is DERIVED from the contraction lengths via
  ``tests/numerics.reassociation_atol`` (``1.16e-04`` here, 81x headroom), never
  pasted. Measured in BOTH collection regimes, because
  ``tests/test_layers/test_attention/test_linear_attention.py`` disables TF32
  process-globally at import and a tolerance-sensitive arm can swing by orders
  of magnitude depending on what was collected first: **file alone
  ``1.430511e-06``, co-collected behind that module ``1.430511e-06``** --
  bit-identical, because with no CUDA device present TF32 is inert anyway.
"""

import inspect
from typing import Any, Dict, Iterator, List, Set

import keras
import numpy as np
import pytest

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.model import DiT

from ...numerics import reassociation_atol
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
    weight_signature,
)
from ..smoke_contract_oracle import assert_contract_rejects_a_broken_forward
from ._dit_helpers import (
    BATCH,
    TINY,
    activate,
    built_model,
    ddpm_training_batch,
    dit_config,
    np_,
    tiny_inputs,
    tiny_model,
)

# ---------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------


def assert_finite_and_shaped(out: Any, config: Dict[str, Any], batch: int) -> np.ndarray:
    """Every entry finite, and the shape DERIVED from the config.

    Interface contract: pure; converts and returns the array so a caller can
    make a further claim on it. The expectation is computed from ``config``,
    never read back from the model -- a model that ignored ``learn_sigma`` would
    otherwise agree with itself.
    """
    array = np_(out)
    channels = config["in_channels"] * (2 if config["learn_sigma"] else 1)
    expected = (batch, config["input_size"], config["input_size"], channels)
    assert tuple(array.shape) == expected, (tuple(array.shape), expected)
    non_finite = int((~np.isfinite(array)).sum())
    assert non_finite == 0, (
        f"{non_finite} of {array.size} output entries are non-finite"
    )
    return array


def assert_not_degenerate(array: np.ndarray, label: str) -> None:
    """A comparison over the exact zero tensor proves nothing."""
    assert np.any(np.asarray(array) != 0.0), (
        f"{label}: the reference output is identically zero, so every claim "
        "made about it below is vacuous. Call activate() first."
    )


# ---------------------------------------------------------------------
# The forward contract, and the proof it can fail
# ---------------------------------------------------------------------


class TestTheForwardContractCanFail:
    """A smoke assertion nothing can break is decoration (v2 §16.1)."""

    def test_the_shape_and_finiteness_contract_rejects_a_broken_forward(self) -> None:
        model = activate(built_model(seed=0))
        inputs = list(tiny_inputs(seed=1))

        def contract(out: Any) -> None:
            assert_finite_and_shaped(out, TINY, BATCH)

        rejections = assert_contract_rejects_a_broken_forward(
            model, inputs, contract
        )
        assert set(rejections) >= {
            "collapse_to_scalar",
            "slice_leading_axis",
            "append_trailing_axis",
        }, sorted(rejections)


# ---------------------------------------------------------------------
# Degenerate geometries
# ---------------------------------------------------------------------

#: Configurations that sit at an edge of the geometry, each with the reason it
#: is an edge. Anything derived from `depth`, `grid_size` or `out_channels`
#: degenerates at one of these and at none of the interior points.
DEGENERATE: Dict[str, Dict[str, Any]] = {
    # A single block: the stack is no longer a loop over several, so an
    # off-by-one in the conditioning fan-out would not show as a mismatch.
    "depth_1": {"depth": 1},
    # input_size == patch_size, so the token grid is 1x1 and `num_patches` is 1.
    # The pos_embed table is (1, 1, D) and the unpatchify fold has nothing to
    # interleave -- an implementation that assumed a >1 grid breaks only here.
    "single_token": {"input_size": 4, "patch_size": 4},
    # `out_channels == in_channels`: the channel split the CFG helper and the
    # loss rely on disappears.
    "no_learned_sigma": {"learn_sigma": False},
    # No null label row at all; `num_classes` rows, not `num_classes + 1`.
    "no_null_row": {"class_dropout_rate": 0.0},
    # Both edges at once: the interaction is not implied by either alone.
    "depth_1_single_token": {"depth": 1, "input_size": 4, "patch_size": 4},
}


class TestTheDegenerateConfigurations:
    """Each edge of the geometry, forward and round trip."""

    @pytest.mark.parametrize("name", sorted(DEGENERATE), ids=sorted(DEGENERATE))
    def test_the_forward_pass_is_finite_and_correctly_shaped(self, name: str) -> None:
        config = dit_config(**DEGENERATE[name])
        model = activate(built_model(seed=0, **DEGENERATE[name]))
        out = model(list(tiny_inputs(seed=2, config=config)), training=False)
        array = assert_finite_and_shaped(out, config, BATCH)
        assert_not_degenerate(array, name)

    @pytest.mark.parametrize("name", sorted(DEGENERATE), ids=sorted(DEGENERATE))
    def test_the_keras_round_trip_reproduces_the_output(
        self, name: str, tmp_path
    ) -> None:
        """Serialization is a property of the CONFIG SPACE, not of one config.

        ``test_dit_model.py`` round-trips :data:`TINY`. A ``get_config`` that
        dropped ``learn_sigma`` or ``patch_size`` would round-trip :data:`TINY`
        perfectly and rebuild every one of these as a different model.
        """
        config = dit_config(**DEGENERATE[name])
        model = activate(built_model(seed=0, **DEGENERATE[name]))
        inputs = list(tiny_inputs(seed=3, config=config))
        before = np_(model(inputs, training=False))

        path = str(tmp_path / f"dit_{name}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        # Weight VALUES first, before the loaded model has ever been called.
        assert len(loaded.weights) == len(model.weights)
        for original, restored in zip(model.weights, loaded.weights):
            np.testing.assert_allclose(
                np_(restored), np_(original), rtol=0, atol=0.0
            )

        after = np_(loaded(inputs, training=False))
        assert_not_degenerate(after, name)
        np.testing.assert_allclose(after, before, rtol=0, atol=0.0)

    def test_the_single_token_grid_really_has_one_token(self) -> None:
        """Anti-vacuity for the ``single_token`` rows above.

        Without this, ``input_size=4, patch_size=4`` could be silently building
        a 2x2 grid and every arm parameterized over it would be testing an
        interior point under an edge's name.
        """
        model = built_model(seed=0, **DEGENERATE["single_token"])
        assert model.grid_size == 1
        assert model.num_patches == 1
        table = [w for w in model.weights if w.path.endswith("pos_embed")][0]
        assert tuple(table.shape) == (1, 1, TINY["hidden_size"])

    def test_depth_one_really_has_one_block(self) -> None:
        model = built_model(seed=0, **DEGENERATE["depth_1"])
        assert len(model.blocks) == 1
        assert not any("block_1/" in w.path for w in model.weights)


# ---------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------

POLICIES = ["float32", "mixed_float16", "float64"]

#: The compute dtype each policy must actually produce. A model that cast
#: everything to float32 internally would pass a finiteness arm under all three
#: while honouring none of them.
COMPUTE_DTYPE = {
    "float32": "float32",
    "mixed_float16": "float16",
    "float64": "float64",
}


@pytest.fixture
def policy(request) -> Iterator[str]:
    """Set a global dtype policy for one test and restore it afterwards.

    The restore is in a ``finally``: a policy leaking out of a FAILING test
    would silently re-dtype every test collected after it in the same process,
    and pytest ordering makes that intermittent.
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


def input_dtype_for(policy_name: str) -> str:
    """The float dtype the caller feeds in. ``mixed_float16`` takes float32."""
    return "float64" if policy_name == "float64" else "float32"


def cast_inputs(inputs: List[np.ndarray], dtype: str) -> List[np.ndarray]:
    """Recast the float members of an ``(x, t, y)`` triple; leave ``y`` int."""
    return [
        value.astype(dtype) if value.dtype.kind == "f" else value
        for value in inputs
    ]


class TestUnderEveryDtypePolicy:
    """Three policies, and the proof they are three regimes and not one."""

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_forward_pass_is_finite(self, policy: str) -> None:
        model = activate(built_model(seed=0))
        inputs = cast_inputs(list(tiny_inputs(seed=4)), input_dtype_for(policy))
        array = assert_finite_and_shaped(
            model(inputs, training=False), TINY, BATCH
        )
        assert_not_degenerate(array, f"policy={policy}")

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_compute_dtype_is_the_one_the_policy_asked_for(
        self, policy: str
    ) -> None:
        """Anti-vacuity for the arm above, asserted at every level of the tree.

        Checking only ``model.compute_dtype`` would miss a sub-layer that was
        constructed with a hard-coded ``dtype=``, which is the shape this defect
        actually takes.
        """
        model = tiny_model()
        expected = COMPUTE_DTYPE[policy]
        assert model.compute_dtype == expected
        for layer in (
            model.x_embedder,
            model.t_embedder,
            model.y_embedder,
            model.final_layer,
            *model.blocks,
        ):
            assert layer.compute_dtype == expected, layer.name

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_output_dtype_follows_the_policy(self, policy: str) -> None:
        model = activate(built_model(seed=0))
        inputs = cast_inputs(list(tiny_inputs(seed=5)), input_dtype_for(policy))
        out = model(inputs, training=False)
        # `np_()` round-trips through NumPy, whose `.dtype.name` is a plain
        # string. `keras.backend.standardize_dtype` would read the same value
        # and is BANNED under `models/` by
        # `test_package_api_contract.py::TestNoKeras2Residues`; not spelling it
        # here keeps the tests off an API the source is not allowed to use.
        assert np_(out).dtype.name == COMPUTE_DTYPE[policy]

    def test_the_float32_and_float16_arms_are_not_the_same_computation(self) -> None:
        """The INERTNESS check the brief asks for, made explicit.

        ``mixed_float16`` keeps float32 master variables, so at one seed the two
        policies hold BIT-IDENTICAL weights. Any output difference is therefore
        attributable to the arithmetic and to nothing else. If this ever reads
        ``0.0`` the two "policy" arms above are one claim written twice.
        """
        outputs = {}
        weights = {}
        for name in ("float32", "mixed_float16"):
            previous = keras.mixed_precision.global_policy()
            keras.mixed_precision.set_global_policy(name)
            try:
                model = activate(built_model(seed=0))
                weights[name] = [np_(w).astype("float64") for w in model.weights]
                outputs[name] = np_(
                    model(list(tiny_inputs(seed=6)), training=False)
                ).astype("float64")
            finally:
                keras.mixed_precision.set_global_policy(previous)

        for a, b in zip(weights["float32"], weights["mixed_float16"]):
            np.testing.assert_allclose(a, b, rtol=0, atol=0.0)

        delta = float(np.max(np.abs(outputs["float32"] - outputs["mixed_float16"])))
        assert delta > 1e-4, (
            f"float32 and mixed_float16 produced outputs agreeing to "
            f"max|delta| = {delta:.3e} on bit-identical weights. The policy is "
            "not reaching the arithmetic: the three 'precision arms' are INERT."
        )
        # And the difference is float16 rounding, not a different model: the
        # relative discrepancy must sit near float16's resolution, not above it.
        scale = float(np.max(np.abs(outputs["float32"])))
        assert delta / scale < 1e-2, (delta, scale)

    @pytest.mark.parametrize("policy", ["mixed_float16"], indirect=True)
    def test_mixed_float16_keeps_its_variables_in_float32(self, policy: str) -> None:
        """That is what "mixed" means; a float16 master copy is a defect."""
        model = built_model(seed=0)
        trainable = [w for w in model.weights if w.trainable]
        assert trainable
        assert {str(w.dtype) for w in trainable} == {"float32"}

    @pytest.mark.parametrize("policy", POLICIES, indirect=True)
    def test_the_frozen_table_stays_float32_under_every_policy(
        self, policy: str
    ) -> None:
        """A measured FACT about this port, asserted rather than assumed.

        ``DiT.build`` creates ``pos_embed`` with an explicit ``dtype="float32"``
        and ``DiT.call`` casts it to the token dtype at the point of use. So the
        table is float32 even under a ``float64`` policy, where it carries
        float32 precision into a float64 computation. This arm exists so that
        fact is written down and any change to it is deliberate; the cast in
        ``call`` is what keeps the forward pass legal in the meantime.
        """
        model = built_model(seed=0)
        table = [w for w in model.weights if w.path.endswith("pos_embed")][0]
        assert str(table.dtype) == "float32"
        inputs = cast_inputs(list(tiny_inputs(seed=7)), input_dtype_for(policy))
        assert_finite_and_shaped(model(inputs, training=False), TINY, BATCH)


# ---------------------------------------------------------------------
# XLA
# ---------------------------------------------------------------------


class TestJitCompileAgreesWithEager:
    """``jit_compile=True`` must be the same model, not merely a fast one."""

    @staticmethod
    def _tolerance(scale: float) -> float:
        """Derived from the contraction lengths on the compared path.

        Both evaluations compute the identical formula; XLA fuses and reorders
        the reductions, so they differ only in rounding. The lengths are the
        three contractions each block performs -- the ``hidden_size`` matmuls,
        the attention softmax over ``num_patches`` tokens and the MLP's
        ``mlp_ratio * hidden_size`` -- applied ``depth + 2`` times (the block
        stack, the patch projection and the read-out).
        """
        hidden = TINY["hidden_size"]
        tokens = (TINY["input_size"] // TINY["patch_size"]) ** 2
        return reassociation_atol(
            reduction_lengths=(hidden, tokens, int(TINY["mlp_ratio"] * hidden)),
            num_steps=TINY["depth"] + 2,
            scale=scale,
        )

    def _compiled(self, jit: bool):
        model = activate(built_model(seed=0))
        loss = DDPMHybridLoss(
            schedule_name="linear", num_timesteps=50,
            in_channels=TINY["in_channels"],
        )
        inputs, y_true = ddpm_training_batch(model, loss, batch=BATCH, seed=8)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1),
            loss=loss,
            jit_compile=jit,
        )
        return model, inputs, y_true

    def test_predict_agrees_between_xla_and_eager(self) -> None:
        results = {}
        for jit in (False, True):
            model, inputs, _ = self._compiled(jit)
            assert model.jit_compile is jit, (
                f"asked for jit_compile={jit} and got {model.jit_compile}; the "
                "two arms of this comparison are the same regime"
            )
            results[jit] = np_(model.predict(inputs, batch_size=BATCH, verbose=0))

        assert_not_degenerate(results[False], "eager predict")
        scale = float(np.max(np.abs(results[False])))
        atol = self._tolerance(scale)
        delta = float(np.max(np.abs(results[True] - results[False])))
        assert delta <= atol, (
            f"XLA and eager predict differ by {delta:.3e}, above the derived "
            f"re-association bound {atol:.3e} at scale {scale:.3f}. Do NOT "
            "widen the bound -- re-derive it, or find the real divergence."
        )

    def test_fit_agrees_between_xla_and_eager(self) -> None:
        """One real optimizer step under each, on the real objective."""
        losses = {}
        moved = {}
        for jit in (False, True):
            model, inputs, y_true = self._compiled(jit)
            before = [np_(w).copy() for w in model.trainable_weights]
            history = model.fit(
                inputs, y_true, batch_size=BATCH, epochs=1, verbose=0
            )
            losses[jit] = float(history.history["loss"][0])
            moved[jit] = sum(
                1
                for w, original in zip(model.trainable_weights, before)
                if np.any(np_(w) != original)
            )

        assert moved[False] > 0 and moved[False] == moved[True], moved
        atol = self._tolerance(max(losses.values()))
        assert abs(losses[True] - losses[False]) <= atol, (losses, atol)


# ---------------------------------------------------------------------
# Knob liveness
# ---------------------------------------------------------------------

#: Every constructor knob, classified by the instrument that can convict it.
#:
#: STRUCTURAL -- changes the parameterisation, so the WEIGHT-SHAPE SIGNATURE
#: must change. An output-difference assertion would be satisfied by the
#: different random draw alone and would pass on a model that dropped the kwarg.
STRUCTURAL_KNOBS: Dict[str, List[Any]] = {
    "input_size": [8, 16],
    "patch_size": [2, 4],
    "in_channels": [4, 6],
    "hidden_size": [32, 64],
    "depth": [2, 3],
    # Stock `keras.layers.MultiHeadAttention` shapes its projections
    # (D, num_heads, key_dim) with key_dim = D // num_heads, so this IS
    # structural here. It is a VALUE knob in packages that carry a custom
    # attention -- classify by the shapes, never by the knob's name.
    "num_heads": [4, 8],
    "mlp_ratio": [4.0, 2.0],
    # 0.0 removes the null label row that classifier-free guidance indexes, so
    # this knob is structural at the 0/non-0 boundary. Its training-mode half is
    # covered below.
    "class_dropout_rate": [0.0, 0.2],
    "num_classes": [10, 20],
    "learn_sigma": [True, False],
    "use_bias": [True, False],
    "frequency_embedding_size": [16, 32],
}

#: VALUE -- same signature, different arithmetic. Under one seed the two models
#: hold bit-identical weights, so an output difference is attributable to the
#: knob. Requires `activate()`: at initialisation both outputs are exactly 0.0.
VALUE_KNOBS: Dict[str, List[Any]] = {
    "norm_epsilon": [1e-6, 0.5],
}

#: TRAINING-MODE -- inert at inference BY DESIGN, which is what dropout means.
#: Liveness is proven under ``training=True``; the inference-time inertness is
#: asserted as a positive claim rather than left as a hole.
TRAINING_MODE_KNOBS = ("dropout_rate", "label_seed")


def structural_builder(knob: str, value: Any):
    """A zero-argument builder returning a BUILT model at ``knob=value``."""
    return lambda: built_model(seed=1234, **{knob: value})


def value_builder(knob: str, value: Any):
    """As above, plus :func:`activate` so the comparison is not over zeros."""
    return lambda: activate(built_model(seed=1234, **{knob: value}), seed=5)


class TestTheKnobCensusIsComplete:
    """A knob added tomorrow must break this file, not slip past it."""

    def test_every_constructor_parameter_is_classified(self) -> None:
        """Enumerated from the SIGNATURE, never from a hand-kept list.

        The three classifications below are the whole population of
        ``DiT.__init__``. If someone adds a knob and does not classify it, this
        arm fails and names it -- which is the only mechanism that stops the
        sweeps below from quietly covering a shrinking fraction of the surface.
        """
        parameters = inspect.signature(DiT.__init__).parameters
        declared: Set[str] = {
            name
            for name, p in parameters.items()
            if name != "self"
            and p.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }
        classified = (
            set(STRUCTURAL_KNOBS) | set(VALUE_KNOBS) | set(TRAINING_MODE_KNOBS)
        )
        assert declared == classified, {
            "unclassified": sorted(declared - classified),
            "classified but gone from the signature": sorted(classified - declared),
        }
        assert len(declared) == 15, sorted(declared)

    def test_no_knob_is_classified_twice(self) -> None:
        names = (
            list(STRUCTURAL_KNOBS) + list(VALUE_KNOBS) + list(TRAINING_MODE_KNOBS)
        )
        assert len(names) == len(set(names)), sorted(names)


class TestTheStructuralKnobs:
    """Pinned on the weight-shape signature, which RNG luck cannot fake."""

    @pytest.mark.parametrize(
        "knob, values",
        sorted(STRUCTURAL_KNOBS.items()),
        ids=sorted(STRUCTURAL_KNOBS),
    )
    def test_the_knob_changes_the_weight_shape_signature(
        self, knob: str, values: List[Any]
    ) -> None:
        assert_structural_knob_changes_weights(
            {v: structural_builder(knob, v) for v in values}, knob=knob
        )

    def test_depth_grows_the_parameter_count_monotonically(self) -> None:
        """A stronger claim than "the signature changed"."""
        totals = []
        for depth in (1, 2, 3):
            model = built_model(seed=0, depth=depth)
            assert len(model.blocks) == depth
            totals.append(sum(int(np.prod(w.shape)) for w in model.weights))
        assert totals[0] < totals[1] < totals[2], totals

    def test_learn_sigma_doubles_exactly_the_read_out_width(self) -> None:
        """Which weight changed, not merely that one did."""
        on = built_model(seed=0, learn_sigma=True)
        off = built_model(seed=0, learn_sigma=False)

        def read_out(model: DiT):
            return [
                tuple(w.shape)
                for w in model.weights
                if w.path.endswith("final_layer/linear/kernel")
            ][0]

        p = TINY["patch_size"]
        c = TINY["in_channels"]
        assert read_out(on) == (TINY["hidden_size"], p * p * 2 * c)
        assert read_out(off) == (TINY["hidden_size"], p * p * c)


class TestTheValueKnobs:
    """Same signature, same seed, therefore bit-identical weights."""

    @pytest.mark.parametrize(
        "knob, values", sorted(VALUE_KNOBS.items()), ids=sorted(VALUE_KNOBS)
    )
    def test_the_knob_changes_the_output(
        self, knob: str, values: List[Any]
    ) -> None:
        inputs = list(tiny_inputs(seed=9))
        assert_not_degenerate(
            np_(activate(built_model(seed=1234))(inputs, training=False)), knob
        )
        deltas = assert_value_knob_changes_output(
            {v: value_builder(knob, v) for v in values},
            inputs,
            knob=knob,
            atol=1e-4,
        )
        assert min(deltas.values()) > 10 * 1e-4, deltas

    def test_norm_epsilon_reaches_the_sub_layers_and_not_just_the_attribute(
        self,
    ) -> None:
        """``assert model.norm_epsilon == x`` is a knob ECHO, not a test."""
        model = built_model(seed=0, norm_epsilon=0.25)
        reached = [
            layer.epsilon
            for layer in model._flatten_layers(include_self=False)
            if type(layer).__name__ == "LayerNormalization"
        ]
        assert reached, "no LayerNormalization found; the walk is broken"
        assert set(reached) == {0.25}, sorted(set(reached))


class TestTheTrainingModeKnobs:
    """Measured on ONE model, ``training=True`` against ``training=False``.

    Two separately constructed models would not do: constructing a ``Dropout``
    at a non-zero rate consumes process-global RNG, so two models built at the
    same seed with different rates hold different weights, and an output
    difference between them would be attributable to the draw.
    """

    def _both_modes(self, **overrides: Any):
        model = activate(built_model(seed=1234, **overrides))
        inputs = list(tiny_inputs(seed=10))
        keras.utils.set_random_seed(99)
        train = np_(model(inputs, training=True))
        infer = np_(model(inputs, training=False))
        return train, infer

    @pytest.mark.parametrize(
        "overrides, floor",
        [
            ({"dropout_rate": 0.5}, 1e-2),
            ({"class_dropout_rate": 0.9}, 1e-2),
        ],
        ids=["dropout_rate", "class_dropout_rate"],
    )
    def test_the_knob_changes_the_output_under_training(
        self, overrides: Dict[str, Any], floor: float
    ) -> None:
        train, infer = self._both_modes(**overrides)
        assert_not_degenerate(infer, str(overrides))
        delta = float(np.max(np.abs(train - infer)))
        assert delta > floor, (
            f"{overrides} left training=True and training=False agreeing to "
            f"max|delta| = {delta:.3e}; the knob does not reach call()"
        )

    def test_at_rate_zero_training_changes_nothing(self) -> None:
        """The control. Without it the arms above could be measuring anything.

        With every stochastic rate at zero, ``training=True`` and
        ``training=False`` must produce the SAME numbers, so the deltas above
        are attributable to the rates and not to some other training-mode branch
        in the graph.
        """
        train, infer = self._both_modes(dropout_rate=0.0, class_dropout_rate=0.0)
        assert_not_degenerate(infer, "rate-zero control")
        np.testing.assert_allclose(train, infer, rtol=0, atol=0.0)

    def test_label_seed_changes_which_labels_are_dropped(self) -> None:
        """Two seeds, bit-identical weights (asserted), different dropout draws."""
        a = activate(built_model(seed=1234, class_dropout_rate=0.5, label_seed=1))
        b = activate(built_model(seed=1234, class_dropout_rate=0.5, label_seed=999))
        assert weight_signature(a) == weight_signature(b)
        for wa, wb in zip(a.weights, b.weights):
            np.testing.assert_allclose(np_(wa), np_(wb), rtol=0, atol=0.0)

        inputs = list(tiny_inputs(seed=11))
        keras.utils.set_random_seed(0)
        out_a = np_(a(inputs, training=True))
        keras.utils.set_random_seed(0)
        out_b = np_(b(inputs, training=True))
        assert float(np.max(np.abs(out_a - out_b))) > 0.0, (
            "two different label_seed values dropped exactly the same labels; "
            "the seed does not reach the dropout RNG"
        )
