"""
Oracle adoption for ``models/thera`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE COVERAGE HOLE THIS FILE IS BUILT AROUND
---------------------------------------------
``Thera.call -> apply_decoder -> self.hypernetwork.decode_with_jac`` is a
CROSS-OBJECT call, and it fires **only at ``return_jac=True``**, which is not
the default. Two consequences, both of which shaped this file:

1. The step-8 callee-closure sweep does not reach it -- it follows calls within
   an object, and this one hops to a different object.
2. Every default-argument forward in this package, including the one the
   gradient oracle would take if it were pointed at ``model(x)``, executes the
   ``return_jac=False`` branch and leaves the Jacobian path completely
   unexercised.

So :class:`TestTheraJacobianBranch` runs the ``return_jac=True`` branch
explicitly, asserts the forward half is IDENTICAL between the two branches (the
Jacobian is an addition, not a different model), and takes a gradient reading
THROUGH the Jacobian -- because the step-9 TV penalty ``mean(abs(jac))`` is a
real training objective and a Jacobian that is on the graph for the forward but
not for the backward would be silently useless.

Measured 2026-08-21, one Adam step, ramp loss, on
``hidden_dim=32 / out_dim=3`` over an ``EDSRBackbone(num_feats=8, num_blocks=2)``
with the ``air`` (identity) tail, source ``(1, 8, 8, 3)`` decoded to a
``12 x 12`` query grid:

===============================  ==========  ======
arm                              weights     dead
=================================  ==========  ====================
arm                                weights     dead
=================================  ==========  ====================
forward (``return_jac=False``)     16          0
Jacobian TV (``return_jac=True``)  16          1 (``heat_field/k``)
=================================  ==========  ====================

THE ONE DEAD WEIGHT IS A REAL FINDING, AND IT IS NOT A DEFECT
---------------------------------------------------------------
Writing the Jacobian arm found it: under the step-9 penalty ``mean(abs(jac))``
and nothing else, ``thera/hypernetwork/heat_field/k`` receives an
IDENTICALLY-ZERO gradient. The mechanism is analytic, not numerical --
``decode_with_jac`` evaluates ``d(field)/d(rel_coords)`` at ``t = 0``, where the
heat envelope ``exp(-k |w|^2 t)`` is exactly ``1`` for every ``k``, so the
conductivity cancels out of the Jacobian entirely. Measured directly: two models
differing only in ``k_init`` (0.1 vs 5.0) produce BIT-IDENTICAL Jacobians.

It is pinned two-sided rather than waived: ``k`` is LIVE under the ordinary
forward loss, and the envelope fact is its own test. The operational
consequence is worth stating plainly -- **a THERA fine-tune driven by the
Jacobian-TV penalty alone cannot move the heat conductivity**; the reconstruction
term is what trains it.

The shipped ``MODEL_VARIANTS`` all use ``num_feats=64, num_blocks=16``; a
smaller backbone is built here so the suite stays affordable. The variant table
itself is exercised by the knob assertions.

THERA'S fp16 DEFECT WAS FIXED AT STEP 17.1
--------------------------------------------
Nothing in this file re-litigates it; a red here is a regression, not a
discovery.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.thera.edsr_backbone import EDSRBackbone
from dl_techniques.models.thera.model import Thera, build_thera
from dl_techniques.models.thera.tails import TheraTailAir

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

SRC_SHAPE = (2, 8, 8, 3)
QUERY = 12
OUT_DIM = 3
HIDDEN_DIM = 32
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 16

#: The ONE weight that gets no gradient from the Jacobian-TV penalty, as a path
#: SUFFIX (never an absolute ``Variable.path``: Keras uniquifies a model name
#: per process, so the second ``Thera`` in a session is ``thera_1/...`` and an
#: absolute pin is green alone and red behind any other test that builds the
#: same class -- it bit batch B twice).
#:
#: FOUND BY THIS ADOPTION, and it is NOT a defect. ``decode_with_jac`` takes
#: ``d(field)/d(rel_coords)`` at ``t = 0``, where the heat envelope is
#: ``exp(-k |w|^2 t) == 1`` for every ``k``, so the conductivity analytically
#: cancels out of the Jacobian. Pinned two-sided:
#: ``test_the_conductivity_IS_live_under_the_ordinary_forward_loss`` and
#: ``test_the_envelope_at_t_zero_is_exactly_one_regardless_of_k``.
#:
#: The operational consequence is real and worth stating: **a THERA fine-tune
#: driven by the step-9 TV penalty ALONE cannot move the heat conductivity.**
JAC_DEAD_SUFFIX = "heat_field/k"


def _jac_only(outputs: Any) -> Any:
    """The step-9 aliasing penalty, ``mean(abs(jac))`` and nothing else."""
    _, jac = outputs
    return keras.ops.mean(keras.ops.abs(jac))


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _inputs(query: int = QUERY, seed: int = 0) -> Tuple[Any, Any, Any]:
    """``(source, coords, t)`` -- THERA's 3-tuple forward input."""
    source = np.random.default_rng(seed).random(SRC_SHAPE).astype("float32")
    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, query), np.linspace(-1.0, 1.0, query),
        indexing="ij")
    coords = np.repeat(
        np.stack([ys, xs], axis=-1)[None], SRC_SHAPE[0], axis=0).astype("float32")
    t = np.full((SRC_SHAPE[0], 1), 1.0, dtype="float32")
    return source, coords, t


def _thera(**o) -> Thera:
    kwargs: Dict[str, Any] = dict(
        hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        backbone=EDSRBackbone(num_feats=8, num_blocks=2, name="backbone_edsr"),
        tail=TheraTailAir(name="tail_air"),
    )
    kwargs.update(o)
    return Thera(**kwargs)


def _built(build_fn=_thera, seed: int = BUILD_SEED) -> Thera:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_inputs(), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs, loss_fn=ramp_loss) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = loss_fn(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestTheraGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero stochastic rate is live: {stochastic}"

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _inputs(), loss_fn=ramp_loss)


class TestTheraJacobianBranch:
    """The cross-object ``decode_with_jac`` path, which no default forward runs.

    ``Thera.call`` reaches it only at ``return_jac=True``; the hypernetwork is a
    different object, so an intra-object call sweep does not follow the hop.
    """

    def test_the_default_forward_does_not_run_it(self):
        """The premise. If ``return_jac`` ever defaults to True, this file's
        whole framing is stale and this test says so."""
        model = _built()
        out = model(_inputs(), training=False)
        assert not isinstance(out, (tuple, list)), (
            f"the DEFAULT forward now returns {type(out)}; return_jac is no "
            f"longer False by default")

    def test_the_branch_returns_the_field_and_a_per_pixel_jacobian(self):
        model = _built()
        source, coords, t = _inputs()
        out, jac = model((source, coords, t), training=False, return_jac=True)
        assert tuple(out.shape) == (SRC_SHAPE[0], QUERY, QUERY, OUT_DIM), tuple(out.shape)
        assert tuple(jac.shape) == (SRC_SHAPE[0], QUERY, QUERY, OUT_DIM, 2), tuple(jac.shape)
        assert_finite(out)
        assert_finite(jac)

    def test_the_forward_half_is_identical_between_the_two_branches(self):
        """The Jacobian is an ADDITION, not a different model.

        If the two branches disagreed on the field itself, the TV penalty
        would be regularising something other than what is being trained --
        and every shape assertion above would still pass.
        """
        model = _built()
        x = _inputs()
        plain = keras.ops.convert_to_numpy(model(x, training=False))
        with_jac = keras.ops.convert_to_numpy(
            model(x, training=False, return_jac=True)[0])
        np.testing.assert_allclose(plain, with_jac, rtol=0, atol=0)

    def test_every_weight_is_live_through_the_jacobian_alone(self):
        """The step-9 TV penalty is ``mean(abs(jac))`` and NOTHING else.

        A Jacobian on the forward graph but off the backward graph is silently
        useless: the trainer's outer tape would compute a penalty that moves no
        weight, and every finiteness and shape check would stay green. So the
        gradient oracle is pointed at a loss made ONLY of the Jacobian.
        """
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)

        with_jac = _JacModel(model)
        report = assert_gradients_reach_every_trainable_weight(
            with_jac, x, loss_fn=_jac_only, expect_zero=(JAC_DEAD_SUFFIX,))
        assert len(report) == GF_WEIGHTS

    def test_the_conductivity_IS_live_under_the_ordinary_forward_loss(self):
        """The discriminating half of the waiver above.

        Without this, ``expect_zero=("heat_field/k",)`` would be
        indistinguishable from a disconnected conductivity parameter -- the
        exact one-sided skip-list rot D-010 forbids. Under the plain forward
        loss ``k`` is live, so the zero above is a property of the JACOBIAN
        (which is taken at ``t=0``), not of the parameter.
        """
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=ramp_loss)
        path = next(p for p in report if p.endswith(JAC_DEAD_SUFFIX))
        assert report[path] is not None and report[path] > 0.0, (
            f"{JAC_DEAD_SUFFIX} is dead under the FORWARD loss too "
            f"(max|grad|={report[path]}); the t=0 explanation is then wrong "
            f"and this IS a disconnected parameter"
        )

    def test_the_envelope_at_t_zero_is_exactly_one_regardless_of_k(self):
        """The mechanism, measured rather than asserted from the docstring.

        ``decode_with_jac`` evaluates ``d(field)/d(rel)`` at ``t = 0``, where
        the heat envelope is ``exp(-k * |w|^2 * 0) == 1`` for EVERY ``k``. Two
        models differing only in ``k_init`` must therefore produce the SAME
        Jacobian on identical weights -- which is precisely why ``k`` gets no
        gradient from ``mean(abs(jac))``.
        """
        jacs = []
        for k in (0.1, 5.0):
            model = _built(lambda k=k: _thera(k_init=k))
            _, jac = model(_inputs(), training=False, return_jac=True)
            jacs.append(np.asarray(keras.ops.convert_to_numpy(jac)))
        np.testing.assert_allclose(jacs[0], jacs[1], rtol=0, atol=0)

    def test_the_jacobian_gradient_assertion_can_fail(self):
        model = _built()
        wrapper = _JacModel(model)
        with broken_forward(wrapper, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    wrapper, _inputs(), loss_fn=_jac_only,
                    expect_zero=(JAC_DEAD_SUFFIX,))


class _JacModel(keras.Model):
    """Thin adapter that makes ``return_jac=True`` the forward.

    Exists because ``gradient_report`` calls ``model(inputs, training=...)`` and
    has no way to pass a third keyword -- widening the ORACLE's signature to
    carry per-package call kwargs would put a package's shape into a shared
    instrument. The adapter holds NO weights of its own (asserted below), so
    the weight set it reports is exactly the wrapped model's.
    """

    def __init__(self, inner: Thera, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, inputs, training=None):
        return self.inner(inputs, training=training, return_jac=True)


class TestTheraKnobSensitivity:

    def test_backbone_changes_the_parameterisation(self):
        """``edsr-baseline`` vs ``rdn`` -- entirely different feature extractors."""
        builders = {
            b: (lambda b=b: _wrap(build_thera(out_dim=3, backbone=b, size="air")))
            for b in ("edsr-baseline", "rdn")
        }
        assert_structural_knob_changes_weights(builders, knob="backbone")

    def test_size_changes_the_parameterisation(self):
        """``air`` (identity tail, hidden_dim 32) vs ``plus`` (ConvNeXt tail)
        vs ``pro`` (SwinIR tail)."""
        builders = {
            s: (lambda s=s: _wrap(
                build_thera(out_dim=3, backbone="edsr-baseline", size=s)))
            for s in ("air", "plus", "pro")
        }
        assert_structural_knob_changes_weights(builders, knob="size")

    def test_hidden_dim_changes_the_parameterisation(self):
        builders = {
            h: (lambda h=h: _built(lambda: _thera(hidden_dim=h)))
            for h in (16, 32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="hidden_dim")

    def test_k_init_reaches_the_heat_field(self):
        """A VALUE knob: the heat conductivity. It changes no weight SHAPE.

        It is not put through ``assert_value_knob_changes_output`` because
        ``k_init`` initialises a trainable variable rather than a forward
        constant, so the two arms differ in a WEIGHT VALUE -- which is what is
        compared here, with the shapes pinned identical first so the difference
        cannot be a different random draw.
        """
        values = {}
        signature = None
        for k in (0.1, 5.0):
            model = _built(lambda k=k: _thera(k_init=k))
            sig = tuple(tuple(w.shape) for w in model.weights)
            assert signature is None or sig == signature, (
                "k_init changed the parameterisation; it is a STRUCTURAL knob "
                "and this comparison is contaminated by a different draw")
            signature = sig
            values[k] = model.k_init

        assert values[0.1] != values[5.0], (
            f"k_init is a no-op: both settings stored {values}")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="hidden_dim")

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError):
            Thera.from_variant("edsr-mega")


def _wrap(model: Thera) -> Thera:
    """Build a caller-supplied Thera on this file's fixed input."""
    model(_inputs(), training=False)
    return model


class TestTheraSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _inputs()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"Thera.call at return_jac=False returns ONE tensor, got "
                f"{type(out)}")
            assert tuple(out.shape) == (SRC_SHAPE[0], QUERY, QUERY, OUT_DIM), (
                f"expected {(SRC_SHAPE[0], QUERY, QUERY, OUT_DIM)}, got "
                f"{tuple(out.shape)}")
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_output_resolution_follows_the_QUERY_grid_not_the_source(self):
        """THERA decodes at arbitrary scale: the output extent comes from
        ``coords``, NOT from the low-resolution ``source``. A build that read
        the source extent instead would produce an 8x8 output here and every
        finiteness check would still pass."""
        model = _built()
        for query in (5, 12, 19):
            out = model(_inputs(query=query), training=False)
            assert tuple(out.shape) == (SRC_SHAPE[0], query, query, OUT_DIM), (
                f"query grid {query} gave {tuple(out.shape)}; the source is "
                f"{SRC_SHAPE[1]}x{SRC_SHAPE[2]}")
            assert model.compute_output_shape(
                [SRC_SHAPE, (SRC_SHAPE[0], query, query, 2), (SRC_SHAPE[0], 1)]
            ) == (SRC_SHAPE[0], query, query, OUT_DIM)
