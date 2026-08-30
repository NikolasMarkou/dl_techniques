"""Every class in ``layers/statistics`` measured against the guide's §16.3 checklist.

§16.3 lists the tests that must exist "before you call it done". Before this file, the
package had **zero** occurrences of five of them: a ``rtol=0`` value round trip with an
explicit ``training=False``, an ``atol=0.0`` weight comparison before the loaded model's
first call, build parity by relative ``w.path``, ``mixed_float16``/``float64`` arms with a
float32 control, and ``jit_compile=True`` versus eager. Per-variable gradient flow existed
only as ``grad is not None``, which §13.2.2 measured reporting green while 61 of 61
weights had identically-zero gradients.

Nine layer classes are swept here. The tenth public name, ``ACFMonitorCallback``, is a
``keras.callbacks.Callback`` and not a layer; ``TestTheCallbackIsNotALayer`` states
item-by-item which §16.3 rows are N/A for it and why, rather than omitting it silently.

Where an item genuinely does not hold, it is pinned with
``@pytest.mark.xfail(strict=True)`` carrying the MEASURED reason (§13.3.2), never a bare
``skip`` and never a silent omission. A strict xfail XPASSes loudly the day someone fixes
the underlying defect. The full xfail table lives in ``XFAILS`` below, one entry per
(class, item), each with the measurement that justifies it.

The shared instruments come from ``v2_compliance_oracle``; their RED proof, which shows
every one of them rejecting a broken subject, is ``test_the_v2_oracle_can_fail.py``.
"""

from typing import Any, Callable, Dict, List, Tuple

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.statistics.deep_kernel_pca import DeepKernelPCA
from dl_techniques.layers.statistics.invertible_kernel_pca import (
    InvertibleKernelPCA,
    InvertibleKernelPCADenoiser,
)
from dl_techniques.layers.statistics.mdn_layer import MDNLayer
from dl_techniques.layers.statistics.moving_std import MovingStd
from dl_techniques.layers.statistics.normalizing_flow import (
    AffineCouplingLayer,
    NormalizingFlowLayer,
)
from dl_techniques.layers.statistics.residual_acf import (
    ACFMonitorCallback,
    ResidualACFLayer,
)
from dl_techniques.layers.statistics.scaler import UnifiedScaler

from .v2_compliance_oracle import (
    DTYPE_POLICIES,
    DtypePolicyScope,
    Tf32Scope,
    MethodAdapter,
    assert_build_parity,
    assert_eager_matches_jit,
    assert_forward_is_finite,
    assert_gradients_reach_every_trainable_weight,
    assert_value_round_trip,
    assert_weights_restored_before_first_call,
    functional_model_from_layer,
    relative_weight_paths,
)

# ---------------------------------------------------------------------
# the subjects
# ---------------------------------------------------------------------

# Every array is drawn from a fixed-seed generator (§13.7.4) so a failure is reproducible.
_RNG_SEED = 1234


def _draw(*shape: int) -> np.ndarray:
    return np.random.default_rng(_RNG_SEED).normal(size=shape).astype("float32")


class Subject:
    """One class under test, with everything the §16.3 instruments need.

    :param name: The class name, used as the pytest parameter id.
    :param factory: Zero-argument callable returning a fresh, unbuilt layer.
    :param make_input: Zero-argument callable returning the sample input. It is a
        callable, not an array, so each dtype arm re-draws rather than sharing one
        already-cast array between policies.
    :param input_shape: The argument ``build()`` takes, for the build-parity item.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        make_input: Callable[[], Any],
        input_shape: Any,
    ) -> None:
        self.name = name
        self.factory = factory
        self.make_input = make_input
        self.input_shape = input_shape


SUBJECTS: List[Subject] = [
    Subject(
        "UnifiedScaler",
        lambda: UnifiedScaler(num_features=4, affine=True, name="scaler"),
        lambda: _draw(8, 16, 4),
        (None, 16, 4),
    ),
    Subject(
        "MovingStd",
        lambda: MovingStd(pool_size=(3, 3), name="moving_std"),
        lambda: _draw(4, 8, 8, 3),
        (None, 8, 8, 3),
    ),
    Subject(
        # batch (6) <= feature width (8) at EVERY level is a hard constraint of the
        # un-fitted fallback: `extract_components` slices the leading `batch_size` rows
        # out of the projection weight, so a wider batch dies inside a matmul. The
        # package's own fixture encodes the same rule.
        "DeepKernelPCA",
        lambda: DeepKernelPCA(
            num_levels=2, components_per_level=[6, 6], name="deep_kernel_pca"
        ),
        lambda: _draw(6, 8),
        (None, 8),
    ),
    Subject(
        "InvertibleKernelPCA",
        lambda: InvertibleKernelPCA(
            n_components=4, n_random_features=16, random_seed=0, name="ikpca"
        ),
        lambda: _draw(8, 6),
        (None, 6),
    ),
    Subject(
        "InvertibleKernelPCADenoiser",
        lambda: InvertibleKernelPCADenoiser(
            n_components=4, n_random_features=16, name="ikpca_denoiser"
        ),
        lambda: _draw(8, 6),
        (None, 6),
    ),
    Subject(
        "MDNLayer",
        lambda: MDNLayer(output_dimension=2, num_mixtures=3, name="mdn"),
        lambda: _draw(8, 5),
        (None, 5),
    ),
    Subject(
        # AffineCouplingLayer has no `call`; `forward` is its public entry point, so it
        # is driven through the oracle's serializable MethodAdapter. Calling `forward`
        # directly bypasses `Layer.__call__` entirely -- see
        # TestForwardBypassesTheLayerNameScope for what that costs.
        "AffineCouplingLayer",
        lambda: MethodAdapter(
            AffineCouplingLayer(input_dim=4, context_dim=3, name="coupling"),
            "forward",
            name="coupling_adapter",
        ),
        lambda: [_draw(8, 4), _draw(8, 3)],
        [(None, 4), (None, 3)],
    ),
    Subject(
        "NormalizingFlowLayer",
        lambda: NormalizingFlowLayer(
            output_dimension=4, num_flow_steps=2, context_dim=3, name="flow"
        ),
        lambda: [_draw(8, 4), _draw(8, 3)],
        [(None, 4), (None, 3)],
    ),
    Subject(
        "ResidualACFLayer",
        lambda: ResidualACFLayer(max_lag=5, name="residual_acf"),
        lambda: [_draw(8, 32, 1), _draw(8, 32, 1)],
        [(None, 32, 1), (None, 32, 1)],
    ),
]

# ---------------------------------------------------------------------
# the measured xfail table
# ---------------------------------------------------------------------

# (class name, item key) -> the MEASURED reason. Each was produced by running the item
# against the class in this worktree, not predicted. `strict=True` throughout, so any of
# these turning green fails the suite and forces this table to be re-derived.
XFAILS: Dict[Tuple[str, str], str] = {
    ("MovingStd", "weights_restored"): (
        "measured: MovingStd has 0 weights of any kind -- it wraps an AveragePooling "
        "and holds only Python config. The oracle's `assert saved` anti-vacuity check "
        "fires, which is the correct outcome: without it the atol=0.0 loop would "
        "iterate zero times and report green on a layer that restored nothing"
    ),
    ("MovingStd", "gradients"): (
        "measured: MovingStd exposes 0 trainable variables, so the §13.2.2 "
        "`len(trainable_variables) > 0` anti-vacuity assertion fires. There is nothing "
        "for a gradient to reach; the item is inapplicable, not failing"
    ),
    ("ResidualACFLayer", "weights_restored"): (
        "measured: ResidualACFLayer creates no weights in __init__ or build (its own "
        "docstring says so) -- it computes an ACF and passes the predictions through. "
        "The anti-vacuity check fires for the same reason as MovingStd"
    ),
    ("ResidualACFLayer", "gradients"): (
        "measured: 0 trainable variables. The layer's gradient contribution is an "
        "add_loss regularizer on its INPUTS, not on weights of its own"
    ),
    ("InvertibleKernelPCA", "gradients"): (
        "measured: `no gradient for ikpca/reconstruction_matrix`. That weight is "
        "trainable=True but is read only by inverse_transform(), which call() never "
        "runs, so a loss on the forward output cannot reach it. Fixing it means "
        "deciding whether the weight should be trainable at all -- out of scope for a "
        "test build-out. Note the sibling InvertibleKernelPCADenoiser PASSES this item, "
        "because its call() runs transform followed by inverse_transform"
    ),
    ("InvertibleKernelPCA", "dtype-mixed_float16"): (
        "measured: InvalidArgumentError at invertible_kernel_pca.py:611, "
        "`ops.sqrt(2.0 / self.n_random_features)` produces a float32 scale that is "
        "multiplied into float16 cosine features"
    ),
    ("InvertibleKernelPCA", "dtype-float64"): (
        "measured: the same site, invertible_kernel_pca.py:611, with float64 features "
        "and a float32 scale"
    ),
    ("InvertibleKernelPCADenoiser", "dtype-mixed_float16"): (
        "measured: inherited -- the denoiser delegates to its InvertibleKernelPCA "
        "child, and the traceback lands on the same invertible_kernel_pca.py:611"
    ),
    ("InvertibleKernelPCADenoiser", "dtype-float64"): (
        "measured: inherited from the same child, invertible_kernel_pca.py:611"
    ),
}


def _params(item: str) -> List[Any]:
    """pytest params for one §16.3 item, carrying that item's measured xfail marks."""
    params = []
    for subject in SUBJECTS:
        reason = XFAILS.get((subject.name, item))
        marks = (
            [pytest.mark.xfail(strict=True, reason=reason)] if reason is not None else []
        )
        params.append(pytest.param(subject, marks=marks, id=subject.name))
    return params


def _dtype_params() -> List[Any]:
    """pytest params for the precision arms, one per (class, policy) pair."""
    params = []
    for subject in SUBJECTS:
        for policy in DTYPE_POLICIES:
            reason = XFAILS.get((subject.name, f"dtype-{policy}"))
            marks = (
                [pytest.mark.xfail(strict=True, reason=reason)]
                if reason is not None
                else []
            )
            params.append(
                pytest.param(
                    subject, policy, marks=marks, id=f"{subject.name}-{policy}"
                )
            )
    return params


# ---------------------------------------------------------------------
# §7.1 -- the round trip, on values
# ---------------------------------------------------------------------


class TestTheKerasRoundTripPreservesValues:
    @pytest.mark.parametrize("subject", _params("round_trip"))
    def test_a_keras_round_trip_reproduces_the_output_values(self, subject: Subject):
        """§16.3 row 1. ``rtol=0`` and an explicit ``training=False``, on VALUES.

        A shape-only round trip is satisfied by a model that restored zero weights, and
        ``assert_allclose``'s default ``rtol=1e-7`` silently loosens a nominally-``atol``
        bound. Both are handled inside the oracle.
        """
        assert_value_round_trip(subject.factory, subject.make_input(), atol=1e-6)


# ---------------------------------------------------------------------
# §8.4 -- weight values, before the loaded model's first call
# ---------------------------------------------------------------------


class TestWeightsAreRestoredBeforeTheFirstCall:
    @pytest.mark.parametrize("subject", _params("weights_restored"))
    def test_saved_weight_values_survive_a_reload_at_atol_zero(
        self, subject: Subject, tmp_path
    ):
        """§16.3 row 2. Compared BEFORE the loaded model's first forward pass.

        After one forward pass a ``build()``-only load path reads the same weight COUNT
        for the correct and the broken variant, because the gap has already been filled
        with fresh random weights.
        """
        sample = subject.make_input()
        assert_weights_restored_before_first_call(
            lambda: functional_model_from_layer(subject.factory, sample),
            sample,
            tmp_path,
        )


# ---------------------------------------------------------------------
# §8.3 -- build parity by relative w.path
# ---------------------------------------------------------------------


class TestExplicitBuildMatchesLazyBuild:
    @pytest.mark.parametrize("subject", _params("build_parity"))
    def test_build_materializes_exactly_the_tree_call_runs(self, subject: Subject):
        """§16.3 row 3. Under-build makes a reload restore into nothing, silently."""
        assert_build_parity(
            subject.factory, subject.input_shape, subject.make_input()
        )

    def test_the_scaler_without_affine_builds_no_affine_weights(self):
        """The §8.3 anti-vacuity sibling: parity alone passes if BOTH paths over-build.

        ``affine=False`` must produce a layer with no ``affine_*`` weight at all, which is
        the "no-sub-layer layout assertion per ``None``/``False`` config" §16.3 asks for.
        """
        layer = UnifiedScaler(num_features=4, affine=False, name="plain")
        layer(_draw(8, 16, 4))
        assert not [w for w in layer.weights if "affine" in w.path], (
            f"affine=False still created {[w.path for w in layer.weights]}"
        )

        affine = UnifiedScaler(num_features=4, affine=True, name="affine")
        affine(_draw(8, 16, 4))
        assert [w for w in affine.weights if "affine" in w.path], (
            "affine=True created no affine weights -- the negative arm above would then "
            "be satisfied by a layer whose affine path is simply dead"
        )


# ---------------------------------------------------------------------
# §13.2.2 -- per-variable gradient flow
# ---------------------------------------------------------------------


class TestGradientsReachEveryTrainableWeight:
    @pytest.mark.parametrize("subject", _params("gradients"))
    def test_every_trainable_variable_receives_a_nonzero_gradient(
        self, subject: Subject
    ):
        """§16.3 row 5. Non-``None`` AND non-zero, named by ``var.path``.

        Most of this package's existing gradient tests assert ``grad is not None`` only.
        §13.2.2 measured that formulation reporting green while 61 of 61 trainable
        weights carried identically-zero gradients.
        """
        assert_gradients_reach_every_trainable_weight(
            subject.factory(), subject.make_input()
        )


# ---------------------------------------------------------------------
# §10.2 / §16.3 -- finiteness on the forward pass
# ---------------------------------------------------------------------


class TestEveryForwardPassIsFinite:
    @pytest.mark.parametrize("subject", _params("finite"))
    def test_the_forward_pass_is_finite(self, subject: Subject):
        """§16.3 row 7. ``ops.all(ops.isfinite(y))``, over every output tensor.

        ``NormalizingFlowLayer`` returns ``(z, log_det)``; a helper that looked at only
        the first output would leave the log-determinant unchecked, which is precisely
        where a degenerate scale would show up first.
        """
        layer = subject.factory()
        assert_forward_is_finite(layer(_as_tensors(subject.make_input())))


def _as_tensors(sample: Any) -> Any:
    if isinstance(sample, (list, tuple)):
        return [ops.convert_to_tensor(a) for a in sample]
    return ops.convert_to_tensor(sample)


# ---------------------------------------------------------------------
# §13.2.6 -- precision arms, with float32 as the control
# ---------------------------------------------------------------------


class TestConstructionAndForwardAcrossDtypePolicies:
    @pytest.mark.parametrize("subject, policy", _dtype_params())
    def test_the_layer_constructs_and_runs_under_the_policy(
        self, subject: Subject, policy: str
    ):
        """§16.3 row 9. ``mixed_float16`` and ``float64``, with a ``float32`` control.

        The float32 arm is not decoration: without it "fp16 is noisy" can masquerade as a
        detected defect (§13.2.6 part 3). The scope also moves ``keras.backend.floatx``
        for the float64 arm, because the policy alone leaves the graph boundary on
        float32 and the arm becomes a fake reading that agrees with float32 to eight
        digits.

        The output dtype is asserted, not just finiteness: a layer that silently casts
        everything back to float32 would pass a finiteness-only arm at every policy.
        """
        with DtypePolicyScope(policy):
            compute = keras.mixed_precision.global_policy().compute_dtype
            layer = subject.factory()
            sample = subject.make_input()
            arrays = sample if isinstance(sample, (list, tuple)) else [sample]
            tensors = [
                ops.convert_to_tensor(np.asarray(a, dtype=compute)) for a in arrays
            ]
            outputs = layer(
                tensors if isinstance(sample, (list, tuple)) else tensors[0]
            )
            assert_forward_is_finite(outputs)

            flat = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            for index, tensor in enumerate(flat):
                assert keras.backend.standardize_dtype(tensor.dtype) == compute, (
                    f"output {index} came back as {tensor.dtype} under the {policy!r} "
                    f"policy, expected the compute dtype {compute!r}"
                )


# ---------------------------------------------------------------------
# §13.2.7 -- XLA versus eager
# ---------------------------------------------------------------------


class TestJitCompiledAgreesWithEager:
    @pytest.mark.parametrize("subject", _params("xla"))
    def test_jit_compile_true_matches_the_eager_result(self, subject: Subject):
        """§16.3 row 10. An eager-only layer is broken on the path ``fit`` takes.

        Keras compiles with ``jit_compile="auto"``, so the compiled path is the default
        training path, not an exotic one.

        The tolerance is 1e-4 on outputs of order 1, and it is MEASURED, not guessed. It
        holds only with TF32 off: §13.6.1 calls TF32 "the default false model defect",
        and this sweep reproduced it exactly. On the RTX 4090 with TF32 ON, four of the
        nine subjects disagree between eager and XLA -- InvertibleKernelPCA 5.42e-04,
        MDNLayer 1.13e-03, AffineCouplingLayer 1.03e-03, NormalizingFlowLayer 3.36e-03 --
        while the identical run on CPU agrees inside 1e-4. Raising the tolerance to
        cover TF32 would have put the bound above a real defect signal; disabling TF32
        removes the confound instead. `Tf32Scope` restores the process-global setting in
        `finally` and asserts the restoration.
        """
        with Tf32Scope(False):
            assert_eager_matches_jit(subject.factory(), subject.make_input(), atol=1e-4)

    def test_the_tf32_scope_restores_the_setting_it_changed(self):
        """Anti-vacuity for the scope above: a scope that leaked would silently change
        the numerics of every test that ran after it in the same process."""
        before = __import__("tensorflow").config.experimental.tensor_float_32_execution_enabled()
        with Tf32Scope(False):
            pass
        after = __import__("tensorflow").config.experimental.tensor_float_32_execution_enabled()
        assert after == before


# ---------------------------------------------------------------------
# a discovery this sweep made: `forward` is not `__call__`
# ---------------------------------------------------------------------


class TestForwardBypassesTheLayerNameScope:
    def test_calling_forward_directly_leaves_the_weights_outside_the_layer_scope(self):
        """``AffineCouplingLayer.forward`` never enters ``Layer.__call__``.

        MEASURED. Built through ``build()``, the weights are
        ``coupling/transformation_net/dense_1/kernel``. Built by calling ``forward``
        directly on a fresh instance, the very same weights are
        ``transformation_net/dense_1/kernel`` -- the owning layer's name segment is
        missing, because ``forward`` bypasses ``__call__`` and therefore its name scope.

        This is the §7.2 hazard ("public methods that bypass lazy build") in its
        weight-path form: two instances of one class disagree about where their weights
        live depending on which entry point built them, which is exactly what breaks
        by-name weight transfer. It is pinned rather than fixed because changing it moves
        weight paths, and that is a checkpoint-affecting change (§6.3).
        """
        sample = [_draw(8, 4), _draw(8, 3)]

        explicit = AffineCouplingLayer(input_dim=4, context_dim=3, name="coupling")
        explicit.build([(None, 4), (None, 3)])

        via_forward = AffineCouplingLayer(input_dim=4, context_dim=3, name="coupling")
        via_forward.forward(
            ops.convert_to_tensor(sample[0]), ops.convert_to_tensor(sample[1])
        )

        assert [w.path for w in explicit.weights][0].startswith("coupling/")
        assert [w.path for w in via_forward.weights][0].startswith("transformation_net/")
        assert relative_weight_paths(explicit) != relative_weight_paths(via_forward)

    def test_the_method_adapter_restores_the_scope(self):
        """The positive arm: routed through ``__call__``, the paths agree again.

        Without this arm the assertion above is also satisfied by an adapter that is
        itself broken, and the sweep's use of ``MethodAdapter`` would be unjustified.
        """
        sample = [_draw(8, 4), _draw(8, 3)]

        def make():
            return MethodAdapter(
                AffineCouplingLayer(input_dim=4, context_dim=3, name="coupling"),
                "forward",
                name="coupling_adapter",
            )

        assert_build_parity(make, [(None, 4), (None, 3)], sample)


# ---------------------------------------------------------------------
# the tenth name: a Callback, not a Layer
# ---------------------------------------------------------------------


class TestTheCallbackIsNotALayer:
    """``ACFMonitorCallback`` measured against §16.3, row by row.

    Stating this explicitly is the point: a silent omission and "N/A" look identical in a
    test report, and the §16.3 checklist is only worth running if every subject is
    accounted for.

    ==================================================== ==============================
    §16.3 row                                            Status for ACFMonitorCallback
    ==================================================== ==============================
    ``.keras`` value round trip                          N/A -- a Callback is not part of
                                                         a saved model; it has no
                                                         ``get_config`` contract and
                                                         ``load_model`` never restores one
    ``atol=0.0`` weight values before first call         N/A -- no weights
    build parity by relative ``w.path``                  N/A -- no ``build``, no weights
    build-through-a-parent probe for constant tables     N/A -- no tables
    per-variable gradient flow                           N/A -- no trainable variables
    every constructor knob pinned (§13.3.2)              APPLIES -- covered below
    ``ops.all(ops.isfinite(y))``                         N/A -- no forward pass
    degenerate lengths on the static path                N/A -- no tensor input
    ``mixed_float16`` / ``float64`` arms                 N/A -- no dtype-carrying state
    ``jit_compile=True`` versus eager                    N/A -- runs in the Python
                                                         callback loop, never traced
    causality / composition / orientation                N/A -- not a tensor transform
    ==================================================== ==============================
    """

    def test_it_is_a_callback_and_not_a_layer(self):
        callback = ACFMonitorCallback(layer_name="acf", log_frequency=10)
        assert isinstance(callback, keras.callbacks.Callback)
        assert not isinstance(callback, keras.layers.Layer), (
            "the N/A column above depends on this: if it were ever made a Layer, every "
            "row marked N/A would become applicable"
        )
        assert not hasattr(callback, "weights")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"layer_name": "acf", "log_frequency": 0}, "log_frequency"),
            ({"layer_name": "acf", "log_frequency": -3}, "log_frequency"),
            ({"layer_name": "", "log_frequency": 10}, "layer_name"),
            ({"layer_name": "   ", "log_frequency": 10}, "layer_name"),
        ],
    )
    def test_its_constructor_knobs_are_validated_at_construction(self, kwargs, match):
        """The one §16.3 row that applies. ``log_frequency=0`` used to construct fine and
        die with a ``ZeroDivisionError`` mid-training, which is a knob validated by the
        first batch rather than by the constructor.
        """
        with pytest.raises(ValueError, match=match):
            ACFMonitorCallback(**kwargs)

    def test_a_valid_configuration_is_accepted(self):
        """The positive arm, so the four rejections above cannot pass on a constructor
        that rejects everything."""
        callback = ACFMonitorCallback(layer_name="acf", log_frequency=1)
        assert callback.layer_name == "acf"
        assert callback.log_frequency == 1
