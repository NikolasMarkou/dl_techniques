"""Guards for the four defects carried into ``plan-2026-08-30T175846-3e8a6ff3``.

One guard per claim, named after the CLAIM rather than the unit (§13.7.1). Every
process-global setting goes through the shared oracle's ``DtypePolicyScope``
(§13.7.4) instead of a fresh one written here (§9.1 reuse order).

MEASURED status at ``f175537a4`` (the commit that introduces this file, all four
defects still present). ``CUDA_VISIBLE_DEVICES=""``, batch 6, input_dim 8:

===== ================================================== =============== ==============
Set   Claim                                              mixed_float16   float64
===== ================================================== =============== ==============
A1    ``DeepKernelPCA`` (1 level) forwards               RED at :714     GREEN
A2    ``DeepKernelPCA`` (2 levels) forwards              RED at :714 *   RED at :1170
A3    ``InvertibleKernelPCA`` forwards                   RED at :611     RED at :611
A4    ``…Denoiser(adaptive_components=True)`` forwards   RED at :611 *   RED at :611 *
A5    no all-scalar ``ops.*`` call remains               RED (4 sites)
B     weight paths agree across entry points            RED (forward / inverse / sample)
C     ``inverse_transform(mean=, std=)``                RED (TypeError, no such kwarg)
D     gradients reach the full public surface           GREEN (a NEW claim, pinned)
===== ================================================== =============== ==============

The two cells marked ``*`` are MASKED: the guard is red, but for an *upstream*
site's reason, not its own. This is the §13.4.9 trap ("a RED proof written against
the reported bug reproduces the wrong one"), so the two dtype guards below report
the raising source line and the masking is measured, not assumed:

* A2 under ``mixed_float16`` dies at ``deep_kernel_pca.py:714`` -- site 1, which sits
  earlier on the same ``call()``. Isolated by shimming ``ops.eye`` to follow the
  compute dtype, site 2 then dies at its own ``deep_kernel_pca.py:1170``, under BOTH
  policies. So A2's true before-value must be re-measured after step 2 lands site 1.
* A4 dies at ``invertible_kernel_pca.py:611`` -- site 3's consuming line
  ``rff_features = scale * cos_features``, inside the ``ikpca`` CHILD --
  under both policies, with the message ``expected to be a float tensor but is a
  half tensor``. That polarity is site 3's (``scale`` is float32, ``cos_features`` is
  half) and is IDENTICAL to the ``adaptive_components=False`` control, which cannot
  reach site 4 at all. Isolated by shimming only site 3's ``ops.sqrt(2.0 / n)`` form,
  site 4 then dies at its own ``invertible_kernel_pca.py:1081`` with the REVERSED
  polarity ``expected to be a half tensor but is a float tensor``. **A4's true
  before-value must be re-measured after step 3 lands site 3** -- today's red is not
  its own.

Guard D is GREEN at HEAD and is not a RED-first proof. It pins a NEW claim (the
weight is reachable through the layer's full public surface, not through ``call()``
alone). Its power is proven separately by injection, in
``test_the_gradient_guard_rejects_a_genuinely_unreachable_weight``.
"""

import ast
import pathlib
import traceback

import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

from dl_techniques.layers.statistics.deep_kernel_pca import DeepKernelPCA
from dl_techniques.layers.statistics.invertible_kernel_pca import (
    InvertibleKernelPCA,
    InvertibleKernelPCADenoiser,
)
from dl_techniques.layers.statistics.normalizing_flow import (
    AffineCouplingLayer,
    NormalizingFlowLayer,
)
from dl_techniques.layers.statistics.scaler import UnifiedScaler

from .v2_compliance_oracle import DtypePolicyScope, assert_forward_is_finite

# The package source location and its anti-vacuity floor already exist, resolved
# from the imported module so a scan can never read a different checkout than the
# one under test. Reuse them rather than deriving a second copy (§9.1).
from .test_the_package_is_v2_compliant import PACKAGE_DIR, _package_sources


# The two policies that expose the never-narrow class. ``float32`` is excluded on
# purpose: it is the control, and it is GREEN at every site by construction, so
# including it here would dilute the RED signal these guards exist to carry. The
# float32 bit-identity control belongs with the fix (plan SC3), not here.
NARROWING_POLICIES = ("mixed_float16", "float64")

# DeepKernelPCA reuses projection-weight rows as per-sample coefficients, so the
# feature dim must be >= the batch size. Same constraint the package's own fixtures
# carry (test_deep_kernel_pca.py:35-43).
BATCH_SIZE = 6
INPUT_DIM = 8


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype("float32")


def _as_compute_dtype(array: np.ndarray):
    """Cast to the ACTIVE policy's compute dtype.

    Feeding float32 under ``mixed_float16`` would let ``Layer.__call__``'s autocast
    hide the defect at the boundary; the guard has to hand the layer a tensor that is
    already narrow.
    """
    return ops.cast(
        ops.convert_to_tensor(array),
        keras.mixed_precision.global_policy().compute_dtype,
    )


def _forward_or_raising_site(build, apply):
    """Run a forward pass and, if it raises, name the source line that raised.

    Returns ``(output, None)`` on success and ``(None, "<file>:<line>")`` on failure,
    where the site is the DEEPEST ``dl_techniques`` frame in the traceback.

    Attribution is the whole point: two of the four dtype sites are masked by an
    upstream site, and a guard that only reports "it raised" would happily accept the
    wrong defect as its own RED proof (§13.4.9).
    """
    try:
        return apply(build()), None
    except Exception:
        frames = [
            frame
            for frame in traceback.extract_tb(traceback.sys.exc_info()[2])
            if "dl_techniques" in frame.filename
        ]
        if not frames:
            raise
        deepest = frames[-1]
        return None, f"{pathlib.Path(deepest.filename).name}:{deepest.lineno}"


def _assert_forwards_at_policy(policy, build, apply, own_site):
    """The forward pass completes and returns the policy's compute dtype.

    ``own_site`` is the ``<file>:<line>`` this guard's defect lives at. When the
    forward raises somewhere ELSE, the failure message says so explicitly, so a
    masked guard cannot be misread as a proof about its own site.
    """
    with DtypePolicyScope(policy):
        expected = keras.mixed_precision.global_policy().compute_dtype
        output, site = _forward_or_raising_site(build, apply)
        assert site is None, (
            f"forward raised under {policy}: the deepest dl_techniques frame is "
            f"{site}. This guard's own defect is at {own_site}."
            + (
                ""
                if site.endswith(own_site.split(":")[-1])
                else f" {site} is a DIFFERENT site -- this red is MASKED and is not "
                f"evidence about {own_site}."
            )
        )
        assert_forward_is_finite(output)
        assert keras.backend.standardize_dtype(output.dtype) == expected, (
            f"under {policy} the output is {output.dtype}, not the policy's compute "
            f"dtype {expected}"
        )


# ---------------------------------------------------------------------
# GUARD SET A -- the four never-narrow dtype sites
# ---------------------------------------------------------------------

class TestEveryAllScalarOpsCallFollowsTheActivePolicy:
    """An ``ops.*`` call built only from Python scalars must not pin float32.

    The MEASURED rule: such a call materializes a tensor at ``backend.floatx()``
    whatever the active policy, and combining it with a differently-dtyped tensor
    raises. Bare scalar arithmetic straight against a tensor (``x * 0.5``) is SAFE --
    the tensor's dtype wins and no promotion tensor is ever created.
    """

    @pytest.mark.parametrize("policy", NARROWING_POLICIES)
    def test_site_1_deep_kernel_pca_eye_follows_the_policy(self, policy, sample_data):
        """``deep_kernel_pca.py:714`` -- ``ops.eye(batch_size)``.

        One level, so the multi-level branch holding site 2 never runs and this
        guard reads site 1 alone.

        MEASURED at HEAD: RED at ``:714`` under ``mixed_float16``; GREEN under
        ``float64``, because ``ops.eye`` with no ``dtype=`` follows
        ``backend.floatx()``, which the float64 arm sets. The defect is real but
        only exposed by a policy whose COMPUTE dtype differs from ``floatx``.
        """
        _assert_forwards_at_policy(
            policy,
            lambda: DeepKernelPCA(num_levels=1, components_per_level=[BATCH_SIZE]),
            lambda layer: layer(_as_compute_dtype(sample_data)),
            own_site="deep_kernel_pca.py:714",
        )

    @pytest.mark.parametrize("policy", NARROWING_POLICIES)
    def test_site_2_deep_kernel_pca_level_weight_follows_the_policy(
        self, policy, sample_data
    ):
        """``deep_kernel_pca.py:1169`` -- ``ops.exp(-0.1 * i)``, used at ``:1170``.

        Needs two levels: the exponential level weighting only runs when there is
        more than one level to combine.

        MEASURED at HEAD: under ``float64`` this is RED at its own ``:1170``. Under
        ``mixed_float16`` it is red at ``:714`` -- site 1, MASKING it. Shimming
        ``ops.eye`` to follow the compute dtype moves the mixed_float16 failure to
        ``:1170`` too, so the true mixed_float16 before-value must be re-read after
        step 2 lands site 1.
        """
        _assert_forwards_at_policy(
            policy,
            lambda: DeepKernelPCA(
                num_levels=2, components_per_level=[BATCH_SIZE, BATCH_SIZE]
            ),
            lambda layer: layer(_as_compute_dtype(sample_data)),
            own_site="deep_kernel_pca.py:1170",
        )

    @pytest.mark.parametrize("policy", NARROWING_POLICIES)
    def test_site_3_invertible_kernel_pca_scale_follows_the_policy(
        self, policy, sample_data
    ):
        """``invertible_kernel_pca.py:610`` -- ``ops.sqrt(2.0 / n_random_features)``.

        MEASURED at HEAD: RED under both policies. The raising line is ``:611``,
        ``rff_features = scale * cos_features`` -- the CONSUMER of the float32
        ``scale`` built one line above. A never-narrow site is always reported at
        the line that combines the pinned tensor with the policy-dtyped one, not at
        the line that created it, and the guard's ``own_site`` has to name the
        former or its own red reads as masked.
        """
        _assert_forwards_at_policy(
            policy,
            lambda: InvertibleKernelPCA(n_components=3, n_random_features=16),
            lambda layer: layer(_as_compute_dtype(sample_data)),
            own_site="invertible_kernel_pca.py:611",
        )

    @pytest.mark.parametrize("policy", NARROWING_POLICIES)
    def test_site_4_the_adaptive_threshold_follows_the_policy(
        self, policy, sample_data
    ):
        """``invertible_kernel_pca.py:1081`` -- ``noise_level * ops.sqrt(2.0)``.

        The MASKED site. It runs only with ``adaptive_components=True`` AND a true
        ``training`` flag, and site 3 -- inside the ``ikpca`` child, reached first on
        the same ``call()`` -- raises before it.

        MEASURED at HEAD: RED at ``:611`` (site 3) under both policies, with the
        message ``expected to be a float tensor but is a half tensor``. That is site
        3's polarity, and it is IDENTICAL to the ``adaptive_components=False``
        control, which cannot execute site 4 at all -- so today's red says nothing
        about site 4.

        With only site 3 shimmed away, this dies at its own ``:1081`` under both
        policies, with the REVERSED polarity ``expected to be a half tensor but is a
        float tensor`` (``noise_level`` is half, ``ops.sqrt(2.0)`` is float32).

        **Its true before-value must be re-measured after step 3.**
        """
        _assert_forwards_at_policy(
            policy,
            lambda: InvertibleKernelPCADenoiser(
                n_components=3, n_random_features=16, adaptive_components=True
            ),
            lambda layer: layer(_as_compute_dtype(sample_data), training=True),
            own_site="invertible_kernel_pca.py:1081",
        )


# ---------------------------------------------------------------------
# GUARD SET A5 -- the package-wide rule scan (plan SC1)
# ---------------------------------------------------------------------

# ``ops.*`` calls that MATERIALIZE a value from their arguments. These are the ones
# whose result dtype cannot be inherited from an input tensor, so an all-scalar
# argument list pins them to ``backend.floatx()``.
#
# Deliberately NOT listed: dtype-PRESERVING ops (``abs``, ``transpose``,
# ``concatenate``, ``sum``, ``where``, ``zeros_like``, ``shape``, ``convert_to_numpy``
# ...). Including them takes the scan from 4 flagged sites to 18, all false positives
# -- measured. Widening this set is not free.
MATERIALIZING_OPS = frozenset({
    "ops.eye", "ops.exp", "ops.sqrt", "ops.log", "ops.power", "ops.zeros",
    "ops.ones", "ops.full", "ops.arange", "ops.linspace", "ops.cast",
    "ops.convert_to_tensor",
})

# Anti-vacuity floor for the scan. It inspects 188 ``ops.*`` calls across the package
# today; a parse failure, a renamed package or a broken matcher would otherwise leave
# it reporting "0 offenders" while inspecting nothing (§13.4.8).
MIN_OPS_CALLS_INSPECTED = 50


def _ops_root(func):
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_ops_call(node):
    return isinstance(node, ast.Call) and _ops_root(node.func) == "ops"


def _ops_call_name(node):
    return ast.unparse(node.func)


def _rhs_is_tensor_valued(value, tensor_names):
    """Whether an assignment's right-hand side yields a TENSOR.

    ``ops.shape(...)`` is excluded on purpose: a dimension read off it is a
    Python/symbolic integer, not a tensor, which is exactly why
    ``ops.eye(ops.shape(k)[0])`` is a defect site and not a safe one.
    """
    calls = [node for node in ast.walk(value) if _is_ops_call(node)]
    if calls:
        return not all(_ops_call_name(call) == "ops.shape" for call in calls)
    return any(
        isinstance(node, ast.Name) and node.id in tensor_names
        for node in ast.walk(value)
    )


def _tensor_valued_names(function_node):
    """Names bound to tensors inside one function body.

    Seeded with the parameters (minus ``self`` and ``training``) and propagated
    through assignments and ``for`` targets. This is what lets the scan tell
    ``ops.eye(batch_size)`` -- an int -- apart from ``ops.exp(-gamma * distances)``,
    where ``distances`` is a tensor. A purely syntactic "does any Name appear"
    rule cannot: it flags only 1 of the 4 known sites, because sites 1-3 all pass a
    Name or Attribute that happens to hold a Python int. Measured.
    """
    names = {
        arg.arg
        for arg in function_node.args.args + function_node.args.kwonlyargs
    } - {"self", "training"}
    for node in ast.walk(function_node):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None:
                continue
            tensor_valued = _rhs_is_tensor_valued(node.value, names)
        elif isinstance(node, ast.For):
            targets = [node.target]
            tensor_valued = _rhs_is_tensor_valued(node.iter, names)
        else:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if not isinstance(sub, ast.Name):
                    continue
                if tensor_valued:
                    names.add(sub.id)
                else:
                    names.discard(sub.id)
    return names


def _argument_is_scalar_only(argument, tensor_names):
    for node in ast.walk(argument):
        if _is_ops_call(node) or isinstance(node, ast.Subscript):
            return False
        if isinstance(node, ast.Name) and node.id in tensor_names:
            return False
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            # ``self.some_int_config`` is a scalar; ``some_tensor.dtype`` is not.
            if not (isinstance(base, ast.Name) and base.id == "self"):
                return False
    return True


def scan_all_scalar_ops_calls(source, filename="<source>"):
    """Return ``(inspected, offenders)`` for one module's source text.

    An offender is a materializing ``ops.*`` call with no ``dtype=`` keyword whose
    every argument is built only from literals, ``self.<scalar>`` attributes and
    names that are not tensor-valued in the enclosing function.
    """
    tree = ast.parse(source)
    inspected = 0
    offenders = []
    for function_node in ast.walk(tree):
        if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tensor_names = _tensor_valued_names(function_node)
        for node in ast.walk(function_node):
            if not _is_ops_call(node):
                continue
            inspected += 1
            if _ops_call_name(node) not in MATERIALIZING_OPS:
                continue
            if any(keyword.arg == "dtype" for keyword in node.keywords):
                continue
            arguments = list(node.args) + [kw.value for kw in node.keywords]
            if arguments and all(
                _argument_is_scalar_only(argument, tensor_names)
                for argument in arguments
            ):
                offenders.append(
                    f"{filename}:{node.lineno} {ast.unparse(node)}"
                )
    return inspected, offenders


class TestNoAllScalarOpsCallRemainsInThePackage:
    """Plan SC1: 0 never-narrow dtype sites under the derived rule (baseline 4)."""

    def test_the_scan_is_not_structurally_blind(self):
        """The scan flags a planted offender and spares its negative controls.

        A scan that reports "0 offenders" is worthless until it has been seen
        rejecting something. Both halves are here: the two planted defects must be
        found, and the five planted safe forms must NOT be, or the scan would pass
        the package by being blind rather than by the package being clean.
        """
        planted = (
            "from keras import ops\n"
            "class Subject:\n"
            "    def call(self, inputs):\n"
            "        n = ops.shape(inputs)[0]\n"
            "        bad_eye = ops.eye(n)\n"
            "        bad_scalar = ops.sqrt(2.0)\n"
            "        good_eye = ops.eye(n, dtype=inputs.dtype)\n"
            "        good_scaled = inputs * 0.5\n"
            "        good_tensor_arg = ops.sqrt(ops.abs(inputs) + 1e-10)\n"
            "        good_from_name = ops.exp(-self.gamma * inputs)\n"
            "        good_cast = ops.cast(good_scaled, inputs.dtype)\n"
            "        return bad_eye, bad_scalar, good_eye, good_scaled, "
            "good_tensor_arg, good_from_name, good_cast\n"
        )
        inspected, offenders = scan_all_scalar_ops_calls(planted, "planted.py")
        assert inspected >= 7, f"the scan only saw {inspected} ops.* calls in the plant"
        found = sorted(offender.split(" ", 1)[1] for offender in offenders)
        assert found == ["ops.eye(n)", "ops.sqrt(2.0)"], (
            f"the scan did not isolate the two planted defects; it reported {found}"
        )

    def test_the_package_holds_no_all_scalar_ops_call(self):
        """MEASURED at HEAD: RED, listing exactly the 4 known sites.

        ``deep_kernel_pca.py:714``, ``:1169``, ``invertible_kernel_pca.py:610`` and
        ``:1081`` -- and nothing else, out of 188 ``ops.*`` calls inspected. No fifth
        site exists in this package under the derived rule.
        """
        sources = _package_sources()
        total_inspected = 0
        offenders = []
        for path in sources:
            inspected, found = scan_all_scalar_ops_calls(
                path.read_text(), path.name
            )
            total_inspected += inspected
            offenders.extend(found)
        assert total_inspected >= MIN_OPS_CALLS_INSPECTED, (
            f"the scan inspected only {total_inspected} ops.* calls across "
            f"{len(sources)} files in {PACKAGE_DIR}; it is not seeing the package"
        )
        assert offenders == [], (
            "these ops.* calls build a tensor from Python scalars alone, so they pin "
            "backend.floatx() whatever the active policy:\n  "
            + "\n  ".join(offenders)
        )


# ---------------------------------------------------------------------
# GUARD SET B -- defect #3, weight-path stability across entry points
# ---------------------------------------------------------------------

# Today's ``build()``-produced relative paths, captured at f175537a4 BEFORE any fix.
# This is the pre-fix truth being PRESERVED, not a new layout being introduced: the
# functional path (``NormalizingFlowLayer.__call__``) already writes exactly these,
# and the fix must leave them untouched (plan SC6, and its "the #3 fix moves weight
# paths -> STOP" falsification signal).
COUPLING_BUILD_PATHS = [
    "transformation_net/dense_1/bias",
    "transformation_net/dense_1/kernel",
    "transformation_net/dense_2/bias",
    "transformation_net/dense_2/kernel",
    "transformation_net/output_dense/bias",
    "transformation_net/output_dense/kernel",
]

FLOW_STEPS = 2
FLOW_BUILD_PATHS = sorted(
    f"affine_coupling_{step}/{leaf}"
    for step in range(FLOW_STEPS)
    for leaf in COUPLING_BUILD_PATHS
)

COUPLING_DIM = 6
CONTEXT_DIM = 3
HIDDEN_UNITS = 8


def _relative_paths(layer):
    """Sorted ``w.path`` with the root segment dropped.

    Same shape as the oracle's ``relative_weight_paths``, which takes a MODEL; this
    takes a bare layer, and the root has to go because Keras auto-increments
    generated names per instance.
    """
    return sorted(weight.path.split("/", 1)[-1] for weight in layer.weights)


@pytest.fixture
def coupling_inputs():
    rng = np.random.default_rng(0)
    return (
        ops.convert_to_tensor(
            rng.standard_normal((4, COUPLING_DIM)).astype("float32")
        ),
        ops.convert_to_tensor(
            rng.standard_normal((4, CONTEXT_DIM)).astype("float32")
        ),
    )


def _make_coupling():
    return AffineCouplingLayer(
        input_dim=COUPLING_DIM,
        context_dim=CONTEXT_DIM,
        hidden_units=HIDDEN_UNITS,
        name="coupling",
    )


def _make_flow():
    return NormalizingFlowLayer(
        output_dimension=COUPLING_DIM,
        num_flow_steps=FLOW_STEPS,
        context_dim=CONTEXT_DIM,
        hidden_units_coupling=HIDDEN_UNITS,
        name="flow",
    )


class TestAWeightPathDoesNotDependOnWhichEntryPointBuiltIt:
    """Defect #3. Every first touch must produce ``build()``'s layout.

    MEASURED at HEAD: ``forward``, ``inverse`` and ``sample`` build the
    ``transformation_net`` sub-layer by calling it directly, outside the owning
    layer's ``build()``/``__call__``, so Keras never enters the owning name scope and
    the weights land at a bare ``dense_1/kernel`` -- losing both the
    ``transformation_net/`` and (for the flow) the ``affine_coupling_i/`` prefixes.
    """

    def test_the_build_reference_layout_is_what_this_pins(self):
        """Anti-vacuity: the reference constant is the layout ``build()`` produces.

        GREEN at HEAD by construction. Without it the three guards below could be
        pinned to a constant that no entry point has ever produced, and would then
        fail forever for the wrong reason.
        """
        layer = _make_coupling()
        layer.build([(None, COUPLING_DIM), (None, CONTEXT_DIM)])
        assert _relative_paths(layer) == COUPLING_BUILD_PATHS
        assert layer.built is True

        flow = _make_flow()
        flow.build([(None, COUPLING_DIM), (None, CONTEXT_DIM)])
        assert _relative_paths(flow) == FLOW_BUILD_PATHS

    def test_forward_as_first_touch_agrees_with_build(self, coupling_inputs):
        """MEASURED at HEAD: RED. ``forward`` yields ``dense_1/kernel`` and leaves
        ``built`` False."""
        z, context = coupling_inputs
        layer = _make_coupling()
        layer.forward(z, context)
        assert _relative_paths(layer) == COUPLING_BUILD_PATHS
        assert layer.built is True, "forward() left the layer unbuilt"

    def test_inverse_as_first_touch_agrees_with_build(self, coupling_inputs):
        """MEASURED at HEAD: RED, identically to ``forward``. The pinned defect
        report only ever covered ``forward``; ``inverse`` has the same exposure."""
        z, context = coupling_inputs
        layer = _make_coupling()
        layer.inverse(z, context)
        assert _relative_paths(layer) == COUPLING_BUILD_PATHS
        assert layer.built is True, "inverse() left the layer unbuilt"

    def test_sample_as_first_touch_agrees_with_build(self, coupling_inputs):
        """MEASURED at HEAD: RED. This is the ORDINARY-use arm.

        ``sample()`` before any ``call()`` is a normal public entry point, not the
        artificial "call ``forward`` directly" case, and it collapses every coupling
        layer's weights to the same bare ``dense_1/kernel`` -- which is why the flow's
        12 weights read as 6 duplicated pairs.
        """
        _, context = coupling_inputs
        flow = _make_flow()
        flow.sample(2, context)
        assert _relative_paths(flow) == FLOW_BUILD_PATHS
        assert flow.built is True, "sample() left the flow unbuilt"

    def test_call_as_first_touch_agrees_with_build(self, coupling_inputs):
        """GREEN at HEAD. ``NormalizingFlowLayer.call`` routes through
        ``Layer.__call__``, which runs ``build()``, so this entry point was never
        exposed. Pinned so a fix for the other three cannot quietly move it."""
        z, context = coupling_inputs
        flow = _make_flow()
        flow([z, context])
        assert _relative_paths(flow) == FLOW_BUILD_PATHS


# ---------------------------------------------------------------------
# GUARD SET C -- defect #4, explicit statistics on inverse_transform
# ---------------------------------------------------------------------

class TestAScalerCanBeInvertedAfterFitWithExplicitStatistics:
    """Defect #4. ``_last_mean``/``_last_std`` are per-sample tensors scoped to the
    trace that made them, so after ``model.fit`` there is nothing left to invert
    with. The caller who kept the statistics must be able to supply them."""

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        return (rng.standard_normal((8, 20, 10)) * 5 + 3).astype("float32")

    def test_explicit_statistics_reconstruct_after_fit(self):
        """MEASURED at HEAD: RED --
        ``TypeError: UnifiedScaler.inverse_transform() got an unexpected keyword
        argument 'mean'``. The kwargs do not exist yet."""
        data = self._data()
        # affine=True so the model fit() below actually has a trainable weight.
        # A Sequential with none warns, and this suite turns warnings into errors.
        scaler = UnifiedScaler(axis=1, store_stats=True, affine=True)
        scaled_eager = scaler(ops.convert_to_tensor(data))
        mean = ops.convert_to_numpy(scaler._last_mean)
        std = ops.convert_to_numpy(scaler._last_std)

        model = keras.Sequential([keras.Input(shape=(20, 10)), scaler])
        model.compile(optimizer="adam", loss="mse")
        model.fit(data, data, epochs=1, batch_size=4, verbose=0)
        scaled = model.predict(data, verbose=0)

        restored = ops.convert_to_numpy(
            scaler.inverse_transform(
                ops.convert_to_tensor(scaled), mean=mean, std=std
            )
        )
        np.testing.assert_allclose(restored, data, atol=1e-4, rtol=0)
        del scaled_eager

    def test_the_no_argument_path_after_fit_still_raises_legibly(self):
        """GREEN at HEAD and must stay green. Adding the kwargs must not turn the
        legible ``RuntimeError`` into silence or into a raw ``TypeError``."""
        data = self._data()
        # affine=True for the same reason as the guard above: a Sequential with no
        # trainable weight warns during fit(), and warnings are errors here.
        scaler = UnifiedScaler(axis=1, store_stats=True, affine=True)
        model = keras.Sequential([keras.Input(shape=(20, 10)), scaler])
        model.compile(optimizer="adam", loss="mse")
        model.fit(data, data, epochs=1, batch_size=4, verbose=0)
        scaled = model.predict(data, verbose=0)

        with pytest.raises(RuntimeError, match="traced call"):
            scaler.inverse_transform(ops.convert_to_tensor(scaled))

    def test_the_eager_no_argument_path_still_reconstructs(self):
        """The positive twin, GREEN at HEAD. Without it the guard above would pass
        just as happily against an ``inverse_transform`` that raises for everyone."""
        data = self._data()
        scaler = UnifiedScaler(axis=1, store_stats=True)
        scaled = scaler(ops.convert_to_tensor(data))
        restored = ops.convert_to_numpy(scaler.inverse_transform(scaled))
        np.testing.assert_allclose(restored, data, atol=1e-4, rtol=0)

    def test_the_batch_averaged_statistics_are_a_trap_not_a_shortcut(self):
        """The rejected option, pinned so nobody re-proposes it.

        ``get_stats()`` returns ``stored_mean``/``stored_std``, which average over
        the BATCH axis. Both documented personas reduce over an axis that excludes
        the batch (``axis=1`` RevIN, ``axis=-1`` StandardScaler), so those summaries
        are simply the wrong numbers. MEASURED: mean absolute error **1.02**
        (``axis=1``) and **1.40** (``axis=-1``), max relative error **128-136x**,
        against an exact per-sample inverse that restores to ~1.9e-06. Exact only at
        ``axis=0``, which neither persona uses.
        """
        data = self._data()
        scaler = UnifiedScaler(axis=1, store_stats=True)
        scaled = ops.convert_to_numpy(scaler(ops.convert_to_tensor(data)))

        exact = ops.convert_to_numpy(
            scaled * scaler._last_std + scaler._last_mean
        )
        batch_averaged = (
            scaled
            * ops.convert_to_numpy(scaler.stored_std)
            + ops.convert_to_numpy(scaler.stored_mean)
        )

        exact_error = float(np.abs(exact - data).mean())
        trap_error = float(np.abs(batch_averaged - data).mean())
        assert exact_error < 1e-4, (
            f"the per-sample inverse is not exact ({exact_error}); this guard's "
            f"comparison is meaningless without it"
        )
        assert trap_error > 0.5, (
            f"the batch-averaged inverse is only off by {trap_error}; the measured "
            f"trap (~1.02 at axis=1) did not reproduce, so re-derive before trusting "
            f"either number"
        )
        assert trap_error > 1000 * exact_error, (
            f"batch-averaged error {trap_error} is not materially worse than the "
            f"exact error {exact_error}"
        )


# ---------------------------------------------------------------------
# GUARD SET D -- defect #2, gradients over the FULL public surface
# ---------------------------------------------------------------------

def _assert_gradients_reach_every_trainable_weight_over_the_public_surface(layer, inputs):
    """A loss built from BOTH ``call()`` and ``inverse_transform()`` reaches every
    trainable weight, non-None and non-zero, named by ``var.path``.

    The oracle's ``assert_gradients_reach_every_trainable_weight`` drives the layer
    through ``call()`` alone, which is exactly why this layer xfails there:
    ``reconstruction_matrix`` is read only inside ``inverse_transform``. This is the
    same assertion shape -- including the ``len(trainable_variables) > 0``
    anti-vacuity check -- over a wider surface.
    """
    with tf.GradientTape() as tape:
        components = layer(inputs, training=True)
        reconstructed = layer.inverse_transform(components)
        loss = ops.mean(ops.square(ops.cast(components, "float32"))) + ops.mean(
            ops.square(ops.cast(inputs - reconstructed, "float32"))
        )

    variables = list(layer.trainable_variables)
    assert len(variables) > 0, (
        f"{type(layer).__name__} exposed no trainable variables -- this gradient "
        f"check would be vacuous"
    )
    gradients = tape.gradient(loss, variables)
    for variable, gradient in zip(variables, gradients):
        assert gradient is not None, f"no gradient for {variable.path}"
        assert np.any(np.asarray(ops.convert_to_numpy(gradient)) != 0.0), (
            f"all-zero gradient for {variable.path}"
        )


class TestEveryTrainableWeightIsReachableFromThePublicSurface:
    """Defect #2. ``reconstruction_matrix`` is trainable and on-path -- through
    ``inverse_transform``, which is public, not through ``call()`` alone.

    MEASURED at HEAD: GREEN. This is a NEW claim being pinned, not a RED-first
    proof. Its power is proven by the injection guard below.
    """

    @staticmethod
    def _subject(cls=InvertibleKernelPCA):
        layer = cls(n_components=3, n_random_features=16)
        layer.build((None, INPUT_DIM))
        return layer

    def test_the_full_public_surface_reaches_every_trainable_weight(self, sample_data):
        """GREEN at HEAD. All 3 default trainable weights receive a live gradient
        once ``inverse_transform`` is on the loss path."""
        _assert_gradients_reach_every_trainable_weight_over_the_public_surface(
            self._subject(), ops.convert_to_tensor(sample_data)
        )

    def test_it_also_holds_with_trainable_frequencies(self, sample_data):
        """GREEN at HEAD. The 4-weight configuration, so the guard is not pinned to
        one arrangement of the layer."""
        layer = InvertibleKernelPCA(
            n_components=3, n_random_features=16, trainable_frequencies=True
        )
        layer.build((None, INPUT_DIM))
        assert len(layer.trainable_variables) == 4
        _assert_gradients_reach_every_trainable_weight_over_the_public_surface(
            layer, ops.convert_to_tensor(sample_data)
        )

    def test_call_alone_does_not_reach_the_reconstruction_weights(self, sample_data):
        """The counterpart fact, pinned so the guard above is not read as claiming
        more than it does.

        MEASURED at HEAD: a loss on ``call()``'s output alone leaves
        ``reconstruction_matrix`` and ``reconstruction_bias`` with ``None``
        gradients. ``call()`` returns components and never runs
        ``inverse_transform``. This is the layer's DESIGN -- the inverse is a learned
        decoder the CALLER must train -- and this guard records it as such, not as a
        defect awaiting a fix.
        """
        layer = self._subject()
        inputs = ops.convert_to_tensor(sample_data)
        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(layer(inputs, training=True)))
        variables = list(layer.trainable_variables)
        assert len(variables) > 0
        gradients = tape.gradient(loss, variables)
        unreached = {
            variable.path.rsplit("/", 1)[-1]
            for variable, gradient in zip(variables, gradients)
            if gradient is None
        }
        assert unreached == {"reconstruction_matrix", "reconstruction_bias"}, (
            f"call()-only reachability changed: {unreached}"
        )

    def test_the_gradient_guard_rejects_a_genuinely_unreachable_weight(
        self, sample_data
    ):
        """The RED proof for guard D. A guard that passes on everything is not a
        guard.

        Injects a trainable weight that nothing reads and asserts the guard rejects
        it. Without this, the GREEN reading above would be indistinguishable from a
        guard that cannot fail.
        """

        class DeadWeightSubject(InvertibleKernelPCA):
            def build(self, input_shape):
                super().build(input_shape)
                self._dead = self.add_weight(
                    name="dead", shape=(3,), initializer="ones", trainable=True
                )

        layer = self._subject(DeadWeightSubject)
        with pytest.raises(AssertionError, match="no gradient for .*dead"):
            _assert_gradients_reach_every_trainable_weight_over_the_public_surface(
                layer, ops.convert_to_tensor(sample_data)
            )
