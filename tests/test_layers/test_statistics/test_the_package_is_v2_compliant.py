"""Guards for the v2-guide (§16.1) compliance violations in ``layers/statistics``.

One guard class per violation recorded in the audit
(`plans/plan-2026-08-30T160738-b462cffc/findings.md`). Each guard is named after the
CLAIM it pins, not after the unit it exercises (§13.7.1).

Status at the commit that introduces this file (unfixed source):

===== ===================================================== ==========
ID    Claim                                                 Pre-fix
===== ===================================================== ==========
V10   ``UnifiedScaler.inverse_transform`` survives tracing   RED (b, c)
V1    ``DeepKernelPCA.compute_output_shape`` works unbuilt   RED (x4)
V2    no sub-layer is created in ``build()``                 RED
V3    every ``ValueError`` names the offending value         RED (19)
V4    cross-parameter contracts raise at construction        RED (x2)
V6    ``training=`` is forwarded in ``normalizing_flow``     RED
V7    ``ACFMonitorCallback`` validates its arguments         RED
V8    no declared-but-unused ``eigenvectors`` field          RED
V11   the in-``call`` assign DOES work under XLA             GREEN by design
===== ===================================================== ==========

V11 is deliberately GREEN before the fix. It pins a MEASURED fact that a source
comment currently denies; the fix is to correct the comment, and this guard exists so
the corrected comment stays true. It is not a RED-first guard and must not be counted
as one.
"""

import ast
import pathlib

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.layers.statistics.scaler import UnifiedScaler
from dl_techniques.layers.statistics.deep_kernel_pca import DeepKernelPCA
from dl_techniques.layers.statistics.invertible_kernel_pca import (
    InvertibleKernelPCA,
    InvertibleKernelPCADenoiser,
)
from dl_techniques.layers.statistics.residual_acf import ACFMonitorCallback


# ---------------------------------------------------------------------
# Where the package's source lives, resolved from the imported module so the
# scan can never read a different checkout than the one under test.
# ---------------------------------------------------------------------
import dl_techniques.layers.statistics.scaler as _scaler_module

PACKAGE_DIR = pathlib.Path(_scaler_module.__file__).resolve().parent

# The scan must see every module in the package. If an import or a path change ever
# shrinks this set, the parametrized guards below would pass while testing nothing.
MIN_SOURCE_FILES = 7


def _package_sources():
    return sorted(p for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py")


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(class_node, name):
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


# =====================================================================
# V10 [HIGH] -- UnifiedScaler.inverse_transform dies after tracing
# =====================================================================
class TestTheScalerCanInvertAfterItHasBeenTraced:
    """``_last_mean`` / ``_last_std`` (``scaler.py`` ~:396) are plain Python attributes.

    Under a traced call they capture GRAPH tensors, so a later
    ``inverse_transform`` raises ``TypeError: <tf.Tensor ...> is out of scope and
    cannot be used here``. MEASURED: eager works, ``tf.function`` and ``model.fit``
    both raise. The fix (step 2) makes them non-trainable ``keras.Variable``s created
    in ``build()``, the idiom the same file already uses for ``stored_mean``.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(1234)
        return rng.normal(loc=3.0, scale=2.0, size=(8, 5)).astype("float32")

    def test_inverse_transform_reconstructs_the_input_eagerly(self):
        """The regime that already works. Pinned so the fix cannot regress it."""
        x = self._data()
        scaler = UnifiedScaler(axis=-1)
        y = scaler(x)
        recovered = keras.ops.convert_to_numpy(scaler.inverse_transform(y))
        np.testing.assert_allclose(recovered, x, atol=1e-4, rtol=0)

    def test_inverse_transform_reconstructs_the_input_after_a_traced_call(self):
        """RED pre-fix: TypeError, the captured graph tensor is out of scope."""
        x = self._data()
        scaler = UnifiedScaler(axis=-1)
        y = keras.ops.convert_to_numpy(scaler(x))

        @tf.function
        def traced(t):
            return scaler(t)

        traced(tf.constant(x))

        recovered = keras.ops.convert_to_numpy(scaler.inverse_transform(y))
        np.testing.assert_allclose(recovered, x, atol=1e-4, rtol=0)

    def test_inverse_transform_reconstructs_the_input_after_model_fit(self):
        """RED pre-fix: the same TypeError, reached through the real training path.

        ``y`` is captured EAGERLY BEFORE ``fit`` and never recomputed afterwards. An
        eager call after ``fit`` would overwrite ``_last_mean`` with a fresh eager
        tensor and the guard would pass on unfixed source -- measured, it did.
        ``batch_size`` covers the whole array and ``shuffle=False``, so the single
        training batch carries exactly the statistics of ``x`` and the post-fix
        reconstruction is the same one the eager arm asserts.
        """
        x = self._data()
        scaler = UnifiedScaler(axis=-1, name="scaler")

        inputs = keras.Input(shape=(5,))
        outputs = keras.layers.Dense(5)(scaler(inputs))
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="sgd", loss="mse")

        y = keras.ops.convert_to_numpy(scaler(x))

        model.fit(x, x, epochs=1, batch_size=x.shape[0], shuffle=False, verbose=0)

        recovered = keras.ops.convert_to_numpy(scaler.inverse_transform(y))
        np.testing.assert_allclose(recovered, x, atol=1e-4, rtol=0)


# =====================================================================
# V1 [HIGH] -- DeepKernelPCA.compute_output_shape requires a built layer
# =====================================================================
class TestDeepKernelPcaComputesItsOutputShapeUnbuilt:
    """§3.4 / Pitfall 14: ``compute_output_shape`` must work from stored config.

    ``deep_kernel_pca.py`` raised unless ``self.built``. 8 of the package's 9
    classes already satisfy this; this was the only failure.
    """

    INPUT_SHAPE = (None, 16)

    # [8, 4], not [3, 2]. The layer's own anchored precondition (the dynamic
    # slice projection_matrix[:batch_size, :]) requires feature_dim >= batch_size
    # at EVERY level, and [3, 2] makes level 1's feature_dim 3 while the probe
    # below runs a batch of 4. That is a property of the fixture, not of the
    # claim under test, and it raised before the layer could produce a shape.
    EXPLICIT = [8, 4]

    @pytest.mark.parametrize(
        "components_per_level", [EXPLICIT, None], ids=["explicit", "adaptive"]
    )
    def test_compute_output_shape_works_before_build(self, components_per_level):
        layer = DeepKernelPCA(num_levels=2, components_per_level=components_per_level)
        assert not layer.built

        shape = layer.compute_output_shape(self.INPUT_SHAPE)

        assert not layer.built, "compute_output_shape must not build the layer"
        assert isinstance(shape, tuple) and len(shape) == 2
        assert shape[0] is None
        assert isinstance(shape[1], int) and shape[1] > 0

    @pytest.mark.parametrize(
        "components_per_level", [EXPLICIT, None], ids=["explicit", "adaptive"]
    )
    def test_the_unbuilt_shape_equals_the_built_shape(self, components_per_level):
        """The real contract: the pre-build answer must be the post-build answer."""
        rng = np.random.default_rng(1234)
        x = rng.normal(size=(4, 16)).astype("float32")

        unbuilt = DeepKernelPCA(num_levels=2, components_per_level=components_per_level)
        predicted = unbuilt.compute_output_shape(self.INPUT_SHAPE)

        built = DeepKernelPCA(num_levels=2, components_per_level=components_per_level)
        y = built(x)
        assert built.built

        assert predicted == built.compute_output_shape(self.INPUT_SHAPE)
        assert predicted[1] == int(keras.ops.convert_to_numpy(y).shape[-1])


# =====================================================================
# V2 [MED] -- a sub-layer created in build()
# =====================================================================
class TestTheDenoiserCreatesItsChildInInit:
    """§1.2 Golden Rule / Pitfall 2: sub-layers are created in ``__init__``.

    ``InvertibleKernelPCADenoiser.__init__`` (~:956) sets ``self.ikpca = None`` and
    ``build()`` (~:982-991) constructs it. Nothing the child needs depends on
    ``input_shape``.
    """

    def test_the_child_exists_before_build(self):
        layer = InvertibleKernelPCADenoiser(n_components=8, n_random_features=32)
        assert not layer.built
        assert layer.ikpca is not None, (
            "InvertibleKernelPCADenoiser.ikpca is None before build(): the child is "
            "created in build(), violating the create-in-__init__ rule"
        )
        assert isinstance(layer.ikpca, InvertibleKernelPCA)


# =====================================================================
# V3 [MED] -- bare ValueErrors that do not name the offending value
# =====================================================================
def _bare_value_error_sites():
    """Return ``(path, lineno, kind)`` for every ``raise ValueError`` that names nothing.

    A message "names the offending value" when its first argument interpolates
    something: an f-string carrying at least one ``{...}`` field, a ``.format(...)``
    call, or a ``%`` BinOp. A plain string constant does not -- and neither does an
    f-string with NO replacement field, which is why ``ast.JoinedStr`` alone is not
    accepted here. Two such empty f-strings exist in ``deep_kernel_pca.py``.
    """
    total = 0
    offenders = []
    for path in _package_sources():
        for node in ast.walk(_parse(path)):
            if not (isinstance(node, ast.Raise) and node.exc is not None):
                continue
            exc = node.exc
            if not (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "ValueError"
            ):
                continue
            total += 1
            if not exc.args:
                offenders.append((path, node.lineno, "no-message"))
                continue
            first = exc.args[0]
            if isinstance(first, ast.JoinedStr):
                if any(isinstance(v, ast.FormattedValue) for v in first.values):
                    continue
                offenders.append((path, node.lineno, "f-string with no field"))
                continue
            if (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Attribute)
                and first.func.attr == "format"
            ):
                continue
            if isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mod):
                continue
            offenders.append((path, node.lineno, "constant string"))
    return total, offenders


class TestEveryValueErrorNamesTheOffendingValue:
    """§16.1: "Argument validation raises ``ValueError`` naming the offending value".

    Package-wide AST scan. Pre-fix MEASURED: 58 ``raise ValueError`` sites, 19 of which
    name nothing -- the 17 constant strings the audit listed, plus two f-strings with no
    replacement field (``deep_kernel_pca.py:356,416``) that the audit's own scan
    accepted because it tested for ``ast.JoinedStr`` rather than for interpolation.
    """

    def test_the_scan_sees_the_whole_package(self):
        """Anti-vacuity (§13.4.8): a failed glob would make the guard below pass."""
        sources = _package_sources()
        assert len(sources) >= MIN_SOURCE_FILES, (
            f"scan found only {len(sources)} source files under {PACKAGE_DIR}"
        )
        total, _ = _bare_value_error_sites()
        assert total >= 40, (
            f"scan found only {total} `raise ValueError` sites; expected >= 40. "
            f"The subject set collapsed -- this guard is testing nothing."
        )

    def test_no_value_error_message_is_bare(self):
        _, offenders = _bare_value_error_sites()
        rendered = "\n".join(
            f"  {p.name}:{line} ({kind})" for p, line, kind in offenders
        )
        assert not offenders, (
            f"{len(offenders)} `raise ValueError` sites do not name the offending "
            f"value:\n{rendered}"
        )


# =====================================================================
# V4 [MED] -- cross-parameter contracts deferred to build()
# =====================================================================
class TestCrossParameterContractsRaiseAtConstruction:
    """§3.5: a contract that needs no ``input_shape`` is checked in ``__init__``.

    Both contracts below are pure functions of constructor arguments, yet both are
    currently only enforced in ``build()``.
    """

    def test_deep_kernel_pca_rejects_a_mismatched_components_length(self):
        with pytest.raises(ValueError):
            DeepKernelPCA(num_levels=2, components_per_level=[1, 2, 3])

    def test_invertible_kernel_pca_rejects_too_many_components(self):
        with pytest.raises(ValueError):
            InvertibleKernelPCA(n_components=99, n_random_features=8)


# =====================================================================
# V6 [MED] -- training= not forwarded in normalizing_flow.py
# =====================================================================
class TestTheFlowForwardsTrainingToItsSubLayers:
    """§4.2 / §16.1: ``training=`` is forwarded explicitly to every sub-layer call.

    This guard is STRUCTURAL (AST) on purpose. ``transformation_net`` is a plain
    ``Dense`` stack with no training-dependent behaviour, so a behavioural probe
    (compare ``training=True`` against ``training=False`` outputs) would be VACUOUS:
    it would pass identically before and after the fix. The guide requires the keyword
    to be forwarded "even where omitting it is currently a no-op", which is a property
    of the source, so the source is what is asserted.
    """

    @staticmethod
    def _coupling_class():
        tree = _parse(PACKAGE_DIR / "normalizing_flow.py")
        node = _find_class(tree, "AffineCouplingLayer")
        assert node is not None, "AffineCouplingLayer not found -- the guard is vacuous"
        return node

    def test_the_conditioner_call_passes_training(self):
        method = _find_method(self._coupling_class(), "_compute_scale_and_shift")
        assert method is not None

        calls = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "transformation_net"
        ]
        assert calls, (
            "no call to self.transformation_net inside _compute_scale_and_shift -- "
            "the guard is vacuous"
        )
        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "training" in kwargs, (
                f"normalizing_flow.py:{call.lineno}: self.transformation_net(...) is "
                f"called without training="
            )

    @pytest.mark.parametrize("method_name", ["forward", "inverse"])
    def test_forward_and_inverse_accept_and_thread_training(self, method_name):
        method = _find_method(self._coupling_class(), method_name)
        assert method is not None

        params = {a.arg for a in method.args.args} | {
            a.arg for a in method.args.kwonlyargs
        }
        assert "training" in params, (
            f"AffineCouplingLayer.{method_name} does not accept a training parameter"
        )

        threaded = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_compute_scale_and_shift"
            and any(kw.arg == "training" for kw in n.keywords)
        ]
        assert threaded, (
            f"AffineCouplingLayer.{method_name} does not pass training= down to "
            f"_compute_scale_and_shift"
        )


# =====================================================================
# V7 [MED] -- ACFMonitorCallback has no validation at all
# =====================================================================
class TestTheAcfCallbackValidatesItsArguments:
    """§3.5: ``log_frequency=0`` currently constructs fine and dies at the first batch
    with ``ZeroDivisionError`` at ``residual_acf.py:612``.
    """

    def test_a_zero_log_frequency_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="log_frequency"):
            ACFMonitorCallback(layer_name="x", log_frequency=0)


# =====================================================================
# V8 [LOW] -- a declared-but-unused field
# =====================================================================
class TestTheKernelPcaHasNoUnusedEigenvectorsField:
    """§5.5: ``InvertibleKernelPCA.eigenvectors`` (~:334) is declared, never written
    and never read. Checked both unbuilt and built, because a field could in principle
    be created later in ``build()`` -- it is not.
    """

    def test_no_eigenvectors_attribute_before_build(self):
        layer = InvertibleKernelPCA(n_components=4, n_random_features=16)
        assert not hasattr(layer, "eigenvectors"), (
            "InvertibleKernelPCA declares self.eigenvectors, which is never written "
            "and never read"
        )

    def test_no_eigenvectors_attribute_after_build(self):
        rng = np.random.default_rng(1234)
        layer = InvertibleKernelPCA(n_components=4, n_random_features=16)
        layer(rng.normal(size=(4, 8)).astype("float32"))
        assert layer.built
        assert not hasattr(layer, "eigenvectors")


# =====================================================================
# V11 [MED] -- EXPECTED GREEN BEFORE THE FIX. Not a RED proof.
# =====================================================================
class TestTheStoredStatsAssignWorksUnderXla:
    """PINS A FACT. This guard is GREEN on unfixed source, by design.

    ``scaler.py:417-421`` claims the in-``call()`` ``.assign`` is "NOT supported under
    TF ``jit_compile=True`` (XLA)". MEASURED FALSE -- it works. The fix (step 3) deletes
    that false sentence from the comment; this guard exists so the replacement text
    stays true. Do NOT count it among the RED proofs for step 1.
    """

    def test_a_jit_compiled_call_updates_the_stored_statistics(self):
        rng = np.random.default_rng(1234)
        x = rng.normal(loc=5.0, scale=1.0, size=(8, 4)).astype("float32")

        scaler = UnifiedScaler(axis=-1, store_stats=True)
        scaler(x)  # eager call: builds the layer and creates the stat weights
        scaler.reset_stats()

        before_mean, before_std = scaler.get_stats()
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(before_mean),
            np.zeros_like(keras.ops.convert_to_numpy(before_mean)),
            atol=0.0,
            rtol=0,
        )

        @tf.function(jit_compile=True)
        def compiled(t):
            return scaler(t)

        compiled(tf.constant(x))

        after_mean = keras.ops.convert_to_numpy(scaler.get_stats()[0])
        assert np.any(np.abs(after_mean) > 1e-6), (
            "the stored mean did not move under jit_compile=True; the source comment's "
            "claim would then be true and this guard's premise wrong"
        )
        expected = np.mean(np.mean(x, axis=-1, keepdims=True), axis=0)
        np.testing.assert_allclose(after_mean, expected, atol=1e-5, rtol=0)
