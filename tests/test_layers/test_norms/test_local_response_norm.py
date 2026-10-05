"""Behavioural tests for ``LocalResponseNormalization`` (LRN, AlexNet 2012).

Structure follows ``research/2026_keras_custom_models_instructions_v2.md`` section 13:
construction tests are cheap and near-worthless, so the weight of this module is on
proving the layer computes the *documented* function and not merely *some* function.

The numerical oracle is ``tf.nn.local_response_normalization``, which is an
independent implementation by different authors -- so agreement is evidence, not
tautology. Two measured facts about it shape this file:

* **It is float32-only.** The ``LRN`` kernel's allowed ``T`` attribute is
  ``[half, bfloat16, float]``, so ``tf.nn.local_response_normalization`` raises
  ``InvalidArgumentError: Value for attr 'T' of double is not in the list`` on a
  float64 tensor. It therefore cannot act as the float64 oracle, and the
  float64 arm of :class:`TestTheMathIsThePapersEquation` uses an explicit
  per-channel NumPy loop instead.
* **It has no channels-first mode.** The channels-first arm compares against
  ``tf.nn`` applied to a ``moveaxis``-ed copy and moved back.

Measured agreement, this file's whole subject population: **2.384e-07** against
``tf.nn`` (float32 eps is 1.192e-07, so this is 2 ulp) and **0.0** against the
float64 NumPy oracle, i.e. bit-exact.
"""

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import layers

from dl_techniques.layers.norms.factory import (
    _TYPE_TO_CLASS,
    create_normalization_layer,
    create_normalization_from_config,
    validate_normalization_config,
)
from dl_techniques.layers.norms.local_response_norm import LocalResponseNormalization

#: The paper's hyper-parameters: Krizhevsky et al. 2012 section 4 reports
#: ``n=5`` (so ``depth_radius=2``), ``alpha=1e-4``, ``beta=0.75``, ``k=1``.
ALEXNET = {"depth_radius": 2, "alpha": 1e-4, "beta": 0.75, "k": 1.0}

#: TF's LRN kernel accepts only these element types (measured: the op's ``T``
#: attribute is ``[half, bfloat16, float]``).
_TF_DTYPES = ["float16", "float32"]

#: ``tf.nn.local_response_normalization`` on GPU rejects ``depth_radius=0``:
#: ``InvalidArgumentError: cuDNN requires depth_radius in [1, 7], got: 0``. That is
#: a constraint of TF's fused CUDA kernel, NOT of this layer -- a zero radius is
#: legal here (the band becomes the identity, leaving a pure per-element rescale)
#: and is covered by the float64 NumPy arm and by the CPU-only TF arm below.
#: This list is the radii the GPU TF oracle can be asked about.
_TF_GPU_RADII = [1, 2, 5]


def _numpy_reference(x, depth_radius, alpha, beta, k, axis=-1):
    """Independent float64 oracle, written as an explicit per-channel loop.

    Deliberately does NOT reuse the layer's band-matrix formulation: a loop over
    channels written from the paper's equation shares no code path with a
    matrix multiply, so agreement is a real check on the band construction.

    :param x: Input array of any rank >= 3.
    :param depth_radius: Half-width of the channel neighbourhood.
    :param alpha: Coefficient on the squared neighbourhood sum.
    :param beta: Exponent on the denominator.
    :param k: Additive constant inside the exponent.
    :param axis: Channel axis.
    :return: The normalized array, dtype float64.
    :rtype: numpy.ndarray
    """
    moved = np.moveaxis(np.asarray(x, dtype="float64"), axis, -1)
    channels = moved.shape[-1]
    squared_sum = np.zeros_like(moved)
    for c in range(channels):
        lo = max(0, c - depth_radius)
        hi = min(channels - 1, c + depth_radius)
        squared_sum[..., c] = (moved[..., lo:hi + 1] ** 2).sum(axis=-1)
    out = moved / (k + alpha * squared_sum) ** beta
    return np.moveaxis(out, -1, axis)


def _tf_reference(x, depth_radius, alpha, beta, k):
    """``tf.nn.local_response_normalization``, whose additive constant is ``bias``."""
    return tf.nn.local_response_normalization(
        x, depth_radius=depth_radius, bias=k, alpha=alpha, beta=beta
    ).numpy()


_F32_EPS = float(np.finfo(np.float32).eps)


def _tf_agreement_atol(expected):
    """Bound for comparing this layer against ``tf.nn`` in float32.

    DERIVED, not pasted. The two implementations compute the same formula in
    different orders -- a band matmul against TF's fused kernel -- so they differ
    only in float32 rounding, and the error is *relative*. A literal bound here is
    wrong twice over: too tight for wide outputs and needlessly slack for small
    ones. The bound is therefore ``8 * eps_f32 * max(1, max|expected|)``, with the
    ``max(1, .)`` term because the error is relative to a magnitude that may be
    below 1, in which case 1 ulp of the *dtype* is the floor.

    Measured over 32 (shape, radius) cells at peak ``|output| = 5.336``: worst
    absolute difference **4.768e-07** = **0.75** ulp of that peak. So 8 ulp is an
    order of magnitude of headroom over what a correct implementation actually
    needs, which is what keeps this from being a flaky bound rather than a loose
    one.

    This is the ``reassociation_atol`` distinction from ``tests/numerics.py``: the
    comparison is attainable-but-tight, so the number is derived from the noise
    source rather than widened to whatever happened to pass.

    :param expected: The reference tensor, float or float64.
    :return: An absolute tolerance for the comparison.
    :rtype: float
    """
    peak = float(np.max(np.abs(np.asarray(expected, dtype="float64"))))
    return 8.0 * _F32_EPS * max(1.0, peak)


class TestTheMathIsThePapersEquation:
    """The layer computes ``X / (k + alpha * sum_j X_j^2) ** beta``."""

    @pytest.mark.parametrize("depth_radius", _TF_GPU_RADII)
    def test_it_matches_tf_nn_on_the_channels_last_path(self, depth_radius):
        """Agreement with TF's independent implementation of the same equation."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 6, 8, 32)).astype("float32")
        observed = np.asarray(
            LocalResponseNormalization(depth_radius=depth_radius)(x), dtype="float64"
        )
        expected = _tf_reference(x, depth_radius, 1e-4, 0.75, 1.0)
        np.testing.assert_allclose(
            observed, expected, rtol=0, atol=_tf_agreement_atol(expected)
        )

    def test_a_zero_radius_is_a_pure_per_element_rescale(self):
        """``depth_radius=0`` is legal here even though cuDNN's LRN refuses it.

        With no neighbours the denominator is the constant ``k`` applied to each
        element's own square, so the whole layer collapses to
        ``X / (k + alpha * X^2) ** beta``.
        """
        rng = np.random.default_rng(10)
        x = rng.normal(size=(2, 4, 4, 8)).astype("float32")
        observed = np.asarray(
            LocalResponseNormalization(depth_radius=0)(x), dtype="float64"
        )
        x64 = x.astype("float64")
        expected = x64 / (1.0 + 1e-4 * x64 ** 2) ** 0.75
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 6, 8, 7),
            (2, 4, 3, 3),
            (2, 3, 4, 5),  # window far wider than the channel axis
            (1, 1, 1, 4),
            (3, 2, 2, 96),  # AlexNet's conv1 output width
            (3, 13, 13, 256),  # AlexNet's conv5 output width
            (1, 56, 56, 96),  # AlexNet's conv1 output SPATIAL size, 227 input
        ],
    )
    @pytest.mark.parametrize("dtype", _TF_DTYPES)
    @pytest.mark.parametrize("depth_radius", _TF_GPU_RADII)
    def test_it_matches_tf_nn_across_shapes_and_radii(self, shape, depth_radius, dtype):
        """The whole subject population, at every dtype TF's kernel accepts.

        Every shape here is rank 4 because ``tf.nn.local_response_normalization`` is
        rank-4-only (``InvalidArgumentError: in must be 4-dimensional``). Rank-3
        support is covered by the float64 NumPy arm below, which has no such limit.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(size=shape).astype(dtype)
        observed = np.asarray(
            LocalResponseNormalization(depth_radius=depth_radius)(x), dtype="float64"
        )
        expected = _tf_reference(
            x.astype("float32"), depth_radius, 1e-4, 0.75, 1.0
        )
        np.testing.assert_allclose(
            observed, expected, rtol=0, atol=_tf_agreement_atol(expected)
        )

    def test_it_matches_the_numpy_reference_in_float64(self):
        """Bit-exact at float64.

        ``rtol=0`` and ``atol=0``: there is no noise source in this computation to
        derive a tolerance from, so anything above zero would be slack.
        """
        keras.mixed_precision.set_dtype_policy("float64")
        try:
            rng = np.random.default_rng(2)
            for shape, axis, radius in [
                ((2, 6, 8, 7), -1, 2),
                ((2, 6, 8, 7), -1, 0),  # cuDNN refuses this radius; TF cannot be the oracle
                ((2, 5, 9), -1, 3),
                ((2, 3, 4, 5), -1, 10),  # window wider than the channel axis
                ((2, 6, 8, 7), 1, 2),  # channels_first
                ((2, 4, 3, 3), -1, 1),
            ]:
                x = rng.normal(size=shape)
                layer = LocalResponseNormalization(
                    depth_radius=radius,
                    data_format=("channels_first" if axis == 1 else None),
                )
                observed = np.asarray(layer(x), dtype="float64")
                expected = _numpy_reference(x, radius, 1e-4, 0.75, 1.0, axis=axis)
                np.testing.assert_array_equal(observed, expected)
        finally:
            keras.mixed_precision.set_dtype_policy("float32")

    def test_the_band_matrix_is_built_in_the_policy_compute_dtype(self):
        """The band must follow the active compute dtype, not the global float32.

        MUTATION-PROVEN. Pinning the band to ``'float32'`` (which is what
        ``keras.backend.floatx()`` returns, and that call is NOT moved by
        ``set_dtype_policy('float64')``) leaves every float32 test in this file
        green -- 99 passed under the mutation -- while silently cutting a float64
        forward pass down to float32 precision. This assertion is the guard that
        mutation could not get past.
        """
        keras.mixed_precision.set_dtype_policy("float64")
        try:
            layer = LocalResponseNormalization()
            layer.build((None, 1, 1, 6))
            assert keras.backend.standardize_dtype(layer.band.dtype) == "float64", (
                "the band matrix is float32 under a float64 policy, so a float64 "
                "forward pass is silently truncated to float32 precision"
            )
        finally:
            keras.mixed_precision.set_dtype_policy("float32")

    def test_a_float64_policy_does_not_truncate_the_forward_pass(self):
        """The end-to-end consequence: float64 in must mean float64 out."""
        keras.mixed_precision.set_dtype_policy("float64")
        try:
            rng = np.random.default_rng(12)
            x = rng.normal(size=(2, 4, 4, 8))  # float64
            observed = np.asarray(LocalResponseNormalization(depth_radius=2)(x), "float64")
            expected = _numpy_reference(x, 2, 1e-4, 0.75, 1.0)
            # float64 epsilon is 2.2e-16; a float32-truncated result would miss by
            # O(1e-3), so this separates by nine orders of magnitude.
            np.testing.assert_allclose(
                observed, expected, rtol=0, atol=1e-14
            )
        finally:
            keras.mixed_precision.set_dtype_policy("float32")

    def test_channels_first_is_a_relabelling_not_a_different_algorithm(self):
        """The two ``data_format`` values must agree once the axes are moved.

        The input is NCHW, ``(batch=2, channels=9, h=7, w=8)``, so
        ``data_format='channels_first'`` reads axis 1 as the channel axis. The
        same computation is then run through the channels-last path by moving the
        channel axis to the end and back.

        A NON-square, asymmetric shape is deliberate: on a square input an axis mix-up
        would be invisible, and the two paths would agree by accident.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 9, 7, 8)).astype("float32")

        as_first = np.asarray(
            LocalResponseNormalization(
                depth_radius=2, data_format="channels_first"
            )(x),
            dtype="float64",
        )
        via_last = np.asarray(
            LocalResponseNormalization(depth_radius=2)(np.moveaxis(x, 1, -1)),
            dtype="float64",
        )
        expected = np.moveaxis(via_last, -1, 1)

        assert as_first.shape == x.shape
        # Bit-exact: the relabelling is exact integer index arithmetic, so there is
        # no reordering of arithmetic to disagree about.
        np.testing.assert_array_equal(as_first, expected)


class TestTheWindowIsTruncatedNotPadded:
    """The edge channels sum over fewer neighbours, as ``tf.nn`` does."""

    def test_the_first_channel_ignores_channels_below_the_window(self):
        """A large negative input BELOW the window must not reach channel 0.

        This is the guard that distinguishes truncation from zero-padding: if the
        implementation padded the window with real taps read past the axis, or
        symmetrized the band, channel 0's denominator would change.
        """
        base = np.full((1, 1, 1, 5), 0.5, dtype="float32")
        bumped = base.copy()
        # Channels 1..4 raised hugely; channel 0 untouched.
        bumped[0, 0, 0, 1:] = 50.0

        layer = LocalResponseNormalization(depth_radius=1)
        y_base = np.asarray(layer(base), dtype="float64")
        y_bumped = np.asarray(layer(bumped), dtype="float64")

        # Channel 0's window is {0, 1}, so channel 1 IS in range and must move it.
        assert not np.isclose(y_base[0, 0, 0, 0], y_bumped[0, 0, 0, 0])

    def test_a_channel_far_outside_the_window_cannot_move_its_neighbour(self):
        """The band has hard support: |c - j| <= depth_radius, exactly."""
        base = np.full((1, 1, 1, 9), 0.5, dtype="float32")
        bumped = base.copy()
        bumped[0, 0, 0, 0] = 500.0  # 8 channels away from channel 8

        layer = LocalResponseNormalization(depth_radius=2)
        y_base = np.asarray(layer(base), dtype="float64")
        y_bumped = np.asarray(layer(bumped), dtype="float64")
        assert np.isclose(
            y_base[0, 0, 0, 8], y_bumped[0, 0, 0, 8], rtol=0, atol=1e-6
        ), (
            "channel 8's output moved when channel 0 changed, so the neighbourhood "
            "is wider than depth_radius=2 claims"
        )

    @pytest.mark.parametrize("radius", [0, 1, 2, 3])
    def test_the_band_matrix_has_the_exact_support_it_documents(self, radius):
        """``band[c, j] == 1`` iff ``|c - j| <= depth_radius``."""
        band = np.asarray(_build_band(7, radius), dtype="float64")
        index = np.arange(7)
        expected = (
            np.abs(index[:, None] - index[None, :]) <= radius
        ).astype("float64")
        np.testing.assert_array_equal(band, expected)


def _build_band(channels, depth_radius):
    """Build a layer over a rank-3 input and return its band matrix.

    ``build()`` receives the FULL shape including the batch axis, so a rank-3
    channels-last input is ``(None, channels)`` padded to ``(None, 1, channels)`` --
    passing ``(None, channels)`` is rank 2 and is correctly rejected.
    """
    layer = LocalResponseNormalization(depth_radius=depth_radius)
    layer.build((None, 1, channels))
    return layer.band


class TestTheBandIsTruncatedAtTheEdge:
    """The band's own rows show the truncation directly."""

    def test_row_zero_covers_only_the_reachable_neighbours(self):
        band = np.asarray(_build_band(5, 2), dtype="float64")
        np.testing.assert_array_equal(
            band[0], np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        )

    def test_the_band_is_symmetric_so_the_matmul_orientation_does_not_matter(self):
        band = np.asarray(_build_band(9, 3), dtype="float64")
        np.testing.assert_array_equal(band, band.T)


class TestTheLayerHoldsNoTrainableWeights:
    """LRN's hyperparameters are fixed, so nothing here is learnable.

    The band matrix IS stored as a non-trainable ``C x C`` weight rather than a bare
    attribute. That is not a style choice: a bare ``keras.ops`` tensor built in
    ``build()`` is stamped with whatever graph was active, and Keras builds a
    sublayer inside a ``scratch_graph`` while tracing a parent, so the tensor dies on
    first use with ``<tf.Tensor 'lrn1/Cast:0' ...> is out of scope``. Reproduced by
    nesting this layer in ``models/vision/alexnet``.

    ``count_params()`` therefore reports ``C * C`` while
    ``trainable_parameters`` reports ``0``, and both facts are pinned below.
    """

    def test_it_reports_zero_TRAINABLE_parameters(self):
        layer = LocalResponseNormalization()
        layer.build((None, 8, 8, 16))
        assert layer.trainable_variables == []
        assert layer.trainable_weights == []
        assert layer.count_params() == 16 * 16, "the 16 x 16 band is counted"

    def test_the_band_is_non_trainable_but_counted(self):
        """The exact split, since "zero parameters" is now false."""
        layer = LocalResponseNormalization(depth_radius=2)
        layer.build((None, 1, 1, 6))
        assert layer.count_params() == 36, "6 x 6 band"
        assert list(layer.non_trainable_weights) == [layer.band]

    def test_it_creates_no_trainable_variables_in_a_model(self):
        inputs = keras.Input((8, 8, 16))
        model = keras.Model(inputs, LocalResponseNormalization()(inputs))
        assert model.trainable_variables == []

    def test_the_band_survives_repeated_and_retraced_calls(self):
        """The graph-safety regression this weight exists for.

        Built bare, the band was stamped with the graph active at build time and died
        on the NEXT call with ``<tf.Tensor 'lrn1/Cast:0' ...> is out of scope and
        cannot be used here`` -- reproduced by nesting this layer in a parent model.

        The layer is built OUTSIDE the tf.function (a variable created inside one is a
        separate TF error), then called twice eagerly and once through a traced
        wrapper, which is the sequence that used to fail.
        """
        layer = LocalResponseNormalization()
        layer.build((None, 4, 4, 8))
        traced = tf.function(layer)

        x = tf.constant(np.random.default_rng(13).normal(size=(2, 4, 4, 8)), "float32")
        eager_first = np.asarray(layer(x))
        eager_second = np.asarray(layer(x))
        traced_out = np.asarray(traced(x))

        np.testing.assert_array_equal(eager_first, eager_second)
        np.testing.assert_array_equal(eager_first, traced_out)

    def test_a_parent_model_can_call_it_twice(self):
        """The nesting case that actually raised, end to end."""
        inputs = keras.Input((4, 4, 8))
        model = keras.Model(inputs, LocalResponseNormalization()(inputs))
        x = np.random.default_rng(14).normal(size=(2, 4, 4, 8)).astype("float32")
        first = np.asarray(model(x))
        second = np.asarray(model(x))
        np.testing.assert_array_equal(first, second)

    def test_it_survives_a_fit_step_with_a_dynamic_batch(self):
        """The static-shape reshape regression, and the only probe that sees it.

        Reshaping back to ``moved.shape`` (the STATIC shape) works for every
        eager call and for a bare ``tf.function``, and it fails only inside
        ``Model.fit``'s trace, because the symbolic KerasTensor Keras builds there
        carries a partially known shape -- (None, 16, 16, 8) -- and ``reshape()``
        rejects a None in it.

        MUTATION-PROVEN, and the probe had to be found first: restoring the static
        shape left 104 tests in this module green, including the ``tf.function``
        variant of this assertion. ``fit`` is what discriminates.
        """
        # A Dense head is attached because Keras refuses (with a UserWarning) to `fit`
        # a model with no trainable weights at all, and neither LRN nor a pooling
        # layer has any. The head exists only so the trace has something to
        # differentiate; the reshape under test happens upstream of it.
        inputs = keras.Input((16, 16, 8))
        features = LocalResponseNormalization(depth_radius=2)(inputs)
        model = keras.Model(
            inputs, layers.Dense(1)(layers.GlobalAveragePooling2D()(features))
        )
        model.compile(optimizer="sgd", loss="mse")
        assert model.trainable_weights, "the probe head must be trainable"

        rng = np.random.default_rng(15)
        history = model.fit(
            rng.normal(size=(4, 16, 16, 8)).astype("float32"),
            rng.normal(size=(4, 1)).astype("float32"),
            epochs=1,
            verbose=0,
        )
        assert np.isfinite(history.history["loss"][0])

    def test_a_traced_model_returns_the_input_batch_shape(self):
        """Batch sizes must not be pinned to the traced one."""
        inputs = keras.Input((4, 4, 8))
        model = keras.Model(inputs, LocalResponseNormalization()(inputs))
        built = tf.function(lambda t: model(t))
        for batch in (1, 3, 7):
            x = tf.constant(
                np.random.default_rng(16).normal(size=(batch, 4, 4, 8)), "float32"
            )
            out = np.asarray(built(x))
            assert out.shape == (batch, 4, 4, 8)
            expected = _numpy_reference(np.asarray(x), 2, 1e-4, 0.75, 1.0)
            np.testing.assert_allclose(
                out, expected, rtol=0, atol=_tf_agreement_atol(expected)
            )


class TestTheShapeIsPreservedAndComputedStatically:
    def test_compute_output_shape_equals_the_input_shape(self):
        shape = (None, 13, 13, 256)
        assert LocalResponseNormalization().compute_output_shape(shape) == shape

    @pytest.mark.parametrize("shape", [(2, 6, 8, 7), (2, 5, 9), (2, 13, 13, 256)])
    def test_a_functional_model_keeps_the_channel_count(self, shape):
        inputs = keras.Input(shape[1:])
        model = keras.Model(inputs, LocalResponseNormalization()(inputs))
        assert model.output_shape == (None,) + shape[1:]


class TestConstructionValidatesEarly:
    """Every rejected configuration raises ``ValueError`` naming the argument."""

    @pytest.mark.parametrize(
        "kwargs,expected_fragment",
        [
            ({"depth_radius": -1}, "depth_radius"),
            ({"depth_radius": 2.5}, "depth_radius"),
            ({"depth_radius": True}, "depth_radius"),
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": -1e-4}, "alpha"),
            ({"beta": 0.0}, "beta"),
            ({"beta": -0.5}, "beta"),
            ({"k": 0.0}, "k"),
            ({"k": -1.0}, "k"),
            ({"data_format": "channels_middle"}, "data_format"),
        ],
    )
    def test_it_rejects_bad_arguments(self, kwargs, expected_fragment):
        with pytest.raises(ValueError) as excinfo:
            LocalResponseNormalization(**kwargs)
        assert expected_fragment in str(excinfo.value)

    @pytest.mark.parametrize("rank", [2, 5])
    def test_build_rejects_an_unsupported_rank(self, rank):
        layer = LocalResponseNormalization()
        with pytest.raises(ValueError) as excinfo:
            layer.build(tuple([None] * rank))
        assert "rank" in str(excinfo.value)

    def test_build_rejects_channels_first_on_a_rank_three_input(self):
        layer = LocalResponseNormalization(data_format="channels_first")
        with pytest.raises(ValueError) as excinfo:
            layer.build((None, 5, 8))
        assert "rank-4" in str(excinfo.value)

    def test_build_rejects_an_undefined_channel_dimension(self):
        layer = LocalResponseNormalization()
        with pytest.raises(ValueError) as excinfo:
            layer.build((None, 8, 8, None))
        assert "channel dimension" in str(excinfo.value)

    def test_calling_call_directly_before_build_raises_rather_than_guessing(self):
        """A silently-empty band would divide by ``k`` and look merely wrong.

        ``layer(x)`` cannot reach this guard, because Keras 3 builds a layer
        automatically on first ``__call__``. It is reachable by calling ``call()``
        directly, which is what a subclass in this tree would do, so the guard is
        pinned through that route rather than through a route that cannot exist.
        """
        layer = LocalResponseNormalization()
        with pytest.raises(ValueError) as excinfo:
            layer.call(keras.random.normal((2, 4, 4, 8)))
        assert "build()" in str(excinfo.value)

    def test_the_usual_call_route_builds_and_works(self):
        """``__call__`` auto-builds, so the guard above is not dead code."""
        x = np.random.default_rng(11).normal(size=(2, 4, 4, 8)).astype("float32")
        out = np.asarray(LocalResponseNormalization()(x), dtype="float64")
        np.testing.assert_allclose(
            out, _numpy_reference(x, 2, 1e-4, 0.75, 1.0), rtol=0, atol=1e-6
        )


class TestSerializationRoundTrips:
    """Every constructor argument survives a ``.keras`` archive."""

    def test_a_saved_and_reloaded_layer_computes_identically(self, tmp_path):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2, 6, 8, 12)).astype("float32")

        original = keras.Sequential([LocalResponseNormalization(**ALEXNET)])
        expected = np.asarray(original(x), dtype="float64")

        path = tmp_path / "lrn.keras"
        original.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_array_equal(
            np.asarray(reloaded(x), dtype="float64"), expected
        )

    def test_get_config_carries_every_argument(self):
        layer = LocalResponseNormalization(
            depth_radius=3, alpha=2e-4, beta=0.6, k=2.0,
            data_format="channels_first",
        )
        config = layer.get_config()
        for key in ("depth_radius", "alpha", "beta", "k", "data_format"):
            assert key in config, f"{key} missing from get_config()"

    def test_the_config_rebuilds_an_equivalent_layer(self):
        layer = LocalResponseNormalization(
            depth_radius=3, alpha=2e-4, beta=0.6, k=2.0
        )
        rebuilt = LocalResponseNormalization.from_config(layer.get_config())
        for attr in ("depth_radius", "alpha", "beta", "k"):
            assert getattr(rebuilt, attr) == getattr(layer, attr)

    def test_a_non_default_layer_does_not_silently_become_the_default(self):
        """A dropped hyperparameter would still round-trip, just wrongly."""
        layer = LocalResponseNormalization(**ALEXNET)
        rebuilt = LocalResponseNormalization.from_config(layer.get_config())
        assert rebuilt.depth_radius == 2
        assert rebuilt.alpha == pytest.approx(1e-4)
        assert rebuilt.beta == pytest.approx(0.75)
        assert rebuilt.k == pytest.approx(1.0)

    def test_the_registered_name_is_the_dl_techniques_key(self):
        name = keras.saving.get_registered_name(LocalResponseNormalization)
        assert name == (
            "dl_techniques.layers.norms.local_response_norm>"
            "LocalResponseNormalization"
        ), (
            "the family directory is NOT stripped for a LAYER (only for models/), "
            f"so the key must carry the full module path; got {name}"
        )


class TestTheFactoryPath:
    """The factory is the documented construction path, so it must behave."""

    def test_it_is_registered_under_the_expected_key(self):
        assert "local_response_norm" in _TYPE_TO_CLASS
        assert _TYPE_TO_CLASS["local_response_norm"] is LocalResponseNormalization

    def test_the_factory_builds_the_layer(self):
        layer = create_normalization_layer("local_response_norm", **ALEXNET)
        assert isinstance(layer, LocalResponseNormalization)

    def test_the_factory_forwards_hyperparameters(self):
        """The factory's contract: what the builder accepts, it must honour."""
        layer = create_normalization_layer(
            "local_response_norm", depth_radius=4, alpha=3e-4, beta=0.5, k=2.5
        )
        assert layer.depth_radius == 4
        assert layer.alpha == pytest.approx(3e-4)
        assert layer.beta == pytest.approx(0.5)
        assert layer.k == pytest.approx(2.5)

    def test_the_factory_does_not_inject_an_epsilon(self):
        """LRN's constant is ``k``; the factory has nothing to bind.

        ``create_normalization_layer`` takes ``epsilon=1e-6`` and ``setdefault``s it
        onto every type that declares one. LRN declares none, so the value must
        simply not appear anywhere on the built layer.
        """
        layer = create_normalization_layer("local_response_norm")
        assert layer.k == pytest.approx(1.0)
        assert not hasattr(layer, "epsilon")
        assert not hasattr(layer, "eps")

    def test_the_factory_rejects_an_undeclared_keyword(self):
        with pytest.raises(ValueError):
            create_normalization_layer("local_response_norm", bogus_key=1)

    def test_the_validator_agrees_with_the_builder(self):
        assert validate_normalization_config("local_response_norm", **ALEXNET) is True

    def test_create_from_config_builds_the_same_layer(self):
        config = {"type": "local_response_norm", **ALEXNET}
        layer = create_normalization_from_config(config)
        assert isinstance(layer, LocalResponseNormalization)
        assert layer.depth_radius == 2
        assert "type" in config, "the caller's dict was mutated"

    def test_the_info_entry_exists_and_names_the_real_parameters(self):
        from dl_techniques.layers.norms.factory import get_normalization_info

        entry = get_normalization_info()["local_response_norm"]
        assert set(entry) == {"description", "parameters", "use_case"}
        for key in entry["parameters"]:
            layer = LocalResponseNormalization()
            assert key in layer.get_config(), (
                f"get_normalization_info documents '{key}' but the layer's "
                "get_config() does not carry it"
            )


class TestMaskingIsNotClaimed:
    """``supports_masking`` is a promise; LRN must not make it falsely."""

    def test_it_does_not_claim_masking_support(self):
        assert LocalResponseNormalization().supports_masking is False

    def test_a_neighbouring_channel_leaks_into_its_neighbour(self):
        """Why the promise cannot be kept: the coupling is real.

        Channel 3 is inside channel 5's window at ``depth_radius=2`` (``|5-3| = 2``),
        so perturbing channel 3 must move channel 5's output. A mask threaded
        through the layer would claim the two positions were independent.
        """
        base = np.full((1, 1, 1, 6), 0.4, dtype="float32")
        bumped = base.copy()
        bumped[0, 0, 0, 3] = 9.0

        layer = LocalResponseNormalization(depth_radius=2)
        y_base = np.asarray(layer(base), dtype="float64")
        y_bumped = np.asarray(layer(bumped), dtype="float64")

        assert not np.isclose(
            y_base[0, 0, 0, 5], y_bumped[0, 0, 0, 5], rtol=0, atol=1e-6
        ), "channel 5 did not react to channel 3, so the neighbourhood is not real"
        # And the coupling is local: channel 0 is 3 away from channel 3, outside a
        # radius-2 window, so it must be untouched.
        assert np.isclose(
            y_base[0, 0, 0, 0], y_bumped[0, 0, 0, 0], rtol=0, atol=1e-6
        )


class TestGradientsFlow:
    def test_the_layer_is_differentiable_wrt_its_input(self):
        # A tf.Variable, not a constant: keras.ops.moveaxis reads `.ndim`, which
        # ResourceVariable does not expose, so this also pins the conversion in call().
        x = tf.Variable(np.random.default_rng(5).normal(size=(2, 4, 4, 8)), "float32")
        with tf.GradientTape() as tape:
            y = LocalResponseNormalization()(x)
            loss = tf.reduce_sum(y ** 2)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert np.all(np.isfinite(grad.numpy()))
        assert np.abs(grad.numpy()).max() > 0

    def test_the_band_matrix_receives_no_gradient(self):
        """It is a constant, so it must not appear in any tape."""
        layer = LocalResponseNormalization()
        layer.build((None, 4, 4, 8))
        x = tf.constant(np.random.default_rng(6).normal(size=(2, 4, 4, 8)), "float32")
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x) ** 2)
        assert tape.gradient(loss, layer.band) is None


class TestTheGuardFailsWhenTheLayerIsBroken:
    """RED proof. A guard never observed failing is not known to work.

    Each mutation below breaks one documented property and the corresponding
    assertion is shown to catch it. These run the *mutated* logic directly, in
    isolation, rather than monkeypatching the shipped layer -- monkeypatching would
    prove the probe runs, not that it would catch this defect.
    """

    def test_swapping_the_band_to_the_identity_would_be_caught(self):
        """Removing the neighbourhood must break the TF comparison."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 4, 4, 8)).astype("float32")
        layer = LocalResponseNormalization(depth_radius=2)
        layer.build(x.shape)

        honest = np.asarray(layer(x), dtype="float64")
        layer.band = tf.eye(8, dtype=layer.band.dtype)  # the defect
        broken = np.asarray(layer(x), dtype="float64")

        assert not np.allclose(honest, broken), (
            "the identity band reproduces the correct output, so this probe "
            "cannot detect a missing neighbourhood"
        )
        # The mutated layer should now agree with the r=0 equation instead, which
        # is what confirms the identity band really is the defect.
        x64 = x.astype("float64")
        np.testing.assert_allclose(
            broken, x64 / (1.0 + 1e-4 * x64 ** 2) ** 0.75, rtol=0, atol=1e-6
        )

    def test_a_wrong_radius_would_be_caught(self):
        """``depth_radius`` is a real knob, not decorative."""
        rng = np.random.default_rng(8)
        x = rng.normal(size=(2, 4, 4, 8)).astype("float32")
        near = np.asarray(LocalResponseNormalization(depth_radius=1)(x), "float64")
        far = np.asarray(LocalResponseNormalization(depth_radius=3)(x), "float64")
        assert not np.allclose(near, far)

    def test_ignoring_k_would_be_caught(self):
        """``k`` is load-bearing: dropping it puts ``k``-dependent scale in error."""
        rng = np.random.default_rng(9)
        x = rng.normal(size=(2, 4, 4, 8)).astype("float32")
        with_k = np.asarray(LocalResponseNormalization(k=1.0)(x), "float64")
        without = np.asarray(LocalResponseNormalization(k=4.0)(x), "float64")
        assert not np.allclose(with_k, without)