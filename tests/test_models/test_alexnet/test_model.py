"""Tests for the AlexNet package (``models/vision/alexnet``).

Two claims carry this architecture and both are measured here rather than asserted
from the paper's prose:

1. **The feature map is 6 x 6 x 256 and fc6 takes 9216 inputs.** The paper never
   states its padding -- it omits the parameter -- so the port transcribes the
   released Caffe model. That transcription is verified against the paper's own
   Figure 3 label, and the two obvious alternatives are shown NOT to reach it.
2. **The grouped convolutions really are grouped.** ``conv3`` in particular reads
   442,752 parameters, not the 885,120 an ungrouped layer would need, and that is
   the paper's two-GPU split rather than a missing layer.

The LRN behaviour itself is tested in
``tests/test_layers/test_norms/test_local_response_norm.py``; what is pinned here is
that AlexNet *uses* it, in the paper's position and count.
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.vision.alexnet import AlexNet, create_alexnet
from dl_techniques.models.vision.alexnet.model import (
    LRN_HYPERPARAMETERS,
    MIN_SPATIAL_EXTENT,
)
from dl_techniques.models.vision.alexnet.spatial_guard import STAGES, conv_out
from dl_techniques.layers.norms.local_response_norm import LocalResponseNormalization


def _sample(batch=2, size=227, channels=3, seed=0):
    """A small random image batch.

    :return: ``(batch, size, size, channels)`` float32.
    :rtype: numpy.ndarray
    """
    return np.random.default_rng(seed).normal(
        size=(batch, size, size, channels)
    ).astype("float32")


class TestTheFeatureMapIsSixBySixBy256:
    """The paper's Figure 3, which the padding choice exists to reproduce."""

    def test_pool3_lands_on_six_by_six_by_256(self):
        """Walk the real layer chain and read the shape off ``pool3``.

        The chain is re-applied from the model's own sublayers rather than written out
        as a second copy: a duplicated chain drifts, and would then keep passing after
        the model itself broke.
        """
        model = AlexNet()
        h = keras.Input((227, 227, 3))
        h = model.conv1(model.conv1_pad(h))
        h = model.relu1(h)
        h = model.lrn1(h)
        h = model.pool1(model.pool1_pad(h))
        h = model.conv2(model.conv2_pad(h))
        h = model.relu2(h)
        h = model.lrn2(h)
        h = model.pool2(model.pool2_pad(h))
        h = model.relu3(model.conv3(model.conv3_pad(h)))
        h = model.relu4(model.conv4(model.conv4_pad(h)))
        h = model.relu5(model.conv5(model.conv5_pad(h)))
        h = model.pool3(h)
        assert tuple(h.shape) == (None, 6, 6, 256)

    def test_the_executed_model_produces_that_shape_too(self):
        """A correct graph is not a correct forward pass; run it."""
        model = create_alexnet(include_top=False)
        out = model(_sample(batch=2))
        assert tuple(out.shape) == (2, 6, 6, 256)

    def test_flatten_is_9216(self):
        """fc6's input width, which the paper states."""
        model = AlexNet()
        assert model.get_feature_extent() == 6
        assert model._flat_features == 9216

    def test_the_convolutions_have_the_papers_filter_counts(self):
        model = AlexNet()
        assert model.conv1.filters == 96
        assert model.conv2.filters == 256
        assert model.conv3.filters == 384
        assert model.conv4.filters == 384
        assert model.conv5.filters == 256

    def test_the_convolutions_have_the_papers_kernel_and_stride(self):
        model = AlexNet()
        for layer, kernel, stride in [
            (model.conv1, 11, 4),
            (model.conv2, 5, 1),
            (model.conv3, 3, 1),
            (model.conv4, 3, 1),
            (model.conv5, 3, 1),
        ]:
            assert tuple(layer.kernel_size) == (kernel, kernel)
            assert tuple(layer.strides) == (stride, stride)

    def test_the_final_pool_is_valid_so_it_can_reach_six(self):
        """The asymmetry that produces the paper's figure."""
        model = AlexNet()
        assert model.pool3.padding == "valid"
        assert model.pool1.padding == "valid"  # padding comes from ZeroPadding2D
        assert model.pool2.padding == "valid"
        # ...and the pads that do the work.
        assert model.pool1_pad.padding == ((1, 1), (1, 1))
        assert model.pool2_pad.padding == ((1, 1), (1, 1))
        assert model.conv1_pad.padding == ((2, 2), (2, 2))

    def test_pool3_has_no_padding_layer(self):
        """``pad=0`` in the table must not leave an identity node in the graph."""
        model = AlexNet()
        assert not hasattr(model, "pool3_pad")

    @pytest.mark.parametrize("size", [224, 225, 226, 227])
    def test_224_through_227_all_reach_the_papers_figure(self, size):
        """227 is the paper's number, but it is not a magic one.

        This CORRECTS a claim that was true of the strict-``valid`` draft and false
        of the shipped padding: with the Caffe pads, 224 does reach 6 x 6, because
        the strides round down to the same extent. Any documentation asserting that
        "224 does not work" is wrong and must say so.
        """
        extent = size
        for _, kernel, stride, pad, _ in STAGES:
            extent = conv_out(extent, kernel, stride, pad)
        assert extent == 6

    def test_256_does_not_reach_it(self):
        """The boundary, so the tolerance above is not vacuous."""
        extent = 256
        for _, kernel, stride, pad, _ in STAGES:
            extent = conv_out(extent, kernel, stride, pad)
        assert extent == 7


class TestTheStageTableMatchesTheModel:
    """The guard and the architecture read one table; this pins they agree."""

    def test_every_stage_is_consumed_and_named_in_the_model(self):
        model = AlexNet()
        for stage_name, _, _, _, _ in STAGES:
            assert hasattr(model, stage_name), f"model has no {stage_name}"

    def test_the_table_reproduces_the_paper_extent_chain(self):
        """56, 28, 26, 13, 13, 13, 13, 6 -- every value the paper labels."""
        expected = [56, 28, 26, 13, 13, 13, 13, 6]
        extent = 227
        observed = []
        for _, kernel, stride, pad, _ in STAGES:
            extent = conv_out(extent, kernel, stride, pad)
            observed.append(extent)
        assert observed == expected

    def test_minimum_spatial_extent_is_55(self):
        assert MIN_SPATIAL_EXTENT == 55
        assert conv_out(54, *STAGES[0][1:4]) >= 1  # conv1 alone survives 54
        assert MIN_SPATIAL_EXTENT > 54


class TestTheConvolutionsAreGrouped:
    """The paper splits conv2-conv5 across two GPUs; ``groups=2`` is that."""

    @pytest.mark.parametrize("name", ["conv2", "conv3", "conv4", "conv5"])
    def test_conv2_through_conv5_are_split_in_two(self, name):
        model = AlexNet()
        assert getattr(model, name).groups == 2

    def test_conv1_is_not_grouped(self):
        """conv1 reads the image directly and was never split."""
        assert AlexNet().conv1.groups == 1

    def test_grouping_halves_the_parameter_count_of_the_affected_layers(self):
        """conv3 reads 442,752, not the 885,120 an ungrouped layer would need."""
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.conv3.count_params() == 384 * (256 // 2) * 9 + 384

    def test_conv5_kernel_records_the_halved_input_width(self):
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.conv5.kernel.shape == (3, 3, 384 // 2, 256)

    def test_grouping_changes_the_output_and_is_not_cosmetic(self):
        """A structural knob: different shapes consume different RNG draws."""
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.conv2.count_params() != (96 * 256 * 5 * 5 + 256)
        assert model.conv2.count_params() == (96 // 2) * 256 * 5 * 5 + 256


class TestTheLocalResponseNormalizationIsWiredIn:
    """AlexNet without LRN is not AlexNet."""

    def test_there_are_exactly_two_lrn_layers(self):
        model = AlexNet()
        lrns = [l for l in model.layers if isinstance(l, LocalResponseNormalization)]
        assert len(lrns) == 2
        assert model.lrn1 is not None and model.lrn2 is not None

    def test_lrn_follows_conv1_and_conv2_only(self):
        model = AlexNet()
        names = [l.name for l in model.layers]
        assert "lrn1" in names and "lrn2" in names
        assert not hasattr(model, "lrn3")
        assert not hasattr(model, "lrn4")
        assert not hasattr(model, "lrn5")

    def test_lrn_comes_after_the_relu_and_before_the_pool(self):
        """The paper's order, which matters: LRN on a negative-valued activation
        would divide by a different denominator than on the rectified one."""
        model = AlexNet()
        names = [l.name for l in model.layers]
        assert names.index("relu1") < names.index("lrn1") < names.index("pool1")
        assert names.index("relu2") < names.index("lrn2") < names.index("pool2")

    def test_the_lrn_hyperparameters_are_the_papers(self):
        """n=5 (radius 2), alpha=1e-4, beta=0.75, k=1."""
        model = AlexNet()
        assert model.lrn1.depth_radius == 2
        assert model.lrn1.alpha == pytest.approx(1e-4)
        assert model.lrn1.beta == pytest.approx(0.75)
        assert model.lrn1.k == pytest.approx(1.0)

    def test_the_default_table_matches_the_layer_defaults(self):
        assert LRN_HYPERPARAMETERS == {
            "depth_radius": 2, "alpha": 1e-4, "beta": 0.75, "k": 1.0
        }

    def test_removing_lrn_changes_the_output(self):
        """If it did not, the layers would be dead weight in the graph."""
        with_lrn = AlexNet()
        _ = with_lrn(_sample(batch=1, seed=1))
        y_with = np.asarray(with_lrn(_sample(batch=1, seed=1)))

        zeroed = AlexNet()
        _ = zeroed(_sample(batch=1, seed=1))  # build, so `band` exists
        for layer in zeroed.layers:
            if isinstance(layer, LocalResponseNormalization):
                layer._band_weight.assign(
                    keras.ops.convert_to_tensor(
                        np.eye(96 if layer.name == "lrn1" else 256, dtype="float32")
                    )
                )
        y_without = np.asarray(zeroed(_sample(batch=1, seed=1)))
        assert not np.allclose(y_with, y_without)

    def test_custom_lrn_hyperparameters_reach_the_layers(self):
        model = AlexNet(lrn_hyperparameters={"depth_radius": 4, "alpha": 1e-3,
                                             "beta": 0.5, "k": 2.0})
        assert model.lrn1.depth_radius == 4
        assert model.lrn1.alpha == pytest.approx(1e-3)


class TestConstructionValidatesEarly:
    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"num_classes": 0}, "num_classes"),
            ({"num_classes": -1}, "num_classes"),
            ({"num_classes": 2.5}, "num_classes"),
            ({"dropout_rate": 1.0}, "dropout_rate"),
            ({"dropout_rate": -0.1}, "dropout_rate"),
            ({"input_shape": (227, 227)}, "input_shape"),
            ({"input_shape": (227, 227, 3, 1)}, "input_shape"),
        ],
    )
    def test_it_rejects_bad_arguments(self, kwargs, fragment):
        with pytest.raises(ValueError) as excinfo:
            AlexNet(**kwargs)
        assert fragment in str(excinfo.value)

    @pytest.mark.parametrize("size", [32, 54])
    def test_it_rejects_an_input_too_small_to_survive_the_final_pool(self, size):
        """A zero-length spatial axis yields all-NaN, not an error, so the
        constructor must refuse it up front."""
        with pytest.raises(ValueError) as excinfo:
            AlexNet(input_shape=(size, size, 3))
        assert "collapses" in str(excinfo.value)
        assert str(MIN_SPATIAL_EXTENT) in str(excinfo.value)

    def test_it_accepts_exactly_the_minimum_extent(self):
        model = AlexNet(input_shape=(MIN_SPATIAL_EXTENT, MIN_SPATIAL_EXTENT, 3))
        assert model is not None

    def test_pretrained_raises_rather_than_returning_random_weights(self):
        with pytest.raises(NotImplementedError) as excinfo:
            create_alexnet(pretrained=True)
        assert "no pretrained weights" in str(excinfo.value).lower()

    def test_a_pretrained_string_also_raises(self):
        with pytest.raises(NotImplementedError):
            create_alexnet(pretrained="imagenet")

    def test_load_pretrained_weights_raises(self):
        with pytest.raises(NotImplementedError):
            AlexNet().load_pretrained_weights("/tmp/does-not-exist.keras")


class TestForwardPassAndShapes:
    def test_the_default_model_emits_class_probabilities(self):
        model = AlexNet()
        out = model(_sample(batch=2))
        assert out.shape == (2, 1000)

    def test_the_output_is_a_probability_distribution(self):
        model = AlexNet()
        out = np.asarray(model(_sample(batch=4)))
        assert np.all(out >= 0.0)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), rtol=0, atol=1e-5)

    def test_num_classes_changes_the_output_width(self):
        model = AlexNet(num_classes=37)
        assert model(_sample(batch=2)).shape == (2, 37)

    def test_include_top_false_returns_the_feature_map(self):
        model = create_alexnet(include_top=False)
        out = model(_sample(batch=2))
        assert tuple(out.shape) == (2, 6, 6, 256)

    def test_include_top_false_drops_the_fully_connected_layers(self):
        model = create_alexnet(include_top=False)
        names = [l.name for l in model.layers]
        assert not any(n.startswith("fc") for n in names)
        assert "conv5" in names

    def test_a_non_default_input_shape_is_honoured(self):
        model = AlexNet(input_shape=(160, 160, 3))
        out = model(_sample(batch=2, size=160))
        assert out.shape == (2, 1000)
        assert model.get_feature_extent() == 4

    def test_dropout_is_inactive_at_inference(self):
        """Two inference passes must agree exactly."""
        model = AlexNet()
        x = _sample(batch=2, seed=2)
        first = np.asarray(model(x, training=False))
        second = np.asarray(model(x, training=False))
        np.testing.assert_array_equal(first, second)

    def test_dropout_is_active_in_training(self):
        model = AlexNet(dropout_rate=0.9)
        x = _sample(batch=8, seed=3)
        first = np.asarray(model(x, training=True))
        second = np.asarray(model(x, training=True))
        assert not np.array_equal(first, second)


class TestGradientsReachEveryStage:
    def test_one_fit_step_updates_the_convolutions_and_the_head(self):
        model = create_alexnet(num_classes=4)
        model.compile(
            optimizer="sgd", loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        rng = np.random.default_rng(4)
        # Build first: reading `.kernel` before any call raises, and the point of the
        # test is to compare BEFORE and AFTER, so both sides need a built model.
        _ = model(rng.normal(size=(1, 227, 227, 3)).astype("float32"))
        before_conv1 = keras.ops.convert_to_numpy(model.conv1.kernel).copy()
        before_fc6 = keras.ops.convert_to_numpy(model.fc6.kernel).copy()

        history = model.fit(
            rng.normal(size=(4, 227, 227, 3)).astype("float32"),
            rng.integers(0, 4, size=4),
            epochs=1,
            verbose=0,
        )
        assert np.isfinite(history.history["loss"][0])
        assert not np.array_equal(
            before_conv1, keras.ops.convert_to_numpy(model.conv1.kernel)
        ), "conv1 received no gradient"
        assert not np.array_equal(
            before_fc6, keras.ops.convert_to_numpy(model.fc6.kernel)
        ), "fc6 received no gradient"

    def test_the_lrn_band_is_not_updated_by_training(self):
        """It is a constant; a gradient step must leave it untouched."""
        model = create_alexnet(num_classes=4)
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
        rng = np.random.default_rng(5)
        _ = model(rng.normal(size=(1, 227, 227, 3)).astype("float32"))
        before = keras.ops.convert_to_numpy(model.lrn1.band).copy()
        model.fit(
            rng.normal(size=(4, 227, 227, 3)).astype("float32"),
            rng.integers(0, 4, size=4),
            epochs=1,
            verbose=0,
        )
        np.testing.assert_array_equal(
            before, keras.ops.convert_to_numpy(model.lrn1.band)
        )


class TestParameterAccounting:
    def test_the_total_matches_the_documented_figure(self):
        """The docstring prints 60,597,608; this keeps it honest."""
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.count_params() == 60_597_608

    def test_the_breakdown_matches_the_documented_table(self):
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.conv1.count_params() == 34_944
        assert model.conv2.count_params() == 307_456
        assert model.conv3.count_params() == 442_752
        assert model.conv4.count_params() == 663_936
        assert model.conv5.count_params() == 442_624
        assert model.fc6.count_params() == 37_752_832
        assert model.fc7.count_params() == 16_781_312
        assert model.fc8.count_params() == 4_097_000

    def test_ninety_seven_percent_of_the_parameters_are_fully_connected(self):
        """The architecture's defining cost, recorded as a number."""
        model = AlexNet()
        _ = model(_sample(batch=1))
        dense = (
            model.fc6.count_params() + model.fc7.count_params()
            + model.fc8.count_params()
        )
        assert dense / model.count_params() > 0.96

    def test_the_lrn_bands_are_non_trainable(self):
        model = AlexNet()
        _ = model(_sample(batch=1))
        # `layer.trainable` is a Keras flag and stays True even when a layer holds no
        # trainable weights, so the meaningful assertion is on the weights themselves.
        assert model.lrn1.trainable_weights == []
        assert model.lrn2.trainable_weights == []
        assert len(model.lrn1.non_trainable_weights) == 1
        assert model.lrn1.count_params() == 96 * 96
        assert model.lrn2.count_params() == 256 * 256

    def test_dropout_and_pools_add_no_parameters(self):
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.dropout1.count_params() == 0
        assert model.dropout2.count_params() == 0
        assert model.pool3.count_params() == 0

    def test_the_include_top_model_is_much_smaller(self):
        model = create_alexnet(include_top=False)
        _ = model(_sample(batch=1))
        assert model.count_params() < 2_000_000


class TestSerialization:
    def test_a_saved_and_reloaded_model_computes_identically(self, tmp_path):
        model = AlexNet(num_classes=11)
        x = _sample(batch=2, seed=6)
        expected = np.asarray(model(x))

        path = tmp_path / "alexnet.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_allclose(
            np.asarray(reloaded(x)), expected, rtol=1e-6, atol=1e-6
        )

    def test_get_config_carries_every_constructor_argument(self):
        config = AlexNet().get_config()
        for key in (
            "num_classes", "input_shape", "dropout_rate", "kernel_initializer",
            "bias_initializer", "include_top", "lrn_hyperparameters",
        ):
            assert key in config, f"{key} missing from get_config()"

    def test_the_config_rebuilds_an_identical_architecture(self):
        model = AlexNet(num_classes=11, dropout_rate=0.25, include_top=False)
        rebuilt = AlexNet.from_config(model.get_config())
        assert rebuilt.num_classes == 11
        assert rebuilt.dropout_rate == pytest.approx(0.25)
        assert rebuilt.include_top is False
        # Build both before comparing weights: a freshly constructed model has no
        # built sublayers, and count_params() raises on those.
        x = _sample(batch=1, seed=9)
        _ = model(x)
        _ = rebuilt(x)
        assert [l.count_params() for l in rebuilt.layers] == [
            l.count_params() for l in model.layers
        ]
        assert rebuilt.get_feature_extent() == model.get_feature_extent()

    def test_a_non_default_setting_survives_the_round_trip(self):
        """A dropped hyperparameter would still round-trip, just wrongly."""
        rebuilt = AlexNet.from_config(
            AlexNet(lrn_hyperparameters={"depth_radius": 5, "alpha": 1e-3,
                                        "beta": 0.6, "k": 3.0}).get_config()
        )
        assert rebuilt.lrn1.depth_radius == 5
        assert rebuilt.lrn1.alpha == pytest.approx(1e-3)
        assert rebuilt.lrn1.k == pytest.approx(3.0)

    def test_the_registered_name_strips_the_family_directory(self):
        """The key must be ``models.alexnet.model``, not ``models.vision....``."""
        name = keras.saving.get_registered_name(AlexNet)
        assert name == "dl_techniques.models.alexnet.model>AlexNet"

    def test_the_package_exports_exactly_the_model_and_the_factory(self):
        import dl_techniques.models.vision.alexnet as pkg

        assert set(pkg.__all__) == {"AlexNet", "create_alexnet"}


class TestTheGuardFailsWhenTheArchitectureIsBroken:
    """RED proofs. Each mutation breaks a claim and a test must catch it."""

    def test_the_pool3_claim_would_be_caught(self):
        """Mutating pool3 to 'same' padding breaks the 6x6 assertion."""
        model = AlexNet()
        assert model.get_feature_extent() == 6, "precondition"

        h = model.conv5(model.conv5_pad(keras.Input((13, 13, 256))))
        padded = model.pool3(h)
        assert tuple(padded.shape)[1] == 6, "precondition: valid pool over 13 -> 6"

        model.pool3.padding = "same"
        mutated = model.pool3(h)
        assert tuple(mutated.shape)[1] == 7, (
            "'same' pool3 no longer gives 7, so this RED proof is stale and the "
            "padding trap it documents may have changed"
        )

    def test_using_same_padding_instead_of_explicit_pads_would_be_caught(self):
        """The trap this package exists to avoid, checked numerically."""
        from keras import layers as L

        x = keras.Input((227, 227, 3))
        h = L.Conv2D(96, 11, strides=4, padding="same")(x)
        assert tuple(h.shape)[1] == 57, (
            "Keras 'same' no longer gives 57 for conv1; the README's warning "
            "about it is stale and should be re-measured"
        )

    def test_dropping_a_group_would_be_caught(self):
        model = AlexNet()
        _ = model(_sample(batch=1))
        assert model.conv3.groups == 2
        grouped = model.conv3.count_params()
        # NOT exactly half: the bias (384) is not halved, so the KERNEL is what
        # halves. Asserting grouped == ungrouped // 2 is off by 192.
        assert grouped != (384 * 256 * 9 + 384)
        assert grouped - 384 == (384 * 256 * 9) // 2, (
            "grouping halves the kernel; if it did not, the groups=2 claim in "
            "the docstring is wrong"
        )

    def test_a_too_small_input_is_refused_rather_than_returning_nans(self):
        """The failure the guard exists to convert into a named error."""
        with pytest.raises(ValueError):
            AlexNet(input_shape=(40, 40, 3))