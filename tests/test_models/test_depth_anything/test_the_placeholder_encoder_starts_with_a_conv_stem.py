"""The placeholder encoder's first stage is one ``ConvStem``, built once, cloned for the teacher.

``DepthAnything(encoder_kind="placeholder")`` wrote its first stage as three loose
layers (``initial_conv``, ``initial_bn``, ``initial_relu``). It is now one ``ConvStem``
named ``initial_stem`` inside the functional encoder. The numeric identity with the
three layers is proved in ``tests/test_models/test_the_conv_stem_replicates_every_hand_rolled_stem.py``;
this file pins what that file cannot see:

* the encoder graph holds the stem and nothing of the old spelling, with 46 weights and
  4,932,544 parameters (the count of the three loose layers);
* the stem survives ``keras.models.clone_model``: ``build`` swallows a clone failure
  into a warning and drops the EMA teacher, so a stem the clone cannot rebuild would
  turn the semi-supervised step into a labeled-only one without a word;
* the model's first seeded draw is the stem's kernel (the instance the model creates in
  ``__init__`` is the stem's, every other convolution takes a clone of it), so no seeded
  tensor of the real DINOv2 model moved;
* the constructor refuses what the model does not take.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.depth_anything.model import DepthAnything

IMAGE_SHAPE = (64, 64, 3)
#: Weights and parameters of the placeholder encoder at ``vit_s``: the stem's conv kernel
#: and four BatchNorm tensors, eight body convolutions with BatchNorm, the projection.
ENCODER_WEIGHTS = 46
ENCODER_PARAMS = 4_932_544


def _model(**changes) -> DepthAnything:
    arguments = dict(
        encoder_type="vit_s",
        encoder_kind="placeholder",
        image_shape=IMAGE_SHAPE,
        use_feature_alignment=False,
    )
    arguments.update(changes)
    model = DepthAnything(**arguments)
    model.build((None, *IMAGE_SHAPE))
    return model


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).uniform(size=(batch, *IMAGE_SHAPE)).astype("float32")


class TestTheEncoderGraph:

    def test_the_first_stage_is_one_conv_stem_named_initial_stem(self) -> None:
        encoder = _model().encoder
        names = [layer.name for layer in encoder.layers]
        stem = encoder.get_layer("initial_stem")
        assert isinstance(stem, ConvStem)
        assert names[:2] == ["encoder_input", "initial_stem"]
        loose = {"initial_conv", "initial_bn", "initial_relu"} & set(names)
        assert not loose, f"the old loose layers are still in the graph: {sorted(loose)}"

    def test_the_stem_keeps_the_weights_and_the_encoder_keeps_its_size(self) -> None:
        encoder = _model().encoder
        assert len(encoder.get_layer("initial_stem").weights) == 5
        assert len(encoder.weights) == ENCODER_WEIGHTS
        assert encoder.count_params() == ENCODER_PARAMS

    def test_the_stem_is_the_reference_stage_one(self) -> None:
        stem = _model().encoder.get_layer("initial_stem")
        assert (stem.filters, stem.kernel_size, stem.strides, stem.padding) == (64, 7, 2, "same")
        assert stem.use_bias is False
        assert stem.normalization_type == "batch_norm"
        assert stem.norm1.epsilon == 1e-5
        assert stem.num_convs == 1 and stem.pool is None


class TestTheTeacherClone:

    def test_the_clone_keeps_the_stem_and_the_teacher_exists(self) -> None:
        model = _model(use_feature_alignment=True, enable_semi_supervised=True)
        assert model.frozen_encoder is not None, (
            "clone_model failed on the encoder: the teacher was dropped with a warning")
        assert isinstance(model.frozen_encoder.get_layer("initial_stem"), ConvStem)
        assert model.use_feature_alignment is True

    def test_the_teacher_reproduces_the_student_exactly_and_is_frozen(self) -> None:
        model = _model(use_feature_alignment=True, enable_semi_supervised=True)
        x = _images()
        student = keras.ops.convert_to_numpy(model.encoder(x, training=False))
        teacher = keras.ops.convert_to_numpy(model.frozen_encoder(x, training=False))
        assert np.array_equal(student, teacher)
        assert model.frozen_encoder.trainable is False
        assert not model.frozen_encoder.trainable_weights

    def test_a_plain_clone_of_the_encoder_round_trips_at_zero(self) -> None:
        encoder = _model().encoder
        clone = keras.models.clone_model(encoder)
        clone.set_weights(encoder.get_weights())
        x = _images()
        for training in (False, True):
            want = keras.ops.convert_to_numpy(encoder(x, training=training))
            got = keras.ops.convert_to_numpy(clone(x, training=training))
            assert np.array_equal(want, got), f"training={training}"


class TestTheSeededDraw:

    def test_the_stems_kernel_is_the_first_draw_the_model_makes(self) -> None:
        """One initializer instance is built in ``__init__``; the stem draws from it.

        Storing a spec instead consumes one default seed less there and moves every seeded
        tensor of the model (measured: 104 of 410 for the real encoder).
        """
        keras.utils.set_random_seed(1234)
        expected = keras.ops.convert_to_numpy(
            keras.initializers.get("he_normal")((7, 7, 3, 64), dtype="float32"))
        keras.utils.set_random_seed(1234)
        kernel = keras.ops.convert_to_numpy(_model().encoder.get_layer("initial_stem").conv1.kernel)
        assert np.array_equal(kernel, expected)

    def test_construction_draws_exactly_one_default_seed(self) -> None:
        """The stream position every later draw (the real DINOv2 encoder, the head) starts from.

        ``__init__`` builds one initializer object from the argument; storing a spec instead builds
        none there, and every tensor created after it, in the real model too, starts one default
        seed earlier (measured: 104 of 410 tensors of the real ``vit_s`` model differ). The next
        initializer's ``seed`` shows how many were drawn.
        """
        keras.utils.set_random_seed(7)
        seeds = [keras.initializers.HeNormal().seed for _ in range(3)]
        keras.utils.set_random_seed(7)
        DepthAnything(encoder_kind="real", encoder_type="vit_s", image_shape=(56, 56, 3))
        assert keras.initializers.HeNormal().seed == seeds[1]

    def test_no_other_convolution_of_the_encoder_repeats_the_stems_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        encoder = _model().encoder
        stem = keras.ops.convert_to_numpy(encoder.get_layer("initial_stem").conv1.kernel).ravel()
        for name in ("conv_block_0_1", "conv_block_0_2", "conv_block_3_2", "feature_projection"):
            other = keras.ops.convert_to_numpy(encoder.get_layer(name).kernel).ravel()
            n = min(256, stem.size, other.size)
            r = float(np.corrcoef(stem[:n], other[:n])[0, 1])
            assert abs(r) < 0.5, f"{name} replays the stem's draw (r={r:.4f})"


class TestTheArmSubjectIsLive:

    def test_the_precision_arm_subject_does_not_output_all_zero_at_the_arms_seed(self) -> None:
        """The arms seed 0 and build; behind the head's ReLU an unlucky draw outputs all zeros,
        and every arm then compares zero with zero. The subject's builder pins its own seed."""
        from tests.test_models.precision_arm_subjects import SUBJECTS

        build, make_inputs, _ = SUBJECTS["depth_anything"]
        keras.utils.set_random_seed(0)
        output = keras.ops.convert_to_numpy(build()(make_inputs(), training=False))
        assert float(np.abs(output).max()) > 0.0
        assert float((output == 0.0).mean()) < 0.9


class TestTheConstructor:

    @pytest.mark.parametrize("keyword, value", [("input_shape", (64, 64, 3)), ("input_dim", 3)])
    def test_a_shape_keyword_is_refused_naming_it(self, keyword, value) -> None:
        with pytest.raises(ValueError, match=keyword):
            DepthAnything(**{keyword: value})

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"output_channels": 0}, "output_channels"),
            ({"output_channels": True}, "output_channels"),
            ({"dpt_features": -4}, "dpt_features"),
            ({"dpt_out_channels": [8, 0, 8, 8]}, "dpt_out_channels"),
            ({"image_shape": (64, 64)}, "image_shape"),
            ({"image_shape": (64, 64, None)}, "image_shape"),
        ],
    )
    def test_a_bad_count_or_shape_is_refused_naming_the_argument(self, changes, message) -> None:
        with pytest.raises(ValueError, match=message):
            DepthAnything(encoder_kind="placeholder", **changes)

    @pytest.mark.parametrize("value", [1, 0, None, "yes"])
    def test_include_top_must_be_a_bool(self, value) -> None:
        with pytest.raises(TypeError, match="include_top"):
            DepthAnything(encoder_kind="placeholder", image_shape=IMAGE_SHAPE, include_top=value)

    def test_a_numpy_count_is_stored_as_a_python_int(self) -> None:
        model = DepthAnything(
            encoder_kind="placeholder", image_shape=IMAGE_SHAPE, output_channels=np.int64(2),
            dpt_features=np.int64(16))
        assert type(model.output_channels) is int and type(model.dpt_features) is int
