"""``--init-from`` fills the STUDENT encoder, counts what it loaded, and refuses a start that loaded none.

The student and the frozen teacher of a ``DepthAnything`` shared one layer name, and
``load_weights_from_checkpoint`` matches top-level layers by name, keeping one layer per name: the
transfer filled the teacher and left the student random while reporting "loaded 1 layer" (R4
finding 2, MEASURED on a model with a teacher: 0 of 10 student kernels equal the checkpoint's, 10
of 10 of the teacher's). The trainer then applied the pretrained-start decoder learning-rate
multiplier to a random encoder. ``test_init_from_skips_the_decoder.py`` builds without a teacher, so
it could not see this. The teacher now has a name of its own (``frozen_<student>``), the report
counts kernels, and a transfer that did not load the student raises.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything.model import create_depth_anything
from dl_techniques.utils.weight_transfer import TransferReport
from train.depth_anything import train_depth_anything as trainer
from train.depth_anything.train_depth_anything import encoder_kernel_counts, init_from_checkpoint

#: Ten convolution kernels in the placeholder encoder: the stem, eight body convolutions, the projection.
ENCODER_KERNELS = 10


def _model(seed: int, teacher: bool = True):
    keras.utils.set_random_seed(seed)
    return create_depth_anything(
        encoder_type="vit_s", encoder_kind="placeholder",
        image_shape=(32, 32, 3), dpt_features=8, dpt_out_channels=[8, 8, 8, 8],
        use_feature_alignment=teacher)


def _kernels(layer):
    return [w.numpy() for w in layer.weights if w.path.endswith("/kernel")]


def _saved_source(tmp_path, teacher: bool = True):
    source = _model(0, teacher)
    if teacher:  # the teacher differs from the student, so a swap between them is visible
        source.frozen_encoder.set_weights([w + 1.0 for w in source.frozen_encoder.get_weights()])
    path = str(tmp_path / "source.keras")
    source.save(path)
    return source, path


def test_the_student_loads_when_the_model_has_a_teacher(tmp_path) -> None:
    source, path = _saved_source(tmp_path)
    target = _model(1)
    fresh_student = _kernels(target.encoder)
    fresh_decoder = [w.copy() for w in target.decoder.get_weights()]

    report = init_from_checkpoint(target, path)

    assert sorted(report.loaded) == ["encoder_vit_s", "frozen_encoder_vit_s"]
    student, want = _kernels(target.encoder), _kernels(source.encoder)
    assert len(student) == len(want) == ENCODER_KERNELS
    for got, expected in zip(student, want):
        np.testing.assert_array_equal(got, expected)
    for got, expected in zip(_kernels(target.frozen_encoder), _kernels(source.frozen_encoder)):
        np.testing.assert_array_equal(got, expected)
    assert all(not np.array_equal(a, b) for a, b in zip(fresh_student, want)), "degenerate probe"
    for got, expected in zip(target.decoder.get_weights(), fresh_decoder):
        np.testing.assert_array_equal(got, expected)


def test_the_kernel_count_reads_the_student_and_names_its_total() -> None:
    target = _model(1)
    loaded_report = TransferReport(loaded=[target.encoder.name])
    assert encoder_kernel_counts(target, loaded_report) == (ENCODER_KERNELS, ENCODER_KERNELS)
    teacher_only = TransferReport(loaded=[target.frozen_encoder.name])
    assert encoder_kernel_counts(target, teacher_only) == (0, ENCODER_KERNELS)


def test_the_log_states_kernels_loaded_of_kernels_expected(tmp_path, caplog) -> None:
    _, path = _saved_source(tmp_path)
    with caplog.at_level("INFO"):
        init_from_checkpoint(_model(1), path)
    assert f"{ENCODER_KERNELS} of {ENCODER_KERNELS} kernels loaded" in caplog.text


def test_a_transfer_that_loaded_only_the_teacher_is_refused_naming_the_student(
        tmp_path, monkeypatch) -> None:
    """The old defect, replayed: the report lists the teacher and not the student."""
    _, path = _saved_source(tmp_path)
    target = _model(1)
    fresh = [w.copy() for w in target.encoder.get_weights()]
    monkeypatch.setattr(
        trainer, "load_weights_from_checkpoint",
        lambda **kwargs: TransferReport(loaded=[target.frozen_encoder.name],
                                        skipped_by_prefix=[target.decoder.name]))
    with pytest.raises(RuntimeError, match=r"0 of 10 kernels of the student encoder 'encoder_vit_s'"):
        init_from_checkpoint(target, path)
    for got, want in zip(target.encoder.get_weights(), fresh):
        np.testing.assert_array_equal(got, want)


def test_a_checkpoint_without_a_teacher_leaves_the_teacher_equal_to_the_student(tmp_path) -> None:
    """The checkpoint carries no teacher; the target's teacher must not stay at the random start."""
    source, path = _saved_source(tmp_path, teacher=False)
    target = _model(1)
    report = init_from_checkpoint(target, path)
    assert report.loaded == ["encoder_vit_s"]
    for got, want in zip(target.frozen_encoder.get_weights(), source.encoder.get_weights()):
        np.testing.assert_array_equal(got, want)
