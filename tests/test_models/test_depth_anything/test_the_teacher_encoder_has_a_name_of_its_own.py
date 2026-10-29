"""The frozen teacher encoder of ``DepthAnything`` has a layer name and weight paths of its own.

The student encoder (``encoder_vit_s``, or ``encoder_vit_s_real``) and the frozen teacher built
beside it used to carry one name, and their weights one set of paths (152 weights on 106 paths at
the placeholder, 410 on 235 with the DINOv2 encoder). Every transfer that matches layers BY NAME
(``load_weights_from_checkpoint``, the trainer's ``--init-from``) keeps one layer per name, so it
filled the teacher and left the student at its random start while reporting ``loaded 1 layer``
(R4 finding 2: 0 of 10 student kernels equal the checkpoint, the frozen teacher took 10 of 10).

The teacher is now ``frozen_<student name>`` and is built under a name scope of that name, so its
weight paths start with it. The ``.keras`` archive is keyed by attribute (``encoder``,
``frozen_encoder``), not by layer name, so it is unchanged: an archive written before this change
still loads (``scratch_iter2/step19_2`` carries the transcripts, the model README quotes them).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything.model import DepthAnything
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

#: ``encoder_kind -> (image side, student name, teacher name)``. The names are written out, not
#: derived from the model: a renamed prefix must fail here.
KINDS = {
    "placeholder": (32, "encoder_vit_s", "frozen_encoder_vit_s"),
    "real": (56, "encoder_vit_s_real", "frozen_encoder_vit_s_real"),
}


def _model(kind: str, seed: int = 0, **changes) -> DepthAnything:
    side = KINDS[kind][0]
    keras.utils.set_random_seed(seed)
    arguments = dict(encoder_type="vit_s", encoder_kind=kind, image_shape=(side, side, 3),
                     use_feature_alignment=True)
    if kind == "placeholder":
        arguments.update(dpt_features=8, dpt_out_channels=[8, 8, 8, 8])
    arguments.update(changes)
    model = DepthAnything(**arguments)
    model.build((None, side, side, 3))
    return model


def _kernels(layer: keras.layers.Layer) -> list:
    return [w.numpy() for w in layer.weights if w.path.endswith("/kernel")]


def _perturb(layer: keras.layers.Layer, by: float = 1.0) -> None:
    layer.set_weights([w + by for w in layer.get_weights()])


@pytest.mark.parametrize("kind", sorted(KINDS))
class TestTheNames:

    def test_the_teacher_is_the_students_name_behind_a_prefix(self, kind: str) -> None:
        _, student, teacher = KINDS[kind]
        model = _model(kind)
        assert model.encoder.name == student
        assert model.frozen_encoder.name == teacher

    def test_the_model_owns_exactly_these_layers_each_once(self, kind: str) -> None:
        _, student, teacher = KINDS[kind]
        model = _model(kind)
        names = sorted(layer.name for layer in model._flatten_layers(include_self=False, recursive=False))
        assert names == sorted([student, teacher, "depth_head", "strong_augmentation"])

    def test_no_two_weights_share_a_path(self, kind: str) -> None:
        model = _model(kind)
        paths = [w.path for w in model.weights]
        assert len(set(paths)) == len(paths), "a weight path repeats: by-path addressing reads one of two"

    def test_the_teachers_weights_live_under_its_name_and_the_students_do_not(self, kind: str) -> None:
        _, student, teacher = KINDS[kind]
        model = _model(kind)
        teacher_paths = [w.path for w in model.frozen_encoder.weights]
        student_paths = [w.path for w in model.encoder.weights]
        assert teacher_paths and all(f"/{teacher}/" in p for p in teacher_paths), teacher_paths[:2]
        assert student_paths and not any(f"/{teacher}/" in p for p in student_paths)

    def test_the_teacher_starts_as_the_students_weights(self, kind: str) -> None:
        model = _model(kind)
        for got, want in zip(model.frozen_encoder.get_weights(), model.encoder.get_weights()):
            np.testing.assert_array_equal(got, want)
        assert not model.frozen_encoder.trainable and model.encoder.trainable

    def test_a_model_without_a_teacher_has_no_prefixed_layer(self, kind: str) -> None:
        _, student, teacher = KINDS[kind]
        model = _model(kind, use_feature_alignment=False)
        assert model.frozen_encoder is None
        names = [layer.name for layer in model._flatten_layers(include_self=False, recursive=False)]
        assert teacher not in names and names.count(student) == 1


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_name_keyed_load_fills_the_student_and_the_teacher_each(kind: str, tmp_path) -> None:
    """The R4 finding: the load filled the teacher and left the student random."""
    _, student, teacher = KINDS[kind]
    source = _model(kind, seed=0)
    _perturb(source.frozen_encoder, 1.0)  # the teacher differs from the student, so a swap is visible
    path = str(tmp_path / "source.keras")
    source.save(path)
    target = _model(kind, seed=1)
    fresh = [k.copy() for k in _kernels(target.encoder)]

    report = load_weights_from_checkpoint(target=target, ckpt_path=path, skip_prefixes=("depth_head",))

    assert sorted(report.loaded) == sorted([student, teacher])
    got_student, want_student = _kernels(target.encoder), _kernels(source.encoder)
    got_teacher, want_teacher = _kernels(target.frozen_encoder), _kernels(source.frozen_encoder)
    assert got_student and len(got_student) == len(want_student)
    for got, want in zip(got_student, want_student):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(got_teacher, want_teacher):
        np.testing.assert_array_equal(got, want)
    assert all(not np.array_equal(a, b) for a, b in zip(fresh, want_student)), (
        "degenerate probe: the target started equal to the source")
    assert all(not np.array_equal(a, b) for a, b in zip(want_student, want_teacher)), (
        "degenerate probe: the source's two encoders are equal")


def test_a_round_trip_keeps_both_names_and_every_weight(tmp_path) -> None:
    source = _model("placeholder", seed=0)
    _perturb(source.frozen_encoder, 1.0)
    path = str(tmp_path / "round_trip.keras")
    source.save(path)
    loaded = keras.models.load_model(path)
    assert loaded.encoder.name == "encoder_vit_s" and loaded.frozen_encoder.name == "frozen_encoder_vit_s"
    # A loaded model's student paths carry no model root (its encoder is deserialized outside the
    # model's scope), a difference that predates the teacher's name; the paths stay unique, and the
    # values are the saved ones.
    paths = [w.path for w in loaded.weights]
    assert len(set(paths)) == len(paths)
    assert len(source.weights) == len(loaded.weights)
    for a, b in zip(source.weights, loaded.weights):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_the_config_names_the_student_only() -> None:
    """The teacher's name is derived at build time, so no archive carries it and none can disagree."""
    config = _model("placeholder").get_config()
    assert config["encoder"]["config"]["name"] == "encoder_vit_s"
    assert "frozen_encoder" not in config
