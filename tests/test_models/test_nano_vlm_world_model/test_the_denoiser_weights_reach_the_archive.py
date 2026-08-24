"""The denoisers' block weights must reach the `.keras` archive.

Rationale
---------
`ConditionalDenoiser.blocks` and `JointDenoiser.joint_blocks` were lists of
dicts of Layers -- layer containers nested two levels deep, owned by a
`keras.layers.Layer`. Keras 3.8 does not write such a container to
`model.weights.h5`. MEASURED before the fix (perturbed weights, CPU):

    ConditionalDenoiser  archive 12 of 44 tensors, 2,848 of 9,408 params
                         12 of 44 identical after reload
                         forward delta 9.118214e-03 (range 2.552134e+00)
    JointDenoiser        archive  8 of 112 tensors,  824 of 18,424 params
                         8 of 112 identical after reload
                         forward delta 1.737189e-02 (range 2.562671e+00)

At full model scale that is the 464 of 1,305 weight tensors batch 3 recorded.

**Both classes already override `build()` and materialise every sub-layer, and
that does NOT help.** The container SHAPE is the property that matters, which is
the second independent confirmation in this repo that "overrides `build()`" is
the wrong predicate for this defect.

Anti-vacuity
------------
Every weight is perturbed before saving. That matters more here than usual: the
denoisers' output projections are `kernel_initializer='zeros'`, so an untrained
denoiser is EXACTLY the identity and every forward assertion compares the input
against itself -- the mechanism that let this package's own `test_round_trip.py`
pass 3/3 against a 464-tensor loss.

See decisions.md D-039 (plan-2026-08-19T163559-499b6f0e).
"""

import io
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.nano_vlm_world_model.denoisers import (
    ConditionalDenoiser,
    JointDenoiser,
    TextDenoiser,
    TimestepEmbedding,
    VisionDenoiser,
)

# ---------------------------------------------------------------------

BATCH, SEQ, DATA_DIM, COND_DIM = 2, 4, 16, 8


@keras.saving.register_keras_serializable(package="test_nano_vlm_world_model")
class _DenoiserHost(keras.Model):
    """Minimal `keras.Model` host so the denoiser can be saved standalone."""

    def __init__(self, kind: str = "conditional", **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        if kind == "conditional":
            self.denoiser = ConditionalDenoiser(
                data_dim=DATA_DIM, condition_dim=COND_DIM, hidden_dim=16,
                num_layers=2, num_attention_heads=2)
        else:
            self.denoiser = JointDenoiser(
                vision_dim=DATA_DIM, text_dim=COND_DIM, hidden_dim=16,
                num_layers=2)

    def build(self, input_shape):
        self.denoiser.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        data, condition, timesteps = inputs
        return self.denoiser(data, condition, timesteps, training=training)

    def get_config(self):
        config = super().get_config()
        config["kind"] = self.kind
        return config


def _flatten(outputs) -> np.ndarray:
    if isinstance(outputs, (list, tuple)):
        return np.concatenate([np.array(t).ravel() for t in outputs])
    return np.array(outputs).ravel()


def _round_trip(kind: str, tmp_path):
    keras.utils.set_random_seed(1234)
    model = _DenoiserHost(kind=kind)
    model.build([(None, SEQ, DATA_DIM), (None, SEQ, COND_DIM), (None,)])

    rng = np.random.RandomState(7)
    for weight in model.weights:
        value = np.array(weight)
        weight.assign(value + rng.randn(*value.shape).astype(value.dtype)
                      * max(0.25 * float(np.std(value)), 1e-3))

    data = np.random.RandomState(0).randn(BATCH, SEQ, DATA_DIM).astype("float32")
    cond = np.random.RandomState(1).randn(BATCH, SEQ, COND_DIM).astype("float32")
    steps = np.zeros((BATCH,), dtype="int32")

    before = _flatten(model([data, cond, steps], training=False))
    weights_before = [np.array(w) for w in model.weights]

    path = tmp_path / f"{kind}.keras"
    model.save(path)

    with zipfile.ZipFile(path) as archive:
        payload = archive.read("model.weights.h5")
    datasets, elements = 0, 0

    def _visit(_name, obj):
        nonlocal datasets, elements
        if isinstance(obj, h5py.Dataset):
            datasets += 1
            elements += int(np.prod(obj.shape)) if obj.shape else 1

    with h5py.File(io.BytesIO(payload), "r") as handle:
        handle.visititems(_visit)

    restored = keras.models.load_model(path)
    after = _flatten(restored([data, cond, steps], training=False))

    return {
        "expected_tensors": len(model.weights),
        "expected_params": model.count_params(),
        "datasets": datasets,
        "elements": elements,
        "weights_before": weights_before,
        "weights_after": [np.array(w) for w in restored.weights],
        "before": before,
        "after": after,
    }


@pytest.fixture(scope="module")
def conditional(tmp_path_factory):
    return _round_trip("conditional", tmp_path_factory.mktemp("cond"))


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    return _round_trip("joint", tmp_path_factory.mktemp("joint"))


@pytest.mark.parametrize("arm", ["conditional", "joint"])
def test_the_archive_holds_every_tensor(arm, request):
    result = request.getfixturevalue(arm)
    assert result["datasets"] == result["expected_tensors"], (
        f"{arm}: archive holds {result['datasets']} datasets against "
        f"{result['expected_tensors']} weights. Keep the block sub-layers in "
        f"FLAT per-role lists -- overriding `build()` does NOT protect against "
        f"this."
    )
    assert result["elements"] == result["expected_params"]


@pytest.mark.parametrize("arm", ["conditional", "joint"])
def test_every_weight_survives_the_round_trip(arm, request):
    result = request.getfixturevalue(arm)
    before, after = result["weights_before"], result["weights_after"]
    mismatched = [
        i for i, (a, b) in enumerate(zip(before, after))
        if a.shape != b.shape or not np.array_equal(a, b)
    ]
    assert not mismatched, (
        f"{arm}: {len(mismatched)} of {len(before)} tensors changed across the "
        f"round trip"
    )


@pytest.mark.parametrize("arm", ["conditional", "joint"])
def test_the_forward_pass_is_bit_identical_after_reload(arm, request):
    result = request.getfixturevalue(arm)
    delta = float(np.max(np.abs(result["before"] - result["after"])))
    assert delta == 0.0, (
        f"{arm}: forward output moved by {delta:.6e}; pre-fix this read "
        f"9.118214e-03 (conditional) / 1.737189e-02 (joint)"
    )


class TestComputeOutputShape:
    """The second half of the row: three `compute_output_shape` returning None."""

    SINGLE = (None, SEQ, DATA_DIM)
    LISTED = [(None, SEQ, DATA_DIM), (None, SEQ, COND_DIM), (None,)]

    @pytest.mark.parametrize("factory", [
        lambda: ConditionalDenoiser(data_dim=DATA_DIM, condition_dim=COND_DIM,
                                    hidden_dim=16, num_layers=1,
                                    num_attention_heads=2),
        lambda: VisionDenoiser(vision_config={"embed_dim": DATA_DIM},
                               text_dim=COND_DIM, num_layers=1),
        lambda: TextDenoiser(text_dim=DATA_DIM, vision_dim=COND_DIM,
                             num_layers=1),
    ], ids=["ConditionalDenoiser", "VisionDenoiser", "TextDenoiser"])
    @pytest.mark.parametrize("spelling", ["single", "listed"])
    def test_it_returns_the_data_shape(self, factory, spelling):
        """MEASURED pre-fix: the `single` spelling returned None."""
        layer = factory()
        shape = self.SINGLE if spelling == "single" else self.LISTED
        assert tuple(layer.compute_output_shape(shape)) == self.SINGLE, (
            f"compute_output_shape({shape}) did not return {self.SINGLE}; the "
            f"pre-fix code tested `isinstance(input_shape, (list, tuple))`, "
            f"which is TRUE for a plain shape tuple, and so returned the batch "
            f"dimension -- None."
        )

    def test_the_two_already_correct_sites_are_untouched(self):
        """Named explicitly so a future sweep does not "fix" them too."""
        assert TimestepEmbedding(embedding_dim=DATA_DIM).compute_output_shape(
            (None,)) == (None, DATA_DIM)
        joint = JointDenoiser(vision_dim=DATA_DIM, text_dim=COND_DIM,
                              hidden_dim=16, num_layers=1)
        assert joint.compute_output_shape(self.LISTED) == (
            self.LISTED[0], self.LISTED[1])
