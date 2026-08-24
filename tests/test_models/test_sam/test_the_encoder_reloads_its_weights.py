"""F-19: a STANDALONE ``ImageEncoderViT`` reloaded 1 of its 65 weights, silently.

``ImageEncoderViT.build`` created ``pos_embed`` and nothing else, on the recorded
reasoning that "sub-layers are built automatically on the first call". That is true of
``__call__`` and false of ``keras.models.load_model``, which builds from the saved
``input_shape`` and restores immediately -- so the restore found a model owning one
weight, and the patch embedding, all four blocks and the neck came back at class
defaults.

Three instruments, and only one of them can see it. This is the point of the module:

* **Archive content** -- extract ``model.weights.h5`` from the ``.keras`` zip and count
  HDF5 datasets. Reads **65 both before and after the fix**: the SAVE side was never
  broken. A save-side check alone is structurally blind to a LOAD-side loss.
* **Post-forward weight count** -- ``len(reloaded.weights)`` after anything has called
  the model. Reads **65 both before and after**: the lazy sub-layers materialize on
  that very forward pass, which is why F-19 was filed NEEDS-PROBE with the sequencing
  called out. Asserted below as an explicit BLINDNESS CONTROL, not as evidence.
* **Pre-forward weight count** and **perturb-save-reload-compare** -- the two that
  actually discriminate. MEASURED pre-fix on CPU at the fixture size below:
  ``len(reloaded.weights) == 1`` before any forward, and ``max|dOut| = 4.628568e+00``
  with 64/65 weights at class defaults. Post-fix: 65 and ``0.000000e+00``, 0/65.

RED-proof: delete the three sub-layer ``build`` calls from ``ImageEncoderViT.build``
and ``test_every_weight_exists_before_the_first_forward`` and
``test_the_reloaded_encoder_reproduces_perturbed_weights`` both fail, while
``test_a_post_forward_count_cannot_see_the_gap`` and
``test_the_archive_itself_was_never_the_problem`` stay green -- which is the finding.

Device: all assertions are exact-equality on shapes / bit-identical reloads and were
taken on CPU (``CUDA_VISIBLE_DEVICES=""``); the perturbation is a deterministic
``+0.137`` on every weight, so no seed-sensitive quantity is compared.
"""

import os
import tempfile
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.SAM.SAM1.image_encoder import ImageEncoderViT

IMG_SIZE = 64
PATCH = 16
EMBED_DIM = 32
DEPTH = 4
OUT_CHANS = 16
#: 1 `pos_embed` + patch embed + 4 blocks + neck, measured.
N_WEIGHTS = 65
#: Deterministic, non-zero, and not a plausible initializer value.
DELTA = 0.137


def _encoder():
    keras.utils.set_random_seed(1234)
    return ImageEncoderViT(
        img_size=IMG_SIZE, patch_size=PATCH, in_chans=3, embed_dim=EMBED_DIM,
        depth=DEPTH, num_heads=4, out_chans=OUT_CHANS,
        # window_size=0 makes every block global, so an empty
        # `global_attn_indexes` is the CORRECT pairing (D-014), not degenerate.
        window_size=0, global_attn_indexes=(),
    )


def _x():
    return np.random.RandomState(0).randn(2, IMG_SIZE, IMG_SIZE, 3).astype("float32")


@pytest.fixture(scope="module")
def perturbed_round_trip(tmp_path_factory):
    """Build, perturb every weight, save, reload. Returns the pieces each arm needs."""
    model = _encoder()
    x = _x()
    model(x)
    assert len(model.weights) == N_WEIGHTS, len(model.weights)

    for w in model.weights:
        w.assign(w + DELTA)
    reference_out = keras.ops.convert_to_numpy(model(x))
    reference_w = {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}

    path = str(tmp_path_factory.mktemp("f19") / "encoder.keras")
    model.save(path)

    reloaded = keras.models.load_model(path)
    n_pre_forward = len(reloaded.weights)          # sampled BEFORE any forward
    reloaded_out = keras.ops.convert_to_numpy(reloaded(x))
    n_post_forward = len(reloaded.weights)

    return {
        "path": path, "x": x,
        "reference_out": reference_out, "reference_w": reference_w,
        "reloaded": reloaded, "reloaded_out": reloaded_out,
        "n_pre_forward": n_pre_forward, "n_post_forward": n_post_forward,
    }


class TestTheEncoderReloadsEveryWeight:
    """The two discriminating arms."""

    def test_every_weight_exists_before_the_first_forward(self, perturbed_round_trip):
        """Sampled at LOAD time. Pre-fix this read 1; the sequencing is the test."""
        n = perturbed_round_trip["n_pre_forward"]
        assert n == N_WEIGHTS, (
            f"the reloaded encoder owned {n} of {N_WEIGHTS} weights BEFORE any "
            f"forward pass. `build` must materialize patch_embed, every block and "
            f"the neck; lazy sub-layers do not exist when load_model restores."
        )

    def test_the_reloaded_encoder_reproduces_perturbed_weights(self, perturbed_round_trip):
        """Perturbation is what makes class defaults distinguishable at all."""
        r = perturbed_round_trip
        mismatched = [
            (w.path, float(np.max(np.abs(
                r["reference_w"][w.path] - keras.ops.convert_to_numpy(w)))))
            for w in r["reloaded"].weights
            if w.path in r["reference_w"]
            and float(np.max(np.abs(
                r["reference_w"][w.path] - keras.ops.convert_to_numpy(w)))) != 0.0
        ]
        assert not mismatched, (
            f"{len(mismatched)} of {N_WEIGHTS} weights came back at values other "
            f"than the perturbed ones (first: {mismatched[:3]})"
        )

        d = float(np.max(np.abs(r["reference_out"] - r["reloaded_out"])))
        assert d == 0.0, (
            f"output moved across the round-trip: max|dOut| = {d:.6e} "
            f"(pre-fix this read 4.628568e+00)"
        )


class TestTheInstrumentsThatCannotSeeIt:
    """Blindness controls. These pass identically with and without the defect."""

    def test_a_post_forward_count_cannot_see_the_gap(self, perturbed_round_trip):
        """Documents WHY F-19 was NEEDS-PROBE: the count is right once anything runs."""
        r = perturbed_round_trip
        assert r["n_post_forward"] == N_WEIGHTS
        assert r["n_post_forward"] >= r["n_pre_forward"], (
            "a forward pass can only ADD lazily built weights; if this ever "
            "inverts, the probe's sequencing assumption is wrong"
        )

    def test_the_archive_itself_was_never_the_problem(self, perturbed_round_trip):
        """`model.weights.h5` held all 65 datasets both before and after the fix."""
        n = 0
        with zipfile.ZipFile(perturbed_round_trip["path"]) as z:
            assert "model.weights.h5" in z.namelist()
            with tempfile.TemporaryDirectory() as td:
                z.extract("model.weights.h5", td)
                with h5py.File(os.path.join(td, "model.weights.h5"), "r") as f:
                    def visit(_name, obj):
                        nonlocal n
                        if isinstance(obj, h5py.Dataset):
                            n += 1
                    f.visititems(visit)
        assert n == N_WEIGHTS, (
            f"the archive holds {n} weight datasets, not {N_WEIGHTS}; that would "
            f"be a SAVE-side defect, which F-19 was not"
        )
