"""The saved `.keras` archive holds each DepthAnything weight exactly once.

DepthAnything used to override `save_own_variables` to dump `list(self.weights)`
under flat `vars/N` keys (`plan_2026-05-10_bd098beb/D-004`), on the premise that
doing so "bypasses Keras' path-walking for these sub-Models entirely". It does
not: Keras' own recursive attribute-tracked save runs anyway, so every archive
carried BOTH families for the same tensors — measured 2026-08-22 at exactly
2.00x (`vit_l`/384: 610 weights, 1220 HDF5 datasets, 4,882,763,410 bytes), while
`load_own_variables` read only the flat half back. The duplicate was write-only.

The half that was load-bearing is the force-build in `load_own_variables`, which
materializes the sub-Models before Keras restores into them. The second test here
is that half's "something changed" twin: with the force-build removed the round
trip stops being exact, which is the 2026-05-10 defect returning.

A single round trip is NOT enough to see this defect class, so both tests do two
(`save -> load -> save -> load`): `load_model` rebuilds from the saved
`input_shape` and restores immediately, so the first reload can be exact even
when the archive layout is wrong.
"""

import re
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.depth_anything.model import (
    DepthAnything,
    create_depth_anything,
)

#: Smallest geometry that still exercises the real ViT encoder + DPT decoder.
ENCODER_TYPE = "vit_s"
IMAGE_SHAPE = (64, 64, 3)

#: Flat `vars/0`, `vars/1`, ... keys — the shape of the removed duplicate dump.
_FLAT_KEY = re.compile(r"^vars/\d+$")


def _weight_dataset_names(keras_path, extract_dir):
    """Every HDF5 dataset name inside a saved `.keras` archive's weight file."""
    h5_path = extract_dir / "model.weights.h5"
    with zipfile.ZipFile(keras_path) as archive:
        with archive.open("model.weights.h5") as src, open(h5_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                dst.write(chunk)
    names = []
    with h5py.File(h5_path, "r") as handle:
        handle.visititems(
            lambda name, obj: names.append(name)
            if isinstance(obj, h5py.Dataset)
            else None
        )
    h5_path.unlink()
    return names


def _seeded_model():
    keras.utils.set_random_seed(42)
    return create_depth_anything(
        encoder_type=ENCODER_TYPE,
        image_shape=IMAGE_SHAPE,
        use_feature_alignment=False,
    )


def test_every_weight_is_written_exactly_once(tmp_path):
    """Dataset count == `len(model.weights)`, and no flat `vars/N` family."""
    model = _seeded_model()
    x = keras.random.normal((1,) + IMAGE_SHAPE, seed=1)
    reference = keras.ops.convert_to_numpy(model(x, training=False))
    n_weights = len(model.weights)

    first = tmp_path / "rt1.keras"
    model.save(first)
    names = _weight_dataset_names(first, tmp_path)

    flat = sorted(name for name in names if _FLAT_KEY.match(name))
    assert not flat, (
        f"{len(flat)} flat `vars/N` datasets are back in the archive "
        f"(e.g. {flat[:3]}). A `save_own_variables` that dumps "
        f"`self.weights` does not replace Keras' recursive save, it "
        f"duplicates it — see D-009 in this model's source."
    )

    groups = {}
    for name in names:
        groups[name.split("/")[0]] = groups.get(name.split("/")[0], 0) + 1
    assert len(names) == n_weights, (
        f"archive holds {len(names)} weight datasets for a model with "
        f"{n_weights} weights (ratio {len(names) / n_weights:.4f}); "
        f"top-level groups: {groups}"
    )

    # Second round trip: the save-side layout is only half the contract.
    reloaded = keras.models.load_model(first)
    first_delta = float(
        np.max(
            np.abs(
                reference - keras.ops.convert_to_numpy(reloaded(x, training=False))
            )
        )
    )
    assert first_delta == 0.0, f"first round trip moved the output by {first_delta}"

    second = tmp_path / "rt2.keras"
    reloaded.save(second)
    assert len(_weight_dataset_names(second, tmp_path)) == n_weights

    twice = keras.models.load_model(second)
    second_delta = float(
        np.max(
            np.abs(reference - keras.ops.convert_to_numpy(twice(x, training=False)))
        )
    )
    assert second_delta == 0.0, f"second round trip moved the output by {second_delta}"


def test_removing_the_force_build_breaks_the_round_trip(tmp_path, monkeypatch):
    """The "something changed" twin: `load_own_variables` is load-bearing.

    Without it, Keras restores into sub-Models whose sub-layers do not exist yet
    and the weights come back re-initialised. If this test ever goes GREEN, the
    framework has stopped needing the hook and the override may be deleted —
    that is the only condition under which deleting it is safe.
    """
    monkeypatch.delattr(DepthAnything, "load_own_variables")

    model = _seeded_model()
    x = keras.random.normal((1,) + IMAGE_SHAPE, seed=1)
    reference = keras.ops.convert_to_numpy(model(x, training=False))

    path = tmp_path / "no_force_build.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    delta = float(
        np.max(
            np.abs(
                reference - keras.ops.convert_to_numpy(reloaded(x, training=False))
            )
        )
    )
    assert delta > 1e-3, (
        "the path-based restore round-tripped exactly WITHOUT the force-build "
        f"in `DepthAnything.load_own_variables` (max|delta| = {delta}). Keras "
        "may have started materializing nested sub-Models on its own; re-check "
        "D-009 before relying on the override."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
