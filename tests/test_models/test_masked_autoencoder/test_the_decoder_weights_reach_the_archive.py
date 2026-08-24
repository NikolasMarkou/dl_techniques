"""The decoder's weights must actually be written to the `.keras` archive.

Rationale
---------
`ConvDecoder` used to store its sub-layers as ``self.decoder_blocks``, a list of
dicts of Layers -- a layer container nested two levels deep, owned by a
``keras.layers.Layer``. Keras 3.8 does NOT write such a container to
``model.weights.h5``. MEASURED on ``MaskedAutoencoder`` before the fix: **11 of
51 tensors and 98,403 of 329,827 parameters in the archive** -- every decoder
convolution kernel and all 32 BatchNorm tensors silently absent -- and only
**27 of 51** tensors surviving a perturb / save / reload comparison. The same
container owned by a ``keras.Model`` is written correctly, which is why the
identical shape in ``models/accunet`` and ``models/cliffordnet`` is harmless.

Why two arms and not one
------------------------
A save-side archive-content check alone cannot see a LOAD-side loss (measured
elsewhere in this repo on ``SHGCNLinkPredictor``: archive complete, reload
lossy). A forward comparison alone cannot see this defect either -- at
``training=False`` the MAE performs no masking and the whole reconstruction
lives inside a ~2e-3 band, so the pre-fix delta (2.028994e-03) is the same order
as the output's own range and a loose ``atol`` passes against it. Both arms are
therefore required, and the forward arm is taken on the DECODER ALONE so it is
not diluted by the masking path.

Anti-vacuity
------------
Every weight is perturbed before saving, so no tensor can match by initializer
coincidence -- the mechanism that hid this defect from 43 of 51 tensors when it
was first measured.

See decisions.md D-026 (plan-2026-08-19T163559-499b6f0e).
"""

import io
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder import MaskedAutoencoder

from .conftest import tiny_encoder

# ---------------------------------------------------------------------


def _tiny_encoder() -> keras.Model:
    """A 4x stride-2 conv stack: 32x32 -> 2x2, matching `decoder_depth=4`.

    The widths RISE (16/32/64/128) rather than staying flat, so no two weight
    tensors share a shape and a mis-ordered restore cannot pass by coincidence.
    """
    return tiny_encoder(
        image_size=32, channels=3, filters=(16, 32, 64, 128), name="tiny_encoder"
    )


def _build_mae() -> MaskedAutoencoder:
    model = MaskedAutoencoder(
        encoder=_tiny_encoder(),
        patch_size=16,
        input_shape=(32, 32, 3),
    )
    model(np.zeros((2, 32, 32, 3), dtype="float32"), training=False)
    return model


def _perturb_every_weight(model: keras.Model, seed: int = 7) -> None:
    """Move every non-moving-statistic weight off its initializer value."""
    rng = np.random.RandomState(seed)
    for weight in model.weights:
        if "moving_" in weight.path:
            continue
        value = np.array(weight)
        sigma = max(0.25 * float(np.std(value)), 1e-3)
        weight.assign(value + rng.randn(*value.shape).astype(value.dtype) * sigma)


def _archive_content(path) -> tuple:
    """Return (dataset count, element count) of the archive's weights file."""
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("model.weights.h5")

    datasets = 0
    elements = 0

    def _visit(_name, obj):
        nonlocal datasets, elements
        if isinstance(obj, h5py.Dataset):
            datasets += 1
            elements += int(np.prod(obj.shape)) if obj.shape else 1

    with h5py.File(io.BytesIO(payload), "r") as handle:
        handle.visititems(_visit)
    return datasets, elements


# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory):
    """Perturb, save, reload -- returns everything both arms need."""
    keras.utils.set_random_seed(1234)
    model = _build_mae()
    _perturb_every_weight(model)

    inputs = np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")
    encoded = np.array(model.encoder(inputs, training=False))
    decoded_before = np.array(model.decoder(encoded, training=False))
    weights_before = [np.array(w) for w in model.weights]

    path = tmp_path_factory.mktemp("mae_archive") / "model.keras"
    model.save(path)
    datasets, elements = _archive_content(path)

    restored = keras.models.load_model(path)
    restored(inputs, training=False)

    return {
        "expected_tensors": len(model.weights),
        "expected_params": model.count_params(),
        "archive_datasets": datasets,
        "archive_elements": elements,
        "weights_before": weights_before,
        "weights_after": [np.array(w) for w in restored.weights],
        "decoded_before": decoded_before,
        "decoded_after": np.array(restored.decoder(encoded, training=False)),
    }


def test_the_archive_holds_every_tensor(round_trip):
    """SAVE-side arm: the archive must hold all 51 tensors, not 11."""
    assert round_trip["archive_datasets"] == round_trip["expected_tensors"], (
        f"`model.weights.h5` holds {round_trip['archive_datasets']} datasets "
        f"against {round_trip['expected_tensors']} weights. A layer container "
        f"nested >=2 deep owned by a `keras.layers.Layer` is not written; "
        f"keep `ConvDecoder`'s sub-layers in FLAT per-role lists."
    )


def test_the_archive_holds_every_parameter(round_trip):
    """SAVE-side arm: element count, so a shape change cannot hide a loss."""
    assert round_trip["archive_elements"] == round_trip["expected_params"], (
        f"archive holds {round_trip['archive_elements']} elements against "
        f"count_params() = {round_trip['expected_params']}"
    )


def test_every_weight_survives_the_round_trip(round_trip):
    """LOAD-side arm: a complete archive can still reload lossily."""
    before = round_trip["weights_before"]
    after = round_trip["weights_after"]
    assert len(before) == len(after)
    mismatched = [
        i for i, (a, b) in enumerate(zip(before, after))
        if a.shape != b.shape or not np.array_equal(a, b)
    ]
    assert not mismatched, (
        f"{len(mismatched)} of {len(before)} tensors changed across the round "
        f"trip (ordinal indices {mismatched[:8]}). Compared by ORDINAL, not by "
        f"`weight.path` -- a reloaded MAE reports unprefixed paths "
        f"(`conv2d/kernel`), so a path-keyed comparison matches almost nothing "
        f"and reads as a failure on both arms."
    )


def test_the_decoder_forward_is_bit_identical_after_reload(round_trip):
    """FORWARD arm, taken on the decoder alone so masking cannot dilute it."""
    before = round_trip["decoded_before"]
    after = round_trip["decoded_after"]
    delta = float(np.max(np.abs(before - after)))
    magnitude = float(np.max(np.abs(before)))
    assert delta == 0.0, (
        f"decoder output moved by {delta:.6e} across the round trip, against "
        f"its own dynamic range {magnitude:.6e}. Pre-fix this read "
        f"2.028994e-03 against a range of 2.626643e-03."
    )
