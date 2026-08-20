"""`SHGCNLinkPredictor` saves completely and used to reload LOSSILY.

Rationale
---------
This is the first case in this plan where an archive-content check is NECESSARY
but NOT SUFFICIENT. MEASURED before the fix, with both decoder scalars perturbed
to 3.75:

    archive datasets                   8 / 8      <-- COMPLETE
    decoder scalars stored             3.75, 3.75 <-- COMPLETE
    decoder scalars after reload       2.0, 1.0   <-- the CLASS DEFAULTS
    tensors identical after reload     6 / 8
    forward delta                      1.497385e-01  (output range 7.310586e-01)

and no warning of any kind. Keras restores a sub-layer's variables only if that
sub-layer is BUILT when the archive is read; `SHGCNLinkPredictor` had no
`build()`, so `FermiDiracDecoder` was unbuilt and its `load_own_variables` was
skipped.

The perturbation is to 3.75 deliberately: the defect returns r=2.0 / t=1.0, the
CLASS DEFAULTS, so any test that leaves the decoder at its initial values
compares defaults against defaults and passes against the defect.

See decisions.md D-029 (plan-2026-08-19T163559-499b6f0e).
"""

import io
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.models.shgcn.model import SHGCNLinkPredictor

# ---------------------------------------------------------------------

PERTURBED_SCALAR = 3.75
CLASS_DEFAULTS = {"threshold": 2.0, "temperature": 1.0}


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory):
    keras.utils.set_random_seed(1234)
    rs = np.random.RandomState(0)
    features = rs.randn(12, 8).astype("float32")
    adjacency = (rs.rand(12, 12) < 0.3).astype("float32")
    adjacency = np.maximum(adjacency, adjacency.T)
    edges = rs.randint(0, 12, size=(5, 2)).astype("int32")

    model = SHGCNLinkPredictor(hidden_dims=[16], embedding_dim=8)
    model([features, adjacency, edges], training=False)

    for weight in model.decoder.weights:
        weight.assign(np.array(PERTURBED_SCALAR, dtype=np.array(weight).dtype))

    before = np.array(model([features, adjacency, edges], training=False))
    weights_before = [np.array(w) for w in model.weights]

    path = tmp_path_factory.mktemp("shgcn_lp") / "model.keras"
    model.save(path)

    with zipfile.ZipFile(path) as archive:
        payload = archive.read("model.weights.h5")
    datasets = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    with h5py.File(io.BytesIO(payload), "r") as handle:
        handle.visititems(_visit)

    restored = keras.models.load_model(path)
    after = np.array(restored([features, adjacency, edges], training=False))

    return {
        "expected_tensors": len(model.weights),
        "archive_datasets": len(datasets),
        "weights_before": weights_before,
        "weights_after": [np.array(w) for w in restored.weights],
        "decoder_after": {
            w.path.split("/")[-1]: float(np.array(w))
            for w in restored.decoder.weights
        },
        "forward_before": before,
        "forward_after": after,
    }


def test_the_archive_is_complete(round_trip):
    """This arm ALREADY PASSED before the fix -- it is here to say so."""
    assert round_trip["archive_datasets"] == round_trip["expected_tensors"], (
        f"archive holds {round_trip['archive_datasets']} datasets against "
        f"{round_trip['expected_tensors']} weights"
    )


def test_the_decoder_scalars_are_not_the_class_defaults_after_reload(round_trip):
    """The isolating arm: the defect returns the class defaults, silently."""
    after = round_trip["decoder_after"]
    for name, default in CLASS_DEFAULTS.items():
        assert name in after, f"decoder weight {name!r} missing after reload"
        assert after[name] == pytest.approx(PERTURBED_SCALAR), (
            f"decoder {name} reloaded as {after[name]!r}; the saved value was "
            f"{PERTURBED_SCALAR}. Reading exactly {default} means "
            f"`FermiDiracDecoder` was unbuilt at load time and its "
            f"`load_own_variables` was skipped -- add/keep `build()` on "
            f"`SHGCNLinkPredictor`."
        )


def test_every_weight_survives_the_round_trip(round_trip):
    before, after = round_trip["weights_before"], round_trip["weights_after"]
    assert len(before) == len(after)
    mismatched = [
        i for i, (a, b) in enumerate(zip(before, after))
        if a.shape != b.shape or not np.array_equal(a, b)
    ]
    assert not mismatched, (
        f"{len(mismatched)} of {len(before)} tensors changed across the round "
        f"trip (ordinal indices {mismatched}); pre-fix this read 2 of 8"
    )


def test_the_forward_pass_is_bit_identical_after_reload(round_trip):
    delta = float(np.max(np.abs(
        round_trip["forward_before"] - round_trip["forward_after"])))
    assert delta == 0.0, (
        f"forward output moved by {delta:.6e} across the round trip against an "
        f"output range of "
        f"{float(np.max(np.abs(round_trip['forward_before']))):.6e}; pre-fix "
        f"this read 1.497385e-01"
    )
