"""The two VQ-VAE READMEs state what their decoders do, and the code still does it.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 13.4 (D-251). Both READMEs
carry a GPU-determinism note for decoders built from ``Conv2DTranspose`` (D-024, D-083:
after another transposed conv warms cuDNN's autotuner, a non-deterministic engine can be
picked). The note is only true while the decoder it describes has transposed convs, so
both halves are checked: the claim in the text and the layers in the code. The rotation
model's class docstring said its auto-built codec used swish; the code builds ReLU and a
linear output (a sigmoid one until plan-2026-11-25T124419-fa94f59c, D-022: a head confined
to [0, 1] cannot reach a standardized target), which the docstring and README now say and
this file checks on a built model. A file written before keeps its own sigmoid decoder.
"""

from pathlib import Path

import keras
import numpy as np

from dl_techniques.models.vision.vq_vae_rotation import (
    VQVAERotationTrick,
    create_vq_vae_rotation,
)

PACKAGES = Path(__file__).resolve().parents[2] / "src/dl_techniques/models/vision"
NOTE = ("UpSampling2D", "TF_CUDNN_DETERMINISTIC=1", "enable_op_determinism",
        "ConvBwdData_eng25", "D-083")
# The best output a head confined to [0, 1] can give scores this on the standardized CIFAR-10
# selection split of `python -m train` (5,000 rows, split seed 42): plan-2026-11-25T124419-fa94f59c, D-023.
FLOOR = "0.5339"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _layers(model: keras.Model):
    return list(model._flatten_layers(include_self=False, recursive=True))


def test_the_vq_vae_readme_note_matches_its_example_decoder():
    text = (PACKAGES / "vq_vae/README.md").read_text()
    assert "keras.layers.Conv2DTranspose(" in text
    for phrase in NOTE:
        assert phrase in _flat(text), phrase


def test_the_rotation_readme_note_matches_the_auto_built_decoder():
    model = create_vq_vae_rotation(
        input_shape=(16, 16, 3), num_embeddings=8, embedding_dim=4,
        hidden_channels=8, num_res_blocks=1)
    decoder = _layers(model.decoder)
    assert any(isinstance(layer, keras.layers.Conv2DTranspose) for layer in decoder)
    text = (PACKAGES / "vq_vae_rotation/README.md").read_text()
    for phrase in NOTE:
        assert phrase in _flat(text), phrase


def test_the_rotation_docs_name_the_activations_the_codec_builds():
    model = create_vq_vae_rotation(
        input_shape=(16, 16, 3), num_embeddings=8, embedding_dim=4,
        hidden_channels=8, num_res_blocks=1)
    activations = {
        layer.get_config()["activation"]
        for part in (model.encoder, model.decoder)
        for layer in _layers(part) if isinstance(layer, keras.layers.Activation)
    }
    assert activations == {"relu"}, activations
    output = model.decoder.layers[-1]
    assert output.get_config()["activation"] == "linear"
    for text in (VQVAERotationTrick.__doc__,
                 (PACKAGES / "vq_vae_rotation/README.md").read_text()):
        text = _flat(text)
        assert "swish" not in text.lower()
        assert "+ sigmoid" not in text and "with a sigmoid" not in text
        for phrase in ("ReLU", "linear", "standardized", FLOOR):
            assert phrase in text, phrase


def test_an_archive_written_with_the_sigmoid_decoder_still_loads_with_it(tmp_path):
    model = create_vq_vae_rotation(
        input_shape=(16, 16, 3), num_embeddings=8, embedding_dim=4,
        hidden_channels=8, num_res_blocks=1)
    head = model.decoder.layers[-1]
    assert head.name == "dec_to_output" and head.get_config()["activation"] == "linear"
    head.activation = keras.activations.sigmoid  # the model a file written before the change holds
    x = np.random.default_rng(0).normal(size=(2, 16, 16, 3)).astype("float32")
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    assert 0.0 < expected.min() and expected.max() < 1.0, "the control: this decoder is the sigmoid one"
    model.save(tmp_path / "old.keras")
    loaded = keras.models.load_model(tmp_path / "old.keras")
    assert loaded.decoder.layers[-1].get_config()["activation"] == "sigmoid"
    got = keras.ops.convert_to_numpy(loaded(x, training=False))
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
