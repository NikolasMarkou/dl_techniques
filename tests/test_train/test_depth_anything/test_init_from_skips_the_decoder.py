"""``--init-from`` loads a DepthAnything checkpoint without its decoder.

The CLI help says the checkpoint's "non-decoder weights" are loaded and the
README says the head is freshly initialized. The trainer skipped the prefix
``"dpt_decoder"``, a name no DepthAnything layer carries since the decoder
became the ``'depth_estimation_head'`` vision head, so the decoder was
transferred too (MEASURED at c39e444a3: ``skipped_by_prefix`` empty, every
decoder weight equal to the checkpoint's). Plan-2026-10-14T062616-81938e81
step 7.1, decision D-031.
"""

import keras
import numpy as np

from dl_techniques.models.vision.depth_anything.model import create_depth_anything
from train.depth_anything.train_depth_anything import init_from_checkpoint


def _model(seed: int):
    keras.utils.set_random_seed(seed)
    return create_depth_anything(
        encoder_type="vit_s", encoder_kind="placeholder",
        image_shape=(32, 32, 3), decoder_dims=[8, 8, 8, 8],
        use_feature_alignment=False)


def test_the_encoder_loads_and_the_decoder_keeps_its_init(tmp_path) -> None:
    source = _model(0)
    path = str(tmp_path / "source.keras")
    source.save(path)
    target = _model(1)
    fresh_decoder = [w.copy() for w in target.decoder.get_weights()]

    report = init_from_checkpoint(target, path)

    assert report.skipped_by_prefix == [target.decoder.name]
    for got, want in zip(target.encoder.get_weights(),
                         source.encoder.get_weights()):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(target.decoder.get_weights(), fresh_decoder):
        np.testing.assert_array_equal(got, want)
    assert any(not np.array_equal(a, b) for a, b in zip(
        source.decoder.get_weights(), fresh_decoder)), (
        "degenerate probe: the two decoders started equal")
