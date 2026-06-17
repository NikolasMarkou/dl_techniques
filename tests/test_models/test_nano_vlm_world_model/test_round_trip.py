"""M2 .keras round-trip test for ScoreBasedNanoVLM (nano_vlm_world_model).

This model's top-level forward is STOCHASTIC (internal ``keras.random.normal``
noise), so a naive output-identity test is impossible. Historically its denoisers
dropped ~600 weights on a ``.keras`` reload because the nested ``MultiHeadAttention``
and ``Sequential`` sub-layers were built lazily on first call (unbuilt at
weight-restore time). The fix gives every denoiser an explicit ``build()`` and has
the model build them in its own ``build()``.

We verify the round-trip robustly:

1. Weight count is preserved (no weights dropped / re-created on reload).
2. Every weighted sub-component (vision encoder, text encoder, and each denoiser
   present for the mode) produces NUMERICALLY IDENTICAL output for fixed inputs at
   ``training=False`` after reload — the authoritative proof that the restored
   weights are wired correctly. (Positional/name weight matching is unreliable here
   because build-vs-forward weight ordering and sub-layer auto-names differ across
   instances, even though all values restore correctly.)
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM


def _tiny_model(mode="joint"):
    vision_config = {
        'img_size': 32, 'patch_size': 16, 'embed_dim': 64,
        'depth': 2, 'num_heads': 4, 'output_mode': 'none',
    }
    text_config = {
        'vocab_size': 64, 'embed_dim': 64,
        'depth': 2, 'num_heads': 4, 'max_seq_len': 32,
    }
    diffusion_config = {'num_timesteps': 100, 'beta_schedule': 'cosine'}
    return ScoreBasedNanoVLM(
        vision_config=vision_config,
        text_config=text_config,
        diffusion_config=diffusion_config,
        vocab_size=64,
        generation_mode=mode,
    )


def _inputs():
    return {
        'images': np.random.rand(2, 32, 32, 3).astype('float32'),
        'text': np.random.randint(0, 64, size=(2, 16)).astype('int32'),
    }


def _close(a, b):
    np.testing.assert_allclose(
        ops.convert_to_numpy(a), ops.convert_to_numpy(b), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("mode", ["text_to_image", "image_to_text", "joint"])
def test_keras_round_trip(mode):
    model = _tiny_model(mode)
    inputs = _inputs()
    _ = model(inputs, training=False)  # build via forward
    n_weights = len(model.weights)
    assert n_weights > 0

    # Fixed deterministic probes (no internal randomness at training=False).
    images = inputs['images']
    nv = np.random.rand(2, 4, 64).astype('float32')   # noisy vision (4 patches)
    nt = np.random.rand(2, 16, 64).astype('float32')  # noisy text
    ts = np.array([3, 7], dtype='int32')

    ve0 = model.vision_encoder(images, training=False)
    te0 = model.text_encoder({'input_ids': inputs['text']}, training=False)
    den0 = {}
    if mode in ('text_to_image', 'joint'):
        den0['vision'] = model.vision_denoiser(nv, te0, ts, training=False)
    if mode in ('image_to_text', 'joint'):
        den0['text'] = model.text_denoiser(nt, ve0, ts, training=False)
    if mode == 'joint':
        den0['joint'] = model.joint_denoiser(nv, nt, ts, training=False)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "swm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        _ = loaded(inputs, training=False)

    # (1) No weights dropped or spuriously added.
    assert len(loaded.weights) == n_weights, (
        f"weight count changed on reload: {n_weights} -> {len(loaded.weights)}"
    )

    # (2) Deterministic component forwards are bit-identical after reload.
    _close(ve0, loaded.vision_encoder(images, training=False))
    _close(te0, loaded.text_encoder({'input_ids': inputs['text']}, training=False))
    if mode in ('text_to_image', 'joint'):
        _close(den0['vision'], loaded.vision_denoiser(nv, te0, ts, training=False))
    if mode in ('image_to_text', 'joint'):
        _close(den0['text'], loaded.text_denoiser(nt, ve0, ts, training=False))
    if mode == 'joint':
        jv0, jt0 = den0['joint']
        jv1, jt1 = loaded.joint_denoiser(nv, nt, ts, training=False)
        _close(jv0, jv1)
        _close(jt0, jt1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
