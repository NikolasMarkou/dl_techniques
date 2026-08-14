"""Coverage for ``ScoreBasedNanoVLM.generate_from_text`` (nano_vlm_world_model).

This entry point had ZERO tests before this module: ``test_smoke.py`` exercises only
``call()`` and ``test_round_trip.py`` calls the denoisers directly. Two defects lived
behind that gap, and they were ordered — the first masked the second. This module pins
the first: the vision-feature shape probe hardcoded a ``(1, 224, 224, 3)`` dummy image
regardless of the model's configured ``img_size``, so at any other size the call died
inside ``PositionalEmbedding.call`` before reaching the generation loop.

Everything here runs at a NON-224 ``img_size`` (32), which is the configuration that
makes that defect observable at all.
"""

import numpy as np
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM

IMG_SIZE = 32
EMBED_DIM = 64


def _tiny_model(prediction_type=None):
    """A ``text_to_image`` model at img_size=32 — deliberately NOT the hardcoded 224."""
    vision_config = {
        'img_size': IMG_SIZE, 'patch_size': 16, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'output_mode': 'none',
    }
    text_config = {
        'vocab_size': 64, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'max_seq_len': 32,
    }
    diffusion_config = {'num_timesteps': 100, 'beta_schedule': 'cosine'}
    if prediction_type is not None:
        diffusion_config['prediction_type'] = prediction_type
    return ScoreBasedNanoVLM(
        vision_config=vision_config,
        text_config=text_config,
        diffusion_config=diffusion_config,
        vocab_size=64,
        generation_mode='text_to_image',
        use_classifier_free_guidance=False,
    )


def _text_features(batch=2, seq_len=8):
    return ops.convert_to_tensor(
        np.random.rand(batch, seq_len, EMBED_DIM).astype('float32')
    )


class TestGenerateFromTextShapeProbe:
    """RED-proof for the hardcoded 224x224 probe (plan step 3)."""

    def test_runs_at_non_224_img_size(self):
        """The probe must follow the CONFIGURED img_size, not the literal 224.

        Pre-fix this raised ``InvalidArgumentError`` out of
        ``layers/embedding/positional_embedding.py:239``
        (``Expected size[1] in [0, 5], but got 197``) — the 224-image's 196+1 tokens
        being sliced against a 4+1-token positional table.
        """
        model = _tiny_model()
        out = model.generate_from_text(_text_features(), num_inference_steps=2)

        # img_size 32 / patch 16 -> 2x2 = 4 patches, + CLS token = 5.
        assert tuple(ops.shape(out)) == (2, 5, EMBED_DIM)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))
