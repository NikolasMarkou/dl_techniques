"""``TextEncoder`` pools every position a 3D attention mask lets some query see.

Guard for plan-2026-10-14T062616-81938e81 step 5.6 (D-205), found while
running the SAM 3 text tower's tests. ``TextEncoder`` turned a 3D
``(batch, query, key)`` mask into the pooling keep-mask by reading its FIRST
ROW. For a causal mask that row keeps position 0 alone. Measured at 7841170b1:
``TextEncoder(output_mode='mean')(ids, attention_mask=causal)`` returned exactly
``x[:, 0]`` (every reducing strategy pooled token 0 only), and after step 5.4's
zeroing ``'none'`` returned zeros everywhere but position 0. The keep-mask is now
"some query may attend to this key". A 3D mask with ``use_cls_token=True``
crashed in a concatenate (a 2D CLS column against a 3D mask); it now raises a
``ValueError`` saying so.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.utils.masking import create_causal_attend_mask

IDS = np.array([[5, 7, 9, 4], [3, 4, 6, 2]], "int32")


def _encoder(output_mode, use_cls_token=False):
    keras.utils.set_random_seed(0)
    return TextEncoder(
        vocab_size=32, embed_dim=8, depth=1, num_heads=2, max_seq_len=8,
        output_mode=output_mode, use_cls_token=use_cls_token,
    )


def _sequence(mask):
    encoder = _encoder("none")
    return np.asarray(encoder(IDS, attention_mask=mask))


def test_a_causal_mask_keeps_every_position_for_pooling():
    causal = create_causal_attend_mask(IDS)
    sequence = _sequence(causal)
    assert np.abs(sequence[:, 1:]).min() > 0.0, "'none' zeroed attended positions"
    mean = np.asarray(_encoder("mean")(IDS, attention_mask=causal))
    np.testing.assert_allclose(mean, sequence.mean(axis=1), rtol=1e-5, atol=1e-6)
    assert np.abs(mean - sequence[:, 0]).max() > 1e-3


def test_a_3d_padding_mask_keeps_exactly_the_real_tokens():
    keys = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], "int32")
    mask3d = np.repeat(keys[:, None, :], 4, axis=1)
    sequence = _sequence(mask3d)
    np.testing.assert_array_equal(sequence[keys == 0], 0.0)
    assert np.abs(sequence[keys == 1]).min() > 0.0


def test_a_3d_mask_with_a_cls_token_is_refused_by_name():
    with pytest.raises(ValueError, match="use_cls_token"):
        _encoder("mean", use_cls_token=True)(
            IDS, attention_mask=create_causal_attend_mask(IDS)
        )
