"""`NanoVLM.generate`'s top-k sampling draws from `softmax(logits)`.

`keras.random.categorical` takes LOGITS. Until plan-2026-10-14T062616-81938e81
step 9 the sampler passed it `softmax(top_k_logits)`, so it drew from
`softmax(softmax(z))`: a flattened distribution in which a token holding
99.995% of the mass was drawn about 73% of the time.
"""

import keras
import numpy as np

from dl_techniques.models.vision_language.nano_vlm import NanoVLM

DRAWS = 4000


def _sampler_model() -> NanoVLM:
    return NanoVLM.from_variant(
        "mini", vocab_size=8,
        vision_config={"img_size": 32, "embed_dim": 16, "depth": 1, "num_heads": 2},
        text_config={"embed_dim": 16, "depth": 1, "num_heads": 2, "max_seq_len": 16},
        fusion_config={"attention_config": {"num_heads": 2}, "num_fusion_layers": 1},
    )


def test_top_k_sampling_follows_the_softmax_of_the_logits():
    keras.utils.set_random_seed(0)
    model = _sampler_model()
    # Token 3 holds softmax([10, 0])[0] = 0.99995 of the top-2 mass; token 5
    # the rest. Every other token is outside the top 2.
    row = np.full((8,), -50.0, dtype="float32")
    row[3], row[5] = 10.0, 0.0
    logits = np.tile(row, (DRAWS, 1))

    drawn = keras.ops.convert_to_numpy(
        model._sample_tokens_batch(logits, temperature=1.0, top_k=2))

    assert set(np.unique(drawn)) <= {3, 5}
    share = float(np.mean(drawn == 3))
    # Expected 0.99995: at most a handful of 5s in 4000 draws. The old
    # softmax-of-softmax sampler gives 1 / (1 + e**-0.9999) = 0.731, which
    # this bound excludes by more than 50 binomial standard deviations.
    assert share > 0.99, share
