"""``NanoVLM.generate`` finishes each sequence on its own EOS.

Until plan-2026-10-14T062616-81938e81 step 11.3 (F-9 W6, D-238) the loop ended
with ``if eos_token_id in next_tokens: break``: the first EOS in ANY row
stopped the whole batch, so a row still mid-sentence was cut off (the F-9
probe: batch 2, row 0 emits EOS at step 0, ``generate`` returned shape
``(2, prompt + 1)``). Now a finished row keeps emitting ``eos_token_id`` and
the loop ends when every row has finished, or after ``max_length`` steps.

The sampler is replaced by a script (the model's logits are irrelevant to the
claim); the decoder still runs every step.
"""

import numpy as np
import pytest

from dl_techniques.models.vision_language.nano_vlm import NanoVLM

EOS = 2
PROMPT = np.array([[5, 6, 7], [7, 6, 5]], dtype="int32")


def _model() -> NanoVLM:
    return NanoVLM.from_variant(
        "mini", vocab_size=16,
        vision_config={"img_size": 32, "embed_dim": 16, "depth": 1, "num_heads": 2},
        text_config={"embed_dim": 16, "depth": 1, "num_heads": 2, "max_seq_len": 16},
    )


def _scripted(model, monkeypatch, rows):
    """Step ``t`` samples ``rows[t]``; counts the decoder steps taken."""
    steps = []

    def sample(logits, temperature, top_k):
        steps.append(len(steps))
        return np.asarray(rows[len(steps) - 1], dtype="int64")

    monkeypatch.setattr(model, "_sample_tokens_batch", sample)
    return steps


def _images():
    return np.random.default_rng(0).uniform(-1, 1, (2, 32, 32, 3)).astype("float32")


def test_an_early_eos_in_one_row_does_not_end_the_other(monkeypatch):
    model = _model()
    script = [[EOS, 9], [10, 11], [12, 13], [14, EOS], [15, 15]]
    steps = _scripted(model, monkeypatch, script)
    out = np.asarray(model.generate(_images(), PROMPT, max_length=8, eos_token_id=EOS))
    # Row 1 runs 4 tokens to its own EOS; row 0 is EOS from step 0 on.
    assert out.shape == (2, 3 + 4)
    np.testing.assert_array_equal(out[0], [5, 6, 7, EOS, EOS, EOS, EOS])
    np.testing.assert_array_equal(out[1], [7, 6, 5, 9, 11, 13, EOS])
    assert len(steps) == 4, "the loop must stop once every row has finished"


def test_without_eos_the_loop_runs_max_length(monkeypatch):
    model = _model()
    steps = _scripted(model, monkeypatch, [[9, 10]] * 5)
    out = np.asarray(model.generate(_images(), PROMPT, max_length=5, eos_token_id=EOS))
    assert out.shape == (2, 3 + 5) and len(steps) == 5
    assert not (out[:, 3:] == EOS).any()


def test_every_row_ending_together_stops_at_once(monkeypatch):
    model = _model()
    steps = _scripted(model, monkeypatch, [[EOS, EOS], [9, 9]])
    out = np.asarray(model.generate(_images(), PROMPT, max_length=6, eos_token_id=EOS))
    assert out.shape == (2, 4) and len(steps) == 1


@pytest.mark.parametrize("batch", [1, 3])
def test_any_batch_size(monkeypatch, batch):
    model = _model()
    rows = [[EOS] + [9] * (batch - 1)] + [[9] * batch] * 2 + [[EOS] * batch]
    _scripted(model, monkeypatch, rows)
    prompt = np.full((batch, 3), 5, dtype="int32")
    images = np.zeros((batch, 32, 32, 3), "float32")
    out = np.asarray(model.generate(images, prompt, max_length=6, eos_token_id=EOS))
    # One row finishes at step 0; with others, they finish at step 3.
    assert out.shape == (batch, 3 + (1 if batch == 1 else 4))
    assert (out[0, 3:] == EOS).all()
