"""What `generate()` and `compute_output_shape` may assume about the fusion layer.

Only the default strategy was ever tested, which is why two defects survived:

* ``generate()`` concatenated vision+text only when ``fused`` was a tuple, which
  happens for ``cross_attention`` alone. For every other strategy ``combined``
  was already at the single shared length, yet the code still sliced off a
  vision-length prefix — leaving an **empty** axis, which
  ``text_logits[:, -1, :]`` then indexed.
* ``NanoVLM.compute_output_shape`` claimed ``vision_seq_len + text_seq_len`` for
  every non-``attention_pooling`` strategy, contradicting
  ``MultiModalFusion.compute_output_shape``, which returns the vision length
  alone for all six sequence-preserving strategies and a per-modality tuple only
  for ``cross_attention``.

Both are now pinned against the **actual forward output**, not against each
other's arithmetic.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.nano_vlm.model import NanoVLM

IMG = 32
PATCH = 16
DIM = 16
VOCAB = 32
BATCH = 2

# The vision tower emits (IMG // PATCH)^2 patch tokens plus a CLS token.
VISION_LEN = (IMG // PATCH) ** 2 + 1

# Strategies that keep a per-position sequence and so can drive generate().
SEQUENCE_STRATEGIES = [
    "cross_attention", "concatenation", "addition", "multiplication",
    "gated", "bilinear", "tensor_fusion",
]


def _model(strategy):
    return NanoVLM(
        vision_config={"img_size": IMG, "patch_size": PATCH, "embed_dim": DIM,
                       "depth": 1, "num_heads": 2, "output_mode": "none"},
        text_config={"vocab_size": VOCAB, "embed_dim": DIM, "depth": 1,
                     "num_heads": 2, "max_seq_len": 32},
        fusion_config={"fusion_strategy": strategy, "dim": DIM,
                       "attention_config": {"num_heads": 2},
                       "num_fusion_layers": 1},
        vocab_size=VOCAB,
    )


def _batch(text_len):
    rng = np.random.default_rng(0)
    return (
        rng.random((BATCH, IMG, IMG, 3), dtype="float32"),
        rng.integers(0, VOCAB, (BATCH, text_len)).astype("int32"),
    )


class TestGenerateEitherWorksOrRefusesByName:

    def test_cross_attention_extends_the_prompt(self):
        model = _model("cross_attention")
        images, prompt = _batch(text_len=3)
        generated = model.generate(images, prompt, max_length=2, top_k=0)
        assert generated.shape[0] == BATCH
        assert generated.shape[1] > prompt.shape[1]

    @pytest.mark.parametrize(
        "strategy", [s for s in SEQUENCE_STRATEGIES if s != "cross_attention"]
        + ["attention_pooling"]
    )
    def test_every_other_strategy_is_refused_by_name(self, strategy):
        """Refusal, not a wrong answer. Before this, the loop ran for all eight
        and sliced a vision-length prefix off a tensor that had none, leaving an
        empty axis that ``text_logits[:, -1, :]`` indexed."""
        model = _model(strategy)
        images, prompt = _batch(text_len=VISION_LEN)
        with pytest.raises(ValueError) as excinfo:
            model.generate(images, prompt, max_length=1)
        message = str(excinfo.value)
        assert strategy in message
        assert "cross_attention" in message

    def test_the_equal_length_precondition_really_does_break_at_step_2(self):
        """Anti-vacuity control for the refusal above: the six length-sensitive
        strategies are not merely awkward here, they are impossible. One fused
        forward at matched lengths SUCCEEDS; appending a single token to the
        text stream then breaks it, and the loop appends one per step."""
        model = _model("concatenation")
        images, prompt = _batch(text_len=VISION_LEN)
        model({"images": images, "text_tokens": prompt}, training=False)

        grown = np.concatenate([prompt, prompt[:, :1]], axis=1)
        with pytest.raises(ValueError, match="same sequence length"):
            model({"images": images, "text_tokens": grown}, training=False)

    def test_unequal_lengths_are_refused_by_name_not_by_concat_op(self):
        """The factory docstring's own example, before it was corrected."""
        model = _model("concatenation")
        images, prompt = _batch(text_len=VISION_LEN + 2)
        with pytest.raises(ValueError, match="concatenation"):
            model({"images": images, "text_tokens": prompt}, training=False)


class TestComputeOutputShapeAgreesWithTheForwardPass:

    @pytest.mark.parametrize("strategy", SEQUENCE_STRATEGIES + ["attention_pooling"])
    def test_claimed_logits_shape_equals_actual(self, strategy):
        model = _model(strategy)
        images, text = _batch(text_len=VISION_LEN)
        logits = model({"images": images, "text_tokens": text}, training=False)

        claimed = model.compute_output_shape({
            "images": (BATCH, IMG, IMG, 3),
            "text_tokens": (BATCH, VISION_LEN),
        })
        assert tuple(claimed) == tuple(logits.shape), (
            f"compute_output_shape disagrees with call() for '{strategy}'"
        )

    def test_only_cross_attention_sums_the_two_lengths(self):
        """The specific arithmetic that was applied to all eight strategies."""
        cross = _model("cross_attention").compute_output_shape({
            "images": (BATCH, IMG, IMG, 3), "text_tokens": (BATCH, VISION_LEN),
        })
        concat = _model("concatenation").compute_output_shape({
            "images": (BATCH, IMG, IMG, 3), "text_tokens": (BATCH, VISION_LEN),
        })
        assert cross[1] == 2 * VISION_LEN
        assert concat[1] == VISION_LEN
