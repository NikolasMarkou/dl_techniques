"""The helpers the head domains share, each against a number written down here.

``heads/validation.py`` (``lookup_head``, ``key_by_name``, ``pool_or_passthrough``),
``heads/masking.py`` (``fill_masked``) and ``VLMTaskConfig.to_dict`` / ``coerce``
replaced code that each of the NLP, VLM and vision domains had written out. The
domain suites (dispatch, multi-task, padding masks, round trips) pin behaviour
through the heads; this file pins the helpers directly, so a helper changed
without a head going red is still caught.

Expected values are literals chosen here (``-10000.0`` is the float16 sentinel,
``-1e9`` the float32 and float64 one), not recomputed through the helper's own
formula.
"""

import ast
from pathlib import Path

import keras
import numpy as np
import pytest

import dl_techniques.layers.heads as heads_pkg
from dl_techniques.layers.heads.masking import fill_masked
from dl_techniques.layers.heads.nlp.task_types import NLPTaskType
from dl_techniques.layers.heads.validation import (
    key_by_name,
    lookup_head,
    pool_or_passthrough,
)
from dl_techniques.layers.heads.vision.task_types import VisionTaskType
from dl_techniques.layers.heads.vlm import VLMTaskConfig, VLMTaskType

HEADS_DIR = Path(heads_pkg.__file__).parent


# ---------------------------------------------------------------------
# lookup_head
# ---------------------------------------------------------------------

_TABLE = {VLMTaskType.IMAGE_CAPTIONING: "cap", VLMTaskType.VISUAL_GROUNDING: "ground"}


def test_lookup_head_returns_the_member_and_its_row_for_a_member_or_a_string() -> None:
    assert lookup_head(VLMTaskType.IMAGE_CAPTIONING, _TABLE, VLMTaskType, "VLM") == (
        VLMTaskType.IMAGE_CAPTIONING, "cap")
    assert lookup_head(" Visual_Grounding ", _TABLE, VLMTaskType, "VLM") == (
        VLMTaskType.VISUAL_GROUNDING, "ground")


def test_lookup_head_refuses_a_member_without_a_row_and_names_the_ones_with_one() -> None:
    with pytest.raises(ValueError) as info:
        lookup_head(VLMTaskType.VIDEO_CAPTIONING, _TABLE, VLMTaskType, "VLM", owner="X")
    message = str(info.value)
    assert message.startswith("X: ") and "VIDEO_CAPTIONING" in message
    assert "IMAGE_CAPTIONING" in message and "VISUAL_GROUNDING" in message
    assert "2 task types" in message


@pytest.mark.parametrize("bad", [3, None, NLPTaskType.SENTIMENT_ANALYSIS])
def test_lookup_head_refuses_a_value_that_is_not_this_domains_task_type(bad) -> None:
    with pytest.raises(ValueError, match="VLMTaskType"):
        lookup_head(bad, _TABLE, VLMTaskType, "VLM")


def test_lookup_head_refuses_a_string_naming_no_task_type() -> None:
    with pytest.raises(ValueError, match="bogus"):
        lookup_head("bogus", _TABLE, VLMTaskType, "VLM")


def test_the_three_dispatchers_route_through_lookup_head() -> None:
    """A fourth copy of the lookup would not show up in any dispatch test."""
    for rel in ("nlp/dispatch.py", "vlm/dispatch.py", "vision/dispatch.py"):
        tree = ast.parse((HEADS_DIR / rel).read_text())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "lookup_head" in called, rel
        defaults = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and len(n.args) == 2
                    and "DISPATCH" in ast.unparse(n.func.value)]
        assert defaults == [], f"{rel} falls back to a default head"


# ---------------------------------------------------------------------
# key_by_name
# ---------------------------------------------------------------------

def test_key_by_name_keys_a_list_and_a_tuple_in_order_and_passes_a_dict_through() -> None:
    items = ["b", "a", "c"]
    keyed = key_by_name(items, str.upper, "f")
    assert list(keyed.items()) == [("B", "b"), ("A", "a"), ("C", "c")]
    assert key_by_name(tuple(items), str.upper, "f") == key_by_name(items, str.upper, "f")
    given = {"x": 1}
    assert key_by_name(given, str.upper, "f") is given


def test_key_by_name_refuses_a_repeated_name_instead_of_keeping_the_last() -> None:
    with pytest.raises(ValueError, match=r"f: names \['A'\] appear more than once"):
        key_by_name(["a", "A", "b"], str.upper, "f")


@pytest.mark.parametrize("bad", [{1, 2}, "abc", 3, None])
def test_key_by_name_refuses_any_other_container(bad) -> None:
    with pytest.raises(ValueError, match="list, tuple, dict"):
        key_by_name(bad, str, "f")


# ---------------------------------------------------------------------
# pool_or_passthrough
# ---------------------------------------------------------------------

def _pool(features, mask):
    return features.sum(axis=1) if mask is None else (features * mask[..., None]).sum(axis=1)


def test_pool_or_passthrough_pools_rank_three_under_the_mask() -> None:
    x = np.arange(12, dtype="float32").reshape(2, 3, 2)
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype="float32")
    out = pool_or_passthrough(x, mask, _pool, owner="H", features_name="f", mask_name="m")
    np.testing.assert_array_equal(out, [[2.0, 4.0], [6.0, 7.0]])


def test_pool_or_passthrough_returns_rank_two_unchanged_and_never_calls_pool() -> None:
    x = np.ones((2, 4), dtype="float32")

    def boom(*_):
        raise AssertionError("pool was called for a pooled input")

    assert pool_or_passthrough(x, None, boom, owner="H", features_name="f", mask_name="m") is x


def test_pool_or_passthrough_refuses_a_mask_for_a_pooled_input_and_other_ranks() -> None:
    x2 = np.ones((2, 4), dtype="float32")
    with pytest.raises(ValueError, match="H: m was given for a rank-2 f"):
        pool_or_passthrough(x2, np.ones((2, 3)), _pool, owner="H", features_name="f", mask_name="m")
    with pytest.raises(ValueError, match="H: f must be rank 2 .*rank 3 .*got rank 4"):
        pool_or_passthrough(np.ones((2, 3, 4, 5)), None, _pool,
                            owner="H", features_name="f", mask_name="m")


# ---------------------------------------------------------------------
# fill_masked
# ---------------------------------------------------------------------

#: dtype -> the sentinel `mask_sentinel` is documented to return for it.
_SENTINELS = {"float16": -10000.0, "float32": -1e9, "float64": -1e9}


@pytest.mark.parametrize("dtype", sorted(_SENTINELS))
def test_fill_masked_selects_the_sentinel_at_masked_positions_and_leaves_the_rest_bit_equal(dtype) -> None:
    scores = np.array([[0.5, -2.0, 3.25, 7.0]], dtype=dtype)
    keep = np.array([[1, 0, 1, 0]])
    out = keras.ops.convert_to_numpy(fill_masked(keras.ops.convert_to_tensor(scores), keep))
    assert out.dtype == np.dtype(dtype)
    expected = np.array([[0.5, _SENTINELS[dtype], 3.25, _SENTINELS[dtype]]], dtype=dtype)
    np.testing.assert_array_equal(out, expected)


def test_the_additive_form_is_nan_at_kept_positions_in_float16_and_fill_masked_is_not() -> None:
    """The hazard the helper exists for, shown first so the arm above means something."""
    scores = np.array([[0.5, -2.0, 3.25]], dtype="float16")
    keep = np.array([[1, 0, 1]], dtype="float16")
    assert np.float16(-1e9) == -np.inf
    additive = scores * keep + (1 - keep) * np.float16(-1e9)
    assert np.isnan(additive).any()
    out = keras.ops.convert_to_numpy(fill_masked(keras.ops.convert_to_tensor(scores), keep))
    assert np.isfinite(out).all()


def test_fill_masked_takes_its_sentinel_from_the_dtype_it_is_told() -> None:
    scores = keras.ops.convert_to_tensor(np.zeros((1, 2), dtype="float32"))
    own = keras.ops.convert_to_numpy(fill_masked(scores, np.array([[1, 0]])))
    forced = keras.ops.convert_to_numpy(fill_masked(scores, np.array([[1, 0]]), "float16"))
    assert own[0, 1] == np.float32(-1e9) and forced[0, 1] == np.float32(-10000.0)


# ---------------------------------------------------------------------
# VLMTaskConfig.to_dict / coerce
# ---------------------------------------------------------------------

def test_a_vlm_task_config_round_trips_through_to_dict_and_coerce() -> None:
    cfg = VLMTaskConfig(name="q", task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
                        num_classes=7, hidden_size=32, dropout_rate=0.25)
    data = cfg.to_dict()
    assert data == {"name": "q", "task_type": "visual_question_answering",
                    "vocab_size": None, "hidden_size": 32, "dropout_rate": 0.25,
                    "num_classes": 7}
    assert VLMTaskConfig.coerce(data) == cfg
    assert VLMTaskConfig.coerce(cfg) is cfg


def test_vlm_coerce_refuses_anything_else() -> None:
    with pytest.raises(TypeError, match="VLMTaskConfig"):
        VLMTaskConfig.coerce([("name", "x")])


def test_the_to_dict_is_json_data() -> None:
    import json
    cfg = VLMTaskConfig(name="c", task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=5)
    assert json.loads(json.dumps(cfg.to_dict())) == cfg.to_dict()


def test_vision_coerce_is_what_the_vision_dispatch_uses_for_enhancement() -> None:
    assert VisionTaskType.coerce("denoising") is VisionTaskType.DENOISING
