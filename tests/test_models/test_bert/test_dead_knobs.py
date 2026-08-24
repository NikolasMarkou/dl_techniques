"""`BERT.__init__`'s two knobs that reached nothing.

Both were stored on `self` and serialized by `get_config()`, and neither was
ever read:

* ``position_embedding_type="absolute"`` -- `BertEmbeddings` has a parameter of
  exactly that name, `BERT._build_architecture` never forwarded it, and
  ``"absolute"`` is not in `VALID_POSITION_EMBEDDING_TYPES`
  (``('learned', 'sinusoidal')``), so the shipped default would have RAISED if
  anyone had wired it up naively. RULED: wire it, copying FNet's D-071 exactly
  -- normalize the legacy ``"absolute"`` spelling to ``"learned"`` once in
  ``__init__``, then forward. `'sinusoidal'` becomes reachable; the default's
  behaviour is unchanged, because `BertEmbeddings` already defaulted to
  ``'learned'``.
* ``use_cache=True``, documented as controlling "caching in attention layers".
  BERT here is a bidirectional encoder with no incremental-decoding path and no
  KV cache anywhere in the stack, so there is nothing for it to control. RULED:
  delete, with `from_config` dropping the legacy key so a `.keras` file written
  before this change still loads.

See decisions.md D-015.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.bert_embeddings import (
    VALID_POSITION_EMBEDDING_TYPES,
)
from dl_techniques.models.bert.bert import BERT

SMALL = dict(
    vocab_size=32, hidden_size=16, num_layers=1, num_heads=2,
    intermediate_size=32, max_position_embeddings=16,
)


def _ids(batch=2, seq=8):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).integers(0, 32, (batch, seq)).astype("int32")
    )


class TestPositionEmbeddingTypeIsWired:

    def test_the_shipped_default_would_have_been_illegal_downstream(self):
        """Pin the reason this was not merely unused but unusable."""
        assert "absolute" not in VALID_POSITION_EMBEDDING_TYPES
        assert set(VALID_POSITION_EMBEDDING_TYPES) == {"learned", "sinusoidal"}

    def test_legacy_absolute_is_normalized_to_learned(self):
        model = BERT(position_embedding_type="absolute", **SMALL)
        assert model.position_embedding_type == "learned"
        assert model.embeddings.position_embedding_type == "learned"

    def test_the_value_reaches_the_embedding_layer(self):
        model = BERT(position_embedding_type="sinusoidal", **SMALL)
        assert model.embeddings.position_embedding_type == "sinusoidal"

    def test_it_changes_the_forward_output(self):
        """Not a stored-attribute assertion: sinusoidal and learned position
        signals must produce different hidden states."""
        ids = _ids()
        keras.utils.set_random_seed(3)
        learned = BERT(position_embedding_type="learned", **SMALL)(ids)
        keras.utils.set_random_seed(3)
        sinus = BERT(position_embedding_type="sinusoidal", **SMALL)(ids)

        a = keras.ops.convert_to_numpy(
            learned["last_hidden_state"] if isinstance(learned, dict) else learned
        )
        b = keras.ops.convert_to_numpy(
            sinus["last_hidden_state"] if isinstance(sinus, dict) else sinus
        )
        delta = float(np.max(np.abs(a - b)))
        assert delta > 1e-5, f"position_embedding_type is still inert: {delta:.6e}"

    def test_an_unrecognized_value_raises(self):
        with pytest.raises(ValueError, match="position_embedding_type"):
            BERT(position_embedding_type="rotary", **SMALL)

    def test_get_config_carries_the_normalized_spelling(self):
        cfg = BERT(position_embedding_type="absolute", **SMALL).get_config()
        assert cfg["position_embedding_type"] == "learned"
        assert BERT.from_config(cfg).position_embedding_type == "learned"


class TestUseCacheIsGone:

    def test_the_constructor_refuses_it(self):
        # MEASURED: `keras.Model.__init__` swallows unknown kwargs from `**kwargs`
        # and re-raises them as a `ValueError`, not the `TypeError` a plain
        # Python signature mismatch would give. Pin what actually fires.
        with pytest.raises(ValueError, match="use_cache"):
            BERT(use_cache=True, **SMALL)

    def test_it_is_absent_from_get_config(self):
        assert "use_cache" not in BERT(**SMALL).get_config()

    def test_a_legacy_config_carrying_use_cache_still_loads(self):
        """`bert/` is the most reachable package in the tree and `from_config`
        is `cls(**config)`, so a bare delete would make every `.keras` file
        written before this change unloadable."""
        cfg = BERT(**SMALL).get_config()
        cfg["use_cache"] = True
        cfg["position_embedding_type"] = "absolute"

        model = BERT.from_config(cfg)
        assert not hasattr(model, "use_cache")
        assert model.position_embedding_type == "learned"
