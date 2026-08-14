"""Tests for the embedding factory: covers all registered keys, parameter
passthrough, validation, and config-driven construction."""

import re
import math
import numpy as np
import pytest
import keras

from dl_techniques.layers.embedding import (
    STRICT_DROPPED_KEY_MARKER,
    create_embedding_layer,
    create_embedding_from_config,
    validate_embedding_config,
)
from dl_techniques.layers.embedding.factory import (
    EMBEDDING_REGISTRY,
    get_embedding_info,
)

# Minimal valid construction kwargs for every registered key.
VALID_CFGS = {
    "patch_1d": dict(patch_size=4, embed_dim=16),
    "patch_2d": dict(patch_size=4, embed_dim=16),
    "positional_learned": dict(max_seq_len=32, dim=16),
    "rope": dict(head_dim=16, max_seq_len=32),
    "dual_rope": dict(head_dim=16, max_seq_len=32),
    "continuous_rope": dict(dim=64, ndim=3),
    "continuous_sincos": dict(dim=64, ndim=3),
    "bert_embeddings": dict(vocab_size=50, hidden_size=16,
                            max_position_embeddings=32, type_vocab_size=2),
    "modern_bert_embeddings": dict(vocab_size=50, hidden_size=16, type_vocab_size=2,
                                   initializer_range=0.02, layer_norm_eps=1e-12,
                                   dropout_rate=0.1, use_bias=False),
    "albert_factorized": dict(vocab_size=50, bottleneck_dim=8, output_dim=16),
    "positional_sine_2d": dict(num_pos_feats=8),
    "scalar_sinusoidal": dict(dim=64),
    "mrope_ideogram4": dict(head_dim=64, rope_theta=10000.0, mrope_section=(2, 3, 3)),
}


class TestEmbeddingFactory:

    def test_all_registry_keys_have_test_config(self):
        # Guards against adding a registry key without test coverage here.
        assert set(VALID_CFGS) == set(EMBEDDING_REGISTRY)

    @pytest.mark.parametrize("key", list(VALID_CFGS))
    def test_construct_all_keys(self, key):
        layer = create_embedding_layer(key, **VALID_CFGS[key])
        assert isinstance(layer, keras.layers.Layer)
        assert type(layer).__name__ == EMBEDDING_REGISTRY[key]["class"].__name__

    def test_name_passthrough(self):
        layer = create_embedding_layer("rope", head_dim=16, max_seq_len=32, name="my_rope")
        assert layer.name == "my_rope"

    def test_optional_param_passthrough(self):
        # modern_bert's ctor requires initializer_range/layer_norm_eps/dropout_rate/use_bias
        # (no defaults), so they are required_params; supply them and confirm the value flows
        # through to the built layer.
        l = create_embedding_layer("modern_bert_embeddings", vocab_size=50,
                                   hidden_size=16, type_vocab_size=2,
                                   initializer_range=0.02, layer_norm_eps=1e-12,
                                   use_bias=True, dropout_rate=0.3)
        assert l.dropout_rate == 0.3
        # albert bottleneck honored.
        l2 = create_embedding_layer("albert_factorized", vocab_size=50,
                                    bottleneck_dim=8, output_dim=16)
        assert l2.bottleneck_dim == 8
        # positional_sine_2d temperature override.
        l3 = create_embedding_layer("positional_sine_2d", num_pos_feats=16, temperature=5000.0)
        assert l3.temperature == 5000.0

    def test_from_config(self):
        layer = create_embedding_from_config(
            {"type": "positional_sine_2d", "num_pos_feats": 16, "temperature": 5000.0})
        assert layer.num_pos_feats == 16 and layer.temperature == 5000.0

    def test_get_embedding_info_covers_registry(self):
        info = get_embedding_info()
        assert set(info) == set(EMBEDDING_REGISTRY)
        for entry in info.values():
            assert "required_params" in entry and "optional_params" in entry

    # ---- validation errors ------------------------------------------

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_embedding_layer("does_not_exist", dim=8)

    def test_missing_required_raises(self):
        with pytest.raises(ValueError):
            validate_embedding_config("albert_factorized", vocab_size=50)  # missing bottleneck/output

    def test_bad_value_raises(self):
        with pytest.raises(ValueError):
            validate_embedding_config("positional_sine_2d", temperature=-1.0)
        with pytest.raises(ValueError):
            validate_embedding_config("rope", head_dim=16, max_seq_len=32, rope_theta=-1.0)
        with pytest.raises(ValueError):
            validate_embedding_config("albert_factorized", vocab_size=50,
                                      bottleneck_dim=0, output_dim=16)


# =====================================================================
# I-5: the three opt-in BertEmbeddings parameters must SURVIVE the factory.
#
# ``create_embedding_layer`` filters kwargs to
# ``required_params | optional_params`` and silently DROPS the rest
# (``factory.py``, the ``final_params`` comprehension). A parameter that is
# not registered is therefore a silent no-op, and a test that only asserts
# "construction succeeded" cannot see it -- it would pass identically with the
# parameter registered and un-registered.
#
# Every test below asserts the parameter's EFFECT: the value read back off
# ``get_config()`` AND a behavioural consequence (which weight exists, whether a
# Keras mask is produced). Each was proven red by un-registering its own
# parameter from ``EMBEDDING_REGISTRY['bert_embeddings']['optional_params']``;
# see ``findings/step5-mutation-log.md`` of
# plans/plan-2026-08-10T183739-b007f435.
# =====================================================================

BERT_BASE = dict(vocab_size=50, hidden_size=16, max_position_embeddings=32)


def _weight_paths(layer):
    """Build the layer on a fixed input and return its weight paths."""
    ids = keras.ops.convert_to_tensor(np.array([[1, 2, 0, 0]], dtype="int32"))
    layer(ids)
    return {w.path for w in layer.weights}


class TestBertEmbeddingsFactoryParams:

    def test_use_token_type_embeddings_false_reaches_the_layer(self):
        layer = create_embedding_layer(
            "bert_embeddings", **BERT_BASE,
            type_vocab_size=2,            # supplied but inert -> normalized to None
            use_token_type_embeddings=False,
        )
        cfg = layer.get_config()
        assert cfg["use_token_type_embeddings"] is False, (
            "use_token_type_embeddings did not survive create_embedding_layer -- it is "
            "missing from EMBEDDING_REGISTRY['bert_embeddings'] and was silently dropped"
        )
        # Behavioural consequence: no segment-embedding weight is allocated at all.
        paths = _weight_paths(layer)
        assert not any("token_type_embeddings" in p for p in paths), (
            f"token_type_embeddings weight exists despite use_token_type_embeddings=False: "
            f"{sorted(paths)}"
        )
        # And the now-inert value is not serialized (D-002).
        assert cfg["type_vocab_size"] is None

    def test_use_token_type_embeddings_default_still_builds_the_segment_weight(self):
        """Control for the test above: the assertion must be able to go both ways."""
        layer = create_embedding_layer("bert_embeddings", **BERT_BASE, type_vocab_size=2)
        assert layer.get_config()["use_token_type_embeddings"] is True
        paths = _weight_paths(layer)
        assert any("token_type_embeddings" in p for p in paths), sorted(paths)

    def test_position_embedding_type_sinusoidal_reaches_the_layer(self):
        layer = create_embedding_layer(
            "bert_embeddings", **BERT_BASE, type_vocab_size=2,
            position_embedding_type="sinusoidal",
        )
        assert layer.get_config()["position_embedding_type"] == "sinusoidal", (
            "position_embedding_type did not survive create_embedding_layer -- it is "
            "missing from EMBEDDING_REGISTRY['bert_embeddings'] and was silently dropped"
        )
        # Behavioural consequence: the sinusoidal branch allocates NO position table.
        paths = _weight_paths(layer)
        assert not any("position_embeddings" in p for p in paths), (
            f"a learned position_embeddings weight exists despite "
            f"position_embedding_type='sinusoidal': {sorted(paths)}"
        )

    def test_position_embedding_type_default_still_builds_the_learned_table(self):
        """Control for the test above."""
        layer = create_embedding_layer("bert_embeddings", **BERT_BASE, type_vocab_size=2)
        assert layer.get_config()["position_embedding_type"] == "learned"
        paths = _weight_paths(layer)
        assert any("position_embeddings" in p for p in paths), sorted(paths)

    def test_mask_zero_false_reaches_the_layer(self):
        layer = create_embedding_layer(
            "bert_embeddings", **BERT_BASE, type_vocab_size=2, mask_zero=False,
        )
        assert layer.get_config()["mask_zero"] is False, (
            "mask_zero did not survive create_embedding_layer -- it is missing from "
            "EMBEDDING_REGISTRY['bert_embeddings'] and was silently dropped"
        )
        # Behavioural consequence: no Keras auto-mask is produced for pad id 0 (I-4).
        ids = keras.ops.convert_to_tensor(np.array([[1, 2, 0, 0]], dtype="int32"))
        layer(ids)
        assert layer.word_embeddings.compute_mask(ids) is None, (
            "word_embeddings still emits a Keras auto-mask despite mask_zero=False"
        )

    def test_mask_zero_default_still_emits_a_mask(self):
        """Control: mask_zero=True really does produce a mask, so the `is None`
        assertion above is discriminating rather than vacuously true."""
        layer = create_embedding_layer("bert_embeddings", **BERT_BASE, type_vocab_size=2)
        assert layer.get_config()["mask_zero"] is True
        ids = keras.ops.convert_to_tensor(np.array([[1, 2, 0, 0]], dtype="int32"))
        layer(ids)
        mask = layer.word_embeddings.compute_mask(ids)
        assert mask is not None
        assert keras.ops.convert_to_numpy(mask).tolist() == [[True, True, False, False]]

    # ---- the conditional-required `type_vocab_size` rule (D-002 / D-010) ----

    def test_omitting_type_vocab_size_with_token_types_enabled_still_raises(self):
        """Regression guard for moving type_vocab_size out of required_params.

        Before this change the static required-params check produced this error. The
        computed rule in validate_embedding_config must keep producing it, or the move
        would be a silent downgrade of an existing loud failure.
        """
        with pytest.raises(ValueError, match="type_vocab_size"):
            create_embedding_layer("bert_embeddings", **BERT_BASE)

    def test_validate_rejects_missing_type_vocab_size_when_token_types_enabled(self):
        """Isolates the FACTORY-level rule from the constructor's own raise: this calls
        validation only, so no BertEmbeddings is ever constructed."""
        with pytest.raises(ValueError, match="type_vocab_size"):
            validate_embedding_config("bert_embeddings", **BERT_BASE)
        with pytest.raises(ValueError, match="type_vocab_size"):
            validate_embedding_config("bert_embeddings", **BERT_BASE, type_vocab_size=None)
        with pytest.raises(ValueError, match="type_vocab_size"):
            validate_embedding_config("bert_embeddings", **BERT_BASE,
                                      use_token_type_embeddings=True, type_vocab_size=None)

    def test_validate_allows_missing_type_vocab_size_when_token_types_disabled(self):
        # No raise: the parameter is genuinely meaningless on this path.
        validate_embedding_config("bert_embeddings", **BERT_BASE,
                                  use_token_type_embeddings=False)

    def test_validate_rejects_bad_position_embedding_type(self):
        """Factory-level enum check, isolated from the constructor's (defence-in-depth)."""
        with pytest.raises(ValueError, match="position_embedding_type"):
            validate_embedding_config("bert_embeddings", **BERT_BASE, type_vocab_size=2,
                                      position_embedding_type="relative")
        # Both legal values pass validation.
        for value in ("learned", "sinusoidal"):
            validate_embedding_config("bert_embeddings", **BERT_BASE, type_vocab_size=2,
                                      position_embedding_type=value)

    def test_bad_position_embedding_type_raises_through_the_factory_too(self):
        with pytest.raises(ValueError, match="position_embedding_type"):
            create_embedding_layer("bert_embeddings", **BERT_BASE, type_vocab_size=2,
                                   position_embedding_type="relative")

    # ---- negative control: unregistered keys now RAISE ----

    def test_bogus_key_now_raises(self):
        """The silent-drop contract is REVERSED: an unregistered key is an error.

        This test previously asserted the opposite (``..._is_still_silently_dropped``,
        "must not turn the factory strict"). That contract is what let
        ``ViT(pos_dropout_rate=0.5)`` build a permanently-0.0 dropout at four
        production call sites, so it was deliberately reversed by
        ``plan-2026-08-14T042537-ff96c6c6`` D-002 (see the ``# DECISION`` anchor in
        ``layers/embedding/factory.py`` at the strict dropped-kwarg raise).

        Keyed on ``STRICT_DROPPED_KEY_MARKER`` rather than on message prose, and
        ``re.escape``d because the marker's ``(s)`` is a regex group.
        """
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)):
            create_embedding_layer(
                "bert_embeddings", **BERT_BASE, type_vocab_size=2,
                definitely_not_a_real_param=123,
            )

    def test_registry_declares_all_three_new_params_with_ctor_defaults(self):
        """Static companion to the effect tests: names AND default values.

        The defaults are injected by create_embedding_layer
        (``params.update(optional_params)``), so a wrong default here is behavioural,
        not documentation.
        """
        optional = EMBEDDING_REGISTRY["bert_embeddings"]["optional_params"]
        assert optional["use_token_type_embeddings"] is True
        assert optional["position_embedding_type"] == "learned"
        assert optional["mask_zero"] is True
        assert optional["type_vocab_size"] is None
        assert "type_vocab_size" not in EMBEDDING_REGISTRY["bert_embeddings"]["required_params"]


# =====================================================================
# D-002 (plan-2026-08-14T042537-ff96c6c6): the strict dropped-kwarg raise.
#
# BOTH arms are tested, because the predicate has two independent ways to be
# wrong:
#   * too weak  -> an unknown key is silently dropped again (the ViT bug).
#   * too broad -> a legitimate OPTIONAL parameter raises. This is the measured
#     D-023 trap: narrowing the predicate's right-hand side from
#     `required | optional` to `required` alone makes every optional parameter
#     anyone passes an error. `test_every_registered_type_accepts_all_its_optional_params`
#     is the control that catches it, and it was proven RED by exactly that
#     narrowing: 11 of the 13 parametrized cases failed -- every type that
#     declares an optional param at all (the two survivors,
#     `modern_bert_embeddings` and `mrope_ideogram4`, have an empty
#     `optional_params`), and 22 tests in this directory went red in total.
# =====================================================================


class TestStrictDroppedKwargs:

    def test_unknown_kwarg_raises_with_the_marker(self):
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)):
            create_embedding_layer("positional_learned", max_seq_len=32, dim=16,
                                   dropout=0.5)

    def test_message_names_the_dropped_key_and_the_accepted_set(self):
        with pytest.raises(ValueError) as excinfo:
            create_embedding_layer("positional_learned", max_seq_len=32, dim=16,
                                   dropout=0.5)
        message = str(excinfo.value)
        assert STRICT_DROPPED_KEY_MARKER in message
        assert "'dropout'" in message          # the key the caller got wrong
        assert "'dropout_rate'" in message     # the key they meant, from the accepted set

    def test_all_unknown_keys_are_reported_not_just_the_first(self):
        with pytest.raises(ValueError) as excinfo:
            create_embedding_layer("rope", head_dim=16, max_seq_len=32,
                                   nope_one=1, nope_two=2)
        message = str(excinfo.value)
        assert "'nope_one'" in message and "'nope_two'" in message
        assert f"2 {STRICT_DROPPED_KEY_MARKER}" in message

    @pytest.mark.parametrize("key", list(VALID_CFGS))
    def test_every_registered_type_accepts_all_its_optional_params(self, key):
        """CONTROL for the over-broad predicate (the measured D-023 trap).

        Passes every one of the type's registry-declared OPTIONAL parameters
        explicitly, at its own registry default value, on top of the type's
        minimal valid config. None of these may raise -- they are precisely the
        parameters the factory itself injects. Narrowing the raise's right-hand
        side to ``set(required_params)`` turns every one of these into an error.
        """
        cfg = dict(VALID_CFGS[key])
        for opt_name, opt_default in EMBEDDING_REGISTRY[key]["optional_params"].items():
            cfg.setdefault(opt_name, opt_default)
        layer = create_embedding_layer(key, **cfg)
        assert isinstance(layer, keras.layers.Layer)

    def test_name_and_embedding_type_are_not_treated_as_dropped_keys(self):
        """`name` is the factory's own signature parameter, not a registry key."""
        layer = create_embedding_layer("rope", head_dim=16, max_seq_len=32,
                                       name="rope_named")
        assert layer.name == "rope_named"

    def test_create_embedding_from_config_also_raises(self):
        """The wrapper forwards `**config`, so it inherits the same contract."""
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)):
            create_embedding_from_config(
                {"type": "positional_sine_2d", "num_pos_feats": 16, "nope": 1})

    def test_unknown_type_still_fails_before_the_kwarg_diff(self):
        """Ordering guard: `validate_embedding_config` runs FIRST, so an unknown
        type keeps its pre-existing failure mode and never reaches the raise."""
        with pytest.raises(ValueError) as excinfo:
            create_embedding_layer("does_not_exist", definitely_bogus=1)
        assert STRICT_DROPPED_KEY_MARKER not in str(excinfo.value)
        assert "Unknown embedding type" in str(excinfo.value)

    def test_missing_required_still_fails_before_the_kwarg_diff(self):
        with pytest.raises(ValueError) as excinfo:
            create_embedding_layer("albert_factorized", vocab_size=50,
                                   definitely_bogus=1)
        assert STRICT_DROPPED_KEY_MARKER not in str(excinfo.value)
        assert "Required parameters missing" in str(excinfo.value)
