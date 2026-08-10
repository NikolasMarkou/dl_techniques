"""
Test suite for DistilBERT (encoder foundation model).

Covers the paths that were never exercised before plan
`plan-2026-08-10-b007f435`: the top-level package import, the sinusoidal
position branch under `float32` and `mixed_float16`, `rms_norm`, the
unknown-`normalization_type` raise (which must come from the EMBEDDING stage,
not from the first `TransformerLayer`), the I-4 no-auto-masking guard, the
D-003 `pad_token_id`-is-advisory guard, `create_distilbert_with_head` end to
end, and the `get_config()` / `.keras` round trips asserted on weight VALUES
and a `training=False` forward output.

Measured facts these tests encode (do not "simplify" them into shape checks):

* `mask_zero=False` is INVISIBLE to a forward-output comparison -- deleting it
  leaves every output sha bit-identical and produces no `_keras_mask` at all
  (`findings/step6-equivalence.md` mutation M-A). The guard must read the
  structural flag.
* `training=None` is NOT inference in this repo, so every round-trip forward is
  taken at an explicit `training=False`.
* A statistic that reads the process-global Keras RNG is coupled to pytest
  COLLECTION ORDER, so `keras.utils.set_random_seed` is called immediately
  before construction wherever values are compared.

`call()` accepts an int32 (B, T) token tensor or a dict with `input_ids` and
returns a dict with `last_hidden_state` + `attention_mask`.
"""

import os
import keras
import pytest
import importlib
import traceback
import numpy as np

from dl_techniques.models.distilbert.model import DistilBERT

SEED = 20260811


def _model(**overrides):
    params = dict(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    params.update(overrides)
    return DistilBERT(**params)


def _tokens(batch=2, seq=16):
    return np.random.default_rng(0).integers(0, 256, (batch, seq)).astype("int32")


def _padded_tokens(batch=2, seq=12, keep=8):
    """Tokens whose tail is `pad_token_id` (0) -- the D-003 setting."""
    ids = np.random.default_rng(3).integers(1, 256, (batch, seq)).astype("int32")
    ids[:, keep:] = 0
    return ids


def _np(tensor):
    return keras.ops.convert_to_numpy(tensor)


def _traceback_files(exc):
    """Every source file in `exc`'s traceback AND in its `__cause__` chain.

    `create_embedding_layer` re-raises with `from`, so the frame that actually
    refused the value lives on the cause, not on the surface exception.
    """
    files = []
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        files.extend(f.filename for f in traceback.extract_tb(cur.__traceback__))
        cur = cur.__cause__ or cur.__context__
    return files


class TestDistilBERT:

    def test_forward_dict(self):
        out = _model()(_tokens(), training=False)
        assert "last_hidden_state" in out
        assert tuple(out["last_hidden_state"].shape) == (2, 16, 64)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _tokens()
        before = keras.ops.convert_to_numpy(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "distilbert.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["last_hidden_state"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DistilBERT differs after .keras round-trip")


class TestTopLevelImport:
    """SC-1. `models/distilbert/__init__.py` was 0 bytes before step 7."""

    def test_public_symbols_are_importable_from_the_package(self):
        pkg = importlib.import_module("dl_techniques.models.distilbert")

        assert hasattr(pkg, "DistilBERT"), (
            "dl_techniques.models.distilbert does not export DistilBERT; "
            "the package __init__ is empty or incomplete"
        )
        assert hasattr(pkg, "create_distilbert_with_head"), (
            "dl_techniques.models.distilbert does not export "
            "create_distilbert_with_head"
        )
        # The exported names must be the real implementations, not shadows.
        from dl_techniques.models.distilbert.model import (
            DistilBERT as _cls,
            create_distilbert_with_head as _fn,
        )
        assert pkg.DistilBERT is _cls
        assert pkg.create_distilbert_with_head is _fn
        assert sorted(pkg.__all__) == [
            "DistilBERT", "create_distilbert_with_head"
        ], pkg.__all__
        # SC-4: the deleted private class must not be resurrected as an export.
        assert not hasattr(pkg, "DistilBertEmbeddings"), (
            "DistilBertEmbeddings was deleted; the package must not export it"
        )


class TestSinusoidalPositions:
    """SC-2. The branch that crashed under `mixed_float16` at HEAD."""

    def test_sinusoidal_float32_forward_and_structure(self):
        assert keras.mixed_precision.global_policy().name == "float32", (
            "environment precondition: a sibling test leaked a global policy"
        )
        keras.utils.set_random_seed(SEED)
        model = _model(sinusoidal_pos_embds=True)

        emb = model.embeddings
        assert emb.position_embedding_type == "sinusoidal", (
            "sinusoidal_pos_embds=True did not reach the embedding layer "
            f"(got position_embedding_type={emb.position_embedding_type!r})"
        )
        assert emb.position_embeddings is None

        # Constant token ids: self-attention is permutation-equivariant, so the
        # output rows can only differ through the POSITIONAL term. This is what
        # makes the assertion non-vacuous -- a shape/finiteness check cannot see
        # a positional term that has been zeroed out.
        const_ids = np.full((1, 12), 5, dtype="int32")
        out = model(const_ids, training=False)["last_hidden_state"]
        arr = _np(out)
        assert keras.backend.standardize_dtype(out.dtype) == "float32"
        assert bool(np.all(np.isfinite(arr)))
        assert not np.allclose(arr[0, 0], arr[0, 1], atol=1e-5), (
            "output rows of a constant-token sequence are identical -- the "
            "sinusoidal positional term is not reaching the output"
        )
        assert not np.allclose(arr[0, 0], arr[0, -1], atol=1e-5)

        # DECISION plan-2026-08-10-b007f435/D-013
        # Weight paths are read AFTER the forward pass, never before: an
        # unbuilt keras.Model reports `weights == []`, so the same assertion
        # placed above the call is VACUOUS (measured this step, mutation M4c3 --
        # a real shadow `position_embeddings` weight left it green).
        paths = [w.path for w in model.weights]
        assert any("word_embeddings" in p for p in paths), (
            "control: the model is not built, so a weight-path assertion here "
            f"cannot see anything; paths={paths}"
        )
        assert not any("position_embeddings" in p for p in paths), (
            "a LEARNED position table was allocated despite "
            f"sinusoidal_pos_embds=True: {paths}"
        )

    def test_sinusoidal_mixed_float16_forward(self):
        assert keras.mixed_precision.global_policy().name == "float32", (
            "environment precondition: a sibling test leaked a global policy"
        )
        previous = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            keras.utils.set_random_seed(SEED)
            model = _model(sinusoidal_pos_embds=True)
            # At HEAD this raised InvalidArgumentError: the float32 sin/cos
            # table was summed with a float16 word embedding (D-008).
            out = model(_tokens(), training=False)["last_hidden_state"]
            assert keras.backend.standardize_dtype(out.dtype) == "float16", (
                "mixed_float16 forward did not produce a float16 output"
            )
            assert bool(np.all(np.isfinite(_np(out)))), (
                "mixed_float16 sinusoidal forward produced NaN/Inf"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
        assert keras.mixed_precision.global_policy().name == "float32"


class TestNormalizationType:

    def test_rms_norm_forward_and_round_trip(self, tmp_path):
        keras.utils.set_random_seed(SEED)
        model = _model(normalization_type="rms_norm")

        assert type(model.embeddings.layer_norm).__name__ == "RMSNorm", (
            "normalization_type='rms_norm' did not reach the embedding stage; "
            f"got {type(model.embeddings.layer_norm).__name__}"
        )
        x = _tokens()
        before = _np(model(x, training=False)["last_hidden_state"])
        assert bool(np.all(np.isfinite(before)))

        path = os.path.join(str(tmp_path), "rms.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert type(loaded.embeddings.layer_norm).__name__ == "RMSNorm", (
            "the reloaded model's embedding normalization is not RMSNorm; "
            "normalization_type did not survive the .keras round trip"
        )
        after = _np(loaded(x, training=False)["last_hidden_state"])
        np.testing.assert_allclose(
            before, after, atol=1e-6,
            err_msg="rms_norm DistilBERT differs after a .keras round trip",
        )

    def test_unknown_normalization_type_raises_from_the_embedding_stage(self):
        """SC-3 (re-scoped iter-1).

        `ValueError` alone is VACUOUS: at HEAD, before this plan, an unknown
        type already raised -- from the first `TransformerLayer`, AFTER the
        embeddings had silently degraded to `LayerNormalization`. The guard
        therefore asserts WHERE the refusal comes from and that the message
        enumerates `BertEmbeddings`' four accepted types, not the
        transformer's eighteen.
        """
        with pytest.raises(ValueError) as excinfo:
            _model(normalization_type="definitely_not_a_norm")

        # DECISION plan-2026-08-10-b007f435/D-013
        # Assertion order is load-bearing: the "not from a TransformerLayer"
        # check must be first, because the pre-plan defect (embeddings degrade
        # silently, the encoder raises later) satisfies neither and would
        # otherwise be attributed to the wrong assertion.
        files = _traceback_files(excinfo.value)
        assert not any(
            os.path.join("layers", "transformers") in f for f in files
        ), (
            "the raise came from a TransformerLayer, which means the embedding "
            "stage accepted the unknown type and silently degraded first "
            f"(SC-3); traceback files: {files}"
        )
        assert any(os.path.join("layers", "embedding") in f for f in files), (
            "the unknown-normalization_type raise did not come from the "
            f"embedding stage; traceback files: {files}"
        )

        message = str(excinfo.value)
        for accepted in ("layer_norm", "rms_norm", "band_rms", "batch_norm"):
            assert accepted in message, (
                "the error message does not enumerate BertEmbeddings' four "
                f"accepted normalization types (missing {accepted!r}): {message}"
            )
        assert "definitely_not_a_norm" in message


class TestMaskingContract:

    def test_no_keras_auto_mask_is_acquired(self):
        """I-4.

        A forward-output comparison is PROVABLY BLIND to this: mutation M-A in
        `findings/step6-equivalence.md` deleted `mask_zero=False` at the call
        site and every output sha stayed bit-identical, with no `_keras_mask`
        anywhere (`LayerNormalization`/`Dropout` do not forward it on the eager
        path). Only the structural flag catches it.
        """
        model = _model()
        emb = model.embeddings

        assert emb.mask_zero is False, (
            "the embedding layer was built with mask_zero=True; DistilBERT "
            "threads an explicit attention_mask and must not also acquire a "
            "Keras auto-mask (I-4)"
        )
        assert emb.word_embeddings.mask_zero is False, (
            "word_embeddings emits a Keras auto-mask despite the layer's "
            "mask_zero=False"
        )

        ids = _padded_tokens()
        embedding_output = emb(input_ids=ids, training=False)
        assert getattr(embedding_output, "_keras_mask", None) is None, (
            "the embedding output carries a _keras_mask; a second, "
            "uncoordinated masking mechanism has appeared (I-4)"
        )
        model_output = model(ids, training=False)["last_hidden_state"]
        assert getattr(model_output, "_keras_mask", None) is None, (
            "the model output carries a _keras_mask (I-4)"
        )
        # Framework identity given the flag (Embedding.compute_mask returns
        # None iff mask_zero is False); kept as executable documentation of
        # what the flag means downstream.
        assert emb.word_embeddings.compute_mask(ids) is None

    def test_attention_mask_is_never_derived_from_pad_token_id(self):
        """D-003, and the dict-input `attention_mask` path.

        Assertion order matters: the "no mask is derived" equality is checked
        BEFORE the "an explicit mask changes the answer" inequality, because a
        mutation that auto-derives a mask would satisfy neither and the
        inequality would shadow the equality.
        """
        # DECISION plan-2026-08-10-b007f435/D-013
        # Do NOT reorder the three assertions below. Measured this step: the
        # pad_token_id-derivation mutation (M14) reds ONLY the equality, and
        # the mask-never-reaches-the-encoder mutation (M15) reds ONLY the
        # inequality. Swapping them leaves one of the two proven zero times.
        keras.utils.set_random_seed(SEED)
        model = _model()
        ids = _padded_tokens()
        pad_mask = (ids != model.pad_token_id).astype("int32")
        assert pad_mask.min() == 0 and pad_mask.max() == 1, "setup: need real padding"

        no_mask = _np(model({"input_ids": ids}, training=False)["last_hidden_state"])
        ones = _np(model(
            {"input_ids": ids, "attention_mask": np.ones_like(pad_mask)},
            training=False,
        )["last_hidden_state"])
        masked = _np(model(
            {"input_ids": ids, "attention_mask": pad_mask}, training=False
        )["last_hidden_state"])

        # (b) Omitting the mask must be exactly the all-attend answer, i.e. no
        # mask is inferred from pad_token_id. Measured residual: 0.0.
        np.testing.assert_allclose(
            no_mask, ones, atol=1e-6,
            err_msg=(
                "a mask-less forward pass differs from an all-ones-mask one -- "
                "a mask is being derived from pad_token_id, which D-003 "
                "forbids"
            ),
        )
        # (a) The explicit mask must actually reach the attention stack.
        # Measured max|diff| at this config: 0.0287.
        assert np.abs(masked - no_mask).max() > 1e-4, (
            "an explicit attention_mask carried on a dict input does not "
            "change the output -- it is not reaching the encoder layers"
        )
        # (c) The mask is passed through unchanged for downstream heads.
        out = model({"input_ids": ids, "attention_mask": pad_mask}, training=False)
        np.testing.assert_array_equal(_np(out["attention_mask"]), pad_mask)
        assert model({"input_ids": ids}, training=False)["attention_mask"] is None


class TestHeadIntegration:

    def test_create_distilbert_with_head_end_to_end(self):
        from dl_techniques.models.distilbert import create_distilbert_with_head
        from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

        num_classes = 3
        task = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.TEXT_CLASSIFICATION,
            num_classes=num_classes,
        )
        keras.utils.set_random_seed(SEED)
        model = create_distilbert_with_head(
            "tiny",
            task,
            distilbert_config_overrides=dict(
                vocab_size=256, max_position_embeddings=64, num_layers=1
            ),
        )

        assert sorted(model.input.keys()) == ["attention_mask", "input_ids"], (
            "the assembled model's input signature changed; "
            f"got {sorted(model.input.keys())}"
        )
        encoders = [l for l in model.layers if isinstance(l, DistilBERT)]
        assert len(encoders) == 1, [type(l).__name__ for l in model.layers]
        assert encoders[0].vocab_size == 256, (
            "distilbert_config_overrides did not reach the encoder "
            f"(vocab_size={encoders[0].vocab_size}, expected the override 256)"
        )

        ids = _padded_tokens()
        mask = (ids != 0).astype("int32")
        out = model.predict({"input_ids": ids, "attention_mask": mask}, verbose=0)

        assert isinstance(out, dict) and sorted(out) == ["logits", "probabilities"], out
        assert out["logits"].shape == (ids.shape[0], num_classes), out["logits"].shape
        assert np.all(np.isfinite(out["logits"]))
        np.testing.assert_allclose(
            out["probabilities"].sum(axis=-1), 1.0, atol=1e-5,
            err_msg="classification head probabilities do not sum to 1",
        )


class TestSerialization:

    NON_DEFAULT = dict(
        sinusoidal_pos_embds=True,
        normalization_type="rms_norm",
        layer_norm_eps=1e-9,
        pad_token_id=7,
        dropout_rate=0.0,
    )

    def test_get_config_from_config_carries_the_non_default_keys(self):
        keras.utils.set_random_seed(SEED)
        model = _model(**self.NON_DEFAULT)
        config = model.get_config()

        for key, value in self.NON_DEFAULT.items():
            assert key in config, f"'{key}' is missing from get_config()"
            assert config[key] == value, (
                f"get_config()['{key}'] == {config[key]!r}, expected {value!r}"
            )

        rebuilt = DistilBERT.from_config(config)
        for key, value in self.NON_DEFAULT.items():
            assert rebuilt.get_config()[key] == value, (
                f"'{key}' did not survive from_config()"
            )
        # The config values must reach the BUILT sub-layers, not just be stored.
        assert rebuilt.embeddings.position_embedding_type == "sinusoidal"
        assert type(rebuilt.embeddings.layer_norm).__name__ == "RMSNorm"
        assert rebuilt.embeddings.layer_norm.epsilon == 1e-9, (
            "layer_norm_eps did not reach the rebuilt embedding normalization "
            f"(I-2); got {rebuilt.embeddings.layer_norm.epsilon}"
        )

    def test_keras_round_trip_preserves_weight_values_and_inference_output(
        self, tmp_path
    ):
        """I-3.

        Weight VALUES and a `training=False` forward output, never shapes or
        parameter counts: a nested/lazy sub-layer store can match on count,
        path and total params while restoring fresh kernels.
        """
        keras.utils.set_random_seed(SEED)
        model = _model(**self.NON_DEFAULT)
        x = _tokens()
        before = _np(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "distilbert_nondefault.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        # Paths are compared with the model-name root stripped: the reloaded
        # model gets its own root (`distil_bert/...` vs `embeddings/...`), which
        # is a naming artifact, not a weight-set difference.
        def _key(weight, owner):
            prefix = owner.name + "/"
            path = weight.path
            return path[len(prefix):] if path.startswith(prefix) else path

        original = {_key(w, model): _np(w) for w in model.weights}
        restored = {_key(w, loaded): _np(w) for w in loaded.weights}
        assert sorted(original) == sorted(restored), (
            f"weight path set changed across the round trip: "
            f"{sorted(set(original) ^ set(restored))}"
        )
        assert len(original) > 0
        for wpath, value in original.items():
            np.testing.assert_array_equal(
                value, restored[wpath],
                err_msg=f"weight VALUES differ after the round trip: {wpath}",
            )

        after = _np(loaded(x, training=False)["last_hidden_state"])
        np.testing.assert_allclose(
            before, after, atol=1e-6,
            err_msg=(
                "training=False forward output differs after the .keras round "
                "trip (training=None is NOT inference in this repo)"
            ),
        )
        # Structural settings must survive too -- they are invisible to the
        # numeric comparison above (findings/step6-equivalence.md, M-A).
        assert loaded.embeddings.mask_zero is False
        assert loaded.embeddings.word_embeddings.mask_zero is False
        assert loaded.embeddings.position_embedding_type == "sinusoidal"
        assert type(loaded.embeddings.layer_norm).__name__ == "RMSNorm"


class TestPretrainedIsBroken:
    """D-012: pins the MEASURED behaviour. This is NOT an endorsement.

    `from_variant(pretrained=<path>)` raises on BOTH routes (unbuilt:
    `keras.random.uniform(..., dtype='int32')`; built: `load_weights(...,
    by_name=True)`), so the whole weight-loading surface is non-functional.
    The user decision at step 8 was to document, not fix. This test fails
    loudly the day someone fixes it, which is the point.
    """

    def test_from_variant_with_a_weights_path_still_raises(self, tmp_path):
        keras.utils.set_random_seed(SEED)
        model = DistilBERT.from_variant(
            "tiny", vocab_size=256, max_position_embeddings=64
        )
        model(_tokens(), training=False)
        path = os.path.join(str(tmp_path), "weights.keras")
        model.save(path)

        with pytest.raises(ValueError) as excinfo:
            DistilBERT.from_variant(
                "tiny", vocab_size=256, max_position_embeddings=64,
                pretrained=path,
            )
        message = str(excinfo.value)
        assert "requires a floating point" in message, (
            "load_pretrained_weights no longer fails the way D-012 measured; "
            f"re-measure and update D-012 before changing this test: {message}"
        )

        # The documented working route, verified in the same test so the
        # replacement advice cannot rot: keras.models.load_model.
        restored = keras.models.load_model(path)
        for a, b in zip(model.weights, restored.weights):
            np.testing.assert_array_equal(_np(a), _np(b), err_msg=a.path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
