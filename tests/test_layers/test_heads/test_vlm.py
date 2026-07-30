"""Tests for the merged ``heads.vlm`` sub-package.

The old ``vlm_heads`` package had ZERO test coverage and a broken (empty)
``__init__.py``; the merge populated ``heads/vlm/__init__.py``. This file asserts
import + Keras registration of all 6 VLM classes, constructs heads via
``create_vlm_head``, and (as of the bugfix plan) exercises the full forward pass
+ a ``.keras`` round-trip for the three previously-broken heads.

The three forward-pass-fixed heads are locked here:

  * ``ImageTextMatchingHead`` — Bug 1 (``ops.l2_normalize`` -> ``ops.normalize``)
    + the fine-grained-fusion 3-D blocker (D-001). Its fusion path requires
    ``vision_dim == text_dim == hidden_size``, so the tests feed ``DIM`` features
    with ``hidden_size=DIM``.
  * ``ImageCaptioningHead`` — Bug 2 (causal mask shape/polarity) + the
    cross-attention type/kwarg blocker. Returns ``logits`` of ``(B, S, vocab)``.
  * ``VQAHead`` — Bug 4 (shared cross-attention type/kwarg). Its DEFAULT
    ``pooling_strategy="attention"`` reuses one cross-attention layer in both
    directions, so the tests use ``vision_dim == text_dim``.
"""

import json
import linecache
import logging
import os
import tempfile

import numpy as np
import pytest
import keras
# `tf` is used ONLY by `TestSite1PostFusionWidthAssertion`'s site-2 guard, for the
# device-invariant exception type and the GPU-presence check (D-030).
import tensorflow as tf
from keras import ops

from dl_techniques.layers.ffn.factory import (
    FFN_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
    create_ffn_layer as _factory_create_ffn_layer,
)
from dl_techniques.layers.heads.vlm import factory as vlm_factory
from dl_techniques.layers.heads.vlm.factory import (
    _accepted_constructor_kwargs,
    _is_single_shape,
)
from dl_techniques.layers.heads.vlm import (
    VLMTaskType,
    VLMTaskConfig,
    create_vlm_head,
    create_multi_task_vlm_head,
    BaseVLMHead,
    ImageCaptioningHead,
    VQAHead,
    VisualGroundingHead,
    ImageTextMatchingHead,
    MultiTaskVLMHead,
)

DIM = 32
B, S = 3, 7
VOCAB = 50
NUM_HEADS = 4
NUM_CLASSES = 11


# ---------------------------------------------------------------------
# Import + registration of all 6 VLM classes
# ---------------------------------------------------------------------

class TestVLMRegistration:

    @pytest.mark.parametrize("name", [
        "BaseVLMHead",
        "ImageCaptioningHead",
        "VQAHead",
        "VisualGroundingHead",
        "ImageTextMatchingHead",
        "MultiTaskVLMHead",
    ])
    def test_class_registered(self, name) -> None:
        assert keras.saving.get_registered_object(f"Custom>{name}") is not None


# ---------------------------------------------------------------------
# Factory dispatch + construction smoke
# ---------------------------------------------------------------------

class TestVLMFactoryConstruction:

    def test_image_captioning_head_constructs(self) -> None:
        cfg = VLMTaskConfig(
            name="cap",
            task_type=VLMTaskType.IMAGE_CAPTIONING,
            vocab_size=VOCAB,
            hidden_size=DIM,
        )
        head = create_vlm_head(
            cfg, vision_dim=DIM, text_dim=DIM, num_layers=1, num_heads=NUM_HEADS
        )
        assert isinstance(head, ImageCaptioningHead)
        assert isinstance(head, keras.layers.Layer)
        assert head.num_layers == 1
        assert head.num_heads == NUM_HEADS

    def test_image_text_matching_head_constructs(self) -> None:
        cfg = VLMTaskConfig(
            name="itm",
            task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
            hidden_size=DIM,
        )
        head = create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)
        assert isinstance(head, ImageTextMatchingHead)
        assert isinstance(head, keras.layers.Layer)

    def test_factory_from_dict_config(self) -> None:
        head = create_vlm_head(
            {
                "name": "itm",
                "task_type": VLMTaskType.IMAGE_TEXT_MATCHING,
                "hidden_size": DIM,
            },
            vision_dim=DIM,
            text_dim=DIM,
        )
        assert isinstance(head, ImageTextMatchingHead)


# ---------------------------------------------------------------------
# Forward-pass builders + dummy inputs
# ---------------------------------------------------------------------

def _captioning_head() -> ImageCaptioningHead:
    cfg = VLMTaskConfig(
        name="cap",
        task_type=VLMTaskType.IMAGE_CAPTIONING,
        vocab_size=VOCAB,
        hidden_size=DIM,
    )
    return create_vlm_head(
        cfg, vision_dim=DIM, text_dim=DIM, num_layers=1, num_heads=NUM_HEADS
    )


def _vqa_head() -> VQAHead:
    cfg = VLMTaskConfig(
        name="vqa",
        task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
        hidden_size=DIM,
        num_classes=NUM_CLASSES,
    )
    # DEFAULT pooling_strategy="attention"; equal dims for the shared cross-attn.
    return create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)


def _itm_head() -> ImageTextMatchingHead:
    cfg = VLMTaskConfig(
        name="itm",
        task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
        hidden_size=DIM,  # D == hidden_size required by the fusion path.
    )
    return create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)


@pytest.fixture
def vision_feats() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((B, S, DIM)).astype("float32")


@pytest.fixture
def text_feats() -> np.ndarray:
    rng = np.random.default_rng(8)
    return rng.standard_normal((B, S, DIM)).astype("float32")


# ---------------------------------------------------------------------
# SC3 / SC4 / SC1 — forward pass on the three fixed heads
# ---------------------------------------------------------------------

class TestVLMForwardPass:

    def test_image_captioning_forward(self, vision_feats, text_feats) -> None:
        head = _captioning_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        })
        assert "logits" in out
        assert tuple(out["logits"].shape) == (B, S, VOCAB)

    def test_vqa_forward_default_attention_pooling(
        self, vision_feats, text_feats
    ) -> None:
        head = _vqa_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "question_features": ops.convert_to_tensor(text_feats),
        })
        assert "answer_logits" in out
        assert tuple(out["answer_logits"].shape) == (B, NUM_CLASSES)

    def test_image_text_matching_forward(self, vision_feats, text_feats) -> None:
        head = _itm_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        })
        # Full output dict contract (SC1).
        assert set(out.keys()) == {
            "similarity_matrix",
            "logits",
            "match_score",
            "vision_embeddings",
            "text_embeddings",
        }
        assert tuple(out["similarity_matrix"].shape) == (B, B)
        assert tuple(out["logits"].shape) == (B, B)
        assert tuple(out["match_score"].shape) == (B,)
        assert np.all(np.isfinite(ops.convert_to_numpy(out["similarity_matrix"])))

    def test_image_captioning_causal_property(
        self, vision_feats, text_feats
    ) -> None:
        """SC3 (optional): perturbing the LAST text position must not change the
        logits at earlier positions (the causal mask blocks future leakage).

        # DECISION plan-2026-07-30T140922-8af1028f/D-026
        MEASURED ON GPU (2026-07-30, RTX 4070, cc 8.9), the three regimes that
        can differ for a float32 matmul stack in this repo:

            TF32 ON, file-scoped (the GPU default) : max|diff| = 0.000000e+00
            TF32 forced OFF                        : max|diff| = 0.000000e+00
            co-collected behind
            test_attention/test_linear_attention.py: max|diff| = 0.000000e+00
              (at the time of measurement that file's import-time, unrestored
               `enable_tensor_float_32_execution(False)` really did leak --
               TF32 was observed False here while the file-scoped run observed
               True, so the regime genuinely changed and the number did not.
               That leak has since been scoped away by the `tf32_disabled`
               fixture in `tests/test_layers/conftest.py` (D-031), so a
               co-collected run now sees the ambient default here -- which is
               the FIRST row above, also 0.0. All three regimes were measured;
               removing the leak cannot move a number that is exactly zero in
               every one of them.)

        So `atol=1e-5` below is NOT a bare epsilon riding on ambient TF32 state.
        The masked positions are bit-identical: an additive -1e9 mask makes the
        future token's softmax weight exactly 0.0 in float32, and adding 0.0 is
        exact, so the earlier positions' reduction is unchanged bit-for-bit
        whatever precision the matmul uses.

        DO NOT "fix" this by porting the `_TF32_ULP = 2**-11` ulp-relative bound
        and the capture/restore/assert harness from
        `test_transformers/test_gated_linear_attention_block.py`. That was the
        recommended remedy before anyone measured; it is unnecessary here (the
        diff is 0.0, not merely small) and a toggle harness would add a
        process-global side effect to a test that provably does not need one.

        NOT VACUOUS, RED-proved the same day: replacing `attention_mask=causal_mask`
        with `attention_mask=None` in `ImageCaptioningHead.call`
        (`layers/heads/vlm/factory.py`) makes this assertion fail at
        max|diff| = 3.27e+00 (TF32 on) / 3.03e+00 (TF32 off), i.e. the defect
        signal is ~5 orders of magnitude above `atol`, against a noise floor of
        exactly zero.
        """
        head = _captioning_head()
        base = {
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        }
        logits0 = ops.convert_to_numpy(head(base)["logits"])

        perturbed = text_feats.copy()
        perturbed[:, S - 1, :] += 5.0  # disturb only the final position's input
        logits1 = ops.convert_to_numpy(head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(perturbed),
        })["logits"])

        # Positions 0..S-2 must be unchanged; only the last position may differ.
        np.testing.assert_allclose(
            logits0[:, : S - 1, :], logits1[:, : S - 1, :], atol=1e-5
        )


# ---------------------------------------------------------------------
# SC5 — .keras save/load round-trip of ImageCaptioningHead
# ---------------------------------------------------------------------

class TestVLMRoundtrip:

    def test_image_captioning_roundtrip(self, vision_feats, text_feats) -> None:
        """A `.keras` round-trip must restore the head from ITS OWN config, with
        identical values.

        REPAIRED twice over:

        1. It asserted SHAPES only, which a round-trip that rebuilt the
           architecture but dropped every learned weight also satisfies.
        2. More seriously, it wrapped the head in a subclassed `keras.Model`
           whose `__init__` called `_captioning_head()` afresh. The head was
           therefore reconstructed FROM CODE on load, never from its serialized
           config -- so the test could not detect head-config lossiness at all.
           Proved: making `ImageCaptioningHead.get_config` emit `num_heads: 1`
           instead of the real value left the old test green.

        It now builds a FUNCTIONAL model, so the head really is serialized and
        revived through `get_config`/`from_config`, and asserts VALUE equality.
        """
        vf = ops.convert_to_tensor(vision_feats)
        tf_ = ops.convert_to_tensor(text_feats)

        v_in = keras.Input(shape=tuple(vf.shape[1:]), name="vision_features")
        t_in = keras.Input(shape=tuple(tf_.shape[1:]), name="text_features")
        head = _captioning_head()
        outputs = head({"vision_features": v_in, "text_features": t_in})
        model = keras.Model({"vision_features": v_in, "text_features": t_in}, outputs)

        inputs = {"vision_features": vf, "text_features": tf_}
        y0 = model(inputs)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cap_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(inputs)

        assert tuple(y1["logits"].shape) == tuple(y0["logits"].shape) == (B, S, VOCAB)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y1["logits"]),
            ops.convert_to_numpy(y0["logits"]),
            rtol=1e-6, atol=1e-6,
            err_msg=(
                "captioning logits changed across a .keras round-trip — the "
                "reloaded head is not the saved one"
            ),
        )


# ---------------------------------------------------------------------
# H8/H9 — get_config is JSON-serializable and from_config round-trips
# (the task_config dataclass holds a VLMTaskType enum)
# ---------------------------------------------------------------------

class TestVLMConfigRoundtrip:

    def _heads(self):
        return {
            "captioning": _captioning_head(),
            "vqa": _vqa_head(),
            "itm": _itm_head(),
            "grounding": create_vlm_head(
                VLMTaskConfig(name="grd", task_type=VLMTaskType.VISUAL_GROUNDING,
                              hidden_size=DIM),
                vision_dim=DIM, text_dim=DIM,
            ),
            "base": BaseVLMHead(
                task_config=VLMTaskConfig(name="base",
                                          task_type=VLMTaskType.VISUAL_DIALOGUE,
                                          hidden_size=DIM),
                vision_dim=DIM, text_dim=DIM,
            ),
        }

    @pytest.mark.parametrize("name", ["captioning", "vqa", "itm", "grounding", "base"])
    def test_config_is_json_serializable_and_reconstructs(self, name):
        head = self._heads()[name]
        config = head.get_config()
        # Must be JSON-safe (enum collapsed to its string value).
        json.dumps(config)
        rebuilt = type(head).from_config(config)
        assert isinstance(rebuilt, type(head))
        assert rebuilt.task_config.task_type == head.task_config.task_type

    def test_multitask_config_round_trip(self):
        mt = create_multi_task_vlm_head(
            [VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                           hidden_size=DIM)],
            shared_vision_dim=DIM, shared_text_dim=DIM,
        )
        config = mt.get_config()
        json.dumps(config)
        rebuilt = MultiTaskVLMHead.from_config(config)
        assert list(rebuilt.task_heads) == list(mt.task_heads)


# ---------------------------------------------------------------------
# H7 — compute_output_shape mirrors call() outputs
# ---------------------------------------------------------------------

class TestVLMComputeOutputShape:

    def test_captioning_shape(self):
        head = _captioning_head()
        shapes = head.compute_output_shape(
            {"vision_features": (B, S, DIM), "text_features": (B, S, DIM)}
        )
        assert shapes["logits"] == (B, S, VOCAB)

    def test_vqa_shape(self):
        head = _vqa_head()
        shapes = head.compute_output_shape(
            {"vision_features": (B, S, DIM), "question_features": (B, S, DIM)}
        )
        assert shapes["answer_logits"] == (B, NUM_CLASSES)

    def test_itm_shape(self, vision_feats, text_feats):
        head = _itm_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        })
        shapes = head.compute_output_shape(
            {"vision_features": (B, S, DIM), "text_features": (B, S, DIM)}
        )
        for key in out:
            assert tuple(out[key].shape) == tuple(shapes[key])


# ---------------------------------------------------------------------
# Graph-safety of the causal mask (regression, `ops.tril` trap)
#
# `ImageCaptioningHead.call()` builds a causal mask. It used to do so with
# `keras.ops.tril`, which routes through a `tf.cond` that rejects a Python-bool
# predicate the moment it is traced:
#
#     TypeError: pred must not be a Python bool
#
# That failure is EAGER-INVISIBLE. A direct call works; every graph path fails
# (`tf.function`, `Model.predict` with a static or a dynamic sequence axis,
# `jit_compile=True`). Worse, Keras downgrades a `call()` crash during
# build-tracing to a `UserWarning`, so before this test the whole VLM suite
# reported PASS while the exception text appeared twice in its own output.
# A green exit code was not evidence; these tests make it evidence.
#
# NOTE ON SCOPE: `.keras` save/load is deliberately NOT asserted here. Fixing
# the mask exposed a SECOND, unrelated defect that the crash had been masking --
# `ImageCaptioningHead` declares no `build()`, so a Functional-model round-trip
# reports "12 objects could not be loaded (<Dense name=kv, built=False>)".
# That is a serialization defect, not a mask defect, and is tracked separately;
# asserting it here would make this test fail for a reason it does not govern.
# ---------------------------------------------------------------------


class TestCaptioningCausalMaskGraphSafety:
    """The causal mask must survive tracing, not just eager execution."""

    @staticmethod
    def _inputs():
        rng = np.random.default_rng(7)
        return (
            rng.normal(size=(B, S, DIM)).astype("float32"),
            rng.normal(size=(B, S, DIM)).astype("float32"),
        )

    def test_eager_call_still_works(self) -> None:
        """Control: the path that was ALWAYS green must stay green.

        Without this, a fix that broke eager while fixing graph would look
        like a pass on the three tests below.
        """
        vf, tf_ = self._inputs()
        out = _captioning_head()(
            {"vision_features": ops.convert_to_tensor(vf),
             "text_features": ops.convert_to_tensor(tf_)}
        )
        assert tuple(out["logits"].shape) == (B, S, VOCAB)

    def test_traced_tf_function(self) -> None:
        """`tf.function` tracing — the minimal reproducer of the trap."""
        import tensorflow as tf

        head = _captioning_head()

        @tf.function
        def run(v, t):
            return head({"vision_features": v, "text_features": t})

        out = run(tf.constant(self._inputs()[0]), tf.constant(self._inputs()[1]))
        assert tuple(out["logits"].shape) == (B, S, VOCAB)

    def test_functional_model_predict_static_sequence(self) -> None:
        vf, tf_ = self._inputs()
        vi = keras.Input(shape=(S, DIM))
        ti = keras.Input(shape=(S, DIM))
        out = _captioning_head()({"vision_features": vi, "text_features": ti})
        model = keras.Model([vi, ti], out)
        assert model.predict([vf, tf_], verbose=0)["logits"].shape == (B, S, VOCAB)

    def test_functional_model_predict_dynamic_sequence(self) -> None:
        """A `None` sequence axis makes the mask size symbolic at trace time."""
        vf, tf_ = self._inputs()
        vi = keras.Input(shape=(None, DIM))
        ti = keras.Input(shape=(None, DIM))
        out = _captioning_head()({"vision_features": vi, "text_features": ti})
        model = keras.Model([vi, ti], out)
        assert model.predict([vf, tf_], verbose=0)["logits"].shape == (B, S, VOCAB)

    def test_jit_compiled(self) -> None:
        import tensorflow as tf

        head = _captioning_head()

        @tf.function(jit_compile=True)
        def run(v, t):
            return head({"vision_features": v, "text_features": t})

        out = run(tf.constant(self._inputs()[0]), tf.constant(self._inputs()[1]))
        assert tuple(out["logits"].shape) == (B, S, VOCAB)

    def test_causal_mask_reaching_self_attention_is_lower_triangular_keep_form(
        self,
    ) -> None:
        """The mask the head ACTUALLY HANDS to self-attention, not a rebuilt copy.

        REPAIRED: this test used to construct the mask from `MaskFactory` in its
        own body and assert on that. It therefore tested `MaskFactory`, not the
        head -- replacing the head's own mask expression with `ones_like` (attend
        to everything, including the future) left it green.

        It now spies on `self_attention_layers[0]` and asserts on the
        `attention_mask` value that arrives there: 1 = attend (key j <= query i),
        0 = future. Complementary to
        `test_image_captioning_causal_property`, which checks the end-to-end
        behaviour; this pins the polarity at the point of use, which is where the
        two possible mistakes (inverted polarity, dropped diagonal) live.
        """
        payload = _family_payload()
        head = _make_single(VLMTaskType.IMAGE_CAPTIONING, "captioning")
        head.build(_shapes_of(payload))

        captured = {}
        real_attn = head.self_attention_layers[0]

        def attn_spy(x, attention_mask=None, training=None):
            captured["mask"] = (
                None if attention_mask is None
                else ops.convert_to_numpy(attention_mask)
            )
            return real_attn(x, attention_mask=attention_mask, training=training)

        head.self_attention_layers[0] = attn_spy
        try:
            head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        finally:
            head.self_attention_layers[0] = real_attn

        assert "mask" in captured, "self-attention was never called"
        mask = captured["mask"]
        assert mask is not None, (
            "self-attention received attention_mask=None — the causal mask is "
            "not reaching the layer at all"
        )
        # (1, S, S) full-mask form; drop the broadcast batch axis.
        mask2d = mask[0] if mask.ndim == 3 else mask
        expected = np.tril(np.ones((_FAMILY_SEQ, _FAMILY_SEQ), dtype=mask2d.dtype))
        assert np.array_equal(mask2d, expected), (
            f"the mask handed to self-attention is not lower-triangular KEEP "
            f"form; got\n{mask2d}"
        )
        assert mask2d[0, 0] == 1.0, "diagonal dropped: a token cannot attend to itself"
        assert mask2d[0, 1] == 0.0, "future not masked"
        assert mask2d[-1].sum() == _FAMILY_SEQ, (
            "last query must attend to the whole prefix"
        )


# ---------------------------------------------------------------------
# Explicit sub-layer build (regression, lazy-sublayer serialization trap)
#
# `ImageCaptioningHead` declared no `build()`, so the sub-layers created in
# `__init__` stayed unbuilt until Keras traced `call()`. A `.keras` round-trip
# through a Functional model then could not restore them:
#
#     ValueError: A total of 12 objects could not be loaded.
#     Example error message for object <Dense name=kv, built=False>
#
# This was masked twice over. First by the `ops.tril` graph-mode trap, which
# crashed `call()` before the loader got this far. Then by the head's own
# round-trip test, which compares only output SHAPES after reload — and a model
# that restored NO weights still emits correctly-shaped output. The tests below
# assert VALUES across the save/load boundary, which is the only assertion that
# can tell a restored model from an empty one.
# ---------------------------------------------------------------------


class TestCaptioningExplicitBuild:
    """Sub-layers must be built by `build()`, not lazily by tracing `call()`."""

    @staticmethod
    def _shapes():
        return {"vision_features": (None, S, DIM), "text_features": (None, S, DIM)}

    @staticmethod
    def _inputs():
        rng = np.random.default_rng(17)
        return (rng.normal(size=(B, S, DIM)).astype("float32"),
                rng.normal(size=(B, S, DIM)).astype("float32"))
    # CONSOLIDATED (F-18): three tests removed from here --
    #   test_build_creates_weights_without_a_forward_pass
    #   test_explicit_build_is_numerically_inert
    #   test_functional_round_trip_preserves_VALUES
    # `TestVLMHeadFamilyExplicitBuild` below runs all three checks over all four
    # heads INCLUDING captioning, so these were captioning-only duplicates. One of
    # them also carried the `len(head.weights) > 0` form that this file's own
    # comments flag as insufficient (captioning satisfies it with `build()`
    # deleted, because `temperature` is created by `add_weight` in `__init__`).
    # Verified by measurement, not by inspection -- and the measurement is NOT
    # a clean wash, so it is recorded honestly: a no-op `ImageCaptioningHead.
    # build()` failed 8 tests before this removal and 6 after. Detection is
    # retained (6 >> 0) but is measurably WEAKER, because the family harness
    # exercises captioning at `_FAMILY_DIM=96` with one fusion strategy while the
    # removed duplicates used this section's own config. The two dead-component
    # probes this file's guards exist for -- captioning ignoring `text_features`,
    # and `gated_linear_scan` returning zeros -- were re-run after consolidation
    # and both still fail (2 and 12 tests respectively), which is the gate that
    # mattered. `test_round_trip_restores_every_weight` below is NOT a duplicate
    # and stays.

    def test_round_trip_restores_every_weight(self) -> None:
        """Weight-level check, independent of the forward pass.

        Value equality could in principle be reached with a subset of weights
        restored if the rest happened not to matter for this input. Compare the
        weight tensors themselves.
        """
        vf, tf_ = self._inputs()
        vi = keras.Input(shape=(S, DIM))
        ti = keras.Input(shape=(S, DIM))
        out = _captioning_head()({"vision_features": vi, "text_features": ti})
        model = keras.Model([vi, ti], out)
        model.predict([vf, tf_], verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cap_weights.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        before = model.get_weights()
        after = restored.get_weights()
        assert len(before) == len(after) and len(before) > 0
        for i, (w0, w1) in enumerate(zip(before, after)):
            np.testing.assert_allclose(
                w0, w1, rtol=1e-6, atol=1e-6,
                err_msg=f"weight {i} (shape {w0.shape}) not restored",
            )


# ---------------------------------------------------------------------
# Explicit sub-layer build across the WHOLE VLM head family
#
# `BaseVLMHead.build()` only calls `super().build()`, and none of the five
# subclasses built their sub-layers. They were created in `__init__` and left
# unbuilt until Keras traced `call()`, so a Functional-model `.keras` round-trip
# could not restore them ("A total of N objects could not be loaded").
#
# Each head is checked three ways, because each catches a different failure:
#   1. `build()` alone must create weights   -> catches a missing/no-op build()
#   2. explicit build must be numerically INERT vs the lazy path, at the same
#      weight count -> catches a build() that builds the WRONG shapes, or builds
#      sub-layers `call()` never uses (which would inflate the checkpoint)
#   3. a round-trip must preserve VALUES     -> catches the original defect
#
# Check 2 is the one that constrains the fix rather than just asserting it ran:
# `VisualGroundingHead` never runs the post-fusion stack, so building it would
# add weights the lazy path never created and check 2 would fail.
# ---------------------------------------------------------------------

# Feature width 96: MultiTaskVLMHead's per-head defaults include num_heads=12,
# and hidden_dim must be divisible by it. (`task_specific_kwargs`, which would
# let a caller override that, is unusable -- MultiTaskVLMHead reads it out of
# **kwargs but forwards kwargs to Layer.__init__, which rejects it.)
_FAMILY_DIM = 96
_FAMILY_SEQ = 7
_FAMILY_BATCH = 3


def _family_payload(text_key: str = "text_features"):
    rng = np.random.default_rng(11)
    return {
        "vision_features": rng.normal(
            size=(_FAMILY_BATCH, _FAMILY_SEQ, _FAMILY_DIM)
        ).astype("float32"),
        text_key: rng.normal(
            size=(_FAMILY_BATCH, _FAMILY_SEQ, _FAMILY_DIM)
        ).astype("float32"),
    }


def _family_config(name: str, task_type: VLMTaskType) -> VLMTaskConfig:
    return VLMTaskConfig(
        name=name, task_type=task_type, vocab_size=VOCAB,
        hidden_size=_FAMILY_DIM, num_classes=NUM_CLASSES,
    )


# (id, task_type, text input key)
_SINGLE_HEADS = [
    ("captioning", VLMTaskType.IMAGE_CAPTIONING, "text_features"),
    ("vqa", VLMTaskType.VISUAL_QUESTION_ANSWERING, "question_features"),
    ("grounding", VLMTaskType.VISUAL_GROUNDING, "text_features"),
    ("matching", VLMTaskType.IMAGE_TEXT_MATCHING, "text_features"),
]


def _make_single(task_type: VLMTaskType, name: str):
    return create_vlm_head(
        _family_config(name, task_type),
        vision_dim=_FAMILY_DIM,
        text_dim=_FAMILY_DIM,
    )


def _flatten_outputs(out) -> np.ndarray:
    """Flatten a head's output (tensor, dict, or dict-of-dicts) to one vector."""
    if isinstance(out, dict):
        return np.concatenate(
            [_flatten_outputs(out[k]).ravel() for k in sorted(out)]
        )
    return ops.convert_to_numpy(out).ravel()


def _shapes_of(payload) -> dict:
    return {k: (None,) + v.shape[1:] for k, v in payload.items()}


class TestVLMHeadFamilyExplicitBuild:
    """Every VLM head must build its sub-layers in `build()`, not lazily."""

    @pytest.mark.parametrize("label,task_type,text_key", _SINGLE_HEADS)
    def test_build_creates_weights_without_a_forward_pass(
        self, label, task_type, text_key
    ) -> None:
        payload = _family_payload(text_key)
        head = _make_single(task_type, label)
        assert not head.built
        head.build(_shapes_of(payload))
        assert head.built

        # `len(weights) > 0` is NOT a sufficient assertion here: this was caught
        # by RED-proofing, where `ImageTextMatchingHead` satisfied it with
        # `build()` deleted, because its `temperature` scalar is created by
        # `add_weight` in `__init__` and exists unbuilt. The real invariant is
        # that build() creates EVERY weight a forward pass would, so compare
        # against a fresh instance driven by an actual call.
        reference = _make_single(task_type, label)
        reference({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        assert len(reference.weights) > 0, "the reference forward pass built nothing"
        assert len(head.weights) == len(reference.weights), (
            f"{label}: build() created {len(head.weights)} weights but a forward "
            f"pass creates {len(reference.weights)} — sub-layers are left unbuilt "
            f"and a .keras round-trip cannot restore them"
        )

    @pytest.mark.parametrize("label,task_type,text_key", _SINGLE_HEADS)
    def test_explicit_build_is_numerically_inert(
        self, label, task_type, text_key
    ) -> None:
        """Same weight count AND bit-identical output vs the lazy path.

        The weight-count half is what forbids building sub-layers that `call()`
        never runs: `VisualGroundingHead` skips the post-fusion stack entirely,
        so building it would show up here as extra weights.
        """
        payload = _family_payload(text_key)
        tensors = {k: ops.convert_to_tensor(v) for k, v in payload.items()}

        keras.utils.set_random_seed(5)
        explicit = _make_single(task_type, label)
        explicit.build(_shapes_of(payload))
        out_explicit = _flatten_outputs(explicit(tensors))

        keras.utils.set_random_seed(5)
        lazy = _make_single(task_type, label)
        out_lazy = _flatten_outputs(lazy(tensors))

        assert len(explicit.weights) == len(lazy.weights), (
            f"{label}: explicit build created {len(explicit.weights)} weights vs "
            f"{len(lazy.weights)} lazily — build() is building sub-layers that "
            f"call() does not use, or missing some that it does"
        )
        assert np.array_equal(out_explicit, out_lazy), (
            f"{label}: explicit build() changed the forward result; it must be inert"
        )

    @pytest.mark.parametrize("label,task_type,text_key", _SINGLE_HEADS)
    def test_functional_round_trip_preserves_values(
        self, label, task_type, text_key
    ) -> None:
        payload = _family_payload(text_key)
        inputs = {
            k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()
        }
        model = keras.Model(
            list(inputs.values()), _make_single(task_type, label)(inputs)
        )
        xs = [payload[k] for k in inputs]
        before = model.predict(xs, verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{label}.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        after = restored.predict(xs, verbose=0)

        np.testing.assert_allclose(
            _flatten_outputs(before), _flatten_outputs(after),
            rtol=1e-6, atol=1e-6,
            err_msg=f"{label}: values changed across a .keras round-trip — "
                    f"weights were not restored",
        )

    def test_multi_task_head_builds_and_round_trips(self) -> None:
        """The wrapper must build every task head it owns."""
        payload = _family_payload()
        cfgs = {
            "cap": _family_config("cap", VLMTaskType.IMAGE_CAPTIONING),
            "itm": _family_config("itm", VLMTaskType.IMAGE_TEXT_MATCHING),
        }
        mk = lambda: create_multi_task_vlm_head(
            cfgs, shared_vision_dim=_FAMILY_DIM, shared_text_dim=_FAMILY_DIM
        )

        head = mk()
        head.build(_shapes_of(payload))
        assert head.built and len(head.weights) > 0

        tensors = {k: ops.convert_to_tensor(v) for k, v in payload.items()}
        keras.utils.set_random_seed(7)
        explicit = mk()
        explicit.build(_shapes_of(payload))
        out_explicit = _flatten_outputs(explicit(tensors))
        keras.utils.set_random_seed(7)
        lazy = mk()
        out_lazy = _flatten_outputs(lazy(tensors))
        assert len(explicit.weights) == len(lazy.weights)
        assert np.array_equal(out_explicit, out_lazy)

        inputs = {
            k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()
        }
        model = keras.Model(list(inputs.values()), mk()(inputs))
        xs = [payload[k] for k in inputs]
        before = model.predict(xs, verbose=0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "multitask.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        np.testing.assert_allclose(
            _flatten_outputs(before),
            _flatten_outputs(restored.predict(xs, verbose=0)),
            rtol=1e-6, atol=1e-6,
            err_msg="multi-task: values changed across a .keras round-trip",
        )


class TestVisualGroundingForwardPass:
    """`VisualGroundingHead` was dead on its forward pass — pin that it runs.

    `call()` gathered the top-scoring region with NumPy-style fancy indexing,
    `fused[batch_indices, top_indices]`. TF tensors reject that outright, so the
    head raised `TypeError` on ANY call, eager included. It is absent from this
    module's list of forward-pass-verified heads, which is why nothing caught it.
    """

    def test_forward_pass_runs_and_returns_the_documented_keys(self) -> None:
        payload = _family_payload()
        head = _make_single(VLMTaskType.VISUAL_GROUNDING, "grounding")
        out = head({k: ops.convert_to_tensor(v) for k, v in payload.items()})

        assert "bbox" in out and "confidence" in out
        assert tuple(out["bbox"].shape) == (_FAMILY_BATCH, 4)
        assert tuple(out["confidence"].shape) == (_FAMILY_BATCH, _FAMILY_SEQ)
        assert np.isfinite(ops.convert_to_numpy(out["bbox"])).all()

    def test_top_region_gather_selects_the_argmax_region(self) -> None:
        """The gather must pick the region the confidence scorer ranked first.

        REPAIRED: this test used to recompute `argmax` + `take_along_axis` in its
        own body on hand-made arrays and never construct a `VisualGroundingHead`
        at all -- it exercised `keras.ops`, not the layer. Forcing the real gather
        to region 0 left it green, which is how it was caught.

        It now drives the REAL head: `confidence_scorer` is stubbed so the ranking
        is known, `fusion` is spied so the expected rows are the layer's own fused
        features, and the assertion is on the head's returned
        `grounded_features`. A gather that took region 0, or transposed the batch
        and region axes, now fails.
        """
        payload = _family_payload()
        head = _make_single(VLMTaskType.VISUAL_GROUNDING, "grd")
        head.build(_shapes_of(payload))

        # Known ranking, one distinct winner per batch row.
        chosen = [1, 0, 4] if _FAMILY_SEQ > 4 else list(range(_FAMILY_BATCH))
        scores = np.full((_FAMILY_BATCH, _FAMILY_SEQ), 0.1, dtype="float32")
        for row, region in enumerate(chosen):
            scores[row, region] = 0.9

        captured = {}
        real_fusion = head.fusion

        def fusion_spy(tensors, training=None):
            out = real_fusion(tensors, training=training)
            captured["fused"] = ops.convert_to_numpy(out)
            return out

        class _ScoreStub:
            """Emits the ranking above, shaped as the scorer's (B, N, 1)."""

            def __call__(self, _fused, *args, **kwargs):
                return ops.convert_to_tensor(scores[..., None])

        head.fusion = fusion_spy
        head.confidence_scorer = _ScoreStub()
        try:
            out = head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        finally:
            del head.fusion
            del head.confidence_scorer

        assert "fused" in captured, "the fusion layer was never called"
        fused = captured["fused"]
        grounded = ops.convert_to_numpy(out["grounded_features"])
        expected = np.stack(
            [fused[row, region] for row, region in enumerate(chosen)]
        )

        np.testing.assert_allclose(
            grounded, expected, rtol=1e-6, atol=1e-6,
            err_msg=(
                f"grounded_features are not the fused features of the "
                f"argmax regions {chosen} — the gather picked the wrong region"
            ),
        )
        # And the head must report the ranking it was given.
        np.testing.assert_allclose(
            ops.convert_to_numpy(out["confidence"]), scores, rtol=1e-6, atol=1e-6
        )


# ---------------------------------------------------------------------
# MultiTaskVLMHead.task_specific_kwargs (regression, dead documented feature)
#
# The parameter was documented on `create_multi_task_vlm_head` but unusable:
# `MultiTaskVLMHead.__init__` read it back out of `**kwargs` while also
# forwarding `kwargs` to `Layer.__init__`, which rejects unknown keys. So every
# call passing it raised `ValueError: Unrecognized keyword arguments`, and the
# per-task overrides were unreachable -- forcing every head onto the shared
# defaults (notably `num_heads=12`, so `hidden_size` had to be divisible by 12).
#
# The same `self.shared_head_kwargs = kwargs` line also leaked this layer's own
# Keras base arguments into every child constructor, where `name` collides with
# the head's auto-generated `f"{task_config.name}_head"`.
#
# The tests use hidden_size=32, which is deliberately NOT divisible by the
# default num_heads=12: construction can only succeed if the override actually
# reaches the head, so these tests cannot pass vacuously.
# ---------------------------------------------------------------------

_TSK_DIM = 32          # not divisible by the default num_heads=12 -- load-bearing
_TSK_OVERRIDE = {"cap": {"num_heads": 4, "num_layers": 1}}


def _tsk_configs(dim: int = _TSK_DIM):
    return {
        "cap": VLMTaskConfig(
            name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
            vocab_size=VOCAB, hidden_size=dim,
        ),
        "itm": VLMTaskConfig(
            name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
            vocab_size=VOCAB, hidden_size=dim,
        ),
    }


def _tsk_payload(dim: int = _TSK_DIM):
    rng = np.random.default_rng(21)
    return {
        "vision_features": rng.normal(size=(B, S, dim)).astype("float32"),
        "text_features": rng.normal(size=(B, S, dim)).astype("float32"),
    }


def _make_multi(dim: int = _TSK_DIM, **extra):
    return create_multi_task_vlm_head(
        _tsk_configs(dim), shared_vision_dim=dim, shared_text_dim=dim, **extra
    )


class TestMultiTaskTaskSpecificKwargs:
    """Per-task constructor overrides must actually reach the per-task heads."""

    def test_override_reaches_the_head(self) -> None:
        """Constructing at all is the proof.

        `_TSK_DIM` (32) is not divisible by the default `num_heads=12`, so if the
        override did not land, `ImageCaptioningHead.__init__` would raise
        "hidden_dim (32) must be divisible by num_heads (12)". The explicit
        attribute assertions then pin WHICH values arrived.
        """
        head = _make_multi(task_specific_kwargs=_TSK_OVERRIDE)
        assert head.task_heads["cap"].num_heads == 4
        assert head.task_heads["cap"].num_layers == 1

    def test_override_does_not_leak_to_other_tasks(self) -> None:
        """A per-task override must apply to THAT task only."""
        head = _make_multi(task_specific_kwargs=_TSK_OVERRIDE)
        itm = head.task_heads["itm"]
        # ImageTextMatchingHead takes no num_heads/num_layers at all; if the
        # override leaked it would have raised on construction. Assert the head
        # exists and is the expected class, so this is not a vacuous check.
        assert isinstance(itm, ImageTextMatchingHead)
        assert not hasattr(itm, "num_layers")

    def test_unknown_task_name_raises(self) -> None:
        """A typo'd task name must not silently apply to nothing."""
        with pytest.raises(ValueError, match="absent from task_configs"):
            _make_multi(96, task_specific_kwargs={"caption": {"num_heads": 4}})

    def test_override_survives_get_config_round_trip(self) -> None:
        head = _make_multi(task_specific_kwargs=_TSK_OVERRIDE)
        config = head.get_config()
        assert config["task_specific_kwargs"] == _TSK_OVERRIDE
        assert json.dumps(config["task_specific_kwargs"])  # JSON-serializable

        rebuilt = MultiTaskVLMHead.from_config(config)
        assert rebuilt.task_heads["cap"].num_heads == 4
        assert rebuilt.task_heads["cap"].num_layers == 1

    def test_override_survives_a_keras_round_trip(self) -> None:
        """Values preserved AND the override still present on the loaded layer."""
        payload = _tsk_payload()
        inputs = {
            k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()
        }
        model = keras.Model(
            list(inputs.values()),
            _make_multi(task_specific_kwargs=_TSK_OVERRIDE)(inputs),
        )
        xs = [payload[k] for k in inputs]
        before = model.predict(xs, verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "multitask_tsk.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        np.testing.assert_allclose(
            _flatten_outputs(before),
            _flatten_outputs(restored.predict(xs, verbose=0)),
            rtol=1e-6, atol=1e-6,
        )
        loaded = next(l for l in restored.layers if hasattr(l, "task_heads"))
        assert loaded.task_heads["cap"].num_heads == 4, (
            "the override was lost across the round-trip"
        )

    def test_keras_base_kwargs_do_not_leak_into_the_heads(self) -> None:
        """`name` belongs to this layer, not to its children.

        `self.shared_head_kwargs = kwargs` used to forward it into every head,
        where it collides with the head's own `f"{task_config.name}_head"`.
        """
        head = _make_multi(96, name="my_multitask_head")
        assert head.name == "my_multitask_head"
        # Each head keeps its OWN generated name, not the wrapper's.
        for task_name, task_head in head.task_heads.items():
            assert task_head.name != "my_multitask_head"
            assert task_name in task_head.name

    def test_shared_kwargs_still_reach_every_head(self) -> None:
        """Control: the shared path must keep working alongside overrides.

        Without this, a fix that dropped shared kwargs entirely would still pass
        every test above.
        """
        head = _make_multi(
            task_specific_kwargs={"cap": {"num_heads": 4, "num_layers": 1}},
            ffn_type="swiglu",
        )
        # `ffn_type` is one of the few kwargs EVERY head class here accepts
        # (ImageCaptioningHead declares it directly; ImageTextMatchingHead
        # inherits it from BaseVLMHead). See the note in
        # `MultiTaskVLMHead.__init__`: a shared kwarg must be accepted by every
        # head class, and `fusion_strategy` for instance would raise on
        # ImageCaptioningHead, which is not a BaseVLMHead subclass.
        assert head.shared_head_kwargs == {"ffn_type": "swiglu"}
        for task_head in head.task_heads.values():
            assert task_head.ffn_type == "swiglu"


# ---------------------------------------------------------------------
# ImageCaptioningHead x ffn_type (regression, missing hidden_dim)
#
# The head built its decoder FFN with
# `create_ffn_from_config({"type": ..., "output_dim": ..., "name": ...})` and no
# `hidden_dim`. 13 of the FFN registry's 21 types REQUIRE `hidden_dim`, so every
# one of those `ffn_type` values raised
# "Required parameters missing for mlp: ['hidden_dim']" and was unusable. Only
# the default `swiglu` (which lists it as optional and derives it) and the 7
# types that take no `hidden_dim` at all ever worked.
#
# `hidden_dim` is now passed CONDITIONALLY, driven by the registry. That
# conditionality is the load-bearing part and has its own control below:
# `swiglu` derives its hidden width internally (2/3 rule from
# `ffn_expansion_factor`, rounded to `ffn_multiple_of`), so passing `hidden_dim`
# unconditionally would override that derivation and silently change the DEFAULT
# configuration's widths.
# ---------------------------------------------------------------------

_HIDDEN_DIM_REQUIRING_FFN_TYPES = sorted(
    t for t, meta in FFN_REGISTRY.items()
    if "hidden_dim" in meta.get("required_params", ())
)


def _ffn_head(ffn_type: str, **extra) -> ImageCaptioningHead:
    cfg = VLMTaskConfig(
        name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
        vocab_size=VOCAB, hidden_size=DIM,
    )
    return ImageCaptioningHead(
        task_config=cfg, vision_dim=DIM, text_dim=DIM,
        num_layers=1, num_heads=NUM_HEADS, ffn_type=ffn_type, **extra
    )


def _ffn_inputs():
    rng = np.random.default_rng(1)
    return {
        "vision_features": ops.convert_to_tensor(
            rng.normal(size=(B, S, DIM)).astype("float32")
        ),
        "text_features": ops.convert_to_tensor(
            rng.normal(size=(B, S, DIM)).astype("float32")
        ),
    }


def _param_count(head) -> int:
    return int(sum(np.prod(w.shape) for w in head.weights))


class TestCaptioningFFNTypes:
    """Every FFN type requiring an explicit `hidden_dim` must be usable."""

    def test_the_registry_still_has_types_that_require_hidden_dim(self) -> None:
        """Guard the guard: if this list empties, the tests below go vacuous."""
        assert len(_HIDDEN_DIM_REQUIRING_FFN_TYPES) > 5, (
            "no FFN types require hidden_dim any more — the parametrized tests "
            "below would silently test nothing"
        )
        assert "mlp" in _HIDDEN_DIM_REQUIRING_FFN_TYPES

    @pytest.mark.parametrize("ffn_type", _HIDDEN_DIM_REQUIRING_FFN_TYPES)
    def test_forward_pass_works(self, ffn_type) -> None:
        head = _ffn_head(ffn_type)
        out = head(_ffn_inputs())
        assert tuple(out["logits"].shape) == (B, S, VOCAB)
        assert np.isfinite(ops.convert_to_numpy(out["logits"])).all()

    @pytest.mark.parametrize("ffn_type", _HIDDEN_DIM_REQUIRING_FFN_TYPES)
    def test_config_round_trip(self, ffn_type) -> None:
        head = _ffn_head(ffn_type)
        rebuilt = ImageCaptioningHead.from_config(head.get_config())
        assert rebuilt.ffn_type == ffn_type
        assert rebuilt.ffn_expansion_factor == head.ffn_expansion_factor

    def test_ffn_expansion_factor_widens_a_hidden_dim_requiring_ffn(self) -> None:
        """The factor must actually reach the FFN, not just be stored."""
        counts = []
        for factor in (2, 4, 8):
            head = _ffn_head("mlp", ffn_expansion_factor=factor)
            head(_ffn_inputs())
            counts.append(_param_count(head))
        assert counts[0] < counts[1] < counts[2], (
            f"ffn_expansion_factor did not widen the mlp FFN: {counts}"
        )

    def test_default_swiglu_ignores_ffn_expansion_factor(self) -> None:
        """The control that forbids passing `hidden_dim` unconditionally.

        `swiglu` derives its own hidden width, so its parameter count must be
        INVARIANT to this head's `ffn_expansion_factor`. If a future change
        passed `hidden_dim` to every FFN type, swiglu would start honouring the
        factor and this test would fail — which is exactly the silent change to
        the default configuration it exists to prevent.
        """
        counts = set()
        for factor in (2, 4, 8):
            head = _ffn_head("swiglu", ffn_expansion_factor=factor)
            head(_ffn_inputs())
            counts.add(_param_count(head))
        assert len(counts) == 1, (
            f"the default swiglu path responded to ffn_expansion_factor "
            f"({sorted(counts)}) — hidden_dim is being passed to an FFN type "
            f"that should derive it, changing the default configuration"
        )

    def test_ffn_expansion_factor_defaults_to_four(self) -> None:
        assert _ffn_head("swiglu").ffn_expansion_factor == 4

    def test_old_configs_without_the_new_key_still_load(self) -> None:
        """Backward compatibility: the parameter has a default.

        A config serialized before `ffn_expansion_factor` existed lacks the key;
        `from_config` must still work rather than raising a missing-argument
        error.
        """
        config = _ffn_head("swiglu").get_config()
        del config["ffn_expansion_factor"]
        rebuilt = ImageCaptioningHead.from_config(config)
        assert rebuilt.ffn_expansion_factor == 4


# ---------------------------------------------------------------------
# Output-WIDTH routing via FFN_REGISTRY['output_dim_param']
#
# Both VLM FFN sites used to hardcode `"output_dim": self.hidden_dim`. That key
# does not EXIST for 4 of the 21 registry types, which name their output width
# `filters` (gated_mlp), `features` (kan) or `units` (power_mlp, tversky). For
# them the hardcoded key was silently dropped by `create_ffn_layer`'s parameter
# filter and the type then died in `validate_ffn_config` on its own width
# parameter, which the head never supplied.
#
# The sites now look the width parameter's NAME up in the registry. The failure
# this section exists to catch is a "fix" that pattern-matches the literal string
# "output_dim" instead: it is a no-op for exactly those 4 types while looking
# identical to the correct routing, so every test would stay green and nothing
# would have been closed. `test_the_width_parameter_names_are_still_renamed` is
# the anti-vacuity guard on the derived list, and the `kan`/`power_mlp`
# forward+round-trip parametrizations are what go RED against that variant.
#
# The honest reachable outcome and the reasons the rest stay closed live in ONE
# home: the `ffn_type` docstring of `ImageCaptioningHead` in
# `src/dl_techniques/layers/heads/vlm/factory.py`. It is not restated here.
# ---------------------------------------------------------------------

_WIDTH_PARAM_BY_FFN_TYPE = {
    t: meta["output_dim_param"] for t, meta in FFN_REGISTRY.items()
}

# The types whose width parameter is NOT called "output_dim". This list is what
# separates the correct registry lookup from the literal-string variant.
_RENAMED_WIDTH_FFN_TYPES = sorted(
    t for t, p in _WIDTH_PARAM_BY_FFN_TYPE.items()
    if p is not None and p != "output_dim"
)

# Newly reachable on BOTH sites: the width parameter was their ONLY missing
# required parameter.
_NEWLY_REACHABLE_FFN_TYPES = ["kan", "power_mlp"]

# Still closed, and WHY -- each value is a substring of the error that keeps it
# closed. Asserting the specific message (not merely "it raises") is what proves
# the width routing landed: `units`/`filters` no longer appear as missing.
_CLOSED_BY_MISSING_HYPERPARAM = {
    "counting": "count_dim",
    "logic": "logic_dim",
    "mixer": "tokens_mlp_dim",
    "tversky": "num_features",
}


def _itm_width_head(ffn_type: str) -> ImageTextMatchingHead:
    """Site 1: `BaseVLMHead._build_common_layers` -> post-fusion FFN (rank-2)."""
    return ImageTextMatchingHead(
        task_config=_family_config("itm", VLMTaskType.IMAGE_TEXT_MATCHING),
        vision_dim=_FAMILY_DIM,
        text_dim=_FAMILY_DIM,
        use_post_fusion_ffn=True,
        ffn_type=ffn_type,
    )


def _cap_width_head(ffn_type: str) -> ImageCaptioningHead:
    """Site 2: `ImageCaptioningHead.__init__`'s per-layer FFN loop (rank-3)."""
    return ImageCaptioningHead(
        task_config=_family_config("cap", VLMTaskType.IMAGE_CAPTIONING),
        vision_dim=_FAMILY_DIM,
        text_dim=_FAMILY_DIM,
        num_layers=1,
        num_heads=NUM_HEADS,
        ffn_type=ffn_type,
    )


def _forward_and_round_trip(head) -> None:
    """Forward pass + `.keras` round-trip asserting VALUES and weight counts.

    Shape-only round-trip assertions are worthless here: this file's own history
    records a model that restored ZERO weights and still emitted
    correctly-shaped output. Compare the actual tensors, and compare the pre/post
    weight COUNT so a partially-restored model cannot pass either.
    """
    payload = _family_payload()
    vi = keras.Input(shape=(_FAMILY_SEQ, _FAMILY_DIM))
    ti = keras.Input(shape=(_FAMILY_SEQ, _FAMILY_DIM))
    out = head({"vision_features": vi, "text_features": ti})
    model = keras.Model([vi, ti], out)
    args = [payload["vision_features"], payload["text_features"]]
    before = model.predict(args, verbose=0)
    n_before = len(model.get_weights())
    assert n_before > 0, "the head created no weights — nothing to round-trip"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "width.keras")
        model.save(path)
        restored = keras.models.load_model(path)
    after = restored.predict(args, verbose=0)
    n_after = len(restored.get_weights())

    assert n_after == n_before, (
        f"weight count changed across the .keras round-trip: "
        f"{n_before} -> {n_after}"
    )
    keys = before.keys() if isinstance(before, dict) else range(len(before))
    for k in keys:
        b = before[k] if isinstance(before, dict) else before[k]
        a = after[k] if isinstance(after, dict) else after[k]
        assert np.isfinite(np.asarray(b)).all(), f"output {k!r} is not finite"
        np.testing.assert_allclose(
            b, a, rtol=1e-5, atol=1e-5,
            err_msg=f"output {k!r} changed across the .keras round-trip",
        )


class TestFFNOutputWidthParamRouting:
    """Both VLM sites must route the output width by the type's OWN param name."""

    def test_the_width_parameter_names_are_still_renamed(self) -> None:
        """Anti-vacuity: if every type called it `output_dim`, this whole
        section would pass against the literal-string no-op variant."""
        assert _RENAMED_WIDTH_FFN_TYPES, (
            "no FFN type names its output width anything other than "
            "'output_dim' any more — the parametrizations below can no longer "
            "distinguish a registry lookup from a literal string match"
        )
        assert _RENAMED_WIDTH_FFN_TYPES == [
            "gated_mlp", "kan", "power_mlp", "tversky"
        ], (
            f"the renamed-width type set moved to {_RENAMED_WIDTH_FFN_TYPES}; "
            f"re-derive the reachability groups below before editing this"
        )

    @pytest.mark.parametrize("ffn_type", _NEWLY_REACHABLE_FFN_TYPES)
    def test_site1_matching_head_forward_and_round_trip(self, ffn_type) -> None:
        """Site 1 (`BaseVLMHead._build_common_layers`), rank-2 post-fusion."""
        _forward_and_round_trip(_itm_width_head(ffn_type))

    @pytest.mark.parametrize("ffn_type", _NEWLY_REACHABLE_FFN_TYPES)
    def test_site2_captioning_head_forward_and_round_trip(self, ffn_type) -> None:
        """Site 2 (`ImageCaptioningHead.__init__`), rank-3 decoder stream."""
        _forward_and_round_trip(_cap_width_head(ffn_type))

    @pytest.mark.parametrize("ffn_type", sorted(_CLOSED_BY_MISSING_HYPERPARAM))
    def test_types_needing_an_unrelated_hyperparameter_stay_closed(
        self, ffn_type
    ) -> None:
        """Closed by a hyperparameter `VLMTaskConfig` does not carry.

        Deliberately NOT closed by inventing a derivation (user ruling): a
        `count_dim`/`logic_dim`/`tokens_mlp_dim`/`channels_mlp_dim`/`num_features`
        default would be new design surface, not a lookup.

        The assertion names the SPECIFIC missing key, which is also what proves
        the width routing landed for `tversky`: its width parameter `units` is no
        longer among the missing ones.
        """
        needle = _CLOSED_BY_MISSING_HYPERPARAM[ffn_type]
        for factory in (_itm_width_head, _cap_width_head):
            with pytest.raises(ValueError, match=needle):
                factory(ffn_type)

    def test_tversky_width_param_reaches_the_factory_even_though_it_stays_closed(
        self,
    ) -> None:
        """`units` must have been supplied; only `num_features` may be missing.

        Without the registry lookup the error names BOTH keys. This is the
        assertion that separates "the routing works but the type needs more" from
        "the routing silently did nothing".
        """
        with pytest.raises(ValueError) as excinfo:
            _itm_width_head("tversky")
        message = str(excinfo.value)
        # Match the MISSING list specifically -- the factory's wrapper message
        # also echoes the full `required`/`provided` lists, in which 'units'
        # legitimately appears either way.
        assert "missing for tversky: ['num_features']" in message, (
            f"'units' is still reported missing for tversky, so the width "
            f"parameter never reached the factory: {message}"
        )

    def test_gated_mlp_stays_closed_on_its_RANK_not_on_its_width_param(self) -> None:
        """A permanent capability gap, not a registry gap.

        `GatedMLP` is 1x1-conv based and needs a rank-4 `(B, H, W, C)` input.
        Site 1 hands it rank-2 and site 2 rank-3, so `filters` now reaching it
        moves the failure from "required parameter missing" to a kernel-rank
        error — and that MOVE is the observable proof the routing works.
        """
        for factory in (_itm_width_head, _cap_width_head):
            with pytest.raises(ValueError) as excinfo:
                head = factory("gated_mlp")
                head({k: ops.convert_to_tensor(v)
                      for k, v in _family_payload().items()})
            message = str(excinfo.value)
            assert "filters" not in message, (
                f"'filters' is still reported missing, so gated_mlp's width "
                f"parameter never reached the factory: {message}"
            )
            assert "Kernel shape" in message or "rank" in message.lower(), message

    def test_both_sites_ask_the_registry_the_same_question(self) -> None:
        """I-4 for the WIDTH rule, mirroring the `hidden_dim` rule's own guard.

        Spies on the single shared helper and asserts both construction sites
        call it with the same arguments and consume the same answer. If one site
        is ever re-hardcoded, the spy records one call instead of two.
        """
        calls = []
        original = vlm_factory._ffn_width_kwargs

        def spy(ffn_type, width):
            result = original(ffn_type, width)
            calls.append((ffn_type, width, tuple(sorted(result.items()))))
            return result

        vlm_factory._ffn_width_kwargs = spy
        try:
            _cap_width_head("kan")
            cap_calls = list(calls)
            calls.clear()
            _itm_width_head("kan")
            itm_calls = list(calls)
        finally:
            vlm_factory._ffn_width_kwargs = original

        assert cap_calls and itm_calls, (
            f"a site did not consult the shared width helper at all: "
            f"captioning={cap_calls}, matching={itm_calls}"
        )
        assert set(cap_calls) == set(itm_calls) == {
            ("kan", _FAMILY_DIM, (("features", _FAMILY_DIM),))
        }, f"the two sites disagree: captioning={cap_calls}, matching={itm_calls}"

    def test_mixer_receives_no_width_key_at_all(self) -> None:
        """`output_dim_param is None` means "pass nothing", not "pass output_dim".

        `mixer`'s output shape is structurally its input shape; there is no width
        to set. It stays closed on its own two hyperparameters (asserted above),
        so this checks the helper directly.
        """
        assert vlm_factory._ffn_width_kwargs("mixer", _FAMILY_DIM) == {}
        assert vlm_factory._ffn_width_kwargs("mlp", _FAMILY_DIM) == {
            "output_dim": _FAMILY_DIM
        }


class TestSite1PostFusionWidthAssertion:
    """I-2: a wrong post-fusion width on site 1 silently REWIRES, so assert it.

    `ImageTextMatchingHead.similarity_head` is a plain `Dense` built from the
    shape `post_fusion_ffn.compute_output_shape(...)` returns, so it accepts any
    width and no Keras error is ever raised. Site 2 needs no such assertion — its
    unprojected residual add raises loudly (pinned by
    `test_site2_residual_add_raises_loudly_on_a_wrong_width`).
    """

    def test_the_assertion_is_green_as_shipped(self) -> None:
        head = _itm_width_head("swiglu")
        head({k: ops.convert_to_tensor(v) for k, v in _family_payload().items()})
        assert head.post_fusion_ffn.compute_output_shape(
            (None, _FAMILY_DIM)
        )[-1] == _FAMILY_DIM

    def test_a_wrong_post_fusion_width_raises_naming_the_ffn_type(self) -> None:
        """RED proof: force a half-width FFN and require OUR error, not a
        downstream shape complaint (there is none — that is the whole point)."""
        original = vlm_factory._ffn_width_kwargs

        def half_width(ffn_type, width):
            return {k: width // 2 for k in original(ffn_type, width)}

        vlm_factory._ffn_width_kwargs = half_width
        try:
            with pytest.raises(ValueError) as excinfo:
                head = _itm_width_head("mlp")
                head({k: ops.convert_to_tensor(v)
                      for k, v in _family_payload().items()})
        finally:
            vlm_factory._ffn_width_kwargs = original
        message = str(excinfo.value)
        assert "post-fusion FFN" in message, message
        assert "ffn_type='mlp'" in message, message
        assert str(_FAMILY_DIM // 2) in message and str(_FAMILY_DIM) in message, (
            f"the error must report both the actual and the expected width: "
            f"{message}"
        )

    def test_site2_residual_add_raises_loudly_on_a_wrong_width(self) -> None:
        """I-1, executed rather than assumed.

        `ImageCaptioningHead.call()` does `x + ffn_output` with no projection, so
        a wrong FFN width must raise. This is why site 2 gets no explicit
        assertion of its own.

        # DECISION plan-2026-07-30T140922-8af1028f/D-030
        This guard used to assert on the backend's ERROR TEXT (both widths named
        in the message). That is DEVICE-DEPENDENT and was measured RED on GPU:
        CPU `AddV2` says `Incompatible shapes: [3,7,96] vs. [3,7,48] [Op:AddV2]`,
        GPU `AddV2` says only `required broadcastable shapes [Op:AddV2]` — no
        dimensions at all. The subject of the guard (it raises, at the residual
        add) holds on both; only the phrasing differs.

        So the discriminating core below is device-INVARIANT: the exception TYPE,
        the failing OP (`AddV2`, which both wordings name), and the deepest Python
        frame — eager execution raises at the line that ISSUED the op, and Keras's
        `traceback_utils` filters its own frames, so that frame is exactly
        `heads/vlm/factory.py`'s `x + ffn_output`. A Python traceback is a property
        of the program, not of the kernel. Together these are STRICTER than the old
        text match, which would have accepted a shape error raised anywhere else as
        long as it happened to mention 48 and 96.

        DO NOT weaken this to a bare `pytest.raises(Exception)`: a guard that
        cannot tell its target from any other failure is close to vacuous.
        """
        original = vlm_factory._ffn_width_kwargs

        # Precondition: the SAME head at the CORRECT width builds and forwards.
        # Without this, a generally-broken `_cap_width_head` would satisfy the
        # `raises` below for entirely the wrong reason.
        control = _cap_width_head("mlp")
        control({k: ops.convert_to_tensor(v)
                 for k, v in _family_payload().items()})

        def half_width(ffn_type, width):
            return {k: width // 2 for k in original(ffn_type, width)}

        vlm_factory._ffn_width_kwargs = half_width
        try:
            with pytest.raises(tf.errors.InvalidArgumentError) as excinfo:
                head = _cap_width_head("mlp")
                head({k: ops.convert_to_tensor(v)
                      for k, v in _family_payload().items()})
        finally:
            vlm_factory._ffn_width_kwargs = original

        message = str(excinfo.value)
        assert "AddV2" in message, (
            f"the failure must come from the residual ADD, not from some other "
            f"op: {message}"
        )

        frames = []
        tb = excinfo.value.__traceback__
        while tb is not None:
            frames.append((
                tb.tb_frame.f_code.co_filename,
                tb.tb_lineno,
                tb.tb_frame.f_code.co_name,
            ))
            tb = tb.tb_next
        frame_file, frame_line, frame_func = frames[-1]
        source = linecache.getline(frame_file, frame_line).strip()
        assert os.path.abspath(frame_file) == os.path.abspath(
            vlm_factory.__file__
        ), f"raised in {frame_file}, not in the VLM factory: {frames}"
        assert frame_func == "call", (
            f"raised in {frame_func!r}, not in `ImageCaptioningHead.call`: {frames}"
        )
        assert "ffn_output" in source and "+" in source, (
            f"raised at {frame_file}:{frame_line} ({source!r}), which is not the "
            f"unprojected residual add this invariant is about"
        )

        # Device-CONDITIONAL extra, stated rather than silently dropped: only the
        # CPU kernel names the two widths (see the docstring). Skipping when a GPU
        # is present cannot produce a false RED in either direction — if the op
        # were placed on CPU anyway the dims would be there and we merely forgo a
        # check.
        if not tf.config.list_physical_devices("GPU"):
            assert (
                str(_FAMILY_DIM // 2) in message and str(_FAMILY_DIM) in message
            ), f"expected a shape-mismatch naming both widths, got: {message}"


class TestSwiGLUComputeOutputShapeHonesty:
    """`SwiGLUFFN.compute_output_shape` must not lie about its output width.

    It returned `input_shape` verbatim while the forward path really projects to
    `output_dim`. Harmless only because every shipped caller happened to pass
    `output_dim == input width`. It is load-bearing HERE:
    `BaseVLMHead._build_fusion_stack` derives every downstream shape from that
    method, `swiglu` is `ImageCaptioningHead`'s default, and the site-1 width
    assertion above would otherwise be checking a lying oracle.

    Swept across all 21 registry types at an output width different from the
    input width; `swiglu` was the ONLY type whose declared width disagreed with
    its forward width.
    """

    def test_declared_width_matches_forward_width(self) -> None:
        from dl_techniques.layers.ffn.factory import create_ffn_layer

        layer = create_ffn_layer("swiglu", output_dim=24)
        x = ops.convert_to_tensor(
            np.zeros((2, 5, 32), dtype="float32")
        )
        declared = tuple(layer.compute_output_shape((2, 5, 32)))
        actual = tuple(layer(x).shape)
        assert declared[-1] == actual[-1] == 24, (
            f"compute_output_shape declares width {declared[-1]} but the "
            f"forward pass produces {actual[-1]}"
        )
        assert declared[:-1] == actual[:-1] == (2, 5)


# ---------------------------------------------------------------------
# MultiTaskVLMHead x heterogeneous head signatures
#
# The five head classes do NOT share a constructor signature:
# ImageCaptioningHead and VQAHead derive straight from keras.layers.Layer, while
# VisualGroundingHead and ImageTextMatchingHead derive from BaseVLMHead and
# inherit its fusion arguments. Shared kwargs used to be forwarded to ALL heads,
# so `fusion_strategy` -- perfectly valid for two of them -- raised
# "Unrecognized keyword arguments" on the other two, making the wrapper unusable
# for any mixed set of tasks.
#
# Shared kwargs are now routed to the heads whose class accepts them. That is
# deliberately best-effort, so it is fenced by three guards, each pinned below:
#   - a shared kwarg accepted by NO head raises (typo guard)
#   - task_specific_kwargs is STRICT, because it names one head explicitly
#   - wrapper-owned args (task_config/vision_dim/text_dim) raise a clear error
#     instead of an opaque duplicate-argument TypeError
# plus a logger.info record of every partial application, so routing is
# discoverable rather than silent.
# ---------------------------------------------------------------------

_HETERO_DIM = 96   # divisible by the default num_heads=12


def _hetero_configs():
    return {
        "cap": VLMTaskConfig(
            name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
            vocab_size=VOCAB, hidden_size=_HETERO_DIM,
        ),
        "itm": VLMTaskConfig(
            name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
            vocab_size=VOCAB, hidden_size=_HETERO_DIM,
        ),
        "vqa": VLMTaskConfig(
            name="vqa", task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
            vocab_size=VOCAB, hidden_size=_HETERO_DIM, num_classes=NUM_CLASSES,
        ),
    }


def _hetero_multi(**kwargs):
    return create_multi_task_vlm_head(
        _hetero_configs(),
        shared_vision_dim=_HETERO_DIM,
        shared_text_dim=_HETERO_DIM,
        **kwargs,
    )


class TestMultiTaskHeterogeneousSignatures:
    """A shared kwarg must reach the heads that accept it, and only those."""

    def test_kwarg_valid_for_only_some_heads_no_longer_raises(self) -> None:
        """The original bug, in one line.

        `fusion_strategy` is a BaseVLMHead argument. Two of these three heads
        are not BaseVLMHead subclasses, and forwarding it to them used to raise.
        """
        head = _hetero_multi(fusion_strategy="concatenation")
        assert head.task_heads["itm"].fusion_strategy == "concatenation"
        # The heads that cannot take it simply do not get it.
        assert not hasattr(head.task_heads["cap"], "fusion_strategy")
        assert not hasattr(head.task_heads["vqa"], "fusion_strategy")

    def test_shared_kwarg_accepted_by_no_head_raises(self) -> None:
        """Routing is best-effort, so a kwarg going NOWHERE must not be silent."""
        with pytest.raises(ValueError, match="not accepted by ANY head"):
            _hetero_multi(fusionn_strategy="concatenation")   # typo

    def test_task_specific_kwargs_are_validated_strictly(self) -> None:
        """A per-task override names ONE head, so it cannot be best-effort.

        This is the deliberate asymmetry with shared kwargs: naming a head and
        handing it something it cannot accept is unambiguously a caller error.
        """
        with pytest.raises(ValueError, match="does not accept"):
            _hetero_multi(
                task_specific_kwargs={"cap": {"fusion_strategy": "concatenation"}}
            )

    @pytest.mark.parametrize("reserved", ["task_config", "vision_dim", "text_dim"])
    def test_wrapper_owned_arguments_are_rejected(self, reserved) -> None:
        """These are supplied by the wrapper; passing them collided opaquely."""
        with pytest.raises(ValueError, match="supplies these to every head"):
            _hetero_multi(**{reserved: _HETERO_DIM})

    def test_wrapper_owned_arguments_rejected_in_task_specific_kwargs_too(self) -> None:
        with pytest.raises(ValueError, match="supplies these to every head"):
            _hetero_multi(task_specific_kwargs={"cap": {"vision_dim": _HETERO_DIM}})

    def test_universally_accepted_kwarg_still_reaches_every_taker(self) -> None:
        """Control: routing must not become "give it to nobody"."""
        head = _hetero_multi(ffn_type="swiglu")
        assert head.task_heads["cap"].ffn_type == "swiglu"
        assert head.task_heads["itm"].ffn_type == "swiglu"
        # VQAHead has no ffn_type at all -- it is skipped, not crashed on.
        assert not hasattr(head.task_heads["vqa"], "ffn_type")

    def test_partial_application_is_logged(self, caplog) -> None:
        """Skipping must be discoverable, not silent.

        Best-effort routing is only defensible if the caller can find out that a
        setting did not reach every head.
        """
        with caplog.at_level(logging.INFO):
            _hetero_multi(fusion_strategy="concatenation")
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "fusion_strategy" in messages
        assert "skipped for" in messages
        assert "'cap'" in messages or "cap" in messages

    def test_shared_and_task_specific_compose(self) -> None:
        """Shared routing and per-task overrides must work together."""
        head = _hetero_multi(
            ffn_type="swiglu",
            task_specific_kwargs={
                "cap": {"num_heads": 4, "num_layers": 1},
                "vqa": {"pooling_strategy": "mean"},
            },
        )
        assert head.task_heads["cap"].num_heads == 4
        assert head.task_heads["cap"].num_layers == 1
        assert head.task_heads["cap"].ffn_type == "swiglu"     # shared still applied
        assert head.task_heads["vqa"].pooling_strategy == "mean"
        assert head.task_heads["itm"].ffn_type == "swiglu"

    def test_accepted_kwargs_helper_excludes_var_keyword(self) -> None:
        """`**kwargs` must NOT count as "accepts anything".

        Every head declares `**kwargs`, but only to forward to Layer.__init__,
        which rejects unknown keys. If the helper treated `**kwargs` as
        universal acceptance, every routing decision above would invert and the
        typo guard would never fire.
        """
        accepted = _accepted_constructor_kwargs(ImageCaptioningHead)
        assert "num_layers" in accepted and "ffn_type" in accepted
        assert "fusion_strategy" not in accepted, (
            "ImageCaptioningHead is not a BaseVLMHead subclass; **kwargs was "
            "wrongly treated as accepting anything"
        )
        assert "kwargs" not in accepted and "self" not in accepted

    def test_accepted_kwargs_helper_includes_inherited_parameters(self) -> None:
        """A BaseVLMHead subclass declaring only `**kwargs` still inherits."""
        accepted = _accepted_constructor_kwargs(VisualGroundingHead)
        assert "fusion_strategy" in accepted, "inherited BaseVLMHead args missed"
        assert "normalization_type" in accepted


# ---------------------------------------------------------------------
# The post-fusion `hidden_dim` rule (one rule, both sites).
#
# `BaseVLMHead._build_common_layers` used to pass `hidden_dim` to
# `create_ffn_layer` UNCONDITIONALLY while `ImageCaptioningHead` passed it only
# to types whose registry entry requires it. Two contradictory rules for the
# same question in one file, and the unconditional one silently overrode the
# internal width derivation of every type that treats `hidden_dim` as optional.
# ---------------------------------------------------------------------


class TestPostFusionFFNHiddenDimRule:
    """`hidden_dim` reaches the FFN factory only when the type requires it."""

    @staticmethod
    def _itm(ffn_type: str, expansion: int):
        head = ImageTextMatchingHead(
            task_config=_family_config("itm", VLMTaskType.IMAGE_TEXT_MATCHING),
            vision_dim=_FAMILY_DIM,
            text_dim=_FAMILY_DIM,
            use_post_fusion_ffn=True,
            ffn_type=ffn_type,
            ffn_expansion_factor=expansion,
        )
        payload = _family_payload()
        head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        return head

    @staticmethod
    def _ffn_params(head) -> int:
        return int(sum(int(np.prod(w.shape)) for w in head.post_fusion_ffn.weights))

    def test_swiglu_width_follows_its_OWN_derivation_not_an_override(self) -> None:
        """An optional-`hidden_dim` type must size itself, from the head's factor.

        REWRITTEN after review-iter-1 WARNING-3. The first version asserted the
        width was INVARIANT to `ffn_expansion_factor`, which fixed the original
        override bug but replaced it with a quieter one: `hidden_dim` was the only
        channel carrying the head's factor to the FFN, so not sending it made a
        documented constructor knob a silent no-op (73728 params at factor 2, 4
        AND 8). D-020 forwards `ffn_expansion_factor` itself to types that accept
        it, so the knob works THROUGH swiglu's own derivation.

        The invariant now pinned is the one that actually matters: the width is
        swiglu's own `2/3 * output_dim * factor` rounded UP to `ffn_multiple_of`,
        and NOT the head's `hidden_dim * factor` override. Reverting to an
        unconditional `hidden_dim=` fails on the `!=` assertion; dropping the
        factor forward fails on the responsiveness assertion.
        """
        assert "hidden_dim" not in FFN_REGISTRY["swiglu"]["required_params"], (
            "premise changed: swiglu now REQUIRES hidden_dim, so this guard no "
            "longer tests what it claims"
        )
        widths, params = {}, {}
        for factor in (2, 4, 8):
            head = self._itm("swiglu", factor)
            widths[factor] = int(head.post_fusion_ffn.hidden_dim)
            params[factor] = self._ffn_params(head)

        multiple_of = 256  # swiglu's default `ffn_multiple_of`
        for factor, width in widths.items():
            raw = int(2 * _FAMILY_DIM * factor / 3)
            expected = ((raw + multiple_of - 1) // multiple_of) * multiple_of
            assert width == expected, (
                f"factor {factor}: swiglu width {width} is not its own "
                f"derivation {expected} (2/3*{_FAMILY_DIM}*{factor} rounded up to "
                f"{multiple_of})"
            )
            assert width != _FAMILY_DIM * factor, (
                f"factor {factor}: width {width} equals the head's "
                f"hidden_dim*factor — the override is back"
            )

        # The knob must be observable at least once across the sweep. Factors 2
        # and 4 both round to 256 here, which is the multiple_of granularity, not
        # a dead knob -- so require variation across the whole sweep, not per-step.
        assert len(set(params.values())) > 1, (
            f"swiglu's width does not respond to ffn_expansion_factor anywhere "
            f"in {sorted(widths)}: {params}. The head's factor is not reaching it"
        )

    def test_mlp_width_still_tracks_ffn_expansion_factor(self) -> None:
        """The conditional must NOT stop feeding types that DO require it.

        Reverting the fix in the other direction -- dropping the kwarg entirely
        -- either raises "Required parameters missing for mlp: ['hidden_dim']"
        or silently stops honouring `ffn_expansion_factor`. Either way this
        fails, so the guard pins both directions together with the one above.
        """
        assert "hidden_dim" in FFN_REGISTRY["mlp"]["required_params"]
        counts = {f: self._ffn_params(self._itm("mlp", f)) for f in (2, 4, 8)}
        assert len(set(counts.values())) == 3, (
            f"mlp's post-fusion width no longer responds to "
            f"ffn_expansion_factor {counts} — the required kwarg stopped flowing"
        )
        # 96*factor hidden units, two kernels + two biases.
        assert counts[4] == 74208, f"expected the measured baseline, got {counts[4]}"

    def test_both_ffn_sites_use_the_same_predicate(self) -> None:
        """Guards the seam itself, not just one side of it.

        `ImageCaptioningHead` and `BaseVLMHead` must agree. If either site
        reverts to an unconditional kwarg, the two heads disagree about swiglu's
        width inside a single wrapper -- the originally measured symptom was
        captioning FFN hidden_dim=256 vs matching hidden_dim=192.
        """
        cap = ImageCaptioningHead(
            task_config=_family_config("cap", VLMTaskType.IMAGE_CAPTIONING),
            vision_dim=_FAMILY_DIM, text_dim=_FAMILY_DIM,
            ffn_type="swiglu", ffn_expansion_factor=4, num_layers=1,
        )
        payload = _family_payload()
        cap({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        cap_hidden = int(cap.ffn_layers[0].hidden_dim)

        itm = self._itm("swiglu", 4)
        itm_hidden = int(itm.post_fusion_ffn.hidden_dim)

        assert cap_hidden == itm_hidden, (
            f"the two FFN construction sites disagree on swiglu's derived "
            f"hidden width: captioning={cap_hidden}, matching={itm_hidden}"
        )


# ---------------------------------------------------------------------
# Fusion-strategy output contract (D-011).
#
# `MultiModalFusion` has 8 strategies. 6 return a single rank-preserving tensor
# and work on both BaseVLMHead subclasses. 2 do not, and used to fail with an
# error naming neither the strategy nor the reason.
# ---------------------------------------------------------------------

_SINGLE_TENSOR_STRATEGIES = [
    "concatenation", "addition", "multiplication",
    "gated", "bilinear", "tensor_fusion",
]
_UNSUPPORTED_STRATEGIES = {
    # strategy -> the phrase the raise must contain
    "cross_attention": "one output per modality",
    "attention_pooling": "pools away an axis",
}

_BASE_SUBCLASSES = [
    ("itm", ImageTextMatchingHead, VLMTaskType.IMAGE_TEXT_MATCHING),
    ("grd", VisualGroundingHead, VLMTaskType.VISUAL_GROUNDING),
]


def _fusion_head(cls, name, task_type, strategy):
    return cls(
        task_config=_family_config(name, task_type),
        vision_dim=_FAMILY_DIM, text_dim=_FAMILY_DIM,
        fusion_strategy=strategy,
    )


class TestFusionStrategyOutputContract:
    """A head must reject a fusion output its post-fusion stack cannot consume."""

    @pytest.mark.parametrize("name,cls,task_type", _BASE_SUBCLASSES)
    @pytest.mark.parametrize("strategy", sorted(_UNSUPPORTED_STRATEGIES))
    def test_unsupported_strategy_raises_naming_the_cause(
        self, name, cls, task_type, strategy
    ) -> None:
        """The message must name the strategy AND why it cannot work.

        Before this guard: `cross_attention` died with `ValueError: Invalid
        dtype: tuple` (Keras choking on a list-of-shapes it was handed as a
        shape), and `attention_pooling` died in a squeeze on ITM but only much
        later inside an ArgMax on VG. None of those name the strategy.

        Deleting either check in `_build_fusion_stack` fails here: the raise
        either does not happen or does not carry the required phrase.
        """
        expected_phrase = _UNSUPPORTED_STRATEGIES[strategy]
        head = _fusion_head(cls, name, task_type, strategy)
        payload = _family_payload()

        with pytest.raises(ValueError, match=strategy) as excinfo:
            head.build(_shapes_of(payload))
        assert expected_phrase in str(excinfo.value), (
            f"raise for {strategy} did not explain the cause; got: "
            f"{excinfo.value}"
        )

    @pytest.mark.parametrize("name,cls,task_type", _BASE_SUBCLASSES)
    @pytest.mark.parametrize("strategy", _SINGLE_TENSOR_STRATEGIES)
    def test_supported_strategies_still_build_and_call(
        self, name, cls, task_type, strategy
    ) -> None:
        """CONTROL: the 6 working strategies must keep working, on both heads.

        Without this, a check that rejected EVERYTHING would satisfy the test
        above. Asserts a real forward pass and finite outputs, not just
        construction, so an over-broad guard or a dead pipeline also fails.
        """
        head = _fusion_head(cls, name, task_type, strategy)
        payload = _family_payload()
        out = head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        flat = _flatten_outputs(out)
        assert flat.size > 0
        assert np.all(np.isfinite(flat)), (
            f"{name}/{strategy} produced non-finite outputs"
        )

    @pytest.mark.parametrize("name,cls,task_type", _BASE_SUBCLASSES)
    @pytest.mark.parametrize("strategy", sorted(_UNSUPPORTED_STRATEGIES))
    def test_lazy_and_explicit_build_fail_identically(
        self, name, cls, task_type, strategy
    ) -> None:
        """The guard must not introduce a build-vs-lazy asymmetry.

        Measured before the fix: these 2 strategies already failed IDENTICALLY
        on both paths, i.e. this is a capability gap, not a `build()`-contract
        defect. The fix must keep it that way -- a guard that fired only from
        `build()` would leave `head(payload)` dying with the old opaque error.
        """
        payload = _family_payload()
        tensors = {k: ops.convert_to_tensor(v) for k, v in payload.items()}

        with pytest.raises(ValueError, match=strategy):
            _fusion_head(cls, name, task_type, strategy).build(
                _shapes_of(payload)
            )
        with pytest.raises(ValueError, match=strategy):
            _fusion_head(cls, name, task_type, strategy)(tensors)

    def test_is_single_shape_discriminates_shape_from_shape_list(self) -> None:
        """The predicate cannot be `isinstance(x, (list, tuple))`.

        Both a shape and a list of shapes are sequences; only the ELEMENT type
        separates them. A dynamic axis is `None`, which must not be mistaken for
        a nested sequence.
        """
        assert _is_single_shape((2, 5, 96))
        assert _is_single_shape((None, 5, 96))
        assert _is_single_shape([None, 96])
        assert not _is_single_shape([(2, 5, 96), (2, 5, 96)])
        assert not _is_single_shape(((None, 96), (None, 96)))
        assert not _is_single_shape("not a shape")

        # NOTE-5 (review-iter-1): a backend shape object is a sequence of ints
        # but NOT a list/tuple, so an isinstance-only predicate misclassifies it
        # in BOTH directions. Latent today (`MultiModalFusion` returns plain
        # tuples) and pinned so it stays fixed.
        import tensorflow as _tf
        assert _is_single_shape(_tf.TensorShape([2, 5, 96])), (
            "a TensorShape is ONE shape; an isinstance-only check calls it a "
            "collection and the contract guard then rejects a valid fusion layer"
        )
        assert not _is_single_shape(
            [_tf.TensorShape([2, 5, 96]), _tf.TensorShape([2, 5, 96])]
        ), "a list OF TensorShapes is a collection, not one shape"


class TestMultiTaskConfigPreservesKerasBaseKwargs:
    """`from_config` must not silently discard `name` / `trainable`.

    `MultiTaskVLMHead.from_config` popped `name`, `trainable` and `dtype` before
    `cls(**config)`, to stop them leaking into per-task head construction. But
    `__init__` already partitions `**kwargs` on `_KERAS_BASE_LAYER_KWARGS` and
    forwards the base ones to `super().__init__` only -- so the pop prevented no
    leak and merely threw the values away. A head saved frozen reloaded
    UNFROZEN, with bit-identical outputs, so nothing about the numbers showed it.
    """

    @staticmethod
    def _cfgs():
        return [
            VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                          hidden_size=_FAMILY_DIM),
            VLMTaskConfig(name="grd", task_type=VLMTaskType.VISUAL_GROUNDING,
                          hidden_size=_FAMILY_DIM),
        ]

    def _head(self, **kw):
        return create_multi_task_vlm_head(
            self._cfgs(), shared_vision_dim=_FAMILY_DIM,
            shared_text_dim=_FAMILY_DIM, **kw
        )

    def test_from_config_preserves_name_and_trainable(self) -> None:
        """Restoring the `config.pop` loop fails here."""
        head = self._head(name="mt", trainable=False)
        rebuilt = type(head).from_config(head.get_config())
        assert rebuilt.name == "mt", (
            f"name was regenerated instead of restored: {rebuilt.name!r}"
        )
        assert rebuilt.trainable is False, (
            "trainable=False was discarded — a frozen head reloads unfrozen"
        )

    def test_trainable_false_actually_freezes(self) -> None:
        """CONTROL: `trainable` must have teeth, or the test above is vacuous.

        If `trainable=False` did not change anything observable, preserving it
        would be a cosmetic assertion. Asserted standalone (0 vs 29 trainable
        weights) rather than through a Functional wrapper, where the model-level
        count read 33 both before and after and is not a reliable discriminator.
        """
        payload = _family_payload()
        tensors = {k: ops.convert_to_tensor(v) for k, v in payload.items()}

        frozen = self._head(name="mt", trainable=False)
        frozen(tensors)
        live = self._head(name="mt2", trainable=True)
        live(tensors)

        assert len(frozen.trainable_weights) == 0
        assert len(live.trainable_weights) > 0, "the live head built nothing"

    def test_keras_round_trip_preserves_name_and_trainable(self) -> None:
        """End-to-end `.keras` save/load, not just `get_config` in memory."""
        payload = _family_payload()
        vin = keras.Input(shape=(_FAMILY_SEQ, _FAMILY_DIM), name="vision_features")
        tin = keras.Input(shape=(_FAMILY_SEQ, _FAMILY_DIM), name="text_features")
        head = self._head(name="mt", trainable=False)
        model = keras.Model(
            {"vision_features": vin, "text_features": tin},
            head({"vision_features": vin, "text_features": tin}),
        )

        def _find(m):
            for layer in m.layers:
                if isinstance(layer, MultiTaskVLMHead):
                    return layer
            raise AssertionError("MultiTaskVLMHead not found in the model")

        assert _find(model).trainable is False
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mt.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        restored = _find(reloaded)
        assert restored.name == "mt"
        assert restored.trainable is False, (
            "a frozen MultiTaskVLMHead came back trainable from .keras"
        )

    def test_base_kwargs_still_do_not_reach_sub_heads(self) -> None:
        """The pop's ORIGINAL purpose must still hold after removing it.

        This is the over-fix direction: forwarding base kwargs must not start
        leaking them into per-task construction, where `name` would collide with
        each head's auto-generated `f"{task_config.name}_head"`.
        """
        head = self._head(name="mt", trainable=False)
        assert head.name == "mt"
        for task_name, sub in head.task_heads.items():
            assert sub.name != "mt", (
                f"the wrapper's name leaked into sub-head {task_name!r}"
            )
            assert sub.name.startswith(task_name), (
                f"sub-head {task_name!r} lost its generated name: {sub.name!r}"
            )


class TestHeadsActuallyConsumeTheirInputs:
    """A head that ignores one of its inputs must fail something.

    Found by a dead-component probe, not by a bug report: replacing
    `ImageCaptioningHead`'s `text_features` with `zeros_like` -- so the head
    generates captions from the image alone and ignores the text stream entirely
    -- left the ENTIRE VLM suite green (126/126 at the time of writing, including
    the round-trip, causality, explicit-build and weight-count tests).

    The repo's standing lesson is to inject a DEAD COMPONENT, not just the
    specific bug. Shape, weight-count and finiteness assertions all survive a
    dead input; only a sensitivity assertion does not. These tests perturb ONE
    input at a time and require the output to move.
    """

    @staticmethod
    def _perturb(head, payload, key):
        """Return (baseline, perturbed) outputs after changing only `key`."""
        base = _flatten_outputs(
            head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        )
        bumped = {k: v.copy() for k, v in payload.items()}
        # A large, structured shift: a tiny epsilon could vanish into a
        # saturating nonlinearity and make a live input look dead.
        bumped[key] = bumped[key] + 7.0
        after = _flatten_outputs(
            head({k: ops.convert_to_tensor(v) for k, v in bumped.items()})
        )
        return base, after

    @pytest.mark.parametrize("label,task_type,text_key", _SINGLE_HEADS)
    @pytest.mark.parametrize("which", ["vision", "text"])
    def test_output_responds_to_each_input(
        self, label, task_type, text_key, which
    ) -> None:
        """Every head must respond to BOTH of its feature inputs.

        Parametrized over both inputs deliberately: asserting only on
        `text_features` would let a head that ignores the IMAGE pass, which is
        the same defect mirrored.

        SCOPE, stated so this is not mistaken for more than it is: this detects
        TOTAL deadness only. The threshold is `max|delta| > 1e-6` after a `+7.0`
        shift -- roughly 1e-7 relative. A head that reads only one position of the
        text stream, mixes the two streams with the wrong weight, or attenuates
        one input by 1e5 all PASS. It is exactly the guard for the dead component
        that was found (a `zeros_like` stream, which moves the output by 0.0); it
        is not a correctness test, and a numerically-correct-fusion guard would
        need a reference implementation, not a perturbation.
        """
        payload = _family_payload(text_key)
        key = "vision_features" if which == "vision" else text_key
        head = _make_single(task_type, label)

        base, after = self._perturb(head, payload, key)

        assert np.all(np.isfinite(base)) and np.all(np.isfinite(after))
        delta = float(np.max(np.abs(after - base)))
        assert delta > 1e-6, (
            f"{label}: perturbing {key!r} by +7.0 changed the output by "
            f"{delta:.3e} — this head is IGNORING that input"
        )

    def test_captioning_logits_depend_on_the_text_stream(self) -> None:
        """The specific dead component the probe found, pinned directly.

        `ImageCaptioningHead` is autoregressive: `text_features` is the decoder
        stream and `vision_features` is what it cross-attends to. Zeroing the
        text stream must not leave the logits unchanged.
        """
        payload = _family_payload()
        head = _make_single(VLMTaskType.IMAGE_CAPTIONING, "captioning")

        live = head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
        zeroed = head({
            "vision_features": ops.convert_to_tensor(payload["vision_features"]),
            "text_features": ops.zeros_like(
                ops.convert_to_tensor(payload["text_features"])
            ),
        })
        live_logits = ops.convert_to_numpy(live["logits"])
        zero_logits = ops.convert_to_numpy(zeroed["logits"])

        delta = float(np.max(np.abs(live_logits - zero_logits)))
        assert delta > 1e-6, (
            "captioning logits are identical with the text stream zeroed — the "
            "decoder input is not reaching the output"
        )


# ---------------------------------------------------------------------
# Step-4 guards: both VLM FFN construction sites now pre-filter their OWN
# generic conveniences against FFN_REGISTRY before handing the config to the
# factory (`assemble_ffn_config`, D-017/D-019).
#
# The hazard was created by D-014, not inherited: opening `kan` and `power_mlp`
# made SITE 1's UNCONDITIONAL `dropout_rate` reachable for two types that do not
# accept it, so those constructions logged
# `create_ffn_layer(...): dropping 1 unsupported parameter(s) ['dropout_rate']`.
# Once the factory raises, that is a hard failure for two types this plan just
# opened.
#
# Instrument note: this used to be a handler on the logger named 'dl'
# (`utils/logger.py`: `logging.getLogger("dl")`, NOT 'dl_techniques'), because
# the factory WARNED about a dropped key. It RAISES now (D-023), so the
# instrument is a raise classifier -- but the vacuity hazard is identical, and
# `test_ffn_type_grid_harness_bites` is what proves it is not blind.
# ---------------------------------------------------------------------

_GRID_ALL_FFN_TYPES = sorted(FFN_REGISTRY)


def _vlm_strictness_break(fn):
    """Run ``fn``; return the strict-factory dropped-key message, or ``None``.

    Replaces a ``logging.Handler`` on the ``dl`` logger: ``create_ffn_layer``
    used to WARN about a key it had to drop and now RAISES (D-023), so a
    warning recorder would capture nothing forever and every zero the grid
    reported would be vacuous while still looking green. Any raise that is NOT
    a dropped key (missing required param, rank mismatch) was already loud
    before the flip and is therefore not a strictness break.
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - classification is the point
        msg = str(exc)
        return msg if STRICT_DROPPED_KEY_MARKER in msg else None
    return None


def _build_site1(ffn_type: str):
    """SITE 1: `BaseVLMHead._build_common_layers`'s post-fusion FFN."""
    head = ImageTextMatchingHead(
        task_config=_family_config("itm", VLMTaskType.IMAGE_TEXT_MATCHING),
        vision_dim=_FAMILY_DIM,
        text_dim=_FAMILY_DIM,
        use_post_fusion_ffn=True,
        ffn_type=ffn_type,
    )
    payload = _family_payload()
    head({k: ops.convert_to_tensor(v) for k, v in payload.items()})
    return head


def _build_site2(ffn_type: str):
    """SITE 2: `ImageCaptioningHead.__init__`'s per-layer FFN loop."""
    head = _ffn_head(ffn_type)
    head(_ffn_inputs())
    return head


class TestVLMFFNTypeGrid:
    """Neither VLM site may hand the factory a key it must drop, for any type.

    MEASURED before this change (`grid.py`, 21 types, site defaults):
    site 1 dropped `dropout_rate` for `kan` and `power_mlp`; site 2 dropped
    NOTHING, because it never injects `dropout_rate` and gates every other key
    on the registry. The asymmetry is real and is recorded in source beside both
    sites -- site 2's pre-filter is a no-op today and is there so that adding one
    unconditional convenience cannot silently re-arm the hazard.
    """

    def test_ffn_type_grid_harness_bites(self) -> None:
        """RED-proof the classifier against a real drop before trusting a None."""
        broke = _vlm_strictness_break(
            lambda: _factory_create_ffn_layer(
                "mlp", hidden_dim=8, output_dim=8, nosuchparam=1
            )
        )
        assert broke is not None and "nosuchparam" in broke, (
            f"the classifier returned {broke!r} for a deliberately unsupported "
            f"key; it is blind, so its None below proves nothing"
        )

    def test_ffn_type_grid_harness_does_not_always_fire(self) -> None:
        """CONTROL: a clean build, and a raise of a DIFFERENT kind, are not breaks."""
        assert _vlm_strictness_break(
            lambda: _factory_create_ffn_layer("mlp", hidden_dim=8, output_dim=8)
        ) is None
        assert _vlm_strictness_break(
            lambda: _factory_create_ffn_layer("mlp")  # missing required params
        ) is None

    def test_ffn_type_grid_covers_every_registry_type(self) -> None:
        assert len(_GRID_ALL_FFN_TYPES) == 21

    @pytest.mark.parametrize("ffn_type", _GRID_ALL_FFN_TYPES)
    @pytest.mark.parametrize("site", ["site1", "site2"])
    def test_ffn_type_grid_is_not_broken_by_strictness(
        self, ffn_type, site
    ) -> None:
        build = _build_site1 if site == "site1" else _build_site2
        broke = _vlm_strictness_break(lambda: build(ffn_type))
        assert broke is None, (
            f"VLM {site} with ffn_type={ffn_type!r} fails construction on a "
            f"key that type does not accept: {broke}"
        )

    @pytest.mark.parametrize("ffn_type", ["kan", "power_mlp"])
    def test_the_two_newly_reachable_types_still_build_on_both_sites(
        self, ffn_type
    ) -> None:
        """Anti-vacuity for the grid above: silence must not mean "it raised".

        `kan` and `power_mlp` are exactly the two cells the pre-filter fixed, so
        if the fix had instead broken their construction, the drop-count would
        also be zero and the grid would look green.
        """
        assert _build_site1(ffn_type).post_fusion_ffn is not None
        assert _build_site2(ffn_type).ffn_layers[0] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
