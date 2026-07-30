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
import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

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
        logits at earlier positions (the causal mask blocks future leakage)."""
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
        vf = ops.convert_to_tensor(vision_feats)
        tf = ops.convert_to_tensor(text_feats)

        @keras.saving.register_keras_serializable()
        class _CapWrapper(keras.Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.head = _captioning_head()

            def call(self, inputs, training=None):
                return self.head(inputs, training=training)

        model = _CapWrapper()
        inputs = {"vision_features": vf, "text_features": tf}
        y0 = model(inputs)  # build before save (LESSONS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cap_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y1["logits"].shape) == tuple(y0["logits"].shape) == (B, S, VOCAB)


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

    def test_causal_mask_is_lower_triangular_keep_form(self) -> None:
        """The mask's SEMANTICS, not just its graph-safety.

        A graph-safe mask with the wrong polarity or a dropped diagonal would
        pass every test above while silently letting the model attend to the
        future. Rebuild the mask the way `call()` does and pin its content:
        1 = attend (key j <= query i), 0 = future.
        """
        from dl_techniques.utils.masking import MaskFactory

        seq_len = 6
        mask = ops.cast(
            ops.logical_not(MaskFactory.create_causal_mask(seq_len, dtype="bool")),
            keras.backend.floatx(),
        )
        got = ops.convert_to_numpy(mask)
        assert np.array_equal(got, np.tril(np.ones((seq_len, seq_len), dtype=got.dtype)))
        assert got[0, 0] == 1.0, "diagonal dropped: a token cannot attend to itself"
        assert got[0, 1] == 0.0, "future not masked"
        assert got[-1].sum() == seq_len, "last query must attend to the whole prefix"


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

    def test_build_creates_weights_without_a_forward_pass(self) -> None:
        """`build()` alone must materialize the weights.

        This is the assertion that fails outright when `build()` is absent:
        with no explicit build, `head.weights` is empty until something traces
        `call()`.
        """
        head = _captioning_head()
        assert not head.built
        head.build(self._shapes())
        assert head.built
        assert len(head.weights) > 0, (
            "build() produced no weights — sub-layers are still unbuilt and a "
            ".keras round-trip cannot restore them"
        )

    def test_explicit_build_is_numerically_inert(self) -> None:
        """Building explicitly must not change any number.

        Same seed, same inputs: a head built via `build()` and one built lazily
        by its first call must agree bit-exactly, and hold the same number of
        weights. A `build()` that created the wrong shapes, or extra/fewer
        sub-layers, would show up here rather than as a silent divergence.
        """
        vf, tf_ = self._inputs()
        payload = {"vision_features": ops.convert_to_tensor(vf),
                   "text_features": ops.convert_to_tensor(tf_)}

        keras.utils.set_random_seed(31)
        eager_built = _captioning_head()
        eager_built.build(self._shapes())
        explicit = ops.convert_to_numpy(eager_built(payload)["logits"])

        keras.utils.set_random_seed(31)
        lazy = _captioning_head()
        lazily = ops.convert_to_numpy(lazy(payload)["logits"])

        assert len(eager_built.weights) == len(lazy.weights)
        assert np.array_equal(explicit, lazily), (
            "explicit build() changed the forward result; it must be inert"
        )

    def test_functional_round_trip_preserves_VALUES(self) -> None:
        """The assertion the pre-existing round-trip test was missing.

        `test_image_captioning_roundtrip` compares only shapes, so it passed
        even while 12 sub-layer objects failed to load. Comparing values is what
        makes this a real round-trip test.
        """
        vf, tf_ = self._inputs()
        vi = keras.Input(shape=(S, DIM))
        ti = keras.Input(shape=(S, DIM))
        out = _captioning_head()({"vision_features": vi, "text_features": ti})
        model = keras.Model([vi, ti], out)
        before = model.predict([vf, tf_], verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cap_functional.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        after = restored.predict([vf, tf_], verbose=0)

        assert set(before) == set(after)
        for key in before:
            np.testing.assert_allclose(
                before[key], after[key], rtol=1e-6, atol=1e-6,
                err_msg=f"'{key}' changed across a .keras round-trip — weights "
                        f"were not restored",
            )

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

        A graph-safe gather that silently took region 0, or transposed the batch
        and region axes, would still return correctly-shaped finite output. This
        reproduces the gather on known values instead of trusting the shape.
        """
        batch, regions, feat = 3, 5, 4
        fused = ops.convert_to_tensor(
            np.arange(batch * regions * feat, dtype="float32").reshape(
                batch, regions, feat
            )
        )
        scores = ops.convert_to_tensor(
            np.array([[0.1, 0.9, 0.2, 0.0, 0.3],
                      [0.5, 0.1, 0.0, 0.2, 0.1],
                      [0.0, 0.1, 0.2, 0.3, 0.7]], dtype="float32")
        )
        expected_rows = [1, 0, 4]

        top_indices = ops.argmax(scores, axis=1)
        gather_index = ops.reshape(ops.cast(top_indices, "int32"), (-1, 1, 1))
        gather_index = ops.broadcast_to(gather_index, (batch, 1, feat))
        got = ops.convert_to_numpy(
            ops.squeeze(ops.take_along_axis(fused, gather_index, axis=1), axis=1)
        )

        fused_np = ops.convert_to_numpy(fused)
        for b, r in enumerate(expected_rows):
            np.testing.assert_array_equal(
                got[b], fused_np[b, r],
                err_msg=f"batch {b}: gathered the wrong region (expected {r})",
            )
