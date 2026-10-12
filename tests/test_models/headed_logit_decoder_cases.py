"""The ``include_top`` contract of a causal decoder whose headless output is LOGITS.

Zamba2 and HNet return tied (or untied) next-token logits from ``call`` when
they own no head, not a hidden-state dict like the other thirteen language
classes, and a ``CausalLanguageModel`` wraps them with ``skip_head=True``. Their
migration (plan-2026-10-11T170958-dc4d2202 D-005) kept that output byte for byte
and added the same head the others own: ``include_top=True`` with a
``task_config`` builds ONE ``layers/heads/nlp`` head named ``task_head``, fed the
final-norm output and a padding mask, and ``call`` returns its finalized output.
The tied logits are then not computed. ``test_zamba2/test_include_top.py`` and
``test_hnet/test_include_top.py`` run this one set of claims against each class.

Interface contract: a subclass named ``Test<Model>IncludeTop`` binds

* ``CLS`` -- the model class; ``MODULE`` -- the module whose
  ``tied_embedding_logits`` the class calls (patched to prove the headed model
  never computes the logits);
* ``N_BACKBONE`` -- the backbone's weight count at ``KW``, headless;
* ``model(**kw)`` -- a vocabulary-64 instance of the tiny configuration, built
  from ``KW`` plus ``kw``; ``factory(**kw)`` -- the same through
  ``create_<model>`` (the ``pretrained=<path>`` route);
* ``logits_from_hidden(twin, hidden)`` -- the headless twin's logits from a
  final hidden state, i.e. what the class's own output stage does;
* ``MISFIT`` -- factory overrides that make a backbone checkpoint of the plain
  configuration NOT fit the target (a different width or depth).

The mixin is not collected itself (no ``Test`` prefix).
"""

import json
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME

# ---------------------------------------------------------------------

SEQ = 8
CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9)
QA = NLPTaskConfig(
    name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2)
#: Same head class and weight shapes as ``CLF``, a different task.
SPAM = NLPTaskConfig(
    name="spam", task_type=NLPTaskType.SPAM_DETECTION, num_classes=3)
#: A generation head: its vocabulary is the model's (64).
GENERATION = {"name": "gen", "task_type": "text_summarization"}


def to_np(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def head_batch() -> Dict[str, np.ndarray]:
    """Row 1 is right-padded: 5 real tokens, then 3 pad ids under a 0 mask."""
    ids = np.random.default_rng(0).integers(1, 64, (2, SEQ)).astype("int32")
    mask = np.ones_like(ids)
    ids[1, 5:] = 0
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def relative_paths(model: keras.Model) -> List[str]:
    """Weight paths without the model-root segment, in order."""
    return [w.path.split("/", 1)[-1] for w in model.weights]


def randomized(model: keras.Model, seed: int) -> keras.Model:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


def backbone_layers(model: keras.Model) -> Dict[str, list]:
    """``{layer name: weight values}`` of every layer that is neither the task
    head nor an untied output projection."""
    return {
        layer.name: [to_np(w) for w in layer.weights]
        for layer in model.layers
        if layer.weights and layer.name not in (TASK_HEAD_NAME, "lm_head")
    }


def assert_same_backbone(a: keras.Model, b: keras.Model) -> None:
    left, right = backbone_layers(a), backbone_layers(b)
    assert left and sorted(left) == sorted(right)
    for name in left:
        assert len(left[name]) == len(right[name]), name
        for x, y in zip(left[name], right[name]):
            np.testing.assert_array_equal(x, y, err_msg=name)


def head_inputs(model: keras.Model, batch) -> Dict[str, np.ndarray]:
    """What the task head receives on a real call: its ``hidden_states`` and
    ``attention_mask``, read by wrapping the head's ``call``."""
    seen: Dict[str, np.ndarray] = {}
    real = model.task_head.call

    def spy(inputs, *args, **kwargs):
        seen.update({k: to_np(v) for k, v in inputs.items()})
        return real(inputs, *args, **kwargs)

    model.task_head.call = spy
    try:
        model(batch, training=False)
    finally:
        del model.task_head.call
    return seen


# ---------------------------------------------------------------------


class HeadedLogitDecoderCases:
    """The ``include_top`` claims, for any class that binds the hooks above."""

    CLS: Any = None
    MODULE: Any = None
    N_BACKBONE: int = 0
    MISFIT: Dict[str, Any] = {}

    def model(self, **kw: Any) -> keras.Model:
        raise NotImplementedError

    def factory(self, **kw: Any) -> keras.Model:
        raise NotImplementedError

    def logits_from_hidden(self, twin: keras.Model, hidden: np.ndarray):
        raise NotImplementedError

    def headless_twin(self, model: keras.Model, batch) -> keras.Model:
        """A headless model with ``model``'s own backbone weights."""
        if not model.built:
            model(batch, training=False)
        config = model.get_config()
        config.update(include_top=False, task_config=None,
                      head_config_overrides=None)
        config.pop("name", None)
        twin = self.CLS.from_config(config)
        twin(batch, training=False)
        for layer in twin.layers:
            if layer.weights and layer.name != "lm_head":
                layer.set_weights(model.get_layer(layer.name).get_weights())
        return twin

    # -- outputs ---------------------------------------------------------

    def test_headless_output_is_still_the_logits_tensor(self):
        """No head: a bare logits tensor, from a tensor or a dict input alike
        (a dict carrying an all-ones mask: a mask that removes positions is the
        backbone's business for HNet, which chunks around it)."""
        model = self.model()
        ids = head_batch()["input_ids"]
        plain = model(ids, training=False)
        assert not isinstance(plain, dict)
        assert tuple(plain.shape) == (2, SEQ, 64)
        as_dict = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        np.testing.assert_allclose(
            to_np(model(as_dict, training=False)), to_np(plain),
            atol=1e-5, rtol=0)

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA; nothing else."""
        batch = head_batch()
        out = self.model(include_top=True, task_config=CLF)(batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = self.model(include_top=True, task_config=NER)(batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, SEQ, 9)

        out = self.model(include_top=True, task_config=QA)(batch, training=False)
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, SEQ)

        out = self.model(include_top=True, task_config=GENERATION)(
            batch, training=False)
        assert tuple(out.shape) == (2, SEQ, 64)

    @pytest.mark.parametrize("task", [None, CLF, NER, QA],
                             ids=["headless", "clf", "ner", "qa"])
    def test_compute_output_shape_is_what_call_returns(self, task):
        kw = {} if task is None else dict(include_top=True, task_config=task)
        model = self.model(**kw)
        batch = head_batch()
        out = model(batch, training=False)

        def shapes(x):
            if isinstance(x, dict):
                return {k: shapes(v) for k, v in x.items()}
            return tuple(getattr(x, "shape", x))

        assert shapes(model.compute_output_shape((2, SEQ))) == shapes(out)
        as_dict = {"input_ids": (2, SEQ), "attention_mask": (2, SEQ)}
        assert shapes(model.compute_output_shape(as_dict)) == shapes(out)

    def test_symbolic_output_matches_the_eager_one(self):
        headed = self.model(include_top=True, task_config=NER)
        symbolic = headed(keras.Input(shape=(SEQ,), dtype="int32"))
        assert not isinstance(symbolic, dict)
        assert tuple(symbolic.shape) == (None, SEQ, 9)
        headless = self.model()(keras.Input(shape=(SEQ,), dtype="int32"))
        assert tuple(headless.shape) == (None, SEQ, 64)

    # -- what the head reads --------------------------------------------

    def test_the_head_reads_the_final_hidden_state_and_the_logits_are_not_computed(
            self, monkeypatch):
        """The hidden state the head receives is the one the headless model
        turns into logits, and the headed model never forms the logits."""
        model = self.model(include_top=True, task_config=CLF)
        batch = head_batch()
        twin = self.headless_twin(model, batch)

        def refuse(*args, **kwargs):
            raise AssertionError("a headed model computed the tied logits")

        with monkeypatch.context() as patch:
            patch.setattr(self.MODULE, "tied_embedding_logits", refuse)
            seen = head_inputs(model, batch)

        hidden = seen["hidden_states"]
        assert hidden.shape == (2, SEQ, model.hidden_size)
        np.testing.assert_allclose(
            self.logits_from_hidden(twin, hidden),
            to_np(twin(batch, training=False)), atol=1e-5, rtol=0)

    def test_the_default_pooling_is_last_and_an_override_wins(self):
        assert self.model(include_top=True, task_config=CLF
                          ).task_head.pooling_type == "last"
        assert self.model(include_top=True, task_config=CLF,
                          head_config_overrides={"pooling_type": "mean"}
                          ).task_head.pooling_type == "mean"

    def test_the_head_reads_the_last_real_token(self):
        """Changing the last real token of row 1 changes only row 1's output."""
        model = self.model(include_top=True, task_config=CLF)
        batch = head_batch()
        base = to_np(model(batch, training=False))
        edited = {k: v.copy() for k, v in batch.items()}
        edited["input_ids"][1, 4] = (edited["input_ids"][1, 4] % 63) + 1
        moved = to_np(model(edited, training=False))
        np.testing.assert_array_equal(moved[0], base[0])
        assert not np.allclose(moved[1], base[1], atol=1e-6, rtol=0)

    @pytest.mark.parametrize("with_mask", [True, False],
                             ids=["attention_mask", "pad_derived"])
    def test_appended_padding_does_not_change_the_head_output(self, with_mask):
        """Right padding after the last real token is invisible to the head."""
        model = self.model(pad_token_id=0, include_top=True, task_config=CLF)
        ids = np.random.default_rng(3).integers(1, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.zeros((2, 4), "int32")], axis=1)

        def run(x):
            if with_mask:
                return to_np(model({"input_ids": x,
                                    "attention_mask": (x != 0).astype("int32")},
                                   training=False))
            return to_np(model(x, training=False))

        np.testing.assert_allclose(run(padded), run(ids), atol=1e-5, rtol=0)

    def test_a_custom_pad_token_id_drives_the_derived_mask(self):
        model = self.model(pad_token_id=7, include_top=True, task_config=CLF)
        ids = np.random.default_rng(4).integers(8, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.full((2, 3), 7, "int32")], axis=1)
        np.testing.assert_allclose(to_np(model(padded, training=False)),
                                   to_np(model(ids, training=False)),
                                   atol=1e-5, rtol=0)

    # -- configuration ---------------------------------------------------

    def test_config_round_trip_with_a_head(self):
        overrides = {"pooling_type": "mean"}
        model = self.model(pad_token_id=5, include_top=True, task_config=CLF,
                           head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["pad_token_id"] == 5
        assert config["head_config_overrides"] == overrides
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = self.model(include_top=True,
                               task_config=dict(config["task_config"]),
                               head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = self.CLS.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.task_head.pooling_type == "mean"

        batch = head_batch()
        model(batch, training=False)
        rebuilt(batch, training=False)
        assert relative_paths(rebuilt) == relative_paths(model)
        rebuilt.set_weights(model.get_weights())
        np.testing.assert_array_equal(
            to_np(rebuilt(batch, training=False)),
            to_np(model(batch, training=False)))

    def test_a_config_without_the_new_keys_still_loads(self):
        """A config written before the knobs existed rebuilds the headless model."""
        config = self.model().get_config()
        for key in ("include_top", "task_config", "head_config_overrides",
                    "pad_token_id"):
            config.pop(key)
        model = self.CLS.from_config(config)
        assert model.include_top is False and model.task_head is None
        assert model.pad_token_id is None

    def test_the_head_vocabulary_is_the_models(self):
        assert self.model(include_top=True, task_config=GENERATION
                          ).task_head.task_config.vocabulary_size == 64
        with pytest.raises(ValueError, match="vocabulary_size=65"):
            self.model(include_top=True,
                       task_config=dict(GENERATION, vocabulary_size=65))

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            self.model(include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            self.model(**dead)

    def test_multiple_choice_is_refused(self):
        task = NLPTaskConfig(name="mc", task_type=NLPTaskType.MULTIPLE_CHOICE,
                             num_classes=4)
        with pytest.raises(ValueError, match="MULTIPLE_CHOICE"):
            self.model(include_top=True, task_config=task)

    # -- serialization and building -------------------------------------

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model."""
        model = self.model(include_top=True, task_config=CLF)
        batch = head_batch()
        before = to_np(model(batch, training=False))
        saved = [to_np(w).copy() for w in model.weights]
        head_saved = [to_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert len(loaded.weights) == len(saved)
        for s, w in zip(saved, loaded.weights):
            np.testing.assert_allclose(s, to_np(w), atol=0.0, rtol=0.0,
                                       err_msg=f"{w.path} not restored")
        for s, w in zip(head_saved, loaded.task_head.weights):
            np.testing.assert_allclose(s, to_np(w), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(to_np(loaded(batch, training=False)), before)

    def test_a_headed_model_compiles_and_fits_with_a_plain_loss(self):
        model = self.model(include_top=True, task_config=CLF)
        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True))
        history = model.fit(head_batch(), np.array([0, 2]), epochs=1,
                            batch_size=2, verbose=0)
        assert np.isfinite(history.history["loss"][0])

    @pytest.mark.parametrize("include_top", [False, True])
    def test_explicit_build_matches_lazy_build(self, include_top):
        """``build()`` materializes exactly the tree ``call()`` runs."""
        kw = dict(include_top=include_top,
                  task_config=CLF if include_top else None)
        explicit = self.model(**kw)
        explicit.build((None, SEQ))
        lazy = self.model(**kw)
        lazy(head_batch(), training=False)
        assert relative_paths(explicit) == relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        headless = self.model()
        headless(head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = self.model(include_top=True, task_config=CLF)
        headed(head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_every_head_weight_lives_under_task_head(self):
        """The backbone is every weight a headless model has; the head only
        appends weights under ``task_head/``, whatever the task."""
        def paths(**kw):
            m = self.model(name="m", **kw)
            m(head_batch(), training=False)
            return relative_paths(m)

        backbone = paths()
        assert len(backbone) == self.N_BACKBONE
        assert not [p for p in backbone if p.startswith("task_head/")]
        for task in (CLF, NER):
            headed = paths(include_top=True, task_config=task)
            assert headed[:len(backbone)] == backbone, task.name
            assert len(headed) > len(backbone)
            assert all(p.startswith("task_head/")
                       for p in headed[len(backbone):]), task.name
        assert len(paths(include_top=True, task_config=CLF)) == self.N_BACKBONE + 8

    def test_a_headless_model_is_reproducible_under_one_seed(self):
        """Two headless builds from one seed are bit-equal: the headless path
        draws nothing the head could have disturbed."""
        batch = head_batch()
        outputs, weights = [], []
        for _ in range(2):
            keras.utils.set_random_seed(0)
            model = self.model()
            outputs.append(to_np(model(batch, training=False)))
            weights.append([to_np(w) for w in model.weights])
        np.testing.assert_array_equal(outputs[0], outputs[1])
        for a, b in zip(*weights):
            np.testing.assert_array_equal(a, b)

    # -- backbone reuse --------------------------------------------------

    def test_the_factory_returns_the_class(self):
        model = self.factory()
        assert type(model) is self.CLS
        assert model.vocab_size == 64 and model.include_top is False

    def test_the_factory_builds_a_headed_model(self):
        model = self.factory(include_top=True, task_config=CLF)
        assert model.include_top and model.task_head is not None

    def test_pretrained_true_raises_and_a_falsy_value_is_ignored(self):
        for kw in ({}, dict(include_top=True, task_config=CLF)):
            with pytest.raises(NotImplementedError, match="pretrained"):
                self.factory(pretrained=True, **kw)
        for falsy in (False, None, ""):
            assert type(self.factory(pretrained=falsy)) is self.CLS

    def test_a_missing_checkpoint_is_refused_before_anything_is_built(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            self.factory(pretrained=str(tmp_path / "nope.keras"))

    def test_a_headless_checkpoint_restores_a_headed_backbone(self, tmp_path):
        source = self.factory()
        batch = head_batch()
        source(batch, training=False)
        randomized(source, 1)
        path = str(tmp_path / "headless.keras")
        source.save(path)

        headed = self.factory(include_top=True, task_config=CLF, pretrained=path)
        assert_same_backbone(source, headed)
        fresh = self.factory(include_top=True, task_config=CLF)
        with pytest.raises(AssertionError):  # anti-vacuity: init is not the source
            assert_same_backbone(source, fresh)

    def test_a_headed_checkpoint_restores_a_headless_backbone(self, tmp_path):
        source = self.factory(include_top=True, task_config=CLF)
        batch = head_batch()
        source(batch, training=False)
        randomized(source, 2)
        path = str(tmp_path / "headed.keras")
        source.save(path)

        headless = self.factory(pretrained=path)
        assert_same_backbone(source, headless)
        assert headless.task_head is None

    def test_a_backbone_trained_under_one_head_is_reused_under_another(self, tmp_path):
        """Same weights, another head: the hidden state the new head reads is
        bit-equal to the one the old head read."""
        source = self.factory(include_top=True, task_config=CLF)
        batch = head_batch()
        source(batch, training=False)
        randomized(source, 3)
        reference = head_inputs(source, batch)["hidden_states"]
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)

        token = self.factory(include_top=True, task_config=NER, pretrained=path)
        assert tuple(token(batch, training=False).shape) == (2, SEQ, 9)
        np.testing.assert_array_equal(
            head_inputs(token, batch)["hidden_states"], reference)

        fresh = self.factory(include_top=True, task_config=NER)
        assert not np.array_equal(
            head_inputs(fresh, batch)["hidden_states"], reference)

    def test_an_equal_shape_head_of_another_task_is_not_copied(self, tmp_path):
        source = self.factory(include_top=True, task_config=CLF)
        batch = head_batch()
        source(batch, training=False)
        randomized(source, 4)
        path = str(tmp_path / "clf.keras")
        source.save(path)

        spam = self.factory(include_top=True, task_config=SPAM, pretrained=path)
        src_head = [to_np(w) for w in source.task_head.weights]
        dst_head = [to_np(w) for w in spam.task_head.weights]
        assert [a.shape for a in src_head] == [b.shape for b in dst_head], (
            "fixture invalid: the two heads must have equal weight shapes")
        assert not any(np.array_equal(a, b) for a, b in zip(src_head, dst_head))
        assert_same_backbone(source, spam)

    def test_a_backbone_that_does_not_fit_raises_for_a_headed_model(self, tmp_path):
        source = self.factory()
        source(head_batch(), training=False)
        path = str(tmp_path / "small.keras")
        source.save(path)
        with pytest.raises(ValueError, match="unloaded|mismatch|no target layer"):
            self.factory(include_top=True, task_config=CLF, pretrained=path,
                         **self.MISFIT)
