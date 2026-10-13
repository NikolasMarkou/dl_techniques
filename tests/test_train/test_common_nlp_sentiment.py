"""The shared sentiment fine-tuning helpers of ``train.common.nlp``.

``build_sentiment_model_from_encoder`` rebuilds a language encoder with its own
``task_head`` (``include_top=True``) from a headless encoder checkpoint, and
``set_backbone_trainable`` is the two-stage recipe's freeze (plan
plan-2026-10-11T170958-dc4d2202, D-010). Both are used by the bert, fnet,
tree_transformer and bert/wikipedia fine-tuning scripts; this file tests them
once, on all three encoder classes the scripts use.

What each group pins, and why it is not satisfied by construction:

``TestBuilder``
    * the architecture is the ARCHIVE's, not the caller's: the tiny encoders
      here use a vocabulary and a sequence length no default has (97 / 12), and
      the builder is given neither;
    * the backbone arrays are the checkpoint's, compared by value;
    * a load that fails RAISES (a checkpoint without its weights file is a real
      failure the loader reports; a builder that swallowed it would hand back a
      random-init model that trains from scratch and reports success).
``TestBuilderTakesOnlyTheArchitecture``
    * the archive's ``head_config_overrides`` never reach the new head (a
      mean-pooled archive head still gives CLS pooling);
    * the class match is exact: a ``DistilBERT`` archive is refused for ``BERT``.
``TestSetBackboneTrainable``
    * head-only means the trainable set IS the head's variables (by identity)
      and one optimizer step moves no backbone value;
    * the unfreeze leaves no variable frozen and a step then moves backbone
      values. ``model.trainable = True`` after a per-layer freeze is a no-op in
      Keras, so a helper written that way passes the freeze arm and fails here.

Everything is written under pytest's ``tmp_path``; repo-root ``results/`` is
never touched.
"""

import zipfile
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskType
from dl_techniques.models.language.bert import BERT
from dl_techniques.models.language.distilbert import DistilBERT
from dl_techniques.models.language.fnet import FNet
from dl_techniques.models.language.tree_transformer import TreeTransformer
from train.common.nlp import build_sentiment_model_from_encoder, set_backbone_trainable

# Odd on purpose: no variant default and no trainer default is 97 or 12.
VOCAB = 97
LENGTH = 12

CLASSES = {
    "bert": (BERT, {"vocab_size": VOCAB, "max_position_embeddings": LENGTH}),
    "fnet": (FNet, {"vocab_size": VOCAB, "max_position_embeddings": LENGTH}),
    "tree": (TreeTransformer, {"vocab_size": VOCAB}),
}
ALL = pytest.mark.parametrize("name", sorted(CLASSES))

# Layers a headless encoder has and a headed one does not (D-011: the tied-output
# projection of TreeTransformer is replaced by the task head).
HEADLESS_ONLY_LAYERS = {"tree": {"lm_head_projection"}}

HEAD_ONLY_KEYS = ("name", "include_top", "task_config", "head_config_overrides")


def _inputs(batch: int = 3) -> Dict[str, Any]:
    rng = np.random.RandomState(0)
    mask = np.ones((batch, LENGTH), "int32")
    mask[:, LENGTH - 3:] = 0
    return {
        "input_ids": rng.randint(1, VOCAB, size=(batch, LENGTH)).astype("int32"),
        "attention_mask": mask,
        "token_type_ids": np.zeros((batch, LENGTH), "int32"),
    }


def _sentiment_task(num_classes: int = 2) -> Dict[str, Any]:
    return {
        "name": "sentiment_analysis",
        "task_type": NLPTaskType.SENTIMENT_ANALYSIS.value,
        "num_classes": num_classes,
    }


def _config_without_head(model: keras.Model) -> Dict[str, Any]:
    return {k: v for k, v in model.get_config().items() if k not in HEAD_ONLY_KEYS}


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Any:
    """``saved(name) -> (headless encoder, path of its .keras archive)``."""
    root = tmp_path_factory.mktemp("sentiment_encoders")
    cache: Dict[str, Tuple[keras.Model, str]] = {}

    def _get(name: str) -> Tuple[keras.Model, str]:
        if name not in cache:
            cls, geometry = CLASSES[name]
            encoder = cls.from_variant("tiny", **geometry)
            encoder(_inputs(1), training=False)
            path = str(root / f"{name}.keras")
            encoder.save(path)
            cache[name] = (encoder, path)
        return cache[name]

    return _get


def _headed(name: str) -> keras.Model:
    cls, geometry = CLASSES[name]
    model = cls.from_variant(
        "tiny", include_top=True, task_config=_sentiment_task(), **geometry
    )
    model(_inputs(1), training=False)
    return model


def _backbone_arrays(model: keras.Model) -> Dict[str, np.ndarray]:
    # Keyed by layer name and position, not by ``Variable.path``: the path carries
    # the model's own (process-wide uniquified) name.
    return {
        f"{layer.name}/{index}": np.array(v.numpy())
        for layer in model.layers
        if layer is not model.task_head
        for index, v in enumerate(layer.weights)
    }


@ALL
class TestBuilder:
    def test_the_architecture_comes_from_the_archive(self, name, saved):
        cls, _ = CLASSES[name]
        encoder, path = saved(name)
        model = build_sentiment_model_from_encoder(cls, path, num_classes=2)
        assert type(model) is cls
        assert model.include_top is True and model.vocab_size == VOCAB
        assert _config_without_head(model) == _config_without_head(encoder)

    def test_the_head_is_the_requested_sentiment_head(self, name, saved):
        cls, _ = CLASSES[name]
        _, path = saved(name)
        model = build_sentiment_model_from_encoder(
            cls, path, num_classes=3, dropout_rate=0.2
        )
        assert model.task_head.name == "task_head"
        task = model.get_config()["task_config"]
        assert task["task_type"] == NLPTaskType.SENTIMENT_ANALYSIS.value
        assert task["num_classes"] == 3 and task["dropout_rate"] == 0.2
        output = model(_inputs(), training=False)
        assert not isinstance(output, dict)
        assert tuple(output.shape) == (3, 3)

    def test_the_backbone_values_are_the_checkpoints(self, name, saved):
        cls, _ = CLASSES[name]
        encoder, path = saved(name)
        model = build_sentiment_model_from_encoder(cls, path, num_classes=2)
        want = {l.name: l for l in encoder.layers}
        compared = 0
        for layer in model.layers:
            if layer is model.task_head:
                continue
            assert len(layer.weights) == len(want[layer.name].weights)
            for got, expected in zip(layer.weights, want[layer.name].weights):
                np.testing.assert_array_equal(
                    np.asarray(got), np.asarray(expected), err_msg=got.path
                )
                compared += 1
        # Only the headless model's own output layer has no home under a head.
        dropped = set(want) - {l.name for l in model.layers}
        assert dropped == HEADLESS_ONLY_LAYERS.get(name, set())
        left_out = sum(len(want[n].weights) for n in dropped)
        assert compared == len(encoder.weights) - left_out > 0

    def test_a_checkpoint_trained_under_a_head_loads_with_a_fresh_head(
        self, name, saved, tmp_path
    ):
        """The head of the checkpoint is never copied (any head, or none)."""
        cls, _ = CLASSES[name]
        _, path = saved(name)
        first = build_sentiment_model_from_encoder(cls, path, num_classes=2)
        headed_path = str(tmp_path / "headed.keras")
        first.save(headed_path)

        second = build_sentiment_model_from_encoder(cls, headed_path, num_classes=2)
        for key, value in _backbone_arrays(first).items():
            np.testing.assert_array_equal(value, _backbone_arrays(second)[key])
        head_before = [np.array(v.numpy()) for v in first.task_head.weights]
        head_after = [np.array(v.numpy()) for v in second.task_head.weights]
        assert any(
            a.shape == b.shape and not np.array_equal(a, b)
            for a, b in zip(head_before, head_after)
        ), "the head of the checkpoint was copied instead of left fresh"

    def test_a_missing_file_raises(self, name, tmp_path):
        cls, _ = CLASSES[name]
        with pytest.raises(FileNotFoundError):
            build_sentiment_model_from_encoder(cls, str(tmp_path / "none.keras"), 2)

    def test_an_archive_of_another_class_raises(self, name, saved):
        cls, _ = CLASSES[name]
        other = next(n for n in sorted(CLASSES) if CLASSES[n][0] is not cls)
        _, path = saved(other)
        with pytest.raises(ValueError, match="not a readable"):
            build_sentiment_model_from_encoder(cls, path, 2)

    def test_a_failed_backbone_load_raises(self, name, saved, tmp_path):
        """An archive whose weights file is gone must raise, never return.

        The config is intact, so the architecture rebuilds; only the load
        fails. A builder that logged the error and returned the random-init
        model would pass every test above.
        """
        cls, _ = CLASSES[name]
        _, path = saved(name)
        broken = str(tmp_path / "no_weights.keras")
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(broken, "w") as dst:
            names = src.namelist()
            assert any(n.endswith(".h5") for n in names), names
            for member in names:
                if not member.endswith(".h5"):
                    dst.writestr(member, src.read(member))
        with pytest.raises(Exception) as caught:
            build_sentiment_model_from_encoder(cls, broken, 2)
        assert not isinstance(caught.value, (FileNotFoundError, KeyboardInterrupt))
        assert "weights" in str(caught.value).lower() or "archive" in str(caught.value).lower()


class TestBuilderTakesOnlyTheArchitecture:
    """The archive gives the architecture; nothing of its head, nor another class.

    Plan plan-2026-10-13T035756-5dd4d94d, step 7 (review item W4).
    """

    def test_the_archive_head_overrides_are_not_carried_into_the_new_head(
        self, tmp_path
    ):
        """A head saved with ``pooling_type="mean"`` gives a CLS-pooled new head.

        The docstring promises "a fresh ``task_head`` (CLS pooling, bare
        logits)". A rebuild that copied every archive config key carried
        ``head_config_overrides`` along, so the new head pooled by mean.
        """
        _, geometry = CLASSES["bert"]
        source = BERT.from_variant(
            "tiny",
            include_top=True,
            task_config=_sentiment_task(),
            head_config_overrides={"pooling_type": "mean"},
            **geometry,
        )
        source(_inputs(1), training=False)
        # Non-vacuous: the archive really records a mean-pooled head.
        assert source.task_head.pooling_type == "mean"
        path = str(tmp_path / "bert_mean_head.keras")
        source.save(path)

        model = build_sentiment_model_from_encoder(BERT, path, num_classes=2)
        assert not model.get_config()["head_config_overrides"]
        assert model.task_head.pooling_type == "cls"

    def test_an_archive_whose_class_name_only_ends_with_the_class_is_refused(
        self, tmp_path
    ):
        """``DistilBERT`` ends with ``BERT``: the class match must be exact.

        A suffix match let a DistilBERT archive through for ``BERT``; it then
        died inside ``from_config`` on DistilBERT's own keyword arguments
        instead of the builder's refusal.
        """
        encoder = DistilBERT.from_variant(
            "tiny", vocab_size=VOCAB, max_position_embeddings=LENGTH
        )
        encoder(_inputs(1), training=False)
        path = str(tmp_path / "distilbert.keras")
        encoder.save(path)
        with pytest.raises(ValueError, match="not a readable BERT archive"):
            build_sentiment_model_from_encoder(BERT, path, 2)


@ALL
class TestSetBackboneTrainable:
    def test_head_only_trains_exactly_the_head(self, name):
        model = _headed(name)
        set_backbone_trainable(model, False)
        head_ids = {id(v) for v in model.task_head.trainable_weights}
        assert head_ids, "the head has no trainable weights: the probe is vacuous"
        assert {id(v) for v in model.trainable_weights} == head_ids
        assert all(not l.trainable for l in model.layers if l is not model.task_head)
        assert model.task_head.trainable

    def test_a_head_only_step_moves_no_backbone_value_and_moves_the_head(self, name):
        model = _headed(name)
        set_backbone_trainable(model, False)
        model.compile(
            optimizer=keras.optimizers.AdamW(1e-2, weight_decay=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        before_backbone = _backbone_arrays(model)
        before_head = [np.array(v.numpy()) for v in model.task_head.weights]
        model.train_on_batch(_inputs(), np.array([0, 1, 0]))
        after = _backbone_arrays(model)
        assert [k for k in before_backbone if not np.array_equal(before_backbone[k], after[k])] == []
        assert any(
            not np.array_equal(b, v.numpy())
            for b, v in zip(before_head, model.task_head.weights)
        )

    def test_the_unfreeze_leaves_nothing_frozen_and_trains_the_backbone(self, name):
        model = _headed(name)
        expected = {id(v) for v in model.trainable_weights}
        assert len(expected) > len(model.task_head.trainable_weights)
        set_backbone_trainable(model, False)
        set_backbone_trainable(model, True)
        assert {id(v) for v in model.trainable_weights} == expected
        assert all(l.trainable for l in model.layers)

        model.compile(
            optimizer=keras.optimizers.AdamW(1e-2),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        before = _backbone_arrays(model)
        model.train_on_batch(_inputs(), np.array([0, 1, 0]))
        after = _backbone_arrays(model)
        assert any(not np.array_equal(before[k], after[k]) for k in before)

    def test_the_freeze_survives_a_parent_flag_set_earlier(self, name):
        """``model.trainable = False`` first, as a hand-written recipe would."""
        model = _headed(name)
        model.trainable = False
        set_backbone_trainable(model, False)
        head_ids = {id(v) for v in model.task_head.trainable_weights}
        assert head_ids and {id(v) for v in model.trainable_weights} == head_ids
        set_backbone_trainable(model, True)
        assert len(model.trainable_weights) > len(head_ids)

    def test_a_headless_model_is_refused(self, name):
        cls, geometry = CLASSES[name]
        headless = cls.from_variant("tiny", **geometry)
        with pytest.raises(ValueError, match="task_head"):
            set_backbone_trainable(headless, False)
