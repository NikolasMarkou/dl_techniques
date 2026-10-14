"""FastVLM follows the language-model load and head-swap contract.

Plan-2026-10-14T062616-81938e81 step 5 (decision D-005, D-019). The pattern
is the one ``models/language`` uses (``GUIDE.md`` section 1, recipe item 13 of
the plan's language-head finding):

* ``from_variant(..., pretrained=False)`` owns ``pretrained``.
  ``True`` raises ``NotImplementedError`` (no weights ship), and a missing
  path raises ``FileNotFoundError`` before anything is built.
* A path copies the BACKBONE only, by layer name: every backbone layer loads
  or the call raises, and the checkpoint's head (``conv_exp`` and
  ``classification_head``) is never copied, so the class count may differ and
  the target's head keeps its fresh initialization.
* ``create_fastvlm(variant, pretrained=False, **kwargs)`` is a one-line
  delegate.
* Silent misuse is refused: an unknown keyword (Keras' functional ``Model``
  constructor would drop it), the deleted ``head_config_preset``, and each
  reserved ``head_config_overrides`` key.
* The head's ``activation_type`` follows the model's ``activation``.

Every load arm compares weights by VALUE (``np.array_equal``), never by
count or path alone (``GUIDE.md`` section 2).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.fastvlm import FastVLM, create_fastvlm

#: Small enough to save and load quickly; 32 px gives a 2 x 2 stage-3 map.
GEOMETRY = dict(input_shape=(32, 32, 3))
VARIANT = "nano"
SOURCE_CLASSES = 10


def _backbone(model: keras.Model) -> dict:
    """Every weight outside the head, by path."""
    return {
        w.path: keras.ops.convert_to_numpy(w)
        for w in model.weights
        if not w.path.startswith(FastVLM.HEAD_LAYER_NAMES)
    }


def _head(model: keras.Model) -> dict:
    return {
        w.path: keras.ops.convert_to_numpy(w)
        for w in model.weights
        if w.path.startswith(FastVLM.HEAD_LAYER_NAMES)
    }


@pytest.fixture(scope="module")
def headed_checkpoint(tmp_path_factory):
    """A headed 10-class nano checkpoint and the model that wrote it."""
    keras.utils.set_random_seed(1)
    model = FastVLM.from_variant(VARIANT, num_classes=SOURCE_CLASSES, **GEOMETRY)
    path = str(tmp_path_factory.mktemp("fastvlm") / "headed.keras")
    model.save(path)
    return path, _backbone(model), _head(model)


@pytest.fixture(scope="module")
def headless_checkpoint(tmp_path_factory):
    keras.utils.set_random_seed(3)
    model = FastVLM.from_variant(VARIANT, include_top=False, **GEOMETRY)
    path = str(tmp_path_factory.mktemp("fastvlm") / "headless.keras")
    model.save(path)
    return path, _backbone(model)


def _assert_backbone_equal(target: keras.Model, expected: dict) -> None:
    loaded = _backbone(target)
    assert set(loaded) == set(expected), (
        f"backbone paths differ: {sorted(set(loaded) ^ set(expected))}")
    unequal = [p for p in expected if not np.array_equal(loaded[p], expected[p])]
    assert not unequal, f"{len(unequal)} backbone weights did not load: {unequal}"


# ---------------------------------------------------------------------
# pretrained: True and a missing path
# ---------------------------------------------------------------------


@pytest.mark.parametrize("build", [FastVLM.from_variant, create_fastvlm],
                         ids=["from_variant", "create_fastvlm"])
def test_pretrained_true_raises(build) -> None:
    with pytest.raises(NotImplementedError, match="not distributed"):
        build(VARIANT, pretrained=True, num_classes=4, **GEOMETRY)


def test_a_missing_path_raises_before_anything_is_built(monkeypatch, tmp_path) -> None:
    def no_build(self, *args, **kwargs):
        raise AssertionError("FastVLM was constructed before the path was checked")

    monkeypatch.setattr(FastVLM, "__init__", no_build)
    with pytest.raises(FileNotFoundError, match="not found"):
        create_fastvlm(VARIANT, pretrained=str(tmp_path / "absent.keras"))


# ---------------------------------------------------------------------
# A path loads the backbone, and only the backbone
# ---------------------------------------------------------------------

#: Targets for the headed 10-class checkpoint: another class count, no head,
#: and the same class count (where a by-name copy of the head would succeed
#: silently if the head were not skipped).
HEADED_TARGETS = {
    "seven_classes": dict(num_classes=7),
    "headless": dict(include_top=False),
    "same_classes": dict(num_classes=SOURCE_CLASSES),
}


@pytest.mark.parametrize("target", sorted(HEADED_TARGETS))
def test_a_headed_checkpoint_loads_the_backbone_only(headed_checkpoint, target) -> None:
    path, source_backbone, source_head = headed_checkpoint
    keras.utils.set_random_seed(2)
    model = create_fastvlm(VARIANT, pretrained=path, **GEOMETRY,
                           **HEADED_TARGETS[target])
    _assert_backbone_equal(model, source_backbone)

    head = _head(model)
    if target == "headless":
        assert model.head is None and model.conv_exp is None
        assert not head
        return
    # Twin assertion: the head is fresh, not the checkpoint's. Only the
    # randomly drawn kernels can tell (BatchNorm gammas start at 1 on both
    # sides). The seeds differ (1 vs 2), so a copied kernel is the only way
    # two of them could be equal.
    kernels = [p for p in head if p.endswith("kernel") and p in source_head
               and head[p].shape == source_head[p].shape]
    expected = 5 if target == "same_classes" else 4  # conv_exp 4, classifier 1
    assert len(kernels) == expected, kernels
    copied = [p for p in kernels if np.array_equal(head[p], source_head[p])]
    assert not copied, f"head kernels were copied from the checkpoint: {copied}"


def test_a_headless_checkpoint_loads_into_a_headed_model(headless_checkpoint) -> None:
    path, source_backbone = headless_checkpoint
    model = create_fastvlm(VARIANT, pretrained=path, num_classes=5, **GEOMETRY)
    _assert_backbone_equal(model, source_backbone)
    assert model.head is not None and model.conv_exp is not None


def test_the_loaded_model_reproduces_the_checkpoint_features(headed_checkpoint) -> None:
    """By output, not only by weight: the backbone computes what it did."""
    path, _, _ = headed_checkpoint
    source = keras.models.load_model(path)
    target = create_fastvlm(VARIANT, pretrained=path, include_top=False, **GEOMETRY)
    x = (3.0 * np.random.default_rng(0).standard_normal((2, 32, 32, 3))).astype("float32")
    expected = source.extract_features(x, training=False)[-1]
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(target(x, training=False)),
        keras.ops.convert_to_numpy(expected))


#: A deeper stage 3, and wider stages. Each must refuse, not truncate or skip.
MISFIT_TARGETS = {
    "deeper": dict(depths=[1, 2, 4]),
    "wider": dict(embed_dims=[32, 48, 96]),
}


@pytest.mark.parametrize("target", sorted(MISFIT_TARGETS))
def test_a_backbone_that_does_not_fit_raises(headed_checkpoint, target) -> None:
    path, _, _ = headed_checkpoint
    with pytest.raises(ValueError, match="Shape mismatch"):
        create_fastvlm(VARIANT, pretrained=path, num_classes=SOURCE_CLASSES,
                       **GEOMETRY, **MISFIT_TARGETS[target])


# ---------------------------------------------------------------------
# Silent misuse is refused
# ---------------------------------------------------------------------


@pytest.mark.parametrize("keyword", ["bogus_kw", "head_config_preset"])
def test_an_unknown_keyword_raises(keyword: str) -> None:
    """Keras' functional ``Model.__init__`` drops unknown keywords silently.

    MEASURED before step 5: ``FastVLM(..., bogus_kw=1)`` built a model. The
    refusal is ``ValueError``, the error a subclassed Keras model gives.
    """
    with pytest.raises(ValueError, match=keyword):
        FastVLM.from_variant(VARIANT, num_classes=4, **GEOMETRY, **{keyword: 1})
    with pytest.raises(ValueError, match=keyword):
        FastVLM(num_classes=4, embed_dims=[8, 16, 32], depths=[1, 1, 1],
                num_heads=[1, 1, 1], **GEOMETRY, **{keyword: 1})


def test_create_fastvlm_is_a_one_line_delegate() -> None:
    import inspect
    params = inspect.signature(create_fastvlm).parameters
    assert list(params) == ["variant", "pretrained", "kwargs"]
    assert "head_config_preset" not in inspect.signature(FastVLM.__init__).parameters


#: A value of the right type for each reserved key, so the refusal cannot be
#: an incidental type error further down (an int ``name`` is refused by Keras).
RESERVED_OVERRIDES = {"name": "renamed_head", "num_classes": 7,
                      "task_type": "classification"}


@pytest.mark.parametrize("key", sorted(RESERVED_OVERRIDES))
def test_a_reserved_head_override_raises(key: str) -> None:
    with pytest.raises(ValueError, match=f"may not set.*{key}"):
        FastVLM.from_variant(VARIANT, num_classes=4, **GEOMETRY,
                             head_config_overrides={key: RESERVED_OVERRIDES[key]})


def test_a_head_override_reaches_the_head() -> None:
    """The control for the refusal above: a legal key is applied."""
    model = FastVLM.from_variant(VARIANT, num_classes=4, **GEOMETRY,
                                 head_config_overrides={"block_dims": (16,)})
    assert [b.units for b in model.head.dense_blocks] == [16]


# ---------------------------------------------------------------------
# The head's activation follows the model
# ---------------------------------------------------------------------


def test_the_head_reports_the_model_activation() -> None:
    relu = FastVLM.from_variant(VARIANT, num_classes=4, activation="relu", **GEOMETRY)
    gelu = FastVLM.from_variant(VARIANT, num_classes=4, **GEOMETRY)
    assert relu.head.activation_type == "relu"
    assert gelu.head.activation_type == "gelu"
    # The reference head itself applies no activation and no normalization.
    assert relu.head.dense_blocks == [] and relu.head.normalization_type is None


def test_a_hidden_head_block_uses_the_model_activation() -> None:
    model = FastVLM.from_variant(VARIANT, num_classes=4, activation="relu",
                                 head_config_overrides={"block_dims": (16,)},
                                 **GEOMETRY)
    (block,) = model.head.dense_blocks
    assert isinstance(block.activation, keras.layers.ReLU)
    assert block.norm is None


def test_an_activation_the_head_cannot_build_raises_only_when_a_block_needs_it() -> None:
    """``'tanh'`` is a Keras activation but not a head-factory key."""
    model = FastVLM.from_variant(VARIANT, num_classes=4, activation="tanh", **GEOMETRY)
    assert model.head.dense_blocks == []
    with pytest.raises(ValueError, match="tanh"):
        FastVLM.from_variant(VARIANT, num_classes=4, activation="tanh",
                             head_config_overrides={"block_dims": (16,)},
                             **GEOMETRY)
    explicit = FastVLM.from_variant(
        VARIANT, num_classes=4, activation="tanh", **GEOMETRY,
        head_config_overrides={"block_dims": (16,), "activation_type": "gelu"})
    assert explicit.head.dense_blocks[0].activation_type == "gelu"
