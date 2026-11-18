"""The transfer engine of ``init``: ``plan_transfer``, ``check``, ``apply`` and ``vit_pos_bicubic``.

Tiny Keras models, no run. The engine copies weight by weight, keyed by the Keras variable path
relative to its model (D-030, D-045: one leading component is dropped on both sides when it is the
model's root, ``<model.name>/`` or the component every weight shares; nothing else is renamed), complete-or-raise in both directions (plan-2026-11-18T072904-0fd54b91 D-005, D-006): every target
weight is copied, resampled by a declared rule, reset as a declared head or grown as a new block,
and every source weight is used or dropped by a declared rule; anything else is a ``ConfigError``
raised before the first ``assign``, so a refusal leaves every target weight as it was (compared
tensor by tensor). The ViT position table is resampled to a larger grid only, checked against the
independent per-pixel loop of ``torch_bicubic_2d`` (D-009). Inside a component a spec names in
``positional`` (D-031), weights whose paths differ pair by their order when the counts, shapes and
dtypes all agree.

``model.weights`` holds no ``SeedGenerator`` state in Keras 3.8 (it is in ``model.variables``),
so a dropout model's seed is never transferred; the first test pins that.
"""

import types
from typing import Any, Dict, List, Optional, Sequence

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from train.api.errors import ConfigError
from train.api.init_from import apply, check, plan_transfer
from train.api.registry import CrossTask, TransferSpec

from ...test_models.test_dino.dinov2_numpy_reference import torch_bicubic_2d

TABLE = "pos_embed/pos_embedding"
VIT = TransferSpec(head=("head",), resample={TABLE: "vit_pos_bicubic"}, grow=r"block_\d+")


def classifier(seed: int, blocks: int = 2, width: int = 4, classes: int = 3, head: str = "head",
               extra: bool = False) -> keras.Model:
    """``stem``, ``norm`` (BatchNorm), ``drop`` (Dropout), ``block_<i>`` dense blocks, a head."""
    keras.utils.set_random_seed(seed)
    inputs = keras.Input((5,))
    x = keras.layers.Dense(width, name="stem")(inputs)
    x = keras.layers.BatchNormalization(name="norm")(x)
    x = keras.layers.Dropout(0.1, name="drop")(x)
    for i in range(blocks):
        x = keras.layers.Dense(width, name=f"block_{i}")(x)
    if extra:
        x = keras.layers.Dense(width, name="extra")(x)
    model = keras.Model(inputs, keras.layers.Dense(classes, name=head)(x))
    for variable in model.weights:  # BatchNorm statistics too, so a copy is visible on every weight
        variable.assign(np.random.default_rng(seed).normal(size=variable.shape).astype(variable.dtype))
    return model


class Backbone(keras.layers.Layer):
    """A layer named ``backbone`` holding a classifier's trunk, so its paths start with ``backbone/``."""

    def __init__(self, blocks: int = 2, width: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.trunk = [keras.layers.Dense(width, name="stem"), keras.layers.BatchNormalization(name="norm")]
        self.trunk += [keras.layers.Dense(width, name=f"block_{i}") for i in range(blocks)]

    def build(self, input_shape):
        for layer in self.trunk:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, x):
        for layer in self.trunk:
            x = layer(x)
        return x


def segmenter(seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)
    inputs = keras.Input((5,))
    return keras.Model(inputs, keras.layers.Dense(2, name="segmentation_head")(Backbone(name="backbone")(inputs)))


def table_model(seed: int, side: int, rows: Optional[int] = None, dim: int = 3) -> keras.Model:
    """A position table at ``pos_embed/pos_embedding`` of ``1 + side * side`` rows (or ``rows``) and a head."""
    keras.utils.set_random_seed(seed)
    n = rows if rows is not None else 1 + side * side
    inputs = keras.Input((n, dim))
    x = PositionalEmbedding(max_seq_len=n, dim=dim, name="pos_embed")(inputs)
    model = keras.Model(inputs, keras.layers.Dense(2, name="head")(x))
    for variable in model.weights:
        variable.assign(np.random.default_rng(seed).normal(size=variable.shape).astype(variable.dtype))
    return model


def holding(model: keras.Model, weights: Sequence[Any]) -> types.SimpleNamespace:
    """``model``'s name with other weights: what ``plan_transfer`` reads of a model."""
    return types.SimpleNamespace(name=model.name, weights=list(weights))


class Cell(keras.layers.Layer):
    """A layer named ``cell`` holding one ``Dense`` per name, so a name suffix can differ inside it."""

    def __init__(self, names: Sequence[str], units: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.inner = [keras.layers.Dense(n, name=name) for name, n in zip(names, units)]

    def build(self, input_shape):
        for layer in self.inner:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, x):
        for layer in self.inner:
            x = layer(x)
        return x


class Toy(keras.Model):
    """A subclassed model: every weight path starts with the model's own name (``<name>/stem/kernel``)."""

    def __init__(self, names: Sequence[str] = ("rnn", "rnn_1"), units: Sequence[int] = (4, 4), **kwargs):
        super().__init__(**kwargs)
        self.stem = keras.layers.Dense(4, name="stem")
        self.cell = Cell(names, units, name="cell")
        self.head = keras.layers.Dense(2, name="head")

    def call(self, x):
        return self.head(self.cell(self.stem(x)))


def toy(seed: int, name: str, **kwargs) -> Toy:
    keras.utils.set_random_seed(seed)
    model = Toy(name=name, **kwargs)
    model(np.zeros((1, 5), "float32"))  # a call builds every sublayer
    for variable in model.weights:
        variable.assign(np.random.default_rng(seed).normal(size=variable.shape).astype(variable.dtype))
    return model


def values(model: keras.Model) -> Dict[str, np.ndarray]:
    return {v.path: np.array(v.numpy()) for v in model.weights}


def assert_unchanged(model: keras.Model, before: Dict[str, np.ndarray]) -> None:
    after = values(model)
    assert list(after) == list(before)
    for path, value in before.items():
        np.testing.assert_array_equal(after[path], value, err_msg=path)


def refuse(source: keras.Model, target: keras.Model, spec: TransferSpec, **kwargs) -> str:
    """Plan, check and apply; require a ``ConfigError`` and every target weight left as it was."""
    before = values(target)
    with pytest.raises(ConfigError) as raised:
        apply(plan_transfer(source, target, spec, **kwargs))
    assert_unchanged(target, before)
    return str(raised.value)


def paths(model: keras.Model, prefix: str) -> List[str]:
    return [v.path for v in model.weights if v.path.startswith(prefix + "/")]


# --- copy ----------------------------------------------------------------------------------------


def test_the_seed_generator_state_is_not_a_weight():
    model = classifier(0)
    assert any("seed_generator" in v.path for v in model.variables)
    assert not any("seed_generator" in v.path for v in model.weights)


def test_an_equal_architecture_is_copied_exactly():
    source, target = classifier(1), classifier(2)
    plan = plan_transfer(source, target, VIT)
    check(plan)
    apply(plan)
    assert plan.copied == tuple(v.path for v in target.weights)
    assert "norm/moving_mean" in plan.copied and "norm/moving_variance" in plan.copied
    assert (plan.reset, plan.grown, plan.dropped, plan.resampled) == ((), (), (), ())
    want = values(source)
    for path, value in values(target).items():
        np.testing.assert_array_equal(value, want[path], err_msg=path)


@pytest.mark.parametrize("classes", [3, 7])
def test_a_reset_head_stays_fresh_and_the_source_head_is_dropped(classes):
    source, target = classifier(1), classifier(2, classes=classes)
    fresh = values(target)
    plan = plan_transfer(source, target, VIT, head_reset=True)
    apply(plan)
    assert plan.reset == ("head/kernel", "head/bias") and plan.dropped == ("head/kernel", "head/bias")
    want = values(source)
    for path, value in values(target).items():
        np.testing.assert_array_equal(value, fresh[path] if path.startswith("head/") else want[path], err_msg=path)


def test_a_cross_task_prefix_map_fills_the_backbone_and_drops_the_classifier_head():
    source = classifier(1, head="classification_head")
    target = segmenter(2)
    fresh = values(target)
    spec = TransferSpec(head=("segmentation_head",),
                        sources={("image_classification", "toy"): CrossTask("backbone/", ("classification_head",))})
    plan = plan_transfer(source, target, spec, relation=spec.sources[("image_classification", "toy")])
    apply(plan)
    assert plan.dropped == ("classification_head/kernel", "classification_head/bias")
    assert plan.reset == ("segmentation_head/kernel", "segmentation_head/bias")
    assert plan.copied == tuple(paths(target, "backbone")) and len(plan.copied) == len(source.weights) - 2
    want = values(source)
    for path, value in values(target).items():
        if path.startswith("backbone/"):
            np.testing.assert_array_equal(value, want[path[len("backbone/"):]], err_msg=path)
        else:
            np.testing.assert_array_equal(value, fresh[path], err_msg=path)


def test_a_cross_task_source_head_reaches_no_target_weight():
    source = classifier(1, head="classification_head")
    message = refuse(source, segmenter(2), TransferSpec(head=("segmentation_head",)),
                     relation=CrossTask("backbone/", ()))
    assert "classification_head/kernel" in message and "neither used nor dropped" in message


# --- grow ----------------------------------------------------------------------------------------


def test_grow_fills_only_the_new_blocks():
    source, target = classifier(1, blocks=2), classifier(2, blocks=4)
    fresh = values(target)
    plan = plan_transfer(source, target, VIT)
    apply(plan)
    assert plan.grown == ("block_2/kernel", "block_2/bias", "block_3/kernel", "block_3/bias")
    want = values(source)
    for path, value in values(target).items():
        np.testing.assert_array_equal(value, fresh[path] if path in plan.grown else want[path], err_msg=path)


def test_a_new_weight_outside_the_block_pattern_is_refused():
    message = refuse(classifier(1), classifier(2, extra=True), VIT)
    assert "2 of 14 weights of the target model have no source weight" in message
    assert "extra/kernel: target (4, 4) float32, source none" in message


def test_without_a_grow_pattern_a_deeper_target_is_refused():
    message = refuse(classifier(1, blocks=2), classifier(2, blocks=3), TransferSpec(head=("head",)))
    assert "block_2/kernel" in message and "nothing was copied" in message


def test_a_grown_block_name_that_the_source_has_is_refused():
    # A block the source holds is not new: its other weights must come from the source, not stay fresh.
    source = classifier(1, blocks=2)
    target = classifier(2, blocks=2)
    plan = plan_transfer(holding(source, [v for v in source.weights if v.path != "block_1/bias"]), target, VIT)
    with pytest.raises(ConfigError, match="block_1/bias: target \\(4,\\) float32, source none"):
        check(plan)


# --- refusals ------------------------------------------------------------------------------------


def test_an_undeclared_shape_mismatch_is_refused_naming_paths_and_shapes():
    message = refuse(classifier(1, classes=3), classifier(2, classes=7), VIT)
    assert message.startswith("Invalid value for 'init': 2 of 12 weights of the target model have no source weight")
    assert "head/kernel: target (4, 7) float32, source (4, 3) float32" in message
    assert "head/bias: target (7,) float32, source (3,) float32" in message
    assert message.endswith("; nothing was copied (paths without the model's root)")


def test_a_refusal_lists_five_paths_and_counts_the_rest():
    message = refuse(classifier(1, width=4), classifier(2, width=6), VIT)
    # Every weight but head/bias, whose shape (3,) does not depend on the width.
    assert "11 of 12 weights" in message and "(+6 more)" in message
    assert message.count("target (") == 5


def test_an_unused_source_weight_is_refused():
    message = refuse(classifier(1, blocks=3), classifier(2, blocks=2), VIT)
    assert message == ("Invalid value for 'init': 2 of 14 weights of the source run are neither used nor dropped "
                       "by a declared rule: block_2/kernel (4, 4) float32; block_2/bias (4,) float32; "
                       "nothing was copied (paths without the model's root)")


def test_a_dtype_mismatch_is_refused():
    source = classifier(1)
    target = classifier(2)
    with keras.name_scope("head"):
        fake = keras.Variable(np.zeros((3,), "float16"), name="bias")
    assert fake.path == "head/bias"
    weights = [fake if v.path == "head/bias" else v for v in source.weights]
    plan = plan_transfer(holding(source, weights), target, VIT)
    with pytest.raises(ConfigError, match=r"head/bias: target \(3,\) float32, source \(3,\) float16"):
        apply(plan)


# --- paths relative to the model (D-030) and positional pairing (D-031) ------------------------------


def test_the_model_name_is_not_part_of_a_path():
    # vit_pico and vit_tiny root their paths at vision_transformer_pico and vision_transformer_tiny.
    source, target = toy(1, "toy_pico"), toy(2, "toy_tiny")
    assert source.weights[0].path == "toy_pico/stem/kernel" and target.weights[0].path == "toy_tiny/stem/kernel"
    plan = plan_transfer(source, target, VIT)
    apply(plan)
    assert plan.copied == ("stem/kernel", "stem/bias", "cell/rnn/kernel", "cell/rnn/bias", "cell/rnn_1/kernel",
                           "cell/rnn_1/bias", "head/kernel", "head/bias")
    want = [np.array(v.numpy()) for v in source.weights]
    for value, variable in zip(want, target.weights):
        np.testing.assert_array_equal(variable.numpy(), value, err_msg=variable.path)


def test_the_root_every_weight_shares_or_the_models_name_is_dropped_and_nothing_else():
    # D-045, measured: accunet (named ACC_UNet) is built under acc_u_net/ and its loaded archive has no root; a loaded
    # depth model keeps its name on the head's weights only. Neither may change a key; a second component stays.
    def model(name: str, *paths: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(name=name, weights=[types.SimpleNamespace(path=p, shape=(2,), dtype="float32")
                                                         for p in paths])

    built = model("ACC_UNet", "acc_u_net/encoder/kernel", "acc_u_net/head/kernel")
    for source in (built, model("ACC_UNet", "encoder/kernel", "head/kernel"),
                   model("depth", "encoder/kernel", "depth/head/kernel")):
        plan = plan_transfer(source, built, TransferSpec(head=("head",)), head_reset=True)
        check(plan)
        assert (plan.copied, plan.reset, plan.dropped) == (("encoder/kernel",), ("head/kernel",), ("head/kernel",))
    for kept in (model("m", "m/block/inner/kernel"), model("m", "block/kernel", "bias")):
        assert plan_transfer(kept, kept, TransferSpec()).copied == tuple(p.split("/", 1)[1] if p.startswith("m/") else p
                                                                         for p in (w.path for w in kept.weights))


class Rooted(keras.Model):
    """A classifier as a subclassed model: the trunk of :class:`Backbone` and a head, under the model's name."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trunk = [keras.layers.Dense(4, name="stem"), keras.layers.BatchNormalization(name="norm")]
        self.trunk += [keras.layers.Dense(4, name=f"block_{i}") for i in range(2)]
        self.head = keras.layers.Dense(3, name="classification_head")

    def call(self, x):
        for layer in self.trunk:
            x = layer(x)
        return self.head(x)


def test_a_cross_task_prefix_acts_on_the_path_relative_to_the_model():
    # resnet (paths under res_net/) into resnet_seg (backbone/): the prefix replaces the dropped model name.
    keras.utils.set_random_seed(1)
    source = Rooted(name="res_net")
    source(np.zeros((1, 5), "float32"))
    assert source.weights[0].path == "res_net/stem/kernel"
    target = segmenter(2)
    spec = TransferSpec(head=("segmentation_head",),
                        sources={("image_classification", "toy"): CrossTask("backbone/", ("classification_head",))})
    plan = plan_transfer(source, target, spec, relation=spec.sources[("image_classification", "toy")])
    apply(plan)
    assert plan.copied == tuple(paths(target, "backbone")) and len(plan.copied) == len(source.weights) - 2
    assert plan.dropped == ("classification_head/kernel", "classification_head/bias")
    for want, variable in zip(source.weights, [v for v in target.weights if v.path.startswith("backbone/")]):
        np.testing.assert_array_equal(variable.numpy(), want.numpy(), err_msg=variable.path)


def test_an_inner_name_suffix_is_refused_without_a_positional_declaration():
    source = toy(1, "toy", names=("rnn_2", "rnn_3"))
    message = refuse(source, toy(2, "toy"), VIT)
    assert "4 of 8 weights of the target model have no source weight" in message
    assert "cell/rnn/kernel: target (4, 4) float32, source none" in message
    assert "toy/" not in message


def test_a_positional_component_pairs_its_weights_by_order():
    source, target = toy(1, "toy", names=("rnn_2", "rnn_3")), toy(2, "toy")
    plan = plan_transfer(source, target, TransferSpec(head=("head",), positional=("cell",)))
    apply(plan)
    assert len(plan.copied) == 8 and plan.missing == () and plan.unused == ()
    for want, variable in zip(source.weights, target.weights):
        np.testing.assert_array_equal(variable.numpy(), want.numpy(), err_msg=variable.path)


@pytest.mark.parametrize("names,units", [(("rnn_2",), (4,)), (("rnn_2", "rnn_3", "rnn_4"), (4, 4, 4)),
                                         (("rnn_2", "rnn_3"), (4, 3))], ids=["fewer", "more", "shape"])
def test_a_positional_component_that_does_not_pair_is_refused(names, units):
    source = toy(1, "toy", names=names, units=units)
    message = refuse(source, toy(2, "toy"), TransferSpec(head=("head",), positional=("cell",)))
    assert "cell/rnn/kernel: target (4, 4) float32, source none" in message
    assert "neither used nor dropped" in message or "have no source weight" in message


def test_a_positional_component_is_one_top_level_name():
    with pytest.raises(ValueError, match="positional"):
        TransferSpec(positional=("cell/rnn",))


# --- resample ------------------------------------------------------------------------------------


@pytest.mark.parametrize("small,large,dim", [(4, 8, 3), (4, 7, 3), (8, 16, 192)])
def test_the_table_is_resampled_as_torch_bicubic(small, large, dim):
    source, target = table_model(1, small, dim=dim), table_model(2, large, dim=dim)
    plan = plan_transfer(source, target, VIT)
    apply(plan)
    assert plan.resampled == ((TABLE, "vit_pos_bicubic", (1, 1 + small * small, dim), (1, 1 + large * large, dim)),)
    assert plan.copied == ("head/kernel", "head/bias")
    table = values(source)[TABLE][0]
    got = values(target)[TABLE][0]
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[0], table[0])
    grid = table[1:].reshape(small, small, dim).transpose(2, 0, 1).astype(np.float64)
    want = torch_bicubic_2d(grid, large, large).transpose(1, 2, 0).reshape(-1, dim)
    assert np.abs(got[1:] - want).max() <= 1e-6


def test_a_smaller_grid_is_refused():
    message = refuse(table_model(1, 8), table_model(2, 4), VIT)
    assert message == ("Invalid value for 'init': vit pos_embed would shrink its grid from 8x8 to 4x4; "
                       "only a larger grid is resampled (bicubic, no antialiasing)")


def test_a_non_square_grid_is_refused():
    message = refuse(table_model(1, 4), table_model(2, 4, rows=1 + 24), VIT)
    assert "1 + 24 rows is not a square grid" in message


def test_an_undeclared_table_change_is_refused():
    message = refuse(table_model(1, 4), table_model(2, 8), TransferSpec(head=("head",)))
    assert f"{TABLE}: target (1, 65, 3) float32, source (1, 17, 3) float32" in message


def test_an_unknown_rule_name_is_a_spec_error():
    with pytest.raises(ValueError, match="nearest"):
        TransferSpec(resample={TABLE: "nearest"})
