"""The transfer engine of ``init``: ``plan_transfer``, ``check``, ``apply`` and its resample rules.

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

A ``resample`` key is a regex that must match the whole path, so one key covers a table in every
encoder layer at any depth; a path two keys match is refused (plan-2026-11-19T081214-5c6cc105 D-005).
The BEiT relative position bias rule is held to unilm's own resize, run verbatim and frozen into
``data/beit_relpos_reference.npz`` by ``data/generate_beit_relpos_reference.py`` (D-004); the Energy
Transformer position table has no reference rule, so its grid change is refused by name (D-006).

``model.weights`` holds no ``SeedGenerator`` state in Keras 3.8 (it is in ``model.variables``),
so a dropout model's seed is never transferred; the first test pins that.
"""

import contextlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.utils.logger import logger
from train.api import init_from
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


# --- regex keys, the BEiT rule against unilm, the Energy Transformer refusal (plan-2026-11-19T081214-5c6cc105) --------

#: unilm's own resize of seeded random tables (``data/generate_beit_relpos_reference.py``, run verbatim by a python3
#: with torch and scipy 1.13.1); case ``<S>_<D>_<heads>``: a table of ``S x S`` patches resized to ``D x D``.
REFERENCE = Path(__file__).parent / "data" / "beit_relpos_reference.npz"
REFERENCE_CASES = ("8_14_3", "8_16_3", "4_8_3", "14_28_3", "8_16_12")
#: The one key that names the table of every BEiT encoder layer.
BEIT_TABLE = r"beit_backbone/encoder_layer_\d+/attention/relative_position_bias_table"
ET_TABLE = "et_backbone/pos_embed/pos_embedding"


@pytest.fixture(scope="module")
def unilm() -> Dict[str, np.ndarray]:
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


def beit_rows(window: int) -> int:
    """The rows of the table of ``window x window`` patches: ``(2 window - 1)^2`` displacements and 3 extra rows."""
    return (2 * window - 1) ** 2 + 3


def weight_at(path: str, value: np.ndarray) -> keras.Variable:
    """A weight at the Keras path ``path``: each component but the last a name scope."""
    *scopes, name = path.split("/")
    with contextlib.ExitStack() as stack:
        for scope in scopes:
            stack.enter_context(keras.name_scope(scope))
        return keras.Variable(value, name=name)


def stand_in(tables: Mapping[str, np.ndarray]) -> types.SimpleNamespace:
    """A model as ``plan_transfer`` reads one: ``tables`` at their paths and ``head/bias``, under no root."""
    weights = [weight_at(path, value) for path, value in tables.items()]
    return types.SimpleNamespace(name="model", weights=[*weights, weight_at("head/bias", np.ones(3, "float32"))])


def beit_layers(layers: int, rows: int, heads: int = 3, seed: int = 0,
                first: Optional[np.ndarray] = None) -> types.SimpleNamespace:
    """A stand-in BEiT trunk: a seeded random table of ``rows`` rows in each encoder layer, ``first`` in layer 0."""
    rng = np.random.default_rng(seed)
    tables = {f"beit_backbone/encoder_layer_{i}/attention/relative_position_bias_table":
              rng.standard_normal((rows, heads)).astype("float32") for i in range(layers)}
    if first is not None:
        tables["beit_backbone/encoder_layer_0/attention/relative_position_bias_table"] = first
    return stand_in(tables)


def beit_spec(extra: Optional[Mapping[str, str]] = None) -> TransferSpec:
    return TransferSpec(head=("head",), resample={BEIT_TABLE: "beit_relpos_cubic", **(extra or {})})


@pytest.mark.parametrize("case", REFERENCE_CASES)
def test_the_beit_rule_equals_unilms_own_resize_and_copies_the_extra_rows(case, unilm):
    table, want = unilm[f"{case}__table"], unilm[f"{case}__resized"]
    got = init_from.beit_relpos_cubic(table, want.shape)
    assert got.shape == want.shape and got.dtype == np.float32
    diff = float(np.abs(got[:-3] - want[:-3]).max())
    logger.info(f"BEIT_UNILM {case}: {table.shape} to {want.shape}, max |beit_relpos_cubic - unilm| = {diff:.3e} "
                f"(unilm's values up to {np.abs(want[:-3]).max():.3f})")
    assert diff <= 1e-6, f"{case}: {diff:.3e} from unilm's own resize"
    np.testing.assert_array_equal(got[-3:], table[-3:])  # cls to token, token to cls, cls to cls: bit for bit
    np.testing.assert_array_equal(want[-3:], table[-3:])


def test_the_reference_holds_the_case_where_unilms_search_saturates(unilm):
    # 4 to 8 patches: three geometric steps reach 7 only at q = 2, so the search stops at its bound 1.5 and the outer
    # targets lie beyond the last source position (4.75); every other case finds q inside the bounds.
    q = {case: float(unilm[f"{case}__q"]) for case in REFERENCE_CASES}
    assert 1.5 - 1e-6 <= q.pop("4_8_3") < 1.5
    assert all(1.01 < value < 1.4 for value in q.values()), q


@pytest.mark.parametrize("layers", [12, 24])
def test_one_regex_key_resamples_the_table_of_every_encoder_layer(layers, unilm):
    # BEiT holds a table per encoder layer (12 at tiny to base, 24 at large, any num_layers): one key full-matches each.
    # Layer 0 holds the reference's table, so what the engine assigns there is unilm's own output.
    source = beit_layers(layers, beit_rows(8), seed=1, first=unilm["8_16_3__table"])
    target = beit_layers(layers, beit_rows(16), seed=2)
    plan = plan_transfer(source, target, beit_spec())
    apply(plan)
    tables = [f"beit_backbone/encoder_layer_{i}/attention/relative_position_bias_table" for i in range(layers)]
    assert plan.resampled == tuple((path, "beit_relpos_cubic", (228, 3), (964, 3)) for path in tables)
    assert plan.copied == ("head/bias",) and (plan.missing, plan.unused) == ((), ())
    got, theirs = values(target), values(source)
    for path in tables:  # every layer from its own source table
        np.testing.assert_array_equal(got[path], init_from.beit_relpos_cubic(theirs[path], (964, 3)), err_msg=path)
    assert np.abs(got[tables[0]][:-3] - unilm["8_16_3__resized"][:-3]).max() <= 1e-6
    np.testing.assert_array_equal(got[tables[0]][-3:], unilm["8_16_3__table"][-3:])


@pytest.mark.parametrize("rows,heads,words", [
    ((beit_rows(16), beit_rows(8)), (3, 3), "would resample a grid of 16x16 patches to 8x8; only a larger grid"),
    ((beit_rows(8), 230), (3, 3), "(230, 3): 230 rows are not the (2W - 1)^2 + 3 of a square grid of W x W patches"),
    ((beit_rows(8), beit_rows(16)), (3, 4), "(228, 3) cannot become (964, 4); only its number of rows"),
    ((beit_rows(2), beit_rows(8)), (3, 3), "(12, 3): 12 rows are not the (2W - 1)^2 + 3 of a square grid of W x W "
                                           "patches with W at least 3"),
], ids=["smaller", "not_square", "heads", "two_by_two"])
def test_a_beit_table_unilms_rule_does_not_cover_is_refused_while_planning(rows, heads, words):
    message = refuse(beit_layers(2, rows[0], heads[0], seed=1), beit_layers(2, rows[1], heads[1], seed=2), beit_spec())
    assert message.startswith("Invalid value for 'init': BEiT relative_position_bias_table "), message
    assert words in message, message


def test_a_path_two_resample_keys_match_is_refused_while_planning():
    other = r"beit_backbone/encoder_layer_0/.+"
    message = refuse(beit_layers(2, beit_rows(8)), beit_layers(2, beit_rows(16)),
                     beit_spec({other: "beit_relpos_cubic"}))
    assert message == ("Invalid value for 'init': beit_backbone/encoder_layer_0/attention/relative_position_bias_table "
                       f"matches 2 resample keys of the family ({BEIT_TABLE}, {other}); a weight takes one rule")


def test_a_key_that_matches_only_the_start_of_a_path_is_not_its_rule():
    spec = TransferSpec(head=("head",), resample={r"beit_backbone/encoder_layer_\d+": "beit_relpos_cubic"})
    message = refuse(beit_layers(1, beit_rows(8)), beit_layers(1, beit_rows(16)), spec)
    assert ("beit_backbone/encoder_layer_0/attention/relative_position_bias_table: target (964, 3) float32, "
            "source (228, 3) float32") in message


def test_a_resample_key_that_is_not_a_regex_is_a_spec_error():
    with pytest.raises(ValueError, match="is not a regex") as raised:
        TransferSpec(resample={r"encoder_layer_(\d+/table": "beit_relpos_cubic"})
    assert "encoder_layer_(" in str(raised.value)


def test_an_energy_transformer_grid_change_is_refused_naming_both_grids_and_nothing_is_assigned():
    rng = np.random.default_rng(0)
    source = stand_in({ET_TABLE: rng.standard_normal((1, 64, 8)).astype("float32")})
    target = stand_in({ET_TABLE: rng.standard_normal((1, 256, 8)).astype("float32")})
    message = refuse(source, target, TransferSpec(head=("head",), resample={ET_TABLE: "et_pos_refused"}))
    assert message == ("Invalid value for 'init': the Energy Transformer position table (1, 64, 8) (8x8 patches) "
                       "cannot become (1, 256, 8) (16x16 patches): no reference rule resamples it (the Energy "
                       "Transformer paper describes no fine-tuning at another resolution); train this run at the "
                       "source's grid (its image_size and patch_size)")


def test_init_from_imports_no_tensorflow_keras_numpy_or_scipy_at_module_scope():
    # The rules import numpy and scipy inside themselves: a registered row that resamples imports init_from, and the
    # help path stays light.
    code = ("import sys\nimport train.api.init_from\n"
            "print(sorted({name.split('.')[0] for name in sys.modules} & {'tensorflow', 'keras', 'numpy', 'scipy'}))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]", result.stdout
