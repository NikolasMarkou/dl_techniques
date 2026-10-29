"""Unit contract for ``dl_techniques.utils.model_build.concretize_axes``.

``concretize_axes`` was added by plan ``plan-2026-08-23T091307-9a110062`` (D-420)
for exactly two callers -- ``Qwen3Next.build`` and ``SCUNet.build`` -- and both
exercise it only through the one shape their own model uses. Its docstring makes
three promises those two paths never touch: a CONCRETE axis is left alone, an
axis a shape is too short to have is ignored, and a negative axis index raises.
Each is asserted here, so a "simplification" of the walk fails on the clause it
broke rather than on a model test three packages away.

The substitution itself is measured where it matters, on the models:
``tests/test_models/test_the_explicit_build_materializes_the_model.py``.
"""

import ast
import inspect
import re
from pathlib import Path

import keras
import pytest

from dl_techniques.utils.model_build import concretize_axes


def test_a_none_axis_becomes_the_requested_extent():
    assert concretize_axes((None, None), {1: 1}) == (None, 1)


def test_a_concrete_axis_is_left_alone():
    """The clause both callers depend on and neither one proves.

    ``Qwen3Next`` is built from ``(None, 16)`` by its own test harness and from
    ``(None, None)`` by the functional factories; only the second substitutes.
    A version that overwrote the concrete 16 with the probe value would still
    materialize 97 weights and pass every model test.
    """
    assert concretize_axes((None, 16), {1: 1}) == (None, 16)
    assert concretize_axes((None, 96, 128, 3), {1: 64, 2: 64}) == (None, 96, 128, 3)


def test_an_axis_the_shape_is_too_short_to_have_is_ignored():
    assert concretize_axes((None,), {1: 1, 2: 1}) == (None,)


def test_the_batch_axis_is_substitutable_like_any_other():
    """No axis is privileged -- axis 0 is not special-cased away."""
    assert concretize_axes((None, None), {0: 4}) == (4, None)


def test_a_dict_nest_is_walked_and_the_input_is_not_mutated():
    given = {"input_ids": (None, None), "attention_mask": (None, None)}
    got = concretize_axes(given, {1: 1})
    assert got == {"input_ids": (None, 1), "attention_mask": (None, 1)}
    assert given == {"input_ids": (None, None), "attention_mask": (None, None)}


def test_a_list_nest_is_walked():
    got = concretize_axes([(None, 16, 3), (None, None, 3)], {1: 8})
    assert got == [(None, 16, 3), (None, 8, 3)]


def test_no_replacements_is_a_structure_preserving_identity():
    assert concretize_axes((None, None, 3), {}) == (None, None, 3)


def test_a_negative_axis_raises_rather_than_wrapping():
    """``-1`` would silently mean "last axis" to Python indexing.

    ``list.__setitem__`` accepts it, so without the guard
    ``concretize_axes((None, None), {-1: 1})`` would substitute the CHANNEL
    axis while the caller meant the batch axis, and nothing downstream would
    notice.
    """
    with pytest.raises(ValueError, match="non-negative"):
        concretize_axes((None, None), {-1: 1})


class TestPlainHeadConfigOverrides:
    def test_none_and_empty_mean_no_overrides(self):
        from dl_techniques.utils.model_build import plain_head_config_overrides

        assert plain_head_config_overrides(None, "M") is None
        assert plain_head_config_overrides({}, "M") is None

    def test_json_data_is_copied_not_aliased(self):
        from dl_techniques.utils.model_build import plain_head_config_overrides

        given = {"hidden_dims": [8, 4], "activation": "gelu",
                 "kernel_regularizer": {"class_name": "L2", "config": {"l2": 1e-4}}}
        got = plain_head_config_overrides(given, "M")
        assert got == given and got is not given

    def test_a_live_object_is_refused_naming_the_key_and_the_type(self):
        import keras

        from dl_techniques.utils.model_build import plain_head_config_overrides

        with pytest.raises(ValueError) as caught:
            plain_head_config_overrides(
                {"activation": "gelu", "kernel_regularizer": keras.regularizers.L2(1e-4)}, "Toy")
        message = str(caught.value)
        assert "Toy" in message and "head_config_overrides['kernel_regularizer']" in message
        assert "L2" in message and "activation" not in message
        assert isinstance(caught.value.__cause__, TypeError)


class TestCoerceIncludeTop:
    """``coerce_include_top`` is the one ``include_top`` rule the vision classifiers share."""

    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_comes_back_unchanged(self, value):
        from dl_techniques.utils.model_build import coerce_include_top

        assert coerce_include_top(value) is value

    def test_a_numpy_bool_becomes_a_python_bool(self):
        import json

        import numpy as np

        from dl_techniques.utils.model_build import coerce_include_top

        got = coerce_include_top(np.False_)
        assert got is False
        assert json.dumps({"include_top": got}) == '{"include_top": false}'

    @pytest.mark.parametrize("value", ["no", "", 0, 1, None, 0.0, [True]])
    def test_anything_else_is_refused_naming_the_value_and_its_type(self, value):
        from dl_techniques.utils.model_build import coerce_include_top

        with pytest.raises(TypeError, match="include_top must be a bool or numpy.bool_") as caught:
            coerce_include_top(value)
        assert repr(value) in str(caught.value)
        assert type(value).__qualname__ in str(caught.value)

    def test_a_numpy_integer_is_refused_with_its_module_qualified_type(self):
        """``type(np.int64(0)).__name__`` is ``int64`` and ``type(np.False_).__name__`` is
        ``bool`` under numpy 2: the message carries the module so the refusal of one
        cannot read as a refusal of the other."""
        import numpy as np

        from dl_techniques.utils.model_build import coerce_include_top

        with pytest.raises(TypeError, match=r"numpy\.int64"):
            coerce_include_top(np.int64(0))


class TestValidateRates:
    """``validate_rates`` names the model's own argument, not the sub-layer's."""

    def test_rates_inside_the_unit_interval_pass(self):
        from dl_techniques.utils.model_build import validate_rates

        validate_rates(dropout_rate=0.0, drop_path_rate=0.999, spatial_dropout_rate=0.5)

    def test_no_rates_pass(self):
        from dl_techniques.utils.model_build import validate_rates

        validate_rates()

    @pytest.mark.parametrize("value", [-1.0, -1e-9, 1.0, 2.0, float("nan")])
    def test_a_rate_outside_zero_one_is_refused_naming_the_argument_and_the_value(self, value):
        from dl_techniques.utils.model_build import validate_rates

        with pytest.raises(ValueError) as caught:
            validate_rates(dropout_rate=0.1, drop_path_rate=value)
        assert str(caught.value) == f"drop_path_rate must be in [0, 1), got {value}"

    def test_the_first_bad_rate_in_call_order_is_the_one_named(self):
        from dl_techniques.utils.model_build import validate_rates

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            validate_rates(dropout_rate=2.0, spatial_dropout_rate=-1.0)


class TestRequireChannelCount:
    """``require_channel_count``: the last axis of an unbatched input shape must be known."""

    @pytest.mark.parametrize("shape", [(32, 48, 3), (None, None, 3), [None, None, 1], (3,)])
    def test_a_known_channel_count_passes_whatever_the_spatial_axes_are(self, shape):
        from dl_techniques.utils.model_build import require_channel_count

        require_channel_count(shape)

    @pytest.mark.parametrize("shape", [(None, None, None), (32, 48, None), (None,), ()])
    def test_an_unknown_channel_count_is_refused_naming_the_argument_and_the_tuple(self, shape):
        from dl_techniques.utils.model_build import require_channel_count

        with pytest.raises(ValueError, match="input_shape") as caught:
            require_channel_count(shape)
        assert f"input_shape={tuple(shape)}" in str(caught.value)

    def test_only_the_last_axis_counts(self):
        from dl_techniques.utils.model_build import require_channel_count

        require_channel_count((None, 7))
        with pytest.raises(ValueError):
            require_channel_count((7, None))


class TestRefuseInputShape:
    """``refuse_input_shape``: the shape keywords Keras would pop and only warn about."""

    @pytest.mark.parametrize("keyword, value", [("input_shape", (8, 8, 3)), ("input_dim", 3)])
    def test_a_shape_keyword_is_refused_naming_the_class_the_keyword_and_the_value(self, keyword, value):
        from dl_techniques.utils.model_build import refuse_input_shape

        with pytest.raises(ValueError) as caught:
            refuse_input_shape({keyword: value}, "Toy", "it has no shape")
        assert str(caught.value) == f"Toy takes no {keyword} (it has no shape); got {keyword}={value!r}"

    def test_other_keywords_and_an_empty_mapping_pass(self):
        from dl_techniques.utils.model_build import refuse_input_shape

        refuse_input_shape({}, "Toy", "why")
        refuse_input_shape({"name": "toy", "trainable": False}, "Toy", "why")

    def test_the_mapping_is_not_mutated(self):
        from dl_techniques.utils.model_build import refuse_input_shape

        kwargs = {"name": "toy"}
        refuse_input_shape(kwargs, "Toy", "why")
        assert kwargs == {"name": "toy"}

    def test_keras_alone_accepts_the_keyword_with_a_warning(self):
        """The control: without the helper the keyword is swallowed, which is why the helper exists."""
        import warnings

        class Toy(keras.layers.Layer):
            pass

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            Toy(input_shape=(8, 8, 3))
        assert any("input_shape" in str(w.message) for w in seen)


# ---------------------------------------------------------------------
# The caller lists in the docstrings are the real callers
# ---------------------------------------------------------------------

_SRC = Path(__file__).resolve().parents[2] / "src" / "dl_techniques"


def _enclosing_callers(name: str, defining: Path) -> set:
    """Names of the classes (or functions, outside a class) whose code calls ``name``.

    Interface contract: parses every module under ``src/dl_techniques`` except ``defining``;
    a call counts when its callee is ``name`` or ``<expr>.name``. Returns the innermost
    enclosing class name, else the enclosing function name.
    """
    found = set()
    for path in _SRC.rglob("*.py"):
        if path == defining:
            continue
        text = path.read_text()
        if name not in text:
            continue
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            called = callee.id if isinstance(callee, ast.Name) else (
                callee.attr if isinstance(callee, ast.Attribute) else None)
            if called != name:
                continue
            cls = fn = None
            cursor = node
            while cursor in parents:
                cursor = parents[cursor]
                if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn is None:
                    fn = cursor.name
                if isinstance(cursor, ast.ClassDef) and cls is None:
                    cls = cursor.name
            found.add(cls or fn)
    return found


def _contract_paragraph(function) -> str:
    """The caller clause of the interface contract: from "Interface contract" to the first "):" or ")."."""
    doc = inspect.getdoc(function)
    start = doc.index("Interface contract")
    return doc[start:start + re.search(r"\)[:.]", doc[start:]).start()]


_HELPERS = [
    ("validate_rates", "model_build"), ("require_channel_count", "model_build"),
    ("refuse_input_shape", "model_build"), ("coerce_include_top", "model_build"),
    ("refuse_pretrained", "weight_transfer")]


@pytest.mark.parametrize("name, module", _HELPERS)
def test_the_documented_callers_are_the_real_callers(name, module):
    """Every class or function that calls the helper is named in its interface contract, and
    every class-like or ``create_*`` name the contract lists really calls it. The lists were
    written by hand and went stale twice (a caller added, a caller omitted)."""
    import importlib

    owner = importlib.import_module(f"dl_techniques.utils.{module}")
    function = getattr(owner, name)
    real = _enclosing_callers(name, Path(owner.__file__))
    paragraph = _contract_paragraph(function)
    named = set(re.findall(r"``([A-Za-z_][A-Za-z0-9_]*)(?:\.\w+)?``", paragraph))
    assert real, "the scan found no caller: the instrument is blind"
    assert real <= named, f"callers missing from the docstring: {sorted(real - named)}"
    listed = {n for n in named if re.fullmatch(r"_?[A-Z]\w+|create_\w+", n)}
    assert listed <= real, f"names in the docstring that do not call it: {sorted(listed - real)}"


def test_the_caller_scan_sees_a_stale_docstring():
    """Control: a function whose contract omits a real caller and names a stranger is caught."""
    def toy():
        """Toy.

        Interface contract (callers: ``Stranger``): the rest.
        """

    real = {"Actual"}
    named = set(re.findall(r"``([A-Za-z_][A-Za-z0-9_]*)(?:\.\w+)?``", _contract_paragraph(toy)))
    assert not real <= named and {n for n in named if re.fullmatch(r"_?[A-Z]\w+", n)} - real
