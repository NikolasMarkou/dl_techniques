"""SqueezeNet's bypass variant must really wire bypasses, and its head must
emit logits at every class count and in both ranks.

Guard for C-9 (plan-2026-08-14T233721-d4f9beb2, step 32).

Two defects, both invisible to the pre-existing suite:

1. ``squeezenet_v1.py`` selected ``bypass_indices = [2, 4, 6, 8]`` for
   ``use_bypass="simple"``. Fire modules are named ``fire{idx+2}``, so fire3/5/7/9
   -- the channel-matched positions -- are ``idx = 1, 3, 5, 7``. The listed
   indices are all channel-MISMATCHED (plus an out-of-range 8), so the
   ``identity.shape[-1] == x.shape[-1]`` guard was false everywhere and
   ``create_squeezenet_v1("1.0_bypass")`` built a graph topologically identical
   to plain ``"1.0"`` while still reporting ``use_bypass == "simple"``.
   ``test_bypass_configurations`` asserts only that the model builds and the
   output shape is right, and ``test_model_variants`` asserts the stored flag --
   which is exactly how this survived.

2. ``squeezenet_v2.py``'s head selected ``'sigmoid'`` at ``num_classes == 2``
   under a comment reading "Softmax activation for binary classification". With
   ``Conv2D(2) -> GAP -> sigmoid`` the two outputs are independent and do not sum
   to 1, so the package's own documented ``num_classes=2`` nodule examples,
   compiled with ``categorical_crossentropy``, optimize something else.
   ``SqueezeNetV1`` softmaxed on the same argument.

Since plan-2026-10-14T062616-81938e81 step 4 neither model applies ANY output
activation: both return the paper head, ``conv10`` (ReLU) averaged over space,
as logits, and the loss applies the softmax. The 2D V2 head had been the shared
``ClassificationHead`` (two hidden blocks, softmax) since 2026-10-09 while the
3D one kept ``conv10`` plus its own ``Activation('softmax')``; both ranks now
build the same ``conv10`` head. ``TestTheHeadEmitsLogits`` pins that, rank by
rank, by recomputing the output from ``conv10``'s own activations.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.squeezenet.squeezenet_v1 import (
    SqueezeNetV1,
    create_squeezenet_v1,
)
from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
from dl_techniques.models.vision.squeezenet.spatial_guard import minimum_spatial_extent


def _add_layers(model):
    return [l for l in model.layers if isinstance(l, keras.layers.Add)]


def _channel_matched_positions(fire_configs):
    """Positions where a residual Add is dimensionally possible, derived from the
    variant config alone (input channels of fire idx == output of fire idx-1)."""
    widths = [c["e1x1"] + c["e3x3"] for c in fire_configs]
    return [idx for idx in range(1, len(widths)) if widths[idx - 1] == widths[idx]]


class TestSimpleBypassIsRealWiring:
    def test_bypass_variant_builds_add_layers(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        adds = _add_layers(model)
        assert len(adds) > 0, "1.0_bypass built no residual Add layers at all"

    def test_add_count_equals_the_channel_matched_positions(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        expected = _channel_matched_positions(
            SqueezeNetV1.MODEL_VARIANTS["1.0_bypass"]["fire_configs"]
        )
        adds = _add_layers(model)
        assert len(adds) == len(expected)
        # fire modules are named fire{idx+2}; the Add is add_fire{idx+2}
        assert sorted(l.name for l in adds) == sorted(
            f"add_fire{idx + 2}" for idx in expected
        )

    def test_bypass_variant_differs_topologically_from_plain(self):
        """Anti-vacuity: the flag alone is not the property. At HEAD the two
        graphs had an identical Add count (zero)."""
        plain = create_squeezenet_v1("1.0", num_classes=10)
        bypass = create_squeezenet_v1("1.0_bypass", num_classes=10)
        assert bypass.use_bypass == "simple"
        assert plain.use_bypass is False or plain.use_bypass == False  # noqa: E712
        assert len(_add_layers(bypass)) > len(_add_layers(plain))

    def test_bypass_model_still_forwards(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        out = model(np.random.rand(2, 224, 224, 3).astype("float32"))
        arr = keras.ops.convert_to_numpy(out)
        assert arr.shape == (2, 10)
        assert np.all(np.isfinite(arr))


#: (case id, builder, per-sample input shape). The 3D row is the one whose
#: ``Activation('softmax', name='predictions')`` (anchor
#: plan-2026-08-14T233721-d4f9beb2/D-063) was deleted at step 4.
_LOGIT_CASES = [
    ("v1_2d_binary", lambda: SqueezeNetV1.from_variant(
        "1.1", num_classes=2, input_shape=(64, 64, 3), dropout_rate=0.0),
     (64, 64, 3)),
    ("v2_2d_binary", lambda: SqueezeNoduleNetV2.from_variant(
        "v2", num_classes=2, input_shape=(64, 64, 1), dropout_rate=0.0),
     (64, 64, 1)),
    ("v2_2d_three", lambda: SqueezeNoduleNetV2.from_variant(
        "v2", num_classes=3, input_shape=(64, 64, 1), dropout_rate=0.0),
     (64, 64, 1)),
    # 64, not 40 or 48: at both of those the final map is 1 x 1 x 1, and at
    # seed 0 both binary conv10 units start below zero, so the ReLU'd logits
    # are exactly 0.0 for both samples (MEASURED, CPU) and nothing here could
    # discriminate. At 64 the map is 2 x 2 x 2.
    ("v2_3d_binary", lambda: SqueezeNoduleNetV2.from_variant(
        "v2_3d", num_classes=2, input_shape=(64, 64, 64, 1), dropout_rate=0.0),
     (64, 64, 64, 1)),
]


class TestTheHeadEmitsLogits:
    """No output activation in either model, in 2D or 3D, at 2 or 3 classes."""

    @pytest.mark.parametrize(
        "case,build,shape", _LOGIT_CASES, ids=[c[0] for c in _LOGIT_CASES])
    def test_no_softmax_or_sigmoid_layer_exists(self, case, build, shape):
        keras.utils.set_random_seed(0)
        model = build()
        for layer in model._flatten_layers(include_self=False):
            assert not isinstance(layer, keras.layers.Softmax), (case, layer.name)
            activation = getattr(layer, "activation", None)
            assert activation not in (
                keras.activations.softmax, keras.activations.sigmoid
            ), (case, layer.name)
        assert model.layers[-1].name == "avgpool", (
            f"{case}: the model's last layer is {model.layers[-1].name}, "
            f"not the global pool"
        )

    @pytest.mark.parametrize(
        "case,build,shape", _LOGIT_CASES, ids=[c[0] for c in _LOGIT_CASES])
    def test_the_output_is_the_spatial_mean_of_relu_conv10(self, case, build, shape):
        """Recompute the output from ``conv10``'s activations; nothing may follow.

        A re-added softmax (or the old binary sigmoid) after the pool fails the
        value comparison, and also the "not normalized" assertion below. The
        tolerance is the float32 reassociation bound for a mean over the
        ``conv10`` map, derived, not pasted.
        """
        from tests.numerics import reassociation_atol

        keras.utils.set_random_seed(0)
        model = build()
        x = (3.0 * np.random.default_rng(0).standard_normal((2, *shape))).astype("float32")
        conv10 = model.get_layer("conv10")
        assert conv10.activation is keras.activations.relu, case
        tap = keras.Model(model.input, conv10.output)

        out = keras.ops.convert_to_numpy(model(x, training=False)).astype(np.float64)
        maps = keras.ops.convert_to_numpy(tap(x, training=False)).astype(np.float64)
        spatial_axes = tuple(range(1, maps.ndim - 1))
        expected = maps.mean(axis=spatial_axes)
        positions = int(np.prod(maps.shape[1:-1]))
        atol = reassociation_atol([positions], 1, float(np.abs(expected).max()))
        np.testing.assert_allclose(out, expected, rtol=0, atol=atol, err_msg=case)

        assert np.all(out >= 0.0), f"{case}: a ReLU'd conv10 averages to >= 0"
        assert float(np.abs(out).max()) > 0.0, f"{case}: every logit is zero"
        rows = out.sum(axis=-1)
        assert np.any(np.abs(rows - 1.0) > reassociation_atol([out.shape[-1]], 1, 1.0)), (
            f"{case}: every output row sums to 1 ({rows}); the model returns "
            f"probabilities, not logits"
        )


class TestDocstringExampleIsConstructible:
    """The class docstring's own example must forward FINITE values.

    Until 2026-08-18 this class pinned `1.0_bypass` at (32, 32, 3) -- the size
    the docstring advertised -- and asserted only ``out.shape == (2, 10)``. That
    assertion passes on an all-NaN tensor, which is exactly what the model
    returned: at 32 px the "1.0" stem family walks 32 -> 13 -> 6 -> 2 -> 0, and
    a zero-length spatial axis makes GlobalAveragePooling2D divide by zero
    without Keras raising anywhere. Shape is not the property; finiteness is
    (its sibling `test_bypass_model_still_forwards` at 224 px already knew).
    """

    def test_corrected_docstring_example_builds_and_forwards_finite(self):
        model = SqueezeNetV1.from_variant(
            "1.0_bypass", num_classes=10, input_shape=(64, 64, 3)
        )
        out = keras.ops.convert_to_numpy(
            model(np.random.rand(2, 64, 64, 3).astype("float32"))
        )
        assert out.shape == (2, 10)
        assert np.all(np.isfinite(out))

    def test_sub_floor_input_is_refused_instead_of_returning_nan(self):
        with pytest.raises(ValueError, match=r"collapses to length 0"):
            SqueezeNetV1.from_variant(
                "1.0_bypass", num_classes=10, input_shape=(32, 32, 3)
            )

    def test_the_floor_is_computed_per_variant_not_hard_coded(self):
        """35 for the 7x7-stem family, 31 for "1.1" -- both derived from the
        variant's own kernel/stride/pool schedule."""
        assert minimum_spatial_extent(SqueezeNetV1.MODEL_VARIANTS["1.0"]) == 35
        assert minimum_spatial_extent(SqueezeNetV1.MODEL_VARIANTS["1.0_bypass"]) == 35
        assert minimum_spatial_extent(SqueezeNetV1.MODEL_VARIANTS["1.1"]) == 31
        for variant in SqueezeNoduleNetV2.MODEL_VARIANTS.values():
            assert minimum_spatial_extent(variant) == 35

    def test_at_the_floor_the_output_is_finite_and_one_below_it_raises(self):
        """Anti-vacuity: the computed floor is exact in both directions."""
        floor = minimum_spatial_extent(SqueezeNetV1.MODEL_VARIANTS["1.1"])
        model = SqueezeNetV1.from_variant(
            "1.1", num_classes=10, input_shape=(floor, floor, 3)
        )
        out = keras.ops.convert_to_numpy(
            model(np.random.rand(2, floor, floor, 3).astype("float32"))
        )
        assert np.all(np.isfinite(out))
        with pytest.raises(ValueError, match=r"minimum legal spatial extent"):
            SqueezeNetV1.from_variant(
                "1.1", num_classes=10, input_shape=(floor - 1, floor - 1, 3)
            )

    def test_three_d_variant_guard_applies_to_every_axis(self):
        with pytest.raises(ValueError, match=r"collapses to length 0"):
            SqueezeNoduleNetV2.from_variant(
                "v2_3d", num_classes=2, input_shape=(48, 48, 32, 1)
            )


class TestUseBypassOverrideIsHonoured:
    """`use_bypass=False` must be able to turn a bypass variant's bypass OFF.

    `use_bypass if use_bypass else variant_config.get("use_bypass")` treats an
    explicit `False` as "unset", so the variant's own `"simple"` won.
    """

    def test_explicit_false_disables_the_variant_bypass(self):
        model = SqueezeNetV1.from_variant(
            "1.0_bypass", num_classes=10, input_shape=(64, 64, 3), use_bypass=False
        )
        assert model.use_bypass is False
        assert len(_add_layers(model)) == 0

    def test_omitting_the_argument_still_defers_to_the_variant(self):
        model = SqueezeNetV1.from_variant(
            "1.0_bypass", num_classes=10, input_shape=(64, 64, 3)
        )
        assert model.use_bypass == "simple"
        assert len(_add_layers(model)) == 4
