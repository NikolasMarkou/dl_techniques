"""
Guards for the SqueezeNet identity pair -- step 27, D-129.

Two defects, one package, and they only matter together:

1. ``SqueezeNetV1.from_config`` / ``SqueezeNoduleNetV2.from_config`` popped
   ``'name'`` into a local dict that was then discarded, so a reload renamed the
   model.
2. ``utils/weight_transfer.py:261-262`` keys its layer map by ``layer.name``.

Measured BEFORE the fix (2026-08-21, GPU:1 / RTX 4070, seeded):

    top-level  : 'my_squeezenet' -> reload 'squeeze_net_v1'
    NESTED     : wrapper sublayer 'backbone' -> reload 'squeeze_net_v1'
                 load_weights_from_checkpoint -> loaded=['head'],
                 missing_in_source=['backbone'], unused_in_source=[...]
                 max|source - target| after transfer = 1.3552e-03
                 (the backbone kept its RANDOM init, and the call returned
                 normally -- the report was the only trace)

Measured AFTER: 'backbone' survives, loaded=['backbone', 'head'], and the
transfer is EXACT (0.0).

``test_nested_backbone_transfer`` is the one that matters: a top-level
``.name`` check alone is GREEN-able by any cosmetic change and does not observe
the weight-transfer consequence at all. The nested arm was verified RED against
the real source by restoring the ``'name'`` pop -- it then reads
``missing_in_source == ['backbone']`` and a non-zero delta.

The third guard covers the sibling gap found while re-deriving this pair:
``from_variant(weights=...)`` swallowed ``weights`` into ``**kwargs`` and
returned a randomly-initialised model with no error at all. The
``NotImplementedError`` lived ONLY in the module-level ``create_squeezenet_*``
factories; it now lives in ``from_variant``, the chokepoint both reach.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1
from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint


def _build_wrapper(seed: int) -> keras.Model:
    """A SqueezeNet backbone nested inside another Functional model."""
    keras.utils.set_random_seed(seed)
    backbone = SqueezeNetV1(
        num_classes=10,
        input_shape=(64, 64, 3),
        include_top=False,
        dropout_rate=0.0,
        name="backbone",
    )
    inp = keras.Input((64, 64, 3))
    h = backbone(inp)
    h = keras.layers.GlobalAveragePooling2D(name="gap")(h)
    return keras.Model(inp, keras.layers.Dense(10, name="head")(h), name="wrapper")


class TestNameSurvivesRoundTrip:
    """``from_config`` must not discard the caller's ``name``."""

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda n: SqueezeNetV1(
                    num_classes=10, input_shape=(64, 64, 3), dropout_rate=0.0, name=n
                ),
                id="v1",
            ),
            pytest.param(
                lambda n: SqueezeNoduleNetV2(
                    num_classes=10, input_shape=(64, 64, 3), dropout_rate=0.0, name=n
                ),
                id="v2",
            ),
        ],
    )
    def test_top_level_name(self, factory, tmp_path):
        model = factory("my_custom_name")
        path = os.path.join(str(tmp_path), "m.keras")
        model.save(path)
        assert keras.models.load_model(path).name == "my_custom_name"

    def test_nested_backbone_transfer(self, tmp_path):
        """The half that has teeth: name loss silently un-transfers a backbone."""
        source = _build_wrapper(seed=0)
        path = os.path.join(str(tmp_path), "w.keras")
        source.save(path)

        target = _build_wrapper(seed=7)
        report = load_weights_from_checkpoint(target=target, ckpt_path=path)

        assert report.missing_in_source == [], report.missing_in_source
        assert report.unused_in_source == [], report.unused_in_source
        assert "backbone" in report.loaded

        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        delta = float(
            np.max(np.abs(source.predict(x, verbose=0) - target.predict(x, verbose=0)))
        )
        assert delta == 0.0, f"transfer left a {delta:.4e} gap (backbone not transferred)"

        # Checked LAST, deliberately: the transfer consequence above is the
        # assertion with teeth, and an early name-equality check would short
        # -circuit the RED proof before it ever ran.
        assert [l.name for l in keras.models.load_model(path).layers] == [
            l.name for l in source.layers
        ]


class TestFromVariantRefusesWeights:
    """``weights=`` must raise, not be swallowed into a random model."""

    @pytest.mark.parametrize(
        "cls,variant",
        [(SqueezeNetV1, "1.1"), (SqueezeNoduleNetV2, "v2")],
        ids=["v1", "v2"],
    )
    def test_from_variant_raises(self, cls, variant):
        with pytest.raises(NotImplementedError, match="No pretrained"):
            cls.from_variant(
                variant, num_classes=10, input_shape=(64, 64, 3), weights="imagenet"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
