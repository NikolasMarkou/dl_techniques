"""F-73 and F-74: `from_variant` must accept its own documented overrides, and
must not damage the class-level variant table while doing so.

**F-73 re-derived: SIX sites, not the carried four.** `fractalnet` and `cbam`
carry the same narrowed form as `swin_transformer` and `resnet` -- named preset
fields splatted alongside `**kwargs` -- and both raised
`TypeError: got multiple values for keyword argument` on a documented override.
All six now use the `wave_field/model.py` house style verbatim:
``config = cls.MODEL_VARIANTS[variant].copy()`` / ``config.pop("description",
None)`` / ``config.update(kwargs)``.

**F-74 re-derived, and half of it is CLOSED-as-refuted.** The carried claim was
that three sites alias a mutable variant LIST, provable by
``m.depths.append(999)`` reaching ``MODEL_VARIANTS[v]["depths"]``. MEASURED at
all three sites, before any fix: the identity probe reads ``False`` AND the
mutation probe reads no leak -- ``MODEL_VARIANTS["tiny"]["depths"]`` stayed
``[2, 2, 6, 2]``, fractalnet's stayed ``[1, 2, 2]``, cbam's stayed
``[64, 128]``. Keras 3 rewraps ``self.depths = depths`` as a ``TrackedList``
whose contents are copied, so the list-level alias does not survive assignment
and the carried RED proof would have passed before the fix. **Both probes are
non-discriminating here; neither is used as a guard below.**

What IS live is the DICT level: all three read ``config =
cls.MODEL_VARIANTS[variant]`` with no ``.copy()``. That was dormant only because
nothing mutated ``config`` -- and the F-73 repair mutates it, by construction.
The two findings are therefore COUPLED: shipping ``config.update(kwargs)``
without the ``.copy()`` converts a dormant hazard into permanent class-table
poisoning on the first overriding call. The guard below is on the table, not on
the instance attribute.

CPU only.
"""

import copy

import pytest

from dl_techniques.models.cbam.model import CBAMNet
from dl_techniques.models.coshnet.model import CoShNet
from dl_techniques.models.fractalnet.model import FractalNet
from dl_techniques.models.resnet.model import ResNet
from dl_techniques.models.swin_transformer.model import SwinTransformer
from dl_techniques.models.tabm.model import TabMModel

#: ``(label, cls, variant, overrides)`` -- each override names a field the
#: variant preset also supplies, which is exactly the collision F-73 was.
OVERRIDE_CASES = [
    ("coshnet", CoShNet, "base", dict(dropout_rate=0.3)),
    ("tabm", TabMModel, "small", dict(n_num_features=4, cat_cardinalities=[], k=4)),
    ("swin", SwinTransformer, "tiny", dict(num_classes=5, embed_dim=64, num_heads=[2, 4, 8, 16])),
    ("resnet", ResNet, "resnet18", dict(num_classes=5, blocks_per_stage=[1, 1, 1, 1])),
    ("fractalnet", FractalNet, "micro", dict(num_classes=5, depths=[1, 1, 1])),
    ("cbam", CBAMNet, "tiny", dict(num_classes=5, dims=[8, 16])),
]
IDS = [case[0] for case in OVERRIDE_CASES]

#: Snapshotted at IMPORT time, before any `from_variant` in this module runs.
#: MEASURED: capturing it inside the test instead lets the two earlier arms of
#: the same parametrized case poison the table first, so `before` is already the
#: poisoned value and the assertion passes with the `.copy()` deleted -- the
#: first draft of this file did exactly that and read 18/18 green in BOTH arms.
PRISTINE_VARIANTS = {label: copy.deepcopy(cls.MODEL_VARIANTS) for label, cls, _, _ in OVERRIDE_CASES}


@pytest.mark.parametrize("label,cls,variant,overrides", OVERRIDE_CASES, ids=IDS)
class TestFromVariantOverrides:
    def test_a_documented_override_does_not_collide(self, label, cls, variant, overrides):
        """F-73. RED before the fix at all six: `got multiple values`."""
        cls.from_variant(variant, **overrides)

    def test_the_override_actually_reaches_the_constructor(self, label, cls, variant, overrides):
        """Anti-vacuity: `no TypeError` is not the same as `the value was used`.

        Without this, a `from_variant` that silently DROPPED `**kwargs`
        would pass the arm above.
        """
        model = cls.from_variant(variant, **overrides)
        field, expected = next(iter(overrides.items()))
        actual = getattr(model, field)
        assert list(actual) == list(expected) if isinstance(expected, list) else actual == expected

    def test_the_class_variant_table_is_not_mutated(self, label, cls, variant, overrides):
        """F-74, dict level. RED once the `.copy()` is dropped from the fix."""
        before = PRISTINE_VARIANTS[label]
        cls.from_variant(variant, **overrides)
        assert cls.MODEL_VARIANTS == before, (
            f"{label}.from_variant poisoned MODEL_VARIANTS: "
            f"{before[variant]} -> {cls.MODEL_VARIANTS[variant]}"
        )
