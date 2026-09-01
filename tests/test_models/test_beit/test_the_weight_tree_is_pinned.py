"""The pinned weight tree of ``models/vision/beit/model.py``.

For every (class, variant) pair in a pinned matrix this file asserts two things
about the BUILT model: the number of weights it carries, and a digest of its
``(weight name, weight shape)`` signature. Nothing else. It is not a behaviour
test and has no opinion about whether BEiT is correct -- only about whether its
weight tree is the same tree it was when the constants below were recorded.

**Provenance of the constants.** They were captured by executing
``src/dl_techniques/models/beit/model.py`` as it existed at::

    d61c1370736efe364114c6d5ebaaf15b675bac48

-- the commit pinned immediately before the ``models/beit/`` -> ``models/vision/beit/``
restructure of 2026-08-24 -- on the CPU ``golden_reference_device`` at seed 1234.
They are therefore an EXTERNAL historical record, not a self-comparison: no part of
this file recomputes them from today's source, so a change to today's source cannot
quietly redefine what "unchanged" means.

**What this file CANNOT catch, stated plainly: any change that moves no weight
shapes and no weight count.** That explicitly includes a change that alters the
forward-pass VALUES -- a reordered sub-layer construction that redraws the same
shapes from the RNG, a constant initializer whose value moved, a norm epsilon that
stopped being passed. All of those leave every assertion here green. This file pins
a tree, not a computation, and it must not be cited as coverage for numerics.

**Why the bitwise-output arm was retired.** This file used to carry a second arm
that hashed the forward output against the same pin, plus a self-guard that
extracted the pinned source into a temporary module and re-executed it. That
arm pinned one FILE but not its dependency graph. BEiT's forward pass runs through
``layers/embedding/class_token.py``, ``layers/embedding/mask_token.py`` and
``layers/transformers/transformer.py``, and since the pin
``src/dl_techniques/layers/`` has moved by 315 files / +53506 / -33083 lines,
those three among them. So every unrelated layer edit re-reddened the arm for a
non-BEiT reason, and each rebaseline would only have reset the clock. The weight
signature is immune to that drift, which is why it survives and the digest does not.

The green sibling ``tests/test_layers/test_attention/
test_window_attention_restructure_is_inert.py`` shows the shape a stable bit-identity
grader has to take: it copies the golden layer's weights into the current layer BY
WEIGHT PATH before comparing, so it pins the COMPUTATION given identical weights
rather than the whole initialization pipeline, and it is scoped to the files its own
restructure edited, so no third party is free to drift underneath it. Rebuilding
BEiT's harness in that shape is a new instrument, not a repair of this one.

**Device.** Everything runs inside the ``golden_reference_device`` fixture (CPU). A
GPU capture cannot answer an inertness question at all in this repo: TF32 and
non-deterministic reductions make the same forward disagree with itself run-to-run
(``test_basic_blocks_work_at_any_stage0_width.py`` records two different digests
from two runs of identical ResNet code).
"""

import hashlib

import keras
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Pinned capture parameters. Changing ANY of these invalidates every constant
# below -- they are inputs to the digests, not preferences.

#: ``git rev-parse HEAD`` immediately before the first restructure edit.
PRE_RESTRUCTURE_COMMIT = "d61c1370736efe364114c6d5ebaaf15b675bac48"

SEED = 1234
INPUT_SHAPE = (32, 32, 3)
PATCH_SIZE = 16  # -> a 2x2 patch grid, 5 tokens with cls
MIM_VOCAB_SIZE = 64
NUM_CLASSES = 10

#: (class name, scale) pairs. ``BeitModel`` at all four variants; the two heads at
#: ``tiny`` only -- a head's own weights do not vary with the trunk scale in any way
#: the trunk rows do not already cover, and both heads compose the SAME trunk.
MATRIX = [
    ("BeitModel", "tiny"),
    ("BeitModel", "small"),
    ("BeitModel", "base"),
    ("BeitModel", "large"),
    ("BeitForMaskedImageModeling", "tiny"),
    ("BeitForImageClassification", "tiny"),
]

#: Captured by EXECUTING ``models/beit/model.py`` as of
#: :data:`PRE_RESTRUCTURE_COMMIT` on the CPU golden reference device at seed 1234.
#: These are historical constants: re-deriving them from today's source would turn
#: this file into a self-comparison that passes against any breakage.
GOLDEN = {
    ("BeitModel", "tiny"): {
        "count": 220,
        "signature": "25444a4c771f7810cf661a83513812ac8f4b2cd6d4bda957cb2074b96485fbca",
    },
    ("BeitModel", "small"): {
        "count": 220,
        "signature": "17071ebc6faad1959ac378db316be2409b7f57ab9bc0e2b2dd1d74da8d264089",
    },
    ("BeitModel", "base"): {
        "count": 220,
        "signature": "0b4b387443f7f5c2f1aae05728bad372d9941056d0556535eb525e01c71bd1c0",
    },
    ("BeitModel", "large"): {
        "count": 436,
        "signature": "2b3dd00e76d58132383f1cfc3c985e0b11bd8baf861567d303abbf8adb99258d",
    },
    ("BeitForMaskedImageModeling", "tiny"): {
        "count": 224,
        "signature": "f267a2824bab5afa14b9f436cd1d6061997099346ef7cb788420870950dbaf41",
    },
    ("BeitForImageClassification", "tiny"): {
        "count": 224,
        "signature": "4a037dff2f32e1284bb97dded44f7aa8aef0e8efb258e6e84bbe692c1e80635f",
    },
}


# ---------------------------------------------------------------------
# The subject


@pytest.fixture(scope="module")
def current_module():
    """The live ``models/vision/beit/model.py`` -- the side under test."""
    from dl_techniques.models.vision.beit import model as current

    return current


# ---------------------------------------------------------------------
# Capture


def _images() -> np.ndarray:
    return np.random.default_rng(0).random((1, *INPUT_SHAPE)).astype("float32")


def _construct(module, class_name: str, variant: str):
    """Build one matrix row from ``module``, consuming RNG in a pinned order.

    The seed is set ONCE, immediately before the trunk, and the head (if any) is then
    constructed from that trunk -- so the two sides consume the global RNG in the same
    order only if the restructure preserved that order. Re-seeding between the trunk
    and the head would hide exactly the reordering this file exists to catch.
    """
    keras.utils.set_random_seed(SEED)
    backbone = module.BeitModel(
        input_shape=INPUT_SHAPE, patch_size=PATCH_SIZE, scale=variant
    )
    if class_name == "BeitModel":
        return backbone
    if class_name == "BeitForMaskedImageModeling":
        return module.BeitForMaskedImageModeling(backbone, vocab_size=MIM_VOCAB_SIZE)
    if class_name == "BeitForImageClassification":
        return module.BeitForImageClassification(backbone, num_classes=NUM_CLASSES)
    raise AssertionError(f"unknown matrix class {class_name!r}")


def _weight_shape_signature(model) -> tuple:
    """``(name, shape)`` per weight, taken AFTER a forward pass.

    Two traps are closed here.

    Before the first call a subclassed model's ``weights`` is EMPTY, so a signature
    taken then is ``()`` for every configuration and cannot fail.

    The leading component of ``weight.path`` is the MODEL INSTANCE name, which Keras
    uniquifies per construction (``beit_backbone``, ``beit_backbone_1``, ...). It is
    not part of the weight tree being compared, and leaving it in makes every entry
    differ between the reference module and the current one -- an arm that fails
    identically against a correct restructure is as useless as one that passes both
    ways.
    """
    assert model.built, "signature taken before the model was built"
    signature = tuple(
        (weight.path.split("/", 1)[-1], tuple(weight.shape)) for weight in model.weights
    )
    assert len(signature) > 0, "empty weight signature -- the arm is vacuous"
    return signature


def _capture(module, class_name: str, variant: str, device: str) -> dict:
    """Build one matrix row and reduce it to ``count`` + ``signature``.

    The forward pass is not compared -- it is what BUILDS the model. A subclassed
    model's ``weights`` is empty until it has been called once, so a signature taken
    before this call is ``()`` for every configuration and cannot fail.
    """
    with keras.device(device):
        model = _construct(module, class_name, variant)
        model(_images(), training=False)
    signature = _weight_shape_signature(model)
    return {
        "count": len(signature),
        "signature": hashlib.sha256(repr(signature).encode()).hexdigest(),
    }


# ---------------------------------------------------------------------
# Arm 1: weight-shape signature


@pytest.mark.parametrize("class_name,variant", MATRIX, ids=lambda v: str(v))
def test_the_weight_shape_signature_is_unchanged(
    class_name, variant, current_module, golden_reference_device
):
    """Arm 1: the weight TREE (names + shapes + count) is bit-for-bit the same.

    Fires on anything that adds, removes, renames or reshapes a weight -- a sub-layer
    created under a different ``name=``, a helper that builds the encoder stack in a
    different order, a width that stopped being read from ``SCALE_CONFIGS``.

    Blind, by construction, to any change that moves no shapes -- including one that
    moves the forward-pass VALUES. Nothing in this repo closes that hole for BEiT;
    see the module docstring for why the arm that used to try was retired.
    """
    golden = GOLDEN[(class_name, variant)]
    current = _capture(current_module, class_name, variant, golden_reference_device)

    assert current["count"] == golden["count"], (
        f"the weight tree of {class_name}/{variant} changed size: "
        f"{current['count']} entries, pre-restructure it had {golden['count']}"
    )
    assert current["signature"] == golden["signature"], (
        f"the weight-shape signature of {class_name}/{variant} no longer matches the "
        f"pre-restructure model at {PRE_RESTRUCTURE_COMMIT}: "
        f"{current['signature']} != {golden['signature']}"
    )
