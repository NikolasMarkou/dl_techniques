"""The instrument that grades the ``models/beit/model.py`` restructure.

This file answers exactly one question, for every (class, variant) pair in a pinned
matrix: **did the restructure move a single bit?** It is not a behaviour test. It has
no opinion about whether BEiT is correct -- only about whether it is the SAME as it
was before the restructure started.

**The reference is EXTERNAL, and that is the whole design.** A restructure test that
compares the current code against itself passes against ANY breakage: this repo has a
measured instance of a bit-identity arm passing 7/7 against a deliberately broken
model because both sides of the comparison went down the same branch
(``test_resnet/test_stem_type.py``, whose method this file ports). So the reference
side is obtained by ``git show``-ing ``models/beit/model.py`` out of a recorded
pre-restructure commit into a scratch file and loading THAT through ``importlib``.
Two distinct module objects, two distinct source texts, one comparison.

Pre-restructure commit, recorded verbatim (``git rev-parse HEAD`` at capture time)::

    d61c1370736efe364114c6d5ebaaf15b675bac48

Reproduce the reference side by hand with::

    git show d61c1370736efe364114c6d5ebaaf15b675bac48:src/dl_techniques/models/beit/model.py > /tmp/beit_pre.py

**Two arms, each with its own injection.** A weight-shape signature is blind to any
change that moves no shapes, and an output digest alone cannot say WHERE a divergence
came from, so both are mandatory and neither is allowed to be the other's proxy:

* **Signature arm** -- :func:`test_the_weight_shape_signature_is_unchanged`.
  RED-proven by injection (i): ``SCALE_CONFIGS['tiny']['num_heads']`` 3 -> 1 in the
  CURRENT ``src/dl_techniques/models/beit/model.py`` (the relative-position-bias table
  is ``(num_relative_distance, num_heads)``, so heads move shapes). Observed, verbatim::

      E  AssertionError: the weight-shape signature of BeitModel/tiny no longer matches the pre-restructure model at d61c1370736efe364114c6d5ebaaf15b675bac48: 816527d14c5c21c26cfa41141361429f680b2ce8b0743e313dd27683bd89292a != 25444a4c771f7810cf661a83513812ac8f4b2cd6d4bda957cb2074b96485fbca

  6 failed / 13 collected: the three ``tiny`` signature rows AND -- because a head
  count also moves values -- the three ``tiny`` digest rows. 7 passed.

* **Digest arm** -- :func:`test_the_forward_output_is_bitwise_unchanged`.
  RED-proven by injection (ii): ``SCALE_CONFIGS['tiny']['layer_scale_init_value']``
  0.1 -> 0.2. LayerScale gammas are constant-initialized, so no shape and no weight
  COUNT moves and no RNG draw is consumed differently -- **the signature arm stays
  fully green** and only the digest arm fires. That is the proof the digest arm is not
  redundant with the signature arm. Observed, verbatim::

      E  AssertionError: the forward output of BeitModel/tiny is not bitwise identical to the pre-restructure model at d61c1370736efe364114c6d5ebaaf15b675bac48: b70428a7e1240bfc06bf84d5e1220ef77202b4deadc3904cfe6cceefa934398c != 464c13da953ccedd1cc0735561973e6763f3b0d3a44a4ea2bf5e0f87c44cc89c
      E    current head: [ 0.000413 -0.073509  0.102383 -0.094769 -0.026035 -0.034262]
      E    pre-restructure head: [ 0.017959 -0.02665   0.083803 -0.055225 -0.022207 -0.019279]

  3 failed / 13 collected: the three ``tiny`` digest rows and nothing else.
  **0 signature tests failed, all 6 stayed green**, 10 passed. Verified by running
  the injection, not by reasoning about it.

**Device.** Everything runs inside the ``golden_reference_device`` fixture (CPU). A GPU
digest cannot answer an inertness question at all in this repo: TF32 and non-
deterministic reductions make the same forward disagree with itself run-to-run
(``test_basic_blocks_work_at_any_stage0_width.py`` records two different digests from
two runs of identical ResNet code).

**Budget.** Step 1's pre-registered rule was to drop ``large`` to ``base`` if the
``large`` CPU trunk build exceeded 240 s. Measured: 3.2 s (32x32 input, 2x2 patch
grid). No reduction was taken; ``large`` stays in the matrix.
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile

import keras
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Pinned capture parameters. Changing ANY of these invalidates every constant
# below -- they are inputs to the digests, not preferences.

#: ``git rev-parse HEAD`` immediately before the first restructure edit.
PRE_RESTRUCTURE_COMMIT = "d61c1370736efe364114c6d5ebaaf15b675bac48"

MODEL_SOURCE_PATH = "src/dl_techniques/models/beit/model.py"

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

#: Captured by EXECUTING the ``git show``n module above on the CPU golden reference
#: device at seed 1234. They are asserted against the live reference capture by
#: :func:`test_the_golden_reference_is_external_and_not_a_self_comparison`, so a wrong
#: or garbage-collected commit hash cannot silently degrade this file into a
#: self-comparison: the reference would stop reproducing these numbers.
GOLDEN = {
    ("BeitModel", "tiny"): {
        "count": 220,
        "signature": "25444a4c771f7810cf661a83513812ac8f4b2cd6d4bda957cb2074b96485fbca",
        "output": "464c13da953ccedd1cc0735561973e6763f3b0d3a44a4ea2bf5e0f87c44cc89c",
        "head": "[ 0.017959 -0.02665   0.083803 -0.055225 -0.022207 -0.019279]",
    },
    ("BeitModel", "small"): {
        "count": 220,
        "signature": "17071ebc6faad1959ac378db316be2409b7f57ab9bc0e2b2dd1d74da8d264089",
        "output": "9219f581158f92676752ded6e5a6841264832d4da0fc53b2f77b860c9af98027",
        "head": "[-0.058658  0.041394  0.091805  0.046795 -0.103322 -0.07007 ]",
    },
    ("BeitModel", "base"): {
        "count": 220,
        "signature": "0b4b387443f7f5c2f1aae05728bad372d9941056d0556535eb525e01c71bd1c0",
        "output": "dbf5d5b0c06534ec24f9dd4eb73da7331596508936d21e5468f5d5de8c44998b",
        "head": "[ 0.148268  0.097023  0.032203 -0.044755 -0.011044 -0.002225]",
    },
    ("BeitModel", "large"): {
        "count": 436,
        "signature": "2b3dd00e76d58132383f1cfc3c985e0b11bd8baf861567d303abbf8adb99258d",
        "output": "bdf9faafd29b468c07e69ba64254ad6a552fcc2bc6850652f20c869b08fe020a",
        "head": "[-0.079896  0.02424   0.038202 -0.001958 -0.068652  0.018452]",
    },
    ("BeitForMaskedImageModeling", "tiny"): {
        "count": 224,
        "signature": "f267a2824bab5afa14b9f436cd1d6061997099346ef7cb788420870950dbaf41",
        "output": "bc82d4295c8deae0bbb137c9a42d79ae7a3a93146bf10c26d8ed099e112be36c",
        "head": "[ 0.198801 -0.183406  0.239409  0.268048 -0.309425 -0.162346]",
    },
    ("BeitForImageClassification", "tiny"): {
        "count": 224,
        "signature": "4a037dff2f32e1284bb97dded44f7aa8aef0e8efb258e6e84bbe692c1e80635f",
        "output": "6a70fdffca68d02619503491d774e5c455a122494a7ad5fcd1411312fde4663f",
        "head": "[-0.500585  0.196606  0.388994  0.444171 -0.047356 -0.501982]",
    },
}


# ---------------------------------------------------------------------
# The external reference


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))


@pytest.fixture(scope="module")
def pre_restructure_module():
    """``models/beit/model.py`` as of :data:`PRE_RESTRUCTURE_COMMIT`, importable.

    The extracted source goes to a process-local temporary directory, never into the
    repository tree: a second importable copy of this module sitting under ``src/``
    would be collected, registered and imported by everything else.

    The Keras global custom-object registry is snapshotted and RESTORED around the
    import. ``@keras.saving.register_keras_serializable()`` writes into a
    process-global dict, so exec'ing a second definition of ``BeitModel`` re-points
    the key ``Custom>BeitModel`` at the pre-restructure class -- after which any OTHER
    test in the same process that deserializes a saved BEiT would silently get the old
    class back. That is a test-suite-wide corruption caused by an instrument, which is
    strictly worse than the bug the instrument is looking for.
    """
    from keras.src.saving import object_registration

    saved_objects = dict(object_registration.GLOBAL_CUSTOM_OBJECTS)
    saved_names = dict(object_registration.GLOBAL_CUSTOM_NAMES)

    tmp_dir = tempfile.mkdtemp(prefix="beit_pre_restructure_")
    pre_path = os.path.join(tmp_dir, "beit_pre_restructure.py")
    source = subprocess.run(
        ["git", "show", f"{PRE_RESTRUCTURE_COMMIT}:{MODEL_SOURCE_PATH}"],
        cwd=_repo_root(),
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert "class BeitModel" in source, (
        f"`git show {PRE_RESTRUCTURE_COMMIT}:{MODEL_SOURCE_PATH}` produced "
        f"{len(source)} characters with no `class BeitModel` in them -- the "
        f"reference side is not the module it claims to be."
    )
    with open(pre_path, "w") as handle:
        handle.write(source)

    spec = importlib.util.spec_from_file_location("beit_pre_restructure", pre_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["beit_pre_restructure"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        # DECISION plan-2026-08-24T074054-247151fd/D-001
        # WHAT NOT TO DO: do not delete this restore, and do not replace it with
        # `keras.saving.custom_object_scope`. `register_keras_serializable` writes
        # into a PROCESS-GLOBAL dict at class-definition time, so a scope manager
        # around the import is a no-op against it, and without the restore the key
        # `Custom>BeitModel` stays pointed at the pre-restructure class for the rest
        # of the pytest process -- every later `.keras` reload in the same process
        # would then deserialize into the OLD class, turning this instrument into a
        # suite-wide corruption. Nothing fails loudly when that happens.
        # See decisions.md D-001.
        sys.modules.pop("beit_pre_restructure", None)
        object_registration.GLOBAL_CUSTOM_OBJECTS.clear()
        object_registration.GLOBAL_CUSTOM_OBJECTS.update(saved_objects)
        object_registration.GLOBAL_CUSTOM_NAMES.clear()
        object_registration.GLOBAL_CUSTOM_NAMES.update(saved_names)


@pytest.fixture(scope="module")
def current_module():
    """The live ``models/beit/model.py`` -- the side under test."""
    from dl_techniques.models.beit import model as current

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
    with keras.device(device):
        model = _construct(module, class_name, variant)
        output = keras.ops.convert_to_numpy(model(_images(), training=False))
    signature = _weight_shape_signature(model)
    return {
        "count": len(signature),
        "signature": hashlib.sha256(repr(signature).encode()).hexdigest(),
        "output": hashlib.sha256(
            np.ascontiguousarray(output, dtype="float32").tobytes()
        ).hexdigest(),
        "head": np.array2string(output.reshape(-1)[:6], precision=6, separator=" "),
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

    Blind, by construction, to any change that moves no shapes. That hole is arm 2's.
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


# ---------------------------------------------------------------------
# Arm 2: bitwise forward output


@pytest.mark.parametrize("class_name,variant", MATRIX, ids=lambda v: str(v))
def test_the_forward_output_is_bitwise_unchanged(
    class_name, variant, current_module, golden_reference_device
):
    """Arm 2: the forward output is bitwise identical at a pinned seed.

    This is the arm that sees a shape-preserving change: a reordered sub-layer
    creation (same shapes, different RNG draws), a constant initializer whose value
    moved, a drop-path ramp that got reversed, a norm epsilon that stopped being
    passed explicitly. None of those move a single entry of arm 1's signature.

    ``training=False`` is passed EXPLICITLY. ``training=None`` is not inference for
    ``StochasticDepth`` -- it short-circuits only on ``training is False`` -- and every
    block here carries a non-zero drop-path rate, which would make this digest random.

    There is no tolerance and no ``allclose``: this plan is a pure reorganisation with
    a CPU-pinned reference, so there is no legitimate nonzero delta to absorb. A
    failure here is a defect to find, never a tolerance to widen.
    """
    golden = GOLDEN[(class_name, variant)]
    current = _capture(current_module, class_name, variant, golden_reference_device)

    assert current["output"] == golden["output"], (
        f"the forward output of {class_name}/{variant} is not bitwise identical to "
        f"the pre-restructure model at {PRE_RESTRUCTURE_COMMIT}: "
        f"{current['output']} != {golden['output']}\n"
        f"current head: {current['head']}\n"
        f"pre-restructure head: {golden['head']}"
    )
    assert current["head"] == golden["head"], (
        f"the plaintext output head of {class_name}/{variant} moved: "
        f"{current['head']} != {golden['head']}"
    )


# ---------------------------------------------------------------------
# The instrument's own guard


def test_the_golden_reference_is_external_and_not_a_self_comparison(
    pre_restructure_module, current_module, golden_reference_device
):
    """The reference really is a DIFFERENT module, and it still reproduces :data:`GOLDEN`.

    Without this test the whole file rests on a commit hash nobody re-checks. If that
    hash were wrong -- pointing at an already-restructured commit -- or if the constants
    had been re-captured from the post-restructure code after a failure, every other
    test here would be a self-comparison that passes against any breakage. Here the
    reference is EXECUTED and its digests are required to reproduce the recorded
    constants, which they can only do if the recorded commit is the one the constants
    came from.

    ``BeitModel`` at ``tiny`` alone is re-executed; the constants for the other five
    rows came from the same capture run, and re-running all six would double this
    file's cost to re-prove one property.
    """
    assert pre_restructure_module is not current_module, (
        "the reference and the subject are the SAME module object -- this file is a "
        "self-comparison and cannot fail"
    )
    assert pre_restructure_module.__file__ != current_module.__file__
    assert not pre_restructure_module.__file__.startswith(_repo_root()), (
        f"the reference module was loaded from inside the repository tree "
        f"({pre_restructure_module.__file__}) -- it must come from `git show`, not "
        f"from a working-tree file that the restructure could edit"
    )

    reference = _capture(
        pre_restructure_module, "BeitModel", "tiny", golden_reference_device
    )
    golden = GOLDEN[("BeitModel", "tiny")]
    assert reference["count"] == golden["count"]
    assert reference["signature"] == golden["signature"], (
        f"the module at {PRE_RESTRUCTURE_COMMIT} no longer reproduces the recorded "
        f"reference signature: {reference['signature']} != {golden['signature']}. "
        f"Either the commit hash is wrong or the constants were re-captured from "
        f"post-restructure code -- in both cases this file is void."
    )
    assert reference["output"] == golden["output"], (
        f"the module at {PRE_RESTRUCTURE_COMMIT} no longer reproduces the recorded "
        f"reference output digest: {reference['output']} != {golden['output']}"
    )
