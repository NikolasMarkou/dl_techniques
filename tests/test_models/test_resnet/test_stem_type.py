"""``ResNet(stem_type=...)`` selects the input stem, and the default is inert.

F-10. ``_build_stem`` was unconditionally the published ImageNet stem -- a 7x7
stride-2 convolution followed by a 3x3 stride-2 max pool -- which downsamples
by 4x before stage 1 runs. MEASURED on this repo: ``resnet18`` on a
``(32, 32, 3)`` input reaches the global average pool at ``(1, 1, 1, 512)``,
i.e. the last two stages stride a feature map that has already collapsed to a
single pixel. The README's own flagship CIFAR-10 example walked the reader into
exactly that. He et al. use a different stem on CIFAR (section 4.2: one 3x3
stride-1 convolution, no pooling) and there was no way to express it here at
all.

``stem_type='cifar'`` adds it. ``stem_type='imagenet'`` is the default and must
be bit-identical to the pre-knob model -- arm (a) MEASURES that against a
golden reference captured from the pre-knob commit, on a weight-SHAPE signature
taken AFTER a forward pass (a pre-forward signature is ``()`` for an un-called
subclassed model and is vacuous by construction) PLUS a bitwise output digest,
because a signature alone is blind to a pooling change.

MEASURED, resnet18 on ``(1, 32, 32, 3)``, ``include_top=False``:

    stem_type='imagenet'  ->  (1, 1, 1, 512)   stem_pool: MaxPooling2D
    stem_type='cifar'     ->  (1, 4, 4, 512)   stem_pool: None

RED PROOFS -- one named injection per assertion, ACTUAL observed text. Six
injections; the FIFTH refuted the arm it was written for and the arm was
rebuilt (see ``test_the_imagenet_stem_reproduces_the_pre_knob_model_exactly``).

Injection 1, ``kernel_size``/``strides`` left unconditional (the 7x7/s2 conv
built for BOTH values, only the pool made conditional) -> **3 failed, 4
passed** -- note THREE, not the two predicted; ``from_variant`` also asserts
the stem geometry:

- ``test_the_cifar_stem_preserves_resolution_into_the_last_stage``::
  ``AssertionError: stem_type='cifar' must preserve resolution into stage 4;
  got (1, 2, 2, 512), expected (1, 4, 4, 512). The paired 'imagenet' reading
  is (1, 1, 1, 512) -- the collapse this knob exists to avoid.``
- ``test_the_cifar_stem_conv_is_3x3_stride_1``::
  ``AssertionError: the cifar stem conv must be 3x3 stride 1, got kernel 7
  stride (2, 2)``
- ``test_from_variant_forwards_the_stem_type``:: ``assert (7, 7) == (3, 3)``

Injection 2, ``self.stem_pool`` built for BOTH stems (only the conv made
conditional) -> **4 failed, 3 passed**:

- ``test_the_cifar_stem_has_no_pooling_layer_at_all``::
  ``AssertionError: stem_type='cifar' must not build a stem_pool at all; found
  <MaxPooling2D name=stem_pool, built=True>``
- ``test_the_cifar_stem_preserves_resolution_into_the_last_stage``:: the same
  ``(1, 2, 2, 512)`` message as injection 1
- ``test_the_stem_type_survives_a_keras_round_trip`` and
  ``test_from_variant_forwards_the_stem_type``, both by
  ``assert <MaxPooling2D name=stem_pool, built=True> is None``

Injection 3, ``stem_type`` dropped from ``get_config()`` -> **1 failed, 6
passed**: ``test_the_stem_type_survives_a_keras_round_trip`` ::
``KeyError: 'stem_type'``. NOTE the predicted text -- an ``AssertionError``
about the reloaded model reporting ``'imagenet'`` -- is WRONG: the
``get_config()`` key assertion sits above the reload and fires first.

Injection 4, the ``stem_type`` validation removed -> **1 failed, 6 passed**:
``test_an_invalid_stem_type_raises_naming_the_value`` ::
``Failed: DID NOT RAISE <class 'ValueError'>``.

Injection 5, the ``'imagenet'`` branch made 5x5 instead of 7x7 -> **7 passed,
0 failed** against the arm's FIRST form, which compared an explicit
``'imagenet'`` against an unspecified ``stem_type``. Both send the same branch
through the same code, so the comparison was satisfied by construction and the
shipped stem could be silently wrong. The arm was rebuilt around a GOLDEN
reference captured from commit 1ffc1ff8e (pre-knob). Re-run against the
rebuilt arm -> **1 failed, 6 passed**::

  AssertionError: the default weight-shape signature no longer matches the
  pre-knob model: 98330cc30798d8a820fc7d94c674228a00dc6f3182d089baaea5dd1a4a74ac1b
  != 67a14ad65a325ecec67b369963d23970bfc4053c0959e636e5f635942fa3e0a6.
  stem_conv is (5, 5) stride (2, 2); pre-knob it was (7, 7) stride (2, 2).

Injection 6, the ``'imagenet'`` pool ``pool_size`` 3 -> 2, which moves NO
weight shape and is invisible to the signature -> **1 failed, 6 passed**, by
the arm's second half::

  AssertionError: the default forward output is no longer bit-identical to the
  pre-knob model: 3618e8f3631e5b73e10eba6ba035785084b01018877da1db3e273b1c359a69c1
  != 9856f1e63e998ac95982bcd47755b970c3b59dd5772c38696129e81410a53e03.
"""

import hashlib
import os

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.resnet.model import ResNet

# ---------------------------------------------------------------------

INPUT_SHAPE = (32, 32, 3)
SEED = 1234

# Golden reference for arm (a), captured by EXECUTING
# `models/vision/resnet/model.py` as of commit 1ffc1ff8e -- the commit before
# `stem_type` existed -- on the golden reference device at seed 1234, with
# `ResNet.from_variant("resnet18", num_classes=10, input_shape=(32, 32, 3))`
# and `np.random.default_rng(0).random((1, 32, 32, 3))` as input.
# Reproduce with:
#   git show 1ffc1ff8e:src/dl_techniques/models/vision/resnet/model.py > /tmp/pre.py
#   # then importlib-load /tmp/pre.py and hash as below.
PRE_KNOB_WEIGHT_COUNT = 102
PRE_KNOB_WEIGHT_SIGNATURE_SHA256 = (
    "67a14ad65a325ecec67b369963d23970bfc4053c0959e636e5f635942fa3e0a6"
)
PRE_KNOB_OUTPUT_SHA256 = (
    "9856f1e63e998ac95982bcd47755b970c3b59dd5772c38696129e81410a53e03"
)
PRE_KNOB_OUTPUT_HEAD = (
    "[[-9.223562 4.7692604 -1.1373646 14.22518 9.336036 "
    "-4.82405 4.7835627 3.0008566 -3.1802382 -0.9986413]]"
)


def _resnet18(stem_type=None, **kwargs) -> ResNet:
    """resnet18 at a pinned seed. ``stem_type=None`` means "do not pass it"."""
    keras.utils.set_random_seed(SEED)
    if stem_type is not None:
        kwargs["stem_type"] = stem_type
    return ResNet.from_variant(
        "resnet18", num_classes=10, input_shape=INPUT_SHAPE, **kwargs
    )


def _weight_shape_signature(model: ResNet) -> tuple:
    """(name, shape) per weight, captured AFTER a forward pass.

    Before the first call a subclassed model's ``weights`` is empty, so a
    signature taken then is ``()`` for every configuration and cannot fail.
    """
    assert model.built, "signature taken before the model was built"
    # The leading component of `weight.path` is the MODEL INSTANCE name, which
    # Keras uniquifies per construction (`res_net`, `res_net_1`, ...). It is not
    # part of the weight tree this arm compares, and leaving it in makes every
    # entry differ for every configuration -- an arm that fails identically
    # against a correct fix is as useless as one that passes both ways.
    signature = tuple(
        (weight.path.split("/", 1)[-1], tuple(weight.shape))
        for weight in model.weights
    )
    assert len(signature) > 0, "empty weight signature -- the arm is vacuous"
    return signature


def _images(batch: int = 1) -> np.ndarray:
    return np.random.default_rng(0).random((batch, *INPUT_SHAPE)).astype("float32")


# --- (a) the default is inert -----------------------------------------------


def test_the_imagenet_stem_reproduces_the_pre_knob_model_exactly(
    golden_reference_device,
):
    """(a) The default stem matches a GOLDEN reference from the pre-knob code.

    THE OBVIOUS FORM OF THIS ARM IS VACUOUS AND WAS MEASURED TO BE. Comparing
    ``stem_type='imagenet'`` against an unspecified ``stem_type`` sends both
    through the SAME branch, so it is satisfied by construction against any
    change to that branch: with the ImageNet stem deliberately changed to a 5x5
    kernel (injection 5), that comparison passed 7/7 while the shipped stem was
    wrong. This is the repo's dominant failure mode, caught here by running the
    injection rather than by reasoning about it.

    The reference is therefore EXTERNAL: both digests below were captured by
    executing ``models/vision/resnet/model.py`` **as of commit 1ffc1ff8e** -- the
    commit before ``stem_type`` existed -- loaded from `git show` through
    ``importlib``, on the golden reference device, at seed 1234. The post-knob
    code reproduces both, and the capture is stable across repeated runs.

    Device: pinned to the golden reference device because
    ``test_basic_blocks_work_at_any_stage0_width.py`` records that the same
    ResNet forward on GPU 1 is NOT reproducible run-to-run at this size (two
    runs of identical code gave two different digests), so a GPU digest cannot
    answer an inertness question at all.
    """
    images = _images()

    with keras.device(golden_reference_device):
        default_model = _resnet18(stem_type=None)
        default_out = keras.ops.convert_to_numpy(
            default_model(images, training=False)
        )
        explicit_model = _resnet18(stem_type="imagenet")
        explicit_out = keras.ops.convert_to_numpy(
            explicit_model(images, training=False)
        )

    default_signature = _weight_shape_signature(default_model)
    assert len(default_signature) == PRE_KNOB_WEIGHT_COUNT, (
        f"the weight tree changed size: {len(default_signature)} entries, "
        f"pre-knob had {PRE_KNOB_WEIGHT_COUNT}"
    )
    signature_digest = hashlib.sha256(
        repr(default_signature).encode()
    ).hexdigest()
    assert signature_digest == PRE_KNOB_WEIGHT_SIGNATURE_SHA256, (
        f"the default weight-shape signature no longer matches the pre-knob "
        f"model: {signature_digest} != {PRE_KNOB_WEIGHT_SIGNATURE_SHA256}. "
        f"stem_conv is {default_model.stem_conv.kernel_size} stride "
        f"{default_model.stem_conv.strides}; pre-knob it was (7, 7) stride "
        f"(2, 2)."
    )

    output_digest = hashlib.sha256(
        np.ascontiguousarray(default_out, dtype="float32").tobytes()
    ).hexdigest()
    assert output_digest == PRE_KNOB_OUTPUT_SHA256, (
        f"the default forward output is no longer bit-identical to the "
        f"pre-knob model: {output_digest} != {PRE_KNOB_OUTPUT_SHA256}. "
        f"Got {np.array2string(default_out, precision=6)}, pre-knob "
        f"{PRE_KNOB_OUTPUT_HEAD}."
    )

    # Secondary: an EXPLICIT 'imagenet' must equal the unspecified default.
    # On its own this proves nothing (see the vacuity note above); it is here
    # so that a future default flip is convicted by name.
    assert _weight_shape_signature(explicit_model) == default_signature
    delta = float(np.max(np.abs(default_out - explicit_out)))
    assert delta == 0.0, (
        f"stem_type='imagenet' is not bit-identical to the default: "
        f"max|delta| = {delta:.6e} (required exactly 0.0)"
    )


# --- (b) the defect this knob fixes, stated as a paired reading -------------


def test_the_cifar_stem_preserves_resolution_into_the_last_stage():
    """(b) ``(1, 4, 4, 512)`` for cifar, PAIRED with imagenet's ``(1,1,1,512)``.

    The pairing is the point: an unpaired ``(1, 4, 4, 512)`` assertion states a
    shape, while the pair states the defect. The ImageNet reading is the
    measured collapse the README's flagship example produced.
    """
    images = _images()

    cifar = _resnet18("cifar", include_top=False)
    cifar_shape = tuple(cifar(images, training=False).shape)

    imagenet = _resnet18("imagenet", include_top=False)
    imagenet_shape = tuple(imagenet(images, training=False).shape)

    assert imagenet_shape == (1, 1, 1, 512), (
        f"the ImageNet stem's collapse on a 32x32 input has moved: "
        f"{imagenet_shape}, expected (1, 1, 1, 512)"
    )
    assert cifar_shape == (1, 4, 4, 512), (
        f"stem_type='cifar' must preserve resolution into stage 4; got "
        f"{cifar_shape}, expected (1, 4, 4, 512). The paired 'imagenet' "
        f"reading is {imagenet_shape} -- the collapse this knob exists to avoid."
    )


def test_the_cifar_stem_conv_is_3x3_stride_1():
    """The stem convolution itself, not only its downstream shape."""
    model = _resnet18("cifar")
    assert (
        model.stem_conv.kernel_size == (3, 3) and model.stem_conv.strides == (1, 1)
    ), (
        f"the cifar stem conv must be 3x3 stride 1, got kernel "
        f"{model.stem_conv.kernel_size[0]} stride {model.stem_conv.strides}"
    )
    # D-041: BOTH stems take their width from filters_per_stage[0].
    assert model.stem_conv.filters == model.filters_per_stage[0]


# --- (c) direct layout assertion, not shape parity --------------------------


def test_the_cifar_stem_has_no_pooling_layer_at_all():
    """(c) No ``stem_pool`` sub-layer exists, and none is tracked.

    A shape check alone is blind to a pool that happens to be a no-op; this
    asserts the layout. Both halves matter: the attribute AND the tracked
    sub-layer tree, since a layer can be built and tracked while the attribute
    is rebound.
    """
    cifar = _resnet18("cifar")
    cifar(_images(), training=False)

    assert cifar.stem_pool is None, (
        f"stem_type='cifar' must not build a stem_pool at all; found "
        f"{cifar.stem_pool}"
    )
    tracked = [layer.name for layer in cifar._flatten_layers(include_self=False)]
    assert "stem_pool" not in tracked, (
        f"a stem_pool layer is still tracked on the cifar model: {tracked[:8]}"
    )

    # Anti-vacuity twin: the imagenet stem DOES have one, so the assertions
    # above are not satisfied by a model that has no sub-layers at all.
    imagenet = _resnet18("imagenet")
    imagenet(_images(), training=False)
    assert imagenet.stem_pool is not None
    assert "stem_pool" in [
        layer.name for layer in imagenet._flatten_layers(include_self=False)
    ]


# --- (d) serialization ------------------------------------------------------


def test_the_stem_type_survives_a_keras_round_trip(tmp_path):
    """(d) ``get_config()`` carries it and the reloaded forward agrees."""
    model = _resnet18("cifar")
    images = _images()
    before = keras.ops.convert_to_numpy(model(images, training=False))

    assert model.get_config()["stem_type"] == "cifar"

    path = os.path.join(str(tmp_path), "resnet_cifar.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    assert loaded.stem_type == "cifar", (
        f"stem_type did not survive get_config(); the reloaded model reports "
        f"{loaded.stem_type!r}"
    )
    assert loaded.stem_pool is None

    after = keras.ops.convert_to_numpy(loaded(images, training=False))
    # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant, test_round_trip.py:41)
    np.testing.assert_allclose(
        before, after, atol=1e-4,
        err_msg="the cifar-stem ResNet differs after a .keras round-trip",
    )


# --- (e) the factory path forwards it ---------------------------------------


def test_from_variant_forwards_the_stem_type():
    """(e) ``from_variant('resnet18', stem_type='cifar')`` reaches the stem."""
    model = ResNet.from_variant(
        "resnet18", num_classes=10, input_shape=INPUT_SHAPE, stem_type="cifar"
    )
    assert model.stem_type == "cifar"
    assert model.stem_pool is None
    assert model.stem_conv.kernel_size == (3, 3)

    default_variant = ResNet.from_variant(
        "resnet18", num_classes=10, input_shape=INPUT_SHAPE
    )
    assert default_variant.stem_type == "imagenet", (
        "from_variant must default to the published ImageNet stem"
    )


# --- (f) validation ---------------------------------------------------------


def test_an_invalid_stem_type_raises_naming_the_value():
    """(f) Axis 2: the ``ValueError`` names the offending value."""
    with pytest.raises(ValueError, match="stem_type"):
        ResNet(input_shape=INPUT_SHAPE, stem_type="bogus")

    with pytest.raises(ValueError, match="bogus"):
        ResNet(input_shape=INPUT_SHAPE, stem_type="bogus")
