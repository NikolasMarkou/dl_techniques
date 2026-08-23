"""Pin the numeric defaults that five ports inherited from Keras instead of their reference.

A port that omits a numeric its reference specifies does not get a neutral value —
it gets *Keras'* value, silently, with no shape symptom, no warning, and a green
test suite. The five cells below were each found that way; this module is the
instrument that keeps them found. Every row carries the URL its expected value was
fetched from, so a reader can re-verify a number without re-running the search.

THE TORCH-VS-KERAS MOMENTUM TRAP, STATED HERE BECAUSE THIS GUARD IS WHERE SOMEONE
WILL COME TO "CORRECT" IT
--------------------------------------------------------------------------------
The two frameworks define BatchNorm momentum as each other's complement:

    Keras   moving = momentum * moving + (1 - momentum) * batch
            https://keras.io/api/layers/normalization_layers/batch_normalization/
    PyTorch moving = (1 - momentum) * moving + momentum * batch
            https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm2d.html

so ``keras_momentum = 1 - torch_momentum``. torchvision's ``nn.BatchNorm2d``
default of ``momentum=0.1`` is therefore ``0.9`` here, and Keras' own default of
``0.99`` corresponds to a torch-side ``0.01`` — a tracking constant ten times
slower than the reference, which is the defect these rows pin. **``0.1`` is the
WRONG value to assert in this file.** If a future reader "fixes" 0.9 to 0.1, this
guard fails and this paragraph is the reason why.

THE CAFFE-XAVIER TRAP
---------------------
Caffe's ``weight_filler { type: "xavier" }`` normalizes by ``fan_in`` by default
(``FillerParameter.variance_norm`` defaults to ``FAN_IN``), giving
``U(+-sqrt(3/fan_in))``. Keras' ``glorot_uniform`` normalizes by
``(fan_in + fan_out)/2``. The Keras name that reproduces Caffe's default xavier is
``lecun_uniform``. The full measured derivation lives at
``models/squeezenet/caffe_reference_init.py``; this file asserts the outcome.

WHAT THIS GUARD DOES NOT DO
---------------------------
It does not assert that these numerics are *good*, only that they are the
reference's. Packages whose cited reference genuinely specifies no value
(``depth_anything``, ``vae``, ``time_series/mdn`` — all classified
REFERENCE-SILENT in the same audit) are deliberately ABSENT: asserting a number
for them would manufacture a citation that does not exist.

RED-proof: each row was reverted to the pre-fix value in turn and this module
failed naming that exact site (recorded in the plan's decision log, D-482).

References:
    - torchvision ``nn.BatchNorm2d``:
      https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm2d.html
    - CliffordNet (Ji 2026): https://arxiv.org/abs/2601.06793
    - SqueezeNet v1.0 prototxt:
      https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/train_val.prototxt
    - SqueezeNet v1.1 prototxt:
      https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt
    - Caffe ``XavierFiller``:
      https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp
"""

from typing import Any, Callable, List, Tuple

import pytest

_TORCHVISION_BN = (
    "https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm2d.html"
)
_CLIFFORDNET = "https://arxiv.org/abs/2601.06793"
_SQUEEZENET_V10 = (
    "https://github.com/forresti/SqueezeNet/blob/master/"
    "SqueezeNet_v1.0/train_val.prototxt"
)
_SQUEEZENET_V11 = (
    "https://github.com/forresti/SqueezeNet/blob/master/"
    "SqueezeNet_v1.1/train_val.prototxt"
)

# Keras' own defaults, asserted as the NEGATIVE control: a row whose expected
# value equals the Keras default cannot distinguish "ported correctly" from
# "never ported at all", and would be a vacuous pin.
_KERAS_BN_MOMENTUM_DEFAULT = 0.99
_KERAS_CONV_INITIALIZER_DEFAULT = "glorot_uniform"


# --- readers -----------------------------------------------------------------
# Each returns the SHIPPED value for one cell. Lazy, so one package failing to
# import cannot mask the other four.


def _resnet_default_momentum() -> float:
    from dl_techniques.models.resnet.model import ResNet

    model = ResNet(
        num_classes=2,
        blocks_per_stage=[1],
        filters_per_stage=[8],
        block_type="bottleneck",
        normalization_type="batch_norm",
        input_shape=(32, 32, 3),
    )
    return model.normalization_kwargs["momentum"]


def _cliffordnet_stem_momentum() -> float:
    from dl_techniques.models.cliffordnet.model import _STEM_BN_MOMENTUM

    return _STEM_BN_MOMENTUM


def _clifford_clip_stem_momentum() -> float:
    from dl_techniques.models.clip.clifford_clip import _VISION_STEM_BN_MOMENTUM

    return _VISION_STEM_BN_MOMENTUM


def _squeezenet_v1_stem_initializer() -> str:
    from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1

    return SqueezeNetV1.STEM_INITIALIZER


def _squeezenet_v1_head_stddev() -> float:
    from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1

    return SqueezeNetV1.HEAD_INITIALIZER["config"]["stddev"]


def _squeezenet_v2_stem_initializer() -> str:
    from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2

    return SqueezeNoduleNetV2.STEM_INITIALIZER


def _squeezenet_v2_head_stddev() -> float:
    from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2

    return SqueezeNoduleNetV2.HEAD_INITIALIZER["config"]["stddev"]


# (site, reader, expected, the-value-that-means-unported, source url)
REFERENCE_PINS: List[Tuple[str, Callable[[], Any], Any, Any, str]] = [
    # --- BatchNorm momentum. Expected 0.9 == torch 0.1; see the module docstring.
    (
        "resnet/model.py ResNet.normalization_kwargs['momentum'] (stem + every block)",
        _resnet_default_momentum,
        0.9,
        _KERAS_BN_MOMENTUM_DEFAULT,
        _TORCHVISION_BN,
    ),
    (
        "cliffordnet/model.py _STEM_BN_MOMENTUM (stem_bn1 x2, stem_norm)",
        _cliffordnet_stem_momentum,
        0.9,
        _KERAS_BN_MOMENTUM_DEFAULT,
        _CLIFFORDNET,
    ),
    (
        "clip/clifford_clip.py _VISION_STEM_BN_MOMENTUM "
        "(vision_stem_bn1 x2, vision_stem_norm)",
        _clifford_clip_stem_momentum,
        0.9,
        _KERAS_BN_MOMENTUM_DEFAULT,
        _CLIFFORDNET,
    ),
    # --- SqueezeNet kernel fillers. 25 xavier convs, conv10 alone gaussian.
    (
        "squeezenet/squeezenet_v1.py SqueezeNetV1.STEM_INITIALIZER (conv1, xavier)",
        _squeezenet_v1_stem_initializer,
        "lecun_uniform",
        _KERAS_CONV_INITIALIZER_DEFAULT,
        _SQUEEZENET_V10,
    ),
    (
        "squeezenet/squeezenet_v1.py SqueezeNetV1.HEAD_INITIALIZER stddev "
        "(conv10, gaussian)",
        _squeezenet_v1_head_stddev,
        0.01,
        None,  # glorot has no stddev, so there is no rival numeric to exclude
        _SQUEEZENET_V10,
    ),
    (
        "squeezenet/squeezenet_v2.py SqueezeNoduleNetV2.STEM_INITIALIZER "
        "(conv1, xavier)",
        _squeezenet_v2_stem_initializer,
        "lecun_uniform",
        _KERAS_CONV_INITIALIZER_DEFAULT,
        _SQUEEZENET_V11,
    ),
    (
        "squeezenet/squeezenet_v2.py SqueezeNoduleNetV2.HEAD_INITIALIZER stddev "
        "(conv10, gaussian)",
        _squeezenet_v2_head_stddev,
        0.01,
        None,
        _SQUEEZENET_V11,
    ),
]

_IDS = [row[0] for row in REFERENCE_PINS]


@pytest.mark.parametrize("site,reader,expected,unported,url", REFERENCE_PINS, ids=_IDS)
def test_the_shipped_numeric_is_the_reference_value(site, reader, expected, unported, url):
    actual = reader()
    assert actual == expected, (
        f"{site} ships {actual!r}, the reference specifies {expected!r}.\n"
        f"Re-verify at: {url}\n"
        f"If this is a momentum row and you are about to write 0.1: read this "
        f"module's docstring. Keras and PyTorch define momentum oppositely."
    )


@pytest.mark.parametrize("site,reader,expected,unported,url", REFERENCE_PINS, ids=_IDS)
def test_the_pin_is_not_vacuous(site, reader, expected, unported, url):
    """The negative control.

    A row whose expected value happens to equal the framework default asserts
    nothing: it passes identically on a package that was never ported. Rows with
    no rival default (the gaussian stddev — ``glorot_uniform`` has no stddev at
    all, which is itself why the pre-fix state could not be expressed as a wrong
    number) declare ``unported=None`` and are skipped rather than faked.
    """
    if unported is None:
        pytest.skip("no framework default to distinguish this row from")
    assert expected != unported, (
        f"{site}: the pinned value {expected!r} IS the framework default, so this "
        f"row cannot fail on an unported package and pins nothing."
    )
