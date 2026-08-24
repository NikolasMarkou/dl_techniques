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
``models/vision/squeezenet/caffe_reference_init.py``; this file asserts the outcome.

THE BARE-STRING-INITIALIZER TRAP
--------------------------------
``kernel_initializer="truncated_normal"`` names the right *distribution family*
and silently carries Keras' own scale: ``TruncatedNormal(mean=0.0, stddev=0.05)``
(``keras/src/initializers/random_initializers.py``), 2.5x wider than the
``std=.02`` every ViT-family reference specifies. That is how the ``dino`` rows
below survived review — the string looks correct. Two further facts the
initializer rows depend on, both MEASURED rather than assumed:

* Keras' ``TruncatedNormal`` resamples outside +-2 sigma, so the REALIZED
  standard deviation is ``stddev * 0.87964``, not ``stddev``. A test asserting
  the nominal 0.02 on drawn weights would be red for a correct implementation.
* A seedless ``RandomInitializer`` resolves ``seed=None`` to a concrete seed at
  CONSTRUCTION time, so one instance REPLAYS the identical draw on every call
  (two calls of one instance at the same shape differ by exactly ``0.0``). The
  reference constants are therefore inert config DICTS, never instances: an
  instance used as a default argument is evaluated once at import and would hand
  every model in the process the same weights.

WHY THE GELU ROWS ARE NOT SCALAR ROWS
-------------------------------------
``bert`` and ``gemma`` name references that specify the **tanh approximation** of
GELU, while Keras' ``"gelu"`` string is ``approximate=False`` — the exact/erf
form. There is no scalar to compare: the defect is *which function the graph
runs*, and a string assertion cannot see that (both forms are spelled "gelu"
somewhere). ``test_the_gelu_form_in_use_is_the_tanh_approximation`` therefore
evaluates the callable the BUILT MODEL holds against an independently written-out
tanh formula. That formula is transcribed here a second time on purpose: the
other transcription (``tests/test_layers/test_activations/test_gelu_tanh.py``)
checks a different subject — whether ``gelu_tanh`` implements it — and an oracle
that imports its expectation from the thing it is judging is not an oracle.

Unlike every other row in this file, the two GELU rows are **INFERENCE-CHANGING**:
they alter the forward pass of an already-trained model. Every initializer and
momentum row is TRAINING-ONLY, which
``test_a_loaded_checkpoint_ignores_the_initializer`` demonstrates rather than
asserts.

WHAT THIS GUARD DOES NOT DO
---------------------------
It does not assert that these numerics are *good*, only that they are the
reference's. Packages whose cited reference genuinely specifies no value
(``depth_anything``, ``vae``, ``time_series/mdn`` — all classified
REFERENCE-SILENT in the same audit) are deliberately ABSENT: asserting a number
for them would manufacture a citation that does not exist.

RED-proof: each row was reverted to the pre-fix value in turn and this module
failed naming that exact site (recorded in the plan's decision log, D-482).
The rows added by D-500..D-504 were RED-proven the same way: ``vit``'s stddev
reverted to Keras' 0.05 failed
``test_the_shipped_numeric_is_the_reference_value[vit/model.py ...]`` and the
realized-draw row beside it; ``dino``'s reverted likewise; the ``bert`` and
``gemma`` GELU defaults reverted to ``"gelu"`` failed the form test naming the
package.

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
    - original BERT ``gelu`` (tanh approximation):
      https://github.com/google-research/bert/blob/master/modeling.py
    - HuggingFace ``Gemma3TextConfig.hidden_activation = "gelu_pytorch_tanh"``:
      https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py
    - HuggingFace ``ViTConfig.initializer_range = 0.02``:
      https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision/vit/configuration_vit.py
    - Swin ``_init_weights``: ``trunc_normal_(m.weight, std=.02)``:
      https://github.com/microsoft/Swin-Transformer/blob/main/models/vision/swin_transformer.py
    - DINO ``_init_weights``: ``trunc_normal_(m.weight, std=.02)``:
      https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    - DINOv3, same ViT/Mlp convention:
      https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py
"""

from typing import Any, Callable, List, Tuple

import keras
import numpy as np
import pytest
from keras import ops

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
_GOOGLE_BERT = "https://github.com/google-research/bert/blob/master/modeling.py"
_HF_GEMMA3 = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/gemma3/configuration_gemma3.py"
)
_HF_VIT = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/vision/vit/configuration_vit.py"
)
_MSFT_SWIN = (
    "https://github.com/microsoft/Swin-Transformer/blob/main/"
    "models/vision/swin_transformer.py"
)
_FAIR_DINO = (
    "https://github.com/facebookresearch/dino/blob/main/vision_transformer.py"
)
_HF_BEIT = (
    "https://github.com/huggingface/transformers/blob/main/"
    "src/transformers/models/vision/beit/configuration_beit.py"
)

# Keras' own defaults, asserted as the NEGATIVE control: a row whose expected
# value equals the Keras default cannot distinguish "ported correctly" from
# "never ported at all", and would be a vacuous pin.
_KERAS_BN_MOMENTUM_DEFAULT = 0.99
_KERAS_CONV_INITIALIZER_DEFAULT = "glorot_uniform"
#: What the bare string ``"truncated_normal"`` resolves to. A real rival: the
#: pre-fix ``dino`` sites named the right family and got this scale.
_KERAS_TRUNCATED_NORMAL_STDDEV = 0.05
#: The Keras activation string whose ``approximate=False`` default is the defect
#: the two GELU rows pin.
_KERAS_EXACT_GELU_STRING = "gelu"


# --- readers -----------------------------------------------------------------
# Each returns the SHIPPED value for one cell. Lazy, so one package failing to
# import cannot mask the other four.


def _resnet_default_momentum() -> float:
    from dl_techniques.models.vision.resnet.model import ResNet

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
    from dl_techniques.models.vision.cliffordnet.model import _STEM_BN_MOMENTUM

    return _STEM_BN_MOMENTUM


def _clifford_clip_stem_momentum() -> float:
    from dl_techniques.models.vision_language.clip.clifford_clip import _VISION_STEM_BN_MOMENTUM

    return _VISION_STEM_BN_MOMENTUM


def _squeezenet_v1_stem_initializer() -> str:
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1

    return SqueezeNetV1.STEM_INITIALIZER


def _squeezenet_v1_head_stddev() -> float:
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1

    return SqueezeNetV1.HEAD_INITIALIZER["config"]["stddev"]


def _squeezenet_v2_stem_initializer() -> str:
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2

    return SqueezeNoduleNetV2.STEM_INITIALIZER


def _squeezenet_v2_head_stddev() -> float:
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2

    return SqueezeNoduleNetV2.HEAD_INITIALIZER["config"]["stddev"]


def _vit_initializer_stddev() -> float:
    from dl_techniques.models.vision.vit.model import REFERENCE_KERNEL_INITIALIZER

    return REFERENCE_KERNEL_INITIALIZER["config"]["stddev"]


def _swin_initializer_stddev() -> float:
    from dl_techniques.models.vision.swin_transformer.model import (
        REFERENCE_KERNEL_INITIALIZER,
    )

    return REFERENCE_KERNEL_INITIALIZER["config"]["stddev"]


def _dino_initializer_stddev() -> float:
    from dl_techniques.models.vision.dino.reference_init import DINO_KERNEL_INITIALIZER

    return DINO_KERNEL_INITIALIZER["config"]["stddev"]


def _bert_default_hidden_act() -> str:
    from dl_techniques.models.language.bert.model import BERT

    return BERT.DEFAULT_HIDDEN_ACT


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
    # --- ViT-family kernel initializers. All three references say std=.02.
    (
        "vit/model.py REFERENCE_KERNEL_INITIALIZER stddev (every layer)",
        _vit_initializer_stddev,
        0.02,
        None,  # the pre-fix rival was he_normal, which has no stddev at all
        _HF_VIT,
    ),
    (
        "swin_transformer/model.py REFERENCE_KERNEL_INITIALIZER stddev "
        "(every layer)",
        _swin_initializer_stddev,
        0.02,
        None,  # the pre-fix rival was glorot_uniform, which has no stddev
        _MSFT_SWIN,
    ),
    (
        "dino/reference_init.py DINO_KERNEL_INITIALIZER stddev "
        "(dino_v1 head + classifier, dino_v3 model + classifier)",
        _dino_initializer_stddev,
        0.02,
        _KERAS_TRUNCATED_NORMAL_STDDEV,
        _FAIR_DINO,
    ),
    # --- GELU form. The scalar here is only the NAME; the form actually in use
    # --- is pinned by test_the_gelu_form_in_use_is_the_tanh_approximation.
    (
        "bert/bert.py BERT.DEFAULT_HIDDEN_ACT (every encoder FFN)",
        _bert_default_hidden_act,
        "gelu_tanh",
        _KERAS_EXACT_GELU_STRING,
        _GOOGLE_BERT,
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


# --- non-scalar rows ---------------------------------------------------------
# Two of the five cells this plan shipped cannot be expressed as "reader() ==
# expected": an activation FORM has no scalar, and an initializer's declared
# stddev does not by itself prove the draw reaches the graph. They live here, in
# the same file, rather than in a rival guard module.

#: Keras resamples outside +-2 sigma, so a TruncatedNormal's REALIZED std is this
#: fraction of its nominal ``stddev``. ONE home for the factor. Never assert the
#: nominal 0.02 against drawn weights.
TRUNCATION_FACTOR = 0.87964
REALIZED_TARGET = 0.02 * TRUNCATION_FACTOR   # ~= 0.017593

#: max|exact-erf GELU - tanh GELU|, float64, over x in [-6, 6]; attained at
#: x ~= 2.699, interior to the realistic post-LayerNorm activation range.
EXPECTED_GELU_FORM_SEPARATION = 4.7324e-04


def _reference_tanh_gelu(x):
    """The tanh approximation, transcribed from google-research/bert modeling.py.

    Deliberately written out rather than imported: this is the oracle for
    whether the shipped ports use that form, and an oracle that imports its
    expectation from the code under test proves nothing.
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _grid():
    return np.linspace(-6.0, 6.0, 20001).astype("float32")


def _built_bert_ffn_activations():
    from dl_techniques.models.language.bert.model import BERT

    ids = np.random.RandomState(0).randint(0, 64, size=(2, 8))
    inputs = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }
    keras.utils.set_random_seed(1234)
    model = BERT(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=2,
        intermediate_size=64, max_position_embeddings=16, type_vocab_size=2,
    )
    model(inputs, training=False)
    return [layer.ffn_layer.activation_fn for layer in model.encoder_layers]


def _built_gemma_ffn_activations():
    from dl_techniques.layers.transformers.gemma3_transformer import Gemma3TransformerBlock

    keras.utils.set_random_seed(7)
    block = Gemma3TransformerBlock(
        hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
        ffn_hidden_size=64, max_seq_len=16,
    )
    block(np.random.RandomState(1).randn(2, 8, 32).astype("float32"), training=False)
    return [block.ffn.activation]


@pytest.mark.parametrize(
    "package, reader, url",
    [
        ("bert (every encoder FFN)", _built_bert_ffn_activations, _GOOGLE_BERT),
        ("gemma (the GeGLU gate)", _built_gemma_ffn_activations, _HF_GEMMA3),
    ],
    ids=["bert", "gemma"],
)
def test_the_gelu_form_in_use_is_the_tanh_approximation(package, reader, url):
    """INFERENCE-CHANGING rows: which function the BUILT GRAPH actually calls.

    A string assertion cannot answer this — Keras' ``"gelu"`` and the tanh
    approximation are both spelled "gelu" somewhere. So the callable held by the
    built model is evaluated and compared against ``_reference_tanh_gelu``.
    """
    grid = _grid()
    expected_tanh = _reference_tanh_gelu(grid.astype("float64")).astype("float32")
    exact_erf = np.asarray(keras.activations.gelu(ops.convert_to_tensor(grid)))

    activations = reader()
    assert activations, package
    for i, fn in enumerate(activations):
        got = np.asarray(fn(ops.convert_to_tensor(grid)))
        assert np.abs(got - expected_tanh).max() < 1e-5, (
            f"{package}, site {i}: the activation in use is not the tanh "
            f"approximation the reference specifies. Re-verify at: {url}"
        )
        separation = float(np.abs(got - exact_erf).max())
        assert separation == pytest.approx(
            EXPECTED_GELU_FORM_SEPARATION, rel=0.05
        ), (
            f"{package}, site {i}: sits {separation:.6e} from the exact/erf GELU; "
            f"0.0 means the port reverted to Keras' approximate=False default"
        )


def _built_vit():
    from dl_techniques.models.vision.vit.model import ViT

    keras.utils.set_random_seed(3)
    model = ViT(input_shape=(32, 32, 3), num_classes=4, scale="pico", patch_size=8)
    model(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    return model, "\x00"


def _built_swin():
    from dl_techniques.models.vision.swin_transformer.model import SwinTransformer

    keras.utils.set_random_seed(3)
    model = SwinTransformer(
        num_classes=4, embed_dim=16, depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8],
        window_size=2, patch_size=4, input_shape=(64, 64, 3),
    )
    model(np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32"))
    return model, "\x00"


def _built_dino_v3():
    from dl_techniques.models.vision.dino.dino_v3 import DINOv3

    keras.utils.set_random_seed(3)
    model = DINOv3(
        image_size=(32, 32), patch_size=(8, 8), embed_dim=32, depth=2,
        num_heads=2, num_classes=4,
    )
    model(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    return model, "\x00"


def _built_dino_v1_head():
    from dl_techniques.models.vision.dino.dino_v1 import DINOHead

    keras.utils.set_random_seed(3)
    head = DINOHead(in_dim=64, out_dim=32, hidden_dim=256, bottleneck_dim=64)
    head(np.random.RandomState(1).randn(8, 64).astype("float32"))
    # ``last_layer`` carries a UnitNorm(axis=0) constraint that rescales its
    # columns to unit norm, so its post-build std reports the CONSTRAINT, not
    # the initializer.
    return head, "last_layer"


def _built_beit():
    """The site D-600 fixed. BEiT declares ``initializer_range=0.02`` and hands
    one ``TruncatedNormal`` to every block, but ``TransformerLayer``'s ``'beit'``
    branch never forwarded it, so q/k/v/proj fell back to ``BeitAttention``'s own
    ``glorot_uniform``. MEASURED pre-fix on exactly this model: attention kernels
    at realized std **0.125238** (glorot at dim=64) beside 0.017609 for every
    other kernel in the SAME model. That within-model split is why a whole-model
    probe would have missed it -- ``other`` alone passes."""
    from dl_techniques.models.vision.beit.model import BeitModel

    keras.utils.set_random_seed(3)
    model = BeitModel(
        input_shape=(32, 32, 3), patch_size=8, hidden_size=64,
        num_layers=4, num_heads=4, intermediate_size=128,
    )
    model(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    return model, "\x00"


def _built_beit_attention_only():
    """The same model, restricted to the attention subtree -- the ONLY part the
    dropped knob could reach. Kept separate from ``_built_beit`` on purpose: the
    whole-model row is diluted by 9 correct kernels against 16 defective ones and
    was MEASURED to still pass at rel=0.05 pre-fix if the split were averaged."""
    model, exclude = _built_beit()
    return _AttentionSubtree(model), exclude


class _AttentionSubtree:
    """Adapter exposing only ``.../attention/...`` weights as ``.weights``."""

    def __init__(self, model):
        self.weights = [w for w in model.weights if "/attention/" in w.path]


def _built_dino_v1_classifier():
    from dl_techniques.models.vision.dino.dino_v1 import DINOv1

    keras.utils.set_random_seed(3)
    model = DINOv1(
        image_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=2,
        num_classes=64, include_top=True,
    )
    model(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    return model, "\x00"


@pytest.mark.parametrize(
    "site, builder, only, rel, url",
    [
        ("vit/model.py (every kernel)", _built_vit, None, 0.05, _HF_VIT),
        ("swin_transformer/model.py (every kernel)", _built_swin, None, 0.05, _MSFT_SWIN),
        ("dino/dino_v3.py (every kernel)", _built_dino_v3, None, 0.05, _FAIR_DINO),
        (
            "dino/dino_v1.py DINOHead (excl. UnitNorm last_layer)",
            _built_dino_v1_head, None, 0.05, _FAIR_DINO,
        ),
        (
            "dino/dino_v1.py classifier head",
            _built_dino_v1_classifier, "classifier/kernel", 0.10, _FAIR_DINO,
        ),
        ("beit/model.py (every kernel)", _built_beit, None, 0.05, _HF_BEIT),
        (
            "beit/model.py attention q/k/v/proj ONLY (D-600)",
            _built_beit_attention_only, None, 0.05, _HF_BEIT,
        ),
    ],
)
def test_the_initializer_draw_reaches_the_graph(site, builder, only, rel, url):
    """A declared constant is not a draw. This asserts the REALIZED std of the
    weights a built model actually holds — ``0.02 * 0.87964``, never 0.02."""
    obj, exclude = builder()
    kernels = [
        np.asarray(w)
        for w in obj.weights
        if (w.path == only if only else w.path.endswith("kernel"))
        and np.asarray(w).ndim >= 2
        and exclude not in w.path
    ]
    assert kernels, f"{site}: probe found no kernels -- the model did not build"
    realized = float(np.concatenate([k.ravel() for k in kernels]).std())
    assert realized == pytest.approx(REALIZED_TARGET, rel=rel), (
        f"{site} draws at std {realized:.6f}; the reference's 0.02 realizes as "
        f"{REALIZED_TARGET:.6f} after truncation. Re-verify at: {url}"
    )


#: The distinctive stddev the per-branch reachability probe below asks for. Far
#: from every fallback in the tree (glorot at dim=64 realizes 0.125, and the
#: MEASURED pre-fix fallbacks ranged 0.058 to 0.131), so "the knob arrived" and
#: "the layer used its own default" cannot be confused.
_PROBE_STDDEV = 0.5
#: ``TransformerLayer``'s nine self-attention branches. ``fnet`` is here on
#: purpose: it is parameter-free (a 2-D DFT), declares no initializer, and must
#: be EXCLUDED BY THE REGISTRY rather than by a hand-kept skip list.
_TRANSFORMER_ATTENTION_TYPES = {
    "multi_head": {},
    "window": {"window_size": 4},
    "beit": {"window_size": (4, 4)},
    "group_query": {"n_kv_head": 2},
    "differential": {"lambda_init": 0.8},
    "multi_head_latent": {},
    "anchor": {},
    "lighthouse": {},
    "fnet": {},
}
#: ``LighthouseAttention`` requires seq_len divisible by pooling_factor**(levels-1).
_PROBE_SEQ_LEN = {"lighthouse": 32}


def _attention_types_declaring_a_kernel_initializer():
    from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY

    return sorted(
        t for t in _TRANSFORMER_ATTENTION_TYPES
        if "kernel_initializer" in ATTENTION_REGISTRY[t].get("optional_params", {})
    )


def test_the_registry_split_of_the_attention_branches_is_what_it_was_measured_to_be():
    """Anti-vacuity for the row below. If a refactor stopped declaring
    ``kernel_initializer`` anywhere, the parametrization would silently shrink to
    nothing and the reachability test would pass by having no cases."""
    declaring = _attention_types_declaring_a_kernel_initializer()
    assert len(declaring) == 8, declaring
    assert "fnet" not in declaring
    assert set(declaring) == set(_TRANSFORMER_ATTENTION_TYPES) - {"fnet"}


@pytest.mark.parametrize("attention_type", _attention_types_declaring_a_kernel_initializer())
def test_the_blocks_kernel_initializer_reaches_its_attention_layer(attention_type):
    """D-600. Every attention type that DECLARES ``kernel_initializer`` must
    actually receive ``TransformerLayer``'s.

    Pre-fix MEASURED realized std with the block asking for stddev=0.5
    (target 0.439813): multi_head 0.441372 (the only branch that forwarded),
    window 0.099118, beit 0.124754, group_query 0.131345, differential 0.123712,
    multi_head_latent 0.058167, anchor 0.123712, lighthouse 0.123712. Eight of
    nine branches dropped it; nothing raised, because each layer falls back to
    its own ``glorot_uniform``.

    This is parametrized off the REGISTRY, not off a literal list, so a tenth
    attention type that declares the parameter joins this guard automatically.
    """
    from dl_techniques.layers.transformers.transformer import TransformerLayer

    dim, heads = 64, 4
    seq = _PROBE_SEQ_LEN.get(attention_type, 17)
    x = np.random.RandomState(0).randn(2, seq, dim).astype("float32")

    keras.utils.set_random_seed(7)
    block = TransformerLayer(
        hidden_size=dim, num_heads=heads, intermediate_size=2 * dim,
        attention_type=attention_type,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=_PROBE_STDDEV),
        **_TRANSFORMER_ATTENTION_TYPES[attention_type],
    )
    block(x)

    kernels = [
        np.asarray(w) for w in block.attention.weights
        if np.asarray(w).ndim >= 2 and w.path.endswith("kernel")
    ]
    assert kernels, (
        f"{attention_type}: the probe found no 2-D attention kernels, so it "
        f"cannot see the initializer at all -- fix the probe, not the assertion"
    )
    realized = float(np.concatenate([k.ravel() for k in kernels]).std())
    expected = _PROBE_STDDEV * TRUNCATION_FACTOR
    assert realized == pytest.approx(expected, rel=0.05), (
        f"attention_type='{attention_type}': the block asked for "
        f"TruncatedNormal(stddev={_PROBE_STDDEV}) (realizing {expected:.6f}) but "
        f"its {len(kernels)} attention kernels drew at std {realized:.6f}. "
        f"_get_attention_params dropped kernel_initializer and the layer fell "
        f"back to its own default. See decisions.md D-600."
    )


def test_the_forwarded_initializer_is_cloned_not_shared():
    """The trade this fix must not make. Forwarding ONE seedless instance to N
    blocks would replay one draw -- the D-540/D-560 defect. Callers really do
    hand a single instance to every block (``models/vision/beit/model.py:409``)."""
    from dl_techniques.layers.transformers.transformer import TransformerLayer

    shared = keras.initializers.TruncatedNormal(stddev=_PROBE_STDDEV)
    x = np.random.RandomState(0).randn(2, 17, 64).astype("float32")
    # Seeded ONCE, outside the loop, and that is load-bearing. A seedless clone
    # draws its seed from the global RNG at CONSTRUCTION, so re-seeding before
    # each block hands both clones the same seed and they collide even with the
    # fix in place -- MEASURED while writing this test. Real models build their
    # blocks in one uninterrupted sequence, which is what is reproduced here.
    keras.utils.set_random_seed(7)
    blocks = []
    for i in range(2):
        b = TransformerLayer(
            hidden_size=64, num_heads=4, intermediate_size=128,
            attention_type="beit", window_size=(4, 4),
            kernel_initializer=shared, name=f"probe_{i}",
        )
        b(x)
        blocks.append(b)

    def kernels(b):
        return {
            w.path.split("/attention/")[1]: np.asarray(w)
            for w in b.attention.weights
            if np.asarray(w).ndim >= 2 and w.path.endswith("kernel")
        }

    a, c = kernels(blocks[0]), kernels(blocks[1])
    assert set(a) == set(c) and len(a) == 4, sorted(a)
    for name in a:
        assert np.abs(a[name] - c[name]).max() > 0.0, (
            f"two blocks handed the SAME initializer instance drew identical "
            f"{name}; clone_initializer was dropped from _get_attention_params"
        )
    within = sorted(a)
    for i in range(len(within)):
        for j in range(i + 1, len(within)):
            assert np.abs(a[within[i]] - a[within[j]]).max() > 0.0, (
                f"{within[i]} == {within[j]} within one block"
            )


def test_the_bare_truncated_normal_string_is_keras_own_scale():
    """The trap the dino rows exist to close: right family, wrong scale."""
    bare = keras.initializers.get("truncated_normal")
    assert bare.stddev == _KERAS_TRUNCATED_NORMAL_STDDEV
    assert bare.stddev == pytest.approx(2.5 * 0.02)


def test_the_reference_initializers_are_inert_dicts_not_shared_instances():
    """A seedless instance bakes its seed at construction and REPLAYS its draw,
    so one shared as a default argument would hand every model in the process
    identical weights. Both halves are measured here, not assumed."""
    from dl_techniques.models.vision.dino.reference_init import DINO_KERNEL_INITIALIZER
    from dl_techniques.models.vision.swin_transformer.model import (
        REFERENCE_KERNEL_INITIALIZER as SWIN_INIT,
    )
    from dl_techniques.models.vision.vit.model import (
        REFERENCE_KERNEL_INITIALIZER as VIT_INIT,
    )

    for config in (VIT_INIT, SWIN_INIT, DINO_KERNEL_INITIALIZER):
        assert isinstance(config, dict), config

    one = keras.initializers.TruncatedNormal(stddev=0.02)
    assert np.abs(np.asarray(one((8, 8))) - np.asarray(one((8, 8)))).max() == 0.0, (
        "Keras stopped replaying a seedless instance; the dict form can be revisited"
    )
    a = keras.initializers.get(dict(VIT_INIT))
    b = keras.initializers.get(dict(VIT_INIT))
    assert np.abs(np.asarray(a((8, 8))) - np.asarray(b((8, 8)))).max() > 0.0


def test_a_loaded_checkpoint_ignores_the_initializer(tmp_path):
    """Why every initializer row here is TRAINING-ONLY and the two GELU rows are
    not. An artifact written under the OLD initializer is reproduced exactly by a
    model built with the NEW one, because loading overwrites what was drawn."""
    from dl_techniques.models.vision.vit.model import ViT

    x = np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")
    keras.utils.set_random_seed(11)
    old = ViT(
        input_shape=(32, 32, 3), num_classes=4, scale="pico", patch_size=8,
        kernel_initializer="he_normal",
    )
    old(x)
    reference = [np.asarray(w) for w in old.weights]
    path = tmp_path / "vit_old_init.keras"
    old.save(path)

    keras.utils.set_random_seed(99)
    fresh = ViT(input_shape=(32, 32, 3), num_classes=4, scale="pico", patch_size=8)
    fresh(x)
    # Non-vacuity: the two DISAGREE before the load, so the assertion after it is
    # not passing on an accident of seeding.
    before = max(
        float(np.abs(np.asarray(a) - b).max()) for a, b in zip(fresh.weights, reference)
    )
    assert before > 1e-3

    fresh.load_weights(path)
    after = max(
        float(np.abs(np.asarray(a) - b).max()) for a, b in zip(fresh.weights, reference)
    )
    assert after == 0.0, "a weight load no longer overrides the initializer"
