"""Every ``models/vision`` classifier returns bare logits.

Census guard for plan-2026-10-14T062616-81938e81 (decisions D-002, D-010). Until
that plan the vision ``ClassificationHead`` ended in a softmax and returned the
same tensor under ``'logits'`` and ``'probabilities'``; eleven of its model
consumers unwrapped ``['probabilities']`` and seven returned the head's dict.
Each consumer now returns ``head_output['logits']`` as one bare tensor.

One row per consumer class, four arms per row:

``test_the_output_is_one_bare_tensor``
    ``call`` returns a single tensor of shape ``(batch, num_classes)``, not a
    dict, list or tuple.
``test_the_output_is_not_normalized``
    On a non-degenerate input (unit-normal scaled by 3, batch 3, 5 classes)
    some row misses 1 by more than ``reassociation_atol([C], 1, 1.0)``, the
    float32 rounding bound inside which a softmax row sums to 1. A model that
    re-applies a softmax, or unwraps ``'probabilities'``, sits inside it.
``test_the_symbolic_shape_is_the_eager_shape``
    ``model(keras.Input(...))`` has the eager shape with an unknown batch.
    Keras builds that symbolic call from ``compute_output_shape`` when the
    class overrides it and by tracing ``call`` otherwise, so the arm reaches a
    hand-written shape method wherever one exists. Where one exists it is also
    compared directly. ResNet, AlexNet and CBAMNet define none; Keras then
    raises ``NotImplementedError`` from ``compute_output_shape``, which reports
    nothing false.
``test_a_stock_from_logits_fit_step_moves_the_classifier``
    ``compile(loss=SparseCategoricalCrossentropy(from_logits=True))`` plus one
    ``fit`` step runs, gives a finite loss and changes the head's classifier
    kernel. ``pyproject.toml`` turns every ``UserWarning`` into an error, so
    Keras' "``from_logits=True`` but the output came from a softmax" warning
    fails this arm too.

Adding a consumer is one entry in :data:`SUBJECTS`. An AST arm
(``test_every_classification_head_owner_has_a_census_row``) fails when a
``models/`` class builds a classification head (``build_model_vision_head``,
``create_vision_head``, ``create_head`` or ``ClassificationHead``) and has no
row, so a new consumer that returns the head's dict cannot slip past the
census; callers of another task type (DarkIR, SCUNet) are exempt by task.

Step 6 added the FFTNet classifier. It returned ``{"logits": ...}`` before that
step. Since step 10.1 (D-229) it is ``FFTNet(include_top=True)`` itself, whose
``head`` is a ``ClassificationHead`` (``pooling_type='cls_token'``,
``block_dims=()``) built through ``build_model_vision_head``, not a
``Sequential`` inside a functional wrapper.

Step 5 added FastVLM (``models/vision_language``, decision D-005), the one
``ClassificationHead`` consumer outside ``models/vision``. Its head is FastViT's
reference classifier: a model-owned ``conv_exp`` (a depthwise 3x3
``MobileOneBlock`` with squeeze-and-excitation, twice as wide as stage 3), then
the shared head as pool, dropout and one linear layer (decision D-019).

Step 4 (Batch C) added CliffordNet, BEiT on both pooling paths, Swin, FractalNet
and SqueezeNet V1/V2, so the census covers all 17 ``models/vision`` classifier
classes (BEiT twice). SqueezeNet is the one family that is NOT a
``ClassificationHead`` consumer any more: its paper head is ``conv10`` (ReLU)
then a global average pool, which the shared head cannot express (decision
D-017), so its rows locate the classifier as the ``conv10`` layer.

Step 3 of the same plan (decisions D-013 to D-015) added two more arms. Each consumer's
head is configured as its paper's or reference implementation's
(``test_the_head_has_the_reference_shape``, one row per class in
:data:`REFERENCE_HEAD_OWNERS`), and the model's own initializer and regularizer
knobs reach the classifier (``test_the_model_knobs_reach_the_classifier``). Two
parameter counts are pinned against external references: ResNet-50 against
torchvision's published count and AlexNet against the paper's layer list
summed by hand.
"""

import ast
from pathlib import Path
from typing import Callable, Dict, Tuple

import keras
import numpy as np
import pytest

from tests.numerics import reassociation_atol

NUM_CLASSES = 5
BATCH = 3


def _resnet(**knobs) -> keras.Model:
    from dl_techniques.models.vision.resnet import ResNet
    return ResNet(num_classes=NUM_CLASSES, blocks_per_stage=[1, 1],
                  filters_per_stage=[8, 16], block_type="basic",
                  input_shape=(32, 48, 3), **knobs)


def _convnext_v1(**knobs) -> keras.Model:
    from dl_techniques.models.vision.convnext import ConvNeXtV1
    return ConvNeXtV1(num_classes=NUM_CLASSES, depths=[1, 1], dims=[8, 16],
                      input_shape=(32, 48, 3), **knobs)


def _convnext_v2(**knobs) -> keras.Model:
    from dl_techniques.models.vision.convnext import ConvNeXtV2
    return ConvNeXtV2(num_classes=NUM_CLASSES, depths=[1, 1], dims=[8, 16],
                      input_shape=(32, 48, 3), **knobs)


def _alexnet(**knobs) -> keras.Model:
    # Square: AlexNet's flatten width is 256 * extent ** 2 for ONE extent.
    # 67 is the smallest input above MIN_SPATIAL_EXTENT (55) that keeps the
    # fc6 kernel small.
    from dl_techniques.models.vision.alexnet import AlexNet
    return AlexNet(num_classes=NUM_CLASSES, input_shape=(67, 67, 3), **knobs)


def _cbam(**knobs) -> keras.Model:
    from dl_techniques.models.vision.cbam import CBAMNet
    return CBAMNet(num_classes=NUM_CLASSES, dims=[8, 16],
                   input_shape=(32, 48, 3), **knobs)


#: class name -> (builder, per-sample input shape).
def _mobilenet_v1(**knobs) -> keras.Model:
    from dl_techniques.models.vision.mobilenet import MobileNetV1
    return MobileNetV1(num_classes=NUM_CLASSES, width_multiplier=0.25,
                       input_shape=(32, 48, 3), **knobs)


def _mobilenet_v2(**knobs) -> keras.Model:
    from dl_techniques.models.vision.mobilenet import MobileNetV2
    return MobileNetV2(num_classes=NUM_CLASSES, width_multiplier=0.35,
                       input_shape=(32, 48, 3), **knobs)


def _mobilenet_v3(**knobs) -> keras.Model:
    from dl_techniques.models.vision.mobilenet import MobileNetV3
    return MobileNetV3(num_classes=NUM_CLASSES, variant="small",
                       width_multiplier=0.5, input_shape=(32, 48, 3), **knobs)


def _mobilenet_v4(**knobs) -> keras.Model:
    from dl_techniques.models.vision.mobilenet import MobileNetV4
    return MobileNetV4.from_variant("small", num_classes=NUM_CLASSES,
                                    input_shape=(32, 48, 3), **knobs)


def _vit(**knobs) -> keras.Model:
    from dl_techniques.models.vision.vit import ViT
    return ViT(input_shape=(32, 48, 3), num_classes=NUM_CLASSES, scale="pico",
               patch_size=16, **knobs)


def _vit_hmlp(**knobs) -> keras.Model:
    from dl_techniques.models.vision.vit_hmlp import ViTHMLP
    return ViTHMLP(input_shape=(32, 48, 3), num_classes=NUM_CLASSES,
                   scale="tiny", patch_size=16, **knobs)


def _cliffordnet(**knobs) -> keras.Model:
    from dl_techniques.models.vision.cliffordnet import CliffordNet
    return CliffordNet(num_classes=NUM_CLASSES, channels=16, depth=2,
                       patch_size=2, shifts=[1, 2], **knobs)


def _hierarchical_cliffordnet(**knobs) -> keras.Model:
    from dl_techniques.models.vision.cliffordnet import HierarchicalCliffordNet
    return HierarchicalCliffordNet(
        num_classes=NUM_CLASSES, stage_depths=[1, 1], stage_dims=[8, 16],
        patch_size=2, shifts=[1, 2], **knobs)


def _beit(**knobs) -> keras.Model:
    # use_mean_pooling=True (the default): the local mean-pooling head.
    from dl_techniques.models.vision.beit import create_beit_classifier
    return create_beit_classifier("tiny", (32, 48, 3), 16,
                                  num_classes=NUM_CLASSES, **knobs)


def _beit_cls(**knobs) -> keras.Model:
    # use_mean_pooling=False: the shared ClassificationHead on the cls state.
    from dl_techniques.models.vision.beit import create_beit_classifier
    return create_beit_classifier("tiny", (32, 48, 3), 16,
                                  num_classes=NUM_CLASSES,
                                  use_mean_pooling=False, **knobs)


def _swin(**knobs) -> keras.Model:
    # 64 x 96 at patch 4, window 2: stage grids 16x24, 8x12, 4x6, 2x3.
    from dl_techniques.models.vision.swin_transformer import SwinTransformer
    return SwinTransformer(num_classes=NUM_CLASSES, embed_dim=16,
                           depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8],
                           window_size=2, patch_size=4, drop_path_rate=0.0,
                           input_shape=(64, 96, 3), **knobs)


def _fractalnet(**knobs) -> keras.Model:
    from dl_techniques.models.vision.fractalnet import FractalNet
    return FractalNet(num_classes=NUM_CLASSES, depths=[1, 2], filters=[8, 16],
                      strides=[2, 2], drop_path_rate=0.0,
                      input_shape=(32, 48, 3), **knobs)


def _squeezenet_v1(**knobs) -> keras.Model:
    # "1.1": its spatial floor is 31, so 48 x 64 clears it on both axes.
    from dl_techniques.models.vision.squeezenet import SqueezeNetV1
    return SqueezeNetV1.from_variant("1.1", num_classes=NUM_CLASSES,
                                     input_shape=(48, 64, 3), **knobs)


def _squeezenet_v2(**knobs) -> keras.Model:
    # Floor 35 on every axis for all four V2 variants.
    from dl_techniques.models.vision.squeezenet import SqueezeNoduleNetV2
    return SqueezeNoduleNetV2.from_variant("v2", num_classes=NUM_CLASSES,
                                           input_shape=(48, 64, 3), **knobs)


def _fftnet(**knobs) -> keras.Model:
    # Square only: the class takes one image_size. 32 px at patch 16 is a
    # 2 x 2 patch grid plus the CLS token.
    from dl_techniques.models.vision.fftnet import FFTNet
    return FFTNet.from_variant("tiny", num_classes=NUM_CLASSES,
                               image_size=32, patch_size=16, **knobs)


def _fastvlm(**knobs) -> keras.Model:
    # 32 x 48 gives a 2 x 3 stage-3 map, so conv_exp's BatchNorms see 6
    # values per channel even at batch 1.
    from dl_techniques.models.vision_language.fastvlm import FastVLM
    return FastVLM(num_classes=NUM_CLASSES, embed_dims=[16, 32, 64],
                   depths=[1, 1, 1], num_heads=[1, 2, 4], mlp_ratio=2.0,
                   drop_path_rate=0.0, input_shape=(32, 48, 3), **knobs)


#: Each builder takes extra constructor keywords (``**knobs``).
SUBJECTS: Dict[str, Tuple[Callable[..., keras.Model], Tuple[int, int, int]]] = {
    "ResNet": (_resnet, (32, 48, 3)),
    "ConvNeXtV1": (_convnext_v1, (32, 48, 3)),
    "ConvNeXtV2": (_convnext_v2, (32, 48, 3)),
    "AlexNet": (_alexnet, (67, 67, 3)),
    "CBAMNet": (_cbam, (32, 48, 3)),
    "MobileNetV1": (_mobilenet_v1, (32, 48, 3)),
    "MobileNetV2": (_mobilenet_v2, (32, 48, 3)),
    "MobileNetV3": (_mobilenet_v3, (32, 48, 3)),
    "MobileNetV4": (_mobilenet_v4, (32, 48, 3)),
    "ViT": (_vit, (32, 48, 3)),
    "ViTHMLP": (_vit_hmlp, (32, 48, 3)),
    "CliffordNet": (_cliffordnet, (32, 48, 3)),
    "HierarchicalCliffordNet": (_hierarchical_cliffordnet, (32, 48, 3)),
    "BeitForImageClassification": (_beit, (32, 48, 3)),
    "BeitForImageClassification_cls": (_beit_cls, (32, 48, 3)),
    "SwinTransformer": (_swin, (64, 96, 3)),
    "FractalNet": (_fractalnet, (32, 48, 3)),
    "SqueezeNetV1": (_squeezenet_v1, (48, 64, 3)),
    "SqueezeNoduleNetV2": (_squeezenet_v2, (48, 64, 3)),
    "FastVLM": (_fastvlm, (32, 48, 3)),
    "FFTNet": (_fftnet, (32, 32, 3)),
}

#: The classes whose classifier is the paper's ``conv10`` layer, not a
#: ``ClassificationHead`` (decision D-017).
CONV10_HEADS = {"SqueezeNetV1", "SqueezeNoduleNetV2"}

def _head_layer(model: keras.Model):
    """The model's classification head, or ``None`` (SqueezeNet's ``conv10``).

    FastVLM holds its head as ``head``; every ``models/vision`` class as
    ``classification_head``.

    :param model: A census subject.
    :type model: keras.Model
    :return: The head layer, or ``None``.
    """
    for attribute in ("classification_head", "head"):
        head = getattr(model, attribute, None)
        if head is not None and hasattr(head, "classifier"):
            return head
    return None


def _classifier(model: keras.Model) -> keras.layers.Layer:
    """The layer that produces the class scores.

    :param model: A census subject, built.
    :type model: keras.Model
    :return: The head's ``classifier`` Dense (``head_classifier`` on BEiT's
        mean-pooling head), or SqueezeNet's ``conv10``.
    :rtype: keras.layers.Layer
    """
    head = _head_layer(model)
    if head is not None:
        return head.classifier
    return model.get_layer("conv10")


def _images(shape: Tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """A non-degenerate batch: unit-normal scaled by 3, fixed seed.

    :param shape: Per-sample input shape.
    :type shape: Tuple[int, int, int]
    :param seed: RNG seed.
    :type seed: int
    :return: ``(BATCH, *shape)`` float32.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    return (3.0 * rng.standard_normal((BATCH, *shape))).astype("float32")


def _build(name: str) -> Tuple[keras.Model, Tuple[int, int, int]]:
    builder, shape = SUBJECTS[name]
    keras.utils.set_random_seed(0)
    return builder(), shape


@pytest.fixture(params=sorted(SUBJECTS))
def subject(request) -> str:
    return request.param


# ---------------------------------------------------------------------


def test_the_output_is_one_bare_tensor(subject: str) -> None:
    model, shape = _build(subject)
    out = model(_images(shape), training=False)
    assert not isinstance(out, (dict, list, tuple)), (
        f"{subject} returned {type(out).__name__}; a classifier returns its "
        f"logits as one tensor"
    )
    assert tuple(out.shape) == (BATCH, NUM_CLASSES), (
        f"{subject} returned shape {tuple(out.shape)}"
    )


def test_the_output_is_not_normalized(subject: str) -> None:
    model, shape = _build(subject)
    out = keras.ops.convert_to_numpy(model(_images(shape), training=False))
    assert np.isfinite(out).all(), f"{subject} produced non-finite logits"
    atol = reassociation_atol([NUM_CLASSES], 1, 1.0)
    rows = out.astype(np.float64).sum(axis=-1)
    assert np.any(np.abs(rows - 1.0) > atol), (
        f"{subject}: every output row sums to 1 within {atol:.2e} ({rows}); "
        f"the model returns probabilities, not logits"
    )


def test_the_symbolic_shape_is_the_eager_shape(subject: str) -> None:
    model, shape = _build(subject)
    eager = tuple(model(_images(shape), training=False).shape)

    symbolic = model(keras.Input(shape))
    assert not isinstance(symbolic, (dict, list, tuple)), (
        f"{subject}: the symbolic call returned {type(symbolic).__name__}"
    )
    assert tuple(symbolic.shape) == (None, *eager[1:]), (
        f"{subject}: symbolic {tuple(symbolic.shape)} vs eager {eager}"
    )

    overrides = type(model).compute_output_shape is not keras.Model.compute_output_shape
    if overrides:
        reported = model.compute_output_shape((BATCH, *shape))
        assert tuple(reported) == eager, (
            f"{subject}.compute_output_shape reports {reported}, call "
            f"returns {eager}"
        )


def test_a_stock_from_logits_fit_step_moves_the_classifier(subject: str) -> None:
    model, shape = _build(subject)
    x = _images(shape, seed=1)
    y = np.arange(BATCH) % NUM_CLASSES
    model(x[:1], training=False)
    kernel = _classifier(model).kernel
    before = keras.ops.convert_to_numpy(kernel).copy()

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    history = model.fit(x, y, batch_size=BATCH, epochs=1, verbose=0)

    loss = history.history["loss"][0]
    assert np.isfinite(loss), f"{subject}: loss {loss}"
    after = keras.ops.convert_to_numpy(kernel)
    assert not np.array_equal(before, after), (
        f"{subject}: one fit step left the classifier kernel unchanged"
    )


def test_a_classifier_bias_shift_moves_the_output_by_exactly_the_shift(subject: str) -> None:
    """N12 (plan-2026-10-14T062616-81938e81 iteration 2, D-122 item 6, D-153).

    Add -50 to the classifier bias. A linear classifier moves every output by
    exactly -50, within float32 rounding at the output's magnitude; a sigmoid,
    softmax or ReLU tail does not (a softmax is invariant to the shift, a
    sigmoid saturates, a ReLU clamps). ``test_the_output_is_not_normalized``
    cannot see a sigmoid or ReLU tail; this arm can.

    SqueezeNet V1/V2 are exempt by name, and the exemption is checked rather
    than assumed: their ``conv10`` carries the paper's ReLU before the pool
    (decision D-017), so the shift drives every output to exactly 0.
    """
    model, shape = _build(subject)
    x = _images(shape)
    out = keras.ops.convert_to_numpy(model(x, training=False)).astype(np.float64)
    bias = _classifier(model).bias
    original = keras.ops.convert_to_numpy(bias)
    shift = -50.0
    bias.assign(original + shift)
    moved = keras.ops.convert_to_numpy(model(x, training=False)).astype(np.float64)
    bias.assign(original)
    if subject in CONV10_HEADS:
        assert (moved == 0.0).all(), (
            f"{subject}: the conv10 ReLU should clamp a -50 bias shift to 0; "
            f"output range [{moved.min()}, {moved.max()}]")
        return
    bound = 8 * float(np.finfo(np.float32).eps) * (np.abs(out).max() + abs(shift))
    error = np.abs(moved - out - shift).max()
    assert error <= bound, (
        f"{subject}: a {shift} classifier bias shift moved the output by "
        f"{(moved - out).min():.4f}..{(moved - out).max():.4f}, not {shift} "
        f"(max error {error:.3e} > {bound:.3e}); something after the classifier "
        f"bends the logits")


# ---------------------------------------------------------------------
# D-013: each head is its reference's, and the model's knobs reach it
# ---------------------------------------------------------------------

#: Which sublayers of ``classification_head`` own weights, per class, at the
#: reference configuration. ResNet: He et al. Sec. 3.4, GAP + one FC (also
#: torchvision ``avgpool`` + ``fc``). CBAMNet: the CBAM paper evaluates on
#: ResNets, so GAP + one FC. ConvNeXt V1/V2: facebookresearch ConvNeXt and
#: ConvNeXt-V2 ``self.head(self.norm(x.mean([-2, -1])))``, GAP + LayerNorm + FC.
#: AlexNet: Krizhevsky et al. Sec. 3.5 and the Caffe ``bvlc_alexnet``,
#: fc6 4096 + fc7 4096 (ReLU, dropout, no normalization) + fc8.
REFERENCE_HEAD_OWNERS: Dict[str, set] = {
    "ResNet": {"classifier"},
    "CBAMNet": {"classifier"},
    "ConvNeXtV1": {"pooled_norm", "classifier"},
    "ConvNeXtV2": {"pooled_norm", "classifier"},
    "AlexNet": {"dense_block", "dense_block_1", "classifier"},
    # MobileNetV1: Howard et al. 2017 Table 1 (Avg Pool, FC, Softmax) as Keras
    # Applications builds it (dropout 1e-3, then the classifier).
    "MobileNetV1": {"classifier"},
    # MobileNetV2: Sandler et al. 2018 Table 2 (avgpool, conv2d 1x1 k), as
    # torchvision builds it (Dropout(0.2), Linear).
    "MobileNetV2": {"classifier"},
    # MobileNetV3: Howard et al. 2019 Tables 1/2 (pool, conv2d 1x1 NBN
    # 1280/1024, hard-swish, conv2d 1x1 NBN k), Keras Applications layout.
    "MobileNetV3": {"dense_block", "classifier"},
    # MobileNetV4: Qin et al. 2024 / timm head_norm=True (pool, 1x1 conv 1280,
    # BatchNorm, ReLU, classifier).
    "MobileNetV4": {"dense_block", "classifier"},
    # ViT and ViT-HMLP: Dosovitskiy et al. 2020 Sec. 3.1, "a single linear
    # layer at fine-tuning time", on the final norm's CLS token.
    "ViT": {"classifier"},
    "ViTHMLP": {"classifier"},
    # CliffordNet: ParaMind2025/CAN model.py, x.mean(dim=[-2, -1]) ->
    # nn.LayerNorm(embed_dim) -> nn.Linear(embed_dim, num_classes).
    "CliffordNet": {"pooled_norm", "classifier"},
    # HierarchicalCliffordNet: the same head on the last stage's map
    # (ParaMind2025/CAN model_hier.py: mean -> nn.LayerNorm -> nn.Linear).
    "HierarchicalCliffordNet": {"pooled_norm", "classifier"},
    # BEiT: microsoft/unilm beit/modeling_finetune.py. Mean path:
    # fc_norm(t[:, 1:].mean(1)) -> head; here head_norm and head_classifier.
    "BeitForImageClassification": {"head_norm", "head_classifier"},
    # Cls path: x[:, 0] after the trunk's own norm -> head. No head norm.
    "BeitForImageClassification_cls": {"classifier"},
    # Swin: microsoft/Swin-Transformer models/swin_transformer.py, norm (per
    # token, the model's own head_norm) -> avgpool -> head. The head itself
    # is pool + one Linear.
    "SwinTransformer": {"classifier"},
    # FractalNet: Larsson et al. 2017 Sec. 4, "A softmax prediction layer
    # attaches at the end of the network": pool, one linear layer.
    "FractalNet": {"classifier"},
    # FastVLM: apple/ml-fastvit and apple/ml-fastvlm (mci.py) FastViT
    # classifier, conv_exp -> gap -> head (nn.Linear); timm final_conv ->
    # ClassifierHead (pool, drop_rate, fc). conv_exp is the model's own layer.
    "FastVLM": {"classifier"},
    # FFTNet: the ViT fine-tuning head on the CLS token (an optional dropout,
    # one linear Dense); the paper (arXiv 2502.18394) names no other.
    "FFTNet": {"classifier"},
}


def _head_owners(head: keras.layers.Layer) -> set:
    """Names of the sublayers that own the head's weights.

    Paths read ``.../<head name>/<owner>/<variable>``.

    :param head: The model's head layer.
    :type head: keras.layers.Layer
    :return: The owner names.
    :rtype: set
    """
    return {
        parts[parts.index(head.name) + 1]
        for parts in (w.path.split("/") for w in head.weights)
    }


def _assert_the_conv10_head(subject: str, model: keras.Model) -> None:
    """SqueezeNet's paper head: conv10 (1x1, num_classes, ReLU), then GAP.

    Iandola et al. 2016 / the official prototxts (conv10, relu_conv10, global
    pool10) and torchvision ``squeezenet1_0``/``1_1`` (``final_conv``, ReLU,
    ``AdaptiveAvgPool2d(1)``). Nothing follows the pool.
    """
    assert getattr(model, "classification_head", None) is None, subject
    conv10 = model.get_layer("conv10")
    assert isinstance(conv10, keras.layers.Conv2D), subject
    assert conv10.kernel_size == (1, 1) and conv10.filters == NUM_CLASSES
    assert conv10.activation is keras.activations.relu, subject
    assert isinstance(model.layers[-1], keras.layers.GlobalAveragePooling2D), (
        f"{subject}: last layer {model.layers[-1].name}"
    )
    assert model.layers[-2] is conv10, subject


def test_the_head_has_the_reference_shape(subject: str) -> None:
    model, shape = _build(subject)
    model(_images(shape), training=False)
    if subject in CONV10_HEADS:
        _assert_the_conv10_head(subject, model)
        return
    head = _head_layer(model)
    owners = _head_owners(head)
    assert owners == REFERENCE_HEAD_OWNERS[subject], (
        f"{subject} head weight owners {sorted(owners)}, the reference head "
        f"has {sorted(REFERENCE_HEAD_OWNERS[subject])}"
    )
    if subject == "BeitForImageClassification":
        # The local mean-pooling head: no hidden block by construction.
        assert head.norm.epsilon == model.backbone.layer_norm_eps
        return
    # No normalization inside any hidden block of these reference heads,
    # except MobileNetV4's BatchNorm at the family epsilon.
    if subject == "MobileNetV4":
        (block,) = head.dense_blocks
        assert block.units == 1280
        assert isinstance(block.norm, keras.layers.BatchNormalization)
        assert block.norm.epsilon == pytest.approx(1e-3)
    else:
        assert all(block.norm is None for block in head.dense_blocks), subject
    if subject == "MobileNetV3":
        (block,) = head.dense_blocks
        # Small's last point is 1024; width_multiplier <= 1 does not narrow it.
        assert block.units == 1024
        assert block.activation_type == "hard_swish"
    if subject.startswith("ConvNeXt"):
        assert isinstance(head.pooled_norm, keras.layers.LayerNormalization)
        assert head.pooled_norm.epsilon == pytest.approx(1e-6)
    if subject == "AlexNet":
        assert [b.units for b in head.dense_blocks] == [4096, 4096]
        assert [b.dropout_rate for b in head.dense_blocks] == [0.5, 0.5]
    if subject in ("CliffordNet", "HierarchicalCliffordNet"):
        assert isinstance(head.pooled_norm, keras.layers.LayerNormalization)
        # torch nn.LayerNorm's default eps, not the 1e-6 of the blocks.
        assert head.pooled_norm.epsilon == pytest.approx(1e-5)
    if subject == "SwinTransformer":
        # The official norm is per token, BEFORE the pool, so the model owns
        # it; the shared head's pooled_norm would norm the pooled vector.
        assert not hasattr(head, "pooled_norm")
        assert isinstance(model.head_norm, keras.layers.LayerNormalization)
        assert model.head_norm.epsilon == pytest.approx(1e-5)
        # Official PatchMerging: nn.Linear(4 * dim, 2 * dim, bias=False).
        assert [m.reduction.use_bias for m in model.patch_merge_layers] == [
            False, False, False]
    if subject == "FractalNet":
        assert isinstance(head.pooling, keras.layers.GlobalAveragePooling2D)
    if subject == "FastVLM":
        assert head.normalization_type is None
        assert isinstance(head.pooling, keras.layers.GlobalAveragePooling2D)
        # conv_exp: in_channels=64, out_channels=int(64 * cls_ratio 2.0),
        # kernel_size=3, stride=1, groups=in_channels, use_se=True,
        # num_conv_branches=1 (the reference constructor call).
        conv_exp = model.conv_exp
        assert (conv_exp.out_channels, conv_exp.kernel_size, conv_exp.stride,
                conv_exp.groups, conv_exp.use_se,
                conv_exp.num_conv_branches) == (128, 3, 1, 64, True, 1)
        assert model.layers[-2] is conv_exp and model.layers[-1] is head


_L2 = 3e-4

#: Per class: the knobs the model advertises, and what each must set on the
#: head's classifier. ResNet and ConvNeXt take no kernel_initializer knob; the
#: ConvNeXt head draws from the class's ``HEAD_INITIALIZER``.
KNOBS: Dict[str, Dict[str, object]] = {
    "ResNet": {"kernel_regularizer": keras.regularizers.L2(_L2)},
    "ConvNeXtV1": {"kernel_regularizer": keras.regularizers.L2(_L2)},
    "ConvNeXtV2": {"kernel_regularizer": keras.regularizers.L2(_L2)},
    "CBAMNet": {"kernel_regularizer": keras.regularizers.L2(_L2),
                "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "AlexNet": {"kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
                "bias_initializer": keras.initializers.Constant(0.5)},
    # MobileNet: ``weight_decay`` is the family's L2 knob; ``dropout_rate`` is
    # the dropout before the classifier (V1/V2) or after the hidden block
    # (V3/V4).
    "MobileNetV1": {"weight_decay": _L2, "dropout_rate": 0.3,
                    "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "MobileNetV2": {"weight_decay": _L2, "dropout_rate": 0.3,
                    "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "MobileNetV3": {"weight_decay": _L2, "dropout_rate": 0.3,
                    "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "MobileNetV4": {"weight_decay": _L2, "dropout_rate": 0.3,
                    "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "ViT": {"kernel_regularizer": keras.regularizers.L2(_L2),
            "bias_regularizer": keras.regularizers.L2(_L2), "dropout_rate": 0.3,
            "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
            "bias_initializer": keras.initializers.Constant(0.5)},
    "ViTHMLP": {"kernel_regularizer": keras.regularizers.L2(_L2),
                "bias_regularizer": keras.regularizers.L2(_L2), "dropout_rate": 0.3,
                "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
                "bias_initializer": keras.initializers.Constant(0.5)},
    "CliffordNet": {"kernel_regularizer": keras.regularizers.L2(_L2),
                    "bias_regularizer": keras.regularizers.L2(_L2),
                    "dropout_rate": 0.3,
                    "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
                    "bias_initializer": keras.initializers.Constant(0.5)},
    "HierarchicalCliffordNet": {
        "kernel_regularizer": keras.regularizers.L2(_L2),
        "bias_regularizer": keras.regularizers.L2(_L2),
        "dropout_rate": 0.3,
        "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
        "bias_initializer": keras.initializers.Constant(0.5)},
    # BEiT takes no initializer or regularizer knob of its own: the head draws
    # from TruncatedNormal(backbone.initializer_range), so that is the knob.
    "BeitForImageClassification": {"dropout_rate": 0.3, "initializer_range": 0.05},
    "BeitForImageClassification_cls": {"dropout_rate": 0.3,
                                       "initializer_range": 0.05},
    "SwinTransformer": {"kernel_regularizer": keras.regularizers.L2(_L2),
                        "bias_regularizer": keras.regularizers.L2(_L2),
                        "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
                        "bias_initializer": keras.initializers.Constant(0.5)},
    # FractalNet: classifier_dropout_rate is its pre-classifier dropout and
    # global_pool picks the head's pooling (dead before step 4).
    "FractalNet": {"kernel_regularizer": keras.regularizers.L2(_L2),
                   "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123),
                   "classifier_dropout_rate": 0.3, "global_pool": "max"},
    # SqueezeNet: the regularizer reaches conv10; the kernel_initializer does
    # NOT, by design -- conv10 keeps the prototxt's gaussian(0.01) filler.
    "SqueezeNetV1": {"kernel_regularizer": keras.regularizers.L2(_L2),
                     "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    "SqueezeNoduleNetV2": {"kernel_regularizer": keras.regularizers.L2(_L2),
                           "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    # FastVLM has no regularizer knob; its kernel_initializer governs every
    # conv and Dense kernel, conv_exp and the classifier included.
    "FastVLM": {"dropout_rate": 0.3,
                "kernel_initializer": keras.initializers.RandomNormal(stddev=0.123)},
    # FFTNet's constructor has no initializer or regularizer knob; its head
    # draws from a fixed TruncatedNormal(0.02), which the arm pins, and head
    # keywords reach it through head_config_overrides
    # (test_fftnet/test_every_head_override_reaches_the_head.py).
    "FFTNet": {},
}

#: Where each class's ``dropout_rate`` must land in its reference head.
CLASSIFIER_DROPOUT = {"MobileNetV1", "MobileNetV2", "ViT", "ViTHMLP",
                      "CliffordNet", "HierarchicalCliffordNet",
                      "BeitForImageClassification_cls", "FastVLM"}
BLOCK_DROPOUT = {"MobileNetV3", "MobileNetV4"}


def test_the_model_knobs_reach_the_classifier(subject: str) -> None:
    builder, shape = SUBJECTS[subject]
    keras.utils.set_random_seed(0)
    model = builder(**KNOBS[subject])
    model(_images(shape), training=False)
    head = _head_layer(model)
    classifier = _classifier(model)
    knobs = KNOBS[subject]
    if "kernel_regularizer" in knobs or "weight_decay" in knobs:
        assert isinstance(classifier.kernel_regularizer, keras.regularizers.L2), (
            f"{subject}: the model's kernel L2 did not reach the classifier"
        )
        assert classifier.kernel_regularizer.l2 == pytest.approx(_L2)
    if "bias_regularizer" in knobs:
        assert isinstance(classifier.bias_regularizer, keras.regularizers.L2), (
            f"{subject}: bias_regularizer did not reach the classifier"
        )
        assert classifier.bias_regularizer.l2 == pytest.approx(_L2)
    if subject in CLASSIFIER_DROPOUT:
        assert head.classifier_dropout.rate == pytest.approx(0.3), subject
    if subject in BLOCK_DROPOUT:
        assert [b.dropout_rate for b in head.dense_blocks] == [0.3], subject
    if subject == "BeitForImageClassification":
        assert head.dropout.rate == pytest.approx(0.3), subject
    if subject.startswith("BeitForImageClassification"):
        config = keras.initializers.serialize(classifier.kernel_initializer)
        assert config["class_name"] == "TruncatedNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.05), (
            f"{subject}: initializer_range did not reach the classifier"
        )
    if subject == "FractalNet":
        assert head.classifier_dropout.rate == pytest.approx(0.3)
        assert isinstance(head.pooling, keras.layers.GlobalMaxPooling2D), (
            "FractalNet: global_pool='max' did not reach the head"
        )
    if subject in CONV10_HEADS:
        config = keras.initializers.serialize(classifier.kernel_initializer)
        assert config["class_name"] == "RandomNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.01), (
            f"{subject}: conv10 must keep the prototxt's gaussian(0.01) filler"
        )
    elif "kernel_initializer" in knobs:
        config = keras.initializers.serialize(classifier.kernel_initializer)
        assert config["class_name"] == "RandomNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.123)
    if "bias_initializer" in knobs:
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(classifier.bias),
            np.full(classifier.bias.shape, 0.5, "float32"))
    if subject.startswith("ConvNeXt"):
        config = keras.initializers.serialize(classifier.kernel_initializer)
        assert config["class_name"] == "TruncatedNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.02), (
            f"{subject}: the reference head draws at trunc_normal_(std=.02)"
        )
    if subject == "FFTNet":
        config = keras.initializers.serialize(classifier.kernel_initializer)
        assert config["class_name"] == "TruncatedNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.02)
    if subject == "FastVLM":
        config = keras.initializers.serialize(model.conv_exp.kernel_initializer)
        assert config["class_name"] == "RandomNormal", (subject, config)
        assert config["config"]["stddev"] == pytest.approx(0.123)


def _trainable_count(model: keras.Model) -> int:
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


#: torchvision ``models.resnet50()``: ``sum(p.numel() for p in m.parameters())``
#: is 25,557,032, the figure torchvision's model table publishes. PyTorch
#: counts BatchNorm gamma/beta and not the running statistics, which is
#: exactly Keras' trainable set here.
TORCHVISION_RESNET50_PARAMETERS = 25_557_032


def _alexnet_paper_parameter_count() -> int:
    """AlexNet's learnable parameters from Krizhevsky et al. 2012, Sec. 3.5.

    conv1 96 kernels 11x11x3; conv2 256 of 5x5x48; conv3 384 of 3x3x256 (the
    one layer "connected to all kernel maps in the second layer"); conv4 384
    of 3x3x192; conv5 256 of 3x3x192; fc 9216 -> 4096 -> 4096 -> 1000. Every
    layer has a bias. Written out here, not imported, so the oracle does not
    come from the code under test.

    :return: 60,965,224.
    :rtype: int
    """
    convs = [(11, 11, 3, 96), (5, 5, 48, 256), (3, 3, 256, 384),
             (3, 3, 192, 384), (3, 3, 192, 256)]
    fcs = [(6 * 6 * 256, 4096), (4096, 4096), (4096, 1000)]
    total = sum(kh * kw * cin * cout + cout for kh, kw, cin, cout in convs)
    total += sum(fin * fout + fout for fin, fout in fcs)
    return total


def test_resnet50_has_the_torchvision_parameter_count() -> None:
    from dl_techniques.models.vision.resnet import ResNet
    model = ResNet.from_variant("resnet50", num_classes=1000,
                                input_shape=(224, 224, 3))
    model.build((None, 224, 224, 3))
    assert _trainable_count(model) == TORCHVISION_RESNET50_PARAMETERS


#: timm ``vit_base_patch16_224`` (and the ViT README's measured table): ViT-B/16
#: at 224 px with a single ``Linear(768, 1000)`` head on the CLS token.
TIMM_VIT_BASE_PATCH16_224_PARAMETERS = 86_567_656


def test_vit_base_has_the_timm_parameter_count() -> None:
    from dl_techniques.models.vision.vit import ViT
    model = ViT(input_shape=(224, 224, 3), num_classes=1000, scale="base",
                patch_size=16)
    model.build((None, 224, 224, 3))
    assert model.count_params() == TIMM_VIT_BASE_PATCH16_224_PARAMETERS


#: torchvision ``squeezenet1_0()`` / ``squeezenet1_1()`` at 1000 classes:
#: ``sum(p.numel() for p in m.parameters())``, the figures torchvision's model
#: table publishes (1.2M; 1,248,424 and 1,235,496 exactly). The two share the
#: Fire widths and the conv10 head; they differ in the stem and pool schedule.
TORCHVISION_SQUEEZENET_PARAMETERS = {"1.0": 1_248_424, "1.1": 1_235_496}


@pytest.mark.parametrize("variant", sorted(TORCHVISION_SQUEEZENET_PARAMETERS))
def test_squeezenet_v1_has_the_torchvision_parameter_count(variant: str) -> None:
    from dl_techniques.models.vision.squeezenet import SqueezeNetV1
    model = SqueezeNetV1.from_variant(variant, num_classes=1000,
                                      input_shape=(224, 224, 3))
    assert _trainable_count(model) == TORCHVISION_SQUEEZENET_PARAMETERS[variant]


def _swin_official_parameter_count(embed_dim: int, depths, num_heads,
                                   num_classes: int = 1000, window: int = 7,
                                   patch: int = 4, in_chans: int = 3) -> int:
    """Swin's parameters, summed by hand from the official module list.

    ``microsoft/Swin-Transformer`` ``models/swin_transformer.py``: a patch-embed
    conv with bias and its LayerNorm; per block ``norm1``, the
    ``(2 * window - 1) ** 2 x heads`` relative-position table, ``qkv`` and
    ``proj`` with bias, ``norm2``, ``fc1``/``fc2`` with bias at ratio 4; per
    merge a LayerNorm over ``4C`` and ``nn.Linear(4C, 2C, bias=False)``; the
    final LayerNorm and ``nn.Linear(num_features, num_classes)``. Written out
    here, not imported, so the oracle does not come from the code under test.
    """
    total = in_chans * patch * patch * embed_dim + embed_dim + 2 * embed_dim
    for stage, (depth, heads) in enumerate(zip(depths, num_heads)):
        c = embed_dim * 2 ** stage
        block = (2 * c + (2 * window - 1) ** 2 * heads + 3 * c * c + 3 * c
                 + c * c + c + 2 * c + c * 4 * c + 4 * c + 4 * c * c + c)
        total += depth * block
        if stage < len(depths) - 1:
            total += 2 * 4 * c + 4 * c * 2 * c
    features = embed_dim * 2 ** (len(depths) - 1)
    return total + 2 * features + features * num_classes + num_classes


def test_swin_tiny_has_the_official_parameter_count() -> None:
    from dl_techniques.models.vision.swin_transformer import SwinTransformer
    expected = _swin_official_parameter_count(96, [2, 2, 6, 2], [3, 6, 12, 24])
    # timm's swin_tiny_patch4_window7_224 publishes 28.29M.
    assert expected == 28_288_354, "the hand sum itself changed"
    model = SwinTransformer.from_variant("tiny", num_classes=1000,
                                         input_shape=(224, 224, 3))
    assert _trainable_count(model) == expected


#: Ji 2026 (arXiv:2601.06793) Table 1, CIFAR-100: CliffordNet-Nano 1.4M,
#: CliffordNet-Lite 2.6M. The paper prints one decimal, so that is the pin.
CLIFFORDNET_PAPER_PARAMETERS_M = {"nano": 1.4, "lite": 2.6}


@pytest.mark.parametrize("variant", sorted(CLIFFORDNET_PAPER_PARAMETERS_M))
def test_cliffordnet_has_the_paper_parameter_count(variant: str) -> None:
    from dl_techniques.models.vision.cliffordnet import CliffordNet
    model = CliffordNet.from_variant(variant, num_classes=100)
    model.build((None, 32, 32, 3))
    assert round(model.count_params() / 1e6, 1) == (
        CLIFFORDNET_PAPER_PARAMETERS_M[variant])


def test_alexnet_has_the_paper_parameter_count() -> None:
    from dl_techniques.models.vision.alexnet import create_alexnet
    expected = _alexnet_paper_parameter_count()
    assert expected == 60_965_224, "the hand sum itself changed"
    model = create_alexnet()
    model.build((None, 227, 227, 3))
    assert _trainable_count(model) == expected


# ---------------------------------------------------------------------
# return_dict is gone (plan-2026-10-14T062616-81938e81 step 3, D-002)
# ---------------------------------------------------------------------

#: The classes that took ``return_dict`` until step 3, plus their factories.
RETURN_DICT_REMOVED = (
    "MobileNetV1", "MobileNetV2", "MobileNetV3", "MobileNetV4", "ViT", "ViTHMLP",
)


@pytest.mark.parametrize("name", RETURN_DICT_REMOVED)
def test_return_dict_is_refused(name: str) -> None:
    """The argument is deleted, not ignored.

    Keras 3.8's ``Layer.__init__`` refuses an unknown keyword with
    ``ValueError("Unrecognized keyword arguments passed to <Class>: ...")``
    (measured), so the refusal is a ``ValueError`` naming the argument, not
    a ``TypeError``.
    """
    builder, _ = SUBJECTS[name]
    with pytest.raises(ValueError, match="return_dict"):
        builder(return_dict=False)


@pytest.mark.parametrize("factory", [
    "dl_techniques.models.vision.mobilenet:create_mobilenetv1",
    "dl_techniques.models.vision.mobilenet:create_mobilenetv2",
    "dl_techniques.models.vision.mobilenet:create_mobilenetv3",
    "dl_techniques.models.vision.mobilenet:create_mobilenetv4",
    "dl_techniques.models.vision.vit:create_vit",
    "dl_techniques.models.vision.vit_hmlp:create_vit_hmlp",
    "dl_techniques.models.vision.cliffordnet:create_cliffordnet",
    "dl_techniques.models.vision.swin_transformer:create_swin_transformer",
    "dl_techniques.models.vision.fractalnet:create_fractal_net",
    "dl_techniques.models.vision.squeezenet:create_squeezenet_v1",
    "dl_techniques.models.vision.squeezenet:create_squeezenodule_net_v2",
    "dl_techniques.models.vision.beit:create_beit_classifier",
    "dl_techniques.models.vision_language.fastvlm:create_fastvlm",
])
def test_no_factory_takes_return_dict(factory: str) -> None:
    import importlib
    import inspect
    module, attr = factory.split(":")
    fn = getattr(importlib.import_module(module), attr)
    assert "return_dict" not in inspect.signature(fn).parameters, (
        f"{attr} still declares return_dict"
    )
    assert "head_config_preset" not in inspect.signature(fn).parameters, (
        f"{attr} still declares the dead head_config_preset"
    )


# ---------------------------------------------------------------------
# The dead head knobs are gone, and refused (plan-2026-10-14T062616-81938e81
# step 4, D-004 and D-017)
# ---------------------------------------------------------------------

#: (subject, dead keyword). ``head_config_preset`` was stored and never read
#: by any Batch C class; SqueezeNetV1's ``head_config_overrides`` configured a
#: shared head the class no longer has.
DEAD_HEAD_KNOBS = [
    ("CliffordNet", "head_config_preset"),
    ("SwinTransformer", "head_config_preset"),
    ("FractalNet", "head_config_preset"),
    ("SqueezeNetV1", "head_config_preset"),
    ("SqueezeNetV1", "head_config_overrides"),
    ("SqueezeNoduleNetV2", "head_config_overrides"),
    ("FastVLM", "head_config_preset"),
]


@pytest.mark.parametrize("name,knob", DEAD_HEAD_KNOBS,
                         ids=[f"{n}-{k}" for n, k in DEAD_HEAD_KNOBS])
def test_a_dead_head_knob_is_refused(name: str, knob: str) -> None:
    """Deleted, not ignored.

    CliffordNet is a subclassed ``keras.Model``, which refuses an unknown
    keyword itself. Swin, FractalNet and SqueezeNet are functional subclasses:
    Keras' ``Functional.__init__`` drops unknown keywords silently (measured),
    so they refuse through ``utils.model_build.refuse_unknown_model_kwargs``,
    as FastVLM (also functional) does since step 5.
    Either way the refusal is ``ValueError`` naming the keyword.
    """
    builder, _ = SUBJECTS[name]
    value = "default" if knob == "head_config_preset" else {}
    with pytest.raises(ValueError, match=knob):
        builder(**{knob: value})


# ---------------------------------------------------------------------
# The census finds its own subjects (plan-2026-10-14T062616-81938e81
# iteration 2 step 6.4, the heads review's A2-3)
# ---------------------------------------------------------------------

#: Calls that build a vision head, with the position of their task argument.
#: ``ClassificationHead`` takes none: building one is always a subject.
_HEAD_BUILDERS = {
    "build_model_vision_head": 0,
    "create_vision_head": 0,
    "create_head": 1,
    "ClassificationHead": None,
}
_MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "models"


def _task_of(call: ast.Call, position: int) -> str:
    """The task a head-building call names, as the ``VisionTaskType`` value.

    :raises AssertionError: If the task is not a literal, so the census
        cannot tell whether the call builds a classifier.
    """
    node = call.args[position] if len(call.args) > position else next(
        (k.value for k in call.keywords if k.arg == "task_type"), None)
    if isinstance(node, ast.Attribute) and ast.unparse(node.value) == "VisionTaskType":
        return node.attr.lower()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    raise AssertionError(
        f"line {call.lineno}: the task of {ast.unparse(call.func)} is not a "
        f"literal ({ast.unparse(node) if node else None}); the census cannot "
        f"tell whether it builds a classifier")


def _classifier_owners(source: str) -> set:
    """Names of the classes (or module-level functions) in ``source`` that
    build a classification head. Other task types (DarkIR and SCUNet build a
    denoising head) are not subjects: the exemption is by task, not by name.
    """
    owners = set()

    def visit(node: ast.AST, owner) -> None:
        for child in ast.iter_child_nodes(node):
            here = owner
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and owner is None:
                here = child.name
            if isinstance(child, ast.Call):
                func = child.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in _HEAD_BUILDERS:
                    position = _HEAD_BUILDERS[name]
                    if position is None or _task_of(child, position) == "classification":
                        owners.add(here)
            visit(child, here)

    visit(ast.parse(source), None)
    return owners


def test_the_owner_scan_sees_a_planted_classifier_and_skips_other_tasks() -> None:
    planted = (
        "class Planted:\n"
        "    def __init__(self):\n"
        "        self.head = build_model_vision_head(\n"
        "            VisionTaskType.CLASSIFICATION, {'num_classes': 3})\n"
        "class Restorer:\n"
        "    def __init__(self):\n"
        "        self.head = build_model_vision_head(VisionTaskType.DENOISING, {})\n"
        "def make_classifier():\n"
        "    return create_vision_head('classification', num_classes=3)\n"
    )
    assert _classifier_owners(planted) == {"Planted", "make_classifier"}
    with pytest.raises(AssertionError, match="not a literal"):
        _classifier_owners("x = build_model_vision_head(task, {})\n")


def test_every_classification_head_owner_has_a_census_row() -> None:
    owners = set()
    for path in sorted(_MODELS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if any(name in source for name in _HEAD_BUILDERS):
            owners |= _classifier_owners(source)
    # Anti-vacuity: 16 classification-head owners (17 calls; ResNet makes two)
    # at 3a89046f8.
    assert len(owners) >= 16, sorted(owners)
    rows = {name.split("_")[0] for name in SUBJECTS}
    assert owners <= rows, (
        f"classification-head owners with no SUBJECTS row: "
        f"{sorted(owners - rows)}; add one so the logits arms reach them")
