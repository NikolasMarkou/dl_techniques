"""A vision classifier built on the shared ``ClassificationHead`` returns bare logits.

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

Adding a consumer is one entry in :data:`SUBJECTS`.

Step 3 of the same plan (decisions D-013 to D-015) added two more arms. Each consumer's
head is configured as its paper's or reference implementation's
(``test_the_head_has_the_reference_shape``, one row per class in
:data:`REFERENCE_HEAD_OWNERS`), and the model's own initializer and regularizer
knobs reach the classifier (``test_the_model_knobs_reach_the_classifier``). Two
parameter counts are pinned against external references: ResNet-50 against
torchvision's published count and AlexNet against the paper's layer list
summed by hand.
"""

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
}


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
    kernel = model.classification_head.classifier.kernel
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
}


def test_the_head_has_the_reference_shape(subject: str) -> None:
    model, shape = _build(subject)
    model(_images(shape), training=False)
    head = model.classification_head
    # Paths read ``<model>/classification_head/<owner>/...``.
    owners = {
        parts[parts.index("classification_head") + 1]
        for parts in (w.path.split("/") for w in head.weights)
    }
    assert owners == REFERENCE_HEAD_OWNERS[subject], (
        f"{subject} head weight owners {sorted(owners)}, the reference head "
        f"has {sorted(REFERENCE_HEAD_OWNERS[subject])}"
    )
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
}

#: Where each class's ``dropout_rate`` must land in its reference head.
CLASSIFIER_DROPOUT = {"MobileNetV1", "MobileNetV2", "ViT", "ViTHMLP"}
BLOCK_DROPOUT = {"MobileNetV3", "MobileNetV4"}


def test_the_model_knobs_reach_the_classifier(subject: str) -> None:
    builder, shape = SUBJECTS[subject]
    keras.utils.set_random_seed(0)
    model = builder(**KNOBS[subject])
    model(_images(shape), training=False)
    head = model.classification_head
    classifier = head.classifier
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
    if "kernel_initializer" in knobs:
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
