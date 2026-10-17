"""DepthAnything's decoder is the reference Depth Anything V1 DPT head.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 8 (D-128). Until
this step the decoder was the shared ``DepthEstimationHead`` in its dpt mode:
one feature map (the ViT's last layer) through Conv-BN-ReLU stages. The
reference (LiheYoung/Depth-Anything ``depth_anything/dpt.py`` ``DPTHead`` with
``blocks.py``, fetched 2026-10-17) reads the patch tokens of the encoder's
last four layers (``get_intermediate_layers(x, 4)``, each through the final
norm), reassembles them at 4x, 2x, 1x and 1/2x the patch grid, fuses them
coarse to fine through ``refinenet4..1`` and has no BatchNorm (``use_bn=False``
in every published ``config.json``). The step-150 one-batch overfit showed the
old decoder's inference-mode loss at 1.06 against 0.39 in training mode, the
gap BatchNorm statistics make (D-127).

``OFFICIAL_VITS_DEPTH_HEAD`` is the module list of the published
``LiheYoung/depth_anything_vits14`` checkpoint (``pytorch_model.bin``, every
``depth_head.*`` tensor shape, read without torch). The reference builds a
``resConfUnit1`` in ``refinenet4`` too, which its forward never calls (no skip
input); this port does not build it (D-128), so the port's count is the
official count less exactly that unit.
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.depth_anything import DepthAnything
from dl_techniques.models.vision.depth_anything.components import DPTHead

OFFICIAL_VITS_DEPTH_HEAD = {
    "projects.0.weight": (48, 384, 1, 1),
    "projects.0.bias": (48,),
    "projects.1.weight": (96, 384, 1, 1),
    "projects.1.bias": (96,),
    "projects.2.weight": (192, 384, 1, 1),
    "projects.2.bias": (192,),
    "projects.3.weight": (384, 384, 1, 1),
    "projects.3.bias": (384,),
    "resize_layers.0.weight": (48, 48, 4, 4),
    "resize_layers.0.bias": (48,),
    "resize_layers.1.weight": (96, 96, 2, 2),
    "resize_layers.1.bias": (96,),
    "resize_layers.3.weight": (384, 384, 3, 3),
    "resize_layers.3.bias": (384,),
    "scratch.layer1_rn.weight": (64, 48, 3, 3),
    "scratch.layer2_rn.weight": (64, 96, 3, 3),
    "scratch.layer3_rn.weight": (64, 192, 3, 3),
    "scratch.layer4_rn.weight": (64, 384, 3, 3),
    "scratch.refinenet1.out_conv.weight": (64, 64, 1, 1),
    "scratch.refinenet1.out_conv.bias": (64,),
    "scratch.refinenet1.resConfUnit1.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet1.resConfUnit1.conv1.bias": (64,),
    "scratch.refinenet1.resConfUnit1.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet1.resConfUnit1.conv2.bias": (64,),
    "scratch.refinenet1.resConfUnit2.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet1.resConfUnit2.conv1.bias": (64,),
    "scratch.refinenet1.resConfUnit2.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet1.resConfUnit2.conv2.bias": (64,),
    "scratch.refinenet2.out_conv.weight": (64, 64, 1, 1),
    "scratch.refinenet2.out_conv.bias": (64,),
    "scratch.refinenet2.resConfUnit1.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet2.resConfUnit1.conv1.bias": (64,),
    "scratch.refinenet2.resConfUnit1.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet2.resConfUnit1.conv2.bias": (64,),
    "scratch.refinenet2.resConfUnit2.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet2.resConfUnit2.conv1.bias": (64,),
    "scratch.refinenet2.resConfUnit2.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet2.resConfUnit2.conv2.bias": (64,),
    "scratch.refinenet3.out_conv.weight": (64, 64, 1, 1),
    "scratch.refinenet3.out_conv.bias": (64,),
    "scratch.refinenet3.resConfUnit1.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet3.resConfUnit1.conv1.bias": (64,),
    "scratch.refinenet3.resConfUnit1.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet3.resConfUnit1.conv2.bias": (64,),
    "scratch.refinenet3.resConfUnit2.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet3.resConfUnit2.conv1.bias": (64,),
    "scratch.refinenet3.resConfUnit2.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet3.resConfUnit2.conv2.bias": (64,),
    "scratch.refinenet4.out_conv.weight": (64, 64, 1, 1),
    "scratch.refinenet4.out_conv.bias": (64,),
    "scratch.refinenet4.resConfUnit1.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet4.resConfUnit1.conv1.bias": (64,),
    "scratch.refinenet4.resConfUnit1.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet4.resConfUnit1.conv2.bias": (64,),
    "scratch.refinenet4.resConfUnit2.conv1.weight": (64, 64, 3, 3),
    "scratch.refinenet4.resConfUnit2.conv1.bias": (64,),
    "scratch.refinenet4.resConfUnit2.conv2.weight": (64, 64, 3, 3),
    "scratch.refinenet4.resConfUnit2.conv2.bias": (64,),
    "scratch.output_conv1.weight": (32, 64, 3, 3),
    "scratch.output_conv1.bias": (32,),
    "scratch.output_conv2.0.weight": (32, 32, 3, 3),
    "scratch.output_conv2.0.bias": (32,),
    "scratch.output_conv2.2.weight": (1, 32, 1, 1),
    "scratch.output_conv2.2.bias": (1,),
}
#: The published vits checkpoint's depth_head total, summed from the table.
OFFICIAL_VITS_DEPTH_HEAD_PARAMS = 2_728_513
#: The tensors of refinenet4.resConfUnit1, which the reference forward never reads.
NEVER_CALLED = {k for k in OFFICIAL_VITS_DEPTH_HEAD if k.startswith("scratch.refinenet4.resConfUnit1.")}

#: (embed dim, features, out_channels) of each published checkpoint's config.json.
PUBLISHED_CONFIGS = {
    "vit_s": (384, 64, (48, 96, 192, 384)),
    "vit_b": (768, 128, (96, 192, 384, 768)),
    "vit_l": (1024, 256, (256, 512, 1024, 1024)),
}


def _official_name(path):
    """`depth_head/refinenet1/res_conf_unit1/conv1/kernel` -> `scratch.refinenet1.resConfUnit1.conv1.weight`."""
    parts = path.split("/")[1:]
    leaf = {"kernel": "weight", "bias": "bias"}[parts[-1]]
    name = ".".join(parts[:-1]).replace("res_conf_unit", "resConfUnit")
    for prefix in ("projects_", "resize_layers_", "output_conv2_"):
        name = name.replace(prefix, prefix[:-1] + ".")
    if not name.startswith(("projects", "resize_layers")):
        name = "scratch." + name
    return f"{name}.{leaf}"


def _torch_shape(shape):
    """A Keras kernel shape in torch's layout.

    Keras Conv2D ``(kh, kw, in, out)`` -> torch Conv2d ``(out, in, kh, kw)``, and
    Keras Conv2DTranspose ``(kh, kw, out, in)`` -> torch ConvTranspose2d
    ``(in, out, kh, kw)``: the same permutation for both.
    """
    if len(shape) != 4:
        return tuple(shape)
    kh, kw, a, b = shape
    return (b, a, kh, kw)


def _reference_param_count(dim, features, out_channels):
    """DPTHead(nclass=1, use_bn=False) parameter count, summed from its module list."""
    c = out_channels
    projects = sum(dim * ci + ci for ci in c)
    resize = (c[0] * c[0] * 16 + c[0]) + (c[1] * c[1] * 4 + c[1]) + (c[3] * c[3] * 9 + c[3])
    layer_rn = sum(ci * features * 9 for ci in c)  # bias=False
    rcu = 2 * (features * features * 9 + features)
    refinenets = 4 * (2 * rcu + features * features + features)
    out1 = features * (features // 2) * 9 + features // 2
    out2 = (features // 2) * 32 * 9 + 32 + 32 + 1
    return projects + resize + layer_rn + refinenets + out1 + out2, rcu


def _head(encoder_type, grid=2):
    dim, features, out_channels = PUBLISHED_CONFIGS[encoder_type]
    head = DPTHead(features, out_channels, name="depth_head")
    head.build([(None, grid, grid, dim)] * 4)
    return head


def test_the_table_is_the_published_checkpoint():
    total = sum(math.prod(s) for s in OFFICIAL_VITS_DEPTH_HEAD.values())
    assert total == OFFICIAL_VITS_DEPTH_HEAD_PARAMS
    reference, rcu = _reference_param_count(*PUBLISHED_CONFIGS["vit_s"])
    assert reference == total
    assert sum(math.prod(OFFICIAL_VITS_DEPTH_HEAD[k]) for k in NEVER_CALLED) == rcu


def test_every_head_weight_is_an_official_tensor_of_the_same_shape():
    head = _head("vit_s")
    ported = {}
    for w in head.weights:
        ported[_official_name(w.path)] = _torch_shape(tuple(w.shape))
    assert set(OFFICIAL_VITS_DEPTH_HEAD) - set(ported) == NEVER_CALLED
    assert set(ported) <= set(OFFICIAL_VITS_DEPTH_HEAD)
    for name, shape in ported.items():
        assert shape == OFFICIAL_VITS_DEPTH_HEAD[name], name
    assert head.count_params() == OFFICIAL_VITS_DEPTH_HEAD_PARAMS - sum(
        math.prod(OFFICIAL_VITS_DEPTH_HEAD[k]) for k in NEVER_CALLED)


@pytest.mark.parametrize("encoder_type", list(PUBLISHED_CONFIGS))
def test_the_head_count_is_the_reference_hand_sum(encoder_type):
    reference, never_called = _reference_param_count(*PUBLISHED_CONFIGS[encoder_type])
    assert _head(encoder_type).count_params() == reference - never_called


def _small_model(**kwargs):
    keras.utils.set_random_seed(0)
    kwargs = dict(dict(encoder_type="vit_s", image_shape=(28, 28, 3), dpt_features=8,
                       dpt_out_channels=[4, 4, 8, 8], use_feature_alignment=False), **kwargs)
    model = DepthAnything(**kwargs)
    model(np.zeros((1,) + kwargs["image_shape"], dtype="float32"))
    return model


def _images(n=2, size=28, seed=0):
    return np.random.default_rng(seed).normal(size=(n, size, size, 3)).astype("float32")


def test_the_head_reads_the_last_four_vit_layers_each_through_the_final_norm():
    # The block outputs are read off the encoder's own functional graph, a path
    # independent of `get_intermediate_layers`, which the head uses.
    model = _small_model()
    encoder, x = model.encoder, _images()
    depth = encoder.depth
    graph = keras.Model(encoder.inputs, [encoder.get_layer(f"block_{i}").output
                                         for i in range(depth - 4, depth)])
    blocks = graph([x, np.zeros((2, encoder.num_patches), bool)])
    by_hand = [encoder.norm(b)[:, 1:, :] for b in blocks]
    read = model._encoder_layers(encoder, x, False)
    assert len(read) == 4
    for got, want in zip(read, by_hand):
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(got),
            keras.ops.convert_to_numpy(keras.ops.reshape(want, (2, 2, 2, -1))))


def test_each_of_the_four_fusion_paths_receives_a_gradient():
    keras.utils.set_random_seed(0)  # an all-zero ReLU output would have no gradient at all
    head = _head("vit_s", grid=3)
    rng = np.random.default_rng(1)
    layers = [tf.Variable(rng.normal(size=(2, 3, 3, 384)).astype("float32")) for _ in range(4)]
    with tf.GradientTape() as tape:
        out = head(layers)
        loss = keras.ops.sum(out)
    assert float(keras.ops.mean(keras.ops.cast(out > 0, "float32"))) > 0.05  # the ReLU passes some
    grads = tape.gradient(loss, layers + head.trainable_variables)
    for i in range(4):
        assert float(tf.reduce_max(tf.abs(grads[i]))) > 0, f"no gradient reaches encoder layer {i}"
    by_path = dict(zip([v.path for v in head.trainable_variables], grads[4:]))
    for i in range(1, 5):
        for part in (f"projects_{i - 1}/", f"layer{i}_rn/", f"refinenet{i}/"):
            moved = [p for p, g in by_path.items() if part in p and float(tf.reduce_max(tf.abs(g))) > 0]
            owned = [p for p in by_path if part in p]
            assert owned and moved == owned, (part, sorted(set(owned) - set(moved)))


@pytest.mark.parametrize("grid", [(2, 2), (3, 5), (4, 4)])
def test_compute_output_shape_is_the_call_shape(grid):
    head = DPTHead(8, (4, 4, 8, 8), output_channels=2)
    shapes = [(1,) + grid + (16,)] * 4
    head.build(shapes)
    out = head([keras.ops.ones(s) for s in shapes])
    assert tuple(out.shape) == head.compute_output_shape(shapes) == (1, 16 * grid[0], 16 * grid[1], 2)


def test_a_keras_round_trip_keeps_every_output_value(tmp_path):
    model = _small_model()
    x = _images()
    before = keras.ops.convert_to_numpy(model(x, training=False))
    assert before.max() > 0, "degenerate probe: an all-zero output"
    path = str(tmp_path / "depth_anything.keras")
    model.save(path)
    loaded = keras.models.load_model(path, compile=False)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(loaded(x, training=False)), before)
    assert loaded.dpt_features == 8 and loaded.dpt_out_channels == [4, 4, 8, 8]


def test_the_output_is_the_same_in_training_and_inference_mode():
    # No BatchNorm, no dropout and no stochastic depth anywhere in the DINOv2
    # model: the two modes compute one function.
    keras.utils.set_random_seed(0)
    model = DepthAnything(encoder_type="vit_s", image_shape=(28, 28, 3),
                          use_feature_alignment=False)
    x = _images()
    inference = keras.ops.convert_to_numpy(model(x, training=False))
    assert inference.max() > 0, "degenerate probe: an all-zero output"
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(model(x, training=True)), inference)
