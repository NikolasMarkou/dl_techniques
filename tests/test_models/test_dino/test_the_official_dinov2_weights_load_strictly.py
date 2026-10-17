"""The official DINOv2 weights load strictly and give the published model's outputs.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.7 (D-134,
D-135, D-137). ``DINOv2VisionTransformer.from_variant(variant,
pretrained=True)`` reads ``facebook/dinov2-{small,base,large}``'s
``model.safetensors`` from the local download (``official_weights.py``) and
maps every tensor by name.

Three groups:

* **No file needed.** The weight map pairs every name of the official
  manifest (``dinov2_numpy_reference.official_manifest``, the pattern of the
  published files) with one backbone variable of the same shape and fills
  every backbone weight exactly once; a synthetic tensor dict loads into
  every weight with torch's layouts transposed; the load refuses a missing,
  a left-over or a mis-shaped tensor and a non-official structure, assigning
  nothing; an absent download raises before anything is built.
* **The downloaded small file** (skipped, with the path, when absent): it is
  the published file (sha256) with the manifest's names and shapes; one
  loaded block and the whole loaded model (resampled 252 px grid and the
  stored 518 px grid) equal the float64 numpy reference to float32 accuracy.
* **The downloaded base file plus transformers' fixture image**: the loaded
  model reproduces the integration values transformers published for
  ``facebook/dinov2-base`` (``test_modeling_dinov2.py``,
  ``test_inference_no_head``) in both of its interpolation conventions: the
  reference offset 0.1 (v4.32-v4.45, atol 1e-4) and the output-size one
  (v4.50, rtol = atol = 1e-3).
"""

import hashlib
import os
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.dino.dino_v2 import DINOv2VisionTransformer
from dl_techniques.models.vision.dino import official_weights as ow
from tests.numerics import reassociation_atol

from . import dinov2_numpy_reference as ref

SMALL = dict(embed_dim=384, depth=12, num_heads=6)
BASE = dict(embed_dim=768, depth=12, num_heads=12)


def _local(variant):
    try:
        return ow.official_dinov2_weights_path(variant)
    except FileNotFoundError:
        return None


SMALL_PATH = _local("small")
BASE_PATH = _local("base")
FIXTURE = Path(os.environ.get(ow.DINOV2_WEIGHTS_DIR_ENV) or ow.DINOV2_WEIGHTS_DIR) / "hf_fixture" / "000000039769.png"

needs_small = pytest.mark.skipif(
    SMALL_PATH is None, reason="official DINOv2-small weights are not downloaded "
    f"(expected under {ow.DINOV2_WEIGHTS_DIR} or ${ow.DINOV2_WEIGHTS_DIR_ENV})")
needs_base_and_fixture = pytest.mark.skipif(
    BASE_PATH is None or not FIXTURE.is_file(),
    reason=f"official DINOv2-base weights or transformers' fixture image {FIXTURE} absent")


def _backbone(image_size=28, **kwargs):
    kwargs = dict(dict(image_size=image_size, patch_size=14, num_register_tokens=0,
                       init_values=1.0, pos_embed_grid_size=37, **SMALL), **kwargs)
    return DINOv2VisionTransformer(**kwargs)


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=shape).astype(np.float32)
            for name, shape in ref.official_manifest(384, 12).items()}


# ---------------------------------------------------------------------------
# No file needed
# ---------------------------------------------------------------------------


def test_the_map_names_every_official_tensor_once_and_fills_every_weight_once():
    backbone = _backbone()
    entries = ow.official_weight_map(backbone)
    names = [name for name, _, _ in entries]
    manifest = ref.official_manifest(384, 12)
    assert len(names) == len(set(names)) == len(manifest) == 223
    assert set(names) == set(manifest)
    filled = {}
    for name, variable, transform in entries:
        if transform is None:   # key / value: fused into the query entry
            assert ".attention.attention.key." in name or ".attention.attention.value." in name
            continue
        filled.setdefault(id(variable), []).append(name)
    assert all(len(v) == 1 for v in filled.values()), [v for v in filled.values() if len(v) > 1]
    assert {id(w) for w in backbone.weights} == set(filled)


def test_a_synthetic_official_dict_fills_every_weight_in_keras_layout():
    backbone = _backbone()
    tensors = _synthetic()
    ow.load_official_dinov2_weights(backbone, tensors)
    blk = backbone.transformer_blocks[3]
    engine = blk.attention.cross_attention
    p = "encoder.layer.3."
    q, k, v = (tensors[p + f"attention.attention.{n}.weight"] for n in ("query", "key", "value"))
    expect = {
        id(engine.qkv_dense.kernel): np.concatenate([q.T, k.T, v.T], axis=1),
        id(engine.qkv_dense.bias): np.concatenate(
            [tensors[p + f"attention.attention.{n}.bias"] for n in ("query", "key", "value")]),
        id(engine.proj_dense.kernel): tensors[p + "attention.output.dense.weight"].T,
        id(blk.ffn.fc1.kernel): tensors[p + "mlp.fc1.weight"].T,
        id(blk.ffn.fc2.kernel): tensors[p + "mlp.fc2.weight"].T,
        id(blk.ls2.gamma): tensors[p + "layer_scale2.lambda1"],
        id(backbone.patch_embed.proj.kernel): np.transpose(
            tensors["embeddings.patch_embeddings.projection.weight"], (2, 3, 1, 0)),
        id(backbone.pos_embed.pos_embed): tensors["embeddings.position_embeddings"],
        id(backbone.mask_token_layer.weights[0]): tensors["embeddings.mask_token"][None],
        id(backbone.norm.gamma): tensors["layernorm.weight"],
    }
    by_id = {id(w): w for w in backbone.weights}
    for key, value in expect.items():
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(by_id[key]), value, by_id[key].path)
    # Every weight now holds official values: none is left at its init.
    pool = np.concatenate([t.ravel() for t in tensors.values()])
    for w in backbone.weights:
        assert np.isin(keras.ops.convert_to_numpy(w).ravel()[:8], pool).all(), w.path


def _weights(model):
    return [keras.ops.convert_to_numpy(w).copy() for w in model.weights]


@pytest.mark.parametrize("edit", ["missing", "left_over", "wrong_shape"])
def test_the_load_refuses_and_assigns_nothing(edit):
    backbone = _backbone()
    before = _weights(backbone)
    tensors = _synthetic()
    if edit == "missing":
        del tensors["encoder.layer.7.mlp.fc2.bias"]
        match = "missing"
    elif edit == "left_over":
        tensors["encoder.layer.12.norm1.weight"] = np.ones(384, np.float32)
        match = "unused"
    else:
        tensors["embeddings.position_embeddings"] = np.zeros((1, 1 + 36 * 36, 384), np.float32)
        match = "shape mismatches"
    with pytest.raises(ValueError, match=match):
        ow.load_official_dinov2_weights(backbone, tensors)
    for a, b in zip(before, _weights(backbone)):
        np.testing.assert_array_equal(a, b)


def test_a_backbone_unlike_the_official_one_is_refused():
    with pytest.raises(ValueError, match="unfilled backbone weights"):
        ow.load_official_dinov2_weights(_backbone(num_register_tokens=4), _synthetic())
    with pytest.raises(ValueError, match="pos_embed_grid_size=37"):
        ow.load_official_dinov2_weights(_backbone(pos_embed_grid_size=None), _synthetic())
    with pytest.raises(ValueError, match="LayerScale"):
        ow.load_official_dinov2_weights(_backbone(init_values=None), _synthetic())


@pytest.mark.parametrize("variant", ["tiny", "giant"])
def test_a_variant_without_official_weights_raises_not_implemented(variant):
    with pytest.raises(NotImplementedError, match=variant):
        DINOv2VisionTransformer.from_variant(variant, image_size=28, pretrained=True)


def test_an_absent_download_raises_before_anything_is_built(tmp_path, monkeypatch):
    monkeypatch.setenv(ow.DINOV2_WEIGHTS_DIR_ENV, str(tmp_path))

    def no_build(*args, **kwargs):
        raise AssertionError("built the backbone before finding the weights")

    monkeypatch.setattr(DINOv2VisionTransformer, "__init__", no_build)
    with pytest.raises(FileNotFoundError, match=ow.DINOV2_WEIGHTS_DIR_ENV):
        DINOv2VisionTransformer.from_variant("small", image_size=28, pretrained=True)


@pytest.mark.parametrize("kwargs,match", [
    (dict(patch_size=16, image_size=32), "patch_size"),
    (dict(num_register_tokens=4), "num_register_tokens"),
    (dict(ffn_type="swiglu"), "ffn_type"),
    (dict(init_values=None), "init_values"),
])
def test_pretrained_refuses_a_structure_the_checkpoint_lacks(tmp_path, monkeypatch, kwargs, match):
    (tmp_path / "dinov2-small").mkdir()
    (tmp_path / "dinov2-small" / "model.safetensors").write_bytes(b"")
    monkeypatch.setenv(ow.DINOV2_WEIGHTS_DIR_ENV, str(tmp_path))
    kwargs = dict(dict(image_size=28), **kwargs)
    with pytest.raises(ValueError, match=match):
        DINOv2VisionTransformer.from_variant("small", pretrained=True, **kwargs)


# ---------------------------------------------------------------------------
# The downloaded small file
# ---------------------------------------------------------------------------


@needs_small
def test_the_downloaded_files_are_the_published_ones():
    from safetensors import safe_open
    dims = {"small": (384, 12), "base": (768, 12), "large": (1024, 24)}
    checked = 0
    for variant, (dim, depth) in dims.items():
        path = _local(variant)
        if path is None:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == ow.OFFICIAL_DINOV2_CHECKPOINTS[variant][1], variant
        with safe_open(str(path), framework="numpy") as handle:
            shapes = {k: tuple(handle.get_slice(k).get_shape()) for k in handle.keys()}
        assert shapes == ref.official_manifest(dim, depth), variant
        checked += 1
    assert checked >= 1


@pytest.fixture(scope="module")
def small_tensors():
    from safetensors.numpy import load_file
    return load_file(str(SMALL_PATH))


def _atol(reference, tokens, depth):
    # Per block: the q/k/v, projection and two MLP contractions (D, D, D, 4D),
    # the score and value reductions (head width 64, tokens), two LayerNorms (D).
    d = 384
    return reassociation_atol([d, d, d, 4 * d, 64, tokens, d, d], depth,
                              float(np.abs(reference).max()))


@needs_small
def test_one_loaded_block_is_the_reference_block(small_tensors, golden_reference_device):
    tokens = np.random.default_rng(3).normal(size=(1, 1 + 18 * 18, 384)).astype(np.float32)
    want = ref.block(small_tensors, 5, tokens[0].astype(np.float64), 6)
    with keras.device(golden_reference_device):
        backbone = DINOv2VisionTransformer.from_variant("small", image_size=252, pretrained=True)
        got = keras.ops.convert_to_numpy(backbone.transformer_blocks[5](tokens))[0]
    err = float(np.abs(got - want).max())
    assert err <= _atol(want, tokens.shape[1], 1), err


@needs_small
@pytest.mark.parametrize("size", [252, 518])
def test_the_loaded_small_model_is_the_reference_model(small_tensors, golden_reference_device, size):
    image = np.random.default_rng(size).normal(size=(size, size, 3)).astype(np.float32)
    final, layers = ref.forward(small_tensors, image, 6)
    with keras.device(golden_reference_device):
        backbone = DINOv2VisionTransformer.from_variant("small", image_size=size, pretrained=True)
        out = backbone([image[None], np.zeros((1, backbone.num_patches), bool)])
        inter = backbone.get_intermediate_layers(image[None], 4)
    tokens = 1 + (size // 14) ** 2
    cls = keras.ops.convert_to_numpy(out["x_norm_clstoken"])[0]
    patches = keras.ops.convert_to_numpy(out["x_norm_patchtokens"])[0]
    assert float(np.abs(cls - final[0]).max()) <= _atol(final[0], tokens, 12)
    assert float(np.abs(patches - final[1:]).max()) <= _atol(final[1:], tokens, 12)
    for got, want in zip(inter, layers):
        got = keras.ops.convert_to_numpy(got)[0]
        assert float(np.abs(got - want[1:]).max()) <= _atol(want[1:], tokens, 12)
    # A statistic of the published model's CLS feature on this image, so a
    # loaded model that matched the reference only by being near zero would fail.
    assert float(np.abs(final[0]).max()) > 1.0


@needs_small
def test_the_classifier_wrapper_and_the_factory_load_the_same_backbone():
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2, create_dino_v2
    reference = DINOv2VisionTransformer.from_variant("small", image_size=56, pretrained=True)
    for model in (DINOv2.from_variant("small", image_size=56, num_classes=3, pretrained=True),
                  create_dino_v2("small", image_size=56, num_classes=3, pretrained=True)):
        assert model.backbone.num_register_tokens == 0
        assert len(model.backbone.weights) == len(reference.weights)
        for got, want in zip(model.backbone.weights, reference.weights):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(got),
                                          keras.ops.convert_to_numpy(want))


# ---------------------------------------------------------------------------
# transformers' published values for facebook/dinov2-base
# ---------------------------------------------------------------------------

#: ``last_hidden_state[0, :3, :3]``: transformers v4.32.0-v4.45.0 (interpolation
#: at scale (n + 0.1) / 37, the reference's), then v4.50.0 (at the output size).
PUBLISHED = {
    0.1: (np.array([[-2.1747, -0.4729, 1.0936], [-3.2780, -0.8269, -0.9210],
                    [-2.9129, 1.1284, -0.7306]]), dict(rtol=0, atol=1e-4)),
    0.0: (np.array([[-2.2005, -0.4495, 1.0964], [-3.3959, -0.8942, -1.0315],
                    [-2.9355, 1.1564, -0.7656]]), dict(rtol=1e-3, atol=1e-3)),
}


def _hf_preprocess(path):
    """transformers' BitImageProcessor for dinov2-base: shortest edge 256 (PIL
    bicubic), center crop 224, rescale 1/255, ImageNet mean and std."""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    w, h = image.size
    short, long_ = min(w, h), max(w, h)
    new_long = int(256 * long_ / short)
    image = image.resize((256, new_long) if w <= h else (new_long, 256), resample=Image.BICUBIC)
    arr = np.asarray(image)
    top, left = (arr.shape[0] - 224) // 2, (arr.shape[1] - 224) // 2
    arr = arr[top:top + 224, left:left + 224].astype(np.float32) / 255.0
    return (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)


@needs_base_and_fixture
@pytest.mark.parametrize("offset", sorted(PUBLISHED))
def test_the_loaded_base_model_gives_the_published_values(golden_reference_device, offset):
    pixels = _hf_preprocess(FIXTURE)
    with keras.device(golden_reference_device):
        backbone = DINOv2VisionTransformer.from_variant(
            "base", image_size=224, pretrained=True, interpolate_offset=offset)
        out = backbone([pixels[None], np.zeros((1, backbone.num_patches), bool)])
    cls = keras.ops.convert_to_numpy(out["x_norm_clstoken"])[0]
    patches = keras.ops.convert_to_numpy(out["x_norm_patchtokens"])[0]
    got = np.stack([cls[:3], patches[0, :3], patches[1, :3]])
    expected, tolerance = PUBLISHED[offset]
    np.testing.assert_allclose(got, expected, **tolerance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
