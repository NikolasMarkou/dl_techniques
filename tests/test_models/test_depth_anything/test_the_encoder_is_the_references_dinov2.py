"""Depth Anything's encoder is the reference's DINOv2, and it loads the official weights.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.7 (D-136,
D-137). The reference (LiheYoung/Depth-Anything ``depth_anything/dpt.py``)
builds ``torch.hub``'s ``dinov2_vit{s,b,l}14`` (no registers, the 37 x 37
position table resampled to the input) and reads ``get_intermediate_layers(x,
4)``; the paper fine-tunes it from DINOv2's weights with a randomly
initialized head. Until this step the 'real' encoder was the in-tree ViT at
patch 16, which no official checkpoint fits.

The value test (skipped, with the path, without the download) runs
``DepthAnything.from_variant('vit_s', pretrained=True)`` at 252 px, the
gate's size, and compares the four maps the head reads with the float64
numpy reference over the official file.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything import DepthAnything
from dl_techniques.models.vision.dino import official_weights as ow
from dl_techniques.models.vision.dino.dino_v2 import DINOv2VisionTransformer
from tests.numerics import reassociation_atol

from ..test_dino import dinov2_numpy_reference as ref


def _small(**kwargs):
    keras.utils.set_random_seed(0)
    kwargs = dict(dict(encoder_type="vit_s", image_shape=(28, 28, 3), dpt_features=8,
                       dpt_out_channels=[4, 4, 8, 8], use_feature_alignment=True,
                       enable_semi_supervised=True), **kwargs)
    model = DepthAnything(**kwargs)
    model(np.zeros((1,) + kwargs["image_shape"], "float32"))
    return model


def test_the_real_encoder_has_the_reference_structure():
    model = _small()
    for encoder in (model.encoder, model.frozen_encoder):
        assert isinstance(encoder, DINOv2VisionTransformer)
        assert encoder.patch_size == (14, 14)
        assert encoder.pos_embed_grid_size == 37
        assert encoder.num_register_tokens == 0
        assert all(b.ls1 is not None for b in encoder.transformer_blocks)
        assert tuple(encoder.pos_embed.pos_embed.shape) == (1, 1 + 37 * 37, 384)
    assert not model.encoder.mask_token_layer.trainable
    assert model.encoder_h == model.encoder_w == 2


def test_the_teacher_is_a_dinov2_copy_the_head_can_read():
    model = _small()
    x = np.random.default_rng(0).normal(size=(2, 28, 28, 3)).astype("float32")
    student = model._encoder_layers(model.encoder, x, False)
    teacher = model._encoder_layers(model.frozen_encoder, x, False)
    for s, t in zip(student, teacher):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(s), keras.ops.convert_to_numpy(t))
    assert tuple(model._pseudo_label_depth(x).shape) == (2, 28, 28, 1)


@pytest.mark.parametrize("side", [32, 64, 256])
def test_a_side_that_is_not_a_multiple_of_14_is_refused(side):
    with pytest.raises(ValueError, match="multiples of 14"):
        DepthAnything(encoder_type="vit_s", image_shape=(side, 28, 3))


def test_the_placeholder_takes_any_side():
    model = DepthAnything(encoder_type="vit_s", image_shape=(64, 72, 3), encoder_kind="placeholder",
                          use_feature_alignment=False, dpt_features=8, dpt_out_channels=[8, 8, 8, 8])
    assert tuple(model(np.zeros((1, 64, 72, 3), "float32")).shape) == (1, 64, 72, 1)


def test_from_variant_refuses_before_building(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown variant"):
        DepthAnything.from_variant("vit_xl")
    with pytest.raises(ValueError, match="encoder_kind"):
        DepthAnything.from_variant("vit_s", pretrained=True, encoder_kind="placeholder")
    monkeypatch.setenv(ow.DINOV2_WEIGHTS_DIR_ENV, str(tmp_path))

    def no_build(*args, **kwargs):
        raise AssertionError("built the model before finding the weights")

    monkeypatch.setattr(DepthAnything, "__init__", no_build)
    with pytest.raises(FileNotFoundError, match=ow.DINOV2_WEIGHTS_DIR_ENV):
        DepthAnything.from_variant("vit_s", pretrained=True)


def _small_weights():
    try:
        return ow.official_dinov2_weights_path("small")
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_small_weights() is None, reason="official DINOv2-small weights are not "
                    f"downloaded (expected under {ow.DINOV2_WEIGHTS_DIR} or ${ow.DINOV2_WEIGHTS_DIR_ENV})")
def test_a_pretrained_model_reads_the_official_dinov2_layers(golden_reference_device):
    from safetensors.numpy import load_file
    tensors = load_file(str(_small_weights()))
    image = np.random.default_rng(7).normal(size=(252, 252, 3)).astype("float32")
    _final, layers = ref.forward(tensors, image, 6)
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(0)
        model = DepthAnything.from_variant("vit_s", pretrained=True, image_shape=(252, 252, 3),
                                           enable_semi_supervised=True)
        read = model._encoder_layers(model.encoder, image[None], False)
        teacher = model._encoder_layers(model.frozen_encoder, image[None], False)
        depth = keras.ops.convert_to_numpy(model(image[None]))
    assert len(read) == 4
    for got, want, t in zip(read, layers, teacher):
        got = keras.ops.convert_to_numpy(got)[0].reshape(-1, 384)
        want = want[1:]
        bound = reassociation_atol([384, 384, 384, 1536, 64, 325, 384, 384], 12,
                                   float(np.abs(want).max()))
        assert float(np.abs(got - want).max()) <= bound
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(t)[0].reshape(-1, 384), got)
    # The head is not loaded (the paper's randomly initialized decoder): finite, non-negative.
    assert np.isfinite(depth).all() and depth.min() >= 0 and depth.shape == (1, 252, 252, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
