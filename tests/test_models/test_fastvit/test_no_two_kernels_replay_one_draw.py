"""No two kernels of the FastViT tower or of one of its blocks replay one initializer draw.

Every FastViT block resolved its ``kernel_initializer`` once and handed that one
seedless instance to all of its sub-layers, ``FastVitImageEncoder`` handed its own to
the stem, every stage and the projection, and ``MobileOneBlock`` /
``ReparamLargeKernelConv`` handed theirs to every branch. A seedless instance reaching
several weights replays one sample at each (``initializers/clone.py``), so the whole
tower drew from ONE ``he_normal`` stream. MEASURED with the shared instrument
(``initializer_replay_oracle``) on an mci0 tower at seed 1234: **1653 of 7626**
kernel pairs replayed (r = 1.0), among them the mixer kernels of every RepMixer
block of a stage, the stem, both large-kernel branches of every downsample, and
attention ``qkv`` against ``proj``. Only ``FastVitConvMlp`` did not replay: its
``MLPBlock`` already cloned per projection, so its row is coverage, not a RED proof.

Each site now takes ``clone_initializer(...)`` (the blocks and the two conv layers),
the encoder stores an ``initializer_spec(...)``, and the attention block passes
``output_kernel_initializer`` so ``qkv`` and ``proj`` separate. A SEEDED initializer
still replays at every site, by contract; :func:`test_a_seeded_initializer_still_replays`
is the positive control that the instrument sees a replay in these blocks at all.
"""

import keras
import numpy as np
import pytest

from ..initializer_replay_oracle import replayed_pairs
from dl_techniques.layers.conv_blocks.mobile_one_block import MobileOneBlock
from dl_techniques.layers.conv_blocks.reparam_large_kernel_conv import ReparamLargeKernelConv
from dl_techniques.models.vision.fastvit import FastVitImageEncoder
from dl_techniques.models.vision.fastvit.components import (
    FastVitAttentionBlock,
    FastVitConvMlp,
    FastVitPatchEmbed,
    FastVitRepMixer,
    FastVitRepMixerBlock,
    FastVitStage,
)


def _wrap(layer: keras.layers.Layer, input_shape) -> keras.Model:
    """A functional model around ``layer``, built, so ``replayed_pairs`` can read it."""
    inputs = keras.Input(input_shape)
    return keras.Model(inputs, layer(inputs, training=False))


#: (id, builder, input shape). Each builder runs under a fixed seed, so a replay is
#: deterministic. Two blocks per stage so a draw replayed ACROSS blocks is seen too.
_BLOCKS = [
    ("conv_mlp", lambda: FastVitConvMlp(dim=32, hidden_dim=96), (8, 8, 32)),
    ("rep_mixer", lambda: FastVitRepMixer(dim=32), (8, 8, 32)),
    ("rep_mixer_block", lambda: FastVitRepMixerBlock(dim=32, mlp_ratio=3.0), (8, 8, 32)),
    ("patch_embed", lambda: FastVitPatchEmbed(embed_dim=64, use_se=True), (16, 16, 32)),
    ("attention_block", lambda: FastVitAttentionBlock(dim=64, head_dim=32), (4, 4, 64)),
    ("stage_repmixer", lambda: FastVitStage(dim=64, depth=2), (16, 16, 32)),
    ("stage_attention", lambda: FastVitStage(
        dim=64, depth=2, token_mixer='attention', use_pos_emb=True,
        normalization_type='layer_norm'), (16, 16, 32)),
    ("mobile_one_block", lambda: MobileOneBlock(
        out_channels=32, kernel_size=3, num_conv_branches=2, use_se=True), (8, 8, 32)),
    ("reparam_large_kernel_conv", lambda: ReparamLargeKernelConv(
        out_channels=32, kernel_size=7, stride=2, group_size=0, small_kernel=3,
        use_se=True), (16, 16, 16)),
]


@pytest.mark.parametrize("build,input_shape", [(b, s) for _, b, s in _BLOCKS],
                         ids=[name for name, _, _ in _BLOCKS])
def test_no_two_block_kernels_replay_one_draw(build, input_shape):
    keras.utils.set_random_seed(1234)
    model = _wrap(build(), input_shape)
    replayed, compared = replayed_pairs(model)
    assert compared > 0, "the block exposes fewer than two comparable kernels"
    assert not replayed, f"{len(replayed)} of {compared} kernel pairs replay one draw: {replayed[:3]}"


def _tower(variant: str, size: int = 64, **overrides) -> FastVitImageEncoder:
    keras.utils.set_random_seed(1234)
    model = FastVitImageEncoder.from_variant(
        variant, input_shape=(size, size, 3), projection_dim=64, **overrides)
    model(np.zeros((1, size, size, 3), dtype="float32"), training=False)
    return model


def test_no_two_mci0_kernels_replay_one_draw():
    """The real mci0 tower: 4 stages, SE downsamples, attention in the last stage."""
    replayed, compared = replayed_pairs(_tower("mci0"))
    assert compared >= 5000, f"only {compared} kernel pairs compared"
    assert not replayed, (
        f"{len(replayed)} of {compared} kernel pairs replay one initializer draw "
        f"(1653 of 7626 before the fix). Pairs: {replayed[:5]}")


def test_no_two_mci3_shaped_kernels_replay_one_draw():
    """mci3's structure (5 stages, two attention stages with RepCPE, layer norm), shrunk."""
    # 128 px, not 64: five stages take 64 px down to a 1 x 1 last stage, where attention
    # softmaxes over a single token and Keras warns (an error under this suite).
    replayed, compared = replayed_pairs(_tower(
        "mci3", size=128, layers=(1, 2, 1, 2, 1), embed_dims=(32, 64, 128, 256, 512)))
    assert compared >= 1000, f"only {compared} kernel pairs compared"
    assert not replayed, f"{len(replayed)} of {compared} kernel pairs replay one draw: {replayed[:5]}"


def test_the_encoder_stores_a_spec_not_an_instance():
    """A stored instance would be the one object every site resolves; a spec is resolved per site."""
    model = FastVitImageEncoder.from_variant(
        "mci0", input_shape=(64, 64, 3),
        kernel_initializer=keras.initializers.HeNormal())
    assert isinstance(model.kernel_initializer, (str, dict))
    assert model.get_config()["kernel_initializer"] == model.kernel_initializer


def test_a_seeded_initializer_still_replays():
    """Positive control: a SEEDED instance replays at every site by contract, and the instrument sees it.

    Without this, a regression that broke the instrument (or the wrapping) would let every
    test above pass vacuously.
    """
    keras.utils.set_random_seed(1234)
    block = FastVitRepMixer(dim=32, kernel_initializer=keras.initializers.HeNormal(seed=7))
    replayed, _ = replayed_pairs(_wrap(block, (8, 8, 32)))
    assert replayed, "a seeded initializer must replay across the mixer's conv and scale branches"
