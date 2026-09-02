"""The token packing must agree with the patch embedder's own conv geometry.

Raised by an orchestrator measurement during step 6 and required by step 7
(``probes/orchestrator_transpose_verification.md``).

**The gap this closes.** ``test_the_token_bridge_is_a_bijection.py`` pins the
packing in ISOLATION: which flat column lands at which ``(row, col, channel)``.
That is a real guard and it is not this one. The bridge tensor is consumed by
``PatchEmbedding2D``, a ``Conv2D`` whose kernel and stride both equal the patch
size, so it reads non-overlapping ``p x p`` blocks; whether a token's payload
lands *inside* whole conv blocks is a JOINT property of ``token_flat_to_bridge``
and the embedder's ``patch_size``, and neither side's own tests can see it.

Measured by the orchestrator on the ``sd`` geometry: with the spatial transpose,
**0 of 16** conv patches draw from more than one token; without it, **16 of 16**
do -- every single visual token would be a blend of two unrelated text tokens
before the transformer ever ran. No shape assertion anywhere notices.

Two arms, deliberately different instruments:

* a **geometry** arm that reads ``kernel_size`` and ``strides`` off the model's
  real ``Conv2D`` and counts multi-token windows;
* a **behavioural** arm that runs that real convolution with an all-ones kernel,
  so a nonzero output position means "this conv patch saw this token" as a
  measured fact rather than an inferred one.

The second injection the orchestrator names -- a patch embedder whose
``patch_size`` disagrees with ``BridgeConfig.patch_size`` -- is invisible to
every other test in this directory, because each side is individually correct.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import (
    BRIDGE_PRESETS,
)
from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA
from dl_techniques.models.vision_language.bit_diffusion.token_bridge import (
    token_flat_to_bridge,
)

from ._ditxa_helpers import np_

#: The `tiny` preset, whose geometry the `tiny` model variant is built on:
#: 8 tokens x 32 dims == 8 * 8 * 4, patch 2, so 2 patches per token, 16 patches.
CONFIG = BRIDGE_PRESETS["tiny"]


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A ``tiny`` model whose ``x_embedder`` -- the object under test -- is built.

    Only the patch embedder is built, deliberately. A full forward pass would
    RAISE under the second injection this file exists to catch (a patch size
    that disagrees with the bridge geometry makes ``pos_embed``'s token count
    wrong), and a crash in an unrelated line is a weaker proof than the
    assertion the guard was written to fire: this repo has already recorded one
    RED proof defeated by an earlier guard.
    """
    m = DiTXA.from_variant("tiny")
    m.x_embedder.build((None, CONFIG.height, CONFIG.width, CONFIG.channels))
    return m


def _label_bridge() -> np.ndarray:
    """A bridge tensor whose every element is the 1-based index of its token."""
    labels = np.repeat(
        np.arange(1, CONFIG.token_seq_len + 1, dtype="float32"),
        CONFIG.token_emb_dim,
    )[None, :]
    return np_(token_flat_to_bridge(keras.ops.convert_to_tensor(labels), CONFIG))


def _windows(bridge: np.ndarray, kernel, strides):
    """Yield the label set of every conv window, using the conv's OWN geometry."""
    kh, kw = int(kernel[0]), int(kernel[1])
    sh, sw = int(strides[0]), int(strides[1])
    height, width = bridge.shape[1], bridge.shape[2]
    for row in range(0, height - kh + 1, sh):
        for col in range(0, width - kw + 1, sw):
            yield set(np.unique(bridge[0, row : row + kh, col : col + kw, :]).tolist())


class TestEveryConvPatchDrawsFromOneToken:
    """The geometry arm."""

    def test_no_conv_window_mixes_two_tokens(self, model):
        """0 of ``num_patches`` windows may see more than one token label."""
        bridge = _label_bridge()
        conv = model.x_embedder.proj
        sets = list(_windows(bridge, conv.kernel_size, conv.strides))
        mixed = [i for i, labels in enumerate(sets) if len(labels) != 1]
        assert not mixed, (
            f"{len(mixed)} of {len(sets)} conv patches draw from more than one "
            f"token: {[(i, sorted(sets[i])) for i in mixed[:4]]}. The packing and "
            "the patch embedder's grid disagree -- either the spatial transpose "
            "in token_bridge.py is gone, or the embedder's patch_size differs "
            f"from BridgeConfig.patch_size ({CONFIG.patch_size})."
        )

    def test_the_windows_tile_the_bridge_exactly(self, model):
        """Anti-vacuity: the window walk must cover the whole tensor once."""
        bridge = _label_bridge()
        conv = model.x_embedder.proj
        sets = list(_windows(bridge, conv.kernel_size, conv.strides))
        assert len(sets) == CONFIG.num_patches, (
            f"the window walk produced {len(sets)} windows, the geometry says "
            f"{CONFIG.num_patches}"
        )
        assert int(conv.kernel_size[0]) > 1, (
            "a 1x1 patch makes 'one token per window' trivially true and this "
            "whole file vacuous"
        )
        assert tuple(conv.kernel_size) == tuple(conv.strides) == (
            CONFIG.patch_size,
            CONFIG.patch_size,
        ), (
            f"the embedder reads {tuple(conv.kernel_size)}-sized patches while "
            f"BridgeConfig packs {CONFIG.patch_size}-sized ones; the two sides "
            "are each internally consistent and jointly wrong"
        )
        assert model.num_patches == CONFIG.num_patches

    def test_each_token_occupies_a_whole_number_of_conv_patches(self, model):
        """Every label must own exactly ``patches_per_token`` windows."""
        bridge = _label_bridge()
        conv = model.x_embedder.proj
        counts = {}
        for labels in _windows(bridge, conv.kernel_size, conv.strides):
            (label,) = labels
            counts[label] = counts.get(label, 0) + 1
        assert sorted(counts) == [
            float(i) for i in range(1, CONFIG.token_seq_len + 1)
        ]
        assert set(counts.values()) == {CONFIG.patches_per_token}, counts

    def test_the_detector_sees_a_deliberately_smeared_packing(self):
        """Dead-component probe, self-contained.

        The transpose removed, reproduced here in NumPy rather than by editing
        the source, so the detector's power is asserted even in a run where no
        injection is active. If this arm ever goes green the arms above have
        stopped discriminating.
        """
        labels = np.repeat(
            np.arange(1, CONFIG.token_seq_len + 1, dtype="float32"),
            CONFIG.token_emb_dim,
        )
        p, h, w, c = (
            CONFIG.patch_size,
            CONFIG.patch_h,
            CONFIG.patch_w,
            CONFIG.channels,
        )
        smeared = labels.reshape(1, h, w, p, p, c).reshape(1, h * p, w * p, c)
        mixed = [
            i
            for i, s in enumerate(_windows(smeared, (p, p), (p, p)))
            if len(s) != 1
        ]
        assert len(mixed) == CONFIG.num_patches, (
            "deleting the spatial transpose must smear EVERY conv patch; the "
            f"detector saw only {len(mixed)} of {CONFIG.num_patches}"
        )


class TestTheRealConvolutionSeesOneTokenPerPatch:
    """The behavioural arm: the actual ``PatchEmbedding2D``, actually run."""

    def _support(self, model) -> dict:
        """Patch indices whose embedding is nonzero, per token.

        The conv kernel is set to ones and the bias to zeros, so each output
        equals the SUM of its window -- a deterministic membership test rather
        than a probabilistic one. A random kernel would make "nonzero" an
        almost-sure statement instead of a certain one.
        """
        conv = model.x_embedder.proj
        conv.kernel.assign(np.ones(np_(conv.kernel).shape, dtype="float32"))
        conv.bias.assign(np.zeros(np_(conv.bias).shape, dtype="float32"))

        support = {}
        for token in range(CONFIG.token_seq_len):
            flat = np.zeros((1, CONFIG.token_flat_dim), dtype="float32")
            flat[0, token * CONFIG.token_emb_dim : (token + 1) * CONFIG.token_emb_dim] = 1.0
            bridge = token_flat_to_bridge(keras.ops.convert_to_tensor(flat), CONFIG)
            out = np_(model.x_embedder(bridge))
            support[token] = set(np.nonzero(np.abs(out[0]).sum(axis=-1))[0].tolist())
        return support

    def test_each_tokens_support_is_disjoint_and_the_right_size(self, model):
        """No two tokens may activate the same conv patch."""
        support = self._support(model)
        for token, patches in support.items():
            assert len(patches) == CONFIG.patches_per_token, (
                f"token {token} activates {len(patches)} conv patches, expected "
                f"{CONFIG.patches_per_token}: {sorted(patches)}"
            )
        for a in range(CONFIG.token_seq_len):
            for b in range(a + 1, CONFIG.token_seq_len):
                overlap = support[a] & support[b]
                assert not overlap, (
                    f"tokens {a} and {b} both reach conv patches "
                    f"{sorted(overlap)}: the patch embedding fuses two "
                    "unrelated text tokens into one visual token"
                )

    def test_the_supports_cover_every_patch(self, model):
        """Anti-vacuity: an all-empty support set would trivially be disjoint."""
        support = self._support(model)
        covered = set().union(*support.values())
        assert covered == set(range(CONFIG.num_patches)), (
            f"the token supports cover {len(covered)} of {CONFIG.num_patches} "
            "conv patches; a measurement that sees nothing cannot see a smear"
        )
