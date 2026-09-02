"""Explicit ``build()`` materializes exactly the tree ``call()`` runs (v2 §8.3).

The guide asks for a **pair** of tests, because neither half is sufficient:

* the parity half compares the relative weight paths of a model built by an
  explicit ``build(shape)`` against one built lazily by a first ``call()``;
* the anti-vacuity half exists because parity is **blind to over-building** --
  it passes whenever BOTH paths build everything, including everything wrong.

A parity failure is a naming problem before it is a build problem: Keras
auto-increments generated names per instance, so two separately constructed
models produce ``block/w`` versus ``block_1/w`` at every unnamed level and
stripping only the root does not normalize that away. Every sub-layer in this
package therefore carries an explicit ``name=``; this file is what would notice
if one stopped.

**The anti-vacuity half here is not the guide's literal "no head config" arm.**
This package has no optional head. Instead
:class:`_DiTXAWithOneSubLayerDroppedFromBuild` reproduces the actual defect:
``build()`` skips ONE sub-layer that ``call()`` still runs. Explicit-build then
materializes 51 tensors where the lazy path materializes 54, and the parity
predicate must convict it. Without that arm, a parity predicate that compared,
say, only the weight COUNT of the root, or that silently normalized both sides
to the same set, would read green forever. There is a second anti-vacuity arm
below (``use_bias=False``) covering the over-building direction the guide names.

Classes covered: ``DiTXA``, ``DiTXABlock``, ``DiTXAFinalLayer``,
``DiTXATimestepEmbedder``, ``SharedTokenDecoder`` and ``ClassLabelEmbedding``
-- every custom class this plan introduced.
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.class_label_embedding import (
    ClassLabelEmbedding,
)
from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    DiTXABlock,
    DiTXATimestepEmbedder,
)
from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    DiTXAFinalLayer,
)
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    SharedTokenDecoder,
)

HIDDEN = 32
HEADS = 4
TOKENS = 9
BATCH = 2


def relative_paths(model: Any):
    """Weight paths with the root segment stripped, so two INSTANCES compare."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def assert_build_parity(make, explicit_shape, lazy_inputs, *, label: str):
    """The v2 §8.3 parity claim, plus the two things that make it non-vacuous.

    :param make: zero-argument builder returning a FRESH, unbuilt instance.
    :param explicit_shape: the argument for ``instance.build(...)``.
    :param lazy_inputs: the argument for ``instance(...)``.
    :param label: name used in the failure message.
    :return: the shared relative-path list.
    """
    explicit = make()
    explicit.build(explicit_shape)
    lazy = make()
    lazy(lazy_inputs)

    a, b = relative_paths(explicit), relative_paths(lazy)
    assert a, f"{label}: explicit build produced NO weights; parity is vacuous"
    assert b, f"{label}: lazy build produced NO weights; parity is vacuous"

    only_lazy = sorted(set(b) - set(a))
    only_explicit = sorted(set(a) - set(b))
    assert a == b, (
        f"{label}: build() does not materialize the tree call() runs.\n"
        f"  built only by call(): {only_lazy}\n"
        f"  built only by build(): {only_explicit}"
    )
    return a


class TestEverySubLayerCarriesAnExplicitName:
    """Parity's precondition, asserted on its own so a failure is legible."""

    def test_no_weight_path_carries_a_keras_autoincremented_segment(self):
        """``dense_3`` / ``layer_normalization_1`` are the tell.

        A generated name is per-instance and per-process, so it makes parity
        fail for a reason that has nothing to do with ``build()``.
        """
        model = DiTXA.from_variant("tiny")
        model.build(
            {
                "x_t": (BATCH, 8, 8, 4),
                "t": (BATCH,),
                "y": (BATCH,),
                "x_cond": (BATCH, 8, 8, 4),
                "direction": (BATCH,),
            }
        )
        # The only digit-suffixed names the model assigns ON PURPOSE, listed
        # exhaustively rather than by prefix: a stray `block_9` on a depth-2
        # model is exactly the kind of thing this arm should catch.
        deliberate = {f"block_{i}" for i in range(model.depth)} | {
            f"drop_path_{i}" for i in range(model.depth)
        }
        segments = {
            segment for w in model.weights for segment in w.path.split("/")[1:-1]
        }
        assert len(segments) >= 20, (
            f"only {len(segments)} intermediate path segments found; this arm "
            "is looking at a tree too shallow to convict anything"
        )
        generated = sorted(
            segment
            for segment in segments
            if segment.rsplit("_", 1)[-1].isdigit() and segment not in deliberate
        )
        assert not generated, (
            "these weight-path segments look Keras-auto-generated rather than "
            f"explicitly named: {generated}. Give the sub-layer a name=."
        )


class TestParityPerClass:
    """One arm per custom class in the package."""

    def test_ditxa(self):
        shape = {
            "x_t": (BATCH, 8, 8, 4),
            "t": (BATCH,),
            "y": (BATCH,),
            "x_cond": (BATCH, 8, 8, 4),
            "direction": (BATCH,),
        }
        rng = np.random.default_rng(0)
        inputs = {
            "x_t": rng.normal(size=(BATCH, 8, 8, 4)).astype("float32"),
            "t": rng.uniform(0.1, 0.9, size=(BATCH,)).astype("float32"),
            "y": np.zeros((BATCH,), dtype="int32"),
            "x_cond": rng.normal(size=(BATCH, 8, 8, 4)).astype("float32"),
            "direction": np.zeros((BATCH,), dtype="float32"),
        }
        paths = assert_build_parity(
            lambda: DiTXA.from_variant("tiny"), shape, inputs, label="DiTXA"
        )
        assert len(paths) == 54, len(paths)

    def test_ditxa_block(self):
        x = (BATCH, TOKENS, HIDDEN)
        c = (BATCH, HIDDEN)
        rng = np.random.default_rng(1)
        tensors = [
            rng.normal(size=x).astype("float32"),
            rng.normal(size=c).astype("float32"),
            rng.normal(size=x).astype("float32"),
        ]
        assert_build_parity(
            lambda: DiTXABlock(hidden_size=HIDDEN, num_heads=HEADS),
            [x, c, x],
            tensors,
            label="DiTXABlock",
        )

    def test_ditxa_final_layer(self):
        x = (BATCH, TOKENS, HIDDEN)
        c = (BATCH, HIDDEN)
        rng = np.random.default_rng(2)
        assert_build_parity(
            lambda: DiTXAFinalLayer(
                hidden_size=HIDDEN,
                patch_size=2,
                out_channels=4,
                grid_height=3,
                grid_width=3,
            ),
            [x, c],
            [
                rng.normal(size=x).astype("float32"),
                rng.normal(size=c).astype("float32"),
            ],
            label="DiTXAFinalLayer",
        )

    def test_ditxa_timestep_embedder(self):
        rng = np.random.default_rng(3)
        assert_build_parity(
            lambda: DiTXATimestepEmbedder(
                hidden_size=HIDDEN, frequency_embedding_size=16
            ),
            (BATCH,),
            rng.uniform(0.0, 1000.0, size=(BATCH,)).astype("float32"),
            label="DiTXATimestepEmbedder",
        )

    def test_shared_token_decoder(self):
        rng = np.random.default_rng(4)
        assert_build_parity(
            lambda: SharedTokenDecoder(
                vocab_size=17, hidden_dim=24, token_seq_len=4, token_emb_dim=8
            ),
            (BATCH, 32),
            rng.normal(size=(BATCH, 32)).astype("float32"),
            label="SharedTokenDecoder",
        )

    def test_class_label_embedding(self):
        assert_build_parity(
            lambda: ClassLabelEmbedding(
                num_classes=5, hidden_size=HIDDEN, dropout_rate=0.1
            ),
            (BATCH,),
            np.zeros((BATCH,), dtype="int32"),
            label="ClassLabelEmbedding",
        )


class _DiTXAWithOneSubLayerDroppedFromBuild(DiTXA):
    """The seeded defect: ``build()`` skips a sub-layer ``call()`` still runs.

    This is the real failure v2 §8.1 describes -- a weight tree that does not
    match the traced graph, which loads a checkpoint short of a sub-layer's
    weights and reports success. ``cond_embedder_reverse`` is the one dropped
    because it is exercised on every forward pass under D-005 (both conditioning
    embedders always run and ``ops.where`` selects), so its absence is a genuine
    defect and not a dead branch.
    """

    def build(self, input_shape: Any) -> None:
        if self.built:
            return
        skipped = self.cond_embedder_reverse
        self.cond_embedder_reverse = _NullBuild(skipped)
        try:
            super().build(input_shape)
        finally:
            self.cond_embedder_reverse = skipped


class _NullBuild:
    """Proxy whose ``build`` is a no-op; everything else delegates."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def build(self, *args, **kwargs):  # the seeded omission
        return None

    def __getattr__(self, item):
        return getattr(self._wrapped, item)


class TestTheAntiVacuityHalf:
    """The parity predicate must be able to FAIL. Two directions."""

    def test_a_sub_layer_missing_from_build_is_convicted(self):
        """Under-building: ``build()`` materializes less than ``call()`` runs.

        The whole point of the pair. If this arm ever goes green, the parity
        arms above are decoration.
        """
        shape = {
            "x_t": (BATCH, 8, 8, 4),
            "t": (BATCH,),
            "y": (BATCH,),
            "x_cond": (BATCH, 8, 8, 4),
            "direction": (BATCH,),
        }
        rng = np.random.default_rng(5)
        inputs = {
            "x_t": rng.normal(size=(BATCH, 8, 8, 4)).astype("float32"),
            "t": rng.uniform(0.1, 0.9, size=(BATCH,)).astype("float32"),
            "y": np.zeros((BATCH,), dtype="int32"),
            "x_cond": rng.normal(size=(BATCH, 8, 8, 4)).astype("float32"),
            "direction": np.zeros((BATCH,), dtype="float32"),
        }

        broken = _DiTXAWithOneSubLayerDroppedFromBuild.from_variant("tiny")
        broken.build(shape)
        explicit = relative_paths(broken)
        assert not any("cond_embedder_reverse" in p for p in explicit), (
            "the seeded removal did not take effect; this arm proves nothing"
        )

        with pytest.raises(AssertionError, match="cond_embedder_reverse"):
            assert_build_parity(
                lambda: _DiTXAWithOneSubLayerDroppedFromBuild.from_variant(
                    "tiny"
                ),
                shape,
                inputs,
                label="seeded",
            )

    def test_a_knob_that_removes_weights_removes_them_from_both_paths(self):
        """Over-building: parity passes when BOTH paths build everything.

        ``use_bias=False`` must produce a model with strictly fewer weight
        tensors, not the same tree with unused biases sitting in it.
        """
        shape = [(BATCH, TOKENS, HIDDEN), (BATCH, HIDDEN), (BATCH, TOKENS, HIDDEN)]

        with_bias = DiTXABlock(hidden_size=HIDDEN, num_heads=HEADS, use_bias=True)
        with_bias.build(shape)
        without = DiTXABlock(hidden_size=HIDDEN, num_heads=HEADS, use_bias=False)
        without.build(shape)

        assert len(without.weights) < len(with_bias.weights), (
            f"use_bias=False built {len(without.weights)} tensors and "
            f"use_bias=True built {len(with_bias.weights)}; the knob is not "
            "removing anything, so build() is over-building on one of the two "
            "and parity cannot see it"
        )
        # The adaLN modulation Dense keeps its bias unconditionally -- upstream
        # always has it, and it is the zero-init shift/scale/gate source.
        assert any("adaln_modulation/bias" in w.path for w in without.weights)
