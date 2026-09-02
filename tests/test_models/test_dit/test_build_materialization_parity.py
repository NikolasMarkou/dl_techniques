"""Explicit ``build()`` materializes exactly the tree ``call()`` runs (v2 §8.3).

The guide asks for a **pair** of tests, because neither half is sufficient:

* the parity half compares the relative weight paths of a model built by an
  explicit ``build(shape)`` against one built lazily by a first ``call()``;
* the anti-vacuity half exists because parity is **blind to over-building** --
  it passes whenever BOTH paths build everything, including everything wrong,
  and it passes trivially whenever the predicate cannot distinguish two sets.

A parity failure is a naming problem before it is a build problem. Keras
auto-increments generated names per instance, so two separately constructed
models produce ``dense/kernel`` versus ``dense_1/kernel`` at every unnamed
level, and stripping only the ROOT segment (which is what
:func:`_dit_helpers.relative_paths` does, deliberately) does not normalize that
away. Every sub-layer in ``dit/`` therefore carries an explicit ``name=``; the
first class below is what would notice if one stopped.

**Why this matters for this package specifically.** ``DiT.build()`` does not
delegate to Keras' lazy machinery: it calls ``build()`` on each of the five
sub-trees by hand, with shapes it derives from ``input_shape`` and its own
config (``token_shape``, ``c_shape``). A hand-written ``build()`` is exactly the
thing that drifts from ``call()``, and the symptom is silent -- a checkpoint
written from the explicitly-built model is short a sub-layer's weights and
``load_weights`` reports success.

**Measured at step 9**: the tiny configuration materializes **41** weight
tensors on both paths, of which 2 are non-trainable (``pos_embed`` and the
timestep frequency ladder ``t_embedder/freqs``).
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.blocks import DiTBlock, DiTFinalLayer
from dl_techniques.models.vision_language.dit.model import DiT

from ._dit_helpers import BATCH, TINY, dit_config, relative_paths, tiny_inputs

HIDDEN = TINY["hidden_size"]
HEADS = TINY["num_heads"]
TOKENS = (TINY["input_size"] // TINY["patch_size"]) ** 2

#: Measured 2026-09-02 on :data:`TINY`. Re-derived by the arms below from the
#: two paths themselves; quoted here so a change is legible in the diff.
TINY_WEIGHT_TENSORS = 41


def model_build_shape(config: dict, batch: int = BATCH) -> list:
    """``[x_shape, t_shape, y_shape]`` for a configuration's geometry."""
    n, c = config["input_size"], config["in_channels"]
    return [(batch, n, n, c), (batch,), (batch,)]


def assert_build_parity(make, explicit_shape, lazy_inputs, *, label: str):
    """The v2 §8.3 parity claim, plus the checks that make it non-vacuous.

    Interface contract: ``make`` is a zero-argument callable returning a FRESH,
    UNBUILT instance -- it is invoked twice and the two instances must not share
    state. ``explicit_shape`` is the argument for ``instance.build(...)`` and
    ``lazy_inputs`` the argument for ``instance(...)``. Returns the shared
    relative-path list so a caller can make a stronger claim on top; raises
    ``AssertionError`` naming the asymmetric paths otherwise.

    :param make: Zero-argument builder returning a fresh, unbuilt instance.
    :param explicit_shape: Argument for the explicit ``build``.
    :param lazy_inputs: Argument for the lazy ``__call__``.
    :param label: Name used in the failure message.
    :return: The shared sorted relative-path list.
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
        f"  built only by call():  {only_lazy}\n"
        f"  built only by build(): {only_explicit}"
    )
    return a


# ---------------------------------------------------------------------
# Parity's precondition
# ---------------------------------------------------------------------


class TestEverySubLayerCarriesAnExplicitName:
    """Asserted on its own so a parity failure is legible when it happens."""

    def test_no_weight_path_carries_a_keras_autoincremented_segment(self) -> None:
        """``dense_3`` / ``layer_normalization_1`` are the tell.

        A generated name is per-instance and per-process, so it makes parity
        fail for a reason that has nothing to do with ``build()``.
        """
        model = DiT(**TINY)
        model.build(model_build_shape(TINY))

        # The only digit-suffixed names this model assigns ON PURPOSE, listed
        # exhaustively rather than by prefix: a stray `block_9` on a depth-2
        # model is exactly what this arm should catch.
        deliberate = {f"block_{i}" for i in range(model.depth)}
        segments = {
            segment for w in model.weights for segment in w.path.split("/")[1:-1]
        }
        assert len(segments) >= 8, (
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


# ---------------------------------------------------------------------
# Parity, one arm per class this package defines
# ---------------------------------------------------------------------


class TestParityPerClass:
    """``DiT`` and the two block classes ``dit/`` owns."""

    def test_dit(self) -> None:
        paths = assert_build_parity(
            lambda: DiT(**TINY),
            model_build_shape(TINY),
            list(tiny_inputs(seed=0)),
            label="DiT",
        )
        assert len(paths) == TINY_WEIGHT_TENSORS, len(paths)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"depth": 1},
            {"learn_sigma": False},
            {"class_dropout_rate": 0.0},
            {"use_bias": False},
            # The degenerate single-token grid: input_size == patch_size, so
            # `build()` derives token_shape = (B, 1, D). A hand-written build
            # that folded the token axis away would only show up here.
            {"input_size": 4, "patch_size": 4},
        ],
        ids=["depth1", "no_sigma", "no_null_row", "no_bias", "single_token"],
    )
    def test_dit_under_a_non_default_configuration(self, overrides: dict) -> None:
        """Parity is a property of the CONFIG SPACE, not of one config.

        ``build()`` derives ``token_shape`` and ``c_shape`` from the config, so
        a config whose derivation is wrong builds a different tree from the one
        ``call()`` runs -- and every arm above would stay green.
        """
        config = dit_config(**overrides)
        assert_build_parity(
            lambda: DiT(**config),
            model_build_shape(config),
            list(tiny_inputs(seed=1, config=config)),
            label=f"DiT{overrides}",
        )

    def test_dit_block(self) -> None:
        rng = np.random.default_rng(1)
        token_shape = (BATCH, TOKENS, HIDDEN)
        c_shape = (BATCH, HIDDEN)
        assert_build_parity(
            lambda: DiTBlock(hidden_size=HIDDEN, num_heads=HEADS),
            [token_shape, c_shape],
            [
                rng.normal(size=token_shape).astype("float32"),
                rng.normal(size=c_shape).astype("float32"),
            ],
            label="DiTBlock",
        )

    def test_dit_final_layer(self) -> None:
        rng = np.random.default_rng(2)
        token_shape = (BATCH, TOKENS, HIDDEN)
        c_shape = (BATCH, HIDDEN)
        assert_build_parity(
            lambda: DiTFinalLayer(
                hidden_size=HIDDEN,
                patch_size=TINY["patch_size"],
                out_channels=2 * TINY["in_channels"],
            ),
            [token_shape, c_shape],
            [
                rng.normal(size=token_shape).astype("float32"),
                rng.normal(size=c_shape).astype("float32"),
            ],
            label="DiTFinalLayer",
        )


# ---------------------------------------------------------------------
# The anti-vacuity half
# ---------------------------------------------------------------------


class _NullBuild:
    """Proxy whose ``build`` is a no-op; everything else delegates."""

    def __init__(self, wrapped: Any) -> None:
        object.__setattr__(self, "_wrapped", wrapped)

    def build(self, *args: Any, **kwargs: Any) -> None:  # the seeded omission
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_wrapped")(*args, **kwargs)


class _DiTWithOneSubLayerDroppedFromBuild(DiT):
    """The seeded defect: ``build()`` skips a sub-layer ``call()`` still runs.

    ``y_embedder`` is the one dropped because it runs unconditionally on every
    forward pass -- ``c = t_emb + y_emb`` -- so its absence from the weight tree
    is a genuine defect and not a dead branch. This is the real v2 §8.1 failure:
    a weight tree that does not match the traced graph, which loads a checkpoint
    short of a sub-layer's weights and reports success.
    """

    def build(self, input_shape: Any) -> None:
        if self.built:
            return
        real = self.y_embedder
        object.__setattr__(self, "y_embedder", _NullBuild(real))
        try:
            super().build(input_shape)
        finally:
            object.__setattr__(self, "y_embedder", real)


class TestTheAntiVacuityHalf:
    """The parity predicate must be able to FAIL. Three directions."""

    def test_a_sub_layer_missing_from_build_is_convicted(self) -> None:
        """Under-building. If this arm ever goes green, parity is decoration."""
        broken = _DiTWithOneSubLayerDroppedFromBuild(**TINY)
        broken.build(model_build_shape(TINY))
        explicit = relative_paths(broken)
        assert not any("y_embedder" in p for p in explicit), (
            "the seeded removal did not take effect; this arm proves nothing. "
            f"explicit paths: {explicit}"
        )

        with pytest.raises(AssertionError, match="y_embedder"):
            assert_build_parity(
                lambda: _DiTWithOneSubLayerDroppedFromBuild(**TINY),
                model_build_shape(TINY),
                list(tiny_inputs(seed=2)),
                label="seeded",
            )

    def test_a_no_sub_layer_control_makes_the_comparison_fail(self) -> None:
        """The guide's literal control: an instance with NOTHING underneath it.

        A model whose whole sub-tree is proxied away builds only its own
        ``pos_embed``, so the two path sets differ by 40 of 41 entries. Without
        this the parity predicate could be comparing, say, only the root's own
        weights and would read green forever.
        """

        class _Hollow(DiT):
            def build(self, input_shape: Any) -> None:
                if self.built:
                    return
                real = {
                    name: getattr(self, name)
                    for name in ("x_embedder", "t_embedder", "y_embedder",
                                 "final_layer")
                }
                for name, layer in real.items():
                    object.__setattr__(self, name, _NullBuild(layer))
                blocks = self.blocks
                object.__setattr__(
                    self, "blocks", [_NullBuild(b) for b in blocks]
                )
                try:
                    super().build(input_shape)
                finally:
                    for name, layer in real.items():
                        object.__setattr__(self, name, layer)
                    object.__setattr__(self, "blocks", blocks)

        hollow = _Hollow(**TINY)
        hollow.build(model_build_shape(TINY))
        assert relative_paths(hollow) == ["pos_embed"], relative_paths(hollow)

        with pytest.raises(AssertionError, match="build\\(\\) does not"):
            assert_build_parity(
                lambda: _Hollow(**TINY),
                model_build_shape(TINY),
                list(tiny_inputs(seed=3)),
                label="hollow",
            )

    def test_a_knob_that_removes_weights_removes_them_from_both_paths(self) -> None:
        """Over-building: parity passes when BOTH paths build everything.

        ``use_bias=False`` must produce a model with strictly fewer weight
        tensors, not the same tree with unused biases sitting in it.
        """
        with_bias = DiT(**dit_config(use_bias=True))
        with_bias.build(model_build_shape(TINY))
        without = DiT(**dit_config(use_bias=False))
        without.build(model_build_shape(TINY))

        assert len(without.weights) < len(with_bias.weights), (
            f"use_bias=False built {len(without.weights)} tensors and "
            f"use_bias=True built {len(with_bias.weights)}; the knob is not "
            "removing anything, so build() is over-building on one of the two "
            "and parity cannot see it"
        )
        # The adaLN modulation Dense keeps its bias unconditionally -- upstream
        # always has it, and it is the zero-init shift/scale/gate source that
        # makes every block an exact identity at initialisation.
        assert any("adaln/linear/bias" in w.path for w in without.weights)


# ---------------------------------------------------------------------
# What the two paths agree ABOUT
# ---------------------------------------------------------------------


class TestTheTwoPathsAgreeOnMoreThanNames:
    """Equal path SETS would still admit two differently-shaped trees."""

    def test_the_shapes_and_trainability_match_path_for_path(self) -> None:
        explicit = DiT(**TINY)
        explicit.build(model_build_shape(TINY))
        lazy = DiT(**TINY)
        lazy(list(tiny_inputs(seed=4)))

        def profile(model: keras.Model) -> dict:
            return {
                w.path.split("/", 1)[-1]: (tuple(w.shape), bool(w.trainable))
                for w in model.weights
            }

        a, b = profile(explicit), profile(lazy)
        assert a == b, {k: (a.get(k), b.get(k)) for k in set(a) ^ set(b)}
        assert len(a) == TINY_WEIGHT_TENSORS
        non_trainable = sorted(k for k, (_, t) in a.items() if not t)
        assert non_trainable == ["pos_embed", "t_embedder/freqs"], non_trainable
