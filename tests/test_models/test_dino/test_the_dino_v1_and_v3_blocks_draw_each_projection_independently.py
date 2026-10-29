"""DINOv1 and DINOv3 give each block's attention output projection its own random draw.

Both models build ``TransformerLayer`` blocks with ``multi_head`` attention, and ``MultiHeadAttention``
hands ONE initializer instance to its ``qkv`` and ``proj`` Dense layers unless told otherwise: the smaller
draw (``proj``) was a rescaled prefix of the larger (``qkv``), one replayed kernel pair per block (2 of 66
kernel pairs of a two-block model of width 64, measured at HEAD 3bd2c0c59 for both, and for DINOv3 also
with an instance ``kernel_initializer``). ``TransformerLayer`` keeps that default on purpose (the GPT-2
residual rule of decision D-160, which every language model builds on), so the two vision models pass
``output_kernel_initializer`` in their ``attention_args`` the way ``DINOv2Block`` does (D-054, D-056).

Pinned here:

* the replay oracle is empty on both models (and on DINOv3 with a seedless instance as its
  ``kernel_initializer``), with the compared pair count pinned;
* each block's ``proj`` is a new instance of the class ``qkv`` draws from;
* the keyword is passed for ``multi_head`` only: DINOv3's rope path (``group_query``, which clones per
  projection itself) and DINOv1's other attention types receive none;
* the seeded draws: ``qkv`` keeps the first default seed of its block and ``proj`` takes the next one
  (carve-out C-8), and every weight keeps its path.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.dino.dino_v1 import DINOv1
from dl_techniques.models.vision.dino.dino_v3 import DINOv3
from tests.test_models.initializer_replay_oracle import replayed_pairs

DEPTH = 2
KWARGS = dict(image_size=32, patch_size=16, embed_dim=64, depth=DEPTH, num_heads=4)


def _images() -> np.ndarray:
    return np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32")


def _v1(**changes) -> DINOv1:
    model = DINOv1(**{**KWARGS, **changes})
    model(_images(), training=False)
    return model


def _v3(**changes) -> DINOv3:
    model = DINOv3(**{**KWARGS, **changes})
    model(_images(), training=False)
    return model


def _blocks(model):
    """DINOv1 keeps its blocks in ``transformer_blocks``, DINOv3 in ``encoder_layers``."""
    return model.transformer_blocks if isinstance(model, DINOv1) else model.encoder_layers


def _engine(block):
    """The ``MultiHeadCrossAttention`` that owns the Dense projections of a ``TransformerLayer`` block."""
    return block.attention.cross_attention


class TestNoDraw:

    def test_dino_v1_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        pairs, compared = replayed_pairs(_v1())
        assert compared == 66
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    def test_dino_v3_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        pairs, compared = replayed_pairs(_v3())
        assert compared == 66
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    def test_dino_v3_with_a_seedless_instance_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        pairs, compared = replayed_pairs(_v3(kernel_initializer=keras.initializers.HeNormal()))
        assert compared == 66
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    def test_dino_v3_on_its_rope_path_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        pairs, compared = replayed_pairs(_v3(positional_embedding_type="rope"))
        assert compared > 0 and not pairs

    def test_the_blocks_of_one_model_do_not_repeat_each_others_projection(self) -> None:
        keras.utils.set_random_seed(1234)
        for model in (_v1(depth=4), _v3(depth=4)):
            kernels = [np.asarray(_engine(block).proj_dense.kernel).ravel() for block in _blocks(model)]
            for i, a in enumerate(kernels):
                for j in range(i + 1, len(kernels)):
                    r = float(np.corrcoef(a[:256], kernels[j][:256])[0, 1])
                    assert abs(r) < 0.5, f"{type(model).__name__}: blocks {i} and {j} share a draw (r={r:.3f})"


class TestTheOutputProjectionHasItsOwnInitializer:

    @pytest.mark.parametrize("make", [_v1, _v3], ids=["dino_v1", "dino_v3"])
    def test_proj_is_a_new_instance_of_the_class_qkv_has(self, make) -> None:
        for block in _blocks(make()):
            engine = _engine(block)
            qkv, proj = engine.qkv_dense.kernel_initializer, engine.proj_dense.kernel_initializer
            assert proj is not qkv
            assert type(proj) is type(qkv)

    def test_dino_v3_proj_follows_the_models_initializer_class(self) -> None:
        for block in _blocks(_v3(kernel_initializer=keras.initializers.HeNormal())):
            engine = _engine(block)
            assert type(engine.proj_dense.kernel_initializer) is keras.initializers.HeNormal
            assert type(engine.qkv_dense.kernel_initializer) is keras.initializers.HeNormal

    def test_the_keyword_is_passed_for_multi_head_only(self) -> None:
        assert all("output_kernel_initializer" in block.attention_args for block in _blocks(_v1()))
        assert all("output_kernel_initializer" in block.attention_args for block in _blocks(_v3()))
        rope = _v3(positional_embedding_type="rope")
        assert all("output_kernel_initializer" not in block.attention_args for block in _blocks(rope))

    def test_dino_v1_with_another_attention_type_gets_no_keyword(self) -> None:
        model = DINOv1(**KWARGS, attention_type="anchor")
        assert all("output_kernel_initializer" not in block.attention_args for block in _blocks(model))


class TestTheSeededDraws:
    """Carve-out C-8: ``qkv`` keeps its default seed and ``proj`` takes the next one."""

    @staticmethod
    def _first_block_seeds(factory):
        keras.utils.set_random_seed(7)
        model = factory()
        engine = _engine(_blocks(model)[0])
        return engine.qkv_dense.kernel_initializer.seed, engine.proj_dense.kernel_initializer.seed

    @pytest.mark.parametrize("factory", [lambda: _v1(), lambda: _v3()], ids=["dino_v1", "dino_v3"])
    def test_the_two_projections_take_different_seeds(self, factory) -> None:
        qkv_seed, proj_seed = self._first_block_seeds(factory)
        assert qkv_seed != proj_seed

    @pytest.mark.parametrize("make", [_v1, _v3], ids=["dino_v1", "dino_v3"])
    def test_no_weight_name_changed(self, make) -> None:
        paths = [w.path for w in make().weights]
        for tail in ("attention/cross_attention/qkv/kernel", "attention/cross_attention/proj/kernel"):
            assert sum(p.endswith(tail) for p in paths) == DEPTH, tail
        assert not any("output_kernel" in p for p in paths)


class TestTheRuntimeContractReplaysBothModels:
    """The shared contract's replay rows are what fails when a block shares a draw again."""

    def test_a_row_exists_for_each_model_and_dino_v3_has_both_arms(self) -> None:
        from tests.test_models.test_the_vision_classifiers_share_one_runtime_contract import (
            REPLAY_SUBJECTS,
        )
        assert REPLAY_SUBJECTS["dino_v1"][1] is False, "DINOv1 has no initializer knob to arm"
        assert REPLAY_SUBJECTS["dino_v3"][1] is True, "DINOv3 takes kernel_initializer: both arms"

    def test_each_row_builds_the_model_it_is_named_for(self) -> None:
        from tests.test_models.test_the_vision_classifiers_share_one_runtime_contract import (
            REPLAY_SUBJECTS,
        )
        assert isinstance(REPLAY_SUBJECTS["dino_v1"][0](None), DINOv1)
        assert isinstance(REPLAY_SUBJECTS["dino_v3"][0](keras.initializers.HeNormal()), DINOv3)
