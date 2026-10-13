"""D-014: every Qwen feed-forward is as wide as `moe_intermediate_size` says.

In the Hugging Face Qwen3-MoE and Qwen3-Next configs, `moe_intermediate_size`
is the intermediate (hidden) width of each routed expert's FFN. Before D-014
both models passed `ffn_expansion_factor = max(1, moe_intermediate_size //
hidden_size)` to a SwiGLU expert, which then applied its own 2/3 rule and
rounded to a multiple of 256: a Qwen3Next `80b_a3b` expert was 1536 wide, not
512, `small` 768 (704) and `tiny` 512 (352). And the dense Qwen3Next variant
(`num_experts=1`) built no feed-forward at all: `Qwen3NextBlock` left all four
slots empty when it got no `MoEConfig`.

The arms:

- every MoE variant of both classes builds experts of width
  `moe_intermediate_size` (layers and experts capped, vocabulary shrunk, so the
  512-expert variant fits a CPU; Qwen3 `tiny` has no MoE layer and is checked
  with one forced on);
- the dense Qwen3Next builds a feed-forward of that width at each of the four
  slots, and zeroing it changes the output;
- Qwen3's dense layers do not read `moe_intermediate_size` (their width comes
  from the SwiGLU 4x rule, unchanged by D-014);
- an archive of the old shapes is refused by `from_variant(pretrained=path)`,
  never half-loaded;
- `Qwen3NextBlock`'s new `ffn_config` round-trips through `get_config`, and a
  block given neither config still builds no feed-forward (old block archives).

The old-shape archives are built with today's code at the widths the old code
built (an expert of 256 for `hidden_size=32`; a dense block with no
feed-forward). Their weight paths and shapes were checked equal to archives
built at the pre-D-014 commit 25a55bcf7.
"""

from typing import Any, Dict, List, Set, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.moe import (
    ExpertConfig,
    GatingConfig,
    MixtureOfExperts,
    MoEConfig,
)
from dl_techniques.models.language.qwen import qwen3_next as qwen3_next_module
from dl_techniques.models.language.qwen.components import Qwen3NextBlock
from dl_techniques.models.language.qwen.qwen3 import Qwen3
from dl_techniques.models.language.qwen.qwen3_next import Qwen3Next

from ..test_sam.dead_component_oracle import zeroed_variables

# Qwen3's MoE layers go through TransformerLayer, whose own advisory says that
# `moe_config` supersedes `ffn_type` / `ffn_args` (see test_qwen3.py).
pytestmark = [
    pytest.mark.filterwarnings("ignore:moe_config is provided:UserWarning"),
]

_SMALL = dict(vocab_size=64)
_IDS = np.random.default_rng(0).integers(1, 64, (2, 8)).astype("int32")


def _kernel_shapes(layer: keras.layers.Layer) -> Set[Tuple[int, ...]]:
    """2-D kernel shapes of a layer's weights."""
    return {
        tuple(w.shape) for w in layer.weights
        if w.path.endswith("kernel") and len(w.shape) == 2
    }


def _expert_shapes(model: keras.Model) -> List[Set[Tuple[int, ...]]]:
    """One set of 2-D kernel shapes per expert, over every MoE layer."""
    moes = [l for l in model._flatten_layers() if isinstance(l, MixtureOfExperts)]
    assert moes, "the model built no MixtureOfExperts layer"
    return [_kernel_shapes(e) for moe in moes for e in moe.experts]


def _qwen3_moe_kwargs(variant: str) -> Dict[str, Any]:
    """Shrink a Qwen3 variant to one MoE layer with two experts."""
    return dict(_SMALL, num_layers=1, moe_layers=[0],
                num_experts=2, num_experts_per_tok=1)


def _qwen3_next_kwargs(variant: str) -> Dict[str, Any]:
    """Shrink a Qwen3Next variant to one block (two experts if MoE)."""
    kwargs = dict(_SMALL, num_layers=1)
    if Qwen3Next.MODEL_VARIANTS[variant]["num_experts"] > 1:
        kwargs.update(num_experts=2, num_experts_per_tok=1)
    return kwargs


class TestTheExpertWidthIsMoeIntermediateSize:

    @pytest.mark.parametrize("variant", sorted(Qwen3.MODEL_VARIANTS))
    def test_qwen3(self, variant):
        cfg = Qwen3.MODEL_VARIANTS[variant]
        model = Qwen3.from_variant(variant, **_qwen3_moe_kwargs(variant))
        model.build((None, None))
        hidden, width = cfg["hidden_size"], cfg["moe_intermediate_size"]
        for shapes in _expert_shapes(model):
            assert shapes == {(hidden, width), (width, hidden)}, (
                f"Qwen3 {variant}: expert kernels {sorted(shapes)}, expected "
                f"hidden width moe_intermediate_size={width}"
            )

    @pytest.mark.parametrize(
        "variant",
        sorted(v for v, c in Qwen3Next.MODEL_VARIANTS.items() if c["num_experts"] > 1),
    )
    def test_qwen3_next(self, variant):
        cfg = Qwen3Next.MODEL_VARIANTS[variant]
        model = Qwen3Next.from_variant(variant, **_qwen3_next_kwargs(variant))
        model.build((None, None))
        hidden, width = cfg["hidden_size"], cfg["moe_intermediate_size"]
        experts = _expert_shapes(model)
        assert len(experts) == 4 * 2  # 4 MoE slots, 2 experts each
        for shapes in experts:
            assert shapes == {(hidden, width), (width, hidden)}, (
                f"Qwen3Next {variant}: expert kernels {sorted(shapes)}, expected "
                f"hidden width moe_intermediate_size={width}"
            )


def _dense_ffns(block: Qwen3NextBlock) -> List[keras.layers.Layer]:
    return [l for l in (*block.delta_ffn_layers, block.attention_ffn) if l is not None]


class TestTheDenseQwen3NextHasAFeedForward:

    def test_every_variant_without_experts_builds_four_ffns_of_the_configured_width(self):
        dense = [v for v, c in Qwen3Next.MODEL_VARIANTS.items() if c["num_experts"] == 1]
        assert dense == ["80b"], dense
        cfg = Qwen3Next.MODEL_VARIANTS["80b"]
        model = Qwen3Next.from_variant("80b", **_qwen3_next_kwargs("80b"))
        model.build((None, None))
        hidden, width = cfg["hidden_size"], cfg["moe_intermediate_size"]
        ffns = _dense_ffns(model.blocks[0])
        assert len(ffns) == 4, f"{len(ffns)} dense FFNs in the block, expected 4"
        for ffn in ffns:
            assert _kernel_shapes(ffn) == {(hidden, width), (width, hidden)}

    def test_zeroing_each_ffn_changes_the_output(self):
        """Each of the four slots is on the forward path, not just built."""
        keras.utils.set_random_seed(3)
        model = Qwen3Next(vocab_size=64, hidden_size=32, num_layers=1,
                          num_attention_heads=2, num_key_value_heads=1,
                          max_seq_len=32, num_experts=1, num_experts_per_tok=1,
                          moe_intermediate_size=48)
        before = keras.ops.convert_to_numpy(model(_IDS))
        ffns = _dense_ffns(model.blocks[0])
        assert len(ffns) == 4
        for ffn in ffns:
            with zeroed_variables(ffn.weights):
                after = keras.ops.convert_to_numpy(model(_IDS))
            assert np.max(np.abs(before - after)) > 1e-4, (
                f"zeroing {ffn.name} left the output unchanged")


class TestQwen3DenseLayersDoNotReadTheExpertWidth:

    def test_the_dense_ffn_is_the_same_for_any_moe_intermediate_size(self):
        def signature(moe_intermediate_size: int):
            model = Qwen3.from_variant("tiny", vocab_size=64, num_layers=1,
                                       moe_intermediate_size=moe_intermediate_size)
            model.build((None, None))
            # Paths without the model's own name, which differs per instance.
            return sorted((w.path.split("/", 1)[1], tuple(w.shape)) for w in model.weights)
        assert signature(2048) == signature(64)


_Q3N = dict(vocab_size=64, hidden_size=32, num_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_seq_len=32)
_Q3 = dict(vocab_size=64, hidden_size=32, num_layers=2, num_attention_heads=2,
           num_key_value_heads=1, max_seq_len=32, moe_layers=[1])

#: The expert width the pre-D-014 code built for hidden_size=32 and any
#: moe_intermediate_size below 64: factor 1, 2/3 rule, rounded up to 256.
_OLD_EXPERT_WIDTH = 256


class TestAnOldShapeArchiveIsRefused:

    def _save(self, model: keras.Model, tmp_path, name: str) -> str:
        model.build((None, None))
        path = str(tmp_path / f"{name}.keras")
        model.save(path)
        return path

    def test_a_dense_qwen3_next_archive_without_ffn(self, tmp_path, monkeypatch):
        real_block = qwen3_next_module.Qwen3NextBlock

        def old_block(**kwargs):
            kwargs.pop("ffn_config", None)
            return real_block(**kwargs)

        monkeypatch.setattr(qwen3_next_module, "Qwen3NextBlock", old_block)
        old = Qwen3Next(**_Q3N, num_experts=1, num_experts_per_tok=1,
                        moe_intermediate_size=48)
        path = self._save(old, tmp_path, "q3n_dense_old")
        monkeypatch.setattr(qwen3_next_module, "Qwen3NextBlock", real_block)
        with pytest.raises(ValueError):
            Qwen3Next.from_variant("80b", pretrained=path, **_Q3N,
                                   moe_intermediate_size=48)

    def test_a_qwen3_next_archive_with_old_expert_width(self, tmp_path):
        old = Qwen3Next(**_Q3N, num_experts=2, num_experts_per_tok=1,
                        moe_intermediate_size=_OLD_EXPERT_WIDTH)
        path = self._save(old, tmp_path, "q3n_moe_old")
        with pytest.raises(ValueError):
            Qwen3Next.from_variant("tiny", pretrained=path, **_Q3N,
                                   num_experts=2, num_experts_per_tok=1,
                                   moe_intermediate_size=48)

    def test_a_qwen3_archive_with_old_expert_width(self, tmp_path):
        old = Qwen3(**_Q3, num_experts=2, num_experts_per_tok=1,
                    moe_intermediate_size=_OLD_EXPERT_WIDTH)
        path = self._save(old, tmp_path, "q3_moe_old")
        with pytest.raises(ValueError):
            Qwen3.from_variant("small", pretrained=path, **_Q3,
                               num_experts=2, num_experts_per_tok=1,
                               moe_intermediate_size=48)

    def test_the_new_shapes_load(self, tmp_path):
        """Control: the same route loads an archive of the current shapes."""
        new = Qwen3Next(**_Q3N, num_experts=1, num_experts_per_tok=1,
                        moe_intermediate_size=48)
        path = self._save(new, tmp_path, "q3n_dense_new")
        loaded = Qwen3Next.from_variant("80b", pretrained=path, **_Q3N,
                                        moe_intermediate_size=48)
        for a, b in zip(new.weights, loaded.weights):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))


_FFN = {"type": "swiglu", "output_dim": 32, "hidden_dim": 48}


class TestTheBlockFfnConfig:

    def test_get_config_round_trips_the_ffn_config(self):
        block = Qwen3NextBlock(dim=32, num_heads=2, max_seq_len=32, ffn_config=_FFN)
        config = block.get_config()
        assert config["ffn_config"] == _FFN
        rebuilt = Qwen3NextBlock.from_config(config)
        for b in (block, rebuilt):
            b.build((None, 8, 32))
        assert sorted((w.path, tuple(w.shape)) for w in block.weights) == \
            sorted((w.path, tuple(w.shape)) for w in rebuilt.weights)
        assert len(_dense_ffns(rebuilt)) == 4

    def test_a_dense_model_round_trips_through_save(self, tmp_path):
        keras.utils.set_random_seed(5)
        model = Qwen3Next(**_Q3N, num_experts=1, num_experts_per_tok=1,
                          moe_intermediate_size=48)
        expected = keras.ops.convert_to_numpy(model(_IDS))
        path = str(tmp_path / "dense.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert len(_dense_ffns(loaded.blocks[0])) == 4
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded(_IDS)), expected, rtol=0, atol=1e-6)

    def test_a_block_given_neither_config_builds_no_ffn(self):
        block = Qwen3NextBlock(dim=32, num_heads=2, max_seq_len=32)
        block.build((None, 8, 32))
        assert _dense_ffns(block) == []
        assert block.get_config()["ffn_config"] is None

    def test_both_configs_are_refused(self):
        moe = MoEConfig(num_experts=2, expert_config=ExpertConfig(ffn_config=dict(_FFN)),
                        gating_config=GatingConfig(top_k=1))
        with pytest.raises(ValueError, match="not both"):
            Qwen3NextBlock(dim=32, num_heads=2, max_seq_len=32,
                           moe_config=moe, ffn_config=_FFN)
