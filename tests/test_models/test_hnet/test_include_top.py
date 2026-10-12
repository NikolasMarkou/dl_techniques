"""H-Net owns one task head when ``include_top=True``.

Headless, :class:`HNet` still returns the logits tensor (tied or untied) and
contributes its ratio loss through ``add_loss``. With a ``task_config`` it
returns the finalized output of a ``layers/heads/nlp`` head named ``task_head``,
fed ``hidden`` (the backbone's output, one vector per INPUT position) and a
padding mask (plan-2026-10-11T170958-dc4d2202 D-005). The logits are not
computed, an untied ``lm_head`` is not built, and the ratio loss is still
contributed, exactly once. The claims shared with Zamba2 live in
``headed_logit_decoder_cases``; the ones below are H-Net's own: the loss, the
tied and untied layouts and their load rule, the mask, and the stage-0 length
guard under a head.
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hnet import model as hnet_module
from dl_techniques.models.language.hnet.model import HNet

from ..headed_logit_decoder_cases import (
    CLF,
    NER,
    HeadedLogitDecoderCases,
    assert_same_backbone,
    head_batch,
    head_inputs,
    randomized,
    relative_paths,
    to_np,
)
from ..tiny_language_models import create_tiny_hnet, tiny_hnet_arch

#: Tied, so a headless model's weights are exactly the backbone's (an untied
#: headless model also holds its `lm_head`, which a headed one does not).
TIED = dict(max_seq_len=16, headdim=8, tie_word_embeddings=True)
UNTIED = dict(TIED, tie_word_embeddings=False)
OUTER_ATTENTION = ["T1", ["T1"], "T1"]


def tiny(layout=None, **kw: Any) -> HNet:
    return HNet(arch_config=tiny_hnet_arch(layout=layout), **dict(TIED, **kw))


class TestHNetIncludeTop(HeadedLogitDecoderCases):
    CLS = HNet
    MODULE = hnet_module
    #: embeddings, the backbone's stages (tied head: no `lm_head`).
    N_BACKBONE = 37
    MISFIT = dict(headdim=16)

    def model(self, **kw: Any) -> keras.Model:
        return tiny(**kw)

    def factory(self, **kw: Any) -> keras.Model:
        return create_tiny_hnet(**dict(TIED, **kw))

    def logits_from_hidden(self, twin: keras.Model, hidden: np.ndarray):
        return np.einsum("bld,vd->blv", hidden, to_np(twin.embeddings.embeddings))

    # -- the ratio loss --------------------------------------------------

    @pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
    @pytest.mark.parametrize("include_top", [False, True], ids=["headless", "headed"])
    def test_the_ratio_loss_is_contributed_exactly_once(self, tied, include_top):
        kw = dict(tie_word_embeddings=tied)
        if include_top:
            kw.update(include_top=True, task_config=CLF)
        model = tiny(**kw)
        model(head_batch(), training=True)
        assert len(model.losses) == 1
        model(head_batch(), training=True)
        assert len(model.losses) == 1, "losses accumulate across calls"

    def test_the_head_does_not_change_the_ratio_loss(self):
        """The loss is the backbone's: the same weights give the same value
        headless and headed, under a mask that removes positions."""
        headed = tiny(include_top=True, task_config=CLF)
        batch = head_batch()
        headed(batch, training=True)
        twin = self.headless_twin(headed, batch)
        twin(batch, training=True)
        assert float(headed.losses[0]) == float(twin.losses[0]) > 0.0

    def test_the_ratio_loss_reaches_a_fit(self):
        """`fit` sums `model.losses` into the trained scalar under a head."""
        model = tiny(include_top=True, task_config=CLF, ratio_loss_alpha=0.0)
        zero = tiny(include_top=True, task_config=CLF, ratio_loss_alpha=5.0)
        batch = head_batch()
        model(batch, training=True)
        zero(batch, training=True)
        assert float(model.losses[0]) == 0.0 < float(zero.losses[0])

    # -- tied and untied -------------------------------------------------

    @pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
    def test_no_lm_head_exists_under_a_head(self, tied):
        model = tiny(tie_word_embeddings=tied, include_top=True, task_config=CLF)
        model(head_batch(), training=False)
        assert model.lm_head is None
        assert not [w.path for w in model.weights if "lm_head" in w.path]
        # the knob itself is kept, so the config says what the model was asked for
        assert model.get_config()["tie_word_embeddings"] is tied
        rebuilt = HNet.from_config(model.get_config())
        assert rebuilt.tie_word_embeddings is tied and rebuilt.lm_head is None

    def test_an_untied_headless_model_keeps_its_lm_head(self):
        """Anti-vacuity for the arm above: the headless untied model has one."""
        model = tiny(tie_word_embeddings=False)
        model(head_batch(), training=False)
        assert model.lm_head is not None
        assert [w.path for w in model.weights if "lm_head" in w.path]

    def test_the_untied_backbone_paths_are_the_headless_ones_minus_lm_head(self):
        def paths(**kw):
            model = tiny(tie_word_embeddings=False, **kw)
            model(head_batch(), training=False)
            return relative_paths(model)

        headless = paths()
        headed = paths(include_top=True, task_config=CLF)
        assert [p for p in headed if not p.startswith("task_head/")] == [
            p for p in headless if not p.startswith("lm_head/")]

    def test_the_untied_head_reads_hidden_not_the_projection(self, monkeypatch):
        """Untied: the headless logits are `lm_head(hidden)` of the very
        hidden state the head receives."""
        model = tiny(tie_word_embeddings=False, include_top=True, task_config=CLF)
        batch = head_batch()
        twin = self.headless_twin(model, batch)
        seen = head_inputs(model, batch)
        np.testing.assert_allclose(
            to_np(twin.lm_head(seen["hidden_states"])),
            to_np(twin(batch, training=False)), atol=1e-5, rtol=0)

    @pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
    def test_a_headless_checkpoint_fills_a_headed_model(self, tied, tmp_path):
        """An untied `lm_head` in the checkpoint has no home in a headed model,
        and that is allowed: the target carries a task head instead."""
        source = create_tiny_hnet(**dict(TIED, tie_word_embeddings=tied))
        source(head_batch(), training=False)
        randomized(source, 1)
        path = str(tmp_path / "headless.keras")
        source.save(path)
        for target_tied in (True, False):
            headed = create_tiny_hnet(
                pretrained=path, include_top=True, task_config=CLF,
                **dict(TIED, tie_word_embeddings=target_tied))
            assert_same_backbone(source, headed)

    @pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
    def test_a_headed_checkpoint_fills_a_headless_model_of_either_layout(
            self, tied, tmp_path):
        """The reverse direction: an untied headless target keeps a FRESH
        `lm_head` (nothing in a headed checkpoint trained one)."""
        source = create_tiny_hnet(include_top=True, task_config=CLF, **TIED)
        source(head_batch(), training=False)
        randomized(source, 2)
        path = str(tmp_path / "headed.keras")
        source.save(path)
        target = create_tiny_hnet(
            pretrained=path, **dict(TIED, tie_word_embeddings=tied))
        assert_same_backbone(source, target)
        assert (target.lm_head is not None) is (not tied)

    def test_an_untied_headless_checkpoint_does_not_fill_a_tied_model(self, tmp_path):
        source = create_tiny_hnet(**UNTIED)
        source(head_batch(), training=False)
        path = str(tmp_path / "untied.keras")
        source.save(path)
        with pytest.raises(ValueError, match=r"no target layer: \['lm_head'\]"):
            create_tiny_hnet(pretrained=path, **TIED)

    def test_a_tied_headless_checkpoint_does_not_fill_an_untied_model(self, tmp_path):
        source = create_tiny_hnet(**TIED)
        source(head_batch(), training=False)
        path = str(tmp_path / "tied.keras")
        source.save(path)
        with pytest.raises(ValueError, match=r"unloaded: \['lm_head'\]"):
            create_tiny_hnet(pretrained=path, **UNTIED)

    def test_a_vocabulary_change_is_read_from_the_architecture(self, tmp_path):
        """H-Net records its vocabulary inside `arch_config`: a checkpoint of
        another vocabulary keeps only the token table (and an untied output
        kernel) fresh, and loads everything else."""
        source = create_tiny_hnet(**UNTIED)
        source(head_batch(), training=False)
        randomized(source, 3)
        path = str(tmp_path / "v64.keras")
        source.save(path)
        target = create_tiny_hnet(pretrained=path, vocab_size=96, **UNTIED)
        assert tuple(target.embeddings.embeddings.shape) == (96, 16)
        np.testing.assert_array_equal(
            to_np(target.get_layer("backbone").weights[0]),
            to_np(source.get_layer("backbone").weights[0]))

    # -- the mask --------------------------------------------------------

    def test_the_padding_mask_argument_and_the_dict_mask_are_one_thing(self):
        model = tiny(include_top=True, task_config=CLF)
        batch = head_batch()
        by_dict = to_np(model(batch, training=False))
        by_argument = to_np(model(batch["input_ids"],
                                  padding_mask=batch["attention_mask"],
                                  training=False))
        np.testing.assert_array_equal(by_argument, by_dict)
        # the argument wins over a dict's own mask
        wrong = dict(batch, attention_mask=np.ones_like(batch["attention_mask"]))
        np.testing.assert_array_equal(
            to_np(model(wrong, padding_mask=batch["attention_mask"],
                        training=False)), by_dict)

    def test_the_mask_is_the_backbones_and_the_heads(self):
        """A mask that drops row 1's tail changes what the head pools:
        the same ids without it give another answer for that row only."""
        model = tiny(include_top=True, task_config=CLF)
        batch = head_batch()
        model(batch, training=False)
        randomized(model, 6)  # initial weights make the head's answer nearly constant
        masked = to_np(model(batch, training=False))
        unmasked = to_np(model(batch["input_ids"], training=False))
        np.testing.assert_allclose(masked[0], unmasked[0], atol=1e-5, rtol=0)
        assert not np.allclose(masked[1], unmasked[1], atol=1e-5, rtol=0)

    def test_a_derived_pad_mask_reaches_the_head_alone(self):
        """`pad_token_id` builds the HEAD's mask; the backbone gets only an
        explicit mask, so its chunking (and ratio loss) is unaffected."""
        batch = head_batch()
        plain = tiny(include_top=True, task_config=CLF)
        derived = tiny(include_top=True, task_config=CLF, pad_token_id=0)
        derived(batch["input_ids"], training=True)
        plain(batch["input_ids"], training=True)
        derived.set_weights(plain.get_weights())
        derived(batch["input_ids"], training=True)
        plain(batch["input_ids"], training=True)
        assert float(derived.losses[0]) == float(plain.losses[0])
        np.testing.assert_array_equal(
            head_inputs(derived, batch["input_ids"])["hidden_states"],
            head_inputs(plain, batch["input_ids"])["hidden_states"])
        assert not np.allclose(
            head_inputs(derived, batch["input_ids"])["attention_mask"],
            head_inputs(plain, batch["input_ids"])["attention_mask"])

    # -- the length guard ------------------------------------------------

    @pytest.mark.parametrize("include_top", [False, True], ids=["headless", "headed"])
    def test_a_stage_0_attention_limit_is_refused_cleanly_in_either_top(
            self, include_top):
        kw = dict(include_top=True, task_config=CLF) if include_top else {}
        model = tiny(layout=OUTER_ATTENTION, **kw)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.build((2, 17))
        assert not model.weights
        legal = np.random.default_rng(6).integers(1, 64, (2, 12)).astype("int32")
        out = model(legal, training=False)
        assert tuple(out.shape) == ((2, 3) if include_top else (2, 12, 64))

    @pytest.mark.parametrize("include_top", [False, True], ids=["headless", "headed"])
    def test_inner_only_attention_has_no_input_length_limit_in_either_top(
            self, include_top):
        kw = dict(include_top=True, task_config=CLF) if include_top else {}
        ids = np.random.default_rng(7).integers(1, 64, (2, 40)).astype("int32")
        out = tiny(**kw)(ids, training=False)
        assert tuple(out.shape) == ((2, 3) if include_top else (2, 40, 64))
