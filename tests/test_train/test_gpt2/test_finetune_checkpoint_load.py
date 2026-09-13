"""``finetune.py`` must actually load the checkpoints ``pretrain.py`` saves.

Both existing gpt2-finetune test modules stub `load_pretrained_model` (see
`test_run_artifacts.py`'s module docstring and `test_cli_contract.py`'s), so
neither ever executed the REAL `keras.models.load_model` /
`load_pretrained_model` / `compile_model` / `model.fit` path -- exactly the
gap `findings/finetune-hnet-loss-deeper-validation.md` Finding #1 identifies.

MEASURED pre-fix failure (plan-2026-09-13T073704-245ab5d5, iter-1, step-1):
building a tiny ``CausalLanguageModel``-wrapped GPT2, saving it, then running
the OLD ``finetune.py``'s real ``load_pretrained_model``/``compile_model``
against the reload, then calling ``model.fit()`` on a dataset shaped like
``load_finetune_datasets``'s old ``(x, {"logits": y})`` output raised::

    TypeError: Expected any non-tensor type, but got a tensor instead.

-- because ``CausalLanguageModel``'s ``pre_shifted=True`` branch treats the
unpacked ``y`` AS the label tensor with no further unwrapping, so a
dict-keyed ``{"logits": y}`` label broke inside ``compute_loss``. This module
exercises the REAL path end-to-end for both checkpoint shapes:

- a CURRENT checkpoint, saved by the post-migration ``pretrain.py`` as a
  ``CausalLanguageModel`` wrapper;
- a LEGACY checkpoint, saved before that migration as a bare ``GPT2`` (per
  the Pre-Mortem signal in plan.md: a fix that repairs the current path must
  not silently break the legacy one).

Both fixtures use a `"tiny"` GPT2 variant with a tiny vocab/seq length so the
whole module runs in seconds, not minutes.

# DECISION plan-2026-09-13T120637-288bad33/D-014: this module is the real
# save/load/compile/fit round trip, permanently complementary to
# test_run_artifacts.py, not redundant with it -- do not consolidate the two.
# (2026-09-13) It deliberately does NOT check `config.json`/
# `training_history.json` content. For that persistence/control-flow guard
# (which deliberately stubs the loader instead), see `test_run_artifacts.py`.
# See decisions.md D-014 of the above plan for the full disjoint-coverage
# analysis.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.language.gpt2 import GPT2
from dl_techniques.models.language.masked_language_model.clm import CausalLanguageModel
from dl_techniques.losses import MaskedCausalLMLoss

from train.gpt2 import finetune as ft

VOCAB_SIZE = 300
MAX_SEQ_LEN = 16


def _build_tiny_backbone() -> GPT2:
    return GPT2.from_variant("tiny", vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)


def _fit_one_step(model: CausalLanguageModel) -> keras.callbacks.History:
    """Real (x, y) pairs, matching `preprocess_clm_dataset`'s pre-shifted contract."""
    x = np.random.randint(0, VOCAB_SIZE, size=(4, MAX_SEQ_LEN - 1)).astype("int32")
    y = np.random.randint(0, VOCAB_SIZE, size=(4, MAX_SEQ_LEN - 1)).astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    return model.fit(ds, epochs=1, verbose=0)


class TestCurrentCheckpointLoadsAndFinetunes:
    """A `CausalLanguageModel`-wrapped checkpoint -- current `pretrain.py`'s shape."""

    @pytest.fixture
    def checkpoint_path(self, tmp_path):
        backbone = _build_tiny_backbone()
        model = CausalLanguageModel(
            backbone=backbone,
            vocab_size=VOCAB_SIZE,
            skip_head=True,
            output_key="logits",
            pre_shifted=True,
            loss_fn=MaskedCausalLMLoss(),
        )
        dummy = np.random.randint(
            0, VOCAB_SIZE, size=(1, MAX_SEQ_LEN - 1)
        ).astype("int32")
        model(dummy, training=False)
        path = str(tmp_path / "current_clm_wrapped.keras")
        model.save(path)
        return path

    def test_loads_as_causal_language_model(self, checkpoint_path):
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path, max_seq_length=MAX_SEQ_LEN,
        )
        loaded = ft.load_pretrained_model(config)
        assert isinstance(loaded, CausalLanguageModel)

    def test_compiles_and_fits_one_step_with_finite_loss(self, checkpoint_path):
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            batch_size=2,
        )
        loaded = ft.load_pretrained_model(config)
        ft.compile_model(loaded, config, steps_per_epoch=10)
        history = _fit_one_step(loaded)
        loss = history.history["loss"][-1]
        assert np.isfinite(loss)

    def test_loss_type_override_selects_a_different_loss_family(self, checkpoint_path):
        """Fine-tuning may select a loss family different from pretrain-time."""
        from dl_techniques.losses import FocalCausalLMLoss

        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            loss_type="focal",
            focal_gamma=2.0,
        )
        loaded = ft.load_pretrained_model(config)
        assert isinstance(loaded.loss_fn, FocalCausalLMLoss)


class TestLegacyCheckpointLoadsAndFinetunes:
    """A bare-`GPT2` checkpoint -- pre-`CausalLanguageModel`-migration shape.

    Pre-Mortem signal #1 (plan.md): a fix that repairs the CURRENT-checkpoint
    path must not silently turn the LEGACY path into the new break.
    """

    @pytest.fixture
    def checkpoint_path(self, tmp_path):
        backbone = _build_tiny_backbone()
        dummy = np.random.randint(
            0, VOCAB_SIZE, size=(1, MAX_SEQ_LEN)
        ).astype("int32")
        backbone(dummy, training=False)
        path = str(tmp_path / "legacy_bare_gpt2.keras")
        backbone.save(path)
        return path

    def test_loads_and_wraps_as_causal_language_model(self, checkpoint_path):
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path, max_seq_length=MAX_SEQ_LEN,
        )
        loaded = ft.load_pretrained_model(config)
        assert isinstance(loaded, CausalLanguageModel)
        assert isinstance(loaded.backbone, GPT2)

    def test_compiles_and_fits_one_step_with_finite_loss(self, checkpoint_path):
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            batch_size=2,
        )
        loaded = ft.load_pretrained_model(config)
        ft.compile_model(loaded, config, steps_per_epoch=10)
        history = _fit_one_step(loaded)
        loss = history.history["loss"][-1]
        assert np.isfinite(loss)

    def test_freeze_embeddings_actually_freezes_the_backbone(self, checkpoint_path):
        """The freeze walk must recurse into the backbone's real sublayers.

        Neither `model.layers` (`[backbone]` on a `CausalLanguageModel`
        wrapper) nor `model.backbone.layers` (`[decoder]` -- GPT2's own
        top-level `.layers` is a single `TextDecoder` sublayer bundling the
        embeddings and block stack) ever contains a layer named
        "embedding": only a recursive walk
        (`backbone._flatten_layers(recursive=True)`) reaches
        `word_embeddings`/`positional_embeddings`. This would silently no-op
        `--freeze-embeddings` if the walk were shallow.
        """
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            freeze_embeddings=True,
        )
        loaded = ft.load_pretrained_model(config)
        sublayers = list(
            loaded.backbone._flatten_layers(include_self=False, recursive=True)
        )
        embedding_layers = [
            layer for layer in sublayers if "embedding" in layer.name.lower()
        ]
        assert embedding_layers, "fixture backbone has no embedding-named layer to probe"
        assert all(not layer.trainable for layer in embedding_layers)

    def test_freeze_n_layers_actually_freezes_decoder_blocks(self, checkpoint_path):
        """GPT2's own transformer blocks are named `decoder_layer_<i>`, not
        "transformer"/"block" -- MEASURED, not assumed from the substring
        check's own vocabulary. `--freeze-n-layers` must still catch them.
        """
        config = ft.FinetuneConfig(
            pretrained_path=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            freeze_n_layers=2,
        )
        loaded = ft.load_pretrained_model(config)
        sublayers = list(
            loaded.backbone._flatten_layers(include_self=False, recursive=True)
        )
        decoder_blocks = [
            layer for layer in sublayers if "decoder_layer" in layer.name.lower()
        ]
        assert decoder_blocks, "fixture backbone has no decoder_layer_* block to probe"
        frozen = [layer for layer in decoder_blocks if not layer.trainable]
        assert len(frozen) == 2
