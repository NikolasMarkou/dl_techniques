"""Execute the README examples that step 29 corrected.

DECISION plan-2026-08-17T183311-79c63e38/D-044

Six package READMEs made claims their code refuses to honour: `distilbert`'s FAQ was
exactly inverted about `pretrained`, `bias_free_denoisers`/`modern_bert` shipped
runnable `pretrained=True` examples that always raise, `masked_autoencoder`'s every
constructor call passed `encoder_dims=`/`encoder_output_shape=` (parameters the
signature does not have) around a /4 encoder the scale contract now rejects, and
`mobilenet_v4` advertised eight `conv_*` variant names that do not exist.

This module RUNS the corrected forms. A prose claim is only testable by running it:
every one of these defects was invisible to grep and visible on the first execution.
Do NOT weaken these to construction-only or shape-only assertions -- the
`pretrained=` route in particular is a VALUE claim (weights actually arrive), and a
shape assertion passes on a model that silently kept its random initialization.

Deliberately small: `"tiny"`-class variants everywhere. Two measured reasons, not
style. `ModernBERT`-base costs 20.1 GB of host RSS to CONSTRUCT (large: 24.2 GB)
because its local layers pad to a single `window_size**2 = 16384` window, and a
forward pass materializes a 16384x16384 score matrix per head. See decisions.md
D-027 for that degeneracy and D-044 for this module.
"""

import os

import keras
import numpy as np
import pytest


# ---------------------------------------------------------------------
# distilbert / bias_free_denoisers / modern_bert: the `pretrained` contract
# ---------------------------------------------------------------------

class TestPretrainedContractAsDocumented:
    """`pretrained=True` raises; `pretrained="<path>"` works. The distilbert FAQ
    asserted the exact opposite of both halves until 2026-08-18."""

    def test_bfunet_true_raises_and_path_restores_by_value(self, tmp_path):
        from dl_techniques.models.bias_free_denoisers.bfunet import (
            create_bfunet_variant,
        )

        model = create_bfunet_variant("tiny", input_shape=(64, 64, 3))
        path = os.path.join(str(tmp_path), "bfunet_tiny.keras")
        model.save(path)

        restored = create_bfunet_variant(
            "tiny", input_shape=(64, 64, 3), pretrained=path
        )
        x = np.random.default_rng(0).random((1, 64, 64, 3)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))
        after = keras.ops.convert_to_numpy(restored(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6)

        with pytest.raises(NotImplementedError):
            create_bfunet_variant("tiny", input_shape=(64, 64, 3), pretrained=True)

    def test_distilbert_true_raises(self):
        from dl_techniques.models.distilbert.model import DistilBERT

        with pytest.raises(NotImplementedError):
            DistilBERT.from_variant("base", pretrained=True)

    def test_modern_bert_true_raises(self):
        from dl_techniques.models.modern_bert.model import ModernBERT

        with pytest.raises(NotImplementedError):
            ModernBERT.from_variant("tiny", pretrained=True)

    def test_modern_bert_local_path_restores_by_value(self, tmp_path):
        """README § 9 Pattern 1, verbatim.

        Two failure modes this pins, both hit while writing the example:
        (a) `load_pretrained_weights` built its probe input with
            `keras.random.uniform(..., dtype="int32")`, which Keras REJECTS, so the
            whole local-path route raised on every unbuilt model;
        (b) an encoder saved before ever being CALLED writes zero weights (lazy
            subclassed build), and the transfer then finds no overlapping layer.
        The control arm is what distinguishes a real transfer from a no-op.
        """
        from dl_techniques.models.modern_bert.model import (
            ModernBERT,
            create_modern_bert_with_head,
        )
        from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

        encoder = ModernBERT.from_variant("tiny")
        ids = keras.random.randint((1, 128), 0, encoder.vocab_size, dtype="int32")
        encoder({"input_ids": ids}, training=False)  # REQUIRED before save
        path = os.path.join(str(tmp_path), "modern_bert_tiny.keras")
        encoder.save(path)

        ner_task = NLPTaskConfig(
            name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
        )
        ner_model = create_modern_bert_with_head(
            bert_variant="tiny", task_config=ner_task, pretrained=path
        )
        ner_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        inner = [l for l in ner_model.layers if isinstance(l, ModernBERT)][0]
        probe = np.random.default_rng(0).integers(0, 50368, (1, 128)).astype("int32")
        source = keras.models.load_model(path)
        a = keras.ops.convert_to_numpy(
            source({"input_ids": probe}, training=False)["last_hidden_state"]
        )
        b = keras.ops.convert_to_numpy(
            inner({"input_ids": probe}, training=False)["last_hidden_state"]
        )
        np.testing.assert_allclose(a, b, atol=1e-6)

        # CONTROL: a fresh encoder must NOT match, or the assertion above is vacuous.
        fresh = ModernBERT.from_variant("tiny")
        c = keras.ops.convert_to_numpy(
            fresh({"input_ids": probe}, training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(a - c)) > 1e-2


# ---------------------------------------------------------------------
# masked_autoencoder: the encoder contract the README now states
# ---------------------------------------------------------------------

def _mae_encoder(input_shape=(64, 64, 3), width=(16, 24, 32, 32)):
    """A /16 encoder -- the README's `Custom Encoder Architecture` shape, shrunk."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in width:
        x = keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = keras.layers.Activation("gelu")(x)
    return keras.Model(inputs, x, name="readme_encoder")


class TestMaskedAutoencoderReadmeContract:

    def test_encoder_is_a_model_not_a_dimension_list(self):
        """Every README example used to pass `encoder_dims=`/`encoder_output_shape=`.
        Neither is a parameter; `encoder` is, and it must be a `keras.Model`."""
        import inspect

        from dl_techniques.models.masked_autoencoder import MaskedAutoencoder

        params = inspect.signature(MaskedAutoencoder.__init__).parameters
        assert "encoder" in params
        assert "encoder_dims" not in params
        assert "encoder_output_shape" not in params

    def test_the_readme_custom_encoder_builds_and_reconstructs(self):
        from dl_techniques.models.masked_autoencoder import MaskedAutoencoder

        mae = MaskedAutoencoder(
            encoder=_mae_encoder(),
            patch_size=16,
            mask_ratio=0.75,
            input_shape=(64, 64, 3),
        )
        x = np.random.default_rng(0).random((2, 64, 64, 3)).astype("float32")
        out = mae(x, training=True)
        # The SPATIAL assertion is the one that matters: rank+channels alone passed
        # against a decoder emitting 4x the input resolution.
        assert tuple(int(d) for d in out["reconstruction"].shape) == (2, 64, 64, 3)

    def test_the_four_times_encoder_the_readme_showed_is_rejected(self):
        """The README's `Custom Encoder Architecture` example was a single
        `Conv2D(64, 7, strides=4)`. Under the default `decoder_depth=4` its decoder
        emits 4x the input resolution, and the constructor now says so."""
        from dl_techniques.models.masked_autoencoder import MaskedAutoencoder

        inputs = keras.Input(shape=(64, 64, 3))
        bad = keras.Model(
            inputs, keras.layers.Conv2D(64, 7, strides=4, padding="same")(inputs)
        )
        with pytest.raises(ValueError):
            MaskedAutoencoder(encoder=bad, patch_size=16, input_shape=(64, 64, 3))


# ---------------------------------------------------------------------
# mobilenet_v4: the variant names
# ---------------------------------------------------------------------

class TestMobileNetV4VariantNames:

    def test_the_documented_key_set_is_the_real_one(self):
        from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4

        assert set(MobileNetV4.MODEL_VARIANTS) == {
            "small", "medium", "large", "hybrid_medium", "hybrid_large",
        }

    @pytest.mark.parametrize("absent", ["conv_small", "conv_medium", "conv_large"])
    def test_the_conv_prefixed_names_raise(self, absent):
        """The class docstring advertised `from_variant("conv_medium")` a hundred
        lines below a module docstring saying it does not exist; the README carried
        six more, including a copy-pasteable `create_mobilenetv4("conv_medium", ...)`
        and an invented `"conv_large"`."""
        from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4

        with pytest.raises(ValueError):
            MobileNetV4.from_variant(absent, num_classes=10, input_shape=(32, 32, 3))

    def test_the_readme_quick_start_runs(self):
        from dl_techniques.models.mobilenet.mobilenet_v4 import create_mobilenetv4

        model = create_mobilenetv4(
            variant="small", num_classes=10, input_shape=(32, 32, 3)
        )
        out = model(np.zeros((2, 32, 32, 3), dtype="float32"), training=False)
        assert tuple(int(d) for d in out.shape) == (2, 10)


# ---------------------------------------------------------------------
# The docstring-example mechanism swept in the same step
# ---------------------------------------------------------------------

def test_keras_random_uniform_rejects_an_integer_dtype():
    """The mechanism behind eight corrected sites.

    `keras.random.uniform(..., dtype="int32")` ALWAYS raises, so every docstring
    example spelling a token batch that way was dead on arrival, and
    `modern_bert::load_pretrained_weights` -- live code -- raised on every unbuilt
    model. `keras.random.randint` is the replacement. Do NOT "restore" `uniform`.
    """
    with pytest.raises(ValueError, match="floating point"):
        keras.random.uniform((2, 4), 0, 100, dtype="int32")

    ids = keras.random.randint((2, 4), 0, 100, dtype="int32")
    assert tuple(int(d) for d in ids.shape) == (2, 4)


# ---------------------------------------------------------------------
# shgcn: the whole-graph invocation, and the batching trap that killed the
# README's only runnable quickstart (step 19, F-72)
# ---------------------------------------------------------------------

class TestSHGCNWholeGraphInvocation:
    """`shgcn`'s README instructed `model.fit([features, adj], labels)` until
    2026-08-19. Keras batches axis 0 of EVERY input, so it slices the [N, N]
    adjacency alongside the features and the run dies in the data pipeline.

    The README ALSO said a `tf.sparse.SparseTensor` adjacency was mandatory and
    that a dense one "will cause OOM errors". Measured: dense is the supported
    form (`SHGCNLayer` aggregates with `keras.ops.matmul`), and sparse happens to
    work too, but only on the TensorFlow backend. Both halves are pinned here.
    """

    @staticmethod
    def _graph(n=40, f=8, c=3, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, f)).astype("float32")
        adj = np.eye(n, dtype="float32")
        y = rng.integers(0, c, size=(n,))
        return x, adj, y

    def _model(self, c=3):
        from dl_techniques.models.shgcn.model import SHGCNNodeClassifier

        model = SHGCNNodeClassifier(num_classes=c, hidden_dims=[8], embedding_dim=8)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        return model

    def test_train_on_batch_is_the_documented_full_graph_step(self):
        x, adj, y = self._graph()
        model = self._model()
        model.train_on_batch([x, adj], y)
        out = model.predict_on_batch([x, adj])
        assert tuple(int(d) for d in out.shape) == (40, 3)

    def test_fit_with_an_explicit_batch_axis_is_the_other_documented_route(self):
        x, adj, y = self._graph()
        model = self._model()
        model.fit(
            [x[None, ...], adj[None, ...]], y[None, ...],
            epochs=1, batch_size=1, verbose=0,
        )

    def test_the_readme_form_that_was_removed_still_does_not_work(self):
        """RED-proof for the correction: `fit` on unbatched whole-graph inputs
        raises. If this ever starts passing, the README's warning is stale."""
        x, adj, y = self._graph()
        model = self._model()
        with pytest.raises(Exception):
            model.fit([x, adj], y, epochs=1, verbose=0)

    def test_predict_batches_too_and_only_looks_right_while_n_is_small(self):
        """The trap that made `model.predict([features, adj])` read as working:
        at N <= the default batch size of 32 nothing is sliced."""
        model = self._model()
        x, adj, _ = self._graph(n=16)
        assert tuple(int(d) for d in model.predict(
            [x, adj], verbose=0).shape) == (16, 3)

        x, adj, _ = self._graph(n=40)
        with pytest.raises(Exception):
            model.predict([x, adj], verbose=0)

    def test_a_dense_adjacency_is_the_supported_form(self):
        x, adj, _ = self._graph(n=12)
        model = self._model()
        assert tuple(int(d) for d in model([x, adj]).shape) == (12, 3)

    def test_a_sparse_adjacency_also_works_on_the_tf_backend(self):
        import tensorflow as tf

        x, adj, _ = self._graph(n=12)
        model = self._model()
        out = model([x, tf.sparse.from_dense(adj)])
        assert tuple(int(d) for d in out.shape) == (12, 3)


# ---------------------------------------------------------------------
# fastvit: the variant-registry alias models/CLAUDE.md asserted (step 19, F-30)
# ---------------------------------------------------------------------

def test_fastvit_exposes_the_canonical_model_variants_alias():
    """`models/CLAUDE.md` claimed `fastvit` carried both `MODEL_VARIANTS` and
    `SCALE_CONFIGS`; it carried neither, so `getattr(cls, "MODEL_VARIANTS")` --
    the pattern `vit`, `vit_hmlp` and `distilbert` all support -- raised
    `AttributeError`. The alias must be the SAME dict, not a copy.
    """
    from dl_techniques.models.fastvit.model import (
        MCI_VARIANTS,
        FastVitImageEncoder,
    )

    assert FastVitImageEncoder.MODEL_VARIANTS is MCI_VARIANTS
    assert "mci0" in FastVitImageEncoder.MODEL_VARIANTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
