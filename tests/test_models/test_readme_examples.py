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

Deliberately small: `"tiny"`-class variants everywhere. The original reason was
measured and is now HISTORICAL: `ModernBERT`-base cost 20.1 GB of host RSS to
CONSTRUCT (large: 24.2 GB) because its local layers padded to a single
`window_size**2 = 16384` window and built an O(window_size**4) relative-position
index in `__init__`. plan-2026-08-25T053412-0f1fa04f fixed both (D-006 moved the
index to `build()`, D-012 routed local layers to the 1-D `'window_band'`), and
construction is 0.69 GB for all three variants as of 2026-08-25. `"tiny"` stays
because a README example suite should be fast, not because base is unaffordable.
See decisions.md D-027 for the original degeneracy and D-044 for this module.
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
        from dl_techniques.models.vision.bias_free_denoisers.bfunet import (
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
        from dl_techniques.models.language.distilbert.model import DistilBERT

        with pytest.raises(NotImplementedError):
            DistilBERT.from_variant("base", pretrained=True)

    def test_modern_bert_true_raises(self):
        from dl_techniques.models.language.modern_bert.model import ModernBERT

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
        The head is in-model since plan-2026-10-11T004932-b83935e1, so the
        restored encoder is read through a headless twin carrying the headed
        model's own encoder weights.
        """
        from dl_techniques.models.language.modern_bert.model import (
            ModernBERT,
            create_modern_bert,
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
        ner_model = create_modern_bert(
            "tiny", include_top=True, task_config=ner_task, pretrained=path
        )
        ner_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        inner = ModernBERT.from_variant("tiny")
        probe = np.random.default_rng(0).integers(0, 50368, (1, 128)).astype("int32")
        inner({"input_ids": probe}, training=False)
        for layer in inner.layers:
            layer.set_weights(ner_model.get_layer(layer.name).get_weights())
        source = keras.models.load_model(path)
        a = keras.ops.convert_to_numpy(
            source({"input_ids": probe}, training=False)["last_hidden_state"]
        )
        b = keras.ops.convert_to_numpy(
            inner({"input_ids": probe}, training=False)["last_hidden_state"]
        )
        np.testing.assert_allclose(a, b, atol=1e-6)
        assert ner_model({"input_ids": probe}, training=False).shape == (1, 128, 9)

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

        from dl_techniques.models.vision.masked_autoencoder import MaskedAutoencoder

        params = inspect.signature(MaskedAutoencoder.__init__).parameters
        assert "encoder" in params
        assert "encoder_dims" not in params
        assert "encoder_output_shape" not in params

    def test_the_readme_custom_encoder_builds_and_reconstructs(self):
        from dl_techniques.models.vision.masked_autoencoder import MaskedAutoencoder

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
        from dl_techniques.models.vision.masked_autoencoder import MaskedAutoencoder

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
        from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4

        assert set(MobileNetV4.MODEL_VARIANTS) == {
            "small", "medium", "large", "hybrid_medium", "hybrid_large",
        }

    @pytest.mark.parametrize("absent", ["conv_small", "conv_medium", "conv_large"])
    def test_the_conv_prefixed_names_raise(self, absent):
        """The class docstring advertised `from_variant("conv_medium")` a hundred
        lines below a module docstring saying it does not exist; the README carried
        six more, including a copy-pasteable `create_mobilenetv4("conv_medium", ...)`
        and an invented `"conv_large"`."""
        from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4

        with pytest.raises(ValueError):
            MobileNetV4.from_variant(absent, num_classes=10, input_shape=(32, 32, 3))

    def test_the_readme_quick_start_runs(self):
        from dl_techniques.models.vision.mobilenet.mobilenet_v4 import create_mobilenetv4

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
        from dl_techniques.models.graph.shgcn.model import SHGCNNodeClassifier

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
# fastvit: the variant-registry alias models/AGENTS.md asserted (step 19, F-30)
# ---------------------------------------------------------------------

def test_fastvit_exposes_the_canonical_model_variants_alias():
    """`models/AGENTS.md` claimed `fastvit` carried both `MODEL_VARIANTS` and
    `SCALE_CONFIGS`; it carried neither, so `getattr(cls, "MODEL_VARIANTS")` --
    the pattern `vit`, `vit_hmlp` and `distilbert` all support -- raised
    `AttributeError`. The alias must be the SAME dict, not a copy.
    """
    from dl_techniques.models.vision.fastvit.model import (
        MCI_VARIANTS,
        FastVitImageEncoder,
    )

    assert FastVitImageEncoder.MODEL_VARIANTS is MCI_VARIANTS
    assert "mci0" in FastVitImageEncoder.MODEL_VARIANTS


# ---------------------------------------------------------------------
# neural_hmm: the leaf README's code blocks and variant table (2026-10-14)
# ---------------------------------------------------------------------

_NEURAL_HMM_README = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "src", "dl_techniques", "models",
    "time_series", "neural_hmm", "README.md",
)


def _neural_hmm_readme():
    with open(_NEURAL_HMM_README, encoding="utf-8") as handle:
        return handle.read()


class TestNeuralHMMReadme:
    """The README was written with the rewrite, so its examples are RUN, not read: every
    fenced ``python`` block executes in order in ONE namespace (later examples reuse the
    quick start's ``inputs``/``model``, as a reader would), the values each block binds are
    checked, and the variant table names exactly the class table's keys."""

    @staticmethod
    def _blocks():
        import re

        return re.findall(r"```python\n(.*?)```", _neural_hmm_readme(), flags=re.DOTALL)

    @pytest.fixture(scope="class")
    def namespace(self):
        namespace = {}
        for index, block in enumerate(self._blocks()):
            exec(compile(block, f"neural_hmm/README.md[block {index}]", "exec"), namespace)
        return namespace

    def test_the_neural_hmm_readme_has_seven_runnable_blocks(self):
        assert len(self._blocks()) == 7

    def test_the_neural_hmm_quick_start_values(self, namespace):
        to_np = keras.ops.convert_to_numpy
        path, gamma, log_px = (to_np(namespace[k]) for k in ("path", "gamma", "log_px"))
        assert path.shape == (8, 20) and (path[4:, 15:] == -1).all() and (path[:, :15] >= 0).all()
        assert gamma.shape == (8, 20, 3) and np.isfinite(log_px).all() and log_px.shape == (8,)

    def test_the_neural_hmm_usage_examples_values(self, namespace):
        to_np = keras.ops.convert_to_numpy
        assert to_np(namespace["log_p_y"]).shape == (8,) and (to_np(namespace["log_p_y"]) <= 0).all()
        assert to_np(namespace["tagged"]).shape == (8, 20)
        assert np.isfinite(to_np(namespace["per_step_nll"])).all()
        segments = to_np(namespace["segments"])
        valid = namespace["mask"] > 0
        for row, keep in zip(segments, valid):
            assert row[keep][0] == 0 and (np.diff(row[keep]) >= 0).all()
        assert np.isfinite(to_np(namespace["log_z"])).all()

    def test_the_neural_hmm_interpretability_values(self, namespace):
        np.testing.assert_allclose(namespace["transition"].sum(axis=1), 1.0, rtol=0, atol=1e-5)
        np.testing.assert_allclose(namespace["initial"].sum(), 1.0, rtol=0, atol=1e-5)
        assert (namespace["dwell"] >= 1.0).all()

    def test_the_neural_hmm_variant_table_names_the_class_table(self):
        import re

        from dl_techniques.models.time_series.neural_hmm import NeuralHMM

        section = _neural_hmm_readme().split("## 8. Configuration & Presets", 1)[1].split("\n## ", 1)[0]
        documented = set(re.findall(r"^\| `([a-z0-9-]+)` \|", section, flags=re.MULTILINE))
        assert documented == set(NeuralHMM.MODEL_VARIANTS)

    def test_the_neural_hmm_unread_argument_table_is_the_rule(self):
        """§7's table of unread arguments, row by row, against ``NeuralHMM.unread_arguments``.

        Each row is evaluated on a base that reads as much as possible (banded, gaussian,
        generative, autoregressive, transformer: only ``kernel_size`` unread). An encoder row
        (including "no encoder") lists the full unread set of its configuration; any other
        row lists what its condition ADDS to the base's unread set.
        """
        import re

        from dl_techniques.models.time_series.neural_hmm import NeuralHMM

        base = {"mode": "generative", "obs_type": "gaussian", "emission_context": "autoregressive",
                "transition": "static", "topology": "banded", "encoder_type": "transformer"}
        rows = {  # row label prefix -> (configs, the row is a full encoder row)
            "no encoder": ([{"emission_context": "none", "encoder_type": "lstm"}], True),
            "encoder `lstm` or `gru`": ([{"encoder_type": "lstm"}, {"encoder_type": "gru"}], True),
            "encoder `mlp`": ([{"encoder_type": "mlp"}], True),
            "encoder `conv`": ([{"encoder_type": "conv"}], True),
            "encoder `transformer`": ([{}], True),
            "`topology` other than `banded`": ([{"topology": "ergodic"}, {"topology": "left_to_right"}], False),
            "`n_states=1`": ([{"n_states": 1}], False),
            "`obs_type=\"categorical\"`": ([{"obs_type": "categorical"}], False),
            "`mode=\"discriminative\"`": ([{"mode": "discriminative", "emission_context": "none"}], False),
        }
        table = _neural_hmm_readme().split("| Configuration | Arguments it does not read |", 1)[1]
        table = table.split("\n\n", 1)[0]
        seen = set()
        for line in table.strip().splitlines()[1:]:
            label, listed = [cell.strip() for cell in line.strip("|").split("|")]
            prefix = next(p for p in rows if label.startswith(p))
            seen.add(prefix)
            configs, full = rows[prefix]
            for config in configs:
                unread = set(NeuralHMM.unread_arguments({**base, **config}))
                if not full:
                    unread -= set(NeuralHMM.unread_arguments(base))
                assert set(re.findall(r"`(\w+)`", listed)) == unread, (label, config)
        assert seen == set(rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
