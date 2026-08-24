"""RED proof for `vq_vae_rotation`'s k-means warm start (review finding C-31).

Two independent defects lived inside `VectorQuantizerRotationTrick.call()`:

1. It read its own flag with
   `float(keras.ops.convert_to_numpy(self.kmeans_init_done))`. On the TF backend
   `convert_to_numpy` is `np.asarray(x)`, which raises on a graph tensor — and
   `model.fit()` traces `train_step` by default.
2. It appended to `self._kmeans_accum`, a plain Python list, from inside that
   traced function. Even with the read made graph-safe, the append happens once
   per TRACE, so `kmeans_init_steps > 1` accumulated one batch, not N.

**The regime is the point.** Every pre-existing k-means test called the layer
eagerly, which is the one regime in which neither defect is visible. These tests
go through `model.fit()` and through an explicit `@tf.function`.

CPU only.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick
from dl_techniques.layers.vector_quantizer_rotation_trick import (
    VectorQuantizerRotationTrick,
)

INPUT_SHAPE = (16, 16, 3)
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 4


def _images(n=8):
    return np.random.default_rng(0).random((n, *INPUT_SHAPE)).astype("float32")


def _model(kmeans_init_steps=1):
    keras.utils.set_random_seed(0)
    return VQVAERotationTrick(
        input_shape=INPUT_SHAPE,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        hidden_channels=8,
        downsample_factor=2,
        num_res_blocks=1,
        kmeans_init=True,
        kmeans_init_steps=kmeans_init_steps,
    )


class TestKMeansSurvivesGraphMode:

    def test_fit_runs_with_kmeans_init_true(self):
        """`model.fit()` must complete. At HEAD it raised inside the trace.

        The exception was `NotImplementedError: Cannot convert a symbolic
        tf.Tensor to a numpy array` (or a TF-version-specific equivalent),
        raised by `np.asarray` on `kmeans_init_done` during tracing.
        """
        model = _model()
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        history = model.fit(_images(8), epochs=1, batch_size=4, verbose=0)
        assert np.isfinite(float(history.history["loss"][-1]))

    def test_fit_actually_warm_starts_the_codebook(self):
        """ANTI-VACUITY: fit() must not merely survive, it must warm-start.

        A "fix" that deleted the k-means path outright would pass the test
        above and fail this one.
        """
        model = _model()
        model(_images(2), training=False)  # build
        before = keras.ops.convert_to_numpy(model.quantizer.embeddings).copy()
        assert not model.quantizer.is_codebook_warm_started

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0))
        model.fit(_images(8), epochs=1, batch_size=4, verbose=0)

        after = keras.ops.convert_to_numpy(model.quantizer.embeddings)
        assert model.quantizer.is_codebook_warm_started
        assert not np.allclose(before, after), (
            "fit() did not warm-start the codebook"
        )

    def test_call_is_graph_safe_under_tf_function(self):
        """The layer alone, under an explicit trace. No numpy read may remain."""
        layer = VectorQuantizerRotationTrick(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=EMBEDDING_DIM,
            kmeans_init=True,
        )
        x = np.random.default_rng(1).random((6, EMBEDDING_DIM)).astype("float32")
        layer(x, training=False)  # build eagerly

        @tf.function
        def traced(v):
            return layer(v, training=True)

        out = traced(tf.convert_to_tensor(x))
        assert tuple(out.shape) == (6, EMBEDDING_DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


class TestKMeansAccumulatesEveryBatch:

    def test_three_steps_accumulate_three_batches(self):
        """`kmeans_init_steps=3` must see 3 batches' worth of vectors, not 1.

        This is the second, quieter half of the defect: the accumulator was a
        Python list mutated inside a traced `call`, so it grew once per trace.
        The count is read from the layer's own log line, which reports the
        number of samples k-means was fitted on.
        """
        model = _model(kmeans_init_steps=3)
        model(_images(2), training=False)  # build

        seen = {}
        original = model.quantizer.warm_start_codebook

        def spy(batches):
            arrays = batches if isinstance(batches, (list, tuple)) else [batches]
            seen["num_batches"] = len(arrays)
            seen["num_vectors"] = sum(
                int(np.asarray(keras.ops.convert_to_numpy(a)).reshape(
                    -1, EMBEDDING_DIM).shape[0])
                for a in arrays
            )
            return original(batches)

        model.quantizer.warm_start_codebook = spy
        try:
            model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0))
            model.fit(_images(12), epochs=1, batch_size=4, verbose=0)
        finally:
            del model.quantizer.warm_start_codebook

        assert seen["num_batches"] == 3, (
            f"kmeans_init_steps=3 accumulated {seen.get('num_batches')} batches"
        )
        # 12 images / 3 chunks = 4 images per chunk, each 8x8 latent positions
        # after a /2 downsample of a 16x16 input.
        assert seen["num_vectors"] == 3 * 4 * 8 * 8

    def test_explicit_warm_start_concatenates_a_sequence(self):
        """The layer-level entry point accepts a list and uses ALL of it."""
        layer = VectorQuantizerRotationTrick(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=EMBEDDING_DIM,
            kmeans_init=True,
        )
        rng = np.random.default_rng(2)
        batches = [rng.random((5, EMBEDDING_DIM)).astype("float32")
                   for _ in range(3)]
        layer(batches[0], training=False)  # build
        before = keras.ops.convert_to_numpy(layer.embeddings).copy()

        layer.warm_start_codebook(batches)

        after = keras.ops.convert_to_numpy(layer.embeddings)
        assert layer.is_codebook_warm_started
        assert not np.allclose(before, after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
