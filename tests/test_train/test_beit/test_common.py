"""Tests for ``train.beit.common`` — the three things all three BEiT trainers share.

Two of those three are RE-EXPORTS from ``train.energy_transformer.common`` (D-008), so
this module does not re-test their internals — ``tests/test_train/test_energy_transformer/
test_build_raw_image_dataset.py`` owns those, 622 lines of them. What IS tested here is
the contract this package depends on: that the re-exported names are the same objects,
that the documented return tuple and the ``element_map_fn`` hook behave as the BEiT MIM
trainer will use them, and that ``build_optimizer`` really builds the schedule it is asked
for.

The third — :func:`load_frozen_tokenizer` — is BEiT-specific and is tested for the
failures it exists to prevent, each of which is otherwise SILENT:

* a checkpoint that is not a VQ-VAE at all (would fail deep inside a ``tf.data`` graph);
* a code grid that does not match the encoder's patch grid (would train on targets read
  from the wrong spatial position, with a finite and entirely plausible loss);
* a tokenizer that is not actually frozen (would drift during MIM pre-training, making the
  target distribution non-stationary — BEiT's tokenizer is frozen by definition).

Everything here is CPU-cheap: cifar10 is built in memory from ``keras.datasets``, and every
VQ-VAE is a 16-channel toy at 32x32. The single imagenette test reads real prepared TFDS
records and is ``skipif``-guarded on their presence — the guard GLOBS for the shards rather
than assuming either way, and a SKIP is explicitly not a pass.
"""

import os
from dataclasses import dataclass
from glob import glob
from typing import Any, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.beit.common import (
    DATASET_NUM_CLASSES,
    SUPPORTED_DATASETS,
    build_optimizer,
    build_raw_image_dataset,
    load_frozen_tokenizer,
)
from train.energy_transformer.common import IMAGENETTE_TFDS_NAME
from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick


# ---------------------------------------------------------------------
# imagenette availability -- PROVED by globbing, not assumed either way
# ---------------------------------------------------------------------

_TFDS_DATA_DIR = os.environ.get(
    "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
_IMAGENETTE_RECORD_DIR = os.path.join(
    _TFDS_DATA_DIR, *IMAGENETTE_TFDS_NAME.split("/"), "1.0.0")
_IMAGENETTE_TRAIN_SHARDS = sorted(
    glob(os.path.join(_IMAGENETTE_RECORD_DIR, "imagenette-train.tfrecord-*")))
_IMAGENETTE_SKIP_REASON = (
    f"imagenette TFDS records are not prepared: no `imagenette-train.tfrecord-*` "
    f"under {_IMAGENETTE_RECORD_DIR!r} (TFDS_DATA_DIR={_TFDS_DATA_DIR!r}). This test "
    f"reads REAL records and must never download anything. A SKIP here is NOT a pass — "
    f"the imagenette branch simply did not run on this machine."
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

@dataclass
class _OptimConfig:
    """The subset of a trainer ``TrainingConfig`` that ``build_optimizer`` reads."""
    lr_schedule_type: str = "cosine_decay"
    learning_rate: float = 1.5e-3
    epochs: int = 4
    warmup_epochs: int = 1
    optimizer_type: str = "adamw"
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0


def _toy_tokenizer(
        image_size: int = 32,
        downsample_factor: int = 8,
        num_embeddings: int = 32,
) -> VQVAERotationTrick:
    """A CPU-cheap auto-built VQ-VAE. At 32/8 the code grid is 4x4."""
    return VQVAERotationTrick(
        num_embeddings=num_embeddings,
        embedding_dim=4,
        input_shape=(image_size, image_size, 3),
        downsample_factor=downsample_factor,
        hidden_channels=8,
        num_res_blocks=1,
    )


def _save_toy_tokenizer(tmp_path: Any, **kwargs: Any) -> str:
    model = _toy_tokenizer(**kwargs)
    # Build it before saving so the checkpoint carries weights, as a real stage-0 run
    # would.
    image_size = kwargs.get("image_size", 32)
    model(np.zeros((1, image_size, image_size, 3), dtype="float32"))
    path = os.path.join(str(tmp_path), "tokenizer.keras")
    model.save(path)
    return path


def _mim_element_map_fn(image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any, Any]:
    """A stand-in for the real BEiT MIM element: ``((image, mask), targets, weights)``."""
    mask = tf.cast(tf.range(16) < 6, tf.bool)
    targets = tf.zeros((16,), dtype=tf.int32)
    return (image, mask), targets, tf.cast(mask, tf.float32)


# ---------------------------------------------------------------------
# 1. the dataset builder -- the documented tuple and the element_map_fn hook
# ---------------------------------------------------------------------

class TestBuildRawImageDataset:
    """The contract the BEiT trainers consume, on the offline cifar10 branch."""

    def test_it_is_the_same_object_as_energy_transformers(self):
        """D-008: re-exported, not re-implemented. A COPY would fail this by identity.

        Why this can fail if the implementation is wrong: someone "makes the BEiT package
        self-contained" by pasting the body in. The copy would then silently drift from
        the original's three in-code branch-ordering decision anchors.
        """
        from train.energy_transformer import common as et_common

        assert build_raw_image_dataset is et_common.build_raw_image_dataset
        assert build_optimizer is et_common.build_optimizer

    def test_it_returns_the_documented_triple(self):
        ds, num_examples, num_classes = build_raw_image_dataset(
            "cifar10", image_size=32, batch_size=4, is_training=False, augment=False,
        )
        assert isinstance(ds, tf.data.Dataset)
        assert num_examples == 10000          # cifar10 test split
        assert num_classes == 10 == DATASET_NUM_CLASSES["cifar10"]

        images, labels = next(iter(ds))
        assert tuple(images.shape) == (4, 32, 32, 3)
        assert tuple(labels.shape) == (4,)
        assert images.dtype == tf.float32

    def test_the_classifier_pipeline_is_the_unmodified_image_label_pair(self):
        """Stage 2 passes NO element_map_fn and must get `(image, label)` back."""
        ds, _, _ = build_raw_image_dataset(
            "cifar10", image_size=32, batch_size=4, is_training=False, augment=False,
        )
        element = next(iter(ds))
        assert isinstance(element, tuple) and len(element) == 2

    def test_the_element_map_fn_hook_reshapes_the_element(self):
        """Stage 1 injects the masking/tokenizing element through this hook.

        Why this can fail if the implementation is wrong: a builder that ignored
        `element_map_fn` would still yield perfectly good `(image, label)` batches, and
        the MIM trainer would silently train on CLASS LABELS instead of code ids.
        """
        ds, _, _ = build_raw_image_dataset(
            "cifar10", image_size=32, batch_size=4, is_training=False, augment=False,
            element_map_fn=_mim_element_map_fn,
        )
        (images, mask), targets, weights = next(iter(ds))
        assert tuple(images.shape) == (4, 32, 32, 3)
        assert tuple(mask.shape) == (4, 16) and mask.dtype == tf.bool
        assert tuple(targets.shape) == (4, 16) and targets.dtype == tf.int32
        # sample_weight is exactly the mask -- BEiT's convention.
        np.testing.assert_array_equal(
            weights.numpy(), mask.numpy().astype("float32"))

    def test_an_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="Unsupported dataset"):
            build_raw_image_dataset(
                "imagenet21k", image_size=32, batch_size=4, is_training=False)

    def test_both_beit_datasets_are_supported(self):
        assert set(SUPPORTED_DATASETS) == {"imagenette", "cifar10"}


@pytest.mark.skipif(not _IMAGENETTE_TRAIN_SHARDS, reason=_IMAGENETTE_SKIP_REASON)
class TestTheImagenetteBranch:
    """A-8: imagenette is the dataset VERIFIED present offline on this machine.

    Bounded to `.take(1)` batch of 4 at `image_size=32` with a small shuffle buffer — the
    4096 default would decode 4096 full-size JPEGs just to build the pipeline. Reads real
    prepared records; never downloads.
    """

    def test_the_training_pipeline_yields_the_documented_triple_offline(self):
        ds, num_examples, num_classes = build_raw_image_dataset(
            "imagenette", image_size=32, batch_size=4, is_training=True,
            augment=False, shuffle_buffer=32, seed=0,
        )
        assert num_examples > 0
        assert num_classes == 10
        images, labels = next(iter(ds.take(1)))
        assert tuple(images.shape) == (4, 32, 32, 3)
        assert tuple(labels.shape) == (4,)


# ---------------------------------------------------------------------
# 2. the optimizer block
# ---------------------------------------------------------------------

class TestBuildOptimizer:

    def test_it_returns_a_real_optimizer_with_the_requested_schedule(self):
        """The schedule must be LIVE on the optimizer, not a float snapshot.

        MEASURED, not assumed: on Keras 3.8 ``optimizer.learning_rate`` is NOT the
        schedule object — it is the schedule EVALUATED at the optimizer's current
        ``iterations`` (a scalar tensor). An ``isinstance(..., LearningRateSchedule)``
        assertion therefore fails on a perfectly wired optimizer. Driving
        ``iterations`` and reading the value back is the real liveness probe: a bare
        float would return the same number at every step.
        """
        optimizer = build_optimizer(_OptimConfig(), steps_per_epoch=10)
        assert isinstance(optimizer, keras.optimizers.Optimizer)

        def lr_at(step: int) -> float:
            optimizer.iterations.assign(step)
            return float(keras.ops.convert_to_numpy(optimizer.learning_rate))

        peak = lr_at(10)                      # end of the 1-epoch warmup
        assert peak == pytest.approx(1.5e-3, rel=1e-3)
        assert lr_at(0) < peak, "no warmup ramp -- the schedule is not live"
        assert lr_at(39) < peak, "no cosine decay over 4 epochs x 10 steps"

    def test_adamw_carries_the_weight_decay_and_the_gradient_clip(self):
        """H10: the decay comes from the optimizer ONLY, so it must actually arrive."""
        config = _OptimConfig(optimizer_type="adamw", weight_decay=0.05)
        optimizer = build_optimizer(config, steps_per_epoch=10)
        assert float(optimizer.weight_decay) == pytest.approx(0.05)
        assert float(optimizer.global_clipnorm) == pytest.approx(1.0)

    def test_a_non_adamw_optimizer_does_not_get_a_weight_decay(self):
        config = _OptimConfig(optimizer_type="adam", weight_decay=0.05)
        optimizer = build_optimizer(config, steps_per_epoch=10)
        assert getattr(optimizer, "weight_decay", None) in (None, 0.0)


# ---------------------------------------------------------------------
# 3. the frozen tokenizer
# ---------------------------------------------------------------------

class TestLoadFrozenTokenizer:

    def test_the_happy_path_returns_the_documented_code_grid(self, tmp_path):
        path = _save_toy_tokenizer(tmp_path)           # 32 / 8 -> 4x4
        tokenize = load_frozen_tokenizer(
            path, expected_grid=(4, 4), image_shape=(32, 32, 3))

        code_ids = tokenize(np.zeros((3, 32, 32, 3), dtype="float32"))
        assert tuple(keras.ops.shape(code_ids)) == (3, 4, 4)
        assert keras.backend.standardize_dtype(code_ids.dtype) == "int32"
        assert tokenize.grid_size == (4, 4)

    def test_the_code_ids_are_valid_codebook_indices(self, tmp_path):
        path = _save_toy_tokenizer(tmp_path, num_embeddings=32)
        tokenize = load_frozen_tokenizer(
            path, expected_grid=(4, 4), image_shape=(32, 32, 3))
        ids = np.asarray(keras.ops.convert_to_numpy(
            tokenize(np.random.rand(2, 32, 32, 3).astype("float32"))))
        assert ids.min() >= 0 and ids.max() < 32

    def test_a_grid_mismatch_raises_loudly(self, tmp_path):
        """The silent killer: targets read from the wrong spatial position.

        Why this can fail if the implementation is wrong: without the probe, a 4x4
        tokenizer paired with a 2x2 patch grid produces a finite MIM loss forever.
        """
        path = _save_toy_tokenizer(tmp_path)           # really 4x4
        with pytest.raises(ValueError, match=r"code-grid mismatch"):
            load_frozen_tokenizer(
                path, expected_grid=(2, 2), image_shape=(32, 32, 3))

    def test_the_grid_check_is_measured_not_derived(self, tmp_path):
        """NOT computed from expected_grid x downsample_factor — that would be circular.

        Both calls below ask for the SAME ``expected_grid=(4, 4)`` at the SAME
        ``image_shape=(32, 32, 3)``. Only the tokenizer's own ``downsample_factor``
        differs (8 -> 4x4, 4 -> 8x8). A check derived from the tokenizer's config would
        agree with itself and accept both; a check that runs the encoder accepts only the
        first. This is the self-referential-oracle failure mode, guarded directly.
        """
        good = _save_toy_tokenizer(tmp_path, downsample_factor=8)
        load_frozen_tokenizer(good, expected_grid=(4, 4), image_shape=(32, 32, 3))

        bad_dir = os.path.join(str(tmp_path), "f4")
        os.makedirs(bad_dir, exist_ok=True)
        bad = _save_toy_tokenizer(bad_dir, downsample_factor=4)
        with pytest.raises(ValueError, match=r"code-grid mismatch"):
            load_frozen_tokenizer(bad, expected_grid=(4, 4), image_shape=(32, 32, 3))

    def test_an_image_shape_the_tokenizer_cannot_encode_raises_an_actionable_error(
            self, tmp_path):
        """MEASURED: the auto-built encoder pins a FIXED input shape.

        A tokenizer auto-built at ``input_shape=(32, 32, 3)`` does not merely produce a
        different grid at 64x64 — its encoder's ``InputSpec`` REFUSES the call with
        ``Input 0 of layer "vqvae_rotation_encoder" is incompatible...``, which names
        neither the tokenizer path nor what the caller should do. The helper re-raises
        with both.
        """
        path = _save_toy_tokenizer(tmp_path)          # native (32, 32, 3)
        with pytest.raises(ValueError, match=r"cannot encode an image of shape"):
            load_frozen_tokenizer(
                path, expected_grid=(4, 4), image_shape=(64, 64, 3))

    def test_a_non_vq_vae_checkpoint_raises_typeerror(self, tmp_path):
        """A `.keras` load returns whatever was saved; only a VQ-VAE has code ids."""
        impostor = keras.Sequential([
            keras.layers.Input((32, 32, 3)),
            keras.layers.Conv2D(4, 3, padding="same"),
        ])
        path = os.path.join(str(tmp_path), "impostor.keras")
        impostor.save(path)

        with pytest.raises(TypeError, match=r"not a VQVAERotationTrick"):
            load_frozen_tokenizer(
                path, expected_grid=(4, 4), image_shape=(32, 32, 3))

    def test_a_missing_checkpoint_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_frozen_tokenizer(
                os.path.join(str(tmp_path), "nope.keras"),
                expected_grid=(4, 4), image_shape=(32, 32, 3))

    def test_a_non_keras_path_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "tokenizer.h5")
        with open(path, "w") as handle:
            handle.write("not a model")
        with pytest.raises(ValueError, match=r"only supports \.keras"):
            load_frozen_tokenizer(
                path, expected_grid=(4, 4), image_shape=(32, 32, 3))

    @pytest.mark.parametrize(
        "grid,image_shape",
        [
            ((4,), (32, 32, 3)),
            ((0, 4), (32, 32, 3)),
            ((4, 4), (32, 32)),
            ((4, 4), (32, 32, 0)),
        ],
    )
    def test_malformed_arguments_raise_before_the_model_is_loaded(
            self, tmp_path, grid, image_shape):
        path = _save_toy_tokenizer(tmp_path)
        with pytest.raises(ValueError):
            load_frozen_tokenizer(
                path, expected_grid=grid, image_shape=image_shape)

    def test_the_tokenizer_is_frozen(self, tmp_path):
        """`trainable is False` AND no trainable variables reach a gradient tape.

        The flag alone is not the property that matters: BEiT's tokenizer is frozen BY
        DEFINITION, and a tokenizer that kept drifting would make the MIM target
        distribution non-stationary while every loss curve still looked fine.
        """
        path = _save_toy_tokenizer(tmp_path)
        tokenize = load_frozen_tokenizer(
            path, expected_grid=(4, 4), image_shape=(32, 32, 3))
        model = tokenize.model

        assert model.trainable is False
        assert model.trainable_weights == []
        assert list(model.trainable_variables) == []
        # Precondition: it HAS weights, so "no trainable weights" is not vacuous.
        assert len(model.weights) > 0

        images = tf.constant(np.random.rand(2, 32, 32, 3).astype("float32"))
        with tf.GradientTape() as tape:
            latents = model.encode(images)
            loss = tf.reduce_sum(latents)
        # A tape over the frozen model exposes NO trainable variable to differentiate,
        # which is what `fit()` and `optimizer.apply_gradients` consume.
        assert tape.gradient(loss, list(model.trainable_variables)) == []

        # MEASURED CAVEAT, asserted so it is not re-discovered as a surprise: Keras 3's
        # `trainable = False` empties `trainable_weights` but leaves the UNDERLYING
        # `tf.Variable.trainable` at True, so a `tf.GradientTape` still auto-WATCHES
        # them. Anything that differentiates w.r.t. `tape.watched_variables()` instead of
        # `model.trainable_variables` would happily update this "frozen" tokenizer.
        assert len(tape.watched_variables()) > 0

    def test_it_works_as_the_beit_mim_map_fn_tokenizer(self, tmp_path):
        """The end-to-end shape the stage-1 trainer will wire: inside a tf.data graph.

        `make_beit_mim_map_fn` hands the tokenizer an UNBATCHED image, so the documented
        `img[None]` wrapper is exercised here rather than assumed.
        """
        from dl_techniques.datasets.vision.beit_masking import make_beit_mim_map_fn

        path = _save_toy_tokenizer(tmp_path)
        tokenize = load_frozen_tokenizer(
            path, expected_grid=(4, 4), image_shape=(32, 32, 3))

        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=lambda img: tokenize(img[None])[0],
            grid_size=(4, 4),
            num_masking_patches=6,
            min_num_patches=2,
        )

        images = np.random.rand(4, 32, 32, 3).astype("float32")
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn).batch(2)
        (image_b, mask_b), targets_b, weights_b = next(iter(ds))

        assert tuple(image_b.shape) == (2, 32, 32, 3)
        assert tuple(mask_b.shape) == (2, 16)
        assert tuple(targets_b.shape) == (2, 16)
        assert targets_b.dtype == tf.int32
        np.testing.assert_array_equal(
            weights_b.numpy(), mask_b.numpy().astype("float32"))
        ids = targets_b.numpy()
        assert ids.min() >= 0 and ids.max() < 32
