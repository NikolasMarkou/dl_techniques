"""Tests for the masked-patch tf.data transform (Energy Transformer MIM).

Every numerical test runs at the REALISTIC size N=196 (image 224, patch 16,
P=768). A toy N is banned here: it hides exactly the class of bug this suite
exists to catch (rounding collapsing the 90/10 split, an off-by-one in the
patch grid).

Covers success criteria C2 (the 90/10 masking rule + the loss-weight scale) and
C6 (I7: patch-order agreement between ``patchify_targets`` and the model's
``PatchEmbedding2D``).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import dl_techniques
from dl_techniques.datasets.vision import masked_patches
from dl_techniques.datasets.vision.masked_patches import (
    make_masked_patch_map_fn,
    make_stateless_masked_patch_map_fn,
    patchify_targets,
)
from dl_techniques.layers.embedding.factory import create_embedding_layer

# Realistic ViT-Base/16 @224 geometry. Do not shrink these.
IMAGE_SIZE = 224
PATCH_SIZE = 16
CHANNELS = 3
GRID = IMAGE_SIZE // PATCH_SIZE          # 14
NUM_PATCHES = GRID * GRID                # 196
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS  # 768

MASK_RATIO = 0.5
MASK_TOKEN_FRAC = 0.9
N_LOSS = int(round(MASK_RATIO * NUM_PATCHES))          # 98
N_INPUT = int(round(MASK_TOKEN_FRAC * N_LOSS))         # 88


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260714)


@pytest.fixture(scope="module")
def sample_image(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")


def _sampled_batches(num_batches: int = 64, batch_size: int = 4, seed: int = 7):
    """Run the map fn over `num_batches` batches and return the stacked outputs."""
    images = np.zeros(
        (num_batches * batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype="float32"
    )
    map_fn = make_masked_patch_map_fn(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        mask_ratio=MASK_RATIO,
        mask_token_frac=MASK_TOKEN_FRAC,
        seed=seed,
    )
    ds = (
        tf.data.Dataset.from_tensor_slices(images)
        .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    masks, weights = [], []
    for (_img, input_mask), _targets, loss_weight in ds:
        masks.append(input_mask.numpy())
        weights.append(loss_weight.numpy())
    return np.concatenate(masks, axis=0), np.concatenate(weights, axis=0)


class TestMaskingRule:
    """C2 / I2: the paper's 90/10 rule and the loss-weight scale."""

    @pytest.fixture(scope="class")
    def sampled(self):
        return _sampled_batches()

    def test_shapes_and_dtypes(self, sampled):
        input_mask, loss_weight = sampled
        assert input_mask.shape == (256, NUM_PATCHES)
        assert loss_weight.shape == (256, NUM_PATCHES)
        assert input_mask.dtype == np.bool_
        assert loss_weight.dtype == np.float32

    def test_element_signature(self):
        """The frozen contract: ((image, mask), targets, weights)."""
        map_fn = make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, seed=0)
        ds = tf.data.Dataset.from_tensor_slices(
            np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32")
        ).map(map_fn).batch(2)
        (image, input_mask), targets, loss_weight = next(iter(ds))
        assert image.shape == (2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        assert image.dtype == tf.float32
        assert input_mask.shape == (2, NUM_PATCHES)
        assert input_mask.dtype == tf.bool
        assert targets.shape == (2, NUM_PATCHES, PATCH_DIM)
        assert targets.dtype == tf.float32
        assert loss_weight.shape == (2, NUM_PATCHES)
        assert loss_weight.dtype == tf.float32

    def test_exact_counts(self, sampled):
        """|S| == 98 and |input_mask| == 88 on every one of the 256 samples."""
        input_mask, loss_weight = sampled
        loss_set = loss_weight > 0.0
        assert np.all(loss_set.sum(axis=1) == N_LOSS)
        assert np.all(input_mask.sum(axis=1) == N_INPUT)

    def test_input_mask_is_strict_subset_of_loss_set(self, sampled):
        """I2: input_mask ⊊ loss_set. The ~10% un-occluded loss tokens must exist."""
        input_mask, loss_weight = sampled
        loss_set = loss_weight > 0.0
        # subset: no token is masked-in-input without being in the loss set
        assert not np.any(input_mask & ~loss_set)
        # STRICT: every sample keeps n_loss - n_input un-occluded loss tokens
        unoccluded = (loss_set & ~input_mask).sum(axis=1)
        assert np.all(unoccluded == N_LOSS - N_INPUT)
        assert N_LOSS - N_INPUT == 10

    def test_loss_weight_sums_to_N(self, sampled):
        """I3: Σ_i loss_weight_i == N per sample (fp32 -> atol 1e-3)."""
        _input_mask, loss_weight = sampled
        np.testing.assert_allclose(
            loss_weight.sum(axis=1), float(NUM_PATCHES), atol=1e-3
        )

    def test_loss_weight_is_zero_off_the_loss_set(self, sampled):
        """Exactly 0.0 off S (not merely small) and exactly N/n_loss on S."""
        _input_mask, loss_weight = sampled
        nonzero = loss_weight[loss_weight != 0.0]
        assert np.all(loss_weight[loss_weight == 0.0] == 0.0)
        np.testing.assert_allclose(
            nonzero, np.float32(NUM_PATCHES / N_LOSS), rtol=0, atol=0
        )

    def test_masks_differ_across_samples(self, sampled):
        """The selection is per-sample, not one mask broadcast over the batch."""
        input_mask, _loss_weight = sampled
        assert len({m.tobytes() for m in input_mask}) > 200


class TestPatchOrder:
    """C6 / I7: patchify_targets must agree with PatchEmbedding2D token-for-token.

    An identity-kernel Conv2D turns PatchEmbedding2D into a pure patchifier, so
    its output must be elementwise equal to patchify_targets. A mismatch means
    the decoder would be trained against spatially scrambled targets — a failure
    that still shows a descending loss curve.
    """

    @staticmethod
    def _identity_patch_embedding():
        layer = create_embedding_layer(
            "patch_2d",
            patch_size=PATCH_SIZE,
            embed_dim=PATCH_DIM,
            use_bias=True,
        )
        layer.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        # Conv2D kernel is (patch_h, patch_w, C, filters) and contracts over
        # (h, w, c). eye(P).reshape(patch, patch, C, P) sets
        # kernel[h, w, c, d] = 1 iff d == (h * patch + w) * C + c, i.e. the
        # output channel d reads patch element d under an HWC row-major flatten.
        kernel = np.eye(PATCH_DIM, dtype="float32").reshape(
            PATCH_SIZE, PATCH_SIZE, CHANNELS, PATCH_DIM
        )
        bias = np.zeros((PATCH_DIM,), dtype="float32")
        layer.proj.kernel.assign(kernel)
        layer.proj.bias.assign(bias)
        return layer

    def test_identity_kernel_reproduces_patchify_targets(self, sample_image):
        layer = self._identity_patch_embedding()
        batch = sample_image[None, ...]

        embedded = np.asarray(layer(batch))            # (1, 196, 768)
        targets = patchify_targets(batch, PATCH_SIZE).numpy()  # (1, 196, 768)

        assert embedded.shape == (1, NUM_PATCHES, PATCH_DIM)
        assert targets.shape == (1, NUM_PATCHES, PATCH_DIM)
        np.testing.assert_allclose(embedded, targets, atol=1e-5)

    def test_token_i_is_the_row_major_grid_cell_i(self, sample_image):
        """Independent check of token order against a plain numpy slice."""
        targets = patchify_targets(sample_image, PATCH_SIZE).numpy()  # (196, 768)
        assert targets.shape == (NUM_PATCHES, PATCH_DIM)
        for token in (0, 1, GRID, GRID + 1, NUM_PATCHES - 1):
            gh, gw = divmod(token, GRID)
            expected = sample_image[
                gh * PATCH_SIZE:(gh + 1) * PATCH_SIZE,
                gw * PATCH_SIZE:(gw + 1) * PATCH_SIZE,
                :,
            ].reshape(-1)  # numpy C-order == HWC row-major
            np.testing.assert_allclose(targets[token], expected, atol=0)

    def test_unbatched_and_batched_agree(self, sample_image):
        single = patchify_targets(sample_image, PATCH_SIZE).numpy()
        batched = patchify_targets(sample_image[None, ...], PATCH_SIZE).numpy()
        np.testing.assert_allclose(single, batched[0], atol=0)


class TestConstructionErrors:
    """Config errors must be raised EAGERLY, at pipeline-build time."""

    def test_non_divisible_image_size(self):
        with pytest.raises(ValueError, match="divisible"):
            make_masked_patch_map_fn(patch_size=16, image_size=225)

    def test_n_loss_zero(self):
        # 196 patches, ratio 0.002 -> round(0.392) = 0
        with pytest.raises(ValueError, match="n_loss=0"):
            make_masked_patch_map_fn(
                patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, mask_ratio=0.002
            )

    def test_n_input_equals_n_loss(self):
        """The rounding edge that silently deletes the 10% un-occluded signal."""
        # 4 patches (image 32, patch 16), ratio 0.5 -> n_loss=2; frac 0.9 -> round(1.8)=2
        with pytest.raises(ValueError, match="n_input == n_loss"):
            make_masked_patch_map_fn(
                patch_size=16,
                image_size=32,
                mask_ratio=0.5,
                mask_token_frac=MASK_TOKEN_FRAC,
            )

    def test_n_input_zero(self):
        with pytest.raises(ValueError, match="n_input=0"):
            make_masked_patch_map_fn(
                patch_size=PATCH_SIZE,
                image_size=IMAGE_SIZE,
                mask_ratio=MASK_RATIO,
                mask_token_frac=0.001,
            )

    def test_bad_ratios(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_ratio=1.5)
        with pytest.raises(ValueError, match="mask_token_frac"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_token_frac=1.5)

    def test_patchify_rejects_bad_rank_and_size(self, sample_image):
        with pytest.raises(ValueError, match="rank"):
            patchify_targets(np.zeros((10, 10), "float32"), PATCH_SIZE)
        with pytest.raises(ValueError, match="divisible"):
            patchify_targets(np.zeros((17, 17, 3), "float32"), PATCH_SIZE)
        with pytest.raises(ValueError, match="patch_size"):
            patchify_targets(sample_image, 0)


# ---------------------------------------------------------------------
# The legacy function is unchanged by the stateless one sharing its checks
# ---------------------------------------------------------------------

# Every refusal of the legacy function, with its COMPLETE message and the kwargs
# that raise it. The messages are copied from the source at the commit before the
# stateless function was added; a shared validation helper that rewords one of them
# changes what the four legacy trainers print.
_LEGACY_REFUSALS = [
    (dict(patch_size=0, image_size=224), "patch_size must be positive, got 0"),
    (dict(patch_size=16, image_size=0), "image_size must be positive, got 0"),
    (
        dict(patch_size=16, image_size=225),
        "image_size (225) must be divisible by patch_size (16)",
    ),
    (
        dict(patch_size=16, image_size=224, mask_ratio=0.0),
        "mask_ratio must be in (0, 1], got 0.0",
    ),
    (
        dict(patch_size=16, image_size=224, mask_token_frac=1.5),
        "mask_token_frac must be in (0, 1], got 1.5",
    ),
    (
        dict(patch_size=16, image_size=224, mask_ratio=0.002),
        "mask_ratio (0.002) gives n_loss=0 at num_patches=196: "
        "the loss would be computed on zero tokens",
    ),
    (
        dict(patch_size=16, image_size=224, mask_token_frac=0.001),
        "mask_token_frac (0.001) gives n_input=0 at n_loss=98: "
        "no token would ever be occluded and the task would be an identity map",
    ),
    (
        dict(patch_size=16, image_size=32, mask_ratio=0.5, mask_token_frac=0.9),
        "mask_token_frac (0.9) gives n_input == n_loss == 2 at num_patches=4: "
        "every loss token would be occluded, deleting the un-occluded patches the "
        "loss set must retain (the paper's 90/10 rule). "
        "Lower mask_token_frac or raise num_patches.",
    ),
]


class TestTheLegacyFunctionIsUnchanged:
    """The legacy stateful function keeps its messages, its log line and its output."""

    @pytest.mark.parametrize("kwargs,message", _LEGACY_REFUSALS)
    def test_every_legacy_refusal_keeps_its_complete_message(self, kwargs, message):
        with pytest.raises(ValueError) as raised:
            make_masked_patch_map_fn(**kwargs)
        assert str(raised.value) == message

    @pytest.mark.parametrize("kwargs,message", _LEGACY_REFUSALS)
    def test_the_stateless_function_refuses_the_same_configs_in_the_same_words(
        self, kwargs, message
    ):
        with pytest.raises(ValueError) as raised:
            make_stateless_masked_patch_map_fn(seed=0, salt="train", **kwargs)
        assert str(raised.value) == message

    def test_the_legacy_log_line_is_the_one_it_always_wrote(self, monkeypatch):
        lines = []
        monkeypatch.setattr(masked_patches.logger, "info", lines.append)
        make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, MASK_RATIO, MASK_TOKEN_FRAC, seed=7)
        assert lines == [
            "masked-patch map fn: image_size=224, patch_size=16, N=196, n_loss=98, "
            "n_input=88, loss_weight_scale=2.000000, seed=7"
        ]

    def test_the_legacy_output_stream_is_byte_for_byte_what_it_was(self):
        """sha256 over mask, weight, targets and image of six images at seed 7.

        The digest was taken at the commit before the stateless function existed.
        """
        import hashlib

        images = np.random.default_rng(5).uniform(
            size=(6, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        ).astype("float32")
        map_fn = make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, 0.5, 0.9, seed=7)
        ds = (
            tf.data.Dataset.from_tensor_slices(images)
            .map(map_fn, num_parallel_calls=1)
            .batch(3)
        )
        digest = hashlib.sha256()
        for (image, mask), targets, weight in ds:
            for part in (mask, weight, targets, image):
                digest.update(part.numpy().tobytes())
        assert digest.hexdigest()[:16] == "14c3dd39685789fb"


# ---------------------------------------------------------------------
# The stateless function: a mask that is a function of (seed, salt, index)
# ---------------------------------------------------------------------

_DIGEST_SOURCE = """
import hashlib
import tensorflow as tf
from dl_techniques.datasets.vision.masked_patches import make_stateless_masked_patch_map_fn

def digest():
    fn = make_stateless_masked_patch_map_fn(16, 224, 0.5, 0.9, seed=7, salt="train")
    image = tf.zeros((224, 224, 3))
    h = hashlib.sha256()
    for i in range(40):
        (_, mask), _, weight = fn(tf.constant(i, tf.int64), image)
        h.update(mask.numpy().tobytes())
        h.update(weight.numpy().tobytes())
    return h.hexdigest()
"""


def _stateless(seed: int = 7, salt: str = "train", **overrides):
    kwargs = dict(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        mask_ratio=MASK_RATIO,
        mask_token_frac=MASK_TOKEN_FRAC,
        seed=seed,
        salt=salt,
    )
    kwargs.update(overrides)
    return make_stateless_masked_patch_map_fn(**kwargs)


def _stateless_masks(fn, count: int, image: np.ndarray = None):
    """``(count, N)`` input masks and weights of ``fn`` over ``count`` rows of one image."""
    if image is None:
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32")
    labels = tf.data.Dataset.range(count)
    images = tf.data.Dataset.from_tensors(image).repeat(count)
    ds = (
        tf.data.Dataset.zip((images, labels))
        .enumerate()
        .map(lambda i, row: fn(i, row[0], row[1]), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
    )
    masks, weights = [], []
    for (_img, input_mask), _targets, loss_weight in ds:
        masks.append(input_mask.numpy())
        weights.append(loss_weight.numpy())
    return np.concatenate(masks), np.concatenate(weights)


class TestTheStatelessMaskIsAFunctionOfItsKey:
    def test_the_element_has_the_legacy_structure_and_the_legacy_weight(self, sample_image):
        legacy = make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, MASK_RATIO, MASK_TOKEN_FRAC, seed=3)
        (l_image, l_mask), l_targets, l_weight = legacy(tf.constant(sample_image))
        (image, mask), targets, weight = _stateless()(tf.constant(11, tf.int64), tf.constant(sample_image))

        np.testing.assert_array_equal(image.numpy(), l_image.numpy())
        np.testing.assert_array_equal(targets.numpy(), l_targets.numpy())
        assert (image.dtype, mask.dtype, targets.dtype, weight.dtype) == (
            l_image.dtype, l_mask.dtype, l_targets.dtype, l_weight.dtype,
        )
        assert mask.shape == l_mask.shape == (NUM_PATCHES,)
        assert weight.shape == l_weight.shape == (NUM_PATCHES,)
        # the legacy formula: 1{i in S} * N / n_loss
        nonzero = weight.numpy()[weight.numpy() != 0.0]
        assert nonzero.shape == (N_LOSS,)
        np.testing.assert_array_equal(nonzero, np.full(N_LOSS, np.float32(NUM_PATCHES / N_LOSS)))
        np.testing.assert_allclose(weight.numpy().sum(), float(NUM_PATCHES), atol=1e-3)

    def test_the_element_spec_is_fully_static_through_a_real_pipeline(self):
        ds = (
            tf.data.Dataset.from_tensors(np.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32"))
            .repeat(4)
            .enumerate()
            .map(_stateless())
            .batch(2)
        )
        (image, mask), targets, weight = ds.element_spec
        assert image.shape.as_list() == [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS]
        assert mask.shape.as_list() == [None, NUM_PATCHES] and mask.dtype == tf.bool
        assert targets.shape.as_list() == [None, NUM_PATCHES, PATCH_DIM]
        assert weight.shape.as_list() == [None, NUM_PATCHES] and weight.dtype == tf.float32

    def test_one_key_gives_one_mask_from_two_separately_built_functions(self):
        index = tf.constant(5, tf.int64)
        image = tf.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        first = _stateless()(index, image)[0][1].numpy()
        second = _stateless()(index, image)[0][1].numpy()
        np.testing.assert_array_equal(first, second)

    def test_two_passes_of_one_stream_are_equal_under_parallel_calls(self):
        first = _stateless_masks(_stateless(), 96)
        second = _stateless_masks(_stateless(), 96)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_the_mask_does_not_depend_on_the_image_or_the_label(self, sample_image):
        fn = _stateless()
        index = tf.constant(9, tf.int64)
        plain = fn(index, tf.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))[0][1].numpy()
        other = fn(index, tf.constant(sample_image), tf.constant(3))[0][1].numpy()
        np.testing.assert_array_equal(plain, other)

    def test_a_fresh_interpreter_gives_the_same_masks(self, tmp_path):
        namespace = {}
        exec(_DIGEST_SOURCE, namespace)
        here = namespace["digest"]()
        src = str(Path(dl_techniques.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
        env.pop("PYTHONHASHSEED", None)
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SOURCE + "\nprint('DIGEST', digest())"],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        lines = [l for l in out.stdout.splitlines() if l.startswith("DIGEST ")]
        assert lines == [f"DIGEST {here}"]

    def test_every_index_over_a_grid_gets_its_own_mask(self):
        masks, _ = _stateless_masks(_stateless(), 128)
        assert len({row.tobytes() for row in masks}) == 128

    def test_a_large_index_is_a_value_and_gets_its_own_mask(self):
        image = tf.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        fn = _stateless()
        rows = {
            fn(tf.constant(i, tf.int64), image)[0][1].numpy().tobytes()
            for i in (0, 1, 2**31, 2**31 + 1, 2**40, 2**40 + 1)
        }
        assert len(rows) == 6

    def test_a_different_salt_gives_a_different_mask_at_every_index(self):
        train, _ = _stateless_masks(_stateless(salt="train"), 32)
        select, _ = _stateless_masks(_stateless(salt="select"), 32)
        assert not np.any(np.all(train == select, axis=1))

    def test_a_different_seed_gives_a_different_mask_at_every_index(self):
        one, _ = _stateless_masks(_stateless(seed=1), 32)
        two, _ = _stateless_masks(_stateless(seed=2), 32)
        assert not np.any(np.all(one == two, axis=1))

    def test_the_seed_and_the_index_are_not_interchangeable(self):
        """(seed 3, index 5) and (seed 5, index 3) are two keys, not one sum."""
        image = tf.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        a = _stateless(seed=3)(tf.constant(5, tf.int64), image)[0][1].numpy()
        b = _stateless(seed=5)(tf.constant(3, tf.int64), image)[0][1].numpy()
        assert not np.array_equal(a, b)

    def test_seed_zero_is_a_seed(self):
        image = tf.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        zero = _stateless(seed=0)(tf.constant(1, tf.int64), image)[0][1].numpy()
        one = _stateless(seed=1)(tf.constant(1, tf.int64), image)[0][1].numpy()
        assert not np.array_equal(zero, one)

    def test_the_counts_are_exact_at_every_index_of_a_grid(self):
        for seed, salt in ((0, "train"), (123, "select"), (2**31 - 1, "test")):
            masks, weights = _stateless_masks(_stateless(seed=seed, salt=salt), 300)
            loss_set = weights > 0.0
            assert np.all(loss_set.sum(axis=1) == N_LOSS)
            assert np.all(masks.sum(axis=1) == N_INPUT)

    def test_the_input_mask_is_a_strict_subset_of_the_loss_set(self):
        masks, weights = _stateless_masks(_stateless(), 200)
        loss_set = weights > 0.0
        assert not np.any(masks & ~loss_set)
        assert np.all((loss_set & ~masks).sum(axis=1) == N_LOSS - N_INPUT)

    def test_the_input_masks_are_spread_over_the_grid_and_not_one_corner(self):
        masks, _ = _stateless_masks(_stateless(), 300)
        share = masks.mean(axis=0)
        assert share.min() > 0.30 and share.max() < 0.60  # expected 88 / 196 = 0.449

    def test_the_stateless_function_validates_its_key(self):
        for bad in (None, 1.5, "7", True):
            with pytest.raises(TypeError, match="seed"):
                _stateless(seed=bad)
        for bad in (None, "", 3):
            with pytest.raises(TypeError, match="salt"):
                _stateless(salt=bad)


class TestTheInstrumentSeesTheDefectOfTheLegacyFunction:
    """The legacy function draws the next numbers of one generator on every pass."""

    @staticmethod
    def _two_passes(element_fn, count: int = 16):
        """Input masks of two full passes over one ``(index, image)`` dataset."""
        ds = (
            tf.data.Dataset.from_tensors(
                np.zeros((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32")
            )
            .repeat(count)
            .enumerate()
            .map(element_fn)
            .batch(count)
        )
        passes = []
        for _ in range(2):
            (_img, mask), _targets, _weight = next(iter(ds))
            passes.append(mask.numpy())
        return passes

    def test_the_legacy_function_gives_a_different_mask_on_the_second_pass(self):
        legacy = make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, MASK_RATIO, MASK_TOKEN_FRAC, seed=7)
        first, second = self._two_passes(lambda _i, image: legacy(image))
        assert not np.any(np.all(first == second, axis=1))

    def test_the_stateless_function_gives_the_same_mask_on_the_second_pass(self):
        first, second = self._two_passes(_stateless())
        np.testing.assert_array_equal(first, second)
