"""``visualize_reconstruction``: the grid LAYOUT, and the GUI it must never open.

Why this file exists
--------------------
``utils.visualize_reconstruction`` had zero coverage anywhere in the tree. Its
sibling ``create_mae_model`` does not: ``test_scale_contract.py`` already drives
its validation contract end to end, so this file deliberately does not duplicate
that and covers only the delegation itself.

Two things are worth pinning about the function.

**It is not a plotting helper.** Despite the name it imports no pyplot and opens
no figure -- it returns a composite ``np.ndarray`` of shape
``(num_samples * H, 3 * W, C)`` laid out ``[original | masked | reconstructed]``
per row. That contract is easy to break silently: the implementation reaches it
via ``transpose(0, 2, 1, 3, 4)`` then ``reshape``, and swapping those two axes
produces an array of the *identical shape* holding interleaved garbage. So the
arms below assert CONTENT at known offsets -- column 0 of row ``i`` must be
``clip(images[i], 0, 1)`` bit-for-bit -- never the shape alone.

**It must not block on a GUI.** The repo forbids library code that opens or shows
a window (a training run would hang on a headless host). ``plt.show`` is patched
to a recorder and the live figure set is compared before and after, so the arm
fails whether the function calls ``show()`` or merely leaks a figure.
"""

import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder import (
    MaskedAutoencoder,
    create_mae_model,
    visualize_reconstruction,
)

from .conftest import tiny_encoder

IMAGE_SIZE, PATCH_SIZE, CHANNELS = 32, 16, 3
SEED = 20260823


@pytest.fixture(scope="module")
def mae() -> MaskedAutoencoder:
    import keras

    keras.utils.set_random_seed(SEED)
    model = create_mae_model(
        encoder=tiny_encoder(image_size=IMAGE_SIZE, channels=CHANNELS),
        patch_size=PATCH_SIZE,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    )
    model(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32"), training=False)
    return model


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    """Six distinguishable images inside [0, 1], so clipping is a no-op on them."""
    rng = np.random.default_rng(SEED)
    return rng.uniform(0.05, 0.95, (6, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype(
        "float32"
    )


class TestCreateMaeModelDelegates:
    """`create_mae_model` is a passthrough; its validation is pinned elsewhere."""

    def test_it_returns_a_masked_autoencoder_carrying_the_arguments(self, mae):
        assert isinstance(mae, MaskedAutoencoder)
        assert mae.patch_size == PATCH_SIZE
        assert mae.input_shape_config == (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_a_non_model_encoder_is_refused(self):
        with pytest.raises(TypeError, match="encoder must be a keras.Model"):
            create_mae_model(encoder="not a model")


class TestTheGridLayout:
    @pytest.mark.parametrize("num_samples", [1, 2, 4])
    def test_the_grid_is_num_samples_rows_by_three_columns(
        self, mae, images, num_samples
    ):
        grid = visualize_reconstruction(mae, images, num_samples=num_samples)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (
            num_samples * IMAGE_SIZE,
            3 * IMAGE_SIZE,
            CHANNELS,
        ), grid.shape

    def test_num_samples_is_clamped_to_the_batch(self, mae, images):
        """Asking for more rows than there are images must not raise or pad."""
        grid = visualize_reconstruction(mae, images, num_samples=len(images) + 10)
        assert grid.shape[0] == len(images) * IMAGE_SIZE

    def test_the_first_column_is_the_original_image_of_that_row(self, mae, images):
        """The layout claim, asserted at exactly 0.0 -- not a correlation."""
        num_samples = 3
        grid = visualize_reconstruction(mae, images, num_samples=num_samples)
        for row in range(num_samples):
            tile = grid[
                row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE, 0:IMAGE_SIZE, :
            ]
            expected = np.clip(images[row], 0.0, 1.0)
            delta = float(np.max(np.abs(tile - expected)))
            assert delta == 0.0, (
                f"row {row}, column 0 is not image {row}: max|delta| = "
                f"{delta:.6e}. Column order is [original | masked | "
                f"reconstructed]; a transposed reshape yields this exact shape "
                f"with interleaved content."
            )

    def test_the_rows_are_distinct(self, mae, images):
        """Anti-vacuity: a broadcast of image 0 into every row would not be caught
        by a per-row check that only ever looks at row 0."""
        grid = visualize_reconstruction(mae, images, num_samples=3)
        rows = [grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE] for i in range(3)]
        assert not np.array_equal(rows[0], rows[1])
        assert not np.array_equal(rows[1], rows[2])

    def test_the_second_and_third_columns_are_not_the_first(self, mae, images):
        """The masked and reconstructed views must differ from the original."""
        grid = visualize_reconstruction(mae, images, num_samples=2)
        original = grid[:, 0:IMAGE_SIZE, :]
        masked = grid[:, IMAGE_SIZE:2 * IMAGE_SIZE, :]
        reconstructed = grid[:, 2 * IMAGE_SIZE:3 * IMAGE_SIZE, :]
        assert not np.array_equal(original, masked), (
            "column 1 equals column 0; `visualize` was called without masking, "
            "so the middle panel shows nothing."
        )
        assert not np.array_equal(original, reconstructed)


class TestTheOutputRange:
    def test_the_grid_is_clipped_to_the_unit_interval(self, mae, images):
        """Reconstructions are unbounded; the returned grid must not be."""
        grid = visualize_reconstruction(mae, images, num_samples=4)
        assert float(grid.min()) >= 0.0
        assert float(grid.max()) <= 1.0

    def test_out_of_range_inputs_are_clipped_rather_than_rescaled(self, mae):
        """The control for the arm above: clipping is SATURATION, not normalization.

        A hidden min-max rescale would also land inside [0, 1] -- and would move
        the in-range pixels, which this arm forbids.
        """
        rng = np.random.default_rng(SEED)
        wild = rng.uniform(0.2, 0.8, (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype(
            "float32"
        )
        wild[0, 0, 0, :] = -5.0
        wild[0, 0, 1, :] = 5.0

        grid = visualize_reconstruction(mae, wild, num_samples=1)
        assert np.all(grid[0, 0, :] == 0.0), grid[0, 0, :]
        assert np.all(grid[0, 1, :] == 1.0), grid[0, 1, :]
        # The untouched interior pixel must survive unchanged, which a rescale
        # to [0, 1] over a [-5, 5] range would not allow.
        assert float(np.max(np.abs(grid[2, 2, :] - wild[0, 2, 2, :]))) == 0.0


class TestItNeverOpensAGui:
    """Library code that blocks on a window hangs a headless training host."""

    def test_no_figure_is_shown_and_none_is_leaked(self, mae, images, monkeypatch):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append((a, k)))
        before = set(plt.get_fignums())

        visualize_reconstruction(mae, images, num_samples=2)

        assert calls == [], (
            "`visualize_reconstruction` called `plt.show()`. It returns an array "
            "for the CALLER to render; showing a window here blocks any headless "
            "run that imports it."
        )
        assert set(plt.get_fignums()) == before, (
            "a matplotlib figure was created and left open; repeated calls would "
            "leak figures until matplotlib's own open-figure warning fires."
        )

    def test_the_probe_can_see_a_show_call(self, monkeypatch):
        """The control: without it the arm above passes when the patch misses."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append((a, k)))
        plt.show()
        assert calls != []
