"""Quirk guard: patchify and unpatchify agree on the SAME token order, and the
model's ``unpatchify`` binds the geometry it claims to.

**The lines this file pins.**
``src/dl_techniques/models/vision_language/dit/model.py``::

    x = keras.ops.reshape(tokens, (batch, h, w, p, p, c))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (batch, h * p, w * p, c))

re-derived (not transcribed) from upstream's NCHW ``'nhwpqc->nchpwq'``
(``reference/models.py``, ``DiT.unpatchify``), plus ``DiT.unpatchify``'s binding
``grid_height=self.grid_size, grid_width=self.grid_size,
patch_size=self.patch_size, channels=self.out_channels``.

**The plausible WRONG alternative.** The permutation ``(0, 2, 4, 1, 3, 5)``. It
produces an IDENTICALLY SHAPED tensor on a square grid and passes every shape
assertion; step 6 measured it RED at **3 failed** with every shape assertion
green.

**WHAT THIS FILE ADDS, and what it does not repeat.**
``test_dit_model.py::TestTheUnpatchifyOrientation`` already owns the
delta-impulse arm at an asymmetric coordinate on a NON-SQUARE ``2x3`` grid, the
exhaustive per-pixel census, and the shape-identity sibling -- all with the
destination index ``row = i*p + pi`` / ``col = j*p + pj`` computed independently
of the function under test. Re-writing those here would be two guards over one
claim. What step 6 left open is the property BEYOND the free function:

1. **Patchify's token order is the same convention unpatchify consumes.** Both
   halves could be row-major, both could be column-major, or they could
   DISAGREE -- and a disagreement is invisible to any test of either half alone.
   The arms here send a spatial delta impulse through the model's own
   ``x_embedder`` and locate the responding TOKEN by an independently computed
   index ``token = patch_row * grid_width + patch_col``, then push a one-hot
   token back through ``unpatchify`` and check it lands on the same patch.
2. **The model's method binds the right geometry.** ``channels`` must be
   ``out_channels`` (``2 * C`` under ``learn_sigma``), not ``in_channels``, and
   ``patch_size`` must be the model's -- an argument-binding slip changes no
   output shape when the grid is square and the payload width happens to match.
3. **The composition is the identity in patch coordinates**, which is the claim
   an end-to-end reader actually needs.

**RED proof (step 10).** Two injections into ``model.py``:

* the transposed permutation ``(0, 2, 4, 1, 3, 5)`` -- **3 failed / 12 passed**:
  ``test_a_one_hot_token_lands_on_the_same_patch[0-3]`` and ``[2-1]`` (the
  asymmetric coordinates; the diagonal ``[0-0]``/``[3-3]`` cases stay green,
  which is the symmetry the parametrization exists to defeat) and
  ``test_a_spatial_impulse_reaches_its_own_patch_in_the_output``.
* ``DiT.unpatchify`` binding ``channels=self.in_channels`` -- **14 failed /
  1 passed**, a loud arithmetic failure rather than a silent one.
"""

from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.model import DiT, unpatchify_tokens

from ._dit_helpers import TINY, built_model, np_


def token_index(patch_row: int, patch_col: int, grid_width: int) -> int:
    """Row-major token index. Written out, never read back from the code."""
    return patch_row * grid_width + patch_col


# ---------------------------------------------------------------------
# Patchify and unpatchify share ONE token order
# ---------------------------------------------------------------------


class TestTheTwoHalvesAgreeOnTheTokenOrder:
    """A disagreement between the two ends is invisible to either end alone."""

    @staticmethod
    def _impulse_model() -> Tuple[DiT, int, int]:
        """A model on a square grid -- the only shape ``DiT`` builds -- plus its
        grid size. The NON-square case is covered by the free-function arms."""
        model = built_model(seed=0)
        grid = model.input_size // model.patch_size
        return model, grid, model.patch_size

    @pytest.mark.parametrize("patch_row,patch_col", [(0, 0), (0, 3), (2, 1), (3, 3)])
    def test_a_spatial_impulse_wakes_the_independently_indexed_token(
        self, patch_row: int, patch_col: int
    ) -> None:
        """Patchify's token axis is ROW-MAJOR over the patch grid.

        The responding token is located by ``patch_row * grid + patch_col``,
        computed here; nothing re-invokes the model's own index arithmetic. A
        transposed patchify would light up ``patch_col * grid + patch_row``,
        which on the asymmetric coordinates above is a different token.
        """
        model, grid, patch = self._impulse_model()
        x = np.zeros(
            (1, model.input_size, model.input_size, model.in_channels), "float32"
        )
        x[0, patch_row * patch, patch_col * patch, :] = 1.0

        tokens = np_(model.x_embedder(keras.ops.convert_to_tensor(x), training=False))
        baseline = np_(
            model.x_embedder(
                keras.ops.convert_to_tensor(np.zeros_like(x)), training=False
            )
        )
        moved = np.max(np.abs(tokens - baseline), axis=-1)[0]

        expected = token_index(patch_row, patch_col, grid)
        assert int(np.argmax(moved)) == expected, moved
        assert float(moved[expected]) > 0.0
        # Exactly ONE token responds: the impulse sits inside a single patch.
        assert int(np.count_nonzero(moved > 1e-8)) == 1, moved

    @pytest.mark.parametrize("patch_row,patch_col", [(0, 0), (0, 3), (2, 1), (3, 3)])
    def test_a_one_hot_token_lands_on_the_same_patch(
        self, patch_row: int, patch_col: int
    ) -> None:
        """The return trip: ``unpatchify`` reads the SAME row-major token axis."""
        model, grid, patch = self._impulse_model()
        payload = patch * patch * model.out_channels
        tokens = np.zeros((1, grid * grid, payload), "float32")
        tokens[0, token_index(patch_row, patch_col, grid), :] = 1.0

        image = np_(model.unpatchify(keras.ops.convert_to_tensor(tokens)))
        rows = slice(patch_row * patch, (patch_row + 1) * patch)
        cols = slice(patch_col * patch, (patch_col + 1) * patch)

        assert np.all(image[0, rows, cols, :] == 1.0)
        # And nothing outside that patch moved.
        mask = np.zeros(image.shape[1:3], bool)
        mask[rows, cols] = True
        assert float(np.max(np.abs(image[0][~mask]))) == 0.0

    def test_the_two_conventions_would_disagree_if_either_flipped(self) -> None:
        """Anti-vacuity: the transposed token index is a DIFFERENT token here."""
        _, grid, _ = self._impulse_model()
        for patch_row, patch_col in [(0, 3), (2, 1), (1, 2)]:
            assert token_index(patch_row, patch_col, grid) != token_index(
                patch_col, patch_row, grid
            )


# ---------------------------------------------------------------------
# The model's method binds the geometry it claims to
# ---------------------------------------------------------------------


class TestTheMethodBindsTheModelsOwnGeometry:
    """``channels=out_channels``, ``patch_size=self.patch_size``, square grid."""

    def test_the_channel_count_is_out_channels_not_in_channels(self) -> None:
        model = built_model(seed=0)
        assert model.out_channels == 2 * model.in_channels
        grid = model.input_size // model.patch_size
        payload = model.patch_size ** 2 * model.out_channels
        tokens = np.random.default_rng(0).standard_normal(
            (2, grid * grid, payload)
        ).astype("float32")

        image = np_(model.unpatchify(keras.ops.convert_to_tensor(tokens)))
        assert image.shape == (
            2, model.input_size, model.input_size, model.out_channels
        )

    def test_a_learn_sigma_false_model_unpatchifies_to_c_channels(self) -> None:
        """The binding tracks the config rather than a hard-coded ``2 * C``."""
        model = built_model(seed=0, learn_sigma=False)
        assert model.out_channels == model.in_channels
        grid = model.input_size // model.patch_size
        payload = model.patch_size ** 2 * model.out_channels
        tokens = np.zeros((1, grid * grid, payload), "float32")
        image = np_(model.unpatchify(keras.ops.convert_to_tensor(tokens)))
        assert image.shape[-1] == model.in_channels

    def test_it_equals_the_free_function_with_the_models_geometry(self) -> None:
        """The binding is checked ARGUMENT BY ARGUMENT, not just by shape."""
        model = built_model(seed=0)
        grid = model.input_size // model.patch_size
        payload = model.patch_size ** 2 * model.out_channels
        tokens = np.random.default_rng(1).standard_normal(
            (2, grid * grid, payload)
        ).astype("float32")
        tensor = keras.ops.convert_to_tensor(tokens)

        np.testing.assert_allclose(
            np_(model.unpatchify(tensor)),
            np_(
                unpatchify_tokens(
                    tensor,
                    grid_height=grid,
                    grid_width=grid,
                    patch_size=model.patch_size,
                    channels=model.out_channels,
                )
            ),
            rtol=0,
            atol=0.0,
        )

    def test_a_wrong_patch_size_binding_would_not_even_be_shape_legal(self) -> None:
        """Why a shape check is not enough for the CHANNELS argument, but is for
        ``patch_size``: swapping ``p`` for ``2 * p`` fails to divide the payload.

        Stated as an executable arm so the asymmetry between the two arguments
        is on record: ``patch_size`` is caught by arithmetic, ``channels`` is
        not, which is why the channel arms above exist.
        """
        model = built_model(seed=0)
        grid = model.input_size // model.patch_size
        payload = model.patch_size ** 2 * model.out_channels
        tokens = keras.ops.convert_to_tensor(
            np.zeros((1, grid * grid, payload), "float32")
        )
        with pytest.raises(Exception):
            unpatchify_tokens(
                tokens,
                grid_height=grid,
                grid_width=grid,
                patch_size=2 * model.patch_size,
                channels=model.out_channels,
            )


# ---------------------------------------------------------------------
# The end-to-end composition is spatially faithful
# ---------------------------------------------------------------------


class TestTheCompositionIsSpatiallyFaithful:
    """Patch ``(i, j)`` in, patch ``(i, j)`` out -- through the real forward path."""

    def test_a_spatial_impulse_reaches_its_own_patch_in_the_output(self) -> None:
        """The whole model, woken, with the read-out replaced by a token copier.

        A DiT emits exactly ``0.0`` at init and its blocks mix tokens once
        awake, so the honest end-to-end orientation probe is the one that keeps
        the token axis intact: patchify the impulse, hand the resulting per-token
        activity straight to ``unpatchify`` as a payload, and check which patch
        lights up. That composes the two halves under test and nothing else.
        """
        model = built_model(seed=0)
        grid = model.input_size // model.patch_size
        patch = model.patch_size
        patch_row, patch_col = 1, 3
        assert patch_row != patch_col, "the coordinate must be asymmetric"

        x = np.zeros(
            (1, model.input_size, model.input_size, model.in_channels), "float32"
        )
        x[0, patch_row * patch, patch_col * patch, :] = 1.0

        tokens = np_(model.x_embedder(keras.ops.convert_to_tensor(x), training=False))
        baseline = np_(
            model.x_embedder(
                keras.ops.convert_to_tensor(np.zeros_like(x)), training=False
            )
        )
        activity = np.max(np.abs(tokens - baseline), axis=-1)[0]

        payload = patch * patch * model.out_channels
        carried = np.broadcast_to(
            activity[None, :, None], (1, grid * grid, payload)
        ).astype("float32")
        image = np_(model.unpatchify(keras.ops.convert_to_tensor(carried)))

        per_patch = image[0].reshape(grid, patch, grid, patch, model.out_channels)
        per_patch = np.max(np.abs(per_patch), axis=(1, 3, 4))
        assert np.unravel_index(int(np.argmax(per_patch)), per_patch.shape) == (
            patch_row,
            patch_col,
        ), per_patch

    def test_the_output_grid_matches_the_input_grid(self) -> None:
        model = built_model(seed=0)
        rng = np.random.default_rng(3)
        x = rng.normal(
            size=(2, model.input_size, model.input_size, model.in_channels)
        ).astype("float32")
        t = rng.integers(0, 1000, size=(2,)).astype("float32")
        y = rng.integers(0, TINY["num_classes"], size=(2,)).astype("int32")
        out = np_(model([x, t, y], training=False))
        assert out.shape[1:3] == x.shape[1:3]
