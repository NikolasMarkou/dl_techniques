import pytest
import numpy as np

from dl_techniques.layers.embedding.sincos_pos_embed_3d import get_3d_sincos_pos_embed
from dl_techniques.layers.embedding.sincos_pos_embed_2d import (
    get_1d_sincos_pos_embed_from_grid,
)


class TestGet3DSincosPosEmbed:
    """Comprehensive test suite for get_3d_sincos_pos_embed (pure NumPy)."""

    def test_output_shape_no_cls(self):
        embed_dim = 16
        grid_size = 4
        grid_depth = 2

        table = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)

        assert table.shape == (grid_depth * grid_size * grid_size, embed_dim)

    def test_output_shape_with_cls_token(self):
        embed_dim = 16
        grid_size = 4
        grid_depth = 2

        table = get_3d_sincos_pos_embed(
            embed_dim, grid_size, grid_depth, cls_token=True
        )

        assert table.shape == (1 + grid_depth * grid_size * grid_size, embed_dim)

    def test_cls_token_row_is_zero(self):
        embed_dim = 16
        table = get_3d_sincos_pos_embed(
            embed_dim, grid_size=4, grid_depth=2, cls_token=True
        )

        np.testing.assert_array_equal(table[0], np.zeros(embed_dim))
        # And the CLS row is the ONLY structurally-guaranteed-zero row: at
        # least one later row must be non-zero, or this assertion would be
        # vacuously satisfied by an all-zero table.
        assert not np.allclose(table[1:], 0.0)

    def test_matches_manual_rederivation_default_bands(self):
        """Manually re-derive the (non-uniform-power) band split and assembly."""
        embed_dim = 16
        grid_size = 3
        grid_depth = 2

        grid_d = np.arange(grid_depth, dtype=float)
        grid_h = np.arange(grid_size, dtype=float)
        grid_w = np.arange(grid_size, dtype=float)
        gh, gd, gw = np.meshgrid(grid_h, grid_d, grid_w)

        h_dim = embed_dim // 4
        w_dim = embed_dim // 4
        d_dim = embed_dim // 2

        emb_h = get_1d_sincos_pos_embed_from_grid(h_dim, gh)
        emb_w = get_1d_sincos_pos_embed_from_grid(w_dim, gw)
        emb_d = get_1d_sincos_pos_embed_from_grid(d_dim, gd)
        expected = np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]

        actual = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)

        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_uniform_power_band_split(self):
        """uniform_power=True must use equal (rounded) bands for all 3 axes."""
        embed_dim = 12
        grid_size = 2
        grid_depth = 2

        grid_d = np.arange(grid_depth, dtype=float)
        grid_h = np.arange(grid_size, dtype=float)
        grid_w = np.arange(grid_size, dtype=float)
        gh, gd, gw = np.meshgrid(grid_h, grid_d, grid_w)

        band = int(np.ceil(embed_dim / 6) * 2)
        emb_h = get_1d_sincos_pos_embed_from_grid(band, gh)
        emb_w = get_1d_sincos_pos_embed_from_grid(band, gw)
        emb_d = get_1d_sincos_pos_embed_from_grid(band, gd)
        expected = np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]

        actual = get_3d_sincos_pos_embed(
            embed_dim, grid_size, grid_depth, uniform_power=True
        )

        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_frame_axis_actually_varies_the_table(self):
        """Two grid positions differing ONLY in frame_id must get different rows."""
        embed_dim = 16
        grid_size = 3
        grid_depth = 3

        table = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)
        # Token flatten order is depth-major, height-mid, width-minor (see the
        # module docstring): row index = d * (grid_size**2) + h * grid_size + w.
        row_d0 = table[0 * grid_size * grid_size + 0 * grid_size + 0]
        row_d1 = table[1 * grid_size * grid_size + 0 * grid_size + 0]

        assert not np.allclose(row_d0, row_d1)

    def test_no_nans_or_infs(self):
        table = get_3d_sincos_pos_embed(embed_dim=32, grid_size=5, grid_depth=4)
        assert not np.any(np.isnan(table))
        assert not np.any(np.isinf(table))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
