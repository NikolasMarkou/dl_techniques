"""RED proof: KAN knot grids must survive Keras 3's stateless build pass.

``KANvolution.grid`` and ``KANLinear.grid`` were created with
``add_weight(initializer='zeros')`` and then written by an ``.assign()`` issued from
``build()`` (for ``KANLinear``, indirectly via ``_set_grid_from_range``). Keras 3 runs
a symbolic build pass inside a ``StatelessScope`` whenever a layer is first reached
from a PARENT layer's ``call()`` -- i.e. in every real model, and on every
``create_ffn_layer(ffn_type='kan')`` path -- and that scope RECORDS the assign and
then DISCARDS it. The knot vector stayed all zeros, collapsing every knot onto a
single point, so the B-spline basis (linear for ``KANvolution``, Cox-de Boor for
``KANLinear``) ran over a degenerate knot vector.

**Every test here builds the layer through a parent layer's ``call()``.** A test that
calls ``layer.build(...)`` directly is precisely the test that missed this: the direct
path never enters the stateless scope and always looked correct.

Assertions compare the grid against its closed form (a linspace with known endpoints
and known uniform spacing), not against a shape and not against "is non-zero" -- an
interior knot is legitimately 0.0 in both layers, so only the full value comparison
discriminates.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.convolutional_kan import KANvolution
from dl_techniques.layers.ffn.kan_linear import KANLinear


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` reaches the child, forcing the stateless build."""

    def __init__(self, child: keras.layers.Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


def _build_through_parent(child: keras.layers.Layer, input_shape) -> keras.layers.Layer:
    parent = _Parent(child)
    parent(keras.Input(shape=input_shape))
    assert child.built, "child was not built through the parent's call()"
    return child


class TestKANvolutionGrid:
    """``KANvolution.grid`` must be the uniform [-1, 1] knot vector after a parent build."""

    def test_grid_equals_the_unit_linspace(self) -> None:
        grid_size = 5
        layer = _build_through_parent(
            KANvolution(filters=4, kernel_size=(3, 3), grid_size=grid_size), (8, 8, 3)
        )
        expected = np.linspace(-1.0, 1.0, grid_size + 1).astype(np.float32)
        np.testing.assert_allclose(np.asarray(layer.grid), expected, atol=1e-6)

    def test_knots_do_not_collapse_onto_one_point(self) -> None:
        layer = _build_through_parent(
            KANvolution(filters=4, kernel_size=(3, 3), grid_size=5), (8, 8, 3)
        )
        spacing = np.diff(np.asarray(layer.grid))
        np.testing.assert_allclose(spacing, np.full(5, 0.4, dtype=np.float32), atol=1e-6)


class TestKANLinearGrid:
    """``KANLinear.grid`` must be the extended knot sequence after a parent build."""

    @staticmethod
    def _expected(grid_range, grid_size: int, spline_order: int) -> np.ndarray:
        inner = np.linspace(grid_range[0], grid_range[1], grid_size + 1)
        h = inner[1] - inner[0]
        left = np.arange(-spline_order, 0) * h + inner[0]
        right = np.arange(1, spline_order + 1) * h + inner[-1]
        return np.concatenate([left, inner, right]).astype(np.float32)

    def test_grid_equals_the_extended_knot_sequence(self) -> None:
        layer = _build_through_parent(
            KANLinear(features=6, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)),
            (4,),
        )
        np.testing.assert_allclose(
            np.asarray(layer.grid), self._expected((-2.0, 2.0), 5, 3), atol=1e-5
        )

    def test_grid_honours_a_non_default_range(self) -> None:
        layer = _build_through_parent(
            KANLinear(features=6, grid_size=4, spline_order=2, grid_range=(0.5, 3.5)),
            (4,),
        )
        np.testing.assert_allclose(
            np.asarray(layer.grid), self._expected((0.5, 3.5), 4, 2), atol=1e-5
        )

    def test_runtime_grid_adaptation_still_assigns(self) -> None:
        """The initializer must not disable ``update_grid_from_samples``.

        That path issues a REAL ``.assign()`` from user code in a real scope, long
        after ``build()``, and is a separate writer from the initial value.
        """
        layer = _build_through_parent(
            KANLinear(features=6, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)),
            (4,),
        )
        before = np.asarray(layer.grid).copy()
        samples = np.linspace(-9.0, 9.0, 64 * 4).reshape(64, 4).astype("float32")
        layer.update_grid_from_samples(keras.ops.convert_to_tensor(samples))
        after = np.asarray(layer.grid)

        assert not np.allclose(before, after), "update_grid_from_samples did not move the grid"
        assert float(after.min()) < -6.0, (
            f"adapted grid did not track the sample range, min={after.min()}"
        )
        assert float(after.max()) > 6.0, (
            f"adapted grid did not track the sample range, max={after.max()}"
        )
