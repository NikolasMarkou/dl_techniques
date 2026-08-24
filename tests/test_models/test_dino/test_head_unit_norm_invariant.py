"""``DINOHead``'s unit-norm invariant must hold in the model users actually build.

D-011 promises that with ``norm_last_layer=True`` every prototype column of
``last_layer.kernel`` has norm 1 even for a never-trained head. A Keras
constraint is applied by the OPTIMIZER, so the invariant is established by a
one-off normalization in ``DINOHead.build``.

That normalization is a ``.assign()`` inside ``build()`` — the shape that was
found DEAD in six other places in this repo (D-021, F-9), because Keras 3 runs
the symbolic build pass of a sublayer first reached from a parent's ``call()``
inside a ``StatelessScope`` that records and discards assigns. The iteration-1
adversarial review flagged this site as a likely seventh.

**Measured 2026-08-17, CPU, and the flag is REFUTED for the shipped path.**
Through a full ``DINOv1(include_projection_head=True)`` the column norms are
[0.999999, 1.000000] (max|norm - 1| = 8.3e-07): the assign survives, because
``DINOv1`` reaches the head in a way that builds it directly rather than from
inside a parent layer's ``call()``.

The hole is real but latent: the same head built from inside a parent LAYER's
``call()`` gives column norms of 0.119-0.245 (max|norm - 1| = 0.881). Nothing in
the tree does that today. This module pins the invariant on the path that ships,
so a future restructuring of ``DINOv1`` that moves the head onto the stateless
path fails here instead of silently voiding D-011.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.dino.dino_v1 import DINOv1, DINOHead


def _column_norms(head):
    kernel = ops.convert_to_numpy(head.last_layer.kernel)
    # UnitNorm(axis=0) normalizes each COLUMN of a Dense kernel, one per
    # prototype.
    return np.linalg.norm(kernel, axis=0)


class TestHeadUnitNormInvariant:

    def test_a_never_trained_head_already_satisfies_the_invariant(self):
        """The D-011 promise, on the real model path."""
        keras.utils.set_random_seed(0)
        model = DINOv1(
            embed_dim=32, depth=1, num_heads=2, image_size=32, patch_size=16,
            include_projection_head=True, dino_out_dim=64,
            dino_hidden_dim=32, dino_bottleneck_dim=16,
        )
        model(ops.zeros((2, 32, 32, 3)))

        norms = _column_norms(model.head)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

        # Anti-vacuity: an all-zero kernel would give norms of 0, and a
        # single-column kernel would make this trivial.
        assert norms.size == 64, norms.size

    def test_norm_last_layer_false_does_not_normalize(self):
        """The isolating case: without the flag the invariant must NOT hold, or
        the test above is measuring something other than the flag."""
        keras.utils.set_random_seed(0)
        head = DINOHead(
            in_dim=32, out_dim=64, hidden_dim=32, bottleneck_dim=16,
            norm_last_layer=False,
        )
        head.build((None, 32))

        norms = _column_norms(head)
        assert not np.allclose(norms, 1.0, atol=1e-3), (
            "columns are unit-norm with norm_last_layer=False; the flag is "
            "not what establishes the invariant"
        )

    @pytest.mark.parametrize("norm_last_layer", [True, False])
    def test_the_flag_reaches_the_constraint(self, norm_last_layer):
        """The constraint itself must be attached or absent accordingly."""
        head = DINOHead(
            in_dim=32, out_dim=64, hidden_dim=32, bottleneck_dim=16,
            norm_last_layer=norm_last_layer,
        )
        head.build((None, 32))

        constraint = head.last_layer.kernel_constraint
        if norm_last_layer:
            assert isinstance(constraint, keras.constraints.UnitNorm)
            assert constraint.axis == 0
        else:
            assert constraint is None
