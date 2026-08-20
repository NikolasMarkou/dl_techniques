"""
``PowerMLPLayer``'s two branches must not start as the same function.

R-123, batch 8: ``powermlp_hidden_1/main_dense/kernel == basis_dense/kernel`` at
``(6,4)``. MEASURED here BEFORE the fix at BOTH ``(6,4)`` and ``(6,12)``:
``max|delta|`` exactly **0.000000e+00**, i.e. shape-independently identical.
AFTER: **1.441834e+00** and **1.789001e+00**.

The cause was one shared seedless initializer instance handed to both
``Dense`` layers -- Keras 3 behaviour, a defect only because these two branches
are what the PowerMLP architecture IS. See decisions.md D-057.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer


def _kernels(units, in_dim, seed=1234):
    keras.utils.set_random_seed(seed)
    layer = PowerMLPLayer(units=units)
    layer.build((None, in_dim))
    by_name = {w.path.rsplit("/", 2)[-2]: np.asarray(ops.convert_to_numpy(w))
               for w in layer.weights if w.path.endswith("kernel")}
    return by_name["main_dense"], by_name["basis_dense"]


@pytest.mark.parametrize("units,in_dim", [(4, 6), (12, 6), (8, 8)])
def test_the_main_and_basis_kernels_do_not_start_identical(units, in_dim):
    main, basis = _kernels(units, in_dim)
    assert main.shape == basis.shape, "the arms must be comparable for this to mean anything"
    delta = float(np.abs(main - basis).max())
    assert delta > 0.0, (
        f"main_dense/kernel == basis_dense/kernel at {main.shape}: the two "
        "branches whose difference IS the architecture start as one function"
    )


def test_an_explicitly_SEEDED_initializer_still_ties_the_branches_by_design():
    """
    The clone preserves a caller-supplied seed, so this is the documented,
    intended behaviour -- and pinning it is what stops a future 'fix' from
    quietly removing reproducibility.
    """
    keras.utils.set_random_seed(1234)
    layer = PowerMLPLayer(units=4,
                          kernel_initializer=keras.initializers.GlorotUniform(seed=7))
    layer.build((None, 6))
    by_name = {w.path.rsplit("/", 2)[-2]: np.asarray(ops.convert_to_numpy(w))
               for w in layer.weights if w.path.endswith("kernel")}
    assert np.array_equal(by_name["main_dense"], by_name["basis_dense"])
