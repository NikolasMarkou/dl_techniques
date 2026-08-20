"""REF-9 at `GraphNeuralNetworkLayer`: every mode's weights must SURVIVE a save.

Keras 3.8 does not write a layer container nested two or more levels deep to
`model.weights.h5` when its owner is a `keras.layers.Layer`. This class is a
`Layer`, and it held GraphSAGE's two per-block transforms as a list of DICTS of
`Dense`. MEASURED before the repair (`num_layers=2`, `concept_dim=4`):

    mode        weights  archived  max|dW| reload   max|dOut| reload
    gcn              16        16  0.000000e+00     0.000000e+00
    graphsage        20        12  1.407037e+00     1.659781e+00
    gat              28        28  0.000000e+00     0.000000e+00
    gin              21        21  0.000000e+00     0.000000e+00

i.e. ALL EIGHT GraphSAGE `Dense` tensors were missing and the reloaded model
was a DIFFERENT model. After the repair (two flat per-role lists) graphsage is
20/20 at exactly 0.0 in both arms, and the other three are unchanged.

Both arms are here on purpose. A save-side count alone cannot see a LOAD-side
loss, and a reload comparison alone cannot distinguish "restored correctly"
from "never perturbed" -- so the donor's weights are ALL perturbed first and
the perturbation is asserted to have happened.
"""

import os
import tempfile
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.layers.graphs.graph_neural_network import (
    GraphNeuralNetworkLayer,
)

MODES = ("gcn", "graphsage", "gat", "gin")
N, F, D = 6, 4, 4  # F == D: the norm layers are built on `node_shape`, so a
                   # feature width other than `concept_dim` raises inside
                   # `LayerNormalization` -- a separate, pre-existing defect,
                   # not this file's subject.


def _model(mode):
    nodes = keras.Input((N, F))
    adj = keras.Input((N, N))
    out = GraphNeuralNetworkLayer(
        concept_dim=D, num_layers=2, message_passing=mode, dropout_rate=0.0,
        name="gnn",
    )([nodes, adj])
    return keras.Model([nodes, adj], out)


def _inputs():
    return [
        np.random.RandomState(0).randn(2, N, F).astype("float32"),
        np.random.RandomState(1).rand(2, N, N).astype("float32"),
    ]


def _archived_dataset_count(path):
    with zipfile.ZipFile(path) as z:
        payload = z.read("model.weights.h5")
    scratch = path + ".weights.h5"
    with open(scratch, "wb") as fh:
        fh.write(payload)
    names = []
    try:
        with h5py.File(scratch, "r") as f:
            f.visititems(
                lambda k, v: names.append(k)
                if isinstance(v, h5py.Dataset) else None
            )
    finally:
        os.remove(scratch)
    return len(names)


@pytest.mark.parametrize("mode", MODES)
def test_every_weight_reaches_the_archive_and_survives_a_reload(mode):
    keras.utils.set_random_seed(7)
    model = _model(mode)
    xs = _inputs()
    model(xs)

    assert model.weights, "vacuity: the layer created no weights at all"

    originals = [np.array(w) for w in model.weights]
    for w in model.weights:
        w.assign(np.array(w) + 0.5)
    moved = sum(
        1 for o, w in zip(originals, model.weights)
        if np.abs(np.array(w) - o).max() > 0.0
    )
    assert moved == len(model.weights), (
        f"vacuity: only {moved} of {len(model.weights)} weights were "
        f"perturbed, so a fresh-init reload could pass by coincidence"
    )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.keras")
        model.save(path)

        n_archived = _archived_dataset_count(path)
        assert n_archived == len(model.weights), (
            f"{mode}: {n_archived} of {len(model.weights)} weight tensors "
            f"reached `model.weights.h5`. A layer container nested two or "
            f"more levels deep under a `keras.layers.Layer` owner is not "
            f"written -- keep the sub-layers in FLAT per-role lists."
        )

        before = np.array(model(xs))
        loaded = keras.models.load_model(path)
        after = np.array(loaded(xs))

    assert len(loaded.weights) == len(model.weights)
    max_dw = max(
        float(np.abs(np.array(a) - np.array(b)).max())
        for a, b in zip(model.weights, loaded.weights)
    )
    assert max_dw == 0.0, (
        f"{mode}: reloaded weights differ from the donor by {max_dw:.6e}; "
        f"the save side may be complete while the LOAD side drops them"
    )
    assert np.abs(before - after).max() == 0.0, (
        f"{mode}: the reloaded model computes a different function"
    )


def test_graphsage_holds_its_transforms_in_flat_per_role_lists():
    """The structural property the fix rests on, pinned by name.

    Without this a future refactor back to `gnn_layers[i]['self']` would only
    be caught by the archive count above, whose message would not say what to
    do about it.
    """
    layer = GraphNeuralNetworkLayer(
        concept_dim=D, num_layers=2, message_passing="graphsage",
    )
    assert len(layer.sage_self_layers) == 2
    assert len(layer.sage_neighbor_layers) == 2
    assert all(isinstance(x, keras.layers.Dense) for x in layer.sage_self_layers)
    assert all(
        isinstance(x, keras.layers.Dense) for x in layer.sage_neighbor_layers
    )
    assert not any(isinstance(x, dict) for x in layer.gnn_layers), (
        "a dict inside `gnn_layers` is the nested container REF-9 names"
    )
