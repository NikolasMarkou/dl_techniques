"""RELGT's output is linear: class logits (plan-2026-10-14T062616-81938e81 D-148).

Until 2026-10-17 ``problem_type='classification'`` (the default) put a softmax on
the final ``Dense``. The repository's logits rule made the head linear for every
task, which left ``problem_type`` selecting nothing, so it was deleted. Pinned:

- a shift of the final bias moves every output by exactly that shift (a softmax
  is invariant to it);
- ``problem_type`` is refused, not silently accepted;
- stock ``fit`` with ``SparseCategoricalCrossentropy(from_logits=True)``
  memorises one fixed batch of subgraphs (the D-122 training bar).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.graph.relgt.model import create_relgt_model

BATCH, NUM_NODES, FEATURE_DIM, OUTPUT_DIM = 8, 6, 8, 3


def _inputs(batch: int = BATCH):
    rng = np.random.default_rng(0)
    return {
        "node_features": rng.random((batch, NUM_NODES, FEATURE_DIM)).astype("float32"),
        "node_types": rng.integers(0, 10, (batch, NUM_NODES)).astype("int32"),
        "hop_distances": rng.integers(0, 3, (batch, NUM_NODES)).astype("int32"),
        "relative_times": rng.random((batch, NUM_NODES, 1)).astype("float32"),
        "subgraph_adjacency": rng.random((batch, NUM_NODES, NUM_NODES)).astype("float32"),
    }


def _final_dense(model):
    return model.prediction_head.get_layer("FinalOutput")


def test_a_bias_shift_moves_every_output_by_exactly_the_shift(monkeypatch):
    # RELGTTokenEncoder draws fresh random GNN positional-encoding features on
    # every call (the paper's random node features), and keras.utils.
    # set_random_seed does not make an unseeded eager draw repeat (measured on
    # GPU: two seeded calls differ by up to 1.07). The draw is pinned to zeros
    # here; the claim is about the head, not the encoder.
    monkeypatch.setattr(keras.random, "normal",
                        lambda shape, dtype=None, **kwargs: keras.ops.zeros(shape, dtype=dtype))
    model = create_relgt_model(output_dim=OUTPUT_DIM, model_size="small")
    x = _inputs(2)
    out = keras.ops.convert_to_numpy(model(x, training=False))
    dense = _final_dense(model)
    assert dense.activation is keras.activations.linear
    shift = -50.0
    dense.bias.assign(dense.bias + shift)
    moved = keras.ops.convert_to_numpy(model(x, training=False))
    bound = 8 * np.finfo(np.float32).eps * (np.abs(out).max() + abs(shift))
    np.testing.assert_allclose(moved - out, shift, rtol=0, atol=bound)


def test_problem_type_is_refused():
    with pytest.raises((TypeError, ValueError)):
        create_relgt_model(output_dim=OUTPUT_DIM, model_size="small", problem_type="classification")


def test_fit_with_from_logits_memorises_one_batch():
    keras.utils.set_random_seed(0)
    model = create_relgt_model(output_dim=OUTPUT_DIM, model_size="small", dropout_rate=0.0)
    x = _inputs()
    y = np.arange(BATCH) % OUTPUT_DIM
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    history = model.fit(x, y, batch_size=BATCH, epochs=150, shuffle=False, verbose=0)
    chance = float(np.log(OUTPUT_DIM))
    assert history.history["loss"][-1] < 0.1 * chance, history.history["loss"][-1]
    assert history.history["accuracy"][-1] == 1.0
