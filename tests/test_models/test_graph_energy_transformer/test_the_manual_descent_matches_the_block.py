"""``descend_capture``'s manual descent must equal ``EnergyTransformer.call``'s own loop.

Variant B (``GraphAnomalyDetector``) needs both the post-step-1 state ``g_1`` and the
final state ``g_T``, but the ET block returns only its final state. So
``GraphEnergyTransformerBackbone.descend_capture`` re-runs the descent BY HAND through
the block's public ``.norm`` / ``._weighted_adjacency`` / ``.attention.update`` /
``.hopfield.update``. That is a SECOND implementation of the same recurrence, and the
package README asserts it is "verified bit-exact against the block's own ``call()``
(max |Δ| = 0.0)".

MEASURED 2026-08-21: **that claim had no test anywhere in the repo.**
``grep -rn "descend_capture" tests/`` reached only
``test_weighted_adjacency.py``, whose subject is a dead PROJECTOR, not trajectory
parity. This module is that missing guard. It is a missing-guard-over-a-correct-path
finding, not a defect: the parity holds at HEAD on both arms.

**Compare the LayerNormed quantity.** ``call`` returns the raw ``x_T``; ``descend_capture``
records ``block.norm(x_T)``. Comparing those two directly is comparing different
quantities: it reads ~1.6e+00 while nothing is wrong. The parity statement is

    descend_capture(...)[T]  ==  block.norm( block(tokens, ...) )

An earlier reading of this exact seam was filed at 1.649e+00 and then reversed to
0.000000e+00 by its own control, for precisely this reason.

Arms:

* ``test_...binary_adjacency`` — the default path.
* ``test_...weighted_adjacency`` — ``use_weighted_adjacency=True``, the eq.-25 opt-in.
  ``descend_capture`` must hoist ``Ŵ`` once from the block INPUT and forward it at every
  step, exactly as ``call`` does; dropping that forward is the defect
  ``plan-2026-07-15T053724-78001af1/D-003`` fixed, and it is invisible on the binary arm.
* ``test_the_parity_probe_can_see_a_divergence`` — the negative control. ``g_1`` (one
  step) must NOT equal ``norm(call(...))`` (``T`` steps), or an exact 0.0 above would
  only mean the probe compares a tensor with itself.

``noise_std=0.0`` throughout (variant B's own contract): the manual loop is noiseless,
so a nonzero ``noise_std`` would make the two paths legitimately disagree. Every build
is seeded. Device-independent in intent, but the bound is EXACT (``atol=0.0,
rtol=0.0``): both paths execute the same ops in the same order on the same tensors, so
any nonzero delta is a real divergence, not reassociation noise.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.graph_energy_transformer.model import (
    GraphEnergyTransformerBackbone,
)

BATCH, N_REAL, N_PAD = 4, 24, 8
N = N_REAL + N_PAD
F, EMBED_DIM, NUM_HEADS, HEAD_DIM, HOPFIELD_DIM = 8, 32, 4, 8, 64
NUM_STEPS = 4
SEED = 20260821


def _backbone(use_weighted_adjacency=False):
    keras.utils.set_random_seed(SEED)
    return GraphEnergyTransformerBackbone(
        node_feature_dim=F, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        head_dim=HEAD_DIM, hopfield_dim=HOPFIELD_DIM, num_blocks=1,
        num_steps=NUM_STEPS, use_pe=False, use_cls=False,
        noise_std=0.0, use_weighted_adjacency=use_weighted_adjacency, seed=SEED,
    )


def _batch():
    rng = np.random.default_rng(SEED)
    node_features = np.zeros((BATCH, N, F), dtype="float32")
    node_features[:, :N_REAL, :] = rng.normal(size=(BATCH, N_REAL, F)).astype("float32")
    adjacency = np.zeros((BATCH, N, N), dtype="float32")
    adjacency[:, :N_REAL, :N_REAL] = (
        rng.random((BATCH, N_REAL, N_REAL)) < 0.2
    ).astype("float32")
    node_mask = np.zeros((BATCH, N), dtype="float32")
    node_mask[:, :N_REAL] = 1.0
    return {
        "node_features": keras.ops.convert_to_tensor(node_features),
        "adjacency": keras.ops.convert_to_tensor(adjacency),
        "node_mask": keras.ops.convert_to_tensor(node_mask),
    }


def _both_paths(backbone, inputs):
    """Return (manual captures, the block's own LayerNormed final state)."""
    tokens = backbone.embed(inputs, training=False)
    adjacency, node_mask = inputs["adjacency"], inputs["node_mask"]

    caps = backbone.descend_capture(
        tokens, adjacency, node_mask, capture_steps={1, NUM_STEPS}, training=False,
    )

    block = backbone.blocks[0]
    x = keras.ops.cast(tokens, block.compute_dtype)
    adj = keras.ops.cast(adjacency, block.compute_dtype)
    own = block(x, attention_mask=adj, mask=node_mask, training=False)
    own_normed = keras.ops.cast(block.norm(own), backbone.compute_dtype)
    return caps, own_normed


def _maxdelta(a, b):
    return float(
        np.max(np.abs(keras.ops.convert_to_numpy(a) - keras.ops.convert_to_numpy(b)))
    )


class TestTheManualDescentMatchesTheBlock:

    @pytest.mark.parametrize("weighted", [False, True], ids=["binary", "weighted"])
    def test_the_manual_trajectory_ends_where_the_block_does(self, weighted):
        backbone = _backbone(use_weighted_adjacency=weighted)
        inputs = _batch()
        caps, own_normed = _both_paths(backbone, inputs)

        assert set(caps) == {1, NUM_STEPS}, sorted(caps)
        delta = _maxdelta(caps[NUM_STEPS], own_normed)
        assert delta == 0.0, (
            f"descend_capture's step-{NUM_STEPS} state diverges from "
            f"EnergyTransformer.call by max|delta| = {delta:.6e} "
            f"(use_weighted_adjacency={weighted})"
        )

    def test_the_parity_probe_can_see_a_divergence(self):
        """Control: g_1 (one step) must NOT equal the T-step result."""
        backbone = _backbone()
        inputs = _batch()
        caps, own_normed = _both_paths(backbone, inputs)

        delta = _maxdelta(caps[1], own_normed)
        assert delta > 1e-4, (
            "the step-1 and step-T states are indistinguishable, so the exact-0.0 "
            f"parity assertion is vacuous: max|delta| = {delta:.6e}"
        )
