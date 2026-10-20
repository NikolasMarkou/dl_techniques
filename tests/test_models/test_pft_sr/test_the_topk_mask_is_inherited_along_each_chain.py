"""Each layer attends only within the keys its chain kept, and the kept width
follows ``num_topk`` by the official rule.

The official code keeps two progressive-focusing chains, ``pfa_values[shift]``
and ``pfa_indices[shift]`` with ``shift = 0`` for the plain-window layers and
``1`` for the shifted ones (:240-316): a shifted window partitions the map
differently, so its map cannot multiply a plain-window one. A layer therefore
inherits from the previous layer OF ITS CHAIN (two layers back within a stage,
and across stage boundaries), not from the layer just before it; the official
``WindowAttention.flops`` reads ``num_topk[layer_id - 2]`` for the same reason.
Within a chain, a layer whose ``num_topk`` is below the current width keeps its
top ``num_topk`` keys per query; otherwise it keeps the width (and the indices).

Read from the running model: every layer's returned mask and map are recorded
by wrapping its ``call``. The expected widths come from ``num_topk`` here, by
the official rule, not from the model's own schedule.
"""

import keras
import numpy as np

from dl_techniques.models.vision.super_resolution.pft_sr import PFTSR

N = 16                                   # window 4
DEPTHS = [3, 4]                          # an odd stage: parity restarts per stage
NUM_TOPK = [16, 12, 12, 9, 14, 5, 8]   # layer 4: 14 is not below 12, reuse


def _chain_of(depths):
    """Stage-local parity: ``shift_size = 0 if i % 2 == 0 else ws // 2``."""
    return [i % 2 for depth in depths for i in range(depth)]


def _expected_widths(num_topk, chains):
    width, out = {0: N, 1: N}, []
    for k, c in zip(num_topk, chains):
        if k < width[c]:
            width[c] = k
        out.append(width[c])
    return out


def _record(model, x):
    records = []
    for layer in model.pft_layers:
        original = layer.call

        def call(*args, _original=original, _layer=layer, **kwargs):
            out = _original(*args, **kwargs)
            records.append((_layer.name, kwargs.get("prev_values") is not None,
                            kwargs.get("prev_mask"), out[1], out[2]))
            return out

        layer.call = call
    model(x, training=False)
    return records


def test_each_layer_keeps_only_keys_its_chain_kept_at_the_scheduled_width():
    keras.utils.set_random_seed(0)
    model = PFTSR(scale=2, embed_dim=16, num_blocks=DEPTHS, num_heads=2, window_size=4,
                  num_topk=NUM_TOPK)
    x = np.random.default_rng(0).random((1, 8, 12, 3)).astype("float32")
    records = _record(model, x)
    chains = _chain_of(DEPTHS)
    widths = _expected_widths(NUM_TOPK, chains)
    assert len(records) == len(NUM_TOPK)

    last_mask = {0: None, 1: None}
    seen = {0: False, 1: False}
    for l, ((name, inherited, prev_mask, values, mask), c, width) in enumerate(
            zip(records, chains, widths)):
        values = np.asarray(values)
        assert inherited == seen[c], f"{name}: inherits={inherited}, chain {c} seen={seen[c]}"
        seen[c] = True
        if prev_mask is not None:
            assert np.array_equal(np.asarray(prev_mask), np.asarray(last_mask[c])), \
                f"{name} did not receive its own chain's mask"
        if width == N:
            assert mask is None, f"{name}: a mask at the full width {N}"
            continue
        mask = np.asarray(mask)
        assert np.all(mask.sum(-1) == width), f"{name}: kept {set(mask.sum(-1).ravel())}, want {width}"
        if last_mask[c] is not None:
            assert not np.any(mask & ~np.asarray(last_mask[c])), \
                f"{name} kept a key its chain had dropped"
        assert np.all(values[~mask] == 0.0), f"{name}: attention outside the kept keys"
        if inherited:
            # (a * a_prev + eps) / (sum + eps) > 0 on every kept key. A chain's
            # FIRST top-k has no eps: in a shifted layer it may keep keys
            # across a region whose weight exp(-100) is 0 in float32.
            assert np.all(values[mask] > 0.0), f"{name}: a kept key with zero weight"
        last_mask[c] = mask
    assert widths == [16, 12, 12, 9, 12, 5, 8], "the schedule this arm was written for"


def test_the_model_schedule_is_the_official_rule():
    model = PFTSR(scale=2, embed_dim=16, num_blocks=DEPTHS, num_heads=2, window_size=4,
                  num_topk=NUM_TOPK)
    keeps = [layer.keep_topk for layer in model.pft_layers]
    assert keeps == [None, 12, 12, 9, None, 5, 8]
    assert [layer.shift_size for layer in model.pft_layers] == [0, 2, 0, 0, 2, 0, 2]


def test_no_schedule_keeps_every_key():
    model = PFTSR(scale=2, embed_dim=16, num_blocks=[2, 2], num_heads=2, window_size=4)
    assert model.num_topk is None
    assert all(layer.keep_topk is None for layer in model.pft_layers)


def test_a_tie_still_keeps_exactly_k_keys():
    """A flat map ties every key. The official ``torch.topk`` still returns
    exactly ``k``; a mask built by thresholding at the k-th value would keep
    them all (D-214)."""
    from dl_techniques.models.vision.super_resolution.pft_sr.blocks import topk_mask
    flat = np.full((3, 2, 16, 16), 1.0 / 16, dtype="float32")
    mask = np.asarray(topk_mask(flat, 5))
    assert np.all(mask.sum(-1) == 5)
    assert np.all(mask[..., :5]), "ties go to the lower index (TensorFlow top_k)"


def test_a_constant_image_keeps_the_scheduled_width():
    """End to end on the tie case: a constant image makes whole windows flat."""
    keras.utils.set_random_seed(0)
    model = PFTSR(scale=2, embed_dim=16, num_blocks=DEPTHS, num_heads=2, window_size=4,
                  num_topk=NUM_TOPK)
    records = _record(model, np.full((1, 8, 12, 3), 0.5, dtype="float32"))
    for (name, _, _, _, mask), width in zip(records, _expected_widths(NUM_TOPK, _chain_of(DEPTHS))):
        if width < N:
            assert np.all(np.asarray(mask).sum(-1) == width), name
