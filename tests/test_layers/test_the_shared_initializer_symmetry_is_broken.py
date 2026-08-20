"""
R-123: three ``layers/`` sites where one initializer INSTANCE built several roles.

D-057 ruled that a shared-instance initializer collision is a defect EXACTLY
when the colliding weights play DIFFERENT architectural roles, and that this
cannot be read off the shape -- every site must be probed. These three were
probed and all three convict:

======================================  ==================================
site                                    what was bit-identical BEFORE
======================================  ==================================
``layers/attention/group_query_...``    ``w_q == w_k == w_v == w_o`` in
                                        every block (``fastvlm``: 66
                                        identical pairs of 84 tensors, of
                                        which 36 were these projections)
``layers/attention/energy_attention``   ``w_key == w_query``, making the
                                        initial score matrix EXACTLY
                                        symmetric in all six
                                        ``energy_transformer`` /
                                        ``graph_energy_transformer`` classes
``layers/memory/baseline_ntm``          ``read_head/key == write_head/key
                                        == write_head/erase ==
                                        write_head/add`` and all six
                                        ``{read,write}_head/{beta,gate,
                                        gamma}`` (``ntm``: 22 identical
                                        pairs of 17 tensors)
======================================  ==================================

``erase`` and ``add`` are OPPOSITE operations on the NTM's memory; a query and
a key are the two halves of a score. None of these is the harmless
same-role case D-057 protects.

AFTER, measured on the same models: ``ntm`` 22 identical pairs -> **0**;
``fastvlm`` 66 -> **30**, and all 30 survivors are the RoPE ``cos_cached`` /
``sin_cached`` tables, which are DETERMINISTIC and identical by construction,
not by initializer sharing.

The seeded arm is not decoration
--------------------------------
``clone_initializer`` deliberately does NOT break the symmetry of a SEEDED
initializer: two clones of ``GlorotUniform(seed=7)`` still produce identical
tensors, because an author who asked for a seed asked for reproducibility.
That arm is asserted here so a later "stronger fix" cannot quietly remove it.
"""

import itertools

import keras
import numpy as np
import pytest
from keras import ops


def _arr(v):
    return np.asarray(ops.convert_to_numpy(ops.cast(v, "float32")))


def _identical_same_shape_pairs(model):
    """Bit-identical, same-shape, NON-CONSTANT weight pairs.

    Constants (all-zeros biases, all-ones norm gammas) are excluded because
    they are identical BY DESIGN and would swamp the signal -- an early draft
    of this probe reported them and made every model look broken.
    """
    tensors = []
    for v in model.weights:
        a = _arr(v)
        if a.size < 2 or np.unique(a).size <= 1:
            continue
        tensors.append((v.path, a))
    return [
        (p1, p2)
        for (p1, a1), (p2, a2) in itertools.combinations(tensors, 2)
        if a1.shape == a2.shape and np.array_equal(a1, a2)
    ]


# ---------------------------------------------------------------------------
# The MECHANISM, pinned. This is the RED half: it is the exact shape the three
# sites had, reproduced on two `Dense` layers, and it must still hold -- if
# Keras ever stops replaying a shared instance, all three fixes below become
# removable rather than load-bearing.
# ---------------------------------------------------------------------------
def test_one_shared_initializer_instance_still_produces_identical_kernels():
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    a = keras.layers.Dense(4, kernel_initializer=shared)
    b = keras.layers.Dense(4, kernel_initializer=shared)
    a.build((None, 6))
    b.build((None, 6))
    delta = float(np.abs(_arr(a.kernel) - _arr(b.kernel)).max())
    assert delta == 0.0, (
        "a shared seedless initializer instance no longer replays its draw "
        f"(max|delta| {delta:.6e}); the three clone_initializer sites are now "
        "dead weight and should be removed rather than left unexplained"
    )

    # And the STRING form does not collide -- the control that shows the cause
    # is the instance, not the name.
    keras.utils.set_random_seed(1234)
    c = keras.layers.Dense(4, kernel_initializer="glorot_uniform")
    d = keras.layers.Dense(4, kernel_initializer="glorot_uniform")
    c.build((None, 6))
    d.build((None, 6))
    assert float(np.abs(_arr(c.kernel) - _arr(d.kernel)).max()) > 0.0


# ---------------------------------------------------------------------------
# Site 1 -- GroupQueryAttention
# ---------------------------------------------------------------------------
def _gqa(**kwargs):
    from dl_techniques.layers.attention.group_query_attention import (
        GroupedQueryAttention,
    )
    layer = GroupedQueryAttention(dim=16, num_heads=4, num_kv_heads=2, **kwargs)
    layer.build((None, 5, 16))
    return layer


def test_the_four_attention_projections_start_different():
    keras.utils.set_random_seed(0)
    layer = _gqa()
    q, o = _arr(layer.w_q.kernel), _arr(layer.w_o.kernel)
    assert q.shape == o.shape, "the probe compares two same-shape kernels"
    assert float(np.abs(q - o).max()) > 0.0, "w_q and w_o are still identical"
    k, v = _arr(layer.w_k.kernel), _arr(layer.w_v.kernel)
    assert float(np.abs(k - v).max()) > 0.0, "w_k and w_v are still identical"


def test_a_seeded_initializer_still_reproduces_across_the_projections():
    """Deliberate: ``clone_initializer`` preserves an EXPLICIT seed."""
    keras.utils.set_random_seed(0)
    layer = _gqa(kernel_initializer=keras.initializers.GlorotUniform(seed=7))
    assert float(np.abs(_arr(layer.w_k.kernel)
                        - _arr(layer.w_v.kernel)).max()) == 0.0


# ---------------------------------------------------------------------------
# Site 2 -- EnergyAttention
# ---------------------------------------------------------------------------
def _energy(**kwargs):
    from dl_techniques.layers.attention.energy_attention import EnergyAttention
    layer = EnergyAttention(dim=16, num_heads=2, head_dim=8, **kwargs)
    layer.build((None, 5, 16))
    return layer


def test_the_energy_key_and_query_start_different():
    keras.utils.set_random_seed(0)
    layer = _energy()
    delta = float(np.abs(_arr(layer.w_key) - _arr(layer.w_query)).max())
    assert delta > 0.0, (
        "w_key and w_query are still bit-identical, so the initial energy "
        "score matrix is exactly symmetric")


def test_the_energy_seeded_initializer_still_reproduces():
    keras.utils.set_random_seed(0)
    layer = _energy(kernel_initializer=keras.initializers.GlorotUniform(seed=7))
    assert float(np.abs(_arr(layer.w_key) - _arr(layer.w_query)).max()) == 0.0


# ---------------------------------------------------------------------------
# Site 3 -- the NTM heads, judged at the MODEL, where the pairs were measured
# ---------------------------------------------------------------------------
def test_the_ntm_heads_have_no_identical_projections_left():
    from dl_techniques.models.ntm import create_ntm_variant
    keras.utils.set_random_seed(0)
    model = create_ntm_variant(variant="tiny", input_shape=(10, 8),
                               output_dim=4)
    model(np.random.RandomState(0).randn(2, 10, 8).astype("float32"),
          training=False)
    pairs = _identical_same_shape_pairs(model)
    assert not pairs, (
        f"{len(pairs)} bit-identical head projections remain: {pairs[:4]}")


def test_the_ntm_probe_can_see_a_collision_it_is_meant_to_reject():
    """Liveness for :func:`_identical_same_shape_pairs`.

    A probe that returned ``[]`` unconditionally would pass the test above
    against a fully collapsed model. This builds a model that IS collapsed --
    two ``Dense`` layers sharing one instance -- and requires the probe to
    report it.
    """
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    # Both kernels must be (4, 4): the probe compares SAME-SHAPE pairs only,
    # and a first draft using `Input((6,))` gave (6, 4) and (4, 4), so the
    # probe correctly reported nothing and the liveness arm read as a failure.
    collapsed = keras.Sequential([
        keras.layers.Input((4,)),
        keras.layers.Dense(4, kernel_initializer=shared, name="a"),
        keras.layers.Dense(4, kernel_initializer=shared, name="b"),
    ])
    assert _identical_same_shape_pairs(collapsed), (
        "the probe cannot see a collision, so every assertion above is vacuous")


# ---------------------------------------------------------------------------
# fastvlm -- the whole-model statement, and the RoPE exemption named explicitly
# ---------------------------------------------------------------------------
def test_fastvlm_has_no_identical_random_pairs_outside_the_rope_caches():
    from dl_techniques.models.fastvlm.model import FastVLM
    keras.utils.set_random_seed(0)
    model = FastVLM(num_classes=4)
    pairs = _identical_same_shape_pairs(model)
    non_rope = [(a, b) for a, b in pairs
                if "rope" not in a and "rope" not in b]
    assert not non_rope, (
        f"{len(non_rope)} identical RANDOM pairs remain: {non_rope[:4]}")
    assert pairs, (
        "not even the RoPE caches match any more -- either the caches were "
        "removed or this probe stopped working; both need explaining")
