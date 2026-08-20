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
import re

import keras
import numpy as np
import pytest
from keras import ops


def _arr(v):
    return np.asarray(ops.convert_to_numpy(ops.cast(v, "float32")))


def _identical_same_shape_pairs(model):
    """Bit-identical, same-SIZE, NON-CONSTANT weight pairs, compared FLAT.

    Constants (all-zeros biases, all-ones norm gammas) are excluded because
    they are identical BY DESIGN and would swamp the signal -- an early draft
    of this probe reported them and made every model look broken.

    DECISION plan-2026-08-19T163559-499b6f0e/D-073
    The comparison is ``a1.ravel() == a2.ravel()`` over equal-SIZE tensors, not
    ``a1 == a2`` over equal-SHAPE ones. The first draft of this probe compared
    shapes and was STRUCTURALLY BLIND to the ``cbam`` site, reporting 0 pairs
    for a model that has two per block: a shared initializer instance hands the
    SAME FLAT DRAW to a ``(64, 8)`` squeeze kernel and an ``(8, 64)`` excite
    kernel, so the two tensors hold the same numbers in the same order at
    different shapes. Calling that 0 a refutation would have been an instrument
    failure. Note also that this is NOT a transpose relationship -- the audit
    called it "transposed" and that is wrong for any shape but ``(n, 1)``:
    ``a2.T`` reorders the elements, the flat draw does not. Do NOT narrow this
    back to a shape comparison. See decisions.md D-073.
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
        if a1.size == a2.size and np.array_equal(a1.ravel(), a2.ravel())
    ]


def _role(path):
    """The last two path segments with their block indices erased.

    D-057's rule -- a collision is a defect exactly when the two weights play
    DIFFERENT architectural roles -- needs a mechanical reading of "role" to
    partition a 140-pair census. Two weights share a role when their leaf path
    matches after ``_0``/``_12`` style indices are normalised away, i.e. they
    are the same named tensor in two different blocks.
    """
    return re.sub(r"_\d+", "_N", "/".join(path.split("/")[-2:]))


def _different_role_pairs(model):
    """Only the pairs D-057 CONVICTS: same numbers, different roles."""
    return [(a, b) for a, b in _identical_same_shape_pairs(model)
            if _role(a) != _role(b)]


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


# ===========================================================================
# The FOUR rows step 18.1 left open, each ruled on its own measurement.
#
# D-057's rule is "different architectural roles", and it cannot be applied by
# grep. Each row below carries the census that decided it. Two of the four
# refute the reading they arrived with:
#
#   cbam    the audit called it a TRANSPOSE and 18.1's shape probe saw 0 of 9.
#           Both are wrong: it is the SAME FLAT DRAW at two shapes, 2 pairs.
#   relgt   the audit charged 5, 18.1 measured 0 and "could not reproduce".
#           The audit is RIGHT; 18.1's construction dodged every shape
#           coincidence. Reproduced verbatim at the audit's geometry.
#   clip    needed a BUILT model. `CLIP`'s `w_q == w_k == w_v` is already gone
#           (D-068's group_query_attention clone), but a built model shows SIX
#           new SwiGLU pairs, and `CliffordCLIP` shows 763.
#   yolo12  161 pairs at scale `n`, of which 155 are SAME-role and 6 are not.
#           Only the 6 are a defect; the 155 stay and are asserted to stay.
# ===========================================================================


# ---------------------------------------------------------------------------
# cbam -- the row a same-shape probe structurally could not see
# ---------------------------------------------------------------------------
def test_the_cbam_squeeze_and_excite_kernels_are_no_longer_one_draw():
    """MEASURED BEFORE: 2 pairs, one per block, at (64, 8) vs (8, 64) and
    (128, 16) vs (16, 128). AFTER: 0."""
    from dl_techniques.models.cbam import CBAMNet
    keras.utils.set_random_seed(1234)
    model = CBAMNet.from_variant("tiny", input_shape=(32, 32, 3), num_classes=4)
    model(np.zeros((1, 32, 32, 3), "float32"), training=False)
    pairs = _identical_same_shape_pairs(model)
    assert not pairs, f"{len(pairs)} identical pairs remain: {pairs}"


def test_the_flat_probe_sees_what_the_shape_probe_could_not():
    """Liveness for the D-073 widening, on the exact shape cbam had.

    Two ``Dense`` layers sharing one instance whose kernels are ``(4, 2)`` and
    ``(2, 4)``. A same-SHAPE probe reports nothing here -- that is the blindness
    being fixed -- while the flat probe must report the pair. Both halves are
    asserted, so narrowing the probe back fails this test.
    """
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    collapsed = keras.Sequential([
        keras.layers.Input((4,)),
        keras.layers.Dense(2, kernel_initializer=shared, name="squeeze"),
        keras.layers.Dense(4, kernel_initializer=shared, name="excite"),
    ])
    kernels = [_arr(v) for v in collapsed.weights if v.path.endswith("kernel")]
    assert [k.shape for k in kernels] == [(4, 2), (2, 4)]
    assert kernels[0].shape != kernels[1].shape, (
        "the shapes now match, so this liveness arm no longer proves the "
        "widening was needed")
    assert _identical_same_shape_pairs(collapsed), (
        "the flat probe cannot see a same-draw pair at two shapes, so the "
        "cbam assertion above is vacuous")


# ---------------------------------------------------------------------------
# relgt -- the 5-vs-0 disagreement, settled by the construction
# ---------------------------------------------------------------------------
def _relgt(embedding_dim, ffn_dim):
    from dl_techniques.models.relgt import RELGT
    keras.utils.set_random_seed(1234)
    model = RELGT(output_dim=2, num_transformer_blocks=1, num_heads=2,
                  num_global_centroids=4, num_node_types=3, gnn_pe_dim=8,
                  embedding_dim=embedding_dim, ffn_dim=ffn_dim)
    model({
        "node_features": np.random.RandomState(0).randn(1, 6, 8).astype("float32"),
        "node_types": np.zeros((1, 6), "int32"),
        "hop_distances": np.zeros((1, 6), "int32"),
        "subgraph_adjacency": np.eye(6, dtype="float32")[None],
        "relative_times": np.zeros((1, 6), "float32"),
    }, training=False)
    return model


@pytest.mark.parametrize("embedding_dim,ffn_dim", [(32, 32), (16, 256)])
def test_relgt_has_no_identical_pairs_at_either_disputed_construction(
        embedding_dim, ffn_dim):
    """Both readings are pinned, because only one of them was ever a refutation.

    ``(32, 32)`` is the audit's geometry, where it charged 5 pairs; those 5
    reproduce EXACTLY at HEAD-before-this-step (``FeatureEncoder ==
    PEProjection`` (8, 32); ``cross_attention/proj == ffn/fc1 == ffn/fc2``
    (32, 32); ``PredictionFFN/fc1 == fc2`` (32, 32)). ``(16, 256)`` is the
    construction step 18.1 used to report "0 of 23 -- not reproduced": with
    ``ffn_dim != embedding_dim`` no FFN shape coincides, so its 0 was a
    property of the geometry and not of the model. Pinning only that one would
    re-freeze the blindness.
    """
    pairs = _identical_same_shape_pairs(_relgt(embedding_dim, ffn_dim))
    assert not pairs, (
        f"embedding_dim={embedding_dim} ffn_dim={ffn_dim}: {len(pairs)} "
        f"identical pairs remain: {pairs[:5]}")


# ---------------------------------------------------------------------------
# clip -- both classes, both needing a BUILT model
# ---------------------------------------------------------------------------
def _clip():
    from dl_techniques.models.clip.model import CLIP
    keras.utils.set_random_seed(1234)
    model = CLIP(image_size=32, patch_size=16, vision_layers=1, vision_width=32,
                 vision_heads=2, vision_kv_heads=1, text_layers=1,
                 text_width=32, text_heads=2, text_kv_heads=1, embed_dim=16,
                 vocab_size=64, context_length=8)
    model({"image": np.zeros((1, 32, 32, 3), "float32"),
           "text": np.zeros((1, 8), "int32")}, training=False)
    return model


def test_clip_has_no_identical_random_pairs_outside_the_rope_caches():
    """MEASURED BEFORE: 8 pairs. The two ``w_q == w_k == w_v`` collisions the
    audit charged are ALREADY gone (D-068 cloned in
    ``group_query_attention``), but a BUILT model -- which 18.1 never had --
    shows six more: ``ffn/gate_proj == up_proj == down_proj`` in both towers.
    SwiGLU is ``silu(gate(x)) * up(x)``; equal branches make it ``silu(u) * u``.
    AFTER: 2, both of them the deterministic RoPE caches."""
    model = _clip()
    pairs = _identical_same_shape_pairs(model)
    non_rope = [(a, b) for a, b in pairs if "rope" not in a and "rope" not in b]
    assert not non_rope, (
        f"{len(non_rope)} identical RANDOM pairs remain: {non_rope[:4]}")
    assert pairs, (
        "not even the RoPE caches match any more -- either the caches were "
        "removed or this probe stopped working; both need explaining")


def _clifford_clip():
    from dl_techniques.models.clip.clifford_clip import CliffordCLIP
    keras.utils.set_random_seed(1234)
    model = CliffordCLIP.from_variant(
        "nano", vocab_size=64, image_size=64, context_length=16,
        vision_patch_size=4, dropout_rate=0.0,
        vision_stochastic_depth_rate=0.0, text_stochastic_depth_rate=0.0)
    model.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    return model


def test_the_clifford_clip_towers_are_no_longer_the_same_function():
    """The worst R-123 site in the tree, and a MODULE-LEVEL cause.

    ``clifford_clip.py``'s ``_DEFAULT_KERNEL_INIT`` is one ``Initializer``
    INSTANCE used as a default argument, so it was shared by every sub-layer of
    every instance. MEASURED BEFORE: **763** identical pairs of 137
    non-constant tensors, including
    ``vision_clifford_block_0/linear_det/kernel ==
    text_clifford_block_0/linear_det/kernel`` -- the image tower and the text
    tower starting as the same function, which is what a contrastive model
    exists NOT to be. AFTER: 234, of which **0 are different-role and 0 cross
    the towers**; every survivor is the same named tensor in two blocks of one
    stage, which D-057 explicitly does not convict.
    """
    model = _clifford_clip()
    diff = _different_role_pairs(model)
    assert not diff, f"{len(diff)} different-role pairs remain: {diff[:4]}"
    cross = [(a, b) for a, b in _identical_same_shape_pairs(model)
             if ("vision" in a) != ("vision" in b)]
    assert not cross, (
        f"{len(cross)} pairs still tie the vision tower to the text tower: "
        f"{cross[:4]}")


# ---------------------------------------------------------------------------
# yolo12 -- the row that is a defect in 6 of its 161 pairs and in no others
# ---------------------------------------------------------------------------
def test_yolo12_detection_head_no_longer_ties_the_box_and_class_branches():
    """D-057 applied per pair, not per model.

    MEASURED BEFORE at ``scale='n'``, ``input_shape=(64, 64, 3)``: 161
    bit-identical pairs of 140 non-constant tensors. **155 are SAME-role**
    (``conv/kernel`` against ``conv/kernel`` in another backbone block, and
    ``bbox_N_pred`` against ``bbox_M_pred`` across the three scales) and D-057
    does not convict those. **6 are DIFFERENT-role**, all in the detection
    head: ``bbox_N_pred/kernel`` against ``cls_0_pw{1,2}/conv/kernel``, i.e.
    the box regressor and the classifier starting as the same function.
    AFTER: 140 pairs, **0 of them different-role**.

    The same-role residue is asserted to REMAIN. Removing it would mean
    cloning through the whole backbone, which this step deliberately did not
    do; if a later change makes it vanish, that is a much larger edit than
    anyone thought they were making and it should fail here first.
    """
    from dl_techniques.models.yolo12 import create_yolov12_multitask
    keras.utils.set_random_seed(1234)
    model = create_yolov12_multitask(num_detection_classes=4,
                                     tasks=["detection"],
                                     input_shape=(64, 64, 3), scale="n")
    model(np.zeros((1, 64, 64, 3), "float32"), training=False)
    all_pairs = _identical_same_shape_pairs(model)
    diff = _different_role_pairs(model)
    assert not diff, (
        f"{len(diff)} different-role pairs remain in the detection head: "
        f"{diff[:4]}")
    assert all_pairs, (
        "the SAME-role backbone residue is gone too. D-057 does not convict "
        "it and this step did not touch it, so either the backbone was cloned "
        "as well -- a far larger change -- or this probe stopped working.")


def test_the_role_partition_can_tell_the_two_kinds_apart():
    """Liveness for :func:`_role`, which is what decides yolo12's ruling.

    Without this, a ``_role`` that returned a constant would mark every pair
    same-role and the yolo12 assertion above would pass against a model whose
    box and class heads are identical.
    """
    assert _role("head/bbox_branch_0/bbox_0_pred/kernel") == \
        _role("head/bbox_branch_2/bbox_2_pred/kernel")
    assert _role("head/bbox_branch_0/bbox_0_pred/kernel") != \
        _role("head/cls_branch_0/cls_0_pw1/conv/kernel")
