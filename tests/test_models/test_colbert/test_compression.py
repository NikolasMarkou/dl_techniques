"""Guards for the ColBERTv2 index-time residual compression codec.

The codec is the one artifact-level difference between ColBERT v1 and v2 in
this package, and it is the one component that must stay *outside* the model.
These tests therefore split into two families:

1. **Codec behaviour** -- the bit-width property (more bits reconstruct
   better), the unit-norm decode contract, round-trip serialization, and the
   input validation surface.
2. **The index-time boundary** -- a structural assertion that no symbol defined
   in ``compression.py`` is statically reachable from ``ColBERT.call`` or from
   either ColBERT loss.

Every random draw here is seeded and the seed is stated at the draw site.

MEASURED FALSIFICATION (recorded in decisions.md D-019). This step was briefed
to assert that "encoding a vector that IS a centroid gives a zero residual and
**decodes back to that centroid**". The first half is exactly true and is
asserted below (the residual is ``0.0``, bit-for-bit). The second half is
**false, and false for the reference codec too**: a quantizer whose
reconstruction levels are quantiles of the residual distribution has no level
equal to zero -- at ``nbits=1`` there are only two levels and neither can be
zero -- so a zero residual is still dequantized to a non-zero offset. Measured
here at ``dim=32, k=16``: mean ``||decode(encode(c)) - c||`` is 0.604 at
``nbits=1`` and 0.305 at ``nbits=2``. Asserting "decodes back to the centroid"
would have failed a correct implementation. What IS true, and is asserted
instead, is the discriminating property: the decoded vector still identifies
its own centroid as its nearest one, by a wide inner-product margin.

RED-PROOF RESULTS (three injections; each restored from a ``cp`` backup and
verified with ``diff -q``; never ``git stash`` / ``git checkout --``):

    (a) drop the final ``_colbert_codec_l2_normalize`` from
        ``ResidualCompressionCodec.decode``
        -> RED: test_decoded_vectors_are_unit_norm
           (assertion "decode() returned vectors that are not unit-norm ...")
    (b) make ``nbits=2`` bucketize with the single-bit codebook (levels forced
        through the ``nbits=1`` cutoff/weight derivation)
        -> RED: test_two_bits_reconstruct_strictly_better_than_one_bit
           (assertion "nbits=2 did not beat nbits=1 on seed ...")
    (c) import and call ``ResidualCompressionCodec`` from inside
        ``ColBERT.call`` in ``model.py``
        -> RED: test_no_codec_symbol_is_reachable_from_the_model_or_the_losses
           (assertion "codec symbol 'ResidualCompressionCodec' is referenced in
           executable code of dl_techniques.models.language.colbert.model")
"""

import ast
import inspect
import sys

import numpy as np
import pytest

from dl_techniques.models.language.colbert import compression as compression_module
from dl_techniques.models.language.colbert.compression import (
    SUPPORTED_NBITS,
    ResidualCompressionCodec,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 32
NUM_CENTROIDS = 16
NUM_VECTORS = 500


def _draw(seed: int, num_vectors: int = NUM_VECTORS, dim: int = DIM) -> np.ndarray:
    """Draw a seeded Gaussian embedding sample.

    Gaussian rows, once L2-normalized, are uniform on the sphere -- the
    hardest case for a centroid codec, since there is no cluster structure for
    k-means to exploit. A clustered draw would flatter the codec.

    :param seed: Explicit seed; every call site states it.
    :type seed: int
    :param num_vectors: Number of rows.
    :type num_vectors: int
    :param dim: Row width.
    :type dim: int
    :returns: ``(num_vectors, dim)`` float64 array.
    :rtype: numpy.ndarray
    """
    return np.random.default_rng(seed).normal(size=(num_vectors, dim))


@pytest.fixture
def fitted_codec() -> ResidualCompressionCodec:
    """A 2-bit codec fitted on seed 0.

    :returns: A fitted codec.
    :rtype: ResidualCompressionCodec
    """
    return ResidualCompressionCodec(
        dim=DIM, nbits=2, num_centroids=NUM_CENTROIDS, seed=0
    ).fit(_draw(seed=0))


# ---------------------------------------------------------------------------
# 1. The bit-width property
# ---------------------------------------------------------------------------


def test_two_bits_reconstruct_strictly_better_than_one_bit() -> None:
    """More residual bits must lower reconstruction error, on every draw.

    Derivation: encode/decode is a scalar quantizer applied independently to
    each residual dimension. ``nbits=1`` partitions the residual line into
    ``2`` cells, ``nbits=2`` into ``4``; both use quantiles of the *same*
    residual distribution, and the two centroid codebooks are identical because
    both codecs are fitted on the same sample with the same seed. A finer
    partition of the same line, with each cell represented by its own
    conditional median, cannot have larger expected distortion -- and for a
    continuous residual distribution the inequality is strict, because at least
    one 1-bit cell is split by a cutoff that lies strictly inside it. So the
    prediction is a **strict inequality on every seed**, not merely on average.

    The assertion is on that property. No sample error value is pinned.
    """
    errors_one = []
    errors_two = []
    for seed in range(20):  # seeds 0..19, one independent draw each
        vectors = _draw(seed=seed)
        codec_one = ResidualCompressionCodec(
            dim=DIM, nbits=1, num_centroids=NUM_CENTROIDS, seed=seed
        ).fit(vectors)
        codec_two = ResidualCompressionCodec(
            dim=DIM, nbits=2, num_centroids=NUM_CENTROIDS, seed=seed
        ).fit(vectors)

        # The two codecs must differ ONLY in bit width; if the centroids
        # diverged, the comparison below would not isolate the quantizer.
        np.testing.assert_allclose(
            codec_one.centroids,
            codec_two.centroids,
            atol=0.0,
            rtol=0.0,
            err_msg=(
                f"the nbits=1 and nbits=2 codecs disagree on their centroids at "
                f"seed {seed}; the bit-width comparison is confounded"
            ),
        )

        error_one = codec_one.reconstruction_error(vectors)
        error_two = codec_two.reconstruction_error(vectors)
        errors_one.append(error_one)
        errors_two.append(error_two)

        assert error_two < error_one, (
            f"nbits=2 did not beat nbits=1 on seed {seed}: "
            f"err(2)={error_two:.6f} >= err(1)={error_one:.6f}"
        )

    assert float(np.mean(errors_two)) < float(np.mean(errors_one)), (
        f"nbits=2 did not beat nbits=1 on the mean over 20 seeds: "
        f"{np.mean(errors_two):.6f} >= {np.mean(errors_one):.6f}"
    )


def test_the_packed_residual_size_matches_the_bit_width() -> None:
    """The packed payload must actually be ``nbits`` bits per dimension.

    Without this, a "2-bit" codec that silently stores 8-bit levels would still
    pass the error-property test above while buying none of the compression the
    codec exists for.
    """
    vectors = _draw(seed=7)  # seed 7
    for nbits in SUPPORTED_NBITS:
        codec = ResidualCompressionCodec(
            dim=DIM, nbits=nbits, num_centroids=NUM_CENTROIDS, seed=7
        ).fit(vectors)
        _, packed = codec.encode(vectors[:5])
        expected_bytes = DIM * nbits // 8
        assert packed.shape == (5, expected_bytes), (
            f"nbits={nbits} packed to shape {packed.shape}, expected "
            f"(5, {expected_bytes})"
        )
        assert packed.dtype == np.uint8, (
            f"nbits={nbits} packed to dtype {packed.dtype}, expected uint8"
        )


# ---------------------------------------------------------------------------
# 2. The decode contract
# ---------------------------------------------------------------------------


def test_decoded_vectors_are_unit_norm() -> None:
    """``decode()`` must re-L2-normalize (H-7, reference-code-only detail).

    Adding a quantized residual to a unit-norm centroid moves the point off the
    sphere. MaxSim is only a sum of cosine similarities while its inputs are
    unit vectors, so a decode that skips the renormalization rescales every
    score in the index by a per-vector factor.
    """
    vectors = _draw(seed=3)  # seed 3
    for nbits in SUPPORTED_NBITS:
        codec = ResidualCompressionCodec(
            dim=DIM, nbits=nbits, num_centroids=NUM_CENTROIDS, seed=3
        ).fit(vectors)
        decoded = codec.decode(*codec.encode(vectors[:64]))
        norms = np.linalg.norm(decoded, axis=-1)
        assert np.all(np.isfinite(decoded)), (
            f"decode() produced non-finite values at nbits={nbits}"
        )
        np.testing.assert_allclose(
            norms,
            np.ones_like(norms),
            atol=1e-6,
            rtol=0.0,
            err_msg=(
                f"decode() returned vectors that are not unit-norm at "
                f"nbits={nbits}: norms range "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            ),
        )


def test_a_centroid_encodes_to_an_exactly_zero_residual() -> None:
    """A centroid is its own nearest centroid, with residual exactly zero.

    This is the half of the briefed premise that survived measurement. See the
    module docstring: the other half ("decodes back to that centroid") is false
    for any quantile-derived codebook, including the reference's, because no
    reconstruction level equals zero.
    """
    vectors = _draw(seed=11)  # seed 11
    for nbits in SUPPORTED_NBITS:
        codec = ResidualCompressionCodec(
            dim=DIM, nbits=nbits, num_centroids=NUM_CENTROIDS, seed=11
        ).fit(vectors)
        codes, _ = codec.encode(codec.centroids)
        np.testing.assert_array_equal(
            codes,
            np.arange(NUM_CENTROIDS, dtype=codes.dtype),
            err_msg=(
                f"a centroid was not assigned to itself at nbits={nbits}; "
                "the maximum-inner-product nearest-centroid rule is broken"
            ),
        )
        residual = codec.centroids - codec.centroids[codes]
        assert np.abs(residual).max() == 0.0, (
            f"a centroid's residual against itself is not exactly zero at "
            f"nbits={nbits}: max|residual|={np.abs(residual).max():.3e}"
        )


def test_a_decoded_centroid_still_identifies_its_own_centroid() -> None:
    """The lossy decode must not move a centroid into another cell.

    Replaces the falsified "decodes back to that centroid" claim with the
    property that actually matters for retrieval: after the round trip, the
    decoded vector's maximum inner product over the codebook is still with the
    centroid it came from.
    """
    vectors = _draw(seed=13)  # seed 13
    for nbits in SUPPORTED_NBITS:
        codec = ResidualCompressionCodec(
            dim=DIM, nbits=nbits, num_centroids=NUM_CENTROIDS, seed=13
        ).fit(vectors)
        decoded = codec.decode(*codec.encode(codec.centroids))
        similarities = decoded @ codec.centroids.T
        winners = np.argmax(similarities, axis=1)
        np.testing.assert_array_equal(
            winners,
            np.arange(NUM_CENTROIDS),
            err_msg=(
                f"a decoded centroid's nearest codebook entry is no longer "
                f"itself at nbits={nbits}"
            ),
        )


# ---------------------------------------------------------------------------
# 3. Serialization round trips
# ---------------------------------------------------------------------------


def test_get_config_round_trip_reproduces_codes_and_decoded_vectors(
    fitted_codec: ResidualCompressionCodec,
) -> None:
    """``from_config(get_config())`` must be the same codec, bit for bit.

    A config that carries the settings but drops the codebook would produce a
    codec that raises, or worse, one that re-fits to a different codebook; both
    are caught by comparing codes and decoded values exactly.
    """
    probe = _draw(seed=21, num_vectors=32)  # seed 21
    codes, packed = fitted_codec.encode(probe)
    decoded = fitted_codec.decode(codes, packed)

    restored = ResidualCompressionCodec.from_config(fitted_codec.get_config())
    restored_codes, restored_packed = restored.encode(probe)

    np.testing.assert_array_equal(
        restored_codes,
        codes,
        err_msg="from_config() codec assigned different centroid codes",
    )
    np.testing.assert_array_equal(
        restored_packed,
        packed,
        err_msg="from_config() codec packed different residual bits",
    )
    np.testing.assert_allclose(
        restored.decode(restored_codes, restored_packed),
        decoded,
        atol=0.0,
        rtol=0.0,
        err_msg="from_config() codec decoded to different vectors",
    )


def test_npz_save_load_round_trip_reproduces_codes_and_decoded_vectors(
    fitted_codec: ResidualCompressionCodec, tmp_path
) -> None:
    """``save()`` / ``load()`` must round-trip the codebook exactly.

    Routed through ``tmp_path`` -- nothing is written under repo-root
    ``results/``.
    """
    probe = _draw(seed=22, num_vectors=32)  # seed 22
    codes, packed = fitted_codec.encode(probe)
    decoded = fitted_codec.decode(codes, packed)

    destination = str(tmp_path / "codec.npz")
    fitted_codec.save(destination)
    restored = ResidualCompressionCodec.load(destination)

    assert restored.nbits == fitted_codec.nbits
    assert restored.dim == fitted_codec.dim
    assert restored.num_centroids == fitted_codec.num_centroids

    restored_codes, restored_packed = restored.encode(probe)
    np.testing.assert_array_equal(
        restored_codes,
        codes,
        err_msg="load()ed codec assigned different centroid codes",
    )
    np.testing.assert_allclose(
        restored.decode(restored_codes, restored_packed),
        decoded,
        atol=0.0,
        rtol=0.0,
        err_msg="load()ed codec decoded to different vectors",
    )


# ---------------------------------------------------------------------------
# 4. Validation surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbits", [0, 3, 4, -1, 8])
def test_an_unsupported_nbits_raises_naming_the_value(nbits: int) -> None:
    """``nbits`` outside ``{1, 2}`` must raise, naming the offending value.

    The reference defines only 1- and 2-bit residuals; anything else would
    bucketize garbage against a codebook of the wrong size.
    """
    with pytest.raises(ValueError, match=str(nbits)):
        ResidualCompressionCodec(dim=DIM, nbits=nbits)


def test_fitting_more_centroids_than_vectors_raises() -> None:
    """``k`` larger than the sample cannot be satisfied and must raise."""
    codec = ResidualCompressionCodec(dim=DIM, nbits=1, num_centroids=64, seed=0)
    with pytest.raises(ValueError, match="64"):
        codec.fit(_draw(seed=31, num_vectors=10))  # seed 31


def test_an_empty_sample_raises() -> None:
    """An empty input has nothing to compress and must raise, not return."""
    codec = ResidualCompressionCodec(dim=DIM, nbits=1, num_centroids=4, seed=0)
    with pytest.raises(ValueError, match="empty"):
        codec.fit(np.zeros((0, DIM)))


def test_a_dim_mismatch_at_encode_time_raises(
    fitted_codec: ResidualCompressionCodec,
) -> None:
    """Encoding vectors of the wrong width must raise, not broadcast."""
    with pytest.raises(ValueError, match="dim"):
        fitted_codec.encode(_draw(seed=32, num_vectors=4, dim=DIM + 1))  # seed 32


def test_an_empty_encode_input_raises(
    fitted_codec: ResidualCompressionCodec,
) -> None:
    """Encoding zero vectors must raise rather than return empty arrays."""
    with pytest.raises(ValueError, match="empty"):
        fitted_codec.encode(np.zeros((0, DIM)))


def test_encoding_before_fitting_raises() -> None:
    """An unfitted codec has no codebook and must say so."""
    codec = ResidualCompressionCodec(dim=DIM, nbits=1, num_centroids=4, seed=0)
    assert not codec.is_fitted
    with pytest.raises(RuntimeError, match="codebook"):
        codec.encode(_draw(seed=33, num_vectors=4))  # seed 33


def test_decoding_an_out_of_range_code_raises(
    fitted_codec: ResidualCompressionCodec,
) -> None:
    """A centroid index outside the codebook must raise, not index-wrap."""
    codes, packed = fitted_codec.encode(_draw(seed=34, num_vectors=4))  # seed 34
    codes = codes.copy()
    codes[0] = NUM_CENTROIDS + 5
    with pytest.raises(ValueError, match="outside"):
        fitted_codec.decode(codes, packed)


# ---------------------------------------------------------------------------
# 5. The index-time boundary (H-7)
# ---------------------------------------------------------------------------


def _codec_symbol_names() -> set:
    """Collect every name ``compression.py`` defines at module level.

    :returns: Set of defined symbol names (classes, functions, constants).
    :rtype: set
    """
    names = set()
    for name, value in vars(compression_module).items():
        if name.startswith("__"):
            continue
        if getattr(value, "__module__", None) == compression_module.__name__:
            names.add(name)
    # Module-level constants carry no ``__module__``; add them explicitly by
    # parsing the module's own AST rather than hard-coding a list.
    tree = ast.parse(inspect.getsource(compression_module))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _executable_identifiers(module) -> set:
    """Every identifier a module references in *executable* code.

    Parsing to an AST is what makes this guard meaningful: docstrings and
    comments are dropped, so a module that merely *mentions* the codec in prose
    (``model.py`` does, in ``create_colbert_v2``'s docstring) is not a hit,
    while a real call or import is.

    :param module: An imported module object.
    :returns: Names appearing as ``Name`` ids, attribute names, or imported
        aliases anywhere in the module's code.
    :rtype: set
    """
    tree = ast.parse(inspect.getsource(module))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(node.module.split("."))
    return found


def test_no_codec_symbol_is_reachable_from_the_model_or_the_losses() -> None:
    """The codec must not be statically reachable from the model or the losses.

    **What this proves.** Neither ``colbert/model.py`` nor
    ``losses/colbert_loss.py`` references, imports, or calls any name defined in
    ``compression.py``, anywhere in its executable code -- including inside
    ``ColBERT.call``, which is checked separately and by name. Prose mentions in
    docstrings are deliberately invisible to this check, because it parses an
    AST rather than grepping text.

    **What this does not prove.** It cannot see dynamic access -- an
    ``importlib.import_module`` of a name assembled at runtime, or a
    ``getattr`` on a string -- and it says nothing about modules other than the
    two checked here. It is a static-reference guard, which is the shape the
    boundary actually takes in this package.
    """
    from dl_techniques.losses import colbert_loss as loss_module
    from dl_techniques.models.language.colbert import model as model_module

    codec_symbols = _codec_symbol_names()
    assert "ResidualCompressionCodec" in codec_symbols, (
        "the codec symbol inventory is empty of the codec class itself; the "
        "guard would be vacuous"
    )

    for module in (model_module, loss_module):
        referenced = _executable_identifiers(module)
        leaked = sorted(codec_symbols & referenced)
        assert not leaked, (
            f"codec symbol {leaked[0]!r} is referenced in executable code of "
            f"{module.__name__}; residual compression is index-time only "
            f"(all leaked symbols: {leaked})"
        )

        # A module-object leak (``from . import compression``) would not show up
        # as a codec symbol, so check the namespace for the module itself too.
        for name, value in vars(module).items():
            assert value is not compression_module, (
                f"{module.__name__} binds the compression module as {name!r}; "
                "residual compression is index-time only"
            )

    # And the forward pass specifically, by name.
    call_identifiers = set()
    call_tree = ast.parse(
        inspect.cleandoc(inspect.getsource(model_module.ColBERT.call))
    )
    for node in ast.walk(call_tree):
        if isinstance(node, ast.Name):
            call_identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            call_identifiers.add(node.attr)
    leaked_in_call = sorted(codec_symbols & call_identifiers)
    assert not leaked_in_call, (
        f"codec symbol {leaked_in_call[0]!r} is referenced inside "
        f"ColBERT.call; the codec must never run in the forward pass"
    )


def test_the_codec_is_not_a_keras_layer() -> None:
    """The codec must be a plain object, outside any computation graph.

    A ``keras.Layer`` subclass would be constructible inside a model and would
    acquire build/serialization semantics it has no use for; the boundary is
    partly enforced by simply not being one.
    """
    import keras

    assert not issubclass(ResidualCompressionCodec, keras.layers.Layer), (
        "ResidualCompressionCodec is a keras Layer; it is an index-time codec "
        "and must not be part of any graph"
    )
    assert not issubclass(ResidualCompressionCodec, keras.Model)
    assert "keras" not in _executable_identifiers(compression_module), (
        "compression.py references keras in executable code; the codec is a "
        "numpy-only index-time component"
    )
    assert compression_module.__name__ in sys.modules


# ---------------------------------------------------------------------------
# 6. End-to-end composition: encoder -> codec -> MaxSim
# ---------------------------------------------------------------------------

# Geometry of the compose guard below. Stated here, and named in the docstring,
# because every number in that docstring was measured at exactly this geometry
# and is meaningless without it.
COMPOSE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
COMPOSE_NUM_DOCS = 8
COMPOSE_DOC_LEN = 48
COMPOSE_NUM_CENTROIDS = 16

# Bound on the RELATIVE MaxSim drift, per bit width, derived from the measured
# population in the docstring below with a ~2x margin. Relative rather than
# absolute because a MaxSim score is a sum over ``query_maxlen`` terms and so
# scales with the query length the test happens to use.
COMPOSE_RELATIVE_DRIFT_BOUND = {1: 0.10, 2: 0.08}

# Floor on the DIFFERENTIAL arm: how many times larger the reconstruction error
# of a centroid-only decode (plain 16-way vector quantization -- i.e. the codec
# with its residual stage dead) must be than the real decode's, per bit width.
# This is the arm that makes the residual stage earn its existence; the drift
# bounds above cannot do it (see the docstring's DIFFERENTIAL ARM section).
# Measured over 12 seeds (0..11) at this geometry: the ratio never fell below
# 1.5846 at nbits=1 or 2.5392 at nbits=2, so these floors carry ~20% margin,
# while a dead residual stage pins the ratio at exactly 1.0.
COMPOSE_RESIDUAL_GAIN_FLOOR = {1: 1.3, 2: 2.0}


def test_maxsim_scores_survive_a_round_trip_through_the_index_time_codec() -> None:
    """The codec's reason to exist, composed end to end, on real embeddings.

    Every other test in this module drives the codec with **synthetic** vectors
    and never touches MaxSim. This one runs the whole index-time path that the
    codec exists for::

        ColBERT.from_variant("tiny").encode_document(...)
            -> ResidualCompressionCodec.fit / encode / decode
            -> MaxSimScorer, against an UNCOMPRESSED query

    and compares the MaxSim score of each document computed against its decoded
    embeddings with the score computed against its original ones, at both
    supported bit widths.

    **What this measures, and what it does not (H-3).** No pretrained ColBERT
    weights exist anywhere in this repository, so the encoder here is randomly
    initialized. Every number below is therefore a **score-stability under
    compression** number -- how far a lossy index moves a score relative to an
    exact one -- and is **not**, and must never be read as, a retrieval-quality
    number. Nothing here says the ranking it produces is good; it says the codec
    perturbs whatever ranking the encoder produces by a bounded amount.

    **MEASURED POPULATION.** 8 seeds (``COMPOSE_SEEDS`` = 0..7), one
    independently initialized ``tiny`` encoder each, ``COMPOSE_NUM_DOCS`` = 8
    documents of ``COMPOSE_DOC_LEN`` = 48 tokens against one 32-token query,
    ``COMPOSE_NUM_CENTROIDS`` = 16. Produced 2026-08-25 on GPU 1 by
    ``plans/plan-2026-08-25T165753-704a9bcb/`` step 2 with the standalone
    script form of this test body, invoked as::

        CUDA_VISIBLE_DEVICES=1 .venv/bin/python measure_codec_compose.py

    Max over the 8 documents of ``|score(decoded) - score(original)|``, then per
    seed, absolute and as a fraction of the original score:

    ==== ================== ================== ================== ==================
    seed nbits=1 abs        nbits=1 rel        nbits=2 abs        nbits=2 rel
    ==== ================== ================== ================== ==================
    0    0.459528           0.022214           0.287338           0.013486
    1    0.793819           0.038107           0.657005           0.030867
    2    0.739096           0.033299           0.600113           0.027757
    3    0.524366           0.024562           0.420313           0.019688
    4    0.829851           0.039862           0.574581           0.026862
    5    0.415518           0.020305           0.412127           0.019818
    6    1.092064           0.050728           0.803324           0.037316
    7    0.510862           0.024132           0.374838           0.017707
    ==== ================== ================== ================== ==================

    Population maxima: **nbits=1 -> 1.092064 absolute / 0.050728 relative**;
    **nbits=2 -> 0.803324 absolute / 0.037316 relative**. The asserted bounds
    are ``COMPOSE_RELATIVE_DRIFT_BOUND`` = 0.10 at 1 bit and 0.08 at 2 bits --
    each roughly **2x** its measured maximum, so an ordinary seed-to-seed swing
    cannot redden this guard. What the bounds do **not** do is detect a broken
    decode: they are one-sided tolerances, and the section below records the
    measurement that proves they tolerate the death of the codec's defining
    mechanism. Only a total decode failure (all-zero output, injection (d))
    moves the drift far enough to redden them.

    **DIFFERENTIAL ARM: the drift bounds alone cannot see a dead residual
    stage.** Replacing the reconstruction with ``centroids[codes] +
    residuals * 0.0`` -- which reduces "residual compression" to plain 16-way
    vector quantization and discards every quantized residual bit at both bit
    widths -- leaves the drift assertions GREEN. Measured relative MaxSim drift
    for that centroid-only decode, same geometry, 12 seeds: 0.033810, 0.028049,
    0.030800, 0.010378, 0.049580, 0.037165, 0.058220, 0.020338, 0.023961,
    0.021655, 0.046885, 0.037876 -- worst 0.0582, under both bounds on 12 of 12
    seeds (the worst is below even the tighter 0.08 bound) and, at 1 bit,
    *below* the real decode's own drift on 6 of those 12.
    The cause is H-3: on a randomly initialized encoder the token embeddings do
    not cluster, so a 1-bit residual quantizer adds noise to a MaxSim score as
    often as it removes it. **So the ordering "real decode drifts less than
    centroid-only" is measurably FALSE here and is deliberately NOT asserted**
    -- 6/12 at nbits=1, 10/12 at nbits=2 (seeds 1 and 3 invert it at 2 bits
    too). Asserting it would ship a flaky guard stating something untrue.

    What IS true at every seed, and is asserted, is the same comparison on
    **reconstruction error** rather than on MaxSim -- the quantity the residual
    stage actually optimizes. Mean ``||v_normalized - decoded||`` over the 12
    seeds, real decode vs the centroid-only decode of the same codec:

    ==== ============ ============ ========== ============ ==========
    seed nbits=1      nbits=2      centroid   gain @1bit   gain @2bit
    ==== ============ ============ ========== ============ ==========
    0    0.500249     0.312397     0.793246   1.5857       2.5392
    1    0.484295     0.299281     0.776897   1.6042       2.5959
    2    0.463910     0.288718     0.741446   1.5983       2.5681
    3    0.480875     0.298135     0.772611   1.6067       2.5915
    4    0.498103     0.310091     0.789788   1.5856       2.5470
    5    0.506204     0.312394     0.805901   1.5920       2.5798
    6    0.463585     0.287452     0.740292   1.5969       2.5754
    7    0.478720     0.297228     0.765056   1.5981       2.5740
    8    0.482068     0.300808     0.763893   1.5846       2.5395
    9    0.470125     0.293195     0.751105   1.5977       2.5618
    10   0.461963     0.286271     0.739599   1.6010       2.5836
    11   0.477578     0.299358     0.761298   1.5941       2.5431
    ==== ============ ============ ========== ============ ==========

    The gain ratio never fell below **1.5846** at 1 bit or **2.5392** at 2 bits
    (and the real decode beat centroid-only on **12 of 12** seeds at both bit
    widths). ``COMPOSE_RESIDUAL_GAIN_FLOOR`` asserts 1.3 / 2.0, ~20% under
    those minima, and a dead residual stage pins the ratio at exactly 1.0.

    **WHAT THIS ARM CANNOT DETECT -- the blind band, stated as a number.** The
    floors detect a **dead** residual stage, not a merely **attenuated** one.
    Sweeping the residual scale (``centroids[codes] + residuals * s``, decoded
    and renormalized exactly as ``decode()`` does) over seeds 0-3:

    ======= ============ ============ ==============================
    scale s gain @1bit   gain @2bit   vs floors 1.3 / 2.0
    ======= ============ ============ ==============================
    1.00    1.5857       2.5392       passes (healthy)
    0.75    1.4494       2.0414       **passes both** (2-bit by ~2%)
    0.50    1.2820       1.5400       fails both
    0.25    1.1266       1.2072       fails both
    0.00    1.0000       1.0000       fails both (the dead-stage injection)
    ======= ============ ============ ==============================

    So a residual stage attenuated by up to about **45%** slips past the 1-bit
    floor, and up to about **26%** past the 2-bit floor. Read the "~20% margin"
    above as protection against seed-to-seed swing, **not** as this guard's
    detection power: what it buys is "the residual bits are being used at
    roughly their designed strength or better", not "at exactly full strength".
    Tightening the floors toward the measured minima (~1.50 / ~2.40) would
    narrow the band and is recorded as available future work in decisions.md
    D-011 -- deliberately NOT done here, because it trades a known, stated blind
    band for an unquantified flake risk on hardware where these ratios have not
    been measured.

    **TOP-1 IS DELIBERATELY NOT ASSERTED, AT EITHER BIT WIDTH.** The same 8
    seeds were checked for it and it does **not** hold: top-1 was preserved on
    3 of 8 seeds at ``nbits=1`` and 4 of 8 at ``nbits=2``, and the full
    8-document ordering was preserved on **0 of 8** seeds at both. That is the
    expected consequence of H-3 rather than a codec defect: with a randomly
    initialized encoder the 8 documents carry no ranking signal, so their scores
    are near-ties -- the whole inter-document score spread was about 0.84 at
    seed 0 (``[20.6865, 21.5226]``), which is **smaller than the 1.092
    compression drift measured at seed 6**. A guard asserting top-1 preservation
    here would be asserting something measurably false and would ship flaky. The
    honest guard is asymmetric: the score delta is bounded and is asserted; the
    ranking is not preserved and is not asserted.

    Also recorded, not asserted: the 2-bit drift was below the 1-bit drift on
    all 8 seeds, but at seed 5 by only 0.0034 absolute -- too thin a margin to
    make a load-bearing assertion out of. The strict bit-width property is
    already guarded, on reconstruction error rather than on MaxSim, by
    ``test_two_bits_reconstruct_strictly_better_than_one_bit``.

    RED-PROOF OF THE DIFFERENTIAL ARM (injection (e), 2026-08-25). Line
    ``reconstructed = self.centroids[code_array] + residuals`` in
    ``ResidualCompressionCodec.decode`` was replaced in place in ``src/`` with
    ``... + residuals * 0.0``. RED: this test, at the named assertion "the
    residual stage did not earn its existence at nbits=1 (seed 0): dropping it
    -- decoding to the bare centroid -- costs only 1.0000x the reconstruction
    error", while every drift assertion above still passed. Restored from a
    ``cp`` backup and verified byte-identical with ``diff -q``.

    RED-PROOF (injection (d), 2026-08-25). ``ResidualCompressionCodec.decode``
    was replaced in place in ``src/`` -- never a scratch copy, because
    ``pyproject.toml``'s ``pythonpath = ["src"]`` overrides ``PYTHONPATH`` and a
    copy reads a false green -- with a stub returning
    ``np.zeros((len(codes), self.dim))``. RED: this test, at the named
    assertion "compression moved a MaxSim score by ... of its uncompressed
    value at nbits=1", observed relative drift 1.000000 against the 0.10 bound.
    Restored from a ``cp`` backup and verified byte-identical with ``diff -q``.

    This test does not weaken
    ``test_no_codec_symbol_is_reachable_from_the_model_or_the_losses``: that
    guard AST-parses ``model.py`` and ``colbert_loss.py``, so its subject is the
    production source's call graph, not any test module's imports. Composition
    at index time, in a caller, is exactly what the boundary licenses.
    """
    import keras

    from dl_techniques.models.language.colbert.components import MaxSimScorer
    from dl_techniques.models.language.colbert.model import ColBERT

    for seed in COMPOSE_SEEDS:
        keras.utils.set_random_seed(seed)
        model = ColBERT.from_variant("tiny")
        rng = np.random.default_rng(seed)

        # Fully unmasked documents: a padded position is zeroed by the
        # participation mask, and an all-zero row has no direction for a
        # nearest-centroid rule to find. Padding behaviour is guarded
        # elsewhere; this test is about the codec round trip.
        doc_ids = rng.integers(
            0, model.vocab_size, (COMPOSE_NUM_DOCS, COMPOSE_DOC_LEN)
        ).astype("int32")
        query_ids = rng.integers(
            0, model.vocab_size, (1, model.query_maxlen)
        ).astype("int32")

        documents = np.asarray(
            keras.ops.convert_to_numpy(
                model.encode_document(
                    {
                        "input_ids": doc_ids,
                        "attention_mask": np.ones(
                            (COMPOSE_NUM_DOCS, COMPOSE_DOC_LEN), dtype="int32"
                        ),
                    },
                    training=False,
                )
            ),
            dtype=np.float64,
        )
        query = np.asarray(
            keras.ops.convert_to_numpy(
                model.encode_query(
                    {
                        "input_ids": query_ids,
                        "attention_mask": np.ones(
                            (1, model.query_maxlen), dtype="int32"
                        ),
                    },
                    training=False,
                )
            ),
            dtype=np.float64,
        )

        scorer = MaxSimScorer(mask_value=model.mask_value)
        queries = np.repeat(query, COMPOSE_NUM_DOCS, axis=0)
        uncompressed = np.asarray(
            keras.ops.convert_to_numpy(scorer(queries, documents)), dtype=np.float64
        )
        assert np.all(np.abs(uncompressed) > 0.0), (
            f"seed {seed}: an uncompressed MaxSim score is exactly zero, so the "
            "relative drift below would be undefined and the guard vacuous"
        )

        flat = documents.reshape(-1, model.dim)
        truth = compression_module._colbert_codec_l2_normalize(flat)
        for nbits in SUPPORTED_NBITS:
            codec = ResidualCompressionCodec(
                dim=model.dim,
                nbits=nbits,
                num_centroids=COMPOSE_NUM_CENTROIDS,
                seed=seed,
            ).fit(flat)
            codes, packed = codec.encode(flat)
            decoded_flat = codec.decode(codes, packed)
            decoded = decoded_flat.reshape(documents.shape)

            compressed = np.asarray(
                keras.ops.convert_to_numpy(scorer(queries, decoded)),
                dtype=np.float64,
            )
            assert np.all(np.isfinite(compressed)), (
                f"seed {seed}, nbits={nbits}: scoring against decoded "
                f"embeddings produced a non-finite MaxSim score: {compressed}"
            )

            relative = float(
                np.max(np.abs(compressed - uncompressed) / np.abs(uncompressed))
            )
            bound = COMPOSE_RELATIVE_DRIFT_BOUND[nbits]
            assert relative <= bound, (
                f"compression moved a MaxSim score by {relative:.6f} of its "
                f"uncompressed value at nbits={nbits} (seed {seed}), above the "
                f"{bound} bound derived from an 8-seed population whose maximum "
                f"was 0.050728 at nbits=1 and 0.037316 at nbits=2"
            )

            # DECISION plan-2026-08-25T165753-704a9bcb/D-009
            # This arm compares RECONSTRUCTION ERROR, not MaxSim drift. Do NOT
            # "simplify" it to `relative_drift(real) <= relative_drift(
            # centroid_only)`: that ordering was measured over 12 seeds at this
            # geometry and HOLDS ONLY 6/12 at nbits=1 and 10/12 at nbits=2 (see
            # the docstring's DIFFERENTIAL ARM table). Under H-3 a randomly
            # initialized encoder produces unclustered embeddings, so a 1-bit
            # residual quantizer perturbs a MaxSim score as often as it helps
            # it. Asserting the MaxSim ordering ships a flaky guard that states
            # something untrue; the reconstruction-error ordering holds 12/12
            # with a >=1.58x margin. See decisions.md D-009.
            # DIFFERENTIAL ARM. The bound above is a one-sided tolerance and a
            # dead residual stage passes it (measured; see the docstring). This
            # arm compares the real decode against the SAME codec with its
            # residual stage removed -- a centroid-only decode, renormalized
            # exactly as decode() renormalizes -- and requires the residual
            # bits to buy a measured factor of reconstruction accuracy.
            centroid_only = compression_module._colbert_codec_l2_normalize(
                codec.centroids[codes]
            )
            error_real = float(
                np.linalg.norm(truth - decoded_flat, axis=-1).mean()
            )
            error_centroid_only = float(
                np.linalg.norm(truth - centroid_only, axis=-1).mean()
            )
            assert error_real > 0.0, (
                f"seed {seed}, nbits={nbits}: the real decode reconstructs the "
                "embeddings exactly, which is impossible for a lossy codec and "
                "makes the gain ratio below undefined"
            )
            gain = error_centroid_only / error_real
            floor = COMPOSE_RESIDUAL_GAIN_FLOOR[nbits]
            assert gain >= floor, (
                f"the residual stage did not earn its existence at nbits="
                f"{nbits} (seed {seed}): dropping it -- decoding to the bare "
                f"centroid -- costs only {gain:.4f}x the reconstruction error "
                f"({error_centroid_only:.6f} vs {error_real:.6f}), below the "
                f"{floor} floor derived from a 12-seed population whose "
                "minimum was 1.5846 at nbits=1 and 2.5392 at nbits=2; a gain "
                "of exactly 1.0 means the residual bits are being discarded"
            )
