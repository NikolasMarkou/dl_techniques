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
