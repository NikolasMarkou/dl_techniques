"""
Residual compression: the one artifact-level addition ColBERTv2 makes to
ColBERT. :class:`ResidualCompressionCodec` is a plain Python object, not a
Keras layer, used to shrink a trained index after training finishes.

A late-interaction index stores one vector per token rather than one per
passage, so a 128-d ``float16`` embedding costs 256 bytes and a large
collection needs terabytes. Token embeddings of a trained ColBERT cluster
tightly, so each one is stored as an integer centroid id from k-means plus a
small residual, crushed to one or two bits per dimension. At ``nbits=1`` a
128-d vector compresses to 16 bytes of residual plus a code.

This is index-time only: nothing here is differentiable or reachable from
:meth:`ColBERT.call` or from either ColBERT loss. Compression runs once, over
an already-trained encoder's outputs; decompression runs in the search
engine when candidate vectors are read back.

:meth:`fit` runs spherical k-means over a sample of embeddings to get
centroids. :meth:`encode` finds the nearest centroid by inner product and
bucketizes the residual per dimension into equiprobable levels. :meth:`decode`
unpacks the levels, adds the centroid, and re-normalizes, since a decoded
vector generally lands off the unit sphere and MaxSim needs it back on it.

The bucket-boundary derivation here is this implementation's own, not a
transcription of the reference's undisclosed cutoff computation: cutoffs are
equiprobable quantiles of the pooled residual distribution.

References:
    - Santhanam et al., 2022. ColBERTv2: Effective and Efficient Retrieval via
      Lightweight Late Interaction. (https://arxiv.org/abs/2112.01488)
    - Khattab and Zaharia, 2020. ColBERT: Efficient and Effective Passage
      Search via Contextualized Late Interaction over BERT.
      (https://arxiv.org/abs/2004.12832)
    - Official implementation, ``colbert/indexing/codecs/residual.py``
      (https://github.com/stanford-futuredata/ColBERT), read 2026-08-25.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: The only residual bit-widths the reference codec supports. Anything else
#: bucketizes garbage rather than failing, so it is rejected at construction.
SUPPORTED_NBITS: Tuple[int, ...] = (1, 2)

#: Numerical floor used when renormalizing, so a reconstructed vector that
#: happens to land on the origin yields zeros rather than ``NaN``.
_NORM_EPSILON: float = 1e-12


# ---------------------------------------------------------------------------
# Private helpers
#
# Names here use the ``_colbert_codec_*`` prefix so the index-time boundary
# test can grep every symbol this module defines against ``model.py`` and the
# loss module.
# ---------------------------------------------------------------------------


def _colbert_codec_l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis with a zero-safe floor.

    :param vectors: Array of shape ``(..., dim)``.
    :type vectors: numpy.ndarray
    :returns: Array of the same shape whose last-axis norms are 1, except for
        rows that were exactly zero, which stay zero.
    :rtype: numpy.ndarray
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, _NORM_EPSILON)


def _colbert_codec_kmeans(
    vectors: np.ndarray,
    num_centroids: int,
    num_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spherical k-means over unit-norm vectors.

    Assignment uses maximum inner product, which coincides with minimum
    Euclidean distance because both the data and the centroids are unit-norm.
    Centroids are re-normalized after every mean update; an empty cluster keeps
    its previous centroid rather than collapsing to a zero vector.

    :param vectors: Unit-norm array of shape ``(num_vectors, dim)``.
    :type vectors: numpy.ndarray
    :param num_centroids: Number of clusters, ``k``.
    :type num_centroids: int
    :param num_iterations: Lloyd iterations to run.
    :type num_iterations: int
    :param rng: Seeded generator used for the initial centroid draw.
    :type rng: numpy.random.Generator
    :returns: Unit-norm centroids of shape ``(num_centroids, dim)``.
    :rtype: numpy.ndarray
    """
    num_vectors = vectors.shape[0]
    initial = rng.choice(num_vectors, size=num_centroids, replace=False)
    centroids = _colbert_codec_l2_normalize(vectors[initial].astype(np.float64))

    for _ in range(num_iterations):
        codes = np.argmax(vectors @ centroids.T, axis=1)
        updated = centroids.copy()
        for cluster in range(num_centroids):
            members = vectors[codes == cluster]
            if members.shape[0] > 0:
                updated[cluster] = members.mean(axis=0)
        centroids = _colbert_codec_l2_normalize(updated)

    return centroids


def _colbert_codec_derive_buckets(
    residuals: np.ndarray, num_levels: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive equiprobable bucket cutoffs and their reconstruction values.

    See this module's docstring for the provenance caveat: this derivation is
    *not* a transcription of the reference implementation.

    :param residuals: Residual array of any shape; quantiles are taken over all
        entries pooled.
    :type residuals: numpy.ndarray
    :param num_levels: ``2 ** nbits`` -- the number of quantization levels.
    :type num_levels: int
    :returns: ``(bucket_cutoffs, bucket_weights)`` of shapes
        ``(num_levels - 1,)`` and ``(num_levels,)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    flat = residuals.reshape(-1)
    cutoff_quantiles = np.arange(1, num_levels, dtype=np.float64) / num_levels
    weight_quantiles = (
        np.arange(num_levels, dtype=np.float64) + 0.5
    ) / num_levels

    bucket_cutoffs = np.quantile(flat, cutoff_quantiles)
    bucket_weights = np.quantile(flat, weight_quantiles)

    # Strictly increasing cutoffs keep ``searchsorted`` single-valued. A
    # degenerate residual distribution (all-identical values) can otherwise
    # produce duplicated quantiles.
    bucket_cutoffs = np.maximum.accumulate(bucket_cutoffs)
    return bucket_cutoffs.astype(np.float64), bucket_weights.astype(np.float64)


def _colbert_codec_pack_levels(levels: np.ndarray, nbits: int) -> np.ndarray:
    """Bit-pack per-dimension level indices, ``nbits`` bits per dimension.

    :param levels: Integer array of shape ``(num_vectors, dim)`` with values in
        ``[0, 2**nbits)``.
    :type levels: numpy.ndarray
    :param nbits: Bits per dimension.
    :type nbits: int
    :returns: ``uint8`` array of shape ``(num_vectors, ceil(dim * nbits / 8))``.
    :rtype: numpy.ndarray
    """
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint8)
    bits = (levels.astype(np.uint8)[..., None] >> shifts) & np.uint8(1)
    bits = bits.reshape(levels.shape[0], -1)
    return np.packbits(bits, axis=-1)


def _colbert_codec_unpack_levels(
    packed: np.ndarray, dim: int, nbits: int
) -> np.ndarray:
    """Inverse of :func:`_colbert_codec_pack_levels`.

    :param packed: ``uint8`` array of shape ``(num_vectors, num_bytes)``.
    :type packed: numpy.ndarray
    :param dim: Vector dimensionality that was packed.
    :type dim: int
    :param nbits: Bits per dimension.
    :type nbits: int
    :returns: Integer array of shape ``(num_vectors, dim)``.
    :rtype: numpy.ndarray
    """
    bits = np.unpackbits(packed.astype(np.uint8), axis=-1)
    bits = bits[:, : dim * nbits].reshape(packed.shape[0], dim, nbits)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.int64)
    return np.sum(bits.astype(np.int64) << shifts, axis=-1)


# ---------------------------------------------------------------------------
# The codec
# ---------------------------------------------------------------------------


class ResidualCompressionCodec:
    """Index-time residual compressor for ColBERTv2 token embeddings.

    A plain Python object, not a :class:`keras.Layer`: it holds no trainable
    state, appears in no computation graph, and must never be reachable from
    :meth:`ColBERT.call` or from any loss. See the module docstring for the
    provenance of the bucket derivation.

    Usage::

        codec = ResidualCompressionCodec(dim=128, nbits=2, num_centroids=256)
        codec.fit(sample_embeddings)          # once, over a sample of the index
        codes, packed = codec.encode(embeddings)
        restored = codec.decode(codes, packed)

    :param dim: Dimensionality of the embeddings this codec handles. Must be
        positive.
    :type dim: int
    :param nbits: Residual bits per dimension. Must be 1 or 2 -- the reference
        codec supports no other width, and any other value would bucketize
        garbage rather than fail.
    :type nbits: int
    :param num_centroids: Number of k-means centroids, ``k``. Must be positive
        and no larger than the number of vectors passed to :meth:`fit`.
    :type num_centroids: int
    :param kmeans_iterations: Lloyd iterations during :meth:`fit`. The
        reference's ``kmeans_niters`` default is 4.
    :type kmeans_iterations: int
    :param seed: Seed for the initial centroid draw. Fixed by default so that
        two codecs fitted on the same sample are identical.
    :type seed: int
    :raises ValueError: If ``nbits`` is not in ``{1, 2}``, or ``dim`` /
        ``num_centroids`` / ``kmeans_iterations`` are not positive.
    """

    def __init__(
        self,
        dim: int = 128,
        nbits: int = 1,
        num_centroids: int = 256,
        kmeans_iterations: int = 4,
        seed: int = 42,
    ) -> None:
        if nbits not in SUPPORTED_NBITS:
            raise ValueError(
                f"nbits must be one of {list(SUPPORTED_NBITS)}, got {nbits}. "
                "The reference ColBERTv2 residual codec quantizes each residual "
                "dimension to 1 or 2 bits; no other width is defined."
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if num_centroids <= 0:
            raise ValueError(
                f"num_centroids must be positive, got {num_centroids}."
            )
        if kmeans_iterations <= 0:
            raise ValueError(
                f"kmeans_iterations must be positive, got {kmeans_iterations}."
            )

        self.dim = int(dim)
        self.nbits = int(nbits)
        self.num_centroids = int(num_centroids)
        self.kmeans_iterations = int(kmeans_iterations)
        self.seed = int(seed)

        self.centroids: Optional[np.ndarray] = None
        self.bucket_cutoffs: Optional[np.ndarray] = None
        self.bucket_weights: Optional[np.ndarray] = None

    # -- properties --------------------------------------------------------

    @property
    def num_levels(self) -> int:
        """Number of quantization levels, ``2 ** nbits``.

        :returns: 2 for ``nbits=1``, 4 for ``nbits=2``.
        :rtype: int
        """
        return 2 ** self.nbits

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` (or :meth:`from_config`) has supplied a codebook.

        :returns: ``True`` when centroids and buckets are present.
        :rtype: bool
        """
        return (
            self.centroids is not None
            and self.bucket_cutoffs is not None
            and self.bucket_weights is not None
        )

    @property
    def bytes_per_vector(self) -> int:
        """Packed residual size in bytes, excluding the centroid code.

        :returns: ``ceil(dim * nbits / 8)``.
        :rtype: int
        """
        return int(np.ceil(self.dim * self.nbits / 8))

    # -- internal validation ----------------------------------------------

    def _validate_matrix(self, vectors: np.ndarray, argument: str) -> np.ndarray:
        """Coerce an input to a 2-D float64 matrix of the codec's width.

        :param vectors: Candidate array.
        :type vectors: numpy.ndarray
        :param argument: Name to quote in error messages.
        :type argument: str
        :returns: Validated ``(num_vectors, dim)`` float64 array.
        :rtype: numpy.ndarray
        :raises ValueError: On a non 2-D input, an empty input, or a width that
            differs from ``self.dim``.
        """
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"{argument} must be a 2-D array of shape (num_vectors, dim), "
                f"got shape {array.shape}."
            )
        if array.shape[0] == 0:
            raise ValueError(
                f"{argument} is empty (shape {array.shape}); there is nothing "
                "to compress."
            )
        if array.shape[1] != self.dim:
            raise ValueError(
                f"{argument} has dim {array.shape[1]}, but this codec was "
                f"configured for dim {self.dim}."
            )
        return array

    def _require_fitted(self) -> None:
        """Raise if no codebook is present.

        :raises RuntimeError: When called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This ResidualCompressionCodec has no codebook. Call fit() on a "
                "sample of document embeddings, or restore one with "
                "from_config()/load()."
            )

    # -- fit / encode / decode --------------------------------------------

    def fit(self, vectors: np.ndarray) -> "ResidualCompressionCodec":
        """Learn centroids and bucket boundaries from a sample of embeddings.

        The sample is L2-normalized on the way in, matching the encoder's own
        output contract and making maximum inner product the correct
        nearest-centroid rule.

        :param vectors: Sample of shape ``(num_vectors, dim)``. Must contain at
            least ``num_centroids`` rows.
        :type vectors: numpy.ndarray
        :returns: ``self``, for chaining.
        :rtype: ResidualCompressionCodec
        :raises ValueError: On an empty sample, a dim mismatch, or a sample
            smaller than ``num_centroids``.
        """
        array = self._validate_matrix(vectors, "vectors")
        if array.shape[0] < self.num_centroids:
            raise ValueError(
                f"Cannot fit {self.num_centroids} centroids to only "
                f"{array.shape[0]} training vectors. Pass a larger sample or "
                "reduce num_centroids."
            )

        normalized = _colbert_codec_l2_normalize(array)
        rng = np.random.default_rng(self.seed)
        self.centroids = _colbert_codec_kmeans(
            normalized, self.num_centroids, self.kmeans_iterations, rng
        )

        codes = np.argmax(normalized @ self.centroids.T, axis=1)
        residuals = normalized - self.centroids[codes]
        self.bucket_cutoffs, self.bucket_weights = _colbert_codec_derive_buckets(
            residuals, self.num_levels
        )

        logger.info(
            "ResidualCompressionCodec fitted: dim=%d, nbits=%d, k=%d, "
            "%d bytes/vector of residual, mean |residual|=%.6f",
            self.dim,
            self.nbits,
            self.num_centroids,
            self.bytes_per_vector,
            float(np.abs(residuals).mean()),
        )
        return self

    def encode(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compress embeddings to centroid codes plus packed residual bits.

        :param vectors: Array of shape ``(num_vectors, dim)``.
        :type vectors: numpy.ndarray
        :returns: ``(codes, packed)`` where ``codes`` is ``int32`` of shape
            ``(num_vectors,)`` and ``packed`` is ``uint8`` of shape
            ``(num_vectors, bytes_per_vector)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises ValueError: On an empty input or a dim mismatch.
        :raises RuntimeError: If the codec is not fitted.
        """
        self._require_fitted()
        array = self._validate_matrix(vectors, "vectors")
        normalized = _colbert_codec_l2_normalize(array)

        codes = np.argmax(normalized @ self.centroids.T, axis=1)
        residuals = normalized - self.centroids[codes]
        levels = np.searchsorted(self.bucket_cutoffs, residuals)
        packed = _colbert_codec_pack_levels(levels, self.nbits)
        return codes.astype(np.int32), packed

    def decode(self, codes: np.ndarray, packed: np.ndarray) -> np.ndarray:
        """Reconstruct unit-norm embeddings from codes and packed residuals.

        :param codes: ``(num_vectors,)`` centroid indices from :meth:`encode`.
        :type codes: numpy.ndarray
        :param packed: ``(num_vectors, bytes_per_vector)`` packed residual bits.
        :type packed: numpy.ndarray
        :returns: ``(num_vectors, dim)`` float64 array of unit-norm vectors.
        :rtype: numpy.ndarray
        :raises ValueError: On a shape mismatch between ``codes`` and ``packed``
            or an out-of-range code.
        :raises RuntimeError: If the codec is not fitted.
        """
        self._require_fitted()
        code_array = np.asarray(codes, dtype=np.int64).reshape(-1)
        packed_array = np.asarray(packed, dtype=np.uint8)
        if packed_array.ndim != 2:
            raise ValueError(
                f"packed must be 2-D, got shape {packed_array.shape}."
            )
        if code_array.shape[0] != packed_array.shape[0]:
            raise ValueError(
                f"codes and packed disagree on the number of vectors: "
                f"{code_array.shape[0]} vs {packed_array.shape[0]}."
            )
        if code_array.size and (
            code_array.min() < 0 or code_array.max() >= self.num_centroids
        ):
            raise ValueError(
                "codes contains an index outside "
                f"[0, {self.num_centroids}); this codec cannot decode it."
            )

        levels = _colbert_codec_unpack_levels(packed_array, self.dim, self.nbits)
        residuals = self.bucket_weights[levels]
        reconstructed = self.centroids[code_array] + residuals

        # DECISION plan-2026-08-25T121346-c71fc3ad/D-018: re-normalize the reconstruction —
        # adding a quantized residual moves it off the unit sphere, which silently rescales MaxSim. See decisions.md.
        return _colbert_codec_l2_normalize(reconstructed)

    def reconstruction_error(self, vectors: np.ndarray) -> float:
        """Mean L2 distance between input vectors and their decoded forms.

        Provided so callers (and tests) can compare bit-widths without
        re-deriving the round trip.

        :param vectors: Array of shape ``(num_vectors, dim)``.
        :type vectors: numpy.ndarray
        :returns: Mean over vectors of ``||v_normalized - decode(encode(v))||``.
        :rtype: float
        """
        array = self._validate_matrix(vectors, "vectors")
        normalized = _colbert_codec_l2_normalize(array)
        restored = self.decode(*self.encode(array))
        return float(np.linalg.norm(normalized - restored, axis=-1).mean())

    # -- serialization -----------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict holding config *and* codebook.

        The learned arrays are included as nested lists, so a config round trip
        restores a fully usable codec rather than an unfitted shell.

        :returns: Constructor arguments plus ``centroids`` / ``bucket_cutoffs``
            / ``bucket_weights`` (``None`` when unfitted).
        :rtype: dict
        """
        return {
            "dim": self.dim,
            "nbits": self.nbits,
            "num_centroids": self.num_centroids,
            "kmeans_iterations": self.kmeans_iterations,
            "seed": self.seed,
            "centroids": (
                None if self.centroids is None else self.centroids.tolist()
            ),
            "bucket_cutoffs": (
                None
                if self.bucket_cutoffs is None
                else self.bucket_cutoffs.tolist()
            ),
            "bucket_weights": (
                None
                if self.bucket_weights is None
                else self.bucket_weights.tolist()
            ),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResidualCompressionCodec":
        """Rebuild a codec from :meth:`get_config` output.

        :param config: Dict as produced by :meth:`get_config`.
        :type config: dict
        :returns: A codec with the same settings and codebook.
        :rtype: ResidualCompressionCodec
        """
        config = dict(config)
        centroids = config.pop("centroids", None)
        bucket_cutoffs = config.pop("bucket_cutoffs", None)
        bucket_weights = config.pop("bucket_weights", None)

        codec = cls(**config)
        if centroids is not None:
            codec.centroids = np.asarray(centroids, dtype=np.float64)
        if bucket_cutoffs is not None:
            codec.bucket_cutoffs = np.asarray(bucket_cutoffs, dtype=np.float64)
        if bucket_weights is not None:
            codec.bucket_weights = np.asarray(bucket_weights, dtype=np.float64)
        return codec

    def save(self, path: str) -> None:
        """Write the codec to a compressed ``.npz`` file.

        :param path: Destination path. ``numpy`` appends ``.npz`` if absent.
        :type path: str
        :raises RuntimeError: If the codec is not fitted.
        """
        self._require_fitted()
        np.savez_compressed(
            path,
            dim=np.int64(self.dim),
            nbits=np.int64(self.nbits),
            num_centroids=np.int64(self.num_centroids),
            kmeans_iterations=np.int64(self.kmeans_iterations),
            seed=np.int64(self.seed),
            centroids=self.centroids,
            bucket_cutoffs=self.bucket_cutoffs,
            bucket_weights=self.bucket_weights,
        )
        logger.info("ResidualCompressionCodec saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ResidualCompressionCodec":
        """Read a codec written by :meth:`save`.

        :param path: Source ``.npz`` path.
        :type path: str
        :returns: The restored codec.
        :rtype: ResidualCompressionCodec
        """
        with np.load(path) as data:
            codec = cls(
                dim=int(data["dim"]),
                nbits=int(data["nbits"]),
                num_centroids=int(data["num_centroids"]),
                kmeans_iterations=int(data["kmeans_iterations"]),
                seed=int(data["seed"]),
            )
            codec.centroids = np.asarray(data["centroids"], dtype=np.float64)
            codec.bucket_cutoffs = np.asarray(
                data["bucket_cutoffs"], dtype=np.float64
            )
            codec.bucket_weights = np.asarray(
                data["bucket_weights"], dtype=np.float64
            )
        return codec

    def __repr__(self) -> str:
        """Readable summary including fitted state.

        :returns: Debug representation.
        :rtype: str
        """
        return (
            f"ResidualCompressionCodec(dim={self.dim}, nbits={self.nbits}, "
            f"num_centroids={self.num_centroids}, fitted={self.is_fitted})"
        )
