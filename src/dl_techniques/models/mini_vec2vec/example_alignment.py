"""
Example script demonstrating MiniVec2VecAligner usage.

This script shows how to:
1. Generate synthetic aligned embedding spaces
2. Use MiniVec2VecAligner to recover the alignment
3. Evaluate alignment quality **in the frame the map was fitted in**
4. Test serialization

The third point is the part that is easy to get wrong, and this script used to
get it wrong. `align` mean-centers both spaces and L2-normalizes every row, and
it does NOT store the two mean vectors — `call` is only `X @ W`. Applying `W`
to raw embeddings therefore evaluates the map in a frame it was never fitted
in. The bug was invisible here because the synthetic fixture is generated
already unit-normed and near-zero-mean, so `align_frame` was very close to the
identity on it. Real embedding spaces are not centered, and there the same code
silently reports a much worse alignment than the model actually learned.
`align_frame` below is the one place that reproduces the fitted frame; both
evaluation and the "transform new embeddings" demo go through it.
"""

import os
import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Dict
from sklearn.neighbors import NearestNeighbors


from dl_techniques.utils.logger import logger
from dl_techniques.models.mini_vec2vec import MiniVec2VecAligner


def generate_synthetic_data(
        n_samples: int = 25000,
        n_eval: int = 5000,
        embed_dim: int = 128,
        n_clusters: int = 20,
        cluster_noise: float = 0.3,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic aligned embedding spaces for testing.

    Creates two embedding spaces where space_B is a random orthogonal
    transformation of space_A. This provides ground truth for evaluation.

    The base cloud is a MIXTURE of `n_clusters` gaussians, not isotropic noise.
    That is not decoration: stage 2 of the algorithm matches the two spaces by
    solving a quadratic assignment between their k-means centroid Gram
    matrices, so a cloud with no cluster structure gives it nothing to match
    and the whole pipeline returns a map no better than chance. MEASURED with
    isotropic data at these defaults: final Frobenius error 5.23 against
    ||Q||_F = sqrt(embed_dim), i.e. no recovery at all.

    Keep `align`'s `approx_clusters` matched to `n_clusters`. Also measured:
    with 20 modes and `approx_clusters=8`, k-means finds a different arbitrary
    8-way grouping in each space, the centroid sets are then not permutations
    of one another, and recovery fails exactly as it does on isotropic data
    (5.31). With both at 20 the error is 0.157.

    Args:
        n_samples: Number of samples for alignment.
        n_eval: Number of samples for evaluation.
        embed_dim: Embedding dimensionality.
        n_clusters: Number of gaussian modes in the base cloud.
        cluster_noise: Standard deviation around each mode.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (XA_align, XB_align, XA_eval, XB_eval, ground_truth_W)
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info("Generating synthetic data...")

    # Create base embeddings from a gaussian mixture, then normalize.
    centers = np.random.randn(n_clusters, embed_dim).astype(np.float32) * 3.0
    labels = np.random.randint(0, n_clusters, size=n_samples + n_eval)
    base_embeddings = (
        centers[labels]
        + np.random.randn(n_samples + n_eval, embed_dim).astype(np.float32)
        * cluster_noise
    )
    base_embeddings /= np.linalg.norm(base_embeddings, axis=1, keepdims=True)

    # Create a random orthogonal matrix (ground truth transformation)
    random_matrix, _ = np.linalg.qr(np.random.randn(embed_dim, embed_dim))

    # Create the two spaces
    space_A = base_embeddings
    space_B = base_embeddings @ random_matrix

    # Split into alignment and evaluation sets
    XA_align, XB_align = space_A[:n_samples], space_B[:n_samples]
    XA_eval, XB_eval = space_A[n_samples:], space_B[n_samples:]

    logger.info(
        f"Data generated: Alignment set size: {n_samples}, "
        f"Eval set size: {n_eval}"
    )

    return XA_align, XB_align, XA_eval, XB_eval, random_matrix


def align_frame(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Reproduce ``align``'s preprocessing for new embeddings.

    ``MiniVec2VecAligner.align`` centers each space by the mean of the arrays
    it was given and then L2-normalizes every row (``model.py``, stage 1). Those
    means are local to that call, so a caller applying ``W`` later must pass the
    SAME mean here — the one computed over the alignment set, not over the new
    batch.

    Args:
        X: Raw embeddings, shape ``(n, embedding_dim)``.
        mean: The ``(1, embedding_dim)`` mean of the space these embeddings
            belong to, computed over the ALIGNMENT set.

    Returns:
        Embeddings in the frame ``W`` maps between.
    """
    Xp = X - mean
    return Xp / np.linalg.norm(Xp, axis=1, keepdims=True)


def compute_top1_accuracy(
        XA_aligned: np.ndarray,
        XB_true: np.ndarray
) -> float:
    """
    Calculate Top-1 retrieval accuracy.

    For each aligned source embedding, checks if its nearest neighbor
    in the target space is the corresponding true target.

    Args:
        XA_aligned: Aligned source embeddings.
        XB_true: True target embeddings.

    Returns:
        Top-1 accuracy (0 to 1).
    """
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    nn.fit(XB_true)
    _, indices = nn.kneighbors(XA_aligned)

    correct_matches = np.sum(indices.flatten() == np.arange(len(XA_aligned)))
    return correct_matches / len(XA_aligned)


def compute_mean_cosine_similarity(
        XA_aligned: np.ndarray,
        XB_true: np.ndarray
) -> float:
    """
    Calculate mean cosine similarity between aligned pairs.

    Computes the average cosine similarity between each aligned
    source embedding and its corresponding true target.

    Args:
        XA_aligned: Aligned source embeddings.
        XB_true: True target embeddings.

    Returns:
        Mean cosine similarity (0 to 1).
    """
    # Normalize embeddings
    XA_norm = XA_aligned / np.linalg.norm(XA_aligned, axis=1, keepdims=True)
    XB_norm = XB_true / np.linalg.norm(XB_true, axis=1, keepdims=True)

    # Element-wise product and sum
    cosine_sims = np.sum(XA_norm * XB_norm, axis=1)
    return np.mean(cosine_sims)


def compute_transformation_error(
        learned_W: np.ndarray,
        ground_truth_W: np.ndarray
) -> float:
    """
    Compute Frobenius norm error between learned and ground truth W.

    Note: Due to sign ambiguity in orthogonal matrices, we compute
    the minimum error considering sign flips.

    Args:
        learned_W: Learned transformation matrix.
        ground_truth_W: Ground truth transformation matrix.

    Returns:
        Frobenius norm error.
    """
    error_pos = np.linalg.norm(learned_W - ground_truth_W, ord='fro')
    error_neg = np.linalg.norm(learned_W + ground_truth_W, ord='fro')
    return min(error_pos, error_neg)


def evaluate_alignment(
        aligner: MiniVec2VecAligner,
        XA_eval: np.ndarray,
        XB_eval: np.ndarray,
        mean_A: np.ndarray,
        mean_B: np.ndarray,
        ground_truth_W: Optional[np.ndarray] = None,
        stage: str = "final"
) -> Dict[str, float]:
    """
    Evaluate alignment quality with multiple metrics.

    Args:
        aligner: Trained MiniVec2VecAligner model.
        XA_eval: Evaluation source embeddings.
        XB_eval: Evaluation target embeddings.
        mean_A: Mean of the source space over the ALIGNMENT set.
        mean_B: Mean of the target space over the ALIGNMENT set.
        ground_truth_W: Optional ground truth transformation for error computation.
        stage: Stage name for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Transform evaluation embeddings IN THE FITTED FRAME. Both sides go
    # through `align_frame`: `W` maps processed-A to processed-B, so comparing
    # against a raw `XB_eval` would be a second frame error on the target side.
    XA_proc = align_frame(XA_eval, mean_A)
    XB_proc = align_frame(XB_eval, mean_B)
    aligned_A = ops.convert_to_numpy(aligner(XA_proc))

    # Compute metrics
    metrics = {
        'top1_accuracy': compute_top1_accuracy(aligned_A, XB_proc),
        'mean_cosine_sim': compute_mean_cosine_similarity(aligned_A, XB_proc)
    }

    # Add transformation error if ground truth available
    if ground_truth_W is not None:
        learned_W = ops.convert_to_numpy(aligner.W)
        metrics['transformation_error'] = compute_transformation_error(
            learned_W, ground_truth_W
        )

    # Log metrics
    logger.info(f"\n--- Evaluation Results ({stage}) ---")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    return metrics


def test_serialization(
        aligner: MiniVec2VecAligner,
        XA_eval: np.ndarray,
        save_dir: str = "temp_models"
) -> None:
    """
    Test model serialization and deserialization.

    Args:
        aligner: Trained MiniVec2VecAligner model.
        XA_eval: Evaluation embeddings for consistency check.
        save_dir: Directory to save the model.

    Raises:
        AssertionError: If loaded model predictions don't match original.
    """
    logger.info("\n--- Testing Model Serialization ---")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "mini_vec2vec_aligner.keras")

    # Get prediction from original model
    original_pred = aligner(XA_eval)

    # Save model
    aligner.save(filepath)
    logger.info(f"Model saved to {filepath}")

    # Load model
    loaded_aligner = keras.models.load_model(filepath)
    logger.info("Model loaded successfully")

    # Get prediction from loaded model
    loaded_pred = loaded_aligner(XA_eval)

    # Verify predictions match
    np.testing.assert_allclose(
        ops.convert_to_numpy(original_pred),
        ops.convert_to_numpy(loaded_pred),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Loaded model predictions should match original"
    )

    logger.info("✓ Serialization test PASSED: Predictions match")

    # Cleanup
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        logger.info(f"Cleaned up temporary directory: {save_dir}")


def run_alignment_example(
        n_samples: int = 25000,
        n_eval: int = 5000,
        embed_dim: int = 128,
        # Alignment hyperparameters
        approx_clusters: int = 20,
        approx_runs: int = 30,
        approx_neighbors: int = 10,
        refine1_iterations: int = 50,
        refine1_sample_size: int = 5000,
        refine1_neighbors: int = 10,
        refine2_clusters: int = 200,
        smoothing_alpha: float = 0.5,
        seed: Optional[int] = 42
) -> Tuple[MiniVec2VecAligner, np.ndarray, np.ndarray]:
    """
    Run complete alignment example with evaluation and testing.

    Args:
        n_samples: Number of samples for alignment.
        n_eval: Number of samples for evaluation.
        embed_dim: Embedding dimensionality.
        approx_clusters: Number of clusters for anchor alignment.
        approx_runs: Number of anchor alignment runs.
        approx_neighbors: Number of neighbors for pseudo-pairs.
        refine1_iterations: Refine-1 iteration count.
        refine1_sample_size: Samples per Refine-1 iteration.
        refine1_neighbors: Neighbors for Refine-1.
        refine2_clusters: Clusters for Refine-2.
        smoothing_alpha: Smoothing factor for updates.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(aligner, mean_A, mean_B)``. The two means are returned, not
        just the model, because ``W`` is only meaningful together with the frame
        it was fitted in and the model does not carry it — see ``align_frame``.
    """
    logger.info("=" * 70)
    logger.info("Mini-Vec2Vec Alignment Example")
    logger.info("=" * 70)

    # ===== Step 1: Generate Data =====
    XA_align, XB_align, XA_eval, XB_eval, ground_truth_W = generate_synthetic_data(
        n_samples=n_samples,
        n_eval=n_eval,
        embed_dim=embed_dim,
        seed=seed
    )

    # The frame `align` will fit in: the mean of each space over the ALIGNMENT
    # set. `align` computes these internally and does not keep them, so anything
    # that applies `W` afterwards has to recompute them from the same arrays.
    mean_A = XA_align.mean(axis=0, keepdims=True)
    mean_B = XB_align.mean(axis=0, keepdims=True)

    # ===== Step 2: Create Model =====
    logger.info("\n--- Initializing MiniVec2VecAligner ---")
    aligner = MiniVec2VecAligner(embedding_dim=embed_dim)
    # `align` builds the model itself; the explicit build here is only so the
    # BEFORE-alignment evaluation below has a `W` to read.
    aligner.build(input_shape=(None, embed_dim))
    logger.info(f"Model created with embedding_dim={embed_dim}")

    # ===== Step 3: Evaluate Before Alignment =====
    evaluate_alignment(
        aligner,
        XA_eval,
        XB_eval,
        mean_A,
        mean_B,
        ground_truth_W,
        stage="BEFORE alignment"
    )

    # ===== Step 4: Run Alignment =====
    logger.info("\n" + "=" * 70)
    logger.info("Starting Alignment Process")
    logger.info("=" * 70)

    history = aligner.align(
        XA=XA_align,
        XB=XB_align,
        approx_clusters=approx_clusters,
        approx_runs=approx_runs,
        approx_neighbors=approx_neighbors,
        refine1_iterations=refine1_iterations,
        refine1_sample_size=refine1_sample_size,
        refine1_neighbors=refine1_neighbors,
        refine2_clusters=refine2_clusters,
        smoothing_alpha=smoothing_alpha
    )

    # ===== Step 5: Evaluate After Alignment =====
    metrics = evaluate_alignment(
        aligner,
        XA_eval,
        XB_eval,
        mean_A,
        mean_B,
        ground_truth_W,
        stage="AFTER alignment"
    )

    # ===== Step 6: Test Serialization =====
    test_serialization(aligner, XA_eval)

    # ===== Summary =====
    logger.info("\n" + "=" * 70)
    logger.info("Alignment Complete - Summary")
    logger.info("=" * 70)
    logger.info(f"Final Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
    logger.info(f"Final Mean Cosine Similarity: {metrics['mean_cosine_sim']:.4f}")
    if 'transformation_error' in metrics:
        logger.info(
            f"Transformation Error (Frobenius): "
            f"{metrics['transformation_error']:.4f}"
        )
    logger.info("=" * 70)

    return aligner, mean_A, mean_B


if __name__ == "__main__":
    """
    Main entry point for the example script.

    Runs alignment with default hyperparameters tuned for good performance
    on synthetic data. For real-world applications, these may need tuning.
    """
    # Run example with default parameters
    trained_aligner, mean_A, mean_B = run_alignment_example()

    # Example of using the trained aligner for new embeddings. The raw batch is
    # put into the fitted frame FIRST — feeding `new_embeddings` straight in
    # would apply `W` in a frame it was not fitted in.
    logger.info("\n--- Example: Transform New Embeddings ---")
    new_embeddings = np.random.randn(100, 128).astype(np.float32)
    new_embeddings /= np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    transformed = trained_aligner(align_frame(new_embeddings, mean_A))
    logger.info(
        f"Transformed {new_embeddings.shape[0]} embeddings: "
        f"{new_embeddings.shape} -> {transformed.shape}"
    )