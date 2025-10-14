"""
Example script demonstrating MiniVec2VecAligner usage.

This script shows how to:
1. Generate synthetic aligned embedding spaces
2. Use MiniVec2VecAligner to recover the alignment
3. Evaluate alignment quality
4. Test serialization
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
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic aligned embedding spaces for testing.

    Creates two embedding spaces where space_B is a random orthogonal
    transformation of space_A. This provides ground truth for evaluation.

    Args:
        n_samples: Number of samples for alignment.
        n_eval: Number of samples for evaluation.
        embed_dim: Embedding dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (XA_align, XB_align, XA_eval, XB_eval, ground_truth_W)
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info("Generating synthetic data...")

    # Create base embeddings (normalized)
    base_embeddings = np.random.randn(
        n_samples + n_eval,
        embed_dim
    ).astype(np.float32)
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
        ground_truth_W: Optional[np.ndarray] = None,
        stage: str = "final"
) -> Dict[str, float]:
    """
    Evaluate alignment quality with multiple metrics.

    Args:
        aligner: Trained MiniVec2VecAligner model.
        XA_eval: Evaluation source embeddings.
        XB_eval: Evaluation target embeddings.
        ground_truth_W: Optional ground truth transformation for error computation.
        stage: Stage name for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Transform evaluation embeddings
    aligned_A = ops.convert_to_numpy(aligner(XA_eval))

    # Compute metrics
    metrics = {
        'top1_accuracy': compute_top1_accuracy(aligned_A, XB_eval),
        'mean_cosine_sim': compute_mean_cosine_similarity(aligned_A, XB_eval)
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
) -> MiniVec2VecAligner:
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
        Trained MiniVec2VecAligner model.
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

    # ===== Step 2: Create and Build Model =====
    logger.info("\n--- Initializing MiniVec2VecAligner ---")
    aligner = MiniVec2VecAligner(embedding_dim=embed_dim)
    aligner.build(input_shape=(None, embed_dim))
    logger.info(f"Model created with embedding_dim={embed_dim}")

    # ===== Step 3: Evaluate Before Alignment =====
    evaluate_alignment(
        aligner,
        XA_eval,
        XB_eval,
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

    return aligner


if __name__ == "__main__":
    """
    Main entry point for the example script.

    Runs alignment with default hyperparameters tuned for good performance
    on synthetic data. For real-world applications, these may need tuning.
    """
    # Run example with default parameters
    trained_aligner = run_alignment_example()

    # Example of using the trained aligner for new embeddings
    logger.info("\n--- Example: Transform New Embeddings ---")
    new_embeddings = np.random.randn(100, 128).astype(np.float32)
    new_embeddings /= np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    transformed = trained_aligner(new_embeddings)
    logger.info(
        f"Transformed {new_embeddings.shape[0]} embeddings: "
        f"{new_embeddings.shape} -> {transformed.shape}"
    )