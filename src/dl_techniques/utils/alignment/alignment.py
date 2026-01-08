"""
Main alignment API for measuring representation similarity.

Provides high-level interface for computing alignment scores between
neural network representations using various metrics.
"""

import os
import keras
import tensorflow as tf
import numpy as np
from typing import Union, List, Optional, Dict, Tuple

from .metrics import AlignmentMetrics
from .utils import prepare_features, compute_score, normalize_features


class Alignment:
    """
    High-level API for computing representation alignment.
    
    This class provides a convenient interface for measuring alignment
    between neural network representations using various metrics.
    
    Example:
        ```python
        # Initialize alignment scorer
        scorer = Alignment(
            reference_features=[feat1, feat2],  # Pre-computed features
            metric="mutual_knn",
            topk=10
        )
        
        # Score new features
        score, layer_idx = scorer.score(new_features)
        
        # Or compute alignment between two models
        score, indices = scorer.compute_pairwise_alignment(
            features_a, features_b
        )
        ```
    """
    
    def __init__(
        self,
        reference_features: Optional[List[Union[tf.Tensor, np.ndarray]]] = None,
        metric: str = "mutual_knn",
        topk: int = 10,
        normalize: bool = True,
        device: str = "auto",
        dtype: str = "float32"
    ):
        """
        Initialize alignment scorer.
        
        Args:
            reference_features: Optional list of reference feature tensors
            metric: Alignment metric to use
            topk: Number of neighbors for k-NN metrics
            normalize: Whether to L2-normalize features
            device: Device to use ('auto', 'cpu', 'cuda')
            dtype: Data type for computations
        """
        self.metric = metric
        self.topk = topk
        self.normalize = normalize
        self.dtype = dtype
        
        # Validate metric
        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Supported: {AlignmentMetrics.SUPPORTED_METRICS}"
            )
        
        # Store reference features
        if reference_features is not None:
            self.reference_features = [
                keras.ops.convert_to_tensor(f, dtype=dtype)
                for f in reference_features
            ]
            
            # Prepare (normalize and remove outliers)
            self.reference_features = [
                prepare_features(f, exact=True)
                for f in self.reference_features
            ]
            
            if self.normalize:
                self.reference_features = [
                    normalize_features(f)
                    for f in self.reference_features
                ]
        else:
            self.reference_features = None
    
    def score(
        self,
        features: Union[tf.Tensor, List[tf.Tensor], np.ndarray],
        return_layer_indices: bool = True,
        **kwargs
    ) -> Union[float, Tuple[float, Tuple[int, int]]]:
        """
        Score features against reference features.
        
        Args:
            features: Features to score (single tensor or list of layers)
            return_layer_indices: Whether to return best layer indices
            **kwargs: Additional arguments for metric
            
        Returns:
            If return_layer_indices=True: (score, (ref_layer_idx, feat_layer_idx))
            If return_layer_indices=False: score
            
        Raises:
            ValueError: If no reference features are set
        """
        if self.reference_features is None:
            raise ValueError(
                "No reference features set. Initialize with reference_features "
                "or use compute_pairwise_alignment() instead."
            )
        
        # Convert to list of tensors
        if not isinstance(features, list):
            if len(features.shape) == 3:
                # (N, L, D) -> list of (N, D)
                features = [features[:, i, :] for i in range(features.shape[1])]
            else:
                features = [features]
        
        # Prepare features
        features = [
            prepare_features(keras.ops.convert_to_tensor(f, dtype=self.dtype), exact=True)
            for f in features
        ]
        
        if self.normalize:
            features = [normalize_features(f) for f in features]
        
        # Find best alignment across all reference features
        best_score = 0.0
        best_indices = (0, 0)
        
        for ref_idx, ref_feats in enumerate(self.reference_features):
            # Convert to list if needed
            if not isinstance(ref_feats, list):
                if len(ref_feats.shape) == 3:
                    ref_layers = [ref_feats[:, i, :] for i in range(ref_feats.shape[1])]
                else:
                    ref_layers = [ref_feats]
            else:
                ref_layers = ref_feats
            
            # Score against each layer
            for feat_idx, feat_layer in enumerate(features):
                for ref_layer_idx, ref_layer in enumerate(ref_layers):
                    # Prepare metric kwargs
                    metric_kwargs = kwargs.copy()
                    if 'knn' in self.metric:
                        metric_kwargs['topk'] = self.topk
                    
                    # Compute score
                    score = AlignmentMetrics.measure(
                        self.metric,
                        ref_layer,
                        feat_layer,
                        **metric_kwargs
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_indices = (ref_layer_idx, feat_idx)
        
        if return_layer_indices:
            return best_score, best_indices
        else:
            return best_score
    
    def compute_pairwise_alignment(
        self,
        features_a: Union[tf.Tensor, List[tf.Tensor], np.ndarray],
        features_b: Union[tf.Tensor, List[tf.Tensor], np.ndarray],
        return_layer_indices: bool = True,
        **kwargs
    ) -> Union[float, Tuple[float, Tuple[int, int]]]:
        """
        Compute alignment between two feature sets.
        
        Args:
            features_a: Features from model A
            features_b: Features from model B
            return_layer_indices: Whether to return best layer indices
            **kwargs: Additional arguments for metric
            
        Returns:
            If return_layer_indices=True: (score, (layer_a_idx, layer_b_idx))
            If return_layer_indices=False: score
        """
        # Prepare features
        features_a = prepare_features(
            keras.ops.convert_to_tensor(features_a, dtype=self.dtype),
            exact=True
        )
        features_b = prepare_features(
            keras.ops.convert_to_tensor(features_b, dtype=self.dtype),
            exact=True
        )
        
        # Compute score
        metric_kwargs = kwargs.copy()
        score, indices = compute_score(
            features_a,
            features_b,
            metric=self.metric,
            topk=self.topk,
            normalize=self.normalize,
            **metric_kwargs
        )
        
        if return_layer_indices:
            return score, indices
        else:
            return score
    
    def compute_alignment_matrix(
        self,
        features_list_a: List[Union[tf.Tensor, np.ndarray]],
        features_list_b: Optional[List[Union[tf.Tensor, np.ndarray]]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute alignment matrix for multiple models.
        
        Args:
            features_list_a: List of feature tensors for models A
            features_list_b: Optional list for models B (default: same as A)
            **kwargs: Additional arguments for metric
            
        Returns:
            Dictionary with keys:
                - 'scores': Alignment scores, shape (len(A), len(B))
                - 'indices': Best layer indices, shape (len(A), len(B), 2)
        """
        if features_list_b is None:
            features_list_b = features_list_a
            symmetric = True
        else:
            symmetric = False
        
        n_a = len(features_list_a)
        n_b = len(features_list_b)
        
        scores = np.zeros((n_a, n_b), dtype=np.float32)
        indices = np.zeros((n_a, n_b, 2), dtype=np.int32)
        
        for i, feats_a in enumerate(features_list_a):
            for j, feats_b in enumerate(features_list_b):
                # Skip if symmetric and already computed
                if symmetric and i > j:
                    scores[i, j] = scores[j, i]
                    indices[i, j] = indices[j, i][::-1]
                    continue
                
                # Compute alignment
                score, layer_indices = self.compute_pairwise_alignment(
                    feats_a,
                    feats_b,
                    return_layer_indices=True,
                    **kwargs
                )
                
                scores[i, j] = score
                indices[i, j] = layer_indices
                
                # Mirror if symmetric
                if symmetric and i != j:
                    scores[j, i] = score
                    indices[j, i] = layer_indices[::-1]
        
        return {
            'scores': scores,
            'indices': indices
        }
    
    def set_reference_features(
        self,
        reference_features: List[Union[tf.Tensor, np.ndarray]]
    ) -> None:
        """
        Set or update reference features.
        
        Args:
            reference_features: List of reference feature tensors
        """
        self.reference_features = [
            keras.ops.convert_to_tensor(f, dtype=self.dtype)
            for f in reference_features
        ]
        
        # Prepare features
        self.reference_features = [
            prepare_features(f, exact=True)
            for f in self.reference_features
        ]
        
        if self.normalize:
            self.reference_features = [
                normalize_features(f)
                for f in self.reference_features
            ]
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported metrics.
        
        Returns:
            List of metric names
        """
        return AlignmentMetrics.SUPPORTED_METRICS.copy()
    
    @staticmethod
    def from_models(
        reference_models: List[keras.Model],
        data: Union[tf.Tensor, np.ndarray],
        layer_names: Optional[List[List[str]]] = None,
        metric: str = "mutual_knn",
        topk: int = 10,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> "Alignment":
        """
        Create alignment scorer from Keras models.
        
        Args:
            reference_models: List of Keras models
            data: Input data for feature extraction
            layer_names: Optional list of layer names for each model
            metric: Alignment metric to use
            topk: Number of neighbors for k-NN metrics
            normalize: Whether to normalize features
            batch_size: Batch size for feature extraction
            **kwargs: Additional arguments for Alignment.__init__
            
        Returns:
            Alignment instance with pre-computed reference features
        """
        from .utils import extract_layer_features
        
        reference_features = []
        
        for i, model in enumerate(reference_models):
            layers = layer_names[i] if layer_names else None
            feats = extract_layer_features(
                model,
                data,
                layer_names=layers,
                batch_size=batch_size
            )
            reference_features.append(feats)
        
        return Alignment(
            reference_features=reference_features,
            metric=metric,
            topk=topk,
            normalize=normalize,
            **kwargs
        )


class AlignmentLogger:
    """
    Logger for tracking alignment scores during training.
    
    Can be used as a Keras callback to monitor representation alignment.
    """
    
    def __init__(
        self,
        alignment_scorer: Alignment,
        validation_data: Union[tf.Tensor, np.ndarray],
        log_freq: int = 1,
        log_dir: Optional[str] = None
    ):
        """
        Initialize alignment logger.
        
        Args:
            alignment_scorer: Alignment instance for scoring
            validation_data: Data for computing features
            log_freq: Frequency of logging (in epochs)
            log_dir: Optional directory for saving logs
        """
        self.alignment_scorer = alignment_scorer
        self.validation_data = validation_data
        self.log_freq = log_freq
        self.log_dir = log_dir
        self.scores = []
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, model: keras.Model, logs: Optional[dict] = None):
        """
        Compute and log alignment score.
        
        Args:
            epoch: Current epoch number
            model: Keras model being trained
            logs: Optional training logs dictionary
        """
        if epoch % self.log_freq != 0:
            return
        
        # Extract features
        from .utils import extract_layer_features
        features = extract_layer_features(model, self.validation_data)
        
        # Compute alignment
        score, indices = self.alignment_scorer.score(
            features,
            return_layer_indices=True
        )
        
        # Log
        self.scores.append({
            'epoch': epoch,
            'score': float(score),
            'layer_indices': indices
        })
        
        print(f"\nEpoch {epoch} - Alignment: {score:.4f} (layers: {indices})")
        
        # Save if log_dir provided
        if self.log_dir:
            save_path = os.path.join(
                self.log_dir,
                f"alignment_epoch_{epoch}.npz"
            )
            np.savez(
                save_path,
                score=score,
                indices=indices,
                epoch=epoch
            )
    
    def get_scores(self) -> List[Dict]:
        """
        Get all logged scores.
        
        Returns:
            List of score dictionaries
        """
        return self.scores.copy()
    
    def plot_scores(self, save_path: Optional[str] = None):
        """
        Plot alignment scores over training.
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if not self.scores:
            print("No scores to plot")
            return
        
        epochs = [s['epoch'] for s in self.scores]
        scores = [s['score'] for s in self.scores]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, scores, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Alignment Score')
        plt.title('Representation Alignment During Training')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
