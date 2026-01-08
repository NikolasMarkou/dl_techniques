"""
Utility functions for alignment computation and feature processing.
"""

import os
import keras
import tensorflow as tf
import numpy as np
from typing import Union, Optional, List, Tuple


def prepare_features(
    feats: Union[tf.Tensor, keras.KerasTensor, List],
    q: float = 0.95,
    exact: bool = False
) -> Union[tf.Tensor, List[tf.Tensor]]:
    """
    Prepare features by removing outliers and converting to tensor.
    
    Args:
        feats: Features as tensor or list of tensors
        q: Quantile for outlier removal (default: 0.95)
        exact: Whether to use exact quantile computation
        
    Returns:
        Processed features
    """
    from .metrics import remove_outliers
    
    if isinstance(feats, list):
        return [
            remove_outliers(
                keras.ops.convert_to_tensor(f, dtype='float32'),
                q=q,
                exact=exact
            )
            for f in feats
        ]
    else:
        feats_tensor = keras.ops.convert_to_tensor(feats, dtype='float32')
        return remove_outliers(feats_tensor, q=q, exact=exact)


def compute_score(
    x_feats: Union[tf.Tensor, List[tf.Tensor]],
    y_feats: Union[tf.Tensor, List[tf.Tensor]],
    metric: str = "mutual_knn",
    topk: int = 10,
    normalize: bool = True,
    **kwargs
) -> Tuple[float, Tuple[int, int]]:
    """
    Find best alignment score across layer combinations.
    
    For multi-layer features, tests all pairwise combinations and returns
    the best alignment score and corresponding layer indices.
    
    Args:
        x_feats: Features from model X, shape (N, L, D) or list of (N, D)
        y_feats: Features from model Y, shape (N, L, D) or list of (N, D)
        metric: Alignment metric to use
        topk: Number of neighbors for k-NN metrics
        normalize: Whether to L2-normalize features
        **kwargs: Additional arguments for the metric
        
    Returns:
        Tuple of (best_score, (x_layer_idx, y_layer_idx))
    """
    from .metrics import AlignmentMetrics
    
    # Convert to list of layers
    if isinstance(x_feats, (tf.Tensor, keras.KerasTensor)):
        if len(x_feats.shape) == 3:
            # Shape (N, L, D) -> list of (N, D)
            x_layers = [x_feats[:, i, :] for i in range(x_feats.shape[1])]
        else:
            x_layers = [x_feats]
    else:
        x_layers = x_feats
    
    if isinstance(y_feats, (tf.Tensor, keras.ops.Tensor)):
        if len(y_feats.shape) == 3:
            y_layers = [y_feats[:, i, :] for i in range(y_feats.shape[1])]
        else:
            y_layers = [y_feats]
    else:
        y_layers = y_feats
    
    # Find best alignment across layers
    best_score = 0.0
    best_indices = (0, 0)
    
    for i, x_layer in enumerate(x_layers):
        for j, y_layer in enumerate(y_layers):
            # Normalize if requested
            if normalize:
                x_aligned = keras.ops.normalize(x_layer, axis=-1)
                y_aligned = keras.ops.normalize(y_layer, axis=-1)
            else:
                x_aligned = x_layer
                y_aligned = y_layer
            
            # Prepare metric kwargs
            metric_kwargs = kwargs.copy()
            if 'knn' in metric:
                metric_kwargs['topk'] = topk
            
            # Compute score
            score = AlignmentMetrics.measure(
                metric,
                x_aligned,
                y_aligned,
                **metric_kwargs
            )
            
            # Update best
            if score > best_score:
                best_score = score
                best_indices = (i, j)
    
    return best_score, best_indices


def compute_alignment_matrix(
    x_feat_list: List[tf.Tensor],
    y_feat_list: List[tf.Tensor],
    metric: str = "mutual_knn",
    topk: int = 10,
    normalize: bool = True,
    precise: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute alignment matrix between all pairs of feature sets.
    
    Args:
        x_feat_list: List of feature tensors for models X
        y_feat_list: List of feature tensors for models Y
        metric: Alignment metric to use
        topk: Number of neighbors for k-NN metrics
        normalize: Whether to L2-normalize features
        precise: Whether to use float32 precision
        **kwargs: Additional metric arguments
        
    Returns:
        Tuple of (scores_matrix, indices_matrix) where:
            - scores_matrix: shape (len(x_feat_list), len(y_feat_list))
            - indices_matrix: shape (len(x_feat_list), len(y_feat_list), 2)
    """
    n_x = len(x_feat_list)
    n_y = len(y_feat_list)
    
    scores = np.zeros((n_x, n_y), dtype=np.float32)
    indices = np.zeros((n_x, n_y, 2), dtype=np.int32)
    
    # Check if symmetric (same model sets)
    symmetric = (x_feat_list is y_feat_list) and (metric != "cycle_knn")
    
    for i, x_feats in enumerate(x_feat_list):
        # Prepare features
        x_prep = prepare_features(x_feats, exact=precise)
        
        for j, y_feats in enumerate(y_feat_list):
            # Skip if already computed (symmetric case)
            if symmetric and i > j:
                scores[i, j] = scores[j, i]
                indices[i, j] = indices[j, i][::-1]
                continue
            
            # Prepare features
            y_prep = prepare_features(y_feats, exact=precise)
            
            # Compute alignment
            score, layer_indices = compute_score(
                x_prep,
                y_prep,
                metric=metric,
                topk=topk,
                normalize=normalize,
                **kwargs
            )
            
            scores[i, j] = score
            indices[i, j] = layer_indices
            
            # Mirror for symmetric case
            if symmetric:
                scores[j, i] = score
                indices[j, i] = layer_indices[::-1]
    
    return scores, indices


def normalize_features(
    feats: Union[tf.Tensor, keras.ops.Tensor, List],
    axis: int = -1
) -> Union[tf.Tensor, List[tf.Tensor]]:
    """
    L2-normalize features.
    
    Args:
        feats: Features as tensor or list of tensors
        axis: Axis along which to normalize
        
    Returns:
        Normalized features
    """
    if isinstance(feats, list):
        return [keras.ops.normalize(f, axis=axis) for f in feats]
    else:
        return keras.ops.normalize(feats, axis=axis)


def extract_layer_features(
    model: keras.Model,
    inputs: Union[tf.Tensor, keras.ops.Tensor],
    layer_names: Optional[List[str]] = None,
    batch_size: int = 32
) -> List[tf.Tensor]:
    """
    Extract features from multiple layers of a Keras model.
    
    Args:
        model: Keras model
        inputs: Input data, shape (N, ...)
        layer_names: List of layer names to extract from (default: all layers)
        batch_size: Batch size for processing
        
    Returns:
        List of feature tensors, one per layer
    """
    if layer_names is None:
        # Extract from all layers
        layer_names = [layer.name for layer in model.layers]
    
    # Create feature extractor model
    outputs = [model.get_layer(name).output for name in layer_names]
    extractor = keras.Model(inputs=model.input, outputs=outputs)
    
    # Extract features in batches
    all_features = [[] for _ in layer_names]
    
    n_samples = inputs.shape[0]
    for i in range(0, n_samples, batch_size):
        batch = inputs[i:i+batch_size]
        batch_features = extractor(batch, training=False)
        
        if not isinstance(batch_features, list):
            batch_features = [batch_features]
        
        for j, feats in enumerate(batch_features):
            # Global average pooling if spatial dimensions exist
            if len(feats.shape) > 2:
                feats = keras.ops.mean(feats, axis=tuple(range(1, len(feats.shape) - 1)))
            all_features[j].append(feats)
    
    # Concatenate batches
    layer_features = [
        keras.ops.concatenate(feats_list, axis=0)
        for feats_list in all_features
    ]
    
    return layer_features


def save_features(
    features: Union[tf.Tensor, List[tf.Tensor]],
    save_path: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save features to disk.
    
    Args:
        features: Features as tensor or list of tensors
        save_path: Path to save file (.npz format)
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy
    if isinstance(features, list):
        features_np = [keras.ops.convert_to_numpy(f) for f in features]
        save_dict = {f"layer_{i}": f for i, f in enumerate(features_np)}
        save_dict["num_layers"] = len(features_np)
    else:
        features_np = keras.ops.convert_to_numpy(features)
        save_dict = {"features": features_np}
    
    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                save_dict[key] = value
    
    # Save
    np.savez_compressed(save_path, **save_dict)


def load_features(
    load_path: str
) -> Union[tf.Tensor, List[tf.Tensor]]:
    """
    Load features from disk.
    
    Args:
        load_path: Path to saved features (.npz format)
        
    Returns:
        Features as tensor or list of tensors
    """
    data = np.load(load_path, allow_pickle=True)
    
    if "features" in data:
        # Single tensor
        return keras.ops.convert_to_tensor(data["features"])
    elif "num_layers" in data:
        # Multiple layers
        num_layers = int(data["num_layers"])
        return [
            keras.ops.convert_to_tensor(data[f"layer_{i}"])
            for i in range(num_layers)
        ]
    else:
        # Legacy format: assume all arrays are features
        return [keras.ops.convert_to_tensor(data[key]) for key in sorted(data.keys())]


def pool_features(
    features: Union[tf.Tensor, keras.ops.Tensor],
    pool_type: str = "mean",
    axis: Union[int, Tuple[int, ...]] = 1
) -> tf.Tensor:
    """
    Pool features along specified axis.
    
    Args:
        features: Features tensor
        pool_type: Type of pooling ('mean', 'max', 'sum')
        axis: Axis or axes to pool over
        
    Returns:
        Pooled features
    """
    features = keras.ops.convert_to_tensor(features)
    
    if pool_type == "mean":
        return keras.ops.mean(features, axis=axis)
    elif pool_type == "max":
        return keras.ops.max(features, axis=axis)
    elif pool_type == "sum":
        return keras.ops.sum(features, axis=axis)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


def create_feature_filename(
    output_dir: str,
    dataset: str,
    subset: str,
    model_name: str,
    pool: Optional[str] = None,
    prompt: Optional[bool] = None,
    caption_idx: Optional[int] = None
) -> str:
    """
    Create standardized filename for features.
    
    Args:
        output_dir: Base output directory
        dataset: Dataset name
        subset: Dataset subset
        model_name: Model identifier
        pool: Pooling method used
        prompt: Whether prompting was used
        caption_idx: Caption index if multiple captions
        
    Returns:
        Full path to feature file
    """
    save_name = model_name.replace('/', '_')
    
    if pool:
        save_name += f"_pool-{pool}"
    if prompt:
        save_name += f"_prompt-{prompt}"
    if caption_idx is not None:
        save_name += f"_cid-{caption_idx}"
    
    save_path = os.path.join(
        output_dir,
        dataset,
        subset,
        f"{save_name}.npz"
    )
    
    return save_path


def create_alignment_filename(
    output_dir: str,
    dataset: str,
    modelset: str,
    modality_x: str,
    pool_x: Optional[str],
    prompt_x: Optional[bool],
    modality_y: str,
    pool_y: Optional[str],
    prompt_y: Optional[bool],
    metric: str,
    topk: Optional[int] = None
) -> str:
    """
    Create standardized filename for alignment results.
    
    Args:
        output_dir: Base output directory
        dataset: Dataset name
        modelset: Model set identifier
        modality_x: Modality of X (e.g., 'vision', 'language')
        pool_x: Pooling for X
        prompt_x: Whether X used prompting
        modality_y: Modality of Y
        pool_y: Pooling for Y
        prompt_y: Whether Y used prompting
        metric: Alignment metric used
        topk: Number of neighbors (for k-NN metrics)
        
    Returns:
        Full path to alignment results file
    """
    subdir = (
        f"{modality_x}_pool-{pool_x}_prompt-{prompt_x}_"
        f"{modality_y}_pool-{pool_y}_prompt-{prompt_y}"
    )
    
    if 'knn' in metric and topk is not None:
        filename = f"{metric}_k{topk}.npz"
    else:
        filename = f"{metric}.npz"
    
    save_path = os.path.join(
        output_dir,
        dataset,
        modelset,
        subdir,
        filename
    )
    
    return save_path


def compute_statistics(
    features: Union[tf.Tensor, keras.ops.Tensor]
) -> dict:
    """
    Compute statistics of features.
    
    Args:
        features: Feature tensor
        
    Returns:
        Dictionary of statistics
    """
    features = keras.ops.convert_to_tensor(features)
    
    stats = {
        "mean": float(keras.ops.mean(features)),
        "std": float(keras.ops.std(features)),
        "min": float(keras.ops.min(features)),
        "max": float(keras.ops.max(features)),
        "shape": tuple(features.shape),
        "norm": float(keras.ops.sqrt(keras.ops.sum(features ** 2)))
    }
    
    return stats


def batch_generator(
    data: Union[tf.Tensor, keras.ops.Tensor, np.ndarray],
    batch_size: int
):
    """
    Generate batches from data.
    
    Args:
        data: Input data
        batch_size: Batch size
        
    Yields:
        Batches of data
    """
    n_samples = data.shape[0]
    
    for i in range(0, n_samples, batch_size):
        yield data[i:i+batch_size]
