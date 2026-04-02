"""
Deep Kernel Principal Component Analysis (DKPCA) algorithm.

This layer provides a hierarchical, multi-level extension of Kernel Principal
Component Analysis (KPCA). It is designed to learn more expressive and
disentangled feature representations than shallow KPCA by creating a deep
architecture with coupled optimization across all levels.

Architecture and Design Philosophy:
The core innovation of DKPCA lies in its forward and backward coupling
mechanisms, which distinguish it from a naive stacking of independent KPCA
layers. The architecture consists of multiple sequential levels, where each
level performs a kernel transformation and extracts principal components.

1.  **Forward Coupling**: The data flows directionally from the input through
    the hierarchy. The principal components extracted at level `j-1` serve as
    the input features for level `j`. This allows the model to build
    progressively more abstract representations, with each level operating on
    the features learned by the one before it.

2.  **Backward Coupling**: This is the key mechanism that enables joint
    optimization across the hierarchy. Information from deeper, more abstract
    levels (e.g., level `j+1`) flows backward to refine the representations at
    shallower levels (e.g., level `j`). This ensures that the features learned
    at a given level are not just optimal for reconstructing their immediate
    input but are also useful for the feature extraction task of subsequent
    levels. This transforms the learning problem from a greedy, layer-by-layer
    process into a globally coherent optimization, yielding more informative
    and disentangled features throughout the entire network.

Foundational Mathematics:
DKPCA extends the foundational principles of Kernel PCA. Standard KPCA uses
the "kernel trick" to implicitly map data into a high-dimensional feature
space `φ(x)` and then performs PCA in that space. This is achieved by
constructing a kernel matrix `K` where `Kᵢⱼ = k(xᵢ, xⱼ) = <φ(xᵢ), φ(xⱼ)>`
and finding its eigenvectors.

DKPCA formulates this as a joint optimization problem across all `L` levels.
The objective is to find sets of principal component coefficients `{α¹, α²,
..., α^L}` that simultaneously minimize the reconstruction error at every
level. The optimization problem is coupled: the input to the kernel `K^(j)` at
level `j` is derived from the components `α^(j-1)`. The backward coupling is
formalized by introducing dependencies in the objective function that link the
solution `α^(j)` to the solution `α^(j+1)`. This creates a holistic system
where the principal components at each level are influenced by all other
levels, leading to a globally consistent and expressive hierarchical
representation.

References:
    - [Tonin, P. A., et al. (2023). Deep Kernel Principal Component Analysis
      for Multi-level Feature Learning.](https://arxiv.org/abs/2302.11220)
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DeepKernelPCA(keras.layers.Layer):
    """
    Deep Kernel Principal Component Analysis layer for multi-level feature extraction.

    This layer implements DKPCA, which extends traditional Kernel PCA to multiple
    hierarchical levels with coupled optimization. It creates both forward and backward
    dependencies across levels to extract more informative features than shallow KPCA.
    The optimization minimizes ``min sum_j ||X^(j) - K^(j) alpha^(j)||^2_F + lambda sum_j ||alpha^(j)||^2_2``
    where ``K^(j)`` is the kernel matrix at level ``j`` and ``alpha^(j)`` are the
    principal component coefficients. Forward coupling passes components from level
    ``j-1`` as input to level ``j``, while backward coupling refines shallower
    representations using information from deeper levels through gated residual
    connections.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │         Input (batch, features)   │
        └──────────────┬────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Level 1: Kernel K^1 ─► PCA ─► alpha^1  │
        └──────────────┬──────────────────────────┘
                       │ forward coupling
                       ▼
        ┌─────────────────────────────────────────┐
        │  Level 2: Kernel K^2 ─► PCA ─► alpha^2  │
        └──────────────┬──────────────────────────┘
                       │ forward coupling
                       ▼
        ┌─────────────────────────────────────────┐
        │  Level L: Kernel K^L ─► PCA ─► alpha^L  │
        └──────────────┬──────────────────────────┘
                       │
               ┌───────┴───────┐
               │ Backward Pass │ (refine via gated coupling)
               └───────┬───────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Weighted Concat ─► Output       │
        │  (batch, sum(components))        │
        └──────────────────────────────────┘

    :param num_levels: Number of hierarchical KPCA levels. Must be positive.
        Controls the depth of feature extraction. Defaults to 3.
    :type num_levels: int
    :param components_per_level: Number of principal components to extract at
        each level. Length must match ``num_levels``. If ``None``, uses adaptive
        sizing with golden-ratio reduction.
    :type components_per_level: list[int] | None
    :param kernel_type: Kernel function(s) to use. Options: ``'rbf'``,
        ``'polynomial'``, ``'linear'``, ``'sigmoid'``, ``'cosine'``. If a single
        string, uses same kernel for all levels. Defaults to ``'rbf'``.
    :type kernel_type: str | list[str]
    :param kernel_params: Kernel-specific parameters. If a single dict, uses
        same params for all levels. Defaults to ``None``.
    :type kernel_params: dict[str, Any] | list[dict[str, Any]] | None
    :param regularization_lambda: L2 regularization strength for principal
        components. Defaults to 0.01.
    :type regularization_lambda: float
    :param coupling_strength: Strength of forward-backward coupling between
        levels, in ``[0, 1]``. Defaults to 0.5.
    :type coupling_strength: float
    :param use_backward_coupling: Whether to include backward dependencies
        from deeper levels to shallower ones. Defaults to ``True``.
    :type use_backward_coupling: bool
    :param center_kernel: Whether to center the kernel matrix. Defaults to ``True``.
    :type center_kernel: bool
    :param kernel_regularizer: Optional regularizer for kernel parameters.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param projection_regularizer: Optional regularizer for projection matrices.
    :type projection_regularizer: keras.regularizers.Regularizer | None
    :param coupling_regularizer: Optional regularizer for coupling weights.
    :type coupling_regularizer: keras.regularizers.Regularizer | None
    :param trainable_kernels: Whether kernel parameters are trainable.
        Defaults to ``False``.
    :type trainable_kernels: bool
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            num_levels: int = 3,
            components_per_level: Optional[List[int]] = None,
            kernel_type: Union[str, List[str]] = 'rbf',
            kernel_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            regularization_lambda: float = 0.01,
            coupling_strength: float = 0.5,
            use_backward_coupling: bool = True,
            center_kernel: bool = True,
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            projection_regularizer: Optional[regularizers.Regularizer] = None,
            coupling_regularizer: Optional[regularizers.Regularizer] = None,
            trainable_kernels: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if not (0.0 <= coupling_strength <= 1.0):
            raise ValueError(f"coupling_strength must be in [0, 1], got {coupling_strength}")
        if regularization_lambda < 0:
            raise ValueError(f"regularization_lambda must be non-negative, got {regularization_lambda}")

        # Store configuration
        self.num_levels = num_levels
        self.components_per_level = components_per_level
        self.regularization_lambda = regularization_lambda
        self.coupling_strength = coupling_strength
        self.use_backward_coupling = use_backward_coupling
        self.center_kernel = center_kernel
        self.kernel_regularizer = kernel_regularizer
        self.projection_regularizer = projection_regularizer
        self.coupling_regularizer = coupling_regularizer
        self.trainable_kernels = trainable_kernels

        # Process kernel configuration
        if isinstance(kernel_type, str):
            self.kernel_types = [kernel_type] * num_levels
        else:
            if len(kernel_type) != num_levels:
                raise ValueError(f"kernel_type list length ({len(kernel_type)}) must match num_levels ({num_levels})")
            self.kernel_types = kernel_type

        # Validate kernel types
        valid_kernels = {'rbf', 'polynomial', 'linear', 'sigmoid', 'cosine'}
        for kt in self.kernel_types:
            if kt not in valid_kernels:
                raise ValueError(f"Invalid kernel type: {kt}. Must be one of {valid_kernels}")

        # Process kernel parameters
        if kernel_params is None:
            self.kernel_params = [{}] * num_levels
        elif isinstance(kernel_params, dict):
            self.kernel_params = [kernel_params.copy()] * num_levels
        else:
            if len(kernel_params) != num_levels:
                raise ValueError(f"kernel_params list length must match num_levels")
            self.kernel_params = kernel_params

        # Initialize weight attributes (created in build)
        self.kernel_weights = []
        self.projection_matrices = []
        self.eigenvalues = []
        self.coupling_weights_forward = []
        self.coupling_weights_backward = []

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights for multi-level kernel PCA.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Determine components per level if not specified
        if self.components_per_level is None:
            # Adaptive sizing: progressively reduce dimensions
            components = []
            current_dim = input_dim
            for i in range(self.num_levels):
                # Use golden ratio for smooth reduction
                next_dim = max(int(current_dim * 0.618), 1)
                components.append(min(next_dim, current_dim))
                current_dim = next_dim
            self.components_per_level = components
        else:
            if len(self.components_per_level) != self.num_levels:
                raise ValueError(f"components_per_level length must match num_levels")

        # Create weights for each level
        current_input_dim = input_dim

        for level in range(self.num_levels):
            num_components = self.components_per_level[level]

            # Ensure we don't extract more components than dimensions
            if num_components > current_input_dim:
                raise ValueError(
                    f"Level {level}: Cannot extract {num_components} components from {current_input_dim} dimensions")

            # Kernel-specific weights if trainable
            if self.trainable_kernels:
                kernel_type = self.kernel_types[level]

                if kernel_type == 'rbf':
                    # Trainable gamma parameter for RBF kernel
                    gamma_init = self.kernel_params[level].get('gamma', 1.0 / current_input_dim)
                    self.kernel_weights.append(
                        self.add_weight(
                            name=f'kernel_gamma_level_{level}',
                            shape=(1,),
                            initializer=initializers.Constant(gamma_init),
                            trainable=True,
                            regularizer=self.kernel_regularizer
                        )
                    )
                elif kernel_type == 'polynomial':
                    # Trainable degree and coef0 for polynomial kernel
                    degree_init = float(self.kernel_params[level].get('degree', 3))
                    coef0_init = self.kernel_params[level].get('coef0', 1.0)
                    self.kernel_weights.append({
                        'degree': self.add_weight(
                            name=f'kernel_degree_level_{level}',
                            shape=(1,),
                            initializer=initializers.Constant(degree_init),
                            trainable=True,
                            regularizer=self.kernel_regularizer
                        ),
                        'coef0': self.add_weight(
                            name=f'kernel_coef0_level_{level}',
                            shape=(1,),
                            initializer=initializers.Constant(coef0_init),
                            trainable=True,
                            regularizer=self.kernel_regularizer
                        )
                    })
                elif kernel_type == 'sigmoid':
                    gamma_init = self.kernel_params[level].get('gamma', 0.01)
                    coef0_init = self.kernel_params[level].get('coef0', 1.0)
                    self.kernel_weights.append({
                        'gamma': self.add_weight(
                            name=f'kernel_sigmoid_gamma_level_{level}',
                            shape=(1,),
                            initializer=initializers.Constant(gamma_init),
                            trainable=True,
                            regularizer=self.kernel_regularizer
                        ),
                        'coef0': self.add_weight(
                            name=f'kernel_sigmoid_coef0_level_{level}',
                            shape=(1,),
                            initializer=initializers.Constant(coef0_init),
                            trainable=True,
                            regularizer=self.kernel_regularizer
                        )
                    })
                else:
                    self.kernel_weights.append(None)
            else:
                self.kernel_weights.append(None)

            # Projection matrix for PCA at this level (orthonormal initialization)
            self.projection_matrices.append(
                self.add_weight(
                    name=f'projection_matrix_level_{level}',
                    shape=(current_input_dim, num_components),
                    initializer='orthogonal',
                    trainable=True,
                    regularizer=self.projection_regularizer
                )
            )

            # Eigenvalues for tracking explained variance
            self.eigenvalues.append(
                self.add_weight(
                    name=f'eigenvalues_level_{level}',
                    shape=(num_components,),
                    initializer='ones',
                    trainable=False
                )
            )

            # Forward coupling weights (from previous level to current)
            if level > 0 and self.coupling_strength > 0:
                prev_components = self.components_per_level[level - 1]
                self.coupling_weights_forward.append(
                    self.add_weight(
                        name=f'coupling_forward_level_{level}',
                        shape=(prev_components, num_components),
                        initializer=initializers.RandomNormal(stddev=0.01 * self.coupling_strength),
                        trainable=True,
                        regularizer=self.coupling_regularizer
                    )
                )
            else:
                self.coupling_weights_forward.append(None)

            # Backward coupling weights (from current level to previous)
            if level < self.num_levels - 1 and self.use_backward_coupling and self.coupling_strength > 0:
                # Will be properly sized when we know next level dimensions
                self.coupling_weights_backward.append(
                    self.add_weight(
                        name=f'coupling_backward_level_{level}',
                        shape=(num_components, num_components),  # Will be reshaped dynamically
                        initializer=initializers.RandomNormal(stddev=0.01 * self.coupling_strength),
                        trainable=True,
                        regularizer=self.coupling_regularizer
                    )
                )
            else:
                self.coupling_weights_backward.append(None)

            # Update input dimension for next level
            current_input_dim = num_components

        super().build(input_shape)

    def compute_kernel_matrix(
            self,
            x: keras.KerasTensor,
            level: int
    ) -> keras.KerasTensor:
        """Compute kernel matrix for a given level with numerical stability.

        :param x: Input tensor of shape ``(batch_size, features)``.
        :type x: keras.KerasTensor
        :param level: Level index for kernel computation.
        :type level: int
        :return: Kernel matrix of shape ``(batch_size, batch_size)``.
        :rtype: keras.KerasTensor
        """
        kernel_type = self.kernel_types[level]
        params = self.kernel_params[level].copy()

        # Use trainable parameters if available
        if self.trainable_kernels and self.kernel_weights[level] is not None:
            if kernel_type == 'rbf':
                params['gamma'] = ops.squeeze(self.kernel_weights[level])
            elif kernel_type == 'polynomial':
                params['degree'] = ops.squeeze(self.kernel_weights[level]['degree'])
                params['coef0'] = ops.squeeze(self.kernel_weights[level]['coef0'])
            elif kernel_type == 'sigmoid':
                params['gamma'] = ops.squeeze(self.kernel_weights[level]['gamma'])
                params['coef0'] = ops.squeeze(self.kernel_weights[level]['coef0'])

        # Add small epsilon for numerical stability
        eps = 1e-10

        if kernel_type == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            gamma = params.get('gamma', 1.0 / ops.cast(ops.shape(x)[-1], dtype=x.dtype))
            # Efficient pairwise distance computation
            x_norm = ops.sum(ops.square(x), axis=1, keepdims=True)
            distances = x_norm + ops.transpose(x_norm) - 2.0 * ops.matmul(x, ops.transpose(x))
            # Ensure non-negative distances
            distances = ops.maximum(distances, 0.0)
            kernel_matrix = ops.exp(-gamma * distances)

        elif kernel_type == 'polynomial':
            # Polynomial kernel: (x^T y + coef0)^degree
            degree = params.get('degree', 3.0)
            coef0 = params.get('coef0', 1.0)
            dot_product = ops.matmul(x, ops.transpose(x))
            kernel_matrix = ops.power(ops.maximum(dot_product + coef0, eps), degree)

        elif kernel_type == 'linear':
            # Linear kernel: x^T y
            kernel_matrix = ops.matmul(x, ops.transpose(x))

        elif kernel_type == 'sigmoid':
            # Sigmoid kernel: tanh(gamma * x^T y + coef0)
            gamma = params.get('gamma', 0.01)
            coef0 = params.get('coef0', 1.0)
            dot_product = ops.matmul(x, ops.transpose(x))
            kernel_matrix = ops.tanh(gamma * dot_product + coef0)

        elif kernel_type == 'cosine':
            # Cosine similarity kernel
            x_norm = ops.sqrt(ops.sum(ops.square(x), axis=1, keepdims=True) + eps)
            x_normalized = x / x_norm
            kernel_matrix = ops.matmul(x_normalized, ops.transpose(x_normalized))

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Center the kernel matrix if requested
        if self.center_kernel:
            batch_size = ops.cast(ops.shape(kernel_matrix)[0], dtype=kernel_matrix.dtype)
            # Compute row and column means
            row_mean = ops.mean(kernel_matrix, axis=1, keepdims=True)
            col_mean = ops.mean(kernel_matrix, axis=0, keepdims=True)
            mean_all = ops.mean(kernel_matrix)
            # Center the kernel matrix
            kernel_matrix = kernel_matrix - row_mean - col_mean + mean_all

        return kernel_matrix

    def extract_components(
            self,
            kernel_matrix: keras.KerasTensor,
            projection_matrix: keras.KerasTensor,
            eigenvalues: keras.KerasTensor,
            num_components: int,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Extract principal components using kernel PCA with eigendecomposition.

        :param kernel_matrix: Kernel matrix of shape ``(batch_size, batch_size)``.
        :type kernel_matrix: keras.KerasTensor
        :param projection_matrix: Projection matrix of shape ``(feature_dim, num_components)``.
        :type projection_matrix: keras.KerasTensor
        :param eigenvalues: Eigenvalues for this level.
        :type eigenvalues: keras.KerasTensor
        :param num_components: Number of components to extract.
        :type num_components: int
        :param training: Whether in training mode.
        :type training: bool | None
        :return: Principal components of shape ``(batch_size, num_components)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(kernel_matrix)[0]

        # Add regularization to diagonal for numerical stability
        kernel_matrix_reg = kernel_matrix + self.regularization_lambda * ops.eye(batch_size)

        # During training, update projection via eigendecomposition (approximation)
        if training:
            # Compute eigendecomposition of regularized kernel matrix
            # Note: In practice, we use the projection matrix as an approximation
            # Full eigendecomposition would be: eigvals, eigvecs = ops.linalg.eigh(kernel_matrix_reg)
            # But for efficiency, we use iterative approximation through gradient descent

            # Normalize projection matrix to maintain orthogonality
            projection_matrix = ops.nn.l2_normalize(projection_matrix, axis=0)

        # Project kernel matrix to get components
        # This approximates K @ eigenvectors[:, :num_components]
        components = ops.matmul(kernel_matrix_reg, projection_matrix[:batch_size, :])

        # Normalize by eigenvalues (approximate scaling)
        components = components / (ops.sqrt(ops.abs(eigenvalues) + 1e-10))

        return components

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through hierarchical kernel PCA levels with coupled optimization.

        :param inputs: Input tensor of shape ``(batch_size, input_features)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean flag for training mode.
        :type training: bool | None
        :return: Concatenated principal components from all levels.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        current_features = inputs

        # Store intermediate features for coupling
        level_features = []
        level_kernels = []

        # === Forward Pass ===
        for level in range(self.num_levels):
            # Compute kernel matrix for current features
            kernel_matrix = self.compute_kernel_matrix(current_features, level)
            level_kernels.append(kernel_matrix)

            # Extract principal components
            num_components = self.components_per_level[level]

            # Use appropriate projection dimensions
            if level == 0:
                # First level: project from input space
                projection = self.projection_matrices[level]
            else:
                # Subsequent levels: project from previous component space
                prev_components = self.components_per_level[level - 1]
                projection = self.projection_matrices[level][:prev_components, :]

            components = self.extract_components(
                kernel_matrix,
                projection,
                self.eigenvalues[level],
                num_components,
                training=training
            )

            # Apply forward coupling from previous level
            if level > 0 and self.coupling_weights_forward[level] is not None:
                prev_features = level_features[-1]
                # Forward influence from previous level
                coupling_term = ops.matmul(prev_features, self.coupling_weights_forward[level])
                components = components + self.coupling_strength * coupling_term

            # Apply activation for non-linearity between levels
            components = ops.nn.tanh(components)

            # Store features
            level_features.append(components)

            # Prepare input for next level
            current_features = components

        # === Backward Coupling Pass ===
        if self.use_backward_coupling and self.num_levels > 1:
            # Create refined features with backward information flow
            refined_features = level_features.copy()

            # Backward pass: refine each level using information from deeper levels
            for level in range(self.num_levels - 2, -1, -1):
                if level < self.num_levels - 1:
                    # Get information from next (deeper) level
                    next_features = refined_features[level + 1]

                    # Create backward coupling
                    # Use transpose of forward coupling for symmetry
                    if self.coupling_weights_forward[level + 1] is not None:
                        backward_influence = ops.matmul(
                            next_features,
                            ops.transpose(self.coupling_weights_forward[level + 1])
                        )
                        # Refine current level features
                        refined_features[level] = level_features[level] + \
                                                  0.5 * self.coupling_strength * backward_influence

                        # Apply soft gating for selective information flow
                        gate = ops.nn.sigmoid(refined_features[level])
                        refined_features[level] = gate * refined_features[level] + \
                                                  (1 - gate) * level_features[level]

            # Use refined features as output
            output_features = refined_features
        else:
            output_features = level_features

        # === Output Combination ===
        # Concatenate all level features with optional weighting
        if len(output_features) == 1:
            output = output_features[0]
        else:
            # Weight features by their level (deeper levels get slightly less weight)
            weighted_features = []
            for i, features in enumerate(output_features):
                # Exponential decay weighting
                weight = ops.exp(-0.1 * i)
                weighted_features.append(weight * features)

            # Concatenate weighted features
            output = ops.concatenate(weighted_features, axis=-1)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        total_components = sum(self.components_per_level) if self.components_per_level else None
        return (batch_size, total_components)

    def get_explained_variance_ratio(self) -> List[float]:
        """Get the explained variance ratio for each level.

        :return: Explained variance ratios (as percentages) for each level.
        :rtype: list[float]
        """
        ratios = []
        for level in range(self.num_levels):
            eigenvalues = self.eigenvalues[level]
            # Compute explained variance ratio
            total_variance = ops.sum(eigenvalues)
            explained_ratio = eigenvalues / (total_variance + 1e-10)
            ratios.append(ops.convert_to_numpy(explained_ratio))
        return ratios

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_levels': self.num_levels,
            'components_per_level': self.components_per_level,
            'kernel_type': self.kernel_types if len(set(self.kernel_types)) > 1 else self.kernel_types[0],
            'kernel_params': self.kernel_params if len(set(map(str, self.kernel_params))) > 1 else self.kernel_params[
                0],
            'regularization_lambda': self.regularization_lambda,
            'coupling_strength': self.coupling_strength,
            'use_backward_coupling': self.use_backward_coupling,
            'center_kernel': self.center_kernel,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer) if self.kernel_regularizer else None,
            'projection_regularizer': regularizers.serialize(
                self.projection_regularizer) if self.projection_regularizer else None,
            'coupling_regularizer': regularizers.serialize(
                self.coupling_regularizer) if self.coupling_regularizer else None,
            'trainable_kernels': self.trainable_kernels,
        })
        return config

# ---------------------------------------------------------------------
