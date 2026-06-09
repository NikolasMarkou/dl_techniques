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
import numpy as np
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

    .. note:: **Working-bar limitations (documented approximations).** This is
        an idiosyncratic "deep" KPCA variant, NOT a textbook eigendecomposition
        kernel PCA. Specifically:

        * The per-level projection does **not** perform a true
          eigendecomposition (``eigh``) of the centered Gram matrix. During
          training it only L2-normalizes the learnable projection columns; the
          principal-component coefficients are learned by gradient descent.
        * ``extract_components`` reuses the leading ``batch_size`` rows of each
          ``(feature_dim, num_components)`` projection weight as the per-sample
          coefficients (because the sample axis is dynamic and cannot be a
          weight dimension). This **requires ``feature_dim >= batch_size`` at
          every level** (i.e. the layer's input dim and each level's component
          count must be at least the batch size); it is a learned approximation,
          not a Nystrom/analytic projection.
        * ``eigenvalues`` are tracked but not fitted from data; the
          explained-variance ratios are nominal.

        These approximations are intentional and out of scope to "fix" into
        canonical kernel PCA; the layer's contract is a non-crashing,
        gradient-trainable, round-trip-serializable hierarchical transform.
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
        # Preserve the raw constructor argument so get_config() serializes the
        # PRE-build value (None triggers the adaptive golden-ratio path in
        # build()); self.components_per_level is mutated in-place there.
        self._components_per_level_init = components_per_level
        self.regularization_lambda = regularization_lambda
        self.coupling_strength = coupling_strength
        self.use_backward_coupling = use_backward_coupling
        self.center_kernel = center_kernel
        self.kernel_regularizer = kernel_regularizer
        self.projection_regularizer = projection_regularizer
        self.coupling_regularizer = coupling_regularizer
        self.trainable_kernels = trainable_kernels

        # Preserve original constructor args for serialization (build does not
        # mutate these, but get_config reconstructs a single-vs-list form from
        # the expanded per-level attrs which can lose the original intent).
        self._kernel_type_init = kernel_type
        self._kernel_params_init = kernel_params

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
            # One independent copy per level (a shared `* num_levels` reference
            # would alias one dict across all levels).
            self.kernel_params = [kernel_params.copy() for _ in range(num_levels)]
        else:
            if len(kernel_params) != num_levels:
                raise ValueError(f"kernel_params list length must match num_levels")
            self.kernel_params = kernel_params

        # Initialize weight attributes (created in build)
        self.kernel_weights = []
        self.projection_matrices = []
        self.eigenvalues = []
        self.coupling_weights_forward = []
        # NOTE: the backward coupling pass (see call()) reuses the TRANSPOSE of
        # coupling_weights_forward[level+1]; no separate backward weights exist.

        # --- Genuine kernel-PCA fitted state (populated by adapt()) ---------
        # When adapt() has run, call() takes a GENUINE Nystrom kernel-PCA path
        # using these per-level weights instead of the un-fitted random
        # projection above. _fit_flag is a non-trainable scalar weight (created
        # in build) so the fitted-vs-unfitted decision survives serialization.
        self._fit_flag = None
        # Per-level fitted tensors (created lazily in adapt via add_weight so
        # they are tracked + serialized): landmark representations, Nystrom
        # alphas, and double-centering stats. See adapt() / D-006.
        self.landmark_reprs = []      # level-input repr of the adapt landmarks
        self.nystrom_alphas = []      # (M, k) eigvecs / sqrt(eigval)
        self.train_kernel_rowmean = []  # (M,) training-Gram row means
        self.train_kernel_allmean = []  # () training-Gram grand mean

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights for multi-level kernel PCA.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        # Functional API may pass a LIST OF SHAPES for multi-input layers; this
        # is single-input, so unwrap only a true nested list-of-shapes. A plain
        # shape serialized as a list (e.g. [None, 8]) must NOT be unwrapped —
        # its first element is an int/None, not a shape.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

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

            # (Backward coupling reuses transpose(coupling_weights_forward[level+1])
            #  in call(); no dedicated backward weights are allocated.)

            # Update input dimension for next level
            current_input_dim = num_components

        # Non-trainable flag: 0.0 = un-fitted (random projection), 1.0 = fitted
        # (genuine Nystrom kernel PCA). Created here so it serializes and the
        # fitted/un-fitted branch is reproducible after a round-trip.
        self._fit_flag = self.add_weight(
            name='fit_flag',
            shape=(),
            initializer='zeros',
            trainable=False,
        )

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
        # Dynamic (graph-safe) batch size of the kernel matrix.
        batch_size = ops.shape(kernel_matrix)[0]

        # Add regularization to diagonal for numerical stability.
        # ops.eye accepts the symbolic batch size under TF graph mode (verified).
        kernel_matrix_reg = kernel_matrix + self.regularization_lambda * ops.eye(batch_size)

        # During training, normalize the projection columns to keep them on the
        # unit sphere (an orthogonality-maintaining approximation of the true
        # eigendecomposition update). NOTE (working-bar limitation): this is the
        # idiosyncratic "deep" KPCA approximation, NOT a textbook eigh-based
        # KPCA update. Use `training is True` so a symbolic/None flag under
        # graph tracing does not silently take the training branch.
        if training is True:
            # ops.nn.l2_normalize does not exist in this Keras build; the
            # canonical column-wise L2 normalization is ops.normalize(x, axis=0).
            projection_matrix = ops.normalize(projection_matrix, axis=0)

        # Project the regularized kernel matrix onto the learned coefficients.
        # The kernel matrix is (batch, batch); to yield (batch, num_components)
        # we right-multiply by a coefficient block whose first axis is the
        # sample axis. The projection weight is allocated as
        # (feature_dim, num_components) because batch is dynamic and cannot be a
        # weight dimension; its first `batch_size` rows are used as the
        # per-sample coefficients.
        #
        # DECISION plan_2026-06-08_a5f40f4f/D-005: this dynamic slice
        # `projection_matrix[:batch_size, :]` is KEPT (NOT replaced by the whole
        # matrix). It is graph-safe under TF (slicing a static weight by a
        # symbolic index is supported and verified). Using the full projection
        # would make the matmul (batch,batch)@(feature_dim,num_components)
        # shape-incoherent whenever feature_dim != batch. WORKING-BAR LIMITATION
        # (see class docstring): this re-uses the projection weight's leading
        # rows as sample coefficients, which is the idiosyncratic "deep KPCA"
        # approximation and REQUIRES feature_dim (the layer's input/component
        # dim) >= batch_size; it is NOT a textbook Nystrom/eigh KPCA projection.
        components = ops.matmul(kernel_matrix_reg, projection_matrix[:batch_size, :])

        # Normalize by eigenvalues (approximate scaling)
        components = components / (ops.sqrt(ops.abs(eigenvalues) + 1e-10))

        return components

    # -----------------------------------------------------------------
    # Genuine kernel-PCA fit (adapt) + fitted Nystrom transform
    # -----------------------------------------------------------------

    def _rbf_gamma(self, level: int, feature_dim: int) -> float:
        """RBF gamma for a level (matches compute_kernel_matrix's default)."""
        return float(self.kernel_params[level].get('gamma', 1.0 / feature_dim))

    @staticmethod
    def _rbf_pairwise_np(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        """exp(-gamma * ||a_i - b_j||^2), numpy, shape (len(a), len(b))."""
        a_sq = np.sum(a * a, axis=1, keepdims=True)            # (n, 1)
        b_sq = np.sum(b * b, axis=1, keepdims=True).T          # (1, m)
        dist = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
        return np.exp(-gamma * dist)

    def _rbf_pairwise_keras(
            self,
            a: keras.KerasTensor,
            b: keras.KerasTensor,
            gamma: float,
    ) -> keras.KerasTensor:
        """exp(-gamma * ||a_i - b_j||^2), keras ops, shape (batch, M)."""
        a_sq = ops.sum(ops.square(a), axis=1, keepdims=True)          # (batch, 1)
        b_sq = ops.transpose(
            ops.sum(ops.square(b), axis=1, keepdims=True)
        )                                                            # (1, M)
        dist = a_sq + b_sq - 2.0 * ops.matmul(a, ops.transpose(b))
        dist = ops.maximum(dist, 0.0)
        return ops.exp(-gamma * dist)

    def adapt(self, data: Union[np.ndarray, keras.KerasTensor]) -> None:
        # DECISION plan_2026-06-09_be55db55/D-006: this is the correctness-
        # establishing GENUINE kernel-PCA fit. It runs eagerly OUTSIDE call()
        # (like keras.layers.Normalization.adapt) and assigns into tracked
        # weights via .assign / add_weight. The un-fitted call() path is a random
        # projection (sklearn corr ~chance, findings/pca-correctness.md); this
        # adapt replaces it with the canonical Nystrom out-of-sample formula
        # (centered training Gram -> eigh -> alphas = eigvecs / sqrt(eigval);
        # transform = centered K(x, landmarks) @ alphas).
        #
        # DO NOT move any of this into call(): a per-batch call() cannot do a
        # dataset-level eigendecomposition, and an in-call .assign is graph-
        # unsafe (deleted by plan_2026-06-08_a5f40f4f/D-006). DO NOT try to
        # recover the paper's JOINT multi-level optimization here: it is NOT
        # recoverable layer-wise and is OUT OF SCOPE (D-007). The fit is GREEDY
        # layer-wise (each level fits on the previous fitted level's output) and
        # the deep forward/backward COUPLING is intentionally disabled on the
        # fitted path because it corrupts the clean Nystrom projection. See
        # decisions.md D-006 / D-007.
        """Fit genuine (Nystrom) kernel PCA to ``data``, greedy layer-wise.

        Mirrors ``keras.layers.Normalization.adapt``. The adapt ``data`` doubles
        as the Nystrom landmark set. For each level (sequentially): compute the
        centered RBF training Gram of the current representation, eigendecompose
        it, store the top-``components_per_level[level]`` eigenvectors (scaled by
        ``1/sqrt(eigenvalue)``) as Nystrom coefficients, then transform the
        landmarks through this level to produce the input representation for the
        next level (greedy forward fit).

        After ``adapt`` the layer's ``call`` performs the genuine fitted
        transform. Until ``adapt`` is called the layer still RUNS (un-fitted
        random-projection fallback, documented), producing meaningless output.

        :param data: Calibration / landmark data ``(n_samples, input_dim)``.
            ``n_samples`` becomes the number of Nystrom landmarks ``M`` and must
            exceed every level's component count.
        :type data: numpy.ndarray | keras.KerasTensor
        :raises ValueError: if ``n_samples`` is too small for any level's
            requested component count.
        """
        data = ops.convert_to_numpy(
            ops.convert_to_tensor(data, dtype="float32")
        ).astype(np.float64)
        if not self.built:
            self.build(tuple(data.shape))

        n_samples = int(data.shape[0])
        for level in range(self.num_levels):
            k = self.components_per_level[level]
            if n_samples - 1 < k:
                raise ValueError(
                    f"adapt requires at least n_components + 1 = {k + 1} "
                    f"samples to fit level {level}, got n_samples = {n_samples}. "
                    f"Provide more data or reduce components_per_level."
                )

        # Reset any prior fit (re-adapt is allowed).
        self.landmark_reprs = []
        self.nystrom_alphas = []
        self.train_kernel_rowmean = []
        self.train_kernel_allmean = []

        current = data  # (M, feature_dim) representation entering this level
        for level in range(self.num_levels):
            k = self.components_per_level[level]
            feature_dim = current.shape[1]
            gamma = self._rbf_gamma(level, feature_dim)

            # Training Gram of the landmarks at this level + double-centering.
            gram = self._rbf_pairwise_np(current, current, gamma)  # (M, M)
            row_mean = np.mean(gram, axis=1, keepdims=True)        # (M, 1)
            all_mean = float(np.mean(gram))
            gram_c = gram - row_mean - row_mean.T + all_mean

            # eigh -> descending; top-k eigenvectors scaled by 1/sqrt(eigval)
            # are the Nystrom out-of-sample coefficients (alphas).
            eigvals, eigvecs = np.linalg.eigh(gram_c)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            top_vals = np.maximum(eigvals[:k], 1e-12)
            top_vecs = eigvecs[:, :k]
            alphas = top_vecs / np.sqrt(top_vals)                  # (M, k)

            # Persist fitted state for this level as tracked weights.
            self.landmark_reprs.append(self._fitted_weight(
                f'landmark_repr_level_{level}', current.astype(np.float32)))
            self.nystrom_alphas.append(self._fitted_weight(
                f'nystrom_alpha_level_{level}', alphas.astype(np.float32)))
            self.train_kernel_rowmean.append(self._fitted_weight(
                f'train_rowmean_level_{level}',
                row_mean.squeeze(-1).astype(np.float32)))
            self.train_kernel_allmean.append(self._fitted_weight(
                f'train_allmean_level_{level}',
                np.array(all_mean, dtype=np.float32)))

            # Record genuine eigenvalues (descending) into the existing weight.
            self.eigenvalues[level].assign(top_vals.astype(np.float32))

            # Greedy: transform the landmarks through this fitted level to get
            # the next level's input representation. The fitted transform is
            # exactly the out-of-sample formula evaluated on the landmarks
            # themselves: centered Gram rows @ alphas.
            gram_oos_c = (
                gram - row_mean - row_mean.T + all_mean
            )  # landmarks-vs-landmarks centered Gram == gram_c
            current = (gram_oos_c @ alphas)  # (M, k)

        self._fit_flag.assign(1.0)

    def _fitted_weight(self, name: str, value: np.ndarray):
        # DECISION plan_2026-06-09_be55db55/D-006: the Nystrom fitted weights are
        # data-shaped (M landmarks unknown until adapt), so they cannot be
        # created in build(). Keras locks the variable tracker after build, so we
        # briefly unlock it to register these tracked, serialized, non-trainable
        # weights, then re-lock. This is the documented escape hatch for
        # adapt-time state (Normalization keeps its adapt stats in build, but its
        # shapes are config-known; ours are not). DO NOT move this into call()
        # (graph-unsafe) and DO NOT pre-allocate in build() (M is unknown there).
        """Create-or-overwrite a tracked, non-trainable fitted-state weight."""
        self._tracker.unlock()
        try:
            w = self.add_weight(
                name=name,
                shape=value.shape,
                initializer='zeros',
                trainable=False,
            )
        finally:
            self._tracker.lock(
                "You cannot add new elements of state (variables or sub-layers) "
                "to a layer that is already built."
            )
        w.assign(value)
        return w

    def _rebuild_fitted_weights(self, m_landmarks: int) -> None:
        # DECISION plan_2026-06-09_be55db55/D-006: the Nystrom fitted weights are
        # created in adapt() with data-dependent shapes (M landmarks is unknown
        # at build time). load_own_variables() therefore re-creates them here
        # BEFORE assigning the saved values, so a round-trip restores the genuine
        # fitted state. The per-level component count k is config-derived
        # (components_per_level), only M comes from the saved arrays.
        self.landmark_reprs = []
        self.nystrom_alphas = []
        self.train_kernel_rowmean = []
        self.train_kernel_allmean = []
        feature_dim = self.projection_matrices[0].shape[0]
        for level in range(self.num_levels):
            k = self.components_per_level[level]
            self.landmark_reprs.append(self._fitted_weight(
                f'landmark_repr_level_{level}',
                np.zeros((m_landmarks, feature_dim), dtype=np.float32)))
            self.nystrom_alphas.append(self._fitted_weight(
                f'nystrom_alpha_level_{level}',
                np.zeros((m_landmarks, k), dtype=np.float32)))
            self.train_kernel_rowmean.append(self._fitted_weight(
                f'train_rowmean_level_{level}',
                np.zeros((m_landmarks,), dtype=np.float32)))
            self.train_kernel_allmean.append(self._fitted_weight(
                f'train_allmean_level_{level}',
                np.array(0.0, dtype=np.float32)))
            feature_dim = k

    def load_own_variables(self, store) -> None:
        """Re-create fitted weights (if the saved layer was adapted) then load.

        The saved store lists base variables first (creation order) followed by
        ``4 * num_levels`` fitted-state variables when the saved layer was
        adapted. We detect that surplus, read the landmark count ``M`` from the
        first saved fitted array's shape, re-create the data-shaped fitted
        weights, then defer to the default index-based loader.
        """
        n_base = len(self._trainable_variables + self._non_trainable_variables)
        n_store = len(store.keys())
        if n_store > n_base and not self.landmark_reprs:
            # First fitted array is landmark_repr_level_0 -> shape (M, feat).
            first_fitted = np.asarray(store[str(n_base)])
            m_landmarks = int(first_fitted.shape[0])
            self._rebuild_fitted_weights(m_landmarks)
        super().load_own_variables(store)

    def _fitted_transform(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Genuine Nystrom kernel-PCA transform (post-adapt path).

        Per level: center the out-of-sample kernel ``K(x, landmarks)`` against
        the stored training-Gram statistics, project through the stored Nystrom
        alphas, and feed the result as the next level's input (greedy forward,
        no deep coupling -- D-007). Concatenates all levels' components.
        """
        current = inputs
        outputs = []
        for level in range(self.num_levels):
            landmarks = self.landmark_reprs[level]
            gamma = self._rbf_gamma(level, int(landmarks.shape[1]))

            k_oos = self._rbf_pairwise_keras(current, landmarks, gamma)  # (b, M)
            # Double-center against stored training-Gram stats:
            # Kc = K - mean_row(K over landmarks) - train_rowmean + train_allmean
            oos_row_mean = ops.mean(k_oos, axis=1, keepdims=True)        # (b, 1)
            k_centered = (
                k_oos
                - oos_row_mean
                - self.train_kernel_rowmean[level][None, :]
                + self.train_kernel_allmean[level]
            )
            current = ops.matmul(k_centered, self.nystrom_alphas[level])  # (b,k)
            outputs.append(current)

        if len(outputs) == 1:
            return outputs[0]
        return ops.concatenate(outputs, axis=-1)

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
        # === Genuine fitted path (post-adapt) ===
        # DECISION plan_2026-06-09_be55db55/D-006: once adapt() has fitted the
        # Nystrom kernel PCA, call() takes the genuine projection path and the
        # un-fitted random-projection forward/coupling block below is bypassed
        # entirely. The branch is on a Python list length (set in adapt /
        # load_own_variables), NOT on the _fit_flag tensor, so it resolves at
        # trace time and stays graph-safe.
        #
        # DECISION plan_2026-06-09_be55db55/D-007: the fitted path does NOT route
        # through the deep forward/backward COUPLING block below. The coupling
        # (gated residual cross-level mixing) corrupts the clean per-level
        # Nystrom kernel-PCA projection and has no textbook kernel-PCA meaning;
        # the paper's JOINT multi-level optimization that the coupling
        # approximates is NOT recoverable greedily and is OUT OF SCOPE. DO NOT
        # re-enable coupling on the fitted path to "match the paper" -- that is a
        # research problem (F1). The fitted stack is greedy-layer-wise correct
        # (each level a genuine kernel PCA on the previous level's output). See
        # decisions.md D-007.
        if len(self.nystrom_alphas) == self.num_levels:
            return self._fitted_transform(inputs)

        batch_size = ops.shape(inputs)[0]
        current_features = inputs

        # Store intermediate features for coupling
        level_features = []
        level_kernels = []

        # === Forward Pass (un-fitted fallback) ===
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
            components = ops.tanh(components)

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
        """Compute the output shape of the layer.

        The total number of output components is only resolved after ``build``
        (the adaptive ``components_per_level=None`` path is filled in there), so
        this method requires the layer to be built.
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]
        if not self.built or self.components_per_level is None:
            raise ValueError(
                "compute_output_shape requires the layer to be built "
                "(components_per_level is resolved in build()). Build the layer "
                "or call it on a concrete input first."
            )
        batch_size = input_shape[0]
        total_components = sum(self.components_per_level)
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
            # Serialize the ORIGINAL constructor args (sentinels preserved), NOT
            # the post-build mutated attributes. components_per_level=None is the
            # adaptive-sizing sentinel that build() overwrites; kernel_type /
            # kernel_params are expanded to per-level lists in __init__.
            # from_config(get_config()) must reconstruct an identical PRE-build
            # layer.
            'components_per_level': self._components_per_level_init,
            'kernel_type': self._kernel_type_init,
            'kernel_params': self._kernel_params_init,
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
