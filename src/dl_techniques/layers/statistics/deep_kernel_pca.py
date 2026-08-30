"""
Deep Kernel PCA: a stack of kernel-PCA levels.

This module ships one layer, ``DeepKernelPCA``. It chains several kernel-PCA
levels so the extracted features are richer than a single shallow kernel PCA
gives you.

**What the layer actually computes.** Call ``adapt(data)`` first. After that
the layer runs a greedy, per-level Nystrom kernel PCA. Each level is an
independent kernel PCA fitted on the previous fitted level's output. Levels are
fitted one after another and nothing flows back. The inter-level coupling is
switched OFF on this path.

Before ``adapt`` the layer still runs, on a random-projection fallback whose
output is meaningless. The forward and backward coupling described in the
reference paper is active only on that fallback path.

.. important::

    This layer does NOT solve the paper's coupled, globally-joint multi-level
    optimization. That objective is OUT OF SCOPE: it cannot be recovered from a
    greedy layer-wise ``adapt``. Wherever the text below says "coupled",
    "joint" or "globally coherent", it is describing the reference paper, not
    this code.

Reference-paper background (NOT implemented here):
The paper's contribution is a forward and a backward coupling between levels.
Forward coupling feeds the components of level `j-1` into level `j`, so deeper
levels work on more abstract features. Backward coupling sends information from
level `j+1` back to level `j`, so a level's features must also serve the levels
after it. Together they turn a greedy layer-by-layer fit into one joint
optimization over all levels.

Foundational Mathematics:
Standard kernel PCA maps data into a high-dimensional feature space `φ(x)`
through the kernel trick, then runs PCA there. You never form `φ(x)`. You form
the kernel matrix `K` with `Kᵢⱼ = k(xᵢ, xⱼ) = <φ(xᵢ), φ(xⱼ)>` and take its
eigenvectors.

The paper extends that to `L` levels at once. It looks for coefficient sets
`{α¹, ..., α^L}` that minimize reconstruction error at every level
simultaneously, with the input to kernel `K^(j)` derived from `α^(j-1)` and
extra terms linking `α^(j)` to `α^(j+1)`. Solving that jointly is what makes
the paper's representation globally consistent. This layer solves the levels
one at a time instead.

References:
    - [Tonin, P. A., et al. (2023). Deep Kernel Principal Component Analysis
      for Multi-level Feature Learning.](https://arxiv.org/abs/2302.11220)
"""

import keras
import numpy as np
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, List, Dict, Any
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.statistics.deep_kernel_pca")
class DeepKernelPCA(keras.layers.Layer):
    """
    Multi-level kernel PCA, fitted greedily one level at a time.

    The layer stacks ``num_levels`` kernel-PCA levels. Level 0 works on the
    input, level 1 on level 0's components, and so on. The output is every
    level's components concatenated.

    Call ``adapt(data)`` before you use the output. ``adapt`` fits a genuine
    Nystrom kernel PCA at each level, in order, each on the previous fitted
    level's output. Inter-level coupling is switched OFF on that fitted path.

    Skip ``adapt`` and the layer still runs, on a random-projection fallback.
    That fallback output is meaningless. It exists so the layer never crashes
    before it is fitted. The reference paper's forward and backward coupling
    lives only there. This layer does not solve the paper's joint multi-level
    objective (``min sum_j ||X^(j) - K^(j) alpha^(j)||^2_F + lambda sum_j
    ||alpha^(j)||^2_2`` over all levels at once); that is out of scope.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │ inputs                  (batch, input_dim) │
        └─────────────────────┬──────────────────────┘
                              ▼
        ┌────────────────────────────────────────────┐
        │ Level 0   kernel matrix -> components      │
        │ weights: projection_matrix_level_0,        │
        │          eigenvalues_level_0               │
        └─────────────────────┬──────────────────────┘
                              │ (batch, components_per_level[0])
                              ▼
        ┌────────────────────────────────────────────┐
        │ Level 1   kernel matrix -> components      │
        │ weights: projection_matrix_level_1,        │
        │          eigenvalues_level_1               │
        └─────────────────────┬──────────────────────┘
                              │ (batch, components_per_level[1])
                              ▼
                             ...
                              ▼
        ┌────────────────────────────────────────────┐
        │ Level num_levels-1   kernel -> components  │
        │ weights: projection_matrix_level_<L-1>,    │
        │          eigenvalues_level_<L-1>           │
        └─────────────────────┬──────────────────────┘
                              │ (batch, components_per_level[-1])
                              ▼
        ┌────────────────────────────────────────────┐
        │ concatenate every level's components       │
        └────────────────────────────────────────────┘
                              ▼
                    (batch, sum(components_per_level))

    What sits inside each level depends on whether ``adapt`` has run. See the
    two diagrams below.

    **One Fitted Level (post-adapt, in _fitted_transform):**

    .. code-block:: text

        current                          landmark_reprs[level]
        (batch, feature_dim)             (M, feature_dim)  [weight]
             │                                 │
             └────────────────┬────────────────┘
                              ▼
        ┌────────────────────────────────────────────┐
        │ RBF kernel  exp(-gamma * ||x - l||^2)      │
        │ gamma = kernel_params[level]['gamma']      │
        └─────────────────────┬──────────────────────┘
                              │ k_oos  (batch, M)
                              ▼
        ┌────────────────────────────────────────────┐
        │ double-center against stored stats         │
        │   k_oos - rowmean(k_oos)                   │
        │         - train_kernel_rowmean[level]      │
        │         + train_kernel_allmean[level]      │
        └─────────────────────┬──────────────────────┘
                              │ (batch, M)
                              ▼
        ┌────────────────────────────────────────────┐
        │ matmul  nystrom_alphas[level]   (M, k)     │
        └────────────────────────────────────────────┘
                              ▼
                          (batch, k)

    ``M`` is the number of ``adapt`` landmarks, ``k`` is
    ``components_per_level[level]``. Only the RBF kernel runs here; the
    ``kernel_type`` setting applies to the un-fitted fallback.

    **Fitted vs Un-fitted Path:**

    .. code-block:: text

                       call(inputs)
                       │
                       │  len(nystrom_alphas) == num_levels ?
                       │
                       ┌─────────────────────────────────┐
                      no                                yes
                       ▼                                 ▼
        ┌────────────────────────────┐    ┌────────────────────────────┐
        │ un-fitted fallback         │    │ _fitted_transform          │
        │ MEANINGLESS OUTPUT         │    │ genuine Nystrom kernel PCA │
        ├────────────────────────────┤    ├────────────────────────────┤
        │ per level:                 │    │ per level:                 │
        │  kernel K   (batch, batch) │    │  K(x, landmarks)  (b, M)   │
        │  + regularization_lambda*I │    │  double-center             │
        │  @ projection rows         │    │  @ nystrom_alphas          │
        │  / sqrt(|eigenvalues|)     │    │                            │
        │  + forward coupling        │    │  NO coupling               │
        │  tanh                      │    │  no tanh                   │
        │ then backward coupling     │    │ no backward pass           │
        │ then exp(-0.1*i) weighting │    │ no level weighting         │
        └──────────────┴─────────────┘    └──────────────┴─────────────┘
                       └─────────────────────────────────┘
                                        │
                                        ▼
                                  concatenate
                         (batch, sum(components_per_level))

    The fork tests a Python list length, so it resolves at trace time and stays
    graph-safe. ``_fit_flag`` is saved state that records the fit; it is never
    read as the branch condition.

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
    :param kwargs: Additional keyword arguments for the ``keras.layers.Layer``
        base class, such as ``name`` and ``dtype``.
    :type kwargs: Any

    :raises ValueError: if ``num_levels`` is not positive, ``coupling_strength``
        is outside ``[0, 1]``, ``regularization_lambda`` is negative, or a
        ``kernel_type`` / ``kernel_params`` / ``components_per_level`` list
        length does not match ``num_levels``.

    :ivar landmark_reprs: Per level, the ``adapt`` landmarks as that level sees
        them, shape ``(M, feature_dim)``. Empty until ``adapt`` runs.
    :vartype landmark_reprs: list[keras.Variable]
    :ivar nystrom_alphas: Per level, the Nystrom out-of-sample coefficients
        ``eigenvectors / sqrt(eigenvalues)``, shape ``(M, k)``. Its length is
        also the fitted / un-fitted branch condition in ``call``.
    :vartype nystrom_alphas: list[keras.Variable]
    :ivar train_kernel_rowmean: Per level, the row means of the training Gram,
        shape ``(M,)``. Used to center out-of-sample kernels.
    :vartype train_kernel_rowmean: list[keras.Variable]
    :ivar train_kernel_allmean: Per level, the grand mean of the training Gram,
        a scalar. Used to center out-of-sample kernels.
    :vartype train_kernel_allmean: list[keras.Variable]

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``. ``input_dim`` must be
        known at build time.

    Output shape:
        2D tensor of shape ``(batch_size, sum(components_per_level))``.

    Example:
        .. code-block:: python

            layer = DeepKernelPCA(num_levels=2, components_per_level=[8, 4])
            layer.adapt(train_x)
            features = layer(batch_x)

    Note:
        The fitted path eigendecomposes (``eigh``) the centered training Gram at
        each level and keeps the top ``k`` eigenvectors. ``eigenvalues`` then
        holds real, descending, data-fitted variances. There is no
        ``feature_dim >= batch_size`` requirement on this path.

    Note:
        The un-fitted fallback is not textbook kernel PCA and is kept only so
        the layer runs before ``adapt``. It never eigendecomposes anything; it
        L2-normalizes the projection columns during training and treats the
        first ``batch_size`` rows of each ``(feature_dim, num_components)``
        projection weight as the per-sample coefficients. That trick
        **requires ``feature_dim >= batch_size`` at every level**.
        ``eigenvalues`` stays at its all-ones init, so explained-variance ratios
        mean nothing until ``adapt`` is called.
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
        """Validate the configuration and store it. No weights are created.

        ``kernel_type`` and ``kernel_params`` are expanded here into one entry
        per level. The raw arguments are kept separately so ``get_config``
        serializes what the caller passed, not the expanded form. Weights arrive
        in ``build``, which also resolves ``components_per_level`` when it was
        passed as "None". See the class docstring for every parameter and for
        the errors raised here.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if not (0.0 <= coupling_strength <= 1.0):
            raise ValueError(f"coupling_strength must be in [0, 1], got {coupling_strength}")
        if regularization_lambda < 0:
            raise ValueError(f"regularization_lambda must be non-negative, got {regularization_lambda}")
        if components_per_level is not None and len(components_per_level) != num_levels:
            raise ValueError(
                f"components_per_level must have one entry per level: got "
                f"{len(components_per_level)} entries ({components_per_level}) "
                f"for num_levels={num_levels}"
            )

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
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.coupling_regularizer = regularizers.get(coupling_regularizer)
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
                raise ValueError(
                    f"kernel_params list length ({len(kernel_params)}) must match "
                    f"num_levels ({num_levels})"
                )
            self.kernel_params = kernel_params

        # Weight attributes for the un-fitted path; build() fills them in.
        self.kernel_weights = []
        self.projection_matrices = []
        self.eigenvalues = []
        self.coupling_weights_forward = []

        # Fitted kernel-PCA state, populated by adapt(). _fit_flag is a
        # non-trainable scalar weight created in build() that records whether
        # adapt() has run. The four lists below are created lazily in adapt()
        # via add_weight so they are tracked and serialized; see the class
        # docstring's :ivar entries for their shapes and meanings.
        self._fit_flag = None
        self.landmark_reprs = []
        self.nystrom_alphas = []
        self.train_kernel_rowmean = []
        self.train_kernel_allmean = []

    @staticmethod
    def _unwrap_input_shape(
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Unwrap a nested list-of-shapes into the single shape this layer takes.

        The functional API may pass a LIST OF SHAPES for multi-input layers; this
        layer is single-input, so unwrap only a true nested list-of-shapes. A
        plain shape serialized as a list (e.g. ``[None, 8]``) must NOT be
        unwrapped: its first element is an int or ``None``, not a shape.

        :param input_shape: Shape tuple, possibly wrapped in a one-element list.
        :type input_shape: tuple[int | None, ...]
        :return: The unwrapped shape.
        :rtype: tuple[int | None, ...]
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            return input_shape[0]
        return input_shape

    def _resolve_components_per_level(self, input_dim: int) -> List[int]:
        """Resolve how many components each level extracts.

        The single place this arithmetic lives. It is a pure function of
        ``input_dim`` and the constructor arguments, which is what lets
        ``compute_output_shape`` answer before ``build`` has run.

        With an explicit ``components_per_level`` the answer is that list.
        With ``None`` the sizes are derived adaptively, shrinking the dimension
        by the golden ratio at each level and never dropping below 1.

        :param input_dim: Size of the last input dimension.
        :type input_dim: int
        :return: One component count per level.
        :rtype: list[int]
        """
        if self._components_per_level_init is not None:
            return list(self._components_per_level_init)

        components = []
        current_dim = input_dim
        for _ in range(self.num_levels):
            # Use golden ratio for smooth reduction
            next_dim = max(int(current_dim * 0.618), 1)
            components.append(min(next_dim, current_dim))
            current_dim = next_dim
        return components

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights for multi-level kernel PCA.

        Also resolves ``components_per_level`` when it was left as "None", using
        ``_resolve_components_per_level``.

        :param input_shape: Shape tuple of the input tensor. A nested
            list-of-shapes from the functional API is unwrapped.
        :type input_shape: tuple[int | None, ...]
        :return: Nothing.
        :rtype: None
        :raises ValueError: if the last input dimension is undefined, or if a
            level asks for more components than it has dimensions. The
            ``components_per_level`` length contract is enforced in ``__init__``,
            which is where it can be seen.
        """
        input_shape = self._unwrap_input_shape(input_shape)

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                f"Last dimension of input must be defined, got "
                f"input_shape={tuple(input_shape)}"
            )

        self.components_per_level = self._resolve_components_per_level(input_dim)

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

            # No backward-coupling weights are allocated. The backward pass in
            # call() reuses transpose(coupling_weights_forward[level + 1]).

            # Update input dimension for next level
            current_input_dim = num_components

        # Non-trainable record of the fit: 0.0 = un-fitted (random projection),
        # 1.0 = fitted (genuine Nystrom kernel PCA). Created here so it
        # serializes. call() branches on len(self.nystrom_alphas), not on this.

        # DECISION plan-2026-08-30T152113-67a8cc1e/D-001: _fit_flag is written
        # and never read. That is intended. DO NOT delete it as dead code:
        # load_own_variables detects a fitted checkpoint by COUNTING weights,
        # so removing it drops n_base 6 -> 5 and every existing .keras file
        # then reads a scalar's .shape[0] and raises IndexError.
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
        """Compute the (batch, batch) kernel matrix for one level.

        Used only by the un-fitted fallback path. Dispatches on
        ``kernel_types[level]`` and, when ``trainable_kernels`` is set, reads the
        kernel's parameters from that level's weights instead of the config.
        Double-centers the result when ``center_kernel`` is set.

        :param x: Input tensor of shape ``(batch_size, features)``.
        :type x: keras.KerasTensor
        :param level: Level index for kernel computation.
        :type level: int
        :return: Kernel matrix of shape ``(batch_size, batch_size)``.
        :rtype: keras.KerasTensor
        :raises ValueError: if ``kernel_types[level]`` is not one of ``"rbf"``,
            ``"polynomial"``, ``"linear"``, ``"sigmoid"``, ``"cosine"``.
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
        """Extract this level's components on the UN-FITTED fallback path.

        This runs no eigendecomposition. It regularizes the kernel matrix
        diagonal, then multiplies by the leading ``batch_size`` rows of the
        learnable projection weight and rescales by the stored eigenvalues. The
        result is only as meaningful as gradient descent has made those weights.
        ``_fitted_transform`` is the genuine kernel-PCA path.

        :param kernel_matrix: Kernel matrix of shape ``(batch_size, batch_size)``.
        :type kernel_matrix: keras.KerasTensor
        :param projection_matrix: Projection matrix of shape ``(feature_dim, num_components)``.
        :type projection_matrix: keras.KerasTensor
        :param eigenvalues: Eigenvalues for this level.
        :type eigenvalues: keras.KerasTensor
        :param num_components: Number of components to extract. Documented for
            the caller's benefit; the shape comes from ``projection_matrix``.
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

        # Keep the projection columns on the unit sphere during training. Test
        # `training is True`, not truthiness: a symbolic or None flag under
        # graph tracing must not take the training branch. ops.nn.l2_normalize
        # does not exist in this Keras build; ops.normalize(x, axis=0) is the
        # column-wise form.
        if training is True:
            projection_matrix = ops.normalize(projection_matrix, axis=0)

        # The kernel matrix is (batch, batch), so the right operand's first axis
        # must be the sample axis. The projection weight is allocated as
        # (feature_dim, num_components) because batch is dynamic and cannot be a
        # weight dimension, so its first batch_size rows stand in as the
        # per-sample coefficients.

        # DECISION plan_2026-06-08_a5f40f4f/D-005: keep the dynamic slice
        # projection_matrix[:batch_size, :]. DO NOT pass the whole weight: the
        # matmul is then (batch,batch)@(feature_dim,k), which fails whenever
        # feature_dim != batch. The slice REQUIRES feature_dim >= batch_size at
        # every level, so a large batch over few features raises here.
        components = ops.matmul(kernel_matrix_reg, projection_matrix[:batch_size, :])

        # Normalize by eigenvalues (approximate scaling)
        components = components / (ops.sqrt(ops.abs(eigenvalues) + 1e-10))

        return components

    # -----------------------------------------------------------------
    # Genuine kernel-PCA fit (adapt) + fitted Nystrom transform
    # -----------------------------------------------------------------

    def _rbf_gamma(self, level: int, feature_dim: int) -> float:
        """RBF gamma for a level, matching ``compute_kernel_matrix``'s default.

        :param level: Level index.
        :type level: int
        :param feature_dim: Feature width entering this level. Used only for the
            ``1 / feature_dim`` fallback when no gamma was configured.
        :type feature_dim: int
        :return: The gamma to use at this level.
        :rtype: float
        """
        return float(self.kernel_params[level].get('gamma', 1.0 / feature_dim))

    @staticmethod
    def _rbf_pairwise_np(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        """Pairwise RBF kernel ``exp(-gamma * ||a_i - b_j||^2)`` in numpy.

        The squared norms broadcast as ``(n, 1)`` against ``(1, m)``.

        :param a: Array of shape ``(n, d)``.
        :type a: numpy.ndarray
        :param b: Array of shape ``(m, d)``.
        :type b: numpy.ndarray
        :param gamma: RBF bandwidth.
        :type gamma: float
        :return: Kernel matrix of shape ``(n, m)``.
        :rtype: numpy.ndarray
        """
        a_sq = np.sum(a * a, axis=1, keepdims=True)
        b_sq = np.sum(b * b, axis=1, keepdims=True).T
        dist = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
        return np.exp(-gamma * dist)

    def _rbf_pairwise_keras(
            self,
            a: keras.KerasTensor,
            b: keras.KerasTensor,
            gamma: float,
    ) -> keras.KerasTensor:
        """Pairwise RBF kernel ``exp(-gamma * ||a_i - b_j||^2)`` in keras ops.

        The graph-safe twin of ``_rbf_pairwise_np``, used on the fitted forward
        path. The squared norms broadcast as ``(batch, 1)`` against ``(1, M)``.

        :param a: Batch tensor of shape ``(batch, d)``.
        :type a: keras.KerasTensor
        :param b: Landmark tensor of shape ``(M, d)``.
        :type b: keras.KerasTensor
        :param gamma: RBF bandwidth.
        :type gamma: float
        :return: Kernel matrix of shape ``(batch, M)``.
        :rtype: keras.KerasTensor
        """
        a_sq = ops.sum(ops.square(a), axis=1, keepdims=True)
        b_sq = ops.transpose(
            ops.sum(ops.square(b), axis=1, keepdims=True)
        )
        dist = a_sq + b_sq - 2.0 * ops.matmul(a, ops.transpose(b))
        dist = ops.maximum(dist, 0.0)
        return ops.exp(-gamma * dist)

    def adapt(self, data: Union[np.ndarray, keras.KerasTensor]) -> None:
        """Fit genuine (Nystrom) kernel PCA to ``data``, one level at a time.

        Mirrors ``keras.layers.Normalization.adapt``. The ``data`` you pass also
        becomes the Nystrom landmark set. Per level, in order:

        1. Compute the RBF training Gram of the current representation and
           double-center it.
        2. Eigendecompose it with ``eigh`` and keep the top
           ``components_per_level[level]`` eigenvectors, each divided by
           ``sqrt(eigenvalue)``. Those are the Nystrom coefficients.
        3. Push the landmarks through this level to get the next level's input.

        Re-adapting is allowed; it discards the previous fit. After ``adapt``,
        ``call`` takes the fitted transform. Before it, ``call`` runs the
        un-fitted fallback and its output is meaningless.

        :param data: Calibration / landmark data ``(n_samples, input_dim)``.
            ``n_samples`` becomes the number of Nystrom landmarks ``M`` and must
            exceed every level's component count.
        :type data: numpy.ndarray | keras.KerasTensor
        :return: Nothing. The fitted state is written to the layer's weights.
        :rtype: None
        :raises ValueError: if ``n_samples`` is too small for any level's
            requested component count.
        """
        # DECISION plan_2026-06-09_be55db55/D-006: the genuine kernel-PCA fit
        # lives here, eager and outside call(). DO NOT move it into call(): a
        # per-batch call() cannot do a dataset-level eigendecomposition, and an
        # in-call .assign is graph-unsafe. The fit is greedy per level; the
        # paper's joint objective is not recoverable this way (see D-007).
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

        # `current` holds the (M, feature_dim) representation entering a level.
        current = data
        for level in range(self.num_levels):
            k = self.components_per_level[level]
            feature_dim = current.shape[1]
            gamma = self._rbf_gamma(level, feature_dim)

            # Training Gram (M, M) of the landmarks at this level, then
            # double-centering with the (M, 1) row means and the grand mean.
            gram = self._rbf_pairwise_np(current, current, gamma)
            row_mean = np.mean(gram, axis=1, keepdims=True)
            all_mean = float(np.mean(gram))
            gram_c = gram - row_mean - row_mean.T + all_mean

            # eigh -> descending; the top-k eigenvectors scaled by 1/sqrt(eigval)
            # are the (M, k) Nystrom out-of-sample coefficients (alphas).
            eigvals, eigvecs = np.linalg.eigh(gram_c)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            top_vals = np.maximum(eigvals[:k], 1e-12)
            top_vecs = eigvecs[:, :k]
            alphas = top_vecs / np.sqrt(top_vals)

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

            # Push the landmarks through this fitted level to get the next
            # level's input. This is the out-of-sample formula evaluated on the
            # landmarks themselves, so the centered landmarks-vs-landmarks Gram
            # below equals gram_c. Result is (M, k).
            gram_oos_c = (
                gram - row_mean - row_mean.T + all_mean
            )
            current = (gram_oos_c @ alphas)

        self._fit_flag.assign(1.0)

    def _fitted_weight(self, name: str, value: np.ndarray):
        """Create-or-overwrite a tracked, non-trainable fitted-state weight.

        Keras locks a layer's variable tracker once it is built. This unlocks it
        just long enough to register one more weight, then re-locks it with the
        same message Keras uses, so a later stray ``add_weight`` still fails.

        :param name: Weight name.
        :type name: str
        :param value: Initial value; also fixes the weight's shape.
        :type value: numpy.ndarray
        :return: The created weight, already assigned ``value``.
        :rtype: keras.Variable
        """
        # DECISION plan_2026-06-09_be55db55/D-006: the fitted weights are
        # data-shaped (M is unknown until adapt), so the tracker is unlocked
        # here to create them. DO NOT pre-allocate them in build() instead: M is
        # not known there. DO NOT move this into call(): it is graph-unsafe.
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
        """Re-create the fitted-state weights as empty zeros, ready to load.

        Called by ``load_own_variables`` when the saved layer had been adapted.
        Only ``M`` comes from the saved file; every other shape is derived from
        the config and from ``projection_matrices[0]``. Any previous fitted
        state is dropped.

        :param m_landmarks: Number of Nystrom landmarks ``M`` in the saved file.
        :type m_landmarks: int
        :return: Nothing. The four fitted-state lists are repopulated in place.
        :rtype: None
        """
        # DECISION plan_2026-06-09_be55db55/D-006: adapt() creates the fitted
        # weights with data-dependent shapes, so load_own_variables must
        # re-create them HERE before the saved values are assigned. DO NOT drop
        # this call and rely on the default loader: the variables would not
        # exist yet and a round-trip would silently lose the fit.
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

        :param store: The saved variable store, keyed by stringified index.
        :type store: Any
        :return: Nothing.
        :rtype: None
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
        """Genuine Nystrom kernel-PCA transform, taken after ``adapt``.

        Per level: build the out-of-sample kernel ``K(x, landmarks)``, center it
        against the stored training-Gram statistics, and project it through the
        stored Nystrom alphas. The result feeds the next level. No coupling runs
        here (D-007). All levels' components are concatenated.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Concatenated components, shape
            ``(batch_size, sum(components_per_level))``.
        :rtype: keras.KerasTensor
        """
        current = inputs
        outputs = []
        for level in range(self.num_levels):
            landmarks = self.landmark_reprs[level]
            gamma = self._rbf_gamma(level, int(landmarks.shape[1]))

            # Out-of-sample kernel, shape (batch, M).
            k_oos = self._rbf_pairwise_keras(current, landmarks, gamma)

            # Double-center against the stored training-Gram stats:
            # Kc = K - mean_row(K over landmarks) - train_rowmean + train_allmean
            # The (batch, 1) row mean broadcasts over the M landmark columns.
            oos_row_mean = ops.mean(k_oos, axis=1, keepdims=True)
            k_centered = (
                k_oos
                - oos_row_mean
                - self.train_kernel_rowmean[level][None, :]
                + self.train_kernel_allmean[level]
            )
            # Project onto the Nystrom alphas -> (batch, k).
            current = ops.matmul(k_centered, self.nystrom_alphas[level])
            outputs.append(current)

        if len(outputs) == 1:
            return outputs[0]
        return ops.concatenate(outputs, axis=-1)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass. Takes the fitted path if ``adapt`` has run.

        After ``adapt`` this delegates to ``_fitted_transform``. Before it, the
        un-fitted fallback below runs the coupling stack and returns meaningless
        output. See the class docstring's fitted-vs-un-fitted diagram.

        :param inputs: Input tensor of shape ``(batch_size, input_features)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean flag for training mode. Only the un-fitted
            fallback reads it, to L2-normalize the projection columns.
        :type training: bool | None
        :return: Concatenated principal components from all levels.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-06-09_be55db55/D-006: once adapt() has fitted the
        # layer, call() returns early and the whole un-fitted block below is
        # skipped. DO NOT branch on the _fit_flag tensor instead: this branch is
        # a Python list length (set by adapt / load_own_variables) so it
        # resolves at trace time and stays graph-safe.

        # DECISION plan_2026-06-09_be55db55/D-007: the fitted path does NOT run
        # the forward/backward coupling below. DO NOT re-enable coupling there
        # to "match the paper": gated cross-level mixing corrupts the clean
        # per-level Nystrom projection, and the paper's joint objective it
        # approximates is not recoverable greedily.
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

        Works on an unbuilt layer. The component counts come from
        ``_resolve_components_per_level``, the same pure helper ``build`` uses, so
        the answer before ``build`` is the answer after it.

        :param input_shape: Shape tuple of the input tensor. A nested
            list-of-shapes from the functional API is unwrapped.
        :type input_shape: tuple[int | None, ...]
        :return: ``(batch_size, sum(components_per_level))``.
        :rtype: tuple[int | None, ...]
        :raises ValueError: if the last input dimension is undefined.
        """
        input_shape = self._unwrap_input_shape(input_shape)
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                f"Last dimension of input must be defined, got "
                f"input_shape={tuple(input_shape)}"
            )
        batch_size = input_shape[0]
        total_components = sum(self._resolve_components_per_level(input_dim))
        return (batch_size, total_components)

    def get_explained_variance_ratio(self) -> List[float]:
        """Get the per-component explained variance ratio at each level.

        Each entry is that level's ``eigenvalues`` divided by their sum, so the
        values are fractions in ``[0, 1]``, not percentages. The declared return
        type says ``list[float]``, but each entry is really a numpy array of
        length ``components_per_level[level]``.

        Before ``adapt`` the eigenvalues are all ones, so every ratio comes back
        uniform and means nothing.

        :return: One array of ratios per level.
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
