"""
Kernel PCA with a reconstruction path, built on Random Fourier Features.

Kernel PCA extracts non-linear features well. It gives you no natural way back
from principal components to the original data. That gap is the classical
"pre-image problem".

The published method attacks the gap by replacing the implicit kernel with an
explicit, finite-dimensional feature map. Ordinary linear PCA then runs in a
space you can write down. This module does the forward half that way. Its
inverse is a learned linear decoder, not the paper's analytic pre-image solve.
Read "What the inverse actually is" below before you trust a reconstruction.

Two classes ship here:

- ``InvertibleKernelPCA`` maps inputs to Random Fourier Features, then projects
  them onto principal components. ``adapt`` fits the projection.
  ``inverse_transform`` maps components back toward input space.
- ``InvertibleKernelPCADenoiser`` wraps one of those. It denoises by running
  transform then inverse transform, optionally zeroing small components first.

**How the forward path works:**

1. Random feature mapping. The input is projected into a higher-dimensional
   space by a fixed non-linear map ``z(x)``. The dot product ``z(x)^T z(y)``
   approximates a shift-invariant kernel ``k(x, y)``.
2. Linear PCA. Standard PCA runs on those explicit features. ``adapt``
   eigendecomposes their covariance and stores the top eigenvectors.

**What the inverse actually is:**

``inverse_transform`` is a LEARNED APPROXIMATION, not a true kernel-PCA
pre-image solve. It maps components back to RFF space with the trainable
``reconstruction_matrix``, then decodes RFF space to input space by multiplying
with the transposed random frequency matrix and dividing by
``n_random_features + regularization``. Both steps are linear. Gradient
descent learns ``projection_matrix``, ``reconstruction_matrix`` and
``reconstruction_bias``. The
frequency decoder is fixed: it reuses ``frequencies``, which trains only when
``trainable_frequencies=True`` (default ``False``). Reconstruction quality is
only as good as ``reconstruction_matrix`` makes it. The method's own docstring
lists the exact five-step path.

Foundational Mathematics:
Bochner's theorem says any shift-invariant kernel `k(x, y) = k(x - y)` is the
Fourier transform of a non-negative measure. Rahimi and Recht turned that into
a sampling scheme: approximate the kernel as the mean of a randomized feature
map.

For a shift-invariant kernel `k`, its approximation is:
`k(x, y) ≈ z(x)ᵀz(y)`
The feature map `z(x)` is defined as:
`z(x) = sqrt(2/D) * [cos(ω₁ᵀx + b₁), ..., cos(ω_Dᵀx + b_D)]`
where:
-   `D` is the number of random features.
-   `ωᵢ` are random frequency vectors sampled from a distribution `p(ω)` which
    is the Fourier transform of the kernel `k`. For the Radial Basis Function
    (RBF) kernel `k(x, y) = exp(-γ||x-y||²)`, `p(ω)` is a Gaussian distribution.
-   `bᵢ` are random phase shifts sampled uniformly from `[0, 2π]`.

Project the input data `X` into this feature space `Z` and the kernel matrix is
`K ≈ ZZᵀ`. The problem is now standard linear PCA on `Z`: the principal
components are the eigenvectors of the covariance of `Z`.

Note:
    The `laplacian` and `cauchy` kernel options draw their frequencies from the
    same Gaussian as `rbf`, not from the Cauchy distribution those kernels call
    for. They are working surrogates, not exact RFF maps.

References:
    - [Gedon, A., et al. (2023). Invertible Kernel PCA with Random Fourier
      Features.](https://arxiv.org/abs/2303.05043)
    - [Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale
      Kernel Machines. In NIPS.](
      https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
"""

import keras
import numpy as np
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.statistics.invertible_kernel_pca")
class InvertibleKernelPCA(keras.layers.Layer):
    """
    Kernel PCA in an explicit Random Fourier Feature space, with a decoder.

    The layer maps its input through the RFF map
    ``z(x) = sqrt(2/D) cos(x @ frequencies + phases)``, where ``D`` is
    ``n_random_features``. The dot product ``z(x)^T z(y)`` approximates a
    shift-invariant kernel, so PCA on ``z(x)`` behaves like kernel PCA while
    staying ordinary linear algebra. ``call`` returns the principal components.

    Call ``adapt(data)`` once before you use the output. It eigendecomposes the
    RFF covariance and writes ``feature_mean``, ``projection_matrix`` and
    ``eigenvalues``. Skip it and the layer still runs, but its output is a
    random projection.

    .. warning::

        ``inverse_transform`` is a **LEARNED APPROXIMATION**, NOT a true
        kernel-PCA pre-image solve. It uses a trainable
        ``reconstruction_matrix`` followed by a fixed linear frequency decoder.
        ``reconstruction_matrix`` and ``reconstruction_bias`` are the trained
        parts. The decoder reuses ``frequencies``, which trains only when
        ``trainable_frequencies=True`` (default ``False``).

        Nothing in this package trains them. A loss on this layer's own output
        gives them no gradient, because they are reached only through
        ``inverse_transform``. Straight after ``adapt``, reconstruction is no
        better than returning zeros: relative error 0.9985 on 20 Gaussian
        samples at seed 0. Put ``inverse_transform`` in a loss first. The ``laplacian``
        and ``cauchy`` kernel options draw Gaussian frequencies, not the exact
        Cauchy frequencies those kernels require. No online eigendecomposition
        runs anywhere. See ``inverse_transform`` for the exact reconstruction
        path. This is a self-consistent working layer, not a canonical ikPCA.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │ inputs (batch, input_dim)                │
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
        │ compute_random_features                  │
        │ weights: frequencies (input_dim, D)      │
        │ and phases (D,)                          │
        │ D = n_random_features                    │
        └────────────────────┬─────────────────────┘
                             ▼ (batch, D)
        ┌──────────────────────────────────────────┐
        │ subtract feature_mean        (optional)  │
        │ only when center_features=True           │
        └────────────────────┬─────────────────────┘
                             ▼ (batch, D)
        ┌──────────────────────────────────────────┐
        │ matmul with projection_matrix            │
        │ (D, n_components)                        │
        └────────────────────┬─────────────────────┘
                             ▼ (batch, n_components)
        ┌──────────────────────────────────────────┐
        │ divide by sqrt(|eigenvalues|)            │
        │ (optional) only when whiten=True         │
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
        │ components (batch, n_components)         │
        └──────────────────────────────────────────┘

    ``adapt`` writes ``feature_mean``, ``projection_matrix`` and
    ``eigenvalues``. Of those three only ``projection_matrix`` is created with
    ``trainable=True``, so gradient descent moves it afterwards. The other two
    are ``trainable=False``. ``phases`` never trains. ``frequencies`` trains
    only when ``trainable_frequencies=True`` (default ``False``).

    **Forward vs Inverse:**

    .. code-block:: text

           transform / call            inverse_transform
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │ inputs (b, input_dim)     │   │ output (b, input_dim)     │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      ▼ (b, D)                        ▲ (b, D)
        ┌───────────────────────────┐   ┌─────────────┴─────────────┐
        │ compute_random_features   │   │ + reconstruction_bias     │
        │ sqrt(2/D) *               │   │ (optional: use_bias)      │
        │ cos(x @ freq + phases)    │   │ @ transpose(frequencies)  │
        │                           │   │ / (D + regularization)    │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      ▼                               ▲
        ┌───────────────────────────┐   ┌─────────────┴─────────────┐
        │ - feature_mean            │   │ + feature_mean            │
        │ (optional: centering)     │   │ (optional: centering)     │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      ▼                               ▲
        ┌───────────────────────────┐   ┌─────────────┴─────────────┐
        │ @ projection_matrix       │   │ @ reconstruction_matrix   │
        │ (D, n_components)         │   │ (n_components, D)         │
        │ adapt() + trainable=True  │   │ LEARNED trained weight    │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      ▼                               ▲
        ┌───────────────────────────┐   ┌─────────────┴─────────────┐
        │ / sqrt(|eigenvalues|)     │   │ * sqrt(|eigenvalues|)     │
        │ (optional: whiten)        │   │ (optional: whiten)        │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      ▼                               ▲
                components (b, n_components)

    The two columns share ``frequencies``, ``phases``, ``feature_mean`` and
    ``eigenvalues``. They do NOT share the projection: the inverse uses its own
    ``reconstruction_matrix``, which is why the round trip is approximate even
    after ``adapt``. The forward column applies a cosine; the inverse column has
    no matching non-linearity, only the transposed frequency matmul.

    :param n_components: Number of principal components to keep. Must be
        positive and no larger than ``n_random_features``. Defaults to "None",
        which resolves in ``build`` to ``min(n_random_features, input_dim)``.
    :type n_components: int | None
    :param n_random_features: Size of the RFF space, ``D``. Larger values give
        a closer kernel approximation and cost more memory. Must be positive.
        Defaults to 256.
    :type n_random_features: int
    :param kernel_type: Kernel to approximate. One of "rbf", "laplacian",
        "cauchy". The last two use a Gaussian frequency surrogate, not their
        exact frequency distribution. Defaults to "rbf".
    :type kernel_type: str
    :param gamma: Kernel bandwidth. Defaults to "None", which resolves in
        ``build`` to ``1.0 / input_dim``.
    :type gamma: float | None
    :param center_features: Subtract ``feature_mean`` from the RFF features
        before projecting, and add it back on the way out. Defaults to "True".
    :type center_features: bool
    :param whiten: Divide components by ``sqrt(|eigenvalues|)`` so each has
        unit variance. Defaults to "False".
    :type whiten: bool
    :param regularization: Ridge term. It is added to the diagonal of the RFF
        covariance in ``adapt`` and to the divisor in ``inverse_transform``.
        Must be non-negative. Defaults to 1e-6.
    :type regularization: float
    :param random_seed: Seed for the frequency and phase draws. The phases use
        ``random_seed + 1`` so the two streams do not alias. Defaults to
        "None", which is nondeterministic.
    :type random_seed: int | None
    :param trainable_frequencies: Let gradient descent update ``frequencies``.
        Doing so breaks the kernel approximation the RFF map is derived from.
        Defaults to "False".
    :type trainable_frequencies: bool
    :param use_bias: Add a trainable ``reconstruction_bias`` at the end of
        ``inverse_transform``. Defaults to "True".
    :type use_bias: bool
    :param kernel_regularizer: Regularizer for ``frequencies``. It is attached
        only when ``trainable_frequencies=True``. Defaults to "None".
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param bias_regularizer: Regularizer for ``reconstruction_bias``. Defaults
        to "None".
    :type bias_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional arguments passed to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: if ``n_components`` is not positive, if
        ``n_random_features`` is not positive, if ``kernel_type`` is not one of
        the three supported names, or if ``regularization`` is negative.

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``. The last dimension must
        be known at build time.

    Output shape:
        2D tensor of shape ``(batch_size, n_components)``.

    Example:
        .. code-block:: python

            layer = InvertibleKernelPCA(n_components=8, n_random_features=128)
            layer.adapt(train_x)
            codes = layer(train_x)

            # `adapt` fits the forward path only. Until the reconstruction
            # weights are trained against a loss, `approx` is close to zeros.
            approx = layer.inverse_transform(codes)

    Note:
        ``build`` overwrites ``self.gamma`` and ``self.n_components`` when they
        were passed as "None". ``get_config`` serializes the raw constructor
        values instead, so ``from_config(get_config())`` rebuilds a layer that
        resolves them the same way against a new input shape.
    """

    def __init__(
            self,
            n_components: Optional[int] = None,
            n_random_features: int = 256,
            kernel_type: Literal['rbf', 'laplacian', 'cauchy'] = 'rbf',
            gamma: Optional[float] = None,
            center_features: bool = True,
            whiten: bool = False,
            regularization: float = 1e-6,
            random_seed: Optional[int] = None,
            trainable_frequencies: bool = False,
            use_bias: bool = True,
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and store it. No weights are created.

        Weights arrive in ``build``, which also resolves ``gamma`` and
        ``n_components`` when they were passed as "None". See the class
        docstring for every parameter and for the errors raised here.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if n_components is not None and n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        if n_random_features <= 0:
            raise ValueError(f"n_random_features must be positive, got {n_random_features}")
        if kernel_type not in ['rbf', 'laplacian', 'cauchy']:
            raise ValueError(f"kernel_type must be 'rbf', 'laplacian', or 'cauchy', got {kernel_type}")
        if regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {regularization}")

        # Store configuration.
        # ``build`` mutates ``self.gamma`` (None -> 1/input_dim) and
        # ``self.n_components`` (None -> min(...)). Keep the RAW constructor
        # arguments as sentinels so ``get_config`` serializes the pre-build
        # values and ``from_config(get_config())`` rebuilds an identical
        # pre-build layer (see get_config).
        self._gamma_init = gamma
        self._n_components_init = n_components
        self.n_components = n_components
        self.n_random_features = n_random_features
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.center_features = center_features
        self.whiten = whiten
        self.regularization = regularization
        self.random_seed = random_seed
        self.trainable_frequencies = trainable_frequencies
        self.use_bias = use_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Initialize weight attributes (created in build)
        self.frequencies = None
        self.phases = None
        self.projection_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.feature_mean = None
        self.reconstruction_matrix = None
        self.reconstruction_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the RFF and PCA weights.

        Resolves ``gamma`` to ``1.0 / input_dim`` and ``n_components`` to
        ``min(n_random_features, input_dim)`` when either was passed as "None",
        then creates ``frequencies``, ``phases``, ``projection_matrix``,
        ``eigenvalues``, ``feature_mean`` (only when ``center_features``),
        ``reconstruction_matrix`` and ``reconstruction_bias`` (only when
        ``use_bias``).

        :param input_shape: Shape tuple of the input tensor. Its last entry
            must be a known integer.
        :type input_shape: tuple[int | None, ...]
        :return: Nothing.
        :rtype: None
        :raises ValueError: if the last dimension of ``input_shape`` is
            ``None``, if ``n_components`` exceeds ``n_random_features``, or if
            ``kernel_type`` is not recognised.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Set gamma if not specified
        if self.gamma is None:
            self.gamma = 1.0 / input_dim

        # Determine actual number of components
        if self.n_components is None:
            self.n_components = min(self.n_random_features, input_dim)
        elif self.n_components > self.n_random_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot be larger than "
                f"n_random_features ({self.n_random_features})"
            )

        # Initialize random number generator
        if self.random_seed is not None:
            initializer_seed = self.random_seed
        else:
            initializer_seed = None

        # Create random frequencies based on kernel type
        if self.kernel_type == 'rbf':
            # For RBF kernel: ω ~ N(0, 2γI)
            freq_stddev = np.sqrt(2 * self.gamma)
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=freq_stddev,
                seed=initializer_seed
            )
        elif self.kernel_type == 'laplacian':
            # DOCUMENTED APPROXIMATION. The exact Laplacian kernel needs
            # frequencies drawn from a Cauchy distribution, which has infinite
            # variance. This branch substitutes a scaled Gaussian
            # (stddev=gamma). It is a working surrogate, not an exact
            # Laplacian RFF map.
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=self.gamma,
                seed=initializer_seed
            )
        elif self.kernel_type == 'cauchy':
            # DOCUMENTED APPROXIMATION. Same Gaussian surrogate as the
            # laplacian branch above, not an exact Cauchy-kernel RFF map.
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=self.gamma,
                seed=initializer_seed
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Random frequency matrix ω
        self.frequencies = self.add_weight(
            name='frequencies',
            shape=(input_dim, self.n_random_features),
            initializer=freq_initializer,
            trainable=self.trainable_frequencies,
            regularizer=self.kernel_regularizer if self.trainable_frequencies else None
        )

        # Random phase vector b ~ Uniform(0, 2π)

        # DECISION plan-2026-08-22T035419-a11304c8/D-304: the seed guard is
        # `is not None`, never truthiness. `random_seed=0` is falsy, so a bare
        # `if initializer_seed` left the phases unseeded while the frequencies
        # above stayed seeded: 60 builds at seed 0 gave 60 DISTINCT sklearn
        # correlations, 0.1244..0.9958. See decisions.md D-304.
        phase_initializer = initializers.RandomUniform(
            minval=0.0,
            maxval=2 * np.pi,
            seed=initializer_seed + 1 if initializer_seed is not None else None
        )

        # Phases stay fixed. Training them would break the RFF kernel
        # approximation, and no option exposes them.
        self.phases = self.add_weight(
            name='phases',
            shape=(self.n_random_features,),
            initializer=phase_initializer,
            trainable=False
        )

        # PCA projection matrix (from RFF space to principal components)
        self.projection_matrix = self.add_weight(
            name='projection_matrix',
            shape=(self.n_random_features, self.n_components),
            initializer='orthogonal',
            trainable=True
        )

        # Eigenvalues for whitening and variance tracking
        self.eigenvalues = self.add_weight(
            name='eigenvalues',
            shape=(self.n_components,),
            initializer='ones',
            trainable=False
        )

        # Mean vector for centering RFF features
        if self.center_features:
            self.feature_mean = self.add_weight(
                name='feature_mean',
                shape=(self.n_random_features,),
                initializer='zeros',
                trainable=False
            )

        # Reconstruction matrix for inverse transform
        # Maps from principal components back to RFF space
        self.reconstruction_matrix = self.add_weight(
            name='reconstruction_matrix',
            shape=(self.n_components, self.n_random_features),
            initializer='orthogonal',
            trainable=True
        )

        # Optional bias for reconstruction
        if self.use_bias:
            self.reconstruction_bias = self.add_weight(
                name='reconstruction_bias',
                shape=(input_dim,),
                initializer='zeros',
                trainable=True,
                regularizer=self.bias_regularizer
            )

        super().build(input_shape)

    def adapt(self, data: Union[np.ndarray, keras.KerasTensor]) -> None:
        """Fit the kernel-PCA projection to ``data``.

        This is the step that makes the layer's output mean anything. It
        mirrors ``keras.layers.Normalization.adapt``: it runs eagerly, outside
        ``call``, and assigns into weights that already exist.

        It computes the Random Fourier Features of ``data``, eigendecomposes
        their covariance, then stores the top ``n_components`` eigenvectors in
        ``projection_matrix``, the matching eigenvalues in descending order in
        ``eigenvalues``, and the RFF mean in ``feature_mean``.

        Until you call it, the layer still runs. Its ``projection_matrix``
        stays at its orthogonal init, so the output is a random projection of
        the RFF features and correlates with true kernel PCA at chance.

        :param data: Calibration data of shape ``(n_samples, input_dim)``.
        :type data: numpy.ndarray | keras.KerasTensor
        :return: Nothing. The fitted values are written into the weights.
        :rtype: None
        :raises ValueError: if ``n_samples`` is too small to estimate the
            RFF covariance for the requested ``n_components``.
        """
        # DECISION plan_2026-06-09_be55db55/D-005: keep this fit here, eager
        # and outside call(). DO NOT move it (or any .assign) into call():
        # in-call variable assignment is unsafe in TF graph mode. The fit is
        # DATASET-LEVEL (eigendecomposition of the full RFF covariance) and
        # cannot be computed per batch.

        # Build (creates the weights) if not yet built.
        data = ops.convert_to_tensor(data, dtype="float32")
        if not self.built:
            self.build(tuple(data.shape))

        n_samples = int(data.shape[0])

        # RFF features -> numpy, shape (n_samples, n_random_features).
        # float64 because eigh on a near-rank-deficient covariance is
        # sensitive to precision.
        rff = ops.convert_to_numpy(self.compute_random_features(data)).astype(
            np.float64
        )

        # Under-determination guard: a reliable covariance estimate needs more
        # samples than the requested components (rank of the empirical
        # covariance is at most n_samples - 1).
        if n_samples - 1 < self.n_components:
            raise ValueError(
                f"adapt requires at least n_components + 1 = "
                f"{self.n_components + 1} samples to estimate the RFF "
                f"covariance, got n_samples = {n_samples}. Provide more data "
                f"or reduce n_components."
            )

        # Center the RFF features. feature_mean_value is (n_random_features,).
        feature_mean_value = np.mean(rff, axis=0)
        rff_centered = rff - feature_mean_value

        # RFF covariance (n_random_features, n_random_features), diagonally
        # regularized for numerical stability (rank-deficient when
        # n_samples < n_random_features).
        cov = (rff_centered.T @ rff_centered) / (n_samples - 1)
        cov += self.regularization * np.eye(cov.shape[0], dtype=cov.dtype)

        # eigh returns ascending eigenvalues; reverse to descending and take
        # the top-n_components eigenvectors (columns).
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # top_eigvecs is (n_random_features, k) and top_eigvals is (k,),
        # where k is n_components.
        top_eigvecs = eigvecs[:, : self.n_components]
        top_eigvals = eigvals[: self.n_components]

        # Assign the fitted state into the existing weights (eager / outside
        # call -> legal). Cast back to the weights' dtype.
        if self.center_features:
            self.feature_mean.assign(
                feature_mean_value.astype(np.float32)
            )
        self.projection_matrix.assign(top_eigvecs.astype(np.float32))
        # Clamp tiny/negative eigenvalues (regularization can push them
        # slightly negative) to keep whitening well-defined.
        self.eigenvalues.assign(
            np.maximum(top_eigvals, 0.0).astype(np.float32)
        )

    def compute_random_features(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute Random Fourier Features for the input.

        Returns ``sqrt(2/D) * cos(inputs @ frequencies + phases)``, with ``D``
        equal to ``n_random_features``. The scale is what makes
        ``z(x)^T z(y)`` approximate the kernel rather than ``D`` times it.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: RFF tensor of shape ``(batch_size, n_random_features)``.
        :rtype: keras.KerasTensor
        """
        # Compute linear projections: Xω
        linear_proj = ops.matmul(inputs, self.frequencies)

        # Add random phases: Xω + b
        proj_with_phase = linear_proj + self.phases

        # Apply cosine transformation: cos(Xω + b)
        cos_features = ops.cos(proj_with_phase)

        # Scale by sqrt(2/D) for proper kernel approximation
        scale = ops.sqrt(2.0 / self.n_random_features)
        rff_features = scale * cos_features

        return rff_features

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Map inputs to principal components.

        RFF map, then optional centering, then the projection, then optional
        whitening. The class docstring draws the whole path. ``training`` is
        accepted for the Keras contract and changes nothing here.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Unused.
        :type training: bool | None
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        # Compute Random Fourier Features
        rff_features = self.compute_random_features(inputs)

        # DECISION plan_2026-06-08_a5f40f4f/D-006: call() runs NO stateful
        # update. DO NOT reintroduce an in-call `.assign` "online update": the
        # removed `update_pca_components` assigned to feature_mean and
        # projection_matrix inside the forward graph, unsafe in TF graph mode.
        # adapt() fits both; projection_matrix is trainable=True. See D-005.

        # Center features if requested
        if self.center_features:
            centered_features = rff_features - self.feature_mean
        else:
            centered_features = rff_features

        # Project to principal components
        components = ops.matmul(centered_features, self.projection_matrix)

        # Whiten if requested (divide by sqrt of eigenvalues)
        if self.whiten:
            components = components / (ops.sqrt(ops.abs(self.eigenvalues) + 1e-10))

        return components

    def transform(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Map inputs to principal components with ``training=False``.

        A thin alias for ``call``, kept for the scikit-learn-shaped API.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        return self.call(inputs, training=False)

    def inverse_transform(
            self,
            components: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reconstruct data from principal components.

        .. note::

            This is a **LEARNED APPROXIMATION**, NOT an analytic kernel-PCA
            pre-image. The path is a single linear chain:

            1. (optional) un-whiten the components,
            2. map components back to RFF space with the trainable
               ``reconstruction_matrix`` ``(n_components, n_random_features)``,
            3. (optional) add back ``feature_mean``,
            4. decode RFF space to input space with the transposed random
               ``frequencies`` ``(input_dim, D)`` as a fixed linear decoder,
               scaled by ``1/(D + regularization)``,
            5. (optional) add the trainable ``reconstruction_bias``.

            Step 4 has no counterpart to the cosine applied on the way in, so
            the round trip is approximate no matter how well ``adapt`` fits the
            forward projection. Reconstruction quality is only as good as the
            trainable weights make it.

        :param components: Principal components of shape ``(batch_size, n_components)``.
        :type components: keras.KerasTensor
        :return: Reconstructed data of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-06-08_a5f40f4f/D-007: one LEARNED inverse path.
        # DO NOT reintroduce the removed arccos + freq_gram/freq_proj
        # pseudo-inverse code: those tensors were computed and discarded, and
        # a real pre-image solver is out of scope.

        # Un-whiten if whitening was applied
        if self.whiten:
            components = components * ops.sqrt(ops.abs(self.eigenvalues) + 1e-10)

        # Map components back to RFF space via the learned reconstruction matrix.
        rff_reconstructed = ops.matmul(components, self.reconstruction_matrix)

        # Add back the mean if centering was used
        if self.center_features:
            rff_reconstructed = rff_reconstructed + self.feature_mean

        # Linearly decode RFF space -> input space using the (transposed)
        # frequency matrix as a fixed linear decoder. frequencies is
        # (input_dim, D); its transpose maps (batch, D) -> (batch, input_dim).
        reconstructed = ops.matmul(
            rff_reconstructed,
            ops.transpose(self.frequencies)
        ) / (self.n_random_features + self.regularization)

        # Add bias if used
        if self.use_bias:
            reconstructed = reconstructed + self.reconstruction_bias

        return reconstructed

    def fit_transform(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Run ``call`` twice and return the second result.

        .. warning::

            This method does NOT fit anything, despite the name. ``call`` is
            stateless, so the discarded ``training=True`` pass changes nothing
            and the result equals ``transform(inputs)``. The fit lives in
            ``adapt``. Call that instead.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        # The first pass is a no-op held over from an earlier in-call fit that
        # D-006 removed. It is kept so the public signature does not change.
        _ = self.call(inputs, training=True)

        # Second pass to transform
        return self.call(inputs, training=False)

    def compute_reconstruction_error(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute per-sample reconstruction error.

        Runs ``transform`` then ``inverse_transform`` and returns the mean
        squared error against the input, one value per sample. Expect it to be
        large until the reconstruction weights have been trained.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Mean squared error of shape ``(batch_size,)``.
        :rtype: keras.KerasTensor
        """
        # Transform to components
        components = self.transform(inputs)

        # Reconstruct
        reconstructed = self.inverse_transform(components)

        # Compute MSE
        error = ops.mean(ops.square(inputs - reconstructed), axis=-1)

        return error

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: ``(batch_size, n_components)``.
        :rtype: tuple[int | None, ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.n_components)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        Emits the RAW ``n_components`` and ``gamma`` the caller passed, not the
        values ``build`` resolved them to, so ``from_config(get_config())``
        rebuilds an identical pre-build layer.

        :return: Serializable config dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            # Serialize the RAW constructor sentinels (not the build-mutated
            # values) so from_config rebuilds an identical pre-build layer.
            'n_components': self._n_components_init,
            'n_random_features': self.n_random_features,
            'kernel_type': self.kernel_type,
            'gamma': self._gamma_init,
            'center_features': self.center_features,
            'whiten': self.whiten,
            'regularization': self.regularization,
            'random_seed': self.random_seed,
            'trainable_frequencies': self.trainable_frequencies,
            'use_bias': self.use_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer) if self.kernel_regularizer else None,
            'bias_regularizer': regularizers.serialize(self.bias_regularizer) if self.bias_regularizer else None,
        })
        return config


@register_dl_technique("dl_techniques.layers.statistics.invertible_kernel_pca")
class InvertibleKernelPCADenoiser(keras.layers.Layer):
    """
    Denoiser that projects through an ``InvertibleKernelPCA`` and back.

    The layer owns one ``InvertibleKernelPCA`` child, built with
    ``whiten=True`` and ``center_features=True``. It transforms the noisy input
    to components, then inverse transforms. Directions the fit gave small
    eigenvalues carry little signal, so squeezing the data through
    ``n_components`` of them drops most of the noise.

    Set ``adaptive_components=True`` and the layer also hard-thresholds the
    components during training. It estimates a per-sample noise level from the
    input and zeroes every component whose magnitude falls at or below
    ``noise_level * sqrt(2.0)``. This fork runs ONLY when ``training`` is true.
    At inference the components pass through untouched.

    Call ``adapt(data)`` before use. It forwards to the child, which is where
    the actual kernel-PCA fit happens.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │ noisy inputs (batch, input_dim)              │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │ self.ikpca(inputs, training)                 │
        │ child InvertibleKernelPCA owns the weights   │
        └──────────────────────┬───────────────────────┘
                               ▼ components (batch, n_components)
               adaptive_components and training ?
                        ┌──────┴────────────────────────┐
                       yes                             no
                        ▼                               ▼
        ┌────────────────────────────────┐    ┌────────────────────┐
        │ estimate_noise_level(inputs)   │    │ components pass    │
        │ threshold = noise * sqrt(2.0)  │    │ through unchanged  │
        │ zero every |c| <= threshold    │    │ (no thresholding)  │
        └───────────────┬────────────────┘    └─────────┬──────────┘
                        └──────┬────────────────────────┘
                               ▼ (batch, n_components)
        ┌──────────────────────────────────────────────┐
        │ self.ikpca.inverse_transform(components)     │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │ denoised (batch, input_dim)                  │
        └──────────────────────────────────────────────┘

    This layer creates no weights of its own. Every weight belongs to the
    ``ikpca`` child, and both boxes that touch it reference the same instance.

    :param n_components: How many components to keep. An int is used as is. A
        float in ``(0, 1]`` is read as a fraction and ``build`` turns it into
        ``max(1, int(n_random_features * fraction))``. Defaults to 0.95.
    :type n_components: int | float
    :param n_random_features: Size of the child's RFF space. Defaults to 512.
    :type n_random_features: int
    :param kernel_type: Kernel passed to the child. Defaults to "rbf".
    :type kernel_type: str
    :param gamma: Kernel bandwidth passed to the child. Defaults to "None",
        which lets the child resolve it to ``1.0 / input_dim``.
    :type gamma: float | None
    :param adaptive_components: Enable the training-time hard threshold on the
        components. Defaults to "False".
    :type adaptive_components: bool
    :param noise_estimation: How ``estimate_noise_level`` works. "mad" uses
        ``1.4826 * median(|x - median(x)|)``. "std" uses the standard deviation
        of the first difference along the last axis. Defaults to "mad".
    :type noise_estimation: str
    :param kwargs: Additional arguments passed to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: if ``n_components`` is a float outside ``(0, 1]``.

    :ivar ikpca: The child ``InvertibleKernelPCA``. ``None`` until ``build``
        creates it, or until ``from_config`` restores a saved one.
    :vartype ikpca: InvertibleKernelPCA | None

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``.

    Output shape:
        Same as the input, ``(batch_size, input_dim)``.

    Example:
        .. code-block:: python

            denoiser = InvertibleKernelPCADenoiser(n_components=0.5)
            denoiser.adapt(clean_x)
            clean = denoiser(noisy_x)

    Note:
        The float ``n_components`` is NOT a variance fraction, despite reading
        like one. ``build`` multiplies it by ``n_random_features`` without
        looking at any eigenvalue, so the default 0.95 keeps 486 of 512
        components.
    """

    def __init__(
            self,
            n_components: Union[int, float] = 0.95,
            n_random_features: int = 512,
            kernel_type: str = 'rbf',
            gamma: Optional[float] = None,
            adaptive_components: bool = False,
            noise_estimation: Literal['mad', 'std'] = 'mad',
            **kwargs: Any
    ) -> None:
        """Store the configuration and split the ``n_components`` argument.

        An int goes straight to ``self.n_components`` and leaves
        ``self.variance_threshold`` at ``None``. A float goes the other way,
        and ``build`` resolves it. The raw argument is kept on
        ``self.n_components_param`` for ``get_config``. No child layer and no
        weights are created here. See the class docstring for the parameters.
        """
        super().__init__(**kwargs)

        self.n_components_param = n_components
        self.n_random_features = n_random_features
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.adaptive_components = adaptive_components
        self.noise_estimation = noise_estimation

        # Determine actual number of components
        if isinstance(n_components, float):
            if not (0 < n_components <= 1):
                raise ValueError(f"When float, n_components must be in (0, 1], got {n_components}")
            # Will be determined based on variance
            self.n_components = None
            self.variance_threshold = n_components
        else:
            self.n_components = n_components
            self.variance_threshold = None

        # Create ikPCA layer
        self.ikpca = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve ``n_components`` and build the ``ikpca`` child.

        Reuses an existing child when ``from_config`` already restored one, so
        deserialization does not silently replace saved weights with a fresh
        layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Nothing.
        :rtype: None
        """
        # Turn a float n_components into a count. This scales by
        # n_random_features only; it does not inspect any eigenvalue.
        if self.variance_threshold is not None:
            self.n_components = max(
                1,
                int(self.n_random_features * self.variance_threshold)
            )

        # Create ikPCA child layer only if it was not already provided by
        # from_config (deserialization rebuilds the child from its saved config).
        # whiten=True puts every retained component on the same scale, so the
        # single adaptive threshold below applies to all of them equally.
        if self.ikpca is None:
            self.ikpca = InvertibleKernelPCA(
                n_components=self.n_components,
                n_random_features=self.n_random_features,
                kernel_type=self.kernel_type,
                gamma=self.gamma,
                whiten=True,
                center_features=True,
                name='ikpca_denoiser'
            )

        # Build ikPCA child before finalizing this layer's build.
        if not self.ikpca.built:
            self.ikpca.build(input_shape)

        # super().build() LAST so the layer is only marked built after all
        # child weights exist.
        super().build(input_shape)

    def estimate_noise_level(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Estimate the per-sample noise level of the input.

        With ``noise_estimation="mad"`` it takes the median absolute deviation
        along the last axis and scales by 1.4826, the factor that makes MAD an
        unbiased standard-deviation estimate for Gaussian data. With ``"std"``
        it takes the standard deviation of the first difference
        ``inputs[:, 1:] - inputs[:, :-1]``, which suppresses smooth structure
        and leaves the high-frequency part.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Noise level of shape ``(batch_size, 1)``.
        :rtype: keras.KerasTensor
        :raises ValueError: if ``noise_estimation`` is neither "mad" nor "std".
        """
        if self.noise_estimation == 'mad':
            # Median Absolute Deviation
            median = ops.median(inputs, axis=-1, keepdims=True)
            mad = ops.median(ops.abs(inputs - median), axis=-1, keepdims=True)
            # Convert MAD to standard deviation estimate (assuming Gaussian noise)
            noise_level = 1.4826 * mad
        elif self.noise_estimation == 'std':
            # Standard deviation of high-frequency components
            diff = inputs[:, 1:] - inputs[:, :-1]
            noise_level = ops.std(diff, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown noise estimation method: {self.noise_estimation}")

        return noise_level

    def adapt(self, data: Union[np.ndarray, keras.KerasTensor]) -> None:
        """Fit the ``ikpca`` child to ``data``.

        Builds this layer and its child if needed, then delegates to
        ``self.ikpca.adapt(data)`` so the transform-and-back path runs in the
        fitted principal subspace. Skip this and the denoiser projects through
        a random subspace.

        :param data: Calibration data of shape ``(n_samples, input_dim)``.
        :type data: numpy.ndarray | keras.KerasTensor
        :return: Nothing.
        :rtype: None
        """
        data = ops.convert_to_tensor(data, dtype="float32")
        if not self.built:
            self.build(tuple(data.shape))
        self.ikpca.adapt(data)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Denoise inputs by projecting through the ikPCA child and back.

        The adaptive threshold branch runs only when
        ``adaptive_components=True`` AND ``training`` is true. The class
        docstring draws both leaves.

        :param inputs: Noisy input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Gates the adaptive threshold.
        :type training: bool | None
        :return: Denoised tensor of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        # Transform to principal components
        components = self.ikpca(inputs, training=training)

        # Adaptive component selection based on noise level
        if self.adaptive_components and training:
            noise_level = self.estimate_noise_level(inputs)
            # Zero every component at or below the noise floor. sqrt(2.0) is
            # the confidence factor: a component must clear the estimated
            # noise by that margin to survive.
            threshold = noise_level * ops.sqrt(2.0)
            mask = ops.abs(components) > threshold
            components = components * ops.cast(mask, components.dtype)

        # Reconstruct denoised signal
        denoised = self.ikpca.inverse_transform(components)

        return denoised

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: The same shape tuple.
        :rtype: tuple[int | None, ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        Emits the raw ``n_components`` argument from ``n_components_param``,
        plus the nested ``ikpca`` config once the child exists. ``from_config``
        consumes that nested entry and restores the exact child instead of
        letting ``build`` derive a new one.

        :return: Serializable config dictionary. ``ikpca_config`` is ``None``
            until the layer is built.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'n_components': self.n_components_param,
            'n_random_features': self.n_random_features,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'adaptive_components': self.adaptive_components,
            'noise_estimation': self.noise_estimation,
            # Nested child config (None until built). Lets from_config rebuild
            # the identical child ikPCA rather than re-deriving it in build().
            'ikpca_config': (
                keras.saving.serialize_keras_object(self.ikpca)
                if self.ikpca is not None else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "InvertibleKernelPCADenoiser":
        """Rebuild the denoiser, restoring the nested ikPCA child if present.

        :param config: Config dictionary from ``get_config``.
        :type config: dict[str, Any]
        :return: A denoiser whose ``ikpca`` attribute is the deserialized
            child, or ``None`` when the saved layer was never built.
        :rtype: InvertibleKernelPCADenoiser
        """
        config = dict(config)
        ikpca_config = config.pop('ikpca_config', None)
        instance = cls(**config)
        if ikpca_config is not None:
            instance.ikpca = keras.saving.deserialize_keras_object(ikpca_config)
        return instance

# ---------------------------------------------------------------------