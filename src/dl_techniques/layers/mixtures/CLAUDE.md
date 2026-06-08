# mixtures

Differentiable soft-clustering / mixture layers for `dl_techniques`.

## Layers
- `radial_basis_function.py` — `RBFLayer`: radial basis function layer with center repulsion.
- `kmeans.py` — `KMeansLayer`: differentiable K-means (temperature softmax assignments, momentum + centroid repulsion).
- `gmm.py` — `GMMLayer`: differentiable Gaussian Mixture Model. Posterior responsibilities over `K` diagonal Gaussians; trainable means / log-variances / mixing logits; isometric-kernel `add_loss` (training-gated) driving covariances toward isotropy.

## Factory
`factory.py` — `MixtureType` Literal (`'rbf'|'kmeans'|'gmm'`), `MIXTURE_REGISTRY`, `create_mixture_layer(mixture_type, **kwargs)`, `create_mixture_from_config(config)`, `get_mixture_info()`, `validate_mixture_config()`.

## Conventions
- All three layers default `*_initializer='orthonormal'`; the string is resolved lazily in `build()` (it is not a registered keras alias — resolving eagerly raises). `n_components`/`n_clusters > feature_dims` falls back to GlorotNormal.
- Public API: `from dl_techniques.layers.mixtures import RBFLayer, KMeansLayer, GMMLayer, create_mixture_layer`.
- Tests: `tests/test_layers/test_mixtures/`.
