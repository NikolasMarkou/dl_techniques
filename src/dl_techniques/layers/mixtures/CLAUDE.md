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
- **Training flag**: training-only side effects (KMeans EMA centroid update; GMM/RBF `add_loss`) are gated via `dl_techniques.utils.tensors.resolve_training_factor`. They fire for python `training=True` AND a symbolic `training=tf.constant(True)` tensor (custom `@tf.function` train loops), and are a true no-op under `None`/`False`/symbolic-False — all graph-safe (no tensor→bool coercion). The python-`True` path is numerically exact (unmasked); the symbolic path masks the delta/loss by a 0/1 factor.
- **Mixed precision**: all three layers run their density / distance / kernel math in `variable_dtype` (float32) under a `mixed_float16` policy — weights are `autocast=False`, inputs are cast to `variable_dtype`, and the output is cast to `compute_dtype` on return — so the numerically-sensitive `exp`/`log`/`softmax`/`division` stay full precision while the layer still emits float16 for downstream chaining. No-op under the default float32 policy. (`GMMLayer`/`KMeansLayer` would otherwise crash from an autocast half-vs-float `Sub`; `RBFLayer` is hardened for uniformity.)
- Public API: `from dl_techniques.layers.mixtures import RBFLayer, KMeansLayer, GMMLayer, create_mixture_layer`.
- Tests: `tests/test_layers/test_mixtures/`.
