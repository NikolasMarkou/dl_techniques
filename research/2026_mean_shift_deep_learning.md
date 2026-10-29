# Mean Shift in Deep Learning: Identities, Components and the Tests That Pin Them

*Date: 2026-10-29. Status: implemented and unit-tested on CPU. No training experiment has been run.*

## 1. Scope

Mean shift moves a point to the kernel-weighted mean of a point set. Repeated, it climbs the kernel density estimate of that set to a mode. The library now holds one set of stateless mean-shift operations and five consumers built on it: a pooling layer that returns the dominant mode of a token set instead of its mean, mode extraction for mixture density network (MDN) outputs, an attention layer whose forward pass is mean-shift steps on tied tokens, a differentiable point-set layer with an eval-only clustering readout, and an instance-embedding loss. This note states the identities they implement, where the code lives, which test pins which claim, and what was measured. It describes what is implemented and nothing else; section 7 lists what is not.

| Component | File | Public name | Registry key |
|---|---|---|---|
| Shared operations | `src/dl_techniques/layers/mixtures/mean_shift_ops.py` | `pairwise_sq_dist`, `kernel_logits`, `mean_shift_step`, `kde_log_density`, `mean_shift_iterate`, `median_heuristic_bandwidth` | none (not exported from `layers.mixtures`) |
| Mode pooling | `src/dl_techniques/layers/sequence_pooling/mode_pooling.py` | `ModePooling` | `mode` in `create_sequence_pooling_layer` |
| MDN modes | `src/dl_techniques/layers/statistics/mdn_layer.py` | `get_mixture_modes`, `get_mode_estimate` | none (functions) |
| Attention | `src/dl_techniques/layers/attention/mean_shift_attention.py` | `MeanShiftAttention` | `mean_shift` in `create_attention_layer` |
| Point-set layer | `src/dl_techniques/layers/mixtures/mean_shift.py` | `MeanShiftLayer`, `assign_clusters` | `mean_shift` in `create_mixture_layer` (the function has none) |
| Loss | `src/dl_techniques/losses/mean_shift_instance_loss.py` | `MeanShiftInstanceLoss` | none; exported from `dl_techniques.losses` |
| Float64 oracle (tests only) | `tests/test_layers/test_mixtures/mean_shift_oracle.py` | `oracle_step`, `oracle_iterate`, `oracle_kde_log_density`, `oracle_median_bandwidth`, `oracle_pairwise_sq_dist` | none |

## 2. Identities

All formulas are plain ASCII. `x` is a query, `x_j` the N keys, `h` the bandwidth, `d` the dimension.

### 2.1 Mean shift is a kernel-weighted mean

```
m(x) = sum_j K(x, x_j) x_j / sum_j K(x, x_j)        K(x, y) = exp(-||x - y||^2 / (2 h^2))
```

One mean-shift step replaces `x` by `m(x)`. A fixed point `x = m(x)` is a stationary point of the kernel density estimate.

### 2.2 The shift is h^2 times the score of the density estimate

```
p_hat(x)       = (1/N) sum_j (2 pi h^2)^(-d/2) exp(-||x - x_j||^2 / (2 h^2))
grad log p_hat = (m(x) - x) / h^2          so   m(x) - x = h^2 grad_x log p_hat(x)
```

This is the Miyasawa/Tweedie form: the shift is a score estimate, scaled by the squared bandwidth.

### 2.3 The attention form

```
||x - x_j||^2 / (2 h^2) = ||x||^2 / (2 h^2) - (x . x_j) / h^2 + ||x_j||^2 / (2 h^2)
logits_ij = (x_i . x_j) / h^2 - ||x_j||^2 / (2 h^2)
m(X)      = softmax_j(logits) @ X
```

The query-norm term is constant along `j` and drops out of the softmax (`drop_query_norm=True`); the key-norm term stays. Mean shift on a token set is therefore attention with tied Q = K = V, temperature `h^2` and a key-norm bias. With `h = d_head ** 0.25`, `h^2 = sqrt(d_head)` and the logits are `q.k / sqrt(d_head) - ||k||^2 / (2 sqrt(d_head))`: the scaled dot-product logits minus a key-norm bias. When all keys have equal norm that bias is constant and the weights are exactly scaled dot-product weights. The modern Hopfield update has the same softmax form on equal-norm patterns, with inverse temperature `1/h^2`.

### 2.4 The von Mises-Fisher case

```
||x_j|| = 1, ||x|| = 1:   ||x - x_j||^2 = 2 - 2 x.x_j     logits_ij = kappa x_i . x_j,   kappa = 1/h^2
```

On the unit sphere the key-norm bias is constant, so the expression is the same softmax without it. Inputs are L2-normalised and every iterate is projected back onto the sphere.

### 2.5 Blurring and non-blurring

```
non-blurring:  z <- m(z; keys = x)      keys stay at the input, z climbs a fixed p_hat to a mode
blurring:      z <- m(z; keys = z)      keys move with the points, the whole set contracts
damped step:   z <- (1 - eta) z + eta m(z),   0 < eta < 2
```

After one step with `eta = 1` the two variants are identical.

### 2.6 The density estimate with valid counts

```
log p(x) = logsumexp_j(-||x - x_j||^2 / (2 h^2)) - log N_valid - (d/2) log(2 pi h^2)
```

`N_valid` counts the kept keys (at least 1). A masked key is excluded from the logsumexp by selection. The query-norm term is kept here: this is a density, not a softmax.

### 2.6a The Welsch M-estimator reading

```
rho(r) = h^2 (1 - exp(-||r||^2 / (2 h^2)))
grad_x sum_j rho(x - x_j) = sum_j K(x, x_j) (x - x_j) = 0     <=>     x = m(x)
```

The fixed point of mean shift is a stationary point of the Welsch loss, and a step is one reweighted least-squares step with weights `K`. A far-away token gets weight near zero, which is why a minority of outliers does not move the mode. No test pins this reading beyond the tests of the iteration itself.

### 2.7 Tweedie and the empirical denoiser

```
y = x + sigma n, n ~ N(0, I):        E[x | y] = y + sigma^2 grad log p_Y(y)
prior = empirical set {x_j}:         p_Y is the KDE with h = sigma,   E[x | y] = m(y)
```

The Bayes denoiser under an empirical prior is one non-blurring mean-shift step with `h = sigma`. This repository does not implement that denoiser yet.

### 2.8 The diagonal Gaussian mixture fixed point

```
r_m(x) = softmax_m(log pi_m + log N(x; mu_m, diag(sigma_m^2)))
x     <- (sum_m r_m Lambda_m)^-1 sum_m r_m Lambda_m mu_m,        Lambda_m = diag(1 / sigma_m^2)
```

This is the Carreira-Perpinan fixed point of a Gaussian mixture density. With axis-aligned components the inverse and the product are elementwise. `get_mixture_modes` starts one iteration at every component mean. The mean of a multi-modal mixture lies between its modes; these starts end at the modes.

## 3. Component map

| Component | What it computes |
|---|---|
| Shared operations, `src/dl_techniques/layers/mixtures/mean_shift_ops.py` | `pairwise_sq_dist(a, b)`: `||a||^2 - 2 a.b + ||b||^2` clamped at 0. `kernel_logits(q, k, h, kernel, drop_query_norm)`: the logits of 2.3 or 2.4. `mean_shift_step(q, k, v, h, mask, kernel)`: softmax weights times `v`, returns `(m, weights)`. `kde_log_density(q, k, h, mask)`: 2.6. `mean_shift_iterate(x, h, num_steps, step_size, blurring, mask, kernel, return_weights)`: 2.5. `median_heuristic_bandwidth(x, mask)`: median pairwise distance among the valid points of each sample |
| `ModePooling`, `src/dl_techniques/layers/sequence_pooling/mode_pooling.py` | Pools `(B, N, D)` to `(B, D)`: start at the masked mean (and optionally at kept tokens at evenly spaced ranks), run `num_steps` non-blurring steps against the fixed tokens, return the start that ends at the highest KDE log density, picked by a hard argmax without gradient |
| `get_mixture_modes`, `get_mode_estimate`, `src/dl_techniques/layers/statistics/mdn_layer.py` | 2.8 from every component mean: returns `(modes, log_density, is_unique)` of shapes `[B, M, D]`, `[B, M]`, `[B, M]`. `get_mode_estimate` runs it on a model's MDN output and keeps the highest-density mode per sample, next to `get_point_estimate`, whose mean lands between modes |
| `MeanShiftAttention`, `src/dl_techniques/layers/attention/mean_shift_attention.py` | One shared projection to `H` heads (tied Q = K = V), `num_steps` damped steps per head, an output projection. Registered as `mean_shift` in the attention factory; `TransformerLayer` has an `attention_type='mean_shift'` branch and `DINOv2` lists it in `SUPPORTED_ATTENTION_TYPES` (`src/dl_techniques/models/vision/dino/dino_v2.py`) |
| `MeanShiftLayer`, `assign_clusters`, `src/dl_techniques/layers/mixtures/mean_shift.py` | `MeanShiftLayer` runs `mean_shift_iterate` on a `(B, N, D)` point set, optionally after a bias-free `Dense` projection. `assign_clusters(shifted, merge_radius)` is NumPy: greedy radius merging in index order, labels from 0 per sample |
| `MeanShiftInstanceLoss`, `src/dl_techniques/losses/mean_shift_instance_loss.py` | Per row over valid points and ordered pairs `i != j`: pull = mean `||z_i - z_j||^2` over same-id pairs, push = mean `max(0, margin - ||z_i - z_j||)^2` over different-id pairs, `loss = pull_weight * pull + push_weight * push`, shape `(B,)` |

| Component | Constructor knobs that matter | Numerics and the fully masked row |
|---|---|---|
| Shared operations | `kernel` (`gaussian` or `vmf`), `bandwidth` (scalar, `(B,1,1)`, `(B,H,1,1)` or per query), `step_size` in (0, 2), `blurring`, `mask` (1 = keep) | Logits, softmax and density run in at least float32 (`accumulation_dtype`), result cast back to the input dtype. Masked keys have exactly zero weight, by `where` selection. A row with no kept key gets uniform weights, so its output is the plain mean of `v`; `kde_log_density` treats every key as kept for such a row |
| `ModePooling` | `num_steps=5`, `bandwidth="auto"` or a float, `bandwidth_scale`, `learnable_bandwidth`, `num_starts`, `kernel`, `exclude_positions`, `return_density` | Same accumulation rule. A row with no kept token, including one emptied by `exclude_positions`, returns the plain mean over all positions |
| `get_mixture_modes` | `num_steps=100`, `merge_tol=1e-3` | Float32 or wider (float16 is promoted); sigma floored at 1e-6; Python-int loop, no early stop; buffers `[B, M, M, D]`. `is_unique` is False for a start that matches any earlier start within `merge_tol` |
| `MeanShiftAttention` | `dim`, `num_heads=8`, `head_dim`, `num_steps=1`, `blurring`, `kernel`, `bandwidth_mode`, `bandwidth_init` (default `head_dim ** 0.25`), `min_bandwidth`, `max_bandwidth`, `step_size`, `learnable_step_size`, `dropout_rate`, `use_bias=False` | Float32 softmax under `mixed_float16`. Mask is a keep mask of rank 2, 3 or 4, applied through `apply_attention_mask`. A query row with no kept key gets output zeros and weights zeros (not the mean). Rank-4 input raises in `build` |
| `MeanShiftLayer` | `num_steps=5`, `bandwidth=1.0` in [1e-2, 1e2], `learnable_bandwidth=True`, `blurring`, `step_size`, `kernel`, `use_projection`, `projection_dim`, `return_affinity` | Iteration in at least float32. A masked point is never a key; its own shifted value is computed and finite, callers ignore it. A batch row with no kept point gets uniform weights, so the plain mean |
| `MeanShiftInstanceLoss` | `bandwidth=1.0`, `margin` (default `2 * bandwidth`), `pull_weight`, `push_weight`, `max_points=1024`, `ignore_label=-1` | Float32 or wider. Points with the ignore id take no part. A row with one instance or no valid point costs 0, not NaN. A row over `max_points` is subsampled at evenly spaced indices (static `N` only) |

## 4. Test map

Test file paths are relative to the repository root. Functions are cited by name; a class name, then `::`, then a name is a method of that class.

| Identity or behaviour | Test file | Test |
|---|---|---|
| Float64 oracle can fail (RED proofs for the three identity checks) | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_identity_check_passes_on_the_real_ops`, `test_identity_check_fails_on_an_injected_defect` |
| `pairwise_sq_dist`: oracle, non-negative, zero diagonal, clamp | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_1_pairwise_sq_dist_matches_oracle_is_nonnegative`, `test_1_pairwise_sq_dist_has_a_zero_diagonal`, `test_1_pairwise_sq_dist_clamps_cancellation_at_zero` |
| One step is the softmax with key-norm bias times `x` | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_2_one_step_is_softmax_with_key_bias_times_x` |
| Dropping the query norm leaves the weights unchanged | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_3_dropping_the_query_norm_leaves_weights_unchanged` |
| `m(x) - x = h^2 grad log p_hat`; density value vs oracle | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_4_shift_equals_h_squared_times_density_gradient`, `test_4_density_value_matches_the_oracle` |
| Two-blob convergence (non-blurring reaches the blob modes) | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_5_non_blurring_iteration_reaches_the_blob_modes` |
| Bandwidth limits: huge `h` gives the masked mean, tiny `h` gives the input points | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_6_huge_bandwidth_returns_the_masked_mean`, `test_6_tiny_bandwidth_returns_the_input_points` |
| Masked keys have zero weight; fully masked row is finite and the plain mean; masked KDE | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_7_masked_keys_have_exactly_zero_weight`, `test_7_fully_masked_row_is_finite_and_the_plain_mean`, `test_kde_masked_and_fully_masked_rows_are_finite_and_match_the_oracle` |
| Permutation equivariance of the blurring iteration | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_8_blurring_iteration_is_permutation_equivariant` |
| Mixed precision (float16 vs float32, squares beyond the float16 range) | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_9_float16_inputs_are_finite_and_close_to_float32`, `test_9_float16_with_squares_beyond_the_float16_range_stays_finite` |
| `tf.function` with unknown batch and length | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_10_graph_mode_with_unknown_batch_and_length_matches_eager` |
| vMF step, sphere projection; iteration vs oracle; median heuristic | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_vmf_one_step_is_softmax_of_the_dot_product_over_h_squared`, `test_vmf_iteration_stays_on_the_sphere_and_matches_the_oracle`, `test_iteration_matches_the_oracle`, `test_median_heuristic_matches_the_oracle`, `test_median_heuristic_has_no_gradient` |
| `ModePooling` robustness: mode stays on the big blob, mean is pulled | `tests/test_layers/test_mode_pooling.py` | `TestRobustness::test_the_mode_stays_on_the_big_blob_while_the_mean_is_pulled`, `TestRobustness::test_red_proof_the_plain_mean_fails_the_robustness_body` |
| `ModePooling` multi-start finds the heavier mode | `tests/test_layers/test_mode_pooling.py` | `TestMultiStart::test_one_start_returns_the_light_blob_and_four_starts_the_heavy_one`, `TestMultiStart::test_the_density_follows_the_selected_start` |
| `ModePooling` masks, exclusions, fully masked row, order invariance at one start | `tests/test_layers/test_mode_pooling.py` | `TestMasking::test_masked_tokens_have_no_influence`, `TestMasking::test_a_fully_masked_row_is_the_plain_mean_and_finite`, `test_one_start_is_invariant_to_token_order` |
| `ModePooling` precision, graph, factory | `tests/test_layers/test_mode_pooling.py` | `TestPrecisionAndGraphs::test_mixed_float16_is_finite_and_close_to_float32`, `TestPrecisionAndGraphs::test_a_graph_with_unknown_batch_and_length_matches_eager_at_two_lengths`, `TestShapesAndFactory::test_the_factory_builds_mode_and_lists_it` |
| MDN: single component, equal separated components, bimodal contrast with the mean | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheModesAreTheMixtureModes::test_a_single_component_has_its_mean_as_mode`, `TestTheModesAreTheMixtureModes::test_the_mode_estimate_of_two_separated_components_has_two_unique_modes`, `TestTheContrastCanFail::test_the_point_estimate_of_the_bimodal_case_fails_the_not_the_mean_check`, `TestTheContrastCanFail::test_the_mode_estimate_passes_the_same_check` |
| MDN: overlap vs float64 grid, unequal weights, float64 fixed point | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheModesAreTheMixtureModes::test_overlapping_components_have_one_mode_equal_to_the_grid_maximum`, `TestTheModesAreTheMixtureModes::test_unequal_weights_select_the_heavier_component`, `TestTheModesAreTheMixtureModes::test_the_modes_match_the_float64_fixed_point` |
| MDN: log-density oracle, uniqueness, tracing | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheLogDensity::test_the_log_density_matches_the_float64_oracle_at_the_returned_modes`, `TestUniqueness::test_identical_components_leave_exactly_one_unique_start`, `TestTracing::test_an_unknown_batch_size_traces_and_matches_eager` |
| `MeanShiftAttention`: vMF is softmax attention, Gaussian is `mean_shift_step`, default bandwidth gives scaled dot-product logits minus the key-norm bias (checks `vmf`, `step`, `default`) | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestIdentities::test_the_identity_holds`, `TestIdentities::test_the_identity_fails_on_an_injected_defect`, `TestIdentities::test_a_fixed_step_size_damps_the_shift` |
| `MeanShiftAttention`: blurring vs non-blurring | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestIteration::test_blurring_and_non_blurring_agree_at_one_step`, `TestIteration::test_blurring_and_non_blurring_differ_from_two_steps`, `TestIteration::test_non_blurring_keys_stay_at_the_projected_tokens` |
| `MeanShiftAttention`: masks, zeroed rows, permutation | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestMasks::test_a_query_row_with_no_kept_key_is_zeroed`, `TestMasks::test_a_fully_masked_key_set_gives_zeros_and_stays_finite`, `TestMasks::test_permuting_the_tokens_permutes_the_output` |
| `MeanShiftAttention`: bandwidth, parameter count, gradients | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestBandwidth::test_the_softplus_inverse_round_trips`, `TestBandwidth::test_the_effective_bandwidth_starts_at_the_default`, `TestSerialization::test_fewer_parameters_than_the_multi_head_attention_of_the_same_width`, `TestGradients::test_every_gradient_is_finite_and_the_weights_are_live` |
| `MeanShiftAttention`: precision, two lengths from one traced function, factory | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `TestPrecision::test_one_concrete_function_serves_two_lengths`, `TestFactory::test_the_key_is_registered_and_listed` |
| `MeanShiftAttention` inside `TransformerLayer` | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestTransformerLayer::test_the_block_builds_and_runs`, `TestTransformerLayer::test_a_padding_mask_changes_nothing_for_the_unpadded_rows` |
| `TransformerLayer` branch and `DINOv2` supported set | `tests/test_layers/test_transformers/test_transformer.py`, `tests/test_models/test_dino/test_the_dinov2_block_draws_each_attention_projection_independently.py` | `TestTransformerLayerExtendedAttention::test_construct_and_forward`, `TestTheSupportedSet::test_the_supported_types_are_exactly_these` |
| `MeanShiftLayer` equals the ops and the float64 oracle | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestIdentities::test_the_layer_equals_the_ops`, `TestIdentities::test_one_step_equals_the_float64_oracle`, `TestIdentities::test_three_steps_equal_the_float64_oracle` |
| `MeanShiftLayer` masks, gradients, liveness | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestMasks::test_a_fully_masked_batch_row_stays_finite`, `TestGradients::test_gradients_are_finite_for_the_inputs_and_the_bandwidth`, `TestLiveness::test_blurring_contracts_the_set_more_than_not_blurring`, `TestLiveness::test_a_large_bandwidth_collapses_to_the_mean` |
| `MeanShiftLayer` precision, two lengths, factory; `assign_clusters` | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `TestPrecision::test_one_concrete_function_serves_two_lengths`, `TestFactory::test_it_builds_with_defaults_and_no_required_parameter`, `TestAssignClusters::test_a_hand_built_two_cluster_set`, `TestAssignClusters::test_it_reads_two_labels_off_the_shifted_output_of_two_blobs` |
| `MeanShiftInstanceLoss`: zero loss when separated, hand-computed value, oracle | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_zero_loss_on_perfectly_separated_clusters`, `test_the_tiny_case_matches_the_hand_computed_value`, `test_matches_the_numpy_oracle_with_ignored_points_and_weights` |
| `MeanShiftInstanceLoss`: `sample_weight` arm, degenerate rows, subsampling | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_zeroing_a_sample_weight_equals_dropping_the_row`, `test_degenerate_rows_cost_a_finite_amount_with_a_finite_gradient`, `test_the_subsample_is_deterministic_and_equals_the_evenly_spaced_subset` |
| End to end: trained embedding, `MeanShiftLayer` and `assign_clusters` recover 3 blobs | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_end_to_end_three_blobs_are_recovered_by_a_trained_embedding_and_mean_shift` |

## 5. Measured numbers

CPU, float32 unless stated. Each row names the test whose setup was used; rows marked "script" were measured by a one-off script that imports that test file's helpers and prints the quantity, rows marked "test comment" are the value recorded in a comment of that test.

| Quantity | Value | Source |
|---|---|---|
| `ModePooling`, 70 inliers (std 0.1) at the origin and 30 outliers around 5 on axis 0 (std 0.5), `D=8`, `num_steps=10`: distance from the blob centre | mean 1.47; mode 0.0024 (`bandwidth=0.5`), 0.0050 (`bandwidth="auto"`); the test bound is 0.01 | script, setup of `TestRobustness::test_the_mode_stays_on_the_big_blob_while_the_mean_is_pulled` in `tests/test_layers/test_mode_pooling.py` |
| Parameters at `dim=64`, 8 heads, no bias | `MeanShiftAttention` 8200 (two 64x64 kernels and 8 raw bandwidths); `MultiHeadAttention` 16384 (four 64x64 kernels) | script (`count_params()` after `build`), comparison asserted by `TestSerialization::test_fewer_parameters_than_the_multi_head_attention_of_the_same_width` in `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| One step vs float64 oracle, max abs error of the output | float32 2.6e-7, float64 1.1e-15 | script, setup of `test_2_one_step_is_softmax_with_key_bias_times_x` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` (seeds 20 and 21, `h=0.9`); the test bounds are 1e-5 and 1e-10 |
| `max abs (m(x) - x - h^2 grad log p_hat)`, with `max abs shift` 2.7 | float32 3.1e-7, float64 8.9e-16 | script, setup of `test_4_shift_equals_h_squared_times_density_gradient` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` (seeds 5 and 6, `h=0.7`) |
| float16 vs float32, max abs difference, ops after 1 and 3 steps | 3.0e-3 and 5.2e-3; bound 1e-2 | script, setup of `test_9_float16_inputs_are_finite_and_close_to_float32` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` |
| float16 vs float32, `ModePooling` (`num_starts=3`, with a mask) | pooled 2.4e-4, log density 4.7e-3; bounds 1e-2 and 2e-2 | script, setup of `TestPrecisionAndGraphs::test_mixed_float16_is_finite_and_close_to_float32` in `tests/test_layers/test_mode_pooling.py` |
| float16 vs float32, `MeanShiftAttention` after 3 steps | 1.5e-3 (Gaussian), 4.4e-4 (vMF); bound 0.5 percent of the output scale | test comment in `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| float16 vs float32, `MeanShiftLayer` after 3 steps | 1.3e-3 (Gaussian), 3.0e-4 (vMF) | test comment in `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `tests/test_layers/test_mixtures/test_mean_shift.py` |
| `MeanShiftAttention` gradients of `sum(out^2)`, the three bandwidth modes, `num_steps` 1, 3, 8, both kernels | every gradient finite. Max abs gradient of the two projection kernels at default bandwidth: Gaussian 29 to 59, vMF 9.3 to 10. Raw-bandwidth gradient (the bias of the bandwidth projection for `per_token`) at `num_steps <= 3`: Gaussian 74 to 130, vMF 0.09 to 0.85 | script, helper `_gradients` and setup of `TestGradients::test_every_gradient_is_finite_and_the_weights_are_live` in `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| Same, bandwidth pinned at `min_bandwidth` | all finite; raw-bandwidth gradient at `num_steps <= 3` is 2e-11 to 1.4e-10 (Gaussian) and 6e-21 to 1.3e-20 (vMF), nonzero but vanishing | same script and test |
| 3 blobs, MLP embedding + `MeanShiftLayer(num_steps=3, bandwidth=1.0)` + `MeanShiftInstanceLoss`, blob scale 0.25, `merge_radius=0.5`, 30 epochs, 4 held-out samples | adjusted Rand index 1.0 on every sample and 3 clusters, for seeds 0, 1 and 2; untrained: 1 cluster, index 0.0; the test pins seeds 0 and 1 with a bound of 0.9 | script (seed 2 added), `test_end_to_end_three_blobs_are_recovered_by_a_trained_embedding_and_mean_shift` in `tests/test_losses/test_mean_shift_instance_loss.py` |

## 6. Parameter and numeric choices

Bandwidth parametrisations:

| Layer | Parametrisation |
|---|---|
| `MeanShiftAttention`, `bandwidth_mode="global"` | one raw scalar, `h = clip(min + softplus(raw), min, max)` |
| `MeanShiftAttention`, `"per_head"` (default) | one raw value per head, same map |
| `MeanShiftAttention`, `"per_token"` | a `Dense(num_heads)` on the projected tokens with zero kernel and bias set to the raw initial value, so `h` starts uniform and varies per head and query token |
| `MeanShiftLayer` | one scalar `log h` (weight created with `autocast=False`), `h = clip(exp(log h), 1e-2, 1e2)`; a fixed bandwidth creates no weight |
| `ModePooling` | `h` from `bandwidth` (a float, or `"auto"` for the median distance), times `bandwidth_scale`, times `exp(log_scale)` when `learnable_bandwidth` (`log_scale` starts at 0); floored at 1e-6 |
| `MeanShiftInstanceLoss` | `bandwidth` only sets the default `margin = 2 * bandwidth` |

- Initial value. The raw bandwidth starts at the stable inverse softplus of `bandwidth_init - min`, `y + log(-expm1(-y))`, which is exact for every `y > 0` in double precision; the default `bandwidth_init` is `head_dim ** 0.25`, the scaled dot-product temperature (`TestBandwidth::test_the_softplus_inverse_round_trips`, `tests/test_layers/test_attention/test_mean_shift_attention.py`).
- Step size. `eta` is a Python float in (0, 2). With `learnable_step_size` it is `2 * sigmoid(raw)` per head, `raw` starting at `log(eta0 / (2 - eta0))`.
- Median heuristic. The median of the off-diagonal pairwise distances among the valid points of each sample (a distance, not a squared distance), under `stop_gradient`, 1.0 for a sample with fewer than two valid points, floored at 1e-6. It uses a sort and a flat-row `take`, so it works with an unknown `N`. When most token pairs are far apart (many outliers scattered in different directions) the median is large and the mode drifts toward the mean; lower `bandwidth_scale` or pass a float.
- Clip gradient. Outside the clip range the gradient to the bandwidth is zero (`TestGradients::test_the_bandwidth_gradient_is_zero_outside_the_clip_range`, `tests/test_layers/test_mixtures/test_mean_shift.py`).
- Dtype. Kernel math runs in at least float32 because a squared coordinate beyond 256 already overflows float16 (maximum 65504) and `exp` overflows float16 beyond about 11; the result is cast back to the compute dtype. `test_9_float16_with_squares_beyond_the_float16_range_stays_finite` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` uses points at 300.
- Selection. Argmax selection uses a flat-row `take`, not `take_along_axis`, and iteration counts are Python ints.

Failure modes and what counters them:

- Blurring collapse. With a wide bandwidth or many steps the blurring set contracts toward one point (`TestLiveness::test_blurring_contracts_the_set_more_than_not_blurring`, `TestLiveness::test_a_large_bandwidth_collapses_to_the_mean` in `tests/test_layers/test_mixtures/test_mean_shift.py`). Counters: `ModePooling` never blurs, `MeanShiftAttention` and `MeanShiftLayer` take `blurring=False`, `num_steps` stays small.
- Bandwidth to 0 or to infinity. `h` to 0 gives one-hot weights (output equals the input point, bandwidth gradient vanishes); `h` to infinity gives uniform weights (output is the masked mean). Counter: the clip range and the softplus floor.
- Mode bifurcations. The set of modes of a density estimate changes with `h`: modes can appear or disappear, so a mode finder's output is not a smooth function of `h` at those values. Nothing in the code tracks this and no test pins it.
- Hard selection. `ModePooling` with `num_starts > 1` picks a start by a hard argmax with no gradient, and the extra starts are tokens chosen by position, so the result depends on token order. Only `num_starts=1` is asserted order-invariant.
- Cost. Weights are `(B, H, N, N)`, so time and memory are O(N^2); `get_mixture_modes` buffers are `[B, M, M, D]`; the loss subsamples rows over `max_points`.
- Cluster ids. `assign_clusters` is NumPy, not differentiable, order-dependent (greedy merge in index order) and must stay outside `tf.function` and any training step.

## 7. Limits and open items

Not implemented:

- No experiment on real data has been run: `MeanShiftAttention` has not been compared with softmax attention in a trained model, `ModePooling` has not been used in a classification head, and `MeanShiftInstanceLoss` has only been run on the synthetic 3-blob toy.
- No empirical denoiser oracle: mean shift on a patch or embedding set has not been compared with a trained Miyasawa denoiser (2.7 is a derivation, not code).
- No kNN or windowed variants: every step is dense.
- `get_mixture_modes` handles diagonal Gaussian components only.

Follow-on ideas, none of them started:

- ModeNorm: normalise a token set around its mode instead of its mean.
- Semantic mode decoding: decode from MDN modes instead of the mixture mean.
- MeanShiftFormer: merge tokens by mean-shift cluster.
- Repulsive DETR queries: a repulsive kernel term that pushes object queries apart.
- Mean-shift self-supervision: targets that are mean-shifted embeddings.
- Iterated soft-argmax: local mean shift on a heatmap as a refined soft-argmax.
- TabM mode ensembling: combine ensemble members at their dominant mode.
- Sparse and linear kernels: neighbour-limited or random-feature kernels against the O(N^2) cost.
- DEQ solve: solve the fixed point with a root finder instead of unrolled steps.

## 8. References

- Fukunaga and Hostetler, 1975. The estimation of the gradient of a density function, with applications in pattern recognition. IEEE Transactions on Information Theory 21(1).
- Cheng, 1995. Mean shift, mode seeking, and clustering. IEEE TPAMI 17(8).
- Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature space analysis. IEEE TPAMI 24(5).
- Carreira-Perpinan, 2000. Mode-finding for mixtures of Gaussian distributions. IEEE TPAMI 22(11).
- Carreira-Perpinan, 2007. Gaussian mean-shift is an EM algorithm. IEEE TPAMI 29(5).
- Carreira-Perpinan, 2015. A review of mean-shift algorithms for clustering. arXiv 1503.00687.
- Banerjee, Dhillon, Ghosh and Sra, 2005. Clustering on the unit hypersphere using von Mises-Fisher distributions. JMLR 6.
- Ramsauer et al., 2021. Hopfield networks is all you need. ICLR. arXiv 2008.02217.
- Vaswani et al., 2017. Attention is all you need. NeurIPS. arXiv 1706.03762.
- Miyasawa, 1961. An empirical Bayes estimator of the mean of a normal population. Bulletin of the International Statistical Institute 38.
- Efron, 2011. Tweedie's formula and selection bias. Journal of the American Statistical Association 106(496).
- De Brabandere, Neven and Van Gool, 2017. Semantic instance segmentation with a discriminative loss function. arXiv 1708.02551.
- Kong and Fowlkes, 2018. Recurrent pixel embedding for instance grouping. CVPR. arXiv 1712.08273.
