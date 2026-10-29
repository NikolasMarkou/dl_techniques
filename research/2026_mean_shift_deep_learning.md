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

Masking uses no sentinel. The row maximum is taken over the kept logits only (a masked entry enters it as negative infinity), masked weights are set to exactly 0 and the kept weights are divided by their own sum. A fixed constant such as -1e9 is not safe: at a small bandwidth the kept logits `-||x - x_j||^2 / (2 h^2)` can be far below any constant (about -1e12 at `h = 1e-6`), the masked keys would win the softmax and the pooled vector would be zero.

### 2.6a Translation safety

```
||x - x_j||^2 = ||(x - c) - (x_j - c)||^2        for any c
```

The Gaussian kernel depends on differences only, but the expanded form `||a||^2 - 2 a.b + ||b||^2` cancels two large squared norms in float32 when the points sit far from the origin. The ops therefore subtract a stop-gradient centre `c` (the mean of the kept keys, so a junk value stored at a masked position cannot set it) before forming the distance, and `mean_shift_step` averages the centred values (`w @ (v - c) + c`, exact because the weights sum to 1). The Gaussian logits are then defined up to a per-query constant, which softmax and logsumexp remove; they no longer equal `q.k/h^2 - ||k||^2/(2h^2)` entry by entry, only along the key axis. The vMF kernel is not centred: its inputs are L2-normalised.

### 2.6b The Welsch M-estimator reading

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

This is the Carreira-Perpinan fixed point of a Gaussian mixture density. With axis-aligned components the inverse and the product are elementwise. `get_mixture_modes` starts one iteration at every component mean, moved off it by `1e-3` standard deviations times a fixed multiplier for each component and dimension (sign either way, magnitude 0.25 to 1, never zero, no two equal), so that coordinate-symmetric mixtures leave their saddles as well; with `num_steps=0` the starts are not moved and the exact means come back. The mean of a multi-modal mixture lies between its modes; these starts end at the modes.

What it returns are stationary points reached from each start, not a guarantee of local maxima. The iteration never lowers the density, so a stationary point that is a minimum or a saddle is unstable and a start leaves it; only a start exactly on one stays, and an offset direction shared by every component would keep a start of a swap-symmetric mixture on its symmetric manifold, which the per-component multipliers prevent. Without the offset, components at -1, 0, +1 in one dimension (sigma 0.5, weights 0.45, 0.1, 0.45) return the middle mean, where the Hessian of `log p` is +4.79, as a unique mode. `get_mode_estimate` takes the highest density among the points and is therefore a true mode. Convergence is linear and slows near the merge point of two components, about two standard deviations apart (section 6).

## 3. Component map

| Component | What it computes |
|---|---|
| Shared operations, `src/dl_techniques/layers/mixtures/mean_shift_ops.py` | `pairwise_sq_dist(a, b, key_mask)`: `||a||^2 - 2 a.b + ||b||^2` clamped at 0, on points centred by the kept-key mean (2.6a). `kernel_logits(q, k, h, kernel, drop_query_norm, key_mask)`: the logits of 2.3 or 2.4, up to a per-query constant for the Gaussian kernel. Both convert every operand to a tensor, so a numpy array or a variable works. `mean_shift_step(q, k, v, h, mask, kernel)`: softmax weights times `v`, returns `(m, weights)`. `kde_log_density(q, k, h, mask)`: 2.6. `mean_shift_iterate(x, h, num_steps, step_size, blurring, mask, kernel, return_weights)`: 2.5. `median_heuristic_bandwidth(x, mask)`: median pairwise distance among the valid points of each sample |
| `ModePooling`, `src/dl_techniques/layers/sequence_pooling/mode_pooling.py` | Pools `(B, N, D)` to `(B, D)`: start at the masked mean (and optionally at kept tokens at evenly spaced ranks), run `num_steps` non-blurring steps against the fixed tokens, return the start that ends at the highest KDE log density, picked by a hard argmax without gradient |
| `get_mixture_modes`, `get_mode_estimate`, `src/dl_techniques/layers/statistics/mdn_layer.py` | 2.8 from every component mean (offset by `1e-3` sigma times a per-component, per-dimension multiplier of magnitude 0.25 to 1; `num_steps=0` returns the exact means): stationary points, returns `(modes, log_density, is_unique)` of shapes `[B, M, D]`, `[B, M]`, `[B, M]`. `get_mode_estimate` runs it on a model's MDN output and keeps the highest-density mode per sample, next to `get_point_estimate`, whose mean lands between modes |
| `MeanShiftAttention`, `src/dl_techniques/layers/attention/mean_shift_attention.py` | One shared projection to `H` heads (tied Q = K = V), `num_steps` damped steps per head, an output projection. Registered as `mean_shift` in the attention factory; `TransformerLayer` has an `attention_type='mean_shift'` branch and `DINOv2` lists it in `SUPPORTED_ATTENTION_TYPES` (`src/dl_techniques/models/vision/dino/dino_v2.py`) |
| `MeanShiftLayer`, `assign_clusters`, `src/dl_techniques/layers/mixtures/mean_shift.py` | `MeanShiftLayer` runs `mean_shift_iterate` on a `(B, N, D)` point set, optionally after a bias-free `Dense` projection. `assign_clusters(shifted, merge_radius, mask)` is NumPy: greedy radius merging in index order, labels from 0 per sample; points with a zero mask get -1 and never open or join a cluster |
| `MeanShiftInstanceLoss`, `src/dl_techniques/losses/mean_shift_instance_loss.py` | Per row over valid points and ordered pairs `i != j`: pull = mean `||z_i - z_j||^2` over same-id pairs, push = mean `max(0, margin - ||z_i - z_j||)^2` over different-id pairs, `loss = pull_weight * pull + push_weight * push`, shape `(B,)`; distances are centred on the valid points; `y_true` and `y_pred` may be tensors, arrays or variables |

| Component | Constructor knobs that matter | Numerics and the fully masked row |
|---|---|---|
| Shared operations | `kernel` (`gaussian` or `vmf`), `bandwidth` (scalar, `(B,1,1)`, `(B,H,1,1)` or per query), `step_size` in (0, 2), `blurring`, `mask` (1 = keep) | Logits, softmax and density run in at least float32 (`accumulation_dtype`), result cast back to the input dtype. Masked keys have exactly zero weight, by `where` selection and a row maximum over kept logits only (no sentinel, 2.6), at any bandwidth. A row with no kept key gets uniform weights, so its output is the plain mean of `v`; `kde_log_density` treats every key as kept for such a row |
| `ModePooling` | `num_steps=5`, `bandwidth="auto"` or a float, `bandwidth_scale`, `learnable_bandwidth`, `num_starts`, `kernel`, `exclude_positions`, `return_density` | Same accumulation rule. A row with no kept token, including one emptied by `exclude_positions`, returns the plain mean over all positions. `bandwidth="auto"` is the median pair distance of the kept tokens, floored at 1e-3 times the RMS radius of the kept tokens about their masked mean (and at the absolute 1e-6); a float bandwidth keeps only the absolute 1e-6 floor. The data floor matters when most tokens are identical: the median is then 0, and without it the density that ranks several starts is float32 rounding noise and a lone outlier can win (section 5 gives the sweep, about 1e-4 is the threshold and 1e-3 leaves a factor of 10). `bandwidth="auto"` sorts `T^2` distances per sample and dominates the cost (section 6); a float bandwidth avoids it. A constant offset of the tokens moves the mode by the offset only |
| `get_mixture_modes` | `num_steps=100`, `merge_tol=1e-3` | Float32 or wider (float16 is promoted); sigma floored at 1e-6; Python-int loop, no early stop; buffers `[B, M, M, D]`. `is_unique` is False for a start that matches any earlier start within `merge_tol`. Starts are offset by `1e-3` sigma times a per-component, per-dimension multiplier (`num_steps > 0`; `num_steps=0` returns the exact means). Near a 2 sigma separation convergence is slow: use more steps or a larger `merge_tol` |
| `MeanShiftAttention` | `dim`, `num_heads=8`, `head_dim`, `num_steps=1`, `blurring`, `kernel`, `bandwidth_mode`, `bandwidth_init` (default `head_dim ** 0.25`), `min_bandwidth`, `max_bandwidth`, `step_size`, `learnable_step_size`, `dropout_rate`, `use_bias=False` | Float32 softmax under `mixed_float16`. Mask is a keep mask of rank 2, 3 or 4, applied through `apply_attention_mask` after the logits of each row are shifted by the row's kept maximum and the masked entries are parked at 0, so the best kept logit is 0 and the fixed mask constant always loses: a masked key has exactly zero weight at any bandwidth, also when the kept keys are hundreds of bandwidths away. A query row with no kept key gets output zeros and weights zeros (not the mean), and under blurring it holds still at every step, so it never leaks a forbidden key into another row. The Gaussian centre is the masked mean of the kept keys under a rank-2 padding mask, and the mean of the keys that at least one query row keeps under a rank-3 mask or, per head, a rank-4 mask, so a value stored at a key that no row keeps cannot set it. The default bandwidth gives a nearly uniform vMF kernel (section 5). Rank-4 input raises in `build` |
| `MeanShiftLayer` | `num_steps=5`, `bandwidth=1.0` in [0.01001, 99.9] (the clip range [1e-2, 1e2] less a relative margin of 1e-3), `learnable_bandwidth=True`, `blurring`, `step_size`, `kernel`, `use_projection`, `projection_dim`, `return_affinity` | Iteration in at least float32. A masked point is never a key; its own shifted value is computed and finite, callers ignore it (`assign_clusters` takes the same mask and labels it -1). A batch row with no kept point gets uniform weights, so the plain mean |
| `MeanShiftInstanceLoss` | `bandwidth=1.0`, `margin` (default `2 * bandwidth`), `pull_weight`, `push_weight`, `max_points=1024`, `ignore_label=-1` | Float32 or wider. Points with the ignore id take no part. A row with one instance or no valid point costs 0, not NaN. A row over `max_points` is subsampled at evenly spaced indices (static `N` only), so the value depends on whether `N` is static once `N > max_points`: an instance whose only point falls between the indices drops out in the static case |

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
| Tiny bandwidth with a mask: the nearest kept key wins, masked weight is exactly zero, subset equivalence, finite density and gradients | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_w1_a_tiny_bandwidth_keeps_the_nearest_kept_key_and_the_masked_weight_is_zero`, `test_w1_a_masked_step_equals_the_step_over_the_kept_subset_at_a_tiny_bandwidth`, `test_w1_the_kde_density_at_a_tiny_bandwidth_is_finite_and_equals_the_subset_density`, `test_w1_a_fully_masked_row_stays_uniform_at_a_tiny_bandwidth`, `test_w1_float16_inputs_at_a_tiny_bandwidth_are_finite_with_exact_zero_masked_weights`, `test_w1_gradients_through_a_masked_step_are_finite` |
| Translation: every op is equivariant to a constant offset in float32 up to 1000, RED proof without the centring, near-coincident pair far from the origin, junk at a masked key does not set the centre, logits up to a per-query constant | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_w2_every_op_is_translation_equivariant_in_float32`, `test_w2_red_proof_without_the_centring_the_translation_check_fails_at_1000`, `test_w2_pairwise_sq_dist_resolves_a_near_coincident_pair_far_from_the_origin`, `test_w2_a_junk_value_in_a_masked_key_does_not_set_the_centre`, `test_w2_the_logits_are_defined_up_to_a_per_query_constant_and_the_weights_are_not_affected` |
| Tensor operands: a numpy array or a variable as the second operand | `tests/test_layers/test_mixtures/test_mean_shift_ops.py` | `test_n12_every_second_operand_may_be_an_array_or_a_variable` |
| `ModePooling` robustness: mode stays on the big blob, mean is pulled | `tests/test_layers/test_mode_pooling.py` | `TestRobustness::test_the_mode_stays_on_the_big_blob_while_the_mean_is_pulled`, `TestRobustness::test_red_proof_the_plain_mean_fails_the_robustness_body` |
| `ModePooling` multi-start finds the heavier mode | `tests/test_layers/test_mode_pooling.py` | `TestMultiStart::test_one_start_returns_the_light_blob_and_four_starts_the_heavy_one`, `TestMultiStart::test_the_density_follows_the_selected_start` |
| `ModePooling` masks, exclusions, fully masked row, order invariance at one start | `tests/test_layers/test_mode_pooling.py` | `TestMasking::test_masked_tokens_have_no_influence`, `TestMasking::test_a_fully_masked_row_is_the_plain_mean_and_finite`, `test_one_start_is_invariant_to_token_order` |
| `ModePooling` precision, graph, factory | `tests/test_layers/test_mode_pooling.py` | `TestPrecisionAndGraphs::test_mixed_float16_is_finite_and_close_to_float32`, `TestPrecisionAndGraphs::test_a_graph_with_unknown_batch_and_length_matches_eager_at_two_lengths`, `TestShapesAndFactory::test_the_factory_builds_mode_and_lists_it` |
| `ModePooling` translation, tiny and float bandwidths with a mask, duplicated tokens, finite gradients | `tests/test_layers/test_mode_pooling.py` | `TestTranslationAndSmallBandwidths::test_the_mode_moves_with_a_constant_offset_of_the_tokens`, `TestTranslationAndSmallBandwidths::test_a_tiny_bandwidth_with_a_mask_returns_the_duplicated_point_not_zeros`, `TestTranslationAndSmallBandwidths::test_a_float_bandwidth_far_below_the_data_scale_with_a_mask_stays_on_the_data`, `TestTranslationAndSmallBandwidths::test_mostly_identical_tokens_pool_to_that_token`, `TestTranslationAndSmallBandwidths::test_the_gradients_stay_finite_with_identical_tokens_and_a_mask`, `TestTranslationAndSmallBandwidths::test_the_gradient_is_finite_for_a_shifted_token_set`, `TestTranslationAndSmallBandwidths::test_a_fully_masked_row_stays_the_plain_mean_at_a_tiny_bandwidth` |
| `ModePooling` auto bandwidth floor follows the data scale: random and named duplicated values pool to the duplicated point at 1, 2 and 4 starts with and without a mask, ordinary data never reaches the floor, a float bandwidth is not floored by the data scale, RED proof with the old absolute floor | `tests/test_layers/test_mode_pooling.py` | `TestTheAutoBandwidthFloorFollowsTheDataScale::test_random_duplicated_values_pool_to_the_duplicated_point`, `TestTheAutoBandwidthFloorFollowsTheDataScale::test_named_duplicated_values_with_a_mask_pool_to_the_duplicated_point`, `TestTheAutoBandwidthFloorFollowsTheDataScale::test_the_plan_layout_with_a_non_representable_value_pools_to_it`, `TestTheAutoBandwidthFloorFollowsTheDataScale::test_ordinary_data_does_not_reach_the_floor`, `TestTheAutoBandwidthFloorFollowsTheDataScale::test_a_float_bandwidth_is_not_floored_by_the_data_scale`, `TestTheAutoBandwidthFloorFollowsTheDataScale::test_the_check_fails_when_the_floor_is_the_old_absolute_constant` |
| MDN: single component, equal separated components, bimodal contrast with the mean | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheModesAreTheMixtureModes::test_a_single_component_has_its_mean_as_mode`, `TestTheModesAreTheMixtureModes::test_the_mode_estimate_of_two_separated_components_has_two_unique_modes`, `TestTheContrastCanFail::test_the_point_estimate_of_the_bimodal_case_fails_the_not_the_mean_check`, `TestTheContrastCanFail::test_the_mode_estimate_passes_the_same_check` |
| MDN: overlap vs float64 grid, unequal weights, float64 fixed point | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheModesAreTheMixtureModes::test_overlapping_components_have_one_mode_equal_to_the_grid_maximum`, `TestTheModesAreTheMixtureModes::test_unequal_weights_select_the_heavier_component`, `TestTheModesAreTheMixtureModes::test_the_modes_match_the_float64_fixed_point` |
| MDN: log-density oracle, uniqueness, tracing | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheLogDensity::test_the_log_density_matches_the_float64_oracle_at_the_returned_modes`, `TestUniqueness::test_identical_components_leave_exactly_one_unique_start`, `TestTracing::test_an_unknown_batch_size_traces_and_matches_eager` |
| MDN: a start at a density minimum leaves it (RED proof without the offset), the estimate is the highest-density start, zero steps keep the means | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestStationaryPointsReachedFromEachComponentMean::test_a_start_at_a_density_minimum_leaves_it`, `TestStationaryPointsReachedFromEachComponentMean::test_red_without_the_start_offset_the_minimum_is_returned_as_a_unique_mode`, `TestStationaryPointsReachedFromEachComponentMean::test_the_mode_estimate_is_the_highest_density_start`, `TestStationaryPointsReachedFromEachComponentMean::test_zero_steps_keep_the_means_without_the_offset` |
| MDN start offsets: the X-shaped mixture (components at (-1, 1), (1, -1), (0, 0)) leaves its swap-symmetric saddle, the saddle is a stationary point, 40 swap-symmetric mixtures return no saddle or minimum as a unique mode, every component and dimension gets its own nonzero offset, RED proof with one shared offset direction, the offset traces with an unknown batch | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheStartOffsetBreaksCoordinateSymmetry::test_the_x_shaped_mixture_centre_start_does_not_stop_on_its_saddle`, `TestTheStartOffsetBreaksCoordinateSymmetry::test_the_x_shaped_saddle_is_one_of_the_stationary_points`, `TestTheStartOffsetBreaksCoordinateSymmetry::test_swap_symmetric_mixtures_return_no_saddle_or_minimum_as_a_unique_mode`, `TestTheStartOffsetBreaksCoordinateSymmetry::test_every_component_and_dimension_gets_its_own_nonzero_offset`, `TestTheStartOffsetBreaksCoordinateSymmetry::test_red_with_one_shared_offset_direction_the_x_shaped_saddle_is_returned`, `TestTheStartOffsetBreaksCoordinateSymmetry::test_the_offset_survives_an_unknown_batch_under_tf_function` |
| MDN near-merge: 1.9 and 1.99 sigma within 0.1 of the true mode after 100 steps, the documented over-count, one unique mode after 3000 steps, `merge_tol`, exactly 2 sigma, separated contrast | `tests/test_layers/test_statistics/test_mdn_modes.py` | `TestTheMergePointConvergesSlowly::test_after_one_hundred_steps_the_starts_are_within_a_tenth_of_the_true_mode`, `TestTheMergePointConvergesSlowly::test_one_hundred_steps_still_report_two_unique_modes_at_1_99_sigma`, `TestTheMergePointConvergesSlowly::test_many_steps_reach_the_single_true_mode`, `TestTheMergePointConvergesSlowly::test_a_larger_merge_tol_merges_them_at_the_documented_step_count`, `TestTheMergePointConvergesSlowly::test_at_exactly_two_sigma_the_convergence_is_still_slow`, `TestTheMergePointConvergesSlowly::test_well_separated_components_stay_two_modes_under_the_larger_merge_tol` |
| `MeanShiftAttention`: vMF is softmax attention, Gaussian is `mean_shift_step`, default bandwidth gives scaled dot-product logits minus the key-norm bias (checks `vmf`, `step`, `default`) | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestIdentities::test_the_identity_holds`, `TestIdentities::test_the_identity_fails_on_an_injected_defect`, `TestIdentities::test_a_fixed_step_size_damps_the_shift` |
| `MeanShiftAttention`: blurring vs non-blurring | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestIteration::test_blurring_and_non_blurring_agree_at_one_step`, `TestIteration::test_blurring_and_non_blurring_differ_from_two_steps`, `TestIteration::test_non_blurring_keys_stay_at_the_projected_tokens` |
| `MeanShiftAttention`: masks, zeroed rows, permutation | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestMasks::test_a_query_row_with_no_kept_key_is_zeroed`, `TestMasks::test_a_fully_masked_key_set_gives_zeros_and_stays_finite`, `TestMasks::test_permuting_the_tokens_permutes_the_output` |
| `MeanShiftAttention`: a forbidden key does not leak into other rows (RED proof when a dead row moves), dead rows have zero output, weights and gradient | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestAForbiddenKeyDoesNotLeak::test_the_other_rows_do_not_read_the_forbidden_key`, `TestAForbiddenKeyDoesNotLeak::test_the_check_fails_when_a_row_with_no_kept_key_moves`, `TestAForbiddenKeyDoesNotLeak::test_the_other_rows_equal_the_run_without_the_dead_row_and_the_forbidden_key`, `TestAForbiddenKeyDoesNotLeak::test_a_row_with_no_kept_key_has_zero_output_and_zero_weights`, `TestAForbiddenKeyDoesNotLeak::test_a_row_with_no_kept_key_receives_no_gradient_and_nothing_goes_nan` |
| `MeanShiftAttention`: junk in padded keys does not set the Gaussian centre | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestPaddedKeysDoNotSetTheCentre::test_junk_in_the_padded_keys_changes_no_unpadded_output`, `TestPaddedKeysDoNotSetTheCentre::test_the_check_fails_when_the_centre_ignores_the_padding` |
| `MeanShiftAttention`: a masked key has exactly zero weight at any bandwidth (padded key at the query in the cross form, forbidden diagonal between distant tokens, learnable bandwidth, float16), RED proofs without the shift by the kept maximum | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_masked_weights_are_exactly_zero_and_kept_weights_sum_to_one`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_a_padded_key_at_the_query_gets_no_weight_in_the_cross_form`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_a_forbidden_diagonal_gets_no_weight_between_distant_tokens`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_the_check_fails_without_the_shift`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_the_review_cases_fail_without_the_shift`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_every_gradient_is_finite_with_a_learnable_bandwidth`, `TestAMaskedKeyHasZeroWeightAtAnyBandwidth::test_mixed_float16_keeps_masked_weights_at_zero_and_the_output_finite` |
| `MeanShiftAttention`: a key that no row keeps does not set the Gaussian centre under rank-3 and rank-4 masks (per head), RED proof with the plain mean | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestAKeyNoRowKeepsDoesNotSetTheCentre::test_junk_in_a_key_forbidden_to_every_row_changes_no_kept_row`, `TestAKeyNoRowKeepsDoesNotSetTheCentre::test_the_centre_of_each_head_follows_that_head_s_mask`, `TestAKeyNoRowKeepsDoesNotSetTheCentre::test_the_check_fails_when_the_centre_ignores_the_mask` |
| `MeanShiftAttention`: a row with no kept key holds still, so other rows read its held token, RED proof when it moves | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestARowWithNoKeptKeyHoldsStill::test_the_other_rows_read_the_held_token_of_that_row`, `TestARowWithNoKeptKeyHoldsStill::test_the_check_fails_when_the_row_is_not_held` |
| `MeanShiftAttention`: bandwidth, parameter count, gradients | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestBandwidth::test_the_softplus_inverse_round_trips`, `TestBandwidth::test_the_effective_bandwidth_starts_at_the_default`, `TestSerialization::test_fewer_parameters_than_the_multi_head_attention_of_the_same_width`, `TestGradients::test_every_gradient_is_finite_and_the_weights_are_live` |
| `MeanShiftAttention`: precision, two lengths from one traced function, factory | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `TestPrecision::test_one_concrete_function_serves_two_lengths`, `TestFactory::test_the_key_is_registered_and_listed` |
| `MeanShiftAttention` inside `TransformerLayer` | `tests/test_layers/test_attention/test_mean_shift_attention.py` | `TestTransformerLayer::test_the_block_builds_and_runs`, `TestTransformerLayer::test_a_padding_mask_changes_nothing_for_the_unpadded_rows` |
| `TransformerLayer` branch and `DINOv2` supported set | `tests/test_layers/test_transformers/test_transformer.py`, `tests/test_models/test_dino/test_the_dinov2_block_draws_each_attention_projection_independently.py` | `TestTransformerLayerExtendedAttention::test_construct_and_forward`, `TestTheSupportedSet::test_the_supported_types_are_exactly_these` |
| `MeanShiftLayer` equals the ops and the float64 oracle | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestIdentities::test_the_layer_equals_the_ops`, `TestIdentities::test_one_step_equals_the_float64_oracle`, `TestIdentities::test_three_steps_equal_the_float64_oracle` |
| `MeanShiftLayer` masks, gradients, liveness | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestMasks::test_a_fully_masked_batch_row_stays_finite`, `TestGradients::test_gradients_are_finite_for_the_inputs_and_the_bandwidth`, `TestLiveness::test_blurring_contracts_the_set_more_than_not_blurring`, `TestLiveness::test_a_large_bandwidth_collapses_to_the_mean` |
| `MeanShiftLayer` bandwidth range: the clip ends and outside are refused, the margin is accepted with the clip inactive, the bandwidth gradient is nonzero across the range | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestConstructor::test_a_bandwidth_at_an_end_of_the_clip_range_or_outside_it_raises`, `TestConstructor::test_a_bandwidth_inside_the_margin_is_accepted_and_the_clip_is_inactive`, `TestGradients::test_the_bandwidth_gradient_is_nonzero_across_the_accepted_range` |
| `MeanShiftLayer` precision, two lengths, factory; `assign_clusters` | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `TestPrecision::test_one_concrete_function_serves_two_lengths`, `TestFactory::test_it_builds_with_defaults_and_no_required_parameter`, `TestAssignClusters::test_a_hand_built_two_cluster_set`, `TestAssignClusters::test_it_reads_two_labels_off_the_shifted_output_of_two_blobs` |
| `assign_clusters` mask: masked outlier gets -1, equals removal, all-masked sample, no-op mask, shape error | `tests/test_layers/test_mixtures/test_mean_shift.py` | `TestAssignClusters::test_a_masked_outlier_gets_minus_one_and_does_not_open_a_cluster`, `TestAssignClusters::test_every_mask_dtype_is_read_as_a_keep_mask`, `TestAssignClusters::test_the_unmasked_points_are_labelled_as_if_the_masked_ones_were_removed`, `TestAssignClusters::test_a_sample_with_no_kept_point_is_all_minus_one`, `TestAssignClusters::test_a_mask_of_ones_changes_nothing`, `TestAssignClusters::test_a_mask_whose_shape_does_not_match_the_leading_axes_raises` |
| `MeanShiftInstanceLoss`: zero loss when separated, hand-computed value, oracle | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_zero_loss_on_perfectly_separated_clusters`, `test_the_tiny_case_matches_the_hand_computed_value`, `test_matches_the_numpy_oracle_with_ignored_points_and_weights` |
| `MeanShiftInstanceLoss`: `sample_weight` arm, degenerate rows, subsampling | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_zeroing_a_sample_weight_equals_dropping_the_row`, `test_degenerate_rows_cost_a_finite_amount_with_a_finite_gradient`, `test_the_subsample_is_deterministic_and_equals_the_evenly_spaced_subset` |
| `MeanShiftInstanceLoss`: a variable, an array and a tensor give the same loss; two close points of different instances push the same at offsets 0 to 1000 (RED proof with an uncentred distance) | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_a_variable_a_numpy_array_and_a_tensor_give_the_same_loss`, `test_two_close_points_of_different_instances_push_the_same_at_any_offset`, `test_the_value_and_gradient_at_every_offset_agree_with_the_offset_zero_result`, `test_red_an_uncentred_distance_loses_the_push_at_a_large_offset` |
| `MeanShiftInstanceLoss`: static and unknown `N` agree up to `max_points` and differ above it | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_static_and_unknown_point_counts_agree_when_n_is_at_most_max_points`, `test_a_static_count_above_max_points_drops_a_lone_off_stride_instance_and_an_unknown_count_does_not` |
| End to end: trained embedding, `MeanShiftLayer` and `assign_clusters` recover 3 blobs | `tests/test_losses/test_mean_shift_instance_loss.py` | `test_end_to_end_three_blobs_are_recovered_by_a_trained_embedding_and_mean_shift` |

## 5. Measured numbers

CPU, float32 unless stated. Each row names the test whose setup was used; rows marked "script" were measured by a one-off script that imports that test file's helpers and prints the quantity, rows marked "test comment" are the value recorded in a comment of that test.

| Quantity | Value | Source |
|---|---|---|
| `ModePooling`, 70 inliers (std 0.1) at the origin and 30 outliers around 5 on axis 0 (std 0.5), `D=8`, `num_steps=10`: distance from the blob centre | mean 1.47; mode 0.0024 (`bandwidth=0.5`), 0.0050 (`bandwidth="auto"`); the test bound is 0.01 | script, setup of `TestRobustness::test_the_mode_stays_on_the_big_blob_while_the_mean_is_pulled` in `tests/test_layers/test_mode_pooling.py` |
| Parameters at `dim=64`, 8 heads, no bias | `MeanShiftAttention` 8200 (two 64x64 kernels and 8 raw bandwidths); `MultiHeadAttention` 16384 (four 64x64 kernels) | script (`count_params()` after `build`), comparison asserted by `TestSerialization::test_fewer_parameters_than_the_multi_head_attention_of_the_same_width` in `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| One step vs float64 oracle, max abs error of the output | float32 3.1e-7, float64 1.3e-15 | script, setup of `test_2_one_step_is_softmax_with_key_bias_times_x` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` (seeds 20 and 21, `h=0.9`); the test bounds are 1e-5 and 1e-10 |
| `max abs (m(x) - x - h^2 grad log p_hat)`, with `max abs shift` 2.7 | float32 3.3e-7, float64 1.2e-15 | script, setup of `test_4_shift_equals_h_squared_times_density_gradient` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` (seeds 5 and 6, `h=0.7`) |
| float16 vs float32, max abs difference, ops after 1 and 3 steps | 3.0e-3 and 5.2e-3; bound 1e-2 | script, setup of `test_9_float16_inputs_are_finite_and_close_to_float32` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` |
| float16 vs float32, `ModePooling` (`num_starts=3`, with a mask) | pooled 2.4e-4, log density 4.7e-3; bounds 1e-2 and 2e-2 | script, setup of `TestPrecisionAndGraphs::test_mixed_float16_is_finite_and_close_to_float32` in `tests/test_layers/test_mode_pooling.py` |
| float16 vs float32, `MeanShiftAttention` after 3 steps | 1.5e-3 (Gaussian), 4.4e-4 (vMF); bound 0.5 percent of the output scale | test comment in `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| float16 vs float32, `MeanShiftLayer` after 3 steps | 1.3e-3 (Gaussian), 3.0e-4 (vMF) | test comment in `TestPrecision::test_mixed_float16_stays_finite_and_close_to_float32`, `tests/test_layers/test_mixtures/test_mean_shift.py` |
| `MeanShiftAttention` gradients of `sum(out^2)`, the three bandwidth modes, `num_steps` 1, 3, 8, both kernels | every gradient finite. Max abs gradient of the two projection kernels at default bandwidth: Gaussian 29 to 59, vMF 9.3 to 10. Raw-bandwidth gradient (the bias of the bandwidth projection for `per_token`) at `num_steps <= 3`: Gaussian 74 to 130, vMF 0.09 to 0.85 | script, helper `_gradients` and setup of `TestGradients::test_every_gradient_is_finite_and_the_weights_are_live` in `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| Same, bandwidth pinned at `min_bandwidth` | all finite; raw-bandwidth gradient at `num_steps <= 3` is 2e-11 to 1.4e-10 (Gaussian) and 6e-21 to 1.3e-20 (vMF), nonzero but vanishing | same script and test |
| 3 blobs, MLP embedding + `MeanShiftLayer(num_steps=3, bandwidth=1.0)` + `MeanShiftInstanceLoss`, blob scale 0.25, `merge_radius=0.5`, 30 epochs, 4 held-out samples | adjusted Rand index 1.0 on every sample and 3 clusters, for seeds 0, 1 and 2; untrained: 1 cluster, index 0.0; the test pins seeds 0 and 1 with a bound of 0.9 | script (seed 2 added), `test_end_to_end_three_blobs_are_recovered_by_a_trained_embedding_and_mean_shift` in `tests/test_losses/test_mean_shift_instance_loss.py` |
| `ModePooling` (`num_steps=10`), the 70 inliers (std 0.05) and 30 outliers (std 0.2, around 2 on axis 0) of `_grid_outlier_data`, `D=8`, everything shifted by a constant 1000 and the shift removed afterwards: mode error against the unshifted run | 0.130 before the centring (2.6 blob standard deviations); after it 2.9e-5 with `bandwidth="auto"` and 2.0e-5 with `bandwidth=0.1` | script, setup of `TestTranslationAndSmallBandwidths::test_the_mode_moves_with_a_constant_offset_of_the_tokens` in `tests/test_layers/test_mode_pooling.py` |
| Median heuristic on the same tokens shifted by 1000 (true value 0.561) | 1.0 before, 0.561 after | same script |
| `ModePooling` with 85 identical tokens, 15 far tokens, 5 positions masked, `bandwidth="auto"` (the median distance is 0, so `h` sits at the 1e-6 floor) | `[0, 0, 0]` before the sentinel-free masking, the duplicated point after (the data floor of section 6 also applies to this layout now) | script, setup of `TestTranslationAndSmallBandwidths::test_a_tiny_bandwidth_with_a_mask_returns_the_duplicated_point_not_zeros` in `tests/test_layers/test_mode_pooling.py` |
| `ModePooling`, `bandwidth="auto"`, 3600 runs of 100 tokens with 75 to 95 percent identical tokens plus far loose tokens, 5 positions masked in half of them: `_random_duplicate_batch(rows=40, seed=100 + i)` for `i` in 0 to 4 with a random, 1.1 and 0.3141 duplicated value, `num_starts` 1, 2 and 4, `num_steps=5`; runs more than 1e-3 off the duplicated point, by the factor `c` of the data floor | `c=0`: 55; `5e-5`: 27; `1e-4`: 0; `1e-3` (the default): 0; worst error with the floor 1.3e-6 | script, helper `_random_duplicate_batch` in `tests/test_layers/test_mode_pooling.py`, `_BANDWIDTH_DATA_FLOOR` set per run |
| `MeanShiftAttention`, Gaussian, `bandwidth_init=1.0`, `blurring=True`, `_set_kernels(_make(num_steps=n, ...))`, `_tokens(1, 6, 71)`, `_pairwise_mask(rank)` (key 5 forbidden to every row, row 0 keeps nothing); 10 added to token 5: largest change of rows 1 to 4 | exactly 0.0 at `num_steps` 1, 2 and 3 for a rank-3 and a rank-4 mask (vMF at 0.5 as well), on two runs. Any nonzero value here would be float32 rounding (1e-7 scale); the forbidden key now enters neither the weights nor the centre, so none appears. With every row allowed to move (the monkeypatch of `test_the_check_fails_when_a_row_with_no_kept_key_moves`) the change is 0 at 1 step and 0.110 at 2 steps | script, setup of `TestAForbiddenKeyDoesNotLeak::test_the_other_rows_do_not_read_the_forbidden_key` in `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| `MeanShiftAttention`, identity projection, `bandwidth_init=0.011`, `num_steps=1`: cross form with one query at the origin, kept keys 1000 and 1001 away and a padded key at the origin; self form with the diagonal forbidden and tokens at 0, 500 and -500 | with the shift by the kept maximum: weights `[1, 0, 0]` (padded key exactly 0) and diagonal weights `[0, 0, 0]`; with the shift removed (the monkeypatch of `test_the_review_cases_fail_without_the_shift`): `[0, 0, 1]` (all the weight on the padded key) and diagonal weights `[1, 1, 1]` | script, helpers `_check_the_cross_form_padded_key_has_zero_weight` and `_check_a_forbidden_diagonal_has_zero_weight` of `tests/test_layers/test_attention/test_mean_shift_attention.py` |
| `MeanShiftAttention` default bandwidth, `dim=64`, 8 heads, 32 tokens: largest attention weight | vMF 0.047 against a uniform 0.031 (nearly uniform); Gaussian 0.72 | docstring of `MeanShiftAttention` (`bandwidth_init`) |
| `MeanShiftLayer(bandwidth=b, num_steps=3)`, gradient of `sum(layer(x) * c)` with respect to `log_bandwidth`, `x = _points(2, 8, 25)`, `c = _points(2, 8, 26)`, at `b` = 0.011, 0.1, 1, 10, 99 | -0.00985, -0.000119, -27.45, -0.04296, -0.000452 (all nonzero, identical on two runs). With `log_bandwidth` forced to `log(0.01)` in float32 the gradient is -0.0119 (exp of it is the clip bound, where the clip passes the gradient); forced to `log(100)` it is exactly 0 (the clip is active). Which side of the end the float32 rounding lands on decides it, which is why the constructor refuses both ends | script, helpers `_make` and `_points` and the loss of `TestGradients::test_the_bandwidth_gradient_is_nonzero_across_the_accepted_range` in `tests/test_layers/test_mixtures/test_mean_shift.py` |
| `MeanShiftLayer(num_steps=5, blurring=True)` followed by `MeanShiftInstanceLoss()` on 9 points in 2 dimensions: three blobs of three points (std 0.1, from `default_rng(0)`) at (0, 0), (3, 0) and (0, 3), one instance id per blob; spread = largest absolute deviation of an output from the mean of the outputs | 1.2e-7 at bandwidth 20 and at 99 (the blurring set collapsed to one point), loss 4.0, gradient with respect to `log_bandwidth` 0 | script on the helper `_make` of `tests/test_layers/test_mixtures/test_mean_shift.py` |
| `MeanShiftInstanceLoss`, two points of different instances 1e-2 apart: value and push gradient at offsets 0, 10, 100, 1000 | value 3.9601 and gradient 3.98 at every offset, within 5e-4 (the pair at 1e-3 and offset 10 gave exactly 4.0 and gradient 0 before the centring) | script, setup of `test_two_close_points_of_different_instances_push_the_same_at_any_offset` in `tests/test_losses/test_mean_shift_instance_loss.py` |
| `MeanShiftInstanceLoss`, `max_points=4`, `N=5`, the fifth point the only one of its instance | 1.6292 for a static `N`, 1.3340 for an unknown `N` (the full-data value) | script, setup of `test_a_static_count_above_max_points_drops_a_lone_off_stride_instance_and_an_unknown_count_does_not` in `tests/test_losses/test_mean_shift_instance_loss.py` |
| `get_mixture_modes`, components at -1, 0, +1 (sigma 0.5, weights 0.45, 0.1, 0.45), 100 steps | the middle start returned 0.0 (a density minimum, Hessian of `log p` +4.79) without the offset, and -0.9658 with it (-0.6543 after 10 steps; the sign is the sign of its offset multiplier, the outer starts end at -0.9658 and +0.9658) | script, setup of `TestStationaryPointsReachedFromEachComponentMean::test_a_start_at_a_density_minimum_leaves_it` in `tests/test_layers/test_statistics/test_mdn_modes.py` |
| `get_mixture_modes`, X-shaped mixture: components at (-1, 1), (1, -1), (0, 0), sigma 0.5, weights 0.45, 0.45, 0.1, 300 steps | the centre start ends at (-0.9958, 0.9958), the end of an arm, and `is_unique` is `[True, True, False]`; with one offset direction shared by all components it stopped at the saddle (0, 0) as a unique mode (`test_red_with_one_shared_offset_direction_the_x_shaped_saddle_is_returned` pins it); `num_steps=0` returns the means exactly | script, setup of `TestTheStartOffsetBreaksCoordinateSymmetry` in `tests/test_layers/test_statistics/test_mdn_modes.py` |
| `get_mixture_modes`, two equal unit components `s` sigma apart (true mode 0 for `s <= 2`): symmetric mode position `m` after 100, 1000, 3000 steps | `s=1.9`: 1.7e-5 at 100; `s=1.99`: 0.0669, 7.5e-6, 1.5e-14; `s=2.0`: 0.1212, 0.0387, 0.0224; `s=2.5`: 1.0997 | script, setup of `TestTheMergePointConvergesSlowly` in `tests/test_layers/test_statistics/test_mdn_modes.py` |
| CPU eager cost of one fixed-bandwidth step, `B=8`, `T=2048`, `D=64` | 9.1 ms with the centring, 4.7 ms before | script on `mean_shift_step` |

## 6. Parameter and numeric choices

Bandwidth parametrisations:

| Layer | Parametrisation |
|---|---|
| `MeanShiftAttention`, `bandwidth_mode="global"` | one raw scalar, `h = clip(min + softplus(raw), min, max)` |
| `MeanShiftAttention`, `"per_head"` (default) | one raw value per head, same map |
| `MeanShiftAttention`, `"per_token"` | a `Dense(num_heads)` on the projected tokens with zero kernel and bias set to the raw initial value, so `h` starts uniform and varies per head and query token |
| `MeanShiftLayer` | one scalar `log h` (weight created with `autocast=False`), `h = clip(exp(log h), 1e-2, 1e2)`; the constructor accepts only values inside the clip range with a relative margin of 1e-3, that is [0.01001, 99.9]; a fixed bandwidth creates no weight |
| `ModePooling` | `h` from `bandwidth` (a float, or `"auto"` for the median distance), times `bandwidth_scale`, times `exp(log_scale)` when `learnable_bandwidth` (`log_scale` starts at 0); floored at 1e-6, and for `"auto"` also at 1e-3 times the RMS radius of the kept tokens about their masked mean |
| `MeanShiftInstanceLoss` | `bandwidth` only sets the default `margin = 2 * bandwidth` |

- Initial value. The raw bandwidth starts at the stable inverse softplus of `bandwidth_init - min`, `y + log(-expm1(-y))`, which is exact for every `y > 0` in double precision; the default `bandwidth_init` is `head_dim ** 0.25`, the scaled dot-product temperature (`TestBandwidth::test_the_softplus_inverse_round_trips`, `tests/test_layers/test_attention/test_mean_shift_attention.py`).
- Step size. `eta` is a Python float in (0, 2). With `learnable_step_size` it is `2 * sigmoid(raw)` per head, `raw` starting at `log(eta0 / (2 - eta0))`.
- Median heuristic. The median of the off-diagonal pairwise distances among the valid points of each sample (a distance, not a squared distance), under `stop_gradient`, 1.0 for a sample with fewer than two valid points, floored at 1e-6. Distances are formed on centred points (2.6a), so the value is translation-safe: 0.561 at an offset of 1000 against 1.0 before. It uses a sort and a flat-row `take`, so it works with an unknown `N`. When most token pairs are far apart (many outliers scattered in different directions) the median is large and the mode drifts toward the mean; lower `bandwidth_scale` or pass a float.
- Clip gradient. Outside the clip range the gradient to the bandwidth is zero, and at an end of the range the clip is active from initialisation, so the layer refuses a bandwidth closer to an end than the relative margin 1e-3 (`TestGradients::test_the_bandwidth_gradient_is_zero_outside_the_clip_range`, `tests/test_layers/test_mixtures/test_mean_shift.py`).
- Dtype. Kernel math runs in at least float32 because a squared coordinate beyond 256 already overflows float16 (maximum 65504) and `exp` overflows float16 beyond about 11; the result is cast back to the compute dtype. `test_9_float16_with_squares_beyond_the_float16_range_stays_finite` in `tests/test_layers/test_mixtures/test_mean_shift_ops.py` uses points at 300.
- Selection. Argmax selection uses a flat-row `take`, not `take_along_axis`, and iteration counts are Python ints.

Failure modes and what counters them:

- Blurring collapse. With a wide bandwidth (about 20 and above in `MeanShiftLayer`) or many steps the blurring set contracts toward one point, an absorbing state with a vanishing training signal (`TestLiveness::test_blurring_contracts_the_set_more_than_not_blurring`, `TestLiveness::test_a_large_bandwidth_collapses_to_the_mean` in `tests/test_layers/test_mixtures/test_mean_shift.py`). Counters: `ModePooling` never blurs, `MeanShiftAttention` and `MeanShiftLayer` take `blurring=False`, `num_steps` stays small.
- Bandwidth to 0 or to infinity. `h` to 0 gives one-hot weights (output equals the input point, bandwidth gradient vanishes); `h` to infinity gives uniform weights (output is the masked mean). Counter: the clip range and the softplus floor.
- Mode bifurcations. The set of modes of a density estimate changes with `h`: modes can appear or disappear, so a mode finder's output is not a smooth function of `h` at those values. Nothing in the code tracks this and no test pins it.
- Hard selection. `ModePooling` with `num_starts > 1` picks a start by a hard argmax with no gradient, and the extra starts are tokens chosen by position, so the result depends on token order. Only `num_starts=1` is asserted order-invariant.
- Cost. Weights are `(B, H, N, N)`, so time and memory are O(N^2); `get_mixture_modes` buffers are `[B, M, M, D]`; the loss subsamples rows over `max_points` (static `N` only; with an unknown `N` it does not, so memory is unbounded there). Centring the distances roughly doubles the eager CPU time of a fixed-bandwidth step (4.7 to 9.1 ms at `B=8`, `T=2048`, `D=64`). `ModePooling(bandwidth="auto")` sorts `T^2` distances per sample and dominates its cost; CPU eager, `B=8`, `D=64`, two runs agree to a few percent:

| `T` | `bandwidth="auto"` | `bandwidth=1.0` |
|---|---|---|
| 256 | 73 ms | 35 ms |
| 1024 | 407 ms | 41 ms |
| 2048 | 1730 ms | 52 ms |

  Pass a float bandwidth for long sequences.
- Stationary points. `get_mixture_modes` returns stationary points reached from each component mean, not a guarantee of local maxima; a start leaves a minimum or a saddle because of the `1e-3` sigma offset, whose multiplier differs per component and dimension so that coordinate-symmetric mixtures leave their saddles too, and `get_mode_estimate` keeps the highest density. Near the merge point (about 2 sigma) convergence is linear and slow, so `is_unique` counts two modes where the density has one until `num_steps` or `merge_tol` is raised (`TestTheMergePointConvergesSlowly` in `tests/test_layers/test_statistics/test_mdn_modes.py`).
- Forbidden keys under blurring. A row with no kept key holds still and outputs zeros, so no forbidden key reaches another row through it. A non-transitive pairwise mask under blurring still allows two-hop reads through kept rows.
- Default vMF bandwidth. `bandwidth_init = head_dim ** 0.25` gives `kappa = 1 / sqrt(head_dim)` on unit vectors, a nearly uniform kernel (section 5); pass a smaller `bandwidth_init` for a sharper vMF attention.
- Cluster ids. `assign_clusters` is NumPy, not differentiable, order-dependent (greedy merge in index order: the same points in another order can give another labelling) and must stay outside `tf.function` and any training step. Its `mask` labels padded or masked points -1 so they cannot open a cluster.

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
