"""
Fixed implementation of denoiser-based prior sampling and linear inverse problem solving.

This version:
1. Uses float32 consistently throughout (no dtype conversion)
2. Uses [-1, +1] range for denoiser compatibility (not [0, 1])
3. Addresses numerical stability issues identified in the original implementation
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class DenoiserPriorSampler:
    """
    Implements the coarse-to-fine stochastic gradient ascent algorithm for sampling
    from the implicit prior embedded in a trained denoiser.

    Based on Miyasawa's result: x̂(y) = y + σ²∇_y log p(y)
    where the denoiser residual f(y) = x̂(y) - y is proportional to the gradient
    of the log probability density.

    This version works with [-1, +1] range images and uses float32 consistently.
    """

    def __init__(
            self,
            denoiser: keras.Model,
            sigma_0: float = 1.0,
            sigma_l: float = 0.005,
            h0: float = 0.01,
            beta: float = 0.5,
            max_iterations: int = 1000
    ):
        """
        Initialize the denoiser prior sampler.

        Args:
            denoiser: Trained bias-free denoiser (operates in [-1, +1] range)
            sigma_0: Initial noise standard deviation
            sigma_l: Final noise standard deviation (stopping criterion)
            h0: Initial step size parameter
            beta: Controls noise injection (0=high noise, 1=no noise)
            max_iterations: Maximum number of iterations
        """
        self.denoiser = denoiser
        self.sigma_0 = float(sigma_0)
        self.sigma_l = float(sigma_l)
        self.h0 = float(h0)
        self.beta = float(beta)
        self.max_iterations = int(max_iterations)

        logger.info(f"Initialized DenoiserPriorSampler with σ₀={sigma_0:.4f}, "
                   f"σₗ={sigma_l:.4f}, h₀={h0:.4f}, β={beta:.4f} (range: [-1, +1])")

    def _compute_denoiser_residual(self, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the denoiser residual f(y) = x̂(y) - y with numerical stability.

        This residual is proportional to the gradient of log p(y).

        Args:
            y: Noisy input tensor in [-1, +1] range

        Returns:
            Clipped denoiser residual tensor
        """
        # Ensure input is in valid range and correct dtype
        y_safe = tf.clip_by_value(tf.cast(y, tf.float32), -1.0, 1.0)

        # Get denoiser output (also in [-1, +1] range)
        x_hat = self.denoiser(y_safe, training=False)
        x_hat = tf.clip_by_value(tf.cast(x_hat, tf.float32), -1.0, 1.0)

        # Compute residual and clip to prevent numerical issues
        residual = x_hat - y_safe
        residual = tf.clip_by_value(residual, -2.0, 2.0)

        return residual

    def _adaptive_step_schedule(self, t: int) -> float:
        """
        Compute adaptive step size to accelerate convergence.

        Uses the schedule: h_t = h₀t / (1 + h₀(t-1))

        Args:
            t: Current iteration number

        Returns:
            Step size for current iteration
        """
        h_t = self.h0 * t / (1.0 + self.h0 * (t - 1))
        return min(h_t, 0.1)  # Cap maximum step size

    def _estimate_effective_noise(self, residual: tf.Tensor) -> tf.Tensor:
        """
        Estimate effective noise standard deviation with numerical stability.

        Args:
            residual: Denoiser residual tensor

        Returns:
            Estimated noise standard deviation (scalar)
        """
        # Compute variance with numerical stability
        residual_squared = tf.square(residual)
        mean_squared = tf.reduce_mean(residual_squared)

        # Clamp to prevent sqrt of negative or very small numbers
        mean_squared = tf.clip_by_value(mean_squared, 1e-8, 100.0)

        sigma_est = tf.sqrt(mean_squared)

        # Additional stability check
        sigma_est = tf.clip_by_value(sigma_est, self.sigma_l, self.sigma_0 * 2)

        return sigma_est

    def sample_prior(
            self,
            shape: Tuple[int, ...],
            seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Sample from the implicit prior using Algorithm 1 from the paper.

        Args:
            shape: Shape of the sample to generate (batch, height, width, channels)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (generated_sample, convergence_info)
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Initialize with random noise centered at 0.0 (middle of [-1, +1])
        initial_std = min(self.sigma_0 * 0.2, 0.1)
        y = tf.random.normal(shape, mean=0.0, stddev=initial_std, dtype=tf.float32)
        y = tf.clip_by_value(y, -1.0, 1.0)

        convergence_info = {
            'iterations': [],
            'sigma_values': [],
            'step_sizes': [],
            'gamma_values': [],
            'y_stats': []
        }

        sigma_prev = self.sigma_0
        t = 1

        logger.info(f"Starting prior sampling with shape {shape} (range: [-1, +1])")
        logger.info(f"Initial y stats: mean={tf.reduce_mean(y):.4f}, std={tf.math.reduce_std(y):.4f}")

        while sigma_prev > self.sigma_l and t <= self.max_iterations:
            # Compute adaptive step size
            h_t = self._adaptive_step_schedule(t)

            # Compute denoiser residual (gradient direction)
            d_t = self._compute_denoiser_residual(y)

            # Estimate effective noise standard deviation
            sigma_t = self._estimate_effective_noise(d_t)
            sigma_t_val = float(sigma_t)

            # Check for numerical issues
            if np.isnan(sigma_t_val) or np.isinf(sigma_t_val):
                logger.warning(f"Invalid sigma at iteration {t}: {sigma_t_val}")
                break

            # Compute noise injection amplitude with enhanced stability
            beta_h = self.beta * h_t
            beta_h = tf.clip_by_value(beta_h, 0.0, 0.99)

            gamma_t_squared = ((1 - beta_h) ** 2 - (1 - h_t) ** 2) * sigma_t ** 2
            gamma_t_squared = tf.maximum(gamma_t_squared, 0.0)
            gamma_t = tf.sqrt(gamma_t_squared)

            # Limit noise injection for stability
            gamma_t = tf.clip_by_value(gamma_t, 0.0, 0.1)

            # Generate random noise
            z_t = tf.random.normal(tf.shape(y), dtype=tf.float32)

            # Update equation with careful numerical handling, clip to [-1, +1]
            y_new = y + h_t * d_t + gamma_t * z_t
            y = tf.clip_by_value(y_new, -1.0, 1.0)

            # Store convergence information
            y_mean = float(tf.reduce_mean(y))
            y_std = float(tf.math.reduce_std(y))

            convergence_info['iterations'].append(t)
            convergence_info['sigma_values'].append(sigma_t_val)
            convergence_info['step_sizes'].append(float(h_t))
            convergence_info['gamma_values'].append(float(gamma_t))
            convergence_info['y_stats'].append({'mean': y_mean, 'std': y_std})

            if t % 25 == 0:
                logger.info(f"Iteration {t}: σ={sigma_t_val:.6f}, h={float(h_t):.6f}, "
                           f"y_mean={y_mean:.3f}, y_std={y_std:.3f}")

            sigma_prev = sigma_t_val
            t += 1

        logger.info(f"Sampling converged after {t - 1} iterations with final σ={sigma_prev:.6f}")

        return y, convergence_info

# ---------------------------------------------------------------------

class LinearInverseProblemSolver:
    """
    Solves linear inverse problems using the denoiser prior via constrained sampling.

    Implements Algorithm 2 from the paper for sampling from p(x|M^T x = x_c).
    This version works with [-1, +1] range images and uses float32 consistently.
    All parameters and initialization logic are adapted to this range.
    """

    def __init__(
            self,
            denoiser: keras.Model,
            sigma_0: float = 1.0,
            sigma_l: float = 0.01,
            h0: float = 0.01,
            beta: float = 0.01,  # Lower beta for inverse problems
            max_iterations: int = 1000
    ):
        """
        Initialize the linear inverse problem solver.

        Args:
            denoiser: Trained bias-free denoiser (operates in [-1, +1] range)
            sigma_0: Initial noise standard deviation (adapted to [-1, +1] range)
            sigma_l: Final noise standard deviation
            h0: Initial step size parameter
            beta: Controls noise injection (lower for inverse problems)
            max_iterations: Maximum number of iterations
        """
        self.denoiser = denoiser
        self.sigma_0 = float(sigma_0)
        self.sigma_l = float(sigma_l)
        self.h0 = float(h0)
        self.beta = float(beta)
        self.max_iterations = int(max_iterations)

        logger.info(f"Initialized LinearInverseProblemSolver with σ₀={sigma_0:.4f}, "
                    f"σₗ={sigma_l:.4f}, h₀={h0:.4f}, β={beta:.4f} (range: [-1, +1])")

    def _adaptive_step_schedule(self, t: int) -> float:
        """
        Compute adaptive step size to accelerate convergence.
        Uses the schedule: h_t = h₀t / (1 + h₀(t-1))
        This function is identical to the one in DenoiserPriorSampler to ensure consistency
        with the paper's use of this schedule for both algorithms.

        Args:
            t: Current iteration number

        Returns:
            Step size for current iteration
        """
        h_t = self.h0 * t / (1.0 + self.h0 * (t - 1))
        return min(h_t, 0.1)  # Cap maximum step size for stability

    def _create_measurement_matrices(
            self,
            measurement_type: str,
            shape: Tuple[int, ...],
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create measurement matrix M and pseudo-inverse for different inverse problems.

        FINAL CORRECTED VERSION: This version applies the most robust and numerically
        stable pseudo-inverse method for each specific task.
        """
        batch_size, height, width, channels = shape
        n_pixels = height * width * channels

        logger.info(f"Creating '{measurement_type}' measurement matrices for shape {shape}")

        if measurement_type == 'inpainting':
            mask_h, mask_w = kwargs.get('mask_size', (height // 3, width // 3))
            start_h = (height - mask_h) // 2
            start_w = (width - mask_w) // 2
            mask = np.ones((height, width, channels), dtype=np.float32)
            mask[start_h:start_h + mask_h, start_w:start_w + mask_w, :] = 0
            mask_flat = mask.flatten()
            measured_indices = np.where(mask_flat == 1)[0]
            n_measurements = len(measured_indices)
            M = np.zeros((n_measurements, n_pixels), dtype=np.float32)
            for i, idx in enumerate(measured_indices):
                M[i, idx] = 1.0

        elif measurement_type == 'random_pixels':
            keep_ratio = kwargs.get('keep_ratio', 0.1)
            n_measurements = max(1, int(n_pixels * keep_ratio))
            measured_indices = np.random.choice(n_pixels, n_measurements, replace=False)
            M = np.zeros((n_measurements, n_pixels), dtype=np.float32)
            for i, idx in enumerate(measured_indices):
                M[i, idx] = 1.0

        elif measurement_type == 'super_resolution':
            factor = kwargs.get('factor', 4)
            new_h, new_w = height // factor, width // factor
            n_measurements = new_h * new_w * channels
            M = np.zeros((n_measurements, n_pixels), dtype=np.float32)
            avg_factor = float(factor * factor)
            for c in range(channels):
                for i in range(new_h):
                    for j in range(new_w):
                        measurement_idx = c * new_h * new_w + i * new_w + j
                        for di in range(factor):
                            for dj in range(factor):
                                pixel_idx = c * height * width + (i * factor + di) * width + (j * factor + dj)
                                M[measurement_idx, pixel_idx] = 1.0 / avg_factor
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

        M = tf.constant(M, dtype=tf.float32)

        # --- FINAL FIX: Use the best pseudo-inverse method for each case ---
        if measurement_type in ['inpainting', 'random_pixels']:
            # For simple selection matrices, the pseudo-inverse is the transpose.
            # This is numerically perfect and avoids instability from regularization.
            logger.info(f"Using simple transpose for M_pinv for '{measurement_type}'.")
            M_pinv = tf.transpose(M)

        elif measurement_type == 'super_resolution':
            # For complex averaging matrices, we must compute the full pseudo-inverse.
            logger.info("Using right pseudo-inverse for super-resolution.")
            m, n = M.shape[0], M.shape[1]
            reg_strength = 1e-6
            MMT = tf.linalg.matmul(M, M, transpose_b=True)
            MMT_reg = MMT + reg_strength * tf.eye(m, dtype=tf.float32)
            MMT_inv = tf.linalg.inv(MMT_reg)
            M_pinv = tf.linalg.matmul(M, MMT_inv, transpose_a=True)
        else:
            # Fallback for any other potential type
            logger.warning("Unknown measurement type for M_pinv, using transpose as a fallback.")
            M_pinv = tf.transpose(M)

        logger.info(f"Created measurement matrix: M={M.shape}, M_pinv={M_pinv.shape}")

        return M, M_pinv

    def solve_inverse_problem(
            self,
            measurement_type: str,
            measurements: tf.Tensor,
            shape: Tuple[int, ...],
            seed: Optional[int] = None,
            **kwargs
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Solve a linear inverse problem using Algorithm 2 from the paper,
        correctly adapted for the [-1, +1] range.

        Args:
            measurement_type: Type of inverse problem ('inpainting', 'super_resolution', etc.)
            measurements: Known measurements x_c = M^T x (in [-1, +1] range)
            shape: Shape of the original signal (batch, height, width, channels)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for measurement matrix creation

        Returns:
            Tuple of (reconstructed_signal, convergence_info)
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        logger.info(f"Solving {measurement_type} inverse problem (range: [-1, +1])")
        logger.info(f"Measurements shape: {measurements.shape}, target shape: {shape}")

        # Create measurement matrices
        try:
            M, M_pinv = self._create_measurement_matrices(measurement_type, shape, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create measurement matrices: {e}")
            raise

        batch_size = shape[0]
        signal_size = np.prod(shape[1:])
        measurements_float = tf.cast(measurements[0], tf.float32)  # Assuming batch size 1 for measurements

        # --- CORRECTED INITIALIZATION for [-1, +1] range (Algorithm 2 adapted) ---
        # The paper's initialization is: y0 ~ N(0.5(I - MMT)e + M x^c, σ₀²I)
        # For [-1, +1] range:
        # 1. The midpoint '0.5' becomes '0.0'. So, `0.0 * (I - MMT)e` effectively becomes zero.
        # 2. `M x^c` (projection of known measurements) is added directly.
        # 3. Noise with stddev `sigma_0` is added.

        logger.info(f"Initializing solution with projected measurements and noise (std={self.sigma_0:.4f})...")

        # Project known measurements back to signal space: M_pinv * x^c
        y_known_flat = tf.linalg.matvec(M_pinv, measurements_float)

        # The neutral/unknown part (0.5 * (I - MMT)e) is 0 for [-1, +1] range.
        # So, the mean for initialization is simply the projected known part.
        y_initial_mean_flat = y_known_flat

        # Add initial noise with standard deviation sigma_0.
        initial_noise = tf.random.normal([signal_size], mean=0.0, stddev=self.sigma_0, dtype=tf.float32)

        # Combine mean and noise
        y_init_flat = y_initial_mean_flat + initial_noise

        # Reshape and clip to the valid range [-1, +1]
        y = tf.reshape(y_init_flat, shape)
        y = tf.clip_by_value(y, -1.0, 1.0)
        # --- END of Corrected Initialization ---

        convergence_info = {
            'iterations': [],
            'sigma_values': [],
            'constraint_errors': [],
            'step_sizes': []
        }

        sigma_prev = self.sigma_0
        t = 1

        logger.info(f"Starting constrained optimization...")

        while sigma_prev > self.sigma_l and t <= self.max_iterations:
            # --- CORRECTED Step Size: Use the adaptive schedule from the paper ---
            h_t = self._adaptive_step_schedule(t)

            # Compute denoiser residual f(y) = x_hat(y) - y
            y_safe = tf.clip_by_value(y, -1.0, 1.0)
            x_hat = self.denoiser(y_safe, training=False)
            x_hat = tf.clip_by_value(x_hat, -1.0, 1.0)  # Ensure denoiser output is clipped
            f_y = x_hat - y_safe
            f_y = tf.clip_by_value(f_y, -2.0, 2.0)  # Clip residuals to prevent extreme values

            # Flatten for matrix operations
            y_flat = tf.reshape(y, [batch_size, -1])
            f_y_flat = tf.reshape(f_y, [batch_size, -1])

            try:
                # Compute constrained gradient (d_t from Algorithm 2)
                # d_t = (I - M M_pinv)f(y) + M_pinv(x^c - M y)
                MM_T = tf.linalg.matmul(M_pinv, M)
                I_minus_MM_T = tf.eye(signal_size, dtype=tf.float32) - MM_T

                term1 = tf.linalg.matvec(I_minus_MM_T, f_y_flat[0])  # Gradient component in null space

                # Constraint satisfaction term: M_pinv(x^c - M y)
                current_measurements = tf.linalg.matvec(M, y_flat[0])
                measurement_error = measurements_float - current_measurements
                term2 = tf.linalg.matvec(M_pinv, measurement_error)  # Corrective gradient component

                d_t_flat = term1 + term2
                d_t = tf.reshape(d_t_flat, shape)
                d_t = tf.clip_by_value(d_t, -1.0, 1.0)  # Clip gradient to prevent explosions

            except Exception as e:
                logger.warning(
                    f"Constrained gradient computation failed at iteration {t}: {e}. Falling back to unconstrained gradient.")
                d_t = f_y  # Fallback to unconstrained denoiser residual

            # Estimate effective noise sigma_t = ||d_t|| / sqrt(N)
            sigma_t_squared = tf.reduce_mean(tf.square(d_t))
            sigma_t_squared = tf.clip_by_value(sigma_t_squared, 1e-8, 100.0)  # Prevent sqrt(0) or extreme values
            sigma_t = tf.sqrt(sigma_t_squared)
            sigma_t_val = float(sigma_t)

            # Compute noise injection amplitude (gamma_t) with stability
            if not (np.isnan(sigma_t_val) or np.isinf(sigma_t_val)):
                beta_h = tf.clip_by_value(self.beta * h_t, 0.0, 0.95)  # Cap beta_h for stability
                gamma_t_squared = ((1 - beta_h) ** 2 - (1 - h_t) ** 2) * sigma_t ** 2
                gamma_t = tf.sqrt(tf.maximum(gamma_t_squared, 0.0))  # Ensure non-negative
                gamma_t = tf.clip_by_value(gamma_t, 0.0, 0.05)  # Limit noise injection for inverse problems
            else:
                logger.warning(f"Invalid sigma at iteration {t}, using minimal noise.")
                gamma_t = 0.001
                sigma_t_val = sigma_prev * 0.9  # Decay sigma to avoid infinite loop

            # Update y_t = y_{t-1} + h_t * d_t + gamma_t * z_t
            z_t = tf.random.normal(tf.shape(y), mean=0.0, stddev=0.1,
                                   dtype=tf.float32)  # Standard dev 0.1 for z_t is common
            y_new = y + h_t * d_t + gamma_t * z_t
            y = tf.clip_by_value(y_new, -1.0, 1.0)  # Clip to the valid range [-1, +1]

            # Compute constraint error (for logging/monitoring)
            try:
                y_measurements = tf.linalg.matvec(M, tf.reshape(y, [-1]))
                constraint_error = tf.reduce_mean(tf.square(y_measurements - measurements_float))
                constraint_error_val = float(constraint_error)
            except Exception as e:
                logger.warning(f"Constraint error computation failed: {e}")
                constraint_error_val = np.nan

            # Store convergence information
            convergence_info['iterations'].append(t)
            convergence_info['sigma_values'].append(sigma_t_val)
            convergence_info['constraint_errors'].append(constraint_error_val)
            convergence_info['step_sizes'].append(float(h_t))

            if t % 25 == 0:
                logger.info(
                    f"Iteration {t}: σ={sigma_t_val:.6f}, constraint_error={constraint_error_val:.8f}, h={float(h_t):.6f}")

            sigma_prev = sigma_t_val
            t += 1

        logger.info(f"Inverse problem solved after {t - 1} iterations. Final σ={sigma_prev:.6f}")

        return y, convergence_info

# ---------------------------------------------------------------------

def create_denoiser_applications(denoiser: keras.Model) -> Dict[str, Any]:
    """
    Create ready-to-use applications for common inverse problems.
    Works with [-1, +1] range images.

    Args:
        denoiser: Trained BF-CNN denoiser

    Returns:
        Dictionary containing sampler and solver instances
    """
    sampler = DenoiserPriorSampler(denoiser)
    solver = LinearInverseProblemSolver(denoiser)

    def inpaint_image(image: tf.Tensor, mask_size: Tuple[int, int] = None) -> tf.Tensor:
        """Inpaint missing rectangular region."""
        if mask_size is None:
            h, w = image.shape[1:3]
            mask_size = (h // 3, w // 3)

        # Create measurements (observed pixels)
        mask = tf.ones((*image.shape[1:3], image.shape[-1]), dtype=tf.float32)
        h, w = mask_size
        start_h, start_w = (image.shape[1] - h) // 2, (image.shape[2] - w) // 2

        # Create mask updates
        mask_updates = tf.zeros((h, w, image.shape[-1]))
        indices = [[i, j, k] for i in range(start_h, start_h + h)
                  for j in range(start_w, start_w + w)
                  for k in range(image.shape[-1])]

        mask = tf.tensor_scatter_nd_update(
            mask, indices, tf.reshape(mask_updates, [-1])
        )

        # Create measurements from observed pixels
        observed_pixels = image * tf.expand_dims(mask, 0)
        measurements = tf.boolean_mask(tf.reshape(observed_pixels, [-1]),
                                     tf.reshape(mask, [-1]) == 1)

        result, _ = solver.solve_inverse_problem(
            'inpainting',
            tf.expand_dims(measurements, 0),
            image.shape,
            mask_size=mask_size
        )
        return result

    def super_resolve_image(low_res_image: tf.Tensor, factor: int = 4) -> tf.Tensor:
        """Super-resolve image by given factor."""
        # Create high-res shape
        batch, h, w, c = low_res_image.shape
        high_res_shape = (batch, h * factor, w * factor, c)

        # Flatten measurements
        measurements = tf.reshape(low_res_image, [-1])

        result, _ = solver.solve_inverse_problem(
            'super_resolution',
            tf.expand_dims(measurements, 0),
            high_res_shape,
            factor=factor
        )
        return result

    def sample_from_prior(shape: Tuple[int, ...], n_samples: int = 1) -> tf.Tensor:
        """Generate samples from the implicit prior."""
        samples = []
        for i in range(n_samples):
            sample, _ = sampler.sample_prior(shape, seed=42 + i)
            samples.append(sample)
        return tf.stack(samples, axis=0)

    return {
        'sampler': sampler,
        'solver': solver,
        'inpaint': inpaint_image,
        'super_resolve': super_resolve_image,
        'sample_prior': sample_from_prior
    }