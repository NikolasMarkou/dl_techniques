"""
Final, corrected, and refined implementation of denoiser-based prior sampling
and linear inverse problem solving, based on the paper by Kadkhodaie and Simoncelli.

This version:
1.  Faithfully implements the paper's algorithms, adapted for a [-1, +1] data range.
2.  Uses float32 consistently and includes numerous clips/clamps for numerical stability.
3.  Chooses the optimal pseudo-inverse method for each inverse problem to ensure stability.
4.  Implements an "early stopping with patience" mechanism for efficiency and to guarantee
    the best-found result is returned.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any
import tensorflow as tf

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class DenoiserPriorSampler:
    """
    Implements Algorithm 1: Coarse-to-fine stochastic gradient ascent for sampling
    from the implicit prior embedded in a trained denoiser.
    """

    def __init__(
            self,
            denoiser: keras.Model,
            sigma_0: float = 1.0,
            sigma_l: float = 0.01,
            h0: float = 0.01,
            beta: float = 0.5,
            max_iterations: int = 1000,
            patience: int = 20
    ):
        """
        Initialize the denoiser prior sampler.

        Args:
            denoiser: Trained bias-free denoiser (operates in [-1, +1] range).
            sigma_0: Initial noise standard deviation.
            sigma_l: Final noise standard deviation (stopping criterion).
            h0: Initial step size parameter.
            beta: Controls noise injection (0=high noise, 1=no noise).
            max_iterations: Maximum number of iterations.
            patience: Number of iterations to wait for sigma to improve before stopping.
        """
        self.denoiser = denoiser
        self.sigma_0 = float(sigma_0)
        self.sigma_l = float(sigma_l)
        self.h0 = float(h0)
        self.beta = float(beta)
        self.max_iterations = int(max_iterations)
        self.patience = int(patience)

        logger.info(f"Initialized DenoiserPriorSampler with σ₀={sigma_0:.4f}, "
                   f"σₗ={sigma_l:.4f}, h₀={h0:.4f}, β={beta:.4f}, patience={patience} "
                   f"(range: [-1, +1])")

    def _compute_denoiser_residual(self, y: tf.Tensor) -> tf.Tensor:
        """Compute the denoiser residual f(y) = x̂(y) - y with numerical stability."""
        y_safe = tf.clip_by_value(tf.cast(y, tf.float32), -1.0, 1.0)
        x_hat = self.denoiser(y_safe, training=False)
        x_hat = tf.clip_by_value(tf.cast(x_hat, tf.float32), -1.0, 1.0)
        return tf.clip_by_value(x_hat - y_safe, -2.0, 2.0)

    def _adaptive_step_schedule(self, t: int) -> float:
        """Compute adaptive step size h_t = h₀t / (1 + h₀(t-1))."""
        h_t = self.h0 * t / (1.0 + self.h0 * (t - 1))
        return min(h_t, 0.1)  # Cap maximum step size for stability

    def sample_prior(
            self,
            shape: Tuple[int, ...],
            seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Sample from the implicit prior using Algorithm 1, adapted for [-1, +1] range,
        with early stopping.
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Initialize from noise centered at 0.0 with full initial std dev.
        y = tf.random.normal(shape, mean=0.0, stddev=self.sigma_0, dtype=tf.float32)
        y = tf.clip_by_value(y, -1.0, 1.0)

        # Early stopping setup
        best_sigma = float('inf')
        best_y = tf.identity(y)
        patience_counter = 0

        convergence_info = {'iterations': [], 'sigma_values': []}
        sigma_prev = self.sigma_0
        t = 1

        logger.info(f"Starting prior sampling with patience={self.patience}...")

        while sigma_prev > self.sigma_l and t <= self.max_iterations:
            h_t = self._adaptive_step_schedule(t)
            d_t = self._compute_denoiser_residual(y)
            sigma_t_squared = tf.reduce_mean(tf.square(d_t))
            sigma_t = tf.sqrt(tf.clip_by_value(sigma_t_squared, 1e-10, 100.0))
            sigma_t_val = float(sigma_t)

            # --- Early stopping logic ---
            if sigma_t_val < best_sigma:
                best_sigma = sigma_t_val
                best_y = tf.identity(y)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Stopping early at iteration {t} after {self.patience} steps with no improvement.")
                break
            # --- End early stopping logic ---

            # Compute noise injection amplitude
            beta_h = tf.clip_by_value(self.beta * h_t, 0.0, 0.99)
            gamma_t_squared = ((1 - beta_h) ** 2 - (1 - h_t) ** 2) * sigma_t ** 2
            gamma_t = tf.sqrt(tf.maximum(gamma_t_squared, 0.0))
            z_t = tf.random.normal(tf.shape(y), dtype=tf.float32)

            # Update step
            y = tf.clip_by_value(y + h_t * d_t + gamma_t * z_t, -1.0, 1.0)

            convergence_info['iterations'].append(t)
            convergence_info['sigma_values'].append(sigma_t_val)

            if t % 50 == 0:
                logger.info(f"Iter {t}: σ={sigma_t_val:.6f}, Best σ={best_sigma:.6f}, Patience={patience_counter}/{self.patience}")

            sigma_prev = sigma_t_val
            t += 1

        logger.info(f"Sampling finished after {t - 1} iterations. Best σ found: {best_sigma:.6f}")
        return best_y, convergence_info


# ---------------------------------------------------------------------

class LinearInverseProblemSolver:
    """
    Solves linear inverse problems using Algorithm 2: Constrained Sampling,
    adapted for [-1, +1] range and with early stopping.
    """

    def __init__(
            self,
            denoiser: keras.Model,
            sigma_0: float = 1.0,
            sigma_l: float = 0.01,
            h0: float = 0.01,
            beta: float = 0.01,
            max_iterations: int = 1000,
            patience: int = 20
    ):
        self.denoiser = denoiser
        self.sigma_0 = float(sigma_0)
        self.sigma_l = float(sigma_l)
        self.h0 = float(h0)
        self.beta = float(beta)
        self.max_iterations = int(max_iterations)
        self.patience = int(patience)
        logger.info(f"Initialized LinearInverseProblemSolver with σ₀={sigma_0:.4f}, "
                   f"σₗ={sigma_l:.4f}, h₀={h0:.4f}, β={beta:.4f}, patience={patience} "
                   f"(range: [-1, +1])")

    def _adaptive_step_schedule(self, t: int) -> float:
        """Compute adaptive step size h_t = h₀t / (1 + h₀(t-1))."""
        h_t = self.h0 * t / (1.0 + self.h0 * (t - 1))
        return min(h_t, 0.1)

    def _compute_denoiser_residual(self, y: tf.Tensor) -> tf.Tensor:
        """Compute the denoiser residual f(y) = x̂(y) - y with numerical stability."""
        y_safe = tf.clip_by_value(tf.cast(y, tf.float32), -1.0, 1.0)
        x_hat = self.denoiser(y_safe, training=False)
        x_hat = tf.clip_by_value(tf.cast(x_hat, tf.float32), -1.0, 1.0)
        return tf.clip_by_value(x_hat - y_safe, -2.0, 2.0)

    def _create_measurement_matrices(
            self,
            measurement_type: str,
            shape: Tuple[int, ...],
            **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Creates M and M_pinv, using the most numerically stable method for each task.
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
            measured_indices = np.where(mask.flatten() == 1)[0]
            M = np.zeros((len(measured_indices), n_pixels), dtype=np.float32)
            for i, idx in enumerate(measured_indices): M[i, idx] = 1.0
        elif measurement_type == 'super_resolution':
            factor = kwargs.get('factor', 4)
            new_h, new_w = height // factor, width // factor
            n_measurements = new_h * new_w * channels
            M = np.zeros((n_measurements, n_pixels), dtype=np.float32)
            avg_factor = float(factor * factor)
            for c in range(channels):
                for i in range(new_h):
                    for j in range(new_w):
                        m_idx = c * new_h * new_w + i * new_w + j
                        for di in range(factor):
                            for dj in range(factor):
                                p_idx = c * height * width + (i * factor + di) * width + (j * factor + dj)
                                M[m_idx, p_idx] = 1.0 / avg_factor
        else: # Covers 'random_pixels' and others
            keep_ratio = kwargs.get('keep_ratio', 0.1)
            n_measurements = max(1, int(n_pixels * keep_ratio))
            measured_indices = np.random.choice(n_pixels, n_measurements, replace=False)
            M = np.zeros((n_measurements, n_pixels), dtype=np.float32)
            for i, idx in enumerate(measured_indices):
                M[i, idx] = 1.0

        M = tf.constant(M, dtype=tf.float32)

        if measurement_type in ['inpainting', 'random_pixels']:
            logger.info(f"Using simple transpose for M_pinv for '{measurement_type}'.")
            M_pinv = tf.transpose(M)
        elif measurement_type == 'super_resolution':
            logger.info("Using right pseudo-inverse for super-resolution.")
            m, n = M.shape[0], M.shape[1]
            reg_strength = 1e-6
            MMT = tf.linalg.matmul(M, M, transpose_b=True)
            MMT_reg = MMT + reg_strength * tf.eye(m, dtype=tf.float32)
            MMT_inv = tf.linalg.inv(MMT_reg)
            M_pinv = tf.linalg.matmul(M, MMT_inv, transpose_a=True)
        else:
            logger.warning("Unknown measurement type for M_pinv, using transpose fallback.")
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
        """Solves a linear inverse problem using Algorithm 2 with early stopping."""
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        M, M_pinv = self._create_measurement_matrices(measurement_type, shape, **kwargs)
        batch_size, signal_size = shape[0], np.prod(shape[1:])
        measurements_float = tf.cast(measurements[0], tf.float32)

        # Principled initialization adapted for [-1, +1] range
        logger.info(f"Initializing solution with projected measurements and noise (std={self.sigma_0:.4f})...")
        y_known_flat = tf.linalg.matvec(M_pinv, measurements_float)
        initial_noise = tf.random.normal([signal_size], mean=0.0, stddev=self.sigma_0, dtype=tf.float32)
        y_init_flat = y_known_flat + initial_noise
        y = tf.clip_by_value(tf.reshape(y_init_flat, shape), -1.0, 1.0)

        # Early stopping setup
        best_sigma = float('inf')
        best_y = tf.identity(y)
        patience_counter = 0

        convergence_info = {'iterations': [], 'sigma_values': [], 'constraint_errors': []}
        sigma_prev = self.sigma_0
        t = 1

        logger.info(f"Starting constrained optimization with patience={self.patience}...")
        while sigma_prev > self.sigma_l and t <= self.max_iterations:
            h_t = self._adaptive_step_schedule(t)
            f_y = self._compute_denoiser_residual(y)
            y_flat, f_y_flat = tf.reshape(y, [batch_size, -1]), tf.reshape(f_y, [batch_size, -1])

            # Constrained gradient (Algorithm 2)
            MM_T = tf.linalg.matmul(M_pinv, M)
            I_minus_MM_T = tf.eye(signal_size, dtype=tf.float32) - MM_T
            term1 = tf.linalg.matvec(I_minus_MM_T, f_y_flat[0])
            measurement_error = measurements_float - tf.linalg.matvec(M, y_flat[0])
            term2 = tf.linalg.matvec(M_pinv, measurement_error)
            d_t = tf.reshape(term1 + term2, shape)

            # Estimate effective noise and check for early stopping
            sigma_t_squared = tf.reduce_mean(tf.square(d_t))
            sigma_t = tf.sqrt(tf.clip_by_value(sigma_t_squared, 1e-10, 100.0))
            sigma_t_val = float(sigma_t)

            if sigma_t_val < best_sigma:
                best_sigma = sigma_t_val
                best_y = tf.identity(y)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Stopping early at iteration {t} after {self.patience} steps with no improvement.")
                break

            # Compute noise injection amplitude
            beta_h = tf.clip_by_value(self.beta * h_t, 0.0, 0.95)
            gamma_t_squared = ((1 - beta_h) ** 2 - (1 - h_t) ** 2) * sigma_t ** 2
            gamma_t = tf.sqrt(tf.maximum(gamma_t_squared, 0.0))
            z_t = tf.random.normal(tf.shape(y), dtype=tf.float32)

            # Update step
            y = tf.clip_by_value(y + h_t * d_t + gamma_t * z_t, -1.0, 1.0)

            # Log convergence
            y_measurements = tf.linalg.matvec(M, tf.reshape(y, [-1]))
            constraint_error = float(tf.reduce_mean(tf.square(y_measurements - measurements_float)))
            convergence_info['iterations'].append(t)
            convergence_info['sigma_values'].append(sigma_t_val)
            convergence_info['constraint_errors'].append(constraint_error)

            if t % 50 == 0:
                logger.info(f"Iter {t}: σ={sigma_t_val:.6f}, Best σ={best_sigma:.6f}, Err={constraint_error:.8f}, Patience={patience_counter}/{self.patience}")

            sigma_prev = sigma_t_val
            t += 1

        logger.info(f"Inverse problem solved after {t - 1} iterations. Best σ found: {best_sigma:.6f}")
        return best_y, convergence_info