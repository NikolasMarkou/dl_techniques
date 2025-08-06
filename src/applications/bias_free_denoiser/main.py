"""
Fixed demonstration of denoiser-based prior sampling and linear inverse problems.

This version:
1. Uses float32 consistently throughout (no dtype conversion)
2. Uses [-1, +1] range for denoiser compatibility (not [0, 1])
3. Addresses numerical stability issues
"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .samplers import (
    DenoiserPriorSampler,
    LinearInverseProblemSolver
)

# ---------------------------------------------------------------------

def load_bf_denoiser() -> keras.Model:
    """
    Load a pre-trained BF-CNN denoiser.

    Returns:
        Trained BF-CNN denoiser model (operates in [-1, +1] range)
    """
    try:
        denoiser = keras.models.load_model(
            '/media/arxwn/data_fast/repositories/dl_techniques/src/results/bfunet_small_20250806_154050/inference_model.keras'
        )
        logger.info("Loaded pre-trained BF-CNN denoiser (range: [-1, +1])")
        return denoiser
    except Exception as e:
        logger.error(f"Could not load denoiser: {e}")
        raise

# ---------------------------------------------------------------------

def demonstrate_prior_sampling(denoiser: keras.Model):
    """
    Demonstrate sampling from the implicit prior with stability improvements.
    """
    logger.info("=== Demonstrating Prior Sampling ===")

    sampler = DenoiserPriorSampler(
        denoiser=denoiser,
        sigma_0=0.5,    # Reduced initial noise
        sigma_l=0.005,   # Slightly higher stopping threshold
        h0=0.005,       # Smaller step size
        beta=0.3        # Reduced noise injection
    )

    # Generate samples - single channel
    shape = (1, 64, 64, 1)

    logger.info("Generating sample from implicit prior...")
    sample, convergence_info = sampler.sample_prior(shape, seed=42)

    logger.info(f"Sample shape: {sample.shape}")
    logger.info(f"Sample range: [{tf.reduce_min(sample):.3f}, {tf.reduce_max(sample):.3f}]")
    logger.info(f"Convergence info: {len(convergence_info['iterations'])} iterations")
    logger.info(f"Final sigma: {convergence_info['sigma_values'][-1]:.6f}")

    return sample, convergence_info

# ---------------------------------------------------------------------

def demonstrate_inpainting(denoiser: keras.Model, test_image: Optional[tf.Tensor] = None):
    """
    Demonstrate image inpainting using the stable denoiser prior.
    """
    logger.info("=== Demonstrating Image Inpainting ===")

    # Create or use test image
    if test_image is None:
        test_image = create_synthetic_test_image((1, 64, 64, 1))

    # --- FINAL FIX: Use the same hyperparameters as the paper and successful SR task ---
    # The solver is robust when using the parameters specified by the authors.
    solver = LinearInverseProblemSolver(
        denoiser=denoiser,
        sigma_0=1.0,        # Correct initial noise level
        sigma_l=0.005,       # Correct final noise level
        h0=0.01,            # Correct initial step size
        beta=0.01,          # Correct noise injection parameter
        max_iterations=200  # More iterations can be helpful
    )
    # --- END OF FIX ---

    # Create damaged image (missing center region)
    damaged_image = test_image.numpy().copy()
    h, w = test_image.shape[1:3]
    mask_h, mask_w = h // 4, w // 4
    start_h, start_w = (h - mask_h) // 2, (w - mask_w) // 2

    damaged_image[0, start_h:start_h + mask_h, start_w:start_w + mask_w, :] = 0.0
    damaged_image = tf.constant(damaged_image, dtype=tf.float32)

    # Create measurements (observed pixels only)
    mask = np.ones((h, w), dtype=np.float32)
    mask[start_h:start_h + mask_h, start_w:start_w + mask_w] = 0
    observed_pixels = test_image * tf.expand_dims(tf.expand_dims(mask, 0), -1)
    measurements = tf.boolean_mask(tf.reshape(observed_pixels, [-1]),
                                   tf.reshape(mask, [-1]) == 1)

    logger.info(f"Inpainting region of size {mask_h}x{mask_w} from {len(measurements)} measurements")

    # Solve inpainting problem
    restored_image, convergence_info = solver.solve_inverse_problem(
        measurement_type='inpainting',
        measurements=tf.expand_dims(measurements, 0),
        shape=test_image.shape,
        mask_size=(mask_h, mask_w)
    )

    logger.info(f"Inpainting completed in {len(convergence_info['iterations'])} iterations")

    if convergence_info['constraint_errors']:
        final_error = convergence_info['constraint_errors'][-1]
        if not np.isnan(final_error) and not np.isinf(final_error):
            logger.info(f"Final constraint error: {final_error:.8f}")

    return test_image, damaged_image, restored_image, convergence_info

# ---------------------------------------------------------------------

def demonstrate_super_resolution(denoiser: keras.Model):
    """
    Demonstrate super-resolution with improved stability.
    """
    logger.info("=== Demonstrating Super-Resolution ===")

    # Create low-resolution test image
    low_res_shape = (1, 32, 32, 1)
    factor = 2  # Reduced factor for stability
    high_res_shape = (1, 64, 64, 1)

    # Create synthetic low-res image
    low_res_image = create_synthetic_test_image(low_res_shape)

    solver = LinearInverseProblemSolver(
        denoiser=denoiser,
        beta=0.01,
        max_iterations=100
    )

    # Flatten for measurements
    measurements = tf.reshape(low_res_image, [-1])

    logger.info(f"Super-resolving {low_res_shape[1:3]} -> {high_res_shape[1:3]} (factor={factor})")

    # Solve super-resolution problem
    high_res_image, convergence_info = solver.solve_inverse_problem(
        measurement_type='super_resolution',
        measurements=tf.expand_dims(measurements, 0),
        shape=high_res_shape,
        factor=factor
    )

    logger.info(f"Super-resolution completed in {len(convergence_info['iterations'])} iterations")

    return low_res_image, high_res_image, convergence_info

# ---------------------------------------------------------------------

def demonstrate_compressive_sensing(denoiser: keras.Model):
    """
    Demonstrate compressive sensing reconstruction with fixed dimensions.
    """
    logger.info("=== Demonstrating Compressive Sensing ===")

    # Create test image - single channel
    test_image = create_synthetic_test_image((1, 64, 64, 1))

    # Create measurements with proper dimensions
    measurement_ratio = 0.2  # Higher ratio for stability
    n_pixels = np.prod(test_image.shape[1:])
    n_measurements = int(n_pixels * measurement_ratio)

    logger.info(f"Creating {n_measurements} measurements from {n_pixels} pixels ({measurement_ratio:.1%})")

    solver = LinearInverseProblemSolver(
        denoiser=denoiser,
        beta=0.01,
        max_iterations=100
    )

    # Take measurements directly
    test_image_flat = tf.reshape(test_image, [-1])

    # Create simple random measurements
    np.random.seed(42)
    measurement_indices = np.random.choice(n_pixels, n_measurements, replace=False)
    measurements = tf.gather(test_image_flat, measurement_indices)

    logger.info(f"Reconstructing from {n_measurements}/{n_pixels} measurements")

    # Use random_pixels for simplicity
    reconstructed_image, convergence_info = solver.solve_inverse_problem(
        measurement_type='random_pixels',
        measurements=tf.expand_dims(measurements, 0),
        shape=test_image.shape,
        keep_ratio=measurement_ratio
    )

    logger.info(f"Compressive sensing completed in {len(convergence_info['iterations'])} iterations")

    # Compute reconstruction metrics
    mse = tf.reduce_mean(tf.square(test_image - reconstructed_image))
    psnr = 20 * tf.math.log(2.0 / tf.sqrt(tf.maximum(mse, 1e-8))) / tf.math.log(10.0)  # 2.0 for [-1,+1] range

    logger.info(f"Reconstruction MSE: {float(mse):.6f}")
    logger.info(f"Reconstruction PSNR: {float(psnr):.2f} dB")

    return test_image, reconstructed_image, convergence_info

# ---------------------------------------------------------------------

def create_synthetic_test_image(shape: Tuple[int, ...]) -> tf.Tensor:
    """
    Create a simple synthetic test image in [-1, +1] range.
    """
    batch, h, w, c = shape

    # Create coordinate grids
    y, x = np.mgrid[0:h, 0:w]
    y = y / h
    x = x / w

    # Create simple single-channel pattern in [0, 1] first
    image = np.zeros((h, w), dtype=np.float32)

    # Circle in top-left
    dist1 = np.sqrt((y - 0.25) ** 2 + (x - 0.25) ** 2)
    circle = (dist1 < 0.15).astype(np.float32)

    # Rectangle in bottom-right
    rect = ((y > 0.6) & (y < 0.9) & (x > 0.6) & (x < 0.9)).astype(np.float32)

    # Gradient background
    gradient = x * 0.3

    # Combine patterns
    image = gradient + circle * 0.5 + rect * 0.7

    # Add single channel dimension
    if c == 1:
        image = np.expand_dims(image, axis=-1)
    else:
        image = np.repeat(np.expand_dims(image, axis=-1), c, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Convert to tensor and normalize to [0, 1] first
    image = tf.constant(image, dtype=tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Light smoothing
    kernel = tf.ones((3, 3, c, c), dtype=tf.float32) / 9.0
    image = tf.nn.conv2d(image, kernel, strides=1, padding='SAME')
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Convert from [0, 1] to [-1, +1] range
    image = image * 2.0 - 1.0

    return tf.clip_by_value(image, -1.0, 1.0)

# ---------------------------------------------------------------------


def _plot_convergence(ax: plt.Axes, conv_info: Dict[str, Any], title: str):
    """Helper function to plot a convergence curve on a given axis."""
    if conv_info and 'sigma_values' in conv_info and conv_info['sigma_values']:
        sigma_vals = conv_info['sigma_values']
        valid_sigma = [s for s in sigma_vals if s and np.isfinite(s) and s > 0]
        if valid_sigma:
            ax.plot(valid_sigma, 'b-', linewidth=2)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Not Run', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(f'{title} Convergence', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Effective σ', fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=8)


def visualize_results(results: dict, save_path: Optional[str] = None):
    """
    Visualize the results using a robust and aesthetically pleasing layout that avoids rendering artifacts.
    """
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1.2, 0.8], height_ratios=[1, 1, 1])

    fig.suptitle("Denoiser-Based Prior Sampling and Inverse Problems", fontsize=18)

    def normalize_for_display(img):
        img_np = img.numpy() if hasattr(img, 'numpy') else img
        return (img_np + 1.0) / 2.0

    # --- Row 1: Prior Sampling ---
    ax_prior_img = fig.add_subplot(gs[0, 0])
    ax_prior_conv = fig.add_subplot(gs[0, 2:])

    if 'prior_sample' in results:
        sample = results['prior_sample'][0, :, :, 0]
        ax_prior_img.imshow(normalize_for_display(sample), cmap='gray', vmin=0, vmax=1)
        ax_prior_img.set_title(f'Prior Sample\n(range: [{tf.reduce_min(sample):.2f}, {tf.reduce_max(sample):.2f}])')
    else:
        ax_prior_img.set_title('Prior Sample')
        ax_prior_img.text(0.5, 0.5, 'Not Run', ha='center', va='center', transform=ax_prior_img.transAxes)
    ax_prior_img.axis('off')
    _plot_convergence(ax_prior_conv, results.get('convergence', {}).get('Prior Sampling'), 'Prior Sampling')

    # --- Row 2: Inpainting ---
    axes_inpainting = [fig.add_subplot(gs[1, i]) for i in range(4)]

    if 'inpainting' in results:
        titles = ['Original', 'Damaged', 'Restored']
        images = results['inpainting'][:3]
        for ax, title, img_tensor in zip(axes_inpainting, titles, images):
            ax.imshow(normalize_for_display(img_tensor[0, :, :, 0]), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
    else:
        for ax, title in zip(axes_inpainting, ['Original', 'Damaged', 'Restored']):
            ax.set_title(title)
            ax.text(0.5, 0.5, 'Not Run', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    _plot_convergence(axes_inpainting[3], results.get('convergence', {}).get('Inpainting'), 'Inpainting')

    # --- Row 3: Super-Resolution or Compressive Sensing ---
    ax_sr1 = fig.add_subplot(gs[2, 0])
    ax_sr2 = fig.add_subplot(gs[2, 1])
    ax_sr_conv = fig.add_subplot(gs[2, 2:])

    task_key, conv_key, title1, title2 = None, None, 'Image 1', 'Image 2'
    if 'super_resolution' in results:
        task_key, conv_key = 'super_resolution', 'Super-Resolution'
        title1, title2 = 'Low Resolution', 'Super-Resolved'
    elif 'compressive_sensing' in results:
        task_key, conv_key = 'compressive_sensing', 'Compressive Sensing'
        title1, title2 = 'Original', 'Reconstructed'

    if task_key:
        img1, img2 = results[task_key][:2]
        ax_sr1.imshow(normalize_for_display(img1[0, :, :, 0]), cmap='gray', vmin=0, vmax=1)
        ax_sr2.imshow(normalize_for_display(img2[0, :, :, 0]), cmap='gray', vmin=0, vmax=1)
    ax_sr1.set_title(title1);
    ax_sr1.axis('off')
    ax_sr2.set_title(title2);
    ax_sr2.axis('off')
    _plot_convergence(ax_sr_conv, results.get('convergence', {}).get(conv_key, {}), conv_key or "Task")

    # Use constrained_layout for better automatic spacing
    fig.set_layout_engine('constrained')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        # logger.info(f"Visualization saved to {save_path}")

    plt.show()

# ---------------------------------------------------------------------

def main():
    """
    Main demonstration with improved error handling and stability.
    """
    logger.info("Starting stable denoiser prior demonstrations ([-1, +1] range)...")

    # Load denoiser
    try:
        denoiser = load_bf_denoiser()
    except Exception as e:
        logger.error(f"Failed to load denoiser: {e}")
        return None

    # Compile denoiser for inference
    if not hasattr(denoiser, 'compiled') or not denoiser.compiled:
        denoiser.compile(optimizer='adam', loss='mse')

    results = {}
    convergence_info = {}

    # Demonstrate prior sampling
    try:
        sample, conv_info = demonstrate_prior_sampling(denoiser)
        results['prior_sample'] = sample
        convergence_info['Prior Sampling'] = conv_info
        logger.info("✓ Prior sampling completed successfully")
    except Exception as e:
        logger.error(f"✗ Prior sampling failed: {e}")

    # Demonstrate inpainting
    try:
        inpainting_results = demonstrate_inpainting(denoiser)
        results['inpainting'] = inpainting_results[:3]
        convergence_info['Inpainting'] = inpainting_results[3]
        logger.info("✓ Inpainting completed successfully")
    except Exception as e:
        logger.error(f"✗ Inpainting failed: {e}")

    # Demonstrate super-resolution
    try:
        sr_results = demonstrate_super_resolution(denoiser)
        results['super_resolution'] = sr_results[:2]
        convergence_info['Super-Resolution'] = sr_results[2]
        logger.info("✓ Super-resolution completed successfully")
    except Exception as e:
        logger.error(f"✗ Super-resolution failed: {e}")

    # Demonstrate compressive sensing
    try:
        cs_results = demonstrate_compressive_sensing(denoiser)
        results['compressive_sensing'] = cs_results[:2]
        convergence_info['Compressive Sensing'] = cs_results[2]
        logger.info("✓ Compressive sensing completed successfully")
    except Exception as e:
        logger.error(f"✗ Compressive sensing failed: {e}")

    # Store convergence info
    results['convergence'] = convergence_info

    # Visualize results
    try:
        visualize_results(results, 'stable_denoiser_prior_results.png')
        logger.info("✓ Visualization completed successfully")
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")

    logger.info("All demonstrations completed!")

    # Print summary
    logger.info("=" * 50)
    logger.info("SUMMARY:")
    for task, info in convergence_info.items():
        if info and 'iterations' in info:
            n_iter = len(info['iterations'])
            final_sigma = info['sigma_values'][-1] if info['sigma_values'] else 'N/A'
            logger.info(f"  {task}: {n_iter} iterations, final σ = {final_sigma}")
    logger.info("=" * 50)

    return results

# ---------------------------------------------------------------------

if __name__ == "__main__":
    results = main()