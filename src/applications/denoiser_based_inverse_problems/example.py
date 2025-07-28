"""
Complete usage example showing how to use a trained BF-CNN denoiser for
prior sampling and solving inverse problems.
"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from dl_techniques.utils.logger import logger
from denoiser_prior_sampling import DenoiserPriorSampler, LinearInverseProblemSolver, create_denoiser_applications


def load_or_create_bfcnn_denoiser() -> keras.Model:
    """
    Load a pre-trained BF-CNN denoiser or create a simple one for demonstration.

    Returns:
        Trained BF-CNN denoiser model
    """
    try:
        # Try to load existing model
        denoiser = keras.models.load_model('path/to/your/bfcnn_denoiser.keras')
        logger.info("Loaded pre-trained BF-CNN denoiser")
        return denoiser
    except:
        logger.info("Creating simple BF-CNN denoiser for demonstration...")
        return create_simple_bfcnn_denoiser()


def create_simple_bfcnn_denoiser(input_shape: Tuple[int, ...] = (None, None, 3)) -> keras.Model:
    """
    Create a simplified bias-free CNN denoiser for demonstration.

    Note: For real applications, you should use a properly trained BF-CNN
    as described in the paper (20 layers, trained on various noise levels).

    Args:
        input_shape: Input shape for the model

    Returns:
        Simple bias-free denoiser model
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Simplified BF-CNN architecture (bias-free)
    for i in range(8):  # Reduced from 20 layers for demo
        x = keras.layers.Conv2D(
            64, 3, padding='same',
            use_bias=False,  # Critical: bias-free
            kernel_initializer='he_normal',
            name=f'conv_{i}'
        )(x)

        if i < 7:  # No BatchNorm on last layer
            x = keras.layers.BatchNormalization(
                center=False,  # Critical: no additive bias
                name=f'bn_{i}'
            )(x)
            x = keras.layers.ReLU(name=f'relu_{i}')(x)

    # Final layer to match input channels
    outputs = keras.layers.Conv2D(
        input_shape[-1], 3, padding='same',
        use_bias=False,
        kernel_initializer='zeros',  # Start with identity-like mapping
        name='output_conv'
    )(x)

    # Residual connection (denoiser learns noise, not clean image)
    outputs = inputs + outputs

    model = keras.Model(inputs, outputs, name='simple_bfcnn_denoiser')
    return model


def demonstrate_prior_sampling(denoiser: keras.Model):
    """
    Demonstrate sampling from the implicit prior.

    Args:
        denoiser: Trained BF-CNN denoiser
    """
    logger.info("=== Demonstrating Prior Sampling ===")

    sampler = DenoiserPriorSampler(
        denoiser=denoiser,
        sigma_0=1.0,  # Initial noise level
        sigma_l=0.01,  # Final noise level
        h0=0.01,  # Step size
        beta=0.5  # Noise injection control
    )

    # Generate samples
    shape = (1, 64, 64, 3)  # Batch, height, width, channels

    logger.info("Generating sample from implicit prior...")
    sample, convergence_info = sampler.sample_prior(shape, seed=42)

    logger.info(f"Sample shape: {sample.shape}")
    logger.info(f"Convergence info: {len(convergence_info['iterations'])} iterations")
    logger.info(f"Final sigma: {convergence_info['sigma_values'][-1]:.4f}")

    return sample, convergence_info


def demonstrate_inpainting(denoiser: keras.Model, test_image: Optional[tf.Tensor] = None):
    """
    Demonstrate image inpainting using the denoiser prior.

    Args:
        denoiser: Trained BF-CNN denoiser
        test_image: Optional test image, creates synthetic if None
    """
    logger.info("=== Demonstrating Image Inpainting ===")

    # Create or use test image
    if test_image is None:
        # Create a simple synthetic test image
        test_image = create_synthetic_test_image((1, 64, 64, 3))

    solver = LinearInverseProblemSolver(
        denoiser=denoiser,
        sigma_0=1.0,
        sigma_l=0.01,
        h0=0.01,
        beta=0.01  # Lower beta for inverse problems
    )

    # Create damaged image (missing center region)
    damaged_image = test_image.numpy().copy()
    h, w = test_image.shape[1:3]
    mask_h, mask_w = h // 3, w // 3
    start_h, start_w = (h - mask_h) // 2, (w - mask_w) // 2

    # Set center region to gray (missing)
    damaged_image[0, start_h:start_h + mask_h, start_w:start_w + mask_w, :] = 0.5
    damaged_image = tf.constant(damaged_image)

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
    logger.info(f"Final constraint error: {convergence_info['constraint_errors'][-1]:.6f}")

    return test_image, damaged_image, restored_image, convergence_info


def demonstrate_super_resolution(denoiser: keras.Model):
    """
    Demonstrate super-resolution using the denoiser prior.

    Args:
        denoiser: Trained BF-CNN denoiser
    """
    logger.info("=== Demonstrating Super-Resolution ===")

    # Create low-resolution test image
    low_res_shape = (1, 16, 16, 3)
    factor = 4
    high_res_shape = (1, 64, 64, 3)

    # Create synthetic low-res image
    low_res_image = create_synthetic_test_image(low_res_shape)

    solver = LinearInverseProblemSolver(denoiser=denoiser, beta=0.01)

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


def demonstrate_compressive_sensing(denoiser: keras.Model):
    """
    Demonstrate compressive sensing reconstruction.

    Args:
        denoiser: Trained BF-CNN denoiser
    """
    logger.info("=== Demonstrating Compressive Sensing ===")

    # Create test image
    test_image = create_synthetic_test_image((1, 64, 64, 3))

    # Create random measurements (10% of pixels)
    measurement_ratio = 0.1
    n_pixels = np.prod(test_image.shape[1:])
    n_measurements = int(n_pixels * measurement_ratio)

    # Create random orthogonal measurement matrix
    np.random.seed(42)
    M = np.random.randn(n_measurements, n_pixels).astype(np.float32)
    M, _ = np.linalg.qr(M)
    M = tf.constant(M)

    # Take measurements
    test_image_flat = tf.reshape(test_image, [-1])
    measurements = tf.linalg.matvec(M, test_image_flat)

    logger.info(f"Reconstructing from {n_measurements}/{n_pixels} measurements ({measurement_ratio:.1%})")

    solver = LinearInverseProblemSolver(denoiser=denoiser, beta=0.01)

    # Solve compressive sensing problem
    reconstructed_image, convergence_info = solver.solve_inverse_problem(
        measurement_type='compressive_sensing',
        measurements=tf.expand_dims(measurements, 0),
        shape=test_image.shape,
        measurement_ratio=measurement_ratio
    )

    logger.info(f"Compressive sensing completed in {len(convergence_info['iterations'])} iterations")

    # Compute reconstruction error
    mse = tf.reduce_mean(tf.square(test_image - reconstructed_image))
    psnr = 20 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)

    logger.info(f"Reconstruction PSNR: {float(psnr):.2f} dB")

    return test_image, reconstructed_image, convergence_info


def create_synthetic_test_image(shape: Tuple[int, ...]) -> tf.Tensor:
    """
    Create a synthetic test image with simple geometric patterns.

    Args:
        shape: Image shape (batch, height, width, channels)

    Returns:
        Synthetic test image
    """
    batch, h, w, c = shape

    # Create coordinate grids
    y, x = np.mgrid[0:h, 0:w]
    y = y / h
    x = x / w

    # Create simple patterns
    image = np.zeros((h, w, c), dtype=np.float32)

    for ch in range(c):
        if ch == 0:  # Red channel - circles
            center_y, center_x = 0.3, 0.3
            dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            image[:, :, ch] = (dist < 0.2).astype(np.float32)

        elif ch == 1:  # Green channel - stripes
            image[:, :, ch] = (np.sin(x * 10) > 0).astype(np.float32)

        else:  # Blue channel - gradient
            image[:, :, ch] = x

    # Smooth the image to make it more natural
    image = tf.constant(image)
    image = tf.nn.conv2d(
        tf.expand_dims(image, 0),
        tf.ones((3, 3, c, c)) / 9.0,
        strides=1,
        padding='SAME'
    )

    return tf.clip_by_value(image, 0.0, 1.0)


def visualize_results(results: dict, save_path: Optional[str] = None):
    """
    Visualize the results from different inverse problems.

    Args:
        results: Dictionary containing results from different demos
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Prior sampling
    if 'prior_sample' in results:
        sample = results['prior_sample'][0, :, :, :]
        for i in range(min(3, sample.shape[-1])):
            axes[0, i].imshow(sample[:, :, i], cmap='viridis')
            axes[0, i].set_title(f'Prior Sample - Channel {i}')
            axes[0, i].axis('off')

    # Row 2: Inpainting
    if 'inpainting' in results:
        original, damaged, restored = results['inpainting'][:3]

        axes[1, 0].imshow(original[0, :, :, :3])
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(damaged[0, :, :, :3])
        axes[1, 1].set_title('Damaged')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(restored[0, :, :, :3])
        axes[1, 2].set_title('Restored')
        axes[1, 2].axis('off')

    # Row 3: Super-resolution
    if 'super_resolution' in results:
        low_res, high_res = results['super_resolution'][:2]

        axes[2, 0].imshow(low_res[0, :, :, :3])
        axes[2, 0].set_title('Low Resolution')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(high_res[0, :, :, :3])
        axes[2, 1].set_title('Super-Resolved')
        axes[2, 1].axis('off')

    # Plot convergence curves
    if 'convergence' in results:
        for i, (name, conv_info) in enumerate(results['convergence'].items()):
            ax = axes[i, 3]
            ax.plot(conv_info['sigma_values'])
            ax.set_title(f'{name} - Convergence')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Effective σ')
            ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    plt.show()


def main():
    """
    Main demonstration of denoiser-based prior sampling and inverse problems.
    """
    logger.info("Starting denoiser prior demonstrations...")

    # Load or create denoiser
    denoiser = load_or_create_bfcnn_denoiser()

    # Compile denoiser if needed (for inference only)
    if not hasattr(denoiser, 'compiled'):
        denoiser.compile(optimizer='adam', loss='mse')

    results = {}
    convergence_info = {}

    # Demonstrate prior sampling
    try:
        sample, conv_info = demonstrate_prior_sampling(denoiser)
        results['prior_sample'] = sample
        convergence_info['Prior Sampling'] = conv_info
    except Exception as e:
        logger.error(f"Prior sampling failed: {e}")

    # Demonstrate inpainting
    try:
        inpainting_results = demonstrate_inpainting(denoiser)
        results['inpainting'] = inpainting_results[:3]
        convergence_info['Inpainting'] = inpainting_results[3]
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")

    # Demonstrate super-resolution
    try:
        sr_results = demonstrate_super_resolution(denoiser)
        results['super_resolution'] = sr_results[:2]
        convergence_info['Super-Resolution'] = sr_results[2]
    except Exception as e:
        logger.error(f"Super-resolution failed: {e}")

    # Demonstrate compressive sensing
    try:
        cs_results = demonstrate_compressive_sensing(denoiser)
        convergence_info['Compressive Sensing'] = cs_results[2]
    except Exception as e:
        logger.error(f"Compressive sensing failed: {e}")

    # Store convergence info
    results['convergence'] = convergence_info

    # Visualize results
    try:
        visualize_results(results, 'denoiser_prior_results.png')
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

    logger.info("Demonstrations completed!")

    return results


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run demonstrations
    results = main()