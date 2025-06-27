"""
Complete Usage Example for YOLOv12 Multi-Task System.

This script demonstrates the complete workflow for training and using
the YOLOv12 multi-task model on the SUT-Crack dataset.

Workflow:
1. Dataset loading and exploration
2. Model training with multi-task learning
3. Model evaluation on test set
4. Full image inference demonstration
5. Results visualization

File: examples/yolo12_multitask_complete_example.py
"""

import os
import sys
import numpy as np
import keras
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_multitask_loss
from dl_techniques.utils.datasets.sut_crack_patch_loader import SUTCrackPatchDataset
from dl_techniques.utils.datasets.inference_utils import create_inference_engine, InferenceConfig
from dl_techniques.utils.logger import logger


def explore_dataset(data_dir: str):
    """Explore the SUT-Crack dataset structure and statistics."""
    logger.info("=== Dataset Exploration ===")

    # Create dataset instance
    dataset = SUTCrackPatchDataset(
        data_dir=data_dir,
        patch_size=256,
        patches_per_image=4,  # Small number for exploration
        include_segmentation=True
    )

    # Get dataset information
    info = dataset.get_dataset_info()
    logger.info(f"Dataset Statistics:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Create a small TF dataset to test data loading
    tf_dataset = dataset.create_tf_dataset(batch_size=2, repeat=False)

    # Sample one batch
    for batch_images, batch_labels in tf_dataset.take(1):
        logger.info(f"Batch shapes:")
        logger.info(f"  Images: {batch_images.shape}")
        logger.info(f"  Detection labels: {batch_labels['detection'].shape}")
        logger.info(f"  Segmentation labels: {batch_labels['segmentation'].shape}")
        logger.info(f"  Classification labels: {batch_labels['classification'].shape}")
        break

    return dataset


def train_multitask_model(
        dataset: SUTCrackPatchDataset,
        tasks: list = ["detection", "segmentation", "classification"],
        epochs: int = 5,  # Small number for demo
        scale: str = "n"
):
    """Train YOLOv12 multi-task model."""
    logger.info("=== Model Training ===")

    # Split dataset (simplified for demo)
    total_annotations = len(dataset.annotations)
    train_size = int(0.7 * total_annotations)
    val_size = int(0.2 * total_annotations)

    train_annotations = dataset.annotations[:train_size]
    val_annotations = dataset.annotations[train_size:train_size + val_size]
    test_annotations = dataset.annotations[train_size + val_size:]

    logger.info(f"Data splits - Train: {len(train_annotations)}, "
                f"Val: {len(val_annotations)}, Test: {len(test_annotations)}")

    # Create training dataset
    train_dataset = SUTCrackPatchDataset(
        data_dir=dataset.data_dir,
        patch_size=256,
        patches_per_image=8,
        include_segmentation=True
    )
    train_dataset.annotations = train_annotations

    # Create validation dataset
    val_dataset = SUTCrackPatchDataset(
        data_dir=dataset.data_dir,
        patch_size=256,
        patches_per_image=4,
        include_segmentation=True
    )
    val_dataset.annotations = val_annotations

    # Create TF datasets
    train_tf_dataset = train_dataset.create_tf_dataset(
        batch_size=4, shuffle=True, repeat=True
    )
    val_tf_dataset = val_dataset.create_tf_dataset(
        batch_size=4, shuffle=False, repeat=True
    )

    # Create model and loss
    model = create_yolov12_multitask(
        num_classes=1,
        input_shape=(256, 256, 3),
        scale=scale,
        tasks=tasks
    )

    loss_fn = create_multitask_loss(
        tasks=tasks,
        patch_size=256,
        use_uncertainty_weighting=True  # Use adaptive weighting
    )

    # Compile model
    optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Build model
    sample_input = keras.ops.zeros((1, 256, 256, 3))
    _ = model(sample_input, training=False)
    model.summary()

    # Calculate steps
    train_info = train_dataset.get_dataset_info()
    val_info = val_dataset.get_dataset_info()

    steps_per_epoch = train_info['total_patches_per_epoch'] // 4
    validation_steps = val_info['total_patches_per_epoch'] // 4

    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        epochs=epochs,
        steps_per_epoch=min(steps_per_epoch, 50),  # Limit for demo
        validation_steps=min(validation_steps, 10),  # Limit for demo
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model_path = "yolov12_multitask_demo.keras"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    return model, history, model_path


def evaluate_model(model_path: str, dataset: SUTCrackPatchDataset):
    """Evaluate the trained model."""
    logger.info("=== Model Evaluation ===")

    # Load model
    model = keras.models.load_model(model_path)

    # Create test dataset (use last annotations)
    test_annotations = dataset.annotations[-10:]  # Last 10 images for demo
    test_dataset = SUTCrackPatchDataset(
        data_dir=dataset.data_dir,
        patch_size=256,
        patches_per_image=4,
        include_segmentation=True
    )
    test_dataset.annotations = test_annotations

    # Create TF dataset
    test_tf_dataset = test_dataset.create_tf_dataset(
        batch_size=4, shuffle=False, repeat=False
    )

    # Evaluate on patches
    logger.info("Evaluating on patch data...")

    all_losses = []
    sample_predictions = []

    for batch_images, batch_labels in test_tf_dataset.take(3):  # Just a few batches
        # Get predictions
        predictions = model(batch_images, training=False)

        # Calculate loss
        loss = model.loss(batch_labels, predictions)
        all_losses.append(float(loss))

        # Store sample predictions for visualization
        if len(sample_predictions) == 0:
            sample_predictions.append({
                'images': batch_images.numpy(),
                'predictions': predictions,
                'ground_truth': batch_labels
            })

    avg_loss = np.mean(all_losses)
    logger.info(f"Average test loss: {avg_loss:.4f}")

    return sample_predictions


def demonstrate_full_image_inference(model_path: str, dataset: SUTCrackPatchDataset):
    """Demonstrate inference on full-resolution images."""
    logger.info("=== Full Image Inference Demo ===")

    # Create inference engine
    config = InferenceConfig(
        patch_size=256,
        overlap=64,
        batch_size=8,
        confidence_threshold=0.1
    )

    # Load model and create inference engine
    model = keras.models.load_model(model_path)

    from dl_techniques.utils.datasets.inference_utils import FullImageInference
    inference_engine = FullImageInference(model, config)

    # Get a few test images
    test_images = [ann.image_path for ann in dataset.annotations[:3]
                   if os.path.exists(ann.image_path)]

    if not test_images:
        logger.warning("No valid test images found for inference demo")
        return

    results = []
    for image_path in test_images:
        logger.info(f"Processing: {os.path.basename(image_path)}")

        try:
            result = inference_engine.predict_image(image_path)
            results.append(result)

            # Log results
            num_detections = len(result.get('detections', []))
            classification = result.get('classification', {})
            cls_score = classification.get('score', 0.0)
            cls_pred = classification.get('prediction', 0)

            logger.info(f"  Detections: {num_detections}")
            logger.info(f"  Classification: {'Crack' if cls_pred else 'No Crack'} "
                        f"(Score: {cls_score:.3f})")

            if 'segmentation' in result and result['segmentation'] is not None:
                seg_coverage = np.mean(result['segmentation'] > 0.5)
                logger.info(f"  Segmentation coverage: {seg_coverage:.3f}")

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")

    return results


def visualize_results(sample_predictions, inference_results):
    """Create visualizations of model predictions."""
    logger.info("=== Creating Visualizations ===")

    # Create output directory
    os.makedirs("demo_results", exist_ok=True)

    # Visualize patch predictions
    if sample_predictions:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sample = sample_predictions[0]
        images = sample['images']
        predictions = sample['predictions']
        ground_truth = sample['ground_truth']

        for i in range(min(3, len(images))):
            # Original patch
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f"Patch {i + 1}")
            axes[0, i].axis('off')

            # Segmentation prediction vs ground truth
            if isinstance(predictions, dict) and 'segmentation' in predictions:
                seg_pred = predictions['segmentation'][i, :, :, 0].numpy()
                seg_gt = ground_truth['segmentation'][i, :, :, 0].numpy()

                # Combine prediction and ground truth for comparison
                comparison = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3))
                comparison[:, :, 0] = seg_pred  # Red channel for predictions
                comparison[:, :, 1] = seg_gt  # Green channel for ground truth

                axes[1, i].imshow(comparison)
                axes[1, i].set_title(f"Seg: Red=Pred, Green=GT")
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, "No Segmentation", ha='center', va='center')
                axes[1, i].axis('off')

        plt.suptitle("Patch-level Predictions")
        plt.tight_layout()
        plt.savefig("demo_results/patch_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Patch predictions saved to: demo_results/patch_predictions.png")

    # Log inference results summary
    if inference_results:
        logger.info("Full image inference summary:")
        for i, result in enumerate(inference_results):
            logger.info(f"  Image {i + 1}:")
            logger.info(f"    Detections: {len(result.get('detections', []))}")

            if 'classification' in result:
                cls = result['classification']
                logger.info(f"    Classification: {cls.get('prediction', 0)} "
                            f"(Score: {cls.get('score', 0.0):.3f})")


def main():
    """Main demonstration function."""
    logger.info("YOLOv12 Multi-Task Complete Example")
    logger.info("=" * 50)

    # Configuration
    data_dir = "/path/to/SUT-Crack"  # Update this path
    tasks = ["detection", "segmentation", "classification"]

    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please update the data_dir path in the script to point to your SUT-Crack dataset")
        return

    try:
        # 1. Explore dataset
        dataset = explore_dataset(data_dir)

        # 2. Train model (with small epochs for demo)
        model, history, model_path = train_multitask_model(
            dataset,
            tasks=tasks,
            epochs=3,  # Small for demo
            scale="n"
        )

        # 3. Evaluate model
        sample_predictions = evaluate_model(model_path, dataset)

        # 4. Demonstrate full image inference
        inference_results = demonstrate_full_image_inference(model_path, dataset)

        # 5. Create visualizations
        visualize_results(sample_predictions, inference_results)

        logger.info("=" * 50)
        logger.info("Demo completed successfully!")
        logger.info("Key files created:")
        logger.info(f"  - Model: {model_path}")
        logger.info("  - Visualizations: demo_results/")

        # Print summary statistics
        if history:
            logger.info(f"Training summary:")
            logger.info(f"  Final loss: {history.history['loss'][-1]:.4f}")
            if 'val_loss' in history.history:
                logger.info(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def production_training_example():
    """Example configuration for production training."""
    logger.info("\n" + "=" * 50)
    logger.info("PRODUCTION TRAINING EXAMPLE")
    logger.info("=" * 50)

    example_command = """
# Complete training command for production use:

python train/yolo12_multitask/train_multitask.py \\
    --data-dir /path/to/SUT-Crack \\
    --tasks detection segmentation classification \\
    --scale s \\
    --epochs 200 \\
    --batch-size 32 \\
    --learning-rate 0.001 \\
    --optimizer adamw \\
    --uncertainty-weighting \\
    --patch-size 256

# Evaluation command:

python train/yolo12_multitask/evaluate_multitask.py \\
    --model-path results/best_model.keras \\
    --data-dir /path/to/SUT-Crack \\
    --output-dir evaluation_results/ \\
    --tasks detection segmentation classification

# Full image inference:

from dl_techniques.utils.datasets.inference_utils import create_inference_engine, InferenceConfig

config = InferenceConfig(
    patch_size=256,
    overlap=64,
    batch_size=16,
    confidence_threshold=0.1
)

engine = create_inference_engine("path/to/model.keras", config)
result = engine.predict_image("path/to/large_image.jpg")
"""

    logger.info(example_command)


if __name__ == "__main__":
    # Run main demo
    main()

    # Show production example
    production_training_example()