"""
Usage Example: Inference with Fine-tuned YOLOv12 Model

This script demonstrates how to use your fine-tuned YOLOv12 model for inference
on new crack detection images. It shows how to:

1. Load the fine-tuned model
2. Preprocess input images
3. Run inference for all tasks (detection, segmentation, classification)
4. Post-process and visualize results
5. Save results in various formats

Usage:
    python inference.py \
        --model-path finetune_results/.../final_finetuned_model.keras \
        --input-path /path/to/test/images \
        --output-dir inference_results \
        --patch-size 256 \
        --visualize

File: inference.py
"""

import os
import sys
import argparse
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dl_techniques.utils.logger import logger


def load_finetuned_model(model_path: str) -> keras.Model:
    """
    Load the fine-tuned YOLOv12 model.

    Args:
        model_path: Path to the saved .keras model file.

    Returns:
        Loaded Keras model.
    """
    try:
        logger.info(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)

        logger.info("✅ Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model outputs: {list(model.output_names) if hasattr(model, 'output_names') else 'Single output'}")

        return model

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def preprocess_image(
        image_path: str,
        target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Preprocess an input image for inference.

    Args:
        image_path: Path to input image.
        target_size: Target size for model input.

    Returns:
        Tuple of (preprocessed_image, original_image, metadata).
    """
    try:
        # Read image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_shape = original_image.shape

        # Resize to target size
        resized_image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        preprocessed_image = resized_image.astype(np.float32) / 255.0

        # Add batch dimension
        batch_image = np.expand_dims(preprocessed_image, axis=0)

        metadata = {
            'original_shape': original_shape,
            'target_size': target_size,
            'scale_x': original_shape[1] / target_size[0],
            'scale_y': original_shape[0] / target_size[1]
        }

        return batch_image, original_image, metadata

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise


def run_inference(
        model: keras.Model,
        preprocessed_image: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Run inference on preprocessed image.

    Args:
        model: Loaded YOLOv12 model.
        preprocessed_image: Preprocessed image batch.

    Returns:
        Dictionary of task predictions.
    """
    try:
        logger.info("Running inference...")

        # Run prediction
        predictions = model(preprocessed_image, training=False)

        # Handle both single output and dictionary outputs
        if isinstance(predictions, dict):
            # Multi-task model with named outputs
            results = {}
            for task_name, prediction in predictions.items():
                results[task_name] = prediction.numpy()
            logger.info(f"Got predictions for tasks: {list(results.keys())}")
        else:
            # Single task model
            results = {'prediction': predictions.numpy()}
            logger.info("Got single task prediction")

        return results

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def decode_detection_predictions(
        detection_pred: np.ndarray,
        confidence_threshold: float = 0.5,
        img_size: int = 256,
        reg_max: int = 16
) -> List[Dict[str, any]]:
    """
    Decode YOLOv12 detection predictions to bounding boxes.

    Args:
        detection_pred: Raw detection predictions.
        confidence_threshold: Minimum confidence threshold.
        img_size: Input image size.
        reg_max: DFL regression maximum value.

    Returns:
        List of detected objects.
    """
    try:
        # detection_pred shape: (1, num_anchors, 4*reg_max + num_classes)
        batch_size, num_anchors, features = detection_pred.shape

        # Split into distance and classification
        dist_pred = detection_pred[0, :, :4 * reg_max]  # (num_anchors, 4*reg_max)
        cls_pred = detection_pred[0, :, 4 * reg_max:]  # (num_anchors, num_classes)

        # Apply sigmoid to classification scores
        cls_scores = 1 / (1 + np.exp(-cls_pred))

        # Find anchors above confidence threshold
        max_scores = np.max(cls_scores, axis=1)
        valid_anchors = max_scores > confidence_threshold

        if not np.any(valid_anchors):
            logger.info("No detections above confidence threshold")
            return []

        # Decode distance predictions for valid anchors
        valid_dist = dist_pred[valid_anchors]  # (num_valid, 4*reg_max)
        valid_scores = cls_scores[valid_anchors]  # (num_valid, num_classes)

        # Reshape and apply softmax to distance predictions
        valid_dist = valid_dist.reshape(-1, 4, reg_max)
        softmax_dist = np.exp(valid_dist) / (np.sum(np.exp(valid_dist), axis=-1, keepdims=True) + 1e-8)

        # Compute weighted distances
        range_vals = np.arange(reg_max)
        decoded_dist = np.sum(softmax_dist * range_vals, axis=-1)  # (num_valid, 4)

        # Generate anchor grid (simplified)
        strides = [8, 16, 32]
        anchors = []
        for stride in strides:
            h, w = img_size // stride, img_size // stride
            for y in range(h):
                for x in range(w):
                    anchors.append([(x + 0.5) * stride, (y + 0.5) * stride])

        anchors = np.array(anchors)
        valid_anchor_indices = np.where(valid_anchors)[0]
        valid_anchor_coords = anchors[valid_anchor_indices]

        # Convert distances to bounding boxes
        detections = []
        for i, (anchor_coords, distances, scores) in enumerate(zip(valid_anchor_coords, decoded_dist, valid_scores)):
            cx, cy = anchor_coords
            l, t, r, b = distances

            # Convert to x1, y1, x2, y2
            x1 = max(0, cx - l)
            y1 = max(0, cy - t)
            x2 = min(img_size, cx + r)
            y2 = min(img_size, cy + b)

            # Get best class and confidence
            best_class = np.argmax(scores)
            confidence = scores[best_class]

            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class': int(best_class),
                'class_name': 'crack'  # For crack detection
            }
            detections.append(detection)

        logger.info(f"Decoded {len(detections)} detections")
        return detections

    except Exception as e:
        logger.error(f"Failed to decode detections: {e}")
        return []


def process_segmentation_predictions(
        segmentation_pred: np.ndarray,
        threshold: float = 0.5
) -> np.ndarray:
    """
    Process segmentation predictions.

    Args:
        segmentation_pred: Raw segmentation predictions.
        threshold: Binary threshold for segmentation mask.

    Returns:
        Binary segmentation mask.
    """
    try:
        # segmentation_pred shape: (1, height, width, 1)
        mask = segmentation_pred[0, :, :, 0]

        # Apply sigmoid if not already applied
        if mask.max() > 1.0 or mask.min() < 0.0:
            mask = 1 / (1 + np.exp(-mask))

        # Apply threshold
        binary_mask = (mask > threshold).astype(np.uint8)

        # Calculate statistics
        total_pixels = binary_mask.size
        crack_pixels = np.sum(binary_mask)
        crack_percentage = (crack_pixels / total_pixels) * 100

        logger.info(f"Segmentation: {crack_percentage:.1f}% crack pixels")

        return binary_mask

    except Exception as e:
        logger.error(f"Failed to process segmentation: {e}")
        return np.zeros((256, 256), dtype=np.uint8)


def process_classification_predictions(
        classification_pred: np.ndarray,
        threshold: float = 0.5
) -> Dict[str, any]:
    """
    Process classification predictions.

    Args:
        classification_pred: Raw classification predictions.
        threshold: Binary classification threshold.

    Returns:
        Classification results.
    """
    try:
        # classification_pred shape: (1, 1) for binary classification
        logit = classification_pred[0, 0]

        # Apply sigmoid
        probability = 1 / (1 + np.exp(-logit))

        # Binary decision
        prediction = probability > threshold

        result = {
            'has_crack': bool(prediction),
            'probability': float(probability),
            'confidence': float(max(probability, 1 - probability))
        }

        logger.info(f"Classification: {'Crack' if result['has_crack'] else 'No crack'} "
                    f"(confidence: {result['confidence']:.3f})")

        return result

    except Exception as e:
        logger.error(f"Failed to process classification: {e}")
        return {'has_crack': False, 'probability': 0.0, 'confidence': 0.0}


def visualize_results(
        original_image: np.ndarray,
        detections: List[Dict[str, any]],
        segmentation_mask: Optional[np.ndarray],
        classification_result: Optional[Dict[str, any]],
        metadata: Dict[str, any],
        save_path: Optional[str] = None
) -> None:
    """
    Visualize inference results.

    Args:
        original_image: Original input image.
        detections: List of detection results.
        segmentation_mask: Segmentation mask.
        classification_result: Classification result.
        metadata: Image metadata.
        save_path: Path to save visualization.
    """
    try:
        # Determine subplot layout
        n_plots = 1  # Original image
        if detections:
            n_plots += 1
        if segmentation_mask is not None:
            n_plots += 1

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Original image with detections
        axes[plot_idx].imshow(original_image)
        axes[plot_idx].set_title("Original Image with Detections")
        axes[plot_idx].axis('off')

        # Draw detection boxes on original image
        if detections:
            scale_x = metadata['scale_x']
            scale_y = metadata['scale_y']

            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                # Scale back to original image coordinates
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y

                # Draw rectangle
                rect = patches.Rectangle(
                    (x1_orig, y1_orig),
                    x2_orig - x1_orig,
                    y2_orig - y1_orig,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[plot_idx].add_patch(rect)

                # Add confidence label
                axes[plot_idx].text(
                    x1_orig, y1_orig - 10,
                    f"Crack: {det['confidence']:.2f}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                    color='white', fontsize=10, fontweight='bold'
                )

        plot_idx += 1

        # Segmentation overlay
        if segmentation_mask is not None and plot_idx < len(axes):
            # Resize mask to original image size
            original_h, original_w = original_image.shape[:2]
            mask_resized = cv2.resize(segmentation_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            # Create overlay
            overlay = original_image.copy()
            overlay[mask_resized > 0] = [255, 0, 0]  # Red for cracks

            # Blend images
            blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

            axes[plot_idx].imshow(blended)
            axes[plot_idx].set_title("Segmentation Overlay")
            axes[plot_idx].axis('off')
            plot_idx += 1

        # Add classification result as text
        if classification_result:
            result_text = f"Classification: {'CRACK DETECTED' if classification_result['has_crack'] else 'NO CRACK'}\n"
            result_text += f"Confidence: {classification_result['confidence']:.3f}"

            fig.suptitle(result_text, fontsize=14, fontweight='bold',
                         color='red' if classification_result['has_crack'] else 'green')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def save_results_json(
        results: Dict[str, any],
        output_path: str
) -> None:
    """
    Save inference results to JSON file.

    Args:
        results: Complete inference results.
        output_path: Path to save JSON file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def process_single_image(
        model: keras.Model,
        image_path: str,
        output_dir: str,
        patch_size: int = 256,
        visualize: bool = True
) -> Dict[str, any]:
    """
    Process a single image through the full inference pipeline.

    Args:
        model: Loaded YOLOv12 model.
        image_path: Path to input image.
        output_dir: Directory to save outputs.
        patch_size: Input patch size.
        visualize: Whether to create visualizations.

    Returns:
        Complete inference results.
    """
    logger.info(f"Processing image: {image_path}")

    # Preprocess image
    preprocessed_image, original_image, metadata = preprocess_image(
        image_path, target_size=(patch_size, patch_size)
    )

    # Run inference
    predictions = run_inference(model, preprocessed_image)

    # Process predictions
    results = {
        'image_path': image_path,
        'metadata': metadata
    }

    # Detection
    if 'detection' in predictions:
        detections = decode_detection_predictions(
            predictions['detection'],
            confidence_threshold=0.3,
            img_size=patch_size
        )
        results['detections'] = detections
    else:
        detections = []

    # Segmentation
    if 'segmentation' in predictions:
        segmentation_mask = process_segmentation_predictions(
            predictions['segmentation'],
            threshold=0.5
        )
        results['segmentation'] = {
            'mask_shape': segmentation_mask.shape,
            'crack_pixels': int(np.sum(segmentation_mask)),
            'total_pixels': int(segmentation_mask.size),
            'crack_percentage': float((np.sum(segmentation_mask) / segmentation_mask.size) * 100)
        }
    else:
        segmentation_mask = None

    # Classification
    if 'classification' in predictions:
        classification_result = process_classification_predictions(
            predictions['classification'],
            threshold=0.5
        )
        results['classification'] = classification_result
    else:
        classification_result = None

    # Create visualizations
    if visualize:
        image_name = Path(image_path).stem
        viz_path = os.path.join(output_dir, f"{image_name}_results.png")

        visualize_results(
            original_image=original_image,
            detections=detections,
            segmentation_mask=segmentation_mask,
            classification_result=classification_result,
            metadata=metadata,
            save_path=viz_path
        )

    # Save JSON results
    json_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
    save_results_json(results, json_path)

    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Run inference with fine-tuned YOLOv12 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to fine-tuned model (.keras file)')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--patch-size', type=int, default=256,
                        help='Input patch size (should match training)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    if not os.path.exists(args.input_path):
        logger.error(f"Input path not found: {args.input_path}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("🚀 Starting YOLOv12 Crack Detection Inference")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_dir}")

    # Load model
    model = load_finetuned_model(args.model_path)

    # Process images
    if os.path.isfile(args.input_path):
        # Single image
        results = process_single_image(
            model=model,
            image_path=args.input_path,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            visualize=args.visualize
        )
        logger.info("✅ Single image processed successfully")

    elif os.path.isdir(args.input_path):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in os.listdir(args.input_path)
            if Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.error("No image files found in input directory")
            return

        logger.info(f"Found {len(image_files)} images to process")

        all_results = []
        for image_file in image_files:
            image_path = os.path.join(args.input_path, image_file)

            try:
                results = process_single_image(
                    model=model,
                    image_path=image_path,
                    output_dir=args.output_dir,
                    patch_size=args.patch_size,
                    visualize=args.visualize
                )
                all_results.append(results)

            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                continue

        # Save batch results
        batch_results_path = os.path.join(args.output_dir, 'batch_results.json')
        save_results_json(all_results, batch_results_path)

        logger.info(f"✅ Processed {len(all_results)}/{len(image_files)} images successfully")

    else:
        logger.error("Input path must be a file or directory")
        return

    logger.info(f"🎉 Inference completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()