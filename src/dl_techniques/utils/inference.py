"""
Full Image Inference Utilities for YOLOv12 Multi-task Model.

This module provides utilities for running inference on full-resolution images
using a patch-based trained multi-task model. It handles sliding window
extraction, patch processing, and result aggregation.

Key Features:
    - Sliding window patch extraction
    - Batch processing for efficiency
    - Multi-task result aggregation
    - Memory-efficient processing of large images
    - Progress tracking and visualization

File: src/dl_techniques/utils/datasets/inference_utils.py
"""

import os
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.patch_transforms import (
    PatchGridGenerator, PatchInfo, PatchPrediction, DetectionResult,
    ResultAggregator, CoordinateTransformer
)
from dl_techniques.models.yolo12_multitask import YOLOv12MultiTask


@dataclass
class InferenceConfig:
    """Configuration for full image inference."""
    patch_size: int = 256
    overlap: int = 64
    batch_size: int = 16
    confidence_threshold: float = 0.1
    nms_iou_threshold: float = 0.5
    nms_score_threshold: float = 0.1
    segmentation_threshold: float = 0.5
    classification_threshold: float = 0.5
    blend_mode: str = "average"  # For segmentation stitching
    device: str = "auto"  # "auto", "cpu", "gpu"


class FullImageInference:
    """
    Full image inference engine for YOLOv12 multi-task model.

    Processes large images by extracting patches, running inference,
    and aggregating results across all tasks.
    """

    def __init__(
            self,
            model: YOLOv12MultiTask,
            config: InferenceConfig = None
    ):
        """
        Initialize full image inference engine.

        Args:
            model: Trained YOLOv12 multi-task model.
            config: Inference configuration.
        """
        self.model = model
        self.config = config or InferenceConfig()

        # Initialize components
        self.patch_generator = PatchGridGenerator(
            patch_size=self.config.patch_size,
            overlap=self.config.overlap
        )

        self.result_aggregator = ResultAggregator(
            nms_iou_threshold=self.config.nms_iou_threshold,
            nms_score_threshold=self.config.nms_score_threshold,
            segmentation_blend_mode=self.config.blend_mode,
            classification_method="weighted"
        )

        # Setup device
        self._setup_device()

        logger.info(f"Full image inference initialized with patch size {self.config.patch_size}")

    def _setup_device(self):
        """Setup computation device."""
        if self.config.device == "auto":
            # Use GPU if available
            if tf.config.list_physical_devices('GPU'):
                self.device = "/GPU:0"
                logger.info("Using GPU for inference")
            else:
                self.device = "/CPU:0"
                logger.info("Using CPU for inference")
        else:
            self.device = f"/{self.config.device.upper()}:0"

    def predict_image(
            self,
            image: Union[str, np.ndarray],
            return_patches: bool = False,
            progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a full image.

        Args:
            image: Image path or numpy array (H, W, C).
            return_patches: Whether to return individual patch predictions.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with aggregated results for all tasks.
        """
        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            try:
                # Use TensorFlow to read and decode the image to an RGB numpy array
                image_data = tf.io.read_file(image)
                # decode_image handles various formats (JPEG, PNG, etc.) and decodes to RGB
                image_tensor = tf.io.decode_image(image_data, channels=3, expand_animations=False)
                image_array = image_tensor.numpy()
            except tf.errors.OpError as e:
                raise ValueError(f"Failed to load image: {image}. Reason: {e}")
        else:
            image_array = image.copy()

        height, width = image_array.shape[:2]
        logger.info(f"Processing image of size {width}x{height}")

        # Generate patch grid
        patch_infos = self.patch_generator.generate_patches(width, height)
        logger.info(f"Generated {len(patch_infos)} patches")

        # Extract and process patches
        patch_predictions = self._process_patches(
            image_array, patch_infos, progress_callback
        )

        # Aggregate results
        final_results = self.result_aggregator.aggregate_predictions(
            patch_predictions, width, height
        )

        # Add image metadata
        final_results['image_shape'] = (height, width, image_array.shape[2])
        final_results['patch_config'] = {
            'patch_size': self.config.patch_size,
            'overlap': self.config.overlap,
            'num_patches': len(patch_infos)
        }

        if return_patches:
            final_results['patch_predictions'] = patch_predictions

        return final_results

    def _process_patches(
            self,
            image: np.ndarray,
            patch_infos: List[PatchInfo],
            progress_callback: Optional[Callable] = None
    ) -> List[PatchPrediction]:
        """Process all patches and return predictions."""
        patch_predictions = []

        # Process patches in batches for efficiency
        for i in range(0, len(patch_infos), self.config.batch_size):
            batch_patch_infos = patch_infos[i:i + self.config.batch_size]

            # Extract batch of patches
            batch_patches = []
            for patch_info in batch_patch_infos:
                patch = self.patch_generator.extract_patch(image, patch_info)
                # Normalize to [0, 1]
                patch_normalized = patch.astype(np.float32) / 255.0
                batch_patches.append(patch_normalized)

            # Convert to tensor and run inference
            batch_tensor = tf.stack(batch_patches, axis=0)

            with tf.device(self.device):
                batch_predictions = self.model(batch_tensor, training=False)

            # Process batch predictions
            for j, patch_info in enumerate(batch_patch_infos):
                patch_pred = self._process_single_patch_prediction(
                    batch_predictions, j, patch_info
                )
                patch_predictions.append(patch_pred)

            # Update progress
            if progress_callback:
                progress = (i + len(batch_patch_infos)) / len(patch_infos)
                progress_callback(progress)

        return patch_predictions

    def _process_single_patch_prediction(
            self,
            batch_predictions: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            patch_idx: int,
            patch_info: PatchInfo
    ) -> PatchPrediction:
        """Process prediction for a single patch."""
        detections = []
        segmentation_mask = None
        classification_score = 0.0

        # Handle multi-task or single-task outputs
        if isinstance(batch_predictions, dict):
            predictions = batch_predictions
        else:
            # Single task output - assume detection
            predictions = {'detection': batch_predictions}

        # Process detection predictions
        if 'detection' in predictions:
            det_pred = predictions['detection'][patch_idx:patch_idx + 1]  # Keep batch dim
            detections = self._decode_detection_predictions(det_pred, patch_info)

        # Process segmentation predictions
        if 'segmentation' in predictions:
            seg_pred = predictions['segmentation'][patch_idx]  # Remove batch dim
            segmentation_mask = tf.nn.sigmoid(seg_pred).numpy()
            segmentation_mask = np.squeeze(segmentation_mask, axis=-1)  # Remove channel dim

        # Process classification predictions
        if 'classification' in predictions:
            cls_pred = predictions['classification'][patch_idx]
            classification_score = float(tf.nn.sigmoid(cls_pred).numpy())

        return PatchPrediction(
            patch_info=patch_info,
            detections=detections,
            segmentation_mask=segmentation_mask,
            classification_score=classification_score
        )

    def _decode_detection_predictions(
            self,
            predictions: keras.KerasTensor,
            patch_info: PatchInfo
    ) -> List[DetectionResult]:
        """Decode detection predictions for a single patch."""
        # Use the detection head's decode method if available
        if hasattr(self.model, 'detection_head') and self.model.detection_head is not None:
            try:
                bboxes, scores, valid = self.model.detection_head.decode_predictions(
                    predictions, confidence_threshold=self.config.confidence_threshold
                )

                detections = []
                batch_idx = 0  # Single patch prediction

                # Extract valid detections
                valid_mask = valid[batch_idx].numpy()
                if np.any(valid_mask):
                    valid_bboxes = bboxes[batch_idx][valid_mask].numpy()
                    valid_scores = scores[batch_idx][valid_mask].numpy()

                    for bbox, score_vec in zip(valid_bboxes, valid_scores):
                        # Get maximum score across classes
                        max_score = np.max(score_vec)
                        class_id = np.argmax(score_vec)

                        if max_score > self.config.confidence_threshold:
                            detections.append(DetectionResult(
                                bbox=bbox.tolist(),
                                confidence=float(max_score),
                                class_id=int(class_id)
                            ))

                return detections

            except Exception as e:
                logger.warning(f"Failed to decode detection predictions: {e}")
                return []

        return []

    def predict_batch_images(
            self,
            image_paths: List[str],
            output_dir: Optional[str] = None,
            save_visualizations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.

        Args:
            image_paths: List of image file paths.
            output_dir: Directory to save results and visualizations.
            save_visualizations: Whether to save visualization images.

        Returns:
            List of inference results for each image.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Run inference
                result = self.predict_image(image_path)

                # Add image metadata
                result['image_path'] = image_path
                result['image_name'] = os.path.basename(image_path)

                results.append(result)

                # Save results and visualizations
                if output_dir:
                    self._save_image_results(
                        image_path, result, output_dir, save_visualizations
                    )

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        return results

    def _save_image_results(
            self,
            image_path: str,
            result: Dict[str, Any],
            output_dir: str,
            save_visualizations: bool
    ):
        """Save results and visualizations for a single image."""
        image_name = Path(image_path).stem

        # Save detection results as JSON
        detection_file = os.path.join(output_dir, f"{image_name}_detections.json")
        detection_data = {
            'image_path': image_path,
            'detections': result.get('detections', []),
            'classification': result.get('classification', {}),
            'stats': result.get('stats', {})
        }

        import json
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f, indent=2)

        # Save segmentation mask
        if 'segmentation' in result and result['segmentation'] is not None:
            seg_mask = result['segmentation']
            seg_file = os.path.join(output_dir, f"{image_name}_segmentation.png")

            # Convert to 8-bit image
            seg_mask_8bit = (seg_mask * 255).astype(np.uint8)
            # Encode to PNG using TensorFlow
            if seg_mask_8bit.ndim == 2: # Grayscale mask
                seg_mask_tensor = tf.expand_dims(seg_mask_8bit, axis=-1)
            else:
                seg_mask_tensor = tf.convert_to_tensor(seg_mask_8bit)

            encoded_image = tf.io.encode_png(seg_mask_tensor)
            tf.io.write_file(seg_file, encoded_image)

        # Create visualizations
        if save_visualizations:
            self._create_visualization(image_path, result, output_dir)

    def _create_visualization(
            self,
            image_path: str,
            result: Dict[str, Any],
            output_dir: str
    ):
        """Create and save visualization of inference results."""
        # Load original image using TensorFlow
        try:
            image_data = tf.io.read_file(image_path)
            image_tensor = tf.io.decode_image(image_data, channels=3, expand_animations=False)
            image = image_tensor.numpy()
        except (tf.errors.OpError, FileNotFoundError) as e:
            logger.error(f"Could not load image for visualization: {image_path}. Reason: {e}")
            return

        image_name = Path(image_path).stem

        # Create visualization figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.subplots_adjust(wspace=0.1, hspace=0.2)

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # Detection results - draw directly on matplotlib axes
        axes[0, 1].imshow(image)
        if 'detections' in result:
            for det in result['detections']:
                bbox = det['bbox']
                conf = det['confidence']

                x1, y1, x2, y2 = [int(x) for x in bbox]
                width, height = x2 - x1, y2 - y1

                # Draw bounding box using a matplotlib patch
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                axes[0, 1].add_patch(rect)

                # Add confidence text
                text = f"Crack: {conf:.2f}"
                # Add a small white background to text for better readability
                axes[0, 1].text(x1, y1 - 10, text, color='red', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

        axes[0, 1].set_title(f"Detections (Count: {len(result.get('detections', []))})")
        axes[0, 1].axis('off')

        # Segmentation results
        if 'segmentation' in result and result['segmentation'] is not None:
            seg_mask = result['segmentation']

            # Create overlay
            seg_overlay = image.copy().astype(np.float32)
            crack_mask = seg_mask > self.config.segmentation_threshold
            # Create a red color mask for the overlay
            red_color = np.array([255, 0, 0], dtype=np.float32)
            seg_overlay[crack_mask] = seg_overlay[crack_mask] * 0.5 + red_color * 0.5
            seg_overlay = np.clip(seg_overlay, 0, 255).astype(np.uint8)

            axes[1, 0].imshow(seg_overlay)
            axes[1, 0].set_title("Segmentation Overlay")
            axes[1, 0].axis('off')

            # Segmentation mask only
            axes[1, 1].imshow(seg_mask, cmap='hot')
            axes[1, 1].set_title("Segmentation Mask")
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "No Segmentation", ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, "No Segmentation", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        # Add classification and statistics
        if 'classification' in result:
            cls_info = result['classification']
            score = cls_info.get('score', 0.0)
            prediction = cls_info.get('prediction', 0)

            fig.suptitle(
                f"Classification: {'Crack' if prediction else 'No Crack'} "
                f"(Score: {score:.3f})",
                fontsize=16
            )

        # Save visualization
        viz_file = os.path.join(output_dir, f"{image_name}_visualization.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close(fig)


class InferenceProfiler:
    """Performance profiler for inference operations."""

    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}

    def profile_inference(
            self,
            inference_engine: FullImageInference,
            test_images: List[str],
            output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Profile inference performance on test images.

        Args:
            inference_engine: Inference engine to profile.
            test_images: List of test image paths.
            output_file: Optional file to save profiling results.

        Returns:
            Profiling results dictionary.
        """
        import time
        import psutil

        results = {
            'image_results': [],
            'summary': {}
        }

        total_start_time = time.time()

        for image_path in test_images:
            # Load image to get size info using TensorFlow
            try:
                image_data = tf.io.read_file(image_path)
                image_tensor = tf.io.decode_image(image_data, channels=3, expand_animations=False)
                height, width, _ = image_tensor.shape
            except (tf.errors.OpError, FileNotFoundError) as e:
                logger.warning(f"Skipping unloadable image in profiler: {image_path}. Reason: {e}")
                continue

            image_size = width * height

            # Profile single image inference
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            try:
                result = inference_engine.predict_image(image_path)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                inference_time = end_time - start_time
                memory_usage = end_memory - start_memory

                image_result = {
                    'image_path': image_path,
                    'image_size': (width, height),
                    'image_pixels': image_size,
                    'inference_time': inference_time,
                    'memory_usage_mb': memory_usage,
                    'patches_processed': result['patch_config']['num_patches'],
                    'detections_found': len(result.get('detections', [])),
                    'pixels_per_second': image_size / inference_time if inference_time > 0 else 0
                }

                results['image_results'].append(image_result)

            except Exception as e:
                logger.error(f"Failed to profile {image_path}: {e}")

        total_time = time.time() - total_start_time

        # Calculate summary statistics
        if results['image_results']:
            times = [r['inference_time'] for r in results['image_results']]
            memories = [r['memory_usage_mb'] for r in results['image_results']]
            pixels = [r['image_pixels'] for r in results['image_results']]

            results['summary'] = {
                'total_images': len(results['image_results']),
                'total_time': total_time,
                'avg_inference_time': np.mean(times),
                'min_inference_time': np.min(times),
                'max_inference_time': np.max(times),
                'avg_memory_usage_mb': np.mean(memories),
                'total_pixels_processed': sum(pixels),
                'avg_pixels_per_second': np.mean([r['pixels_per_second'] for r in results['image_results']])
            }

        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        return results


def create_inference_engine(
        model_path: str,
        config: InferenceConfig = None
) -> FullImageInference:
    """
    Create inference engine from saved model.

    Args:
        model_path: Path to saved YOLOv12 multi-task model.
        config: Inference configuration.

    Returns:
        FullImageInference engine.
    """
    # Load model
    model = keras.models.load_model(model_path)

    # Create inference engine
    engine = FullImageInference(model, config)

    logger.info(f"Inference engine created from model: {model_path}")
    return engine


# Example usage and testing
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Run full-image inference with a YOLOv12 multi-task model."
    )
    parser.add_argument(
        "-m", "--model_path", type=str, required=True,
        help="Path to the saved Keras model file."
    )
    parser.add_argument(
        "-i", "--image_path", type=str, required=True,
        help="Path to the input image for inference."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Directory to save the inference results and visualizations."
    )

    # Add optional arguments for InferenceConfig
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches for inference (default: 256).")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between patches (default: 64).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model prediction (default: 8).")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Confidence threshold for detections (default: 0.1).")

    args = parser.parse_args()

    # Create InferenceConfig from arguments
    config = InferenceConfig(
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold
    )
    logger.info(f"Using inference config: {config}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {args.output_dir}")

    try:
        # Create inference engine from model
        logger.info(f"Loading model from: {args.model_path}")
        engine = create_inference_engine(args.model_path, config)

        # Run single image inference
        logger.info(f"Running inference on image: {args.image_path}")
        start_time = time.time()
        result = engine.predict_image(args.image_path)
        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")

        # Save results and visualizations
        logger.info("Saving results and visualizations...")
        engine._save_image_results(
            image_path=args.image_path,
            result=result,
            output_dir=args.output_dir,
            save_visualizations=True
        )
        logger.info(f"Successfully saved results to {args.output_dir}")

    except FileNotFoundError as e:
        logger.error(f"Error: A required file was not found. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        # exc_info=True will log the full traceback for better debugging