"""
Comprehensive Evaluation Script for YOLOv12 Multi-Task Model.

This script provides detailed evaluation of the trained multi-task model
across all tasks: object detection, segmentation, and classification.
It includes metrics computation, visualization, and comprehensive reporting.

Features:
    - Task-specific evaluation metrics
    - Full image inference evaluation
    - Patch-level and image-level assessment
    - Comprehensive visualizations
    - Performance profiling
    - Detailed reporting

Usage:
    python evaluate_multitask.py --model-path /path/to/model.keras \
                                 --data-dir /path/to/SUT-Crack \
                                 --output-dir /path/to/results

File: src/train/yolo12_multitask/evaluate_multitask.py
"""

import argparse
import os
import sys
import numpy as np
import keras
import tensorflow as tf
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dl_techniques.models.yolo12_multitask import YOLOv12MultiTask
from dl_techniques.utils.datasets.sut_crack_patch_loader import SUTCrackPatchDataset
from dl_techniques.utils.datasets.inference_utils import (
    create_inference_engine, InferenceConfig, FullImageInference
)
from dl_techniques.utils.datasets.patch_transforms import (
    NonMaximumSuppression, CoordinateTransformer
)
from dl_techniques.utils.logger import logger


class DetectionEvaluator:
    """Evaluator for object detection performance."""

    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize detection evaluator.

        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation.
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.nms = NonMaximumSuppression()

    def compute_metrics(
            self,
            predictions: List[Dict[str, Any]],
            ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute detection metrics including mAP.

        Args:
            predictions: List of prediction dictionaries with 'detections'.
            ground_truths: List of ground truth annotations.

        Returns:
            Dictionary of detection metrics.
        """
        # Collect all predictions and ground truths
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        for pred, gt in zip(predictions, ground_truths):
            # Process predictions
            detections = pred.get('detections', [])
            for det in detections:
                all_pred_boxes.append(det['bbox'])
                all_pred_scores.append(det['confidence'])

            # Process ground truths
            if 'bboxes' in gt:
                for bbox in gt['bboxes']:
                    all_gt_boxes.append([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])

        if not all_pred_boxes or not all_gt_boxes:
            return {
                'mAP@0.5': 0.0,
                'mAP@0.5:0.95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'num_predictions': len(all_pred_boxes),
                'num_ground_truths': len(all_gt_boxes)
            }

        # Calculate metrics for different IoU thresholds
        aps = []
        for iou_thresh in self.iou_thresholds:
            ap = self._calculate_ap(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh)
            aps.append(ap)

        # Calculate precision, recall, F1 at IoU=0.5
        precision, recall, f1 = self._calculate_precision_recall_f1(
            all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5
        )

        return {
            'mAP@0.5': aps[0],  # AP at IoU=0.5
            'mAP@0.5:0.95': np.mean(aps),  # Mean AP across all IoU thresholds
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_predictions': len(all_pred_boxes),
            'num_ground_truths': len(all_gt_boxes)
        }

    def _calculate_ap(
            self,
            pred_boxes: List[List[float]],
            pred_scores: List[float],
            gt_boxes: List[List[float]],
            iou_threshold: float
    ) -> float:
        """Calculate Average Precision for given IoU threshold."""
        if not pred_boxes or not gt_boxes:
            return 0.0

        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
        sorted_scores = [pred_scores[i] for i in sorted_indices]

        # Track which ground truths have been matched
        gt_matched = [False] * len(gt_boxes)

        tp = []  # True positives
        fp = []  # False positives

        for pred_box in sorted_pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = self.nms.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if prediction is TP or FP
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (len(gt_boxes) + 1e-8)

        # Calculate AP using interpolation
        return self._interpolate_ap(precision, recall)

    def _interpolate_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Calculate AP using 11-point interpolation."""
        ap = 0.0
        for thresh in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= thresh) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= thresh])
            ap += p / 11.0
        return ap

    def _calculate_precision_recall_f1(
            self,
            pred_boxes: List[List[float]],
            pred_scores: List[float],
            gt_boxes: List[List[float]],
            iou_threshold: float = 0.5,
            score_threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not pred_boxes or not gt_boxes:
            return 0.0, 0.0, 0.0

        # Filter predictions by score threshold
        filtered_preds = [(box, score) for box, score in zip(pred_boxes, pred_scores)
                          if score >= score_threshold]

        if not filtered_preds:
            return 0.0, 0.0, 0.0

        filtered_boxes = [item[0] for item in filtered_preds]
        gt_matched = [False] * len(gt_boxes)
        tp = 0

        # Count true positives
        for pred_box in filtered_boxes:
            for gt_idx, gt_box in enumerate(gt_boxes):
                if not gt_matched[gt_idx]:
                    iou = self.nms.compute_iou(pred_box, gt_box)
                    if iou >= iou_threshold:
                        tp += 1
                        gt_matched[gt_idx] = True
                        break

        fp = len(filtered_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1


class SegmentationEvaluator:
    """Evaluator for segmentation performance."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize segmentation evaluator.

        Args:
            threshold: Threshold for binary segmentation.
        """
        self.threshold = threshold

    def compute_metrics(
            self,
            predictions: List[np.ndarray],
            ground_truths: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Args:
            predictions: List of predicted masks.
            ground_truths: List of ground truth masks.

        Returns:
            Dictionary of segmentation metrics.
        """
        if not predictions or not ground_truths:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'pixel_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

        ious = []
        dices = []
        pixel_accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for pred_mask, gt_mask in zip(predictions, ground_truths):
            # Ensure same shape
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

            # Binarize predictions
            pred_binary = (pred_mask > self.threshold).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)

            # Calculate metrics
            intersection = np.sum(pred_binary & gt_binary)
            union = np.sum(pred_binary | gt_binary)
            pred_sum = np.sum(pred_binary)
            gt_sum = np.sum(gt_binary)

            # IoU
            iou = intersection / (union + 1e-8)
            ious.append(iou)

            # Dice coefficient
            dice = 2 * intersection / (pred_sum + gt_sum + 1e-8)
            dices.append(dice)

            # Pixel accuracy
            correct_pixels = np.sum(pred_binary == gt_binary)
            total_pixels = pred_binary.size
            pixel_acc = correct_pixels / total_pixels
            pixel_accuracies.append(pixel_acc)

            # Precision, Recall, F1
            tp = intersection
            fp = pred_sum - intersection
            fn = gt_sum - intersection

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            'iou': np.mean(ious),
            'dice': np.mean(dices),
            'pixel_accuracy': np.mean(pixel_accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores)
        }


class ClassificationEvaluator:
    """Evaluator for classification performance."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize classification evaluator.

        Args:
            threshold: Threshold for binary classification.
        """
        self.threshold = threshold

    def compute_metrics(
            self,
            predictions: List[float],
            ground_truths: List[int]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: List of prediction scores.
            ground_truths: List of ground truth labels.

        Returns:
            Dictionary of classification metrics.
        """
        if not predictions or not ground_truths:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc': 0.0,
                'average_precision': 0.0
            }

        pred_array = np.array(predictions)
        gt_array = np.array(ground_truths)

        # Binary predictions
        pred_binary = (pred_array > self.threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(gt_array, pred_binary)
        precision = precision_score(gt_array, pred_binary, zero_division=0)
        recall = recall_score(gt_array, pred_binary, zero_division=0)
        f1 = f1_score(gt_array, pred_binary, zero_division=0)

        # AUC and Average Precision (if we have both classes)
        auc = 0.0
        avg_precision = 0.0
        if len(np.unique(gt_array)) > 1:
            try:
                auc = roc_auc_score(gt_array, pred_array)
                avg_precision = average_precision_score(gt_array, pred_array)
            except Exception as e:
                logger.warning(f"Failed to calculate AUC/AP: {e}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'average_precision': avg_precision,
            'confusion_matrix': confusion_matrix(gt_array, pred_binary).tolist()
        }


class MultiTaskEvaluator:
    """Comprehensive evaluator for multi-task model."""

    def __init__(self, tasks: List[str]):
        """
        Initialize multi-task evaluator.

        Args:
            tasks: List of tasks to evaluate.
        """
        self.tasks = tasks
        self.detection_evaluator = DetectionEvaluator()
        self.segmentation_evaluator = SegmentationEvaluator()
        self.classification_evaluator = ClassificationEvaluator()

    def evaluate_model(
            self,
            model_path: str,
            test_dataset: SUTCrackPatchDataset,
            output_dir: str,
            patch_evaluation: bool = True,
            full_image_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model_path: Path to trained model.
            test_dataset: Test dataset.
            output_dir: Output directory for results.
            patch_evaluation: Whether to evaluate on patches.
            full_image_evaluation: Whether to evaluate on full images.

        Returns:
            Comprehensive evaluation results.
        """
        logger.info("Starting comprehensive model evaluation...")

        # Load model
        model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from: {model_path}")

        results = {'model_path': model_path, 'tasks': self.tasks}

        # Patch-level evaluation
        if patch_evaluation:
            logger.info("Running patch-level evaluation...")
            patch_results = self._evaluate_patches(model, test_dataset)
            results['patch_evaluation'] = patch_results

        # Full image evaluation
        if full_image_evaluation:
            logger.info("Running full image evaluation...")
            full_image_results = self._evaluate_full_images(model, test_dataset, output_dir)
            results['full_image_evaluation'] = full_image_results

        # Save results
        self._save_evaluation_results(results, output_dir)

        return results

    def _evaluate_patches(self, model: keras.Model, test_dataset: SUTCrackPatchDataset) -> Dict[str, Any]:
        """Evaluate model on patch-level data."""
        # Create test TF dataset
        test_tf_dataset = test_dataset.create_tf_dataset(
            batch_size=32,
            shuffle=False,
            repeat=False
        )

        all_predictions = {task: [] for task in self.tasks}
        all_ground_truths = {task: [] for task in self.tasks}

        logger.info("Collecting patch predictions...")
        for batch_images, batch_labels in tqdm(test_tf_dataset):
            # Get model predictions
            batch_preds = model(batch_images, training=False)

            if isinstance(batch_preds, dict):
                predictions = batch_preds
            else:
                # Single task output
                predictions = {self.tasks[0]: batch_preds}

            # Process each task
            batch_size = tf.shape(batch_images)[0]

            for i in range(batch_size):
                # Detection
                if 'detection' in self.tasks and 'detection' in predictions:
                    # Process detection predictions and ground truth
                    det_pred = predictions['detection'][i:i + 1]
                    det_gt = batch_labels['detection'][i]

                    # Decode predictions (simplified)
                    all_predictions['detection'].append({'detections': []})  # Placeholder
                    all_ground_truths['detection'].append({'bboxes': []})  # Placeholder

                # Segmentation
                if 'segmentation' in self.tasks and 'segmentation' in predictions:
                    seg_pred = predictions['segmentation'][i].numpy()
                    seg_gt = batch_labels['segmentation'][i].numpy()

                    all_predictions['segmentation'].append(seg_pred)
                    all_ground_truths['segmentation'].append(seg_gt)

                # Classification
                if 'classification' in self.tasks and 'classification' in predictions:
                    cls_pred = float(predictions['classification'][i].numpy())
                    cls_gt = int(batch_labels['classification'][i].numpy())

                    all_predictions['classification'].append(cls_pred)
                    all_ground_truths['classification'].append(cls_gt)

        # Calculate metrics for each task
        patch_results = {}

        if 'detection' in self.tasks:
            det_metrics = self.detection_evaluator.compute_metrics(
                all_predictions['detection'], all_ground_truths['detection']
            )
            patch_results['detection'] = det_metrics

        if 'segmentation' in self.tasks:
            seg_metrics = self.segmentation_evaluator.compute_metrics(
                all_predictions['segmentation'], all_ground_truths['segmentation']
            )
            patch_results['segmentation'] = seg_metrics

        if 'classification' in self.tasks:
            cls_metrics = self.classification_evaluator.compute_metrics(
                all_predictions['classification'], all_ground_truths['classification']
            )
            patch_results['classification'] = cls_metrics

        return patch_results

    def _evaluate_full_images(
            self,
            model: keras.Model,
            test_dataset: SUTCrackPatchDataset,
            output_dir: str
    ) -> Dict[str, Any]:
        """Evaluate model on full images."""
        # Create inference engine
        config = InferenceConfig(
            patch_size=256,
            overlap=64,
            batch_size=16
        )
        inference_engine = FullImageInference(model, config)

        # Get unique images from test dataset
        image_paths = list(set(ann.image_path for ann in test_dataset.annotations))
        logger.info(f"Evaluating on {len(image_paths)} full images...")

        # Run inference on all images
        inference_results = []
        for image_path in tqdm(image_paths, desc="Processing full images"):
            try:
                result = inference_engine.predict_image(image_path)
                result['image_path'] = image_path
                inference_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")

        # Create visualizations
        viz_dir = os.path.join(output_dir, 'full_image_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Save sample visualizations
        for i, result in enumerate(inference_results[:10]):  # First 10 images
            self._create_full_image_visualization(result, viz_dir, f"sample_{i}")

        # Calculate image-level metrics
        image_metrics = self._calculate_image_level_metrics(inference_results, test_dataset)

        return {
            'num_images_processed': len(inference_results),
            'metrics': image_metrics,
            'sample_results': inference_results[:5]  # Store first 5 for inspection
        }

    def _calculate_image_level_metrics(
            self,
            inference_results: List[Dict[str, Any]],
            test_dataset: SUTCrackPatchDataset
    ) -> Dict[str, Any]:
        """Calculate metrics at image level."""
        # Create mapping from image path to ground truth
        gt_mapping = {ann.image_path: ann for ann in test_dataset.annotations}

        image_metrics = {}

        # Classification metrics (image-level)
        if 'classification' in self.tasks:
            cls_predictions = []
            cls_ground_truths = []

            for result in inference_results:
                image_path = result['image_path']
                if image_path in gt_mapping:
                    gt_ann = gt_mapping[image_path]

                    # Prediction
                    cls_score = result.get('classification', {}).get('score', 0.0)
                    cls_predictions.append(cls_score)

                    # Ground truth
                    has_crack = 1 if gt_ann.has_crack else 0
                    cls_ground_truths.append(has_crack)

            if cls_predictions:
                cls_metrics = self.classification_evaluator.compute_metrics(
                    cls_predictions, cls_ground_truths
                )
                image_metrics['classification'] = cls_metrics

        return image_metrics

    def _create_full_image_visualization(
            self,
            result: Dict[str, Any],
            output_dir: str,
            filename: str
    ):
        """Create visualization for full image result."""
        image_path = result['image_path']

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # Detection results
        det_image = image.copy()
        detections = result.get('detections', [])
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']

            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(det_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(det_image, f"Crack: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        axes[0, 1].imshow(det_image)
        axes[0, 1].set_title(f"Detections (Count: {len(detections)})")
        axes[0, 1].axis('off')

        # Segmentation
        if 'segmentation' in result and result['segmentation'] is not None:
            seg_mask = result['segmentation']
            axes[1, 0].imshow(seg_mask, cmap='hot')
            axes[1, 0].set_title("Segmentation Mask")
            axes[1, 0].axis('off')

            # Overlay
            seg_overlay = image.copy().astype(np.float32)
            crack_pixels = seg_mask > 0.5
            seg_overlay[crack_pixels] = seg_overlay[crack_pixels] * 0.6 + np.array([255, 0, 0]) * 0.4
            seg_overlay = np.clip(seg_overlay, 0, 255).astype(np.uint8)

            axes[1, 1].imshow(seg_overlay)
            axes[1, 1].set_title("Segmentation Overlay")
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "No Segmentation", ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, "No Segmentation", ha='center', va='center')
            axes[1, 1].axis('off')

        # Add classification info
        if 'classification' in result:
            cls_info = result['classification']
            score = cls_info.get('score', 0.0)
            prediction = cls_info.get('prediction', 0)

            fig.suptitle(
                f"Classification: {'Crack' if prediction else 'No Crack'} "
                f"(Score: {score:.3f})",
                fontsize=16
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """Save comprehensive evaluation results."""
        # Save JSON results
        results_file = os.path.join(output_dir, 'evaluation_results.json')

        # Create serializable copy
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Create summary report
        self._create_summary_report(results, output_dir)

        # Create metric plots
        self._create_metric_plots(results, output_dir)

    def _create_summary_report(self, results: Dict[str, Any], output_dir: str):
        """Create human-readable summary report."""
        report_file = os.path.join(output_dir, 'evaluation_summary.txt')

        with open(report_file, 'w') as f:
            f.write("YOLOv12 Multi-Task Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Tasks: {', '.join(results['tasks'])}\n\n")

            # Patch-level results
            if 'patch_evaluation' in results:
                f.write("Patch-Level Evaluation:\n")
                f.write("-" * 30 + "\n")

                patch_results = results['patch_evaluation']
                for task, metrics in patch_results.items():
                    f.write(f"\n{task.title()} Metrics:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")

            # Full image results
            if 'full_image_evaluation' in results:
                f.write("\n\nFull Image Evaluation:\n")
                f.write("-" * 30 + "\n")

                full_results = results['full_image_evaluation']
                f.write(f"Images processed: {full_results['num_images_processed']}\n")

                if 'metrics' in full_results:
                    for task, metrics in full_results['metrics'].items():
                        f.write(f"\n{task.title()} Metrics:\n")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {metric}: {value:.4f}\n")

    def _create_metric_plots(self, results: Dict[str, Any], output_dir: str):
        """Create visualization plots for metrics."""
        plots_dir = os.path.join(output_dir, 'metric_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Create comparison plots for different evaluation levels
        if 'patch_evaluation' in results and 'full_image_evaluation' in results:
            self._plot_metric_comparison(results, plots_dir)

    def _plot_metric_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot comparison between patch and full image metrics."""
        patch_results = results.get('patch_evaluation', {})
        full_results = results.get('full_image_evaluation', {}).get('metrics', {})

        # Focus on classification metrics which are available at both levels
        if 'classification' in patch_results and 'classification' in full_results:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']

            patch_values = [patch_results['classification'].get(m, 0) for m in metrics]
            full_values = [full_results['classification'].get(m, 0) for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, patch_values, width, label='Patch-level')
            ax.bar(x + width / 2, full_values, width, label='Full Image')

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Classification Metrics: Patch vs Full Image')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metric_comparison.png'), dpi=150)
            plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv12 Multi-Task Model')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to SUT-Crack dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--tasks', nargs='+',
                        choices=['detection', 'segmentation', 'classification'],
                        default=['detection', 'segmentation', 'classification'],
                        help='Tasks to evaluate')
    parser.add_argument('--patch-size', type=int, default=256,
                        help='Patch size for evaluation')
    parser.add_argument('--no-patch-eval', action='store_true',
                        help='Skip patch-level evaluation')
    parser.add_argument('--no-full-image-eval', action='store_true',
                        help='Skip full image evaluation')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = SUTCrackPatchDataset(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patches_per_image=8,
        include_segmentation=True
    )

    # Create evaluator
    evaluator = MultiTaskEvaluator(args.tasks)

    # Run evaluation
    try:
        results = evaluator.evaluate_model(
            model_path=args.model_path,
            test_dataset=test_dataset,
            output_dir=args.output_dir,
            patch_evaluation=not args.no_patch_eval,
            full_image_evaluation=not args.no_full_image_eval
        )

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()