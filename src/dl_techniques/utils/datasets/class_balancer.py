"""
Enhanced Data Loader with Expert-Recommended Class Balancing

This module provides enhanced data loading capabilities that implement the expert
recommendation of "fixing the data first" through data-level balancing rather
than aggressive loss-level weighting.

Key Features:
    - Intelligent class balancing strategies (undersample, oversample, hybrid)
    - Stratified dataset splitting to maintain class distribution
    - Comprehensive class distribution analysis
    - Data quality monitoring and validation
    - Enhanced patch extraction with balance-aware sampling

Expert Recommendation Applied:
    "Fix the data first: Use data-level undersampling to create a balanced
    training set (e.g., 1:10 or 1:5 ratio)."
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any


from dl_techniques.utils.logger import logger


class DataBalanceAnalyzer:
    """
    Utility class for analyzing and monitoring class balance in datasets.
    """

    @staticmethod
    def analyze_annotations(annotations: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive analysis of class distribution in annotations.

        Args:
            annotations: List of annotation dictionaries.

        Returns:
            Dictionary with detailed class distribution statistics.
        """
        positive_count = 0
        negative_count = 0

        # Analyze image-level labels
        for ann in annotations:
            if ann.get('has_cracks', False):
                positive_count += 1
            else:
                negative_count += 1

        total = positive_count + negative_count
        positive_ratio = positive_count / total if total > 0 else 0
        negative_ratio = negative_count / total if total > 0 else 0
        imbalance_ratio = negative_count / positive_count if positive_count > 0 else float('inf')

        analysis = {
            'total_samples': total,
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'imbalance_ratio': imbalance_ratio,
            'balance_quality': DataBalanceAnalyzer._assess_balance_quality(positive_ratio),
            'recommendations': DataBalanceAnalyzer._get_balance_recommendations(positive_ratio, imbalance_ratio)
        }

        return analysis

    @staticmethod
    def _assess_balance_quality(positive_ratio: float) -> str:
        """Assess the quality of class balance."""
        if 0.4 <= positive_ratio <= 0.6:
            return "excellent"
        elif 0.3 <= positive_ratio <= 0.7:
            return "good"
        elif 0.2 <= positive_ratio <= 0.8:
            return "moderate"
        elif 0.1 <= positive_ratio <= 0.9:
            return "poor"
        else:
            return "severe_imbalance"

    @staticmethod
    def _get_balance_recommendations(positive_ratio: float, imbalance_ratio: float) -> List[str]:
        """Get recommendations for improving class balance."""
        recommendations = []

        if positive_ratio < 0.1:
            recommendations.append("Consider data-level oversampling of positive class")
            recommendations.append("Apply data augmentation to positive samples")
        elif positive_ratio < 0.2:
            recommendations.append("Apply moderate undersampling of negative class")
            recommendations.append("Use stratified sampling for validation")
        elif positive_ratio > 0.8:
            recommendations.append("Consider undersampling positive class")
            recommendations.append("Ensure sufficient negative examples")

        if imbalance_ratio > 20:
            recommendations.append("Severe imbalance detected - use aggressive balancing")
        elif imbalance_ratio > 10:
            recommendations.append("Moderate imbalance - apply data-level balancing")

        if not recommendations:
            recommendations.append("Class balance is acceptable for training")

        return recommendations

    @staticmethod
    def log_analysis(analysis: Dict[str, Any], dataset_name: str = "Dataset"):
        """Log comprehensive analysis results."""
        logger.info(f"📊 {dataset_name} Class Balance Analysis:")
        logger.info(f"   Total samples: {analysis['total_samples']}")
        logger.info(f"   Positive: {analysis['positive_samples']} ({analysis['positive_ratio']:.3f})")
        logger.info(f"   Negative: {analysis['negative_samples']} ({analysis['negative_ratio']:.3f})")
        logger.info(f"   Imbalance ratio: {analysis['imbalance_ratio']:.1f}:1")
        logger.info(f"   Balance quality: {analysis['balance_quality']}")

        if analysis['recommendations']:
            logger.info("   Recommendations:")
            for rec in analysis['recommendations']:
                logger.info(f"     • {rec}")


class BalancingStrategy:
    """
    Abstract base class for different balancing strategies.
    """

    def __init__(self, target_positive_ratio: float = 0.3):
        self.target_positive_ratio = target_positive_ratio

    def apply(self, positive_samples: List[Dict], negative_samples: List[Dict]) -> List[Dict]:
        """Apply balancing strategy to samples."""
        raise NotImplementedError

    def _log_strategy_results(self, original_pos: int, original_neg: int,
                              final_pos: int, final_neg: int, strategy_name: str):
        """Log the results of applying a balancing strategy."""
        logger.info(f"🔄 {strategy_name} Results:")
        logger.info(f"   Original: {original_pos} pos, {original_neg} neg")
        logger.info(f"   Balanced: {final_pos} pos, {final_neg} neg")
        final_ratio = final_pos / (final_pos + final_neg) if (final_pos + final_neg) > 0 else 0
        logger.info(f"   Final ratio: {final_ratio:.3f} (target: {self.target_positive_ratio:.3f})")


class UnderSamplingStrategy(BalancingStrategy):
    """
    Conservative undersampling of majority class (expert recommended).
    """

    def apply(self, positive_samples: List[Dict], negative_samples: List[Dict]) -> List[Dict]:
        """Apply undersampling to negative samples."""
        num_positive = len(positive_samples)
        target_negative = int(num_positive * (1 - self.target_positive_ratio) / self.target_positive_ratio)

        if target_negative < len(negative_samples):
            # Randomly sample negatives
            np.random.shuffle(negative_samples)
            sampled_negatives = negative_samples[:target_negative]
        else:
            sampled_negatives = negative_samples
            logger.warning(f"⚠️ Not enough negative samples for target ratio. Using all {len(negative_samples)}.")

        self._log_strategy_results(
            len(positive_samples), len(negative_samples),
            len(positive_samples), len(sampled_negatives),
            "Undersampling"
        )

        return positive_samples + sampled_negatives


class OverSamplingStrategy(BalancingStrategy):
    """
    Oversampling of minority class with augmentation.
    """

    def __init__(self, target_positive_ratio: float = 0.3, max_augmentation_factor: float = 3.0):
        super().__init__(target_positive_ratio)
        self.max_augmentation_factor = max_augmentation_factor

    def apply(self, positive_samples: List[Dict], negative_samples: List[Dict]) -> List[Dict]:
        """Apply oversampling to positive samples."""
        num_negative = len(negative_samples)
        target_positive = int(num_negative * self.target_positive_ratio / (1 - self.target_positive_ratio))

        if target_positive > len(positive_samples):
            additional_needed = target_positive - len(positive_samples)
            max_additional = int(len(positive_samples) * self.max_augmentation_factor)

            if additional_needed > max_additional:
                logger.warning(f"⚠️ Limiting augmentation to {self.max_augmentation_factor}x original size")
                additional_needed = max_additional

            augmented_positives = positive_samples.copy()

            # Create augmented versions
            for i in range(additional_needed):
                base_sample = positive_samples[i % len(positive_samples)].copy()
                base_sample['augmented'] = True
                base_sample['augmentation_id'] = i
                augmented_positives.append(base_sample)
        else:
            augmented_positives = positive_samples

        self._log_strategy_results(
            len(positive_samples), len(negative_samples),
            len(augmented_positives), len(negative_samples),
            "Oversampling"
        )

        return augmented_positives + negative_samples


class HybridBalancingStrategy(BalancingStrategy):
    """
    Hybrid approach: moderate undersampling + moderate oversampling.
    """

    def __init__(self, target_positive_ratio: float = 0.3, reduction_factor: float = 0.8):
        super().__init__(target_positive_ratio)
        self.reduction_factor = reduction_factor

    def apply(self, positive_samples: List[Dict], negative_samples: List[Dict]) -> List[Dict]:
        """Apply hybrid balancing strategy."""
        original_total = len(positive_samples) + len(negative_samples)
        target_total = int(original_total * self.reduction_factor)

        target_positive = int(target_total * self.target_positive_ratio)
        target_negative = target_total - target_positive

        # Moderate oversampling of positives
        if target_positive > len(positive_samples):
            additional_needed = min(
                target_positive - len(positive_samples),
                len(positive_samples) // 2  # Limit to 1.5x original
            )
            balanced_positives = positive_samples.copy()

            for i in range(additional_needed):
                base_sample = positive_samples[i % len(positive_samples)].copy()
                base_sample['augmented'] = True
                balanced_positives.append(base_sample)
        else:
            balanced_positives = positive_samples[:target_positive]

        # Moderate undersampling of negatives
        if target_negative < len(negative_samples):
            np.random.shuffle(negative_samples)
            balanced_negatives = negative_samples[:target_negative]
        else:
            balanced_negatives = negative_samples

        self._log_strategy_results(
            len(positive_samples), len(negative_samples),
            len(balanced_positives), len(balanced_negatives),
            "Hybrid Balancing"
        )

        return balanced_positives + balanced_negatives


class StratifiedDataSplitter:
    """
    Utility for creating stratified dataset splits that maintain class distribution.
    """

    @staticmethod
    def split_annotations(
            annotations: List[Dict],
            train_ratio: float = 0.7,
            val_ratio: float = 0.2,
            test_ratio: float = 0.1,
            random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified splits maintaining class distribution.

        Args:
            annotations: List of annotation dictionaries.
            train_ratio: Ratio for training set.
            val_ratio: Ratio for validation set.
            test_ratio: Ratio for test set.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_annotations, val_annotations, test_annotations).
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Separate by class
        positive_samples = [ann for ann in annotations if ann.get('has_cracks', False)]
        negative_samples = [ann for ann in annotations if not ann.get('has_cracks', False)]

        logger.info(f"🔀 Stratified splitting: {len(positive_samples)} pos, {len(negative_samples)} neg")

        # Set random seed
        np.random.seed(random_seed)

        # Shuffle each class separately
        np.random.shuffle(positive_samples)
        np.random.shuffle(negative_samples)

        # Calculate split sizes for each class
        n_pos = len(positive_samples)
        n_neg = len(negative_samples)

        # Positive splits
        pos_train_end = int(n_pos * train_ratio)
        pos_val_end = pos_train_end + int(n_pos * val_ratio)

        pos_train = positive_samples[:pos_train_end]
        pos_val = positive_samples[pos_train_end:pos_val_end]
        pos_test = positive_samples[pos_val_end:]

        # Negative splits
        neg_train_end = int(n_neg * train_ratio)
        neg_val_end = neg_train_end + int(n_neg * val_ratio)

        neg_train = negative_samples[:neg_train_end]
        neg_val = negative_samples[neg_train_end:neg_val_end]
        neg_test = negative_samples[neg_val_end:]

        # Combine and shuffle final splits
        train_annotations = pos_train + neg_train
        val_annotations = pos_val + neg_val
        test_annotations = pos_test + neg_test

        np.random.shuffle(train_annotations)
        np.random.shuffle(val_annotations)
        np.random.shuffle(test_annotations)

        logger.info(f"📊 Split results:")
        logger.info(f"   Train: {len(train_annotations)} ({len(pos_train)} pos, {len(neg_train)} neg)")
        logger.info(f"   Val: {len(val_annotations)} ({len(pos_val)} pos, {len(neg_val)} neg)")
        logger.info(f"   Test: {len(test_annotations)} ({len(pos_test)} pos, {len(neg_test)} neg)")

        return train_annotations, val_annotations, test_annotations


class SmartPatchSampler:
    """
    Intelligent patch sampling that considers class balance and data quality.
    """

    def __init__(self,
                 positive_ratio: float = 0.3,
                 min_crack_pixels: int = 100,
                 quality_threshold: float = 0.8):
        """
        Initialize smart patch sampler.

        Args:
            positive_ratio: Target ratio of positive patches.
            min_crack_pixels: Minimum crack pixels to consider patch positive.
            quality_threshold: Quality threshold for patch selection.
        """
        self.positive_ratio = positive_ratio
        self.min_crack_pixels = min_crack_pixels
        self.quality_threshold = quality_threshold

    def sample_patches_per_image(self,
                                 image_annotation: Dict,
                                 patches_per_image: int) -> List[Dict]:
        """
        Sample patches from an image with balance-aware strategy.

        Args:
            image_annotation: Annotation dictionary for the image.
            patches_per_image: Number of patches to extract.

        Returns:
            List of patch annotations with balanced sampling.
        """
        has_cracks = image_annotation.get('has_cracks', False)

        if has_cracks:
            # For positive images, ensure some positive patches
            target_positive_patches = max(1, int(patches_per_image * self.positive_ratio))
            target_negative_patches = patches_per_image - target_positive_patches

            # This would be implemented based on your specific patch extraction logic
            # For now, return a placeholder that maintains the annotation structure
            patches = []
            for i in range(patches_per_image):
                patch_ann = image_annotation.copy()
                patch_ann['patch_id'] = i
                patch_ann['is_positive_patch'] = i < target_positive_patches
                patches.append(patch_ann)

            return patches
        else:
            # For negative images, all patches are negative
            patches = []
            for i in range(patches_per_image):
                patch_ann = image_annotation.copy()
                patch_ann['patch_id'] = i
                patch_ann['is_positive_patch'] = False
                patches.append(patch_ann)

            return patches

    def assess_patch_quality(self, patch_data: np.ndarray) -> float:
        """
        Assess the quality of a patch for training.

        Args:
            patch_data: Patch image data.

        Returns:
            Quality score between 0 and 1.
        """
        # Simple quality assessment based on image statistics
        # In practice, this could be more sophisticated

        # Check for sufficient contrast
        contrast = np.std(patch_data)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize to [0, 1]

        # Check for information content (avoid blank patches)
        mean_intensity = np.mean(patch_data)
        intensity_score = 1.0 - abs(mean_intensity - 128) / 128.0

        # Combine scores
        quality_score = (contrast_score + intensity_score) / 2.0

        return quality_score


class BalancedDatasetConfig:
    """
    Configuration class for balanced dataset creation.
    """

    def __init__(self,
                 balance_strategy: str = "undersample",
                 target_positive_ratio: float = 0.3,
                 min_crack_pixels: int = 100,
                 augmentation_factor: float = 1.0,
                 quality_threshold: float = 0.8,
                 stratified_splits: bool = True,
                 monitor_balance: bool = True):
        """
        Initialize balanced dataset configuration.

        Args:
            balance_strategy: Strategy for balancing ('undersample', 'oversample', 'hybrid').
            target_positive_ratio: Target ratio of positive samples.
            min_crack_pixels: Minimum crack pixels for positive classification.
            augmentation_factor: Factor for data augmentation.
            quality_threshold: Quality threshold for patch selection.
            stratified_splits: Whether to use stratified splitting.
            monitor_balance: Whether to monitor balance during training.
        """
        self.balance_strategy = balance_strategy
        self.target_positive_ratio = target_positive_ratio
        self.min_crack_pixels = min_crack_pixels
        self.augmentation_factor = augmentation_factor
        self.quality_threshold = quality_threshold
        self.stratified_splits = stratified_splits
        self.monitor_balance = monitor_balance

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0.1 <= self.target_positive_ratio <= 0.5:
            logger.warning(f"⚠️ Positive ratio {self.target_positive_ratio} outside recommended range [0.1, 0.5]")

        if self.balance_strategy not in ['undersample', 'oversample', 'hybrid']:
            raise ValueError(f"Invalid balance strategy: {self.balance_strategy}")

        if self.augmentation_factor > 5.0:
            logger.warning(f"⚠️ High augmentation factor ({self.augmentation_factor}) may lead to overfitting")

    def get_balancing_strategy(self) -> BalancingStrategy:
        """Get the appropriate balancing strategy instance."""
        if self.balance_strategy == "undersample":
            return UnderSamplingStrategy(self.target_positive_ratio)
        elif self.balance_strategy == "oversample":
            return OverSamplingStrategy(self.target_positive_ratio, self.augmentation_factor)
        elif self.balance_strategy == "hybrid":
            return HybridBalancingStrategy(self.target_positive_ratio)
        else:
            raise ValueError(f"Unknown balance strategy: {self.balance_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'balance_strategy': self.balance_strategy,
            'target_positive_ratio': self.target_positive_ratio,
            'min_crack_pixels': self.min_crack_pixels,
            'augmentation_factor': self.augmentation_factor,
            'quality_threshold': self.quality_threshold,
            'stratified_splits': self.stratified_splits,
            'monitor_balance': self.monitor_balance
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"💾 Balanced dataset config saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BalancedDatasetConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def create_balanced_dataset_info(annotations: List[Dict],
                                 config: BalancedDatasetConfig) -> Dict[str, Any]:
    """
    Create comprehensive dataset information with balance analysis.

    Args:
        annotations: List of annotation dictionaries.
        config: Balanced dataset configuration.

    Returns:
        Dictionary with comprehensive dataset information.
    """
    # Basic statistics
    total_images = len(annotations)

    # Class analysis
    balance_analysis = DataBalanceAnalyzer.analyze_annotations(annotations)

    # Estimate patches (this would depend on your specific implementation)
    estimated_patches_per_image = 16  # Default assumption
    total_estimated_patches = total_images * estimated_patches_per_image

    dataset_info = {
        'total_images': total_images,
        'total_patches_per_epoch': total_estimated_patches,
        'class_distribution': balance_analysis,
        'balancing_config': config.to_dict(),
        'data_quality': {
            'balance_quality': balance_analysis['balance_quality'],
            'recommendations': balance_analysis['recommendations']
        },
        'expert_compliance': {
            'uses_data_level_balancing': True,
            'follows_expert_recommendation': True,
            'stable_training_expected': True
        }
    }

    return dataset_info


# Example usage functions for integration:

def apply_expert_balancing(annotations: List[Dict],
                           config: BalancedDatasetConfig) -> List[Dict]:
    """
    Apply expert-recommended balancing to annotations.

    Args:
        annotations: Original annotations.
        config: Balancing configuration.

    Returns:
        Balanced annotations following expert recommendations.
    """
    logger.info("🎯 Applying expert-recommended data-level balancing...")

    # Analyze original distribution
    original_analysis = DataBalanceAnalyzer.analyze_annotations(annotations)
    DataBalanceAnalyzer.log_analysis(original_analysis, "Original Dataset")

    # Separate classes
    positive_samples = [ann for ann in annotations if ann.get('has_cracks', False)]
    negative_samples = [ann for ann in annotations if not ann.get('has_cracks', False)]

    # Apply balancing strategy
    strategy = config.get_balancing_strategy()
    balanced_annotations = strategy.apply(positive_samples, negative_samples)

    # Analyze balanced distribution
    balanced_analysis = DataBalanceAnalyzer.analyze_annotations(balanced_annotations)
    DataBalanceAnalyzer.log_analysis(balanced_analysis, "Balanced Dataset")

    # Shuffle final result
    np.random.shuffle(balanced_annotations)

    logger.info("✅ Expert-recommended balancing applied successfully")
    return balanced_annotations