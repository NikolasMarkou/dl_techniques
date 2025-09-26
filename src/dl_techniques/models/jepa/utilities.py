"""
JEPA Masking Utilities for Semantic Block-Based Masking.

This module implements the sophisticated masking strategy that makes JEPA effective:
multi-block semantic masking with spatial distribution constraints and progressive
curriculum learning for optimal representation learning.
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Tuple, List, Optional, Dict
import numpy as np

from dl_techniques.utils.logger import logger


class JEPAMaskingStrategy:
    """
    Advanced masking strategy for JEPA training with semantic block-based approach.

    This class implements the core masking innovation that differentiates JEPA from
    other masked models: instead of random patch masking, it creates large semantic
    blocks that force the model to learn high-level representations rather than
    low-level texture completion.

    Args:
        img_size: Input image dimensions (height, width).
        patch_size: Patch size for tokenization.
        num_mask_blocks: Number of semantic blocks to mask.
        block_size_range: Range of block sizes as fraction of image (min, max).
        aspect_ratio_range: Range of aspect ratios for blocks (min, max).
        context_coverage: Minimum fraction of image that must remain visible.
        progressive_masking: Enable curriculum learning with progressive masking.
        spatial_distribution_weight: Weight for spatial distribution constraint.

    Example:
        ```python
        masker = JEPAMaskingStrategy(
            img_size=(224, 224),
            patch_size=16,
            num_mask_blocks=4,
            block_size_range=(0.15, 0.20),
            context_coverage=0.85
        )

        # Generate masks for batch
        context_mask, target_mask = masker.generate_masks(batch_size=32)
        ```
    """

    def __init__(
            self,
            img_size: Tuple[int, int],
            patch_size: int,
            num_mask_blocks: int = 4,
            block_size_range: Tuple[float, float] = (0.15, 0.20),
            aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
            context_coverage: float = 0.85,
            progressive_masking: bool = True,
            spatial_distribution_weight: float = 1.0
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_mask_blocks = num_mask_blocks
        self.block_size_range = block_size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.context_coverage = context_coverage
        self.progressive_masking = progressive_masking
        self.spatial_distribution_weight = spatial_distribution_weight

        # Compute patch grid dimensions
        self.patch_h = img_size[0] // patch_size
        self.patch_w = img_size[1] // patch_size
        self.num_patches = self.patch_h * self.patch_w

        # Progressive masking curriculum
        self.current_difficulty = 0.0  # 0 = easy, 1 = hard

        logger.info(f"Initialized JEPA masking: {self.patch_h}x{self.patch_w} patches, "
                    f"{num_mask_blocks} blocks, coverage≥{context_coverage}")

    def set_difficulty(self, progress: float):
        """
        Set curriculum learning difficulty.

        Args:
            progress: Training progress from 0 to 1.
        """
        if self.progressive_masking:
            # Start easy (smaller blocks, fewer of them) and progress to hard
            self.current_difficulty = min(1.0, progress * 2.0)  # Ramp up quickly

    def _sample_block_parameters(self) -> Tuple[float, float, float]:
        """
        Sample block size, aspect ratio, and orientation.

        Returns:
            Tuple of (scale, aspect_ratio, rotation).
        """
        # Adjust block size based on curriculum
        if self.progressive_masking:
            min_scale, max_scale = self.block_size_range
            # Start with smaller blocks, increase size with difficulty
            scale_range = (
                min_scale + (max_scale - min_scale) * self.current_difficulty * 0.3,
                min_scale + (max_scale - min_scale) * (0.7 + 0.3 * self.current_difficulty)
            )
        else:
            scale_range = self.block_size_range

        scale = np.random.uniform(scale_range[0], scale_range[1])
        aspect_ratio = np.random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

        return scale, aspect_ratio, 0.0  # No rotation for simplicity

    def _create_block_mask(
            self,
            scale: float,
            aspect_ratio: float,
            center_h: int,
            center_w: int
    ) -> np.ndarray:
        """
        Create a rectangular block mask centered at given position.

        Args:
            scale: Block size as fraction of image.
            aspect_ratio: Block aspect ratio (width/height).
            center_h: Center position in patch coordinates (height).
            center_w: Center position in patch coordinates (width).

        Returns:
            Boolean mask array of shape (patch_h, patch_w).
        """
        # Compute block dimensions in patches
        block_area = scale * self.num_patches
        block_h = int(np.sqrt(block_area / aspect_ratio))
        block_w = int(np.sqrt(block_area * aspect_ratio))

        # Clamp to valid dimensions
        block_h = max(1, min(block_h, self.patch_h))
        block_w = max(1, min(block_w, self.patch_w))

        # Create mask
        mask = np.zeros((self.patch_h, self.patch_w), dtype=bool)

        # Compute bounds
        h_start = max(0, center_h - block_h // 2)
        h_end = min(self.patch_h, h_start + block_h)
        w_start = max(0, center_w - block_w // 2)
        w_end = min(self.patch_w, w_start + block_w)

        mask[h_start:h_end, w_start:w_end] = True

        return mask

    def _sample_block_centers(self, avoid_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Sample centers for mask blocks with spatial distribution constraints.

        Args:
            avoid_mask: Existing mask to avoid overlap with.

        Returns:
            List of (center_h, center_w) coordinates.
        """
        centers = []

        # For better spatial distribution, divide image into regions
        if self.spatial_distribution_weight > 0:
            # Create grid of preferred regions
            region_h = self.patch_h // 2
            region_w = self.patch_w // 2
            regions = [(i, j) for i in range(2) for j in range(2)]
            np.random.shuffle(regions)

        for block_idx in range(self.num_mask_blocks):
            max_attempts = 50
            best_center = None
            best_overlap = float('inf')

            for attempt in range(max_attempts):
                if self.spatial_distribution_weight > 0 and block_idx < len(regions):
                    # Bias toward specific region for better distribution
                    region_i, region_j = regions[block_idx]
                    center_h = np.random.randint(
                        region_i * region_h,
                        (region_i + 1) * region_h
                    )
                    center_w = np.random.randint(
                        region_j * region_w,
                        (region_j + 1) * region_w
                    )
                else:
                    # Random placement
                    center_h = np.random.randint(0, self.patch_h)
                    center_w = np.random.randint(0, self.patch_w)

                # Check overlap with existing masks and avoid regions
                overlap = 0
                if avoid_mask is not None:
                    scale, aspect_ratio, _ = self._sample_block_parameters()
                    temp_mask = self._create_block_mask(scale, aspect_ratio, center_h, center_w)
                    overlap = np.sum(temp_mask & avoid_mask)

                if overlap < best_overlap:
                    best_overlap = overlap
                    best_center = (center_h, center_w)

                # Accept if good enough
                if overlap == 0:
                    break

            if best_center is not None:
                centers.append(best_center)

        return centers

    @tf.function
    def _ensure_context_coverage(
            self,
            target_mask: tf.Tensor,
            min_coverage: float
    ) -> tf.Tensor:
        """
        Ensure minimum context coverage by removing mask regions if necessary.

        Args:
            target_mask: Boolean mask tensor of shape (batch, num_patches).
            min_coverage: Minimum fraction that must remain unmasked.

        Returns:
            Adjusted mask ensuring minimum coverage.
        """
        batch_size = tf.shape(target_mask)[0]

        # Compute current coverage
        masked_ratio = tf.reduce_mean(tf.cast(target_mask, tf.float32), axis=1)
        context_ratio = 1.0 - masked_ratio

        # Find batches with insufficient context
        insufficient_context = context_ratio < min_coverage

        if tf.reduce_any(insufficient_context):
            # For batches with insufficient context, randomly unmask some regions
            target_ratio = 1.0 - min_coverage

            # Generate random priorities for each patch
            random_priorities = tf.random.uniform(tf.shape(target_mask))

            # Sort to get top-k masks to keep
            num_to_keep = tf.cast(target_ratio * tf.cast(self.num_patches, tf.float32), tf.int32)

            # Use top-k to select which masks to keep
            _, top_k_indices = tf.nn.top_k(
                tf.where(target_mask, random_priorities, -1.0),
                k=num_to_keep,
                sorted=False
            )

            # Create new mask
            new_mask = tf.zeros_like(target_mask, dtype=tf.bool)
            batch_indices = tf.range(batch_size)[:, None]
            indices = tf.stack([
                tf.tile(batch_indices, [1, num_to_keep]),
                top_k_indices
            ], axis=2)

            updates = tf.ones([batch_size, num_to_keep], dtype=tf.bool)
            new_mask = tf.tensor_scatter_nd_update(new_mask, indices, updates)

            # Apply correction only where needed
            target_mask = tf.where(
                insufficient_context[:, None],
                new_mask,
                target_mask
            )

        return target_mask

    def generate_masks(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate context and target masks for a batch.

        Args:
            batch_size: Number of samples in batch.

        Returns:
            Tuple of (context_mask, target_mask) where True indicates visible/masked.
            Both have shape (batch_size, num_patches).
        """
        context_masks = []
        target_masks = []

        for _ in range(batch_size):
            # Initialize masks
            context_mask = np.ones((self.patch_h, self.patch_w), dtype=bool)
            target_mask = np.zeros((self.patch_h, self.patch_w), dtype=bool)

            # Sample block centers with spatial distribution
            centers = self._sample_block_centers()

            # Create mask blocks
            for center_h, center_w in centers:
                scale, aspect_ratio, _ = self._sample_block_parameters()
                block_mask = self._create_block_mask(scale, aspect_ratio, center_h, center_w)

                # Update masks
                target_mask = target_mask | block_mask
                context_mask = context_mask & (~block_mask)

            # Flatten to patch sequence format
            context_masks.append(context_mask.flatten())
            target_masks.append(target_mask.flatten())

        # Convert to tensors
        context_mask = tf.constant(np.array(context_masks), dtype=tf.bool)
        target_mask = tf.constant(np.array(target_masks), dtype=tf.bool)

        # Ensure minimum context coverage
        target_mask = self._ensure_context_coverage(target_mask, self.context_coverage)
        context_mask = ~target_mask

        return context_mask, target_mask

    def visualize_masks(
            self,
            context_mask: tf.Tensor,
            target_mask: tf.Tensor,
            sample_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize masks for debugging and analysis.

        Args:
            context_mask: Context mask tensor.
            target_mask: Target mask tensor.
            sample_idx: Sample index to visualize.

        Returns:
            Tuple of (context_image, target_image) as numpy arrays.
        """
        # Extract single sample
        ctx = context_mask[sample_idx].numpy()
        tgt = target_mask[sample_idx].numpy()

        # Reshape to 2D
        ctx_2d = ctx.reshape(self.patch_h, self.patch_w)
        tgt_2d = tgt.reshape(self.patch_h, self.patch_w)

        # Create visualization images
        context_img = np.zeros((self.patch_h, self.patch_w, 3))
        context_img[ctx_2d] = [0, 1, 0]  # Green for context
        context_img[~ctx_2d] = [0.3, 0.3, 0.3]  # Gray for masked

        target_img = np.zeros((self.patch_h, self.patch_w, 3))
        target_img[tgt_2d] = [1, 0, 0]  # Red for target
        target_img[~tgt_2d] = [0.3, 0.3, 0.3]  # Gray for unmasked

        return context_img, target_img

    def get_mask_statistics(
            self,
            context_mask: tf.Tensor,
            target_mask: tf.Tensor
    ) -> Dict[str, float]:
        """
        Compute masking statistics for monitoring.

        Args:
            context_mask: Context mask tensor.
            target_mask: Target mask tensor.

        Returns:
            Dictionary of statistics.
        """
        context_ratio = tf.reduce_mean(tf.cast(context_mask, tf.float32))
        target_ratio = tf.reduce_mean(tf.cast(target_mask, tf.float32))

        # Check for overlap (should be zero)
        overlap = tf.reduce_sum(tf.cast(context_mask & target_mask, tf.float32))
        total_coverage = context_ratio + target_ratio

        stats = {
            "context_ratio": float(context_ratio.numpy()),
            "target_ratio": float(target_ratio.numpy()),
            "total_coverage": float(total_coverage.numpy()),
            "overlap_patches": float(overlap.numpy()),
            "difficulty": self.current_difficulty,
        }

        return stats


class VideoMaskingStrategy(JEPAMaskingStrategy):
    """
    Extended masking strategy for video data (V-JEPA).

    Maintains spatial consistency across temporal dimension while allowing
    for motion-aware masking patterns.
    """

    def __init__(self, num_frames: int = 8, temporal_consistency: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.temporal_consistency = temporal_consistency

    def generate_video_masks(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate spatiotemporally consistent masks for video.

        Returns masks of shape (batch_size, num_frames, num_patches).
        """
        context_masks = []
        target_masks = []

        for _ in range(batch_size):
            # Generate base spatial mask
            spatial_context, spatial_target = self.generate_masks(1)
            spatial_context = spatial_context[0]  # Remove batch dim
            spatial_target = spatial_target[0]

            # Extend to temporal dimension
            frame_context_masks = []
            frame_target_masks = []

            for frame_idx in range(self.num_frames):
                if np.random.random() < self.temporal_consistency:
                    # Use consistent spatial pattern
                    frame_context_masks.append(spatial_context)
                    frame_target_masks.append(spatial_target)
                else:
                    # Generate new pattern for motion modeling
                    new_context, new_target = self.generate_masks(1)
                    frame_context_masks.append(new_context[0])
                    frame_target_masks.append(new_target[0])

            context_masks.append(tf.stack(frame_context_masks))
            target_masks.append(tf.stack(frame_target_masks))

        return tf.stack(context_masks), tf.stack(target_masks)


class AudioMaskingStrategy:
    """
    Masking strategy adapted for audio spectrograms (A-JEPA).

    Applies time-frequency aware masking with curriculum learning
    from simple frequency bands to complex time-frequency patterns.
    """

    def __init__(
            self,
            spec_shape: Tuple[int, int],  # (freq_bins, time_frames)
            patch_size: Tuple[int, int] = (8, 8),
            time_mask_ratio: float = 0.15,
            freq_mask_ratio: float = 0.15,
            curriculum_stages: int = 3
    ):
        self.spec_shape = spec_shape
        self.patch_size = patch_size
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0

        # Compute patch dimensions
        self.patch_freq = spec_shape[0] // patch_size[0]
        self.patch_time = spec_shape[1] // patch_size[1]
        self.num_patches = self.patch_freq * self.patch_time

    def set_curriculum_stage(self, stage: int):
        """Set current curriculum learning stage."""
        self.current_stage = min(stage, self.curriculum_stages - 1)

    def generate_audio_masks(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate time-frequency aware masks for audio spectrograms."""
        # Implementation would go here - simplified for space
        # This would implement progressive masking from frequency bands
        # to time segments to complex time-frequency patterns
        pass