"""
Losses Module
===========================

This package provides a comprehensive collection of advanced and specialized loss
functions implemented in Keras 3 for a wide range of deep learning tasks. Each
loss function is designed to be a self-contained, serializable Keras object,
making it easy to integrate into any Keras workflow.

The collection includes losses for:
    - Robust Regression (Huber, MASE, SMAPE)
    - Probabilistic Forecasting (MQLoss)
    - Imbalanced Classification (AnyLoss Framework, FocalUncertaintyLoss)
    - Self-Supervised Learning (CLIP, DINO, iBOT, SigLIP)
    - Information-Theoretic Regularization (Goodhart, DecoupledInformationLoss)
    - Generative Adversarial Networks (WassersteinLoss, WGAN-GP)
    - Specialized Architectures (Capsule Networks, YOLOv12, NanoVLM)
    - Computer Vision Tasks (Segmentation, Feature Alignment, Affine Invariance)

All implementations are compatible with Python 3.11+ and Keras 3.8+.

Example Usage:
--------------
You can directly import and use any loss function in your model compilation step:

.. code-block:: python

    import keras
    from dl_techniques.losses import F1Loss, DINOLoss

    # Example for imbalanced classification
    model.compile(
        optimizer='adam',
        loss=F1Loss(amplifying_scale=73.0),
        metrics=['accuracy']
    )

    # Example for self-supervised learning
    student_model.compile(
        optimizer='adam',
        loss=DINOLoss(out_dim=65536)
    )
"""

# ---------------------------------------------------------------------
# Define the public API for the losses module
# ---------------------------------------------------------------------

from .affine_invariant_loss import AffineInvariantLoss

from .any_loss import (
    AnyLoss,
    ApproximationFunction,
    AccuracyLoss,
    F1Loss,
    FBetaLoss,
    GeometricMeanLoss,
    BalancedAccuracyLoss,
    WeightedCrossEntropyWithAnyLoss,
)

from .brier_spiegelhalters_ztest_loss import (
    BrierScoreLoss,
    SpiegelhalterZLoss,
    CombinedCalibrationLoss,
    BrierScoreMetric,
    SpiegelhalterZMetric,
)

from .capsule_margin_loss import (
    CapsuleMarginLoss,
    capsule_margin_loss,
    analyze_margin_loss_components,
)

from .clip_contrastive_loss import CLIPContrastiveLoss

from .clustering_loss import (
    ClusteringLoss,
    ClusteringMetrics,
    ClusteringMetricsCallback,
    compute_clustering_metrics,
)

from .decoupled_information_loss import (
    DecoupledInformationLoss,
    analyze_decoupled_information_loss,
)

from .dino_loss import DINOLoss, iBOTPatchLoss, KoLeoLoss

from .feature_alignment_loss import FeatureAlignmentLoss

from .focal_uncertainty_loss import (
    FocalUncertaintyLoss,
    analyze_focal_uncertainty_loss,
)

from .goodhart_loss import GoodhartAwareLoss, analyze_loss_components

from .hrm_loss import HRMLoss, StableMaxCrossEntropy, create_hrm_loss

from .huber_loss import HuberLoss

from .mase_loss import MASELoss, mase_metric

from .quantile_loss import MQLoss, QuantileLoss

from .nano_vlm_loss import NanoVLMLoss

from .segmentation_loss import (
    LossConfig as SegmentationLossConfig, # Renamed to avoid conflicts
    SegmentationLosses,
    create_loss_function as create_segmentation_loss_function, # Renamed
)

# from .siglip_contrastive_loss
from .siglip_contrastive_loss import (
    SigLIPContrastiveLoss,
    AdaptiveSigLIPLoss,
    HybridContrastiveLoss,
    create_siglip_loss,
    create_adaptive_siglip_loss,
    create_hybrid_loss,
)

# from .smape_loss
from .smape_loss import SMAPELoss, smape_metric

# from .tabm_loss
from .tabm_loss import TabMLoss

# from .wasserstein_loss
from .wasserstein_loss import (
    WassersteinLoss,
    WassersteinGradientPenaltyLoss,
    WassersteinDivergence,
    compute_gradient_penalty,
    create_wgan_losses,
    create_wgan_gp_losses,
)

# from .yolo12_multitask_loss
from .yolo12_multitask_loss import (
    YOLOv12MultiTaskLoss,
    create_yolov12_multitask_loss,
    create_yolov12_coco_loss,
    create_yolov12_crack_loss,
)


# Define __all__ for a clean public API
__all__ = [
    # affine_invariant_loss
    "AffineInvariantLoss",
    # any_loss
    "AnyLoss",
    "ApproximationFunction",
    "AccuracyLoss",
    "F1Loss",
    "FBetaLoss",
    "GeometricMeanLoss",
    "BalancedAccuracyLoss",
    "WeightedCrossEntropyWithAnyLoss",
    # brier_spiegelhalters_ztest_loss
    "BrierScoreLoss",
    "SpiegelhalterZLoss",
    "CombinedCalibrationLoss",
    "BrierScoreMetric",
    "SpiegelhalterZMetric",
    # capsule_margin_loss
    "CapsuleMarginLoss",
    "capsule_margin_loss",
    "analyze_margin_loss_components",
    # clip_contrastive_loss
    "CLIPContrastiveLoss",
    # clustering_loss
    "ClusteringLoss",
    "ClusteringMetrics",
    "ClusteringMetricsCallback",
    "compute_clustering_metrics",
    # decoupled_information_loss
    "DecoupledInformationLoss",
    "analyze_decoupled_information_loss",
    # dino_loss
    "DINOLoss",
    "iBOTPatchLoss",
    "KoLeoLoss",
    # feature_alignment_loss
    "FeatureAlignmentLoss",
    # focal_uncertainty_loss
    "FocalUncertaintyLoss",
    "analyze_focal_uncertainty_loss",
    # goodhart_loss
    "GoodhartAwareLoss",
    "analyze_loss_components",
    # hrm_loss
    "HRMLoss",
    "StableMaxCrossEntropy",
    "create_hrm_loss",
    # huber_loss
    "HuberLoss",
    # mase_loss
    "MASELoss",
    "mase_metric",
    # Quantile loss
    "MQLoss",
    "QuantileLoss",
    # nano_vlm_loss
    "NanoVLMLoss",
    # segmentation_loss
    "SegmentationLossConfig",
    "SegmentationLosses",
    "create_segmentation_loss_function",
    # siglip_contrastive_loss
    "SigLIPContrastiveLoss",
    "AdaptiveSigLIPLoss",
    "HybridContrastiveLoss",
    "create_siglip_loss",
    "create_adaptive_siglip_loss",
    "create_hybrid_loss",
    # smape_loss
    "SMAPELoss",
    "smape_metric",
    # tabm_loss
    "TabMLoss",
    # wasserstein_loss
    "WassersteinLoss",
    "WassersteinGradientPenaltyLoss",
    "WassersteinDivergence",
    "compute_gradient_penalty",
    "create_wgan_losses",
    "create_wgan_gp_losses",
    # yolo12_multitask_loss
    "YOLOv12MultiTaskLoss",
    "create_yolov12_multitask_loss",
    "create_yolov12_coco_loss",
    "create_yolov12_crack_loss",
]