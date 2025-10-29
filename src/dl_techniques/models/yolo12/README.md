# YOLOv12: A Keras 3 Multi-Task Learning Framework

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of a **YOLOv12-based multi-task learning model**. This framework is designed to simultaneously perform object detection, instance segmentation, and image classification using a shared, efficient feature extraction backbone. It is built for flexibility, allowing any combination of tasks to be enabled or disabled, and supports separate class configurations for each task.

This implementation uses the Keras Functional API with named dictionary outputs, providing a clean and interpretable interface for complex multi-task scenarios. It is fully serializable and works seamlessly across TensorFlow, PyTorch, and JAX backends.

---

## Table of Contents

1. [Overview: What is YOLOv12 Multi-Task and Why It Matters](#1-overview-what-is-yolov12-multi-task-and-why-it-matters)
2. [The Problem Multi-Task Learning Solves](#2-the-problem-multi-task-learning-solves)
3. [How YOLOv12 Multi-Task Works: Core Concepts](#3-how-yolov12-multi-task-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is YOLOv12 Multi-Task and Why It Matters

### What is YOLOv12 Multi-Task?

This is a powerful computer vision model that can perform multiple, distinct tasks from a single input image in one forward pass. It leverages the highly efficient **YOLOv12 backbone and neck** as a shared feature extractor and attaches specialized "heads" to perform different jobs.

### Key Innovations

1.  **Shared Backbone Efficiency**: By sharing the most computationally expensive part of the network (the feature extractor), the model avoids redundant calculations. This makes it significantly faster and more memory-efficient than running separate models for each task.
2.  **Implicit Regularization**: Training multiple tasks simultaneously encourages the shared backbone to learn more general and robust features. One task's learning process can help regularize and improve the performance of another, a phenomenon known as "knowledge transfer."
3.  **Flexible Task Composition**: The architecture is fully modular. You can enable or disable tasks like `DETECTION`, `SEGMENTATION`, and `CLASSIFICATION` with a simple configuration flag. Crucially, it supports **separate class counts for each task** (e.g., 80 detection classes, but only 1 segmentation class).
4.  **Named Dictionary Outputs**: For clarity in multi-task scenarios, the model returns a dictionary where keys correspond to task names (e.g., `{'detection': ..., 'segmentation': ...}`).

### Why It Matters

**The Single-Task Model Problem**:
```
Problem: Analyze a scene comprehensively (e.g., find all cars, segment the road, and classify the weather).
Traditional Approach:
  1. Train a model for object detection (find cars).
  2. Train a separate model for semantic segmentation (segment road).
  3. Train a third model for image classification (classify weather).
  4. Limitation: This is slow (3x inference time), memory-intensive, and misses
     out on potential synergies between tasks.
```

**YOLOv12 Multi-Task's Solution**:
```
YOLOv12's Approach:
  1. Use one powerful backbone to understand the image's content.
  2. Attach three small, specialized heads to the backbone's features.
  3. In a single forward pass, get all three outputs simultaneously.
  4. Benefit: Drastically improved efficiency and the potential for one task's
     knowledge to improve another's (e.g., knowing where cars are can help segment
     the road better).
```

---

## 2. The Problem Multi-Task Learning Solves

### The Inefficiency of Siloed Models

In many real-world applications, a single image needs to answer multiple questions. Deploying separate, specialized models for each question leads to a "pipeline" of models that is slow, costly, and difficult to maintain.

```
┌─────────────────────────────────────────────────────────────┐
│  The Problem of Model Pipelines                             │
│                                                             │
│  - Redundant Computation: Each model independently runs a   │
│    heavy backbone to extract features from the same image.  │
│                                                             │
│  - High Latency: Total inference time is the sum of all     │
│    models in the pipeline.                                  │
│                                                             │
│  - Large Memory Footprint: Each model's weights must be     │
│    loaded into memory, which can be prohibitive on edge     │
│    devices.                                                 │
│                                                             │
│  - No Knowledge Sharing: The models are trained in isolation│
│    and cannot benefit from each other's learned features.   │
└─────────────────────────────────────────────────────────────┘
```

Multi-task learning solves these issues by unifying the feature extraction process and optimizing for multiple objectives at once.

---

## 3. How YOLOv12 Multi-Task Works: Core Concepts

### The Shared Backbone and Task-Specific Heads

The architecture is simple and powerful, consisting of a single trunk with multiple branches.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YOLOv12 Multi-Task Complete Data Flow                │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: SHARED FEATURE EXTRACTION
─────────────────────────────────
Input Image (H, W, 3)
    │
    ├─► YOLOv12 Backbone & Neck (Feature Extractor)
    │   (Generates a pyramid of feature maps at different scales: P3, P4, P5)
    │
    └─► Multi-Scale Features [P3, P4, P5]


STEP 2: PARALLEL TASK-SPECIFIC HEADS
───────────────────────────────────
Multi-Scale Features
    │
    ├─► [Enabled] Detection Head ───► Detection Output (BBoxes, Classes)
    │
    ├─► [Enabled] Segmentation Head ──► Segmentation Mask
    │
    └─► [Enabled] Classification Head ──► Classification Logits


STEP 3: COMBINED OUTPUT
───────────────────────
Individual Task Outputs
    │
    └─► Dictionary Output: {'detection': ..., 'segmentation': ...}
```

---

## 4. Architecture Deep Dive

### 4.1 `YOLOv12FeatureExtractor`

-   **Purpose**: The shared engine of the model. It contains the YOLOv12 backbone (CSP-style blocks) and neck (PANet).
-   **Functionality**: It takes a raw image and produces three feature maps at different scales:
    -   **P3**: High-resolution, fine-grained features (at 1/8th of input size).
    -   **P4**: Medium-resolution, balanced features (at 1/16th of input size).
    -   **P5**: Low-resolution, semantic-rich features (at 1/32nd of input size).
    -   This feature pyramid allows each head to access the most appropriate level of detail for its task.

### 4.2 `YOLOv12DetectionHead`

-   **Purpose**: To perform object detection.
-   **Functionality**: It takes the multi-scale features and passes them through separate convolutional branches for bounding box regression and object classification. It uses Distribution Focal Loss (DFL) for more precise box localization.

### 4.3 `YOLOv12SegmentationHead`

-   **Purpose**: To perform pixel-level semantic segmentation.
-   **Functionality**: It acts as a lightweight decoder. Starting from the most semantic feature map (P5), it progressively upsamples and fuses features from the P4 and P3 levels via skip connections, gradually rebuilding a full-resolution segmentation mask.

### 4.4 `YOLOv12ClassificationHead`

-   **Purpose**: To perform global image classification.
-   **Functionality**: It applies global average and max pooling to the multi-scale features to create a fixed-size feature vector, which is then passed through a simple MLP to produce classification logits.

---

## 5. Quick Start Guide

### Installation

```bash
# Install Keras 3 and a backend (e.g., tensorflow)
pip install keras tensorflow numpy
```

### Your First Multi-Task Model (30 seconds)

Let's build a model for both detection and segmentation.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.yolo12.multitask import YOLOv12MultiTask
from dl_techniques.layers.vision_heads.task_types import TaskType

# 1. Create a multi-task model for detection and segmentation
# Configure separate class counts for each task.
model = YOLOv12MultiTask(
    num_detection_classes=80,    # e.g., for COCO detection
    num_segmentation_classes=1,  # e.g., for binary road segmentation
    task_config=[TaskType.DETECTION, TaskType.SEGMENTATION],
    scale='s'  # Use the 'small' variant
)

# 2. Compile the model
# For multi-task, provide a dictionary of losses and weights.
model.compile(
    optimizer="adamw",
    loss={
        "detection": some_detection_loss_fn,
        "segmentation": keras.losses.BinaryCrossentropy(from_logits=True)
    },
    loss_weights={"detection": 1.0, "segmentation": 0.5}
)
print("✅ YOLOv12 Multi-Task model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 2
dummy_images = np.random.rand(batch_size, 640, 640, 3).astype("float32")

# 4. Run inference
# The output is a dictionary with keys matching the task names.
predictions = model.predict(dummy_images)
print(f"\n✅ Prediction keys: {list(predictions.keys())}")
print(f"Detection output shape: {predictions['detection'].shape}")
print(f"Segmentation output shape: {predictions['segmentation'].shape}")
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`YOLOv12MultiTask`** | `...yolo12.multitask.YOLOv12MultiTask` | The main Keras `Model` class assembling the architecture. |
| **`...FeatureExtractor`** | `...yolo12.feature_extractor.YOLOv12FeatureExtractor` | The shared backbone and neck. |
| **`...DetectionHead`** | `...layers.yolo12_heads.YOLOv12DetectionHead` | The head for object detection. |
| **`...SegmentationHead`**|`...layers.yolo12_heads.YOLOv12SegmentationHead`| The head for semantic segmentation. |
| **`...ClassificationHead`**|`...layers.yolo12_heads.YOLOv12ClassificationHead`| The head for image classification. |
| **`TaskType`** | `...layers.vision_heads.task_types.TaskType` | Enum for defining which tasks to enable. |

---

## 7. Configuration & Model Variants

The feature extractor backbone comes in several scales, trading speed for accuracy.

| Scale | `depth_multiple` | `width_multiple` | Description |
|:---:|:---:|:---:|:---|
| **`n`** | 0.50 | 0.25 | **Nano**: Fastest, for edge devices. |
| **`s`** | 0.50 | 0.50 | **Small**: A good balance for general use. |
| **`m`** | 0.50 | 1.00 | **Medium**: More accurate, for server-side inference. |
| **`l`** | 1.00 | 1.00 | **Large**: High accuracy, for demanding tasks. |
| **`x`** | 1.00 | 1.50 | **Extra-Large**: Maximum accuracy, for research. |

---

## 8. Comprehensive Usage Examples

### Example 1: Detection Only

The framework gracefully handles single-task configurations.

```python
from dl_techniques.models.yolo12.multitask import create_yolov12_multitask

# The output will be a single tensor, not a dictionary.
model = create_yolov12_multitask(
    num_detection_classes=80,
    tasks="detection", # or tasks=TaskType.DETECTION
    scale="m"
)
model.compile(optimizer="adamw", loss=some_detection_loss_fn)
```

### Example 2: Three-Task Model (Det + Seg + Class)

Configure all three heads with different class counts.

```python
model = create_yolov12_multitask(
    num_detection_classes=20,    # PASCAL VOC classes
    num_segmentation_classes=1,  # Binary segmentation
    num_classification_classes=5,  # 5 scene types
    tasks=["detection", "segmentation", "classification"],
    scale="l"
)

# Compile with a loss for each head
model.compile(
    optimizer="adamw",
    loss={
        "detection": det_loss,
        "segmentation": seg_loss,
        "classification": class_loss
    }
)
```

---

## 9. Advanced Usage Patterns

### Transfer Learning and Fine-Tuning

A common workflow is to pre-train on a large dataset (like COCO) and then fine-tune on a smaller, specialized dataset.

```python
# 1. Load a pre-trained model (e.g., from COCO)
pretrained_model = YOLOv12MultiTask(
    num_detection_classes=80, # COCO classes
    num_segmentation_classes=80,
    task_config=["detection", "segmentation"],
    scale="s"
)
# pretrained_model.load_weights('coco_weights.h5')

# 2. Freeze the backbone for initial fine-tuning
feature_extractor = pretrained_model.get_feature_extractor()
feature_extractor.trainable = False

# 3. Create a new model for a different task (e.g., 1 class)
# The heads will be randomly initialized, while the backbone is frozen.
fine_tune_model = YOLOv12MultiTask(
    num_detection_classes=1,
    num_segmentation_classes=1,
    task_config=["detection", "segmentation"],
    scale="s"
)
# Copy the frozen backbone weights
fine_tune_model.get_feature_extractor().set_weights(feature_extractor.get_weights())

# 4. Compile and fine-tune the heads
fine_tune_model.compile(...)
# fine_tune_model.fit(..., epochs=10)

# 5. Unfreeze the backbone and fine-tune end-to-end with a low learning rate
fine_tune_model.get_feature_extractor().trainable = True
fine_tune_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5), ...)
# fine_tune_model.fit(..., epochs=5)
```

---

## 10. Performance Optimization

-   **Mixed Precision**: The convolutional nature of the model makes it an excellent candidate for `mixed_float16` training, which can provide significant speedups on modern GPUs.
-   **Choose the Right Scale**: Select the smallest model scale (`n`, `s`, `m`, etc.) that meets your accuracy requirements to maximize speed and minimize memory usage.
-   **Input Resolution**: Reducing the input resolution (e.g., from 640x640 to 416x416) will dramatically improve performance at the cost of detecting smaller objects.

---

## 11. Training and Best Practices

-   **Loss Weighting**: When training multiple tasks, the `loss_weights` in `model.compile()` are crucial. You may need to experiment to find the right balance. A common strategy is to give the primary task a weight of `1.0` and adjust the weights of auxiliary tasks.
-   **Data Augmentation**: Use standard vision augmentations like random flips, crops, color jittering, and mosaic augmentation.
-   **Optimizer**: **AdamW** is a strong default.
-   **Learning Rate Schedule**: A **cosine decay** schedule is highly recommended for stable training.

---

## 12. Serialization & Deployment

The `YOLOv12MultiTask` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train a multi-task model
model = create_yolov12_multitask(
    num_detection_classes=1, num_segmentation_classes=1,
    tasks=["detection", "segmentation"]
)
# ... model.fit(...)

# Save the entire model to a single file
model.save('my_yolo_multitask.keras')

# Load the model, including all heads and custom layers
loaded_model = keras.models.load_model('my_yolo_multitask.keras')
print("✅ YOLOv12 Multi-Task model loaded successfully!")
```

---

## 13. Testing & Validation

A `pytest` test to ensure the multi-task output structure and serialization are correct.

```python
import pytest
import numpy as np
import keras
import tempfile
import os
from dl_techniques.models.yolo12.multitask import YOLOv12MultiTask

def test_multitask_serialization_cycle():
    """CRITICAL TEST: Ensures a multi-task model can be saved and loaded."""
    model = YOLOv12MultiTask(
        num_detection_classes=5,
        num_segmentation_classes=2,
        task_config=["detection", "segmentation"],
        scale="n"
    )
    dummy_input = np.random.rand(2, 256, 256, 3).astype("float32")
    
    original_preds = model.predict(dummy_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.keras")
        model.save(filepath)
        loaded_model = keras.models.load_model(filepath)

    loaded_preds = loaded_model.predict(dummy_input)

    assert isinstance(original_preds, dict)
    assert "detection" in original_preds
    assert "segmentation" in original_preds
    np.testing.assert_allclose(
        original_preds["detection"], loaded_preds["detection"], rtol=1e-5
    )
    np.testing.assert_allclose(
        original_preds["segmentation"], loaded_preds["segmentation"], rtol=1e-5
    )
    print("✓ Serialization cycle test passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: How do I structure my `y_true` data for `model.fit()`?**

-   **Answer**: When using multiple heads, `y_true` should be a dictionary with keys matching the output names of the model.
    ```python
    # For a model with detection and segmentation heads
    y_true = {
        "detection": y_true_detection_data,
        "segmentation": y_true_segmentation_data
    }
    model.fit(x_train, y_true)
    ```

### Frequently Asked Questions

**Q: Can I add my own custom task head?**

A: Yes. The functional design makes this straightforward. You would get the multi-scale features from the `YOLOv12FeatureExtractor` and then pass them to your custom head layer, adding its output to the `outputs` dictionary before creating the final `keras.Model`.

**Q: Why use YOLOv12 as a backbone?**

A: YOLO architectures are renowned for their excellent balance of speed and accuracy. The CSP-inspired blocks and PANet neck are highly efficient and effective at producing the multi-scale features needed for dense prediction tasks like detection and segmentation.

---

## 15. Technical Details

### Multi-Scale Feature Fusion

The `YOLOv12FeatureExtractor` uses a Path Aggregation Network (PANet) neck. This involves two pathways:
1.  **Top-Down Path**: High-level semantic features from deeper layers are upsampled and fused with features from shallower layers, propagating strong semantic information to finer resolutions.
2.  **Bottom-Up Path**: The resulting features are then processed in a bottom-up direction, propagating strong localization features from shallower layers back to deeper ones.

This bidirectional fusion creates feature maps that are rich in both semantic meaning and precise localization information at every scale, which is ideal for multi-task learning.

---

## 16. Citation

This implementation is based on the principles of the YOLO family of architectures and common multi-task learning patterns. For foundational concepts, please refer to the original YOLO papers by Joseph Redmon et al. and subsequent works by the Ultralytics team.