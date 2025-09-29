# Vision Task Heads Integration Guide

## Overview

This guide demonstrates how to integrate vision task heads with foundation models using the `dl_techniques` framework. The task heads provide a model-agnostic interface that can work with any vision foundation model (ViT, ResNet, EfficientNet, YOLO, DINOv2, etc.).

## Architecture Overview

```
Foundation Model → Feature Maps → Task Head → Task Outputs
       ↓              ↓              ↓           ↓
  (CNN, ViT)    [B, H, W, C]   (Vision Head)  {masks, boxes}
```

- **Foundation Model**: Produces visual feature representations
- **Feature Maps**: Shape `[batch_size, height, width, channels]` or `[batch_size, patches, features]`
- **Task Head**: Transforms features for specific vision tasks
- **Task Outputs**: Task-specific predictions (masks, boxes, depth maps, etc.)

## Core Concepts

### 1. Input/Output Contract

All vision heads expect inputs in one of these formats:

```python
# Format 1: Direct tensor - CNN feature maps
features = backbone(images)  # [batch_size, height, width, channels]

# Format 2: ViT-style patch features 
patch_features = vit_backbone(images)  # [batch_size, num_patches, hidden_dim]

# Format 3: Multi-scale features for FPN-style architectures
multi_scale_features = [
    feat_p3,  # [batch_size, H/8, W/8, C1]
    feat_p4,  # [batch_size, H/16, W/16, C2]
    feat_p5   # [batch_size, H/32, W/32, C3]
]
```

### 2. Task Head Outputs

Each head produces a dictionary with task-specific outputs:

```python
# Detection tasks
outputs = {
    'classifications': tensor,  # [batch_size, num_anchors, num_classes]
    'regressions': tensor       # [batch_size, num_anchors, 4]
}

# Segmentation
outputs = {
    'masks': tensor,    # [batch_size, height, width, num_classes]
    'logits': tensor    # Raw logits before softmax
}

# Depth estimation
outputs = {
    'depth': tensor,       # [batch_size, height, width, 1]
    'confidence': tensor   # [batch_size, height, width, 1]
}

# Instance segmentation
outputs = {
    'classifications': tensor,  # From detection branch
    'regressions': tensor,     # Bounding boxes
    'instance_masks': tensor   # Instance-level masks
}
```

## Integration Examples

### Example 1: ViT with Classification Head

```python
import keras
from dl_techniques.models.vit import ViT
from dl_techniques.layers.vision_heads import create_vision_head, TaskType

# Step 1: Create foundation model
vit = ViT(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    num_classes=None  # Don't add classification head in backbone
)

# Step 2: Create task head
classification_head = create_vision_head(
    TaskType.CLASSIFICATION,
    num_classes=1000,  # ImageNet classes
    hidden_dim=768,
    use_global_pooling=True,
    pooling_type='avg',
    use_attention=True,
    attention_type='multi_head'
)

# Step 3: Build complete model
@keras.saving.register_keras_serializable()
class ImageClassifier(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = vit
        self.head = classification_head
    
    def call(self, images, training=None):
        # Extract features from ViT
        features = self.backbone(images, training=training)
        
        # Pass to classification head
        outputs = self.head(features, training=training)
        return outputs

# Usage
model = ImageClassifier()
images = keras.random.uniform((32, 224, 224, 3))
outputs = model(images)
# outputs['logits'] shape: (32, 1000)
# outputs['probabilities'] shape: (32, 1000)
```

### Example 2: ResNet with Detection Head

```python
from dl_techniques.models.resnet import ResNet50
from dl_techniques.layers.vision_heads import DetectionHead

# Step 1: Create ResNet backbone with FPN
backbone = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(640, 640, 3)
)

# Step 2: Create detection head
detection_head = DetectionHead(
    num_classes=80,  # COCO classes
    num_anchors=9,   # 3 scales × 3 aspect ratios
    bbox_dims=4,
    hidden_dim=256,
    use_ffn=True,
    ffn_type='swiglu'
)

# Step 3: Build detection model
@keras.saving.register_keras_serializable()
class ObjectDetector(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.fpn = self._build_fpn()  # Feature Pyramid Network
        self.head = detection_head
    
    def _build_fpn(self):
        # Simplified FPN implementation
        return keras.Sequential([
            keras.layers.Conv2D(256, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])
    
    def call(self, images, training=None):
        # Extract multi-scale features
        features = self.backbone(images, training=training)
        
        # Apply FPN
        fpn_features = self.fpn(features, training=training)
        
        # Detection head
        outputs = self.head(fpn_features, training=training)
        
        return outputs

# Usage
detector = ObjectDetector()
images = keras.random.uniform((8, 640, 640, 3))
outputs = detector(images)
# outputs['classifications'] shape: (8, num_anchors, 80)
# outputs['regressions'] shape: (8, num_anchors, 4)
```

### Example 3: DINOv2 with Segmentation Head

```python
from dl_techniques.models.dino import DINOv2Model
from dl_techniques.layers.vision_heads import SegmentationHead

# Step 1: Create DINOv2 backbone
dinov2 = DINOv2Model(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    depth=24,
    num_heads=16
)

# Step 2: Create segmentation head
segmentation_head = SegmentationHead(
    num_classes=21,  # Pascal VOC classes
    upsampling_factor=16,  # Upsample from patches to image resolution
    use_skip_connections=True,
    hidden_dim=512,
    use_attention=True,
    attention_type='cbam'
)

# Step 3: Build segmentation model
@keras.saving.register_keras_serializable()
class SemanticSegmenter(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = dinov2
        self.head = segmentation_head
        
        # Reshape patches to spatial format
        self.patch_to_spatial = keras.layers.Lambda(
            lambda x: keras.ops.reshape(x, (-1, 37, 37, 1024))  # 518/14 = 37
        )
    
    def call(self, images, training=None):
        # Get patch features from DINOv2
        patch_features = self.backbone(images, training=training)
        
        # Reshape to spatial format
        spatial_features = self.patch_to_spatial(patch_features)
        
        # Segmentation head
        seg_masks = self.head(spatial_features, training=training)
        
        return seg_masks

# Usage
segmenter = SemanticSegmenter()
images = keras.random.uniform((4, 518, 518, 3))
seg_output = segmenter(images)
# seg_output shape: (4, 518, 518, 21)
```

### Example 4: Depth Estimation with EfficientNet

```python
from dl_techniques.models.efficientnet import EfficientNetB4
from dl_techniques.layers.vision_heads import DepthEstimationHead

# Step 1: Create EfficientNet backbone
efficientnet = EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=(384, 384, 3)
)

# Step 2: Create depth estimation head
depth_head = DepthEstimationHead(
    output_channels=1,
    min_depth=0.1,
    max_depth=100.0,
    use_log_depth=True,
    hidden_dim=256,
    use_attention=False
)

# Step 3: Build depth estimation model
@keras.saving.register_keras_serializable()
class DepthEstimator(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = efficientnet
        self.head = depth_head
    
    def call(self, images, training=None):
        features = self.backbone(images, training=training)
        depth_outputs = self.head(features, training=training)
        return depth_outputs

# Usage
depth_model = DepthEstimator()
images = keras.random.uniform((8, 384, 384, 3))
outputs = depth_model(images)
# outputs['depth'] shape: (8, 384, 384, 1)
# outputs['confidence'] shape: (8, 384, 384, 1)
```

### Example 5: Multi-Task Learning with YOLO

```python
from dl_techniques.layers.vision_heads import create_multi_task_head, TaskType
from dl_techniques.models.yolo12 import YOLOv12FeatureExtractor

# Step 1: Create YOLO backbone
yolo_backbone = YOLOv12FeatureExtractor(
    variant='n',  # nano variant
    input_shape=(640, 640, 3),
    width_mult=0.25,
    depth_mult=0.33
)

# Step 2: Define multiple tasks
task_configs = {
    'detection': {
        'task_type': TaskType.DETECTION,
        'num_classes': 80,
        'num_anchors': 3
    },
    'segmentation': {
        'task_type': TaskType.SEGMENTATION,
        'num_classes': 80,
        'upsampling_factor': 4
    },
    'depth': {
        'task_type': TaskType.DEPTH_ESTIMATION,
        'min_depth': 0.1,
        'max_depth': 50.0
    }
}

# Step 3: Create multi-task head
multi_task_head = create_multi_task_head(
    task_configs,
    shared_backbone_dim=256,
    use_task_specific_attention=True
)

# Step 4: Build multi-task model
@keras.saving.register_keras_serializable()
class MultiTaskVisionModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = yolo_backbone
        self.heads = multi_task_head
    
    def call(self, images, training=None):
        # Extract multi-scale features
        p3, p4, p5 = self.backbone(images, training=training)
        
        # Use P5 for all tasks (simplified)
        features = p5
        
        # Run all task heads
        outputs = self.heads(features, training=training)
        return outputs

# Usage
multi_model = MultiTaskVisionModel()
images = keras.random.uniform((4, 640, 640, 3))
all_outputs = multi_model(images)
# all_outputs['detection']['classifications'] shape: (4, num_anchors, 80)
# all_outputs['segmentation'] shape: (4, 640, 640, 80)
# all_outputs['depth']['depth'] shape: (4, 640, 640, 1)
```

### Example 6: Instance Segmentation

```python
from dl_techniques.layers.vision_heads import InstanceSegmentationHead

# Create instance segmentation head
instance_head = InstanceSegmentationHead(
    num_classes=80,
    num_instances=100,
    mask_size=(28, 28),
    hidden_dim=256,
    use_attention=True,
    attention_type='cbam'
)

@keras.saving.register_keras_serializable()
class InstanceSegmenter(keras.Model):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = instance_head
    
    def call(self, images, training=None):
        features = self.backbone(images, training=training)
        outputs = self.head(features, training=training)
        
        # Post-process masks to full resolution
        if not training:
            outputs['instance_masks'] = self.upsample_masks(
                outputs['instance_masks']
            )
        
        return outputs
    
    def upsample_masks(self, masks):
        # Upsample from 28x28 to image resolution
        return keras.layers.UpSampling2D(size=(16, 16))(masks)

# Usage with any backbone
backbone = create_backbone('resnet50')  # Your choice
model = InstanceSegmenter(backbone)
outputs = model(images)
# outputs['classifications'], outputs['regressions'], outputs['instance_masks']
```

## Advanced Configurations

### Attention Mechanisms

Different tasks benefit from different attention strategies:

```python
# CBAM for spatial tasks (segmentation, depth)
spatial_head = create_vision_head(
    TaskType.SEGMENTATION,
    num_classes=150,  # ADE20K classes
    use_attention=True,
    attention_type='cbam'  # Channel and spatial attention
)

# Multi-head attention for global understanding
global_head = create_vision_head(
    TaskType.CLASSIFICATION,
    num_classes=1000,
    use_attention=True,
    attention_type='multi_head',
    hidden_dim=768
)

# Differential attention for fine-grained tasks
fine_head = create_vision_head(
    TaskType.KEYPOINT_DETECTION,
    num_classes=17,  # COCO keypoints
    use_attention=True,
    attention_type='differential'
)
```

### FFN Options

Enhance heads with different feed-forward networks:

```python
# SwiGLU for best performance
high_perf_head = create_vision_head(
    TaskType.DETECTION,
    num_classes=80,
    use_ffn=True,
    ffn_type='swiglu',
    ffn_expansion_factor=8
)

# GLU for efficiency
efficient_head = create_vision_head(
    TaskType.DETECTION,
    num_classes=80,
    use_ffn=True,
    ffn_type='glu',
    ffn_expansion_factor=2
)
```

### Normalization Strategies

```python
# Layer normalization (standard)
standard_head = create_vision_head(
    task_type,
    normalization_type='layer_norm'
)

# Batch normalization (CNN-friendly)
cnn_head = create_vision_head(
    task_type,
    normalization_type='batch_norm'
)

# RMS normalization (efficient)
efficient_head = create_vision_head(
    task_type,
    normalization_type='rms_norm'
)
```

## Task-Specific Configurations

### Detection Configuration

```python
from dl_techniques.vision.heads import HeadConfiguration

# Get optimized detection config
detection_config = HeadConfiguration.get_default_config(TaskType.DETECTION)
# {'num_classes': 80, 'num_anchors': 9, 'bbox_dims': 4, ...}

# High-performance variant
hp_config = HeadConfiguration.get_high_performance_config(TaskType.DETECTION)
detection_head = create_vision_head(TaskType.DETECTION, **hp_config)

# Efficient variant for mobile
efficient_config = HeadConfiguration.get_efficient_config(TaskType.DETECTION)
mobile_head = create_vision_head(TaskType.DETECTION, **efficient_config)
```

### Segmentation with Skip Connections

```python
# For U-Net style architectures
segmentation_head = SegmentationHead(
    num_classes=19,  # Cityscapes
    use_skip_connections=True,
    hidden_dim=256
)

# In model, provide multi-scale features
def call(self, images, training=None):
    # Extract features at multiple scales
    feat_1_4 = self.layer1(images)   # 1/4 resolution
    feat_1_8 = self.layer2(feat_1_4)  # 1/8 resolution
    feat_1_16 = self.layer3(feat_1_8) # 1/16 resolution
    feat_1_32 = self.layer4(feat_1_16) # 1/32 resolution
    
    # Pass all scales to head
    multi_scale = [feat_1_4, feat_1_8, feat_1_16, feat_1_32]
    seg_output = self.seg_head(multi_scale, training=training)
    return seg_output
```

## Training Considerations

### Loss Functions

Different tasks require appropriate loss functions:

```python
# Detection - Focal loss for classification, smooth L1 for regression
def detection_loss(y_true, y_pred):
    cls_loss = focal_loss(
        y_true['classifications'],
        y_pred['classifications']
    )
    reg_loss = smooth_l1_loss(
        y_true['regressions'],
        y_pred['regressions']
    )
    return cls_loss + reg_loss

# Segmentation - Cross entropy with class weights
class_weights = compute_class_weights(train_labels)
seg_loss = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    class_weight=class_weights
)

# Depth - Scale-invariant loss
def depth_loss(y_true, y_pred):
    d = keras.ops.log(y_true) - keras.ops.log(y_pred)
    term1 = keras.ops.mean(keras.ops.square(d))
    term2 = keras.ops.square(keras.ops.mean(d)) * 0.5
    return term1 - term2
```

### Multi-Task Loss Balancing

```python
class MultiTaskLoss(keras.losses.Loss):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {
            'detection': 1.0,
            'segmentation': 1.0,
            'depth': 1.0
        }
    
    def call(self, y_true, y_pred):
        total_loss = 0
        
        for task, weight in self.task_weights.items():
            if task in y_pred:
                task_loss = self.compute_task_loss(
                    task, y_true[task], y_pred[task]
                )
                total_loss += weight * task_loss
        
        return total_loss
```

### Data Augmentation

```python
# Task-aware augmentation
def augment_for_task(images, labels, task_type):
    if task_type == TaskType.DETECTION:
        # Preserve bounding box validity
        images, labels = random_flip_with_boxes(images, labels)
        images, labels = random_crop_with_boxes(images, labels)
    
    elif task_type == TaskType.SEGMENTATION:
        # Apply same transforms to masks
        images, labels = random_crop_and_resize(images, labels)
        images = color_jitter(images)  # Masks unaffected
    
    elif task_type == TaskType.DEPTH_ESTIMATION:
        # Avoid geometric transforms that break depth
        images = color_augmentation_only(images)
    
    return images, labels
```

## Performance Optimization

### Mixed Precision Training

```python
# Enable mixed precision for faster training
keras.mixed_precision.set_global_policy('mixed_float16')

# Ensure head outputs are float32 for stability
class StableVisionHead(BaseVisionHead):
    def call(self, inputs, training=None):
        outputs = super().call(inputs, training)
        
        # Cast outputs to float32 for loss computation
        return {
            k: keras.ops.cast(v, 'float32') 
            for k, v in outputs.items()
        }
```

### Gradient Checkpointing

```python
# For memory-efficient training with large models
@keras.saving.register_keras_serializable()
class CheckpointedHead(BaseVisionHead):
    def call(self, inputs, training=None):
        if training:
            # Use gradient checkpointing for memory efficiency
            return keras.ops.recompute_grad(
                lambda x: super().call(x, training),
                inputs
            )
        return super().call(inputs, training)
```

### Model Pruning

```python
# Prune heads for deployment
def prune_vision_head(head, sparsity=0.5):
    """Prune vision head weights for efficiency."""
    pruning_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=sparsity,
        begin_step=0,
        end_step=1000
    )
    
    pruned_head = keras_pruning.prune_low_magnitude(
        head,
        pruning_schedule=pruning_schedule
    )
    
    return pruned_head
```

## Deployment Considerations

### ONNX Export

```python
# Export vision model with heads to ONNX
def export_to_onnx(model, save_path):
    dummy_input = keras.random.uniform((1, 224, 224, 3))
    
    # Trace model
    traced = keras.export.trace(model, dummy_input)
    
    # Export to ONNX
    keras.export.onnx.export(
        traced,
        save_path,
        input_names=['images'],
        output_names=['predictions'],
        opset_version=13
    )
```

### TensorRT Optimization

```python
# Optimize for NVIDIA GPUs
def optimize_with_tensorrt(onnx_path):
    import tensorrt as trt
    
    builder = trt.Builder()
    config = builder.create_builder_config()
    
    # Enable FP16 for better performance
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build optimized engine
    engine = builder.build_engine(network, config)
    return engine
```

## Troubleshooting

### Common Issues

1. **Feature Dimension Mismatch**
```python
# Error: Expected hidden_dim=256 but got 512
# Solution: Match head hidden_dim to backbone output
backbone_dim = backbone.output_shape[-1]
head = create_vision_head(
    task_type,
    hidden_dim=backbone_dim
)
```

2. **Resolution Mismatch**
```python
# Error: Segmentation mask doesn't match input size
# Solution: Adjust upsampling factor
input_size = 512
backbone_stride = 32
upsampling_factor = backbone_stride

seg_head = SegmentationHead(
    num_classes=21,
    upsampling_factor=upsampling_factor
)
```

3. **Multi-Scale Feature Format**
```python
# Error: Expected list of features but got single tensor
# Solution: Wrap single feature in list or disable skip connections
seg_head = SegmentationHead(
    num_classes=21,
    use_skip_connections=False  # For single-scale features
)
```

4. **Anchor Configuration**
```python
# Error: Anchor dimensions don't match predictions
# Solution: Ensure consistency between anchors and head
num_anchors = len(scales) * len(aspect_ratios)
detection_head = DetectionHead(
    num_classes=80,
    num_anchors=num_anchors  # Must match anchor generator
)
```

## Best Practices

1. **Start Simple**: Begin with default configurations
2. **Profile First**: Identify bottlenecks before optimization
3. **Task-Specific Tuning**: Each task has unique requirements
4. **Validate Outputs**: Check output shapes and ranges
5. **Monitor Metrics**: Use task-appropriate evaluation metrics

## Summary

The Vision Task Head system provides:

1. **Model Agnostic**: Works with any vision backbone
2. **Task Optimized**: Specialized architectures for each task
3. **Highly Configurable**: Extensive customization options
4. **Production Ready**: Full serialization and optimization support
5. **Multi-Task Support**: Unified interface for complex models

Key integration steps:
1. Choose vision backbone (ViT, ResNet, YOLO, etc.)
2. Select appropriate task head
3. Match dimensions between backbone and head
4. Configure task-specific parameters
5. Connect backbone outputs to head inputs
6. Process task outputs for your application

The heads handle the complexity of task-specific transformations while maintaining a clean, consistent interface across all vision tasks.