# Metrics Module

The `dl_techniques.metrics` module provides a collection of specialized Keras metrics for evaluating a wide range of advanced deep learning models, from language models and capsule networks to multimodal and multi-output architectures.

## Overview

This module offers stateful Keras `Metric` subclasses that are fully serializable and integrate seamlessly with `model.compile()`. These metrics provide more insightful performance measures than standard accuracy, especially for tasks like language modeling, vision-language alignment, and image restoration.

## Available Metrics

| Name | Class | Description | Use Case |
|------|-------|-------------|----------|
| `perplexity` | `Perplexity` | Measures prediction uncertainty in language models, defined as `exp(cross_entropy)`. A lower value is better. | Evaluating language models, neural machine translation, and speech recognition systems. |
| `clip_accuracy` | `CLIPAccuracy` | Measures top-k retrieval accuracy for contrastive vision-language models by checking similarity matrix predictions. | Evaluating the alignment quality of multimodal models like CLIP. |
| `capsule_accuracy` | `CapsuleAccuracy` | Computes classification accuracy based on the length of output capsule vectors, not probabilities. | Evaluating Capsule Networks where vector magnitude indicates presence. |
| `psnr` | `PsnrMetric` | Computes Peak Signal-to-Noise Ratio (PSNR), specifically for the primary output of multi-output models. | Evaluating image quality in restoration/super-resolution tasks with deep supervision. |
| `hrm_metrics` | `HRMMetrics` | A container for a suite of metrics for the Hierarchical Reasoning Model, including accuracy, Q-halt accuracy, and step counts. | Custom evaluation within the training loop of an HRM model. |

## Basic Usage

Most metrics can be directly added to the `metrics` list in `model.compile()`.

```python
import keras
from dl_techniques.metrics import Perplexity, CapsuleAccuracy

# For a language model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        Perplexity(from_logits=True, name='perplexity')
    ]
)

# For a Capsule Network
capsule_model.compile(
    optimizer='adam',
    loss=CapsuleMarginLoss(), # from dl_techniques.losses
    metrics=[CapsuleAccuracy()]
)
```

## Metric-Specific Parameters & Usage

### Perplexity Metric
Calculates perplexity, the exponentiated cross-entropy loss. It supports masking of padding tokens.

**Key Params:** `from_logits` (bool, default: True), `ignore_class` (int, default: None).

```python
from dl_techniques.metrics import Perplexity

# Standard perplexity for a model outputting logits
perplexity_metric = Perplexity(from_logits=True)

# Perplexity that ignores the padding token with ID 0
masked_perplexity = Perplexity(from_logits=True, ignore_class=0)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[masked_perplexity]
)
```

A stateless functional version, `perplexity()`, is also available for use in custom training loops.

### CLIP Accuracy Metric
Measures the retrieval accuracy for CLIP-style models. It expects the model to output a dictionary or tuple containing `logits_per_image` and `logits_per_text` similarity matrices.

**Key Params:** `top_k` (int, default: 1), `track_directions` (bool, default: False), `average_directions` (bool, default: True).

```python
from dl_techniques.metrics import CLIPAccuracy

# Top-1 accuracy, averaged across image-to-text and text-to-image
acc_top1 = CLIPAccuracy(top_k=1)

# Top-5 accuracy, with separate tracking for each direction
acc_top5_directional = CLIPAccuracy(top_k=5, track_directions=True)

# --- In your model definition ---
# The model must output the similarity matrices
outputs = {
    "logits_per_image": image_to_text_sim,
    "logits_per_text": text_to_image_sim
}
clip_model = keras.Model(inputs=[...], outputs=outputs)

# --- In model compilation ---
# Note: y_true is ignored by this metric
clip_model.compile(
    optimizer='adam',
    loss=CLIPContrastiveLoss(), # from dl_techniques.losses
    metrics=[acc_top1, acc_top5_directional]
)
```

When `track_directions=True`, you can access the directional accuracies during or after training:
```python
# After fitting the model
print(f"I2T Top-5 Accuracy: {acc_top5_directional.i2t_accuracy.numpy():.4f}")
print(f"T2I Top-5 Accuracy: {acc_top5_directional.t2i_accuracy.numpy():.4f}")
```

### PSNR Metric for Multi-Output Models
Computes PSNR on only the *first* output of a model that returns a list of tensors. This is ideal for deep supervision architectures where intermediate outputs are used for loss calculation but only the final output's quality is of interest.

```python
from dl_techniques.metrics import PsnrMetric

# The model returns a list of images [output_main, output_aux1, output_aux2]
multi_output_model = ...

# The PsnrMetric will automatically use only output_main for evaluation
multi_output_model.compile(
    optimizer='adam',
    loss=custom_multi_output_loss,
    metrics=[PsnrMetric(name='final_psnr')]
)
```

### HRM Metrics
`HRMMetrics` is a helper class, not a standard Keras `Metric`. It's designed to be used inside a custom training loop to manage multiple metrics specific to the Hierarchical Reasoning Model.

```python
from dl_techniques.metrics import HRMMetrics

# Instantiate the metrics container
hrm_metrics = HRMMetrics(ignore_index=-100)

class HRMModel(keras.Model):
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y_true, y_pred=y_pred)
        
        # ... apply gradients ...
        
        # Update the suite of HRM metrics
        hrm_metrics.update_state(y_true, y_pred)
        
        # Return results
        results = {"loss": loss}
        results.update(hrm_metrics.result())
        return results

    def on_epoch_begin(self, epoch, logs=None):
        # Reset metrics at the start of each epoch
        hrm_metrics.reset_state()
```