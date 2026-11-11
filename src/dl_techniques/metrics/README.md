# Metrics Module

The `dl_techniques.metrics` module provides a comprehensive collection of specialized metrics for evaluating advanced deep learning architectures. It covers language modeling, vision-language contrastive alignment, generative text evaluation, time-series forecasting, depth estimation, and probabilistic calibration.

## Architectural Design

The module is divided into two strict paradigms:
1. **Streaming Keras Metrics:** Subclasses of `keras.metrics.Metric`. These implement `update_state` and `result`, use backend-agnostic `keras.ops`, and are designed directly for `model.compile()`. They are fully serializable via `@register_dl_technique`.
2. **Pool-Level / Batch-Agnostic Evaluators:** Pure Python/NumPy functions. These are used when a metric is a functional of the entire dataset (e.g., SVD for embedding geometry, corpus-level IDF for CIDEr, or subset counting for `pass@k`) and cannot be expressed as a streaming accumulation.

## Available Metrics

| Name | Type | Description | Primary Use Case |
|------|------|-------------|------------------|
| `Perplexity` | Streaming | `exp(cross_entropy)`. Measures prediction uncertainty. | Language modeling, NMT, speech recognition. |
| `BitsPerToken` | Streaming | `log2(perplexity)`. AMP-safe float32 cross-entropy accumulator. | LLM evaluation. |
| `BrierScore` | Streaming | Proper scoring rule for probabilistic calibration. | Binary/Multi-label classification confidence. |
| `CategoricalBrierScore`| Streaming | Multi-class Brier score with a sparse algebraic fast path. | Segmentation mask calibration. |
| `CLIPAccuracy` | Streaming | Top-k retrieval accuracy via similarity matrix diagonal. | Vision-language contrastive models (CLIP). |
| `CLIPRecallAtK` | Streaming | Measures if correct match appears in top-K predictions. | Multimodal retrieval tasks. |
| `CoverageMetric` | Streaming | Empirical interval coverage for quantile forecasts. | Probabilistic time-series forecasting. |
| `PsnrMetric` | Streaming | Peak Signal-to-Noise Ratio evaluated only on primary output. | Image restoration with deep supervision. |
| `SsimMetric` | Streaming | Structural Similarity Index. (TensorFlow backend required). | Image quality assessment. |
| `VoidAwareAccuracy` | Streaming | Pixel accuracy safely ignoring a specific void class index. | Semantic segmentation with ignore masks. |
| `HalfPrecisionTopKAccuracy` | Streaming | Top-k accuracy safely casting `float16` logits to `float32`. | Mixed-precision classifiers. |
| `KeypointMatchMetric` | Streaming | Precision/Recall/F1 of mutual-nearest matches vs homography. | LightGlue-style local feature matching. |
| `AbsRelMetric` (etc) | Streaming | Depth estimation metrics masking invalid ground truth pixels. | Monocular depth estimation. |
| `generation_report` | Pool-level | Unified BLEU, ROUGE, chrF, METEOR, and CIDEr metrics. | Corpus-level text generation evaluation. |
| `ranking_metrics` | Pool-level | MRR, nDCG, and Recall@K with pessimistic tie-breaking. | Text embedding retrieval evaluation. |
| `anisotropy` (etc) | Pool-level | Embedding geometry diagnostics (uniformity, alignment, rank). | Diagnosing embedding space collapse. |
| `pass_at_k` | Pool-level | Unbiased coverage estimator (probability of success in k draws). | Code generation, best-of-k verifiers. |
| `morans_i_grid` | Pool-level | Spatial autocorrelation (Queen contiguity) without dense matrices. | Topographic map evaluation. |

## Pool-Level Coverage: `pass@k`

`pass_at_k` measures **coverage**—whether a correct solution exists in a candidate pool. This is distinct from mean success rate and requires the entire outcome matrix to evaluate correctly without bias. It is not a `keras.metrics.Metric`.

```python
import numpy as np
from dl_techniques.metrics.pass_at_k import pass_at_k, pass_at_k_curve

# (n_problems, n_samples); > 0 marks a sample as solved.
outcomes = np.array([[1, 0, 1, 0],
                     [0, 0, 0, 0]], dtype=float)

# Mean single-sample success rate
pass_at_k(outcomes, k=1)   # 0.25

# Unbiased coverage estimate (Chen et al. 2021)
pass_at_k(outcomes, k=2)   # 0.4167 (The plug-in estimator would overreport as 0.5)

# Vectorized computation across a ladder of k values
pass_at_k_curve(outcomes)  # {1: 0.25, 2: 0.4167, 4: 1.0}
```

By default, it uses the `unbiased` estimator. Always threshold partial credit upstream (e.g., `outcomes > 0.5`) before passing to this function.

## Basic Keras Integration

Streaming metrics are natively compatible with `model.compile()`. Because they are registered via `@register_dl_technique`, models saved in the `.keras` format will automatically restore these metrics upon loading, provided the library is importable.

```python
import keras
from dl_techniques.metrics import Perplexity, BitsPerToken

model.compile(
    optimizer='adamw',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        Perplexity(from_logits=True, ignore_class=-100, name='perplexity'),
        BitsPerToken(from_logits=True, ignore_class=-100, name='bpt')
    ]
)
```

## Metric-Specific Parameters & Usage

### Contrastive Vision-Language Alignment (CLIP)

The CLIP metrics evaluate dual-encoder similarity matrices. They expect a dictionary output containing `logits_per_image` and `logits_per_text`. Ground truth is implicit (the matrix diagonal).

```python
from dl_techniques.metrics import CLIPAccuracy, CLIPRecallAtK

# I2T = Image-to-Text retrieval, T2I = Text-to-Image retrieval
metrics = [
    CLIPAccuracy(direction='i2t', name='i2t_acc'),
    CLIPAccuracy(direction='t2i', name='t2i_acc'),
    CLIPRecallAtK(k=5, direction='i2t', name='i2t_r@5'),
    CLIPRecallAtK(k=10, direction='t2i', name='t2i_r@10')
]

# The model must output the similarity matrices as a dictionary
clip_model.compile(
    optimizer='adamw',
    loss=CLIPContrastiveLoss(), 
    metrics=metrics
)
```

### Probabilistic Calibration (Brier Score)

The Brier Score complements Cross-Entropy by explicitly measuring calibration (mean squared error between probabilities and targets). Lower is better.

```python
from dl_techniques.metrics import CategoricalBrierScore, BrierScore

# For multi-label classification (e.g., 80 classes, Sigmoid output)
multi_label_metric = BrierScore(from_logits=True, name="brier")

# For dense semantic segmentation masks, use the sparse fast-path
# to avoid OOM errors from one-hot tensor expansion.
segmentation_metric = CategoricalBrierScore(
    from_logits=True, 
    sparse_labels=True, 
    name="seg_brier"
)
```

### Text Generation Reporting (Pool-Level)

Metrics like CIDEr, METEOR, and corpus-BLEU cannot be computed sequentially per-batch. CIDEr requires a global document-frequency index, and BLEU requires corpus-wide length ratios for the brevity penalty.

```python
from dl_techniques.metrics.generation_metrics import generation_report

candidates = ["the cat is on the mat", "there is a cat"]
references = [
    ["the cat sits on the mat", "a feline rests"],
    ["a cat is right here"]
]

# Computes all metrics in a single pass
report = generation_report(
    candidates=candidates,
    references=references,
    max_order=4,
    cider_variant="d"
)

print(f"Corpus BLEU: {report['scaled']['bleu']:.2f}")
print(f"CIDEr-D: {report['scaled']['cider']:.2f}")
print(f"ROUGE-Lsum: {report['scaled']['rougeLsum']:.2f}")
```

### Custom Training Loops & Nested Metrics

The `HRMMetrics` class acts as a container for managing multiple related metrics in custom training steps (e.g., Hierarchical Reasoning Models).

```python
from dl_techniques.metrics import HRMMetrics
import tensorflow as tf

hrm_metrics = HRMMetrics(ignore_index=-100)

class HRMModel(keras.Model):
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y_true, y_pred=y_pred)
        
        # Apply gradients (omitted for brevity)
        
        # Update state directly
        hrm_metrics.update_state(y_true, y_pred)
        
        results = {"loss": loss}
        results.update(hrm_metrics.result())
        return results

    def on_epoch_begin(self, epoch, logs=None):
        hrm_metrics.reset_state()
```