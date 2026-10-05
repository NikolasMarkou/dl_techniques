# DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric

A Keras 3 + NumPy implementation of **DeepSORT** — SORT extended with a
learned person appearance descriptor and matching cascade.

## 1. Overview

**DeepSORT** (Wojke et al., ICIP 2017) tracks multiple targets by combining
a constant-velocity Kalman filter with a deep appearance metric. Each step
predicts all tracks, associates confirmed tracks by *gated appearance*
(minimum cosine distance to a per-track embedding gallery) through a
matching cascade that prefers recently-seen tracks, associates leftovers
and unconfirmed tracks by IoU, initiates tracks from orphan detections, and
refreshes the gallery. Track states run Tentative → Confirmed (`n_init=3`)
→ Deleted (`max_age=30` misses).

The appearance network (`mars-small128`, WACV 2018) maps 128×64 crops to
L2-normalized 128-dim descriptors through ELU convolutions and residual
blocks, trained with cosine-softmax identity classification.

The paper reports strongly reduced identity switches versus SORT on MOT16
at real-time rates; no such claim is made for this code, which has not
been trained or benchmarked here.

## 2. Usage

```python
import numpy as np
from dl_techniques.models.vision.deepsort import (
    DeepSortTracker,
    Detection,
    NearestNeighborDistanceMetric,
    create_deepsort_embedding,
)

# Appearance descriptors for crops (B, 128, 64, 3) in [0, 1].
embed = create_deepsort_embedding()
features = embed(np.zeros((4, 128, 64, 3), dtype="float32"), training=False)

# Multi-target tracking over detections.
tracker = DeepSortTracker(NearestNeighborDistanceMetric("cosine", 0.2, 100))
detections = [
    Detection([100.0, 100.0, 50.0, 100.0], 0.9, features[0]),
    Detection([300.0, 200.0, 50.0, 100.0], 0.8, features[1]),
]
tracker.predict()
tracker.update(detections)
for track in tracker.tracks:
    if track.is_confirmed():
        print(track.track_id, track.to_tlwh())
```

Training the embedding (identity labels required):

```python
import keras
from dl_techniques.models.vision.deepsort import create_deepsort_embedding

model = create_deepsort_embedding(include_top=True, num_classes=751)
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

`pretrained=True` raises `NotImplementedError`; pass a local checkpoint
path instead.

## 3. Components

| Name | Purpose |
| :--- | :--- |
| **`DeepSortAppearanceNet`** | mars-small128 embedding; `include_top` appends the cosine classifier. |
| **`DeepSortResidualBlock`** | Pre-activation residual block (first/projection variants). |
| **`CosineClassifier`** | Normalized class vectors + learned softplus scale → cosine logits. |
| **`create_deepsort_embedding`** | Thin factory. |
| **`KalmanFilter`** | Constant-velocity image-space filter + chi-square gating table. |
| **`NearestNeighborDistanceMetric`** | Per-identity gallery, cosine/Euclidean nearest distance, budget. |
| **`min_cost_matching` / `matching_cascade` / `gate_cost_matrix`** | Hungarian assignment, age cascade, Mahalanobis gating. |
| **`iou` / `iou_cost`** | IoU fallback matching. |
| **`Detection` / `Track` / `TrackState`** | Lifecycle state holders. |
| **`DeepSortTracker`** | `predict()` / `update()` orchestration. |

One deliberate deviation from the paper text: association is
appearance-only with Mahalanobis gating (what the reference code does),
not the λ-blended motion+appearance cost. There is no `MODEL_VARIANTS`
table: the reference ships one network.

## 4. Training notes

- Descriptors are exactly unit-norm in training mode (batch statistics);
  at random weights inference-mode activations vanish (transcribed 1e-3
  init) — train before trusting inference norms.
- Batches should be PK (several shots per identity) for the triplet mode.
- The reference trains with Adam at lr 1e-3; see `src/train/deepsort/`.

## 5. Citation

```bibtex
@inproceedings{wojke2017simple,
  title={Simple online and realtime tracking with a deep association metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={ICIP},
  year={2017}
}
```
