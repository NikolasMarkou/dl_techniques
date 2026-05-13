# ML Metrics Reference

Definition reference for every metric name-dropped in `EMBEDDINGS_BENCHMARKS.md`, `LLM_BENCHMARKS.md`, `VISION_BENCHMARKS.md`, and `VLM_BENCHMARKS.md`. Math is plain-text; when in doubt, check the original paper linked at the bottom.

> **Last reviewed**: 2026-05-13

Conventions used in this file:

- `log` means natural log unless explicitly written `log2` or `log10`.
- `|S|` means cardinality of set `S`.
- `1[cond]` is the indicator function (1 if `cond` else 0).
- `TP / FP / TN / FN` are true positive / false positive / true negative / false negative counts.
- "Higher is better" / "Lower is better" notes refer to the convention used in leaderboards, not to the sign of any loss function the metric might double as.
- Where a metric exists in `dl_techniques`, the implementation path is noted under **In `dl_techniques`**.

---

## 1. Classification (Binary and Multiclass)

### Accuracy

**What it measures**: The fraction of examples whose predicted class equals the true class. Simple, intuitive, and the single most over-used metric in ML. Meaningful only when classes are roughly balanced and all confusions are equally costly. On imbalanced data (e.g. fraud, medical screening) accuracy is dominated by the majority class and a constant predictor can post deceptively high scores.

**Formula**:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
        = (1/N) * sum_i 1[y_pred_i == y_true_i]
```

**Edge cases / pitfalls**:

- Useless on imbalanced binary problems (predict-majority baseline can exceed 95%).
- Multiclass accuracy is implicitly micro-averaged; per-class behaviour is hidden.
- Sensitive to label noise. A 1% label flip caps accuracy at 99% regardless of the model.

**Typical reporting**: Percentage with one or two decimals. Higher is better. Default eval for ImageNet, CIFAR, MNIST.

### Top-k Accuracy

**What it measures**: Counts a prediction as correct if the ground-truth class appears anywhere in the model's top-`k` highest-probability predictions. Used when the label space is large (1000-class ImageNet) and "the right answer is in the shortlist" is acceptable. Top-5 has been the standard companion to top-1 on ImageNet since 2012.

**Formula**:

```
TopK = (1/N) * sum_i 1[y_true_i in argsort(p_i)[-k:]]
```

**Edge cases / pitfalls**:

- Top-1 and top-5 trends can diverge for highly confident wrong-class predictions.
- For `k >= number_of_classes`, the metric trivially equals 1.
- Some leaderboards report top-5 error (`1 - top5_accuracy`) rather than the accuracy itself.

**Typical reporting**: Percentage. Higher is better. Standard pair "Top-1 / Top-5" on ImageNet-1K.

### Precision

**What it measures**: Of all examples the model called positive, what fraction were actually positive. Punishes false alarms. The metric you optimize when acting on a positive is expensive (sending a takedown notice, prescribing a treatment).

**Formula**:

```
Precision = TP / (TP + FP)
```

**Edge cases / pitfalls**:

- Undefined when the model never predicts positive (`TP + FP = 0`); conventionally treated as 0 or as NaN depending on the library.
- Easily gamed by predicting positive on the single most confident sample only.
- Multiclass: see "averaging conventions" below.

**Typical reporting**: Percentage. Higher is better. Always co-reported with recall.

### Recall (Sensitivity, True Positive Rate)

**What it measures**: Of all true positives, what fraction the model caught. Punishes misses. The metric you optimize when missing a positive is expensive (disease screening, security scanning).

**Formula**:

```
Recall = TP / (TP + FN)
```

**Edge cases / pitfalls**:

- Undefined when there are no positives in the eval set.
- Trivially 1.0 by predicting positive on every example (paired with terrible precision).

**Typical reporting**: Percentage. Higher is better. Always co-reported with precision.

### F1

**What it measures**: Harmonic mean of precision and recall. Balances false alarms against misses. Used when classes are imbalanced and you want a single number; preferred over accuracy in retrieval-style and medical-screening settings.

**Formula**:

```
F1 = 2 * P * R / (P + R)
```

**Edge cases / pitfalls**:

- Harmonic mean punishes the lower of the two; F1 is low if either P or R is low.
- F1 = 0 when both are 0; undefined if both are 0 by some libraries.
- Multiclass averaging matters enormously: macro-F1 and micro-F1 can differ by >20 points on imbalanced data.

**Typical reporting**: Percentage. Higher is better. Default eval for span extraction (SQuAD), NER, many GLUE tasks.

### F-beta

**What it measures**: Generalized F-score that weights recall `beta` times as heavily as precision. `beta = 1` recovers F1; `beta = 2` (F2) emphasizes recall (useful for screening); `beta = 0.5` (F0.5) emphasizes precision.

**Formula**:

```
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
```

**Edge cases / pitfalls**:

- Same undefined-when-both-zero behaviour as F1.
- Choice of `beta` is application-specific and should be justified.

**Typical reporting**: Decimal in [0, 1] or percentage. Higher is better.

### Specificity (True Negative Rate)

**What it measures**: Of all true negatives, what fraction the model correctly rejected. Counterpart to recall on the negative class. Critical in screening contexts where false positives drive cost (e.g. confirmatory testing).

**Formula**:

```
Specificity = TN / (TN + FP)
```

**Edge cases / pitfalls**:

- Undefined when there are no negatives.
- ROC curves implicitly trade `1 - Specificity` (FPR) against Recall (TPR).

**Typical reporting**: Percentage. Higher is better. Common in medical ML.

### Balanced Accuracy

**What it measures**: Mean of per-class recall. Equivalent to accuracy on a class-balanced test set, even when the actual test set is imbalanced. The lazy fix for class imbalance when you cannot resample.

**Formula**:

```
BalancedAccuracy = (1/K) * sum_k Recall_k
                = 0.5 * (Recall_pos + Recall_neg)   # binary
```

where `K` is the number of classes and `Recall_k` is recall on class `k`.

**Edge cases / pitfalls**:

- Treats every class as equally important - this is sometimes wrong (rare-class recall may matter more).
- Same as macro-recall, which is sometimes a clearer name.

**Typical reporting**: Decimal. Higher is better. scikit-learn default for imbalanced cls.

### Matthews Correlation Coefficient (MCC)

**What it measures**: A correlation coefficient between observed and predicted binary classifications. Returns +1 for perfect prediction, 0 for random, -1 for total disagreement. Considered the most robust single-number summary for binary classification because it uses all four cells of the confusion matrix and is symmetric under class swapping.

**Formula**:

```
MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
```

**Edge cases / pitfalls**:

- Denominator is 0 if any single row or column of the confusion matrix sums to 0 (e.g., the model never predicts positive, or there are no negatives in the eval set). Conventionally set to 0 in that case.
- Multiclass MCC is defined (Gorodkin 2004) but rarely reported; macro-F1 dominates instead.

**Typical reporting**: Decimal in [-1, 1]. Higher is better. Standard on imbalanced binary benchmarks (e.g. CoLA in GLUE).

### Cohen's kappa

**What it measures**: Agreement between two annotators (or model vs ground truth) corrected for agreement expected by chance. Useful when classes are imbalanced because the "by chance" baseline accounts for the marginal class frequencies.

**Formula**:

```
kappa = (p_o - p_e) / (1 - p_e)
```

where `p_o` is observed agreement (= accuracy) and `p_e` is expected agreement under independence of the two label marginals.

**Edge cases / pitfalls**:

- Sensitive to the prevalence and bias of the two raters; the so-called "kappa paradox".
- Undefined when `p_e = 1` (one rater always predicts the same class with probability 1).

**Typical reporting**: Decimal in [-1, 1]. Higher is better. Common in medical annotation, NLP inter-annotator agreement.

### Quadratic-weighted kappa

**What it measures**: Cohen's kappa with a quadratic penalty on the distance between predicted and true ordinal labels. Designed for ordinal-regression-style classification (e.g. diabetic-retinopathy grades 0-4, essay scoring).

**Formula**:

```
QWKappa = 1 - sum_ij w_ij * O_ij / sum_ij w_ij * E_ij

w_ij = (i - j)^2 / (K - 1)^2
```

where `O` is the observed and `E` the expected confusion matrix, `K` is the number of ordinal levels, and `w_ij` penalizes farther-apart confusions more.

**Edge cases / pitfalls**:

- Only meaningful for ordinal labels; meaningless for nominal classes.
- Kaggle competitions in medical imaging (APTOS, Prostate) use it heavily.

**Typical reporting**: Decimal in [-1, 1]. Higher is better.

### ROC-AUC

**What it measures**: Area under the Receiver Operating Characteristic curve, which plots True Positive Rate (recall) against False Positive Rate (`1 - specificity`) as the decision threshold sweeps from `+inf` to `-inf`. Probabilistic interpretation: ROC-AUC is the probability that a randomly chosen positive scores higher than a randomly chosen negative.

**Formula**:

```
ROC-AUC = P(score(x_pos) > score(x_neg))
        = (1 / (|P| * |N|)) * sum_{i in P, j in N} 1[score_i > score_j] + 0.5 * 1[score_i == score_j]
```

**Edge cases / pitfalls**:

- Insensitive to class imbalance, which is sometimes good (compare models across base rates) and sometimes bad (a heavily imbalanced precision-recall regime can look healthy in ROC space).
- Undefined when there is only one class in `y_true`.
- Multiclass: typically reported one-vs-rest (OvR) or one-vs-one (OvO), then macro- or weighted-averaged.

**Typical reporting**: Decimal in [0, 1] (random = 0.5). Higher is better. Default eval for binary classification with continuous scores.

### PR-AUC / Average Precision (binary)

**What it measures**: Area under the Precision-Recall curve. More sensitive than ROC-AUC under severe class imbalance because both axes use only positives. Approximates the expected precision over a range of recall levels.

**Formula** (binary, sklearn convention):

```
AP = sum_n (R_n - R_{n-1}) * P_n
```

where `(P_n, R_n)` are the precision and recall at the n-th threshold, sorted by descending score.

**Edge cases / pitfalls**:

- Numerically different from trapezoidal integration of the PR curve; AP is a step function with a left-Riemann sum.
- Undefined when no positives exist.
- Multiclass / multilabel: micro, macro, samples, weighted variants exist; pick deliberately.

**Typical reporting**: Decimal. Higher is better. Standard on imbalanced binary tasks and detection (see Object Detection section).

### Log Loss / Cross-Entropy

**What it measures**: Negative log-likelihood of the true labels under the model's probability distribution. Strictly proper scoring rule: minimized only by reporting the true posterior. Doubles as the training loss for most classifiers.

**Formula** (multiclass):

```
LogLoss = -(1/N) * sum_i sum_k y_i_k * log(p_i_k)
```

where `y_i_k` is the one-hot true label and `p_i_k` is the predicted probability for class `k` on example `i`.

**Edge cases / pitfalls**:

- `log(0)` is `-inf`; libraries clip predictions to `[eps, 1 - eps]` (sklearn default `eps = 1e-15`). Different clipping changes the number.
- Highly sensitive to overconfident wrong predictions; a single confident mistake dominates the average.

**Typical reporting**: Lower is better. Reported as is, no percentage. Used as the primary leaderboard metric on Kaggle classification with submission probabilities.

### Brier Score

**What it measures**: Mean squared error between predicted probabilities and one-hot labels. Strictly proper scoring rule. Less aggressive than log loss in penalizing overconfidence and bounded in `[0, 1]` for binary, so easier to interpret. Frequently used to evaluate forecasts of probabilities (weather, medicine).

**Formula** (binary):

```
Brier = (1/N) * sum_i (p_i - y_i)^2
```

Multiclass: sum the squared error over all class indicators per example.

**Edge cases / pitfalls**:

- Bounded `[0, 1]` for binary, `[0, 2]` for multiclass-with-one-hot in the usual convention.
- Decomposes into reliability + resolution + uncertainty (Murphy 1973); see Calibration section.

**Typical reporting**: Lower is better. Reported alongside ECE for calibration analyses.

### Confusion Matrix

**What it measures**: A `K x K` table where entry `(i, j)` is the count of examples whose true class is `i` and predicted class is `j`. Diagonal entries are correct predictions; off-diagonal entries identify systematic confusions (e.g. cats predicted as dogs). All scalar classification metrics are derived from it.

**Formula**:

```
C[i, j] = sum_n 1[y_true_n == i and y_pred_n == j]
```

**Edge cases / pitfalls**:

- Row-normalized (divide by row sum) shows per-class recall; column-normalized shows per-class precision.
- For `K = 2` the canonical labeling is `[[TN, FP], [FN, TP]]` (sklearn) but other libraries transpose.

**Typical reporting**: Heatmap in papers. The basis of every other classification metric in this section.

### Averaging Conventions (Macro / Micro / Weighted / Samples)

Crucial and frequently misreported.

- **Micro**: Pool TP/FP/FN globally across classes, then compute the metric once. For multiclass single-label, micro-precision = micro-recall = micro-F1 = accuracy.
- **Macro**: Compute the metric per class, then unweighted-mean. Each class contributes equally regardless of support. Penalizes models that ignore rare classes.
- **Weighted**: Macro but weighted by class support (number of true instances). A compromise between micro and macro; same support-weighting as micro, but per-class computation can still surface rare-class issues.
- **Samples**: For multilabel only. Compute the metric per example over the label set, then mean over examples.

When `dl_techniques` references "F1 macro" or "F1 weighted" in a leaderboard table, this is the convention being invoked.

---

## 2. Object Detection

### IoU (Intersection over Union, Jaccard)

**What it measures**: Overlap between predicted and ground-truth bounding boxes (or masks). The atomic matching criterion behind every detection metric. Independent of object size.

**Formula**:

```
IoU(A, B) = |A intersect B| / |A union B|
```

**Edge cases / pitfalls**:

- IoU is 0 whenever boxes do not overlap and provides no gradient signal in that regime - which is why GIoU/DIoU/CIoU were invented.
- For rotated boxes IoU is more expensive to compute; libraries vary.

**Typical reporting**: Decimal in [0, 1]. Higher is better. Reported as a per-image diagnostic, or as the threshold underlying mAP.

### Average Precision (per class, detection sense)

**What it measures**: Area under the per-class precision-recall curve, where a detection is a true positive only if its IoU with a ground-truth box of the same class exceeds a threshold. Distinct from the binary-classification AP (above) because of the IoU matching step. Each class has its own AP; mAP aggregates across classes.

**Formula** (COCO 101-point interpolation):

```
AP = (1 / 101) * sum_{r in {0.00, 0.01, ..., 1.00}} P_interp(r)
P_interp(r) = max_{r' >= r} P(r')
```

PASCAL VOC 2007 used 11-point interpolation; VOC 2010+ used the all-points (area-under-curve) integral.

**Edge cases / pitfalls**:

- Greedy matching: detections are sorted by confidence, then matched to the highest-IoU unused ground-truth box above the threshold; subsequent matches to the same GT are false positives.
- Crowd / "iscrowd" annotations in COCO are excluded from FP counting.

**Typical reporting**: Decimal in [0, 100] (COCO multiplies by 100). Higher is better.

### mAP@0.5, mAP@0.75

**What it measures**: Mean (over classes) of AP computed at a single IoU threshold (0.5 = "PASCAL", 0.75 = "strict"). The 0.5 number is permissive and saturates on modern detectors; 0.75 is the discriminating midpoint.

**Formula**:

```
mAP@T = (1 / K) * sum_k AP_k(IoU >= T)
```

**Edge cases / pitfalls**:

- Comparing 0.5-only numbers across modern detectors is uninformative; everyone scores within a few points.
- VOC mAP@0.5 != COCO mAP@0.5 even at the same threshold because of different interpolation and matching rules.

**Typical reporting**: Percentage on COCO. Higher is better.

### mAP@[0.5:0.05:0.95] (COCO "AP")

**What it measures**: The COCO primary metric. Mean of mAP computed at ten IoU thresholds from 0.5 to 0.95 in steps of 0.05. Rewards detectors with tight box localization.

**Formula**:

```
AP_coco = (1/10) * sum_{T in {0.50, 0.55, ..., 0.95}} mAP@T
```

**Edge cases / pitfalls**:

- The 101-point recall interpolation is part of the COCO definition; do not confuse with 11-point VOC.
- Sometimes simply labelled `AP` in COCO papers; do not confuse with binary AP.

**Typical reporting**: Percentage. Higher is better. Default eval on COCO.

### AP_small, AP_medium, AP_large

**What it measures**: COCO `AP` restricted to objects whose area is `< 32^2`, in `[32^2, 96^2]`, or `> 96^2` pixels respectively. Diagnoses scale-specific failure modes.

**Formula**: Same as `AP_coco`, restricted to the relevant area bucket.

**Edge cases / pitfalls**:

- Areas are in *original* image pixels; aggressive image rescaling changes which bucket a GT falls into.
- Very few small objects in some classes -> high-variance AP_small numbers.

**Typical reporting**: Percentage. Higher is better. Companion columns to COCO AP.

### AR@1, AR@10, AR@100

**What it measures**: Average Recall when at most 1, 10, or 100 detections per image are kept. Probes the recall ceiling and the quality of low-confidence proposals (matters for downstream re-ranking, e.g. two-stage detectors and tracking).

**Formula**:

```
AR@k = 2 * integral_{T = 0.5 to 1.0} recall(T, max_det = k) dT
```

(The factor 2 normalizes the IoU range `[0.5, 1.0]` to `[0, 1]`.)

**Edge cases / pitfalls**:

- AR@100 is essentially recall ceiling; AR@1 is "top-1 detection" quality.
- For dense-prediction detectors `AR@100` saturates and is unhelpful.

**Typical reporting**: Percentage. Higher is better. Standard COCO companion columns.

### GIoU (Generalized IoU)

**What it measures**: Extension of IoU that remains informative when the two boxes do not overlap. Used primarily as a loss; can also be reported as a quality metric. Penalizes empty space between boxes.

**Formula**:

```
GIoU = IoU - |C \ (A union B)| / |C|
```

where `C` is the smallest axis-aligned enclosing box of `A` and `B`.

**Edge cases / pitfalls**:

- Range is `[-1, 1]` (vs IoU's `[0, 1]`); negative when boxes are disjoint.
- As a loss: `L_GIoU = 1 - GIoU`.

**Typical reporting**: Most commonly a loss. Source: Rezatofighi et al., arXiv:1902.09630.

### DIoU (Distance-IoU)

**What it measures**: IoU minus a normalized distance between box centers. Converges faster than GIoU during training because it adds a direct gradient on center-point displacement even when boxes overlap.

**Formula**:

```
DIoU = IoU - rho^2(b_pred, b_gt) / c^2
```

where `rho` is the Euclidean distance between box centers and `c` is the diagonal length of the smallest enclosing box `C`.

**Edge cases / pitfalls**:

- Same `[-1, 1]` range issue as GIoU.

**Typical reporting**: Loss. Source: Zheng et al., arXiv:1911.08287.

### CIoU (Complete IoU)

**What it measures**: DIoU plus an aspect-ratio consistency penalty. Default regression loss in YOLOv5/v7/v8/v12 and several recent two-stage detectors.

**Formula**:

```
CIoU = IoU - rho^2(b_pred, b_gt) / c^2 - alpha * v

v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2
alpha = v / ((1 - IoU) + v)
```

**Edge cases / pitfalls**:

- `alpha` is treated as a constant during back-propagation in the original paper.

**Typical reporting**: Loss. Source: Zheng et al., arXiv:1911.08287.

### FPS / Latency / Params (co-reported)

Detection leaderboards almost always report runtime stats alongside mAP. **FPS** is frames-per-second on a specified device (TRT-FP16 on A100 is the modern default); **Latency** is per-image inference time in ms (`Latency_ms = 1000 / FPS` when batch=1); **Params** is the number of model parameters in millions (not counting EMA copies or auxiliary heads). All three are sensitive to: input resolution, batch size, half-precision vs full-precision, post-processing (NMS) inclusion, and device generation. Always cite the exact configuration.

---

## 3. Semantic Segmentation

### Pixel Accuracy

**What it measures**: Fraction of pixels whose predicted class equals their true class. The segmentation analogue of classification accuracy and shares its weaknesses on imbalanced classes (e.g. road dominates Cityscapes).

**Formula**:

```
PixelAcc = sum_i n_ii / sum_i t_i
```

where `n_ij` is the number of pixels of class `i` predicted as class `j` and `t_i = sum_j n_ij`.

**Edge cases / pitfalls**:

- Dominated by majority classes. Reportedly meaningless on driving and aerial scenes.

**Typical reporting**: Percentage. Higher is better. Lightweight sanity check; never the primary metric.

### Mean Pixel Accuracy

**What it measures**: Per-class pixel accuracy averaged unweighted across classes.

**Formula**:

```
MeanPixelAcc = (1 / K) * sum_i (n_ii / t_i)
```

**Edge cases / pitfalls**:

- Undefined when a class is absent from the eval image; libraries skip or NaN.
- Not the same as mIoU - it ignores false positives.

**Typical reporting**: Percentage. Higher is better.

### IoU per class

**What it measures**: Pixel-level IoU between predicted and true masks for a single class. The building block for mIoU.

**Formula**:

```
IoU_i = n_ii / (t_i + sum_j n_ji - n_ii)
```

**Edge cases / pitfalls**:

- Undefined when class `i` is absent from both prediction and ground truth (denominator = 0); usually skipped in the average.

**Typical reporting**: Decimal or percentage. Higher is better. Per-class column in segmentation papers.

### mean IoU (mIoU)

**What it measures**: Unweighted mean of per-class IoU. The primary metric on most semantic-segmentation benchmarks (Cityscapes, ADE20K, PASCAL VOC, PASCAL Context).

**Formula**:

```
mIoU = (1 / K) * sum_i IoU_i
```

**Edge cases / pitfalls**:

- Classes absent from a particular split are typically excluded from the average; check the benchmark protocol.
- Class imbalance is hidden: every class contributes equally regardless of pixel count.

**Typical reporting**: Percentage. Higher is better. Default eval on ADE20K, Cityscapes.

### Frequency-Weighted IoU

**What it measures**: Per-class IoU weighted by the class's pixel frequency. A middle ground between Pixel Accuracy (frequency-dominated) and mIoU (class-uniform).

**Formula**:

```
FW-IoU = (1 / sum_k t_k) * sum_i t_i * IoU_i
```

**Edge cases / pitfalls**:

- Rare classes contribute almost nothing; high score does not imply rare-class competence.

**Typical reporting**: Percentage. Less common than mIoU on modern benchmarks.

### Dice / F1 (pixel sense)

**What it measures**: Twice the overlap divided by the sum of cardinalities. Equivalent to F1 at the pixel level. Differs from IoU monotonically (`Dice = 2 * IoU / (1 + IoU)`) but the loss-gradient behaviour differs significantly. Standard in medical imaging.

**Formula**:

```
Dice = 2 * |A intersect B| / (|A| + |B|)
    = 2 * TP / (2 * TP + FP + FN)
```

**Edge cases / pitfalls**:

- Undefined for empty GT and empty prediction; libraries set to 1 by convention (both sets are empty so they "match").
- Dice loss = `1 - Dice` is non-convex but widely used.

**Typical reporting**: Decimal. Higher is better. Default eval on most medical-imaging benchmarks (BraTS, ACDC, KiTS).

### Boundary F1 / BF score

**What it measures**: F1 score on the boundary pixels of the mask, after a distance tolerance is applied. Rewards models that produce crisp object boundaries rather than blobby silhouettes.

**Formula**:

```
BF_theta = 2 * P_theta * R_theta / (P_theta + R_theta)

P_theta = (1 / |B_pred|) * sum_{x in B_pred} 1[exists y in B_gt with d(x, y) < theta]
R_theta = symmetric
```

where `B_pred` and `B_gt` are the predicted and ground-truth boundary pixel sets and `theta` is the distance tolerance in pixels.

**Edge cases / pitfalls**:

- Sensitive to `theta`; report it (commonly 1, 2, or `0.0075 * diag(image)`).
- Original boundary metric of Csurka et al. 2013.

**Typical reporting**: Percentage. Higher is better. Used as a companion to mIoU when fine boundaries matter.

---

## 4. Instance and Panoptic Segmentation

### Mask AP

**What it measures**: COCO `AP` (mean AP at IoU `[0.5:0.05:0.95]`) where the matching IoU is computed between predicted and ground-truth instance masks rather than bounding boxes. Sometimes called "segm AP" in the COCO API.

**Formula**: Identical to detection `AP` (section 2) with the IoU computed pixelwise.

**Edge cases / pitfalls**:

- Mask AP can be lower than box AP even when masks look reasonable, because pixel IoU is stricter than box IoU.
- Mask APs from one library to another can differ slightly due to mask encoding (RLE vs polygon).

**Typical reporting**: Percentage. Higher is better. Standard instance-segmentation metric.

### Panoptic Quality (PQ)

**What it measures**: Single metric for the panoptic task (every pixel gets a class + an instance id). Decomposes into segmentation quality (how well matched segments overlap) times recognition quality (how often segments are detected correctly).

**Formula**:

```
PQ = SQ * RQ
  = (sum_{(p, g) in TP} IoU(p, g)) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
```

where a predicted segment `p` and ground-truth segment `g` of the same class are matched if `IoU(p, g) > 0.5`. This threshold guarantees at most one match per GT.

**Edge cases / pitfalls**:

- PQ averages over classes (each class's PQ computed separately, then mean).
- "Thing" and "stuff" classes are typically reported separately as `PQ_th` and `PQ_st`.
- Some papers report `PQ^dagger` which substitutes IoU per stuff class for the recognition step; check protocol.

**Typical reporting**: Percentage. Higher is better. Default eval on COCO Panoptic.

### Segmentation Quality (SQ)

**What it measures**: Mean IoU over matched TP segments. Captures localization quality conditional on recognition.

**Formula**:

```
SQ = (sum_{(p, g) in TP} IoU(p, g)) / |TP|
```

**Edge cases / pitfalls**:

- Bounded `[0.5, 1.0]` because the matching threshold is 0.5.

**Typical reporting**: Companion column to PQ.

### Recognition Quality (RQ)

**What it measures**: F1 score over segments (matched / unmatched). Captures detection quality independent of localization tightness.

**Formula**:

```
RQ = |TP| / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
```

**Edge cases / pitfalls**:

- Identical in form to segment-level F1.

**Typical reporting**: Companion column to PQ. Source: Kirillov et al., arXiv:1801.00868.

---

## 5. Depth Estimation

Following the protocol of Eigen et al., 2014 (arXiv:1406.2283). All metrics are computed only over pixels with valid ground-truth depth.

### AbsRel (Absolute Relative Error)

**What it measures**: Mean absolute relative depth error. Scale-aware. Standard primary metric for KITTI / NYUv2 depth.

**Formula**:

```
AbsRel = (1 / N) * sum_i |d_pred_i - d_gt_i| / d_gt_i
```

**Edge cases / pitfalls**:

- Division by GT depth means very near surfaces dominate; clipped to a min-depth in most protocols.
- `d_gt = 0` pixels must be masked out.

**Typical reporting**: Decimal (e.g., 0.105 on NYUv2 for strong models). Lower is better.

**In `dl_techniques`**: `dl_techniques/metrics/depth_metrics.py` (`AbsRelMetric`).

### SqRel (Squared Relative Error)

**What it measures**: Mean squared relative depth error. Emphasizes large pointwise errors more than AbsRel.

**Formula**:

```
SqRel = (1 / N) * sum_i (d_pred_i - d_gt_i)^2 / d_gt_i
```

**Edge cases / pitfalls**:

- Same near-surface domination as AbsRel, squared.

**Typical reporting**: Decimal. Lower is better. KITTI standard.

**In `dl_techniques`**: `dl_techniques/metrics/depth_metrics.py` (`SqRelMetric`).

### RMSE (Depth)

**What it measures**: Root mean squared error in metric units (meters). Scale-aware and sensitive to outliers.

**Formula**:

```
RMSE = sqrt((1 / N) * sum_i (d_pred_i - d_gt_i)^2)
```

**Edge cases / pitfalls**:

- Reported in meters (KITTI) or in the dataset's metric unit. Dataset cap matters: KITTI evaluates to 80 m, NYUv2 to 10 m.

**Typical reporting**: Decimal in meters. Lower is better.

**In `dl_techniques`**: `dl_techniques/metrics/depth_metrics.py` (`RMSEMetric`).

### RMSE-log

**What it measures**: RMSE in log space. Scale-aware but emphasizes relative error more than RMSE.

**Formula**:

```
RMSE_log = sqrt((1 / N) * sum_i (log(d_pred_i) - log(d_gt_i))^2)
```

**Edge cases / pitfalls**:

- `log(0)` requires masking and a minimum-depth floor.

**Typical reporting**: Decimal. Lower is better.

**In `dl_techniques`**: `dl_techniques/metrics/depth_metrics.py` (`RMSELogMetric`).

### log10 error

**What it measures**: Mean absolute log10 difference between predicted and ground-truth depth. Used in NYUv2 evaluations.

**Formula**:

```
log10 = (1 / N) * sum_i |log10(d_pred_i) - log10(d_gt_i)|
```

**Edge cases / pitfalls**: Same masking issues as RMSE-log.

**Typical reporting**: Decimal. Lower is better.

### delta_1, delta_2, delta_3 (Threshold accuracies)

**What it measures**: Fraction of pixels whose ratio `max(d_pred / d_gt, d_gt / d_pred)` is below a threshold (1.25, 1.25^2 = 1.5625, 1.25^3 = 1.953125). Coarse, intuitive notion of "how often the depth is within X percent of truth".

**Formula**:

```
delta_n = (1 / N) * sum_i 1[max(d_pred_i / d_gt_i, d_gt_i / d_pred_i) < 1.25^n]
```

**Edge cases / pitfalls**:

- Higher is better, opposite of the error metrics.
- Strong models post `delta_1 > 0.95` on NYUv2.

**Typical reporting**: Decimal. Higher is better. Eigen standard.

**In `dl_techniques`**: `dl_techniques/metrics/depth_metrics.py` (`DeltaThresholdMetric`).

### Scale-Invariant Logarithmic Error (SILog, Eigen)

**What it measures**: Loss / metric that is invariant to a global multiplicative scale of the prediction. Lets you compare relative-depth predictors against scale-aware ones on a level playing field. The official KITTI online server metric.

**Formula**:

```
SILog(d_pred, d_gt) = (1 / n) * sum_i e_i^2 - (lambda / n^2) * (sum_i e_i)^2
e_i = log(d_pred_i) - log(d_gt_i)
```

`lambda = 1` recovers exact scale invariance; `lambda = 0` is plain L2 in log space. KITTI evaluation reports it multiplied by 100 with `lambda = 1`.

**Edge cases / pitfalls**:

- KITTI's reported "SILog" is `sqrt(...) * 100`; check what scale the leaderboard uses before comparing numbers.
- Eigen et al.'s recommended training loss uses `lambda = 0.5`.

**Typical reporting**: Decimal multiplied by 100 on KITTI. Lower is better. Source: Eigen et al., arXiv:1406.2283.

### iRMSE (Inverse RMSE)

**What it measures**: RMSE on inverse depth (`1 / d`) in `1 / km` units. Standard on KITTI's depth-completion track because LiDAR is sparser at long range and inverse depth equalizes the cost.

**Formula**:

```
iRMSE = sqrt((1 / N) * sum_i (1 / d_pred_i - 1 / d_gt_i)^2)
```

**Edge cases / pitfalls**:

- Reported in `1 / km` on KITTI: multiply meters^-1 by 1000.

**Typical reporting**: Decimal. Lower is better.

---

## 6. Regression

### MAE (Mean Absolute Error)

**What it measures**: Average absolute deviation between predicted and true value. Robust to outliers compared to MSE. Equivalent to L1 loss.

**Formula**:

```
MAE = (1 / N) * sum_i |y_pred_i - y_true_i|
```

**Edge cases / pitfalls**:

- Same units as the target.
- Sub-gradient at zero; L1 minimization yields the conditional median.

**Typical reporting**: Real number. Lower is better.

### MSE (Mean Squared Error)

**What it measures**: Average squared deviation. Penalizes large errors quadratically. Equivalent to L2 loss.

**Formula**:

```
MSE = (1 / N) * sum_i (y_pred_i - y_true_i)^2
```

**Edge cases / pitfalls**:

- Outliers dominate; not robust.
- Units are squared - hard to interpret directly.

**Typical reporting**: Real number. Lower is better.

### RMSE (Root Mean Squared Error)

**What it measures**: Square root of MSE. Same units as the target; same outlier-sensitivity as MSE.

**Formula**:

```
RMSE = sqrt(MSE) = sqrt((1 / N) * sum_i (y_pred_i - y_true_i)^2)
```

**Edge cases / pitfalls**:

- Always `>= MAE` for the same predictions; the gap signals tail-error magnitude.

**Typical reporting**: Real number in target units. Lower is better.

### MAPE (Mean Absolute Percentage Error)

**What it measures**: Average relative absolute error, expressed as a percentage. Intuitive but pathological when targets are near zero.

**Formula**:

```
MAPE = (100 / N) * sum_i |y_pred_i - y_true_i| / |y_true_i|
```

**Edge cases / pitfalls**:

- Undefined when any `y_true_i = 0`.
- Asymmetric: penalizes over-prediction more than under-prediction.

**Typical reporting**: Percentage. Lower is better. Common in business forecasting; discouraged in statistics literature.

### sMAPE (Symmetric MAPE)

**What it measures**: A symmetric variant of MAPE that bounds error to `[0, 200]%`.

**Formula** (M4 competition convention):

```
sMAPE = (100 / N) * sum_i 2 * |y_pred_i - y_true_i| / (|y_pred_i| + |y_true_i|)
```

**Edge cases / pitfalls**:

- Multiple incompatible definitions in the wild; M4 (Makridakis) uses the one above; some use `(|y_pred + y_true|) / 2` denominator.
- Undefined when `y_pred = y_true = 0`.

**Typical reporting**: Percentage. Lower is better. Standard on M4 / M5.

### R^2 (Coefficient of Determination)

**What it measures**: Fraction of variance in the target explained by the model. 1.0 = perfect, 0.0 = predicting the mean, negative = worse than predicting the mean.

**Formula**:

```
R^2 = 1 - sum_i (y_pred_i - y_true_i)^2 / sum_i (y_true_i - y_bar)^2
```

where `y_bar` is the mean of the true values.

**Edge cases / pitfalls**:

- Undefined when `y` has zero variance.
- Can be negative on test sets where the model is worse than the mean predictor; this regularly confuses newcomers.

**Typical reporting**: Decimal in (-inf, 1]. Higher is better.

### Adjusted R^2

**What it measures**: R^2 penalized for the number of features. Adjusts for the fact that adding features cannot decrease R^2.

**Formula**:

```
R^2_adj = 1 - (1 - R^2) * (N - 1) / (N - p - 1)
```

where `N` is the sample size and `p` is the number of features.

**Edge cases / pitfalls**:

- Defined only when `N > p + 1`.
- Diagnostic for linear regression; rarely informative for deep nets.

**Typical reporting**: Decimal. Higher is better.

### Explained Variance

**What it measures**: Like R^2 but only the variance-explained part - does not penalize biased estimators.

**Formula**:

```
ExplVar = 1 - Var(y_true - y_pred) / Var(y_true)
```

**Edge cases / pitfalls**:

- Equals R^2 only when the mean residual is 0.
- Inflated for biased predictors.

**Typical reporting**: Decimal. Higher is better.

### Huber Loss

**What it measures**: Quadratic for small residuals and linear for large ones, controlled by `delta`. Robust to outliers while remaining differentiable. Doubles as a regression loss (the default in many RL algorithms and some object-detection regressors).

**Formula**:

```
Huber(r) = 0.5 * r^2                  if |r| <= delta
        = delta * (|r| - 0.5 * delta) otherwise
r = y_pred - y_true
```

**Edge cases / pitfalls**:

- `delta` is a hyperparameter; common choices are 1.0 or the median absolute residual.
- Not scale-free.

**Typical reporting**: Real number. Lower is better. As a metric, rare in papers; as a loss, ubiquitous.

### Quantile / Pinball Loss

**What it measures**: Asymmetric loss that elicits the conditional quantile at level `tau`. The proper scoring rule for quantile regression and probabilistic forecasting (cf. Time Series section).

**Formula**:

```
PinballLoss_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
```

**Edge cases / pitfalls**:

- `tau = 0.5` recovers half-MAE (median).
- "Multi-quantile pinball loss" averages over multiple `tau` values.

**Typical reporting**: Real number, often averaged over `tau in {0.1, 0.5, 0.9}` or a finer grid. Lower is better.

---

## 7. Time Series Forecasting

### MASE (Mean Absolute Scaled Error)

**What it measures**: Forecast MAE divided by the in-sample MAE of a naive seasonal forecast. Scale-free, well-defined for intermittent demand, comparable across series. Hyndman & Koehler 2006.

**Formula** (seasonal, period `m`):

```
MASE = (1 / J) * sum_j |e_j| / ((1 / (T - m)) * sum_{t = m+1}^{T} |Y_t - Y_{t-m}|)
```

where `e_j` is the out-of-sample forecast error and the denominator is the in-sample MAE of the seasonal-naive forecast `Y_t-hat = Y_{t-m}`. For non-seasonal data set `m = 1` (random-walk denominator).

**Edge cases / pitfalls**:

- Denominator is computed on the *training* set only; using the test set is a common bug.
- Undefined when the series is constant on the training segment.

**Typical reporting**: Decimal (MASE = 1 means as good as the naive baseline). Lower is better. Default on M4, M5.

### RMSSE (Root Mean Squared Scaled Error)

**What it measures**: RMSE version of MASE - forecast RMSE divided by in-sample RMSE of the naive forecast. Primary metric in M5.

**Formula**:

```
RMSSE = sqrt((1 / J) * sum_j e_j^2 / ((1 / (T - 1)) * sum_{t = 2}^{T} (Y_t - Y_{t-1})^2))
```

**Edge cases / pitfalls**:

- Same denominator-on-training-only caveat as MASE.

**Typical reporting**: Decimal. Lower is better.

### OWA (M4 Overall Weighted Average)

**What it measures**: M4-specific aggregation: arithmetic mean of (forecast sMAPE / naive-2 sMAPE) and (forecast MASE / naive-2 MASE).

**Formula**:

```
OWA = 0.5 * (sMAPE / sMAPE_naive2 + MASE / MASE_naive2)
```

**Edge cases / pitfalls**:

- Naive-2 = naive forecast with seasonal adjustment via classical decomposition.
- Specific to the M4 competition setup.

**Typical reporting**: Decimal (< 1 means beat the baseline). Lower is better.

### WAPE (Weighted Absolute Percentage Error)

**What it measures**: MAE divided by the total absolute target value. Avoids MAPE's blow-up at zeros by aggregating the denominator.

**Formula**:

```
WAPE = sum_i |y_pred_i - y_true_i| / sum_i |y_true_i|
```

**Edge cases / pitfalls**:

- Still undefined if `sum_i |y_true_i| = 0`.
- Sometimes called "MAD/Mean".

**Typical reporting**: Percentage. Lower is better. Common in retail / demand forecasting.

### CRPS (Continuous Ranked Probability Score)

**What it measures**: Strictly proper scoring rule for probabilistic forecasts of a univariate continuous variable. Generalizes MAE: CRPS of a deterministic forecast (Dirac mass at the prediction) equals the absolute error.

**Formula** (integral form):

```
CRPS(F, y) = integral_{-inf}^{inf} (F(z) - 1[z >= y])^2 dz
```

Equivalent kernel form (Gneiting & Raftery):

```
CRPS(F, y) = E_F |X - y| - 0.5 * E_F |X - X'|
```

where `X, X'` are independent samples from `F`.

**Edge cases / pitfalls**:

- Requires the predicted CDF (or many samples) - point forecasts cannot be scored.
- For Gaussian `F = N(mu, sigma^2)` a closed form exists in terms of the standard normal pdf/cdf.

**Typical reporting**: Real number in target units. Lower is better. Standard on weather and probabilistic forecasting (M5 Uncertainty, GEFCom).

### Pinball Loss (cross-link)

See section 6 (Regression). In time series the average over multiple quantile levels is sometimes called WQL (Weighted Quantile Loss) and is the de-facto standard for probabilistic forecast evaluation in Amazon's GluonTS and several recent foundation-model TS papers.

---

## 8. Information Retrieval and Ranking

### Precision@k

**What it measures**: Fraction of the top-`k` retrieved items that are relevant. Captures the user's experience of "first page of results".

**Formula**:

```
P@k = (1 / k) * sum_{j = 1}^{k} rel_j
```

where `rel_j in {0, 1}` is the relevance of the j-th retrieved item.

**Edge cases / pitfalls**:

- Penalizes systems with too few relevant items in the database (cannot reach 1.0 if fewer than `k` relevant items exist).
- Insensitive to ordering within the top-`k`.

**Typical reporting**: Decimal. Higher is better.

### Recall@k

**What it measures**: Fraction of all relevant items that appear in the top-`k`. Captures how much of the relevant set the system can find.

**Formula**:

```
R@k = (number of relevant items in top k) / (total relevant items)
```

**Edge cases / pitfalls**:

- Trivially 1.0 if `k >= total relevant items`.
- Undefined when the query has no relevant items.

**Typical reporting**: Decimal. Higher is better.

### MAP (Mean Average Precision)

**What it measures**: Mean of Average Precision (AP) across queries, where AP averages precision at every recall level. Captures both ordering and recall.

**Formula**:

```
AP_q = sum_{k = 1}^{n} P@k * rel_k / R_q
MAP = (1 / |Q|) * sum_q AP_q
```

where `R_q` is the total number of relevant items for query `q`.

**Edge cases / pitfalls**:

- Binary relevance only; for graded relevance use nDCG.
- Different libraries truncate to different `k` (MAP@100, MAP@10).

**Typical reporting**: Decimal. Higher is better. Default eval on MTEB Reranking and on TREC ad-hoc retrieval.

### MRR (Mean Reciprocal Rank)

**What it measures**: Mean of `1 / rank_first_relevant`. Captures the position of the first correct answer. Suited to "I want one good answer" tasks (QA, sentence retrieval).

**Formula**:

```
MRR = (1 / |Q|) * sum_q 1 / rank_first_relevant_q
```

If no relevant item appears in the returned list, the reciprocal rank is 0.

**Edge cases / pitfalls**:

- Insensitive to anything after the first relevant hit.
- Typical truncation: MRR@10.

**Typical reporting**: Decimal. Higher is better. Voorhees / TREC heritage.

### nDCG@k (Normalized Discounted Cumulative Gain)

**What it measures**: Position-discounted relevance, normalized by the ideal ordering. Handles graded relevance (not just binary). The standard retrieval metric in MTEB (nDCG@10) and BEIR.

**Formula**:

```
DCG_lin@k    = sum_{i = 1}^{k} rel_i / log2(i + 1)
DCG_exp@k    = sum_{i = 1}^{k} (2^rel_i - 1) / log2(i + 1)
IDCG@k       = DCG@k computed on the ideal ranking
nDCG@k       = DCG@k / IDCG@k
```

The exponential-gain variant is more common in modern leaderboards (TREC, MS MARCO, BEIR).

**Edge cases / pitfalls**:

- Different log bases (`log2` vs natural) produce numerically different but rank-equivalent scores.
- IDCG = 0 (no relevant items) typically yields nDCG = 0 by convention.

**Typical reporting**: Decimal. Higher is better. Source: Jarvelin & Kekalainen, TOIS 2002.

### Hit@k (HitRate@k)

**What it measures**: Indicator: whether at least one relevant item appears in the top-`k`. Coarse but interpretable. Used heavily in knowledge-graph completion (Hit@1, Hit@3, Hit@10).

**Formula**:

```
Hit@k_q = 1[any rel_i = 1 for i in 1..k]
HitRate@k = (1 / |Q|) * sum_q Hit@k_q
```

**Edge cases / pitfalls**:

- Hit@1 is equivalent to top-1 accuracy when there is exactly one correct answer.
- Information-poor: a system improving the *position* of correct answers within `[2, 10]` gets no credit.

**Typical reporting**: Decimal. Higher is better.

### R-Precision

**What it measures**: Precision at rank `R`, where `R` is the number of relevant items for the query. Self-normalizing across queries with different relevance-set sizes.

**Formula**:

```
R-Precision_q = (number of relevant items in top R_q) / R_q
```

**Edge cases / pitfalls**:

- Undefined when `R_q = 0`; usually excluded.

**Typical reporting**: Decimal. Higher is better. TREC tradition.

---

## 9. Sentence Similarity and Embeddings (MTEB-relevant)

### Cosine Similarity

**What it measures**: Cosine of the angle between two embedding vectors. The default similarity for normalized embeddings.

**Formula**:

```
cos(u, v) = (u . v) / (||u|| * ||v||)
```

**Edge cases / pitfalls**:

- Undefined when either vector is zero.
- Equivalent to dot product when both vectors are L2-normalized; many libraries pre-normalize.

**Typical reporting**: Decimal in `[-1, 1]`. Used in STS, retrieval, clustering.

### Dot Product

**What it measures**: Raw inner product. Magnitude-sensitive: longer vectors are "more similar" to everything. Used in DPR and some ColBERT-style retrievers that rely on learned magnitudes.

**Formula**:

```
dot(u, v) = sum_i u_i * v_i
```

**Edge cases / pitfalls**:

- Not bounded; not a true similarity.
- Sensitive to embedding norm collapse during training.

### Euclidean Distance

**What it measures**: L2 distance. Lower is more similar. Less common for normalized embeddings, more common in metric-learning settings (triplet loss).

**Formula**:

```
d(u, v) = sqrt(sum_i (u_i - v_i)^2)
```

**Edge cases / pitfalls**:

- For L2-normalized vectors, `d(u, v)^2 = 2 * (1 - cos(u, v))`; ranking-equivalent to cosine.

### Pearson Correlation

**What it measures**: Linear correlation between predicted and true similarities. Default on the older STS benchmarks (STS-12 through STS-16).

**Formula**:

```
r = sum_i (x_i - x_bar) * (y_i - y_bar) / (sqrt(sum_i (x_i - x_bar)^2) * sqrt(sum_i (y_i - y_bar)^2))
```

**Edge cases / pitfalls**:

- Sensitive to outliers and to nonlinear-but-monotonic relationships.
- Undefined when either variable has zero variance.

**Typical reporting**: Decimal in `[-1, 1]`. Higher is better.

### Spearman Correlation

**What it measures**: Pearson correlation of the ranks of the two variables. Captures monotonic but not necessarily linear relationships. The modern default for STS (MTEB and SentEval).

**Formula**: Pearson applied to `rank(x)`, `rank(y)`.

**Edge cases / pitfalls**:

- Tied ranks reduce to averaged ranks; libraries differ in tie handling.

**Typical reporting**: Decimal in `[-1, 1]`, often multiplied by 100 in MTEB. Higher is better. Default eval for MTEB STS tasks.

### V-measure

**What it measures**: External clustering quality. Harmonic mean of homogeneity (each cluster contains a single class) and completeness (each class is grouped together). Default in MTEB clustering.

**Formula**:

```
h = 1 - H(C | K) / H(C)       # homogeneity
c = 1 - H(K | C) / H(K)       # completeness
V = 2 * h * c / (h + c)
```

where `H` is conditional / marginal entropy and `C`, `K` are class / cluster labels.

**Edge cases / pitfalls**:

- Symmetric in homogeneity and completeness; weighted variants exist.
- Bounded `[0, 1]`. Source: Rosenberg & Hirschberg 2007.

**Typical reporting**: Decimal. Higher is better.

### Adjusted Rand Index (ARI)

**What it measures**: Pairwise agreement between two clusterings, adjusted for chance.

**Formula**:

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

where `RI` is the Rand Index (agreement on whether each pair belongs together / apart).

**Edge cases / pitfalls**:

- Range `[-1, 1]`; 0 is chance, 1 is identical clustering.
- Not chance-corrected for very imbalanced cluster sizes (use AMI in that case).

**Typical reporting**: Decimal. Higher is better.

### Normalized Mutual Information (NMI)

**What it measures**: Mutual information between cluster and class labels, normalized to `[0, 1]`. Multiple normalization choices exist (arithmetic, geometric, max).

**Formula** (geometric normalization):

```
NMI(C, K) = I(C; K) / sqrt(H(C) * H(K))
```

**Edge cases / pitfalls**:

- Different normalizations produce different numbers; specify which.
- Not adjusted for chance (use AMI for that).

**Typical reporting**: Decimal. Higher is better.

### Average Precision (pair-classification sense)

**What it measures**: AP applied to pair-classification tasks (given a pair of sentences, predict whether they are paraphrase/entailment). Each pair has a binary label; the model produces a score; AP is computed as in section 1.

**Formula**: Same as binary AP; see section 1.

**Edge cases / pitfalls**:

- Distinct from detection AP and from MAP.
- MTEB pair-classification tasks (e.g. SprintDuplicateQuestions) report AP as the primary metric.

**Typical reporting**: Decimal. Higher is better.

---

## 10. NLP Generation (Translation / Summarization / Captioning)

### BLEU

**What it measures**: N-gram precision against one or more references, geometric-mean-aggregated across `n = 1..4`, multiplied by a brevity penalty that punishes too-short candidates. The de-facto MT metric since Papineni et al. 2002.

**Formula**:

```
BLEU = BP * exp(sum_{n=1}^{4} w_n * log p_n)

p_n = (sum over candidate sentences of clipped n-gram matches) / (total n-grams in candidate)
BP  = 1            if c > r
    = exp(1 - r/c) if c <= r
```

where `c` is candidate length, `r` is effective reference length, and `w_n = 1/4` are uniform weights.

**Smoothing methods (Chen & Cherry 2014, methods 0-7)**:

- Method 0: no smoothing; geometric mean is 0 if any p_n is 0 (catastrophic at sentence level).
- Method 1: add `epsilon` to zero-precision counts.
- Method 2: add `1` to numerator and denominator for `n > 1`.
- Method 3: NIST geometric sequence smoothing - replace zero precisions with `1 / 2^k`.
- Method 4: Method 3 scaled by `1 / ln(len(candidate))` for short sentences.

**SacreBLEU**: Standardized BLEU implementation (Post 2018, arXiv:1804.08771) that pins tokenization, casing, and smoothing into a reproducible signature string (e.g. `BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+tok.13a+version.2.x`). Always quote the SacreBLEU signature; raw "BLEU = 28.5" without it is unreproducible.

**Edge cases / pitfalls**:

- Sentence-level BLEU with method 0 is famously brittle.
- BLEU is precision-only; recall is implicit via the brevity penalty.
- Multiple references reduce variance; cross-system comparisons need the same reference count.

**Typical reporting**: BLEU * 100. Higher is better. Default on WMT, IWSLT.

### ROUGE-1, ROUGE-2

**What it measures**: N-gram recall / F1 of system summaries against reference summaries. ROUGE-N looks at n-grams of length `n`. Recall variants weight recall; F1 variants are the harmonic mean.

**Formula** (recall variant):

```
ROUGE-N_recall = (sum over reference summaries of matched n-grams) / (sum over reference summaries of total n-grams)
```

F1 combines precision and recall analogously.

**Edge cases / pitfalls**:

- Multiple references averaged or max-pooled depending on configuration.
- Lin's original implementation differed from the now-standard `rouge-score` (Google) and `pyrouge` (Perl).

**Typical reporting**: F1 percentage. Higher is better. Default on CNN/DailyMail, XSum, MEETING.

### ROUGE-L

**What it measures**: F1 over the Longest Common Subsequence (LCS) between candidate and reference. LCS allows non-contiguous matches and rewards sentence-level word ordering.

**Formula**:

```
R_lcs = LCS(cand, ref) / |ref|
P_lcs = LCS(cand, ref) / |cand|
F_lcs = (1 + beta^2) * R_lcs * P_lcs / (R_lcs + beta^2 * P_lcs)
```

Lin's original used `beta -> inf` (effectively recall); modern implementations report F1 with `beta = 1`. Summary-level ROUGE-L (ROUGE-Lsum) computes LCS per sentence and unions.

**Edge cases / pitfalls**:

- ROUGE-L vs ROUGE-Lsum can differ by several points; specify which.

**Typical reporting**: F1 percentage. Higher is better.

### METEOR

**What it measures**: Unigram F-mean (harmonic mean weighted toward recall) with stemming and synonym matching, plus a "fragmentation penalty" for word-order differences. Better human correlation than BLEU on MT.

**Formula** (sketch):

```
P     = matches / |cand|
R     = matches / |ref|
F_mean = 10 * P * R / (R + 9 * P)
Penalty = 0.5 * (chunks / matches)^3
METEOR  = F_mean * (1 - Penalty)
```

**Edge cases / pitfalls**:

- Language-specific resources (WordNet for English); coverage varies for low-resource languages.
- Hyperparameters tuned per language by the original authors.

**Typical reporting**: Decimal. Higher is better.

### chrF / chrF++

**What it measures**: Character n-gram F-score, optionally augmented with word n-grams (chrF++). Robust to morphology; performs well on agglutinative and morphologically rich languages where BLEU is weak.

**Formula**:

```
chrF_beta = (1 + beta^2) * chrP * chrR / (beta^2 * chrP + chrR)
```

`chrF` averages character n-grams up to order 6; `chrF++` adds word n-grams up to order 2. WMT uses `beta = 2` (recall-weighted).

**Edge cases / pitfalls**:

- Tokenization-agnostic, which is a major advantage over BLEU.
- Source: Popovic 2015.

**Typical reporting**: chrF * 100. Higher is better. WMT modern default alongside BLEU.

### TER (Translation Edit Rate)

**What it measures**: Minimum number of edits (insertions, deletions, substitutions, shifts) needed to transform the candidate into the reference, normalized by reference length.

**Formula**:

```
TER = (edits) / (|reference|)
```

**Edge cases / pitfalls**:

- Lower is better, unlike most other generation metrics.
- HTER (human-targeted TER) requires post-editing data.

**Typical reporting**: Percentage. Lower is better. WMT companion metric.

### BERTScore

**What it measures**: Token-level cosine similarity between contextual embeddings of candidate and reference, aggregated into P/R/F1 over a greedy alignment. Correlates better with human judgment than n-gram metrics for paraphrase-style generation.

**Formula**:

```
R_BERT = (1 / |ref|) * sum_{x_i in ref} max_{x_j in cand} cos(e(x_i), e(x_j))
P_BERT = symmetric
F_BERT = 2 * P_BERT * R_BERT / (P_BERT + R_BERT)
```

**Edge cases / pitfalls**:

- Numbers depend strongly on the encoder + layer used; specify model (e.g. `roberta-large` layer 17 for English).
- Default rescaling against random baseline shifts scores; specify whether `rescale_with_baseline=True`.

**Typical reporting**: F1 in [0, 1]. Higher is better. Source: Zhang et al., arXiv:1904.09675.

### BLEURT

**What it measures**: Learned regression metric: a BERT-derived encoder fine-tuned on human MT-quality ratings. Direct human-likeness score.

**Formula**: Model output; no closed form.

**Edge cases / pitfalls**:

- Specific version matters (BLEURT-20, BLEURT-base, BLEURT-large).
- Anchored to specific human-rating distributions; numbers across versions are not directly comparable.

**Typical reporting**: Real number. Higher is better. Source: Sellam et al., arXiv:2004.04696.

### COMET / COMETKIWI

**What it measures**: Cross-lingual encoder-decoder model trained to predict human MT quality scores from `<source, candidate, reference>` triples. COMETKIWI is the reference-free variant (predicts quality from `<source, candidate>` only) and is suitable for online MT QA.

**Formula**: Model output.

**Edge cases / pitfalls**:

- Score range differs by checkpoint (`wmt22-comet-da` returns roughly `[0, 1]`).
- Reference-free COMETKIWI is convenient but typically lower-correlated with humans than reference-based COMET.

**Typical reporting**: Decimal. Higher is better. Source: Rei et al., arXiv:2009.09025. WMT 2022+ official metric.

### Exact Match (SQuAD-style)

**What it measures**: Fraction of predictions that match the reference answer exactly (after normalization: lowercasing, article removal, punctuation stripping, whitespace collapse). Strict; partial credit goes to Token-level F1.

**Formula**:

```
EM = (1 / N) * sum_i 1[normalize(pred_i) == normalize(ref_i)]
```

**Edge cases / pitfalls**:

- "Normalization" rules differ between SQuAD, TriviaQA, NaturalQuestions; check the eval script.
- Multiple reference answers: take max EM across references.

**Typical reporting**: Percentage. Higher is better. SQuAD primary metric pair with token F1.

### Token-level F1 (SQuAD)

**What it measures**: F1 over the bag of tokens shared between predicted and reference answer. Partial credit for partially correct spans.

**Formula**:

```
TokenF1 = 2 * |tokens(pred) intersect tokens(ref)| / (|tokens(pred)| + |tokens(ref)|)
```

**Edge cases / pitfalls**:

- "No answer" case: F1 = 1 if both predict no-answer, else 0.
- Multiple references: max F1 across references.

**Typical reporting**: Percentage. Higher is better.

### Perplexity

**What it measures**: Exponentiated average negative log-likelihood per token under the model. Probabilistic measure of how surprised the model is by held-out text. Intrinsic LM eval.

**Formula**:

```
PPL = exp((1 / N) * sum_i -log P(x_i | x_{<i}))
```

where the sum is over tokens.

**Edge cases / pitfalls**:

- Tokenization-dependent; perplexities across models with different tokenizers are not directly comparable. Use bits-per-byte instead.
- Sliding-window vs full-context evaluation produces different numbers on long documents.

**Typical reporting**: Real number, often on WikiText-103 or The Pile validation. Lower is better.

### Bits per Character / Byte (BPC / BPB)

**What it measures**: Tokenization-invariant version of perplexity, expressed in bits per raw character (BPC) or byte (BPB) of the original text.

**Formula**:

```
BPB = (1 / num_bytes) * sum_i -log2 P(token_i | ...)
```

**Edge cases / pitfalls**:

- Multiply the cross-entropy (in nats per token) by `num_tokens / num_bytes / log(2)`.
- The standard in modern LM scaling-law papers because it factors out tokenizer choice.

**Typical reporting**: Decimal. Lower is better. Default on The Pile, RedPajama.

---

## 11. Code Generation

### pass@k

**What it measures**: Probability that at least one of `k` independently sampled completions passes all unit tests. Standard on HumanEval, MBPP, LiveCodeBench. The Codex paper popularized the unbiased estimator below; the naive `1 - (1 - p)^k` is biased.

**Formula** (Chen et al. 2021, unbiased):

```
pass@k = E_problems [ 1 - C(n - c, k) / C(n, k) ]
```

where for each problem `n` samples are drawn, `c` of them are correct, and `C(., .)` is the binomial coefficient. Set to 1.0 if `n - c < k`.

**Edge cases / pitfalls**:

- Requires `n >= k`. Conventionally `n = 200` for pass@1 / pass@10 / pass@100.
- Pass@1 with greedy decoding (n = 1) reduces to a simple accuracy; with sampling at `n > 1` it is the unbiased estimator.

**Typical reporting**: Percentage. Higher is better. Source: Chen et al., arXiv:2107.03374.

### pass@1

**What it measures**: Special case of pass@k. Reported either with greedy decoding (single deterministic sample) or with `n` samples and the unbiased estimator. The two protocols can differ by several points; specify which.

**Formula**: See pass@k with `k = 1`. With greedy decoding it is binary per problem.

**Typical reporting**: Percentage. Higher is better. Headline number on HumanEval and SWE-bench (where the "agent" generates a patch).

### Exact Match (code)

**What it measures**: Whether the generated code matches the reference exactly after some normalization (whitespace, comments). Weak proxy for correctness because many correct solutions are not byte-identical to the reference; mostly used in code-completion benchmarks where reference matching is appropriate.

**Formula**: As in section 10.

**Typical reporting**: Percentage. Higher is better. Common on line / token completion datasets (CodeXGLUE).

### CodeBLEU

**What it measures**: BLEU augmented with syntactic-AST overlap and dataflow-graph overlap, weighted across the four components. Designed to be more code-aware than plain BLEU but still cheap. Auxiliary metric only; pass@k dominates.

**Formula** (sketch):

```
CodeBLEU = a * BLEU + b * BLEU_weighted + g * MatchAST + d * MatchDataflow
```

with weights `(a, b, g, d)` typically `(0.25, 0.25, 0.25, 0.25)`.

**Edge cases / pitfalls**:

- AST and dataflow parsers exist only for a small set of languages; check support before reporting.
- Frequently mis-implemented; verify against the original repo.

**Typical reporting**: Decimal. Higher is better. Auxiliary.

---

## 12. Speech

### WER (Word Error Rate)

**What it measures**: Edit distance (insertions + deletions + substitutions) between hypothesis and reference word sequences, divided by reference length. Primary ASR metric.

**Formula**:

```
WER = (I + D + S) / N_ref
```

**Edge cases / pitfalls**:

- Can exceed 100% (lots of insertions on short references).
- Sensitive to normalization (punctuation, casing, contractions). Whisper / OpenAI use a custom normalizer; LibriSpeech leaderboards use a strict normalizer.

**Typical reporting**: Percentage. Lower is better. Standard on LibriSpeech, CommonVoice, AMI.

### CER (Character Error Rate)

**What it measures**: Same as WER but at the character level. More appropriate for languages without clear word boundaries (Chinese, Japanese) and for OCR.

**Formula**:

```
CER = (I + D + S) / N_chars_ref
```

**Edge cases / pitfalls**:

- Same normalization sensitivity as WER.
- CER is typically much lower than WER (characters share more between hyp and ref).

**Typical reporting**: Percentage. Lower is better.

### PER (Phoneme Error Rate)

**What it measures**: WER computed at the phoneme level. Used in speech recognition research and in TTS pronunciation evaluation.

**Formula**: As WER, with `N_phonemes_ref` in the denominator.

**Typical reporting**: Percentage. Lower is better.

### MOS (Mean Opinion Score)

**What it measures**: Subjective rating, typically a 1-5 Likert scale collected from crowd-sourced listeners, averaged across raters and utterances. The classical TTS / speech-codec quality metric.

**Formula**: No closed form. Conventionally:

```
MOS = mean over listener-utterance pairs of integer ratings in {1, 2, 3, 4, 5}
```

**Protocol caveats**:

- ITU-T P.800 specifies test conditions (anchored / non-anchored, headphone monitoring, quiet booth, training stimuli).
- Crowd-source MOS (Amazon MTurk) has higher variance and lower reliability than lab MOS.
- Confidence intervals matter: small (<0.1) differences are usually within noise.

**Typical reporting**: Decimal in [1, 5], plus 95% CI. Higher is better. Standard final metric in TTS papers.

---

## 13. Image Generation and Restoration

### PSNR (Peak Signal-to-Noise Ratio)

**What it measures**: Logarithmic ratio of peak signal to mean squared reconstruction error. Standard pixel-level fidelity metric in denoising, super-resolution, compression.

**Formula**:

```
PSNR = 10 * log10(MAX^2 / MSE)
```

where `MAX` is the maximum possible pixel value: 255 for uint8 images and 1.0 for float images normalized to `[0, 1]`. **This matters**: PSNR computed with `MAX = 1` for a normalized image is `20 * log10(255)` dB (about 48 dB) below the same image's PSNR with `MAX = 255`. Mixing conventions across papers is a common bug.

**Edge cases / pitfalls**:

- `MSE = 0` -> PSNR is infinite; libraries return a sentinel or a large number.
- Insensitive to perceptual structure; a uniformly-shifted image keeps PSNR but looks worse.

**Typical reporting**: Decibels (real number, typically 25-45 dB for image restoration). Higher is better.

### SSIM (Structural Similarity)

**What it measures**: Locally windowed comparison of luminance, contrast, and structure. Designed to track perceptual quality better than PSNR.

**Formula**:

```
SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) /
             ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

C1 = (K1 * L)^2,   K1 = 0.01
C2 = (K2 * L)^2,   K2 = 0.03
L  = dynamic range of the pixel values (255 for uint8, 1 for normalized float)
```

Computed with an 11x11 Gaussian window (`sigma = 1.5`) and averaged across the image.

**Edge cases / pitfalls**:

- Constants `C1`, `C2` are scale-dependent through `L` - same uint8-vs-float trap as PSNR.
- Mean / variance / covariance via Gaussian weighting, not uniform - some libraries use a uniform window and produce slightly different numbers.

**Typical reporting**: Decimal in `[-1, 1]`, typically `[0.5, 1.0]` for restoration. Higher is better. Source: Wang et al., IEEE TIP 2004.

### MS-SSIM (Multi-Scale SSIM)

**What it measures**: SSIM applied at multiple downsampled scales, then geometrically combined. Better correlation with human perception across viewing distances.

**Formula**:

```
MS-SSIM = (l_M(x, y))^alpha_M * prod_{j=1}^{M} (c_j(x, y))^beta_j * (s_j(x, y))^gamma_j
```

with default weights from Wang et al. 2003 (5 scales) and luminance computed only at the coarsest scale.

**Edge cases / pitfalls**:

- Image must be large enough for `M` downsamplings (5 scales needs `>= 161 px` per side with default 11x11 window).

**Typical reporting**: Decimal in [0, 1]. Higher is better.

### LPIPS (Learned Perceptual Image Patch Similarity)

**What it measures**: L2 distance between deep-feature activations of a pretrained network (AlexNet, VGG, or SqueezeNet), with per-channel linear weights learned to match human perceptual judgments.

**Formula**:

```
LPIPS(x, y) = sum_l (1 / H_l W_l) * sum_{h, w} ||w_l * (f_l_hw(x) - f_l_hw(y))||_2^2
```

where `f_l` is the l-th conv feature map and `w_l` are learned per-channel weights.

**Edge cases / pitfalls**:

- Backbone matters: AlexNet, VGG, and SqueezeNet variants give numerically different LPIPS scores.
- Inputs must be normalized to the backbone's expected range (typically `[-1, 1]`).

**Typical reporting**: Decimal, typically `[0, 1]`. Lower is better. Source: Zhang et al., arXiv:1801.03924.

### FID (Frechet Inception Distance)

**What it measures**: Wasserstein-2 distance between two Gaussians fit to Inception-v3 `pool3` features (2048-dim) of real and generated images. The dominant unconditional-generation metric since 2017.

**Formula**:

```
FID = ||mu_r - mu_g||_2^2 + Tr(C_r + C_g - 2 * (C_r * C_g)^(1/2))
```

where `(mu_r, C_r)` and `(mu_g, C_g)` are the empirical mean and covariance of real and generated features.

**Edge cases / pitfalls**:

- Highly biased at small sample sizes; report >= 10k samples per side. 50k is the de-facto standard on CIFAR / FFHQ / ImageNet.
- Numerically unstable matrix-square-root computation - use the off-the-shelf reference implementation (TTUR / clean-fid).
- Inception-v3 weights and preprocessing differ between TF and PyTorch reference implementations; numbers across implementations can differ by 1-2 points.

**Typical reporting**: Real number. Lower is better. Source: Heusel et al., arXiv:1706.08500.

### Inception Score (IS)

**What it measures**: Exponentiated KL divergence between conditional class distributions and the marginal class distribution, evaluated with Inception-v3 logits. Rewards images that are both confidently classified and diverse across classes.

**Formula**:

```
IS = exp(E_x KL(p(y | x) || p(y)))
```

**Edge cases / pitfalls**:

- Mode collapse to one image per class produces high IS but low diversity within class.
- Inception-v3-specific - IS is meaningless outside natural-image distributions resembling ImageNet.
- Numerically split-dependent; the canonical protocol is 10 splits of 5k samples.

**Typical reporting**: Real number. Higher is better. Largely superseded by FID, but co-reported on CIFAR-10 generation. Source: Salimans et al., arXiv:1606.03498.

### KID (Kernel Inception Distance)

**What it measures**: Unbiased estimator of the squared Maximum Mean Discrepancy between Inception features, using a polynomial kernel. Unbiased at any sample size, unlike FID.

**Formula**:

```
KID = MMD^2(P_r, P_g)
```

estimated with a degree-3 polynomial kernel on Inception features.

**Edge cases / pitfalls**:

- Same Inception-v3 caveats as FID.
- Numerically tiny (`~ 1e-3`); reported scaled by 1000 in some papers.

**Typical reporting**: Decimal. Lower is better. Source: Binkowski et al., arXiv:1801.01401.

### CLIP score (text-to-image)

**What it measures**: Cosine similarity between CLIP image and text embeddings, averaged across prompts. Standard text-image alignment metric for T2I generation.

**Formula**:

```
CLIP-score = (1 / N) * sum_i max(0, cos(CLIP_img(x_i), CLIP_txt(prompt_i)))
```

**Edge cases / pitfalls**:

- Encoder choice matters (CLIP ViT-B/32, ViT-L/14, OpenCLIP). Report explicitly.
- CLIP is biased toward its own training distribution; high CLIP-score does not equal photorealism.

**Typical reporting**: Decimal scaled by 100 in some libraries. Higher is better.

### NIQE (Natural Image Quality Evaluator)

**What it measures**: No-reference image quality metric based on the deviation of MSCN coefficients from a "naturalness" multivariate Gaussian fit to pristine images.

**Formula**: Mahalanobis distance between MSCN-feature statistics of the test image and a reference natural-image MVG.

**Edge cases / pitfalls**:

- Reference statistics are dataset-dependent (Mittal et al. fit on undistorted LIVE images).

**Typical reporting**: Decimal. Lower is better. Used in image-restoration leaderboards for no-reference quality.

### BRISQUE

**What it measures**: Another no-reference image quality metric using spatial MSCN coefficients passed through a learned SVR regressor trained on human DMOS ratings.

**Formula**: SVR output on a 36-dim MSCN feature vector.

**Edge cases / pitfalls**:

- Score range typically [0, 100]; lower means better quality.
- Tied to LIVE dataset training; less reliable on far-OOD distortions.

**Typical reporting**: Decimal. Lower is better. Companion to NIQE.

---

## 14. Calibration

### Expected Calibration Error (ECE)

**What it measures**: Average gap between confidence and accuracy across confidence bins. The standard scalar calibration metric.

**Formula** (equal-width binning):

```
ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|

B_m   = {i : confidence_i in ((m-1)/M, m/M]}
acc   = (1 / |B_m|) * sum_{i in B_m} 1[y_pred_i == y_true_i]
conf  = (1 / |B_m|) * sum_{i in B_m} p_i
```

**Equal-width** uses fixed bin boundaries (default `M = 15`). **Equal-mass** (sometimes "equal-frequency", or "adaptive binning") uses quantile-based bin edges so each bin has the same number of samples, which is more robust under skewed confidence distributions but harder to compare across models.

**Edge cases / pitfalls**:

- Number of bins matters: too few hides miscalibration; too many produces noisy per-bin estimates.
- Equal-width ECE is dominated by the most-populated bin, which is typically the confidence-near-1 bin for modern overconfident classifiers.
- Reliability of ECE estimates degrades with bin sparsity at small `N`.

**Typical reporting**: Decimal, often * 100. Lower is better. Source: Guo et al., arXiv:1706.04599.

### Maximum Calibration Error (MCE)

**What it measures**: Maximum gap between confidence and accuracy across bins. Worst-case calibration.

**Formula**:

```
MCE = max_m |acc(B_m) - conf(B_m)|
```

**Edge cases / pitfalls**:

- Dominated by sparse bins; very noisy with `M = 15` and small `N`.

**Typical reporting**: Decimal. Lower is better.

### Adaptive Calibration Error (ACE)

**What it measures**: ECE computed with equal-mass (adaptive) bins so each bin has the same number of samples. Robust to confidence-distribution skew.

**Formula**:

```
ACE = (1 / R * K) * sum_k sum_r |acc(B_r_k) - conf(B_r_k)|
```

where `R` is the number of equal-mass bins per class and `K` is the number of classes (Nixon et al. 2019, class-wise adaptive ECE).

**Edge cases / pitfalls**:

- More bins than ECE typically: `R = 100` is common.
- Class-wise vs top-label variants exist.

**Typical reporting**: Decimal. Lower is better. Source: Nixon et al., arXiv:1904.01685.

### Reliability Diagrams

**What it measures**: Visual companion to ECE. Bar plot of `(conf(B_m), acc(B_m))` pairs with the identity line as the perfectly-calibrated reference. Gap above the line indicates underconfidence; gap below indicates overconfidence.

**Formula**: Just the per-bin statistics from ECE; no scalar.

**Typical reporting**: Figure, not a number. Show alongside ECE.

### Brier Score Decomposition

**What it measures**: Decomposes Brier into three additive terms - reliability (calibration), resolution (separation of classes), and uncertainty (intrinsic label noise).

**Formula**:

```
Brier = Reliability - Resolution + Uncertainty

Reliability  = sum_m (|B_m| / N) * (conf(B_m) - acc(B_m))^2
Resolution   = sum_m (|B_m| / N) * (acc(B_m) - y_bar)^2
Uncertainty  = y_bar * (1 - y_bar)        # binary
```

with `y_bar` the marginal positive rate.

**Edge cases / pitfalls**:

- Reliability is closely related to ECE (squared rather than absolute differences).
- Uncertainty is dataset-intrinsic and not improvable by the model.

**Typical reporting**: Three numbers. Lower reliability, higher resolution are good.

---

## 15. Anomaly / Open-Set / OOD Detection

### AUROC (OOD)

**What it measures**: ROC-AUC where the positive class is OOD (or anomalous) and the score is a model-derived novelty score (max-softmax-probability complement, energy, Mahalanobis distance, etc.). Standard headline metric on OOD-detection benchmarks (CIFAR-10 vs CIFAR-100, ImageNet-1k vs SUN/Places/Textures).

**Formula**: As ROC-AUC in section 1.

**Edge cases / pitfalls**:

- Direction of the score matters; flip if "higher = OOD" vs "higher = ID".
- Score regimes vary wildly across methods; AUROC is invariant to monotonic transforms so this is fine.

**Typical reporting**: Percentage. Higher is better.

### FPR @ 95% TPR

**What it measures**: False positive rate when the threshold is set to capture 95% of in-distribution samples (true positive rate = 0.95). Captures the cost of high-recall OOD detection.

**Formula**:

```
FPR95 = FPR at threshold tau* where TPR(tau*) = 0.95
```

**Edge cases / pitfalls**:

- Very sensitive to score estimation in the tail; report with confidence intervals.
- Convention: ID = positive in this definition (so "TPR" is recall on ID).

**Typical reporting**: Percentage. Lower is better. Standard companion to AUROC on OpenOOD.

### AUPR (positive vs negative class variant)

**What it measures**: PR-AUC. Two variants depending on whether the positive class is OOD ("AUPR-out") or ID ("AUPR-in"). Useful when classes are extremely imbalanced (e.g. anomaly detection with 0.1% anomalies).

**Formula**: As binary AP in section 1.

**Edge cases / pitfalls**:

- AUPR-in and AUPR-out are NOT complementary; both are typically reported.
- Highly sensitive to the base rate; not directly comparable across datasets with different OOD prevalence.

**Typical reporting**: Percentage. Higher is better.

---

## 16. Reinforcement Learning

### Mean Episodic Return

**What it measures**: Average undiscounted (or sometimes discounted) reward summed over an episode, averaged across evaluation episodes. The primary RL metric.

**Formula**:

```
ReturnEpisodic = (1 / N_episodes) * sum_e sum_t r_e_t
```

**Edge cases / pitfalls**:

- Reward scaling varies wildly across environments; not comparable across tasks.
- High variance: 10+ seeds are needed for credible comparisons (Henderson et al. 2018).

**Typical reporting**: Real number. Higher is better.

### Success Rate

**What it measures**: Fraction of episodes that achieve the task goal (binary success per episode). Common in goal-conditioned RL, robotics, and game agents.

**Formula**:

```
SR = (1 / N_episodes) * sum_e 1[goal_achieved_e]
```

**Edge cases / pitfalls**:

- "Success" definitions can be lenient or strict; specify.
- Less informative than partial returns when difficulty varies across episodes.

**Typical reporting**: Percentage. Higher is better.

### Sample Efficiency

**What it measures**: Performance per unit of environment interaction (env steps) or per gradient step. Crucial in sparse-reward and physical-robot RL.

**Formula** (one common operational definition):

```
SampleEfficiency = ReturnEpisodic(at step T) / T_env_steps
```

Or the inverse: steps to reach a target return:

```
StepsToTarget = min { T : ReturnEpisodic(T) >= R_target }
```

**Edge cases / pitfalls**:

- "Steps" can be env-frames, env-steps, or wall-clock; specify which.
- Off-policy methods often report against gradient steps; on-policy methods against env steps.

**Typical reporting**: Two-panel learning curve (return vs steps), or a scalar like AUC under the learning curve.

### Atari Human-Normalized Score

**What it measures**: Per-game score linearly normalized to `0 = random policy`, `1 = trained human`. Aggregated across the 26 / 57 / 200 Atari games of the chosen suite via mean or median.

**Formula**:

```
HumanNormScore_g = (Score_g - Score_random_g) / (Score_human_g - Score_random_g)
```

**Edge cases / pitfalls**:

- Mean-HNS is dominated by a few games where agents massively exceed human (Atlantis, Video Pinball); median is more robust.
- "Random" and "human" baselines from Mnih et al. 2015 are still in use 11 years later; cite the table you use.

**Typical reporting**: Decimal (1.0 = human level). Higher is better.

### IQM (Interquartile Mean of Normalized Scores)

**What it measures**: Mean of normalized scores after dropping the bottom 25% and top 25%. Recommended by Agarwal et al. 2021 (Rliable) as a robust aggregate that beats mean (outlier-sensitive) and median (high-variance) on Atari, DM Control, and other RL suites.

**Formula**:

```
IQM(x) = (2 / N) * sum_{i = ceil(N/4) + 1}^{floor(3N/4)} x_(i)
```

where `x_(i)` is the i-th order statistic.

**Edge cases / pitfalls**:

- Always report with stratified bootstrap confidence intervals (Rliable's `rliable.metrics.aggregate_iqm`).
- Defined on normalized scores - raw rewards don't make sense across games.

**Typical reporting**: Decimal with 95% CI. Higher is better. Source: Agarwal et al., arXiv:2108.13264.

---

## Cross-Cutting Pitfalls

Failure modes that span every section above. Read this list before believing any leaderboard number.

- **Averaging conventions**: Micro / macro / weighted / samples are not interchangeable. A 10-point gap between micro-F1 and macro-F1 on imbalanced data is normal, not a bug. Always state which is reported.
- **Train-eval distribution shift**: A model "tuned" on the test set (or on a dataset trained with the same web crawl that surfaces eval samples) inflates every metric in this document. Contamination audits matter; the Big-Bench, MMLU, and SWE-bench leaderboards all carry contamination warnings.
- **Single-seed reporting**: A 1-point gap on a single seed is almost always within run-to-run noise. Modern RL recommendations (Agarwal et al.) and NLP recommendations (Dodge et al. 2019) call for >=5 seeds and bootstrap CIs. Most leaderboards still report single-seed numbers; treat them as upper bounds.
- **Floating-point precision in eval**: Some metrics (FID, Inception Score, MMD, KID) involve large-matrix linear algebra that is non-deterministic in fp16/fp32 across hardware. Differences of 0.1-0.3 are common between repeated evaluations on different GPUs.
- **Top-1 vs top-5 sloppy reporting**: Pre-2018 ImageNet papers sometimes reported top-5 error without saying so; modern reporting is top-1 unless noted. When citing a number, double-check the column header.
- **"Validation set" is overloaded**: ImageNet-1K val IS the test set used in papers; GLUE dev != test; SQuAD dev != test. Same model can have very different numbers on "val" depending on which dataset you mean.
- **Eval-time augmentation**: TTA (multi-crop, horizontal-flip averaging) adds 0.3-1.0 ImageNet points "for free" and is sometimes silently enabled. Check the inference protocol.
- **Ensembling**: Headline numbers are often single-checkpoint; ensembles of 5-10 models add another 0.5-2 points and are usually flagged but not always. Compare like to like.
- **Tokenizer-dependent metrics**: Perplexity, BLEU (before SacreBLEU), pass@k normalization, and any token-F1 metric depend on tokenization. Cross-model comparisons need fixed tokenization or token-free metrics (BPB, chrF).
- **Reference set size**: BLEU with 4 references can be 5-10 points higher than BLEU with 1 reference on the same translations. NIST 2008 (4 refs) and WMT (1 ref) are not comparable.
- **Decoding hyperparameters**: pass@k, BLEU, ROUGE on summarization all shift with temperature, top-p, top-k, beam size, and length penalty. Without these, "we got 80 BLEU" is unreproducible.
- **Image preprocessing**: ImageNet "224x224" can mean (a) `Resize(256) + CenterCrop(224)` (PyTorch), (b) `Resize(232) + CenterCrop(224)` (timm modern recipe), or (c) `RandomResizedCrop(224)` train-time stats applied at test (wrong). Numbers can differ by 0.5-1.0 points.
- **Round-trip serialization**: A model that scores X on the trainer's in-memory copy can score X - epsilon after `model.save() / model.load()` if the layer's `get_config()` drops state. Always validate post-load metrics before reporting.

---

## Sources

- [scikit-learn model evaluation reference](https://scikit-learn.org/stable/modules/model_evaluation.html) - canonical definitions of classification, regression, clustering metrics
- [torchmetrics documentation](https://lightning.ai/docs/torchmetrics/stable/) - PyTorch reference implementations covering most metrics in this file
- [pycocotools / COCO eval](https://cocodataset.org/#detection-eval) - official COCO detection / segmentation / panoptic protocols
- Muennighoff et al., **MTEB** (arXiv:[2210.07316](https://arxiv.org/abs/2210.07316)) - embedding-model evaluation task taxonomy
- Eigen et al., **depth estimation evaluation protocol** (arXiv:[1406.2283](https://arxiv.org/abs/1406.2283)) - SILog, threshold accuracies, AbsRel
- Papineni et al., **BLEU** (ACL 2002) - [Original paper](https://aclanthology.org/P02-1040/)
- Post, **SacreBLEU** (arXiv:[1804.08771](https://arxiv.org/abs/1804.08771)) - reproducible BLEU signature
- Lin, **ROUGE** (ACL 2004 Workshop) - [paper](https://aclanthology.org/W04-1013/)
- Popovic, **chrF** (WMT 2015) - [paper](https://aclanthology.org/W15-3049/)
- Zhang et al., **BERTScore** (arXiv:[1904.09675](https://arxiv.org/abs/1904.09675))
- Sellam et al., **BLEURT** (arXiv:[2004.04696](https://arxiv.org/abs/2004.04696))
- Rei et al., **COMET** (arXiv:[2009.09025](https://arxiv.org/abs/2009.09025))
- Heusel et al., **FID** (arXiv:[1706.08500](https://arxiv.org/abs/1706.08500))
- Salimans et al., **Inception Score** (arXiv:[1606.03498](https://arxiv.org/abs/1606.03498))
- Binkowski et al., **KID** (arXiv:[1801.01401](https://arxiv.org/abs/1801.01401))
- Zhang et al., **LPIPS** (arXiv:[1801.03924](https://arxiv.org/abs/1801.03924))
- Wang et al., **SSIM** (IEEE TIP 2004) - [paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- Kirillov et al., **Panoptic Quality** (arXiv:[1801.00868](https://arxiv.org/abs/1801.00868))
- Guo et al., **calibration / ECE / temperature scaling** (arXiv:[1706.04599](https://arxiv.org/abs/1706.04599))
- Nixon et al., **Adaptive Calibration Error (ACE)** (arXiv:[1904.01685](https://arxiv.org/abs/1904.01685))
- Hyndman & Koehler, **MASE** (International Journal of Forecasting 2006) - [paper](https://robjhyndman.com/papers/mase.pdf)
- Gneiting & Raftery, **CRPS / strictly proper scoring rules** (J. Amer. Stat. Assoc. 2007)
- Chen et al., **Codex / pass@k** (arXiv:[2107.03374](https://arxiv.org/abs/2107.03374))
- Agarwal et al., **IQM / Rliable** (arXiv:[2108.13264](https://arxiv.org/abs/2108.13264))
- Cohen, **kappa** (Educ. Psychol. Meas. 1960); Matthews, **MCC** (Biochim. Biophys. Acta 1975)
- Jarvelin & Kekalainen, **nDCG** (ACM TOIS 2002)
- Voorhees, **MRR / TREC**; Manning, Raghavan, Schutze, **MAP** (Introduction to Information Retrieval, 2008)
- Rezatofighi et al., **GIoU** (arXiv:[1902.09630](https://arxiv.org/abs/1902.09630))
- Zheng et al., **DIoU / CIoU** (arXiv:[1911.08287](https://arxiv.org/abs/1911.08287))
- Chen & Cherry, **BLEU smoothing methods** (WMT 2014) - [paper](https://aclanthology.org/W14-3346/)
- Rosenberg & Hirschberg, **V-measure** (EMNLP 2007) - [paper](https://aclanthology.org/D07-1043/)
