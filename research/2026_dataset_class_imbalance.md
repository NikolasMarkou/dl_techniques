# Comprehensive Guide to Handling Class Imbalance in Machine Learning

Class imbalance remains one of machine learning's most persistent challenges, causing models to systematically underperform on minority classes. **Three fundamentally different paradigms** have emerged: modifying the loss function to penalize errors asymmetrically, manipulating the dataset itself to achieve better representation, and applying calibration-based methods with mathematical validity guarantees. This guide provides an exhaustive survey of all three perspectives, including both mainstream techniques with proven empirical success and an increasingly influential critique arguing that proper calibration—not data manipulation—is the principled solution.

---

## Part I: Loss Function Approaches

The loss function perspective addresses imbalance by adjusting how errors are weighted during optimization, without modifying the underlying data distribution. These methods range from simple class weighting to sophisticated margin-based formulations with theoretical guarantees.

### Cost-Sensitive Learning: The Theoretical Foundation

Cost-sensitive learning acknowledges that misclassification errors carry different penalties—a false negative for cancer detection costs more than a false positive. Charles Elkan's seminal 2001 IJCAI paper, "The Foundations of Cost-Sensitive Learning," proved that **any cost-insensitive learner can be made cost-sensitive** through threshold adjustment or sample reweighting.

The framework uses a cost matrix C where C(i,j) represents the cost of predicting class j when the true label is i. For binary classification, the optimal decision threshold becomes:

$$t^* = \frac{C(+1,-1) - C(-1,-1)}{C(+1,-1) - C(+1,+1) + C(-1,+1) - C(-1,-1)}$$

Implementation takes three forms: threshold adjustment at prediction time, sample reweighting during training, or direct incorporation into the objective function. This theoretical framework underpins nearly all subsequent loss-based methods for imbalanced learning.

### Class-Weighted Cross-Entropy

The most straightforward adaptation multiplies each class's contribution to the loss by a weight inversely proportional to its frequency:

$$\mathcal{L}_{WCE} = -\sum_i w_i \cdot y_i \cdot \log(\hat{y}_i)$$

**Three weighting schemes dominate practice:**

| Scheme | Formula | Characteristics |
|--------|---------|-----------------|
| Inverse frequency | $w_j = \frac{N}{C \times n_j}$ | N = total samples, C = class count, $n_j$ = samples in class j |
| Inverse square root | $w_j = \frac{1}{\sqrt{n_j}}$ | Gentler correction |
| Median frequency | Normalized by median class size | Balanced approach |

While simple to implement, class weighting can destabilize training under extreme imbalance ratios exceeding 1:100, often requiring combination with other techniques.

### Focal Loss

Lin et al.'s 2017 ICCV paper introduced Focal Loss to address the extreme foreground-background imbalance (often 1:1000) in object detection. The key insight: standard cross-entropy treats all correctly classified examples equally, allowing easy negatives to dominate gradients.

The focal loss adds a modulating factor:

$$FL(p_t) = -\alpha_t(1 - p_t)^\gamma \cdot \log(p_t)$$

The focusing parameter γ controls down-weighting of easy examples:
- When γ = 0: reduces to standard cross-entropy
- At γ = 2 (recommended): **well-classified examples with $p_t = 0.9$ have their loss reduced by 100×**

The balance factor α typically equals 0.25 for positives. RetinaNet with focal loss achieved state-of-the-art on COCO detection.

**Notable variants include:**
- Focal Tversky Loss for segmentation
- Adaptive Focal Loss with dynamic γ adjustment
- Unified Focal Loss generalizing across Dice and cross-entropy families

### Effective Number of Samples

Cui et al.'s CVPR 2019 paper observed that inverse frequency weighting performs poorly on real-world long-tailed distributions because it ignores **diminishing marginal returns** from additional samples. As sample count increases, new samples increasingly overlap with existing ones in feature space.

The effective number of samples is defined as:

$$E_n = \frac{1 - \beta^n}{1 - \beta}$$

where n is sample count and β ∈ [0,1) controls overlap assumptions. Class-balanced weights become:

$$w_j = \frac{1 - \beta}{1 - \beta^{n_j}}$$

This formula interpolates smoothly:
- When β → 0: $E_n$ → 1 (no reweighting)
- When β → 1: $E_n$ → n (standard inverse frequency)
- **β = 0.999 works well empirically** across datasets

Combining class-balanced weighting with focal loss (CB-Focal) achieved significant improvements on ImageNet-LT and iNaturalist 2018.

### LDAM Loss: Theoretical Generalization Guarantees

Cao et al.'s NeurIPS 2019 paper derived Label-Distribution-Aware Margin (LDAM) loss from first principles, minimizing a margin-based generalization bound. The key theoretical result establishes that **optimal per-class margins scale as $n_j^{-1/4}$**.

The LDAM loss modifies the softmax with class-dependent margins:

$$\mathcal{L}_{LDAM} = -\log\left[\frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y} e^{z_j}}\right]$$

where $\Delta_j = \frac{C}{n_j^{1/4}}$ and C is a tunable hyperparameter. Larger margins for minority classes provide stronger regularization, preventing overfitting on limited samples.

**Deferred Re-Weighting (DRW):** Train initially with LDAM alone to learn good representations, then apply class-balanced weighting in later epochs. This two-stage approach prevents early overfitting to minority classes.

### Asymmetric Loss for Multi-Label Classification

Ben-Baruch et al.'s ICCV 2021 paper addressed a distinct imbalance problem: in multi-label classification, each image has few positive labels but many negatives.

Asymmetric Loss (ASL) uses different focusing parameters:

$$\mathcal{L}^+ = (1 - p)^{\gamma^+} \cdot \log(p) \quad \text{for positive labels}$$
$$\mathcal{L}^- = (p_m)^{\gamma^-} \cdot \log(1 - p_m) \quad \text{for negative labels}$$

The probability shifting mechanism $p_m = \max(p - m, 0)$ hard-thresholds easy negatives. Recommended settings: γ⁺ = 0, γ⁻ = 4, m = 0.05.

### Region-Based Losses for Segmentation

For semantic segmentation with small regions of interest, **Dice Loss** directly optimizes the Dice coefficient:

$$\mathcal{L}_{Dice} = 1 - \frac{2\sum p_i \cdot g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$

**Tversky Loss** (Salehi et al., 2017) generalizes Dice by controlling the precision-recall trade-off:

$$TI(\alpha, \beta) = \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}$$

Setting β > α penalizes false negatives more heavily, favoring recall—critical for lesion detection. Common settings: α = 0.3, β = 0.7.

**Focal Tversky Loss** adds focal modulation: $(1 - TI)^{1/\gamma}$

**Compound losses** combining Dice and cross-entropy consistently outperform either alone:
$$\mathcal{L}_{Combo} = \alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{Dice}$$

### Long-Tail Recognition Losses

Instance segmentation on datasets like LVIS with 1,000+ categories spawned dedicated loss functions:

**Seesaw Loss** (Wang et al., CVPR 2021) applies:
- Mitigation factors $M_{ij} = (\sigma_j/\sigma_i)^p$ that reduce punishment on tail classes
- Compensation factors $C_{ij} = 1 + \alpha \cdot q_j^q$ that increase penalties for misclassification

**Equalization Loss (EQL)** ignores negative gradients from rare categories entirely during training. This method won 1st place in the LVIS Challenge 2019.

**Equalized Focal Loss** makes the focal parameter category-dependent based on frequency.

---

## Part II: Dataset-Level Methods

Rather than changing how errors are weighted, dataset methods alter the data itself through selective removal (undersampling), duplication (oversampling), or synthesis of new examples.

### Undersampling Methods

#### Random Undersampling (RUS)

Simply removes majority class samples until a target ratio is achieved. Despite obvious information loss, Drummond and Holte's influential 2003 workshop paper demonstrated that **undersampling often outperforms oversampling** for decision trees, particularly when the majority class contains redundant samples.

**Advantages:** Computational efficiency, reduced overfitting risk
**Disadvantages:** May discard informative boundary samples, results vary across seeds

#### Tomek Links

Tomek's 1976 IEEE paper defined a Tomek link: two samples from different classes are each other's nearest neighbors. Such pairs identify **borderline or noisy samples** at class boundaries.

Formally, samples $(x_i, x_j)$ form a Tomek link if:
- $d(x_i, x_j) < d(x_i, x_k)$ for all other samples $x_k$
- $d(x_i, x_j) < d(x_j, x_k)$ for all other samples $x_k$

Removing the majority class member from each Tomek link cleans the decision region conservatively.

#### Edited Nearest Neighbors (ENN)

Wilson's 1972 editing rule removes any sample whose class differs from the majority vote of its k nearest neighbors:

$$\text{Remove } x_i \text{ where } \text{mode}(\{\text{class}(x_j) : x_j \in kNN(x_i)\}) \neq \text{class}(x_i)$$

**Variants:**
- 'all' variant: remove if any neighbor differs (more aggressive)
- 'mode' variant: remove only if most neighbors differ
- Repeated ENN: apply iteratively until convergence
- All-kNN: extend editing across k=1,2,...,K values

#### Neighborhood Cleaning Rule (NCL)

Laurikkala's 2001 method combines ENN with additional boundary cleaning:
1. **Phase 1:** Apply ENN to remove misclassified majority samples
2. **Phase 2:** For each minority sample, if all k neighbors belong to majority class and majority exceeds half the minority size, remove those majority neighbors

NCL prioritizes **data quality over aggressive balancing**.

#### One-Sided Selection (OSS)

Kubat and Matwin's 1997 ICML paper combined Tomek links with the Condensed Nearest Neighbor (CNN) rule:
- Tomek links remove borderline/noisy samples
- CNN identifies the minimal consistent subset

This removes both noise at boundaries and redundant majority samples far from the decision boundary.

#### NearMiss Algorithms

Mani and Zhang's 2003 work introduced three variants:

| Variant | Selection Criterion | Characteristics |
|---------|---------------------|-----------------|
| NearMiss-1 | Smallest average distance to k nearest minority samples | Close to boundary, sensitive to noise |
| NearMiss-2 | Distance to k farthest minority samples | Class overlap region, less noise sensitive |
| NearMiss-3 | Two-stage: pre-select M nearest majority neighbors, then select by largest average distance | **Most robust to noise** |

### Oversampling Methods

#### Random Oversampling (ROS)

Duplicates minority samples with replacement until achieving target balance. While adding no new information, it provides a simple baseline that surprisingly works well with robust classifiers like Random Forest.

**Primary concern:** Overfitting through exact duplication, causing models to memorize specific instances.

#### SMOTE (Synthetic Minority Over-sampling Technique)

Chawla et al.'s 2002 JAIR paper introduced SMOTE, generating synthetic samples by interpolating between minority instances and their k-nearest neighbors:

$$x_{new} = x_i + \lambda \cdot (x_j - x_i)$$

where $\lambda \sim U(0,1)$ and $x_j$ is a randomly selected neighbor.

**Variants addressing SMOTE limitations:**

| Variant | Modification | Purpose |
|---------|--------------|---------|
| Borderline-SMOTE | Only synthesize from borderline samples | Focus on decision boundary |
| Safe-Level-SMOTE | Weight interpolation by "safe level" | Avoid noise generation |
| SMOTE-ENN | Apply ENN after SMOTE | Clean noisy synthetics |
| SMOTE-Tomek | Apply Tomek links after SMOTE | Remove borderline noise |

#### ADASYN (Adaptive Synthetic Sampling)

He et al.'s 2008 IJNN paper generates more samples for harder-to-learn minority instances. The algorithm computes a density ratio:

$$r_i = \frac{\Delta_i}{k}$$

where $\Delta_i$ is the count of majority samples among k nearest neighbors. After normalizing to $\hat{r}_i = r_i/\sum r_i$, synthetic samples per instance: $g_i = \hat{r}_i \times G$.

ADASYN adapts to local data characteristics, **shifting the decision boundary toward difficult examples**. With over 4,000 citations, it remains among the most influential oversampling methods.

**Concern:** Generating samples in overlapping regions may introduce noise.

#### Cluster-Based Methods

Cluster-based undersampling replaces majority samples with K-means centroids, preserving distributional structure. This **prototype generation approach** creates new representative points rather than selecting existing ones.

Variations include:
- Selecting samples nearest to centroids (preserving original data)
- Combining clustering with instance selection for diversity

### Modern Augmentation Techniques

#### Mixup and Variants

**Mixup** (Zhang et al., 2017) creates virtual examples through convex combinations:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**Remix** adapts Mixup for imbalance by assigning labels favoring the minority class.

**CutMix** pastes rectangular regions between images with proportional label mixing.

#### GAN-Based Generation

Learns the minority class distribution directly:
- **GAMO** (Generative Adversarial Minority Oversampling): three-player game generating samples near class boundaries
- **BAGAN**: pre-trains autoencoder before GAN fine-tuning for stability

**Advantages:** Diverse, realistic samples; handles high-dimensional data
**Disadvantages:** Requires substantial minority samples; significant computational resources

### Curriculum Learning

Bengio et al.'s 2009 ICML paper showed that **presenting samples from easy to hard** improves learning.

**Self-Paced Learning (SPL):** Model determines difficulty based on current loss, dynamically selecting samples.

**Dynamic Curriculum Learning:** Combines sampling and loss schedulers, adjusting distributions and difficulty throughout training.

**Hard Example Mining / OHEM:** Opposite approach—focusing on highest-loss samples. Focal loss implements soft hard example mining through its modulating factor.

---

## Part III: The Calibration Perspective—Conformal Prediction and Venn-Abers

A minority but increasingly influential voice argues that the standard approach to class imbalance—resampling the data—is fundamentally misguided. **Dr. Valeriy Manokhin**, who completed his PhD in Machine Learning at Royal Holloway under Professor Vladimir Vovk (the inventor of Conformal Prediction), has emerged as a vocal proponent of this perspective.

### The Core Thesis

Manokhin's central argument: **the problem with imbalanced data isn't the imbalance itself—it's poor probability calibration**. Rather than manipulating training data through resampling, conformal methods provide validity guarantees that SMOTE fundamentally cannot.

His most comprehensive treatment appears in Chapter 11 ("Handling Imbalanced Data") of his book *"Practical Guide to Applied Conformal Prediction in Python"* (Packt Publishing, 2024). The chapter argues:

> "While a significant portion of resources in the field suggest using resampling methods, including undersampling, oversampling, and techniques such as SMOTE, it's crucial to note that **these recommendations often sidestep foundational theory and practical application**."

### Critique of SMOTE and Resampling

Manokhin has cited research showing that SMOTE, random undersampling, and random oversampling:
- Did not improve classification metrics
- **Destroyed classifier calibration** in the process
- Were only tested on weak learners (C4.5, Ripper, Naive Bayes) in the original paper
- Do not work well with modern ML models like XGBoost, LightGBM, and Random Forests

Key paper cited: "The harm of class imbalance corrections for risk prediction models" (van Smeden et al.), which concludes that **"outcome imbalance is not a problem in itself, imbalance correction may even worsen results."**

### Conformal Prediction as the Alternative

Conformal Prediction, developed by Vovk, Gammerman, and Shafer, provides **mathematically guaranteed coverage** regardless of class distribution. The framework maintains validity without manipulating training data.

**Key properties:**
- Distribution-free: no assumptions about underlying data distribution
- Finite-sample validity guarantees
- Works with any underlying model (model-agnostic)
- Handles classes independently through Mondrian (class-conditional) approach

The **Mondrian approach** handles classes independently with separate calibration sets, providing valid confidence estimates for both majority and minority classes without requiring resampling.

### Venn-Abers Predictors for Calibration

Venn-ABERS predictors provide superior calibration compared to:
- Platt's scaling
- Isotonic regression
- Standard calibration methods

Manokhin's academic work includes "Multi-class probabilistic classification using inductive and cross Venn–Abers predictors" (COPA 2017), demonstrating these methods are **more accurate than both uncalibrated predictors and existing calibration methods**.

**Multi-class extensions** are particularly valuable since class imbalance is especially challenging in multi-class settings.

### The Recommended Framework

Synthesizing Manokhin's approach:

1. **Understand the problem first:** Determine whether imbalance is static or dynamic; consider whether the problem might be better framed as anomaly detection rather than classification

2. **Avoid SMOTE and resampling:** These methods change the data distribution, destroy calibration, and lack theoretical foundations—they address a symptom rather than the underlying problem

3. **Use cost-sensitive learning:** Apply class-weighted cost functions rather than synthetic data generation

4. **Apply Conformal Prediction:** The framework provides mathematically guaranteed coverage regardless of class distribution

5. **Calibrate with Venn-Abers:** For probability estimates, Venn-ABERS predictors provide superior calibration, particularly important when classes are imbalanced

### Resources for Implementation

**GitHub repositories (github.com/valeman):**
- **Awesome Conformal Prediction** (3,300+ stars): definitive resource, cited in Kevin Murphy's *"Probabilistic Machine Learning: An Introduction"*
- **Venn-Abers-Predictor**: implementation with offline, online, and cross-validation modes
- **Multi-class-probabilistic-classification**: code for multi-class Venn-ABERS

**Courses:** Applied Conformal Prediction course on Maven, with participants from Amazon, Apple, Google, Meta, Nike, BlackRock, Goldman Sachs, and Morgan Stanley. Notable testimonial: *"I learned to use methods like Venn-Abers and saw how conformal prediction tackles imbalanced data. It inspired my Python library, TinyCP, now used in my work. I've dropped SMOTE."*

---

## Part IV: Method Selection Guide

### By Imbalance Severity

| Imbalance Ratio | Recommended Approaches |
|-----------------|------------------------|
| Mild (< 1:10) | Class-weighted cross-entropy; potentially no intervention needed |
| Moderate (1:10 to 1:100) | Focal loss or class-balanced losses + informed undersampling (Tomek links, ENN); consider Venn-Abers calibration |
| Severe (> 1:100) | Multiple techniques: LDAM + DRW, or ADASYN, or GAN-based generation; strongly consider conformal methods |
| Long-tailed (1000+ classes) | Seesaw or EQL + curriculum learning; Mondrian conformal prediction |

### By Task Type

| Task | Preferred Methods |
|------|-------------------|
| Binary classification | Weighted CE, focal loss, Venn-Abers calibration |
| Multi-class classification | Class-balanced losses, LDAM, multi-class Venn-Abers |
| Semantic segmentation | Dice loss, Tversky loss, compound losses |
| Object detection | Focal loss (foreground-background imbalance) |
| Multi-label classification | Asymmetric loss |
| Medical/high-stakes | Conformal prediction (validity guarantees), cost-sensitive learning |

### Combining Approaches

Dataset methods and loss modifications are not mutually exclusive:
- Informed undersampling + focal loss
- ADASYN + class-balanced losses
- Any approach + Venn-Abers post-hoc calibration

**Critical consideration:** If well-calibrated probability estimates are required (medical diagnosis, risk assessment, financial decisions), the calibration-based approach may be essential regardless of other methods used.

---

## Conclusion

Three decades of research have produced a rich toolkit for imbalanced learning:

**Loss function approaches** excel when preserving all data is important. Key methods include focal loss (object detection), LDAM with theoretical guarantees (long-tailed distributions), and region-based losses (segmentation).

**Dataset methods** provide complementary benefits through noise removal (Tomek links, ENN), adaptive synthesis (ADASYN, SMOTE variants), and structure preservation (cluster-based methods).

**The calibration perspective** represents a paradigm shift: rather than treating imbalance as a data problem requiring data manipulation, it frames imbalance as causing poor calibration—solvable through methods with mathematical validity guarantees like Conformal Prediction and Venn-Abers.

The most impactful methods share common traits: **solid theoretical grounding**, **intuitive mechanisms**, and **empirical validation**. Understanding each method's theoretical assumptions and matching them to specific problem characteristics remains the key to success.

For practitioners, the choice isn't necessarily between these paradigms. Modern best practice may involve: using appropriate loss functions during training, applying informed dataset modifications where beneficial, and ensuring proper calibration through conformal methods—particularly when probability estimates matter for downstream decisions.