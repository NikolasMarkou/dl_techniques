# Mathematically Grounded Methods for Fixing Dataset Imbalance

Class imbalance remains one of machine learning's most persistent challenges, causing models to systematically underperform on minority classes. **Two fundamentally different approaches** have emerged: modifying the loss function to penalize errors asymmetrically, or manipulating the dataset itself to achieve better representation. This report provides a comprehensive survey of both perspectives, emphasizing mathematically rigorous methods with proven empirical success across thousands of publications.

## Loss function approaches reweight the learning signal itself

The loss function perspective addresses imbalance by adjusting how errors are weighted during optimization, without modifying the underlying data distribution. These methods range from simple class weighting to sophisticated margin-based formulations with theoretical guarantees.

### Cost-sensitive learning established the theoretical foundation

Cost-sensitive learning acknowledges that misclassification errors carry different penalties—a false negative for cancer detection costs more than a false positive. Charles Elkan's seminal 2001 IJCAI paper, "The Foundations of Cost-Sensitive Learning," proved that **any cost-insensitive learner can be made cost-sensitive** through threshold adjustment or sample reweighting.

The framework uses a cost matrix C where C(i,j) represents the cost of predicting class j when the true label is i. For binary classification, the optimal decision threshold becomes:

**t* = [C(+1,-1) - C(-1,-1)] / [C(+1,-1) - C(+1,+1) + C(-1,+1) - C(-1,-1)]**

Implementation takes three forms: threshold adjustment at prediction time, sample reweighting during training, or direct incorporation into the objective function. This theoretical framework underpins nearly all subsequent loss-based methods for imbalanced learning.

### Class-weighted cross-entropy provides the simplest practical solution

The most straightforward adaptation multiplies each class's contribution to the loss by a weight inversely proportional to its frequency. Standard cross-entropy extends to:

**L_WCE = -Σᵢ wᵢ · yᵢ · log(ŷᵢ)**

Three weighting schemes dominate practice. **Inverse frequency weighting** sets wⱼ = N/(C × nⱼ), where N is total samples, C is class count, and nⱼ is samples in class j. **Inverse square root weighting** uses wⱼ = 1/√nⱼ for a gentler correction. **Median frequency weighting** normalizes by the median class size. While simple to implement, class weighting can destabilize training under extreme imbalance ratios exceeding 1:100, often requiring combination with other techniques.

### Focal loss revolutionized dense object detection

Lin et al.'s 2017 ICCV paper introduced Focal Loss to address the extreme foreground-background imbalance (often 1:1000) in object detection. The key insight: standard cross-entropy treats all correctly classified examples equally, allowing easy negatives to dominate gradients.

The focal loss adds a modulating factor:

**FL(pₜ) = -αₜ(1 - pₜ)^γ · log(pₜ)**

The focusing parameter γ controls down-weighting of easy examples. When γ = 0, this reduces to standard cross-entropy. At γ = 2 (the recommended value), **well-classified examples with pₜ = 0.9 have their loss reduced by 100×** compared to cross-entropy. The balance factor α typically equals 0.25 for positives. RetinaNet with focal loss achieved state-of-the-art on COCO detection, demonstrating effectiveness for extreme imbalance ratios.

Notable variants include **Focal Tversky Loss** for segmentation, **Adaptive Focal Loss** with dynamic γ adjustment, and **Unified Focal Loss** which generalizes across Dice and cross-entropy families.

### Effective number of samples corrects for data overlap

Cui et al.'s CVPR 2019 paper observed that inverse frequency weighting performs poorly on real-world long-tailed distributions because it ignores **diminishing marginal returns** from additional samples. As sample count increases, new samples increasingly overlap with existing ones in feature space.

The effective number of samples is defined as:

**Eₙ = (1 - β^n) / (1 - β)**

where n is sample count and β ∈ [0,1) controls overlap assumptions. Class-balanced weights become:

**wⱼ = (1 - β) / (1 - β^nⱼ)**

This formula interpolates smoothly: when β → 0, Eₙ → 1 (no reweighting); when β → 1, Eₙ → n (standard inverse frequency). **β = 0.999 works well empirically** across datasets. Combining class-balanced weighting with focal loss (CB-Focal) achieved significant improvements on ImageNet-LT and iNaturalist 2018.

### LDAM loss provides theoretical generalization guarantees

Cao et al.'s NeurIPS 2019 paper derived Label-Distribution-Aware Margin (LDAM) loss from first principles, minimizing a margin-based generalization bound. The key theoretical result establishes that **optimal per-class margins scale as nⱼ^(-1/4)**.

The LDAM loss modifies the softmax with class-dependent margins:

**L_LDAM = -log[e^(zᵧ - Δᵧ) / (e^(zᵧ - Δᵧ) + Σⱼ≠ᵧ e^zⱼ)]**

where **Δⱼ = C / nⱼ^(1/4)** and C is a tunable hyperparameter. Larger margins for minority classes provide stronger regularization, preventing overfitting on limited samples.

The authors introduced **Deferred Re-Weighting (DRW)**: train initially with LDAM alone to learn good representations, then apply class-balanced weighting in later epochs. This two-stage approach prevents early overfitting to minority classes and consistently outperforms immediate reweighting across CIFAR-LT, Tiny ImageNet-LT, and iNaturalist.

### Asymmetric loss handles multi-label positive-negative imbalance

Ben-Baruch et al.'s ICCV 2021 paper addressed a distinct imbalance problem: in multi-label classification, each image has few positive labels but many negatives. Standard losses cause optimization to be dominated by negative gradients.

Asymmetric Loss (ASL) uses different focusing parameters for positives and negatives:

**L⁺ = (1 - p)^γ⁺ · log(p)** for positive labels  
**L⁻ = (pₘ)^γ⁻ · log(1 - pₘ)** for negative labels

The probability shifting mechanism **pₘ = max(p - m, 0)** hard-thresholds easy negatives, also helping with label noise. Recommended settings (γ⁺ = 0, γ⁻ = 4, m = 0.05) achieved state-of-the-art on MS-COCO, Pascal-VOC, and Open Images.

### Region-based losses dominate medical image segmentation

For semantic segmentation with small regions of interest, **Dice Loss** directly optimizes the Dice coefficient:

**L_Dice = 1 - (2Σpᵢ·gᵢ + ε) / (Σpᵢ + Σgᵢ + ε)**

This relative metric inherently handles imbalance since it measures overlap rather than per-pixel accuracy. **Tversky Loss** (Salehi et al., 2017) generalizes Dice by controlling the precision-recall trade-off:

**TI(α, β) = TP / (TP + α·FP + β·FN)**

Setting β > α penalizes false negatives more heavily, favoring recall—critical for lesion detection. Common settings use α = 0.3, β = 0.7. **Focal Tversky Loss** adds focal modulation: (1 - TI)^(1/γ), further emphasizing hard examples. Compound losses combining Dice and cross-entropy (L_Combo = α·L_CE + (1-α)·L_Dice) consistently outperform either alone.

### Long-tail recognition requires specialized formulations

Instance segmentation on datasets like LVIS with 1,000+ categories and extreme imbalance spawned dedicated loss functions. **Seesaw Loss** (Wang et al., CVPR 2021) observes that negative gradients from head classes overwhelm positive gradients for tail classes. It applies mitigation factors Mᵢⱼ = (σⱼ/σᵢ)^p that reduce punishment on tail classes and compensation factors Cᵢⱼ = 1 + α·qⱼ^q that increase penalties for misclassification.

**Equalization Loss (EQL)** takes a simpler approach: ignore negative gradients from rare categories entirely during training. This method won 1st place in the LVIS Challenge 2019. **Equalized Focal Loss** further refines this by making the focal parameter category-dependent based on frequency.

## Dataset-level methods modify the training distribution directly

Rather than changing how errors are weighted, dataset methods alter the data itself through selective removal (undersampling), duplication (oversampling), or synthesis of new examples. These techniques can be combined with loss modifications for compounded effect.

### Random undersampling trades information for speed

Random Undersampling (RUS) simply removes majority class samples until a target ratio is achieved. Despite obvious information loss, Drummond and Holte's influential 2003 workshop paper demonstrated that **undersampling often outperforms oversampling** for decision trees, particularly when the majority class contains redundant samples.

The method's appeal lies in computational efficiency—smaller datasets train faster—and reduced risk of overfitting. It works best on large datasets with redundant majority samples and forms the basis of ensemble methods like EasyEnsemble and BalanceCascade. However, random selection may discard informative boundary samples, and results vary across different random seeds.

### Tomek links clean decision boundaries

Tomek's 1976 IEEE paper defined a Tomek link: two samples from different classes are each other's nearest neighbors. Such pairs identify **borderline or noisy samples** at class boundaries. Removing the majority class member from each Tomek link cleans the decision region without aggressive undersampling.

Formally, samples (xᵢ, xⱼ) form a Tomek link if d(xᵢ, xⱼ) < d(xᵢ, xₖ) for all other samples xₖ, and d(xᵢ, xⱼ) < d(xⱼ, xₖ) for all other samples xₖ. The method is conservative—removing few samples—and is typically combined with other techniques. Computational cost scales with pairwise distance calculations, making it expensive for large datasets.

### Edited nearest neighbors removes noisy samples

Wilson's 1972 editing rule removes any sample whose class differs from the majority vote of its k nearest neighbors. For the majority class, this eliminates:

**xᵢ where mode({class(xⱼ) : xⱼ ∈ kNN(xᵢ)}) ≠ class(xᵢ)**

This identifies and removes noisy or mislabeled majority samples embedded in minority regions. The **'all' variant** (remove if any neighbor differs) is more aggressive than the **'mode' variant** (remove only if most neighbors differ). Repeated ENN applies editing iteratively until convergence, while All-kNN extends editing across k=1,2,...,K values.

### Neighborhood Cleaning Rule focuses on data quality

Laurikkala's 2001 NCL method combines ENN with additional boundary cleaning in two phases. Phase 1 applies ENN to remove misclassified majority samples. Phase 2 examines each minority sample's neighbors: if all k neighbors belong to the majority class and the majority class exceeds half the minority class size, those majority neighbors are removed.

NCL prioritizes **data quality over aggressive balancing**, particularly effective for identifying difficult small classes while preserving minority representation.

### One-sided selection combines cleaning and condensing

Kubat and Matwin's 1997 ICML paper introduced One-Sided Selection (OSS), combining Tomek links with the Condensed Nearest Neighbor (CNN) rule. Tomek links remove borderline/noisy samples, while CNN identifies the minimal consistent subset—majority samples that would be misclassified if removed.

This removes both noise at boundaries and redundant majority samples far from the decision boundary, achieving substantial dataset reduction while preserving classification accuracy.

### NearMiss algorithms use distance-based selection heuristics

Mani and Zhang's 2003 work introduced three NearMiss variants for controlled undersampling. **NearMiss-1** selects majority samples with smallest average distance to k nearest minority samples—keeping samples close to the boundary but sensitive to noise. **NearMiss-2** uses distance to k farthest minority samples, selecting samples in the class overlap region with less noise sensitivity.

**NearMiss-3** operates in two stages: first keep M nearest majority neighbors for each minority sample, then from this subset select samples with largest average distance to nearest minority neighbors. This **pre-selection makes NearMiss-3 most robust to noise**.

### Random oversampling risks overfitting through duplication

Random Oversampling (ROS) duplicates minority samples with replacement until achieving target balance. While adding no new information, it provides a simple baseline that surprisingly works well with robust classifiers like Random Forest.

The primary concern is overfitting: exact duplication can cause models to memorize specific instances, creating overly specific decision boundaries. Risk increases with higher oversampling ratios. Combining ROS with regularization techniques or using it alongside ensemble methods mitigates these concerns.

### ADASYN adapts synthesis to learning difficulty

He et al.'s 2008 IJNN paper introduced Adaptive Synthetic Sampling (ADASYN), which **generates more samples for harder-to-learn minority instances**. The algorithm computes a density ratio for each minority sample:

**rᵢ = Δᵢ/k**

where Δᵢ is the count of majority samples among k nearest neighbors. Higher ratios indicate samples in difficult, overlapping regions. After normalizing to r̂ᵢ = rᵢ/Σrᵢ, the number of synthetic samples per instance becomes gᵢ = r̂ᵢ × G, where G is total samples needed.

ADASYN adapts to local data characteristics, **shifting the decision boundary toward difficult examples**. With over 4,000 citations, it remains among the most influential oversampling methods. However, generating samples in overlapping regions may introduce noise, and performance depends on k and balance parameters.

### Cluster-based methods preserve data structure

Cluster-based undersampling replaces majority samples with K-means centroids, reducing the class while preserving distributional structure. The number of clusters equals the desired majority sample count, with centroids serving as representative prototypes.

This **prototype generation approach** differs from selection-based methods—it creates new points rather than choosing existing ones. Effectiveness depends on data naturally forming clusters. Variations include selecting samples nearest to centroids (preserving original data points) or combining clustering with instance selection for both inter-class and intra-class diversity.

### Modern augmentation extends beyond geometric transforms

Traditional augmentation (rotation, flipping, scaling) applied preferentially to minority classes increases their representation without exact duplication. **Mixup** (Zhang et al., 2017) creates virtual examples through convex combinations:

**x̃ = λxᵢ + (1-λ)xⱼ,  ỹ = λyᵢ + (1-λ)yⱼ**

where λ ~ Beta(α, α). **Remix** adapts Mixup for imbalance by assigning labels favoring the minority class while maintaining feature mixing. **CutMix** pastes rectangular regions between images with proportional label mixing.

**GAN-based generation** learns the minority class distribution directly. GAMO (Generative Adversarial Minority Oversampling) uses a three-player game to generate samples near class boundaries. BAGAN pre-trains an autoencoder before GAN fine-tuning for more stable generation. These methods generate diverse, realistic samples and handle high-dimensional data (images, text) effectively, but require substantial minority samples to learn distributions and significant computational resources.

### Curriculum learning orders samples by difficulty

Bengio et al.'s foundational 2009 ICML paper showed that **presenting samples from easy to hard** improves learning. Self-Paced Learning (SPL) lets the model determine difficulty based on current loss, dynamically selecting training samples. For imbalanced data, this starts with minority samples and easy majority samples, gradually incorporating harder examples.

**Dynamic Curriculum Learning** combines sampling and loss schedulers, adjusting from imbalanced→balanced distributions and easy→hard examples throughout training. **Hard Example Mining** takes the opposite approach—focusing on highest-loss samples—with Online Hard Example Mining (OHEM) selecting hardest examples per mini-batch. Focal loss implements soft hard example mining through its modulating factor.

## Choosing the right method depends on context and constraints

Method selection requires considering dataset size, imbalance ratio, computational budget, and task type. For **mild imbalance (< 1:10)**, class-weighted cross-entropy often suffices. **Moderate imbalance (1:10 to 1:100)** benefits from focal loss or class-balanced losses combined with informed undersampling like Tomek links or ENN.

**Severe imbalance (> 1:100)** typically requires multiple techniques: LDAM with deferred reweighting, ADASYN oversampling, or GAN-based generation. For **long-tailed recognition** with thousands of classes, specialized losses like Seesaw or EQL combined with curriculum learning show strongest results.

**Segmentation tasks** favor region-based losses (Dice, Tversky) over pixel-wise cross-entropy. **Object detection** benefits most from focal loss addressing foreground-background imbalance. **Multi-label classification** requires asymmetric losses handling positive-negative skew.

Dataset methods and loss modifications are not mutually exclusive—combining informed undersampling with focal loss or class-balanced losses often outperforms either approach alone. The key is understanding each method's theoretical assumptions and matching them to your specific imbalance characteristics.

## Conclusion

Two decades of research have produced a rich toolkit for imbalanced learning, from Elkan's cost-sensitive foundations to modern LDAM margins and seesaw losses. The most impactful methods share common traits: **solid theoretical grounding** (effective number of samples, margin-based generalization bounds), **intuitive mechanisms** (focal modulation, adaptive sampling), and **empirical validation** across diverse domains.

Loss function approaches excel when preserving all data is important and computational resources allow full dataset training. Dataset methods provide complementary benefits through noise removal (Tomek links, ENN), adaptive synthesis (ADASYN), and structure preservation (cluster-based methods). Curriculum learning bridges both perspectives by controlling how samples contribute to learning over time.

The field continues advancing rapidly, with recent work on equalized losses for extreme long-tail scenarios and GAN-based augmentation for high-dimensional minority classes. Yet classical methods like weighted cross-entropy and informed undersampling remain competitive baselines, often sufficient for practical applications with moderate imbalance.