# MNIST Activation Function Comparison - Experimental Results Analysis

## Executive Summary

The experiment compared six activation functions on MNIST digit classification: **ReLU**, **Tanh**, **GELU**, **Mish**, **SaturatedMish (α=1.0)**, and **SaturatedMish (α=2.0)**. All activation functions achieved exceptional performance with test accuracies exceeding 99.5%, demonstrating that activation choice has minimal impact on final accuracy for this relatively simple dataset. However, notable differences emerged in calibration quality, loss values, and convergence characteristics.

**Key Finding**: While **Mish** achieved the highest test accuracy (99.69%), **GELU** demonstrated the best overall balance of accuracy (99.66%), calibration (lowest mean entropy of 0.0043), and loss optimization (lowest test loss of 0.0508).

## Detailed Performance Analysis

### 1. Test Set Accuracy Rankings

| Rank | Activation | Test Accuracy | Top-5 Accuracy | Test Loss |
|------|------------|---------------|----------------|-----------|
| 1    | **Mish**   | **99.69%**    | 100.00%        | 0.0524    |
| 2    | GELU       | 99.66%        | 100.00%        | **0.0508**|
| 3    | ReLU       | 99.65%        | 99.99%         | 0.0573    |
| 3    | Sat_Mish_1 | 99.65%        | 100.00%        | 0.0541    |
| 5    | Sat_Mish_2 | 99.62%        | 100.00%        | 0.0549    |
| 6    | Tanh       | 99.59%        | 100.00%        | 0.0546    |

**Insights:**
- The accuracy differences are marginal (0.10% spread), indicating all activations are highly effective for MNIST
- Mish variants (including saturated versions) consistently perform in the top tier
- Traditional ReLU remains highly competitive despite its simplicity
- All activations except ReLU achieved perfect top-5 accuracy

### 2. Model Calibration Analysis

Calibration metrics reveal how well predicted probabilities align with actual outcomes:

| Activation | ECE ↓ | Brier Score ↓ | Mean Entropy |
|------------|-------|---------------|--------------|
| **Sat_Mish_1** | **0.0021** | **0.0024** | 0.0052 |
| GELU | 0.0028 | 0.0040 | **0.0043** |
| Tanh | 0.0029 | 0.0038 | 0.0072 |
| Mish | 0.0032 | 0.0051 | 0.0095 |
| Sat_Mish_2 | 0.0032 | 0.0052 | 0.0074 |
| ReLU | 0.0039 | 0.0049 | 0.0055 |

**Key Observations:**
- **SaturatedMish (α=1.0)** shows the best calibration with lowest ECE and Brier Score
- **GELU** has the lowest mean entropy, indicating more confident (less uncertain) predictions
- Standard **Mish** has the highest mean entropy (0.0095), suggesting more conservative probability estimates
- All models are well-calibrated (ECE < 0.004), indicating reliable probability predictions

### 3. Training Dynamics

Final validation metrics provide insights into convergence behavior:

| Activation | Val Accuracy | Val Loss | Loss Gap (Test - Val) |
|------------|--------------|----------|----------------------|
| GELU       | 99.64%       | 0.0495   | 0.0013              |
| ReLU       | 99.63%       | 0.0564   | 0.0009              |
| Mish       | 99.61%       | 0.0463   | 0.0061              |
| Sat_Mish_1 | 99.61%       | 0.0505   | 0.0036              |
| Sat_Mish_2 | 99.58%       | 0.0504   | 0.0045              |
| Tanh       | 99.57%       | 0.0537   | 0.0009              |

**Training Insights:**
- **GELU** and **ReLU** achieved the highest validation accuracies
- **Mish** shows the largest generalization gap (0.0061), suggesting slight overfitting
- **ReLU** and **Tanh** have the smallest generalization gaps, indicating stable training

## Activation Function Characteristics

### Mish Family Analysis
- **Standard Mish**: Highest test accuracy but also highest uncertainty (mean entropy)
- **SaturatedMish (α=1.0)**: Best calibration with excellent accuracy
- **SaturatedMish (α=2.0)**: Slightly lower performance, suggesting α=1.0 is optimal

### Traditional vs Modern Activations
- **ReLU**: Remains highly competitive with simple, efficient computation
- **Tanh**: Lowest accuracy but stable training and good calibration
- **GELU**: Best modern activation, balancing all metrics effectively

## Recommendations

### For Different Use Cases:

1. **Maximum Accuracy**: Choose **Mish**
   - Best for: Competition scenarios where every 0.01% matters
   - Trade-off: Slightly higher computational cost and uncertainty

2. **Best Calibration**: Choose **SaturatedMish (α=1.0)**
   - Best for: Applications requiring reliable probability estimates
   - Trade-off: Marginally lower accuracy than Mish

3. **Best Overall Balance**: Choose **GELU**
   - Best for: Production systems requiring good accuracy and calibration
   - Benefits: Low loss, confident predictions, strong validation performance

4. **Computational Efficiency**: Choose **ReLU**
   - Best for: Resource-constrained environments
   - Benefits: Fastest computation, competitive accuracy, simple implementation

## Statistical Significance

Given the small differences in accuracy (0.10% range), multiple runs with different random seeds would be needed to establish statistical significance. The consistent patterns across multiple metrics (accuracy, calibration, loss) suggest these differences are meaningful but modest.

## Conclusions

1. **All modern activations are viable for MNIST** - the dataset is sufficiently simple that activation choice has minimal impact on final accuracy

2. **Calibration differences are more pronounced** than accuracy differences, making this a better criterion for selection

3. **GELU emerges as the best general-purpose choice**, offering excellent performance across all metrics

4. **SaturatedMish with α=1.0** shows promise for applications requiring well-calibrated probabilities

5. **The Mish family** (standard and saturated variants) demonstrates strong performance, validating its effectiveness as a modern activation function

## Future Research Directions

1. Test on more complex datasets (CIFAR-100, ImageNet) where activation differences may be more pronounced
2. Analyze computational efficiency and memory usage
3. Investigate the impact of activation functions on adversarial robustness
4. Study the interaction between activation functions and other architectural choices (normalization, depth)
5. Examine activation behavior in different parts of the network (early vs late layers)