# ResidualACFLayer: Enforcing White Noise Residuals in Time Series Forecasting

## Overview

The `ResidualACFLayer` is a Keras layer designed to monitor and regularize the autocorrelation function (ACF) of residuals in time series forecasting models. It addresses a critical but often overlooked aspect of forecasting: ensuring that model residuals approximate white noise, which indicates that the model has captured all systematic patterns in the data.

## Motivation

In time series forecasting, a well-specified model should produce residuals that are:
- Uncorrelated across time (no autocorrelation)
- Centered around zero (unbiased)
- Homoscedastic (constant variance)

The presence of significant autocorrelation in residuals indicates that the model has failed to capture some temporal structure in the data. This layer helps detect and optionally penalize such patterns during training.

## Key Features

1. **ACF Computation**: Efficiently computes the autocorrelation function of residuals up to a specified maximum lag using Keras operations.

2. **Monitoring Mode**: Can be used purely for diagnostic purposes without affecting model training, allowing you to track residual patterns.

3. **Regularization Mode**: Optionally adds a regularization term to the loss function that penalizes autocorrelation at specified lags.

4. **Flexible Targeting**: Allows focusing on specific lags (e.g., daily, weekly patterns) rather than all lags equally.

5. **Numerical Stability**: Includes epsilon parameter and careful normalization to handle edge cases.

## Mathematical Background

The autocorrelation function at lag k is defined as:

```
ACF(k) = Cov(r_t, r_{t-k}) / Var(r_t)
```

Where:
- `r_t` are the residuals at time t
- `Cov` is the covariance
- `Var` is the variance

For white noise, ACF(k) ≈ 0 for all k > 0.

The layer computes this efficiently using vectorized operations and adds a regularization term:

```
L_acf = λ * (Σ ACF(k)² + Σ max(|ACF(k)| - threshold, 0)²)
```

## Usage Examples

### 1. Monitoring Only (Diagnostic Mode)

```python
# Add to existing model for residual monitoring
predictions = model(inputs)
monitored_predictions = ResidualACFLayer(
    max_lag=40,
    regularization_weight=None,  # No regularization
    name="acf_monitor"
)([predictions, targets])
```

### 2. With Regularization

```python
# Enforce white noise residuals during training
regularized_predictions = ResidualACFLayer(
    max_lag=20,
    regularization_weight=0.1,
    target_lags=[1, 2, 3, 7, 14],  # Focus on specific patterns
    acf_threshold=0.1,
    name="acf_regularizer"
)([predictions, targets])
```

### 3. With Monitoring Callback

```python
# Track ACF statistics during training
acf_callback = ACFMonitorCallback(
    layer_name="acf_regularizer",
    log_frequency=100  # Log every 100 batches
)

model.fit(data, callbacks=[acf_callback])
```

## Parameters

- **max_lag** (int): Maximum lag to compute ACF for. Higher values capture longer-range dependencies but increase computation.

- **regularization_weight** (float or None): Weight for the ACF regularization loss. If None, the layer only monitors without affecting training.

- **target_lags** (list or None): Specific lags to target for regularization. Useful for focusing on known patterns (e.g., [7, 14] for weekly patterns).

- **acf_threshold** (float): Threshold above which ACF values are considered significant and penalized more heavily. Default is 0.1.

- **use_absolute_acf** (bool): Whether to penalize both positive and negative autocorrelations. Default is True.

- **epsilon** (float): Small constant for numerical stability in variance computation. Default is 1e-7.

## Best Practices

1. **Start with Monitoring**: Begin by using the layer in monitoring mode to understand your model's residual patterns.

2. **Target Known Patterns**: If you know your data has weekly seasonality, target lags 7, 14, 21, etc.

3. **Balance Regularization**: Too much weight on ACF regularization might hurt predictive performance. Start with small weights (0.01-0.1).

4. **Combine with Other Diagnostics**: ACF is one diagnostic among many. Also check residual distributions, heteroscedasticity, etc.

5. **Computational Considerations**: For very long sequences, consider using a smaller max_lag to reduce memory usage.

## Integration with Existing Models

The layer is designed as a pass-through, making it easy to add to existing architectures:

```python
# Original model
inputs = keras.Input(shape=(seq_len, features))
lstm = keras.layers.LSTM(64, return_sequences=True)(inputs)
predictions = keras.layers.Dense(1)(lstm)
model = keras.Model(inputs, predictions)

# Add ACF regularization
targets = keras.Input(shape=(seq_len, 1))
regularized_predictions = ResidualACFLayer(
    max_lag=20,
    regularization_weight=0.05
)([predictions, targets])
model_with_acf = keras.Model([inputs, targets], regularized_predictions)
```

## Theoretical Justification

The Box-Jenkins methodology emphasizes that residual analysis is crucial for model validation. The Ljung-Box test, commonly used for testing autocorrelation, essentially tests whether the sum of squared ACF values is significantly different from zero. This layer implements a differentiable version of this concept, allowing gradient-based optimization to directly minimize residual autocorrelation.

## Limitations and Considerations

1. **Computational Cost**: Computing ACF adds overhead, especially for long sequences and large batches.

2. **Trade-offs**: Minimizing ACF might sometimes conflict with minimizing prediction error. The regularization weight controls this trade-off.

3. **Non-stationarity**: The layer assumes stationarity within each sequence. For highly non-stationary data, consider detrending first.

4. **Interpretation**: Low ACF doesn't guarantee a good model—it's necessary but not sufficient for model adequacy.

## References

1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control.

2. Brockwell, P. J., & Davis, R. A. (2016). Introduction to time series and forecasting.

3. Harvey, A. C. (1990). The econometric analysis of time series.

## Conclusion

The `ResidualACFLayer` brings classical time series diagnostics into the deep learning era, enabling neural networks to not just minimize prediction error but also produce statistically well-behaved residuals. This is particularly important in domains where forecast reliability and uncertainty quantification matter as much as point accuracy.