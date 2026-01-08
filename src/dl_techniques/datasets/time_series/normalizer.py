"""
Time Series Normalization Module.

This module provides utilities for normalizing time series data and converting
back to original scale for evaluation. It includes advanced methods like
quantile transforms, robust scaling, and memory-efficient fitting for
large datasets.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

class NormalizationMethod(Enum):
    """
    Enumeration of available normalization methods.
    """
    DO_NOTHING = "do_nothing"
    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"
    UNIT_VECTOR = "unit_vector"
    MAX_ABS = "max_abs"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    MAD = "mad"
    POWER = "power"
    TANH = "tanh"
    DECIMAL = "decimal"
    PERCENT_CHANGE = "percent_change"


class TimeSeriesNormalizer:
    """
    Normalizer for time series data with proper scaling and inverse scaling.

    This class provides utilities for normalizing time series data and converting
    back to original scale. It supports memory optimizations (subsampling)
    for very large datasets during the fitting phase.

    :param method: Normalization method (string or enum).
    :type method: Union[str, NormalizationMethod]
    :param feature_range: Target range for minmax scaling.
    :type feature_range: Tuple[float, float]
    :param epsilon: Small value to avoid division by zero.
    :type epsilon: float
    :param n_quantiles: Number of quantiles for quantile-based methods.
    :type n_quantiles: int
    :param subsample: Max samples to use for fitting (memory optimization).
    :type subsample: int
    """

    def __init__(
            self,
            method: Union[str, NormalizationMethod] = NormalizationMethod.MINMAX,
            feature_range: Tuple[float, float] = (0.0, 1.0),
            epsilon: float = 1e-8,
            n_quantiles: int = 1000,
            subsample: int = 100000
    ) -> None:
        self.method = self._validate_method(method)
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.n_quantiles = n_quantiles
        self.subsample = subsample

        # Fitted parameters
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.mean_val: Optional[float] = None
        self.std_val: Optional[float] = None
        self.median_val: Optional[float] = None
        self.q25_val: Optional[float] = None
        self.q75_val: Optional[float] = None
        self.max_abs_val: Optional[float] = None
        self.mad_val: Optional[float] = None
        self.power_lambda: Optional[float] = None
        self.decimal_factor: Optional[float] = None
        self.first_val: Optional[float] = None
        self.quantiles: Optional[np.ndarray] = None
        self.fitted: bool = False

        logger.debug(f"TimeSeriesNormalizer initialized: {self.method.value}")

    def _validate_method(self, method: Union[str, NormalizationMethod]) -> NormalizationMethod:
        """Validate and convert method to NormalizationMethod enum."""
        if isinstance(method, str):
            try:
                return NormalizationMethod(method.lower())
            except ValueError:
                valid_methods = [m.value for m in NormalizationMethod]
                raise ValueError(
                    f"Unknown normalization method: '{method}'. "
                    f"Valid methods are: {valid_methods}"
                )
        elif isinstance(method, NormalizationMethod):
            return method
        else:
            raise ValueError("Method must be a string or NormalizationMethod enum.")

    @property
    def available_methods(self) -> List[str]:
        """List available normalization methods."""
        return [method.value for method in NormalizationMethod]

    @property
    def supports_perfect_inverse(self) -> bool:
        """Check if current method supports perfect reconstruction."""
        perfect_methods = {
            NormalizationMethod.DO_NOTHING,
            NormalizationMethod.MINMAX,
            NormalizationMethod.STANDARD,
            NormalizationMethod.ROBUST,
            NormalizationMethod.MAX_ABS,
            NormalizationMethod.MAD,
            NormalizationMethod.TANH,
            NormalizationMethod.DECIMAL,
            NormalizationMethod.PERCENT_CHANGE
        }
        return self.method in perfect_methods

    def fit(self, data: np.ndarray) -> 'TimeSeriesNormalizer':
        """
        Fit the normalizer to the data.

        Uses random subsampling for datasets larger than 1M elements
        to conserve memory and improve performance.

        :param data: Input data array.
        :type data: np.ndarray
        :return: Self for chaining.
        :rtype: TimeSeriesNormalizer
        :raises ValueError: If data is empty or all NaN.
        """
        if data.size == 0:
            raise ValueError("Cannot fit normalizer on empty data")

        # Memory optimization for large datasets
        LARGE_DATA_THRESHOLD = 1_000_000

        if data.size > LARGE_DATA_THRESHOLD:
            logger.info(
                f"Data size {data.size} exceeds threshold. "
                f"Subsampling {self.subsample} points for fitting."
            )
            # Use choice on indices to avoid copying the full array via flatten()
            flat_indices = np.random.choice(
                data.size,
                min(data.size, self.subsample),
                replace=False
            )
            valid_data = data.flat[flat_indices]
            valid_data = valid_data[~np.isnan(valid_data)]
        else:
            flat_data = data.flatten()
            valid_data = flat_data[~np.isnan(flat_data)]

        if len(valid_data) == 0:
            raise ValueError("No valid (non-NaN) data points found")

        # Compute statistics based on method
        if self.method == NormalizationMethod.MINMAX:
            self.min_val = float(np.min(valid_data))
            self.max_val = float(np.max(valid_data))

        elif self.method == NormalizationMethod.STANDARD:
            self.mean_val = float(np.mean(valid_data))
            self.std_val = float(np.std(valid_data))

        elif self.method == NormalizationMethod.ROBUST:
            self.median_val = float(np.median(valid_data))
            self.q25_val = float(np.percentile(valid_data, 25))
            self.q75_val = float(np.percentile(valid_data, 75))

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            self.max_val = float(np.linalg.norm(valid_data))

        elif self.method == NormalizationMethod.MAX_ABS:
            self.max_abs_val = float(np.max(np.abs(valid_data)))

        elif self.method in [NormalizationMethod.QUANTILE_UNIFORM, NormalizationMethod.QUANTILE_NORMAL]:
            self.quantiles = np.quantile(valid_data, np.linspace(0, 1, self.n_quantiles))

        elif self.method == NormalizationMethod.MAD:
            self.median_val = float(np.median(valid_data))
            self.mad_val = float(np.median(np.abs(valid_data - self.median_val)))

        elif self.method == NormalizationMethod.POWER:
            self.power_lambda = 0.0  # Simple log-like approximation
            self.mean_val = float(np.mean(valid_data))
            self.std_val = float(np.std(valid_data))

        elif self.method == NormalizationMethod.TANH:
            self.mean_val = float(np.mean(valid_data))
            self.std_val = float(np.std(valid_data))

        elif self.method == NormalizationMethod.DECIMAL:
            max_abs = np.max(np.abs(valid_data))
            self.decimal_factor = 10 ** np.ceil(np.log10(max_abs)) if max_abs > 0 else 1.0

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            self.first_val = float(valid_data[0]) if len(valid_data) > 0 else 0.0

        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.

        :param data: Input data array.
        :type data: np.ndarray
        :return: Normalized data.
        :rtype: np.ndarray
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.method == NormalizationMethod.DO_NOTHING:
            return data.copy()

        elif self.method == NormalizationMethod.MINMAX:
            data_range = self.max_val - self.min_val
            if data_range == 0:
                return np.full_like(data, self.feature_range[0])
            scale = (data - self.min_val) / (data_range + self.epsilon)
            target_range = self.feature_range[1] - self.feature_range[0]
            return scale * target_range + self.feature_range[0]

        elif self.method == NormalizationMethod.STANDARD:
            if self.std_val == 0:
                return data - self.mean_val
            return (data - self.mean_val) / (self.std_val + self.epsilon)

        elif self.method == NormalizationMethod.ROBUST:
            iqr = self.q75_val - self.q25_val
            if iqr == 0:
                return data - self.median_val
            return (data - self.median_val) / (iqr + self.epsilon)

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            norm = np.linalg.norm(data)
            return data / (norm + self.epsilon)

        elif self.method == NormalizationMethod.MAX_ABS:
            return data / (self.max_abs_val + self.epsilon)

        elif self.method == NormalizationMethod.QUANTILE_UNIFORM:
            return self.transform_quantile_uniform(data)

        elif self.method == NormalizationMethod.QUANTILE_NORMAL:
            uniform_data = self.transform_quantile_uniform(data)
            return self._norm_ppf(uniform_data)

        elif self.method == NormalizationMethod.MAD:
            return (data - self.median_val) / (self.mad_val + self.epsilon)

        elif self.method == NormalizationMethod.POWER:
            transformed = np.sign(data) * np.log(np.abs(data) + 1)
            mean_t = np.nanmean(transformed)
            std_t = np.nanstd(transformed)
            return (transformed - mean_t) / (std_t + self.epsilon)

        elif self.method == NormalizationMethod.TANH:
            standardized = (data - self.mean_val) / (self.std_val + self.epsilon)
            return np.tanh(standardized)

        elif self.method == NormalizationMethod.DECIMAL:
            return data / self.decimal_factor

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            return (data - self.first_val) / (self.first_val + self.epsilon)

        return data

    def transform_quantile_uniform(self, data: np.ndarray) -> np.ndarray:
        """Helper for quantile transform."""
        result = np.zeros_like(data)
        flat_data = data.flatten()
        for i, val in enumerate(flat_data):
            if np.isnan(val):
                result.flat[i] = val
            else:
                idx = np.searchsorted(self.quantiles, val)
                result.flat[i] = idx / (len(self.quantiles) - 1)
        return result

    def _norm_ppf(self, data: np.ndarray) -> np.ndarray:
        """Inverse normal CDF approximation."""
        data = np.clip(data, self.epsilon, 1 - self.epsilon)
        result = np.zeros_like(data)
        for i, p in enumerate(data.flat):
            if np.isnan(p):
                result.flat[i] = p
            else:
                if p < 0.5:
                    t = np.sqrt(-2 * np.log(p))
                    result.flat[i] = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                                       (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
                else:
                    t = np.sqrt(-2 * np.log(1 - p))
                    result.flat[i] = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                                     (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        :param data: Normalized data array.
        :type data: np.ndarray
        :return: Data in original scale.
        :rtype: np.ndarray
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        if self.method == NormalizationMethod.DO_NOTHING:
            return data.copy()

        elif self.method == NormalizationMethod.MINMAX:
            target_range = self.feature_range[1] - self.feature_range[0]
            unscaled = (data - self.feature_range[0]) / (target_range + self.epsilon)
            data_range = self.max_val - self.min_val
            return unscaled * data_range + self.min_val

        elif self.method == NormalizationMethod.STANDARD:
            return data * (self.std_val + self.epsilon) + self.mean_val

        elif self.method == NormalizationMethod.ROBUST:
            iqr = self.q75_val - self.q25_val
            return data * (iqr + self.epsilon) + self.median_val

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            logger.warning(
                "Inverse transform for UNIT_VECTOR is mathematically impossible "
                "(magnitude lost). Returning scaled data."
            )
            return data.copy()

        elif self.method == NormalizationMethod.MAX_ABS:
            return data * (self.max_abs_val + self.epsilon)

        elif self.method == NormalizationMethod.QUANTILE_UNIFORM:
            return self.inverse_transform_quantile_uniform(data)

        elif self.method == NormalizationMethod.QUANTILE_NORMAL:
            uniform_data = self._norm_cdf(data)
            return self.inverse_transform_quantile_uniform(uniform_data)

        elif self.method == NormalizationMethod.MAD:
            return data * (self.mad_val + self.epsilon) + self.median_val

        elif self.method == NormalizationMethod.POWER:
            logger.warning("Power transformation inverse is approximate.")
            unstandardized = data * (self.std_val + self.epsilon) + self.mean_val
            return np.sign(unstandardized) * (np.exp(np.abs(unstandardized)) - 1)

        elif self.method == NormalizationMethod.TANH:
            inverse_tanh = np.arctanh(np.clip(data, -0.99, 0.99))
            return inverse_tanh * (self.std_val + self.epsilon) + self.mean_val

        elif self.method == NormalizationMethod.DECIMAL:
            return data * self.decimal_factor

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            return data * (self.first_val + self.epsilon) + self.first_val

        return data

    def inverse_transform_quantile_uniform(self, data: np.ndarray) -> np.ndarray:
        """Helper for inverse quantile transform."""
        result = np.zeros_like(data)
        flat_data = data.flatten()
        n_q = len(self.quantiles)
        for i, val in enumerate(flat_data):
            if np.isnan(val):
                result.flat[i] = val
            else:
                val = np.clip(val, 0, 1)
                idx = int(val * (n_q - 1))
                idx = np.clip(idx, 0, n_q - 1)
                result.flat[i] = self.quantiles[idx]
        return result

    def _norm_cdf(self, data: np.ndarray) -> np.ndarray:
        """Normal CDF approximation."""
        result = np.zeros_like(data)
        for i, x in enumerate(data.flat):
            if np.isnan(x):
                result.flat[i] = x
            else:
                t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
                d = 0.3989423 * np.exp(-x * x / 2)
                p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 +
                                                                t * (-1.821256 + t * 1.330274))))
                if x >= 0:
                    result.flat[i] = 1.0 - p
                else:
                    result.flat[i] = p
        return result

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(data).transform(data)

    def get_statistics(self) -> Dict[str, Optional[float]]:
        """Get the fitted statistics."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted to get statistics")

        stats = {'method': self.method.value, 'fitted': self.fitted}

        if self.method == NormalizationMethod.MINMAX:
            stats.update({'min_val': self.min_val, 'max_val': self.max_val})
        elif self.method == NormalizationMethod.STANDARD:
            stats.update({'mean_val': self.mean_val, 'std_val': self.std_val})
        elif self.method == NormalizationMethod.ROBUST:
            stats.update({
                'median_val': self.median_val,
                'q25_val': self.q25_val,
                'q75_val': self.q75_val
            })

        return stats

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        perfect_inv = "✓" if self.supports_perfect_inverse else "⚠"
        return (f"TimeSeriesNormalizer(method={self.method.value}, "
                f"feature_range={self.feature_range}, {status}, "
                f"perfect_inverse={perfect_inv})")