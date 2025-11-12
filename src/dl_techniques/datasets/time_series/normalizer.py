import numpy as np
from enum import Enum
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


class NormalizationMethod(Enum):
    """Enumeration of available normalization methods.

    Attributes:
        DO_NOTHING: No normalization applied, data passes through unchanged.
        MINMAX: Min-max normalization to specified feature range.
        STANDARD: Z-score standardization (zero mean, unit variance).
        ROBUST: Robust scaling using median and IQR.
        UNIT_VECTOR: L2 normalization - scales to unit vector (norm=1).
        MAX_ABS: Maximum absolute scaling - divides by maximum absolute value.
        QUANTILE_UNIFORM: Quantile transformation to uniform distribution [0,1].
        QUANTILE_NORMAL: Quantile transformation to standard normal distribution.
        MAD: Median Absolute Deviation normalization (robust alternative to standard).
        POWER: Power transformation (Yeo-Johnson) to make data more Gaussian.
        TANH: Hyperbolic tangent scaling for bounded output [-1,1].
        DECIMAL: Decimal scaling - divides by power of 10.
        PERCENT_CHANGE: Converts to percentage change from first value.
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

# ---------------------------------------------------------------------

class TimeSeriesNormalizer:
    """Normalizer for time series data with proper scaling and inverse scaling.

    This class provides utilities for normalizing time series data and converting
    back to original scale for evaluation. Supports multiple normalization methods
    including min-max scaling, standardization, robust scaling, and specialized
    transformations for different data characteristics.

    Args:
        method: Normalization method. Can be a string or NormalizationMethod enum.
                Available methods:
                - 'do_nothing': Pass-through (no transformation)
                - 'minmax': Min-max scaling to specified range
                - 'standard': Z-score standardization (zero mean, unit variance)
                - 'robust': Robust scaling using median and IQR
                - 'unit_vector': L2 normalization (unit norm)
                - 'max_abs': Maximum absolute scaling
                - 'quantile_uniform': Quantile transformation to uniform [0,1]
                - 'quantile_normal': Quantile transformation to standard normal
                - 'mad': Median Absolute Deviation normalization
                - 'power': Power transformation (Yeo-Johnson)
                - 'tanh': Hyperbolic tangent scaling [-1,1]
                - 'decimal': Decimal scaling
                - 'percent_change': Percentage change from first value
        feature_range: Target range for minmax scaling. Only used when method is 'minmax'.
        epsilon: Small value to avoid division by zero during normalization.
        n_quantiles: Number of quantiles for quantile-based methods.
        subsample: Maximum number of samples to use for fitting quantile methods.

    Raises:
        ValueError: If an unknown normalization method is provided.

    Example:
        >>> normalizer = TimeSeriesNormalizer(method='robust')
        >>> data = np.array([1, 2, 3, 4, 5, 100])  # Contains outlier
        >>> normalized = normalizer.fit_transform(data)
        >>> original = normalizer.inverse_transform(normalized)

    Note:
        Some methods like 'unit_vector' and 'quantile_*' may not perfectly
        reconstruct original data through inverse_transform due to information loss.
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
        self.n_quantiles = n_quantiles  # For quantile methods
        self.subsample = subsample  # For quantile methods on large datasets

        # Fitted parameters - will be set during fit()
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

        logger.info(f"TimeSeriesNormalizer initialized with method: {self.method.value}")

    def _validate_method(self, method: Union[str, NormalizationMethod]) -> NormalizationMethod:
        """Validate and convert method to NormalizationMethod enum.

        Args:
            method: Method to validate. Can be string or enum.

        Returns:
            Validated NormalizationMethod enum.

        Raises:
            ValueError: If method is not recognized.
        """
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
            valid_methods = [m.value for m in NormalizationMethod]
            raise ValueError(
                f"Method must be a string or NormalizationMethod enum. "
                f"Valid methods are: {valid_methods}"
            )

    @property
    def available_methods(self) -> list[str]:
        """Get list of available normalization methods.

        Returns:
            List of method names as strings.
        """
        return [method.value for method in NormalizationMethod]

    @property
    def supports_perfect_inverse(self) -> bool:
        """Check if current method supports perfect inverse transformation.

        Returns:
            True if inverse_transform can perfectly reconstruct original data.
        """
        perfect_inverse_methods = {
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
        return self.method in perfect_inverse_methods

    @classmethod
    def get_method_info(cls) -> dict[str, dict]:
        """Get detailed information about all available methods.

        Returns:
            Dictionary with method names as keys and info dictionaries as values.
        """
        return {
            'do_nothing': {
                'description': 'Pass-through transformation (no change)',
                'output_range': 'Same as input',
                'perfect_inverse': True,
                'robust_to_outliers': True,
                'use_case': 'When no normalization is needed'
            },
            'minmax': {
                'description': 'Linear scaling to specified range',
                'output_range': 'User-defined (default: [0,1])',
                'perfect_inverse': True,
                'robust_to_outliers': False,
                'use_case': 'When you need bounded output in specific range'
            },
            'standard': {
                'description': 'Z-score standardization (zero mean, unit variance)',
                'output_range': 'Unbounded (typically [-3,3])',
                'perfect_inverse': True,
                'robust_to_outliers': False,
                'use_case': 'When data should be normally distributed'
            },
            'robust': {
                'description': 'Scaling using median and IQR',
                'output_range': 'Unbounded',
                'perfect_inverse': True,
                'robust_to_outliers': True,
                'use_case': 'When data contains outliers'
            },
            'unit_vector': {
                'description': 'L2 normalization to unit vector',
                'output_range': 'Same shape, norm=1',
                'perfect_inverse': False,
                'robust_to_outliers': True,
                'use_case': 'When only direction matters, not magnitude'
            },
            'max_abs': {
                'description': 'Scaling by maximum absolute value',
                'output_range': '[-1, 1]',
                'perfect_inverse': True,
                'robust_to_outliers': False,
                'use_case': 'When you want symmetric bounds around zero'
            },
            'quantile_uniform': {
                'description': 'Transform to uniform distribution [0,1]',
                'output_range': '[0, 1]',
                'perfect_inverse': False,
                'robust_to_outliers': True,
                'use_case': 'When you want uniform distribution'
            },
            'quantile_normal': {
                'description': 'Transform to standard normal distribution',
                'output_range': 'Unbounded (normally distributed)',
                'perfect_inverse': False,
                'robust_to_outliers': True,
                'use_case': 'When you want normal distribution from any distribution'
            },
            'mad': {
                'description': 'Median Absolute Deviation normalization',
                'output_range': 'Unbounded',
                'perfect_inverse': True,
                'robust_to_outliers': True,
                'use_case': 'Robust alternative to standard normalization'
            },
            'power': {
                'description': 'Power transformation to make data more normal',
                'output_range': 'Unbounded',
                'perfect_inverse': False,
                'robust_to_outliers': False,
                'use_case': 'When data is highly skewed'
            },
            'tanh': {
                'description': 'Hyperbolic tangent scaling',
                'output_range': '(-1, 1)',
                'perfect_inverse': True,
                'robust_to_outliers': False,
                'use_case': 'When you want smooth bounded output'
            },
            'decimal': {
                'description': 'Scaling by appropriate power of 10',
                'output_range': 'Scaled to manageable range',
                'perfect_inverse': True,
                'robust_to_outliers': True,
                'use_case': 'When you want to reduce magnitude without changing distribution'
            },
            'percent_change': {
                'description': 'Convert to percentage change from first value',
                'output_range': 'Unbounded',
                'perfect_inverse': True,
                'robust_to_outliers': False,
                'use_case': 'For time series relative change analysis'
            }
        }

    def fit(self, data: np.ndarray) -> 'TimeSeriesNormalizer':
        """Fit the normalizer to the data.

        Computes and stores the necessary statistics for the chosen normalization
        method. For 'do_nothing' method, no statistics are computed.

        Args:
            data: Input data array. Can be of any shape. Statistics
                  are computed over the entire array.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If data is empty or contains only NaN values.
        """
        if data.size == 0:
            raise ValueError("Cannot fit normalizer on empty data")

        if np.all(np.isnan(data)):
            raise ValueError("Cannot fit normalizer on data containing only NaN values")

        flat_data = data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]

        if len(valid_data) == 0:
            raise ValueError("No valid (non-NaN) data points found")

        if self.method == NormalizationMethod.DO_NOTHING:
            logger.debug("Using do_nothing method - no parameters to fit")

        elif self.method == NormalizationMethod.MINMAX:
            self.min_val = float(np.nanmin(data))
            self.max_val = float(np.nanmax(data))
            logger.debug(f"Fitted minmax: min={self.min_val:.6f}, max={self.max_val:.6f}")

        elif self.method == NormalizationMethod.STANDARD:
            self.mean_val = float(np.nanmean(data))
            self.std_val = float(np.nanstd(data))
            logger.debug(f"Fitted standard: mean={self.mean_val:.6f}, std={self.std_val:.6f}")

        elif self.method == NormalizationMethod.ROBUST:
            self.median_val = float(np.nanmedian(data))
            self.q25_val = float(np.nanpercentile(data, 25))
            self.q75_val = float(np.nanpercentile(data, 75))
            logger.debug(f"Fitted robust: median={self.median_val:.6f}, "
                         f"q25={self.q25_val:.6f}, q75={self.q75_val:.6f}")

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            # For unit vector, we only need to check that data is not all zeros
            self.max_val = float(np.linalg.norm(valid_data))
            logger.debug(f"Fitted unit_vector: norm={self.max_val:.6f}")

        elif self.method == NormalizationMethod.MAX_ABS:
            self.max_abs_val = float(np.nanmax(np.abs(data)))
            logger.debug(f"Fitted max_abs: max_abs={self.max_abs_val:.6f}")

        elif self.method == NormalizationMethod.QUANTILE_UNIFORM:
            # Subsample if data is too large
            if len(valid_data) > self.subsample:
                indices = np.random.choice(len(valid_data), self.subsample, replace=False)
                sample_data = valid_data[indices]
            else:
                sample_data = valid_data

            self.quantiles = np.quantile(sample_data, np.linspace(0, 1, self.n_quantiles))
            logger.debug(f"Fitted quantile_uniform with {len(self.quantiles)} quantiles")

        elif self.method == NormalizationMethod.QUANTILE_NORMAL:
            # Same as quantile_uniform but we'll transform to normal in transform()
            if len(valid_data) > self.subsample:
                indices = np.random.choice(len(valid_data), self.subsample, replace=False)
                sample_data = valid_data[indices]
            else:
                sample_data = valid_data

            self.quantiles = np.quantile(sample_data, np.linspace(0, 1, self.n_quantiles))
            logger.debug(f"Fitted quantile_normal with {len(self.quantiles)} quantiles")

        elif self.method == NormalizationMethod.MAD:
            self.median_val = float(np.nanmedian(data))
            self.mad_val = float(np.nanmedian(np.abs(data - self.median_val)))
            logger.debug(f"Fitted MAD: median={self.median_val:.6f}, mad={self.mad_val:.6f}")

        elif self.method == NormalizationMethod.POWER:
            # Simple Yeo-Johnson estimation (lambda=0 for log-like transform)
            self.power_lambda = 0.0  # Could be optimized but 0 is often good
            self.mean_val = float(np.nanmean(data))
            self.std_val = float(np.nanstd(data))
            logger.debug(f"Fitted power: lambda={self.power_lambda}")

        elif self.method == NormalizationMethod.TANH:
            self.mean_val = float(np.nanmean(data))
            self.std_val = float(np.nanstd(data))
            logger.debug(f"Fitted tanh: mean={self.mean_val:.6f}, std={self.std_val:.6f}")

        elif self.method == NormalizationMethod.DECIMAL:
            max_abs = np.nanmax(np.abs(data))
            self.decimal_factor = 10 ** np.ceil(np.log10(max_abs)) if max_abs > 0 else 1.0
            logger.debug(f"Fitted decimal: factor={self.decimal_factor}")

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            # Use first valid value as baseline
            self.first_val = float(valid_data[0]) if len(valid_data) > 0 else 0.0
            logger.debug(f"Fitted percent_change: first_val={self.first_val:.6f}")

        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters.

        Args:
            data: Input data array to transform.

        Returns:
            Normalized data array with the same shape as input.

        Raises:
            ValueError: If normalizer has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.method == NormalizationMethod.DO_NOTHING:
            return data.copy()

        elif self.method == NormalizationMethod.MINMAX:
            data_range = self.max_val - self.min_val
            if data_range == 0:
                logger.warning("Data range is zero, returning feature_range minimum")
                return np.full_like(data, self.feature_range[0])

            scale = (data - self.min_val) / (data_range + self.epsilon)
            target_range = self.feature_range[1] - self.feature_range[0]
            return scale * target_range + self.feature_range[0]

        elif self.method == NormalizationMethod.STANDARD:
            if self.std_val == 0:
                logger.warning("Standard deviation is zero, returning zero-mean data")
                return data - self.mean_val

            return (data - self.mean_val) / (self.std_val + self.epsilon)

        elif self.method == NormalizationMethod.ROBUST:
            iqr = self.q75_val - self.q25_val
            if iqr == 0:
                logger.warning("IQR is zero, returning median-centered data")
                return data - self.median_val

            return (data - self.median_val) / (iqr + self.epsilon)

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            norm = np.linalg.norm(data)
            if norm == 0:
                logger.warning("Vector norm is zero, returning original data")
                return data.copy()
            return data / (norm + self.epsilon)

        elif self.method == NormalizationMethod.MAX_ABS:
            if self.max_abs_val == 0:
                logger.warning("Max absolute value is zero, returning original data")
                return data.copy()
            return data / (self.max_abs_val + self.epsilon)

        elif self.method == NormalizationMethod.QUANTILE_UNIFORM:
            # Map data to quantiles then to uniform [0,1]
            result = np.zeros_like(data)
            flat_data = data.flatten()

            for i, val in enumerate(flat_data):
                if np.isnan(val):
                    result.flat[i] = val
                else:
                    # Find position in quantiles
                    idx = np.searchsorted(self.quantiles, val)
                    result.flat[i] = idx / (len(self.quantiles) - 1)

            return result

        elif self.method == NormalizationMethod.QUANTILE_NORMAL:
            # Map to uniform then to standard normal
            uniform_data = self.transform_quantile_uniform(data)
            # Use inverse normal CDF approximation
            return self._norm_ppf(uniform_data)

        elif self.method == NormalizationMethod.MAD:
            if self.mad_val == 0:
                logger.warning("MAD is zero, returning median-centered data")
                return data - self.median_val
            return (data - self.median_val) / (self.mad_val + self.epsilon)

        elif self.method == NormalizationMethod.POWER:
            # Simple Yeo-Johnson with lambda=0 (log-like)
            transformed = np.sign(data) * np.log(np.abs(data) + 1)
            # Standardize the result
            mean_t = np.nanmean(transformed)
            std_t = np.nanstd(transformed)
            return (transformed - mean_t) / (std_t + self.epsilon)

        elif self.method == NormalizationMethod.TANH:
            # Standardize then apply tanh
            standardized = (data - self.mean_val) / (self.std_val + self.epsilon)
            return np.tanh(standardized)

        elif self.method == NormalizationMethod.DECIMAL:
            return data / self.decimal_factor

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            if self.first_val == 0:
                logger.warning("First value is zero, cannot compute percent change")
                return np.zeros_like(data)
            return (data - self.first_val) / (self.first_val + self.epsilon)

    def transform_quantile_uniform(self, data: np.ndarray) -> np.ndarray:
        """Helper method for quantile uniform transformation."""
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
        """Approximation of inverse normal CDF (percent point function)."""
        # Clip to avoid infinite values
        data = np.clip(data, self.epsilon, 1 - self.epsilon)

        # Beasley-Springer-Moro approximation
        # Simple approximation for inverse normal CDF
        result = np.zeros_like(data)

        for i, p in enumerate(data.flat):
            if np.isnan(p):
                result.flat[i] = p
            else:
                if p < 0.5:
                    # Use symmetry
                    t = np.sqrt(-2 * np.log(p))
                    result.flat[i] = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                                       (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
                else:
                    t = np.sqrt(-2 * np.log(1 - p))
                    result.flat[i] = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                                     (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)

        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data array to inverse transform.

        Returns:
            Data in original scale with the same shape as input.

        Raises:
            ValueError: If normalizer has not been fitted.
            NotImplementedError: For methods that don't support inverse transform.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        if self.method == NormalizationMethod.DO_NOTHING:
            return data.copy()

        elif self.method == NormalizationMethod.MINMAX:
            target_range = self.feature_range[1] - self.feature_range[0]
            if target_range == 0:
                logger.warning("Target range is zero, returning min_val")
                return np.full_like(data, self.min_val)

            unscaled = (data - self.feature_range[0]) / (target_range + self.epsilon)
            data_range = self.max_val - self.min_val
            return unscaled * data_range + self.min_val

        elif self.method == NormalizationMethod.STANDARD:
            return data * (self.std_val + self.epsilon) + self.mean_val

        elif self.method == NormalizationMethod.ROBUST:
            iqr = self.q75_val - self.q25_val
            return data * (iqr + self.epsilon) + self.median_val

        elif self.method == NormalizationMethod.UNIT_VECTOR:
            # Cannot reliably inverse transform unit vector normalization
            # because original magnitude is lost
            logger.warning("Unit vector normalization loses magnitude information - "
                           "inverse transform returns normalized data")
            return data.copy()

        elif self.method == NormalizationMethod.MAX_ABS:
            return data * (self.max_abs_val + self.epsilon)

        elif self.method == NormalizationMethod.QUANTILE_UNIFORM:
            # Map from uniform [0,1] back to original distribution
            result = np.zeros_like(data)
            flat_data = data.flatten()

            for i, val in enumerate(flat_data):
                if np.isnan(val):
                    result.flat[i] = val
                else:
                    # Clip to valid range
                    val = np.clip(val, 0, 1)
                    # Find corresponding quantile
                    idx = int(val * (len(self.quantiles) - 1))
                    idx = np.clip(idx, 0, len(self.quantiles) - 1)
                    result.flat[i] = self.quantiles[idx]

            return result

        elif self.method == NormalizationMethod.QUANTILE_NORMAL:
            # Convert from standard normal back to uniform, then to original
            uniform_data = self._norm_cdf(data)
            return self.inverse_transform_quantile_uniform(uniform_data)

        elif self.method == NormalizationMethod.MAD:
            return data * (self.mad_val + self.epsilon) + self.median_val

        elif self.method == NormalizationMethod.POWER:
            # Inverse of power transformation is complex and approximate
            logger.warning("Power transformation inverse is approximate")
            # Reverse standardization first
            unstandardized = data * (self.std_val + self.epsilon) + self.mean_val
            # Approximate inverse of log-like transform
            return np.sign(unstandardized) * (np.exp(np.abs(unstandardized)) - 1)

        elif self.method == NormalizationMethod.TANH:
            # Inverse tanh then reverse standardization
            inverse_tanh = np.arctanh(np.clip(data, -0.99, 0.99))
            return inverse_tanh * (self.std_val + self.epsilon) + self.mean_val

        elif self.method == NormalizationMethod.DECIMAL:
            return data * self.decimal_factor

        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            return data * (self.first_val + self.epsilon) + self.first_val

    def inverse_transform_quantile_uniform(self, data: np.ndarray) -> np.ndarray:
        """Helper method for inverse quantile uniform transformation."""
        result = np.zeros_like(data)
        flat_data = data.flatten()

        for i, val in enumerate(flat_data):
            if np.isnan(val):
                result.flat[i] = val
            else:
                val = np.clip(val, 0, 1)
                idx = int(val * (len(self.quantiles) - 1))
                idx = np.clip(idx, 0, len(self.quantiles) - 1)
                result.flat[i] = self.quantiles[idx]

        return result

    def _norm_cdf(self, data: np.ndarray) -> np.ndarray:
        """Approximation of normal CDF."""
        # Approximation of standard normal CDF
        result = np.zeros_like(data)

        for i, x in enumerate(data.flat):
            if np.isnan(x):
                result.flat[i] = x
            else:
                # Abramowitz and Stegun approximation
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
        """Fit and transform data in one step.

        Convenience method that combines fit() and transform().

        Args:
            data: Input data array to fit and transform.

        Returns:
            Normalized data array.
        """
        return self.fit(data).transform(data)

    def get_statistics(self) -> dict[str, Optional[float]]:
        """Get the fitted statistics for the normalizer.

        Returns:
            Dictionary containing the fitted statistics based on method.

        Raises:
            ValueError: If normalizer has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted to get statistics")

        stats = {
            'method': self.method.value,
            'fitted': self.fitted
        }

        if self.method == NormalizationMethod.MINMAX:
            stats.update({
                'min_val': self.min_val,
                'max_val': self.max_val,
                'feature_range': self.feature_range
            })
        elif self.method == NormalizationMethod.STANDARD:
            stats.update({
                'mean_val': self.mean_val,
                'std_val': self.std_val
            })
        elif self.method == NormalizationMethod.ROBUST:
            stats.update({
                'median_val': self.median_val,
                'q25_val': self.q25_val,
                'q75_val': self.q75_val
            })
        elif self.method == NormalizationMethod.UNIT_VECTOR:
            stats.update({
                'original_norm': self.max_val
            })
        elif self.method == NormalizationMethod.MAX_ABS:
            stats.update({
                'max_abs_val': self.max_abs_val
            })
        elif self.method in [NormalizationMethod.QUANTILE_UNIFORM, NormalizationMethod.QUANTILE_NORMAL]:
            stats.update({
                'n_quantiles': len(self.quantiles) if self.quantiles is not None else 0,
                'quantile_range': [float(self.quantiles[0]),
                                   float(self.quantiles[-1])] if self.quantiles is not None else None
            })
        elif self.method == NormalizationMethod.MAD:
            stats.update({
                'median_val': self.median_val,
                'mad_val': self.mad_val
            })
        elif self.method == NormalizationMethod.POWER:
            stats.update({
                'power_lambda': self.power_lambda,
                'mean_val': self.mean_val,
                'std_val': self.std_val
            })
        elif self.method == NormalizationMethod.TANH:
            stats.update({
                'mean_val': self.mean_val,
                'std_val': self.std_val
            })
        elif self.method == NormalizationMethod.DECIMAL:
            stats.update({
                'decimal_factor': self.decimal_factor
            })
        elif self.method == NormalizationMethod.PERCENT_CHANGE:
            stats.update({
                'first_val': self.first_val
            })

        return stats

    def __repr__(self) -> str:
        """String representation of the normalizer."""
        status = "fitted" if self.fitted else "not fitted"
        perfect_inv = "✓" if self.supports_perfect_inverse else "⚠"
        return (f"TimeSeriesNormalizer(method={self.method.value}, "
                f"feature_range={self.feature_range}, {status}, "
                f"perfect_inverse={perfect_inv})")

    def summary(self) -> str:
        """Get a detailed summary of the normalizer state.

        Returns:
            Multi-line string with normalizer information.
        """
        lines = [
            f"TimeSeriesNormalizer Summary",
            f"=" * 30,
            f"Method: {self.method.value}",
            f"Status: {'✓ Fitted' if self.fitted else '✗ Not fitted'}",
            f"Perfect inverse: {'✓ Yes' if self.supports_perfect_inverse else '⚠ Approximate'}",
            f"Feature range: {self.feature_range}",
            f"Epsilon: {self.epsilon}"
        ]

        if self.fitted:
            lines.append(f"\nFitted Parameters:")
            stats = self.get_statistics()
            for key, value in stats.items():
                if key not in ['method', 'fitted']:
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.6f}")
                    else:
                        lines.append(f"  {key}: {value}")

        # Add method description
        method_info = self.get_method_info().get(self.method.value, {})
        if method_info:
            lines.extend([
                f"\nMethod Info:",
                f"  Description: {method_info.get('description', 'N/A')}",
                f"  Output range: {method_info.get('output_range', 'N/A')}",
                f"  Robust to outliers: {method_info.get('robust_to_outliers', 'N/A')}",
                f"  Use case: {method_info.get('use_case', 'N/A')}"
            ])

        return "\n".join(lines)

# ---------------------------------------------------------------------
