"""
N-BEATS Data Loader and Preprocessing Utilities.

This module provides data loading and preprocessing utilities for N-BEATS models,
including synthetic data generation, real dataset loading, and proper normalization.
"""


import os
import csv
import numpy as np
from typing import Optional, Tuple, Generator, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class TimeSeriesNormalizer:
    """Normalizer for time series data with proper scaling and inverse scaling.

    This class provides utilities for normalizing time series data and converting
    back to original scale for evaluation.

    Args:
        method: String, normalization method ('minmax', 'standard', 'robust').
        feature_range: Tuple, target range for minmax scaling.
        epsilon: Float, small value to avoid division by zero.
    """

    def __init__(
            self,
            method: str = 'minmax',
            feature_range: Tuple[float, float] = (0.0, 1.0),
            epsilon: float = 1e-8
    ) -> None:
        self.method = method
        self.feature_range = feature_range
        self.epsilon = epsilon

        # Fitted parameters
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.median_val = None
        self.q25_val = None
        self.q75_val = None
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'TimeSeriesNormalizer':
        """Fit the normalizer to the data.

        Args:
            data: Input data array. Can be of any shape. Statistics
                  are computed over the entire array.

        Returns:
            Self for method chaining.
        """
        if self.method == 'minmax':
            self.min_val = np.min(data)
            self.max_val = np.max(data)

        elif self.method == 'standard':
            self.mean_val = np.mean(data)
            self.std_val = np.std(data)

        elif self.method == 'robust':
            self.median_val = np.median(data)
            self.q25_val = np.percentile(data, 25)
            self.q75_val = np.percentile(data, 75)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters.

        Args:
            data: Input data array.

        Returns:
            Normalized data array.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.method == 'minmax':
            data_range = self.max_val - self.min_val
            scale = (data - self.min_val) / (data_range + self.epsilon)
            target_range = self.feature_range[1] - self.feature_range[0]
            return scale * target_range + self.feature_range[0]

        elif self.method == 'standard':
            return (data - self.mean_val) / (self.std_val + self.epsilon)

        elif self.method == 'robust':
            iqr = self.q75_val - self.q25_val
            return (data - self.median_val) / (iqr + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data array.

        Returns:
            Data in original scale.
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        if self.method == 'minmax':
            target_range = self.feature_range[1] - self.feature_range[0]
            unscaled = (data - self.feature_range[0]) / (target_range + self.epsilon)
            data_range = self.max_val - self.min_val
            return unscaled * data_range + self.min_val

        elif self.method == 'standard':
            return data * self.std_val + self.mean_val

        elif self.method == 'robust':
            iqr = self.q75_val - self.q25_val
            return data * iqr + self.median_val

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step.

        Args:
            data: Input data array.

        Returns:
            Normalized data array.
        """
        return self.fit(data).transform(data)

# ---------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generator for synthetic time series data for N-BEATS training and testing.
    ...
    """

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            batch_size: int = 32,
            signal_type: str = 'mixed',
            add_noise: bool = True,
            noise_std: float = 0.1,
            random_seed: Optional[int] = None
    ) -> None:
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.signal_type = signal_type
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = np.random.RandomState(random_seed)

        logger.info(f"Initialized synthetic data generator for {signal_type} signals")

    def _generate_trend_signal(self, length: int) -> np.ndarray:
        """Generate a trend signal."""
        time = np.arange(length) / length
        slope = self.random_state.uniform(-0.1, 0.1)
        offset = self.random_state.uniform(-1.0, 1.0)
        return slope * time + offset

    def _generate_seasonality_signal(self, length: int) -> np.ndarray:
        """Generate a seasonality signal."""
        time = np.arange(length)
        signal = np.zeros(length)
        for _ in range(2):  # Sum of two seasonal components
            amplitude = self.random_state.uniform(0.2, 1.0)
            period = self.random_state.choice([length / 4, length / 2, length])
            phase = self.random_state.uniform(0, 2 * np.pi)
            signal += amplitude * np.cos(2 * np.pi * time / period + phase)
        return signal

    def _generate_mixed_signal(self, length: int) -> np.ndarray:
        """Generate a mixed trend and seasonality signal."""
        trend = self._generate_trend_signal(length)
        seasonality = self._generate_seasonality_signal(length)
        trend_weight = self.random_state.uniform(0.3, 0.7)
        return trend_weight * trend + (1 - trend_weight) * seasonality

    def _generate_autoregressive_signal(self, length: int) -> np.ndarray:
        """Generate an autoregressive signal."""
        phi1 = self.random_state.uniform(0.3, 0.7)
        phi2 = self.random_state.uniform(-0.3, 0.2)
        if phi1 + phi2 >= 1 or phi2 - phi1 >= 1 or abs(phi1) >= 1: # Stationarity
            phi1, phi2 = 0.5, 0.2

        signal = np.zeros(length)
        signal[0] = self.random_state.normal(0, 1)
        signal[1] = phi1 * signal[0] + self.random_state.normal(0, 1)
        for i in range(2, length):
            noise = self.random_state.normal(0, 0.1)
            signal[i] = phi1 * signal[i - 1] + phi2 * signal[i - 2] + noise
        return signal

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1] range for this specific instance."""
        min_val, max_val = np.min(signal), np.max(signal)
        range_val = max_val - min_val
        return (signal - min_val) / (range_val + 1e-8)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of synthetic time series data."""
        X = np.zeros((self.batch_size, self.backcast_length, 1))
        y = np.zeros((self.batch_size, self.forecast_length, 1))
        total_length = self.backcast_length + self.forecast_length

        for i in range(self.batch_size):
            signal_generators = {
                'trend': self._generate_trend_signal,
                'seasonality': self._generate_seasonality_signal,
                'mixed': self._generate_mixed_signal,
                'autoregressive': self._generate_autoregressive_signal
            }
            generator_func = signal_generators.get(self.signal_type)
            if generator_func is None:
                raise ValueError(f"Unknown signal type: {self.signal_type}")
            signal = generator_func(total_length)

            if self.add_noise:
                signal += self.random_state.normal(0, self.noise_std, total_length)

            # Each signal in the batch is normalized independently
            signal = self._normalize_signal(signal)

            X[i, :, 0] = signal[:self.backcast_length]
            y[i, :, 0] = signal[self.backcast_length:]
        return X, y

    def create_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Create a data generator for training."""
        while True:
            yield self.generate_batch()


class RealDataLoader:
    """Loader for real-world time series datasets.
    ...
    """
    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            validation_split: float = 0.2,
            test_split: float = 0.2,
            normalize: bool = True,
            normalization_method: str = 'minmax'
    ) -> None:
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.validation_split = validation_split
        self.test_split = test_split
        self.normalize = normalize
        self.normalizer = TimeSeriesNormalizer(method=normalization_method) if normalize else None

    def _create_sequences(
            self,
            data: np.ndarray,
            stride: int = 1,
            random_sampling: bool = False,
            num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        ts_length, num_features = data.shape
        seq_length = self.backcast_length + self.forecast_length

        if ts_length < seq_length:
            raise ValueError(f"Data length {ts_length} is less than required sequence length {seq_length}")

        max_possible_samples = ts_length - seq_length + 1

        if random_sampling:
            if num_samples is None:
                num_samples = max_possible_samples
            else:
                num_samples = min(num_samples, max_possible_samples)

            indices = np.random.choice(max_possible_samples, size=num_samples, replace=False)
        else:
            indices = np.arange(0, max_possible_samples, stride)
            num_samples = len(indices)

        X = np.zeros((num_samples, self.backcast_length, num_features))
        y = np.zeros((num_samples, self.forecast_length, num_features))

        for i, start_idx in enumerate(indices):
            end_backcast = start_idx + self.backcast_length
            end_forecast = end_backcast + self.forecast_length
            X[i] = data[start_idx:end_backcast]
            y[i] = data[end_backcast:end_forecast]

        return X, y

    def load_csv_data(
            self,
            filepath: str,
            value_column: Union[str, int] = 1,
            header: bool = True,
            delimiter: str = ',',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load time series data from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        logger.info(f"Loading data from {filepath}")

        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)

            if header:
                header_row = next(reader)
                if isinstance(value_column, str):
                    try:
                        value_column = header_row.index(value_column)
                    except ValueError:
                        raise ValueError(f"Column '{value_column}' not found in header: {header_row}")

            for row in reader:
                if len(row) > value_column:
                    value = row[value_column].strip()
                    if value:
                        try:
                            data.append(float(value))
                        except ValueError:
                            logger.warning(f"Could not convert '{value}' to float. Skipping row.")

        if not data:
            raise ValueError("No valid data found in the specified column of the file.")

        data = np.array(data)
        logger.info(f"Loaded {len(data)} data points")
        return self._split_and_process_data(data)

    def load_numpy_data(
            self,
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load time series data from numpy array."""
        logger.info(f"Loading data from numpy array with shape {data.shape}")
        return self._split_and_process_data(data)

    def _split_and_process_data(
            self,
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets and process."""
        total_length = len(data)
        test_size = int(total_length * self.test_split)
        val_size = int(total_length * self.validation_split)
        train_size = total_length - test_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        if self.normalize:
            # Fit normalizer ONLY on the training data to prevent data leakage
            self.normalizer.fit(train_data)

            # Transform all splits using the fitted normalizer
            train_data = self.normalizer.transform(train_data)
            val_data = self.normalizer.transform(val_data)
            test_data = self.normalizer.transform(test_data)

        return train_data, val_data, test_data

    def create_datasets(
            self,
            train_data: np.ndarray,
            val_data: np.ndarray,
            test_data: np.ndarray,
            train_samples: Optional[int] = 2000 # Default to a reasonable number for random sampling
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Create train/validation/test datasets."""
        # Create sequences with random sampling for training for better generalization
        X_train, y_train = self._create_sequences(train_data, random_sampling=True, num_samples=train_samples)

        # Use a fixed stride for validation and test sets for consistent evaluation
        # Stride by forecast_length to get non-overlapping forecast windows
        X_val, y_val = self._create_sequences(val_data, stride=self.forecast_length)
        X_test, y_test = self._create_sequences(test_data, stride=self.forecast_length)

        logger.info(f"Created datasets - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------------------------------------------------------------------

class MultivariateSyntheticGenerator:
    """Generator for multivariate synthetic time series data.
    ...
    """

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            num_variables: int = 3,
            batch_size: int = 32,
            correlation_strength: float = 0.5,
            noise_std: float = 0.1,
            random_seed: Optional[int] = None
    ) -> None:
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_variables = num_variables
        self.batch_size = batch_size
        self.correlation_strength = correlation_strength
        self.noise_std = noise_std
        self.random_state = np.random.RandomState(random_seed)
        self.correlation_matrix = self._generate_correlation_matrix()
        logger.info(f"Initialized multivariate generator with {num_variables} variables")

    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate a valid positive semi-definite correlation matrix."""
        # Generate a random matrix
        A = self.random_state.rand(self.num_variables, self.num_variables)
        # Create a covariance matrix
        cov = A @ A.T
        # Convert covariance to correlation matrix
        std_devs = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(std_devs, std_devs)
        # Interpolate with identity matrix to control correlation strength
        identity = np.eye(self.num_variables)
        final_corr = self.correlation_strength * corr_matrix + (1 - self.correlation_strength) * identity
        return final_corr

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of multivariate time series data."""
        X = np.zeros((self.batch_size, self.backcast_length, self.num_variables))
        y = np.zeros((self.batch_size, self.forecast_length, self.num_variables))
        total_length = self.backcast_length + self.forecast_length

        for i in range(self.batch_size):
            # Generate independent base signals
            base_signals = np.zeros((self.num_variables, total_length))
            for j in range(self.num_variables):
                trend = np.linspace(0, 1, total_length) * self.random_state.uniform(-0.1, 0.1)
                seasonality = np.sin(2 * np.pi * self.random_state.uniform(1, 4) * np.linspace(0, 1, total_length))
                base_signals[j] = trend + seasonality

            # Apply correlation using Cholesky decomposition
            L = np.linalg.cholesky(self.correlation_matrix)
            correlated_signals = L @ base_signals

            # Add noise and normalize each variable's time series
            for j in range(self.num_variables):
                signal = correlated_signals[j] + self.random_state.normal(0, self.noise_std, total_length)
                signal_range = np.max(signal) - np.min(signal)
                signal = (signal - np.min(signal)) / (signal_range + 1e-8)
                X[i, :, j] = signal[:self.backcast_length]
                y[i, :, j] = signal[self.backcast_length:]

        return X, y

    def create_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Create a data generator for training."""
        while True:
            yield self.generate_batch()


def create_synthetic_dataset(
        backcast_length: int,
        forecast_length: int,
        num_samples: int = 1000,
        signal_type: str = 'mixed',
        add_noise: bool = True,
        noise_std: float = 0.1,
        validation_split: float = 0.2,
        test_split: float = 0.2,
        random_seed: Optional[int] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Create a complete synthetic dataset for N-BEATS training."""
    generator = SyntheticDataGenerator(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=num_samples,
        signal_type=signal_type,
        add_noise=add_noise,
        noise_std=noise_std,
        random_seed=random_seed
    )

    X_all, y_all = generator.generate_batch()

    val_size = int(num_samples * validation_split)
    test_size = int(num_samples * test_split)
    train_size = num_samples - val_size - test_size

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:train_size + val_size], y_all[train_size:train_size + val_size]
    X_test, y_test = X_all[train_size + val_size:], y_all[train_size + val_size:]

    logger.info(f"Created synthetic dataset - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------------------------------------------------------------------
