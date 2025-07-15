"""
N-BEATS Data Loader and Preprocessing Utilities.

This module provides data loading and preprocessing utilities for N-BEATS models,
including synthetic data generation, real dataset loading, and proper normalization.
"""


import os
import csv
import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Generator, Union, List, Dict, Any

from dl_techniques.utils.logger import logger


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
            data: Input data array of shape (samples, time_steps, features).

        Returns:
            Self for method chaining.
        """
        if self.method == 'minmax':
            self.min_val = np.min(data, axis=(0, 1), keepdims=True)
            self.max_val = np.max(data, axis=(0, 1), keepdims=True)

        elif self.method == 'standard':
            self.mean_val = np.mean(data, axis=(0, 1), keepdims=True)
            self.std_val = np.std(data, axis=(0, 1), keepdims=True) + self.epsilon

        elif self.method == 'robust':
            self.median_val = np.median(data, axis=(0, 1), keepdims=True)
            self.q25_val = np.percentile(data, 25, axis=(0, 1), keepdims=True)
            self.q75_val = np.percentile(data, 75, axis=(0, 1), keepdims=True)

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
            data_range = self.max_val - self.min_val + self.epsilon
            normalized = (data - self.min_val) / data_range
            target_range = self.feature_range[1] - self.feature_range[0]
            return normalized * target_range + self.feature_range[0]

        elif self.method == 'standard':
            return (data - self.mean_val) / self.std_val

        elif self.method == 'robust':
            iqr = self.q75_val - self.q25_val + self.epsilon
            return (data - self.median_val) / iqr

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
            unnormalized = (data - self.feature_range[0]) / target_range
            data_range = self.max_val - self.min_val + self.epsilon
            return unnormalized * data_range + self.min_val

        elif self.method == 'standard':
            return data * self.std_val + self.mean_val

        elif self.method == 'robust':
            iqr = self.q75_val - self.q25_val + self.epsilon
            return data * iqr + self.median_val

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step.

        Args:
            data: Input data array.

        Returns:
            Normalized data array.
        """
        return self.fit(data).transform(data)


class SyntheticDataGenerator:
    """Generator for synthetic time series data for N-BEATS training and testing.

    This class generates various types of synthetic time series including trend,
    seasonality, and mixed patterns.

    Args:
        backcast_length: Integer, length of the input sequence.
        forecast_length: Integer, length of the forecast sequence.
        batch_size: Integer, batch size for data generation.
        signal_type: String, type of signal to generate.
        add_noise: Boolean, whether to add noise to the signal.
        noise_std: Float, standard deviation of noise.
        random_seed: Optional integer, random seed for reproducibility.
    """

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            batch_size: int = 32,
            signal_type: str = 'seasonality',
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

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Initialized synthetic data generator for {signal_type} signals")

    def _generate_trend_signal(self, length: int) -> np.ndarray:
        """Generate a trend signal.

        Args:
            length: Length of the signal.

        Returns:
            Trend signal array.
        """
        lin_space = np.linspace(-self.backcast_length, self.forecast_length, length)
        slope = np.random.uniform(-0.1, 0.1)
        offset = np.random.uniform(-1.0, 1.0)
        return slope * lin_space + offset

    def _generate_seasonality_signal(self, length: int) -> np.ndarray:
        """Generate a seasonality signal.

        Args:
            length: Length of the signal.

        Returns:
            Seasonality signal array.
        """
        lin_space = np.linspace(-self.backcast_length, self.forecast_length, length)

        # Multiple seasonal components
        signal = np.zeros(length)

        # Primary seasonal component
        freq1 = np.random.randint(1, 4)
        amp1 = np.random.uniform(0.5, 1.0)
        phase1 = np.random.uniform(0, 2 * np.pi)
        signal += amp1 * np.cos(2 * np.pi * freq1 * lin_space / length + phase1)

        # Secondary seasonal component
        freq2 = np.random.randint(2, 6)
        amp2 = np.random.uniform(0.2, 0.5)
        phase2 = np.random.uniform(0, 2 * np.pi)
        signal += amp2 * np.cos(2 * np.pi * freq2 * lin_space / length + phase2)

        # Add slight trend
        trend_strength = np.random.uniform(-0.05, 0.05)
        signal += trend_strength * lin_space

        return signal

    def _generate_mixed_signal(self, length: int) -> np.ndarray:
        """Generate a mixed trend and seasonality signal.

        Args:
            length: Length of the signal.

        Returns:
            Mixed signal array.
        """
        trend = self._generate_trend_signal(length)
        seasonality = self._generate_seasonality_signal(length)

        # Random weights for mixing
        trend_weight = np.random.uniform(0.3, 0.7)
        seasonality_weight = 1.0 - trend_weight

        return trend_weight * trend + seasonality_weight * seasonality

    def _generate_autoregressive_signal(self, length: int) -> np.ndarray:
        """Generate an autoregressive signal.

        Args:
            length: Length of the signal.

        Returns:
            Autoregressive signal array.
        """
        # AR(2) process
        phi1 = np.random.uniform(0.3, 0.7)
        phi2 = np.random.uniform(-0.3, 0.3)

        # Ensure stationarity
        if phi1 + phi2 >= 1:
            phi1 = 0.5
            phi2 = 0.3

        signal = np.zeros(length)
        signal[0] = np.random.normal(0, 1)
        signal[1] = phi1 * signal[0] + np.random.normal(0, 1)

        for i in range(2, length):
            signal[i] = phi1 * signal[i - 1] + phi2 * signal[i - 2] + np.random.normal(0, 0.1)

        return signal

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1] range.

        Args:
            signal: Input signal array.

        Returns:
            Normalized signal array.
        """
        min_val = np.min(signal)
        max_val = np.max(signal)

        if max_val - min_val == 0:
            return np.zeros_like(signal)

        return (signal - min_val) / (max_val - min_val)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of synthetic time series data.

        Returns:
            Tuple of (X, y) arrays where X is backcast and y is forecast.
        """
        X = np.zeros((self.batch_size, self.backcast_length, 1))
        y = np.zeros((self.batch_size, self.forecast_length, 1))

        total_length = self.backcast_length + self.forecast_length

        for i in range(self.batch_size):
            # Generate signal based on type
            if self.signal_type == 'trend':
                signal = self._generate_trend_signal(total_length)
            elif self.signal_type == 'seasonality':
                signal = self._generate_seasonality_signal(total_length)
            elif self.signal_type == 'mixed':
                signal = self._generate_mixed_signal(total_length)
            elif self.signal_type == 'autoregressive':
                signal = self._generate_autoregressive_signal(total_length)
            else:
                raise ValueError(f"Unknown signal type: {self.signal_type}")

            # Add noise if requested
            if self.add_noise:
                noise = np.random.normal(0, self.noise_std, total_length)
                signal += noise

            # Normalize signal
            signal = self._normalize_signal(signal)

            # Split into backcast and forecast
            X[i, :, 0] = signal[:self.backcast_length]
            y[i, :, 0] = signal[self.backcast_length:]

        return X, y

    def create_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Create a data generator for training.

        Returns:
            Generator yielding (X, y) batches.
        """
        while True:
            yield self.generate_batch()


class RealDataLoader:
    """Loader for real-world time series datasets.

    This class provides utilities for loading and preprocessing real-world
    time series data for N-BEATS models.

    Args:
        backcast_length: Integer, length of the input sequence.
        forecast_length: Integer, length of the forecast sequence.
        validation_split: Float, fraction of data to use for validation.
        test_split: Float, fraction of data to use for testing.
        normalize: Boolean, whether to normalize the data.
        normalization_method: String, normalization method to use.
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
        self.normalization_method = normalization_method

        self.normalizer = None
        if normalize:
            self.normalizer = TimeSeriesNormalizer(method=normalization_method)

    def _create_sequences(
            self,
            data: np.ndarray,
            stride: int = 1,
            random_sampling: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data.

        Args:
            data: Input time series data.
            stride: Stride for sequence creation.
            random_sampling: Whether to use random sampling for training.

        Returns:
            Tuple of (X, y) arrays.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        total_length = self.backcast_length + self.forecast_length

        if data.shape[0] < total_length:
            raise ValueError(f"Data length {data.shape[0]} is less than required {total_length}")

        if random_sampling:
            # Random sampling for training
            num_samples = min(1000, data.shape[0] - total_length + 1)
            indices = np.random.choice(
                range(self.backcast_length, data.shape[0] - self.forecast_length + 1),
                size=num_samples,
                replace=False
            )

            X = np.zeros((num_samples, self.backcast_length, data.shape[1]))
            y = np.zeros((num_samples, self.forecast_length, data.shape[1]))

            for i, idx in enumerate(indices):
                X[i] = data[idx - self.backcast_length:idx]
                y[i] = data[idx:idx + self.forecast_length]

        else:
            # Sequential sampling for validation/test
            num_samples = (data.shape[0] - total_length) // stride + 1

            X = np.zeros((num_samples, self.backcast_length, data.shape[1]))
            y = np.zeros((num_samples, self.forecast_length, data.shape[1]))

            for i in range(num_samples):
                start_idx = i * stride
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
            skip_empty: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load time series data from CSV file.

        Args:
            filepath: Path to the CSV file.
            value_column: Column index or name containing values.
            header: Whether the CSV has a header row.
            delimiter: CSV delimiter.
            skip_empty: Whether to skip empty values.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading data from {filepath}")

        data = []
        with open(filepath, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)

            if header:
                next(reader)  # Skip header

            for row in reader:
                if len(row) > value_column:
                    value = row[value_column]
                    if skip_empty and (value == '' or value is None):
                        continue
                    try:
                        data.append(float(value))
                    except ValueError:
                        if not skip_empty:
                            raise ValueError(f"Cannot convert '{value}' to float")
                        continue

        if len(data) == 0:
            raise ValueError("No valid data found in file")

        data = np.array(data)
        logger.info(f"Loaded {len(data)} data points")

        return self._split_and_process_data(data)

    def load_numpy_data(
            self,
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load time series data from numpy array.

        Args:
            data: Input numpy array.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        logger.info(f"Loading data from numpy array with shape {data.shape}")
        return self._split_and_process_data(data)

    def _split_and_process_data(
            self,
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets and process.

        Args:
            data: Input data array.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        # Calculate split indices
        total_length = len(data)
        test_size = int(total_length * self.test_split)
        val_size = int(total_length * self.validation_split)
        train_size = total_length - test_size - val_size

        # Split data
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Normalize if requested
        if self.normalize:
            # Fit normalizer on training data
            if len(train_data.shape) == 1:
                train_data_3d = train_data.reshape(1, -1, 1)
            else:
                train_data_3d = train_data.reshape(1, -1, train_data.shape[-1])

            self.normalizer.fit(train_data_3d)

            # Transform all splits
            train_data = self.normalizer.transform(train_data_3d).squeeze()

            if len(val_data.shape) == 1:
                val_data_3d = val_data.reshape(1, -1, 1)
            else:
                val_data_3d = val_data.reshape(1, -1, val_data.shape[-1])
            val_data = self.normalizer.transform(val_data_3d).squeeze()

            if len(test_data.shape) == 1:
                test_data_3d = test_data.reshape(1, -1, 1)
            else:
                test_data_3d = test_data.reshape(1, -1, test_data.shape[-1])
            test_data = self.normalizer.transform(test_data_3d).squeeze()

        return train_data, val_data, test_data

    def create_datasets(
            self,
            train_data: np.ndarray,
            val_data: np.ndarray,
            test_data: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Create train/validation/test datasets.

        Args:
            train_data: Training data.
            val_data: Validation data.
            test_data: Test data.

        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
        """
        # Create sequences
        X_train, y_train = self._create_sequences(train_data, random_sampling=True)
        X_val, y_val = self._create_sequences(val_data, stride=self.forecast_length)
        X_test, y_test = self._create_sequences(test_data, stride=self.forecast_length)

        logger.info(f"Created datasets - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class MultivariateSyntheticGenerator:
    """Generator for multivariate synthetic time series data.

    This class generates multivariate time series with controllable relationships
    between variables.

    Args:
        backcast_length: Integer, length of the input sequence.
        forecast_length: Integer, length of the forecast sequence.
        num_variables: Integer, number of variables to generate.
        batch_size: Integer, batch size for data generation.
        correlation_strength: Float, strength of correlation between variables.
        noise_std: Float, standard deviation of noise.
        random_seed: Optional integer, random seed for reproducibility.
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

        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate correlation matrix
        self.correlation_matrix = self._generate_correlation_matrix()

        logger.info(f"Initialized multivariate generator with {num_variables} variables")

    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate a valid correlation matrix.

        Returns:
            Correlation matrix.
        """
        # Generate random correlation matrix
        A = np.random.randn(self.num_variables, self.num_variables)
        correlation_matrix = np.corrcoef(A)

        # Adjust correlation strength
        correlation_matrix = (correlation_matrix * self.correlation_strength +
                              np.eye(self.num_variables) * (1 - self.correlation_strength))

        return correlation_matrix

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of multivariate time series data.

        Returns:
            Tuple of (X, y) arrays.
        """
        X = np.zeros((self.batch_size, self.backcast_length, self.num_variables))
        y = np.zeros((self.batch_size, self.forecast_length, self.num_variables))

        total_length = self.backcast_length + self.forecast_length

        for i in range(self.batch_size):
            # Generate base signals
            signals = np.zeros((self.num_variables, total_length))

            for j in range(self.num_variables):
                # Generate individual signal
                lin_space = np.linspace(0, 1, total_length)

                # Mix of trend and seasonality
                trend = np.random.uniform(-0.1, 0.1) * lin_space
                seasonality = (np.random.uniform(0.5, 1.0) *
                               np.sin(2 * np.pi * np.random.uniform(1, 3) * lin_space))

                signals[j] = trend + seasonality

            # Apply correlation
            L = np.linalg.cholesky(self.correlation_matrix)
            correlated_signals = L @ signals

            # Add noise and normalize
            for j in range(self.num_variables):
                # Add noise
                noise = np.random.normal(0, self.noise_std, total_length)
                signal = correlated_signals[j] + noise

                # Normalize
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

                # Split into backcast and forecast
                X[i, :, j] = signal[:self.backcast_length]
                y[i, :, j] = signal[self.backcast_length:]

        return X, y

    def create_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Create a data generator for training.

        Returns:
            Generator yielding (X, y) batches.
        """
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
    """Create a complete synthetic dataset for N-BEATS training.

    Args:
        backcast_length: Length of input sequences.
        forecast_length: Length of forecast sequences.
        num_samples: Total number of samples to generate.
        signal_type: Type of signal to generate.
        add_noise: Whether to add noise.
        noise_std: Standard deviation of noise.
        validation_split: Fraction for validation.
        test_split: Fraction for test.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    generator = SyntheticDataGenerator(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=num_samples,
        signal_type=signal_type,
        add_noise=add_noise,
        noise_std=noise_std,
        random_seed=random_seed
    )

    # Generate all data
    X_all, y_all = generator.generate_batch()

    # Split data
    val_size = int(num_samples * validation_split)
    test_size = int(num_samples * test_split)
    train_size = num_samples - val_size - test_size

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]

    X_val = X_all[train_size:train_size + val_size]
    y_val = y_all[train_size:train_size + val_size]

    X_test = X_all[train_size + val_size:]
    y_test = y_all[train_size + val_size:]

    logger.info(f"Created synthetic dataset - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)