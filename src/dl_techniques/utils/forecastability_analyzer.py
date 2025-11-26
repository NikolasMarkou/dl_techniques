"""
Forecastability Assessment Framework.

Implements rigorous forecastability assessment using:
- Permutation Entropy (PE) with automatic parameter selection
- Forecast Value Added (FVA) analysis
- Naive baseline benchmarks with time series cross-validation
- Distribution-free complexity measures

Based on Valeriy Manokhin's forecasting science framework.
"""

import numpy as np
from typing import Dict, Tuple, Literal

# ---------------------------------------------------------------------

class ForecastabilityAssessor:
    """
    Implements the Forecastability Assessment Framework.

    Key principles:
    - Forecastability = range of achievable forecast errors (not just stability)
    - Stability ≠ Predictability
    - CoV fundamentally misleads (ignores temporal structure)
    - Use PE for complexity + naive benchmarks for actual forecastability

    Attributes
    ----------
    None

    Methods
    -------
    permutation_entropy : Calculate complexity measure for predictability
    auto_permutation_entropy : PE with automatic parameter selection
    calculate_naive_baselines : Compute standard naive benchmarks
    forecast_value_added : Calculate FVA for model comparison
    assess_forecastability : Complete assessment pipeline

    Notes
    -----
    PE interpretation:
    - PE → 0: Deterministic, highly predictable
    - PE → 1: Random noise, low predictability
    - Lower PE → Higher predictability
    """

    @staticmethod
    def _estimate_time_delay_ami(
            time_series: np.ndarray,
            max_delay: int = 20
    ) -> int:
        """
        Estimate optimal time delay using Average Mutual Information (AMI).

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        max_delay : int, default=20
            Maximum delay to consider

        Returns
        -------
        int
            Optimal time delay (first local minimum of AMI)

        Notes
        -----
        AMI measures shared information between X(t) and X(t+τ).
        First local minimum indicates optimal delay for phase space reconstruction.
        """
        n = len(time_series)
        ami_values = []

        for delay in range(1, min(max_delay, n // 2)):
            # Create delayed series
            x = time_series[:-delay]
            y = time_series[delay:]

            # Discretize into bins for histogram-based MI estimation
            bins = max(10, int(np.sqrt(len(x))))
            hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
            hist_x, _ = np.histogram(x, bins=bins)
            hist_y, _ = np.histogram(y, bins=bins)

            # Normalize to probabilities
            p_xy = hist_xy / len(x)
            p_x = hist_x / len(x)
            p_y = hist_y / len(y)

            # Compute mutual information
            ami = 0.0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        ami += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

            ami_values.append(ami)

        # Find first local minimum
        for i in range(1, len(ami_values) - 1):
            if ami_values[i] < ami_values[i - 1] and ami_values[i] < ami_values[i + 1]:
                return i + 1

        # Default to 1 if no minimum found
        return 1

    @staticmethod
    def _estimate_embedding_dimension_cao(
            time_series: np.ndarray,
            delay: int = 1,
            max_dim: int = 10
    ) -> int:
        """
        Estimate optimal embedding dimension using Cao's method.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        delay : int, default=1
            Time delay for embedding
        max_dim : int, default=10
            Maximum dimension to test

        Returns
        -------
        int
            Optimal embedding dimension

        Notes
        -----
        Cao's method uses E1(d) and E2(d) statistics.
        When E1 stops changing significantly, optimal dimension reached.
        """
        n = len(time_series)
        e1_values = []

        for d in range(1, max_dim + 1):
            # Create d-dimensional embeddings
            max_idx = n - (d + 1) * delay
            if max_idx < 1:
                break

            embeddings_d = np.array([
                time_series[i:i + d * delay:delay]
                for i in range(max_idx + 1)
            ])

            embeddings_d1 = np.array([
                time_series[i:i + (d + 1) * delay:delay]
                for i in range(max_idx)
            ])

            # Find nearest neighbors
            a_d = 0.0
            for i in range(len(embeddings_d) - 1):
                # Find nearest neighbor in d dimensions
                distances = np.linalg.norm(embeddings_d - embeddings_d[i], axis=1)
                distances[i] = np.inf
                nn_idx = np.argmin(distances)

                # Compute ratio of distances in d+1 dimensions
                dist_d1 = np.linalg.norm(embeddings_d1[i] - embeddings_d1[nn_idx])
                dist_d = distances[nn_idx]

                if dist_d > 0:
                    a_d += dist_d1 / dist_d

            a_d /= (len(embeddings_d) - 1)
            e1_values.append(a_d)

        # Find where E1 saturates
        if len(e1_values) < 2:
            return 3

        # Look for plateau in E1 values
        for i in range(1, len(e1_values)):
            e1_ratio = e1_values[i] / (e1_values[i - 1] + 1e-12)
            if 0.95 <= e1_ratio <= 1.05:  # Within 5% change
                return i + 1

        # Default to middle value if no clear plateau
        return max(3, len(e1_values) // 2)

    @staticmethod
    def permutation_entropy(
            time_series: np.ndarray,
            embedding_dim: int = 3,
            delay: int = 1
    ) -> float:
        """
        Calculate Permutation Entropy to measure complexity.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        embedding_dim : int, default=3
            Embedding dimension (D). Number of values in ordinal pattern.
            Typical range: 3-7
        delay : int, default=1
            Time delay (τ) for phase space reconstruction.
            Use AMI or autocorrelation for selection.

        Returns
        -------
        float
            Normalized permutation entropy in [0, 1]
            0 = deterministic (perfectly predictable)
            1 = random noise (maximum entropy)

        Notes
        -----
        PE quantifies predictability through ordinal pattern complexity.

        Advantages over CoV:
        - Non-parametric (no distributional assumptions)
        - Robust to noise and outliers
        - Invariant to monotonic transformations
        - Captures temporal ordering and causal structure

        Interpretation:
        - PE < 0.3: Highly predictable (strong patterns)
        - PE 0.3-0.7: Moderate predictability
        - PE > 0.7: Low predictability (near random)

        References
        ----------
        Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural
        complexity measure for time series. Physical Review Letters, 88(17).
        """
        n = len(time_series)

        # Minimum data requirement
        if n < embedding_dim:
            return 0.0

        # Create m-dimensional phase space vectors with time delay
        max_idx = n - (embedding_dim - 1) * delay
        embedded = np.array([
            time_series[i:i + embedding_dim * delay:delay]
            for i in range(max_idx)
        ])

        # Get ordinal patterns (rank permutations)
        # argsort gives indices that would sort each vector
        ordinal_patterns = np.argsort(embedded, axis=1)

        # Convert patterns to unique identifiers
        pattern_ids = np.apply_along_axis(
            lambda x: ''.join(map(str, x)),
            axis=1,
            arr=ordinal_patterns
        )

        # Count pattern frequencies
        unique_patterns, counts = np.unique(pattern_ids, return_counts=True)
        probabilities = counts / len(pattern_ids)

        # Compute Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))

        # Normalize by maximum possible entropy log2(D!)
        max_entropy = np.log2(np.math.factorial(embedding_dim))
        normalized_entropy = entropy / (max_entropy + 1e-12)

        return float(np.clip(normalized_entropy, 0.0, 1.0))

    @staticmethod
    def auto_permutation_entropy(
            time_series: np.ndarray,
            max_dim: int = 7,
            max_delay: int = 20
    ) -> Tuple[float, int, int]:
        """
        Calculate PE with automatic parameter selection.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        max_dim : int, default=7
            Maximum embedding dimension to consider
        max_delay : int, default=20
            Maximum time delay to consider

        Returns
        -------
        Tuple[float, int, int]
            (permutation_entropy, optimal_embedding_dim, optimal_delay)

        Notes
        -----
        Uses:
        - AMI (Average Mutual Information) for delay selection
        - Cao's method for embedding dimension selection
        """
        # Estimate optimal delay using AMI
        optimal_delay = ForecastabilityAssessor._estimate_time_delay_ami(
            time_series, max_delay
        )

        # Estimate optimal embedding dimension using Cao's method
        optimal_dim = ForecastabilityAssessor._estimate_embedding_dimension_cao(
            time_series, optimal_delay, max_dim
        )

        # Calculate PE with optimal parameters
        pe = ForecastabilityAssessor.permutation_entropy(
            time_series, optimal_dim, optimal_delay
        )

        return pe, optimal_dim, optimal_delay

    @staticmethod
    def calculate_naive_baselines(
            time_series: np.ndarray,
            season_length: int = 1,
            n_folds: int = 5,
            horizon: int = 1
    ) -> Dict[str, float]:
        """
        Compute naive benchmarks using time series cross-validation.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        season_length : int, default=1
            Seasonal period length. Use 1 for non-seasonal data.
        n_folds : int, default=5
            Number of CV folds for expanding window validation
        horizon : int, default=1
            Forecast horizon (steps ahead)

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'naive_mae': Random walk forecast error
            - 'snaive_mae': Seasonal naive forecast error
            - 'ma_mae': Moving average forecast error
            - 'benchmark_mae': Best baseline MAE

        Notes
        -----
        Naive benchmarks are critical for assessing true forecastability.
        If sophisticated models barely beat naive forecasts, series has
        low inherent forecastability.

        Time series CV uses expanding window:
        - Fold 1: Train [0:n1], Test [n1:n1+h]
        - Fold 2: Train [0:n2], Test [n2:n2+h]
        - etc.
        """
        n = len(time_series)

        # Ensure sufficient data
        min_train = max(season_length * 2, 30)
        if n < min_train + horizon * n_folds:
            # Fallback to simple train/test split
            n_folds = 1

        # Calculate fold sizes
        test_size = horizon
        fold_size = (n - min_train - test_size * n_folds) // n_folds if n_folds > 1 else n - min_train - test_size

        naive_errors = []
        snaive_errors = []
        ma_errors = []

        for fold in range(n_folds):
            # Expanding window split
            train_end = min_train + fold * fold_size
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_end - test_start < horizon:
                break

            train = time_series[:train_end]
            test = time_series[test_start:test_end]

            # Naive forecast (random walk)
            naive_pred = np.full(len(test), train[-1])
            naive_errors.append(np.mean(np.abs(test - naive_pred)))

            # Seasonal naive
            if season_length > 1 and len(train) >= season_length:
                last_season = train[-season_length:]
                snaive_pred = np.tile(last_season, (len(test) // season_length) + 1)[:len(test)]
                snaive_errors.append(np.mean(np.abs(test - snaive_pred)))

            # Moving average (last 3 values)
            ma_window = min(3, len(train))
            ma_pred = np.full(len(test), np.mean(train[-ma_window:]))
            ma_errors.append(np.mean(np.abs(test - ma_pred)))

        # Average across folds
        naive_mae = np.mean(naive_errors) if naive_errors else np.inf
        snaive_mae = np.mean(snaive_errors) if snaive_errors else np.inf
        ma_mae = np.mean(ma_errors) if ma_errors else np.inf

        return {
            'naive_mae': float(naive_mae),
            'snaive_mae': float(snaive_mae),
            'ma_mae': float(ma_mae),
            'benchmark_mae': float(min(naive_mae, snaive_mae, ma_mae))
        }

    @staticmethod
    def forecast_value_added(
            time_series: np.ndarray,
            model_predictions: np.ndarray,
            season_length: int = 1,
            metric: Literal['mae', 'rmse'] = 'mae'
    ) -> Dict[str, float]:
        """
        Calculate Forecast Value Added (FVA) for model comparison.

        Parameters
        ----------
        time_series : np.ndarray
            Ground truth time series, shape (n_timesteps,)
        model_predictions : np.ndarray
            Model predictions, shape (n_timesteps,)
            Should align with time_series (same test period)
        season_length : int, default=1
            Seasonal period for seasonal naive benchmark
        metric : {'mae', 'rmse'}, default='mae'
            Error metric to use

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'fva_vs_naive': FVA relative to random walk (%)
            - 'fva_vs_snaive': FVA relative to seasonal naive (%)
            - 'fva_vs_best': FVA relative to best naive baseline (%)
            - 'model_error': Model error
            - 'naive_error': Naive baseline error
            - 'interpretation': Suggested interpretation

        Notes
        -----
        FVA = (Naive_Error - Model_Error) / Naive_Error × 100%

        Interpretation:
        - FVA > 10%: Model adds substantial value
        - FVA 0-10%: Marginal improvement
        - FVA < 0%: Model destroys value (use naive instead)

        If FVA is low even with sophisticated models, series has
        low inherent forecastability regardless of complexity.
        """
        n = len(time_series)
        if len(model_predictions) != n:
            raise ValueError("time_series and model_predictions must have same length")

        # Calculate model error
        if metric == 'mae':
            model_error = np.mean(np.abs(time_series - model_predictions))

            # Naive baseline
            naive_pred = np.roll(time_series, 1)
            naive_pred[0] = time_series[0]
            naive_error = np.mean(np.abs(time_series - naive_pred))

            # Seasonal naive
            if season_length > 1 and n >= season_length:
                snaive_pred = np.roll(time_series, season_length)
                snaive_pred[:season_length] = time_series[:season_length]
                snaive_error = np.mean(np.abs(time_series - snaive_pred))
            else:
                snaive_error = np.inf

        else:  # rmse
            model_error = np.sqrt(np.mean((time_series - model_predictions) ** 2))

            naive_pred = np.roll(time_series, 1)
            naive_pred[0] = time_series[0]
            naive_error = np.sqrt(np.mean((time_series - naive_pred) ** 2))

            if season_length > 1 and n >= season_length:
                snaive_pred = np.roll(time_series, season_length)
                snaive_pred[:season_length] = time_series[:season_length]
                snaive_error = np.sqrt(np.mean((time_series - snaive_pred) ** 2))
            else:
                snaive_error = np.inf

        # Calculate FVA
        fva_naive = ((naive_error - model_error) / (naive_error + 1e-12)) * 100
        fva_snaive = ((snaive_error - model_error) / (
                    snaive_error + 1e-12)) * 100 if snaive_error != np.inf else -np.inf
        fva_best = max(fva_naive, fva_snaive)

        # Interpretation
        if fva_best > 10:
            interpretation = "Substantial value added"
        elif fva_best > 0:
            interpretation = "Marginal improvement"
        else:
            interpretation = "Model destroys value - use naive forecast"

        return {
            'fva_vs_naive': float(fva_naive),
            'fva_vs_snaive': float(fva_snaive),
            'fva_vs_best': float(fva_best),
            'model_error': float(model_error),
            'naive_error': float(naive_error),
            'interpretation': interpretation
        }

    @staticmethod
    def assess_forecastability(
            time_series: np.ndarray,
            season_length: int = 1,
            auto_pe: bool = True
    ) -> Dict[str, any]:
        """
        Complete forecastability assessment pipeline.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series, shape (n_timesteps,)
        season_length : int, default=1
            Seasonal period (1 for non-seasonal)
        auto_pe : bool, default=True
            If True, use automatic parameter selection for PE

        Returns
        -------
        Dict[str, any]
            Comprehensive assessment containing:
            - 'permutation_entropy': PE value and parameters
            - 'naive_baselines': Naive forecast errors
            - 'forecastability_score': Combined score [0-100]
            - 'recommendation': Suggested approach

        Notes
        -----
        Integrated assessment combining:
        1. PE for temporal complexity
        2. Naive baselines for practical forecastability
        3. Combined score for decision-making

        Forecastability score interpretation:
        - 80-100: Highly forecastable
        - 60-80: Moderately forecastable
        - 40-60: Low forecastability
        - 0-40: Very low forecastability
        """
        # Calculate PE
        if auto_pe:
            pe, dim, delay = ForecastabilityAssessor.auto_permutation_entropy(time_series)
            pe_info = {
                'value': pe,
                'embedding_dim': dim,
                'delay': delay,
                'method': 'auto'
            }
        else:
            pe = ForecastabilityAssessor.permutation_entropy(time_series)
            pe_info = {
                'value': pe,
                'embedding_dim': 3,
                'delay': 1,
                'method': 'default'
            }

        # Calculate naive baselines
        baselines = ForecastabilityAssessor.calculate_naive_baselines(
            time_series, season_length
        )

        # Combined forecastability score (0-100)
        # Lower PE → Higher score (more predictable)
        # Lower baseline error relative to scale → Higher score
        pe_score = (1 - pe) * 100

        # Normalize baseline error by series scale
        series_scale = np.std(time_series) + 1e-12
        normalized_baseline = baselines['benchmark_mae'] / series_scale
        baseline_score = max(0, 100 * (1 - min(normalized_baseline, 1.0)))

        # Combined score (weighted average)
        forecastability_score = 0.6 * pe_score + 0.4 * baseline_score

        # Recommendation
        if forecastability_score >= 80:
            recommendation = "Highly forecastable - sophisticated models recommended"
        elif forecastability_score >= 60:
            recommendation = "Moderately forecastable - test multiple approaches"
        elif forecastability_score >= 40:
            recommendation = "Low forecastability - naive baselines may suffice"
        else:
            recommendation = "Very low forecastability - use naive forecasts or reconsider forecasting"

        return {
            'permutation_entropy': pe_info,
            'naive_baselines': baselines,
            'forecastability_score': float(forecastability_score),
            'recommendation': recommendation
        }

# ---------------------------------------------------------------------
