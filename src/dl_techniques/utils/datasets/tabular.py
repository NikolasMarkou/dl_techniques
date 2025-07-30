import numpy as np
from typing import List, Optional, Tuple, Union, Literal
from sklearn.preprocessing import StandardScaler, LabelEncoder

from dl_techniques.utils.logger import logger


class TabularDataProcessor:
    """Preprocessor for tabular data compatible with TabM models.

    Args:
        categorical_columns: List of categorical column names or indices.
        numerical_columns: List of numerical column names or indices.
        handle_missing: Strategy for handling missing values ('drop', 'mean', 'median').
        scale_numerical: Whether to standardize numerical features.
    """

    def __init__(
            self,
            categorical_columns: Optional[List[Union[str, int]]] = None,
            numerical_columns: Optional[List[Union[str, int]]] = None,
            handle_missing: Literal['drop', 'mean', 'median'] = 'mean',
            scale_numerical: bool = True
    ):
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.handle_missing = handle_missing
        self.scale_numerical = scale_numerical

        # Fitted preprocessing objects
        self.label_encoders_ = {}
        self.numerical_scaler_ = None
        self.feature_names_ = None
        self.cat_cardinalities_ = []
        self.fitted_ = False

    def fit(self, X: np.ndarray, column_names: Optional[List[str]] = None):
        """Fit preprocessing transformations.

        Args:
            X: Input data array.
            column_names: Optional column names for the features.
        """
        self.feature_names_ = column_names

        # Auto-detect column types if not specified
        if not self.categorical_columns and not self.numerical_columns:
            self._auto_detect_types(X)

        # Fit categorical encoders
        self.cat_cardinalities_ = []
        for col_idx, col in enumerate(self.categorical_columns):
            if isinstance(col, str):
                if column_names is None:
                    raise ValueError("Column names required when using string column identifiers")
                col_idx = column_names.index(col)

            # Handle missing values in categorical columns
            cat_data = X[:, col_idx].copy()
            if self.handle_missing == 'drop':
                cat_data = cat_data[~np.isnan(cat_data)]
            else:
                # Fill with most frequent value
                unique_vals, counts = np.unique(cat_data[~np.isnan(cat_data)], return_counts=True)
                if len(unique_vals) > 0:
                    most_frequent = unique_vals[np.argmax(counts)]
                    cat_data[np.isnan(cat_data)] = most_frequent

            encoder = LabelEncoder()
            encoder.fit(cat_data.astype(str))
            self.label_encoders_[col_idx] = encoder
            self.cat_cardinalities_.append(len(encoder.classes_))

        # Fit numerical scaler
        if self.scale_numerical and self.numerical_columns:
            num_indices = []
            for col in self.numerical_columns:
                if isinstance(col, str):
                    if column_names is None:
                        raise ValueError("Column names required when using string column identifiers")
                    col_idx = column_names.index(col)
                else:
                    col_idx = col
                num_indices.append(col_idx)

            if num_indices:
                num_data = X[:, num_indices]
                # Handle missing values
                if self.handle_missing == 'mean':
                    num_data = np.where(np.isnan(num_data), np.nanmean(num_data, axis=0), num_data)
                elif self.handle_missing == 'median':
                    num_data = np.where(np.isnan(num_data), np.nanmedian(num_data, axis=0), num_data)

                self.numerical_scaler_ = StandardScaler()
                self.numerical_scaler_.fit(num_data)

        self.fitted_ = True

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data into numerical and categorical arrays.

        Args:
            X: Input data array.

        Returns:
            Tuple of (numerical_features, categorical_features).
        """
        if not self.fitted_:
            raise ValueError("Processor must be fitted before transform")

        X = X.copy()

        # Process numerical features
        X_num = None
        if self.numerical_columns:
            num_indices = []
            for col in self.numerical_columns:
                if isinstance(col, str):
                    col_idx = self.feature_names_.index(col)
                else:
                    col_idx = col
                num_indices.append(col_idx)

            X_num = X[:, num_indices].astype(np.float32)

            # Handle missing values
            if self.handle_missing == 'mean':
                col_means = np.nanmean(X_num, axis=0)
                X_num = np.where(np.isnan(X_num), col_means, X_num)
            elif self.handle_missing == 'median':
                col_medians = np.nanmedian(X_num, axis=0)
                X_num = np.where(np.isnan(X_num), col_medians, X_num)

            # Scale if fitted
            if self.numerical_scaler_ is not None:
                X_num = self.numerical_scaler_.transform(X_num)

        # Process categorical features
        X_cat = None
        if self.categorical_columns:
            cat_features = []
            for col_idx, col in enumerate(self.categorical_columns):
                if isinstance(col, str):
                    col_idx_actual = self.feature_names_.index(col)
                else:
                    col_idx_actual = col

                cat_data = X[:, col_idx_actual].copy()

                # Handle missing values
                if self.handle_missing != 'drop':
                    encoder = self.label_encoders_[col_idx_actual]
                    # Fill with most frequent class
                    most_frequent = encoder.classes_[0]  # Use first class as default
                    cat_data = np.where(np.isnan(cat_data), most_frequent, cat_data)

                # Encode
                encoded = self.label_encoders_[col_idx_actual].transform(cat_data.astype(str))
                cat_features.append(encoded)

            if cat_features:
                X_cat = np.column_stack(cat_features).astype(np.int32)

        return X_num, X_cat

    def fit_transform(self, X: np.ndarray, column_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform data in one step.

        Args:
            X: Input data array.
            column_names: Optional column names.

        Returns:
            Tuple of (numerical_features, categorical_features).
        """
        self.fit(X, column_names)
        return self.transform(X)

    def _auto_detect_types(self, X: np.ndarray):
        """Auto-detect feature types based on data characteristics."""
        n_features = X.shape[1]

        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            # Remove NaN values for type detection
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                continue

            # Check if all values are integers with small range (likely categorical)
            if np.all(valid_data == valid_data.astype(int)):
                unique_vals = np.unique(valid_data)
                if len(unique_vals) <= 20:  # Heuristic for categorical
                    self.categorical_columns.append(col_idx)
                else:
                    self.numerical_columns.append(col_idx)
            else:
                self.numerical_columns.append(col_idx)

        logger.info(
            f"Auto-detected {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical features")

