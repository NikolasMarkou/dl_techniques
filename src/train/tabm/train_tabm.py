import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from typing import List, Optional, Tuple, Union, Any, Dict

from dl_techniques.losses.tabm_loss import TabMLoss
from dl_techniques.models.tabm import (
    create_tabm_model, create_tabm_ensemble, TabMModel
)
from dl_techniques.datasets import TabularDataProcessor
from dl_techniques.utils.logger import logger



class TabMTrainer:
    """Trainer class for TabM models with ensemble evaluation and selection.

    Args:
        model: TabM model to train.
        validation_split: Fraction of training data to use for validation.
        early_stopping_patience: Patience for early stopping.
        ensemble_selection: Whether to perform ensemble member selection.
    """

    def __init__(
            self,
            model: 'TabMModel',
            validation_split: float = 0.2,
            early_stopping_patience: int = 10,
            ensemble_selection: bool = True
    ):
        self.model = model
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.ensemble_selection = ensemble_selection

        self.history_ = None
        self.best_ensemble_members_ = None
        self.ensemble_scores_ = None

    def train(
            self,
            X_num: Optional[np.ndarray],
            X_cat: Optional[np.ndarray],
            y: np.ndarray,
            batch_size: int = 256,
            epochs: int = 100,
            verbose: int = 1,
            **fit_kwargs
    ) -> Dict[str, Any]:
        """Train the TabM model.

        Args:
            X_num: Numerical features.
            X_cat: Categorical features.
            y: Target labels.
            batch_size: Batch size for training.
            epochs: Maximum number of epochs.
            verbose: Verbosity level.
            **fit_kwargs: Additional arguments for model.fit().

        Returns:
            Training history and results.
        """
        # Prepare input data
        if X_num is not None and X_cat is not None:
            X = {'x_num': X_num, 'x_cat': X_cat}
        elif X_num is not None:
            X = X_num
        elif X_cat is not None:
            X = X_cat
        else:
            raise ValueError("At least one of X_num or X_cat must be provided")

        # Split data
        if self.validation_split > 0:
            if isinstance(X, dict):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=42,
                    stratify=y if len(np.unique(y)) < 20 else None
                )
                # Handle dict splitting
                if isinstance(X, dict):
                    X_train_dict = {}
                    X_val_dict = {}
                    for key in X.keys():
                        X_data = X[key]
                        X_train_split, X_val_split = train_test_split(
                            X_data, test_size=self.validation_split, random_state=42,
                            stratify=y if len(np.unique(y)) < 20 else None
                        )
                        X_train_dict[key] = X_train_split
                        X_val_dict[key] = X_val_split
                    X_train, X_val = X_train_dict, X_val_dict
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=42,
                    stratify=y if len(np.unique(y)) < 20 else None
                )
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
            validation_data = None

        # Setup callbacks
        callbacks = []
        if validation_data is not None and self.early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            )
            callbacks.append(early_stopping)

        # Train model
        self.history_ = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            **fit_kwargs
        )

        # Perform ensemble evaluation if applicable
        results = {'history': self.history_.history}

        if self.ensemble_selection and self.model.k is not None and validation_data is not None:
            ensemble_results = self._evaluate_ensemble_members(X_val, y_val)
            results.update(ensemble_results)

        return results

    def _evaluate_ensemble_members(self, X_val: Any, y_val: np.ndarray) -> Dict[str, Any]:
        """Evaluate individual ensemble members and select best combinations.

        Args:
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dictionary with ensemble evaluation results.
        """
        # Get ensemble predictions
        val_predictions = self.model.predict(X_val, verbose=0)  # (batch_size, k, n_outputs)
        k = val_predictions.shape[1]

        # Evaluate individual ensemble members
        member_scores = []
        for i in range(k):
            member_pred = val_predictions[:, i]
            if self.model.n_classes is None:
                # Regression
                score = -mean_squared_error(y_val, member_pred.squeeze())
            elif self.model.n_classes == 2:
                # Binary classification
                if member_pred.shape[-1] == 1:
                    member_pred = member_pred.squeeze()
                else:
                    member_pred = member_pred[:, 1]
                score = roc_auc_score(y_val, member_pred)
            else:
                # Multiclass classification
                score = accuracy_score(y_val, np.argmax(member_pred, axis=-1))

            member_scores.append(score)

        self.ensemble_scores_ = np.array(member_scores)

        # Find best single member
        best_member_idx = np.argmax(self.ensemble_scores_)

        # Greedy ensemble selection
        selected_members = [best_member_idx]
        best_score = self.ensemble_scores_[best_member_idx]

        for _ in range(k - 1):
            best_candidate = None
            best_candidate_score = best_score

            for candidate in range(k):
                if candidate in selected_members:
                    continue

                # Test adding this candidate
                candidate_members = selected_members + [candidate]
                ensemble_pred = np.mean(val_predictions[:, candidate_members], axis=1)

                if self.model.n_classes is None:
                    # Regression
                    candidate_score = -mean_squared_error(y_val, ensemble_pred.squeeze())
                elif self.model.n_classes == 2:
                    # Binary classification
                    if ensemble_pred.shape[-1] == 1:
                        ensemble_pred = ensemble_pred.squeeze()
                    else:
                        ensemble_pred = ensemble_pred[:, 1]
                    candidate_score = roc_auc_score(y_val, ensemble_pred)
                else:
                    # Multiclass classification
                    candidate_score = accuracy_score(y_val, np.argmax(ensemble_pred, axis=-1))

                if candidate_score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = candidate_score

            if best_candidate is not None:
                selected_members.append(best_candidate)
                best_score = best_candidate_score
            else:
                break  # No improvement found

        self.best_ensemble_members_ = selected_members

        return {
            'individual_scores': member_scores,
            'best_member_idx': best_member_idx,
            'best_member_score': self.ensemble_scores_[best_member_idx],
            'selected_members': selected_members,
            'ensemble_score': best_score,
            'mean_ensemble_score': np.mean(self.ensemble_scores_)
        }

    def predict(self, X: Any, use_best_ensemble: bool = True) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Input features.
            use_best_ensemble: Whether to use selected ensemble members only.

        Returns:
            Predictions array.
        """
        predictions = self.model.predict(X, verbose=0)

        if self.model.k is None:
            # Plain model - remove ensemble dimension
            return predictions.squeeze(axis=1)

        if use_best_ensemble and self.best_ensemble_members_ is not None:
            # Use selected ensemble members
            predictions = predictions[:, self.best_ensemble_members_]

        # Average ensemble predictions
        return np.mean(predictions, axis=1)


def create_and_train_tabm(
        X: np.ndarray,
        y: np.ndarray,
        categorical_columns: Optional[List[Union[str, int]]] = None,
        column_names: Optional[List[str]] = None,
        arch_type: str = 'tabm',
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        batch_size: int = 256,
        epochs: int = 100,
        validation_split: float = 0.2,
        test_split: float = 0.2,
        verbose: int = 1,
        **model_kwargs
) -> Tuple['TabMModel', TabMTrainer, Dict[str, Any]]:
    """Complete pipeline to create, preprocess data, and train a TabM model.

    Args:
        X: Input features.
        y: Target labels.
        categorical_columns: Categorical column identifiers.
        column_names: Column names for features.
        arch_type: TabM architecture type.
        k: Number of ensemble members.
        hidden_dims: Hidden layer dimensions.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        validation_split: Validation split fraction.
        test_split: Test split fraction.
        verbose: Verbosity level.
        **model_kwargs: Additional model arguments.

    Returns:
        Tuple of (trained_model, trainer, results).

    Example:
        >>> # Generate sample data
        >>> X = np.random.randn(1000, 12)
        >>> y = np.random.randint(0, 3, 1000)

        >>> # Train TabM model
        >>> model, trainer, results = create_and_train_tabm(
        ...     X, y,
        ...     categorical_columns=[0, 1, 2],  # First 3 columns are categorical
        ...     arch_type='tabm',
        ...     k=8,
        ...     epochs=50
        ... )

        >>> # Make predictions
        >>> predictions = trainer.predict(X_test)
    """
    from dl_techniques.models.tabm import create_tabm_for_dataset

    # Split data into train/test
    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42,
            stratify=y if len(np.unique(y)) < 20 else None
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    # Preprocess data
    processor = TabularDataProcessor(categorical_columns=categorical_columns)
    X_num_train, X_cat_train = processor.fit_transform(X_train, column_names)

    if X_test is not None:
        X_num_test, X_cat_test = processor.transform(X_test)

    # Create model
    categorical_cardinalities = processor.cat_cardinalities_
    n_num_features = X_num_train.shape[1] if X_num_train is not None else 0

    model = create_tabm_for_dataset(
        X_train=X_train,
        y_train=y_train,
        categorical_indices=categorical_columns,
        categorical_cardinalities=categorical_cardinalities,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **model_kwargs
    )

    # Compile model
    if len(np.unique(y_train)) == 2:
        loss = 'binary_crossentropy' if arch_type == 'plain' else 'binary_crossentropy'
        metrics = ['binary_accuracy']
    elif len(np.unique(y_train)) > 2 and np.all(np.unique(y_train) == np.arange(len(np.unique(y_train)))):
        loss = 'sparse_categorical_crossentropy' if arch_type == 'plain' else 'sparse_categorical_crossentropy'
        metrics = ['sparse_categorical_accuracy']
    else:
        loss = 'mse' if arch_type == 'plain' else 'mse'
        metrics = ['mae']

    if arch_type != 'plain':
        from dl_techniques.models.tabm import TabMLoss
        loss = TabMLoss(loss)

    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    # Train model
    trainer = TabMTrainer(model, validation_split=validation_split)
    training_results = trainer.train(
        X_num_train, X_cat_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )

    # Evaluate on test set if available
    if X_test is not None:
        test_predictions = trainer.predict({'x_num': X_num_test,
                                            'x_cat': X_cat_test} if X_num_test is not None and X_cat_test is not None else X_num_test if X_num_test is not None else X_cat_test)

        if len(np.unique(y_test)) > 2:
            test_accuracy = accuracy_score(y_test, np.argmax(test_predictions,
                                                             axis=-1) if test_predictions.ndim > 1 else test_predictions.round())
            training_results['test_accuracy'] = test_accuracy
        else:
            if test_predictions.ndim > 1 and test_predictions.shape[-1] > 1:
                test_predictions = test_predictions[:, 1]
            test_auc = roc_auc_score(y_test, test_predictions)
            training_results['test_auc'] = test_auc

    if verbose:
        logger.info("Training completed successfully!")
        if 'test_accuracy' in training_results:
            logger.info(f"Test accuracy: {training_results['test_accuracy']:.4f}")
        if 'test_auc' in training_results:
            logger.info(f"Test AUC: {training_results['test_auc']:.4f}")

    return model, trainer, training_results


def example_synthetic_classification():
    """Example with synthetic classification data."""
    logger.info("=== Synthetic Classification Example ===")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    # Convert some features to categorical
    categorical_indices = [0, 1, 2, 3]  # First 4 features as categorical
    for idx in categorical_indices:
        # Convert to categorical by binning
        X[:, idx] = pd.cut(X[:, idx], bins=5, labels=False)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess data
    processor = TabularDataProcessor(
        categorical_columns=categorical_indices,
        scale_numerical=True
    )
    X_num_train, X_cat_train = processor.fit_transform(X_train)
    X_num_test, X_cat_test = processor.transform(X_test)

    logger.info(f"Data shapes - X_num: {X_num_train.shape}, X_cat: {X_cat_train.shape}")
    logger.info(f"Categorical cardinalities: {processor.cat_cardinalities_}")

    # Compare different architectures
    architectures = ['plain', 'tabm', 'tabm-mini']
    results = {}

    for arch in architectures:
        logger.info(f"\n--- Training {arch.upper()} model ---")

        # Create model
        model = create_tabm_model(
            n_num_features=X_num_train.shape[1],
            cat_cardinalities=processor.cat_cardinalities_,
            n_classes=3,
            arch_type=arch,
            k=None if arch == 'plain' else 8,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )

        # Compile model
        if arch == 'plain':
            loss = 'sparse_categorical_crossentropy'
        else:
            loss = TabMLoss('sparse_categorical_crossentropy')

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['sparse_categorical_accuracy']
        )

        # Train model
        trainer = TabMTrainer(model, validation_split=0.2)
        training_results = trainer.train(
            X_num_train, X_cat_train, y_train,
            batch_size=128,
            epochs=50,
            verbose=0
        )

        # Evaluate
        if arch == 'plain':
            # For plain model, combine features
            X_combined_test = np.concatenate([X_num_test, X_cat_test], axis=1)
            test_pred = model.predict(X_combined_test, verbose=0)
            test_pred = test_pred.squeeze(axis=1)  # Remove ensemble dimension
        else:
            test_pred = trainer.predict({'x_num': X_num_test, 'x_cat': X_cat_test})

        test_accuracy = accuracy_score(y_test, np.argmax(test_pred, axis=-1))

        results[arch] = {
            'model': model,
            'trainer': trainer,
            'test_accuracy': test_accuracy,
            'history': training_results['history']
        }

        logger.info(f"{arch} test accuracy: {test_accuracy:.4f}")

        # Print ensemble results if applicable
        if arch != 'plain' and 'individual_scores' in training_results:
            logger.info(f"Individual ensemble scores: {training_results['individual_scores']}")
            logger.info(f"Best ensemble score: {training_results['ensemble_score']:.4f}")

    return results


def example_real_dataset():
    """Example with real dataset (breast cancer)."""
    logger.info("\n=== Breast Cancer Dataset Example ===")

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Classes: {np.unique(y)}")

    # Use the simplified training pipeline
    model, trainer, results = create_and_train_tabm(
        X, y,
        categorical_columns=[],  # All features are numerical
        arch_type='tabm',
        k=6,
        hidden_dims=[64, 32],
        epochs=100,
        validation_split=0.2,
        test_split=0.2,
        verbose=1
    )

    logger.info(f"Final results: {results}")

    return model, trainer, results


def example_regression():
    """Example with regression data."""
    logger.info("\n=== Regression Example ===")

    # Generate regression data
    X, y = make_regression(
        n_samples=1500,
        n_features=12,
        n_informative=8,
        noise=0.1,
        random_state=42
    )

    # Add some categorical features
    categorical_indices = [0, 1]
    for idx in categorical_indices:
        # Convert to categorical by binning
        X[:, idx] = pd.cut(X[:, idx], bins=4, labels=False)

    # Train TabM model
    model, trainer, results = create_and_train_tabm(
        X, y,
        categorical_columns=categorical_indices,
        arch_type='tabm-mini',
        k=4,
        hidden_dims=[96, 48],
        epochs=80,
        validation_split=0.15,
        test_split=0.2,
        verbose=1
    )

    return model, trainer, results


def example_ensemble_analysis():
    """Example showing detailed ensemble analysis."""
    logger.info("\n=== Ensemble Analysis Example ===")

    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create TabM ensemble
    model = create_tabm_ensemble(
        n_num_features=10,
        cat_cardinalities=[],
        n_classes=2,
        k=10,  # Larger ensemble for analysis
        hidden_dims=[64, 32]
    )

    # Compile and train
    model.compile(
        optimizer='adam',
        loss=TabMLoss('binary_crossentropy'),
        metrics=['binary_accuracy']
    )

    trainer = TabMTrainer(model, validation_split=0.2, ensemble_selection=True)
    results = trainer.train(
        X_train, None, y_train,
        batch_size=64,
        epochs=50,
        verbose=1
    )

    # Analyze ensemble members
    if 'individual_scores' in results:
        individual_scores = results['individual_scores']
        selected_members = results['selected_members']

        logger.info(f"Individual member scores: {individual_scores}")
        logger.info(f"Selected members: {selected_members}")
        logger.info(f"Best single member score: {results['best_member_score']:.4f}")
        logger.info(f"Ensemble score: {results['ensemble_score']:.4f}")

        # Plot ensemble member performance
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(individual_scores)), individual_scores)
        plt.axhline(y=np.mean(individual_scores), color='r', linestyle='--', label='Mean')
        plt.xlabel('Ensemble Member')
        plt.ylabel('Validation Score')
        plt.title('Individual Ensemble Member Performance')
        plt.legend()

        plt.subplot(1, 2, 2)
        cumulative_scores = []
        current_members = []
        for member in selected_members:
            current_members.append(member)
            # Simulate cumulative ensemble performance
            cumulative_scores.append(np.mean([individual_scores[m] for m in current_members]))

        plt.plot(range(1, len(cumulative_scores) + 1), cumulative_scores, 'o-')
        plt.xlabel('Number of Selected Members')
        plt.ylabel('Ensemble Score')
        plt.title('Greedy Ensemble Selection Progress')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('ensemble_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    return model, trainer, results


def example_custom_data_pipeline():
    """Example with custom data preprocessing pipeline."""
    logger.info("\n=== Custom Data Pipeline Example ===")

    # Create a more complex synthetic dataset
    np.random.seed(42)
    n_samples = 2000

    # Numerical features
    num_features = np.random.randn(n_samples, 8)

    # Categorical features with different cardinalities
    cat1 = np.random.randint(0, 5, n_samples)  # 5 categories
    cat2 = np.random.randint(0, 3, n_samples)  # 3 categories
    cat3 = np.random.randint(0, 10, n_samples)  # 10 categories

    # Combine features
    X = np.column_stack([num_features, cat1, cat2, cat3])

    # Create target with some relationship to features
    y = (
            2 * num_features[:, 0] +
            num_features[:, 1] -
            0.5 * num_features[:, 2] +
            0.3 * cat1 +
            np.random.normal(0, 0.5, n_samples)
    )

    # Convert to classification
    y = (y > np.median(y)).astype(int)

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Feature composition: 8 numerical + 3 categorical")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Custom preprocessing
    processor = TabularDataProcessor(
        categorical_columns=[8, 9, 10],  # Last 3 columns
        scale_numerical=True,
        handle_missing='mean'
    )

    # Split and preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_num_train, X_cat_train = processor.fit_transform(X_train)
    X_num_test, X_cat_test = processor.transform(X_test)

    # Compare multiple TabM variants
    variants = [
        {'arch_type': 'tabm', 'k': 8, 'name': 'TabM'},
        {'arch_type': 'tabm-mini', 'k': 6, 'name': 'TabM-mini'},
        {'arch_type': 'tabm-normal', 'k': 8, 'name': 'TabM-normal'},
        {'arch_type': 'plain', 'k': None, 'name': 'Plain MLP'}
    ]

    results_comparison = {}

    for variant in variants:
        logger.info(f"\n--- Training {variant['name']} ---")

        model = create_tabm_model(
            n_num_features=X_num_train.shape[1],
            cat_cardinalities=processor.cat_cardinalities_,
            n_classes=2,
            arch_type=variant['arch_type'],
            k=variant['k'],
            hidden_dims=[128, 64, 32],
            dropout_rate=0.15
        )

        # Compile
        if variant['arch_type'] == 'plain':
            loss = 'binary_crossentropy'
        else:
            loss = TabMLoss('binary_crossentropy')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['binary_accuracy']
        )

        # Train
        trainer = TabMTrainer(model, validation_split=0.15)
        training_results = trainer.train(
            X_num_train, X_cat_train, y_train,
            batch_size=128,
            epochs=60,
            verbose=0
        )

        # Evaluate
        if variant['arch_type'] == 'plain':
            X_combined_test = np.concatenate([X_num_test, X_cat_test], axis=1)
            test_pred = model.predict(X_combined_test, verbose=0).squeeze()
        else:
            test_pred = trainer.predict({'x_num': X_num_test, 'x_cat': X_cat_test})

        # Calculate metrics
        if test_pred.ndim > 1 and test_pred.shape[-1] > 1:
            test_pred_proba = test_pred[:, 1]
            test_pred_class = np.argmax(test_pred, axis=-1)
        else:
            test_pred_proba = test_pred.squeeze()
            test_pred_class = (test_pred_proba > 0.5).astype(int)

        test_accuracy = accuracy_score(y_test, test_pred_class)
        test_auc = roc_auc_score(y_test, test_pred_proba)

        results_comparison[variant['name']] = {
            'accuracy': test_accuracy,
            'auc': test_auc,
            'model': model,
            'trainer': trainer
        }

        logger.info(f"{variant['name']} - Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")

    # Summary comparison
    logger.info("\n=== Results Summary ===")
    for name, result in results_comparison.items():
        logger.info(f"{name:12} - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")

    return results_comparison


def main():
    """Run all examples."""
    logger.info("TabM Model Examples")
    logger.info("==================")

    # Run examples
    try:
        example_synthetic_classification()
    except Exception as e:
        logger.error(f"Error in synthetic classification example: {e}")

    try:
        example_real_dataset()
    except Exception as e:
        logger.error(f"Error in real dataset example: {e}")

    try:
        example_regression()
    except Exception as e:
        logger.error(f"Error in regression example: {e}")

    try:
        example_ensemble_analysis()
    except Exception as e:
        logger.error(f"Error in ensemble analysis example: {e}")

    try:
        example_custom_data_pipeline()
    except Exception as e:
        logger.error(f"Error in custom pipeline example: {e}")

    logger.info("\nAll examples completed!")


if __name__ == "__main__":
    main()