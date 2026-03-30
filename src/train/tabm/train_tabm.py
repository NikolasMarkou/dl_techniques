"""Training pipeline for TabM models with ensemble evaluation and selection."""

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

from train.common import setup_gpu


class TabMTrainer:
    """Trainer for TabM models with ensemble evaluation and selection.

    Args:
        model: TabM model to train.
        validation_split: Fraction of training data for validation.
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
        if X_num is not None and X_cat is not None:
            X = {'x_num': X_num, 'x_cat': X_cat}
        elif X_num is not None:
            X = X_num
        elif X_cat is not None:
            X = X_cat
        else:
            raise ValueError("At least one of X_num or X_cat must be provided")

        # Split data for validation
        if self.validation_split > 0:
            if isinstance(X, dict):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=42,
                    stratify=y if len(np.unique(y)) < 20 else None
                )
                if isinstance(X, dict):
                    X_train_dict, X_val_dict = {}, {}
                    for key in X.keys():
                        X_train_split, X_val_split = train_test_split(
                            X[key], test_size=self.validation_split, random_state=42,
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

        callbacks = []
        if validation_data is not None and self.early_stopping_patience > 0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ))

        self.history_ = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            **fit_kwargs
        )

        results = {'history': self.history_.history}
        if self.ensemble_selection and self.model.k is not None and validation_data is not None:
            results.update(self._evaluate_ensemble_members(X_val, y_val))

        return results

    def _evaluate_ensemble_members(self, X_val: Any, y_val: np.ndarray) -> Dict[str, Any]:
        """Evaluate individual ensemble members and select best combinations."""
        val_predictions = self.model.predict(X_val, verbose=0)
        k = val_predictions.shape[1]

        member_scores = []
        for i in range(k):
            member_pred = val_predictions[:, i]
            if self.model.n_classes is None:
                score = -mean_squared_error(y_val, member_pred.squeeze())
            elif self.model.n_classes == 2:
                if member_pred.shape[-1] == 1:
                    member_pred = member_pred.squeeze()
                else:
                    member_pred = member_pred[:, 1]
                score = roc_auc_score(y_val, member_pred)
            else:
                score = accuracy_score(y_val, np.argmax(member_pred, axis=-1))
            member_scores.append(score)

        self.ensemble_scores_ = np.array(member_scores)
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
                candidate_members = selected_members + [candidate]
                ensemble_pred = np.mean(val_predictions[:, candidate_members], axis=1)

                if self.model.n_classes is None:
                    candidate_score = -mean_squared_error(y_val, ensemble_pred.squeeze())
                elif self.model.n_classes == 2:
                    if ensemble_pred.shape[-1] == 1:
                        ensemble_pred = ensemble_pred.squeeze()
                    else:
                        ensemble_pred = ensemble_pred[:, 1]
                    candidate_score = roc_auc_score(y_val, ensemble_pred)
                else:
                    candidate_score = accuracy_score(y_val, np.argmax(ensemble_pred, axis=-1))

                if candidate_score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = candidate_score

            if best_candidate is not None:
                selected_members.append(best_candidate)
                best_score = best_candidate_score
            else:
                break

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
            return predictions.squeeze(axis=1)

        if use_best_ensemble and self.best_ensemble_members_ is not None:
            predictions = predictions[:, self.best_ensemble_members_]

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
    """
    from dl_techniques.models.tabm import create_tabm_for_dataset

    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42,
            stratify=y if len(np.unique(y)) < 20 else None
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    processor = TabularDataProcessor(categorical_columns=categorical_columns)
    X_num_train, X_cat_train = processor.fit_transform(X_train, column_names)
    if X_test is not None:
        X_num_test, X_cat_test = processor.transform(X_test)

    model = create_tabm_for_dataset(
        X_train=X_train,
        y_train=y_train,
        categorical_indices=categorical_columns,
        categorical_cardinalities=processor.cat_cardinalities_,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **model_kwargs
    )

    # Select loss based on task type
    n_unique = len(np.unique(y_train))
    if n_unique == 2:
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    elif n_unique > 2 and np.all(np.unique(y_train) == np.arange(n_unique)):
        loss = 'sparse_categorical_crossentropy'
        metrics = ['sparse_categorical_accuracy']
    else:
        loss = 'mse'
        metrics = ['mae']

    if arch_type != 'plain':
        from dl_techniques.models.tabm import TabMLoss
        loss = TabMLoss(loss)

    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    trainer = TabMTrainer(model, validation_split=validation_split)
    training_results = trainer.train(
        X_num_train, X_cat_train, y_train,
        batch_size=batch_size, epochs=epochs, verbose=verbose
    )

    # Evaluate on test set
    if X_test is not None:
        if X_num_test is not None and X_cat_test is not None:
            test_input = {'x_num': X_num_test, 'x_cat': X_cat_test}
        elif X_num_test is not None:
            test_input = X_num_test
        else:
            test_input = X_cat_test
        test_predictions = trainer.predict(test_input)

        if n_unique > 2:
            test_accuracy = accuracy_score(
                y_test,
                np.argmax(test_predictions, axis=-1) if test_predictions.ndim > 1 else test_predictions.round()
            )
            training_results['test_accuracy'] = test_accuracy
        else:
            if test_predictions.ndim > 1 and test_predictions.shape[-1] > 1:
                test_predictions = test_predictions[:, 1]
            training_results['test_auc'] = roc_auc_score(y_test, test_predictions)

    if verbose:
        logger.info("Training completed")
        if 'test_accuracy' in training_results:
            logger.info(f"Test accuracy: {training_results['test_accuracy']:.4f}")
        if 'test_auc' in training_results:
            logger.info(f"Test AUC: {training_results['test_auc']:.4f}")

    return model, trainer, training_results


# ---------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------

def example_synthetic_classification():
    """Example with synthetic classification data."""
    logger.info("=== Synthetic Classification Example ===")

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=3, random_state=42
    )

    categorical_indices = [0, 1, 2, 3]
    for idx in categorical_indices:
        X[:, idx] = pd.cut(X[:, idx], bins=5, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    processor = TabularDataProcessor(categorical_columns=categorical_indices, scale_numerical=True)
    X_num_train, X_cat_train = processor.fit_transform(X_train)
    X_num_test, X_cat_test = processor.transform(X_test)

    logger.info(f"Data shapes - X_num: {X_num_train.shape}, X_cat: {X_cat_train.shape}")
    logger.info(f"Categorical cardinalities: {processor.cat_cardinalities_}")

    architectures = ['plain', 'tabm', 'tabm-mini']
    results = {}

    for arch in architectures:
        logger.info(f"\n--- Training {arch.upper()} ---")

        model = create_tabm_model(
            n_num_features=X_num_train.shape[1],
            cat_cardinalities=processor.cat_cardinalities_,
            n_classes=3,
            arch_type=arch,
            k=None if arch == 'plain' else 8,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )

        loss = 'sparse_categorical_crossentropy' if arch == 'plain' else TabMLoss('sparse_categorical_crossentropy')
        model.compile(optimizer='adam', loss=loss, metrics=['sparse_categorical_accuracy'])

        trainer = TabMTrainer(model, validation_split=0.2)
        training_results = trainer.train(
            X_num_train, X_cat_train, y_train,
            batch_size=128, epochs=50, verbose=0
        )

        if arch == 'plain':
            X_combined_test = np.concatenate([X_num_test, X_cat_test], axis=1)
            test_pred = model.predict(X_combined_test, verbose=0).squeeze(axis=1)
        else:
            test_pred = trainer.predict({'x_num': X_num_test, 'x_cat': X_cat_test})

        test_accuracy = accuracy_score(y_test, np.argmax(test_pred, axis=-1))
        results[arch] = {
            'model': model, 'trainer': trainer,
            'test_accuracy': test_accuracy, 'history': training_results['history']
        }
        logger.info(f"{arch} test accuracy: {test_accuracy:.4f}")

        if arch != 'plain' and 'individual_scores' in training_results:
            logger.info(f"Ensemble scores: {training_results['individual_scores']}")
            logger.info(f"Best ensemble score: {training_results['ensemble_score']:.4f}")

    return results


def example_real_dataset():
    """Example with breast cancer dataset."""
    logger.info("\n=== Breast Cancer Dataset Example ===")

    data = load_breast_cancer()
    X, y = data.data, data.target
    logger.info(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

    model, trainer, results = create_and_train_tabm(
        X, y,
        categorical_columns=[],
        arch_type='tabm', k=6,
        hidden_dims=[64, 32],
        epochs=100, validation_split=0.2, test_split=0.2,
        verbose=1
    )
    logger.info(f"Results: {results}")
    return model, trainer, results


def example_regression():
    """Example with regression data."""
    logger.info("\n=== Regression Example ===")

    X, y = make_regression(
        n_samples=1500, n_features=12, n_informative=8,
        noise=0.1, random_state=42
    )

    categorical_indices = [0, 1]
    for idx in categorical_indices:
        X[:, idx] = pd.cut(X[:, idx], bins=4, labels=False)

    model, trainer, results = create_and_train_tabm(
        X, y,
        categorical_columns=categorical_indices,
        arch_type='tabm-mini', k=4,
        hidden_dims=[96, 48],
        epochs=80, validation_split=0.15, test_split=0.2,
        verbose=1
    )
    return model, trainer, results


def example_ensemble_analysis():
    """Example showing detailed ensemble analysis."""
    logger.info("\n=== Ensemble Analysis Example ===")

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = create_tabm_ensemble(
        n_num_features=10, cat_cardinalities=[], n_classes=2,
        k=10, hidden_dims=[64, 32]
    )
    model.compile(
        optimizer='adam',
        loss=TabMLoss('binary_crossentropy'),
        metrics=['binary_accuracy']
    )

    trainer = TabMTrainer(model, validation_split=0.2, ensemble_selection=True)
    results = trainer.train(X_train, None, y_train, batch_size=64, epochs=50, verbose=1)

    if 'individual_scores' in results:
        individual_scores = results['individual_scores']
        selected_members = results['selected_members']

        logger.info(f"Individual scores: {individual_scores}")
        logger.info(f"Selected members: {selected_members}")
        logger.info(f"Best single: {results['best_member_score']:.4f}, Ensemble: {results['ensemble_score']:.4f}")

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

    np.random.seed(42)
    n_samples = 2000

    num_features = np.random.randn(n_samples, 8)
    cat1 = np.random.randint(0, 5, n_samples)
    cat2 = np.random.randint(0, 3, n_samples)
    cat3 = np.random.randint(0, 10, n_samples)
    X = np.column_stack([num_features, cat1, cat2, cat3])

    y = (2 * num_features[:, 0] + num_features[:, 1] - 0.5 * num_features[:, 2]
         + 0.3 * cat1 + np.random.normal(0, 0.5, n_samples))
    y = (y > np.median(y)).astype(int)

    logger.info(f"Dataset: {X.shape}, 8 numerical + 3 categorical, classes: {np.bincount(y)}")

    processor = TabularDataProcessor(
        categorical_columns=[8, 9, 10], scale_numerical=True, handle_missing='mean'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_num_train, X_cat_train = processor.fit_transform(X_train)
    X_num_test, X_cat_test = processor.transform(X_test)

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

        loss = 'binary_crossentropy' if variant['arch_type'] == 'plain' else TabMLoss('binary_crossentropy')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss, metrics=['binary_accuracy']
        )

        trainer = TabMTrainer(model, validation_split=0.15)
        training_results = trainer.train(
            X_num_train, X_cat_train, y_train,
            batch_size=128, epochs=60, verbose=0
        )

        if variant['arch_type'] == 'plain':
            X_combined_test = np.concatenate([X_num_test, X_cat_test], axis=1)
            test_pred = model.predict(X_combined_test, verbose=0).squeeze()
        else:
            test_pred = trainer.predict({'x_num': X_num_test, 'x_cat': X_cat_test})

        if test_pred.ndim > 1 and test_pred.shape[-1] > 1:
            test_pred_proba = test_pred[:, 1]
            test_pred_class = np.argmax(test_pred, axis=-1)
        else:
            test_pred_proba = test_pred.squeeze()
            test_pred_class = (test_pred_proba > 0.5).astype(int)

        test_accuracy = accuracy_score(y_test, test_pred_class)
        test_auc = roc_auc_score(y_test, test_pred_proba)

        results_comparison[variant['name']] = {
            'accuracy': test_accuracy, 'auc': test_auc,
            'model': model, 'trainer': trainer
        }
        logger.info(f"{variant['name']} - Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")

    logger.info("\n=== Results Summary ===")
    for name, result in results_comparison.items():
        logger.info(f"{name:12} - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")

    return results_comparison


def main():
    """Run all examples."""
    setup_gpu()

    logger.info("TabM Model Examples")

    examples = [
        ("synthetic classification", example_synthetic_classification),
        ("real dataset", example_real_dataset),
        ("regression", example_regression),
        ("ensemble analysis", example_ensemble_analysis),
        ("custom pipeline", example_custom_data_pipeline),
    ]

    for name, fn in examples:
        try:
            fn()
        except Exception as e:
            logger.error(f"Error in {name} example: {e}")

    logger.info("All examples completed")


if __name__ == "__main__":
    main()
