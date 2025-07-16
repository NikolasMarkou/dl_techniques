"""
Probabilistic N-BEATS Multi-Task Time Series Forecasting with Uncertainty Quantification

This module implements a comprehensive multi-task learning framework for probabilistic
time series forecasting using Probabilistic N-BEATS models with MDN integration.
The implementation provides uncertainty quantification through mixture density networks
and extensive visualization of prediction intervals.

Key Features:
    - Probabilistic forecasting with uncertainty quantification
    - Multi-task learning across diverse time series patterns
    - Aleatoric and epistemic uncertainty decomposition
    - Prediction intervals with configurable confidence levels
    - Comprehensive evaluation with probabilistic metrics
    - Extensive visualization including uncertainty bands
    - Scalable data processing with consistent normalization for stable training
"""

import os
import keras
import matplotlib
import dataclasses
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats

# Use a non-interactive backend for saving plots to files
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
# This import assumes the TimeSeriesNormalizer is in this location
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.models.nbeats_probabilistic import ProbabilisticNBeatsNet
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig

# ---------------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------

@dataclass
class ProbabilisticNBeatsConfig:
    """Configuration class for probabilistic N-BEATS forecasting experiments."""
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "probabilistic_nbeats"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    backcast_length: int = 96
    forecast_horizons: List[int] = field(default_factory=lambda: [12, 24])
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic"])
    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"], "generic": ["generic", "generic"]})
    nb_blocks_per_stack: int = 2
    thetas_dim: Dict[str, List[int]] = field(default_factory=lambda: {"interpretable": [3, 6], "generic": [64, 64]})
    hidden_layer_units: int = 128
    share_weights_in_stack: bool = False
    num_mixtures: int = 5
    mdn_hidden_units: int = 128
    aggregation_mode: str = "concat"
    diversity_regularizer_strength: float = 0.02
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 50
    learning_rate: float = 1e-3
    optimizer: str = 'adamw'
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95])
    num_prediction_samples: int = 1000
    epoch_plot_freq: int = 10


@dataclass
class ProbabilisticForecastMetrics:
    """Comprehensive probabilistic forecasting metrics container."""
    task_name: str; task_category: str; model_type: str; horizon: int
    mse: float; rmse: float; mae: float; mape: float
    crps: float; log_likelihood: float
    coverage_68: float; coverage_90: float; coverage_95: float
    interval_width_68: float; interval_width_90: float; interval_width_95: float
    total_uncertainty: float; aleatoric_uncertainty: float; epistemic_uncertainty: float
    mean_num_active_mixtures: float; mixture_entropy: float
    samples_count: int


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class MixtureMonitoringCallback(keras.callbacks.Callback):
    """Callback to monitor mixture weights and detect model collapse."""
    def __init__(self, val_data_sample, processor, task_name, save_dir):
        super().__init__()
        self.val_data_sample, self.processor, self.task_name, self.save_dir = val_data_sample, processor, task_name, save_dir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            try:
                X_sample = self.val_data_sample[:min(10, len(self.val_data_sample))]
                mixture_params = self.model.predict(X_sample, verbose=0)
                _, _, pi_logits = self.model.mdn_layer.split_mixture_params(mixture_params)
                pi_np = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))
                mean_weights, max_w = np.mean(pi_np, axis=0), np.max(pi_np)
                entropy = -np.sum(mean_weights * np.log(mean_weights + 1e-8))
                if (epoch + 1) % 10 == 0: logger.info(f"E{epoch+1}: MixW={np.round(mean_weights,3)},MaxW={max_w:.3f},Entr={entropy:.3f}")
                if max_w > 0.95: logger.warning(f"E{epoch+1}: Potential mixture collapse! Max weight: {max_w:.3f}")
            except Exception as e: logger.warning(f"Mixture monitoring failed at epoch {epoch+1}: {e}")


# In train_probabilistic.py

class ProbabilisticEpochVisualizationCallback(keras.callbacks.Callback):
    """Keras callback for visualizing probabilistic forecasts during training."""

    def __init__(self, val_data_dict, processor, config, model_type, horizon, save_dir):
        super().__init__()
        self.val_data, self.processor, self.config = val_data_dict, processor, config
        self.model_type, self.horizon, self.save_dir = model_type, horizon, save_dir
        self.plot_indices, self.random_state = {}, np.random.RandomState(42)

    def on_train_begin(self, logs=None):
        os.makedirs(self.save_dir, exist_ok=True)
        available_tasks = [name for name, data in self.val_data.items() if len(data[0]) > 0]
        plot_tasks = self.random_state.choice(available_tasks, min(4, len(available_tasks)), replace=False)
        for task in plot_tasks:
            self.plot_indices[task] = self.random_state.randint(0, len(self.val_data[task][0]))
        logger.info(f"Callback will visualize: {list(self.plot_indices.keys())}")

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.config.epoch_plot_freq != 0: return
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        fig.suptitle(f'Epoch {epoch + 1:03d}: Forecasts for {self.model_type.title()} (H={self.horizon})', fontsize=16)
        axes = axes.flatten()

        for i, (task_name, sample_idx) in enumerate(self.plot_indices.items()):
            ax = axes[i]

            # --- START: EXPLICIT AND CORRECT LOGIC ---

            # 1. Get SCALED data for the model from the validation set
            X_val_scaled, y_val_scaled = self.val_data[task_name]
            x_sample_scaled = X_val_scaled[np.newaxis, sample_idx]
            y_true_scaled = y_val_scaled[sample_idx]

            # 2. Get SCALED predictions from the model
            preds = self.model.predict_probabilistic(x_sample_scaled, num_samples=50)
            point_pred_scaled = preds['point_estimate']
            total_var_scaled = preds['total_variance']

            # 3. Inverse transform ALL relevant variables to ORIGINAL scale
            #    This is the only place we do inverse transforms.
            point_pred_orig = self.processor.inverse_transform_data(task_name, point_pred_scaled)
            y_true_orig = self.processor.inverse_transform_data(task_name, y_true_scaled)
            x_input_orig = self.processor.inverse_transform_data(task_name, x_sample_scaled[0])

            # 4. Correctly scale the standard deviation to ORIGINAL units
            scaler = self.processor.scalers[task_name]
            # We need the scale factor (the original std dev) to convert the predicted std dev
            scale_factor = 1.0
            if scaler.method == 'standard' and scaler.std_val is not None:
                scale_factor = scaler.std_val
            elif scaler.method == 'minmax' and scaler.max_val is not None:
                # For minmax, the effective scale is the range of the data
                scale_factor = scaler.max_val - scaler.min_val

            # std_dev_orig = predicted_std_dev_in_scaled_units * original_data_std_dev
            std_dev_orig = np.sqrt(total_var_scaled) * scale_factor

            # 5. Calculate interval bounds using ONLY original-scale data
            point_flat = point_pred_orig.flatten()
            std_flat = std_dev_orig.flatten()
            lower_68 = point_flat - std_flat
            upper_68 = point_flat + std_flat
            lower_95 = point_flat - 1.96 * std_flat
            upper_95 = point_flat + 1.96 * std_flat

            # 6. Plot ONLY original-scale data
            backcast_time = np.arange(-self.config.backcast_length, 0)
            forecast_time = np.arange(self.horizon)

            ax.plot(backcast_time, x_input_orig.flatten(), color='gray', label='Backcast', linewidth=2)
            ax.plot(forecast_time, y_true_orig.flatten(), 'o-', color='blue', label='True Future', linewidth=2)
            ax.plot(forecast_time, point_flat, '--', color='red', label='Point Forecast', linewidth=2)

            ax.fill_between(forecast_time, lower_68, upper_68, alpha=0.3, color='red', label='68% PI')
            ax.fill_between(forecast_time, lower_95, upper_95, alpha=0.2, color='red', label='95% PI')

            ax.set_title(f'Task: {task_name.replace("_", " ").title()}');
            ax.legend();
            ax.grid(True, linestyle='--', alpha=0.5)

            # --- END: EXPLICIT AND CORRECT LOGIC ---

        # Hide unused plots
        for j in range(len(self.plot_indices), 4):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png"), dpi=100)
        plt.close(fig)

# ---------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------

class ProbabilisticNBeatsDataProcessor:
    def __init__(self, config: ProbabilisticNBeatsConfig):
        self.config = config; self.scalers: Dict[str, TimeSeriesNormalizer] = {}
    def create_sequences(self, data, backcast_length, forecast_length):
        X, y = [], []; rng = range(len(data) - backcast_length - forecast_length + 1)
        for i in rng: X.append(data[i:i+backcast_length]); y.append(data[i+backcast_length:i+backcast_length+forecast_length])
        return np.array(X), np.array(y)
    def fit_scalers(self, task_data: Dict[str, np.ndarray]):
        for task, data in task_data.items():
            scaler = TimeSeriesNormalizer(method='minmax')
            scaler.fit(data[:int(self.config.train_ratio * len(data))])
            self.scalers[task] = scaler
    def transform_data(self, task_name, data): return self.scalers[task_name].transform(data)
    def inverse_transform_data(self, task_name, data): return self.scalers[task_name].inverse_transform(data)

# ---------------------------------------------------------------------
# Probabilistic N-BEATS Trainer
# ---------------------------------------------------------------------

class ProbabilisticNBeatsTrainer:
    def __init__(self, config: ProbabilisticNBeatsConfig, ts_config: TimeSeriesConfig):
        self.config, self.ts_config = config, ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = ProbabilisticNBeatsDataProcessor(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()
        self.random_state = np.random.RandomState(42)

    def prepare_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        logger.info("Preparing multi-task data..."); raw_data = {name: self.generator.generate_task_data(name) for name in self.task_names}
        self.processor.fit_scalers(raw_data); prepared_data = {}
        for h in self.config.forecast_horizons:
            prepared_data[h] = {}
            for task, data in raw_data.items():
                train_size, val_size = int(self.config.train_ratio*len(data)), int(self.config.val_ratio*len(data))
                sets = np.split(data, [train_size, train_size + val_size])
                scaled_sets = [self.processor.transform_data(task, s) for s in sets]
                prepared_data[h][task] = {
                    "train": self.processor.create_sequences(scaled_sets[0], self.config.backcast_length, h),
                    "val":   self.processor.create_sequences(scaled_sets[1], self.config.backcast_length, h),
                    "test":  self.processor.create_sequences(scaled_sets[2], self.config.backcast_length, h),
                }
        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> ProbabilisticNBeatsNet:
        return ProbabilisticNBeatsNet(backcast_length=self.config.backcast_length, forecast_length=forecast_length,
            stack_types=self.config.stack_types[model_type], nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=self.config.thetas_dim[model_type], hidden_layer_units=self.config.hidden_layer_units,
            num_mixtures=self.config.num_mixtures, mdn_hidden_units=self.config.mdn_hidden_units,
            aggregation_mode=self.config.aggregation_mode,
            diversity_regularizer_strength=self.config.diversity_regularizer_strength, input_dim=1)

    def train_model(self, model, train_data, val_data, horizon, model_type, exp_dir):
        X_train, y_train = [np.concatenate(d, axis=0) for d in zip(*[v for v in train_data.values() if len(v[0])>0])]
        X_val, y_val = [np.concatenate(d, axis=0) for d in zip(*[v for v in val_data.values() if len(v[0])>0])]
        X_train, y_train = X_train[self.random_state.permutation(len(X_train))], y_train[self.random_state.permutation(len(y_train))]
        X_train += self.random_state.normal(0, 0.05, X_train.shape)
        logger.info(f"Training data: {X_train.shape}, Validation: {X_val.shape}, Augmentation noise: 0.05")
        optimizer = keras.optimizers.AdamW(learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay, clipnorm=self.config.gradient_clip_norm)
        model.compile(optimizer=optimizer, loss=model.mdn_loss)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            keras.callbacks.TerminateOnNaN(),
            ProbabilisticEpochVisualizationCallback(val_data, self.processor, self.config, model_type, horizon, os.path.join(exp_dir, 'visuals', 'epoch_plots', f'{model_type}_h{horizon}')),
            MixtureMonitoringCallback(X_val, self.processor, f"{model_type}_h{horizon}", exp_dir)]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.config.epochs, batch_size=self.config.batch_size, callbacks=callbacks, verbose=2)
        return {"history": history.history, "model": model}

    def evaluate_model(self, model, test_data, horizon, model_type):
        all_task_metrics = {}
        for task, (X_test, y_test) in test_data.items():
            if len(X_test) > 0:
                all_task_metrics[task] = self._calculate_probabilistic_metrics(model, X_test, y_test, task, model_type, horizon)
        return all_task_metrics

    def _calculate_probabilistic_metrics(self, model, X_test, y_test_scaled, task_name, model_type, horizon):
        preds = model.predict_probabilistic(X_test, num_samples=self.config.num_prediction_samples)
        point_scaled, total_var_scaled, aleatoric_var_scaled = preds['point_estimate'], preds['total_variance'], preds['aleatoric_variance']

        scaler = self.processor.scalers[task_name]
        point_orig = self.processor.inverse_transform_data(task_name, point_scaled)
        y_true_orig = self.processor.inverse_transform_data(task_name, y_test_scaled)

        # --- START OF THE FIX ---
        # Ensure y_true_orig has the same shape as point_orig by removing the trailing dimension.
        # From (batch, timesteps, 1) -> (batch, timesteps)
        if y_true_orig.ndim == 3 and y_true_orig.shape[-1] == 1:
            y_true_orig = np.squeeze(y_true_orig, axis=-1)
        # Now, both y_true_orig and point_orig should have shape (643, 12)
        # --- END OF THE FIX ---

        if scaler.method == 'standard' and scaler.std_val is not None:
            scale_sq = scaler.std_val ** 2
        else:
            scale_sq = (scaler.max_val - scaler.min_val) ** 2 if scaler.max_val is not None else 1.0
            if scaler.method != 'standard': logger.warning(f"Variance scaling for '{scaler.method}' may be inaccurate.")

        total_var_orig = total_var_scaled * scale_sq
        aleatoric_var_orig = aleatoric_var_scaled * scale_sq

        errors = y_true_orig - point_orig
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        non_zero_mask = np.abs(y_true_orig) > 1e-8
        mape = np.mean(np.abs(errors[non_zero_mask] / y_true_orig[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0

        mixture_params = model.predict(X_test, verbose=0)
        try:
            log_likelihood = -float(model.mdn_loss(y_test_scaled, mixture_params))
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}")
            log_likelihood = -np.inf

        std_dev_orig = np.sqrt(total_var_orig)
        confidence_intervals = {c: (point_orig - stats.norm.ppf(1-(1-c)/2)*std_dev_orig, point_orig + stats.norm.ppf(1-(1-c)/2)*std_dev_orig) for c in self.config.confidence_levels}
        coverage = {c: np.mean((y_true_orig >= ci[0]) & (y_true_orig <= ci[1])) for c, ci in confidence_intervals.items()}
        widths = {c: np.mean(ci[1] - ci[0]) for c, ci in confidence_intervals.items()}

        crps = self._calculate_crps(model, X_test, y_true_orig, task_name)

        _, _, pi_logits = model.mdn_layer.split_mixture_params(mixture_params)
        pi = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits))
        return ProbabilisticForecastMetrics(
            task_name=task_name, task_category=next((c for c,ts in self.generator.get_task_categories().items() if task_name in ts),""),
            model_type=model_type, horizon=horizon, mse=mse, rmse=np.sqrt(mse), mae=mae, mape=mape, crps=crps,
            log_likelihood=log_likelihood, coverage_68=coverage.get(0.68,0), coverage_90=coverage.get(0.90,0), coverage_95=coverage.get(0.95,0),
            interval_width_68=widths.get(0.68,0), interval_width_90=widths.get(0.90,0), interval_width_95=widths.get(0.95,0),
            total_uncertainty=np.mean(total_var_orig), aleatoric_uncertainty=np.mean(aleatoric_var_orig), epistemic_uncertainty=np.mean(total_var_orig - aleatoric_var_orig),
            mean_num_active_mixtures=np.mean(np.sum(pi>0.1,axis=-1)), mixture_entropy=np.mean(-np.sum(pi*np.log(pi+1e-8),axis=-1)),
            samples_count=y_true_orig.size)

    def _calculate_crps(self, model, X_test, y_true_orig, task_name):
        preds = model.predict_probabilistic(X_test, num_samples=self.config.num_prediction_samples)
        samples_orig = np.zeros_like(preds['samples'])
        for i in range(preds['samples'].shape[1]):
            samples_orig[:,i,:] = self.processor.inverse_transform_data(task_name, preds['samples'][:,i,:])
        term1 = np.mean(np.abs(samples_orig - y_true_orig[:,np.newaxis,:]), axis=1)
        term2 = np.mean(np.abs(samples_orig[:,:,np.newaxis,:] - samples_orig[:,np.newaxis,:,:]), axis=(1,2))
        return np.mean(term1 - 0.5 * term2)

    def plot_probabilistic_forecasts(self, models, prepared_data, save_dir):
        logger.info("Creating probabilistic forecast visualizations..."); plot_dir = os.path.join(save_dir, 'probabilistic_forecasts'); os.makedirs(plot_dir, exist_ok=True)
        for category, tasks in self.generator.get_task_categories().items():
            for h in self.config.forecast_horizons:
                fig, axes = plt.subplots(2,2,figsize=(20,12),squeeze=False); fig.suptitle(f'Forecasts - {category.title()} (H={h})',fontsize=16)
                axes = axes.flatten()
                for i, task in enumerate(self.random_state.choice(tasks,min(4,len(tasks)),replace=False)):
                    ax = axes[i]
                    if task not in prepared_data[h] or len(prepared_data[h][task]["test"][0])==0: continue
                    X_test, y_test = prepared_data[h][task]["test"]; idx = self.random_state.randint(len(X_test))
                    y_o = self.processor.inverse_transform_data(task, y_test[idx]); ax.plot(np.arange(h), y_o.flatten(), 'o-', c='blue', label='True')
                    for m_type, model in models[h].items():
                        preds = model.predict_probabilistic(X_test[np.newaxis, idx])
                        point_o = self.processor.inverse_transform_data(task, preds['point_estimate'])
                        scaler = self.processor.scalers[task]
                        std_o = np.sqrt(preds['total_variance']) * (scaler.std_val if scaler.method=='standard' and scaler.std_val is not None else 1.0)
                        ax.plot(np.arange(h), point_o.flatten(), '--', label=f'{m_type} Pred')
                        ax.fill_between(np.arange(h), (point_o-1.96*std_o).flatten(), (point_o+1.96*std_o).flatten(), alpha=0.2, label=f'{m_type} 95% PI')
                    ax.set_title(task.replace("_"," ").title()); ax.legend(); ax.grid(True, alpha=0.4)
                for j in range(i + 1, 4): axes[j].set_visible(False)
                plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(plot_dir,f'forecasts_{category}_h{h}.png')); plt.close()

    def plot_uncertainty_analysis(self, models, prepared_data, save_dir):
        logger.info("Creating uncertainty analysis visualizations..."); plot_dir = os.path.join(save_dir, 'uncertainty_analysis'); os.makedirs(plot_dir, exist_ok=True)
        for m_type in self.config.model_types:
            for h in self.config.forecast_horizons:
                model = models[h][m_type]; fig, axes = plt.subplots(2,2,figsize=(20,12)); fig.suptitle(f'Uncertainty: {m_type.title()} H={h}',fontsize=16)
                all_total_std, all_aleatoric_std, all_epistemic_std, all_errors = [],[],[],[]
                for task, data in prepared_data[h].items():
                    if len(data["test"][0])==0: continue
                    X_test, y_test = data["test"]; preds = model.predict_probabilistic(X_test)
                    scaler = self.processor.scalers[task]; scale = scaler.std_val if scaler.method=='standard' and scaler.std_val is not None else 1.0
                    point_o, y_o = self.processor.inverse_transform_data(task, preds['point_estimate']), self.processor.inverse_transform_data(task, y_test)
                    all_total_std.append(np.sqrt(preds['total_variance'].flatten())*scale); all_aleatoric_std.append(np.sqrt(preds['aleatoric_variance'].flatten())*scale)
                    all_epistemic_std.append(np.sqrt((preds['total_variance']-preds['aleatoric_variance']).flatten())*scale); all_errors.append(np.abs(y_o - point_o).flatten())
                if not all_total_std: continue
                total_std, aleatoric_std, epistemic_std, errors = map(np.concatenate, [all_total_std, all_aleatoric_std, all_epistemic_std, all_errors])
                axes[0,0].hist(total_std, bins=50, alpha=0.7, density=True, label='Total'); axes[0,0].hist(aleatoric_std, bins=50, alpha=0.7, density=True, label='Aleatoric'); axes[0,0].legend(); axes[0,0].set_title('Uncertainty Distribution')
                axes[0,1].scatter(aleatoric_std, epistemic_std, alpha=0.1); axes[0,1].set_title('Aleatoric vs Epistemic'); axes[0,1].set_xlabel('Aleatoric Std'); axes[0,1].set_ylabel('Epistemic Std')
                idx = self.random_state.choice(len(errors), min(1000, len(errors)), replace=False); axes[1,0].scatter(total_std[idx], errors[idx], alpha=0.1); axes[1,0].set_title('Uncertainty vs. Error'); axes[1,0].set_xlabel('Total Std'); axes[1,0].set_ylabel('Abs Error')
                _,_,pi_logits = model.mdn_layer.split_mixture_params(model.predict(X_test[:100], verbose=0)); axes[1,1].bar(range(self.config.num_mixtures), np.mean(keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits)), axis=0)); axes[1,1].set_title('Avg. Mixture Weights')
                plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(os.path.join(plot_dir, f'uncertainty_{m_type}_h{h}.png')); plt.close()

    def run_experiment(self) -> Dict[str, Any]:
        exp_dir = os.path.join(self.config.result_dir, f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(exp_dir, exist_ok=True); logger.info(f"Starting experiment: {exp_dir}")
        prepared_data = self.prepare_data(); trained_models, all_metrics = {}, {}
        for h in self.config.forecast_horizons:
            trained_models[h], all_metrics[h] = {}, {}
            for m_type in self.config.model_types:
                logger.info(f"\n{'='*60}\nTraining {m_type} model for horizon {h}\n{'='*60}")
                model = self.create_model(m_type, h)
                data_splits = {s: {t: d[s] for t, d in prepared_data[h].items()} for s in ["train", "val", "test"]}
                res = self.train_model(model, data_splits["train"], data_splits["val"], h, m_type, exp_dir)
                trained_models[h][m_type] = res["model"]
                all_metrics[h][m_type] = self.evaluate_model(res["model"], data_splits["test"], h, m_type)
                if self.config.save_results: res["model"].save(os.path.join(exp_dir, f"{m_type}_h{h}.keras"))
        if self.config.save_results:
            visuals_dir = os.path.join(exp_dir, 'visuals')
            self.plot_probabilistic_forecasts(trained_models, prepared_data, visuals_dir)
            self.plot_uncertainty_analysis(trained_models, prepared_data, visuals_dir)
            self._generate_results_summary(all_metrics, exp_dir)
        logger.info(f"Experiment complete. Results saved to: {exp_dir}")
        return {"results_dir": exp_dir, "metrics": all_metrics}

    def _generate_results_summary(self, all_metrics: Dict, exp_dir: str):
        results = [dataclasses.asdict(m) for h in all_metrics.values() for t in h.values() for m in t.values()]
        if not results: return
        df = pd.DataFrame(results)
        summary_cols = ['rmse', 'crps', 'log_likelihood', 'coverage_95', 'interval_width_95', 'total_uncertainty']
        summary = df.groupby(['model_type', 'horizon'])[summary_cols].mean().round(4)
        logger.info("\n" + "="*80 + "\nSUMMARY BY MODEL AND HORIZON\n" + "="*80 + f"\n{summary}")
        df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
        summary.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))

# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main():
    """Main function to run the Probabilistic N-BEATS experiment."""
    config = ProbabilisticNBeatsConfig(
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        early_stopping_patience=50,
        diversity_regularizer_strength=0.01
    )
    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=42,
        default_noise_level=0.01
    )
    try:
        trainer = ProbabilisticNBeatsTrainer(config, ts_config)
        trainer.run_experiment()
        logger.info("Experiment finished successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()