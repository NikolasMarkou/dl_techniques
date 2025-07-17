"""
BERT MNLI BandRMS Normalization Experiment - Using Custom BERT Model
===================================================================

This experiment extends the BandRMS normalization study to Natural Language Processing
by using the custom BERT model with configurable normalization and fine-tuning on
the MNLI (Multi-Genre Natural Language Inference) task from GLUE.

This version uses the custom BERT model from dl_techniques.models.custom_bert,
providing a cleaner and more maintainable implementation.

Key features:
- Uses custom BERT model with configurable normalization
- MNLI dataset loading and preprocessing
- Fine-tuning with proper learning rate schedules
- Calibration analysis for NLP classification tasks
- Comprehensive evaluation on matched and mismatched validation sets
"""

import json
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

try:
    import datasets
    import transformers
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# Import custom BERT model components
from dl_techniques.models.custom_bert import (
    CustomBertModel,
    BertConfig,
    create_bert_for_classification,
)


# ---------------------------------------------------------------------
# DATASET LOADING AND PREPROCESSING
# ---------------------------------------------------------------------

@dataclass
class MNLIDataset:
    """Container for MNLI dataset."""
    x_train: Dict[str, np.ndarray]  # Contains input_ids, attention_mask, token_type_ids
    y_train: np.ndarray
    x_validation_matched: Dict[str, np.ndarray]
    y_validation_matched: np.ndarray
    x_validation_mismatched: Dict[str, np.ndarray]
    y_validation_mismatched: np.ndarray
    label_names: List[str]
    tokenizer: Any


def load_and_preprocess_mnli(
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
) -> MNLIDataset:
    """
    Load and preprocess MNLI dataset.

    Args:
        model_name: Name of the pre-trained model to use for tokenization
        max_length: Maximum sequence length
        cache_dir: Directory to cache dataset
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        MNLIDataset containing preprocessed data
    """
    logger.info("📊 Loading MNLI dataset...")

    if not TRANSFORMERS_AVAILABLE:
        logger.warning("🔄 Transformers not available, creating synthetic dataset...")
        return create_synthetic_mnli_dataset(max_length, max_samples or 10000)

    try:
        # Load dataset
        dataset = datasets.load_dataset('glue', 'mnli', cache_dir=cache_dir)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Label mapping
        label_names = ['entailment', 'neutral', 'contradiction']

        def tokenize_function(examples):
            """Tokenize premise and hypothesis pairs."""
            return tokenizer(
                examples['premise'],
                examples['hypothesis'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='np'
            )

        # Tokenize datasets
        logger.info("🔤 Tokenizing datasets...")

        # Training set
        train_dataset = dataset['train']
        if max_samples:
            train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Validation sets
        val_matched = dataset['validation_matched']
        if max_samples:
            val_matched = val_matched.select(range(min(max_samples // 4, len(val_matched))))

        val_matched_tokenized = val_matched.map(
            tokenize_function,
            batched=True,
            remove_columns=val_matched.column_names
        )

        val_mismatched = dataset['validation_mismatched']
        if max_samples:
            val_mismatched = val_mismatched.select(range(min(max_samples // 4, len(val_mismatched))))

        val_mismatched_tokenized = val_mismatched.map(
            tokenize_function,
            batched=True,
            remove_columns=val_mismatched.column_names
        )

        # Convert to arrays
        def extract_features(tokenized_dataset, original_dataset):
            """Extract features and labels from tokenized dataset."""
            features = {
                'input_ids': np.array(tokenized_dataset['input_ids']),
                'attention_mask': np.array(tokenized_dataset['attention_mask']),
                'token_type_ids': np.array(tokenized_dataset['token_type_ids'])
            }

            labels = np.array(original_dataset['label'])
            # Convert to categorical
            labels = keras.utils.to_categorical(labels, num_classes=3)

            return features, labels

        # Extract features and labels
        x_train, y_train = extract_features(train_tokenized, train_dataset)
        x_val_matched, y_val_matched = extract_features(val_matched_tokenized, val_matched)
        x_val_mismatched, y_val_mismatched = extract_features(val_mismatched_tokenized, val_mismatched)

        logger.info("✅ MNLI dataset loaded successfully!")
        logger.info(f"   Training: {len(y_train)} samples")
        logger.info(f"   Validation (matched): {len(y_val_matched)} samples")
        logger.info(f"   Validation (mismatched): {len(y_val_mismatched)} samples")

        return MNLIDataset(
            x_train=x_train,
            y_train=y_train,
            x_validation_matched=x_val_matched,
            y_validation_matched=y_val_matched,
            x_validation_mismatched=x_val_mismatched,
            y_validation_mismatched=y_val_mismatched,
            label_names=label_names,
            tokenizer=tokenizer
        )

    except Exception as e:
        logger.error(f"❌ Failed to load MNLI dataset: {e}")
        logger.info("🔄 Falling back to synthetic dataset...")
        return create_synthetic_mnli_dataset(max_length, max_samples or 10000)


def create_synthetic_mnli_dataset(max_length: int, num_samples: int) -> MNLIDataset:
    """Create synthetic MNLI dataset for testing."""
    logger.info(f"🔧 Creating synthetic MNLI dataset with {num_samples} samples...")

    # Create synthetic tokenizer-like object
    class SyntheticTokenizer:
        def __init__(self):
            self.vocab_size = 30522  # BERT vocab size
            self.pad_token_id = 0
            self.cls_token_id = 101
            self.sep_token_id = 102

    tokenizer = SyntheticTokenizer()

    # Generate synthetic data
    def generate_synthetic_features(n_samples):
        """Generate synthetic tokenized features."""
        # Random input IDs (avoid padding token for most positions)
        input_ids = np.random.randint(1, tokenizer.vocab_size, (n_samples, max_length))

        # Add CLS and SEP tokens
        input_ids[:, 0] = tokenizer.cls_token_id
        sep_positions = np.random.randint(max_length // 2, max_length - 1, n_samples)
        for i, sep_pos in enumerate(sep_positions):
            input_ids[i, sep_pos] = tokenizer.sep_token_id
            input_ids[i, sep_pos + 1:] = tokenizer.pad_token_id

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int32)

        # Token type IDs (0 for first sentence, 1 for second sentence)
        token_type_ids = np.zeros((n_samples, max_length), dtype=np.int32)
        for i, sep_pos in enumerate(sep_positions):
            token_type_ids[i, sep_pos + 1:] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

    # Generate features
    x_train = generate_synthetic_features(num_samples)
    x_val_matched = generate_synthetic_features(num_samples // 4)
    x_val_mismatched = generate_synthetic_features(num_samples // 4)

    # Generate labels (3 classes: entailment, neutral, contradiction)
    y_train = keras.utils.to_categorical(
        np.random.randint(0, 3, num_samples), num_classes=3
    )
    y_val_matched = keras.utils.to_categorical(
        np.random.randint(0, 3, num_samples // 4), num_classes=3
    )
    y_val_mismatched = keras.utils.to_categorical(
        np.random.randint(0, 3, num_samples // 4), num_classes=3
    )

    label_names = ['entailment', 'neutral', 'contradiction']

    return MNLIDataset(
        x_train=x_train,
        y_train=y_train,
        x_validation_matched=x_val_matched,
        y_validation_matched=y_val_matched,
        x_validation_mismatched=x_val_mismatched,
        y_validation_mismatched=y_val_mismatched,
        label_names=label_names,
        tokenizer=tokenizer
    )


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class BertMNLIExperimentConfig:
    """Configuration for BERT MNLI experiment."""

    # Model parameters
    model_name: str = "bert-base-uncased"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    max_length: int = 128

    # Training parameters
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # Normalization variants
    normalization_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'BandRMS_010': lambda: ('band_rms', {'max_band_width': 0.10}),
        'BandRMS_020': lambda: ('band_rms', {'max_band_width': 0.20}),
        'BandRMS_030': lambda: ('band_rms', {'max_band_width': 0.30}),
        'RMSNorm': lambda: ('rms_norm', {}),
        'LayerNorm': lambda: ('layer_norm', {}),
    })

    # Experiment settings
    output_dir: Path = Path("results")
    experiment_name: str = "bert_mnli_bandrms_study"
    random_seed: int = 42
    n_runs: int = 2  # Reduced for computational efficiency

    # Dataset parameters
    max_train_samples: int = 50000  # Limit for computational efficiency
    cache_dir: Optional[str] = None

    # Analysis configuration
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_calibration=True,
        analyze_training_dynamics=True,
        calibration_bins=10,
        save_plots=True,
        verbose=True
    ))


# ---------------------------------------------------------------------
# LEARNING RATE SCHEDULING
# ---------------------------------------------------------------------

def create_bert_learning_rate_schedule(
        config: BertMNLIExperimentConfig,
        num_train_steps: int
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule for BERT fine-tuning."""

    def lr_schedule(step):
        """Linear warmup followed by linear decay."""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(config.warmup_steps, tf.float32)
        num_train_steps_f = tf.cast(num_train_steps, tf.float32)

        # Linear warmup
        warmup_lr = config.learning_rate * (step / warmup_steps)

        # Linear decay
        decay_lr = config.learning_rate * (
                (num_train_steps_f - step) / (num_train_steps_f - warmup_steps)
        )

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    return keras.optimizers.schedules.LambdaCallback(lr_schedule)


# ---------------------------------------------------------------------
# MODEL BUILDING
# ---------------------------------------------------------------------

def build_bert_model(
        config: BertMNLIExperimentConfig,
        norm_type: str,
        norm_params: Dict[str, Any],
        name: str
) -> CustomBertModel:
    """
    Build BERT model with custom normalization using the custom BERT implementation.

    Args:
        config: Experiment configuration
        norm_type: Normalization type
        norm_params: Normalization parameters
        name: Model name

    Returns:
        CustomBertModel instance
    """
    logger.info(f"🏗️ Building BERT model '{name}' with {norm_type} normalization...")

    # Create BERT configuration
    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        type_vocab_size=2,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        activation='gelu',
        layer_norm_epsilon=1e-12,
        initializer_range=0.02,
        use_bias=True,
        num_classes=3  # MNLI has 3 classes
    )

    # Create model using convenience function
    model = create_bert_for_classification(
        num_classes=3,
        normalization_type=norm_type,
        normalization_params=norm_params,
        bert_config=bert_config
    )

    # Estimate number of training steps
    num_train_steps = (config.max_train_samples // config.batch_size) * config.epochs

    # Create learning rate schedule
    lr_schedule = create_bert_learning_rate_schedule(config, num_train_steps)

    # Configure optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )

    # Set model name
    model._name = f"{name}_model"

    return model


# ---------------------------------------------------------------------
# TRAINING UTILITIES
# ---------------------------------------------------------------------

def train_bert_model(
        model: CustomBertModel,
        dataset: MNLIDataset,
        config: BertMNLIExperimentConfig,
        model_name: str,
        run_idx: int
) -> keras.callbacks.History:
    """Train BERT model on MNLI dataset."""

    logger.info(f"🚀 Training {model_name} (Run {run_idx + 1})...")

    # Prepare callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(dataset.x_validation_matched, dataset.y_validation_matched),
        callbacks=callbacks,
        verbose=1
    )

    return history


# ---------------------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------------------

def calculate_run_statistics(results_per_run: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Calculate statistics across multiple runs."""
    statistics = {}

    for model_name, run_results in results_per_run.items():
        if not run_results:
            continue

        statistics[model_name] = {}
        metrics = run_results[0].keys()

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result]

            if values:
                statistics[model_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

    return statistics


# ---------------------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------------------

def run_bert_mnli_experiment(config: BertMNLIExperimentConfig) -> Dict[str, Any]:
    """
    Run the BERT MNLI experiment using the custom BERT model.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing experimental results
    """
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting BERT MNLI BandRMS Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_and_preprocess_mnli(
        model_name=config.model_name,
        max_length=config.max_length,
        cache_dir=config.cache_dir,
        max_samples=config.max_train_samples
    )

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    # Multiple runs for statistical significance
    all_trained_models = {}
    all_histories = {}
    results_per_run = {}

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")

        # Set different seed for each run
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_models = {}
        run_histories = {}

        # Train models for this run
        for norm_name, norm_factory in config.normalization_variants.items():
            logger.info(f"--- Training {norm_name} (Run {run_idx + 1}) ---")

            # Get normalization configuration
            norm_type, norm_params = norm_factory()

            # Build model using custom BERT implementation
            model = build_bert_model(
                config, norm_type, norm_params, f"{norm_name}_run{run_idx}"
            )

            # Log model info for first run
            if run_idx == 0:
                logger.info(f"Model {norm_name} parameters: {model.count_params():,}")

                # Log normalization details
                logger.info(f"   Normalization type: {norm_type}")
                logger.info(f"   Normalization params: {norm_params}")

            # Train model
            history = train_bert_model(model, dataset, config, norm_name, run_idx)

            run_models[norm_name] = model
            run_histories[norm_name] = history.history

            logger.info(f"✅ {norm_name} (Run {run_idx + 1}) training completed!")

        # Evaluate models for this run
        logger.info(f"📊 Evaluating models for run {run_idx + 1}...")

        for norm_name, model in run_models.items():
            try:
                # Evaluate on matched validation set
                matched_results = model.evaluate(
                    dataset.x_validation_matched,
                    dataset.y_validation_matched,
                    verbose=0
                )
                matched_metrics = dict(zip(model.metrics_names, matched_results))

                # Evaluate on mismatched validation set
                mismatched_results = model.evaluate(
                    dataset.x_validation_mismatched,
                    dataset.y_validation_mismatched,
                    verbose=0
                )
                mismatched_metrics = dict(zip(model.metrics_names, mismatched_results))

                # Store results
                if norm_name not in results_per_run:
                    results_per_run[norm_name] = []

                results_per_run[norm_name].append({
                    'matched_accuracy': matched_metrics['accuracy'],
                    'mismatched_accuracy': mismatched_metrics['accuracy'],
                    'matched_loss': matched_metrics['loss'],
                    'mismatched_loss': mismatched_metrics['loss'],
                    'run_idx': run_idx
                })

                logger.info(f"✅ {norm_name} (Run {run_idx + 1}): "
                            f"Matched={matched_metrics['accuracy']:.4f}, "
                            f"Mismatched={mismatched_metrics['accuracy']:.4f}")

            except Exception as e:
                logger.error(f"❌ Error evaluating {norm_name} (Run {run_idx + 1}): {e}")

        # Store models and histories from last run
        if run_idx == config.n_runs - 1:
            all_trained_models = run_models
            all_histories = run_histories

        # Memory cleanup
        del run_models
        tf.keras.backend.clear_session()

    # Calculate statistics
    logger.info("📈 Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    # Model analysis
    logger.info("🔬 Performing comprehensive analysis...")
    model_analysis_results = None

    try:
        # Prepare data for analysis
        analysis_data = DataInput(
            x_train=dataset.x_train,
            y_train=dataset.y_train,
            x_test=dataset.x_validation_matched,
            y_test=dataset.y_validation_matched,
            class_names=dataset.label_names
        )

        analyzer = ModelAnalyzer(
            models=all_trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis",
            training_history=all_histories
        )

        model_analysis_results = analyzer.analyze(data=analysis_data)
        logger.info("✅ Model analysis completed!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}")

    # Generate visualizations
    logger.info("📊 Generating visualizations...")

    # Training history comparison
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='bert_mnli_training_comparison',
        subdir='training_plots',
        title='BERT MNLI Normalization Training Comparison'
    )

    # Performance comparison visualization
    create_bert_performance_comparison(run_statistics, experiment_dir / "visualizations")

    # Compile results
    results = {
        'run_statistics': run_statistics,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': all_trained_models,
        'config': config,
        'experiment_dir': experiment_dir,
        'dataset_info': {
            'train_samples': len(dataset.y_train),
            'val_matched_samples': len(dataset.y_validation_matched),
            'val_mismatched_samples': len(dataset.y_validation_mismatched),
            'label_names': dataset.label_names
        }
    }

    # Save results
    save_bert_results(results, experiment_dir)

    # Print summary
    print_bert_summary(results)

    return results


# ---------------------------------------------------------------------
# VISUALIZATION UTILITIES
# ---------------------------------------------------------------------

def create_bert_performance_comparison(statistics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Create BERT performance comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Setup plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = list(statistics.keys())

        # Matched Accuracy
        ax1 = axes[0, 0]
        matched_acc = [statistics[model]['matched_accuracy']['mean'] for model in models]
        matched_std = [statistics[model]['matched_accuracy']['std'] for model in models]

        bars1 = ax1.bar(models, matched_acc, yerr=matched_std, capsize=5, alpha=0.7)
        ax1.set_title('Matched Validation Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, acc, std in zip(bars1, matched_acc, matched_std):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                     f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        # Mismatched Accuracy
        ax2 = axes[0, 1]
        mismatched_acc = [statistics[model]['mismatched_accuracy']['mean'] for model in models]
        mismatched_std = [statistics[model]['mismatched_accuracy']['std'] for model in models]

        bars2 = ax2.bar(models, mismatched_acc, yerr=mismatched_std, capsize=5, alpha=0.7, color='orange')
        ax2.set_title('Mismatched Validation Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, acc, std in zip(bars2, mismatched_acc, mismatched_std):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                     f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        # Matched Loss
        ax3 = axes[1, 0]
        matched_loss = [statistics[model]['matched_loss']['mean'] for model in models]
        matched_loss_std = [statistics[model]['matched_loss']['std'] for model in models]

        bars3 = ax3.bar(models, matched_loss, yerr=matched_loss_std, capsize=5, alpha=0.7, color='green')
        ax3.set_title('Matched Validation Loss')
        ax3.set_ylabel('Loss')
        ax3.tick_params(axis='x', rotation=45)

        # Robustness (difference between matched and mismatched)
        ax4 = axes[1, 1]
        robustness = [matched_acc[i] - mismatched_acc[i] for i in range(len(models))]
        robustness_std = [np.sqrt(matched_std[i] ** 2 + mismatched_std[i] ** 2) for i in range(len(models))]

        bars4 = ax4.bar(models, robustness, yerr=robustness_std, capsize=5, alpha=0.7, color='red')
        ax4.set_title('Robustness Gap (Matched - Mismatched)')
        ax4.set_ylabel('Accuracy Difference')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Add value labels
        for bar, rob, std in zip(bars4, robustness, robustness_std):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + std + 0.002,
                     f'{rob:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'bert_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ BERT performance comparison plot saved")

    except Exception as e:
        logger.error(f"❌ Failed to create performance comparison plot: {e}")


# ---------------------------------------------------------------------
# RESULTS SAVING AND REPORTING
# ---------------------------------------------------------------------

def save_bert_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """Save BERT experiment results."""
    try:
        # Convert numpy types to Python native types
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj

        # Save configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'model_name': results['config'].model_name,
            'normalization_variants': list(results['config'].normalization_variants.keys()),
            'epochs': results['config'].epochs,
            'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate,
            'n_runs': results['config'].n_runs,
            'max_train_samples': results['config'].max_train_samples,
            'hidden_size': results['config'].hidden_size,
            'num_layers': results['config'].num_layers,
            'num_heads': results['config'].num_heads,
            'dataset_info': results['dataset_info']
        }

        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save statistical results
        statistical_results = convert_numpy_to_python(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(statistical_results, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for name, model in results['trained_models'].items():
            model_path = models_dir / f"{name}.keras"
            model.save(model_path)

        logger.info("✅ BERT experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save BERT results: {e}")


def print_bert_summary(results: Dict[str, Any]) -> None:
    """Print BERT experiment summary."""
    logger.info("=" * 80)
    logger.info("📋 BERT MNLI BANDRMS EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Dataset info
    if 'dataset_info' in results:
        info = results['dataset_info']
        logger.info("📊 DATASET INFORMATION:")
        logger.info(f"   Training samples: {info['train_samples']:,}")
        logger.info(f"   Validation (matched): {info['val_matched_samples']:,}")
        logger.info(f"   Validation (mismatched): {info['val_mismatched_samples']:,}")
        logger.info(f"   Classes: {', '.join(info['label_names'])}")
        logger.info("")

    # Statistical results
    if 'run_statistics' in results:
        logger.info("📊 STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<15} {'Matched Acc':<12} {'Mismatched Acc':<15} {'Robustness':<12} {'Runs':<8}")
        logger.info("-" * 75)

        for model_name, stats in results['run_statistics'].items():
            matched_mean = stats['matched_accuracy']['mean']
            matched_std = stats['matched_accuracy']['std']
            mismatched_mean = stats['mismatched_accuracy']['mean']
            mismatched_std = stats['mismatched_accuracy']['std']
            robustness = matched_mean - mismatched_mean
            n_runs = stats['matched_accuracy']['count']

            logger.info(f"{model_name:<15} {matched_mean:.3f}±{matched_std:.3f}  "
                        f"{mismatched_mean:.3f}±{mismatched_std:.3f}    "
                        f"{robustness:.3f}      {n_runs:<8}")

    # Calibration results
    if results.get('model_analysis') and hasattr(results['model_analysis'], 'calibration_metrics'):
        logger.info("")
        logger.info("🎯 CALIBRATION ANALYSIS:")
        logger.info(f"{'Model':<15} {'ECE':<12} {'Brier':<12} {'Entropy':<12}")
        logger.info("-" * 55)

        for model_name, cal_metrics in results['model_analysis'].calibration_metrics.items():
            if cal_metrics:
                ece = cal_metrics.get('ece', 0.0)
                brier = cal_metrics.get('brier_score', 0.0)
                entropy = cal_metrics.get('mean_entropy', 0.0)

                logger.info(f"{model_name:<15} {ece:<12.4f} {brier:<12.4f} {entropy:<12.4f}")

    # Key insights
    logger.info("")
    logger.info("🔍 KEY INSIGHTS:")

    if 'run_statistics' in results:
        # Best matched accuracy
        best_matched = max(results['run_statistics'].items(),
                           key=lambda x: x[1]['matched_accuracy']['mean'])
        logger.info(f"   🏆 Best Matched Accuracy: {best_matched[0]} "
                    f"({best_matched[1]['matched_accuracy']['mean']:.4f})")

        # Best mismatched accuracy (out-of-domain robustness)
        best_mismatched = max(results['run_statistics'].items(),
                              key=lambda x: x[1]['mismatched_accuracy']['mean'])
        logger.info(f"   🌐 Best Mismatched Accuracy: {best_mismatched[0]} "
                    f"({best_mismatched[1]['mismatched_accuracy']['mean']:.4f})")

        # Most robust model (smallest gap)
        most_robust = min(results['run_statistics'].items(),
                          key=lambda x: x[1]['matched_accuracy']['mean'] - x[1]['mismatched_accuracy']['mean'])
        robustness_gap = most_robust[1]['matched_accuracy']['mean'] - most_robust[1]['mismatched_accuracy']['mean']
        logger.info(f"   🛡️ Most Robust: {most_robust[0]} "
                    f"(gap: {robustness_gap:.4f})")

        # Most stable model
        most_stable = min(results['run_statistics'].items(),
                          key=lambda x: x[1]['matched_accuracy']['std'])
        logger.info(f"   📊 Most Stable: {most_stable[0]} "
                    f"(std: {most_stable[1]['matched_accuracy']['std']:.4f})")

        # BandRMS analysis
        band_rms_models = {k: v for k, v in results['run_statistics'].items() if k.startswith('BandRMS')}
        if band_rms_models:
            logger.info("   📐 BandRMS Analysis:")
            for model_name, stats in band_rms_models.items():
                alpha_str = model_name.split('_')[1]
                alpha_value = float(alpha_str) / 100
                logger.info(f"      α={alpha_value:.1f}: Matched={stats['matched_accuracy']['mean']:.4f} "
                            f"± {stats['matched_accuracy']['std']:.4f}, "
                            f"Robustness={stats['matched_accuracy']['mean'] - stats['mismatched_accuracy']['mean']:.4f}")

    logger.info("=" * 80)


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main() -> None:
    """Main execution function."""
    logger.info("🚀 BERT MNLI BandRMS Normalization Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU configuration: {e}")

    # Initialize configuration
    config = BertMNLIExperimentConfig()

    # Log configuration
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Model: {config.model_name}")
    logger.info(f"   Dataset: MNLI")
    logger.info(f"   Architecture: {config.num_layers} layers, {config.num_heads} heads, {config.hidden_size} hidden")
    logger.info(f"   Max length: {config.max_length}")
    logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"   Learning rate: {config.learning_rate}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Max train samples: {config.max_train_samples}")
    logger.info("")

    try:
        # Run experiment
        results = run_bert_mnli_experiment(config)
        logger.info("✅ BERT MNLI BandRMS experiment completed successfully!")

        # Additional analysis
        if results.get('model_analysis'):
            logger.info("📊 Additional analysis results are available in the model_analysis directory")

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()