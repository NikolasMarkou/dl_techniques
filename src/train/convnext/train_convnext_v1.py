"""Train ConvNeXt V1 model with ImageNet support using TensorFlow Datasets."""

import os
import keras
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1, create_convnext_v1

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig


# ---------------------------------------------------------------------

def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


# ---------------------------------------------------------------------

def load_imagenet_dataset(
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        data_dir: Optional[str] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[int, int, int], int]:
    """
    Load ImageNet dataset using TensorFlow Datasets.

    Parameters
    ----------
    image_size : Tuple[int, int], optional
        Target image size (height, width), by default (224, 224)
    batch_size : int, optional
        Batch size, by default 32
    shuffle_buffer_size : int, optional
        Buffer size for shuffling, by default 10000
    cache : bool, optional
        Whether to cache the dataset in memory, by default False
    data_dir : Optional[str], optional
        Directory to download/load data from, by default None

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[int, int, int], int]
        Training dataset, validation dataset, input shape, number of classes
    """
    logger.info("Loading ImageNet dataset from TensorFlow Datasets...")

    # Load datasets
    train_ds, train_info = tfds.load(
        "imagenet2012",
        split="train",
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
    )

    val_ds = tfds.load(
        "imagenet2012",
        split="validation",
        as_supervised=True,
        data_dir=data_dir,
    )

    num_classes = train_info.features['label'].num_classes
    input_shape = (image_size[0], image_size[1], 3)

    def preprocess_train(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess training image with data augmentation."""
        # Random crop and resize
        image = tf.image.resize(image, (int(image_size[0] * 1.15), int(image_size[1] * 1.15)))
        image = tf.image.random_crop(image, [image_size[0], image_size[1], 3])

        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    def preprocess_val(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess validation image."""
        # Center crop
        image = tf.image.resize(image, (int(image_size[0] * 1.15), int(image_size[1] * 1.15)))
        h, w = image_size
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=(tf.shape(image)[0] - h) // 2,
            offset_width=(tf.shape(image)[1] - w) // 2,
            target_height=h,
            target_width=w
        )

        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    # Preprocess and batch training data
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Preprocess and batch validation data
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Get dataset sizes
    train_size = train_info.splits['train'].num_examples
    val_size = train_info.splits['validation'].num_examples

    logger.info(f"ImageNet dataset loaded: {train_size} train, {val_size} validation samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return train_ds, val_ds, input_shape, num_classes


# ---------------------------------------------------------------------

def load_dataset(
        dataset_name: str,
        batch_size: int = 32,
        image_size: Optional[Tuple[int, int]] = None
) -> Tuple[Any, Any, Tuple[int, int, int], int]:
    """
    Load and preprocess dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load
    batch_size : int, optional
        Batch size, by default 32
    image_size : Optional[Tuple[int, int]], optional
        Target image size for ImageNet, by default None

    Returns
    -------
    Tuple[Any, Any, Tuple[int, int, int], int]
        Training data, test/validation data, input shape, number of classes
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == 'imagenet':
        if image_size is None:
            image_size = (224, 224)
        train_ds, val_ds, input_shape, num_classes = load_imagenet_dataset(
            image_size=image_size,
            batch_size=batch_size,
        )
        return train_ds, val_ds, input_shape, num_classes

    elif dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Convert grayscale to RGB
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        input_shape = (28, 28, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 100

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# ---------------------------------------------------------------------

def get_class_names(dataset: str, num_classes: int) -> List[str]:
    """Get class names for the dataset."""
    if dataset.lower() == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset.lower() == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset.lower() == 'cifar100':
        return [f'class_{i}' for i in range(num_classes)]
    elif dataset.lower() == 'imagenet':
        # ImageNet has 1000 classes - use generic names or load from TFDS
        try:
            info = tfds.builder('imagenet2012').info
            return info.features['label'].names
        except:
            return [f'class_{i}' for i in range(num_classes)]
    else:
        return [f'class_{i}' for i in range(num_classes)]


# ---------------------------------------------------------------------

def create_model_config(
        dataset: str,
        variant: str,
        input_shape: Tuple[int, int, int],
        num_classes: int
) -> Dict[str, Any]:
    """Create ConvNeXt V1 model configuration based on dataset."""

    config = {
        'include_top': True,
        'kernel_regularizer': None,
    }

    if dataset.lower() == 'imagenet':
        # Standard ImageNet configuration
        config.update({
            'drop_path_rate': 0.1 if variant in ['tiny', 'small'] else 0.2,
            'dropout_rate': 0.0,  # Modern practice: no dropout in final layer
            'use_gamma': True,
        })
    elif dataset.lower() == 'mnist':
        config.update({
            'drop_path_rate': 0.1,
            'dropout_rate': 0.1,
            'use_gamma': True,
        })
    elif dataset.lower() in ['cifar10', 'cifar100']:
        config.update({
            'drop_path_rate': 0.1 if num_classes == 10 else 0.2,
            'dropout_rate': 0.1 if num_classes == 10 else 0.2,
            'use_gamma': True,
        })
    else:
        config.update({
            'drop_path_rate': 0.1,
            'dropout_rate': 0.1,
            'use_gamma': True,
        })

    return config


# ---------------------------------------------------------------------

def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        steps_per_epoch: Optional[int] = None
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule."""
    if schedule_type == 'cosine':
        decay_steps = total_epochs if steps_per_epoch is None else total_epochs * steps_per_epoch
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.01
        )
    elif schedule_type == 'exponential':
        decay_steps = (total_epochs // 4) if steps_per_epoch is None else (total_epochs // 4) * steps_per_epoch
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.9
        )
    else:  # constant
        return initial_lr


# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        monitor: str = 'val_accuracy',
        patience: int = 15,
        use_lr_schedule: bool = True
) -> Tuple[List, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"convnext_v1_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
    ]

    if not use_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )

    return callbacks, results_dir


# ---------------------------------------------------------------------

def convert_keras_history_to_training_history(history: keras.callbacks.History) -> TrainingHistory:
    """Convert Keras history to TrainingHistory object."""
    return TrainingHistory(
        epochs=list(range(1, len(history.history['loss']) + 1)),
        train_loss=history.history['loss'],
        val_loss=history.history.get('val_loss', []),
        train_metrics={'accuracy': history.history.get('accuracy', [])},
        val_metrics={'accuracy': history.history.get('val_accuracy', [])},
        learning_rates=history.history.get('lr', [])
    )


# ---------------------------------------------------------------------

def create_classification_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        model_name: str
) -> ClassificationResults:
    """Create ClassificationResults object."""
    return ClassificationResults(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name
    )


# ---------------------------------------------------------------------

def generate_comprehensive_visualizations(
        viz_manager: VisualizationManager,
        training_history: TrainingHistory,
        classification_results: ClassificationResults,
        model: keras.Model,
        show_plots: bool = False
) -> None:
    """Generate comprehensive visualizations."""
    logger.info("Generating comprehensive visualizations...")

    # Training curves
    logger.info("  - Training curves")
    training_viz = TrainingCurvesVisualization(
        plot_config=PlotConfig(
            style=PlotStyle.PROFESSIONAL,
            color_scheme=ColorScheme.VIBRANT
        )
    )
    training_viz.plot(training_history, show=show_plots)
    viz_manager.save_visualization(training_viz, "training_curves")

    # Confusion matrix
    logger.info("  - Confusion matrix")
    cm_viz = ConfusionMatrixVisualization(
        plot_config=PlotConfig(
            style=PlotStyle.PROFESSIONAL,
            color_scheme=ColorScheme.VIBRANT
        )
    )
    cm_viz.plot(classification_results, show=show_plots)
    viz_manager.save_visualization(cm_viz, "confusion_matrix")

    # ROC and PR curves (only for binary or small multiclass)
    if len(classification_results.class_names) <= 10:
        logger.info("  - ROC and PR curves")
        roc_pr_viz = ROCPRCurves(
            plot_config=PlotConfig(
                style=PlotStyle.PROFESSIONAL,
                color_scheme=ColorScheme.VIBRANT
            )
        )
        roc_pr_viz.plot(classification_results, show=show_plots)
        viz_manager.save_visualization(roc_pr_viz, "roc_pr_curves")

    # Network architecture
    logger.info("  - Network architecture")
    arch_viz = NetworkArchitectureVisualization(
        plot_config=PlotConfig(
            style=PlotStyle.CLEAN
        )
    )
    arch_viz.plot(model, show=show_plots)
    viz_manager.save_visualization(arch_viz, "architecture")

    logger.info("Visualizations generated successfully")


# ---------------------------------------------------------------------

def run_model_analysis(
        model: keras.Model,
        test_data: Tuple[Any, np.ndarray],
        training_history: keras.callbacks.History,
        model_name: str,
        results_dir: str
) -> Any:
    """Run comprehensive model analysis."""
    logger.info("Running model analysis...")

    analysis_config = AnalysisConfig(
        analyze_gradients=False,  # Skip gradient analysis for speed
        analyze_activations=False,
        analyze_weights=True,
        analyze_performance=True,
        generate_plots=True,
        save_results=True
    )

    analyzer = ModelAnalyzer(
        output_dir=os.path.join(results_dir, "model_analysis"),
        config=analysis_config
    )

    try:
        # For ImageNet (tf.data.Dataset), extract subset for analysis
        if isinstance(test_data[0], tf.data.Dataset):
            logger.info("Extracting subset from tf.data.Dataset for analysis...")
            # Take first batch for quick analysis
            for batch_images, batch_labels in test_data[0].take(1):
                x_test_subset = batch_images.numpy()
                y_test_subset = batch_labels.numpy()
                break
        else:
            x_test_subset = test_data[0][:1000]
            y_test_subset = test_data[1][:1000]

        analysis_results = analyzer.analyze_model(
            model=model,
            model_name=model_name,
            test_data=(x_test_subset, y_test_subset),
            history=training_history.history
        )

        logger.info("Model analysis completed successfully")
        return analysis_results

    except Exception as e:
        logger.warning(f"Model analysis failed: {e}")
        return None


# ---------------------------------------------------------------------

def validate_model_loading(
        model_path: str,
        test_sample: Any,
        expected_output: np.ndarray,
        tolerance: float = 1e-5
) -> bool:
    """Validate that a saved model loads correctly and produces expected outputs."""
    try:
        custom_objects = {"ConvNeXtV1": ConvNeXtV1, "ConvNextV1Block": ConvNextV1Block}
        loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Convert test_sample if it's a tensor
        if isinstance(test_sample, tf.Tensor):
            test_sample = test_sample.numpy()

        loaded_output = loaded_model.predict(test_sample, verbose=0)

        # Compare outputs
        if np.allclose(expected_output, loaded_output, atol=tolerance):
            logger.info("Model loading validation passed")
            return True
        else:
            logger.warning("Model outputs differ after loading")
            return False

    except Exception as e:
        logger.warning(f"Model loading validation failed: {e}")
        return False


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    """Train ConvNeXt V1 model."""
    setup_gpu()

    # Determine image size for ImageNet
    image_size = None
    if args.dataset.lower() == 'imagenet':
        image_size = (args.image_size, args.image_size)

    # Load dataset
    train_data, test_data, input_shape, num_classes = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        image_size=image_size
    )

    # Get class names
    class_names = get_class_names(args.dataset, num_classes)

    # Create model
    logger.info(f"Creating ConvNeXt V1 model (variant: {args.variant})...")
    model_config = create_model_config(args.dataset, args.variant, input_shape, num_classes)

    model = create_convnext_v1(
        variant=args.variant,
        input_shape=input_shape,
        num_classes=num_classes,
        kernel_size=args.kernel_size,
        strides=args.strides,
        **model_config
    )

    # Calculate steps per epoch for ImageNet
    steps_per_epoch = None
    if args.dataset.lower() == 'imagenet':
        # ImageNet has ~1.28M training images
        steps_per_epoch = 1281167 // args.batch_size

    # Create learning rate schedule
    use_lr_schedule = args.lr_schedule != 'constant'
    if use_lr_schedule:
        lr = create_learning_rate_schedule(
            initial_lr=args.learning_rate,
            schedule_type=args.lr_schedule,
            total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch
        )
    else:
        lr = args.learning_rate

    # Compile model
    logger.info("Compiling model...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=args.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.variant}_{args.dataset}",
        monitor='val_accuracy',
        patience=args.patience,
        use_lr_schedule=use_lr_schedule
    )

    # Initialize visualization manager
    viz_manager = VisualizationManager(
        experiment_name="visualizations", output_dir=os.path.join(results_dir))

    # Train model
    logger.info("\nStarting training...")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Variant: {args.variant}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Initial learning rate: {args.learning_rate}")
    logger.info(f"  LR schedule: {args.lr_schedule}")
    logger.info(f"  Weight decay: {args.weight_decay}")

    # Handle different data types (numpy arrays vs tf.data.Dataset)
    is_imagenet = args.dataset.lower() == 'imagenet'

    if is_imagenet:
        # For ImageNet, train_data and test_data are tf.data.Dataset
        history = model.fit(
            train_data,
            epochs=args.epochs,
            validation_data=test_data,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # For other datasets, unpack numpy arrays
        x_train, y_train = train_data
        x_test, y_test = test_data

        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

    # Validate model before loading for evaluation
    logger.info("Validating model serialization...")
    if is_imagenet:
        # Get a small sample from ImageNet
        for test_sample, _ in test_data.take(1):
            test_sample = test_sample[:4]
            break
    else:
        test_sample = x_test[:4]

    pre_save_output = model.predict(test_sample, verbose=0)

    # Save final model
    final_model_path = os.path.join(results_dir, f"convnext_v1_{args.dataset}_{args.variant}_final.keras")
    try:
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        is_loading_valid = validate_model_loading(final_model_path, test_sample, pre_save_output)
        if not is_loading_valid:
            logger.warning("Model loading validation failed - using current model for evaluation")

    except Exception as e:
        logger.warning(f"Failed to save final model: {e}")
        final_model_path = None

    # Load best model for evaluation
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        try:
            is_loading_valid = validate_model_loading(best_model_path, test_sample, pre_save_output)

            if is_loading_valid:
                custom_objects = {"ConvNeXtV1": ConvNeXtV1, "ConvNextV1Block": ConvNextV1Block}
                best_model = keras.models.load_model(best_model_path, custom_objects=custom_objects)
                logger.info("Successfully loaded best model from checkpoint")
            else:
                logger.warning("Best model loading validation failed - using current model")
                best_model = model

        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}")
            logger.warning("Using the current model state instead.")
            best_model = model
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    # Final evaluation
    logger.info("Evaluating final model on test set...")
    if is_imagenet:
        test_results = best_model.evaluate(test_data, verbose=1, return_dict=True)
    else:
        test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Generate predictions for visualization (limited for ImageNet)
    logger.info("Generating predictions for visualization...")

    if is_imagenet:
        # For ImageNet, only evaluate on a subset due to memory constraints
        logger.info("Note: Using validation subset for detailed analysis due to dataset size")

        # Collect predictions on subset
        y_true_list = []
        y_pred_list = []
        y_prob_list = []

        max_samples = min(10000, 50000)  # Limit to 10k samples
        sample_count = 0

        for images, labels in test_data:
            if sample_count >= max_samples:
                break

            predictions = best_model.predict(images, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)

            y_true_list.append(labels.numpy())
            y_pred_list.append(pred_classes)
            y_prob_list.append(predictions)

            sample_count += len(labels)

        y_test_subset = np.concatenate(y_true_list)
        predicted_classes = np.concatenate(y_pred_list)
        all_predictions = np.concatenate(y_prob_list)

        accuracy = np.mean(predicted_classes == y_test_subset)
        logger.info(f"Subset accuracy: {accuracy:.4f} (on {len(y_test_subset)} samples)")

    else:
        all_predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=1)
        predicted_classes = np.argmax(all_predictions, axis=1)
        y_test_subset = y_test
        accuracy = np.mean(predicted_classes == y_test)
        logger.info(f"Manually calculated accuracy: {accuracy:.4f}")

    # Convert training history for visualization
    training_history_viz = convert_keras_history_to_training_history(history)

    # Create classification results
    classification_results = create_classification_results(
        y_true=y_test_subset,
        y_pred=predicted_classes,
        y_prob=all_predictions,
        class_names=class_names if not is_imagenet else class_names[:10],  # Limit class names for ImageNet
        model_name=f"{args.variant}_{args.dataset}"
    )

    # Generate comprehensive visualizations
    generate_comprehensive_visualizations(
        viz_manager=viz_manager,
        training_history=training_history_viz,
        classification_results=classification_results,
        model=best_model,
        show_plots=args.show_plots
    )

    # Run model analysis (on subset for ImageNet)
    if is_imagenet:
        analysis_test_data = (test_data, y_test_subset)
    else:
        analysis_test_data = (x_test, y_test)

    analysis_results = run_model_analysis(
        model=best_model,
        test_data=analysis_test_data,
        training_history=history,
        model_name=f"convnext_v1_{args.variant}_{args.dataset}",
        results_dir=results_dir
    )

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ConvNeXt V1 Training Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n")

        if hasattr(model, 'depths'):
            f.write(f"\nModel Configuration:\n")
            f.write(f"  Depths: {model.depths}\n")
            f.write(f"  Dimensions: {model.dims}\n")
            f.write(f"  Drop Path Rate: {model.drop_path_rate}\n")

        f.write(f"\nTraining Configuration:\n")
        f.write(f"  Epochs: {len(history.history['loss'])}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"  LR Schedule: {args.lr_schedule}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")

        f.write(f"\nFinal Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        if not is_imagenet:
            f.write(f"\nManually Calculated Accuracy: {accuracy:.4f}\n")
        else:
            f.write(f"\nSubset Accuracy: {accuracy:.4f}\n")

        f.write(f"\nTraining History:\n")
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        f.write(f"  Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"  Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n")

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")


# ---------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ConvNeXt V1 model with ImageNet support.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for ImageNet (default: 224)')

    # Model arguments
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['cifar10', 'tiny', 'small', 'base', 'large', 'xlarge'],
                        help='Model variant')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='Depthwise kernel size')
    parser.add_argument('--strides', type=int, default=4,
                        help='Downsampling strides')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (use 128-256 for ImageNet)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate (use 4e-3 for ImageNet)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (use 0.05 for ImageNet)')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'],
                        help='Learning rate schedule')

    # Training control
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # Visualization arguments
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots interactively')

    args = parser.parse_args()

    # Adjust defaults for ImageNet
    if args.dataset.lower() == 'imagenet':
        if args.batch_size == 64:
            logger.info("Note: Using default batch size 64. Consider 128-256 for ImageNet.")
        if args.learning_rate == 1e-3:
            logger.info("Note: Using default lr 1e-3. Consider 4e-3 for ImageNet.")
        if args.weight_decay == 1e-4:
            logger.info("Note: Using default weight decay 1e-4. Consider 0.05 for ImageNet.")

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()