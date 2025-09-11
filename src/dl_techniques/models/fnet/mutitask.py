"""
Multi-Task FNet NLP Model Implementation

This module provides a flexible multi-task learning framework built on top of the FNet model.
It supports various NLP tasks including classification, token classification, regression,
and question answering with shared Fourier-based representations and task-specific heads.

The multi-task approach leverages FNet's efficient Fourier transform token mixing to achieve
better performance with faster training compared to traditional BERT-based multi-task models.
"""

import keras
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .model import FNet


# ---------------------------------------------------------------------
# task types
# ---------------------------------------------------------------------

class TaskType(Enum):
    """Enumeration of supported task types."""
    CLASSIFICATION = "classification"
    TOKEN_CLASSIFICATION = "token_classification"
    REGRESSION = "regression"
    QUESTION_ANSWERING = "question_answering"
    SIMILARITY = "similarity"


# ---------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Configuration for a specific task.

    Args:
        name: Unique identifier for the task
        task_type: Type of NLP task
        num_classes: Number of output classes (for classification)
        dropout_rate: Dropout rate for task-specific head
        loss_weight: Weight for this task's loss in multi-task training
        label_smoothing: Label smoothing parameter (for classification)
    """
    name: str
    task_type: TaskType
    num_classes: Optional[int] = None
    dropout_rate: float = 0.1
    loss_weight: float = 1.0
    label_smoothing: float = 0.0


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TaskHead(keras.layers.Layer):
    """
    Task-specific head for multi-task learning with FNet.

    This layer creates appropriate output layers based on the task type
    and handles task-specific transformations. Designed to work efficiently
    with FNet's Fourier-based representations.

    Args:
        task_config: Configuration for this task
        hidden_size: Hidden dimension size from FNet
        initializer_range: Standard deviation for weight initialization
        **kwargs: Additional keyword arguments
    """

    def __init__(
            self,
            task_config: TaskConfig,
            hidden_size: int,
            initializer_range: float = 0.02,
            **kwargs: Any
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        self.task_config = task_config
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

        # Initialize layers in __init__ - CREATE pattern
        self._build_task_layers()

    def _build_task_layers(self) -> None:
        """Build task-specific layers based on task type."""

        # Common dropout layer
        self.dropout = keras.layers.Dropout(
            self.task_config.dropout_rate,
            name=f"{self.task_config.name}_dropout"
        )

        # Task-specific layers
        if self.task_config.task_type == TaskType.CLASSIFICATION:
            if self.task_config.num_classes is None:
                raise ValueError("num_classes must be specified for classification tasks")

            self.classifier = keras.layers.Dense(
                self.task_config.num_classes,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_classifier"
            )

        elif self.task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
            if self.task_config.num_classes is None:
                raise ValueError("num_classes must be specified for token classification tasks")

            self.token_classifier = keras.layers.Dense(
                self.task_config.num_classes,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_token_classifier"
            )

        elif self.task_config.task_type == TaskType.REGRESSION:
            self.regressor = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_regressor"
            )

        elif self.task_config.task_type == TaskType.QUESTION_ANSWERING:
            # Start and end position classifiers for span-based QA
            self.start_classifier = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_start_classifier"
            )
            self.end_classifier = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_end_classifier"
            )

        elif self.task_config.task_type == TaskType.SIMILARITY:
            # Similarity scoring between sentence pairs
            self.similarity_head = keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.task_config.name}_similarity"
            )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for task head.

        Args:
            inputs: Either pooled output (for sentence-level tasks) or
                   (sequence_output, pooled_output) tuple
            training: Whether in training mode

        Returns:
            Task-specific predictions
        """
        if self.task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
            # Use sequence output for token-level tasks
            if isinstance(inputs, tuple):
                sequence_output, _ = inputs
            else:
                sequence_output = inputs

            hidden_states = self.dropout(sequence_output, training=training)
            return self.token_classifier(hidden_states)

        elif self.task_config.task_type == TaskType.QUESTION_ANSWERING:
            # Use sequence output for QA span prediction
            if isinstance(inputs, tuple):
                sequence_output, _ = inputs
            else:
                sequence_output = inputs

            hidden_states = self.dropout(sequence_output, training=training)
            start_logits = keras.ops.squeeze(self.start_classifier(hidden_states), axis=-1)
            end_logits = keras.ops.squeeze(self.end_classifier(hidden_states), axis=-1)
            return {"start_logits": start_logits, "end_logits": end_logits}

        else:
            # Use pooled output for sentence-level tasks
            if isinstance(inputs, tuple):
                _, pooled_output = inputs
            else:
                pooled_output = inputs

            hidden_states = self.dropout(pooled_output, training=training)

            if self.task_config.task_type == TaskType.CLASSIFICATION:
                return self.classifier(hidden_states)
            elif self.task_config.task_type == TaskType.REGRESSION:
                return self.regressor(hidden_states)
            elif self.task_config.task_type == TaskType.SIMILARITY:
                return self.similarity_head(hidden_states)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "task_config": {
                "name": self.task_config.name,
                "task_type": self.task_config.task_type.value,
                "num_classes": self.task_config.num_classes,
                "dropout_rate": self.task_config.dropout_rate,
                "loss_weight": self.task_config.loss_weight,
                "label_smoothing": self.task_config.label_smoothing,
            },
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TaskHead":
        """Create layer from configuration."""
        task_config_dict = config.pop("task_config")
        task_config = TaskConfig(
            name=task_config_dict["name"],
            task_type=TaskType(task_config_dict["task_type"]),
            num_classes=task_config_dict.get("num_classes"),
            dropout_rate=task_config_dict.get("dropout_rate", 0.1),
            loss_weight=task_config_dict.get("loss_weight", 1.0),
            label_smoothing=task_config_dict.get("label_smoothing", 0.0),
        )
        return cls(task_config=task_config, **config)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskFNet(keras.Model):
    """
    Multi-Task FNet model for simultaneous training on multiple NLP tasks.

    This model uses a shared FNet backbone with task-specific heads for different
    NLP tasks. It supports flexible loss weighting and various training strategies
    while leveraging FNet's efficient Fourier transform token mixing for faster
    training compared to traditional BERT-based multi-task models.

    **Key Advantages over Multi-Task BERT:**
    - **Speed**: ~7x faster training due to Fourier transforms vs. self-attention
    - **Memory**: Lower memory footprint due to parameter-free token mixing
    - **Scalability**: Better performance on longer sequences
    - **Efficiency**: Maintains competitive accuracy with reduced computational cost

    **Architecture Overview:**
    ```
    Input(input_ids, attention_mask, position_ids)
           │
           ▼
    Shared FNet Backbone (Fourier Transform + FFN layers)
           │
           ├─→ Task 1 Head (Classification/Regression/etc.)
           ├─→ Task 2 Head (Token Classification/NER/etc.)
           ├─→ Task 3 Head (Question Answering/etc.)
           └─→ Task N Head (Similarity/etc.)
    ```

    Args:
        fnet_config: Configuration dictionary for the FNet backbone
        task_configs: List of TaskConfig objects defining the tasks
        share_fnet_weights: Whether to share FNet weights across tasks
        gradient_balancing: Strategy for balancing gradients across tasks
        **kwargs: Additional keyword arguments for keras.Model

    Example:
        >>> # Define tasks
        >>> tasks = [
        ...     TaskConfig("sentiment", TaskType.CLASSIFICATION, num_classes=3),
        ...     TaskConfig("ner", TaskType.TOKEN_CLASSIFICATION, num_classes=9),
        ...     TaskConfig("similarity", TaskType.REGRESSION)
        ... ]
        >>>
        >>> # Create model
        >>> model = MultiTaskFNet.from_variant("base", tasks)
        >>>
        >>> # Compile with multiple losses
        >>> model.compile_multitask(
        ...     optimizer="adam",
        ...     learning_rate=2e-5
        ... )
    """

    def __init__(
            self,
            fnet_config: Dict[str, Any],
            task_configs: List[TaskConfig],
            share_fnet_weights: bool = True,
            gradient_balancing: str = "equal",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.fnet_config = fnet_config
        self.task_configs = {task.name: task for task in task_configs}
        self.share_fnet_weights = share_fnet_weights
        self.gradient_balancing = gradient_balancing

        # Create shared FNet backbone - CREATE in __init__
        self.fnet = FNet(**fnet_config, add_pooling_layer=True, name="shared_fnet")

        # Create task-specific heads - CREATE in __init__
        self.task_heads: Dict[str, TaskHead] = {}
        for task_config in task_configs:
            self.task_heads[task_config.name] = TaskHead(
                task_config=task_config,
                hidden_size=fnet_config["hidden_size"],
                initializer_range=fnet_config.get("initializer_range", 0.02)
            )

        # Loss tracking - CREATE in __init__
        self.task_losses: Dict[str, keras.metrics.Mean] = {}
        for task_name in self.task_configs:
            self.task_losses[task_name] = keras.metrics.Mean(name=f"{task_name}_loss")

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
            task_name: Optional[str] = None,
            training: Optional[bool] = None
    ) -> Union[Dict[str, keras.KerasTensor], keras.KerasTensor]:
        """
        Forward pass through the multi-task FNet model.

        Args:
            inputs: Input tensors (input_ids, attention_mask, position_ids)
            task_name: Specific task to run (if None, runs all tasks)
            training: Whether in training mode

        Returns:
            Dictionary of task predictions or single task prediction
        """
        # Get FNet representations
        fnet_outputs = self.fnet(inputs, training=training)
        sequence_output, pooled_output = fnet_outputs

        # Run task-specific heads
        task_outputs = {}

        if task_name is not None:
            # Single task inference
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")

            task_head = self.task_heads[task_name]
            task_config = self.task_configs[task_name]

            if task_config.task_type in [TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING]:
                inputs_for_head = (sequence_output, pooled_output)
            else:
                inputs_for_head = pooled_output

            return task_head(inputs_for_head, training=training)

        else:
            # Multi-task inference
            for task_name, task_head in self.task_heads.items():
                task_config = self.task_configs[task_name]

                if task_config.task_type in [TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING]:
                    inputs_for_head = (sequence_output, pooled_output)
                else:
                    inputs_for_head = pooled_output

                task_outputs[task_name] = task_head(inputs_for_head, training=training)

            return task_outputs

    def compute_loss(
            self,
            predictions: Dict[str, keras.KerasTensor],
            targets: Dict[str, keras.KerasTensor],
            sample_weight: Optional[Dict[str, keras.KerasTensor]] = None
    ) -> keras.KerasTensor:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            sample_weight: Optional sample weights per task

        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        sample_weight = sample_weight or {}

        for task_name, pred in predictions.items():
            if task_name not in targets:
                continue

            task_config = self.task_configs[task_name]
            target = targets[task_name]
            weight = sample_weight.get(task_name, None)

            # Compute task-specific loss
            if task_config.task_type == TaskType.CLASSIFICATION:
                if task_config.label_smoothing > 0:
                    loss_fn = keras.losses.CategoricalCrossentropy(
                        label_smoothing=task_config.label_smoothing,
                        from_logits=True
                    )
                else:
                    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            elif task_config.task_type == TaskType.TOKEN_CLASSIFICATION:
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            elif task_config.task_type == TaskType.REGRESSION:
                loss_fn = keras.losses.MeanSquaredError()

            elif task_config.task_type == TaskType.QUESTION_ANSWERING:
                # Combine start and end losses
                start_loss = keras.losses.sparse_categorical_crossentropy(
                    target["start_positions"], pred["start_logits"], from_logits=True
                )
                end_loss = keras.losses.sparse_categorical_crossentropy(
                    target["end_positions"], pred["end_logits"], from_logits=True
                )
                task_loss = (start_loss + end_loss) / 2.0

            elif task_config.task_type == TaskType.SIMILARITY:
                loss_fn = keras.losses.BinaryCrossentropy()

            if task_config.task_type != TaskType.QUESTION_ANSWERING:
                task_loss = loss_fn(target, pred, sample_weight=weight)

            # Apply task weight
            weighted_loss = task_loss * task_config.loss_weight
            total_loss += weighted_loss

            # Update metrics
            self.task_losses[task_name].update_state(task_loss)

        return total_loss

    def compile_multitask(
            self,
            optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
            learning_rate: float = 2e-5,
            metrics: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Compile the multi-task model with appropriate optimizer and metrics.

        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate for training
            metrics: Optional metrics per task
        """
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == "adamw":
                opt = keras.optimizers.AdamW(learning_rate=learning_rate)
            else:
                opt = keras.optimizers.get(optimizer)
                opt.learning_rate = learning_rate
        else:
            opt = optimizer

        self.compile(optimizer=opt)

        # Add task-specific metrics
        if metrics:
            for task_name, task_metrics in metrics.items():
                for metric_name in task_metrics:
                    metric = keras.metrics.get(metric_name)
                    metric.name = f"{task_name}_{metric_name}"
                    self.compiled_metrics._metrics.append(metric)

    @classmethod
    def from_variant(
            cls,
            variant: str = "base",
            task_configs: List[TaskConfig] = None,
            **kwargs: Any
    ) -> "MultiTaskFNet":
        """
        Create MultiTaskFNet from an FNet variant.

        Args:
            variant: FNet variant ("base", "large", "small", "tiny")
            task_configs: List of task configurations
            **kwargs: Additional arguments

        Returns:
            MultiTaskFNet instance

        Raises:
            ValueError: If task_configs is None or variant is unknown
        """
        if task_configs is None:
            raise ValueError("task_configs must be provided")

        # Get FNet configuration for the variant
        if variant not in FNet.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        fnet_config = FNet.MODEL_VARIANTS[variant].copy()
        fnet_config.pop("description", None)

        return cls(
            fnet_config=fnet_config,
            task_configs=task_configs,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "fnet_config": self.fnet_config,
            "task_configs": [
                {
                    "name": task.name,
                    "task_type": task.task_type.value,
                    "num_classes": task.num_classes,
                    "dropout_rate": task.dropout_rate,
                    "loss_weight": task.loss_weight,
                    "label_smoothing": task.label_smoothing,
                }
                for task in self.task_configs.values()
            ],
            "share_fnet_weights": self.share_fnet_weights,
            "gradient_balancing": self.gradient_balancing,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiTaskFNet":
        """Create model from configuration."""
        # Reconstruct task_configs
        task_configs_data = config.pop("task_configs")
        task_configs = []
        for task_data in task_configs_data:
            task_config = TaskConfig(
                name=task_data["name"],
                task_type=TaskType(task_data["task_type"]),
                num_classes=task_data.get("num_classes"),
                dropout_rate=task_data.get("dropout_rate", 0.1),
                loss_weight=task_data.get("loss_weight", 1.0),
                label_smoothing=task_data.get("label_smoothing", 0.0),
            )
            task_configs.append(task_config)

        return cls(task_configs=task_configs, **config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional multi-task FNet information."""
        super().summary(**kwargs)

        # Print additional model information
        num_tasks = len(self.task_configs)
        task_types = [config.task_type.value for config in self.task_configs.values()]

        logger.info("Multi-Task FNet Model Configuration:")
        logger.info(f"  - Backbone: FNet-{self.fnet_config.get('hidden_size', 768) // 256 * 'Base' or 'Custom'}")
        logger.info(f"  - Number of tasks: {num_tasks}")
        logger.info(f"  - Task types: {', '.join(set(task_types))}")
        logger.info(f"  - Shared weights: {self.share_fnet_weights}")
        logger.info(f"  - Gradient balancing: {self.gradient_balancing}")
        logger.info(f"  - Efficiency advantage: ~7x faster than Multi-Task BERT")

        # Print task details
        logger.info("  - Task Details:")
        for task_name, task_config in self.task_configs.items():
            logger.info(f"    • {task_name}: {task_config.task_type.value}")
            if task_config.num_classes:
                logger.info(f"      Classes: {task_config.num_classes}")
            logger.info(f"      Loss weight: {task_config.loss_weight}")

# ---------------------------------------------------------------------