"""
Multi-Task BERT Training Script
================================

A comprehensive training script for multi-task learning with BERT on multiple NLP tasks.
Designed to be modular, extensible, and production-ready.

Features:
---------
- Multi-task learning with shared BERT encoder
- Task-specific heads for different NLP tasks
- Support for various datasets from tensorflow-datasets
- Customizable hyperparameters and task configurations
- Per-task metrics tracking and evaluation
- Model checkpointing and resumption
- Gradient accumulation support
- Mixed precision training
- Learning rate scheduling

Usage:
------
.. code-block:: bash

    python train.py --config config.yaml
    python train.py --bert-variant base --epochs 10 --batch-size 32

"""

import json
import copy
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional, Any, Callable


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.bert import BERT
from dl_techniques.layers.nlp_heads import (
    create_nlp_head,
    NLPTaskConfig,
    NLPTaskType
)
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------


@dataclass
class TaskConfiguration:
    """Configuration for a single NLP task.

    :param name: Unique identifier for the task.
    :type name: str
    :param task_type: Type of NLP task (classification, NER, QA, etc.).
    :type task_type: NLPTaskType
    :param dataset_name: Name of the dataset in tensorflow-datasets.
    :type dataset_name: str
    :param num_classes: Number of classes for classification tasks.
    :type num_classes: Optional[int]
    :param max_sequence_length: Maximum sequence length for this task.
    :type max_sequence_length: int
    :param batch_size: Batch size for this task.
    :type batch_size: int
    :param loss_weight: Weight for this task's loss in multi-task training.
    :type loss_weight: float
    :param head_config: Additional configuration for the task head.
    :type head_config: Dict[str, Any]
    :param preprocessing_fn: Name of preprocessing function to use.
    :type preprocessing_fn: Optional[str]
    """
    name: str
    task_type: NLPTaskType
    dataset_name: str
    num_classes: Optional[int] = None
    max_sequence_length: int = 128
    batch_size: int = 32
    loss_weight: float = 1.0
    head_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_fn: Optional[str] = None
    dataset_split_train: str = "train"
    dataset_split_val: str = "validation"
    dataset_split_test: str = "test"


@dataclass
class TrainingConfiguration:
    """Global training configuration.

    :param bert_variant: BERT variant to use ('base', 'large', 'small', 'tiny').
    :type bert_variant: str
    :param epochs: Number of training epochs.
    :type epochs: int
    :param learning_rate: Initial learning rate.
    :type learning_rate: float
    :param warmup_steps: Number of warmup steps for learning rate.
    :type warmup_steps: int
    :param weight_decay: Weight decay for AdamW optimizer.
    :type weight_decay: float
    :param gradient_accumulation_steps: Steps for gradient accumulation.
    :type gradient_accumulation_steps: int
    :param max_grad_norm: Maximum gradient norm for clipping.
    :type max_grad_norm: float
    :param mixed_precision: Whether to use mixed precision training.
    :type mixed_precision: bool
    :param save_dir: Directory to save models and logs.
    :type save_dir: str
    :param checkpoint_every: Save checkpoint every N steps.
    :type checkpoint_every: int
    :param eval_every: Evaluate every N steps.
    :type eval_every: int
    :param logging_steps: Log metrics every N steps.
    :type logging_steps: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :param vocab_size: Vocabulary size for BERT.
    :type vocab_size: int
    """
    bert_variant: str = "base"
    epochs: int = 10
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    save_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    eval_every: int = 500
    logging_steps: int = 100
    seed: int = 42
    vocab_size: int = 30522
    use_task_sampling: bool = True
    task_sampling_temperature: float = 1.0


# ---------------------------------------------------------------------
# Tokenizer (Simple Mock - Replace with Real Tokenizer)
# ---------------------------------------------------------------------


class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes.

    In production, replace this with a proper tokenizer like:
    - transformers.BertTokenizer
    - tokenizers.Tokenizer
    - keras_nlp.tokenizers

    :param vocab_size: Size of vocabulary.
    :type vocab_size: int
    :param max_length: Maximum sequence length.
    :type max_length: int
    """

    def __init__(self, vocab_size: int = 30522, max_length: int = 128) -> None:
        """Initialize the tokenizer.

        :param vocab_size: Vocabulary size.
        :type vocab_size: int
        :param max_length: Maximum sequence length.
        :type max_length: int
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def encode(self, text: str) -> Dict[str, np.ndarray]:
        """Encode text to input IDs.

        :param text: Input text string.
        :type text: str
        :return: Dictionary with input_ids, attention_mask, token_type_ids.
        :rtype: Dict[str, np.ndarray]
        """
        # Simple word-based tokenization (replace with real tokenizer)
        words = text.lower().split()[:self.max_length - 2]

        # Create fake token IDs
        token_ids = [self.cls_token_id] + [
            hash(word) % (self.vocab_size - 103) + 103 for word in words
        ] + [self.sep_token_id]

        # Pad to max_length
        seq_length = len(token_ids)
        attention_mask = [1] * seq_length

        if seq_length < self.max_length:
            padding_length = self.max_length - seq_length
            token_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        return {
            "input_ids": np.array(token_ids, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
            "token_type_ids": np.zeros(self.max_length, dtype=np.int32)
        }

    def encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Encode batch of texts.

        :param texts: List of text strings.
        :type texts: List[str]
        :return: Dictionary with batched arrays.
        :rtype: Dict[str, np.ndarray]
        """
        encoded = [self.encode(text) for text in texts]
        return {
            "input_ids": np.stack([e["input_ids"] for e in encoded]),
            "attention_mask": np.stack([e["attention_mask"] for e in encoded]),
            "token_type_ids": np.stack([e["token_type_ids"] for e in encoded])
        }


# ---------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------


class TaskDataLoader:
    """Data loader for a specific NLP task.

    :param task_config: Configuration for the task.
    :type task_config: TaskConfiguration
    :param tokenizer: Tokenizer instance.
    :type tokenizer: SimpleTokenizer
    """

    def __init__(
            self,
            task_config: TaskConfiguration,
            tokenizer: SimpleTokenizer
    ) -> None:
        """Initialize the data loader.

        :param task_config: Task configuration.
        :type task_config: TaskConfiguration
        :param tokenizer: Tokenizer to use.
        :type tokenizer: SimpleTokenizer
        """
        self.task_config = task_config
        self.tokenizer = tokenizer

    def load_dataset(
            self,
            split: str = "train"
    ) -> tf.data.Dataset:
        """Load and preprocess dataset for the task.

        :param split: Dataset split to load.
        :type split: str
        :return: TensorFlow dataset.
        :rtype: tf.data.Dataset
        """
        logger.info(
            f"Loading dataset '{self.task_config.dataset_name}' "
            f"for task '{self.task_config.name}' (split: {split})"
        )

        try:
            # Load from tensorflow-datasets
            ds = tfds.load(
                self.task_config.dataset_name,
                split=split,
                shuffle_files=(split == "train")
            )
        except Exception as e:
            logger.warning(
                f"Could not load dataset '{self.task_config.dataset_name}': {e}"
            )
            logger.info("Generating synthetic dataset instead")
            ds = self._generate_synthetic_dataset(split)

        # Apply task-specific preprocessing
        ds = ds.map(
            self._get_preprocessing_fn(),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Note: Batching and prefetching is now handled in the trainer
        # to allow for dynamic batch sizes per task before interleaving.
        return ds

    def _get_preprocessing_fn(self) -> Callable:
        """Returns the appropriate preprocessing function wrapped for tf.data."""
        task_type = self.task_config.task_type

        if task_type == NLPTaskType.SENTIMENT_ANALYSIS:
            return self._preprocess_classification
        elif task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
            return self._preprocess_token_classification
        elif task_type == NLPTaskType.QUESTION_ANSWERING:
            return self._preprocess_question_answering
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _py_encode_text(self, text_tensor: tf.Tensor) -> Dict[str, np.ndarray]:
        """Wrapper to run tokenizer's encode method in tf.py_function."""
        text = text_tensor.numpy().decode("utf-8")
        encoded = self.tokenizer.encode(text)
        return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]

    def _preprocess_classification(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess classification example using tf.py_function."""
        text = example.get("text", example.get("sentence", ""))
        label = tf.cast(example.get("label", 0), tf.int32)

        input_ids, attention_mask, token_type_ids = tf.py_function(
            func=self._py_encode_text,
            inp=[text],
            Tout=[tf.int32, tf.int32, tf.int32]
        )

        max_len = self.tokenizer.max_length
        input_ids.set_shape([max_len])
        attention_mask.set_shape([max_len])
        token_type_ids.set_shape([max_len])

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        return inputs, label

    def _preprocess_token_classification(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess token classification example using tf.py_function."""
        tokens = example.get("tokens", tf.constant([], dtype=tf.string))
        ner_tags = example.get("ner_tags", tf.constant([], dtype=tf.int32))

        def _py_process_tokens(tokens_tensor, ner_tags_tensor):
            token_list = [t.decode("utf-8") for t in tokens_tensor.numpy()]
            ner_tag_list = ner_tags_tensor.numpy().tolist()

            text = " ".join(token_list)
            encoded = self.tokenizer.encode(text)

            max_len = self.tokenizer.max_length
            seq_len = min(len(ner_tag_list) + 2, max_len)
            labels = [0] + ner_tag_list[:seq_len - 2] + [0]
            labels += [0] * (max_len - len(labels))

            return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"], np.array(labels,
                                                                                                        dtype=np.int32)

        input_ids, attention_mask, token_type_ids, labels = tf.py_function(
            func=_py_process_tokens,
            inp=[tokens, ner_tags],
            Tout=[tf.int32, tf.int32, tf.int32, tf.int32]
        )

        max_len = self.tokenizer.max_length
        input_ids.set_shape([max_len])
        attention_mask.set_shape([max_len])
        token_type_ids.set_shape([max_len])
        labels.set_shape([max_len])

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        return inputs, labels

    def _preprocess_question_answering(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess QA example using tf.py_function."""
        question = example.get("question", "")
        context = example.get("context", "")
        answers = example.get("answers", {})
        answer_start = answers.get("answer_start", tf.constant([0], dtype=tf.int32))

        def _py_process_qa(question_t, context_t, start_t):
            q = question_t.numpy().decode("utf-8")
            c = context_t.numpy().decode("utf-8")
            start_pos = int(start_t.numpy()[0] if start_t.shape[0] > 0 else 0)

            text = f"{q} {c}"
            encoded = self.tokenizer.encode(text)

            max_len = self.tokenizer.max_length
            start_pos = min(start_pos, max_len - 1)
            end_pos = min(start_pos + 10, max_len - 1)

            return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"], start_pos, end_pos

        input_ids, attention_mask, token_type_ids, start_pos, end_pos = tf.py_function(
            func=_py_process_qa,
            inp=[question, context, answer_start],
            Tout=[tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
        )

        max_len = self.tokenizer.max_length
        input_ids.set_shape([max_len])
        attention_mask.set_shape([max_len])
        token_type_ids.set_shape([max_len])
        start_pos.set_shape([])
        end_pos.set_shape([])

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        labels = {"start_positions": start_pos, "end_positions": end_pos}

        return inputs, labels

    def _generate_synthetic_dataset(self, split: str) -> tf.data.Dataset:
        """Generate synthetic dataset for testing.

        :param split: Dataset split.
        :type split: str
        :return: Synthetic TensorFlow dataset.
        :rtype: tf.data.Dataset
        """
        num_examples = 1000 if split == "train" else 200
        task_type = self.task_config.task_type

        logger.info(f"Generating {num_examples} synthetic examples for {split}")

        if task_type == NLPTaskType.SENTIMENT_ANALYSIS:
            return self._generate_synthetic_classification(num_examples)
        elif task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
            return self._generate_synthetic_token_classification(num_examples)
        elif task_type == NLPTaskType.QUESTION_ANSWERING:
            return self._generate_synthetic_qa(num_examples)
        else:
            raise ValueError(f"Cannot generate synthetic data for {task_type}")

    def _generate_synthetic_classification(
            self,
            num_examples: int
    ) -> tf.data.Dataset:
        """Generate synthetic classification data.

        :param num_examples: Number of examples to generate.
        :type num_examples: int
        :return: Synthetic dataset.
        :rtype: tf.data.Dataset
        """
        texts = [
            f"This is synthetic example {i} for classification"
            for i in range(num_examples)
        ]
        labels = np.random.randint(
            0,
            self.task_config.num_classes,
            size=num_examples
        )

        dataset = tf.data.Dataset.from_tensor_slices({
            "text": texts,
            "label": labels
        })

        return dataset

    def _generate_synthetic_token_classification(
            self,
            num_examples: int
    ) -> tf.data.Dataset:
        """Generate synthetic token classification data.

        :param num_examples: Number of examples to generate.
        :type num_examples: int
        :return: Synthetic dataset.
        :rtype: tf.data.Dataset
        """
        data = []
        for i in range(num_examples):
            tokens = [f"word{j}" for j in range(10)]
            ner_tags = np.random.randint(0, self.task_config.num_classes, size=10)
            data.append({"tokens": tokens, "ner_tags": ner_tags})

        dataset = tf.data.Dataset.from_generator(
            lambda: iter(data),
            output_signature={
                "tokens": tf.TensorSpec(shape=(None,), dtype=tf.string),
                "ner_tags": tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )

        return dataset

    def _generate_synthetic_qa(self, num_examples: int) -> tf.data.Dataset:
        """Generate synthetic QA data.

        :param num_examples: Number of examples to generate.
        :type num_examples: int
        :return: Synthetic dataset.
        :rtype: tf.data.Dataset
        """
        questions = [f"What is example {i}?" for i in range(num_examples)]
        contexts = [
            f"This is context for example {i} with some text"
            for i in range(num_examples)
        ]

        dataset = tf.data.Dataset.from_tensor_slices({
            "question": questions,
            "context": contexts,
            "answers": {
                "answer_start": np.zeros(num_examples, dtype=np.int32)
            }
        })

        return dataset


# ---------------------------------------------------------------------
# Multi-Task Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiTaskBERTModel(keras.Model):
    """Multi-task BERT model with task-specific heads.

    This model is adapted for use with Keras `model.fit()` by including a
    custom `train_step`. This allows for efficient multi-task training where
    loss is only calculated for the active task in each batch.

    :param bert_config: Configuration for BERT encoder.
    :type bert_config: Dict[str, Any]
    :param task_configs: Dictionary of task configurations.
    :type task_configs: Dict[str, TaskConfiguration]
    """

    def __init__(
            self,
            bert_config: Dict[str, Any],
            task_configs: Dict[str, TaskConfiguration],
            **kwargs: Any
    ) -> None:
        """Initialize the multi-task model.

        :param bert_config: BERT configuration dictionary.
        :type bert_config: Dict[str, Any]
        :param task_configs: Task configurations.
        :type task_configs: Dict[str, TaskConfiguration]
        :param kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)

        self.bert_config = bert_config
        self.task_configs = task_configs
        self.task_configs_dict = {
            name: asdict(config) for name, config in task_configs.items()
        }

        # Create shared BERT encoder
        self.bert_encoder = BERT(**bert_config)

        # Create task-specific heads
        self.task_heads: Dict[str, keras.Model] = {}
        for task_name, task_config in task_configs.items():
            self._build_task_head(task_name, task_config)

        # Loss functions used in the custom train_step
        self.loss_fns: Dict[str, Callable] = {}
        for task_name, task_config in self.task_configs.items():
            self.loss_fns[task_name] = self._get_loss_function(task_config)

        logger.info(
            f"Created multi-task model with {len(self.task_heads)} tasks: "
            f"{list(self.task_heads.keys())}"
        )

    def _build_task_head(
            self,
            task_name: str,
            task_config: TaskConfiguration
    ) -> None:
        """Build a task-specific head.

        :param task_name: Name of the task.
        :type task_name: str
        :param task_config: Task configuration.
        :type task_config: TaskConfiguration
        """
        nlp_task_config_fields = {f.name for f in fields(NLPTaskConfig)}
        nlp_task_constructor_args = {
            k: v for k, v in task_config.head_config.items()
            if k in nlp_task_config_fields
        }
        nlp_task_config = NLPTaskConfig(
            name=task_config.name,
            task_type=task_config.task_type,
            num_classes=task_config.num_classes,
            **nlp_task_constructor_args
        )
        head_constructor_args = {
            k: v for k, v in task_config.head_config.items()
            if k not in nlp_task_config_fields
        }
        head = create_nlp_head(
            task_config=nlp_task_config,
            input_dim=self.bert_config["hidden_size"],
            **head_constructor_args
        )
        self.task_heads[task_name] = head
        logger.info(f"Created head for task '{task_name}': {task_config.task_type}")

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            task_name: Optional[str] = None,
            training: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Forward pass through the model.

        :param inputs: Input dictionary with input_ids, attention_mask, etc.
        :type inputs: Dict[str, keras.KerasTensor]
        :param task_name: Specific task to run (None for all tasks).
        :type task_name: Optional[str]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Dictionary of outputs per task.
        :rtype: Dict[str, Any]
        """
        bert_outputs = self.bert_encoder(inputs, training=training)
        hidden_states = bert_outputs["last_hidden_state"]
        attention_mask = bert_outputs.get("attention_mask")

        head_inputs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask
        }

        outputs = {}
        if task_name is not None:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")
            outputs[task_name] = self.task_heads[task_name](
                head_inputs,
                training=training
            )
        else:
            for name, head in self.task_heads.items():
                outputs[name] = head(head_inputs, training=training)

        return outputs

    def train_step(self, data: tuple) -> Dict[str, tf.Tensor]:
        """Custom train step for multi-task learning.

        This method overrides the default Keras `train_step`. It expects data
        in the format `(x, y)`, where `x` is a dictionary of inputs that
        includes a special 'task_name' key, and `y` is a dictionary of labels
        for the active task.

        :param data: A tuple of (inputs, labels).
        :type data: tuple
        :return: A dictionary of metrics.
        :rtype: Dict[str, tf.Tensor]
        """
        x, y = data
        task_name_tensor = x.pop("task_name")
        task_name = tf.constant(list(self.task_configs.keys())[0], dtype=tf.string)

        # Get the Python string value of the task name for this batch.
        # This is needed because we can't use a symbolic tensor to index
        # Python dictionaries (like self.task_heads).
        # We use tf.switch_case to branch execution based on the tensor value.

        # Define a function for each possible task
        def make_task_fn(name: str):
            def task_fn():
                # Identify trainable variables for this task
                active_head = self.task_heads[name]
                trainable_vars = self.bert_encoder.trainable_weights + active_head.trainable_weights

                with tf.GradientTape() as tape:
                    outputs = self(x, task_name=name, training=True)
                    task_outputs = outputs[name]
                    labels = y[name]

                    # Compute loss for the active task
                    loss = self._compute_loss_for_task(name, labels, task_outputs, x.get("attention_mask"))

                # Compute and apply gradients
                gradients = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Update compiled metrics
                self.compiled_metrics.update_state(y, outputs)
                return {m.name: m.result() for m in self.metrics}

            return task_fn

        # Create branches for tf.switch_case
        branches = [
            (tf.equal(task_name_tensor[0], name), make_task_fn(name))
            for name in self.task_configs.keys()
        ]

        # Execute the correct branch
        return tf.switch_case(branches)

    def _compute_loss_for_task(self, task_name, y_true, y_pred, attention_mask=None):
        """Helper to compute loss for a specific task."""
        task_config = self.task_configs[task_name]
        loss_fn = self.loss_fns[task_name]

        if task_config.task_type == NLPTaskType.QUESTION_ANSWERING:
            loss = loss_fn(y_true, y_pred)
        elif task_config.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
            loss = self._token_classification_loss(y_true, y_pred, attention_mask)
        else:
            loss = loss_fn(y_true, y_pred["logits"])

        return loss

    def _get_loss_function(self, task_config: TaskConfiguration) -> Callable:
        """Get loss function for a task."""
        task_type = task_config.task_type
        if task_type == NLPTaskType.SENTIMENT_ANALYSIS:
            return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
            return self._token_classification_loss
        elif task_type == NLPTaskType.QUESTION_ANSWERING:
            return self._qa_loss
        else:
            return keras.losses.MeanSquaredError()

    def _token_classification_loss(self, y_true, y_pred, attention_mask=None):
        """Compute token classification loss with masking."""
        logits = y_pred["logits"]
        loss = keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True)
        if attention_mask is not None:
            mask = keras.ops.cast(attention_mask, loss.dtype)
            loss = loss * mask
            return keras.ops.sum(loss) / keras.ops.maximum(keras.ops.sum(mask), 1.0)
        return keras.ops.mean(loss)

    def _qa_loss(self, y_true, y_pred):
        """Compute question answering loss."""
        start_loss = keras.losses.sparse_categorical_crossentropy(y_true["start_positions"], y_pred["start_logits"],
                                                                  from_logits=True)
        end_loss = keras.losses.sparse_categorical_crossentropy(y_true["end_positions"], y_pred["end_logits"],
                                                                from_logits=True)
        return (keras.ops.mean(start_loss) + keras.ops.mean(end_loss)) / 2.0

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        task_configs_serializable = copy.deepcopy(self.task_configs_dict)
        for task_dict in task_configs_serializable.values():
            if 'task_type' in task_dict and isinstance(task_dict['task_type'], NLPTaskType):
                task_dict['task_type'] = task_dict['task_type'].value
        config.update({
            "bert_config": self.bert_config,
            "task_configs_dict": task_configs_serializable
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiTaskBERTModel":
        """Create model from configuration."""
        task_configs_dict_deserializable = copy.deepcopy(config["task_configs_dict"])
        for task_dict in task_configs_dict_deserializable.values():
            if 'task_type' in task_dict:
                task_dict['task_type'] = NLPTaskType(task_dict['task_type'])
        task_configs = {
            name: TaskConfiguration(**task_dict)
            for name, task_dict in task_configs_dict_deserializable.items()
        }
        config_copy = config.copy()
        config_copy.pop("task_configs_dict", None)
        return cls(
            bert_config=config["bert_config"],
            task_configs=task_configs,
            **config_copy
        )


# ---------------------------------------------------------------------
# Training Loop using Keras model.fit()
# ---------------------------------------------------------------------


class MultiTaskTrainerKeras:
    """Trainer for multi-task BERT models using the Keras `fit` method.

    :param model: Multi-task BERT model.
    :type model: MultiTaskBERTModel
    :param task_configs: Dictionary of task configurations.
    :type task_configs: Dict[str, TaskConfiguration]
    :param training_config: Training configuration.
    :type training_config: TrainingConfiguration
    """

    def __init__(
            self,
            model: MultiTaskBERTModel,
            task_configs: Dict[str, TaskConfiguration],
            training_config: TrainingConfiguration
    ) -> None:
        """Initialize the trainer.

        :param model: Model to train.
        :type model: MultiTaskBERTModel
        :param task_configs: Task configurations.
        :type task_configs: Dict[str, TaskConfiguration]
        :param training_config: Training configuration.
        :type training_config: TrainingConfiguration
        """
        self.model = model
        self.task_configs = task_configs
        self.config = training_config

        self.save_dir = Path(training_config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        global_max_length = max(c.max_sequence_length for c in task_configs.values())
        logger.info(f"Padding all task batches to global max sequence length: {global_max_length}")

        self.tokenizer = SimpleTokenizer(
            vocab_size=training_config.vocab_size,
            max_length=global_max_length
        )

    def _create_multitask_dataset(self, split: str = "train") -> tf.data.Dataset:
        """Creates a single dataset by sampling from all task-specific datasets.

        :param split: The dataset split to load ('train' or 'validation').
        :type split: str
        :return: A `tf.data.Dataset` that yields batches for multi-task training.
        :rtype: tf.data.Dataset
        """
        task_datasets = []
        for task_name, task_config in self.task_configs.items():
            loader = TaskDataLoader(task_config, self.tokenizer)
            dataset_split = task_config.dataset_split_train if split == "train" else task_config.dataset_split_val

            ds = loader.load_dataset(dataset_split)

            # Add the task name to the input dictionary and format labels
            def format_for_multitask(inputs, labels):
                inputs['task_name'] = tf.constant(task_name, dtype=tf.string)
                return inputs, {task_name: labels}

            ds = ds.map(format_for_multitask, num_parallel_calls=tf.data.AUTOTUNE)
            task_datasets.append(ds.repeat())

        if not task_datasets:
            raise ValueError("No datasets were loaded. Check task configurations.")

        # Sample from the different task datasets
        if self.config.use_task_sampling and split == 'train':
            weights = [
                config.loss_weight ** (1.0 / self.config.task_sampling_temperature)
                for config in self.task_configs.values()
            ]
            sampling_weights = np.array(weights) / np.sum(weights)
            logger.info(f"Using task sampling with weights: {sampling_weights}")

            # For tf < 2.16, use tf.data.experimental.sample_from_datasets
            if hasattr(tf.data.experimental, "sample_from_datasets"):
                combined_dataset = tf.data.experimental.sample_from_datasets(
                    task_datasets, weights=sampling_weights
                )
            else:  # For tf >= 2.16
                combined_dataset = tf.data.Dataset.sample_from_datasets(
                    task_datasets, weights=sampling_weights
                )
        else:
            # For validation or if sampling is off, interleave datasets
            combined_dataset = tf.data.Dataset.from_tensor_slices(task_datasets).interleave(
                lambda x: x,
                cycle_length=len(task_datasets),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Determine a shared batch size or use the first task's
        batch_size = next(iter(self.task_configs.values())).batch_size

        return combined_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train(self) -> None:
        """Configures and runs the training using `model.fit()`."""
        logger.info("Starting training with Keras `model.fit()`...")

        # 1. Create the combined datasets
        train_dataset = self._create_multitask_dataset(split="train")
        val_dataset = self._create_multitask_dataset(split="validation")

        # 2. Prepare for model.compile()
        losses = {name: self.model.loss_fns[name] for name in self.task_configs}
        loss_weights = {name: config.loss_weight for name, config in self.task_configs.items()}
        metrics = {
            name: keras.metrics.SparseCategoricalAccuracy(name=f"{name}_accuracy")
            for name, config in self.task_configs.items()
            if config.task_type in [NLPTaskType.SENTIMENT_ANALYSIS, NLPTaskType.NAMED_ENTITY_RECOGNITION]
        }

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.epochs * 1000,  # Approximate steps
            warmup_target=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay,
            clipnorm=self.config.max_grad_norm
        )

        # 3. Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            # IMPORTANT: This ensures the custom train_step is used.
            # It also disables the default Keras loss calculation.
            run_eagerly=False
        )

        # 4. Set up callbacks
        checkpoint_path = self.save_dir / "ckpt-{epoch:02d}-{val_loss:.2f}.keras"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.save_dir / "logs"),
                update_freq=self.config.logging_steps
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                verbose=1
            )
        ]

        # 5. Start training
        # Approximate steps per epoch for display
        steps_per_epoch = 1000
        validation_steps = 200

        self.model.fit(
            train_dataset,
            epochs=self.config.epochs,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        final_model_path = self.save_dir / "final_model.keras"
        self.model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Save training config
        config_path = self.save_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def create_default_tasks() -> Dict[str, TaskConfiguration]:
    """Create default task configurations.

    :return: Dictionary of default task configurations.
    :rtype: Dict[str, TaskConfiguration]
    """
    tasks = {
        "sentiment": TaskConfiguration(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            dataset_name="imdb_reviews",
            num_classes=2,
            max_sequence_length=128,
            batch_size=32,
            loss_weight=1.0,
            head_config={
                "dropout_rate": 0.1,
                "use_intermediate": True,
                "intermediate_size": 256
            }
        ),
        "ner": TaskConfiguration(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            dataset_name="conll2003",
            num_classes=9,
            max_sequence_length=128,
            batch_size=32,
            loss_weight=0.8,
            head_config={
                "dropout_rate": 0.1,
                "use_task_attention": True,
                "use_intermediate": True
            }
        ),
    }

    return tasks


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train BERT model on multiple NLP tasks"
    )
    parser.add_argument(
        "--bert-variant",
        type=str,
        default="tiny",
        choices=["base", "large", "small", "tiny"],
        help="BERT model variant"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints_keras_fit",
        help="Directory to save models"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    training_config = TrainingConfiguration(
        bert_variant=args.bert_variant,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        seed=args.seed
    )

    task_configs = create_default_tasks()
    if args.batch_size:
        for task_config in task_configs.values():
            task_config.batch_size = args.batch_size

    bert_config = BERT.MODEL_VARIANTS[args.bert_variant].copy()
    bert_config.pop("description", None)

    logger.info("Creating multi-task BERT model...")
    model = MultiTaskBERTModel(
        bert_config=bert_config,
        task_configs=task_configs
    )

    # Use the new Keras-native trainer
    trainer = MultiTaskTrainerKeras(
        model=model,
        task_configs=task_configs,
        training_config=training_config
    )

    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()