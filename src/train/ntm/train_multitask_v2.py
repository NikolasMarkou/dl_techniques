"""Multi-Task NTM Training Script."""

import os
import keras
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask
from train.common import setup_gpu

from .data_generators import (
    CopyTaskGenerator, CopyTaskConfig,
    AssociativeRecallGenerator, AssociativeRecallConfig,
    RepeatCopyGenerator, PriorityAccessGenerator,
    DynamicNGramGenerator, AlgorithmicTaskGenerator,
    AlgorithmicTaskConfig, TaskData
)


@dataclass
class MultitaskNTMConfig:
    """Configuration for Multi-Task NTM Training."""
    memory_size: int = 128
    memory_dim: int = 20
    controller_dim: int = 256
    controller_type: str = "lstm"
    num_read_heads: int = 1
    num_write_heads: int = 1
    shift_range: int = 3
    epsilon: float = 1e-6
    max_seq_length: int = 100
    max_vector_size: int = 16
    batch_size: int = 64
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    validation_steps: int = 100
    learning_rate: float = 1e-4
    clip_norm: float = 1.0
    save_dir: str = "results/multitask_ntm"
    checkpoint_dir: str = "results/multitask_ntm/checkpoints"
    log_dir: str = "results/multitask_ntm/logs"
    num_eval_samples: int = 1000
    task_map: Dict[str, int] = field(default_factory=lambda: {
        "copy": 0, "associative_recall": 1, "repeat_copy": 2,
        "priority_access": 3, "dynamic_ngram": 4, "insertion_sort": 5
    })

    @property
    def num_tasks(self) -> int:
        return len(self.task_map)


class UnifiedTaskGenerator(keras.utils.Sequence):
    """Unified Keras Sequence for multi-task data generation."""

    def __init__(self, config: MultitaskNTMConfig, mode: str = 'train') -> None:
        self.config = config
        self.mode = mode
        self.batch_size = config.batch_size
        self.steps = config.steps_per_epoch if mode == 'train' else config.validation_steps

        seed = 42 if mode == 'val' else None
        self.rng = np.random.default_rng(seed)

        self.generators = {
            "copy": CopyTaskGenerator(CopyTaskConfig(sequence_length=10, vector_size=8, random_seed=seed)),
            "associative_recall": AssociativeRecallGenerator(AssociativeRecallConfig(num_items=6, key_size=8, value_size=8, random_seed=seed)),
            "repeat_copy": RepeatCopyGenerator(vector_size=8, random_seed=seed),
            "priority_access": PriorityAccessGenerator(vector_size=8, random_seed=seed),
            "dynamic_ngram": DynamicNGramGenerator(vocab_size=8, random_seed=seed),
            "insertion_sort": AlgorithmicTaskGenerator(AlgorithmicTaskConfig(task_name="insertion_sort", train_size=16, test_size=16, random_seed=seed))
        }

    def __len__(self) -> int:
        return self.steps

    def _generate_raw_data(self, task_key: str) -> TaskData:
        gen = self.generators[task_key]
        if task_key == "copy":
            return gen.generate(self.batch_size, sequence_length=self.rng.integers(5, 20))
        elif task_key == "associative_recall":
            return gen.generate(self.batch_size, num_items=self.rng.integers(2, 6))
        elif task_key == "repeat_copy":
            return gen.generate(self.batch_size, sequence_length=self.rng.integers(3, 8), num_repeats=self.rng.integers(2, 4))
        elif task_key == "priority_access":
            return gen.generate(self.batch_size, num_items=self.rng.integers(3, 10))
        elif task_key == "dynamic_ngram":
            seq_len = self.rng.integers(10, 30)
            data = gen.generate(self.batch_size, sequence_length=seq_len)
            if data.masks is None:
                mask = np.ones((self.batch_size, seq_len))
                mask[:, :2] = 0.0
                data.masks = mask
            return data
        elif task_key == "insertion_sort":
            return gen.generate(self.batch_size, problem_size=self.rng.integers(5, 16))
        else:
            raise ValueError(f"Unknown task key: {task_key}")

    def _pad_and_normalize(self, data: TaskData, task_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs_raw = data.inputs
        targets_raw = data.targets
        masks_raw = data.masks
        curr_batch, curr_seq, _ = inputs_raw.shape
        target_dim = targets_raw.shape[-1]

        if task_key == "associative_recall":
            seq_targets = np.zeros((curr_batch, curr_seq, target_dim), dtype=np.float32)
            seq_targets[:, -1, :] = targets_raw
            targets_raw = seq_targets
            masks_raw = np.zeros((curr_batch, curr_seq), dtype=np.float32)
            masks_raw[:, -1] = 1.0

        if task_key == "insertion_sort" and masks_raw is None:
            masks_raw = np.ones((curr_batch, curr_seq), dtype=np.float32)

        max_seq = self.config.max_seq_length
        max_vec = self.config.max_vector_size

        inputs_padded = np.zeros((curr_batch, max_seq, max_vec), dtype=np.float32)
        targets_padded = np.zeros((curr_batch, max_seq, max_vec), dtype=np.float32)
        masks_padded = np.zeros((curr_batch, max_seq), dtype=np.float32)

        limit_seq = min(curr_seq, max_seq)
        limit_in_dim = min(inputs_raw.shape[-1], max_vec)
        limit_out_dim = min(target_dim, max_vec)

        inputs_padded[:, :limit_seq, :limit_in_dim] = inputs_raw[:, :limit_seq, :limit_in_dim]
        targets_padded[:, :limit_seq, :limit_out_dim] = targets_raw[:, :limit_seq, :limit_out_dim]

        if masks_raw is not None:
            masks_padded[:, :limit_seq] = masks_raw[:, :limit_seq]
        else:
            masks_padded[:, :limit_seq] = 1.0

        return inputs_padded, targets_padded, masks_padded

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        task_key = self.rng.choice(list(self.config.task_map.keys()))
        task_id = self.config.task_map[task_key]
        data = self._generate_raw_data(task_key)
        inputs, targets, masks = self._pad_and_normalize(data, task_key)

        task_one_hot = np.zeros((self.batch_size, self.config.num_tasks), dtype=np.float32)
        task_one_hot[:, task_id] = 1.0

        return {"sequence_in": inputs, "task_id_in": task_one_hot}, targets, masks


def create_generators(config: MultitaskNTMConfig) -> Tuple[UnifiedTaskGenerator, UnifiedTaskGenerator]:
    logger.info("Initializing task generators...")
    return UnifiedTaskGenerator(config, mode='train'), UnifiedTaskGenerator(config, mode='val')


def create_multitask_ntm_model(config: MultitaskNTMConfig) -> keras.Model:
    """Create the Multi-Task NTM Keras model."""
    ntm_config = NTMConfig(
        memory_size=config.memory_size, memory_dim=config.memory_dim,
        controller_dim=config.controller_dim, controller_type=config.controller_type,
        num_read_heads=config.num_read_heads, num_write_heads=config.num_write_heads,
        shift_range=config.shift_range, epsilon=config.epsilon
    )

    seq_input = keras.Input(shape=(config.max_seq_length, config.max_vector_size), name="sequence_in")
    task_input = keras.Input(shape=(config.num_tasks,), name="task_id_in")

    ntm_model = NTMMultiTask(ntm_config=ntm_config, output_dim=config.max_vector_size, num_tasks=config.num_tasks)
    outputs = ntm_model([seq_input, task_input])
    final_output = keras.layers.Activation("sigmoid", name="output_sigmoid")(outputs)

    model = keras.Model(inputs=[seq_input, task_input], outputs=final_output, name="multitask_ntm")
    logger.info(f"Model built with {model.count_params():,} parameters.")
    return model


def compile_model(model: keras.Model, config: MultitaskNTMConfig) -> None:
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.clip_norm)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.MeanSquaredError(name="mse")]
    )


def create_callbacks(config: MultitaskNTMConfig) -> List[keras.callbacks.Callback]:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.checkpoint_dir, "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(filename=os.path.join(config.save_dir, "training_log.csv"), append=True)
    ]


def train_multitask_ntm(config: MultitaskNTMConfig) -> Tuple[keras.Model, keras.callbacks.History]:
    """Execute the full training pipeline."""
    train_gen, val_gen = create_generators(config)
    model = create_multitask_ntm_model(config)
    compile_model(model, config)
    callbacks = create_callbacks(config)

    logger.info("Starting training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=config.num_epochs, callbacks=callbacks, verbose=1)

    final_path = os.path.join(config.save_dir, "multitask_ntm_final.keras")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    return model, history


def evaluate_tasks(model: keras.Model, config: MultitaskNTMConfig) -> None:
    """Evaluate the model on all tasks."""
    logger.info(f"Evaluating on {config.num_eval_samples} samples per task...")
    eval_gen = UnifiedTaskGenerator(config, mode='val')
    results = {}

    for task_name, task_id in config.task_map.items():
        raw_data = eval_gen._generate_raw_data(task_name)
        inputs, targets, masks = eval_gen._pad_and_normalize(raw_data, task_name)

        task_one_hot = np.zeros((config.batch_size, config.num_tasks), dtype=np.float32)
        task_one_hot[:, task_id] = 1.0

        preds = model.predict({"sequence_in": inputs, "task_id_in": task_one_hot}, verbose=0)
        mask_bool = masks.astype(bool)
        preds_flat = preds[mask_bool]
        targets_flat = targets[mask_bool]

        if task_name == "insertion_sort":
            mse = np.mean((preds_flat - targets_flat) ** 2)
            logger.info(f"Task: {task_name:<20} | MSE: {mse:.6f}")
            results[task_name] = mse
        else:
            accuracy = np.mean((preds_flat > 0.5).astype(float) == targets_flat)
            logger.info(f"Task: {task_name:<20} | Acc: {accuracy:.4f} | Error: {1.0 - accuracy:.4f}")
            results[task_name] = accuracy


def main() -> None:
    config = MultitaskNTMConfig()
    logger.info(f"Tasks: {list(config.task_map.keys())}, Memory: {config.memory_size}x{config.memory_dim}, "
                f"Batch: {config.batch_size}, Epochs: {config.num_epochs}")

    try:
        model, history = train_multitask_ntm(config)
        evaluate_tasks(model, config)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
