"""Neural Turing Machine training on the copy task."""

import keras
import numpy as np

from train.common import setup_gpu, create_base_argument_parser, create_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory import create_ntm
from . import CopyTaskGenerator, CopyTaskConfig


# =====================================================================
# Training Pipeline
# =====================================================================

def main():
    parser = create_base_argument_parser(
        description="Train NTM on copy task",
        default_dataset="copy",
        dataset_choices=["copy"],
    )
    parser.add_argument('--memory-size', type=int, default=128)
    parser.add_argument('--memory-dim', type=int, default=20)
    parser.add_argument('--controller-dim', type=int, default=100)
    parser.add_argument('--controller-type', type=str, default='lstm', choices=['lstm', 'mlp'])
    parser.add_argument('--num-read-heads', type=int, default=1)
    parser.add_argument('--num-write-heads', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=20)
    parser.add_argument('--vector-size', type=int, default=8)
    parser.add_argument('--num-samples', type=int, default=100000)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--validation-split', type=float, default=0.1)
    parser.add_argument('--num-eval-samples', type=int, default=20)
    parser.add_argument('--success-threshold', type=float, default=0.9)
    args = parser.parse_args()

    if CopyTaskGenerator is None:
        logger.error("Cannot proceed without CopyTaskGenerator.")
        return

    setup_gpu(args.gpu)

    # Generate data
    logger.info("Generating Copy Task data...")
    config = CopyTaskConfig(
        sequence_length=args.sequence_length,
        vector_size=args.vector_size,
        num_samples=args.num_samples,
    )
    generator = CopyTaskGenerator(config)
    data = generator.generate()
    logger.info(f"Input: {data.inputs.shape}, Target: {data.targets.shape}")

    # Build model
    seq_len = data.inputs.shape[1]
    input_dim = data.inputs.shape[2]
    output_dim = data.targets.shape[2]

    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    ntm_layer = create_ntm(
        memory_size=args.memory_size, memory_dim=args.memory_dim, output_dim=output_dim,
        controller_dim=args.controller_dim, controller_type=args.controller_type,
        num_read_heads=args.num_read_heads, num_write_heads=args.num_write_heads,
        return_sequences=True,
    )
    x = ntm_layer(inputs)
    outputs = keras.layers.Activation('sigmoid', name="binary_output")(x)
    model = keras.Model(inputs, outputs, name="ntm_copy_task")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clip_norm),
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.summary()

    # Callbacks
    callbacks, results_dir = create_callbacks(
        model_name="copy_task",
        results_dir_prefix="ntm",
        monitor="val_loss",
        patience=args.patience,
        use_lr_schedule=False,
    )

    # Train
    logger.info("Starting training...")
    history = model.fit(
        data.inputs, data.targets, sample_weight=data.masks,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_split=args.validation_split,
        callbacks=callbacks, verbose=1,
    )

    evaluate_model(model, data, num_eval=args.num_eval_samples,
                   success_threshold=args.success_threshold)


def evaluate_model(model, data, num_eval=20, success_threshold=0.9):
    """Detailed evaluation of NTM copy task performance.

    Scores the output phase only: per-sequence exact match and per-bit
    accuracy over the positions the task mask marks as supervised.

    :param model: Object exposing ``predict(inputs, verbose=...)``.
    :param data: ``TaskData`` with ``inputs``, ``targets`` and per-timestep ``masks``.
    :param num_eval: Number of sequences to draw, without replacement.
    :param success_threshold: Sequence accuracy above which the task counts as solved.
    :return: Dict with ``bit_accuracy``, ``sequence_accuracy`` and
        ``num_evaluated`` (rows that were not fully masked out).
    """
    indices = np.random.choice(len(data.inputs), num_eval, replace=False)
    eval_inputs = data.inputs[indices]
    eval_targets = data.targets[indices]
    eval_masks = data.masks[indices]

    preds = model.predict(eval_inputs, verbose=0)
    preds_binary = (preds > 0.5).astype(float)

    seq_accs = []
    bit_accs = []

    for i in range(num_eval):
        # The mask is per-TIMESTEP, shape (seq_len,), while a prediction is
        # per-ELEMENT, shape (seq_len, output_dim). Broadcast the mask across
        # the output dimension before flattening -- flattening the mask instead
        # only lines up when output_dim == 1.
        mask_boolean = np.broadcast_to(
            eval_masks[i].astype(bool)[:, None], preds_binary[i].shape
        ).ravel()
        p_valid = preds_binary[i].ravel()[mask_boolean]
        t_valid = eval_targets[i].ravel()[mask_boolean]

        if len(p_valid) == 0:
            continue

        seq_accs.append(1.0 if np.array_equal(p_valid, t_valid) else 0.0)
        bit_accs.append(np.mean(p_valid == t_valid))

    mean_bit_acc = float(np.mean(bit_accs)) if bit_accs else float("nan")
    mean_seq_acc = float(np.mean(seq_accs)) if seq_accs else float("nan")

    logger.info(f"Evaluation (N={len(seq_accs)}/{num_eval}): "
                f"Bit Acc={mean_bit_acc:.2%}, Sequence Acc={mean_seq_acc:.2%}")

    if mean_seq_acc > success_threshold:
        logger.info("SUCCESS: NTM has solved the copy task!")
    else:
        logger.info("STATUS: NTM needs more training or tuning.")

    return {
        "bit_accuracy": mean_bit_acc,
        "sequence_accuracy": mean_seq_acc,
        "num_evaluated": len(seq_accs),
    }


if __name__ == '__main__':
    main()
