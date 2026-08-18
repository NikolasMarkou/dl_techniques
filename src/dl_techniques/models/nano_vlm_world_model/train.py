"""
Training Infrastructure for Score-Based nanoVLM

Implements Denoising Score Matching (DSM) training and loss functions.
The key insight: training an optimal denoiser via MSE is equivalent to
learning the score function ∇ log p by Miyasawa's theorem.
"""

import keras
from keras import ops
from typing import Dict, Optional, Any
import tensorflow as tf

from dl_techniques.utils.logger import logger
from .model import create_score_based_nanovlm

@keras.saving.register_keras_serializable()
class DenoisingScoreMatchingLoss(keras.losses.Loss):
    """
    Denoising Score Matching loss for learning score functions.

    By Miyasawa's theorem, training a denoiser D(x_t, c, t) to predict
    the clean data x_0 from noisy x_t is equivalent to learning the score:

        L_DSM = E[||D(x_t, c, t) - x_0||²]

    This simple MSE loss implicitly trains the model to estimate
    ∇ log p(x_t | c), enabling score-based generation.

    This loss is **objective-agnostic and timestep-agnostic**: it is the plain
    mean squared error between what the denoiser emitted and whatever target
    the caller handed it. It does not know which parameterisation produced
    ``y_pred`` and it does not weight timesteps.

    .. note::
       It used to advertise ``prediction_type`` ('epsilon'/'sample'/
       'v_prediction'), ``loss_weight_type`` ('uniform'/'snr'/'truncated_snr',
       citing Hang et al. 2023) and ``min_snr_gamma``. All three were stored,
       serialized and never read -- ``call()`` has always been an
       unconditional MSE -- so ``prediction_type='epsilon'`` on a package whose
       denoisers are all x_0 predictors was harmless only because the field was
       dead. Removed 2026-08-19 rather than implemented, because the weighting
       needs per-sample timesteps this signature cannot receive and the
       parameterisation is chosen by
       :class:`~dl_techniques.models.nano_vlm_world_model.scheduler.DiffusionScheduler`,
       which has a live ``prediction_type`` of its own. :meth:`from_config`
       still accepts and drops the three legacy keys. See decisions.md
       plan-2026-08-18T140459-7991552f/D-034.

    Args:
        reduction: Keras reduction type. Defaults to 'sum_over_batch_size'.
        name: Loss name. Defaults to 'dsm_loss'.
        **kwargs: Additional loss arguments.

    References:
        - Ho et al. (2020): Simple L2 loss works best
    """

    # DECISION plan-2026-08-18T140459-7991552f/D-034
    # Deliberately NO `prediction_type` / `loss_weight_type` / `min_snr_gamma`
    # arguments. Do NOT re-add them as stored-and-serialized fields: that is
    # exactly the state this class shipped in, where a caller could set
    # `loss_weight_type='snr'`, see it in the reloaded config, and train under
    # uniform weighting. `call(y_true, y_pred)` receives no timesteps and no
    # alphas, so SNR weighting is not implementable at this signature; the
    # parameterisation lives on `DiffusionScheduler.prediction_type`, which is
    # read. If min-SNR weighting is wanted, pass per-sample `sample_weight`
    # from the trainer, which does have the timesteps. See decisions.md.
    _LEGACY_DEAD_KEYS = ('prediction_type', 'loss_weight_type', 'min_snr_gamma')

    def __init__(
            self,
            reduction: str = 'sum_over_batch_size',
            name: str = 'dsm_loss',
            **kwargs
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)

        logger.info("Initialized DSM loss: unweighted MSE over all timesteps")

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute DSM loss.

        Args:
            y_true: The target the denoiser is fit against -- whichever
                parameterisation the caller trained (x_0 for every denoiser in
                this package). This loss does not branch on it.
            y_pred: Predicted data from denoiser

        Returns:
            Scalar loss value
        """
        # Simple MSE - the magic is in what we're predicting.
        # There is no timestep-dependent weighting here and no branch on a
        # parameterisation; see the class-level D-032 anchor.
        loss = ops.mean(ops.square(y_pred - y_true), axis=list(range(1, len(y_pred.shape))))

        return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DenoisingScoreMatchingLoss":
        """Rebuild from a ``get_config()`` dict, tolerating pre-2026-08-19 ones.

        Configs saved before the three dead knobs were removed carry them. They
        are dropped rather than rejected: the values never reached ``call()``,
        so a restored loss computes exactly what the saved one computed.
        """
        dropped = [k for k in cls._LEGACY_DEAD_KEYS if k in config]
        if dropped:
            config = {k: v for k, v in config.items()
                      if k not in cls._LEGACY_DEAD_KEYS}
            logger.warning(
                f"DenoisingScoreMatchingLoss: ignoring legacy config key(s) "
                f"{dropped}; they never affected the loss (unweighted MSE)."
            )
        return cls(**config)


@keras.saving.register_keras_serializable()
class VLMDenoisingLoss(keras.losses.Loss):
    """
    Combined loss for vision-language denoising.

    Supports multiple denoising objectives simultaneously:
    - Vision denoising (text → image)
    - Text denoising (image → text)
    - Joint denoising (unified world model)

    Args:
        vision_weight: Weight for vision denoising loss. Defaults to 1.0.
        text_weight: Weight for text denoising loss. Defaults to 1.0.
        joint_weight: Weight for joint denoising loss. Defaults to 0.5.
        **kwargs: Additional loss arguments.
    """

    def __init__(
            self,
            vision_weight: float = 1.0,
            text_weight: float = 1.0,
            joint_weight: float = 0.5,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.joint_weight = joint_weight

        # Component losses
        # The `prediction_type='sample'` this used to pass was a no-op on a
        # class that never read it (D-032); the sub-loss is a plain MSE.
        self.dsm_loss = DenoisingScoreMatchingLoss()

    def call(
            self,
            y_true: Dict[str, keras.KerasTensor],
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Compute combined VLM loss.

        Args:
            y_true: Dictionary of true targets
            y_pred: Dictionary of predictions

        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0

        # Vision denoising loss
        if 'denoised_vision' in y_pred and 'target_vision' in y_true:
            vision_loss = self.dsm_loss(
                y_true['target_vision'], y_pred['denoised_vision']
            )
            total_loss += self.vision_weight * vision_loss

        # Text denoising loss
        if 'denoised_text' in y_pred and 'target_text' in y_true:
            text_loss = self.dsm_loss(
                y_true['target_text'], y_pred['denoised_text']
            )
            total_loss += self.text_weight * text_loss

        # Joint denoising losses
        if 'joint_denoised_vision' in y_pred:
            joint_vision_loss = self.dsm_loss(
                y_true['joint_target_vision'], y_pred['joint_denoised_vision']
            )
            total_loss += self.joint_weight * joint_vision_loss

        if 'joint_denoised_text' in y_pred:
            joint_text_loss = self.dsm_loss(
                y_true['joint_target_text'], y_pred['joint_denoised_text']
            )
            total_loss += self.joint_weight * joint_text_loss

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vision_weight': self.vision_weight,
            'text_weight': self.text_weight,
            'joint_weight': self.joint_weight,
        })
        return config


class ScoreVLMTrainer:
    """
    Custom trainer for Score-Based nanoVLM.

    Implements the Denoising Score Matching training loop with support
    for mixed precision, gradient accumulation, and EMA.

    Args:
        model: ScoreBasedNanoVLM instance
        optimizer: Keras optimizer
        loss_fn: VLMDenoisingLoss instance
        use_ema: Whether to use Exponential Moving Average. Defaults to True.
        ema_decay: EMA decay rate. Defaults to 0.9999.
        gradient_accumulation_steps: Steps to accumulate gradients. Defaults to 1.
    """

    def __init__(
            self,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss_fn: VLMDenoisingLoss,
            use_ema: bool = True,
            ema_decay: float = 0.9999,
            gradient_accumulation_steps: int = 1
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-022
        # Every piece of trainer state that owns a variable is created by
        # `_ensure_state`, EAGERLY, on the first step — not here and not inside
        # `_train_step_fn`.
        #
        # WHAT NOT TO DO:
        #   * Do NOT go back to `self.ema_model = keras.models.clone_model(model)`
        #     followed by `set_weights(model.get_weights())` in this constructor.
        #     `model` is an UNBUILT subclassed `keras.Model` at this point, so
        #     `clone_model` returns an unbuilt clone and `get_weights()` returns
        #     `[]` — the copy was `set_weights([])`, `_update_ema` zipped two
        #     empty lists forever, and `train_score_vlm` saved that inert clone as
        #     EVERY epoch checkpoint.
        #   * Do NOT let the model, the optimizer slots or the accumulators be
        #     created by the first trace of `_train_step_fn`. TensorFlow rejects
        #     that outright on the second trace: "tf.function only supports
        #     singleton tf.Variables created on the first call".
        # See decisions.md D-022.
        self.ema_model = None
        self._state_ready = False

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.vision_loss = keras.metrics.Mean(name='vision_loss')
        self.text_loss = keras.metrics.Mean(name='text_loss')

        # Gradient accumulation. Both become variables in `_ensure_state`; a
        # Python int here would be folded to a constant at trace time (D-023).
        self.accumulated_gradients = None
        self.accumulation_counter = None

        logger.info(
            f"Initialized ScoreVLM trainer with EMA={use_ema}, "
            f"grad_accum={gradient_accumulation_steps}"
        )

    def _ensure_state(
            self,
            images: keras.KerasTensor,
            text_tokens: keras.KerasTensor
    ) -> None:
        """Build the model, the optimizer slots, the EMA clone and the
        accumulators — once, eagerly, before the first trace.

        Interface contract (1 call site, :meth:`train_step`; it is a separate
        method so the whole variable-creating half stays OUTSIDE the
        ``tf.function``): idempotent, no return value, safe to call on every
        step. Requires a real batch because a subclassed ``keras.Model`` has no
        other way to be built.
        """
        if self._state_ready:
            return

        sample = {'images': images, 'text': text_tokens}
        if not self.model.built:
            self.model(sample, training=False)

        # Adam/AdamW slot variables, created here rather than by the first
        # `apply_gradients` inside the traced step.
        self.optimizer.build(self.model.trainable_variables)

        if self.use_ema:
            ema_model = keras.models.clone_model(self.model)
            # Unconditional: whether `clone_model` returns a BUILT clone for a
            # subclassed model is a Keras implementation detail, and the whole
            # defect was a `set_weights` against an unbuilt pair. A redundant
            # forward pass is cheaper than a guard that cannot be falsified.
            ema_model(sample, training=False)
            ema_model.set_weights(self.model.get_weights())
            self.ema_model = ema_model
            logger.info(
                f"EMA clone built with {len(self.ema_model.weights)} weights"
            )

        if self.gradient_accumulation_steps > 1:
            self.accumulated_gradients = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.model.trainable_variables
            ]
        self.accumulation_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

        self._state_ready = True

    def train_step(
            self,
            images: keras.KerasTensor,
            text_tokens: keras.KerasTensor
    ) -> Dict[str, float]:
        """
        Single training step with DSM.

        Args:
            images: Batch of images [batch, H, W, C]
            text_tokens: Batch of text tokens [batch, seq_len]

        Returns:
            Dictionary of metrics
        """
        self._ensure_state(images, text_tokens)
        return self._train_step_fn(images, text_tokens)

    def _apply_accumulated_gradients(self) -> tf.Tensor:
        """Apply the accumulated gradients, then zero them. A ``tf.cond`` branch."""
        self.optimizer.apply_gradients(
            zip(
                [tf.convert_to_tensor(g) for g in self.accumulated_gradients],
                self.model.trainable_variables
            )
        )
        for accumulator in self.accumulated_gradients:
            accumulator.assign(tf.zeros_like(accumulator))
        self.accumulation_counter.assign(0)
        if self.use_ema:
            self._update_ema()
        return tf.constant(0)

    @tf.function
    def _train_step_fn(
            self,
            images: keras.KerasTensor,
            text_tokens: keras.KerasTensor
    ) -> Dict[str, float]:
        """The traced half of :meth:`train_step`. Creates no variables."""
        with tf.GradientTape() as tape:
            # Forward pass: model adds noise internally and denoises
            outputs = self.model(
                {'images': images, 'text': text_tokens},
                training=True
            )

            # Compute DSM loss
            loss = self.loss_fn(outputs, outputs)

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Accumulate or apply gradients
        if self.gradient_accumulation_steps > 1:
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-023
            # The counter is a `tf.Variable` and the release is a `tf.cond`.
            #
            # WHAT NOT TO DO: do NOT reinstate the Python-int counter and the
            # Python `if self.accumulation_counter >= ...`. Inside this
            # `@tf.function` that comparison is evaluated ONCE, at trace time,
            # against the constant 0 — so at the shipped
            # `gradient_accumulation_steps=4` it folded to `False` and
            # `optimizer.apply_gradients` was never emitted into the graph at
            # all. Nothing on the Python side then changes, so no retrace ever
            # happens and the model never trains, while `train_loss.result()`
            # keeps returning a moving number. Eager execution HIDES this
            # entirely; a probe for it must run traced. Likewise the
            # accumulators must be variables: a Python list re-bound to graph
            # tensors does not survive the step.
            # See decisions.md D-023.
            for accumulator, grad in zip(self.accumulated_gradients, gradients):
                if grad is not None:
                    accumulator.assign_add(grad)

            self.accumulation_counter.assign_add(1)

            tf.cond(
                self.accumulation_counter >= tf.constant(
                    self.gradient_accumulation_steps, dtype=tf.int64
                ),
                self._apply_accumulated_gradients,
                lambda: tf.constant(0)
            )
        else:
            # Direct gradient application
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            # Update EMA
            if self.use_ema:
                self._update_ema()

        # Update metrics
        self.train_loss.update_state(loss * self.gradient_accumulation_steps)

        # Component-specific losses
        if 'denoised_vision' in outputs:
            v_loss = ops.mean(ops.square(
                outputs['denoised_vision'] - outputs['target_vision']
            ))
            self.vision_loss.update_state(v_loss)

        if 'denoised_text' in outputs:
            t_loss = ops.mean(ops.square(
                outputs['denoised_text'] - outputs['target_text']
            ))
            self.text_loss.update_state(t_loss)

        return {
            'loss': self.train_loss.result(),
            'vision_loss': self.vision_loss.result(),
            'text_loss': self.text_loss.result(),
        }

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        for ema_weight, model_weight in zip(
                self.ema_model.weights, self.model.weights
        ):
            ema_weight.assign(
                self.ema_decay * ema_weight + (1 - self.ema_decay) * model_weight
            )

    def reset_metrics(self) -> None:
        """Reset all metrics.

        The method is ``reset_state``, singular. ``keras.metrics.Metric`` has
        never defined ``reset_states`` in Keras 3 (only ``layers/rnn`` does), so
        the plural spelling raised ``AttributeError`` at the first statement of
        epoch 0 in :func:`train_score_vlm` and hid every other defect behind it.
        """
        self.train_loss.reset_state()
        self.vision_loss.reset_state()
        self.text_loss.reset_state()

    def get_model_for_inference(self) -> keras.Model:
        """Get model for inference (EMA if available)."""
        return self.ema_model if self.use_ema else self.model


def train_score_vlm(
        model: keras.Model,
        train_dataset: keras.utils.Sequence,
        epochs: int = 100,
        optimizer_config: Optional[Dict] = None,
        checkpoint_dir: str = 'checkpoints/',
        log_frequency: int = 100,
        sample_every_n_epochs: int = 5,
        num_sample_steps: int = 50
) -> None:
    """
    Main training loop for Score-Based nanoVLM.

    Args:
        model: ScoreBasedNanoVLM instance
        train_dataset: Training dataset
        epochs: Number of epochs
        optimizer_config: Optimizer configuration, in the shape
            :func:`dl_techniques.optimization.optimizer_builder` accepts, plus
            an optional ``learning_rate`` (a float or a
            ``LearningRateSchedule``) which is forwarded as that builder's
            separate ``lr_schedule`` argument. Defaults to 1e-4.
        checkpoint_dir: Directory for checkpoints
        log_frequency: Log every N steps
        sample_every_n_epochs: Generate monitoring samples every N epochs.
        num_sample_steps: Reverse-diffusion steps used for those samples.
    """
    logger.info("Starting Score-Based nanoVLM training")

    # Setup optimizer
    if optimizer_config is None:
        optimizer_config = {
            'type': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
        }

    from dl_techniques.optimization import optimizer_builder

    # DECISION plan-2026-08-17T183311-79c63e38/D-016
    # `optimizer_builder(config, lr_schedule)` takes the learning rate as its
    # OWN positional argument and never reads it out of `config`; `lr_schedule`
    # has no default.
    #
    # WHAT NOT TO DO: do NOT go back to `optimizer_builder(optimizer_config)`.
    # It raised `TypeError: missing 1 required positional argument` two
    # statements before `ScoreVLMTrainer` was even constructed, so it hid every
    # trainer-internal defect behind it -- and because this function has no
    # callers in `src/`, nothing surfaced it. Do NOT leave `learning_rate`
    # inside the dict either: it is not a key `optimizer_builder` consumes, so
    # a caller-supplied rate would silently do nothing.
    # See decisions.md D-016.
    optimizer_config = dict(optimizer_config)
    lr_schedule = optimizer_config.pop('learning_rate', 1e-4)
    optimizer = optimizer_builder(optimizer_config, lr_schedule)

    # Setup loss and trainer
    loss_fn = VLMDenoisingLoss(
        vision_weight=1.0,
        text_weight=1.0,
        joint_weight=0.5
    )

    trainer = ScoreVLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        use_ema=True,
        gradient_accumulation_steps=4
    )

    # Training loop
    global_step = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        trainer.reset_metrics()

        last_text_tokens = None
        # DECISION plan-2026-08-17T183311-79c63e38/D-016
        # Index the dataset by `len()`. `keras.utils.Sequence` is
        # `PyDataset` in Keras 3 and defines NO `__iter__` (MEASURED against
        # keras 3.8) -- so `for ... in train_dataset` falls back to Python's
        # legacy `__getitem__` protocol, which walks 0, 1, 2, ... until
        # `IndexError` and IGNORES `__len__` entirely.
        #
        # WHAT NOT TO DO: do not restore
        # `for step, (images, text_tokens) in enumerate(train_dataset)`. A
        # correctly-written `PyDataset` -- including this module's own
        # `example_training` `DummyDataset`, whose `__getitem__` generates a
        # fresh random batch for any index and never raises -- makes that loop
        # run FOREVER. Epoch 0 never ends, so no checkpoint is ever written and
        # the run looks like a hang rather than a failure.
        # See decisions.md D-016.
        for step in range(len(train_dataset)):
            images, text_tokens = train_dataset[step]
            last_text_tokens = text_tokens
            metrics = trainer.train_step(images, text_tokens)

            global_step += 1

            # Logging
            if step % log_frequency == 0:
                logger.info(
                    f"Step {global_step}: "
                    f"Loss={float(metrics['loss']):.4f}, "
                    f"VisionLoss={float(metrics.get('vision_loss', 0)):.4f}, "
                    f"TextLoss={float(metrics.get('text_loss', 0)):.4f}"
                )

        # Epoch end: save checkpoint
        checkpoint_path = f"{checkpoint_dir}/score_vlm_epoch_{epoch + 1}.keras"
        inference_model = trainer.get_model_for_inference()
        inference_model.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Generate samples for monitoring. Guarded because generation needs the
        # vision denoiser, which a text-only configuration does not build; a
        # monitoring hook must never be able to abort a training run.
        if epoch % sample_every_n_epochs == 0 and last_text_tokens is not None:
            try:
                text_features = inference_model.text_encoder(
                    {'input_ids': last_text_tokens}, training=False
                )
                samples = inference_model.generate_from_text(
                    text_features, num_inference_steps=num_sample_steps
                )
                logger.info(
                    f"Generated monitoring samples with shape {tuple(samples.shape)}"
                )
            except Exception as e:
                logger.warning(f"Sample generation skipped: {e}")

    logger.info("Training completed!")


# === Example usage ===

def example_training():
    """Example of training a score-based VLM."""


    # Create model
    model = create_score_based_nanovlm(
        variant='base',
        mode='joint',
        vocab_size=32000
    )

    # Dummy dataset
    class DummyDataset(keras.utils.Sequence):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            images = keras.random.normal((8, 224, 224, 3))
            text = keras.random.randint((8, 77), minval=0, maxval=32000, dtype='int32')
            return images, text

    dataset = DummyDataset()

    # Train
    train_score_vlm(
        model=model,
        train_dataset=dataset,
        epochs=10,
        log_frequency=10
    )

    logger.info("Example training completed!")


if __name__ == '__main__':
    # Enable mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    example_training()