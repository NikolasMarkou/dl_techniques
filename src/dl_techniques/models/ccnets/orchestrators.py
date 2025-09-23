import keras
import tensorflow as tf
import warnings
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .losses import L1Loss, L2Loss, HuberLoss, PolynomialLoss
from .base import CCNetModule, CCNetConfig, CCNetLosses, CCNetModelErrors
from .control import ConvergenceControlStrategy, StaticThresholdStrategy


# ---------------------------------------------------------------------

class CCNetOrchestrator:
    """
    Main orchestrator for Causal Cooperative Networks.
    Manages the three neural modules and their cooperative training.
    """

    def __init__(
            self,
            explainer: CCNetModule,
            reasoner: CCNetModule,
            producer: CCNetModule,
            config: Optional[CCNetConfig] = None,
            control_strategy: Optional[ConvergenceControlStrategy] = None
    ):
        """
        Initialize the CCNet orchestrator.

        Args:
            explainer: Neural network module modeling P(E|X). Must output a
                       tuple of (mu, log_var) for latent space regularization.
            reasoner: Neural network module modeling P(Y|X,E).
            producer: Neural network module modeling P(X|Y,E).
            config: Configuration object for the framework.
            control_strategy: A strategy object that determines when to train the
                              Reasoner to prevent premature convergence. Defaults
                              to StaticThresholdStrategy for backward compatibility.
        """
        self.explainer = explainer
        self.reasoner = reasoner
        self.producer = producer
        self.config = config or CCNetConfig()
        self.control = control_strategy or StaticThresholdStrategy()

        # Initialize core components
        self._init_loss_function()
        self._init_optimizers()

        # Setup mixed precision if requested
        if self.config.use_mixed_precision:
            self._setup_mixed_precision()

    def _init_loss_function(self):
        """
        Initialize the loss function based on configuration, providing
        backward compatibility for the legacy 'loss_type' attribute.
        """
        loss_name = None
        if hasattr(self.config, 'loss_fn'):
            loss_name = self.config.loss_fn
        elif hasattr(self.config, 'loss_type'):
            loss_name = self.config.loss_type
            warnings.warn(
                "'loss_type' is deprecated. Please update your CCNetConfig to use 'loss_fn' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        else:
            # Default to 'l2' if neither is found, matching the config default
            loss_name = 'l2'

        loss_map = {
            'l1': L1Loss,
            'l2': L2Loss,
            'huber': HuberLoss,
            'polynomial': PolynomialLoss
        }
        loss_class = loss_map.get(loss_name)
        if loss_class is None:
            raise ValueError(f"Unsupported loss function '{loss_name}'. "
                             f"Supported types are: {list(loss_map.keys())}")

        # Instantiate the class with parameters from config
        self.loss_fn = loss_class(**self.config.loss_fn_params)

    def _init_optimizers(self):
        """Initialize optimizers for each module."""
        self.optimizers = {
            'explainer': keras.optimizers.Adam(learning_rate=self.config.learning_rates['explainer']),
            'reasoner': keras.optimizers.Adam(learning_rate=self.config.learning_rates['reasoner']),
            'producer': keras.optimizers.Adam(learning_rate=self.config.learning_rates['producer'])
        }

    def _setup_mixed_precision(self):
        """Setup mixed precision training if configured."""
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

    def forward_pass(
            self,
            x_input: tf.Tensor,
            y_truth: tf.Tensor,
            training: bool = True
    ) -> Dict[str, tf.Tensor]:
        """
        Perform a complete forward pass through the CCNet.

        Args:
            x_input: Input observation tensor.
            y_truth: Ground truth label tensor.
            training: Whether in training mode.

        Returns:
            Dictionary containing all intermediate tensors.
        """
        # Step 1: Extract latent explanation distribution (mu, log_var)
        mu, log_var = self.explainer(x_input, training=training)

        # Step 1.5: Sample from the latent distribution using reparameterization trick
        std = keras.ops.exp(0.5 * log_var)
        epsilon = keras.random.normal(shape=keras.ops.shape(mu))
        e_latent = mu + epsilon * std

        # Step 2: Infer label using observation and explanation
        y_inferred = self.reasoner(x_input, e_latent, training=training)

        # Step 3: Reconstruct observation from inferred label and explanation
        x_reconstructed = self.producer(y_inferred, e_latent, training=training)

        # Step 4: Generate observation from true label and explanation
        x_generated = self.producer(y_truth, e_latent, training=training)

        return {
            'x_input': x_input,
            'y_truth': y_truth,
            'mu': mu,
            'log_var': log_var,
            'e_latent': e_latent,
            'y_inferred': y_inferred,
            'x_reconstructed': x_reconstructed,
            'x_generated': x_generated
        }

    def compute_losses(self, tensors: Dict[str, tf.Tensor]) -> CCNetLosses:
        """
        Compute the three fundamental CCNet losses.

        Args:
            tensors: Dictionary of tensors from forward pass.

        Returns:
            CCNetLosses object containing the three losses.
        """
        generation_loss = self.loss_fn(tensors['x_generated'], tensors['x_input'])
        reconstruction_loss = self.loss_fn(tensors['x_reconstructed'], tensors['x_input'])
        inference_loss = self.loss_fn(tensors['x_reconstructed'], tensors['x_generated'])

        return CCNetLosses(
            generation_loss=generation_loss,
            reconstruction_loss=reconstruction_loss,
            inference_loss=inference_loss
        )

    def compute_model_errors(
            self,
            losses: CCNetLosses,
            tensors: Dict[str, tf.Tensor]
    ) -> CCNetModelErrors:
        """
        Compute model-specific error signals from the fundamental losses.
        This uses a stable, additive formulation and includes KL regularization.

        Args:
            losses: CCNetLosses object.
            tensors: Dictionary of tensors from forward pass, for KL loss calc.

        Returns:
            CCNetModelErrors object containing error signals for each module.
        """
        # KL divergence loss to regularize the latent space
        kl_loss = -0.5 * keras.ops.mean(
            1 + tensors['log_var'] - keras.ops.square(tensors['mu']) - keras.ops.exp(tensors['log_var'])
        )

        explainer_error = (
                self.config.explainer_weights['inference'] * losses.inference_loss +
                self.config.explainer_weights['generation'] * losses.generation_loss +
                self.config.kl_weight * kl_loss
        )
        reasoner_error = (
                self.config.reasoner_weights['reconstruction'] * losses.reconstruction_loss +
                self.config.reasoner_weights['inference'] * losses.inference_loss
        )
        producer_error = (
                self.config.producer_weights['generation'] * losses.generation_loss +
                self.config.producer_weights['reconstruction'] * losses.reconstruction_loss
        )

        return CCNetModelErrors(
            explainer_error=explainer_error,
            reasoner_error=reasoner_error,
            producer_error=producer_error
        )

    @tf.function
    def train_step(
            self,
            x_input: tf.Tensor,
            y_truth: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step with causal credit assignment.
        This implementation is compiled to a static graph for performance.

        Args:
            x_input: Input observation batch.
            y_truth: Ground truth label batch.

        Returns:
            Dictionary of scalar loss, error, and gradient norm tensors.
        """
        with tf.GradientTape() as explainer_tape, \
                tf.GradientTape() as reasoner_tape, \
                tf.GradientTape() as producer_tape:

            tensors = self.forward_pass(x_input, y_truth, training=True)
            losses = self.compute_losses(tensors)
            errors = self.compute_model_errors(losses, tensors)

        # Calculate batch accuracy for control strategy
        y_true_labels = keras.ops.argmax(y_truth, axis=-1)
        y_pred_labels = keras.ops.argmax(tensors['y_inferred'], axis=-1)
        batch_accuracy = keras.ops.mean(
            keras.ops.cast(keras.ops.equal(y_true_labels, y_pred_labels), dtype='float32')
        )

        # Delegate training decision to the control strategy (read-only in graph)
        current_metrics = {'batch_accuracy': batch_accuracy}
        train_reasoner = self.control.should_train_reasoner(current_metrics)

        # Compute gradients for all modules
        explainer_vars = self.explainer.trainable_variables
        explainer_grads = explainer_tape.gradient(errors.explainer_error, explainer_vars)
        producer_vars = self.producer.trainable_variables
        producer_grads = producer_tape.gradient(errors.producer_error, producer_vars)
        reasoner_vars = self.reasoner.trainable_variables

        if train_reasoner:
            reasoner_grads = reasoner_tape.gradient(errors.reasoner_error, reasoner_vars)
        else:
            reasoner_grads = [tf.zeros_like(v) for v in reasoner_vars]

        # Compute gradient norms for monitoring
        explainer_grad_norm = tf.linalg.global_norm(explainer_grads)
        reasoner_grad_norm = tf.linalg.global_norm(reasoner_grads)
        producer_grad_norm = tf.linalg.global_norm(producer_grads)

        # Apply gradient clipping if configured
        if self.config.gradient_clip_norm:
            explainer_grads = [tf.clip_by_norm(g, self.config.gradient_clip_norm) if g is not None else None for g in
                               explainer_grads]
            reasoner_grads = [tf.clip_by_norm(g, self.config.gradient_clip_norm) if g is not None else None for g in
                              reasoner_grads]
            producer_grads = [tf.clip_by_norm(g, self.config.gradient_clip_norm) if g is not None else None for g in
                              producer_grads]

        # Apply gradients
        self.optimizers['explainer'].apply_gradients(zip(explainer_grads, explainer_vars))
        self.optimizers['producer'].apply_gradients(zip(producer_grads, producer_vars))
        if train_reasoner:
            self.optimizers['reasoner'].apply_gradients(zip(reasoner_grads, reasoner_vars))

        # Return a comprehensive dictionary of results
        return {
            'generation_loss': losses.generation_loss,
            'reconstruction_loss': losses.reconstruction_loss,
            'inference_loss': losses.inference_loss,
            'explainer_error': errors.explainer_error,
            'reasoner_error': errors.reasoner_error,
            'producer_error': errors.producer_error,
            'explainer_grad_norm': explainer_grad_norm,
            'reasoner_grad_norm': reasoner_grad_norm,
            'producer_grad_norm': producer_grad_norm,
            'batch_accuracy': batch_accuracy,
            'reasoner_is_training': tf.cast(train_reasoner, dtype=tf.float32)
        }

    def evaluate(
            self,
            x_input: tf.Tensor,
            y_truth: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Evaluate the CCNet without training.

        Args:
            x_input: Input observation batch.
            y_truth: Ground truth label batch.

        Returns:
            Dictionary of scalar loss and error tensors.
        """
        tensors = self.forward_pass(x_input, y_truth, training=False)
        losses = self.compute_losses(tensors)
        errors = self.compute_model_errors(losses, tensors)

        return {
            'generation_loss': losses.generation_loss,
            'reconstruction_loss': losses.reconstruction_loss,
            'inference_loss': losses.inference_loss,
            'explainer_error': errors.explainer_error,
            'reasoner_error': errors.reasoner_error,
            'producer_error': errors.producer_error
        }

    def counterfactual_generation(
            self,
            x_reference: tf.Tensor,
            y_target: tf.Tensor
    ) -> tf.Tensor:
        """
        Generate counterfactual observations.

        Args:
            x_reference: Reference observation (provides style).
            y_target: Target label (what to generate).

        Returns:
            Generated counterfactual observation.
        """
        mu, _ = self.explainer(x_reference, training=False)
        e_style = mu
        x_counterfactual = self.producer(y_target, e_style, training=False)
        return x_counterfactual

    def style_transfer(
            self,
            x_content: tf.Tensor,
            x_style: tf.Tensor
    ) -> tf.Tensor:
        """
        Perform style transfer between observations.

        Args:
            x_content: Observation providing the content (label).
            x_style: Observation providing the style.

        Returns:
            Generated observation with content from x_content and style from x_style.
        """
        mu_style, _ = self.explainer(x_style, training=False)
        e_style = mu_style
        mu_content, _ = self.explainer(x_content, training=False)
        e_content_latent = mu_content
        y_content = self.reasoner(x_content, e_content_latent, training=False)
        x_transferred = self.producer(y_content, e_style, training=False)
        return x_transferred

    def disentangle_causes(
            self,
            x_input: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Disentangle the explicit and latent causes of an observation.

        Args:
            x_input: Input observation.

        Returns:
            Tuple of (explicit_cause, latent_cause).
        """
        mu, _ = self.explainer(x_input, training=False)
        e_latent = mu
        y_explicit = self.reasoner(x_input, e_latent, training=False)
        return y_explicit, e_latent

    def verify_consistency(
            self,
            x_input: tf.Tensor,
            threshold: float = 0.01
    ) -> bool:
        """
        Verify the internal consistency of the CCNet's reasoning.

        Args:
            x_input: Input observation to verify.
            threshold: Maximum allowed reconstruction error.

        Returns:
            True if the network's reasoning is consistent.
        """
        y_explicit, e_latent = self.disentangle_causes(x_input)
        x_reconstructed = self.producer(y_explicit, e_latent, training=False)
        error = keras.ops.mean(keras.ops.abs(x_reconstructed - x_input))
        return bool(keras.ops.convert_to_numpy(error) < threshold)

    def save_models(self, base_path: str):
        """
        Save all three models to disk.
        """
        self.explainer.save(f"{base_path}_explainer.keras")
        self.reasoner.save(f"{base_path}_reasoner.keras")
        self.producer.save(f"{base_path}_producer.keras")

    def load_models(self, base_path: str):
        """
        Load all three models from disk.
        """
        self.explainer = keras.models.load_model(f"{base_path}_explainer.keras")
        self.reasoner = keras.models.load_model(f"{base_path}_reasoner.keras")
        self.producer = keras.models.load_model(f"{base_path}_producer.keras")


# ---------------------------------------------------------------------

class SequentialCCNetOrchestrator(CCNetOrchestrator):
    """
    Extended orchestrator for sequential data.
    Implements reverse causality for the Producer via sequence reversal.
    """

    def __init__(
            self,
            explainer: CCNetModule,
            reasoner: CCNetModule,
            producer: CCNetModule,
            config: Optional[CCNetConfig] = None,
            control_strategy: Optional[ConvergenceControlStrategy] = None
    ):
        super().__init__(explainer, reasoner, producer, config, control_strategy)
        if not self.config.sequential_data:
            self.config.sequential_data = True

    def reverse_sequence(self, sequence: tf.Tensor) -> tf.Tensor:
        """
        Reverse a sequence along the time dimension (axis 1).
        """
        return keras.ops.flip(sequence, axis=1)

    def forward_pass(
            self,
            x_input: tf.Tensor,
            y_truth: tf.Tensor,
            training: bool = True
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass with sequential data handling. The Producer uses the
        sequence reversal trick to achieve reverse-causal modeling.
        """
        # Standard forward pass for Explainer and Reasoner
        mu, log_var = self.explainer(x_input, training=training)
        std = keras.ops.exp(0.5 * log_var)
        epsilon = keras.random.normal(shape=keras.ops.shape(mu))
        e_latent = mu + epsilon * std
        y_inferred = self.reasoner(x_input, e_latent, training=training)

        # Producer uses reverse causality via sequence reversal trick
        y_inferred_rev = self.reverse_sequence(y_inferred)
        y_truth_rev = self.reverse_sequence(y_truth)
        e_latent_rev = self.reverse_sequence(e_latent)
        x_reconstructed_rev = self.producer(y_inferred_rev, e_latent_rev, training=training)
        x_generated_rev = self.producer(y_truth_rev, e_latent_rev, training=training)

        # Reverse outputs back to original order
        x_reconstructed = self.reverse_sequence(x_reconstructed_rev)
        x_generated = self.reverse_sequence(x_generated_rev)

        return {
            'x_input': x_input,
            'y_truth': y_truth,
            'mu': mu,
            'log_var': log_var,
            'e_latent': e_latent,
            'y_inferred': y_inferred,
            'x_reconstructed': x_reconstructed,
            'x_generated': x_generated
        }