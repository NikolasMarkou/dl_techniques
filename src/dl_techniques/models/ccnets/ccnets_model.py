"""
Causal Cooperative Networks (CCNets) - Main Model Implementation

This module implements the main CCNets model that orchestrates the three networks
and implements the cooperative learning algorithm with proper gradient management.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import keras
from keras import ops
from dl_techniques.utils.logger import logger


class CCNetsLoss:
    """
    Loss computation utilities for CCNets.

    Implements the three fundamental loss functions:
    - Inference Loss: Consistency between reconstructed and generated observations
    - Generation Loss: Fidelity of generated output to original input
    - Reconstruction Loss: Fidelity of reconstructed output to original input
    """

    @staticmethod
    def compute_inference_loss(
            reconstructed_observation: keras.KerasTensor,
            generated_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute inference loss: |reconstructed_observation - generated_observation|

        This measures consistency between the Producer network's two pathways.
        Lower values indicate better agreement between reasoning and generation paths.

        Args:
            reconstructed_observation: Output from Producer(y', e)
            generated_observation: Output from Producer(y, e)

        Returns:
            Scalar inference loss
        """
        return ops.mean(ops.abs(reconstructed_observation - generated_observation))

    @staticmethod
    def compute_generation_loss(
            generated_observation: keras.KerasTensor,
            input_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute generation loss: |generated_observation - input_observation|

        This measures how well the Producer can generate data from ground truth labels.
        Lower values indicate better generation capability.

        Args:
            generated_observation: Output from Producer(y, e)
            input_observation: Original input data

        Returns:
            Scalar generation loss
        """
        return ops.mean(ops.abs(generated_observation - input_observation))

    @staticmethod
    def compute_reconstruction_loss(
            reconstructed_observation: keras.KerasTensor,
            input_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute reconstruction loss: |reconstructed_observation - input_observation|

        This measures how well the Producer can reconstruct data from predicted labels.
        Lower values indicate better reconstruction capability.

        Args:
            reconstructed_observation: Output from Producer(y', e)
            input_observation: Original input data

        Returns:
            Scalar reconstruction loss
        """
        return ops.mean(ops.abs(reconstructed_observation - input_observation))

    @staticmethod
    def compute_network_errors(
            inference_loss: keras.KerasTensor,
            generation_loss: keras.KerasTensor,
            reconstruction_loss: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Compute network-specific error functions for cooperative learning.

        The error functions are designed to promote cooperation:
        - Explainer improves when inference and generation improve, but reduces reconstruction dependency
        - Reasoner improves when reconstruction and inference improve, but reduces generation dependency
        - Producer improves when generation and reconstruction improve, but reduces inference dependency

        Args:
            inference_loss: Consistency loss between reconstruction and generation
            generation_loss: Fidelity loss for generation
            reconstruction_loss: Fidelity loss for reconstruction

        Returns:
            Tuple of (explainer_error, reasoner_error, producer_error)
        """
        explainer_error = inference_loss + generation_loss - reconstruction_loss
        reasoner_error = reconstruction_loss + inference_loss - generation_loss
        producer_error = generation_loss + reconstruction_loss - inference_loss

        return explainer_error, reasoner_error, producer_error


class CCNetsModel(keras.Model):
    """
    Causal Cooperative Networks (CCNets) Main Model.

    Implements the three-network cooperative learning architecture inspired by
    the Three Kingdoms political philosophy. The model combines explanation,
    reasoning, and faithful reproduction through collaboration rather than competition.

    Args:
        explainer_network: Network that creates explanation vectors from inputs
        reasoner_network: Network that makes predictions from inputs and explanations
        producer_network: Network that generates/reconstructs data from labels and explanations
        loss_weights: Weights for different loss components [inference, generation, reconstruction]
        **kwargs: Additional arguments for keras.Model
    """

    def __init__(
            self,
            explainer_network: keras.Model,
            reasoner_network: keras.Model,
            producer_network: keras.Model,
            loss_weights: Optional[List[float]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.explainer_network = explainer_network
        self.reasoner_network = reasoner_network
        self.producer_network = producer_network
        self.loss_weights = loss_weights or [1.0, 1.0, 1.0]  # [inference, generation, reconstruction]

        # Loss computation utilities
        self.loss_computer = CCNetsLoss()

        logger.info("Initialized CCNetsModel with three cooperative networks")

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through all three networks.

        Args:
            inputs: Tuple of (observations, labels)
                observations: Input data [batch_size, input_dim]
                labels: Ground truth labels [batch_size, label_dim]
            training: Whether in training mode

        Returns:
            Dictionary containing:
                - explanation_vector: Output from Explainer
                - inferred_label: Output from Reasoner
                - generated_observation: Output from Producer(y, e)
                - reconstructed_observation: Output from Producer(y', e)
        """
        observations, labels = inputs

        # Step 1: Explainer Network - Create explanation vector
        explanation_vector = self.explainer_network(observations, training=training)

        # Step 2: Reasoner Network - Make predictions
        inferred_label = self.reasoner_network([observations, explanation_vector], training=training)

        # Step 3: Producer Network - Generate and reconstruct
        # Generation path: use ground truth labels
        generated_observation = self.producer_network([labels, explanation_vector], training=training)

        # Reconstruction path: use inferred labels
        reconstructed_observation = self.producer_network([inferred_label, explanation_vector], training=training)

        return {
            'explanation_vector': explanation_vector,
            'inferred_label': inferred_label,
            'generated_observation': generated_observation,
            'reconstructed_observation': reconstructed_observation
        }

    def compute_losses(
            self,
            observations: keras.KerasTensor,
            labels: keras.KerasTensor,
            outputs: Dict[str, keras.KerasTensor]
    ) -> Dict[str, keras.KerasTensor]:
        """
        Compute all loss components for CCNets.

        Args:
            observations: Original input observations
            labels: Ground truth labels
            outputs: Dictionary of network outputs from forward pass

        Returns:
            Dictionary containing all loss components and network errors
        """
        # Extract outputs
        generated_observation = outputs['generated_observation']
        reconstructed_observation = outputs['reconstructed_observation']

        # Compute fundamental losses
        inference_loss = self.loss_computer.compute_inference_loss(
            reconstructed_observation, generated_observation
        )
        generation_loss = self.loss_computer.compute_generation_loss(
            generated_observation, observations
        )
        reconstruction_loss = self.loss_computer.compute_reconstruction_loss(
            reconstructed_observation, observations
        )

        # Apply loss weights
        weighted_inference_loss = self.loss_weights[0] * inference_loss
        weighted_generation_loss = self.loss_weights[1] * generation_loss
        weighted_reconstruction_loss = self.loss_weights[2] * reconstruction_loss

        # Compute network-specific errors
        explainer_error, reasoner_error, producer_error = self.loss_computer.compute_network_errors(
            weighted_inference_loss, weighted_generation_loss, weighted_reconstruction_loss
        )

        # Total loss for monitoring
        total_loss = weighted_inference_loss + weighted_generation_loss + weighted_reconstruction_loss

        return {
            'inference_loss': inference_loss,
            'generation_loss': generation_loss,
            'reconstruction_loss': reconstruction_loss,
            'weighted_inference_loss': weighted_inference_loss,
            'weighted_generation_loss': weighted_generation_loss,
            'weighted_reconstruction_loss': weighted_reconstruction_loss,
            'explainer_error': explainer_error,
            'reasoner_error': reasoner_error,
            'producer_error': producer_error,
            'total_loss': total_loss
        }

    def train_step(self, data):
        """
        Custom training step implementing cooperative learning.

        This implements the core CCNets training algorithm with proper gradient isolation
        to ensure each network optimizes its own cooperative objective.

        Args:
            data: Tuple of (inputs, targets) where inputs are (observations, labels)

        Returns:
            Dictionary of metrics for this training step
        """
        (observations, labels), targets = data

        # Forward pass through all networks
        with keras.utils.custom_object_scope():
            outputs = self([observations, labels], training=True)

        # Compute all losses
        losses = self.compute_losses(observations, labels, outputs)

        # Explainer Network Training
        with keras.utils.custom_object_scope():
            # Create detached versions for proper gradient isolation
            detached_reconstructed = ops.stop_gradient(outputs['reconstructed_observation'])
            detached_generated = ops.stop_gradient(outputs['generated_observation'])

            # Recompute explainer error with gradients
            explainer_vars = self.explainer_network.trainable_variables
            with keras.backend.GradientTape() as explainer_tape:
                # Forward pass for explainer
                explanation = self.explainer_network(observations, training=True)

                # Generate outputs using current explanation
                inferred = self.reasoner_network([observations, explanation], training=True)
                generated = self.producer_network([labels, explanation], training=True)
                reconstructed = self.producer_network([inferred, explanation], training=True)

                # Compute explainer-specific losses
                inf_loss = self.loss_computer.compute_inference_loss(reconstructed, generated)
                gen_loss = self.loss_computer.compute_generation_loss(generated, observations)
                rec_loss = self.loss_computer.compute_reconstruction_loss(reconstructed, observations)

                explainer_loss = inf_loss + gen_loss - rec_loss

            # Apply gradients to explainer
            explainer_grads = explainer_tape.gradient(explainer_loss, explainer_vars)
            if explainer_grads:
                self.optimizer.apply_gradients(zip(explainer_grads, explainer_vars))

        # Reasoner Network Training
        with keras.utils.custom_object_scope():
            reasoner_vars = self.reasoner_network.trainable_variables
            with keras.backend.GradientTape() as reasoner_tape:
                # Forward pass for reasoner
                explanation = ops.stop_gradient(self.explainer_network(observations, training=True))
                inferred = self.reasoner_network([observations, explanation], training=True)
                generated = ops.stop_gradient(self.producer_network([labels, explanation], training=True))
                reconstructed = self.producer_network([inferred, explanation], training=True)

                # Compute reasoner-specific losses
                inf_loss = self.loss_computer.compute_inference_loss(reconstructed, generated)
                gen_loss = self.loss_computer.compute_generation_loss(generated, observations)
                rec_loss = self.loss_computer.compute_reconstruction_loss(reconstructed, observations)

                reasoner_loss = rec_loss + inf_loss - gen_loss

            # Apply gradients to reasoner
            reasoner_grads = reasoner_tape.gradient(reasoner_loss, reasoner_vars)
            if reasoner_grads:
                self.optimizer.apply_gradients(zip(reasoner_grads, reasoner_vars))

        # Producer Network Training
        with keras.utils.custom_object_scope():
            producer_vars = self.producer_network.trainable_variables
            with keras.backend.GradientTape() as producer_tape:
                # Forward pass for producer
                explanation = ops.stop_gradient(self.explainer_network(observations, training=True))
                inferred = ops.stop_gradient(self.reasoner_network([observations, explanation], training=True))
                generated = self.producer_network([labels, explanation], training=True)
                reconstructed = self.producer_network([inferred, explanation], training=True)

                # Compute producer-specific losses
                inf_loss = self.loss_computer.compute_inference_loss(reconstructed, generated)
                gen_loss = self.loss_computer.compute_generation_loss(generated, observations)
                rec_loss = self.loss_computer.compute_reconstruction_loss(reconstructed, observations)

                producer_loss = gen_loss + rec_loss - inf_loss

            # Apply gradients to producer
            producer_grads = producer_tape.gradient(producer_loss, producer_vars)
            if producer_grads:
                self.optimizer.apply_gradients(zip(producer_grads, producer_vars))

        # Return metrics
        return {
            'loss': losses['total_loss'],
            'inference_loss': losses['inference_loss'],
            'generation_loss': losses['generation_loss'],
            'reconstruction_loss': losses['reconstruction_loss'],
            'explainer_error': losses['explainer_error'],
            'reasoner_error': losses['reasoner_error'],
            'producer_error': losses['producer_error']
        }

    def test_step(self, data):
        """Custom test step for evaluation."""
        (observations, labels), targets = data

        # Forward pass
        outputs = self([observations, labels], training=False)

        # Compute losses
        losses = self.compute_losses(observations, labels, outputs)

        return {
            'loss': losses['total_loss'],
            'inference_loss': losses['inference_loss'],
            'generation_loss': losses['generation_loss'],
            'reconstruction_loss': losses['reconstruction_loss']
        }

    def predict_step(self, data):
        """Custom prediction step."""
        observations = data

        # For prediction, we only need explainer and reasoner
        explanation_vector = self.explainer_network(observations, training=False)
        inferred_label = self.reasoner_network([observations, explanation_vector], training=False)

        return {
            'predictions': inferred_label,
            'explanations': explanation_vector
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'loss_weights': self.loss_weights,
        })
        return config


def create_ccnets_model(
        input_dim: int,
        explanation_dim: int,
        output_dim: int,
        explainer_kwargs: Optional[Dict] = None,
        reasoner_kwargs: Optional[Dict] = None,
        producer_kwargs: Optional[Dict] = None,
        loss_weights: Optional[List[float]] = None
) -> CCNetsModel:
    """
    Factory function to create a complete CCNets model.

    Args:
        input_dim: Dimensionality of input observations
        explanation_dim: Dimensionality of explanation vectors
        output_dim: Dimensionality of predictions/labels
        explainer_kwargs: Additional arguments for ExplainerNetwork
        reasoner_kwargs: Additional arguments for ReasonerNetwork
        producer_kwargs: Additional arguments for ProducerNetwork
        loss_weights: Weights for [inference, generation, reconstruction] losses

    Returns:
        Configured CCNetsModel instance ready for training
    """
    from .ccnets_base import (
        create_explainer_network,
        create_reasoner_network,
        create_producer_network
    )

    # Create individual networks
    explainer = create_explainer_network(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        **(explainer_kwargs or {})
    )

    reasoner = create_reasoner_network(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        **(reasoner_kwargs or {})
    )

    producer = create_producer_network(
        label_dim=output_dim,
        explanation_dim=explanation_dim,
        output_dim=input_dim,
        **(producer_kwargs or {})
    )

    # Create CCNets model
    model = CCNetsModel(
        explainer_network=explainer,
        reasoner_network=reasoner,
        producer_network=producer,
        loss_weights=loss_weights
    )

    logger.info(
        f"Created CCNetsModel: input_dim={input_dim}, explanation_dim={explanation_dim}, output_dim={output_dim}")

    return model