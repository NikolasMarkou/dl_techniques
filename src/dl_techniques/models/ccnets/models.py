"""
Causal Cooperative Networks (CCNets) - Main Model Implementation

This module implements the main CCNets model that orchestrates the three networks
and implements the cooperative learning algorithm with proper gradient management.

The architecture implements cooperative learning inspired by the Three Kingdoms
philosophy, where three networks collaborate rather than compete to achieve
explainability, reasoning, and faithful data reproduction.

**Mathematical Framework (Corrected for Stability):**

Core Loss Functions:
- classification_loss = CCE(ground_truth_labels, inferred_labels)
- inference_loss = |reconstructed_observation - generated_observation|
- generation_loss = |generated_observation - input_observation|
- reconstruction_loss = |reconstructed_observation - input_observation|

Network-Specific Cooperative Objectives:
- Explainer Objective: classification_loss + generation_loss
- Reasoner Objective: classification_loss + reconstruction_loss
- Producer Objective: generation_loss + reconstruction_loss + inference_loss
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import (
    create_explainer_network,
    create_reasoner_network,
    create_producer_network
)
# ---------------------------------------------------------------------


class CCNetsLoss:
    """
    Loss computation utilities for CCNets implementing the cooperative framework.

    This class implements the three fundamental loss functions that measure
    the consistency and fidelity of the cooperative learning process.
    """

    @staticmethod
    def compute_inference_loss(
        reconstructed_observation: keras.KerasTensor,
        generated_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute inference loss: |reconstructed_observation - generated_observation|
        """
        return ops.mean(ops.abs(reconstructed_observation - generated_observation))

    @staticmethod
    def compute_generation_loss(
        generated_observation: keras.KerasTensor,
        input_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute generation loss: |generated_observation - input_observation|
        """
        return ops.mean(ops.abs(generated_observation - input_observation))

    @staticmethod
    def compute_reconstruction_loss(
        reconstructed_observation: keras.KerasTensor,
        input_observation: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute reconstruction loss: |reconstructed_observation - input_observation|
        """
        return ops.mean(ops.abs(reconstructed_observation - input_observation))

    # This function is kept for potential diagnostic use but is not used in the stable training loop.
    @staticmethod
    def compute_network_errors(
        inference_loss: keras.KerasTensor,
        generation_loss: keras.KerasTensor,
        reconstruction_loss: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Computes the original (unstable) network-specific error functions for monitoring.
        """
        explainer_error = inference_loss + generation_loss - reconstruction_loss
        reasoner_error = reconstruction_loss + inference_loss - generation_loss
        producer_error = generation_loss + reconstruction_loss - inference_loss
        return explainer_error, reasoner_error, producer_error

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CCNetsModel(keras.Model):
    """
    Causal Cooperative Networks (CCNets) Main Model.
    """

    def __init__(
        self,
        explainer_network: keras.Model,
        reasoner_network: keras.Model,
        producer_network: keras.Model,
        loss_weights: Optional[List[float]] = None,
        use_mixed_precision: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.explainer_network = explainer_network
        self.reasoner_network = reasoner_network
        self.producer_network = producer_network

        if loss_weights is None:
            # Add weight for the new classification loss
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        if len(loss_weights) != 4:
            raise ValueError("loss_weights must contain 4 values [classification, inference, generation, reconstruction]")
        if any(w < 0 for w in loss_weights):
            raise ValueError("All loss weights must be non-negative")

        self.loss_weights = loss_weights
        self.use_mixed_precision = use_mixed_precision

        self.loss_computer = CCNetsLoss()
        self.classification_loss_fn = keras.losses.CategoricalCrossentropy()

        # Metrics tracking
        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.inference_loss_tracker = keras.metrics.Mean(name="inference_loss")
        self.generation_loss_tracker = keras.metrics.Mean(name="generation_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

        logger.info("Initialized CCNetsModel with three cooperative networks")

    @property
    def metrics(self):
        """Return list of metrics for monitoring."""
        return [
            self.total_loss_tracker,
            self.classification_loss_tracker,
            self.inference_loss_tracker,
            self.generation_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def call(
        self,
        inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            observations, labels = inputs
        else:
            observations = inputs
            labels = None

        explanation_vector = self.explainer_network(observations, training=training)
        inferred_label = self.reasoner_network(explanation_vector, training=training)

        if labels is not None:
            generated_observation = self.producer_network([labels, explanation_vector], training=training)
        else:
            generated_observation = self.producer_network([inferred_label, explanation_vector], training=training)

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
        generated_observation = outputs['generated_observation']
        reconstructed_observation = outputs['reconstructed_observation']
        inferred_label = outputs['inferred_label']

        classification_loss = self.classification_loss_fn(labels, inferred_label)
        inference_loss = self.loss_computer.compute_inference_loss(reconstructed_observation, generated_observation)
        generation_loss = self.loss_computer.compute_generation_loss(generated_observation, observations)
        reconstruction_loss = self.loss_computer.compute_reconstruction_loss(reconstructed_observation, observations)

        total_loss = (self.loss_weights[0] * classification_loss +
                      self.loss_weights[1] * inference_loss +
                      self.loss_weights[2] * generation_loss +
                      self.loss_weights[3] * reconstruction_loss)

        return {
            'classification_loss': classification_loss,
            'inference_loss': inference_loss,
            'generation_loss': generation_loss,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': total_loss
        }

    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            inputs, _ = data
            observations, labels = inputs
        else:
            raise ValueError("Expected data format: ((observations, labels), targets)")

        # --- Explainer Training ---
        with tf.GradientTape() as tape:
            explanation = self.explainer_network(observations, training=True)
            inferred_label = self.reasoner_network(explanation, training=True)
            generated_obs = self.producer_network([labels, explanation], training=True)

            class_loss = self.classification_loss_fn(labels, inferred_label)
            gen_loss = self.loss_computer.compute_generation_loss(generated_obs, observations)

            explainer_loss = (self.loss_weights[0] * class_loss +
                              self.loss_weights[2] * gen_loss)

        explainer_grads = tape.gradient(explainer_loss, self.explainer_network.trainable_variables)

        # --- Reasoner Training ---
        with tf.GradientTape() as tape:
            explanation = ops.stop_gradient(self.explainer_network(observations, training=True))
            inferred_label = self.reasoner_network(explanation, training=True)
            reconstructed_obs = self.producer_network([inferred_label, explanation], training=True)

            class_loss = self.classification_loss_fn(labels, inferred_label)
            rec_loss = self.loss_computer.compute_reconstruction_loss(reconstructed_obs, observations)

            reasoner_loss = (self.loss_weights[0] * class_loss +
                             self.loss_weights[3] * rec_loss)

        reasoner_grads = tape.gradient(reasoner_loss, self.reasoner_network.trainable_variables)

        # --- Producer Training ---
        with tf.GradientTape() as tape:
            explanation = ops.stop_gradient(self.explainer_network(observations, training=True))
            inferred_label = ops.stop_gradient(self.reasoner_network(explanation, training=True))

            generated_obs = self.producer_network([labels, explanation], training=True)
            reconstructed_obs = self.producer_network([inferred_label, explanation], training=True)

            inf_loss = self.loss_computer.compute_inference_loss(reconstructed_obs, generated_obs)
            gen_loss = self.loss_computer.compute_generation_loss(generated_obs, observations)
            rec_loss = self.loss_computer.compute_reconstruction_loss(reconstructed_obs, observations)

            producer_loss = (self.loss_weights[1] * inf_loss +
                             self.loss_weights[2] * gen_loss +
                             self.loss_weights[3] * rec_loss)

        producer_grads = tape.gradient(producer_loss, self.producer_network.trainable_variables)

        # CORRECTED: Apply all gradients in a single step
        all_grads = explainer_grads + reasoner_grads + producer_grads
        all_vars = (self.explainer_network.trainable_variables +
                    self.reasoner_network.trainable_variables +
                    self.producer_network.trainable_variables)

        self.optimizer.apply_gradients(zip(all_grads, all_vars))

        # Update and return metrics after applying gradients
        outputs = self([observations, labels], training=False)
        losses = self.compute_losses(observations, labels, outputs)

        self.total_loss_tracker.update_state(losses['total_loss'])
        self.classification_loss_tracker.update_state(losses['classification_loss'])
        self.inference_loss_tracker.update_state(losses['inference_loss'])
        self.generation_loss_tracker.update_state(losses['generation_loss'])
        self.reconstruction_loss_tracker.update_state(losses['reconstruction_loss'])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            inputs, _ = data
            observations, labels = inputs
        else:
            raise ValueError("Expected data format: ((observations, labels), targets)")

        outputs = self([observations, labels], training=False)
        losses = self.compute_losses(observations, labels, outputs)

        self.total_loss_tracker.update_state(losses['total_loss'])
        self.classification_loss_tracker.update_state(losses['classification_loss'])
        self.inference_loss_tracker.update_state(losses['inference_loss'])
        self.generation_loss_tracker.update_state(losses['generation_loss'])
        self.reconstruction_loss_tracker.update_state(losses['reconstruction_loss'])

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        if isinstance(data, (tuple, list)):
            observations = data[0]
        else:
            observations = data

        outputs = self(observations, training=False)

        return {
            'predictions': outputs['inferred_label'],
            'explanations': outputs['explanation_vector'],
            'reconstructions': outputs['reconstructed_observation']
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "explainer_network": keras.saving.serialize_keras_object(self.explainer_network),
            "reasoner_network": keras.saving.serialize_keras_object(self.reasoner_network),
            "producer_network": keras.saving.serialize_keras_object(self.producer_network),
            'loss_weights': self.loss_weights,
            'use_mixed_precision': self.use_mixed_precision,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        explainer_network = keras.saving.deserialize_keras_object(config.pop("explainer_network"))
        reasoner_network = keras.saving.deserialize_keras_object(config.pop("reasoner_network"))
        producer_network = keras.saving.deserialize_keras_object(config.pop("producer_network"))
        return cls(explainer_network=explainer_network, reasoner_network=reasoner_network, producer_network=producer_network, **config)

# ---------------------------------------------------------------------

def create_ccnets_model(
    input_dim: int,
    explanation_dim: int,
    output_dim: int,
    explainer_kwargs: Optional[Dict] = None,
    reasoner_kwargs: Optional[Dict] = None,
    producer_kwargs: Optional[Dict] = None,
    loss_weights: Optional[List[float]] = None, # Expects 4 weights now
    use_mixed_precision: bool = False
) -> CCNetsModel:
    explainer_kwargs = explainer_kwargs or {}
    reasoner_kwargs = reasoner_kwargs or {}
    producer_kwargs = producer_kwargs or {}

    explainer = create_explainer_network(input_dim=input_dim, explanation_dim=explanation_dim, **explainer_kwargs)
    reasoner = create_reasoner_network(explanation_dim=explanation_dim, output_dim=output_dim, **reasoner_kwargs)
    producer = create_producer_network(label_dim=output_dim, explanation_dim=explanation_dim, output_dim=input_dim, **producer_kwargs)

    model = CCNetsModel(
        explainer_network=explainer,
        reasoner_network=reasoner,
        producer_network=producer,
        loss_weights=loss_weights,
        use_mixed_precision=use_mixed_precision
    )
    logger.info(f"Created CCNetsModel: input_dim={input_dim}, explanation_dim={explanation_dim}, output_dim={output_dim}")
    return model

# ---------------------------------------------------------------------