"""Tests for MANNUtilizationLoss and GNNUtilizationLoss."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.losses.utilization_loss import (
    MANNUtilizationLoss,
    GNNUtilizationLoss,
)


class TestMANNUtilizationLoss:
    """Tests for MANNUtilizationLoss."""

    def test_init_defaults(self):
        loss = MANNUtilizationLoss()
        assert loss.name == "mann_utilization_loss"
        assert loss.entropy_weight == 0.01
        assert loss.write_weight == 0.01
        assert loss.variance_weight == 0.01

    def test_init_custom(self):
        loss = MANNUtilizationLoss(
            entropy_weight=0.1, write_weight=0.05, variance_weight=0.02
        )
        assert loss.entropy_weight == 0.1
        assert loss.write_weight == 0.05
        assert loss.variance_weight == 0.02

    def test_forward_pass(self):
        """y_pred is treated as the memory vectors tensor."""
        loss_fn = MANNUtilizationLoss()
        batch_size, time_steps, memory_dim = 4, 10, 32
        y_true = np.zeros((batch_size, time_steps, memory_dim), dtype=np.float32)
        memory_vectors = np.random.rand(
            batch_size, time_steps, memory_dim
        ).astype(np.float32)
        loss = loss_fn(y_true, memory_vectors)
        assert np.isfinite(float(loss))

    def test_collapsed_memory_higher_loss(self):
        """Collapsed (constant) memory should produce higher loss than diverse memory."""
        loss_fn = MANNUtilizationLoss(
            entropy_weight=1.0, write_weight=1.0, variance_weight=1.0
        )
        batch_size, time_steps, memory_dim = 4, 10, 16

        # Collapsed: near-constant memory
        collapsed = np.ones(
            (batch_size, time_steps, memory_dim), dtype=np.float32
        ) * 0.01 + np.random.rand(batch_size, time_steps, memory_dim).astype(np.float32) * 0.001

        # Diverse: varied memory patterns
        diverse = np.random.rand(
            batch_size, time_steps, memory_dim
        ).astype(np.float32) * 5.0

        dummy = np.zeros((batch_size, time_steps, memory_dim), dtype=np.float32)
        loss_collapsed = float(loss_fn(dummy, collapsed))
        loss_diverse = float(loss_fn(dummy, diverse))

        # Diverse memory should have lower (more negative) utilization loss
        assert loss_diverse < loss_collapsed

    def test_get_config_roundtrip(self):
        loss = MANNUtilizationLoss(
            entropy_weight=0.1, write_weight=0.05, variance_weight=0.02
        )
        config = loss.get_config()
        restored = MANNUtilizationLoss.from_config(config)
        assert restored.entropy_weight == 0.1
        assert restored.write_weight == 0.05
        assert restored.variance_weight == 0.02

    def test_serialization(self):
        loss = MANNUtilizationLoss(entropy_weight=0.5)
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, MANNUtilizationLoss)
        assert restored.entropy_weight == 0.5


class TestGNNUtilizationLoss:
    """Tests for GNNUtilizationLoss."""

    def test_init_defaults(self):
        loss = GNNUtilizationLoss()
        assert loss.name == "gnn_utilization_loss"
        assert loss.diversity_weight == 0.01
        assert loss.attention_weight == 0.01
        assert loss.activation_weight == 0.01

    def test_forward_pass(self):
        """y_pred is treated as the entity embeddings tensor."""
        loss_fn = GNNUtilizationLoss()
        batch_size, num_entities, embed_dim = 4, 8, 32
        y_true = np.zeros((batch_size, num_entities, embed_dim), dtype=np.float32)
        entity_embeddings = np.random.rand(
            batch_size, num_entities, embed_dim
        ).astype(np.float32)
        loss = loss_fn(y_true, entity_embeddings)
        assert np.isfinite(float(loss))

    def test_collapsed_entities_higher_loss(self):
        """Collapsed (identical) entities should produce higher loss than diverse ones."""
        loss_fn = GNNUtilizationLoss(
            diversity_weight=1.0, activation_weight=1.0, attention_weight=1.0
        )
        batch_size, num_entities, embed_dim = 4, 8, 16

        # Collapsed: near-identical entities
        base = np.random.rand(batch_size, 1, embed_dim).astype(np.float32) * 0.01
        collapsed = np.tile(base, (1, num_entities, 1))
        collapsed += np.random.rand(batch_size, num_entities, embed_dim).astype(np.float32) * 0.001

        # Diverse: distinct entities with large norms
        diverse = np.random.rand(
            batch_size, num_entities, embed_dim
        ).astype(np.float32) * 5.0

        dummy = np.zeros((batch_size, num_entities, embed_dim), dtype=np.float32)
        loss_collapsed = float(loss_fn(dummy, collapsed))
        loss_diverse = float(loss_fn(dummy, diverse))

        assert loss_diverse < loss_collapsed

    def test_get_config_roundtrip(self):
        loss = GNNUtilizationLoss(
            diversity_weight=0.1, attention_weight=0.05, activation_weight=0.02
        )
        config = loss.get_config()
        restored = GNNUtilizationLoss.from_config(config)
        assert restored.diversity_weight == 0.1
        assert restored.attention_weight == 0.05
        assert restored.activation_weight == 0.02

    def test_serialization(self):
        loss = GNNUtilizationLoss(diversity_weight=0.5)
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, GNNUtilizationLoss)
        assert restored.diversity_weight == 0.5
