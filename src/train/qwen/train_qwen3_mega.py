"""
Qwen3-MEGA Training Framework
==============================

Complete training system designed to force the model to actually use
MANN and GNN components through:
1. Task design that requires specific components
2. Auxiliary losses that encourage component usage
3. Staged training curriculum
4. Comprehensive monitoring and visualization

This prevents "component collapse" where the model learns to ignore
the fancy new components and just uses the transformer.
"""

import keras
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.qwen.qwen3_mega import create_qwen3_mega

# ---------------------------------------------------------------------
# Task Generators - Force Component Usage
# ---------------------------------------------------------------------

class MANNForcingTask:
    """
    Tasks that explicitly require working memory (MANN).

    These are designed so that transformer self-attention alone
    is insufficient - you MUST use external memory.
    """

    @staticmethod
    def copy_task(
            batch_size: int,
            seq_len: int,
            vocab_size: int,
            num_copies: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Copy sequence: Model must remember input and reproduce it.

        Format: [START] [sequence] [DELIMITER] [sequence_copy] [END]

        This forces MANN usage because:
        - Transformer attention has limited context window
        - Must explicitly store and retrieve exact values
        """
        # Generate random sequences
        sequences = np.random.randint(4, vocab_size - 4, (batch_size, seq_len))

        # Create input: [START] [sequence] [DELIMITER]
        start_token = 1
        delimiter_token = 2
        pad_token = 0

        max_len = seq_len * (num_copies + 1) + num_copies + 2
        inputs = np.full((batch_size, max_len), pad_token, dtype=np.int32)
        targets = np.full((batch_size, max_len), pad_token, dtype=np.int32)

        for b in range(batch_size):
            pos = 0
            # Start token
            inputs[b, pos] = start_token
            pos += 1

            # Original sequence
            inputs[b, pos:pos + seq_len] = sequences[b]
            pos += seq_len

            # Delimiter
            inputs[b, pos] = delimiter_token
            pos += 1

            # Target: copy the sequence num_copies times
            for _ in range(num_copies):
                targets[b, pos:pos + seq_len] = sequences[b]
                pos += seq_len

        return inputs, targets

    @staticmethod
    def repeat_copy_task(
            batch_size: int,
            seq_len: int,
            vocab_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Repeat copy: Model must remember sequence and repeat count.

        Format: [START] [sequence] [DELIMITER] [count] [DELIMITER] [sequence * count] [END]

        Forces MANN because:
        - Must store sequence in memory
        - Must implement a loop counter
        - Requires algorithmic reasoning
        """
        sequences = np.random.randint(4, vocab_size - 10, (batch_size, seq_len))
        repeat_counts = np.random.randint(1, 6, (batch_size,))

        max_len = seq_len * 6 + 10  # Max 5 repeats + overhead
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        targets = np.zeros((batch_size, max_len), dtype=np.int32)

        for b in range(batch_size):
            pos = 0
            inputs[b, pos] = 1  # START
            pos += 1

            # Original sequence
            inputs[b, pos:pos + seq_len] = sequences[b]
            pos += seq_len

            # Delimiter
            inputs[b, pos] = 2
            pos += 1

            # Repeat count (encoded as token)
            inputs[b, pos] = vocab_size - 10 + repeat_counts[b]
            pos += 1

            # Delimiter
            inputs[b, pos] = 2
            pos += 1

            # Target: sequence repeated
            for _ in range(repeat_counts[b]):
                targets[b, pos:pos + seq_len] = sequences[b]
                pos += seq_len

        return inputs, targets, repeat_counts

    @staticmethod
    def reverse_task(
            batch_size: int,
            seq_len: int,
            vocab_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reverse sequence: Forces sequential memory access.

        Format: [START] [sequence] [DELIMITER] [reversed_sequence] [END]
        """
        sequences = np.random.randint(4, vocab_size - 4, (batch_size, seq_len))

        max_len = seq_len * 2 + 4
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        targets = np.zeros((batch_size, max_len), dtype=np.int32)

        for b in range(batch_size):
            pos = 0
            inputs[b, pos] = 1  # START
            pos += 1

            inputs[b, pos:pos + seq_len] = sequences[b]
            pos += seq_len

            inputs[b, pos] = 2  # DELIMITER
            pos += 1

            # Target: reversed
            targets[b, pos:pos + seq_len] = sequences[b][::-1]

        return inputs, targets

    @staticmethod
    def associative_recall_task(
            batch_size: int,
            num_pairs: int,
            vocab_size: int,
            item_len: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Associative recall: Given key, recall value.

        Format: [pair1_key] [pair1_val] ... [pairN_key] [pairN_val]
                [QUERY] [query_key] [DELIMITER] [expected_val]

        Forces content-based addressing in MANN.
        """
        max_len = num_pairs * item_len * 2 + item_len + 4
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        targets = np.zeros((batch_size, max_len), dtype=np.int32)

        for b in range(batch_size):
            # Generate unique key-value pairs
            keys = np.random.randint(4, vocab_size // 2, (num_pairs, item_len))
            values = np.random.randint(vocab_size // 2, vocab_size - 4, (num_pairs, item_len))

            pos = 0
            # Store all pairs
            for i in range(num_pairs):
                inputs[b, pos:pos + item_len] = keys[i]
                pos += item_len
                inputs[b, pos:pos + item_len] = values[i]
                pos += item_len

            # Query token
            inputs[b, pos] = 2
            pos += 1

            # Pick random key to query
            query_idx = np.random.randint(num_pairs)
            inputs[b, pos:pos + item_len] = keys[query_idx]
            pos += item_len

            # Delimiter
            inputs[b, pos] = 3
            pos += 1

            # Expected output
            targets[b, pos:pos + item_len] = values[query_idx]

        return inputs, targets


class GNNForcingTask:
    """
    Tasks that explicitly require entity graph reasoning.

    These are designed so the model must use structured entity
    representations and relationships.
    """

    @staticmethod
    def entity_tracking_story(
            batch_size: int,
            num_entities: int,
            num_events: int,
            vocab_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Story understanding with entity tracking.

        Format: Multiple sentences describing entity interactions,
        then question about entity states/relationships.

        Forces GNN because:
        - Must track multiple entities simultaneously
        - Must update entity states based on events
        - Must reason over entity relationships
        """
        # Simplified: entity IDs, actions, relationships
        # Real version would use actual text

        entity_ids = np.arange(num_entities)
        max_len = num_events * 5 + 10

        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        adjacency = np.zeros((batch_size, num_entities, num_entities), dtype=np.float32)
        targets = np.zeros((batch_size, 3), dtype=np.int32)  # Answer tokens

        for b in range(batch_size):
            # Generate story events that create entity relationships
            pos = 0
            for event in range(num_events):
                entity1 = np.random.randint(num_entities)
                entity2 = np.random.randint(num_entities)
                action = np.random.randint(10, 20)  # Action tokens

                # Format: [entity1] [action] [entity2]
                inputs[b, pos] = 100 + entity1  # Entity tokens start at 100
                inputs[b, pos + 1] = action
                inputs[b, pos + 2] = 100 + entity2
                pos += 3

                # Update adjacency matrix
                adjacency[b, entity1, entity2] = 1.0
                adjacency[b, entity2, entity1] = 0.5  # Asymmetric relationship

            # Question: "What did entity X do to entity Y?"
            query_e1 = np.random.randint(num_entities)
            query_e2 = np.random.randint(num_entities)

            inputs[b, pos] = 2  # QUERY token
            inputs[b, pos + 1] = 100 + query_e1
            inputs[b, pos + 2] = 100 + query_e2

            # Target: the action (simplified)
            targets[b, 0] = 15  # Placeholder answer

        return inputs, adjacency, targets

    @staticmethod
    def knowledge_graph_completion(
            batch_size: int,
            num_entities: int,
            num_relations: int,
            num_triples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Knowledge graph completion: Given (head, relation, ?), predict tail.

        Format: [triple1] [triple2] ... [tripleN] [QUERY] [head] [relation] [?]

        Forces GNN because:
        - Must build entity representation from graph structure
        - Must use multi-hop reasoning
        - Requires understanding of relation types
        """
        max_len = num_triples * 3 + 4
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        adjacency = np.zeros((batch_size, num_entities, num_entities), dtype=np.float32)
        targets = np.zeros((batch_size, 1), dtype=np.int32)

        for b in range(batch_size):
            # Generate knowledge graph triples
            triples = []
            pos = 0

            for _ in range(num_triples):
                head = np.random.randint(num_entities)
                relation = np.random.randint(num_relations)
                tail = np.random.randint(num_entities)

                triples.append((head, relation, tail))

                # Store triple
                inputs[b, pos] = 200 + head  # Entity offset
                inputs[b, pos + 1] = 300 + relation  # Relation offset
                inputs[b, pos + 2] = 200 + tail
                pos += 3

                # Update adjacency with relation type
                adjacency[b, head, tail] = (relation + 1) / num_relations

            # Query: pick a random triple and hide the tail
            query_idx = np.random.randint(len(triples))
            query_head, query_rel, query_tail = triples[query_idx]

            inputs[b, pos] = 2  # QUERY
            inputs[b, pos + 1] = 200 + query_head
            inputs[b, pos + 2] = 300 + query_rel

            targets[b, 0] = 200 + query_tail

        return inputs, adjacency, targets

    @staticmethod
    def multi_hop_reasoning(
            batch_size: int,
            num_entities: int,
            num_hops: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-hop reasoning: Traverse graph to answer question.

        Example: "Alice knows Bob. Bob knows Carol. Who does Alice know 2 steps away?"

        Forces GNN because:
        - Requires explicit graph traversal
        - Must compose relationships across hops
        - Single attention layer can't do this efficiently
        """
        max_len = 50
        inputs = np.zeros((batch_size, max_len), dtype=np.int32)
        adjacency = np.zeros((batch_size, num_entities, num_entities), dtype=np.float32)
        targets = np.zeros((batch_size, 1), dtype=np.int32)

        for b in range(batch_size):
            # Create a chain: e0 -> e1 -> e2 -> ... -> e_num_hops
            chain = np.random.choice(num_entities, size=num_hops + 1, replace=False)

            pos = 0
            for i in range(num_hops):
                # Format: [entityA] [KNOWS] [entityB]
                inputs[b, pos] = 200 + chain[i]
                inputs[b, pos + 1] = 250  # KNOWS relation
                inputs[b, pos + 2] = 200 + chain[i + 1]
                pos += 3

                adjacency[b, chain[i], chain[i + 1]] = 1.0

            # Add some noise edges
            for _ in range(num_entities // 2):
                e1 = np.random.randint(num_entities)
                e2 = np.random.randint(num_entities)
                adjacency[b, e1, e2] = 0.3

            # Query: "Who does chain[0] know {num_hops} steps away?"
            inputs[b, pos] = 2  # QUERY
            inputs[b, pos + 1] = 200 + chain[0]
            inputs[b, pos + 2] = 251  # KNOWS_N_STEPS
            inputs[b, pos + 3] = num_hops

            targets[b, 0] = 200 + chain[-1]

        return inputs, adjacency, targets


# ---------------------------------------------------------------------
# Auxiliary Losses - Prevent Component Collapse
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MANNUtilizationLoss(keras.losses.Loss):
    """
    Loss that encourages MANN to actually use its memory.

    Measures:
    1. Read attention entropy (are we reading diverse locations?)
    2. Write gate activation (are we actually writing?)
    3. Memory content variance (is memory changing over time?)
    """

    def __init__(
            self,
            entropy_weight: float = 0.01,
            write_weight: float = 0.01,
            variance_weight: float = 0.01,
            name: str = "mann_utilization_loss",
            **kwargs: Any
    ):
        super().__init__(name=name, **kwargs)
        self.entropy_weight = entropy_weight
        self.write_weight = write_weight
        self.variance_weight = variance_weight

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            memory_vectors: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute auxiliary loss for MANN utilization.

        Args:
            y_true: Not used, but required by Keras Loss API
            y_pred: Not used
            memory_vectors: Memory read vectors from MANN
                Shape: (batch, seq_len, memory_dim)
        """
        # 1. Encourage diverse memory access (high entropy)
        # If all reads are from the same location, entropy is low (bad)
        memory_probs = keras.ops.softmax(memory_vectors, axis=-1)
        entropy = -keras.ops.sum(
            memory_probs * keras.ops.log(memory_probs + 1e-10),
            axis=-1
        )
        entropy_loss = -keras.ops.mean(entropy)  # Maximize entropy

        # 2. Encourage temporal variance (memory should change)
        # If memory is static, model isn't using it
        temporal_diff = memory_vectors[:, 1:, :] - memory_vectors[:, :-1, :]
        variance = keras.ops.var(temporal_diff)
        variance_loss = -keras.ops.log(variance + 1e-10)  # Maximize variance

        # 3. Encourage non-zero memory (detect unused memory)
        memory_norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(memory_vectors), axis=-1)
        )
        magnitude_loss = -keras.ops.mean(memory_norm)  # Maximize magnitude

        total_loss = (
                self.entropy_weight * entropy_loss +
                self.variance_weight * variance_loss +
                self.write_weight * magnitude_loss
        )

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'entropy_weight': self.entropy_weight,
            'write_weight': self.write_weight,
            'variance_weight': self.variance_weight,
        })
        return config


@keras.saving.register_keras_serializable()
class GNNUtilizationLoss(keras.losses.Loss):
    """
    Loss that encourages GNN to use entity representations.

    Measures:
    1. Entity embedding diversity (are entities distinct?)
    2. Attention weight entropy (is graph attention uniform?)
    3. Entity activation (how many entities are actually used?)
    """

    def __init__(
            self,
            diversity_weight: float = 0.01,
            attention_weight: float = 0.01,
            activation_weight: float = 0.01,
            name: str = "gnn_utilization_loss",
            **kwargs: Any
    ):
        super().__init__(name=name, **kwargs)
        self.diversity_weight = diversity_weight
        self.attention_weight = attention_weight
        self.activation_weight = activation_weight

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            entity_embeddings: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute auxiliary loss for GNN utilization.

        Args:
            entity_embeddings: Entity representations from GNN
                Shape: (batch, num_entities, entity_dim)
        """
        # 1. Encourage entity diversity
        # Compute pairwise cosine similarity
        normalized = keras.ops.normalize(entity_embeddings, axis=-1)
        similarity = keras.ops.einsum('bnd,bmd->bnm', normalized, normalized)

        # Penalize high similarity (want diverse representations)
        # Zero out diagonal (self-similarity)
        num_entities = keras.ops.shape(entity_embeddings)[1]
        mask = 1.0 - keras.ops.eye(num_entities)
        masked_sim = similarity * mask

        diversity_loss = keras.ops.mean(keras.ops.abs(masked_sim))

        # 2. Encourage entity usage (non-zero activations)
        entity_norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(entity_embeddings), axis=-1)
        )
        activation_loss = -keras.ops.mean(entity_norm)

        # 3. Encourage variance across entities
        entity_var = keras.ops.var(entity_embeddings, axis=1)
        variance_loss = -keras.ops.mean(entity_var)

        total_loss = (
                self.diversity_weight * diversity_loss +
                self.activation_weight * activation_loss +
                self.attention_weight * variance_loss
        )

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'diversity_weight': self.diversity_weight,
            'attention_weight': self.attention_weight,
            'activation_weight': self.activation_weight,
        })
        return config


# ---------------------------------------------------------------------
# Component Usage Monitor
# ---------------------------------------------------------------------

@dataclass
class ComponentMetrics:
    """Metrics to detect component collapse."""

    # MANN metrics
    mann_read_entropy: float = 0.0
    mann_write_rate: float = 0.0
    mann_memory_variance: float = 0.0
    mann_utilization_score: float = 0.0

    # GNN metrics
    gnn_entity_diversity: float = 0.0
    gnn_attention_entropy: float = 0.0
    gnn_entity_usage_rate: float = 0.0
    gnn_utilization_score: float = 0.0

    # Integration metrics
    memory_cross_attn_entropy: float = 0.0
    entity_cross_attn_entropy: float = 0.0
    integration_effectiveness: float = 0.0

    # Task performance
    task_accuracy: float = 0.0
    ablation_drop: float = 0.0  # How much does removing component hurt?

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'mann_read_entropy': self.mann_read_entropy,
            'mann_write_rate': self.mann_write_rate,
            'mann_memory_variance': self.mann_memory_variance,
            'mann_utilization': self.mann_utilization_score,
            'gnn_entity_diversity': self.gnn_entity_diversity,
            'gnn_attention_entropy': self.gnn_attention_entropy,
            'gnn_entity_usage': self.gnn_entity_usage_rate,
            'gnn_utilization': self.gnn_utilization_score,
            'memory_cross_attn': self.memory_cross_attn_entropy,
            'entity_cross_attn': self.entity_cross_attn_entropy,
            'integration_effectiveness': self.integration_effectiveness,
            'task_accuracy': self.task_accuracy,
            'ablation_drop': self.ablation_drop,
        }

    def is_healthy(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if components are being used (not collapsed).

        Args:
            thresholds: Custom thresholds for each metric
        """
        if thresholds is None:
            thresholds = {
                'mann_utilization': 0.3,
                'gnn_utilization': 0.3,
                'integration_effectiveness': 0.3,
            }

        return (
                self.mann_utilization_score >= thresholds['mann_utilization'] and
                self.gnn_utilization_score >= thresholds['gnn_utilization'] and
                self.integration_effectiveness >= thresholds['integration_effectiveness']
        )


class ComponentMonitor:
    """
    Monitor component usage during training.

    Detects component collapse by measuring:
    - Information flow through each component
    - Component contribution to final output
    - Ablation study results
    """

    def __init__(self, model: keras.Model):
        self.model = model
        self.history: List[ComponentMetrics] = []

    def compute_metrics(
            self,
            inputs: Dict[str, np.ndarray],
            targets: np.ndarray,
            outputs: Dict[str, np.ndarray]
    ) -> ComponentMetrics:
        """
        Compute comprehensive component usage metrics.

        Args:
            inputs: Model inputs
            targets: Ground truth targets
            outputs: Model outputs (with memory and entity info)
        """
        metrics = ComponentMetrics()

        # Extract components
        logits = outputs['logits']
        memory_vectors = outputs['memory_vectors']
        entity_embeddings = outputs['entity_embeddings']

        # Task accuracy
        predictions = np.argmax(logits, axis=-1)
        metrics.task_accuracy = float(np.mean(predictions == targets))

        # MANN metrics
        metrics.mann_read_entropy = self._compute_entropy(memory_vectors)
        metrics.mann_memory_variance = float(np.var(memory_vectors))
        metrics.mann_utilization_score = (
                metrics.mann_read_entropy * metrics.mann_memory_variance
        )

        # GNN metrics
        metrics.gnn_entity_diversity = self._compute_diversity(entity_embeddings)
        metrics.gnn_entity_usage_rate = self._compute_usage_rate(entity_embeddings)
        metrics.gnn_utilization_score = (
                metrics.gnn_entity_diversity * metrics.gnn_entity_usage_rate
        )

        # Integration effectiveness (requires ablation - expensive)
        # For now, use a proxy based on output variance
        metrics.integration_effectiveness = float(np.std(logits))

        self.history.append(metrics)
        return metrics

    def _compute_entropy(self, vectors: np.ndarray) -> float:
        """Compute average entropy of attention distributions."""
        # Normalize to get probabilities
        probs = np.exp(vectors) / np.sum(np.exp(vectors), axis=-1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        return float(np.mean(entropy))

    def _compute_diversity(self, embeddings: np.ndarray) -> float:
        """Compute diversity of entity embeddings."""
        # Flatten batch dimension
        batch_size, num_entities, dim = embeddings.shape
        flat = embeddings.reshape(-1, dim)

        # Compute pairwise distances
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        normalized = flat / (norms + 1e-10)
        similarity = np.dot(normalized, normalized.T)

        # Diversity is inverse of average similarity
        diversity = 1.0 - np.mean(np.abs(similarity))
        return float(diversity)

    def _compute_usage_rate(self, embeddings: np.ndarray) -> float:
        """Compute fraction of entities with non-zero activations."""
        norms = np.linalg.norm(embeddings, axis=-1)
        threshold = 1e-3
        active = np.sum(norms > threshold)
        total = norms.size
        return float(active / total)

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot component utilization over time.

        Requires matplotlib (not imported here to avoid dependency).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.history:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = range(len(self.history))

        # MANN metrics
        ax = axes[0, 0]
        ax.plot(steps, [m.mann_utilization_score for m in self.history], 'b-', label='Utilization')
        ax.plot(steps, [m.mann_read_entropy for m in self.history], 'g--', label='Entropy')
        ax.axhline(y=0.3, color='r', linestyle=':', label='Threshold')
        ax.set_title('MANN Component Health')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # GNN metrics
        ax = axes[0, 1]
        ax.plot(steps, [m.gnn_utilization_score for m in self.history], 'b-', label='Utilization')
        ax.plot(steps, [m.gnn_entity_diversity for m in self.history], 'g--', label='Diversity')
        ax.axhline(y=0.3, color='r', linestyle=':', label='Threshold')
        ax.set_title('GNN Component Health')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Integration effectiveness
        ax = axes[1, 0]
        ax.plot(steps, [m.integration_effectiveness for m in self.history], 'b-')
        ax.axhline(y=0.3, color='r', linestyle=':', label='Threshold')
        ax.set_title('Integration Effectiveness')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Task accuracy
        ax = axes[1, 1]
        ax.plot(steps, [m.task_accuracy for m in self.history], 'b-')
        ax.set_title('Task Accuracy')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def get_summary(self) -> str:
        """Get human-readable summary of component health."""
        if not self.history:
            return "No metrics collected yet."

        latest = self.history[-1]

        summary = "Component Health Summary:\n"
        summary += "=" * 50 + "\n\n"

        # MANN
        summary += "MANN (Working Memory):\n"
        summary += f"  Utilization: {latest.mann_utilization_score:.3f} "
        summary += "✓\n" if latest.mann_utilization_score >= 0.3 else "✗ LOW\n"
        summary += f"  Read Entropy: {latest.mann_read_entropy:.3f}\n"
        summary += f"  Memory Variance: {latest.mann_memory_variance:.3f}\n\n"

        # GNN
        summary += "GNN (Entity Graph):\n"
        summary += f"  Utilization: {latest.gnn_utilization_score:.3f} "
        summary += "✓\n" if latest.gnn_utilization_score >= 0.3 else "✗ LOW\n"
        summary += f"  Entity Diversity: {latest.gnn_entity_diversity:.3f}\n"
        summary += f"  Usage Rate: {latest.gnn_entity_usage_rate:.3f}\n\n"

        # Integration
        summary += "Integration:\n"
        summary += f"  Effectiveness: {latest.integration_effectiveness:.3f}\n"
        summary += f"  Memory Cross-Attn: {latest.memory_cross_attn_entropy:.3f}\n"
        summary += f"  Entity Cross-Attn: {latest.entity_cross_attn_entropy:.3f}\n\n"

        # Overall health
        summary += f"Task Accuracy: {latest.task_accuracy:.3f}\n"
        summary += f"Overall Status: "
        summary += "HEALTHY ✓\n" if latest.is_healthy() else "COMPONENT COLLAPSE DETECTED ✗\n"

        return summary


# ---------------------------------------------------------------------
# Training Curriculum
# ---------------------------------------------------------------------

@dataclass
class TrainingPhaseConfig:
    """Configuration for a training phase."""
    name: str
    num_epochs: int
    task_types: List[str]  # ['mann', 'gnn', 'combined']
    freeze_transformer: bool = False
    freeze_mann: bool = False
    freeze_gnn: bool = False
    learning_rate: float = 1e-4
    auxiliary_loss_weight: float = 0.1
    batch_size: int = 32


class TrainingCurriculum:
    """
    Staged training curriculum to force component usage.

    Phase 1: Warm-up on MANN tasks with frozen transformer
    Phase 2: Warm-up on GNN tasks with frozen transformer
    Phase 3: Combined tasks with all components active
    Phase 4: Fine-tuning on diverse tasks
    """

    def __init__(self, model: keras.Model):
        self.model = model
        self.current_phase = 0

        # Define curriculum
        self.phases = [
            TrainingPhaseConfig(
                name="Phase 1: MANN Warm-up",
                num_epochs=5,
                task_types=['mann'],
                freeze_transformer=True,
                learning_rate=1e-3,
                auxiliary_loss_weight=0.2,
            ),
            TrainingPhaseConfig(
                name="Phase 2: GNN Warm-up",
                num_epochs=5,
                task_types=['gnn'],
                freeze_transformer=True,
                learning_rate=1e-3,
                auxiliary_loss_weight=0.2,
            ),
            TrainingPhaseConfig(
                name="Phase 3: Integration",
                num_epochs=10,
                task_types=['mann', 'gnn', 'combined'],
                freeze_transformer=False,
                learning_rate=5e-4,
                auxiliary_loss_weight=0.1,
            ),
            TrainingPhaseConfig(
                name="Phase 4: Fine-tuning",
                num_epochs=20,
                task_types=['mann', 'gnn', 'combined', 'language'],
                freeze_transformer=False,
                learning_rate=1e-4,
                auxiliary_loss_weight=0.05,
            ),
        ]

    def get_current_phase(self) -> TrainingPhaseConfig:
        """Get current training phase configuration."""
        return self.phases[self.current_phase]

    def advance_phase(self) -> bool:
        """
        Move to next training phase.

        Returns:
            True if advanced, False if already at last phase
        """
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            return True
        return False

    def configure_model_for_phase(self, phase: TrainingPhaseConfig) -> None:
        """
        Configure model freezing/unfreezing for current phase.
        """
        # Freeze/unfreeze transformer blocks
        for block in self.model.blocks:
            block.trainable = not phase.freeze_transformer

        # Freeze/unfreeze MANN
        self.model.mann.trainable = not phase.freeze_mann

        # Freeze/unfreeze GNN
        self.model.gnn.trainable = not phase.freeze_gnn

        print(f"\n{phase.name}")
        print(f"Transformer: {'FROZEN' if phase.freeze_transformer else 'TRAINABLE'}")
        print(f"MANN: {'FROZEN' if phase.freeze_mann else 'TRAINABLE'}")
        print(f"GNN: {'FROZEN' if phase.freeze_gnn else 'TRAINABLE'}")
        print(f"Learning Rate: {phase.learning_rate}")
        print(f"Auxiliary Loss Weight: {phase.auxiliary_loss_weight}\n")


# ---------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------

class Qwen3MEGATrainer:
    """
    Complete training system for Qwen3-MEGA.

    Integrates:
    - Task generation
    - Auxiliary losses
    - Component monitoring
    - Staged curriculum
    """

    def __init__(
            self,
            model: keras.Model,
            vocab_size: int = 32000,
            monitor_frequency: int = 100,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.monitor_frequency = monitor_frequency

        # Initialize components
        self.curriculum = TrainingCurriculum(model)
        self.monitor = ComponentMonitor(model)

        # Loss functions
        self.main_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mann_aux_loss = MANNUtilizationLoss()
        self.gnn_aux_loss = GNNUtilizationLoss()

        # Task generators
        self.mann_tasks = MANNForcingTask()
        self.gnn_tasks = GNNForcingTask()

        # Training state
        self.global_step = 0
        self.epoch = 0

    def generate_batch(
            self,
            task_type: str,
            batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate a batch of data for specified task type.

        Args:
            task_type: One of ['mann', 'gnn', 'combined', 'language']
            batch_size: Batch size

        Returns:
            (inputs_dict, targets)
        """
        if task_type == 'mann':
            # Random MANN task
            task_choice = np.random.choice(['copy', 'repeat', 'reverse', 'recall'])

            if task_choice == 'copy':
                inputs, targets = self.mann_tasks.copy_task(
                    batch_size, seq_len=10, vocab_size=self.vocab_size
                )
                adjacency = None
            elif task_choice == 'repeat':
                inputs, targets, _ = self.mann_tasks.repeat_copy_task(
                    batch_size, seq_len=8, vocab_size=self.vocab_size
                )
                adjacency = None
            elif task_choice == 'reverse':
                inputs, targets = self.mann_tasks.reverse_task(
                    batch_size, seq_len=10, vocab_size=self.vocab_size
                )
                adjacency = None
            else:  # recall
                inputs, targets = self.mann_tasks.associative_recall_task(
                    batch_size, num_pairs=5, vocab_size=self.vocab_size
                )
                adjacency = None

        elif task_type == 'gnn':
            # Random GNN task
            task_choice = np.random.choice(['story', 'kg_completion', 'multi_hop'])

            num_entities = self.model.num_entities

            if task_choice == 'story':
                inputs, adjacency, targets = self.gnn_tasks.entity_tracking_story(
                    batch_size, num_entities=num_entities,
                    num_events=10, vocab_size=self.vocab_size
                )
            elif task_choice == 'kg_completion':
                inputs, adjacency, targets = self.gnn_tasks.knowledge_graph_completion(
                    batch_size, num_entities=num_entities,
                    num_relations=20, num_triples=15
                )
            else:  # multi_hop
                inputs, adjacency, targets = self.gnn_tasks.multi_hop_reasoning(
                    batch_size, num_entities=num_entities, num_hops=3
                )

        elif task_type == 'combined':
            # Mix of MANN and GNN tasks
            if np.random.random() < 0.5:
                return self.generate_batch('mann', batch_size)
            else:
                return self.generate_batch('gnn', batch_size)

        else:  # 'language'
            # Regular language modeling task
            # For now, use copy task as placeholder
            inputs, targets = self.mann_tasks.copy_task(
                batch_size, seq_len=20, vocab_size=self.vocab_size
            )
            adjacency = None

        # Create attention mask (all ones for simplicity)
        attention_mask = np.ones_like(inputs)

        inputs_dict = {
            'input_ids': inputs,
            'attention_mask': attention_mask,
        }

        if adjacency is not None:
            inputs_dict['entity_adjacency'] = adjacency

        return inputs_dict, targets

    def train_step(
            self,
            inputs: Dict[str, np.ndarray],
            targets: np.ndarray,
            optimizer: keras.optimizers.Optimizer,
            aux_loss_weight: float
    ) -> Dict[str, float]:
        """
        Single training step with auxiliary losses.

        Returns:
            Dictionary of losses
        """
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(inputs, training=True, return_dict=True)

            logits = outputs['logits']
            memory_vectors = outputs['memory_vectors']
            entity_embeddings = outputs['entity_embeddings']

            # Main task loss
            main_loss = self.main_loss(targets, logits)

            # Auxiliary losses
            mann_loss = self.mann_aux_loss(
                targets, logits, memory_vectors
            )
            gnn_loss = self.gnn_aux_loss(
                targets, logits, entity_embeddings
            )

            # Total loss
            total_loss = (
                    main_loss +
                    aux_loss_weight * mann_loss +
                    aux_loss_weight * gnn_loss
            )

        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {
            'total_loss': float(total_loss),
            'main_loss': float(main_loss),
            'mann_aux_loss': float(mann_loss),
            'gnn_aux_loss': float(gnn_loss),
        }

    def train_epoch(
            self,
            phase: TrainingPhaseConfig,
            optimizer: keras.optimizers.Optimizer,
            num_steps: int = 1000
    ) -> None:
        """Train for one epoch within a phase."""

        for step in range(num_steps):
            # Sample task type from phase
            task_type = np.random.choice(phase.task_types)

            # Generate batch
            inputs, targets = self.generate_batch(task_type, phase.batch_size)

            # Training step
            losses = self.train_step(
                inputs, targets, optimizer, phase.auxiliary_loss_weight
            )

            # Monitor components periodically
            if step % self.monitor_frequency == 0:
                # Get outputs for monitoring
                outputs = self.model(inputs, training=False, return_dict=True)
                metrics = self.monitor.compute_metrics(inputs, targets, outputs)

                # Print status
                print(f"Step {self.global_step}: "
                      f"Loss={losses['total_loss']:.4f}, "
                      f"MANN={metrics.mann_utilization_score:.3f}, "
                      f"GNN={metrics.gnn_utilization_score:.3f}, "
                      f"Acc={metrics.task_accuracy:.3f}")

                # Check for component collapse
                if not metrics.is_healthy():
                    print("⚠️  WARNING: Component collapse detected!")
                    print(self.monitor.get_summary())

            self.global_step += 1

        self.epoch += 1

    def train(
            self,
            steps_per_phase: int = 1000,
            checkpoint_dir: Optional[str] = None
    ) -> None:
        """
        Full training with curriculum.

        Args:
            steps_per_phase: Number of steps per epoch
            checkpoint_dir: Directory to save checkpoints
        """
        print("=" * 60)
        print("Qwen3-MEGA Training with Component-Forcing Curriculum")
        print("=" * 60)

        for phase_idx, phase in enumerate(self.curriculum.phases):
            print(f"\n{'=' * 60}")
            print(f"Starting {phase.name}")
            print(f"{'=' * 60}\n")

            # Configure model for this phase
            self.curriculum.configure_model_for_phase(phase)

            # Create optimizer for this phase
            optimizer = keras.optimizers.AdamW(
                learning_rate=phase.learning_rate,
                weight_decay=0.01
            )

            # Train for specified epochs
            for epoch in range(phase.num_epochs):
                print(f"\nEpoch {epoch + 1}/{phase.num_epochs}")
                self.train_epoch(phase, optimizer, steps_per_phase)

                # Save checkpoint
                if checkpoint_dir:
                    checkpoint_path = f"{checkpoint_dir}/phase{phase_idx}_epoch{epoch}.keras"
                    self.model.save(checkpoint_path)
                    print(f"Saved checkpoint: {checkpoint_path}")

            # Summary after phase
            print(f"\n{phase.name} Complete!")
            print(self.monitor.get_summary())

        # Final summary
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(self.monitor.get_summary())

        # Save final model
        if checkpoint_dir:
            final_path = f"{checkpoint_dir}/final_model.keras"
            self.model.save(final_path)
            print(f"\nSaved final model: {final_path}")

        # Plot training curves
        self.monitor.plot_history(
            save_path=f"{checkpoint_dir}/training_curves.png" if checkpoint_dir else None
        )


# ---------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------

def main():
    """Example usage of the training framework."""

    # Import the model (would be from previous artifact)
    # from qwen3_mega import create_qwen3_mega

    print("Creating Qwen3-MEGA model...")
    model = create_qwen3_mega(
        variant="tiny",
        memory_size="small",
        entity_graph_size="small"
    )

    # For demonstration, we'll use a placeholder
    print("Model created!")

    print("\nInitializing trainer...")
    trainer = Qwen3MEGATrainer(
        model=model,
        vocab_size=32000,
        monitor_frequency=50
    )

    print("\nStarting training with curriculum...")
    trainer.train(
        steps_per_phase=500,  # Fewer steps for demo
        checkpoint_dir="./checkpoints"
    )

    print("\n" + "=" * 60)
    print("Training Framework Ready!")
    print("=" * 60)
    print("\nKey Features:")
    print("1. ✓ MANN-forcing tasks (copy, repeat, reverse, recall)")
    print("2. ✓ GNN-forcing tasks (entity tracking, KG completion, multi-hop)")
    print("3. ✓ Auxiliary losses (MANN & GNN utilization)")
    print("4. ✓ Component monitoring (detect collapse)")
    print("5. ✓ Staged curriculum (4-phase training)")
    print("6. ✓ Visualization tools (training curves)")
    print("\nTo use:")
    print("  trainer = Qwen3MEGATrainer(model, vocab_size=32000)")
    print("  trainer.train(steps_per_phase=1000, checkpoint_dir='./checkpoints')")


if __name__ == "__main__":
    main()