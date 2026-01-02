"""
Data generators for Memory-Augmented Neural Network benchmark tasks.

This module provides generators for classic NTM/DNC tasks including
copy, associative recall, and algorithmic reasoning tasks.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (
    AlgorithmicTaskConfig,
    AssociativeRecallConfig,
    CopyTaskConfig,
    LengthGeneralizationConfig,
)


@dataclass
class TaskData:
    """Container for task inputs and targets.
    
    :param inputs: Input sequences of shape (batch, seq_len, features).
    :param targets: Target sequences of shape (batch, seq_len, features).
    :param masks: Optional mask indicating valid output positions.
    :param metadata: Optional dictionary with task-specific metadata.
    """
    inputs: np.ndarray
    targets: np.ndarray
    masks: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class CopyTaskGenerator:
    """Generator for the Copy Task benchmark.
    
    The copy task is a fundamental test of memory capacity where the model
    must store a sequence of binary vectors and reproduce them after a delay.
    
    :param config: Configuration for the copy task.
    
    Example::
    
        config = CopyTaskConfig(sequence_length=10, vector_size=8)
        generator = CopyTaskGenerator(config)
        data = generator.generate(num_samples=100)
    """
    
    def __init__(self, config: CopyTaskConfig) -> None:
        """Initialize the copy task generator.
        
        :param config: Copy task configuration.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
    
    def generate(
        self,
        num_samples: Optional[int] = None,
        sequence_length: Optional[int] = None
    ) -> TaskData:
        """Generate copy task samples.
        
        :param num_samples: Number of samples to generate. Uses config default if None.
        :param sequence_length: Length of sequences. Uses config default if None.
        :return: TaskData containing inputs and targets.
        """
        num_samples = num_samples or self.config.num_samples
        seq_len = sequence_length or self.config.sequence_length
        vec_size = self.config.vector_size
        delay = self.config.delay_length
        
        # Input structure: [start_marker, sequence, delimiter, delay, end_marker]
        # Total input length = 1 + seq_len + 1 + delay + 1 = seq_len + delay + 3
        # Total features = vec_size + 2 (for start/end markers)
        total_features = vec_size + 2
        total_input_len = seq_len + delay + 3
        
        inputs = np.zeros((num_samples, total_input_len, total_features), dtype=np.float32)
        targets = np.zeros((num_samples, total_input_len, vec_size), dtype=np.float32)
        masks = np.zeros((num_samples, total_input_len), dtype=np.float32)
        
        for i in range(num_samples):
            # Generate random binary sequence
            sequence = self._rng.integers(0, 2, size=(seq_len, vec_size)).astype(np.float32)
            
            # Start marker (channel vec_size)
            inputs[i, 0, vec_size] = 1.0
            
            # Input sequence
            inputs[i, 1:seq_len + 1, :vec_size] = sequence
            
            # Delimiter/end marker (channel vec_size + 1)
            inputs[i, seq_len + 1, vec_size + 1] = 1.0
            
            # Target: reproduce sequence after delay
            output_start = seq_len + delay + 2
            targets[i, output_start:output_start + seq_len, :] = sequence
            
            # Mask: only compute loss on output positions
            masks[i, output_start:output_start + seq_len] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            masks=masks,
            metadata={
                "sequence_length": seq_len,
                "vector_size": vec_size,
                "delay_length": delay
            }
        )
    
    def generate_length_generalization_set(
        self,
        config: LengthGeneralizationConfig
    ) -> Dict[int, TaskData]:
        """Generate datasets for length generalization evaluation.
        
        :param config: Length generalization configuration.
        :return: Dictionary mapping sequence lengths to TaskData.
        """
        datasets = {}
        all_lengths = config.train_lengths + config.test_lengths
        
        for length in all_lengths:
            datasets[length] = self.generate(
                num_samples=config.num_samples_per_length,
                sequence_length=length
            )
        
        return datasets


class AssociativeRecallGenerator:
    """Generator for the Associative Recall Task benchmark.
    
    Tests the ability to store and retrieve key-value associations,
    a fundamental capability for memory-augmented networks.
    
    :param config: Configuration for associative recall task.
    
    Example::
    
        config = AssociativeRecallConfig(num_items=6, key_size=4)
        generator = AssociativeRecallGenerator(config)
        data = generator.generate(num_samples=100)
    """
    
    def __init__(self, config: AssociativeRecallConfig) -> None:
        """Initialize the associative recall generator.
        
        :param config: Associative recall configuration.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
    
    def generate(
        self,
        num_samples: Optional[int] = None,
        num_items: Optional[int] = None
    ) -> TaskData:
        """Generate associative recall samples.
        
        :param num_samples: Number of samples to generate.
        :param num_items: Number of key-value pairs per sample.
        :return: TaskData containing inputs and targets.
        """
        num_samples = num_samples or self.config.num_samples
        n_items = num_items or self.config.num_items
        key_size = self.config.key_size
        value_size = self.config.value_size
        
        # Input: key-value pairs + delimiter + query key
        # Structure: [k1, v1, k2, v2, ..., kn, vn, delimiter, query_key]
        item_size = key_size + value_size
        input_len = n_items * 2 + 2  # pairs + delimiter + query
        total_features = max(key_size, value_size) + 1  # +1 for delimiter
        
        inputs = np.zeros((num_samples, input_len, total_features), dtype=np.float32)
        targets = np.zeros((num_samples, value_size), dtype=np.float32)
        
        for i in range(num_samples):
            # Generate random keys and values
            keys = self._rng.random((n_items, key_size)).astype(np.float32)
            values = self._rng.random((n_items, value_size)).astype(np.float32)
            
            # Place key-value pairs
            for j in range(n_items):
                inputs[i, j * 2, :key_size] = keys[j]
                inputs[i, j * 2 + 1, :value_size] = values[j]
            
            # Delimiter
            inputs[i, n_items * 2, -1] = 1.0
            
            # Query key (randomly select one of the stored keys)
            query_idx = self._rng.integers(0, n_items)
            inputs[i, n_items * 2 + 1, :key_size] = keys[query_idx]
            
            # Target is the corresponding value
            targets[i] = values[query_idx]
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "num_items": n_items,
                "key_size": key_size,
                "value_size": value_size
            }
        )
    
    def generate_capacity_test(
        self,
        item_counts: List[int]
    ) -> Dict[int, TaskData]:
        """Generate datasets for memory capacity evaluation.
        
        :param item_counts: List of item counts to test.
        :return: Dictionary mapping item counts to TaskData.
        """
        datasets = {}
        for count in item_counts:
            datasets[count] = self.generate(
                num_samples=100,
                num_items=count
            )
        return datasets


class RepeatCopyGenerator:
    """Generator for the Repeat Copy Task.
    
    An extension of the copy task where the model must output the
    sequence a specified number of times, testing working memory
    and counting capabilities.
    
    :param vector_size: Dimensionality of sequence vectors.
    :param random_seed: Seed for reproducibility.
    """
    
    def __init__(
        self,
        vector_size: int = 8,
        random_seed: int = 42
    ) -> None:
        """Initialize the repeat copy generator.
        
        :param vector_size: Size of each vector in the sequence.
        :param random_seed: Random seed for reproducibility.
        """
        self.vector_size = vector_size
        self._rng = np.random.default_rng(random_seed)
    
    def generate(
        self,
        num_samples: int,
        sequence_length: int,
        num_repeats: int
    ) -> TaskData:
        """Generate repeat copy samples.
        
        :param num_samples: Number of samples to generate.
        :param sequence_length: Length of the sequence to copy.
        :param num_repeats: Number of times to repeat the output.
        :return: TaskData containing inputs and targets.
        """
        vec_size = self.vector_size
        # Features: vec_size + 2 (start marker, end marker) + 1 (repeat count scalar)
        total_features = vec_size + 3
        input_len = sequence_length + 2  # start + sequence + end (with repeat info)
        output_len = sequence_length * num_repeats
        total_len = input_len + output_len
        
        inputs = np.zeros((num_samples, total_len, total_features), dtype=np.float32)
        targets = np.zeros((num_samples, total_len, vec_size), dtype=np.float32)
        masks = np.zeros((num_samples, total_len), dtype=np.float32)
        
        for i in range(num_samples):
            sequence = self._rng.integers(0, 2, size=(sequence_length, vec_size)).astype(np.float32)
            
            # Start marker
            inputs[i, 0, vec_size] = 1.0
            
            # Sequence
            inputs[i, 1:sequence_length + 1, :vec_size] = sequence
            
            # End marker with normalized repeat count
            inputs[i, sequence_length + 1, vec_size + 1] = 1.0
            inputs[i, sequence_length + 1, vec_size + 2] = num_repeats / 10.0  # Normalize
            
            # Target: repeated sequence
            for r in range(num_repeats):
                start_idx = input_len + r * sequence_length
                targets[i, start_idx:start_idx + sequence_length, :] = sequence
                masks[i, start_idx:start_idx + sequence_length] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            masks=masks,
            metadata={
                "sequence_length": sequence_length,
                "num_repeats": num_repeats,
                "vector_size": vec_size
            }
        )


class PriorityAccessGenerator:
    """Generator for the Priority Access Task.
    
    Tests content-addressable memory where items have associated priorities
    and must be retrieved in priority order.
    
    :param vector_size: Dimensionality of item vectors.
    :param random_seed: Seed for reproducibility.
    """
    
    def __init__(
        self,
        vector_size: int = 8,
        random_seed: int = 42
    ) -> None:
        """Initialize the priority access generator.
        
        :param vector_size: Size of each item vector.
        :param random_seed: Random seed for reproducibility.
        """
        self.vector_size = vector_size
        self._rng = np.random.default_rng(random_seed)
    
    def generate(
        self,
        num_samples: int,
        num_items: int
    ) -> TaskData:
        """Generate priority access samples.
        
        :param num_samples: Number of samples to generate.
        :param num_items: Number of items with priorities.
        :return: TaskData containing inputs and targets.
        """
        vec_size = self.vector_size
        # Input: (item, priority) pairs + delimiter
        # Output: items sorted by priority (descending)
        input_features = vec_size + 1  # vector + priority
        input_len = num_items + 1  # items + delimiter
        output_len = num_items
        total_len = input_len + output_len
        
        inputs = np.zeros((num_samples, total_len, input_features + 1), dtype=np.float32)
        targets = np.zeros((num_samples, total_len, vec_size), dtype=np.float32)
        masks = np.zeros((num_samples, total_len), dtype=np.float32)
        
        for i in range(num_samples):
            items = self._rng.random((num_items, vec_size)).astype(np.float32)
            priorities = self._rng.random(num_items).astype(np.float32)
            
            # Input: items with priorities
            inputs[i, :num_items, :vec_size] = items
            inputs[i, :num_items, vec_size] = priorities
            
            # Delimiter
            inputs[i, num_items, -1] = 1.0
            
            # Target: items sorted by priority (descending)
            sorted_indices = np.argsort(-priorities)
            targets[i, input_len:, :] = items[sorted_indices]
            masks[i, input_len:] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            masks=masks,
            metadata={
                "num_items": num_items,
                "vector_size": vec_size
            }
        )


class TraversalGenerator:
    """Generator for Graph Traversal Tasks.
    
    Tests graph-based reasoning by requiring the model to traverse
    a graph structure and report visited nodes.
    
    :param max_nodes: Maximum number of nodes in the graph.
    :param random_seed: Seed for reproducibility.
    """
    
    def __init__(
        self,
        max_nodes: int = 16,
        random_seed: int = 42
    ) -> None:
        """Initialize the traversal task generator.
        
        :param max_nodes: Maximum nodes in generated graphs.
        :param random_seed: Random seed for reproducibility.
        """
        self.max_nodes = max_nodes
        self._rng = np.random.default_rng(random_seed)
    
    def generate(
        self,
        num_samples: int,
        num_nodes: int,
        edge_probability: float = 0.3
    ) -> TaskData:
        """Generate graph traversal samples.
        
        :param num_samples: Number of samples to generate.
        :param num_nodes: Number of nodes in each graph.
        :param edge_probability: Probability of edge between any two nodes.
        :return: TaskData containing inputs and targets.
        """
        # Input: flattened adjacency matrix + start node one-hot
        # Output: nodes reachable from start (BFS order)
        adj_size = num_nodes * num_nodes
        input_features = adj_size + num_nodes  # adj + start one-hot
        
        inputs = np.zeros((num_samples, 1, input_features), dtype=np.float32)
        targets = np.zeros((num_samples, num_nodes), dtype=np.float32)
        
        for i in range(num_samples):
            # Generate random adjacency matrix (symmetric for undirected)
            adj = (self._rng.random((num_nodes, num_nodes)) < edge_probability).astype(np.float32)
            adj = np.maximum(adj, adj.T)  # Make symmetric
            np.fill_diagonal(adj, 0)  # No self-loops
            
            # Random start node
            start_node = self._rng.integers(0, num_nodes)
            
            # BFS to find reachable nodes
            visited = self._bfs(adj, start_node, num_nodes)
            
            # Pack input
            inputs[i, 0, :adj_size] = adj.flatten()
            inputs[i, 0, adj_size + start_node] = 1.0
            
            # Target: binary mask of reachable nodes
            targets[i] = visited
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "num_nodes": num_nodes,
                "edge_probability": edge_probability
            }
        )
    
    def _bfs(
        self,
        adj: np.ndarray,
        start: int,
        num_nodes: int
    ) -> np.ndarray:
        """Perform BFS from start node.
        
        :param adj: Adjacency matrix.
        :param start: Starting node index.
        :param num_nodes: Total number of nodes.
        :return: Binary array indicating reachable nodes.
        """
        visited = np.zeros(num_nodes, dtype=np.float32)
        queue = [start]
        visited[start] = 1.0
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(num_nodes):
                if adj[node, neighbor] > 0 and visited[neighbor] == 0:
                    visited[neighbor] = 1.0
                    queue.append(neighbor)
        
        return visited


class DynamicNGramGenerator:
    """Generator for Dynamic N-Gram Prediction Task.
    
    Tests the ability to learn and apply contextual patterns
    where the prediction rule changes based on context.
    
    :param vocab_size: Size of the vocabulary.
    :param random_seed: Seed for reproducibility.
    """
    
    def __init__(
        self,
        vocab_size: int = 10,
        random_seed: int = 42
    ) -> None:
        """Initialize the dynamic N-gram generator.
        
        :param vocab_size: Number of distinct tokens.
        :param random_seed: Random seed for reproducibility.
        """
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng(random_seed)
    
    def generate(
        self,
        num_samples: int,
        sequence_length: int,
        n_gram_order: int = 2
    ) -> TaskData:
        """Generate dynamic N-gram samples.
        
        :param num_samples: Number of samples to generate.
        :param sequence_length: Length of each sequence.
        :param n_gram_order: Order of N-gram (context size).
        :return: TaskData containing inputs and targets.
        """
        vocab_size = self.vocab_size
        
        inputs = np.zeros((num_samples, sequence_length, vocab_size), dtype=np.float32)
        targets = np.zeros((num_samples, sequence_length, vocab_size), dtype=np.float32)
        
        for i in range(num_samples):
            # Generate random N-gram transition table for this sample
            transition_table = self._rng.integers(
                0, vocab_size, 
                size=(vocab_size,) * n_gram_order
            )
            
            # Generate sequence following the transition table
            sequence = np.zeros(sequence_length, dtype=np.int32)
            sequence[:n_gram_order] = self._rng.integers(0, vocab_size, size=n_gram_order)
            
            for t in range(n_gram_order, sequence_length):
                context = tuple(sequence[t - n_gram_order:t])
                sequence[t] = transition_table[context]
            
            # One-hot encode
            for t in range(sequence_length):
                inputs[i, t, sequence[t]] = 1.0
                if t < sequence_length - 1:
                    targets[i, t, sequence[t + 1]] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "vocab_size": vocab_size,
                "n_gram_order": n_gram_order,
                "sequence_length": sequence_length
            }
        )


class AlgorithmicTaskGenerator:
    """Generator for CLRS-style algorithmic reasoning tasks.
    
    Generates data for sorting, searching, and graph algorithm tasks
    used to evaluate neural algorithmic reasoning capabilities.
    
    :param config: Configuration for algorithmic tasks.
    """
    
    SUPPORTED_TASKS = [
        "insertion_sort",
        "bubble_sort",
        "binary_search",
        "linear_search",
        "bfs",
        "dfs",
        "dijkstra",
        "minimum",
        "maximum",
        "reverse"
    ]
    
    def __init__(self, config: AlgorithmicTaskConfig) -> None:
        """Initialize the algorithmic task generator.
        
        :param config: Algorithmic task configuration.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        
        if config.task_name not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{config.task_name}' not supported. "
                f"Available tasks: {self.SUPPORTED_TASKS}"
            )
    
    def generate(
        self,
        num_samples: Optional[int] = None,
        problem_size: Optional[int] = None
    ) -> TaskData:
        """Generate algorithmic task samples.
        
        :param num_samples: Number of samples to generate.
        :param problem_size: Size of the problem (nodes/elements).
        :return: TaskData containing inputs and targets.
        """
        num_samples = num_samples or self.config.num_samples
        size = problem_size or self.config.train_size
        
        generator_map = {
            "insertion_sort": self._generate_sorting,
            "bubble_sort": self._generate_sorting,
            "binary_search": self._generate_search,
            "linear_search": self._generate_search,
            "minimum": self._generate_minmax,
            "maximum": self._generate_minmax,
            "reverse": self._generate_reverse,
            "bfs": self._generate_graph_traversal,
            "dfs": self._generate_graph_traversal,
            "dijkstra": self._generate_shortest_path
        }
        
        return generator_map[self.config.task_name](num_samples, size)
    
    def _generate_sorting(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate sorting task data.
        
        :param num_samples: Number of samples.
        :param size: Array size to sort.
        :return: TaskData for sorting.
        """
        inputs = self._rng.random((num_samples, size, 1)).astype(np.float32)
        targets = np.sort(inputs, axis=1)
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "task": self.config.task_name,
                "size": size
            }
        )
    
    def _generate_search(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate search task data.
        
        :param num_samples: Number of samples.
        :param size: Array size to search.
        :return: TaskData for search.
        """
        # For binary search, inputs must be sorted
        arrays = np.sort(
            self._rng.random((num_samples, size)).astype(np.float32),
            axis=1
        )
        
        # Random query (existing element for positive examples)
        query_indices = self._rng.integers(0, size, size=num_samples)
        queries = arrays[np.arange(num_samples), query_indices]
        
        # Input: concatenate array and query
        inputs = np.zeros((num_samples, size + 1, 1), dtype=np.float32)
        inputs[:, :size, 0] = arrays
        inputs[:, size, 0] = queries
        
        # Target: one-hot position of query
        targets = np.zeros((num_samples, size), dtype=np.float32)
        targets[np.arange(num_samples), query_indices] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "task": self.config.task_name,
                "size": size
            }
        )
    
    def _generate_minmax(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate min/max finding task data.
        
        :param num_samples: Number of samples.
        :param size: Array size.
        :return: TaskData for min/max.
        """
        inputs = self._rng.random((num_samples, size, 1)).astype(np.float32)
        
        if self.config.task_name == "minimum":
            target_indices = np.argmin(inputs[:, :, 0], axis=1)
        else:
            target_indices = np.argmax(inputs[:, :, 0], axis=1)
        
        targets = np.zeros((num_samples, size), dtype=np.float32)
        targets[np.arange(num_samples), target_indices] = 1.0
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "task": self.config.task_name,
                "size": size
            }
        )
    
    def _generate_reverse(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate sequence reversal task data.
        
        :param num_samples: Number of samples.
        :param size: Sequence size.
        :return: TaskData for reversal.
        """
        inputs = self._rng.random((num_samples, size, 1)).astype(np.float32)
        targets = inputs[:, ::-1, :].copy()
        
        return TaskData(
            inputs=inputs,
            targets=targets,
            metadata={
                "task": "reverse",
                "size": size
            }
        )
    
    def _generate_graph_traversal(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate graph traversal task data.
        
        :param num_samples: Number of samples.
        :param size: Number of nodes.
        :return: TaskData for graph traversal.
        """
        adj_size = size * size
        inputs = np.zeros((num_samples, adj_size + size), dtype=np.float32)
        targets = np.zeros((num_samples, size), dtype=np.float32)
        
        for i in range(num_samples):
            # Random adjacency matrix
            adj = (self._rng.random((size, size)) < 0.3).astype(np.float32)
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0)
            
            start = self._rng.integers(0, size)
            
            inputs[i, :adj_size] = adj.flatten()
            inputs[i, adj_size + start] = 1.0
            
            # BFS/DFS traversal order
            if self.config.task_name == "bfs":
                order = self._bfs_order(adj, start, size)
            else:
                order = self._dfs_order(adj, start, size)
            
            targets[i] = order
        
        return TaskData(
            inputs=inputs.reshape(num_samples, -1, 1),
            targets=targets,
            metadata={
                "task": self.config.task_name,
                "size": size
            }
        )
    
    def _generate_shortest_path(
        self,
        num_samples: int,
        size: int
    ) -> TaskData:
        """Generate shortest path (Dijkstra) task data.
        
        :param num_samples: Number of samples.
        :param size: Number of nodes.
        :return: TaskData for shortest path.
        """
        adj_size = size * size
        inputs = np.zeros((num_samples, adj_size + size), dtype=np.float32)
        targets = np.zeros((num_samples, size), dtype=np.float32)
        
        for i in range(num_samples):
            # Weighted adjacency matrix
            adj = self._rng.random((size, size)).astype(np.float32)
            mask = (self._rng.random((size, size)) < 0.3).astype(np.float32)
            adj = adj * mask
            adj = (adj + adj.T) / 2  # Symmetric
            np.fill_diagonal(adj, 0)
            
            start = self._rng.integers(0, size)
            
            inputs[i, :adj_size] = adj.flatten()
            inputs[i, adj_size + start] = 1.0
            
            # Compute shortest distances
            distances = self._dijkstra(adj, start, size)
            targets[i] = distances / (np.max(distances) + 1e-8)  # Normalize
        
        return TaskData(
            inputs=inputs.reshape(num_samples, -1, 1),
            targets=targets,
            metadata={
                "task": "dijkstra",
                "size": size
            }
        )
    
    def _bfs_order(
        self,
        adj: np.ndarray,
        start: int,
        size: int
    ) -> np.ndarray:
        """Compute BFS visitation order.
        
        :param adj: Adjacency matrix.
        :param start: Start node.
        :param size: Number of nodes.
        :return: Normalized visitation order.
        """
        visited = np.zeros(size, dtype=np.float32)
        order = np.zeros(size, dtype=np.float32)
        queue = [start]
        visited[start] = 1.0
        visit_order = 1
        
        while queue:
            node = queue.pop(0)
            order[node] = visit_order / size
            visit_order += 1
            
            for neighbor in range(size):
                if adj[node, neighbor] > 0 and visited[neighbor] == 0:
                    visited[neighbor] = 1.0
                    queue.append(neighbor)
        
        return order
    
    def _dfs_order(
        self,
        adj: np.ndarray,
        start: int,
        size: int
    ) -> np.ndarray:
        """Compute DFS visitation order.
        
        :param adj: Adjacency matrix.
        :param start: Start node.
        :param size: Number of nodes.
        :return: Normalized visitation order.
        """
        visited = np.zeros(size, dtype=np.float32)
        order = np.zeros(size, dtype=np.float32)
        stack = [start]
        visit_order = 1
        
        while stack:
            node = stack.pop()
            if visited[node] == 0:
                visited[node] = 1.0
                order[node] = visit_order / size
                visit_order += 1
                
                for neighbor in range(size - 1, -1, -1):
                    if adj[node, neighbor] > 0 and visited[neighbor] == 0:
                        stack.append(neighbor)
        
        return order
    
    def _dijkstra(
        self,
        adj: np.ndarray,
        start: int,
        size: int
    ) -> np.ndarray:
        """Compute shortest distances using Dijkstra's algorithm.
        
        :param adj: Weighted adjacency matrix.
        :param start: Start node.
        :param size: Number of nodes.
        :return: Array of shortest distances.
        """
        distances = np.full(size, np.inf, dtype=np.float32)
        distances[start] = 0
        visited = np.zeros(size, dtype=bool)
        
        for _ in range(size):
            # Find minimum distance unvisited node
            min_dist = np.inf
            min_node = -1
            for node in range(size):
                if not visited[node] and distances[node] < min_dist:
                    min_dist = distances[node]
                    min_node = node
            
            if min_node == -1:
                break
            
            visited[min_node] = True
            
            # Update distances
            for neighbor in range(size):
                if adj[min_node, neighbor] > 0:
                    new_dist = distances[min_node] + adj[min_node, neighbor]
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
        
        # Replace inf with large value
        distances = np.where(np.isinf(distances), size, distances)
        return distances
