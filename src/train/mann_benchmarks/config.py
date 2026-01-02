"""
Configuration dataclasses for Memory-Augmented Neural Network benchmarks.

This module provides configuration classes for benchmark task generation
including copy tasks, associative recall, and compositional generalization.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CopyTaskConfig:
    """Configuration for the Copy Task benchmark.
    
    The copy task requires the model to store an input sequence in memory
    and reproduce it after a delay period.
    
    :param sequence_length: Length of the sequence to copy.
    :param vector_size: Dimensionality of each vector in the sequence.
    :param min_sequence_length: Minimum sequence length for variable-length tasks.
    :param max_sequence_length: Maximum sequence length for length generalization tests.
    :param num_samples: Number of samples to generate.
    :param delay_length: Number of blank timesteps between input and output.
    :param random_seed: Seed for reproducibility.
    """
    sequence_length: int = 10
    vector_size: int = 8
    min_sequence_length: int = 1
    max_sequence_length: int = 20
    num_samples: int = 1000
    delay_length: int = 1
    random_seed: int = 42


@dataclass
class AssociativeRecallConfig:
    """Configuration for the Associative Recall Task benchmark.
    
    The task requires the model to store key-value pairs and retrieve
    the correct value when prompted with a query key.
    
    :param num_items: Number of key-value pairs to store.
    :param key_size: Dimensionality of key vectors.
    :param value_size: Dimensionality of value vectors.
    :param min_items: Minimum number of items for variable difficulty.
    :param max_items: Maximum number of items for capacity testing.
    :param num_samples: Number of samples to generate.
    :param random_seed: Seed for reproducibility.
    """
    num_items: int = 6
    key_size: int = 4
    value_size: int = 4
    min_items: int = 2
    max_items: int = 16
    num_samples: int = 1000
    random_seed: int = 42


@dataclass
class BabiTaskConfig:
    """Configuration for bAbI-style Question Answering tasks.
    
    The bAbI tasks test various reasoning capabilities including
    fact retrieval, counting, path-finding, and spatial reasoning.
    
    :param task_ids: List of task IDs to generate (1-20).
    :param vocab_size: Size of the vocabulary.
    :param max_story_length: Maximum number of sentences in a story.
    :param max_sentence_length: Maximum tokens per sentence.
    :param num_samples_per_task: Number of samples per task.
    :param random_seed: Seed for reproducibility.
    """
    task_ids: List[int] = field(default_factory=lambda: list(range(1, 21)))
    vocab_size: int = 200
    max_story_length: int = 70
    max_sentence_length: int = 15
    num_samples_per_task: int = 1000
    random_seed: int = 42


@dataclass
class ScanTaskConfig:
    """Configuration for SCAN compositional generalization benchmark.
    
    SCAN tests systematic recombination of navigation commands,
    a key test for compositional generalization.
    
    :param split_type: Type of train/test split.
        Options: 'simple', 'length', 'add_prim_jump', 'add_prim_turn_left'.
    :param max_input_length: Maximum command sequence length.
    :param max_output_length: Maximum action sequence length.
    :param num_samples: Number of samples to generate.
    :param random_seed: Seed for reproducibility.
    """
    split_type: str = "simple"
    max_input_length: int = 10
    max_output_length: int = 48
    num_samples: int = 10000
    random_seed: int = 42


@dataclass
class AlgorithmicTaskConfig:
    """Configuration for CLRS-style algorithmic reasoning tasks.
    
    Tests algorithmic reasoning on sorting, searching, graph algorithms,
    and dynamic programming problems.
    
    :param task_name: Name of the algorithm to test.
    :param train_size: Number of nodes/elements for training.
    :param test_size: Number of nodes/elements for OOD testing.
    :param num_samples: Number of samples to generate.
    :param random_seed: Seed for reproducibility.
    """
    task_name: str = "insertion_sort"
    train_size: int = 16
    test_size: int = 64
    num_samples: int = 1000
    random_seed: int = 42


@dataclass
class LengthGeneralizationConfig:
    """Configuration for length generalization evaluation.
    
    :param train_lengths: Sequence lengths used during training.
    :param test_lengths: Sequence lengths for evaluation.
    :param num_samples_per_length: Samples per length configuration.
    :param random_seed: Seed for reproducibility.
    """
    train_lengths: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    test_lengths: List[int] = field(default_factory=lambda: [25, 30, 40, 50, 100])
    num_samples_per_length: int = 100
    random_seed: int = 42


@dataclass
class BenchmarkSuiteConfig:
    """Master configuration for the complete benchmark suite.
    
    :param copy_config: Configuration for copy task.
    :param associative_recall_config: Configuration for associative recall.
    :param babi_config: Configuration for bAbI tasks.
    :param scan_config: Configuration for SCAN tasks.
    :param algorithmic_config: Configuration for algorithmic tasks.
    :param length_gen_config: Configuration for length generalization.
    :param output_dir: Directory to save benchmark results.
    :param verbose: Whether to print progress information.
    """
    copy_config: CopyTaskConfig = field(default_factory=CopyTaskConfig)
    associative_recall_config: AssociativeRecallConfig = field(
        default_factory=AssociativeRecallConfig
    )
    babi_config: BabiTaskConfig = field(default_factory=BabiTaskConfig)
    scan_config: ScanTaskConfig = field(default_factory=ScanTaskConfig)
    algorithmic_config: AlgorithmicTaskConfig = field(
        default_factory=AlgorithmicTaskConfig
    )
    length_gen_config: LengthGeneralizationConfig = field(
        default_factory=LengthGeneralizationConfig
    )
    output_dir: str = "./benchmark_results"
    verbose: bool = True
