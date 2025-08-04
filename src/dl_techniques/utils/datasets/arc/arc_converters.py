"""
This module provides a comprehensive suite of utilities for converting, augmenting,
and managing datasets for the Abstraction and Reasoning Corpus (ARC), specifically
designed to work with the data formats used by the Hierarchical Reasoning Model (HRM).

The ARC dataset's unique structure requires significant data wrangling to prepare it
for deep learning models. This module handles the entire pipeline, from parsing the
original JSON files to generating augmented data and splitting it into robust
training, validation, and test sets.

The module is composed of several powerful, single-responsibility classes:

1.  **`ARCFormatConverter` (The Core Transformation Engine):**
    -   This is the central class for converting between different data representations.
    -   It handles the crucial transformation from the standard ARC 2D grid format
        (as found in the original JSON files) into the 1D flattened sequence format
        required by Transformer-based models. This process includes adding special
        `PAD` and `EOS` (End-of-Sequence) tokens and applying a color offset.
    -   It can also perform the reverse operation, converting a model's 1D output
        sequence back into a human-readable 2D grid.

2.  **`ARCDataAugmenter` (Intelligent Data Expansion):**
    -   This class implements data augmentation techniques that are semantically
        meaningful for the abstract tasks in ARC.
    -   It goes beyond simple image flips by applying transformations that preserve
        the underlying logic of a puzzle, such as:
        -   **Dihedral Transformations:** Applying all 8 symmetries (rotations and
          reflections) to the input and output grids consistently.
        -   **Color Permutations:** Systematically swapping colors in a way that
          maintains the puzzle's structure.
    -   This allows for a significant expansion of the training data, helping models
        to learn more generalizable reasoning rules.

3.  **`ARCDatasetMerger` (Combining Datasets):**
    -   A utility for combining ARC tasks from multiple sources (e.g., the official
        dataset plus community-contributed puzzles).
    -   It includes a robust de-duplication mechanism based on hashing the canonical
        representation of a task's examples, ensuring that the final merged dataset
        is clean and free of redundant puzzles.

4.  **`ARCDatasetSplitter` (Leakage-Free Data Splitting):**
    -   This class provides a sophisticated method for splitting a list of ARC tasks
        into training, validation, and test sets.
    -   Crucially, it is designed to prevent data leakage. When dealing with augmented
        data, it ensures that all augmented versions of a single base puzzle are kept
        within the *same* split. This prevents the model from being trained on one
        version of a puzzle and tested on a simple transformation of it, which would
        lead to inflated and misleading performance metrics.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional,  Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Import the dihedral transforms from the project
def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr

# ---------------------------------------------------------------------

def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply inverse dihedral transformation"""
    DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

# ---------------------------------------------------------------------

@dataclass
class ARCTaskData:
    """
    Represents an ARC task in standard JSON format.

    Attributes:
        task_id: Unique identifier for the task
        train: List of train examples (input-output pairs)
        test: List of test examples (may have missing outputs)
    """
    task_id: str
    train: List[Dict[str, List[List[int]]]]
    test: List[Dict[str, List[List[int]]]]

# ---------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Attributes:
        enable_dihedral: Whether to apply dihedral transformations
        enable_color_permutation: Whether to apply color permutations
        enable_translation: Whether to apply translation augmentation
        num_augmentations: Number of augmentations per original example
        preserve_black: Whether to preserve black color (0) during permutation
    """
    enable_dihedral: bool = True
    enable_color_permutation: bool = True
    enable_translation: bool = True
    num_augmentations: int = 8
    preserve_black: bool = True

# ---------------------------------------------------------------------

class ARCFormatConverter:
    """
    Converter between different ARC data formats.

    This class provides methods to convert between the standard ARC JSON format
    and the HRM dataset format, as well as other useful conversions.
    """

    def __init__(self, max_grid_size: int = 30):
        """
        Initialize the converter.

        Args:
            max_grid_size: Maximum grid size for padding
        """
        self.max_grid_size = max_grid_size
        self.pad_token = 0
        self.eos_token = 1
        self.color_offset = 2

    def json_to_task_data(self, json_data: Dict[str, Any], task_id: str) -> ARCTaskData:
        """
        Convert JSON data to ARCTaskData format.

        Args:
            json_data: Raw JSON data from ARC dataset
            task_id: Task identifier

        Returns:
            ARCTaskData object
        """
        return ARCTaskData(
            task_id=task_id,
            train=json_data.get("train", []),
            test=json_data.get("test", [])
        )

    def load_tasks_from_directory(self, directory_path: str) -> List[ARCTaskData]:
        """
        Load all tasks from a directory of JSON files.

        Args:
            directory_path: Path to directory containing JSON files

        Returns:
            List of ARCTaskData objects
        """
        directory = Path(directory_path)
        tasks = []

        for json_file in directory.glob("*.json"):
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            task_id = json_file.stem
            task = self.json_to_task_data(json_data, task_id)
            tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks from {directory_path}")
        return tasks

    def grid_to_sequence(self, grid: List[List[int]], apply_translation: bool = False) -> np.ndarray:
        """
        Convert a 2D grid to a flattened sequence with padding and EOS tokens.

        Args:
            grid: 2D grid as list of lists
            apply_translation: Whether to apply random translation

        Returns:
            Flattened sequence as numpy array
        """
        # Convert to numpy array
        grid_array = np.array(grid, dtype=np.uint8)

        # Apply random translation if requested
        if apply_translation:
            max_pad_r = self.max_grid_size - grid_array.shape[0]
            max_pad_c = self.max_grid_size - grid_array.shape[1]

            if max_pad_r > 0 and max_pad_c > 0:
                pad_r = np.random.randint(0, max_pad_r + 1)
                pad_c = np.random.randint(0, max_pad_c + 1)
            else:
                pad_r = pad_c = 0
        else:
            pad_r = pad_c = 0

        # Create padded grid
        nrow, ncol = grid_array.shape
        padded_grid = np.full((self.max_grid_size, self.max_grid_size), self.pad_token, dtype=np.uint8)

        # Place original grid
        padded_grid[pad_r:pad_r + nrow, pad_c:pad_c + ncol] = grid_array + self.color_offset

        # Add EOS tokens
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < self.max_grid_size:
            padded_grid[eos_row, pad_c:eos_col] = self.eos_token
        if eos_col < self.max_grid_size:
            padded_grid[pad_r:eos_row, eos_col] = self.eos_token

        return padded_grid.flatten()

    def sequence_to_grid(self, sequence: np.ndarray) -> np.ndarray:
        """
        Convert a flattened sequence back to a 2D grid.

        Args:
            sequence: Flattened sequence

        Returns:
            2D grid with padding and EOS tokens removed
        """
        # Reshape to 2D
        grid = sequence.reshape(self.max_grid_size, self.max_grid_size)

        # Find content boundaries
        content_mask = (grid != self.pad_token) & (grid != self.eos_token)

        if np.any(content_mask):
            rows, cols = np.where(content_mask)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1

            # Extract content and convert back to original color space
            content_grid = grid[min_row:max_row, min_col:max_col]
            content_grid = np.maximum(content_grid - self.color_offset, 0)
        else:
            # Empty grid
            content_grid = np.array([[0]], dtype=np.uint8)

        return content_grid.astype(np.uint8)

    def task_to_hrm_format(self, task: ARCTaskData,
                           augmentation_config: Optional[AugmentationConfig] = None) -> Dict[str, Any]:
        """
        Convert ARCTaskData to HRM dataset format.

        Args:
            task: ARCTaskData to convert
            augmentation_config: Configuration for augmentation

        Returns:
            Dictionary with HRM format data
        """
        if augmentation_config is None:
            augmentation_config = AugmentationConfig(num_augmentations=0)

        # Collect all examples
        all_examples = []

        # Process train examples
        for example in task.train:
            input_grid = example["input"]
            output_grid = example["output"]
            all_examples.append((input_grid, output_grid))

        # Process test examples (if they have outputs)
        for example in task.test:
            if "output" in example:
                input_grid = example["input"]
                output_grid = example["output"]
                all_examples.append((input_grid, output_grid))

        # Generate base examples and augmentations
        inputs = []
        labels = []

        for input_grid, output_grid in all_examples:
            # Add original example
            input_seq = self.grid_to_sequence(input_grid, apply_translation=augmentation_config.enable_translation)
            output_seq = self.grid_to_sequence(output_grid, apply_translation=augmentation_config.enable_translation)

            inputs.append(input_seq)
            labels.append(output_seq)

            # Add augmentations
            if augmentation_config.num_augmentations > 0:
                for _ in range(augmentation_config.num_augmentations):
                    aug_input, aug_output = self._apply_augmentation(
                        input_grid, output_grid, augmentation_config
                    )

                    input_seq = self.grid_to_sequence(aug_input,
                                                      apply_translation=augmentation_config.enable_translation)
                    output_seq = self.grid_to_sequence(aug_output,
                                                       apply_translation=augmentation_config.enable_translation)

                    inputs.append(input_seq)
                    labels.append(output_seq)

        return {
            "task_id": task.task_id,
            "inputs": np.array(inputs),
            "labels": np.array(labels),
            "num_examples": len(inputs)
        }

    def _apply_augmentation(self, input_grid: List[List[int]], output_grid: List[List[int]],
                            config: AugmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to input and output grids.

        Args:
            input_grid: Input grid as list of lists
            output_grid: Output grid as list of lists
            config: Augmentation configuration

        Returns:
            Tuple of augmented (input, output) grids
        """
        # Convert to numpy arrays
        input_array = np.array(input_grid, dtype=np.uint8)
        output_array = np.array(output_grid, dtype=np.uint8)

        # Apply dihedral transformation
        if config.enable_dihedral:
            transform_id = np.random.randint(0, 8)
            input_array = dihedral_transform(input_array, transform_id)
            output_array = dihedral_transform(output_array, transform_id)

        # Apply color permutation
        if config.enable_color_permutation:
            # Create color mapping
            if config.preserve_black:
                # Keep black (0) unchanged, permute colors 1-9
                colors_to_permute = np.arange(1, 10)
                permuted_colors = np.random.permutation(colors_to_permute)
                color_mapping = np.arange(10)
                color_mapping[1:] = permuted_colors
            else:
                # Permute all colors 0-9
                color_mapping = np.random.permutation(10)

            # Apply mapping
            input_array = color_mapping[input_array]
            output_array = color_mapping[output_array]

        return input_array, output_array

# ---------------------------------------------------------------------

class ARCDatasetMerger:
    """
    Utility for merging multiple ARC datasets.

    This class provides methods to combine datasets from different sources
    while maintaining proper indexing and avoiding duplicates.
    """

    def __init__(self):
        """Initialize the dataset merger."""
        self.converter = ARCFormatConverter()

    def merge_task_lists(self, task_lists: List[List[ARCTaskData]],
                         deduplicate: bool = True) -> List[ARCTaskData]:
        """
        Merge multiple lists of tasks.

        Args:
            task_lists: List of task lists to merge
            deduplicate: Whether to remove duplicate tasks

        Returns:
            Merged list of tasks
        """
        merged_tasks = []
        seen_hashes = set()

        for task_list in task_lists:
            for task in task_list:
                if deduplicate:
                    task_hash = self._compute_task_hash(task)
                    if task_hash in seen_hashes:
                        logger.debug(f"Skipping duplicate task: {task.task_id}")
                        continue
                    seen_hashes.add(task_hash)

                merged_tasks.append(task)

        logger.info(f"Merged {len(merged_tasks)} tasks from {len(task_lists)} sources")
        return merged_tasks

    def _compute_task_hash(self, task: ARCTaskData) -> str:
        """
        Compute a hash for a task to detect duplicates.

        Args:
            task: Task to hash

        Returns:
            Hash string
        """
        # Create a canonical representation of the task
        task_repr = {
            "train": sorted([self._example_to_tuple(ex) for ex in task.train]),
            "test": sorted([self._example_to_tuple(ex) for ex in task.test])
        }

        # Convert to string and hash
        task_str = json.dumps(task_repr, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()

    def _example_to_tuple(self, example: Dict[str, List[List[int]]]) -> Tuple:
        """Convert an example to a hashable tuple."""
        input_tuple = tuple(tuple(row) for row in example["input"])
        output_tuple = tuple(tuple(row) for row in example.get("output", []))
        return (input_tuple, output_tuple)

# ---------------------------------------------------------------------

class ARCDataAugmenter:
    """
    Advanced data augmentation for ARC datasets.

    This class provides sophisticated augmentation techniques specifically
    designed for ARC puzzles, including semantic-preserving transformations.
    """

    def __init__(self, max_grid_size: int = 30):
        """
        Initialize the augmenter.

        Args:
            max_grid_size: Maximum grid size
        """
        self.max_grid_size = max_grid_size
        self.converter = ARCFormatConverter(max_grid_size)

    def augment_task(self, task: ARCTaskData, config: AugmentationConfig) -> List[ARCTaskData]:
        """
        Generate augmented versions of a task.

        Args:
            task: Original task
            config: Augmentation configuration

        Returns:
            List of augmented tasks (including original)
        """
        augmented_tasks = [task]  # Include original

        if config.num_augmentations == 0:
            return augmented_tasks

        # Generate augmentations
        for aug_id in range(config.num_augmentations):
            augmented_task = self._create_augmented_task(task, config, aug_id)
            augmented_tasks.append(augmented_task)

        return augmented_tasks

    def _create_augmented_task(self, task: ARCTaskData, config: AugmentationConfig,
                               aug_id: int) -> ARCTaskData:
        """
        Create a single augmented version of a task.

        Args:
            task: Original task
            config: Augmentation configuration
            aug_id: Augmentation identifier

        Returns:
            Augmented task
        """
        # Determine transformations to apply
        transform_id = aug_id % 8 if config.enable_dihedral else 0
        apply_color_perm = config.enable_color_permutation and (aug_id % 2 == 1)

        # Generate color mapping
        if apply_color_perm:
            if config.preserve_black:
                colors_to_permute = np.arange(1, 10)
                permuted_colors = np.random.permutation(colors_to_permute)
                color_mapping = np.arange(10)
                color_mapping[1:] = permuted_colors
            else:
                color_mapping = np.random.permutation(10)
        else:
            color_mapping = np.arange(10)  # Identity mapping

        # Create transformation name
        transform_name = f"t{transform_id}_{''.join(map(str, color_mapping))}"

        # Apply transformations
        augmented_train = []
        for example in task.train:
            aug_example = self._transform_example(example, transform_id, color_mapping)
            augmented_train.append(aug_example)

        augmented_test = []
        for example in task.test:
            aug_example = self._transform_example(example, transform_id, color_mapping)
            augmented_test.append(aug_example)

        return ARCTaskData(
            task_id=f"{task.task_id}_{transform_name}",
            train=augmented_train,
            test=augmented_test
        )

    def _transform_example(self, example: Dict[str, List[List[int]]],
                           transform_id: int, color_mapping: np.ndarray) -> Dict[str, List[List[int]]]:
        """
        Transform a single example.

        Args:
            example: Original example
            transform_id: Dihedral transformation ID
            color_mapping: Color permutation mapping

        Returns:
            Transformed example
        """
        # Transform input
        input_array = np.array(example["input"], dtype=np.uint8)
        input_array = dihedral_transform(input_array, transform_id)
        input_array = color_mapping[input_array]
        transformed_input = input_array.tolist()

        # Transform output (if present)
        transformed_example = {"input": transformed_input}

        if "output" in example:
            output_array = np.array(example["output"], dtype=np.uint8)
            output_array = dihedral_transform(output_array, transform_id)
            output_array = color_mapping[output_array]
            transformed_example["output"] = output_array.tolist()

        return transformed_example

# ---------------------------------------------------------------------

class ARCDatasetSplitter:
    """
    Utility for splitting ARC datasets into train/validation/test sets.

    This class provides methods to split datasets while maintaining
    proper balance and avoiding data leakage.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the splitter.

        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def split_tasks(self, tasks: List[ARCTaskData],
                    train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                    ensure_no_augmented_leakage: bool = True) -> Dict[str, List[ARCTaskData]]:
        """
        Split tasks into train/validation/test sets.

        Args:
            tasks: List of tasks to split
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            ensure_no_augmented_leakage: Whether to ensure augmented tasks don't leak across splits

        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        # Group tasks by base ID to avoid leakage
        if ensure_no_augmented_leakage:
            task_groups = self._group_tasks_by_base_id(tasks)
            base_ids = list(task_groups.keys())
            np.random.shuffle(base_ids)

            # Split base IDs
            n_total = len(base_ids)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_base_ids = base_ids[:n_train]
            val_base_ids = base_ids[n_train:n_train + n_val]
            test_base_ids = base_ids[n_train + n_val:]

            # Collect tasks for each split
            train_tasks = []
            val_tasks = []
            test_tasks = []

            for base_id in train_base_ids:
                train_tasks.extend(task_groups[base_id])
            for base_id in val_base_ids:
                val_tasks.extend(task_groups[base_id])
            for base_id in test_base_ids:
                test_tasks.extend(task_groups[base_id])

        else:
            # Simple random split
            shuffled_tasks = tasks.copy()
            np.random.shuffle(shuffled_tasks)

            n_total = len(shuffled_tasks)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_tasks = shuffled_tasks[:n_train]
            val_tasks = shuffled_tasks[n_train:n_train + n_val]
            test_tasks = shuffled_tasks[n_train + n_val:]

        logger.info(f"Split {len(tasks)} tasks into: train={len(train_tasks)}, "
                    f"val={len(val_tasks)}, test={len(test_tasks)}")

        return {
            "train": train_tasks,
            "val": val_tasks,
            "test": test_tasks
        }

    def _group_tasks_by_base_id(self, tasks: List[ARCTaskData]) -> Dict[str, List[ARCTaskData]]:
        """
        Group tasks by their base ID (without augmentation suffix).

        Args:
            tasks: List of tasks to group

        Returns:
            Dictionary mapping base IDs to lists of tasks
        """
        groups = {}

        for task in tasks:
            base_id = task.task_id.split("_")[0]  # Remove augmentation suffix
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(task)

        return groups

# ---------------------------------------------------------------------
