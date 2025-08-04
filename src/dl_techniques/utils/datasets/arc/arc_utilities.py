"""
This module provides a comprehensive, object-oriented toolkit for loading, analyzing,
visualizing, and validating datasets for the Abstraction and Reasoning Corpus (ARC),
specifically those formatted for the Hierarchical Reasoning Model (HRM) project.

The ARC dataset presents unique data handling challenges due to its structure: a
collection of "puzzles," each containing multiple input-output "examples" represented
as 2D grids. This module provides a set of high-level, structured utilities to
abstract away the complexities of file I/O, data parsing, and format conversion,
allowing researchers and developers to focus on model development.

The module is built around a few core concepts and classes:

1.  **Data Structures (`ARCPuzzle`, `ARCExample`, `ARCDatasetStats`):**
    -   These `dataclass` objects provide a clean, type-hinted, and intuitive way to
        represent the hierarchical structure of the ARC dataset in memory. An `ARCPuzzle`
        contains a list of `ARCExample` objects, making it easy to iterate through and
        work with individual task instances.

2.  **`ARCDatasetLoader` (The Core Data Access Layer):**
    -   This is the central class for interacting with a pre-processed ARC dataset.
    -   It handles the loading of all necessary files, including the raw `.npy` arrays
        (inputs, labels, identifiers) and the JSON metadata files.
    -   Its most important function is to reconstruct the original `ARCPuzzle` and
        `ARCExample` structures from the flattened, pre-processed numpy arrays. It
        uses the stored index files to correctly group examples into puzzles and
        handles the conversion of 1D token sequences back into their 2D grid format.

3.  **`ARCDatasetAnalyzer` (Data Insights and Understanding):**
    -   This class uses the `ARCDatasetLoader` to perform statistical analysis on the
        dataset.
    -   It can compute high-level statistics like the number of puzzles and examples,
        as well as more detailed metrics such as the distribution of grid sizes,
        the frequency of different colors, and heuristic-based measures of puzzle
        complexity.

4.  **`ARCDatasetVisualizer` (Visual Inspection):**
    -   A powerful tool for visually inspecting the data. It uses `matplotlib` to
        render the 2D grids with the correct ARC color palette.
    -   It can plot a single puzzle with all its input-output examples, create
        comparison grids of multiple puzzles, and generate charts from the statistics
        computed by the `ARCDatasetAnalyzer`.

5.  **`ARCDatasetValidator` (Data Integrity and Health Checks):**
    -   This class provides a suite of validation checks to ensure a dataset is
        well-formed and free of corruption.
    -   It verifies file structures, checks for consistency between metadata and
        actual data, validates data types and value ranges, and ensures that the
        indexing arrays are structured correctly. This is crucial for debugging
        data pipelines and ensuring reproducible results.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# ARC color palette for visualization
ARC_COLOR_MAP = mcolors.ListedColormap([
    "#000000",  # symbol_0: black
    "#0074D9",  # symbol_1: blue
    "#FF4136",  # symbol_2: red
    "#2ECC40",  # symbol_3: green
    "#FFDC00",  # symbol_4: yellow
    "#AAAAAA",  # symbol_5: grey
    "#F012BE",  # symbol_6: fuchsia
    "#FF851B",  # symbol_7: orange
    "#7FDBFF",  # symbol_8: teal
    "#870C25"  # symbol_9: brown
])

# Constants from the project
ARC_MAX_GRID_SIZE = 30
ARC_VOCAB_SIZE = 12  # PAD + EOS + colors 0-9
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2


@dataclass
class ARCExample:
    """
    Represents a single ARC example with input and output grids.

    Attributes:
        input_grid: Input grid as numpy array
        output_grid: Output grid as numpy array
        puzzle_id: Identifier of the parent puzzle
        example_id: Unique identifier for this example
    """
    input_grid: np.ndarray
    output_grid: np.ndarray
    puzzle_id: str
    example_id: int


@dataclass
class ARCPuzzle:
    """
    Represents an ARC puzzle with multiple examples.

    Attributes:
        puzzle_id: Unique identifier for the puzzle
        examples: List of examples (input-output pairs)
        metadata: Additional metadata about the puzzle
    """
    puzzle_id: str
    examples: List[ARCExample]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ARCDatasetStats:
    """
    Statistics about an ARC dataset.

    Attributes:
        num_puzzles: Total number of puzzles
        num_examples: Total number of examples
        avg_examples_per_puzzle: Average examples per puzzle
        grid_size_stats: Statistics about grid sizes
        color_usage_stats: Statistics about color usage
    """
    num_puzzles: int
    num_examples: int
    avg_examples_per_puzzle: float
    grid_size_stats: Dict[str, Any]
    color_usage_stats: Dict[str, Any]


class ARCDatasetLoader:
    """
    Utility class for loading ARC datasets in HRM format.

    This class provides methods to load processed ARC datasets that were
    created using the build_arc_dataset.py script.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.

        Args:
            dataset_path: Path to the processed ARC dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.identifiers_map: Optional[Dict[int, str]] = None
        self._validate_dataset_structure()

    def _validate_dataset_structure(self) -> None:
        """
        Validate that the dataset has the expected structure.

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If dataset structure is invalid
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Check for identifiers.json
        identifiers_file = self.dataset_path / "identifiers.json"
        if not identifiers_file.exists():
            raise FileNotFoundError(f"Missing identifiers.json: {identifiers_file}")

        # Check for train and test directories
        for split in ["train", "test"]:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                logger.warning(f"Missing split directory: {split_dir}")
                continue

            # Check for required files
            required_files = [
                "dataset.json",
                "all__inputs.npy",
                "all__labels.npy",
                "all__puzzle_identifiers.npy",
                "all__puzzle_indices.npy",
                "all__group_indices.npy"
            ]

            for file_name in required_files:
                file_path = split_dir / file_name
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing required file: {file_path}")

    def load_identifiers_map(self) -> Dict[int, str]:
        """
        Load the puzzle identifiers mapping.

        Returns:
            Dictionary mapping puzzle ID integers to puzzle name strings
        """
        if self.identifiers_map is None:
            identifiers_file = self.dataset_path / "identifiers.json"
            with open(identifiers_file, 'r') as f:
                identifiers_list = json.load(f)
            self.identifiers_map = {i: name for i, name in enumerate(identifiers_list)}

        return self.identifiers_map

    def load_split_metadata(self, split: str) -> Dict[str, Any]:
        """
        Load metadata for a specific split.

        Args:
            split: Dataset split ('train' or 'test')

        Returns:
            Metadata dictionary
        """
        metadata_file = self.dataset_path / split / "dataset.json"
        with open(metadata_file, 'r') as f:
            return json.load(f)

    def load_split_data(self, split: str, subset: str = "all") -> Dict[str, np.ndarray]:
        """
        Load raw data arrays for a specific split and subset.

        Args:
            split: Dataset split ('train' or 'test')
            subset: Dataset subset (default: 'all')

        Returns:
            Dictionary containing loaded numpy arrays
        """
        split_dir = self.dataset_path / split

        # Load all required arrays
        data = {}
        array_names = ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]

        for array_name in array_names:
            file_path = split_dir / f"{subset}__{array_name}.npy"
            data[array_name] = np.load(file_path)

        return data

    def load_puzzles(self, split: str, subset: str = "all",
                     include_augmented: bool = False) -> List[ARCPuzzle]:
        """
        Load puzzles as structured objects.

        Args:
            split: Dataset split ('train' or 'test')
            subset: Dataset subset (default: 'all')
            include_augmented: Whether to include augmented puzzles

        Returns:
            List of ARCPuzzle objects
        """
        data = self.load_split_data(split, subset)
        identifiers_map = self.load_identifiers_map()

        puzzles = []

        # Process each group (puzzle)
        group_indices = data["group_indices"]
        num_groups = len(group_indices) - 1

        for group_idx in range(num_groups):
            group_start = group_indices[group_idx]
            group_end = group_indices[group_idx + 1]

            # Get puzzles in this group
            group_puzzle_ids = data["puzzle_identifiers"][group_start:group_end]
            unique_puzzle_ids = np.unique(group_puzzle_ids)

            for puzzle_id_int in unique_puzzle_ids:
                puzzle_name = identifiers_map.get(puzzle_id_int, f"<unknown_{puzzle_id_int}>")

                # Skip augmented puzzles if not requested
                if not include_augmented and "_" in puzzle_name:
                    continue

                # Find all examples for this puzzle
                puzzle_mask = group_puzzle_ids == puzzle_id_int
                puzzle_indices_in_group = np.where(puzzle_mask)[0] + group_start

                examples = []
                for puzzle_idx in puzzle_indices_in_group:
                    # Get example range
                    example_start = data["puzzle_indices"][puzzle_idx]
                    example_end = data["puzzle_indices"][puzzle_idx + 1]

                    # Load examples for this puzzle
                    for example_idx in range(example_start, example_end):
                        input_seq = data["inputs"][example_idx]
                        output_seq = data["labels"][example_idx]

                        # Convert sequences back to grids
                        input_grid = self._sequence_to_grid(input_seq)
                        output_grid = self._sequence_to_grid(output_seq)

                        example = ARCExample(
                            input_grid=input_grid,
                            output_grid=output_grid,
                            puzzle_id=puzzle_name,
                            example_id=example_idx
                        )
                        examples.append(example)

                if examples:  # Only add puzzles with examples
                    puzzle = ARCPuzzle(
                        puzzle_id=puzzle_name,
                        examples=examples
                    )
                    puzzles.append(puzzle)

        return puzzles

    def _sequence_to_grid(self, sequence: np.ndarray) -> np.ndarray:
        """
        Convert a flattened sequence back to a 2D grid.

        Args:
            sequence: Flattened sequence of length ARC_MAX_GRID_SIZE^2

        Returns:
            2D grid with padding and EOS tokens removed
        """
        # Reshape to 2D grid
        grid = sequence.reshape(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)

        # Find the actual content area (before EOS tokens)
        # EOS tokens mark the end of content
        eos_positions = np.where(grid == EOS_TOKEN)

        if len(eos_positions[0]) > 0:
            # Find the bounding box of actual content
            max_row = np.min(eos_positions[0]) if len(eos_positions[0]) > 0 else ARC_MAX_GRID_SIZE
            max_col = np.min(eos_positions[1]) if len(eos_positions[1]) > 0 else ARC_MAX_GRID_SIZE
        else:
            # No EOS tokens found, find content boundary
            content_mask = grid != PAD_TOKEN
            if np.any(content_mask):
                content_rows, content_cols = np.where(content_mask)
                max_row = np.max(content_rows) + 1
                max_col = np.max(content_cols) + 1
            else:
                max_row = max_col = 1  # Minimum size

        # Extract content and convert back to original color space
        content_grid = grid[:max_row, :max_col]
        content_grid = np.maximum(content_grid - COLOR_OFFSET, 0)  # Remove color offset, clip negatives

        return content_grid.astype(np.uint8)


class ARCDatasetAnalyzer:
    """
    Utility class for analyzing ARC datasets.

    Provides methods to compute statistics, analyze patterns, and
    generate insights about ARC datasets.
    """

    def __init__(self, dataset_loader: ARCDatasetLoader):
        """
        Initialize the analyzer.

        Args:
            dataset_loader: Initialized ARCDatasetLoader instance
        """
        self.loader = dataset_loader

    def compute_dataset_statistics(self, split: str, subset: str = "all") -> ARCDatasetStats:
        """
        Compute comprehensive statistics for a dataset split.

        Args:
            split: Dataset split ('train' or 'test')
            subset: Dataset subset (default: 'all')

        Returns:
            ARCDatasetStats object with computed statistics
        """
        puzzles = self.loader.load_puzzles(split, subset, include_augmented=False)

        # Basic counts
        num_puzzles = len(puzzles)
        num_examples = sum(len(puzzle.examples) for puzzle in puzzles)
        avg_examples_per_puzzle = num_examples / num_puzzles if num_puzzles > 0 else 0

        # Grid size analysis
        input_sizes = []
        output_sizes = []

        # Color usage analysis
        color_counts = np.zeros(10, dtype=int)  # Colors 0-9

        for puzzle in puzzles:
            for example in puzzle.examples:
                # Grid sizes
                input_sizes.append(example.input_grid.shape)
                output_sizes.append(example.output_grid.shape)

                # Color usage
                for grid in [example.input_grid, example.output_grid]:
                    unique_colors, counts = np.unique(grid, return_counts=True)
                    for color, count in zip(unique_colors, counts):
                        if 0 <= color <= 9:
                            color_counts[color] += count

        # Compute grid size statistics
        input_heights = [size[0] for size in input_sizes]
        input_widths = [size[1] for size in input_sizes]
        output_heights = [size[0] for size in output_sizes]
        output_widths = [size[1] for size in output_sizes]

        grid_size_stats = {
            "input_height": {"mean": np.mean(input_heights), "std": np.std(input_heights),
                             "min": np.min(input_heights), "max": np.max(input_heights)},
            "input_width": {"mean": np.mean(input_widths), "std": np.std(input_widths),
                            "min": np.min(input_widths), "max": np.max(input_widths)},
            "output_height": {"mean": np.mean(output_heights), "std": np.std(output_heights),
                              "min": np.min(output_heights), "max": np.max(output_heights)},
            "output_width": {"mean": np.mean(output_widths), "std": np.std(output_widths),
                             "min": np.min(output_widths), "max": np.max(output_widths)},
        }

        # Color usage statistics
        total_pixels = np.sum(color_counts)
        color_usage_stats = {
            "counts": color_counts.tolist(),
            "frequencies": (color_counts / total_pixels).tolist() if total_pixels > 0 else [0] * 10,
            "most_common": int(np.argmax(color_counts)),
            "least_common": int(np.argmin(color_counts[color_counts > 0])) if np.any(color_counts > 0) else 0
        }

        return ARCDatasetStats(
            num_puzzles=num_puzzles,
            num_examples=num_examples,
            avg_examples_per_puzzle=avg_examples_per_puzzle,
            grid_size_stats=grid_size_stats,
            color_usage_stats=color_usage_stats
        )

    def analyze_puzzle_complexity(self, puzzle: ARCPuzzle) -> Dict[str, float]:
        """
        Analyze the complexity of a specific puzzle.

        Args:
            puzzle: ARCPuzzle to analyze

        Returns:
            Dictionary with complexity metrics
        """
        if not puzzle.examples:
            return {"error": "No examples in puzzle"}

        # Grid size complexity
        input_sizes = [example.input_grid.size for example in puzzle.examples]
        output_sizes = [example.output_grid.size for example in puzzle.examples]

        # Color complexity
        input_unique_colors = []
        output_unique_colors = []

        for example in puzzle.examples:
            input_unique_colors.append(len(np.unique(example.input_grid)))
            output_unique_colors.append(len(np.unique(example.output_grid)))

        # Transformation complexity (simple heuristics)
        size_changes = []
        for example in puzzle.examples:
            input_size = example.input_grid.size
            output_size = example.output_grid.size
            size_changes.append(abs(output_size - input_size) / input_size)

        return {
            "avg_input_size": np.mean(input_sizes),
            "avg_output_size": np.mean(output_sizes),
            "avg_input_colors": np.mean(input_unique_colors),
            "avg_output_colors": np.mean(output_unique_colors),
            "avg_size_change_ratio": np.mean(size_changes),
            "num_examples": len(puzzle.examples)
        }


class ARCDatasetVisualizer:
    """
    Utility class for visualizing ARC datasets.

    Provides methods to create visualizations of puzzles, grids,
    and dataset statistics.
    """

    def __init__(self, dataset_loader: ARCDatasetLoader):
        """
        Initialize the visualizer.

        Args:
            dataset_loader: Initialized ARCDatasetLoader instance
        """
        self.loader = dataset_loader

    def plot_puzzle(self, puzzle: ARCPuzzle, max_examples: Optional[int] = None,
                    figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a complete puzzle with all its examples.

        Args:
            puzzle: ARCPuzzle to visualize
            max_examples: Maximum number of examples to show
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        examples_to_show = puzzle.examples[:max_examples] if max_examples else puzzle.examples
        num_examples = len(examples_to_show)

        if num_examples == 0:
            logger.warning(f"No examples to show for puzzle {puzzle.puzzle_id}")
            return plt.figure(figsize=figsize)

        # Create subplot grid: 2 columns (input, output) per example
        fig, axes = plt.subplots(num_examples, 2, figsize=figsize)
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"ARC Puzzle: {puzzle.puzzle_id}", fontsize=16, fontweight='bold')

        for i, example in enumerate(examples_to_show):
            # Input grid
            self._plot_grid(axes[i, 0], example.input_grid, f"Input {i + 1}")

            # Output grid
            self._plot_grid(axes[i, 1], example.output_grid, f"Output {i + 1}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Puzzle visualization saved to {save_path}")

        return fig

    def _plot_grid(self, ax: plt.Axes, grid: np.ndarray, title: str) -> None:
        """
        Plot a single grid on the given axes.

        Args:
            ax: Matplotlib axes to plot on
            grid: 2D numpy array representing the grid
            title: Title for the subplot
        """
        # Create image
        im = ax.imshow(grid, cmap=ARC_COLOR_MAP, vmin=0, vmax=9)

        # Set title
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.5)

        # Remove tick labels but keep ticks for grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add size annotation
        ax.text(0.02, 0.98, f"{grid.shape[0]}×{grid.shape[1]}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def plot_dataset_statistics(self, stats: ARCDatasetStats,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot dataset statistics.

        Args:
            stats: ARCDatasetStats object
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ARC Dataset Statistics', fontsize=16, fontweight='bold')

        # Grid size distribution
        ax = axes[0, 0]
        sizes = ['Input H', 'Input W', 'Output H', 'Output W']
        means = [
            stats.grid_size_stats['input_height']['mean'],
            stats.grid_size_stats['input_width']['mean'],
            stats.grid_size_stats['output_height']['mean'],
            stats.grid_size_stats['output_width']['mean']
        ]
        ax.bar(sizes, means)
        ax.set_title('Average Grid Dimensions')
        ax.set_ylabel('Average Size')

        # Color frequency
        ax = axes[0, 1]
        colors = list(range(10))
        frequencies = stats.color_usage_stats['frequencies']
        bars = ax.bar(colors, frequencies, color=[ARC_COLOR_MAP.colors[i] for i in colors])
        ax.set_title('Color Usage Frequency')
        ax.set_xlabel('Color ID')
        ax.set_ylabel('Frequency')
        ax.set_xticks(colors)

        # Basic statistics
        ax = axes[1, 0]
        basic_stats = [
            ('Puzzles', stats.num_puzzles),
            ('Examples', stats.num_examples),
            ('Avg Examples/Puzzle', stats.avg_examples_per_puzzle)
        ]
        labels, values = zip(*basic_stats)
        ax.bar(labels, values)
        ax.set_title('Dataset Overview')

        # Color usage counts (log scale)
        ax = axes[1, 1]
        counts = stats.color_usage_stats['counts']
        bars = ax.bar(colors, counts, color=[ARC_COLOR_MAP.colors[i] for i in colors])
        ax.set_title('Color Usage Counts (Log Scale)')
        ax.set_xlabel('Color ID')
        ax.set_ylabel('Count (log scale)')
        ax.set_yscale('log')
        ax.set_xticks(colors)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {save_path}")

        return fig

    def create_puzzle_grid_comparison(self, puzzles: List[ARCPuzzle],
                                      max_puzzles: int = 9,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a grid comparison of multiple puzzles.

        Args:
            puzzles: List of ARCPuzzle objects to compare
            max_puzzles: Maximum number of puzzles to show
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        puzzles_to_show = puzzles[:max_puzzles]
        num_puzzles = len(puzzles_to_show)

        # Determine grid layout
        cols = min(3, num_puzzles)
        rows = (num_puzzles + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('ARC Puzzles Comparison', fontsize=16, fontweight='bold')

        for i, puzzle in enumerate(puzzles_to_show):
            row = i // cols
            col = i % cols

            if puzzle.examples:
                # Show first example of each puzzle
                example = puzzle.examples[0]

                # Create subplot for this puzzle showing input and output side by side
                ax = axes[row, col]

                # Combine input and output grids for display
                input_grid = example.input_grid
                output_grid = example.output_grid

                # Pad grids to same height
                max_height = max(input_grid.shape[0], output_grid.shape[0])
                input_padded = np.pad(input_grid, ((0, max_height - input_grid.shape[0]), (0, 0)),
                                      mode='constant', constant_values=0)
                output_padded = np.pad(output_grid, ((0, max_height - output_grid.shape[0]), (0, 0)),
                                       mode='constant', constant_values=0)

                # Combine with separator
                separator = np.full((max_height, 1), -1)  # Use -1 as separator, will be white
                combined = np.concatenate([input_padded, separator, output_padded], axis=1)

                # Plot
                im = ax.imshow(combined, cmap=ARC_COLOR_MAP, vmin=0, vmax=9)
                ax.set_title(f"{puzzle.puzzle_id}\n({len(puzzle.examples)} examples)",
                             fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

                # Add input/output labels
                ax.text(input_grid.shape[1] / 2, -0.5, 'Input', ha='center', va='top', fontsize=8)
                ax.text(input_grid.shape[1] + 1 + output_grid.shape[1] / 2, -0.5, 'Output',
                        ha='center', va='top', fontsize=8)

        # Hide unused subplots
        for i in range(num_puzzles, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Puzzle grid comparison saved to {save_path}")

        return fig


class ARCDatasetValidator:
    """
    Utility class for validating ARC datasets.

    Provides methods to check data integrity, validate formats,
    and ensure consistency across the dataset.
    """

    def __init__(self, dataset_loader: ARCDatasetLoader):
        """
        Initialize the validator.

        Args:
            dataset_loader: Initialized ARCDatasetLoader instance
        """
        self.loader = dataset_loader

    def validate_split(self, split: str, subset: str = "all") -> Dict[str, Any]:
        """
        Validate a specific dataset split.

        Args:
            split: Dataset split to validate
            subset: Dataset subset to validate

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating dataset split: {split}/{subset}")

        validation_results = {
            "split": split,
            "subset": subset,
            "errors": [],
            "warnings": [],
            "stats": {}
        }

        try:
            # Load data
            data = self.loader.load_split_data(split, subset)
            metadata = self.loader.load_split_metadata(split)

            # Validate array shapes and types
            self._validate_arrays(data, validation_results)

            # Validate metadata consistency
            self._validate_metadata_consistency(data, metadata, validation_results)

            # Validate puzzle structure
            self._validate_puzzle_structure(data, validation_results)

            # Validate content ranges
            self._validate_content_ranges(data, validation_results)

            # Compute validation statistics
            validation_results["stats"] = self._compute_validation_stats(data)

        except Exception as e:
            validation_results["errors"].append(f"Failed to load data: {str(e)}")

        # Log results
        if validation_results["errors"]:
            logger.error(f"Validation errors found: {len(validation_results['errors'])}")
            for error in validation_results["errors"]:
                logger.error(f"  - {error}")

        if validation_results["warnings"]:
            logger.warning(f"Validation warnings: {len(validation_results['warnings'])}")
            for warning in validation_results["warnings"]:
                logger.warning(f"  - {warning}")

        if not validation_results["errors"] and not validation_results["warnings"]:
            logger.info("Dataset validation passed successfully")

        return validation_results

    def _validate_arrays(self, data: Dict[str, np.ndarray], results: Dict[str, Any]) -> None:
        """Validate array shapes and data types."""
        required_arrays = ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]

        for array_name in required_arrays:
            if array_name not in data:
                results["errors"].append(f"Missing required array: {array_name}")
                continue

            array = data[array_name]

            # Check data type
            if not isinstance(array, np.ndarray):
                results["errors"].append(f"{array_name} is not a numpy array")
                continue

            # Check for empty arrays
            if array.size == 0:
                results["errors"].append(f"{array_name} is empty")
                continue

            # Validate specific array properties
            if array_name in ["inputs", "labels"]:
                if array.ndim != 2:
                    results["errors"].append(f"{array_name} should be 2D, got {array.ndim}D")
                if array.shape[1] != ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE:
                    results["errors"].append(
                        f"{array_name} should have {ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE} columns, got {array.shape[1]}")

            elif array_name in ["puzzle_identifiers", "puzzle_indices", "group_indices"]:
                if array.ndim != 1:
                    results["errors"].append(f"{array_name} should be 1D, got {array.ndim}D")

    def _validate_metadata_consistency(self, data: Dict[str, np.ndarray],
                                       metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate metadata consistency with actual data."""
        # Check sequence length
        expected_seq_len = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE
        if metadata.get("seq_len") != expected_seq_len:
            results["errors"].append(f"Metadata seq_len {metadata.get('seq_len')} != expected {expected_seq_len}")

        # Check vocabulary size
        expected_vocab_size = ARC_VOCAB_SIZE
        if metadata.get("vocab_size") != expected_vocab_size:
            results["errors"].append(
                f"Metadata vocab_size {metadata.get('vocab_size')} != expected {expected_vocab_size}")

        # Check group count
        actual_groups = len(data["group_indices"]) - 1
        if metadata.get("total_groups") != actual_groups:
            results["warnings"].append(
                f"Metadata total_groups {metadata.get('total_groups')} != actual {actual_groups}")

    def _validate_puzzle_structure(self, data: Dict[str, np.ndarray], results: Dict[str, Any]) -> None:
        """Validate puzzle structure integrity."""
        # Check that puzzle_indices are monotonically increasing
        puzzle_indices = data["puzzle_indices"]
        if not np.all(puzzle_indices[1:] >= puzzle_indices[:-1]):
            results["errors"].append("puzzle_indices are not monotonically increasing")

        # Check that group_indices are monotonically increasing
        group_indices = data["group_indices"]
        if not np.all(group_indices[1:] >= group_indices[:-1]):
            results["errors"].append("group_indices are not monotonically increasing")

        # Check bounds
        max_puzzle_idx = puzzle_indices[-1]
        num_examples = len(data["inputs"])
        if max_puzzle_idx != num_examples:
            results["errors"].append(f"Last puzzle_indices value {max_puzzle_idx} != num_examples {num_examples}")

        max_group_idx = group_indices[-1]
        num_puzzles = len(data["puzzle_identifiers"])
        if max_group_idx != num_puzzles:
            results["errors"].append(f"Last group_indices value {max_group_idx} != num_puzzles {num_puzzles}")

    def _validate_content_ranges(self, data: Dict[str, np.ndarray], results: Dict[str, Any]) -> None:
        """Validate content value ranges."""
        # Check inputs range
        inputs = data["inputs"]
        if np.any(inputs < 0) or np.any(inputs >= ARC_VOCAB_SIZE):
            results["errors"].append(f"inputs contain values outside valid range [0, {ARC_VOCAB_SIZE - 1}]")

        # Check labels range
        labels = data["labels"]
        if np.any(labels < 0) or np.any(labels >= ARC_VOCAB_SIZE):
            results["errors"].append(f"labels contain values outside valid range [0, {ARC_VOCAB_SIZE - 1}]")

        # Check puzzle identifiers
        puzzle_ids = data["puzzle_identifiers"]
        if np.any(puzzle_ids < 0):
            results["errors"].append("puzzle_identifiers contain negative values")

    def _compute_validation_stats(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute validation statistics."""
        return {
            "num_examples": len(data["inputs"]),
            "num_puzzles": len(data["puzzle_identifiers"]),
            "num_groups": len(data["group_indices"]) - 1,
            "unique_puzzle_ids": len(np.unique(data["puzzle_identifiers"])),
            "input_value_range": [int(np.min(data["inputs"])), int(np.max(data["inputs"]))],
            "label_value_range": [int(np.min(data["labels"])), int(np.max(data["labels"]))]
        }

# ---------------------------------------------------------------------
