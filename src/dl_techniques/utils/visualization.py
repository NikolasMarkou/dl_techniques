"""
Visualization utilities for comparing confusion matrices across multiple models.

This module provides functions for generating and plotting confusion matrix
comparisons for multiple models in a single figure.
"""
import io
import keras
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------

def collage(
        images_batch):
    """
    Create a collage of image from a batch

    :param images_batch:
    :return:
    """
    shape = images_batch.shape
    no_images = shape[0]
    images = []
    result = None
    width = np.ceil(np.sqrt(no_images))

    for i in range(no_images):
        images.append(images_batch[i, :, :, :])

        if len(images) % width == 0:
            if result is None:
                result = np.hstack(images)
            else:
                tmp = np.hstack(images)
                result = np.vstack([result, tmp])
            images.clear()
    return result

# ---------------------------------------------------------------------

def draw_figure_to_buffer(
        fig: matplotlib.figure.Figure,
        dpi: int) -> np.ndarray:
    """
    draw figure into numpy buffer

    :param fig: figure to draw
    :param dpi: dots per inch to draw
    :return: np.ndarray buffer
    """
    # open binary buffer
    with io.BytesIO() as io_buf:
        # save figure and reset to start
        fig.savefig(io_buf, format="raw", dpi=dpi)
        io_buf.seek(0)
        # load buffer into numpy and reshape
        img_arr = \
            np.reshape(
                a=np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    return img_arr

# ---------------------------------------------------------------------

def plot_confusion_matrices(
        models: Dict[str, keras.Model],
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        output_path: Path,
        figsize: tuple = (15, 5),
        cmap: str = 'Blues'
) -> None:
    """Plot confusion matrices for multiple models side by side.

    Args:
        models: Dictionary mapping model names to keras Model instances
        x_test: Test data features
        y_test: Test data labels (one-hot encoded)
        class_names: List of class names for axis labels
        output_path: Path to save the comparison figure
        figsize: Figure size (width, height)
        cmap: Color map for confusion matrices
    """
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=figsize)

    # Convert one-hot encoded y_test to class indices
    y_true = np.argmax(y_test, axis=1)

    # Ensure axes is always a list even with one model
    if num_models == 1:
        axes = [axes]

    vmax = 0  # For consistent color scaling across matrices
    matrices = []

    # First pass: compute all matrices and find max value for scaling
    for model_name, model in models.items():
        y_pred = np.argmax(model.predict(x_test), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        matrices.append(cm)
        vmax = max(vmax, np.max(cm))

    # Second pass: plot matrices with consistent scaling
    for ax, (model_name, cm) in zip(axes, zip(models.keys(), matrices)):
        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=vmax,
            cbar=False  # One colorbar for all plots
        )

        # Customize appearance
        ax.set_title(f'{model_name} Model')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if ax is axes[0] else '')  # Only leftmost ylabel

        # Rotate tick labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax)

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------------------------------------------------
