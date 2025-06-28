import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------

def create_dummy_coco_dataset(
    num_samples: int,
    img_size: int,
    num_classes: int = 80,
    max_boxes: int = 20,
    min_boxes: int = 1
) -> tf.data.Dataset:
    """
    Create a dummy COCO-style dataset for object detection training.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.

    Returns:
        TensorFlow dataset with (image, labels) pairs.
        Labels format: (class_id, x1, y1, x2, y2) in absolute coordinates.
    """

    def generator():
        for _ in range(num_samples):
            # Generate dummy image
            img = np.random.rand(img_size, img_size, 3).astype(np.float32)

            # Generate random number of boxes
            num_boxes = np.random.randint(min_boxes, max_boxes + 1)

            # Initialize labels array
            labels = np.zeros((max_boxes, 5), dtype=np.float32)

            for i in range(num_boxes):
                # Random class
                cls_id = np.random.randint(0, num_classes)

                # Random box coordinates (ensure valid boxes)
                x1 = np.random.uniform(0, img_size * 0.8)
                y1 = np.random.uniform(0, img_size * 0.8)
                x2 = np.random.uniform(x1 + 20, min(x1 + 200, img_size))
                y2 = np.random.uniform(y1 + 20, min(y1 + 200, img_size))

                labels[i] = [cls_id, x1, y1, x2, y2]

            yield img, labels

    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(max_boxes, 5), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset

# ---------------------------------------------------------------------
