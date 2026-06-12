"""THERA arbitrary-scale super-resolution training package.

Houses the pure ``tf.data`` arbitrary-scale data pipeline (port of the THERA
reference ``data.py`` ``ArbitraryScaleWrapper``) and, from step 11 onward, the
THERA trainer.
"""

from .data import build_arbitrary_scale_dataset

__all__ = ["build_arbitrary_scale_dataset"]
