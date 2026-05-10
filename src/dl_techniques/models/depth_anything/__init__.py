"""Depth Anything model package — public API."""

from .model import DepthAnything, create_depth_anything
from .components import DPTDecoder
from .teacher_ema import (
    TeacherEMACallback,
    cosine_ema_schedule,
    linear_ema_schedule,
)

__all__ = [
    "DepthAnything",
    "create_depth_anything",
    "DPTDecoder",
    "TeacherEMACallback",
    "cosine_ema_schedule",
    "linear_ema_schedule",
]
