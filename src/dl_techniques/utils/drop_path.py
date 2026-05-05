"""Drop-path (stochastic depth) rate schedules."""

from typing import List


def linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Return linearly spaced drop-path rates from 0 to ``max_rate``.

    :param num_blocks: Total number of blocks.
    :param max_rate: Maximum (last-block) drop probability.
    :return: List of per-block drop-path rates of length ``num_blocks``.
    """
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]
