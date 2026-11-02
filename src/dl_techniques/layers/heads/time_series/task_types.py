"""Task types for the time-series heads.

This module says what a forecasting head task is. It builds no layer and
imports no head. ``time_series/dispatch.py`` holds the table from task type to
head class and ``time_series/factory.py`` builds the head.

There is no task config dataclass. Like the vision domain, this one takes a
task type plus head keyword arguments, because the five heads share no common
field.
"""

from enum import Enum, unique

# ---------------------------------------------------------------------


@unique
class TimeSeriesTaskType(Enum):
    """
    The five forecasting tasks a head can be asked to serve.

    Each member carries a lowercase string value. That string is the
    serialization form, and :meth:`from_string` parses it back. Every member
    has a head; there are no task names waiting for an implementation.

    .. code-block:: text

        Member                        Head class
        ----------------------------  ------------------------------
        QUANTILE_FORECAST             QuantileHead
        QUANTILE_SEQUENCE             QuantileSequenceHead
        GAUSSIAN_LIKELIHOOD           GaussianLikelihoodHead
        NEGATIVE_BINOMIAL_LIKELIHOOD  NegativeBinomialLikelihoodHead
        TIMESTRAL_QUANTILE            TimestralQuantileHead

    ``time_series/dispatch.py``'s ``TIME_SERIES_HEAD_DISPATCH`` is the arbiter
    if this table and that one disagree.
    """

    QUANTILE_FORECAST = "quantile_forecast"
    QUANTILE_SEQUENCE = "quantile_sequence"
    GAUSSIAN_LIKELIHOOD = "gaussian_likelihood"
    NEGATIVE_BINOMIAL_LIKELIHOOD = "negative_binomial_likelihood"
    TIMESTRAL_QUANTILE = "timestral_quantile"

    @classmethod
    def from_string(cls, task_str: str) -> "TimeSeriesTaskType":
        """
        Parse a task type from its string value.

        The input is lowercased and stripped first, so ``" Quantile_Sequence "``
        resolves to :attr:`QUANTILE_SEQUENCE`.

        :param task_str: A task string value, such as ``"gaussian_likelihood"``.
        :type task_str: str
        :return: The member carrying that value.
        :rtype: TimeSeriesTaskType
        :raises ValueError: If no member carries that value. The message lists
            all five valid strings.
        """
        task_str = task_str.lower().strip()
        for task in cls:
            if task.value == task_str:
                return task

        valid_tasks = [task.value for task in cls]
        raise ValueError(
            f"Invalid task type: '{task_str}'. " f"Valid options are: {valid_tasks}"
        )
