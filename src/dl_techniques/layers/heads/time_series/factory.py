"""Factory function for the time-series heads.

``create_time_series_head`` builds one head from a task type. The classes live
in their own modules and the table from task type to class in ``dispatch``.
"""

import keras
from typing import Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .task_types import TimeSeriesTaskType
from .dispatch import get_head_class


def create_time_series_head(
        task_type: Union[TimeSeriesTaskType, str],
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Build the time-series head that serves a task type.

    The task type picks the class through :func:`get_head_class`. Every keyword
    goes to that class's constructor unchanged, so each task needs its own
    head's required arguments:

    .. code-block:: text

        Task type                     Head                            Required
        ----------------------------  ------------------------------  ---------------------
        QUANTILE_FORECAST             QuantileHead                    num_quantiles,
                                                                      output_length
        QUANTILE_SEQUENCE             QuantileSequenceHead            num_quantiles
        GAUSSIAN_LIKELIHOOD           GaussianLikelihoodHead          (none, units=1)
        NEGATIVE_BINOMIAL_LIKELIHOOD  NegativeBinomialLikelihoodHead  (none, units=1)
        TIMESTRAL_QUANTILE            TimestralQuantileHead           d_model, patch_size,
                                                                      quantile_levels,
                                                                      width_factor

    The forecasting models build their heads through this function, not through
    ``create_head``.

    :param task_type: Task the head is for, as an enum member or its string
        value.
    :type task_type: Union[TimeSeriesTaskType, str]
    :param kwargs: Forwarded to the chosen head class.
    :return: Configured time-series head.
    :rtype: keras.layers.Layer
    :raises ValueError: If the task type is unknown, a keyword is one the head
        does not declare, or an argument is invalid.
    :raises TypeError: If a required argument of the head is missing.
    """
    head_class = get_head_class(task_type)
    return head_class(**kwargs)
