"""Task type to head class."""

from typing import Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import lookup_head
from .task_types import TimeSeriesTaskType
from .quantile_head_fixed_io import QuantileHead
from .quantile_head_variable_io import QuantileSequenceHead
from .likelihood_heads import (
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
)
from .timestral_quantile_head import TimestralQuantileHead


#: The one dispatch table of the time-series heads, read by
#: :func:`get_head_class` and therefore by ``create_time_series_head``.
TIME_SERIES_HEAD_DISPATCH: Dict[TimeSeriesTaskType, type] = {
    TimeSeriesTaskType.QUANTILE_FORECAST: QuantileHead,
    TimeSeriesTaskType.QUANTILE_SEQUENCE: QuantileSequenceHead,
    TimeSeriesTaskType.GAUSSIAN_LIKELIHOOD: GaussianLikelihoodHead,
    TimeSeriesTaskType.NEGATIVE_BINOMIAL_LIKELIHOOD: NegativeBinomialLikelihoodHead,
    TimeSeriesTaskType.TIMESTRAL_QUANTILE: TimestralQuantileHead,
}


def get_head_class(task_type: Union[TimeSeriesTaskType, str]) -> type:
    """
    Return the head class for a time-series task type, or raise.

    The class comes from :data:`TIME_SERIES_HEAD_DISPATCH`. To add a task type,
    implement its head and add both the enum member and the table row. There
    is no silent fallback, matching the other three domains.

    :param task_type: The task type, as a member or its string value.
    :type task_type: Union[TimeSeriesTaskType, str]
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If the string names no task type, or the value is
        neither a member nor a string.
    """
    _, head_class = lookup_head(
        task_type, TIME_SERIES_HEAD_DISPATCH, TimeSeriesTaskType, 'time series')
    return head_class
