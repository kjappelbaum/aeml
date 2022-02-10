# -*- coding: utf-8 -*-
"""In some time series there is a trend component that does not interest us, e.g., because we have domain knowledge that this trend is due to another phenomenon like instrument drift. In this case, we might want to remove the trend component for furhter modeling.
The same is the case for the variance. If the variance increases over time, one might want to remove this effect using a Box-Cox transformation [1]

References:
[1] https://otexts.com/fpp2/transformations.html#mathematical-transformations
"""

from copy import deepcopy
from typing import Union

import pandas as pd
import statsmodels

__all__ = ["detrend_stochastic", "detrend_linear_deterministc"]


def detrend_stochastic(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Detrends time series data using the difference method y_t - y_{t-1}.
    This is useful to remove stochastic trends (random walk with trend).

    Args:
        data (Union[pd.Series, pd.DataFrame]): Time series data to detrend

    Returns:
        Union[pd.Series, pd.DataFrame]: Differenced data
    """
    new_data = data.diff()
    new_data = new_data.iloc[1:]
    return new_data


def _detrend_series(series):
    clean_data = statsmodels.tsa.tsatools.detrend(series.values, order=1, axis=0)
    return pd.Series(clean_data, index=series.index, name=series.name)


def detrend_linear_deterministc(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Removes a deterministic linear trend from a series.
    Note that we assume that the data is sampled on a regular grid and
    we estimate the trend as

    np.arange(
        len(series) * (series.iloc[end] - series.iloc[start]) / (end - start)
    )

    Args:
        data (Union[pd.Series, pd.DataFrame]): Data to detrend. In case of
            dataframes we detrend every column separately.

    Returns:
        Union[pd.Series, pd.DataFrame]: Detrended data
    """
    data_ = deepcopy(data)

    if isinstance(data_, pd.DataFrame):
        for column in data_:
            data_[column] = _detrend_series(data_[column])
        return data_
    else:
        return _detrend_series(data)
