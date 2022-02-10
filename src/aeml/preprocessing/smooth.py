# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["exponential_window_smoothing", "z_score_filter"]


def exponential_window_smoothing(
    data: Union[pd.Series, pd.DataFrame], window_size: int, aggregation: str = "mean"
) -> Union[pd.Series, pd.DataFrame]:
    """

    Args:
        data (Union[pd.Series, pd.DataFrame]): Data to smoothen
        window_size (int): size for the exponential window
        aggregation (str, optional): Aggregation function. Defaults to "mean".

    Returns:
        Union[pd.Series, pd.DataFrame]: Smoothned data
    """
    if aggregation == "median":
        new_data = data.ewm(span=window_size).median()
    else:
        new_data = data.ewm(span=window_size).mean()
    return new_data


def _despike_series(series: pd.Series, threshold, window) -> pd.Series:
    indices = series.index
    series_ = deepcopy(series.values)
    mask = np.abs(stats.zscore(series_)) > threshold
    indices_masked = np.where(mask)[0]
    for index in indices_masked:
        series_[index] = np.median(series_[int(index - window) : index - 1])
    return pd.Series(series_, index=indices, name=series.name)


def z_score_filter(
    data: Union[pd.Series, pd.DataFrame], threshold: float = 2, window: int = 10
) -> Union[pd.Series, pd.DataFrame]:
    """Replaces spikes (values > threshold * z_score) with the median
    of the window values before.

    Args:
        data (Union[pd.Series, pd.DataFrame]): Series to despike
        threshold (float, optional): Threshold on the z-score. Defaults to 2.
        window (int, option): Window that is used for the median with which
            the spike value is replaced. This mean only looks back.

    Returns:
        Union[pd.Series, pd.DataFrame]: Despiked series
    """

    if isinstance(data, pd.Series):
        return _despike_series(data, threshold, window)

    else:
        data_ = deepcopy(data)
        for col in data_:
            data_[col] = _despike_series(data_[col], threshold, window)
        return data
