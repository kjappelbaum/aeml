# -*- coding: utf-8 -*-
"""Often, data is not sampled on a regular grid.
This module provides to regularize such data"""
from typing import Union

import pandas as pd

__all__ = ["resample_regular"]


def _interpolate(resampled, interpolation):

    if isinstance(interpolation, int):
        result = resampled.interpolate(method="spline", order=interpolation)
    else:
        result = resampled.interpolate(method="linear")
    return result


def resample_regular(
    df: pd.DataFrame,
    interval: str = "10min",
    interpolation: Union[str, int] = "linear",
    start_time=None,
) -> pd.DataFrame:
    """Resamples the dataframe at a desired interval.

    Args:
        df (pd.DataFrame): input dataframne
        interval (str, optional): Resampling intervall. Defaults to "10min".
        interpolation (Union[str, int], optional): Interpolation method.
            If you provide an integer, spline interpolation of that order will be used.
            Defaults to "linear".

    Returns:
        pd.DataFrame: Output data.
    """
    oidx = df.index
    start = oidx.min() if start_time is None else start_time
    nidx = pd.date_range(start, oidx.max(), freq=interval)
    res = df.reindex(oidx.union(nidx))

    result = _interpolate(res, interpolation).reindex(nidx)
    return result
