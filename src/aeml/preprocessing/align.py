# -*- coding: utf-8 -*-
"""Sometimes, different kinds of measurements are sampled at different intervals. This module provides utilities to combine such data.
We will always operate on pandas dataframes with datatime indexing
"""
from typing import Union

import pandas as pd

from .resample import resample_regular

__all__ = ["align_two_dfs"]


def align_two_dfs(
    df_a: pd.DataFrame, df_b: pd.DataFrame, interpolation: Union[str, int] = "linear"
) -> pd.DataFrame:
    """Alignes to dataframes with datatimeindex
    Resamples both dataframes on the dataframe with the lowest frequency timestep.
    The first timepoint in the new dataframe will be the later one of the first
    observations of the dataframes.

    https://stackoverflow.com/questions/47148446/pandas-resample-interpolate-is-producing-nans
    https://stackoverflow.com/questions/66967998/pandas-interpolation-giving-odd-results

    Args:
        df_a (pd.DataFrame): Dataframe
        df_b (pd.DataFrame): Dataframe
        interpolation (Union[str, int], optional): Interpolation method.
            If you provide an integer, spline interpolation of that order will be used.
            Defaults to "linear".

    Returns:
        pd.DataFrame: merged dataframe
    """
    assert isinstance(df_a, pd.DataFrame)
    assert isinstance(df_b, pd.DataFrame)

    index_series_a = pd.Series(df_a.index, df_a.index)
    index_series_b = pd.Series(df_b.index, df_b.index)
    timestep_a = min(index_series_a.diff().dropna())
    timestep_b = min(index_series_b.diff().dropna())

    if timestep_a > timestep_b:
        resample_step = timestep_a
    else:
        resample_step = timestep_b

    start_time = max([df_a.index[0], df_b.index[0]])

    resampled_a = resample_regular(
        df_a, resample_step, interpolation, start_time=start_time
    )

    resampled_b = resample_regular(
        df_b, resample_step, interpolation, start_time=start_time
    )

    merged = pd.merge(resampled_a, resampled_b, left_index=True, right_index=True)

    return merged
