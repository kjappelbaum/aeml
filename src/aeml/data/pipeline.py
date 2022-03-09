from ..preprocessing.smooth import z_score_filter, exponential_window_smoothing
import datetime as dt
import pandas as pd
import numpy as np


def preprocessing_pipeline(df, z_score_threshold: float = 2, window_size: int = 10):
    good_days = [15, 16, 17, 20, 21, 22, 23, 24]
    good_rows = []

    df["delta_t"] = df["TI-35"] - df["TI-4"]
    df["delta_t_2"] = df["TI-22"] - df["TI-19"]
    df["FI-20/FI-23"] = df["FI-20"] / df["FI-23"]

    new_df = z_score_filter(df, z_score_threshold)
    if window_size is not None:
        new_df = exponential_window_smoothing(new_df, window_size)

    for i in range(len(new_df)):
        if new_df.index[i].day in good_days:
            good_rows.append(new_df.iloc[i])

    df_downsampled = pd.DataFrame(good_rows)

    df_downsampled.index = dt.datetime(2010, 1, 1) + pd.TimedeltaIndex(
        np.arange(len(df_downsampled)) * 2, unit="min"
    )

    return df_downsampled
