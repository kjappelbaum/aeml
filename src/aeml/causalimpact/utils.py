import pandas as pd 
from ..eda.statistics import check_granger_causality

__all__ = ("run_causal_impact_analysis")

def get_timestep_tuples(df, times, i):
    times = sorted(times, key=lambda d: d["start"])
    day = times[i]["start"].day
    if i == 0:
        s_0 = df.index[1]
    else:
        s_0 = df[df.index.day == day].index[0]

    end = df[df.index.day == day].index[-1]

    s_1 = df.index[df.index.get_loc(pd.to_datetime(times[i]["start"]), method="nearest")]
    e_0 = df.index[df.index.get_loc(pd.to_datetime(times[i]["end"]), method="nearest")]
    e_1 = end

    pre_intervention_period = [s_0, s_1]
    intervention_period = [e_0, e_1]

    return pre_intervention_period, intervention_period


def get_causalimpact_splits(x, y, day, times, df):
    # if day>0:
    #     a0, b0 = get_timestep_tuples(df, times, day-1)
    #     a, b = get_timestep_tuples(df, times, day)
    #     a[0] = b0[0]
    # else:

    a, b = get_timestep_tuples(df, times, day)
    a[0] = df.index[1]
    x_way_before, x_way_after = x.split_before(b[1])
    y_way_before, y_way_after = y.split_before(pd.Timestamp(b[1]))

    _, x_before_ = x_way_before.split_before(a[0])
    _, y_before_ = y_way_before.split_before(a[0])

    x_before, x_after = x_before_.split_before(a[1])
    y_before, y_after = y_before_.split_before(a[1])

    x_during, x_test = x_after.split_before(b[0])
    y_during, y_test = y_after.split_before(b[0])

    return (
        (x_before, y_before),
        (x_during, y_during),
        (x_test, y_test),
        (x_way_after, y_way_after),
    )







def _select_unrelated_x(df, x_columns, intervention_column, p_value_threshold, lag=10):
    unrelated_x = []

    for x_column in x_columns:
        if x_column != intervention_column:
            granger_result = check_granger_causality(
                df[x_column], df[intervention_column], lag
            )
            if granger_result["min_p_value"] > p_value_threshold:
                unrelated_x.append(x_column)

    return unrelated_x
