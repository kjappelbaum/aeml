# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

__all__ = [
    "check_stationarity",
    "check_granger_causality",
    "computer_granger_causality_matrix",
]


def check_stationarity(
    series: pd.Series, threshold: float = 0.05, regression="c"
) -> dict:
    """Performs the Augmented-Dickey fuller and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
    for stationarity.

    Args:
        series (pd.Series): Time series data
        threshold (float, optional): p-value thresholds for the statistical tests.
            Defaults to 0.05.
        regression (str, optional): If regression="c" then the tests check for stationarity around a constant.
            For "ct" the test check for stationarity around a trend.
            Defaults to "c".

    Returns:
        dict: Results dictionary with key "stationary" that has a bool as value
    """

    assert regression in ["c", "ct"]

    adf_results = adfuller(series, regression=regression)
    kpss_results = kpss(series, regression=regression, nlags="auto")

    # null hypothesis for ADF is non-sationarity for KPSS null hypothesis is stationarity
    conclusion = (kpss_results[1] > threshold) & (adf_results[1] < threshold)
    results = {
        "adf": {
            "statistic": adf_results[0],
            "p_value": adf_results[1],
            "stationary": adf_results[1] < threshold,
        },
        "kpss": {
            "statistic": kpss_results[0],
            "p_value": kpss_results[1],
            "stationary": kpss_results[1] > threshold,
        },
        "stationary": conclusion,
    }

    return results


def check_granger_causality(
    x: pd.Series, y: pd.Series, max_lag: int = 20, add_constant: bool = True
) -> dict:
    """Check if series x is Granger causal for series y
    We reject the null hypothesis that x does *not* Granger cause y
    if the pvalues are below a desired size of the test.

    Args:
        x (pd.Series): Time series.
        y (pd.Series): Time series.
        max_lag (int, optional): Maximum lag to use for the causality checks.
            Defaults to 20.
        add_constant (bool, optional): [description]. Defaults to True.

    Returns:
        dict: results dictionary
    """
    results = {}
    test_result = grangercausalitytests(
        np.hstack([x.values.reshape(-1, 1), y.values.reshape(-1, 1)]),
        maxlag=max_lag,
        addconst=add_constant,
        verbose=False,
    )
    results["detail"] = test_result
    p_values = []

    for _, v in test_result.items():
        p_values.append(v[0]["ssr_chi2test"][1])

    results["min_p_value"] = min(p_values)
    results["lag_w_min_p_value"] = np.argmin(p_values)
    return results


def computer_granger_causality_matrix(
    df: pd.DataFrame, xs: List[str], ys: List[str]
) -> pd.DataFrame:
    results_matrix = defaultdict(list)

    for x in xs:
        for y in ys:
            results_matrix[x].append(
                check_granger_causality(df[x], df[y])["min_p_value"]
            )

    return pd.DataFrame.from_dict(results_matrix, orient="index", columns=ys)
