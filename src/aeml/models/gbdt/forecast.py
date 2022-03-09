import pandas as pd 
import numpy as np 
from functools import partial 

def summarize_results(results):
    values = []

    for df in results:
        values.append(df.pd_dataframe().values)

    df = df.pd_dataframe()
    columns = df.columns

    return (
        pd.DataFrame(np.mean(values, axis=0), columns=columns, index=df.index),
        pd.DataFrame(np.std(values, axis=0), columns=columns, index=df.index),
    )


def _run_backtest(rep, model, x_test, y_test, start=0.3, stride=1, horizon=4):
    backtest = model.historical_forecasts(
        y_test,
        past_covariates=x_test,
        start=start,
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        verbose=False,
    )
    return backtest


def parallelized_inference(model, x, y, repeats=100, start=0.3, stride=1, horizon=6):
    results = []

    backtest_partial = partial(
        _run_backtest,
        model=model,
        x_test=x,
        y_test=y,
        start=start,
        stride=stride,
        horizon=horizon,
    )

    return results

def _run_forcast(_,model, x_full, y_past, future_len, enable_mc_dropout=True): 

    return model.predict(future_len, series=y_past, past_covariates=x_full, enable_mc_dropout=enable_mc_dropout)

def forecast(model, x_full, y_past, future_len, repeats=100, enable_mc_dropout=True):
    results = []

    backtest_partial = partial(
        _run_forcast,
        model=model,
        x_full=x_full,
        y_past=y_past,
        future_len=future_len,
        enable_mc_dropout=enable_mc_dropout
    )

    for res in map(backtest_partial, range(repeats)):
        results.append(res)

    return results
