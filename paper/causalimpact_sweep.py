from pyexpat import features
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
import wandb
from darts.models import TCNModel
import pandas as pd
from darts.metrics import mape, mae
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from copy import deepcopy
import numpy as np
import logging
import click
from functools import partial
from aeml.models.utils import split_data, choose_index


from aeml.causalimpact.utils import get_timestep_tuples, get_causalimpact_splits
import pickle
from aeml.causalimpact.utils import _select_unrelated_x
from aeml.models.gbdt.gbmquantile import LightGBMQuantileRegressor
from aeml.models.gbdt.run import run_model
from aeml.models.gbdt.settings import *

from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd
from copy import deepcopy
import time


log = logging.getLogger(__name__)

MEAS_COLUMNS = ["TI-19", "TI-3", "FI-19", "FI-11", "TI-1213", "TI-35", "delta_t"]

to_exclude = {
    0: ["TI-19"],
    1: ["FI-19"],
    2: ["TI-3"],
    3: ["FI-11"],
    4: ["FI-11"],
    5: ["TI-1213", "TI-19"],
    6: [],
}

TARGETS_clean = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]

sweep_config = {
    "metric": {"goal": "minimize", "name": "mae_valid"},
    "method": "bayes",
    "parameters": {
        "lags": {"min": 1, "max": 200, "distribution": "int_uniform"},
        "feature_lag": {"max": -1, "min": -200},
        "n_estimators": {"min": 50, "max": 1000},
        "bagging_freq": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "bagging_fraction": {"min": 0.001, "max": 1.0},
        "num_leaves": {"min": 1, "max": 200, "distribution": "int_uniform"},
        "extra_trees": {"values": [True, False]},
        "max_depth": {"values": [-1, 10, 20, 40, 80, 160, 320]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="aeml")


def get_data(x, y, target, targets_clean):
    targets = targets_clean[target]
    train, valid, test, ts, ts1 = split_data(x, y, targets, 0.5)

    return (train, valid, test)


def load_data(datafile="../../../paper/20210624_df_cleaned.pkl"):
    df = pd.read_pickle(datafile)
    Y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean).astype(np.float32)
    X = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS).astype(np.float32)

    transformer = Scaler()
    X = transformer.fit_transform(X)

    y_transformer = Scaler()
    Y = y_transformer.fit_transform(Y)

    return X, Y, transformer, y_transformer


def select_columns(day):
    feat_to_exclude = to_exclude[day]
    feats = [f for f in MEAS_COLUMNS if f not in feat_to_exclude]
    return feats


with open("step_times.pkl", "rb") as handle:
    times = pickle.load(handle)

DF = pd.read_pickle("20210508_df_for_causalimpact.pkl")


def inner_train_test(x, y, day, target):
    run = wandb.init()
    features = select_columns(day)

    y = y[TARGETS_clean[target]]
    x = x[features]

    x_trains = []
    y_trains = []   

    before, during, after, way_after = get_causalimpact_splits(x, y, day, times, DF)

    # if len(before[0]) > len(way_after[0]):
    x_trains.append(before[0])
    y_trains.append(before[1])
    # else:
    x_trains.append(way_after[0])
    y_trains.append(way_after[1])

    xscaler = Scaler(name="x-scaler")
    yscaler = Scaler(name="y-scaler")

    longer = np.argmax([len(x_trains[0]), len(x_trains[1])])
    shorter = np.argmin([len(x_trains[0]), len(x_trains[1])])
    print(len(x_trains[0]), len(x_trains[1]))
    x_trains[longer] = xscaler.fit_transform(x_trains[longer])
    y_trains[longer] = yscaler.fit_transform(y_trains[longer])

    x_trains[shorter] = xscaler.transform(x_trains[shorter])
    y_trains[shorter] = yscaler.transform(y_trains[shorter])

    steps =len(during[0]) 

    if steps >  len(x_trains[shorter]):
        ts = choose_index(x, 0.3)
        x_before, x_after = x_trains[longer].split_before(ts)
        y_before, y_after = y_trains[longer].split_before(ts)

        y_trains[shorter] = y_before
        y_trains[longer] = y_after

        x_trains[shorter] = x_before
        x_trains[longer] = x_after

    print(steps, len(x_trains[shorter]))
    train = (x_trains[longer], y_trains[longer])
    valid = (x_trains[shorter], y_trains[shorter])

    log.info("initialize model")
    model = LightGBMModel(
        lags=run.config.lags,
        lags_past_covariates=[
            run.config.feature_lag,
        ]
        * len(features),
        n_estimators=run.config.n_estimators,
        bagging_freq=run.config.bagging_freq,
        bagging_fraction=run.config.bagging_fraction,
        num_leaves=run.config.num_leaves,
        extra_trees=run.config.extra_trees,
        max_depth=run.config.max_depth,
        output_chunk_length=steps,
        objective="quantile",
        alpha=0.5,
    )

    log.info("fit")

    model.fit(series=train[1], past_covariates=train[0], verbose=False)

    log.info("historical forecast train set")
    backtest_train = model.historical_forecasts(
        train[1],
        past_covariates=train[0],
        start=0.3,
        forecast_horizon=steps,
        stride=1,
        retrain=False,
        verbose=False,
    )

    log.info("historical forecast valid")
    backtest_valid = model.historical_forecasts(
        valid[1],
        past_covariates=valid[0],
        start=0.5,
        forecast_horizon=steps,
        stride=1,
        retrain=False,
        verbose=False,
    )

    log.info("getting scores")
    # mape_valid = mape(valid[1][TARGETS_clean[0]], backtest_valid["0"])
    # mape_train = mape(train[1][TARGETS_clean[0]], backtest_train["0"])

    mae_valid = mae(valid[1][TARGETS_clean[target]], backtest_valid["0"])
    mae_train = mae(train[1][TARGETS_clean[target]], backtest_train["0"])

    # wandb.log({"mape_valid": mape_valid})
    # wandb.log({"mape_train": mape_train})

    log.info(f"MAE valid {mae_valid}")

    wandb.log({"mae_valid": mae_valid})
    wandb.log({"mae_train": mae_train})


@click.command()
@click.argument("day", type=click.INT, default=1)
@click.argument("target", type=click.INT, default=0)
def train_test(day, target):
    print("get data")
    x, y, _, _ = load_data("20210508_df_for_causalimpact.pkl")

    optimizer_func = partial(inner_train_test, day=day, x=x, y=y, target=target)
    wandb.agent(sweep_id, function=optimizer_func, project="aeml")


if __name__ == "__main__":
    train_test()
