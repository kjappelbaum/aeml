from darts.models.forecasting.gradient_boosted_model import LightGBMModel
import wandb
from darts.models import TCNModel
import pandas as pd
from darts.metrics import mape, mae
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from copy import deepcopy
import numpy as np
import click
from functools import partial
from aeml.models.utils import split_data
import torch
from loguru import logger

MEAS_COLUMNS = [
    "TI-19",
    "TI-3",
    "FI-19",
    "FI-11",
    "TI-1213",
    "TI-35",
]

TARGETS_clean = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]

sweep_config = {
    "metric": {"goal": "minimize", "name": "mae_valid"},
    "method": "bayes",
    "parameters": {
        "lags": {"min": 0, "max": 200, "distribution": "int_uniform"},
        "lag_1": {"max": 0, "min": -200, "distribution": "int_uniform"},
        "lag_2": {"max": 0, "min": -200, "distribution": "int_uniform"},
        "lag_3": {"max": 0, "min": -200, "distribution": "int_uniform"},
        "lag_4": {"max": 0, "min": -200, "distribution": "int_uniform"},
        "lag_5": {"max": 0, "min": -200, "distribution": "int_uniform"},
        "lag_6": {"max": 0, "min": -200, "distribution": "int_uniform"},
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


def inner_train_test(x, y, output_seq_length, target):
    run = wandb.init()
    train, valid, _ = get_data(x, y, target, targets_clean=TARGETS_clean)

    logger.info("initialize model")
    model = LightGBMModel(
        lags=run.config.lags,
        lags_past_covariates=[
            run.config.lag_1,
            run.config.lag_2,
            run.config.lag_3,
            run.config.lag_4,
            run.config.lag_5,
            run.config.lag_6,
        ],
        n_estimators=run.config.n_estimators,
        bagging_freq=run.config.bagging_freq,
        bagging_fraction=run.config.bagging_fraction,
        num_leaves=run.config.num_leaves,
        extra_trees=run.config.extra_trees,
        max_depth=run.config.max_depth,
        output_chunk_length=output_seq_length,
        objective="quantile",
        alpha=0.5,
    )

    logger.info("fit")

    model.fit(series=train[1], past_covariates=train[0], verbose=False)

    logger.info("historical forecast train set")
    backtest_train = model.historical_forecasts(
        train[1],
        past_covariates=train[0],
        start=0.3,
        forecast_horizon=output_seq_length,
        stride=1,
        retrain=False,
        verbose=False,
    )

    logger.info("historical forecast valid")
    backtest_valid = model.historical_forecasts(
        valid[1],
        past_covariates=valid[0],
        start=0.3,
        forecast_horizon=output_seq_length,
        stride=1,
        retrain=False,
        verbose=False,
    )

    logger.info("getting scores")
    mape_valid = mape(valid[1][TARGETS_clean[0]], backtest_valid["0"])
    mape_train = mape(train[1][TARGETS_clean[0]], backtest_train["0"])

    mae_valid = mae(valid[1][TARGETS_clean[0]], backtest_valid["0"])
    mae_train = mae(train[1][TARGETS_clean[0]], backtest_train["0"])

    wandb.log({"mape_valid": mape_valid})
    wandb.log({"mape_train": mape_train})

    logger.info(f"MAPE valid {mape_valid}")

    wandb.log({"mae_valid": mae_valid})
    wandb.log({"mae_train": mae_train})
    torch.cuda.empty_cache()


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("output_seq_length", type=click.INT, default=5)
@click.argument("target", type=click.INT, default=0)
def train_test(file, output_seq_length, target):
    print("get data")
    x, y, _, _ = load_data(file)

    optimizer_func = partial(
        inner_train_test, output_seq_length=output_seq_length, x=x, y=y, target=target
    )
    wandb.agent("26gi3tth", function=optimizer_func, project="aeml")


if __name__ == "__main__":
    train_test()
