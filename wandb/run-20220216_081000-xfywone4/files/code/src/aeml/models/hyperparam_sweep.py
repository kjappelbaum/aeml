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
from .utils import split_data, get_data

log = logging.getLogger(__name__)

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
        "num_layers": {"values": [2, 4, 8, 16]},
        "num_filters": {"values": [8, 16, 32, 64]},
        "weight_norm": {"values": [True, False]},
        "kernel_size": {"values": [2, 3, 4, 5]},
        "dropout": {
            "min": 0.1,
            "max": 0.9,
        },
        "batch_size": {"values": [32, 64, 128]},
        "num_outputs": {"values": [0, 1]},
        "n_epochs": {"values": [100, 200, 300, 400]},
        "input_chunk_length": {"values": [31, 40, 60, 80]},
        "lr": {"min": -5, "max": -1, "distribution": "log_uniform"},
    },
}

sweep_id = wandb.sweep(sweep_config, project="aeml")


def load_data(datafile="../../../paper/20210624_df_cleaned.pkl"):
    df = pd.read_pickle(datafile)
    Y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean).astype(np.float32)
    X = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS).astype(np.float32)

    transformer = Scaler()
    X = transformer.fit_transform(X)

    y_transformer = Scaler()
    Y = y_transformer.fit_transform(Y)

    return X, Y, transformer, y_transformer





def inner_train_test(x, y, output_seq_length):
    run = wandb.init()
    train, valid, _ = get_data(x, y, run.config.num_outputs, targets_clean=TARGETS_clean)

    log.info("initialize model")
    model_cov = TCNModel(
        input_chunk_length=run.config.input_chunk_length,
        output_chunk_length=output_seq_length,
        num_layers=run.config.num_layers,
        num_filters=run.config.num_filters,
        kernel_size=run.config.kernel_size,
        dropout=run.config.dropout,
        weight_norm=run.config.weight_norm,
        batch_size=run.config.batch_size,
        n_epochs=run.config.n_epochs,
        log_tensorboard=False,
        optimizer_kwargs={"lr": run.config.lr},  # run.config.lr},
    )

    log.info("fit")

    model_cov.fit(series=train[1], past_covariates=train[0], verbose=False)

    log.info("historical forecast train set")
    backtest_train = model_cov.historical_forecasts(
        train[1],
        past_covariates=train[0],
        start=0.1,
        forecast_horizon=30,
        stride=1,
        retrain=False,
        verbose=False,
    )

    log.info("historical forecast valid")
    backtest_valid = model_cov.historical_forecasts(
        valid[1],
        past_covariates=valid[0],
        start=0.1,
        forecast_horizon=30,
        stride=1,
        retrain=False,
        verbose=False,
    )

    log.info("getting scores")
    mape_valid = mape(valid[1][TARGETS_clean[0]], backtest_valid["0"])
    mape_train = mape(train[1][TARGETS_clean[0]], backtest_train["0"])

    mae_valid = mae(valid[1][TARGETS_clean[0]], backtest_valid["0"])
    mae_train = mae(train[1][TARGETS_clean[0]], backtest_train["0"])

    wandb.log({"mape_valid": mape_valid})
    wandb.log({"mape_train": mape_train})

    log.info(f"MAPE valid {mape_valid}")

    wandb.log({"mae_valid": mae_valid})
    wandb.log({"mae_train": mae_train})


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("output_seq_length", type=click.INT, default=5)
def train_test(file, output_seq_length):
    print("get data")
    x, y, _, _ = load_data(file)

    optimizer_func = partial(inner_train_test, output_seq_length=output_seq_length, x=x, y=y)
    wandb.agent(sweep_id, function=optimizer_func, project="aeml")


if __name__ == "__main__":
    train_test()
