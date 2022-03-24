# -*- coding: utf-8 -*-
from curses.ascii import TAB
import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy

import click
import joblib
import numpy as np
import pandas as pd
from darts import TimeSeries


from aeml.utils import choose_index

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


def dump_pickle(object, filename):
    with open(filename, "wb") as handle:
        pickle.dump(object, handle)


MEAS_COLUMNS = [
    "TI-19",
    #      "FI-16",
    #     "TI-33",
    #     "FI-2",
    #     "FI-151",
    #     "TI-8",
    #     "FI-241",
    #  "valve-position-12",  # dry-bed
    #     "FI-38",  # strippera
    #     "PI-28",  # stripper
    #     "TI-28",  # stripper
    #      "FI-20",
    #     "FI-30",
    "TI-3",
    "FI-19",
    #     "FI-211",
    "FI-11",
    #     "TI-30",
    #     "PI-30",
    "TI-1213",
    #     "TI-4",
    #    "FI-23",
    #    "FI-20",
    #   "FI-20/FI-23",
    #    "TI-22",
    #    "delta_t",
    "TI-35",
    #     "delta_t_2"
]

# First, we train a model on *all* data
# Then, we do a partial-denpendency plot approach and change one variable and see how the model predictions change

# load the trained model

FEAT_NUM_MAPPING = dict(zip(MEAS_COLUMNS, [str(i) for i in range(len(MEAS_COLUMNS))]))
UPDATE_MAPPING = {
    "amp": {
        "scaler": joblib.load(
            "20220312_y_transformer"
        ),
        "model": joblib.load("20220312_model_all_data_0"),
        "name": ["2-Amino-2-methylpropanol C4H11NO"],
    },
    "pz": {
        "scaler": joblib.load(
            "20220312_y_transformer"
        ),
        "model": joblib.load("20220312_model_all_data_1"),
        "name": [ "Piperazine C4H10N2"],
    },
}
SCALER = joblib.load(
    "20220312_x_transformer"
)

TARGETS_clean = ['2-Amino-2-methylpropanol C4H11NO', 'Piperazine C4H10N2'] 

# making the input one item longer as "safety margin"
def calculate_initialization_percentage(timeseries_length: int, input_sequence_length: int = 61):
    fraction_of_input = input_sequence_length / timeseries_length
    res = max([fraction_of_input, 0.1])
    return res


def run_update_historical_forecast(df, x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(TimeSeries.from_dataframe(df, value_cols=TARGETS_clean))

    predictions = model_dict["model"].historical_forecasts(
        past_covariates=x,
        series=y[model_dict['name']],
        start=calculate_initialization_percentage(len(y)),
        forecast_horizon=1,
        retrain=False
    )
    return predictions


def run_update_forecast(df, x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(TimeSeries.from_dataframe(df, value_cols=TARGETS_clean))

    fraction = calculate_initialization_percentage(len(y))
    ts = choose_index(x, fraction)

    y_past, _ = y.split_before(ts)

    predictions = model_dict["model"].forecast(
        past_covariates=x,
        series=y_past[model_dict['name']],
        n=len(x) - len(y_past),
    )
    return predictions


def run_targets(df, feature_levels, target: str = "auine", forecast: bool = False):

    df = deepcopy(df)

    for k, v in feature_levels.items():
        if k == "valve-position-12":
            df[k] = v
        else:
            df[k] = df[k] + v / 100 * np.abs(df[k])

    df["delta_t"] = df["TI-35"] - df["TI-4"]
    df["delta_t_2"] = df["TI-22"] - df["TI-19"]
    df["FI-20/FI-23"] = df["FI-20"] / df["FI-23"]

    X_ = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)
    X_ = SCALER.transform(X_)

    if forecast:
        res = run_update_forecast(df, X_, target)
    else:
        res = run_update_historical_forecast(df, X_, target)
    return res


def run_grid(
    df,
    feature_a: str = "TI-19",
    feature_b: str = "FI-19",
    lower: float = -20,
    upper: float = 20,
    num_points: int = 21,
    objective: str = "amine",
    forecast: bool = False,
):
    grid = np.linspace(lower, upper, num_points)
    results_double_new = defaultdict(dict)

    if feature_a == "valve-position-12":
        grid_a = [0, 1]
    else:
        grid_a = grid
    if feature_b == "valve-position-12":
        grid_b = [0, 1]
    else:
        grid_b = grid

    for point_a in grid_a:
        for point_b in grid_b:
            print(f"Running point {feature_a}: {point_a} {feature_b}: {point_b}")
            results_double_new[point_a][point_b] = run_targets(
                df, {feature_a: point_a, feature_b: point_b}, objective, forecast
            )
    return results_double_new


@click.command("cli")
@click.argument("feature_a", type=str, default="TI-19")
@click.argument("feature_b", type=str, default="FI-19")
@click.argument("objective", type=str, default="amine")
@click.option("--forecast", is_flag=True)
def main(feature_a, feature_b, objective, forecast):
    lower = -20
    upper = 20
    num_points = 21

    df = pd.read_pickle("20220210_smooth_window_16.pkl")

    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    print("starting run")
    results = run_grid(df, feature_a, feature_b, lower, upper, num_points, objective, forecast)
    dump_pickle(
        results,
        f"{TIMESTR}_{feature_a}_{feature_b}_{lower}_{upper}_{num_points}_{objective}_{str(forecast)}".replace(
            "/", "*"
        ),
    )


if __name__ == "__main__":
    main()
