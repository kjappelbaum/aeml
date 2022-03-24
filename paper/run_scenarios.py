# -*- coding: utf-8 -*-
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
import torch


from aeml.models.forecast import parallelized_inference, forecast, summarize_results
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

HORIZON = 5


def load_model(self, path: str, map_location: str = "cpu"):
    """loads a model from a given file path. The file name should end with '.pth.tar'
    Parameters
    ----------
    path
        Path under which to save the model at its current state. The path should end with '.pth.tar'
    """

    with open(path, "rb") as fin:
        model = torch.load(fin, map_location=map_location)
    return model


model1 = torch.load(
    os.path.join(
        THIS_DIR,
        "20220303-114151_2-Amino-2-methylpropanol C4H11NO_model_reduced_feature_set.pth.tar",
    ),
    map_location=torch.device("cpu"),
)
model1.device = torch.device("cpu")


FEAT_NUM_MAPPING = dict(zip(MEAS_COLUMNS, [str(i) for i in range(len(MEAS_COLUMNS))]))
UPDATE_MAPPING = {
    "amp": {
        "scaler": joblib.load(
            "20220303-114151_2-Amino-2-methylpropanol C4H11NO_transformer__reduced_feature_set"
        ),
        "model": model1,
        "name": ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"],
    },
}
SCALER = joblib.load(
    "20220303-114151_2-Amino-2-methylpropanol C4H11NO_x_scaler_reduced_feature_set"
)


# making the input one item longer as "safety margin"
def calculate_initialization_percentage(timeseries_length: int, input_sequence_length: int = 61):
    fraction_of_input = input_sequence_length / timeseries_length
    res = max([fraction_of_input, 0.1])
    return res


def run_update_historical_forecast(df, x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(TimeSeries.from_dataframe(df, value_cols=model_dict["name"]))

    df = parallelized_inference(
        model_dict["model"],
        x,
        y,
        repeats=1,
        start=calculate_initialization_percentage(len(y)),
        horizon=HORIZON,
        enable_mc_dropout=False,
    )
    means, stds = summarize_results(df)
    return {"means": means, "stds": stds}


def run_update_forecast(df, x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(TimeSeries.from_dataframe(df, value_cols=model_dict["name"]))

    fraction = calculate_initialization_percentage(len(y))
    ts = choose_index(x, fraction)

    y_past, _ = y.split_before(ts)

    df = forecast(
        model_dict["model"],
        x_full=x,
        y_past=y_past,
        future_len=len(x) - len(y_past),
        repeats=1,
        enable_mc_dropout=False,
    )
    means, stds = summarize_results(df)
    return {"means": means, "stds": stds}


def run_targets(df, feature_levels, target: str = "amine", forecast: bool = False):

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
    objectives: str = "amine",
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
                df, {feature_a: point_a, feature_b: point_b}, objectives, forecast
            )
    return results_double_new


@click.command("cli")
@click.argument("feature_a", type=str, default="TI-19")
@click.argument("feature_b", type=str, default="FI-19")
@click.argument("objectives", type=str, default="amine")
@click.option("--forecast", is_flag=True)
def main(feature_a, feature_b, objectives, forecast):
    lower = -20
    upper = 20
    num_points = 21

    df = pd.read_pickle("20220210_smooth_window_16.pkl")

    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    print("starting run")
    results = run_grid(df, feature_a, feature_b, lower, upper, num_points, objectives, forecast)
    dump_pickle(
        results,
        f"{TIMESTR}_{feature_a}_{feature_b}_{lower}_{upper}_{num_points}_{objectives}_{str(forecast)}".replace(
            "/", "*"
        ),
    )


if __name__ == "__main__":
    main()
