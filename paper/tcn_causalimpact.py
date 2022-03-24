from statistics import quantiles
import sys
import traceback

sys.path.append("../src")
from aeml.causalimpact.utils import get_timestep_tuples
import pickle
from aeml.causalimpact.utils import _select_unrelated_x
from aeml.models.forecast import (
    forecast,
    parallelized_inference,
)

from aeml.models.tcn_dropout import TCNDropout

from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd
from copy import deepcopy
import time

TIMESTR = time.strftime("%Y%m%d-%H%M%S")

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
    # "delta_t",
    "TI-35",
    #     "delta_t_2"
]

df = pd.read_pickle("20210508_df_for_causalimpact.pkl")

with open("step_times.pkl", "rb") as handle:
    times = pickle.load(handle)


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


# The experiments with the faster response on emissions are:

# - Day 1: water wash temperature step increase. 6~min time delay in amine and CO2 emissions.
# - Day 5: lean solvent and flue gas flow step decrease. 6~min time delay in amine emissions.
# - Day 7: lean solvent and water wash temperature step decrease. 6 min time delay in PZ emissions and 12~min delay in the response of AMP emissions.
# - Day 8: capture rate step decrease. And consequently reboiler level step decrease.
# - Day 9: dry bed operation


# If the time delay and the magnitude of emissions are both taken into consideration, then the experiments with the highest effect on emissions (ranking the one with the highest effect starting from the top to the bottom) become:

# - Day 9: dry bed operation
# - Day 7: lean solvent and water wash temperature step decrease
# - Day 1: water wash temperature step increase

step_changes = [
    ["TI-19"],
    ["FI-19"],
    ["TI-3"],
    ["FI-11"],
    ["FI-11", "FI-2"],
    ["TI-1213"],
    ["TI-1213", "TI-19"],
    ["capture rate"],
    # # ["capture rate"],
    # ["valve-position-12"],
]


causalimpact_models = {}


def run_model(x_trains, y_trains, input_chunk_length, output_chunk_length):

    # run = wandb.init(project='process_ml', reinit=True, sync_tensorboard=True)
    # with run:

    model_cov = TCNDropout(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_layers=4,
        num_filters=32,
        kernel_size=2,
        dropout=0.5616575899759398,
        weight_norm=False,
        batch_size=64,
        n_epochs=100,
        optimizer_kwargs={"lr": 0.019245430880992124},
    )

    model_cov.fit(series=y_trains, past_covariates=x_trains, verbose=False)

    return model_cov


if __name__ == "__main__":
    TARGETS = ["2-Amino-2-methylpropanol C4H11NO"]  # , "Piperazine C4H10N2"]
    # TARGETS = ["Carbon dioxide CO2", "Ammonia NH3"]
    for day in [1, 2]:  # range(len(step_changes)):
        cols = deepcopy(MEAS_COLUMNS)
        if step_changes[day][0] in MEAS_COLUMNS:
            for var in step_changes[day]:
                try:
                    cols = _select_unrelated_x(df, cols, var, 0.00)
                    print(TARGETS, day, var, cols)
                    y = TimeSeries.from_dataframe(df[TARGETS])
                    x = TimeSeries.from_dataframe(df[cols])

                    x_trains = []
                    y_trains = []

                    before, during, after, way_after = get_causalimpact_splits(x, y, day, times, df)

                    # #if len(before[0]) > len(way_after[0]):
                    x_trains.append(before[0])
                    y_trains.append(before[1])
                    # else:
                    # x_trains.append(way_after[0])
                    # y_trains.append(way_after[1])

                    xscaler = Scaler(name="x-scaler")
                    yscaler = Scaler(name="y-scaler")

                    x_trains[0] = xscaler.fit_transform(x_trains[0])
                    y_trains[0] = yscaler.fit_transform(y_trains[0])

                    before = (
                        xscaler.transform(before[0]),
                        yscaler.transform(before[1]),
                    )

                    during = (
                        xscaler.transform(during[0]),
                        yscaler.transform(during[1]),
                    )

                    print(len(during[0]))

                    after = (xscaler.transform(after[0]), yscaler.transform(after[1]))

                    before_x_df, before_y_df = (
                        before[0].pd_dataframe(),
                        before[1].pd_dataframe(),
                    )
                    during_x_df, during_y_df = (
                        during[0].pd_dataframe(),
                        during[1].pd_dataframe(),
                    )
                    after_x_df, after_y_df = (
                        after[0].pd_dataframe(),
                        after[1].pd_dataframe(),
                    )

                    day_x_df = pd.concat([before_x_df, during_x_df, after_x_df], axis=0)
                    day_x_ts = TimeSeries.from_dataframe(day_x_df)

                    day_y_df = pd.concat([before_y_df, during_y_df, after_y_df], axis=0)
                    day_y_ts = TimeSeries.from_dataframe(day_y_df)

                    chunk_length = 60  # math.floor(len(during[0]))
                    model = run_model(
                        x_trains,
                        y_trains,
                        input_chunk_length=61,  # hunk_length * 2,
                        output_chunk_length=chunk_length,
                    )

                    causalimpact_models[day] = {
                        "before": before,
                        "during": during,
                        "after": after,
                        "model": model,
                        "x_all": day_x_ts,
                        "y_all": day_y_ts,
                        "cols": cols,
                        "output_chunk_length": chunk_length,
                    }

                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                    pass

    for day, values in causalimpact_models.items():
        if step_changes[day][0] in MEAS_COLUMNS:
            print(
                (len(values["before"][0]) - 0.1 * len(values["before"][0])) / len(values["y_all"])
            )
            print(len(values["before"][0]) * 0.6)
            try:
                model = values["model"]

                predictions = forecast(
                    model,
                    values["x_all"],
                    values["before"][1],
                    future_len=len(values["during"][0]) + len(values["after"][0]),
                    repeats=50,
                    # horizon=values["output_chunk_length"],
                    # stride=values["output_chunk_length"]
                )
                print(type(predictions[0]))
                causalimpact_models[day]["predictions"] = predictions
            except Exception as e:
                print(e)
                pass

        with open(
            f"{TIMESTR}-_{day}_causalimpact_{'_'.join(TARGETS)}.pkl".replace("/", "*"),
            "wb",
        ) as handle:
            pickle.dump(causalimpact_models, handle)
