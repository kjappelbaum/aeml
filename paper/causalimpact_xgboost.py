from aeml.causalimpact.utils import get_timestep_tuples, get_causalimpact_splits
import pickle
from aeml.causalimpact.utils import _select_unrelated_x
from aeml.models.gbdt.gbmquantile import LightGBMQuantileRegressor
from aeml.models.gbdt.run import run_ci_model
from aeml.models.gbdt.settings import *

from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd
from copy import deepcopy
import time
import numpy as np 
import click
import math 

settings = {
    0: {0: ci_0_0, 1: ci_0_1}, 
1: {0: ci_1_0, 1: ci_1_1},
2: {0: ci_2_0, 1: ci_2_1},
3: {0: ci_3_0, 1: ci_3_1},
4: {0: ci_4_0, 1: ci_4_1},
5: {0: ci_5_0, 1: ci_5_1},
6: {0: ci_6_0, 1: ci_6_1}
}

TIMESTR = time.strftime("%Y%m%d-%H%M%S")
TARGETS_clean = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]
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
    "delta_t",
    "TI-35",
    #     "delta_t_2"
]


DF = pd.read_pickle("20210508_df_for_causalimpact.pkl")

with open("step_times.pkl", "rb") as handle:
    times = pickle.load(handle)


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

to_exclude = {
    0: ["TI-19"],
    1: ["FI-19"],
    2: ["TI-3"],
    3: ["FI-11"],
    4: ["FI-11"],
    5: ["TI-1213", "TI-19"],
    6: [],
}

def select_columns(day):
    feat_to_exclude = to_exclude[day]
    feats = [f for f in MEAS_COLUMNS if f not in feat_to_exclude]
    return feats


@click.command('cli')
@click.argument('day', type=click.INT)
@click.argument('target', type=click.INT)
def run_causalimpact_analysis(day, target): 
    cols = select_columns(day)
    y = TimeSeries.from_dataframe(DF)[TARGETS_clean[target]]
    x = TimeSeries.from_dataframe(DF[cols])

    x_trains = []
    y_trains = []

    before, during, after, way_after = get_causalimpact_splits(
        x, y, day, times, DF
    )

    # We do multiseries training 
    x_trains.append(before[0])
    y_trains.append(before[1])
    x_trains.append(way_after[0])
    y_trains.append(way_after[1])

    xscaler = Scaler(name="x-scaler")
    yscaler = Scaler(name="y-scaler")

    longer = np.argmax([len(x_trains[0]), len(x_trains[1])])
    shorter = np.argmin([len(x_trains[0]), len(x_trains[1])])

    x_trains[longer] = xscaler.fit_transform(x_trains[longer])
    y_trains[longer] = yscaler.fit_transform(y_trains[longer])

    x_trains[shorter] = xscaler.transform(x_trains[shorter])
    y_trains[shorter] = yscaler.transform(y_trains[shorter])

    if len(x_trains[shorter]) < 300: 
        x_trains.pop(shorter)
        y_trains.pop(shorter)

    before = (
        xscaler.transform(before[0]),
        yscaler.transform(before[1]),
    )

    during = (
        xscaler.transform(during[0]),
        yscaler.transform(during[1]),
    )

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
    
    steps = math.ceil(len(during[0])/2)# * 2

    model = run_ci_model(
        x_trains,
        y_trains,
        **settings[day][target],
        num_features=len(cols),
                quantiles=(0.05, 0.5, 0.95), 
                output_chunk_length=steps
    )
    buffer = math.ceil(len(during[0])/3)
    b = before[1][:-buffer]
    predictions = model.forecast( 
                      n = len(during[0]) + 2* buffer,
        series =  b,
        past_covariates = day_x_ts,
                
)

    results = {
        'predictions': predictions, 
        'x_all': day_x_ts, 
        'before': before, 
        'during': during, 
        'after': after
    }

    with open(
            f"{TIMESTR}-causalimpact_{day}_{target}",
            "wb",
        ) as handle:
            pickle.dump(results, handle)

if __name__ == '__main__': 
    run_causalimpact_analysis()