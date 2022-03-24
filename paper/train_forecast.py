from aeml.models.run import run_model
from aeml.models.forecast import forecast, parallelized_inference, summarize_results
from aeml.models.utils import split_data
from aeml.models.plotting import make_forecast_plot
from aeml.utils.io import dump_pickle

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pandas as pd

import matplotlib.pyplot as plt
import hydra
import time
import os
from matplotlib import rcParams
import traceback

plt.style.use("science")
rcParams["font.family"] = "sans-serif"

MEAS_COLUMNS = [
    "TI-19",
    "TI-3",
    "FI-19",
    "FI-11",
    "TI-1213",
    "TI-35",
]

TARGETS_clean = [
    "2-Amino-2-methylpropanol C4H11NO",
    "Piperazine C4H10N2",
    "Carbon dioxide CO2",
    "Ammonia NH3",
]

from aeml.models.run import run_model


STEP_INDICES = [75, 645, 2075, 2880, 3520, 4245]


def make_step_predictions(
    model, step_index, step_number, y_scaled, x_scaled, plot_dir, timestr, horizon
):
    before_step, after_step = y_scaled.split_before(x_scaled.get_timestamp_at_point(step_index))

    forecasts = forecast(model, x_scaled, before_step, len(after_step), repeats=20)
    mean_forecast, std_forecast = summarize_results(forecasts)

    fraction = len(before_step) / len(y_scaled)
    historical_forecasts = parallelized_inference(
        model, x_scaled, y_scaled, start=fraction, repeats=20, horizon=horizon
    )
    mean_historical_forecast, std_historical_forecast = summarize_results(historical_forecasts)

    # Plot the results
    for target in [0, 1]:
        make_forecast_plot(
            y_scaled,
            mean_forecast,
            std_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(
                plot_dir, f"{timestr}_forecast_scenario_{step_number}_{target}.pdf"
            ),
        )
        make_forecast_plot(
            y_scaled,
            mean_historical_forecast,
            std_historical_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(
                plot_dir, f"{timestr}_historical_forecast_scenario_{step_number}_{target}.pdf"
            ),
        )

    return {
        "scenario": step_number,
        "forecast": {
            "mean": mean_forecast,
            "std": std_forecast,
        },
        "historical_forecast": {"mean": mean_historical_forecast, "std": std_historical_forecast},
    }


def constant_coavariate_step_forecast(
    model, step_index, step_number, y_scaled, x_scaled, plot_dir, timestr, horizon
):
    before_step_y, after_step_y = y_scaled.split_before(x_scaled.get_timestamp_at_point(step_index))
    before_step_x, after_step_x = x_scaled.split_before(x_scaled.get_timestamp_at_point(step_index))
    after_step_x_df = after_step_x.pd_dataframe()
    after_step_x_df[MEAS_COLUMNS] = before_step_x[MEAS_COLUMNS].values()[0]
    after_step_x = TimeSeries.from_dataframe(after_step_x_df)

    x = before_step_x.concatenate(after_step_x)

    forecasts = forecast(model, x_scaled, before_step_y, len(after_step_y), repeats=20)
    mean_forecast, std_forecast = summarize_results(forecasts)

    fraction = len(before_step_y) / len(y_scaled)
    historical_forecasts = parallelized_inference(
        model, x_scaled, y_scaled, start=fraction, repeats=20, horizon=horizon
    )
    mean_historical_forecast, std_historical_forecast = summarize_results(historical_forecasts)

    # Plot the results
    for target in [0, 1]:
        make_forecast_plot(
            y_scaled,
            mean_forecast,
            std_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(
                plot_dir,
                f"{timestr}_forecast_scenario_constant_covariates_{step_number}_{target}.pdf",
            ),
        )
        make_forecast_plot(
            y_scaled,
            mean_historical_forecast,
            std_historical_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(
                plot_dir,
                f"{timestr}_historical_forecast_scenario_constant_covariates_{step_number}_{target}.pdf",
            ),
        )

    return {
        "scenario": step_number,
        "forecast": {
            "mean": mean_forecast,
            "std": std_forecast,
        },
        "historical_forecast": {"mean": mean_historical_forecast, "std": std_historical_forecast},
    }


@hydra.main(config_path="../conf/tcn", config_name="default")
def main(cfg):

    plot_dir = os.path.join(cfg.outdir, "plots")
    model_dir = os.path.join(cfg.outdir, "models")
    result_dir = os.path.join(cfg.outdir, "results")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    df = pd.read_pickle(cfg.dataframe)

    x = TimeSeries.from_dataframe(df[MEAS_COLUMNS])
    y = TimeSeries.from_dataframe(df[TARGETS_clean[0:2]])

    train, valid, test, ts, ts_2 = split_data(x, y, TARGETS_clean[0:2], 0.5)

    x_scaler = Scaler(name="x")
    y_scaler = Scaler(name="y")

    x_train = x_scaler.fit_transform(train[0])
    x_valid = x_scaler.transform(valid[0])
    x_test = x_scaler.transform(test[0])

    y_train = y_scaler.fit_transform(train[1])
    y_valid = y_scaler.transform(valid[1])
    y_test = y_scaler.transform(test[1])

    y_scaled = y_scaler.transform(y)
    x_scaled = x_scaler.transform(x)

    # Train model with the provided hyperparameters
    model = run_model(
        x_train,
        y_train,
        input_chunk_length=cfg.trainer.model.input_chunk_length,
        output_chunk_length=cfg.trainer.model.output_chunk_length,
        num_layers=cfg.trainer.model.num_layers,
        num_filters=cfg.trainer.model.num_filters,
        kernel_size=cfg.trainer.model.kernel_size,
        dropout=cfg.trainer.model.dropout,
        lr=cfg.trainer.model.lr,
        weight_norm=cfg.trainer.model.weight_norm,
        batch_size=cfg.trainer.model.batch_size,
        n_epochs=cfg.trainer.model.n_epochs,
    )

    model.save_model(os.path.join(model_dir, f"{timestr}_model.pth.tar"))

    # Run historical forecasts and forecasts on valid and test set
    forecasts = forecast(model, x_scaled, y_train, len(y_valid) + len(y_test), repeats=20)
    mean_forecast, std_forecast = summarize_results(forecasts)

    historical_forecasts = parallelized_inference(
        model,
        x_scaled,
        y_scaled,
        start=0.3,
        repeats=20,
        horizon=cfg.trainer.model.output_chunk_length,
    )
    mean_historical_forecast, std_historical_forecast = summarize_results(historical_forecasts)

    # Plot the results
    for target in [0, 1]:
        make_forecast_plot(
            y_scaled,
            mean_forecast,
            std_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(plot_dir, f"{timestr}_forecast_{target}.pdf"),
        )
        make_forecast_plot(
            y_scaled,
            mean_historical_forecast,
            std_historical_forecast,
            target_names=TARGETS_clean[0:2],
            target=target,
            outname=os.path.join(plot_dir, f"{timestr}_historical_forecast_{target}.pdf"),
        )

    # Run historical forecasts and forecasts for the scenario steps
    step_results = []

    for step_number, step_index in enumerate(STEP_INDICES):
        try:
            step_results.append(
                make_step_predictions(
                    model,
                    step_index,
                    step_number,
                    y_scaled,
                    x_scaled,
                    plot_dir,
                    timestr,
                    horizon=cfg.trainer.model.output_chunk_length,
                )
            )
        except Exception as e:
            print(e)

    # Run "historical forecasts" and "forecasts" for the steps, leaving covariates constant
    constant_covariate_step_results = []

    for step_number, step_index in enumerate(STEP_INDICES):
        try:
            constant_covariate_step_results.append(
                constant_coavariate_step_forecast(
                    model,
                    step_index,
                    step_number,
                    y_scaled,
                    x_scaled,
                    plot_dir,
                    timestr,
                    horizon=cfg.trainer.model.output_chunk_length,
                )
            )
        except Exception as e:
            print(traceback.format_exc())

    # save the results
    results = {
        "cfg": cfg,
        "timestamp": timestr,
        "foretcasts": {"mean": mean_forecast, "std": std_forecast},
        "historical_forecasts": {"mean": mean_historical_forecast, "std": std_historical_forecast},
        "steps": step_results,
        "constant_covariate_steps": constant_covariate_step_results,
    }
    dump_pickle(os.path.join(result_dir, f"{timestr}_results.pkl"), results)


if __name__ == "__main__":
    main()
