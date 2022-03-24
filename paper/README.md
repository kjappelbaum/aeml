# Scripts/Notebooks to reproduce the findings in "Machine learning for industrial processes: Forecasting amine emissions from a carbon capture plant"

## Notebooks
- `20220310_plot_causalimpact.ipynb` used to plots the results of the causal impact analysis (e.g. generated via `xgboost_causalimpact.py`)
- `20220310_plot_forecast_overview.ipynb` used to plot an overview of the historical forecasts 
- `20220310_train_gbdt_on_all.ipynb` used to train the GBDT model on all data (subsequently used to compute the scenarios)
- `20220306_predict_w_gbdt.ipynb` example for training a GBDT model 

## Scripts 
### Causaul impact analysis 
- `causalimpact_sweep.py` run the hyperparamter sweep 
- `causalimpact_xgboost.py` run the causal impact analysis using GBDT models 
- `tcn_causalimpact.py` run the analysis using TCN models
- `step_times.pkl` contains the timestamps for the step changes in our study 

### Scenarios 
- `loop_over_maps_gbdt.py` / `loop_over_maps_scitas.py` used to create and submit slurm script for "scenario" analysis 
- `plot_effects_gbdt.py` / `plot_effects.py` used to convert the outputs of the scenario scripts into heatmaps 
- `run_gbdt_scenarios.py` / `run_scenarios.py` contain the logic for running the scenarios 
