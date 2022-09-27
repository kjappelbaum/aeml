# Scripts/Notebooks to reproduce the findings in "Machine learning for industrial processes: Forecasting amine emissions from a carbon capture plant"

## Notebooks
- `20220310_plot_causalimpact.ipynb` used to plots the results of the causal impact analysis (e.g. generated via `xgboost_causalimpact.py`)
- `20220310_plot_forecast_overview.ipynb` used to plot an overview of the historical forecasts (Figure 7 in the main text as well as the model evaluation)
- `20220310_train_gbdt_on_all.ipynb` used to train the GBDT model on all data (subsequently used to compute the scenarios)
- `20220306_predict_w_gbdt.ipynb` example for training a GBDT model 

## Scripts 
### Causal impact analysis 
- `causalimpact_sweep.py` run the hyperparamter sweep (assumes [Weights and Biases](https://wandb.ai/site) is set up)
- `causalimpact_xgboost.py` run the causal impact analysis using GBDT models 
- `tcn_causalimpact.py` run the analysis using TCN models
- `step_times.pkl` contains the timestamps for the step changes in our study 

### Scenarios 
- `loop_over_maps_gbdt.py` / `loop_over_maps_scitas.py` used to create and submit slurm script for "scenario" analysis 
- `plot_effects_gbdt.py` / `plot_effects.py` used to convert the outputs of the scenario scripts into heatmaps 
- `run_gbdt_scenarios.py` / `run_scenarios.py` contain the logic for running the scenarios 

### Models

Model checkpoints are archived on Zenodo (DOI: [https://dx.doi.org/10.5281/zenodo.5153417](10.5281/zenodo.5153417)) but also available in the `model` subdirectory. 
Unfortunately, we could only serialize the models as pickle files wherefore the same Python version and package versions are needed for reusing the models.

### Results 

The `results` subdirectory contains pre-computed results that are used in the notebooks that plot the results.