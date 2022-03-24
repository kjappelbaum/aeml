# -*- coding: utf-8 -*-
from .gbmquantile import LightGBMQuantileRegressor


def run_ci_model(
    x_train,
    y_train,
    lags,
    feature_lag,
    n_estimators,
    bagging_freq,
    bagging_fraction,
    num_leaves,
    extra_trees,
    max_depth,
    num_features,
    output_chunk_length,
    quantiles=(0.05, 0.5, 0.95),
):
    model = LightGBMQuantileRegressor(
        quantiles[0],
        quantiles[1],
        quantiles[2],
        lags=lags,
        lags_past_covariates=[feature_lag] * num_features,
        n_estimators=n_estimators,
        bagging_freq=bagging_freq,
        bagging_fraction=bagging_fraction,
        num_leaves=num_leaves,
        extra_trees=extra_trees,
        max_depth=max_depth,
        output_chunk_length=output_chunk_length,
    )

    model.fit(y_train, x_train)

    return model


def run_model(
    x_train,
    y_train,
    lags,
    lag_1,
    lag_2,
    lag_3,
    lag_4,
    lag_5,
    lag_6,
    n_estimators,
    bagging_freq,
    bagging_fraction,
    num_leaves,
    extra_trees,
    max_depth,
    output_chunk_length,
    quantiles=(0.05, 0.5, 0.95),
):
    model = LightGBMQuantileRegressor(
        quantiles[0],
        quantiles[1],
        quantiles[2],
        lags=lags,
        lags_past_covariates=[lag_1, lag_2, lag_3, lag_4, lag_5, lag_6],
        n_estimators=n_estimators,
        bagging_freq=bagging_freq,
        bagging_fraction=bagging_fraction,
        num_leaves=num_leaves,
        extra_trees=extra_trees,
        max_depth=max_depth,
        output_chunk_length=output_chunk_length,
    )

    model.fit(y_train, x_train)

    return model
