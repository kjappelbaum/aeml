# -*- coding: utf-8 -*-
from abc import ABC, ABCMeta, abstractmethod
from functools import partial
from itertools import product
from random import sample
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from darts.logging import get_logger, raise_deprecation_warning, raise_if, raise_if_not
from darts.models import TCNModel
from darts.timeseries import TimeSeries
from darts.utils import _build_tqdm_iterator, _with_sanity_checks
from darts.utils.data.inference_dataset import InferenceDataset
from darts.utils.torch import random_method
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from aeml.models.utils import enable_dropout

logger = get_logger(__name__)


class TCNDropout(TCNModel):
    @random_method
    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        enable_mc_dropout: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Predict the ``n`` time step following the end of the training series, or of the specified ``series``.
        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .
        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try calling :func:`predict()` on a model with the wrong covariates argument.
        Darts will also complain if the provided covariates do not have a sufficient time span.
        In general, not all models require the same covariates' time spans:
        * | Models relying on past covariates require the last ``input_chunk_length`` of the ``past_covariates``
          | points to be known at prediction time. For horizon values ``n > output_chunk_length``, these models
          | require at least the next ``n - output_chunk_length`` future values to be known as well.
        * | Models relying on future covariates require the next ``n`` values to be known.
          | In addition (for :class:`DualCovariatesTorchModel` and :class:`MixedCovariatesTorchModel`), they also
          | require the "historic" values of these future covariates (over the past ``input_chunk_length``).
        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.
        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            Optionally, a series or sequence of series, representing the history of the target series whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        past_covariates
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models' training ``batch_size`` value.
        verbose
            Optionally, whether to print progress.
            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of ``series``, or the forecast of the training series
            if ``series`` is not specified and the model has been trained on a single series.
        """
        super().predict(n, series, past_covariates, future_covariates)

        if series is None:
            raise_if(
                self.training_series is None,
                "Input series has to be provided after fitting on multiple series.",
            )
            series = self.training_series

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = self.past_covariate_series
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = self.future_covariate_series

        called_with_single_series = False
        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]

        past_covariates = (
            [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates
        )
        future_covariates = (
            [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates
        )

        # encoders are set when calling fit(), but not when calling fit_from_dataset()
        if self.encoders is not None and self.encoders.encoding_available:
            past_covariates, future_covariates = self.encoders.encode_inference(
                n=n,
                target=series,
                past_covariate=past_covariates,
                future_covariate=future_covariates,
            )

        dataset = self._build_inference_dataset(
            target=series,
            n=n,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        predictions = self.predict_from_dataset(
            n,
            dataset,
            trainer=trainer,
            verbose=verbose,
            batch_size=batch_size,
            n_jobs=n_jobs,
            roll_size=roll_size,
            num_samples=num_samples,
            enable_mc_dropout=enable_mc_dropout,
        )

        return predictions[0] if called_with_single_series else predictions

    @random_method
    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        enable_mc_dropout: bool = False,
    ) -> Sequence[TimeSeries]:

        """
        This method allows for predicting with a specific :class:`darts.utils.data.InferenceDataset` instance.
        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced
        for inference. In most cases, you'll rather want to call :func:`predict()` instead, which will create an
        appropriate :class:`InferenceDataset` for you.
        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .
        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        input_series_dataset
            Optionally, a series or sequence of series, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction.  Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models ``batch_size`` value.
        verbose
            Optionally, whether to print progress.
            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self._verify_inference_dataset_type(input_series_dataset)

        # check that covariates and dimensions are matching what we had during training
        self._verify_predict_sample(input_series_dataset[0])

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(
                0 < roll_size <= self.output_chunk_length,
                "`roll_size` must be an integer between 1 and `self.output_chunk_length`.",
            )

        # check that `num_samples` is a positive integer
        raise_if_not(num_samples > 0, "`num_samples` must be a positive integer.")

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size

        # set prediction parameters
        self.model.set_predict_parameters(
            n=n,
            num_samples=num_samples,
            roll_size=roll_size,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

        if enable_mc_dropout:
            enable_dropout(self.model)

        pred_loader = DataLoader(
            input_series_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        if verbose is not None:
            raise_deprecation_warning(
                "kwarg `verbose` is deprecated and will be removed in a future Darts version. "
                "Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, "
                "`progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict "
                "at model creation.",
                logger,
            )
        verbose = True if verbose is None else verbose

        # setup trainer. will only be re-instantiated if both `trainer` and `self.trainer` are `None`
        trainer = trainer if trainer is not None else self.trainer
        self._setup_trainer(trainer=trainer, verbose=verbose, epochs=self.n_epochs)

        # prediction output comes as nested list: list of predicted `TimeSeries` for each batch.
        predictions = self.trainer.predict(self.model, pred_loader)
        # flatten and return
        return [ts for batch in predictions for ts in batch]

    def _predict_wrapper(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        enable_mc_dropout: bool = False,
    ) -> TimeSeries:
        return self.predict(n, num_samples=num_samples, enable_mc_dropout=enable_mc_dropout)

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: bool = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        enable_mc_dropout: bool = False,
    ) -> Union[TimeSeries, List[TimeSeries]]:

        """Compute the historical forecasts that would have been obtained by this model on the `series`.
        This method uses an expanding training window;
        it repeatedly builds a training set from the beginning of `series`. It trains the
        model on the training set, emits a forecast of length equal to forecast_horizon, and then moves
        the end of the training set forward by `stride` time steps.
        By default, this method will return a single time series made up of the last point of each
        historical forecast. This time series will thus have a frequency of ``series.freq * stride``.
        If `last_points_only` is set to False, it will instead return a list of the
        historical forecasts series.
        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False, the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. This is not
        supported by all models.
        Parameters
        ----------
        series
            The target time series to use to successively train and evaluate the historical forecasts.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        forecast_horizon
            The forecast horizon for the predictions
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Not all models support setting
            `retrain` to `False`. Notably, this is supported by neural networks based models.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to retain only the last point of each historical forecast.
            If set to True, the method returns a single ``TimeSeries`` containing the successive point forecasts.
            Otherwise returns a list of historical ``TimeSeries`` forecasts.
        verbose
            Whether to print progress
        Returns
        -------
        TimeSeries or List[TimeSeries]
            By default, a single ``TimeSeries`` instance created from the last point of each individual forecast.
            If `last_points_only` is set to False, a list of the historical forecasts.
        """

        # TODO: do we need a check here? I'd rather leave these checks to the models/datasets.
        # if covariates:
        #     raise_if_not(
        #         series.end_time() <= covariates.end_time() and covariates.start_time() <= series.start_time(),
        #         'The provided covariates must be at least as long as the target series.'
        #     )

        # only GlobalForecastingModels support historical forecastings without retraining the model
        base_class_name = self.__class__.__base__.__name__
        raise_if(
            not retrain and not self._supports_non_retrainable_historical_forecasts(),
            f"{base_class_name} does not support historical forecastings with `retrain` set to `False`. "
            f"For now, this is only supported with GlobalForecastingModels such as TorchForecastingModels. "
            f"Fore more information, read the documentation for `retrain` in `historical_forecastings()`",
            logger,
        )

        # prepare the start parameter -> pd.Timestamp
        start = series.get_timestamp_at_point(start)

        # build the prediction times in advance (to be able to use tqdm)
        last_valid_pred_time = self._get_last_prediction_time(series, forecast_horizon, overlap_end)

        pred_times = [start]
        while pred_times[-1] < last_valid_pred_time:
            # compute the next prediction time and add it to pred times
            pred_times.append(pred_times[-1] + series.freq * stride)

        # the last prediction time computed might have overshot last_valid_pred_time
        if pred_times[-1] > last_valid_pred_time:
            pred_times.pop(-1)

        iterator = _build_tqdm_iterator(pred_times, verbose)

        # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
        forecasts = []

        last_points_times = []
        last_points_values = []

        # iterate and forecast
        for pred_time in iterator:
            train = series.drop_after(pred_time)  # build the training series

            # train_cov = covariates.drop_after(pred_time) if covariates else None

            if retrain:
                self._fit_wrapper(
                    series=train,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )

            forecast = self._predict_wrapper(
                n=forecast_horizon,
                series=train,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                enable_mc_dropout=enable_mc_dropout,
            )

            if last_points_only:
                last_points_values.append(forecast.all_values()[-1])
                last_points_times.append(forecast.end_time())
            else:
                forecasts.append(forecast)

        if last_points_only:
            if series.has_datetime_index:
                return TimeSeries.from_times_and_values(
                    pd.DatetimeIndex(last_points_times, freq=series.freq * stride),
                    np.array(last_points_values),
                )
            else:
                return TimeSeries.from_times_and_values(
                    pd.RangeIndex(
                        start=last_points_times[0],
                        stop=last_points_times[-1] + 1,
                        step=1,
                    ),
                    np.array(last_points_values),
                )

        return forecasts
