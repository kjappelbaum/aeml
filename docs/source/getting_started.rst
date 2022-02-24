Installation
---------------

We recommend installing pyprocessta in a dedicated `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ or `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. Note that we tested the code on Python 3.8.

The latest version of aeml can be installed from GitHub using

.. code-block:: bash

    pip install git+https://github.com/kjappelbaum/aeml.git


Preprocessing
--------------

For basic preprocessing functions the :py:mod:`aeml.preprocessing` subpackage can be used.


Aligning to dataframes
========================

To align two dataframes, use

.. code-block:: python

    from aeml.preprocessing.align import align_two_dfs

    aligned_dataframe = align_two_dfs(dataframe_a, dataframe_b)


Filtering and smoothing
========================

To perform basic filtering operations you can use

.. code-block:: python

    from aeml.preprocessing.smooth import z_score_filter, exponential_window_smoothing

    dataframe_no_spikes = z_score_filter(dataframe)
    dataframe_smoothed = exponential_window_smoothing(dataframe)


Detrending
===========

Often, it can be useful to remove trend components from time series data. One can distinguish stochastic and deterministic trend components, and we provide utilities to remove both


.. code-block:: python

    from aeml.preprocessing.detrend import detrend_stochastic, detrend_linear_deterministc

    dataframe_no_linear_trend = detrend_linear_deterministc(input_dataframe)
    dataframe_no_stochastic_trend = detrend_stochastic(input_dataframe)


Resampling
=============

For many applications it is important to have data sampled on a regular grid. To resample data onto such a grid you can use

.. code-block:: python

    from aeml.preprocessing.resample import resample_regular

    data_resampled = resample_regular(input_dataframe, interval='2min')

EDA
----

Test for stationarity
======================

One of the most important tests before modeling time series data is to check for `stationarity <https://people.duke.edu/~rnau/411diff.htm>`_ (since many of the "simple" time series models assume stationarity).

.. code-block:: python

    from aeml.eda.statistics import check_stationarity

    test_results = check_stationarity(input_dataseries)

This will perform the `Augmented-Dickey Fuller <https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test>`_ and `Kwiatkowski–Phillips–Schmidt–Shin (KPSS) <https://en.wikipedia.org/wiki/KPSS_test>`_.

Granger causality
===================

One interesting analysis is to check for "correlations" between different timeseries. In timeseries speak, this means to look for `Granger causality <https://en.wikipedia.org/wiki/Granger_causality>`_.
To perform this analysis, you can use

.. code-block:: python

    from aeml.eda.statistics import compute_granger_causality_matrix

    causality_matrix = compute_granger_causality_matrix(input_dataframe)

The matrix can, for example, be plotted as heatmap and highlights the maximum "correlation" between two series (up to some maximum lag).


Training a TCN model
----------------------

The `Temporal convolutional neural network <https://unit8.co/resources/temporal-convolutional-networks-and-forecasting/>`_ implementation uses the darts library. The only change is that we make it possible to also enable dropout for inference.

.. code-block:: python

    from aeml.models.run import run_model
    from aeml.preprocessing.transform import transform_data

    x_timeseries, y_timeseries = get_data(my_dataframe, targets=my_targets, features=my_features_
    train_tuple, test_tuple = get_train_test_data(x_timeseries, y_timeseries, split_date="2010-01-18 12:59:15")
    train_tuple, test_tuples, transformers = transform_data(train_tuple, [test_tuple])

    model = run_model(train_tuple)


Causal impact analysis
--------------------------

Causal impact analysis allows to estimate the effect of some intervention in the absence of a control experiment. For doing so, one builds a model of what the behavior of the system would be without the intervention. The approach used in the `original causal impact paper <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full>`_ uses Bayesian structured time series models which, simply speaking, model time series via two key equation: a state equation that connects a latent, unobserved, state to the observations and once equation that describes the transition between states. The model is then defined by a model for the state and transitions between the states (e.g., local level and seasonality).

