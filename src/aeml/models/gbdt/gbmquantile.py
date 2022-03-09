from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from loguru import logger

class LightGBMQuantileRegressor:
    """Helper class to do quantile regression with LightGBM models.
    For simplicity (and to avoid too much multiple testing) we assume that
    the hyperparameters for all submodels are identical.
    """

    def __init__(self, lower_quantile, mid_quantile, upper_quantile, **kwargs):
        self.lower_model = LightGBMModel(objective="quantile", alpha=lower_quantile, **kwargs)
        self.mid_model = LightGBMModel(objective="quantile", alpha=mid_quantile, **kwargs)
        self.upper_model = LightGBMModel(objective="quantile", alpha=upper_quantile, **kwargs)

    def fit(self, y_train, X_train):
        for name, model in [('lower percentile', self.lower_model), ('middle percentile', self.mid_model), ('upper percentile', self.upper_model)]:
            logger.info(f'fitting model {name}')
            model.fit(y_train, X_train)

    def forecast(self, **kwargs):
        lower = self.lower_model.predict(**kwargs)
        mid = self.mid_model.predict(**kwargs)
        upper = self.upper_model.predict(**kwargs)

        return lower, mid, upper

    def historical_forecasts(self, **kwargs):
        lower = self.lower_model.historical_forecasts(**kwargs)
        mid = self.mid_model.historical_forecasts(**kwargs)
        upper = self.upper_model.historical_forecasts(**kwargs)

        return lower, mid, upper
