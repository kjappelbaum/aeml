from darts.models.forecasting.gradient_boosted_model import LightGBMModel
class LightGBMQuantileRegressor:
    """Helper class to do quantile regression with LightGBM models. 
    For simplicity (and to avoid too much multiple testing) we assume that 
    the hyperparameters for all submodels are identical. 
    """
    def __init__(self, lower_quantile, mid_quantile, upper_quantile, **kwargs): 
        self.lower_model = LightGBMModel(objective='quantile', alpha=lower_quantile, **kwargs)
        self.mid_model = LightGBMModel(objective='quantile', alpha=mid_quantile, **kwargs)
        self.upper_model = LightGBMModel(objective='quantile', alpha=upper_quantile, **kwargs)

    def fit(self, y_train, X_train): 
        for model in [self.lower_model, self.mid_model, self.upper_model]:
            model.fit(y_train, X_train)

    def forecast(self, **kwargs): 
        lower = self.lower_model.predict(**kwargs)
        mid = self.mid_model.predict(**kwargs)
        upper = self.upper_model.predict(**kwargs)

        return lower, mid, upper

    
    def historical_forecasts(self, **kwargs): 
        lower = self.lower_model.historical_forecasts( **kwargs)
        mid = self.mid_model.historical_forecasts( **kwargs)
        upper = self.upper_model.historical_forecasts( **kwargs)

        return lower, mid, upper
