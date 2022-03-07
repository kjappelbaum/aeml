from darts.models.forecasting.gradient_boosted_model import LightGBMModel
class LightGBMQuantileRegressor:
    """Helper class to do quantile regression with LightGBM models. 
    For simplicity (and to avoid too much multiple testing) we assume that 
    the hyperparameters for all submodels are identical. 
    """
    def __init__(self, lower_quantile, mid_quantile, upper_quantile, *kwargs): 
        self.lower_model = 