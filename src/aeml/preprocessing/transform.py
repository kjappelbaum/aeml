from darts.dataprocessing.transformers import Scaler
from typing import List 

def transform_data(train_tuple: tuple, test_tuples: List[tuple]):
    """Scale data using minmax scaling

    Args:
        train_tuple (tuple): tuple of darts time series for training
        test_tuples (List[tuple]): tuples (x,y) of darts time series for testing

    Returns:
        tuple: tuple of time series for training, test tuples and transformers
    """
    x_train, y_train = train_tuple

    transformer = Scaler()

    x_train = transformer.fit_transform(x_train)

    y_transformer = Scaler(name="YScaler")
    y_train = y_transformer.fit_transform(y_train)

    transformed_test_tuples = []
    for x_test, y_test in test_tuples:
        print(x_test.pd_dataframe().shape, y_test.pd_dataframe().shape)
        x_test = transformer.transform(x_test)
        y_test = y_transformer.transform(y_test)
        transformed_test_tuples.append((x_test, y_test))

    return (x_train, y_train), transformed_test_tuples, (transformer, y_transformer)