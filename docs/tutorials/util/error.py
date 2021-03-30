from sklearn.metrics import mean_squared_error


def rmse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs) ** 0.5
