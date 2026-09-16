from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from numpy import sqrt


def deterministic_metrics(ytrue, yhat):
    mae = mean_absolute_error(ytrue, yhat)
    mape = mean_absolute_percentage_error(ytrue, yhat)
    rmse = sqrt(mean_squared_error(ytrue, yhat))
    r2 = r2_score(ytrue, yhat)

    dict = {
        "mae" : mae,
        "mape" : mape,
        "rmse" : rmse,
        "r2" : r2_score
    }

    return dict