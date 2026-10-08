"""Select parameters on validation, then refit only the pre-test prefix."""
import numpy as np
from src.models.classical import create_classical_model


def select_and_forecast(name, training, validation, horizon, candidates):
    """No test targets are accepted by this interface, including during selection."""
    train = np.asarray(training, dtype=float)
    valid = np.asarray(validation, dtype=float)
    trials, best, best_score = [], None, np.inf
    for parameters in candidates:
        record = {'parameters': parameters}
        try:
            estimator = create_classical_model(name, parameters).fit(train)
            predictions = estimator.predict(len(valid))
            rmse = float(np.sqrt(np.mean((predictions-valid)**2)))
            if not np.isfinite(rmse):
                raise ValueError('Nonfinite validation score')
            record.update(status='ok', validation_rmse=rmse)
            if rmse < best_score:
                best, best_score = parameters, rmse
        except (ValueError, np.linalg.LinAlgError, RuntimeError, Warning) as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        trials.append(record)
    if best is None:
        raise ValueError(f'All {name} candidates failed: {trials}')
    pre_test = np.concatenate([train, valid])
    final_model = create_classical_model(name, best).fit(pre_test)
    predictions = final_model.predict(horizon)
    return predictions, {'selected': best, 'validation_rmse': best_score,
                         'selection_training_size': len(train), 'validation_size': len(valid),
                         'final_fit_size': len(pre_test), 'trials': trials}
