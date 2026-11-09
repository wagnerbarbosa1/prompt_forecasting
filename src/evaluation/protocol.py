"""Adapt original splitters to pipeline metadata and observed-target checks."""
from src.evaluation.cross_validation import ExpandingWindow


class ForecastWindow:
    """Plain metadata record; boundaries are supplied by the original splitter."""
    def __init__(self, context_start, origin, horizon):
        self.context_start = context_start
        self.origin = origin
        self.horizon = horizon
        self.history = slice(context_start, origin)
        self.target = slice(origin, origin + horizon)

    def to_dict(self):
        return {'context_start': self.context_start, 'origin': self.origin,
                'horizon': self.horizon}


def build_windows(df, context: int, horizon: int = 7, step: int = 7) -> list[ForecastWindow]:
    """Context is the initial history length. Every later history starts at zero."""
    if min(context, horizon, step) < 1:
        raise ValueError('Window parameters must be positive')
    splitter = ExpandingWindow(len(df), context, step, horizon)
    windows = [ForecastWindow(trainidxs.start, testidxs.start, horizon)
               for trainidxs, testidxs in splitter.split()
               if df.iloc[testidxs].observed.all()]
    if not windows:
        raise ValueError('No complete observed evaluation windows')
    return windows


def validation_split(training_size: int, fraction: float = 0.2) -> tuple[slice, slice]:
    """Last 20% of a supplied training prefix; never split a test interval."""
    if training_size < 2 or not 0 < fraction < 1:
        raise ValueError('Invalid training size or validation fraction')
    size = max(1, int(training_size * fraction))
    return slice(0, training_size - size), slice(training_size - size, training_size)


class ExpandingValidationFold:
    """Metadata for the separate classical-model validation protocol."""
    def __init__(self, number, train_end, validation_end, test_end):
        self.number = number
        self.train_end = train_end
        self.validation_end = validation_end
        self.test_end = test_end
        self.training = slice(0, train_end)
        self.validation = slice(train_end, validation_end)
        self.test = slice(validation_end, test_end)

    def to_dict(self):
        return {'number': self.number, 'train_end': self.train_end,
                'validation_end': self.validation_end, 'test_end': self.test_end}


def expanding_validation_folds(df, initial_training=355, validation_days=30, test_days=30):
    """Chronological folds with a rolling validation block and growing train prefix.

    Full test blocks only. Advancing by test_days reveals that block's actual
    observations for the next origin, never for predictions within that block.
    """
    if min(initial_training, validation_days, test_days) < 1:
        raise ValueError('Fold lengths must be positive')
    folds = []
    splitter = ExpandingWindow(len(df), initial_training, test_days,
                               validation_days + test_days)
    for number, (trainidxs, nextidxs) in enumerate(splitter.split(), 1):
        train_end = trainidxs.stop
        validation_end = nextidxs.start + validation_days
        test_end = nextidxs.stop
        if df.iloc[train_end:test_end].observed.all():
            folds.append(ExpandingValidationFold(number, train_end, validation_end, test_end))
    if not folds:
        raise ValueError('No complete train/validation/test folds')
    return folds
