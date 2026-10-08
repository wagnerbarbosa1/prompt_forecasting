"""Shared chronological evaluation origins; each model selects its allowed history."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastWindow:
    context_start: int
    origin: int
    horizon: int

    @property
    def history(self) -> slice:
        return slice(self.context_start, self.origin)

    @property
    def target(self) -> slice:
        return slice(self.origin, self.origin + self.horizon)


def build_windows(df, context: int, horizon: int = 7, step: int = 7) -> list[ForecastWindow]:
    """Context is the initial history length. Every later history starts at zero."""
    if min(context, horizon, step) < 1:
        raise ValueError('Window parameters must be positive')
    windows = [ForecastWindow(0, origin, horizon)
               for origin in range(context, len(df) - horizon + 1, step)
               if df.iloc[origin:origin + horizon].observed.all()]
    if not windows:
        raise ValueError('No complete observed evaluation windows')
    return windows


def validation_split(training_size: int, fraction: float = 0.2) -> tuple[slice, slice]:
    """Last 20% of a supplied training prefix; never split a test interval."""
    if training_size < 2 or not 0 < fraction < 1:
        raise ValueError('Invalid training size or validation fraction')
    size = max(1, int(training_size * fraction))
    return slice(0, training_size - size), slice(training_size - size, training_size)


@dataclass(frozen=True)
class ExpandingValidationFold:
    number: int
    train_end: int
    validation_end: int
    test_end: int

    @property
    def training(self):
        return slice(0, self.train_end)

    @property
    def validation(self):
        return slice(self.train_end, self.validation_end)

    @property
    def test(self):
        return slice(self.validation_end, self.test_end)


def expanding_validation_folds(df, initial_training=355, validation_days=30, test_days=30):
    """Chronological folds with a rolling validation block and growing train prefix.

    Full test blocks only. Advancing by test_days reveals that block's actual
    observations for the next origin, never for predictions within that block.
    """
    if min(initial_training, validation_days, test_days) < 1:
        raise ValueError('Fold lengths must be positive')
    folds = []
    for number, train_end in enumerate(range(initial_training, len(df)-validation_days-test_days+1, test_days), 1):
        validation_end = train_end + validation_days
        test_end = validation_end + test_days
        if df.iloc[train_end:test_end].observed.all():
            folds.append(ExpandingValidationFold(number, train_end, validation_end, test_end))
    if not folds:
        raise ValueError('No complete train/validation/test folds')
    return folds
