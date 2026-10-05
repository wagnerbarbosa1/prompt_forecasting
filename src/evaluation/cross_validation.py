"""Identical, chronological, expanding-history rolling origins for all models."""
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
