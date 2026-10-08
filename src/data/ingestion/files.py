"""Filesystem input and atomic artifact writes."""
import json
from pathlib import Path
import pandas as pd
from src.data.validation.schema import validate_series


def read_series(path: str | Path, *, textual_calendar: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if textual_calendar:
        from src.data.preprocessing.calendar_text import date_from_text
        frame['timestamp'] = frame.timestamp.map(date_from_text)
    return validate_series(frame)


def write_json(path: str | Path, value: dict) -> None:
    """Replace a complete JSON file atomically; a crash cannot leave half a cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
