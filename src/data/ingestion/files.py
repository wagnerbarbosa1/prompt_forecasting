"""Filesystem input and atomic artifact writes."""
import json
from pathlib import Path
import pandas as pd
from src.data.validation.schema import validate_series


def read_series(path: str | Path) -> pd.DataFrame:
    return validate_series(pd.read_csv(path))


def write_json(path: str | Path, value: dict) -> None:
    """Replace a complete JSON file atomically; a crash cannot leave half a cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
