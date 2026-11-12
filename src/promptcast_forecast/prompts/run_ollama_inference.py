#!/usr/bin/env python3
"""
Generate PromptCast-style prompts for urban water-demand forecasting.

Experimental design
-------------------
Scenarios:
    S1: water demand only
    S2: water demand + calendar context
    S3: water demand + calendar + meteorological forecast for the target days

Prompting strategies:
    zero: zero-shot
    few_random: few-shot with randomly selected historical examples
    few_similar: few-shot with historically similar examples

IMPORTANT LEAKAGE CONTROL
-------------------------
For a forecast origin t:
    - The input history ends at t-1.
    - The target is t ... t+h-1.
    - For S3, target-day meteorological values are treated as WEATHER FORECASTS
      available at the forecast origin, not as observed weather.
    - Calendar information for the target dates is allowed because it is
      deterministic/known in advance (weekday, month, holiday flags).
    - Few-shot demonstrations are selected ONLY from examples whose complete
      future target is already in the past relative to the forecast origin.
    - Similarity is calculated ONLY from information available in the
      demonstration input windows; never from their future targets.
    - The first two years of the series are never evaluated. They are available
      only as historical context.
    - All evaluation origins are in the final two years.

Expected input columns:
    timestamp
    water_produced
    temperature_mean
    radiation_mean
    relative_humidity_mean
    precipitation_mean

The *_std columns are deliberately ignored.

For S3, the four meteorological columns are interpreted as weather forecasts
for their corresponding dates, available at the forecast origin.

Output directory:
    <output>/
        manifest.csv
        s1/
            zero/
                h1/
                    prompt_<window_id>_<scenario>_<strategy>.txt
                    ...
                h3/
                h7/
            few_random/
                h1/
                ...
            few_similar/
                h1/
                ...
        s2/
            ...
        s3/
            ...
        metadata/
            windows.csv
            demonstrations.csv

The generated prompts can subsequently be sent to Ollama by a separate
inference script. This keeps prompt generation and LLM inference separate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET = "water_produced"
DATE = "timestamp"

WEATHER_FORECAST_COLUMNS = [
    "temperature_mean",
    "radiation_mean",
    "relative_humidity_mean",
    "precipitation_mean",
]

EXPECTED_COLUMNS = [DATE, TARGET] + WEATHER_FORECAST_COLUMNS

SCENARIOS = {
    "s1": "demand",
    "s2": "demand_calendar",
    "s3": "demand_calendar_weather",
}

STRATEGIES = {
    "zero": 0,
    "few_random": 3,
    "few_similar": 3,
}


@dataclass(frozen=True)
class ForecastWindow:
    """One forecasting window, uniquely identified by origin + horizon."""

    window_id: str
    origin_id: str
    origin_idx: int
    history_start_idx: int
    history_end_idx: int       # exclusive
    target_start_idx: int
    target_end_idx: int        # exclusive


def make_prompt_id(window_id: str, scenario: str, strategy: str) -> str:
    """Globally unique identifier for one generated prompt."""
    return f"{window_id}_{scenario}_{strategy}"


@dataclass(frozen=True)
class Demonstration:
    """A past example used in a few-shot prompt."""

    demo_id: str
    input_start_idx: int
    input_end_idx: int
    target_start_idx: int
    target_end_idx: int


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load, sort and validate the dataset.

    Only the six variables needed by the experiment are retained.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep the six modeling variables plus any pre-existing deterministic
    # calendar flags. The *_std columns are intentionally excluded.
    keep = EXPECTED_COLUMNS + [
        "is_holiday_ctba_gtba_jve",
        "is_carnival",
        "is_school_recess_pr",
    ]
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    df = df[keep].copy()
    df[DATE] = pd.to_datetime(df[DATE], errors="raise")
    df = df.sort_values(DATE).drop_duplicates(subset=[DATE]).reset_index(drop=True)

    for col in [TARGET] + WEATHER_FORECAST_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[[TARGET] + WEATHER_FORECAST_COLUMNS].isna().any().any():
        bad = df[[TARGET] + WEATHER_FORECAST_COLUMNS].isna().sum()
        raise ValueError(f"Missing/non-numeric values found:\n{bad[bad > 0]}")

    if not df[DATE].is_monotonic_increasing:
        raise ValueError("Timestamp is not sorted.")

    # Calendar features are deterministic and do not use future observations.
    add_calendar_features(df)

    return df


def add_calendar_features(df: pd.DataFrame) -> None:
    """Add deterministic calendar descriptors."""
    df["weekday"] = df[DATE].dt.day_name()
    df["month"] = df[DATE].dt.month
    df["month_name"] = df[DATE].dt.month_name()
    df["day_of_year"] = df[DATE].dt.dayofyear
    df["is_weekend"] = (df[DATE].dt.dayofweek >= 5).astype(int)

    # These flags reproduce the calendar variables used in the previous
    # Ollama experiment when they exist in the source data.
    # If they are absent, they are conservatively set to zero.
    for col in [
        "is_holiday_ctba_gtba_jve",
        "is_carnival",
        "is_school_recess_pr",
    ]:
        if col not in df.columns:
            df[col] = 0


# ---------------------------------------------------------------------
# Forecast windows
# ---------------------------------------------------------------------

def get_evaluation_start(df: pd.DataFrame, evaluation_years: int = 2) -> pd.Timestamp:
    """Return the first date belonging to the final N calendar years."""
    return df[DATE].max() - pd.DateOffset(years=evaluation_years) + pd.Timedelta(days=1)


def build_forecast_windows(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    evaluation_years: int = 2,
    step: int = 1,
) -> List[ForecastWindow]:
    """Build rolling forecast windows.

    A window ending at index `origin_idx` uses rows
    [origin_idx-lookback, origin_idx) as input and
    [origin_idx, origin_idx+horizon) as the future target.

    Only target dates inside the final evaluation period are retained.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    eval_start = get_evaluation_start(df, evaluation_years)
    n = len(df)
    windows: List[ForecastWindow] = []

    for origin_idx in range(lookback, n - horizon + 1, step):
        target_start = df.iloc[origin_idx][DATE]

        if target_start < eval_start:
            continue

        origin_id = target_start.strftime("%Y%m%d")
        window_id = f"{origin_id}_h{horizon}"

        windows.append(
            ForecastWindow(
                window_id=window_id,
                origin_id=origin_id,
                origin_idx=origin_idx,
                history_start_idx=origin_idx - lookback,
                history_end_idx=origin_idx,
                target_start_idx=origin_idx,
                target_end_idx=origin_idx + horizon,
            )
        )

    return windows


# ---------------------------------------------------------------------
# Few-shot candidate construction
# ---------------------------------------------------------------------

def candidate_demonstrations(
    df: pd.DataFrame,
    window: ForecastWindow,
    lookback: int,
    horizon: int,
    min_gap: int = 0,
) -> List[Demonstration]:
    """Return demonstrations whose complete target was known at forecast time.

    For a current forecast origin O, a demonstration is valid when its
    complete target ends at or before O. Its target may therefore overlap
    the current *historical input* window. That is not leakage: those target
    values were already observed by time O.

    `min_gap` optionally enforces extra separation between the demonstration
    target end and the current forecast origin.
    """
    candidates: List[Demonstration] = []

    # target_end is exclusive. A target ending exactly at O is fully known
    # when the forecast at O is made.
    latest_allowed_target_end = window.origin_idx - min_gap

    for demo_origin in range(lookback, latest_allowed_target_end + 1):
        demo_input_start = demo_origin - lookback
        demo_input_end = demo_origin
        demo_target_start = demo_origin
        demo_target_end = demo_origin + horizon

        if demo_target_end > latest_allowed_target_end:
            continue

        demo_origin_id = df.iloc[demo_origin][DATE].strftime("%Y%m%d")
        demo_id = f"{demo_origin_id}_h{horizon}"

        candidates.append(
            Demonstration(
                demo_id=demo_id,
                input_start_idx=demo_input_start,
                input_end_idx=demo_input_end,
                target_start_idx=demo_target_start,
                target_end_idx=demo_target_end,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def fmt_number(value: float) -> str:
    """Stable numeric formatting."""
    return f"{float(value):.2f}"


def format_demand_history(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> str:
    """Format historical demand as date/value pairs."""
    lines = []
    for _, row in df.iloc[start_idx:end_idx].iterrows():
        lines.append(
            f"{row[DATE].date()}: water_produced={fmt_number(row[TARGET])} m3"
        )
    return "\n".join(lines)


def format_weather_forecast(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> str:
    """Format meteorological forecasts for the corresponding dates."""
    lines = []
    for _, row in df.iloc[start_idx:end_idx].iterrows():
        lines.append(
            f"{row[DATE].date()}: "
            f"temperature={row['temperature_mean']:.2f} C, "
            f"radiation={row['radiation_mean']:.2f}, "
            f"relative_humidity={row['relative_humidity_mean']:.2f} %, "
            f"precipitation={row['precipitation_mean']:.2f} mm"
        )
    return "\n".join(lines)


def calendar_description(row: pd.Series) -> str:
    """Describe deterministic calendar information."""
    tags = [row["weekday"]]

    if row["is_weekend"]:
        tags.append("weekend")
    if row["is_holiday_ctba_gtba_jve"]:
        tags.append("public_holiday")
    if row["is_carnival"]:
        tags.append("carnival_period")
    if row["is_school_recess_pr"]:
        tags.append("school_recess")

    return ", ".join(tags)


def format_calendar_history(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> str:
    """Format calendar context for the historical input window."""
    lines = []
    for _, row in df.iloc[start_idx:end_idx].iterrows():
        lines.append(
            f"{row[DATE].date()}: "
            f"{calendar_description(row)}, month={row['month_name']}"
        )
    return "\n".join(lines)


def format_target_calendar(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> str:
    """Format calendar information known in advance for target dates.

    This is allowed because calendar attributes are deterministic and known
    before the forecast is made. No observed future demand/weather is used.
    """
    lines = []
    for _, row in df.iloc[start_idx:end_idx].iterrows():
        lines.append(
            f"{row[DATE].date()}: "
            f"{calendar_description(row)}, month={row['month_name']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------

def z_normalize_vector(x: np.ndarray) -> np.ndarray:
    """Z-normalize a vector; return zeros for a constant vector."""
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()

    if std < 1e-12:
        return np.zeros_like(x)

    return (x - mean) / std


def build_similarity_vector(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    scenario: str,
) -> np.ndarray:
    """Build the vector used ONLY to select similar historical examples.

    Crucially, this function only accesses the demonstration/input window.
    It never accesses the demonstration future target.

    S1:
        normalized demand history

    S2:
        normalized demand history + deterministic calendar encoding

    S3:
        normalized demand history + normalized weather history
    """
    block = df.iloc[start_idx:end_idx]

    parts = [z_normalize_vector(block[TARGET].to_numpy(dtype=float))]

    if scenario in {"s2", "s3"}:
        # Calendar is deterministic. Cyclic encodings avoid arbitrary
        # numerical ordering of weekdays/months.
        dow = block[DATE].dt.dayofweek.to_numpy(dtype=float)
        month = block[DATE].dt.month.to_numpy(dtype=float)

        parts.extend([
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            block["is_weekend"].to_numpy(dtype=float),
            block["is_holiday_ctba_gtba_jve"].to_numpy(dtype=float),
            block["is_carnival"].to_numpy(dtype=float),
            block["is_school_recess_pr"].to_numpy(dtype=float),
        ])

    if scenario == "s3":
        for col in WEATHER_FORECAST_COLUMNS:
            parts.append(z_normalize_vector(block[col].to_numpy(dtype=float)))

    return np.concatenate(parts)


def euclidean_similarity_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two input representations."""
    return float(np.linalg.norm(a - b))


def select_random_demonstrations(
    candidates: Sequence[Demonstration],
    n_examples: int,
    rng: random.Random,
) -> List[Demonstration]:
    """Select few-shot examples randomly."""
    if len(candidates) < n_examples:
        raise ValueError(
            f"Only {len(candidates)} historical demonstrations available; "
            f"{n_examples} requested."
        )

    return rng.sample(list(candidates), n_examples)


def select_similar_demonstrations(
    df: pd.DataFrame,
    window: ForecastWindow,
    candidates: Sequence[Demonstration],
    scenario: str,
    n_examples: int,
) -> List[Tuple[Demonstration, float]]:
    """Select nearest historical demonstrations using input data only."""
    current_vec = build_similarity_vector(
        df,
        window.history_start_idx,
        window.history_end_idx,
        scenario,
    )

    scored = []

    for demo in candidates:
        demo_vec = build_similarity_vector(
            df,
            demo.input_start_idx,
            demo.input_end_idx,
            scenario,
        )
        distance = euclidean_similarity_distance(current_vec, demo_vec)
        scored.append((demo, distance))

    scored.sort(key=lambda x: x[1])

    if len(scored) < n_examples:
        raise ValueError(
            f"Only {len(scored)} historical demonstrations available; "
            f"{n_examples} requested."
        )

    return scored[:n_examples]


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def format_demonstration(
    df: pd.DataFrame,
    demo: Demonstration,
    scenario: str,
    distance: Optional[float] = None,
) -> str:
    """Format one few-shot example.

    The future target is shown here because it is a historical outcome of
    the demonstration. It is NOT used to select the demonstration.
    """
    pieces = [
        "Example:",
        "Historical observations:",
        format_demand_history(
            df,
            demo.input_start_idx,
            demo.input_end_idx,
        ),
    ]

    if scenario in {"s2", "s3"}:
        pieces.extend([
            "",
            "Historical calendar context:",
            format_calendar_history(
                df,
                demo.input_start_idx,
                demo.input_end_idx,
            ),
        ])

    if scenario == "s3":
        pieces.extend([
            "",
            "Meteorological forecast for the demonstrated target dates:",
            format_weather_forecast(
                df,
                demo.target_start_idx,
                demo.target_end_idx,
            ),
        ])

    target_text = []
    for _, row in df.iloc[demo.target_start_idx:demo.target_end_idx].iterrows():
        target_text.append(
            f"{row[DATE].date()}: {fmt_number(row[TARGET])} m3"
        )

    pieces.extend([
        "",
        "Observed water production in the following days:",
        "\n".join(target_text),
    ])

    if distance is not None:
        pieces.insert(1, f"Similarity distance: {distance:.6f}")

    return "\n".join(pieces)


def build_prompt(
    df: pd.DataFrame,
    window: ForecastWindow,
    scenario: str,
    strategy: str,
    demonstrations: Optional[List[Tuple[Demonstration, Optional[float]]]],
) -> str:
    """Build the complete prompt for one forecasting window."""
    horizon = window.target_end_idx - window.target_start_idx
    lookback = window.history_end_idx - window.history_start_idx

    parts = [
        "You are forecasting daily water production for a municipal "
        "water utility.",
        "",
    ]

    # Few-shot demonstrations come before the query.
    if demonstrations:
        parts.append(
            "Below are historical examples. Use them as examples of how "
            "historical patterns relate to subsequent water production."
        )
        parts.append("")

        for demo, distance in demonstrations:
            parts.append(
                format_demonstration(
                    df=df,
                    demo=demo,
                    scenario=scenario,
                    distance=distance if strategy == "few_similar" else None,
                )
            )
            parts.append("")

    # Current query.
    parts.extend([
        "Now forecast the following case.",
        "",
        f"The historical input contains the last {lookback} days, ending "
        f"before the forecast period.",
        "",
        "Historical water production:",
        format_demand_history(
            df,
            window.history_start_idx,
            window.history_end_idx,
        ),
    ])

    if scenario in {"s2", "s3"}:
        parts.extend([
            "",
            "Historical calendar context:",
            format_calendar_history(
                df,
                window.history_start_idx,
                window.history_end_idx,
            ),
            "",
            "Calendar information known in advance for the forecast dates:",
            format_target_calendar(
                df,
                window.target_start_idx,
                window.target_end_idx,
            ),
        ])

    if scenario == "s3":
        parts.extend([
            "",
            "Meteorological forecast for the forecast dates:",
            format_weather_forecast(
                df,
                window.target_start_idx,
                window.target_end_idx,
            ),
            "",
            "The meteorological values above are forecasts available "
            "at the forecast origin, not observed weather values. Use "
            "them together with the historical demand and calendar "
            "information to forecast water production.",
        ])

    parts.extend([
        "",
        f"Predict water production for the next {horizon} day(s), in m3.",
        "Return ONLY the predicted numerical values, comma-separated, "
        "in chronological order, with no explanation.",
        "Example output format: 12345.67, 12876.30",
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def target_values(
    df: pd.DataFrame,
    window: ForecastWindow,
) -> List[float]:
    """Return future target values for evaluation metadata."""
    return df.iloc[
        window.target_start_idx:window.target_end_idx
    ][TARGET].astype(float).tolist()


def make_window_record(
    df: pd.DataFrame,
    window: ForecastWindow,
    scenario: str,
    strategy: str,
    horizon: int,
    lookback: int,
) -> Dict:
    """Metadata for one generated prompt."""
    return {
        "prompt_id": make_prompt_id(window.window_id, scenario, strategy),
        "window_id": window.window_id,
        "origin_id": window.origin_id,
        "scenario": scenario,
        "strategy": strategy,
        "lookback_days": lookback,
        "horizon_days": horizon,
        "history_start": str(df.iloc[window.history_start_idx][DATE].date()),
        "history_end": str(
            df.iloc[window.history_end_idx - 1][DATE].date()
        ),
        "forecast_start": str(
            df.iloc[window.target_start_idx][DATE].date()
        ),
        "forecast_end": str(
            df.iloc[window.target_end_idx - 1][DATE].date()
        ),
        "actual_target": json.dumps(
            target_values(df, window)
        ),
    }


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------

def generate_all_prompts(
    df: pd.DataFrame,
    output_dir: str | Path,
    lookback: int = 14,
    horizons: Sequence[int] = (1, 3, 7),
    evaluation_years: int = 2,
    step: int = 1,
    few_shot_examples: int = 3,
    random_seed: int = 42,
    min_demo_gap: int = 0,
) -> None:
    """Generate all prompt files and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    demo_records = []
    windows_records = []

    # Separate RNG streams make the random few-shot selection reproducible
    # and independent of scenario/horizon processing order.
    base_rng = random.Random(random_seed)
    window_counts_by_horizon = {}

    for horizon in horizons:
        windows = build_forecast_windows(
            df=df,
            lookback=lookback,
            horizon=horizon,
            evaluation_years=evaluation_years,
            step=step,
        )
        window_counts_by_horizon[horizon] = len(windows)

        for window in windows:
            windows_records.append({
                "window_id": window.window_id,
                "origin_id": window.origin_id,
                "horizon": horizon,
                "origin_date": str(
                    df.iloc[window.origin_idx][DATE].date()
                ),
                "history_start": str(
                    df.iloc[window.history_start_idx][DATE].date()
                ),
                "history_end": str(
                    df.iloc[window.history_end_idx - 1][DATE].date()
                ),
                "forecast_start": str(
                    df.iloc[window.target_start_idx][DATE].date()
                ),
                "forecast_end": str(
                    df.iloc[window.target_end_idx - 1][DATE].date()
                ),
            })

            for scenario in SCENARIOS:
                candidates = candidate_demonstrations(
                    df=df,
                    window=window,
                    lookback=lookback,
                    horizon=horizon,
                    min_gap=min_demo_gap,
                )

                for strategy in STRATEGIES:
                    n_examples = (
                        0 if strategy == "zero" else few_shot_examples
                    )

                    selected: List[Tuple[Demonstration, Optional[float]]] = []

                    if strategy == "few_random":
                        rng = random.Random(
                            base_rng.randint(0, 2**31 - 1)
                        )
                        demos = select_random_demonstrations(
                            candidates,
                            n_examples,
                            rng,
                        )
                        selected = [(d, None) for d in demos]

                    elif strategy == "few_similar":
                        selected = [
                            (d, dist)
                            for d, dist in select_similar_demonstrations(
                                df=df,
                                window=window,
                                candidates=candidates,
                                scenario=scenario,
                                n_examples=n_examples,
                            )
                        ]

                    prompt = build_prompt(
                        df=df,
                        window=window,
                        scenario=scenario,
                        strategy=strategy,
                        demonstrations=selected,
                    )

                    prompt_dir = (
                        output_dir
                        / scenario
                        / strategy
                        / f"h{horizon}"
                    )
                    prompt_dir.mkdir(parents=True, exist_ok=True)

                    prompt_id = make_prompt_id(
                        window.window_id, scenario, strategy
                    )
                    prompt_file = (
                        prompt_dir
                        / f"prompt_{prompt_id}.txt"
                    )
                    prompt_file.write_text(
                        prompt,
                        encoding="utf-8",
                    )

                    record = make_window_record(
                        df=df,
                        window=window,
                        scenario=scenario,
                        strategy=strategy,
                        horizon=horizon,
                        lookback=lookback,
                    )

                    record["prompt_file"] = str(
                        prompt_file.relative_to(output_dir)
                    )
                    record["n_demonstrations"] = len(selected)

                    if selected:
                        record["demonstration_ids"] = json.dumps([
                            d.demo_id for d, _ in selected
                        ])
                        record["demonstration_distances"] = json.dumps([
                            None if dist is None else float(dist)
                            for _, dist in selected
                        ])

                        for d, dist in selected:
                            demo_records.append({
                                "forecast_window_id": window.window_id,
                                "scenario": scenario,
                                "strategy": strategy,
                                "horizon": horizon,
                                "demo_id": d.demo_id,
                                "demo_input_start": str(
                                    df.iloc[d.input_start_idx][DATE].date()
                                ),
                                "demo_input_end": str(
                                    df.iloc[d.input_end_idx - 1][DATE].date()
                                ),
                                "demo_target_start": str(
                                    df.iloc[d.target_start_idx][DATE].date()
                                ),
                                "demo_target_end": str(
                                    df.iloc[d.target_end_idx - 1][DATE].date()
                                ),
                                "similarity_distance": dist,
                            })

                    manifest_records.append(record)

    manifest_df = pd.DataFrame(manifest_records)

    # A window is intentionally reused across scenarios and prompting
    # strategies. Therefore window_id is NOT expected to be globally unique.
    # The globally unique unit is prompt_id = window + scenario + strategy.
    if manifest_df["prompt_id"].duplicated().any():
        duplicates = manifest_df.loc[
            manifest_df["prompt_id"].duplicated(keep=False),
            "prompt_id"
        ].tolist()
        raise RuntimeError(
            f"Non-unique prompt_id detected: {duplicates[:10]}"
        )

    # Keep an explicit consistency check for the experimental cell key.
    if manifest_df.duplicated(
        subset=["window_id", "scenario", "strategy"]
    ).any():
        raise RuntimeError(
            "Duplicate (window_id, scenario, strategy) combination detected."
        )

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    pd.DataFrame(manifest_records).to_csv(
        output_dir / "manifest.csv",
        index=False,
    )
    pd.DataFrame(windows_records).drop_duplicates().to_csv(
        metadata_dir / "windows.csv",
        index=False,
    )
    pd.DataFrame(demo_records).to_csv(
        metadata_dir / "demonstrations.csv",
        index=False,
    )

    config = {
        "lookback_days": lookback,
        "horizons": list(horizons),
        "evaluation_years": evaluation_years,
        "step": step,
        "few_shot_examples": few_shot_examples,
        "random_seed": random_seed,
        "min_demo_gap": min_demo_gap,
        "scenarios": SCENARIOS,
        "strategies": {
            "zero": 0,
            "few_random": few_shot_examples,
            "few_similar": few_shot_examples,
        },
        "input_columns": EXPECTED_COLUMNS,
        "std_columns_excluded": [
            "temperature_std",
            "radiation_std",
            "relative_humidity_std",
            "precipitation_std",
        ],
        "leakage_policy": {
            "future_demand": "never included in current prompt",
            "future_weather": "included in S3 as meteorological forecasts for target dates",
            "future_calendar": (
                "allowed because deterministic/known in advance"
            ),
            "few_shot_targets": (
                "allowed only when their complete target is historical "
                "relative to the current forecast origin"
            ),
            "similarity_features": (
                "input window only; demonstration future targets are "
                "never used for selection"
            ),
        },
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("Prompt generation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Prompts generated: {len(manifest_records):,}")
    print("Forecast windows per horizon:")
    for horizon in horizons:
        print(f"  H{horizon}: {window_counts_by_horizon[horizon]:,}")
    print(f"Total forecast windows: {sum(window_counts_by_horizon.values()):,}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_horizons(text: str) -> Tuple[int, ...]:
    """Parse comma-separated horizons."""
    values = tuple(
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    )
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            "horizons must contain positive integers"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe PromptCast-style prompts for "
            "urban water-demand forecasting."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Input CSV containing the Guaratuba dataset.",
    )
    parser.add_argument(
        "--output",
        default="promptcast_prompts",
        help="Output directory.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=14,
        help="Number of historical days in each prompt.",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=(7,),
        help="Comma-separated forecast horizons, e.g. 1,3,7. Default is 7 "
             "(predicts t+1..t+7), matching the rolling-window design used "
             "in comparable time-series forecasting papers.",
    )
    parser.add_argument(
        "--evaluation-years",
        type=int,
        default=2,
        help="Final number of years used as the evaluation period.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=7,
        help="Rolling-window step in days. Default is 7, matching the "
             "default horizon -- non-overlapping windows (window hop == "
             "forecast horizon), as used in comparable time-series "
             "forecasting papers. Set to 1 for a fully overlapping rolling "
             "window instead.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=3,
        help="Number of few-shot demonstrations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for random few-shot selection.",
    )
    parser.add_argument(
        "--min-demo-gap",
        type=int,
        default=0,
        help=(
            "Optional number of days separating the demonstration target "
            "from the current history. 0 is strict enough to prevent "
            "future leakage."
        ),
    )

    args = parser.parse_args()

    df = load_dataset(args.data)

    generate_all_prompts(
        df=df,
        output_dir=args.output,
        lookback=args.lookback,
        horizons=args.horizons,
        evaluation_years=args.evaluation_years,
        step=args.step,
        few_shot_examples=args.few_shot_examples,
        random_seed=args.random_seed,
        min_demo_gap=args.min_demo_gap,
    )


if __name__ == "__main__":
    main()