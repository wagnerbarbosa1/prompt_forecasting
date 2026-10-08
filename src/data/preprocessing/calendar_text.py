"""Render calendar features in English without changing the numerical datasets.

Run with ``python -m src.data.preprocessing.calendar_text`` to export text copies.
Cyclic features are decoded from their supplied values, not approximated by month.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December')
WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
SEASONS = ('summer', 'autumn', 'winter', 'spring')
FLAGS = {
    'is_holiday': ('not a public holiday', 'a public holiday'),
    'holiday': ('not a public holiday', 'a public holiday'),
    'is_holiday_ctba_gtba_jve': ('not a holiday in the Curitiba–Guaratuba–Joinville calendar',
                               'a holiday in the Curitiba–Guaratuba–Joinville calendar'),
    'is_weekend': ('a weekday', 'a weekend day'),
    'is_carnival': ('outside Carnival', 'during Carnival'),
    'is_school_recess_pr': ('outside the Paraná school recess', 'during the Paraná school recess'),
}


def number_to_words(value: int) -> str:
    """Spell out calendar integers (zero through 9999), independent of locale."""
    small = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
             'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
             'sixteen', 'seventeen', 'eighteen', 'nineteen')
    tens = ('', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
    integer = int(value)
    if integer != value or not 0 <= integer <= 9999:
        raise ValueError(f'Invalid calendar integer: {value}')
    if integer < 20:
        return small[integer]
    if integer < 100:
        quotient, remainder = divmod(integer, 10)
        return tens[quotient] + ('-' + small[remainder] if remainder else '')
    base = 100 if integer < 1000 else 1000
    quotient, remainder = divmod(integer, base)
    result = number_to_words(quotient) + (' hundred' if base == 100 else ' thousand')
    return result + ((' and ' if remainder < 100 else ' ') + number_to_words(remainder) if remainder else '')


def words_to_number(text: str) -> int:
    """Parse only the canonical integer spelling emitted by this exporter."""
    units = {number_to_words(n): n for n in (*range(20), *range(20, 100, 10))}
    total, group = 0, 0
    for word in text.replace('-', ' ').split():
        if word == 'and':
            continue
        if word == 'hundred':
            group *= 100
        elif word == 'thousand':
            total += group * 1000
            group = 0
        elif word in units:
            group += units[word]
        else:
            raise ValueError(f'Invalid calendar number: {text}')
    value = total + group
    if number_to_words(value) != text:
        raise ValueError(f'Noncanonical calendar number: {text}')
    return value


def date_from_text(text: str) -> pd.Timestamp:
    """Recover a timestamp solely to align splits; prompts retain word dates."""
    match = re.fullmatch(r'(.+), day (.+) of ([A-Za-z]+)', text)
    if not match or match[3] not in MONTHS:
        raise ValueError(f'Invalid textual date: {text}')
    return pd.Timestamp(year=words_to_number(match[1]), month=MONTHS.index(match[3])+1,
                        day=words_to_number(match[2]))


def date_to_text(value) -> str:
    date = date_from_text(value) if isinstance(value, str) and ', day ' in value else pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError('Missing calendar date')
    return f'{number_to_words(date.year)}, day {number_to_words(date.day)} of {MONTHS[date.month - 1]}'


def _cycle_labels(frame, prefix, labels):
    sine = frame[f'{prefix}_sin'].to_numpy(dtype=float)
    cosine = frame[f'{prefix}_cos'].to_numpy(dtype=float)
    phase = np.mod(np.arctan2(sine, cosine), 2 * np.pi)
    indices = np.rint(phase * len(labels) / (2 * np.pi)).astype(int) % len(labels)
    angles = indices * 2 * np.pi / len(labels)
    if not (np.allclose(sine, np.sin(angles), atol=1e-6) and
            np.allclose(cosine, np.cos(angles), atol=1e-6)):
        raise ValueError(f'Invalid {prefix} cyclic encoding')
    return [labels[index] for index in indices]


def calendar_to_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy, replacing only calendar fields and merging cyclic pairs."""
    result = frame.copy()
    result['timestamp'] = result.timestamp.map(date_to_text)
    for prefix, labels in [('dayofweek', WEEKDAYS), ('season', SEASONS)]:
        pair = [f'{prefix}_sin', f'{prefix}_cos']
        if any(column in result for column in pair):
            if not all(column in result for column in pair):
                raise ValueError(f'Both {prefix} cyclic components are required')
            result[prefix] = _cycle_labels(frame, prefix, labels)
            result = result.drop(columns=pair)
    for column in ('day', 'year'):
        if column in result:
            result[column] = result[column].map(lambda value: number_to_words(words_to_number(value))
                                               if isinstance(value, str) else number_to_words(value))
    if 'month' in result:
        def month_name(value):
            if int(value) != value or not 1 <= value <= 12:
                raise ValueError(f'Invalid month: {value}')
            return MONTHS[int(value) - 1]
        result['month'] = result.month.map(month_name)
    for column, labels in FLAGS.items():
        if column in result:
            def flag_label(value):
                if value in labels:
                    return value
                normalized = str(value).strip().lower()
                if normalized in {'0', '0.0', 'false'}:
                    return labels[0]
                if normalized in {'1', '1.0', 'true'}:
                    return labels[1]
                raise ValueError(f'Invalid calendar flag {column}: {value}')
            result[column] = result[column].map(flag_label)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', default=[
        Path('data/processed/joinville_processed_1day.csv'),
        Path('data/processed/guaratuba_processed_1daysin.csv')])
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/calendar_text'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.input:
        output = args.output_dir / f'{source.stem}_calendar_text.csv'
        if output.resolve() == source.resolve():
            raise ValueError('Text export must not overwrite its source')
        converted = calendar_to_text(pd.read_csv(source))
        converted.to_csv(output, index=False)
        print(f'{output}: {len(converted)} rows')


if __name__ == '__main__':
    main()
