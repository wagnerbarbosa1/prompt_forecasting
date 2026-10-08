"""Direct textual CSV ingestion must preserve calendar meaning and split boundaries."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.ingestion.files import read_series
from src.data.preprocessing.calendar_text import calendar_to_text, date_from_text, date_to_text
from src.data.preprocessing.generate_prompts import build_prompt


class TextDatasetTests(unittest.TestCase):
    def test_text_roundtrip_and_prompt_target_isolation(self):
        dates = pd.date_range('2016-12-29', periods=10)
        original = pd.DataFrame({'timestamp': dates, 'water_produced': np.arange(10.)+100,
                                 'day': dates.day, 'is_holiday': [False]*10})
        words = calendar_to_text(original)
        pd.testing.assert_frame_equal(words, calendar_to_text(words))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.csv'
            words.to_csv(path, index=False)
            frame = read_series(path, textual_calendar=True)
        pd.testing.assert_series_equal(frame.timestamp, original.timestamp)
        self.assertEqual(frame.day.iloc[0], 'twenty-nine')
        future = frame.iloc[3:].drop(columns='water_produced')
        prompt = build_prompt(frame.iloc[:3], 7, future_covariates=future, region='Guaratuba')
        self.assertIn('twenty-nine; thirty; thirty-one', prompt)
        self.assertIn('not a public holiday', prompt)
        self.assertNotIn('2016-12-29', prompt)
        frame.loc[3:, 'water_produced'] = 999999
        self.assertEqual(prompt, build_prompt(frame.iloc[:3], 7,
                         future_covariates=frame.iloc[3:].drop(columns='water_produced'), region='Guaratuba'))

    def test_date_parser_preserves_year(self):
        for date in ('2016-01-01', '2026-01-01', '2020-02-29'):
            self.assertEqual(date_from_text(date_to_text(date)), pd.Timestamp(date))
