"""Calendar semantics, natural language alignment, and target isolation."""
import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing.calendar_text import calendar_to_text, date_to_text
from src.data.preprocessing.generate_prompts import build_prompt


class CalendarPromptTests(unittest.TestCase):
    def frame(self):
        dates = pd.date_range('2016-01-01', periods=10)
        phase = dates.dayofweek * 2 * np.pi / 7
        return pd.DataFrame({
            'timestamp': dates, 'water_produced': np.arange(10.) + 100,
            'temperature_mean': np.arange(10.) + 20, 'day': dates.day,
            'dayofweek_sin': np.sin(phase), 'dayofweek_cos': np.cos(phase),
            'season_sin': 0., 'season_cos': 1., 'is_holiday': [1] + [0]*9,
            'is_school_recess_pr': 1,
        })

    def test_date_words_preserve_year_and_leap_day(self):
        self.assertEqual(date_to_text('2016-01-01'), 'two thousand and sixteen, day one of January')
        self.assertEqual(date_to_text('2026-01-01'), 'two thousand and twenty-six, day one of January')
        self.assertEqual(date_to_text('2020-02-29'), 'two thousand and twenty, day twenty-nine of February')

    def test_conversion_preserves_measurements_and_source(self):
        source = self.frame()
        before = source.copy(deep=True)
        result = calendar_to_text(source)
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result[['water_produced', 'temperature_mean']],
                                      source[['water_produced', 'temperature_mean']])
        self.assertEqual(result.dayofweek.iloc[:3].tolist(), ['Friday', 'Saturday', 'Sunday'])
        self.assertTrue(result.season.eq('summer').all())
        self.assertEqual(result.day.iloc[0], 'one')
        self.assertEqual(result.is_holiday.iloc[:2].tolist(), ['a public holiday', 'not a public holiday'])
        self.assertTrue(result.is_school_recess_pr.eq('during the Paraná school recess').all())
        self.assertFalse(any(c.endswith(('_sin', '_cos')) for c in result))

    def test_seasons_decode_supplied_encoding_at_boundaries(self):
        frame = pd.DataFrame({'timestamp': pd.to_datetime(['2016-03-19', '2016-03-20',
                              '2016-06-20', '2016-09-22']),
                              'season_sin': [0., 1., 0., -1.], 'season_cos': [1., 0., -1., 0.]})
        self.assertEqual(calendar_to_text(frame).season.tolist(), ['summer', 'autumn', 'winter', 'spring'])
        with self.assertRaisesRegex(ValueError, 'Both season'):
            calendar_to_text(frame.drop(columns='season_cos'))

    def test_prompt_narrates_all_covariates_and_questions(self):
        frame = self.frame()
        prompt = build_prompt(frame.iloc[:3], 7,
                              future_covariates=frame.iloc[3:].drop(columns='water_produced'), region='Guaratuba')
        self.assertIn('From two thousand and sixteen, day one of January to '
                      'two thousand and sixteen, day three of January, the average temperature '
                      'of region Guaratuba was 20; 21; 22 degrees Celsius on each day, respectively.', prompt)
        self.assertIn('100.00; 101.00; 102.00', prompt)
        self.assertIn('Friday; Saturday; Sunday', prompt)
        self.assertIn('December through March', prompt)
        self.assertIn('Brazilian holidays and school recesses', prompt)
        self.assertEqual(prompt.count('What is the water_produced going to be on'), 7)
        self.assertNotIn('CSV', prompt)
        self.assertNotIn('2016-01-01', prompt)
        self.assertNotIn('season_sin', prompt)
        self.assertNotIn('tourist city', build_prompt(frame.iloc[:3], 1, region='Joinville'))

    def test_future_targets_cannot_change_prompt(self):
        frame = self.frame()
        def prompt(data):
            return build_prompt(data.iloc[:3], 7,
                                future_covariates=data.iloc[3:].drop(columns='water_produced'))
        original = prompt(frame)
        frame.loc[3:, 'water_produced'] = 99999999
        self.assertEqual(original, prompt(frame))
        with self.assertRaisesRegex(ValueError, 'Future target'):
            build_prompt(frame.iloc[:3], 7, future_covariates=frame.iloc[3:])
        with self.assertRaisesRegex(ValueError, 'dates must match'):
            build_prompt(frame.iloc[:3], 7, future_covariates=frame.iloc[2:9].drop(columns='water_produced'))
