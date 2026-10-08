"""LLM history stays fixed while evaluation origins advance."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.features.covariates import model_inputs
from src.evaluation.cross_validation import ForecastWindow
from src.pipelines.forecasting_pipeline import qwen_history_days, run_dataset


class QwenFixedContextTests(unittest.TestCase):
    def test_invalid_context_rejected_before_loading_data_or_model(self):
        for config in [{}, *({'qwen_history_days': value} for value in
                            (None, 0, 30, 365, 730, True, 7.0, '15'))]:
            with self.subTest(config=config), patch('src.pipelines.forecasting_pipeline.prepare_dataset') as prepare, \
                    patch('src.pipelines.forecasting_pipeline.create_model') as create:
                with self.assertRaisesRegex(ValueError, 'must be 7 or 15'):
                    run_dataset('joinville', {}, config, 'qwen', Path('unused'))
                prepare.assert_not_called()
                create.assert_not_called()

    def test_fixed_context_preserves_evaluation_dates_and_target_isolation(self):
        frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=70),
                              'water_produced': np.arange(70.)})
        dataset = {'numerical_covariates': [], 'categorical_covariates': []}
        for origin in (30, 37, 44):
            window = ForecastWindow(0, origin, 7)
            baseline, expected_future = model_inputs(frame, window, dataset)
            self.assertEqual(len(baseline), origin)
            for days in (7, 15):
                validated = qwen_history_days({'qwen_history_days': days})
                history, future = model_inputs(frame, window, dataset, history_days=validated)
                self.assertEqual(len(history), days)
                self.assertEqual(history.timestamp.iloc[0], frame.timestamp.iloc[origin-days])
                self.assertLess(history.timestamp.max(), future.timestamp.min())
                self.assertNotIn('water_produced', future)
                pd.testing.assert_frame_equal(future, expected_future)
