"""Regression test: days 3–7 must contribute to the actual comparison CSV."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.pipelines.compare_models import compare


class SevenDayMetricsTests(unittest.TestCase):
    def test_comparison_includes_errors_after_day_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset, initial in [('joinville', 365), ('guaratuba', 730)]:
                for model in ['qwen', 'chronos', 'timesfm']:
                    folder = root / dataset / model
                    folder.mkdir(parents=True)
                    run = {'limit': 0, 'qwen_history_days': 15,
                           'config': {'strategy': 'zero'}, 'dataset_sha256': dataset,
                           'source_sha256': 'same-code', 'n_windows': 1,
                           'model': {'model': model}, 'covariates': ['temperature']}
                    status = {'failures': [], 'successful_windows': 1, 'expected_windows': 1}
                    (folder/'run.json').write_text(json.dumps(run))
                    (folder/'status.json').write_text(json.dumps(status))
                    pd.DataFrame({
                        'origin': ['2020-01-01']*7,
                        'timestamp': pd.date_range('2020-01-01', periods=7).strftime('%Y-%m-%d'),
                        'lead': range(1,8), 'y_true': [10.,20.,30.,40.,50.,60.,70.],
                        'y_pred': [10.,20.,40.,50.,60.,70.,80.],
                        'initial_context': [initial]*7,
                        'context': [15 if model == 'qwen' else initial]*7,
                        'history_start': ['2018-01-01']*7,
                        'horizon': [7]*7, 'elapsed_seconds_window': [1.]*7
                    }).to_csv(folder/'predictions.csv', index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                compare(root)
            aggregate = pd.read_csv(root/'metrics_comparison.csv')
            by_day = pd.read_csv(root/'metrics_by_horizon.csv')
            self.assertEqual(len(aggregate), 6)
            self.assertEqual(len(by_day), 42)
            np.testing.assert_allclose(aggregate.rmse, np.sqrt(500/7))
            np.testing.assert_allclose(by_day.loc[by_day.lead <= 2, 'rmse'], 0.)
            np.testing.assert_allclose(by_day.loc[by_day.lead >= 3, 'rmse'], 10.)
            self.assertEqual(set(by_day.lead), set(range(1,8)))
