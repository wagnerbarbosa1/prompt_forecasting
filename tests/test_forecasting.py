"""Leakage, evaluation and malformed-output regression tests; no model downloads."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data.preprocessing.generate_prompts import build_prompt, select_demonstrations
from src.data.preprocessing.missing_values import fill_past_only
from src.data.preprocessing.resampling import daily_grid
from src.data.validation.schema import validate_series
from src.evaluation.cross_validation import build_windows, validation_split
from src.evaluation.metrics import deterministic_metrics
from src.models.promptcast import OllamaConfig, QwenForecaster, parse_prediction
from src.pipelines.data_pipeline import prepare_dataset
from src.pipelines.forecasting_pipeline import ROOT
from src.data.features.covariates import model_inputs, covariate_columns, timesfm_covariates, validate_covariates


class ForecastTests(unittest.TestCase):
    def test_real_datasets_shared_protocol(self):
        for name, context in [('joinville_processed_1day',365), ('guaratuba_processed_1daysin',730)]:
            df, windows = prepare_dataset(ROOT / f'data/processed/{name}.csv', context)
            for window in windows:
                self.assertEqual(window.context_start, 0)
                self.assertGreaterEqual(window.origin, context)
                self.assertEqual(window.history.stop, window.target.start)
                self.assertEqual(window.horizon, 7)
                self.assertTrue(df.iloc[window.target].observed.all())
                self.assertLess(df.iloc[window.history].timestamp.max(), df.iloc[window.target].timestamp.min())
            self.assertTrue(all(b.origin-a.origin == 7 for a,b in zip(windows,windows[1:])))

    def test_future_target_mutation_does_not_change_prompt(self):
        df = pd.DataFrame({'timestamp': pd.date_range('2020-01-01',periods=30),
                           'water_produced': np.arange(30,dtype=float)})
        before = build_prompt(df.iloc[:20],7)
        df.loc[20:, 'water_produced'] = 987654321
        self.assertEqual(before,build_prompt(df.iloc[:20],7))
        self.assertNotIn('987654321',before)
        self.assertIn('2020-01-21',before)

    def test_demonstrations_remain_before_frozen_test(self):
        for strategy in ['few_random','few_similar']:
            examples = select_demonstrations(np.arange(80.), np.arange(20.), 7, strategy=strategy)
            self.assertEqual(len(examples),3)
            self.assertTrue(all(e.target_end <= 80 for e in examples))
            self.assertTrue(all(max(e.target)<80 for e in examples))

    def test_missing_day_is_filled_from_past_only(self):
        df = pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01','2020-01-03']), 'water_produced':[10.,999.]})
        result = fill_past_only(daily_grid(df))
        self.assertEqual(result.water_produced.iloc[1],10.)
        self.assertFalse(result.observed.iloc[1])

    def test_imputed_test_days_not_scored(self):
        df = pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=30),
                           'water_produced':np.arange(30.), 'observed':True})
        df.loc[12,'observed'] = False
        windows=build_windows(df,10,horizon=7,step=7)
        self.assertTrue(all(not(w.origin<=12<w.origin+7) for w in windows))

    def test_covariates_and_history_grow_without_future_targets(self):
        config = json.loads((ROOT / 'configs/benchmark.json').read_text())
        for dataset in config['datasets'].values():
            df, windows = prepare_dataset(ROOT / dataset['path'], dataset['context'])
            validate_covariates(df, dataset)
            self.assertEqual(windows[0].origin, dataset['context'])
            columns = covariate_columns(dataset)
            for offset, window in enumerate(windows[:3]):
                history, future = model_inputs(df, window, dataset)
                self.assertEqual(len(history), dataset['context'] + offset*7)
                self.assertEqual(history.timestamp.iloc[0], df.timestamp.iloc[0])
                self.assertEqual(list(future.columns), ['timestamp'] + columns)
                self.assertNotIn('water_produced', future)
                numerical, categorical = timesfm_covariates(history, future, dataset)
                self.assertEqual(set(numerical) | set(categorical), set(columns))
                for values in {**numerical, **categorical}.values():
                    self.assertEqual(len(values[0]), len(history)+7)
                short, same_future = model_inputs(df, window, dataset, history_days=15)
                self.assertEqual(len(short), 15)
                pd.testing.assert_frame_equal(future, same_future)
                prompt = build_prompt(short, 7, future_covariates=future)
                self.assertTrue(all(c in prompt for c in columns))
                poisoned = df.copy()
                poisoned.loc[window.origin:, 'water_produced'] = 987654321
                other_history, other_future = model_inputs(poisoned, window, dataset, history_days=15)
                self.assertEqual(prompt, build_prompt(other_history, 7, future_covariates=other_future))

    def test_future_target_column_is_rejected(self):
        history = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=3), 'water_produced':[1.,2.,3.]})
        future = pd.DataFrame({'timestamp':pd.date_range('2020-01-04',periods=7), 'water_produced':99.})
        with self.assertRaisesRegex(ValueError, 'Future target'):
            build_prompt(history, 7, future_covariates=future)

    def test_chronos_receives_entire_growing_history_and_covariates(self):
        from src.models.chronos import ChronosForecaster
        from unittest.mock import Mock
        config = json.loads((ROOT / 'configs/benchmark.json').read_text())
        dataset = config['datasets']['joinville']
        df, windows = prepare_dataset(ROOT / dataset['path'], dataset['context'])
        adapter = object.__new__(ChronosForecaster)
        adapter.dataset = dataset
        adapter.pipeline = Mock()
        adapter.pipeline.predict_df.return_value = pd.DataFrame({'0.5': [1.]*7})
        for window in windows[:3]:
            history, future = model_inputs(df, window, dataset)
            adapter.predict(history, 7, future_covariates=future)
            call = adapter.pipeline.predict_df.call_args
            self.assertEqual(call.kwargs['context_length'], window.origin)
            self.assertEqual(len(call.args[0]), window.origin)
            self.assertNotIn('water_produced', call.kwargs['future_df'])
            self.assertTrue(set(covariate_columns(dataset)).issubset(call.kwargs['future_df']))

    def test_invalid_model_responses(self):
        for text in ['{"predictions":[true]}','{"predictions":["1"]}',
                     '{"predictions":[NaN]}','{"predictions":[-1]}',
                     '{"predictions":[1,2]}','[1]','answer: 1',
                     '{"predictions":[1],"extra":0}']:
            with self.subTest(text=text), self.assertRaises((ValueError,TypeError)):
                parse_prediction(text,1)
        self.assertEqual(parse_prediction('{"predictions":[1.5]}',1),[1.5])

    def test_retry_and_raw_response_audit(self):
        model = object.__new__(QwenForecaster)
        model.config = OllamaConfig(retries=1)
        model.last_audit = {}
        answers=[{'done':True,'response':'{}'}, {'done':True,'response':'{"predictions":[2]}'}]
        with patch('src.models.promptcast.post_json',side_effect=answers), patch('src.models.promptcast.time.sleep'):
            self.assertEqual(model.predict(None,1,prompt='history'),[2.])
        self.assertEqual(len(model.last_audit['attempts']),2)
        self.assertIn('error',model.last_audit['attempts'][0])

    def test_context_budget_rejects_silent_truncation(self):
        model=object.__new__(QwenForecaster)
        model.config=OllamaConfig(num_ctx=800)
        with self.assertRaises(ValueError):
            model.predict(None,7,prompt='x'*1000)

    def test_metric_units(self):
        metrics=deterministic_metrics([10.,20.],[12.,18.])
        self.assertEqual(metrics['mae'],2.)
        self.assertEqual(metrics['rmse'],2.)
        self.assertAlmostEqual(metrics['mape_percent'],15.)
        self.assertAlmostEqual(metrics['r2'],0.84)
        self.assertEqual(deterministic_metrics([0.,10.],[0.,12.])['mape_zero_targets_excluded'],1)
        with self.assertRaises(ValueError):
            deterministic_metrics([1.,2.],[1.,float('nan')])

    def test_validation_uses_training_prefix(self):
        train,validation=validation_split(100)
        self.assertEqual((train.start,train.stop,validation.start,validation.stop),(0,80,80,100))

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            validate_series(pd.DataFrame({'timestamp':['2020-01-01']*2,'water_produced':[1,2]}))


if __name__ == '__main__':
    unittest.main()
