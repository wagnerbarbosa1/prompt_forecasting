"""Temporal boundaries, training-only preprocessing and recursive test isolation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data.features.autoregressive import lagged_samples
from src.evaluation.cross_validation import expanding_validation_folds
from src.models.classical import LagRegressor
from src.training.classical import select_and_forecast


class ClassicalBaselineTests(unittest.TestCase):
    def test_folds_expand_by_thirty_without_overlap(self):
        frame = pd.DataFrame({'observed': [True]*500})
        folds = expanding_validation_folds(frame)
        self.assertEqual([(f.train_end,f.validation_end,f.test_end) for f in folds],
                         [(355,385,415),(385,415,445),(415,445,475)])
        for previous, current in zip(folds, folds[1:]):
            self.assertEqual(previous.test_end,current.validation_end)
            self.assertEqual(current.train_end-previous.train_end,30)

    def test_features_and_scaler_only_use_training_history(self):
        values = np.arange(60.,dtype=float)
        features, targets = lagged_samples(values[:40],7)
        np.testing.assert_array_equal(features[-1],values[32:39])
        self.assertEqual(targets[-1],39)
        model=LagRegressor('knn',lags=7,neighbors=3).fit(values[:40])
        np.testing.assert_allclose(model.estimator.named_steps['standardscaler'].mean_,features.mean(axis=0))

    def test_selection_and_final_refit_do_not_receive_test_targets(self):
        seen=[]
        class Fake:
            def fit(self, values):
                self.values=np.asarray(values);seen.append(self.values.copy());return self
            def predict(self,horizon):
                return np.repeat(self.values[-1],horizon)
        full=np.arange(445.,dtype=float)
        def forecast(values):
            with patch('src.training.classical.create_classical_model',side_effect=lambda *args:Fake()):
                return select_and_forecast('knn',values[:355],values[355:385],30,[{'lags':7}])[0]
        original=forecast(full)
        full[385:]=999999
        np.testing.assert_array_equal(original,forecast(full))
        self.assertEqual([len(x) for x in seen],[355,385,355,385])
        self.assertTrue(all(x.max()<385 for x in seen))

    def test_real_lag_models_recursively_forecast_a_fixed_block(self):
        series=100+np.sin(np.arange(90.)/3)
        for name,params in [('linear_regression',{'lags':7}),('knn',{'lags':7,'neighbors':3})]:
            predictions,audit=select_and_forecast(name,series[:50],series[50:60],30,[params])
            self.assertEqual(len(predictions),30)
            self.assertTrue(np.isfinite(predictions).all())
            self.assertEqual(audit['final_fit_size'],60)
