"""Original public split API and its integration with pipeline metadata."""
import inspect
import unittest

import pandas as pd

from src.evaluation.cross_validation import ExpandingWindow, SlidingWindow
from src.evaluation.protocol import build_windows, expanding_validation_folds


class OriginalCrossValidationTests(unittest.TestCase):
    def test_original_expanding_and_sliding_boundaries(self):
        for splitter, starts in [(ExpandingWindow, [0, 0, 0]),
                                 (SlidingWindow, [0, 7, 14])]:
            split = splitter(n_samples=386, trainw=365, step=7, horizon=7).split()
            self.assertTrue(inspect.isgenerator(split))
            pairs = list(split)
            self.assertEqual([(a.start, a.stop, b.start, b.stop) for a, b in pairs],
                             [(start, origin, origin, origin + 7)
                              for start, origin in zip(starts, [365, 372, 379])])

    def test_incomplete_tail_is_excluded_and_exact_last_window_is_included(self):
        self.assertEqual(list(ExpandingWindow(371, 365, 7, 7).split()), [])
        self.assertEqual(list(ExpandingWindow(372, 365, 7, 7).split()),
                         [(slice(0, 365), slice(365, 372))])

    def test_pipeline_uses_original_slices_for_both_initial_contexts(self):
        for context in (365, 730):
            frame = pd.DataFrame({'observed': [True] * (context + 35)})
            windows = build_windows(frame, context)
            expected = list(ExpandingWindow(len(frame), context, 7, 7).split())
            self.assertEqual([(w.history, w.target) for w in windows], expected)
            self.assertEqual(windows[0].to_dict(),
                             {'context_start': 0, 'origin': context, 'horizon': 7})

    def test_observation_filter_belongs_to_adapter_only(self):
        frame = pd.DataFrame({'observed': [True] * 31})
        frame.loc[11, 'observed'] = False
        self.assertEqual(len(list(ExpandingWindow(len(frame), 10, 7, 7).split())), 3)
        windows = build_windows(frame, 10)
        self.assertEqual([w.origin for w in windows], [17, 24])

    def test_classical_folds_preserve_existing_metadata(self):
        fold = expanding_validation_folds(pd.DataFrame({'observed': [True] * 415}))[0]
        self.assertEqual(fold.to_dict(), {'number': 1, 'train_end': 355,
                                         'validation_end': 385, 'test_end': 415})
        self.assertEqual((fold.training, fold.validation, fold.test),
                         (slice(0, 355), slice(355, 385), slice(385, 415)))
