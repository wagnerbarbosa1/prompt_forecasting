"""Original few-shot behavior and generic Ollama model routing."""
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.preprocessing.generate_prompts import build_similarity_vector, select_demonstrations, build_prompt
from src.models.promptcast import OllamaConfig, OllamaForecaster
from src.pipelines.llm_experiment import configurations
from src.pipelines.forecasting_pipeline import ROOT


class LLMStrategyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=80),
                                   'water_produced': np.arange(80.),
                                   'temperature_mean': np.arange(80.) % 7,
                                   'is_weekend': [0, 1] * 40})

    def test_random_generator_advances_and_is_reproducible(self):
        def selections():
            rng = random.Random(42)
            return [[h.index[0] for h, _ in select_demonstrations(
                self.frame.iloc[:60], self.frame.iloc[60:67], 7, strategy='few_random', rng=rng)]
                    for _ in range(2)]
        first = selections()
        self.assertEqual(first, selections())
        self.assertNotEqual(first[0], first[1])

    def test_similarity_uses_weather_and_calendar_and_original_distance(self):
        history = self.frame.iloc[60:67]
        training = self.frame.iloc[:60]
        query = build_similarity_vector(history)
        expected = sorted(range(7, 54), key=lambda o: np.linalg.norm(
            query - build_similarity_vector(training.iloc[o-7:o])))[:3]
        examples = select_demonstrations(training, history, 7, strategy='few_similar')
        self.assertEqual([target.index[0] for _, target in examples], expected)
        changed = history.copy()
        changed['temperature_mean'] = changed.temperature_mean.to_numpy()[::-1]
        self.assertFalse(np.array_equal(query, build_similarity_vector(changed)))
        changed = history.copy()
        changed['is_weekend'] = 1 - changed.is_weekend
        self.assertFalse(np.array_equal(query, build_similarity_vector(changed)))

    def test_future_target_mutation_does_not_change_few_shot_prompt(self):
        def prompt(frame):
            examples = select_demonstrations(frame.iloc[:60], frame.iloc[60:67], 7, strategy='few_similar')
            self.assertTrue(all(target.index.max() < 60 for _, target in examples))
            return build_prompt(frame.iloc[60:67], 7, demonstrations=examples,
                                future_covariates=frame.iloc[67:74].drop(columns='water_produced'))
        before = prompt(self.frame)
        self.frame.loc[67:, 'water_produced'] = 987654321
        self.assertEqual(before, prompt(self.frame))
        self.assertIn('Historical training example:', before)
        self.assertIn('average temperature', before)
        self.assertNotIn('987654321', before)

    def test_matrix_contains_four_models_two_contexts_three_strategies(self):
        spec = json.loads((ROOT / 'configs/llm_experiment.json').read_text())
        runs = list(configurations(spec))
        self.assertEqual(len(runs), 24)
        self.assertEqual({r[3]['ollama']['model'] for r in runs}, set(spec['models'].values()))

    def test_non_reasoning_model_does_not_receive_think_option(self):
        model = object.__new__(OllamaForecaster)
        model.config = OllamaConfig(model='llama3.2:3b')
        model.supports_thinking = False
        response = {'done': True, 'response': '{"predictions": [1, 2]}'}
        with patch('src.models.promptcast.post_json', return_value=response) as post:
            self.assertEqual(model.predict(None, 2, prompt='test'), [1., 2.])
            self.assertNotIn('think', post.call_args.args[1])
