"""Ensure the context ablation changes only history and reports signed improvement."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.pipelines.compare_qwen_contexts import compare_contexts,relative_reduction


class QwenContextComparisonTests(unittest.TestCase):
    def test_positive_and_negative_improvement(self):
        self.assertEqual(relative_reduction(100,80),20)
        self.assertEqual(relative_reduction(100,120),-20)
        self.assertIsNone(relative_reduction(0,1))

    def test_comparison_uses_same_checkpoint_and_all_seven_days(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for dataset,initial in [('joinville',365),('guaratuba',730)]:
                for policy,context,error in [('15_days',15,10),('expanding',initial,5)]:
                    folder=root/policy/dataset/'qwen';folder.mkdir(parents=True)
                    run={'limit':0,'dataset_sha256':dataset,'source_sha256':'same',
                         'model':{'model':'same-qwen','digest':'same-digest'},'covariates':['weather'],
                         'first_test_origin':initial,'config':{'qwen':{'num_ctx':131072},
                         'qwen_history_days':15 if policy=='15_days' else None}}
                    (folder/'run.json').write_text(json.dumps(run))
                    (folder/'status.json').write_text(json.dumps({'expected_windows':1,'successful_windows':1,'failures':[]}))
                    pd.DataFrame({'origin':['2020-01-01']*7,'timestamp':pd.date_range('2020-01-01',periods=7).strftime('%Y-%m-%d'),
                        'lead':range(1,8),'y_true':range(10,17),'y_pred':range(10+error,17+error),
                        'initial_context':[initial]*7,'context':[context]*7,'history_start':['2018-01-01']*7,
                        'horizon':[7]*7,'elapsed_seconds_window':[1.]*7}).to_csv(folder/'predictions.csv',index=False)
            with contextlib.redirect_stdout(io.StringIO()):compare_contexts(root)
            scores=pd.read_csv(root/'improvement.csv')
            self.assertTrue(scores.rmse_reduction_percent.eq(50).all())
            self.assertEqual(len(pd.read_csv(root/'metrics_by_horizon.csv')),28)
            changed=root/'expanding/guaratuba/qwen/run.json'
            run=json.loads(changed.read_text());run['model']['digest']='different';changed.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError,'changes model'):
                compare_contexts(root)

    def test_tabular_prompt_preserves_values_and_expanding_prefix(self):
        import numpy as np
        from src.data.preprocessing.generate_prompts import build_tabular_prompt
        frame=pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=40),
                            'water_produced':np.arange(40.)+100,'weather':np.arange(40.)/2,'holiday':[0]*40})
        first=build_tabular_prompt(frame.iloc[:20],7,future_covariates=frame.iloc[20:27].drop(columns='water_produced'))
        second=build_tabular_prompt(frame.iloc[:27],7,future_covariates=frame.iloc[27:34].drop(columns='water_produced'))
        prefix=first.split('\nEND HISTORY:')[0]
        self.assertTrue(second.startswith(prefix+'\n'))
        table=prefix.split('(CSV, oldest to newest):\n')[1]
        restored=pd.read_csv(io.StringIO(table))
        np.testing.assert_allclose(restored.water_produced,frame.iloc[:20].water_produced)
        np.testing.assert_allclose(restored.weather,frame.iloc[:20].weather)
        changed=frame.copy();changed.loc[20:,'water_produced']=999999
        self.assertEqual(first,build_tabular_prompt(changed.iloc[:20],7,future_covariates=changed.iloc[20:27].drop(columns='water_produced')))
