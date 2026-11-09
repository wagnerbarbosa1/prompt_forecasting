import unittest
from experiments.copy_review.reforecast import copied_sequences
from src.evaluation.cross_validation import SlidingWindow

class ReviewTests(unittest.TestCase):
    def test_full_and_partial_sequences(self):
        prompt='From a to b, the water production (water_produced) was 10; 20; 30; 40; 50; 60; 70 cubic metres on each day, respectively.'
        self.assertEqual(copied_sequences([10,20,30,40,50,60,70],prompt)[0]['length'],7)
        self.assertEqual(copied_sequences([1,20,30,40,5,6,7],prompt)[0]['length'],3)
        self.assertEqual(copied_sequences([10,2,30,4,50,6,70],prompt),[])

    def test_few_shot_output_and_covariate(self):
        prompt='Observed production on those example forecast dates: {"predictions": [1, 2, 3, 4, 5, 6, 7]}'
        self.assertEqual(copied_sequences([1,2,3,4,5,6,7],prompt)[0]['source'],'example_answer')
        prompt='From a to b, the supplied temperature is 1; 2; 3 degrees Celsius on each day, respectively.'
        self.assertEqual(copied_sequences([1,2,3,4,5,6,7],prompt)[0]['source'],'covariate')

    def test_sliding_offsets_keep_evaluation_dates(self):
        for initial in (365,730):
            for days in (7,15):
                offset=initial-days
                pairs=list(SlidingWindow(initial+21-offset,days,7,7).split())
                self.assertEqual([b.start+offset for a,b in pairs],[initial,initial+7,initial+14])
                self.assertTrue(all(a.stop-a.start==days and a.stop==b.start for a,b in pairs))
