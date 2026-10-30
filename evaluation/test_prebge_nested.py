"""Adversarial checks for training scopes and paired query-group confidence intervals."""
import unittest
import numpy as np

from .prebge_nested import safe_partition,gate_mask
from .study_bge_score_calibration import summary


class NestedScopeTests(unittest.TestCase):
    def test_action_training_cannot_reach_outer_holdout(self):
        groups=np.array(['a','b','c','d','e','f'])
        safe_partition([0,1],[2,3],[0,1,2,3],groups)
        with self.assertRaises(ValueError):safe_partition([0,1,4],[2,3],[0,1,2,3],groups)
        with self.assertRaises(ValueError):safe_partition([0,1],[1,2],[0,1,2],groups)

    def test_ids_and_groups_cannot_hide_duplicates(self):
        groups=np.array(['same','same','other'])
        with self.assertRaises(ValueError):safe_partition([0],[1,2],[0,1,2],groups)
        with self.assertRaises(ValueError):safe_partition([0,0],[2],[0,2],groups)

    def test_paired_baseline_is_not_replaced_by_its_mean(self):
        baseline=np.array([[.1,.2],[.9,.8],[.1,.2],[.9,.8]])
        actual=baseline+.05
        report=summary(actual,baseline,np.array(['a','b','c','d']))
        for value in report.values():np.testing.assert_allclose(value['delta_ci']['ci95'],[.05,.05])
        wrong=summary(actual,np.tile(baseline.mean(0),(4,1)),np.array(['a','b','c','d']))
        self.assertLess(wrong['ndcg']['delta_ci']['ci95'][0],0.)

    def test_gate_uses_query_features_only(self):
        gate=dict(feature=1,direction=-1,threshold=.2)
        np.testing.assert_array_equal(gate_mask(gate,np.array([[99,.1],[99,.3]])),[True,False])


if __name__=='__main__':unittest.main()
