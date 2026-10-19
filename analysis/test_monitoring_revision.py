"""Tests for calendar, decision and unit errors that would change the research."""
import unittest
import numpy as np
import pandas as pd
from ncp_monitoring_core import design,seed,select
from ncp_monitoring_diagnostics import block_indices,keep_dates
from ncp_monitoring_inference import wilson,choose

class MonitoringTests(unittest.TestCase):
    def test_calendar_gap_does_not_compress_slope(self):
        dates=pd.date_range('2019-01-01','2024-08-01',freq='MS');X=design(dates);y=X@np.array([2.,3.,.2,.4,.1,-.2])
        keep=keep_dates(dates,'exclude_both');coef=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0]
        self.assertAlmostEqual(coef[1],3.,places=12)
    def test_truncated_blocks_never_cross_excluded_run(self):
        keep=np.array([True]*8+[False]*3+[True]*20);ix,info=block_indices(keep,n=50,rng=np.random.default_rng(1))
        self.assertTrue((ix[:,:8]<8).all());self.assertTrue((ix[:,11:]>=11).all());self.assertEqual(info['run_lengths'],[8,20]);self.assertGreater(info['truncated_fraction'],0)
    def test_wilson_count_boundary(self):
        self.assertLessEqual(wilson(16,500,True)[1],.05);self.assertGreater(wilson(17,500,True)[1],.05)
    def test_state_water_unit_integration(self):
        theta=np.ones(4)*.25
        self.assertAlmostEqual(theta@np.array([7,21,72,100]),50.)
        self.assertAlmostEqual(theta@np.array([7,21,72,189]),72.25)
        self.assertEqual(.1*1000,100.)
    def test_head_rule_uses_signs_not_storage_gain(self):
        c=dict(date='2020-01-01',probability=.9,local_pre_slope=-.001,local_post_slope=.001,pre_slope=-.001,post_slope=.001)
        self.assertIsNotNone(select([c],local=True));self.assertIsNone(select([c],gain=.5))
    def test_threshold_reselection_and_ties(self):
        d=pd.DataFrame([dict(task_id='x',date='2020-01-01',probability=.9,pre_slope=-2,post_slope=1),dict(task_id='x',date='2021-01-01',probability=.9,pre_slope=-1,post_slope=1)])
        self.assertEqual(choose(d,.5,.5).iloc[0].selected_date,'2021-01-01');self.assertEqual(choose(d,.5,.5,'earliest').iloc[0].selected_date,'2020-01-01')
        self.assertEqual(len(choose(d,.95,.5)),0)
    def test_seed_pairing_and_holdout_separation(self):
        a={seed('calibration','no_recovery','low',r,'calibration','2018-01-01',.6,'noise') for r in range(500)}
        b={seed('calibration','no_recovery','low',r,'validation','2018-01-01',.6,'noise') for r in range(500)}
        self.assertTrue(a.isdisjoint(b));self.assertEqual(seed('pair',1),seed('pair',1))
    def test_right_edge_counts_eighteen_including_candidate(self):
        for end,expected in [('2023-12-01','2022-07-01'),('2024-08-01','2023-03-01')]:
            dates=pd.date_range('2005-01-01',end,freq='MS');positions=[i for i in range(len(dates)) if len(dates[max(0,i-36):i])>=18 and len(dates[i:i+36])>=18]
            self.assertEqual(str(dates[max(positions)].date()),expected)

if __name__=='__main__':unittest.main()
