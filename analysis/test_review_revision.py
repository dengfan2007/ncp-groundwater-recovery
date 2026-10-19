"""Scientific invariants for new review diagnostics."""
import unittest
import numpy as np
import pandas as pd
from ncp_review_revision import fit_shape, spatial_contrast


class ReviewTests(unittest.TestCase):
    def test_step_and_slope_are_distinguished_with_a_gap(self):
        dates=pd.date_range('2015-01-01',periods=116,freq='MS')
        t=np.arange(116)/12
        y=-t+8*(t>=6.5)+2*np.sin(2*np.pi*t)
        y[29:40]=np.nan
        step=fit_shape(dates,y,'step')
        hinge=fit_shape(dates,y,'hinge')
        self.assertEqual(step['date'],'2021-07-01')
        self.assertAlmostEqual(step['step_cm'],8.,places=8)
        self.assertAlmostEqual(step['post_slope'],-1.,places=8)
        self.assertLess(step['bic'],hinge['bic'])

    def test_continuous_change_recovers_units_and_date(self):
        dates=pd.date_range('2015-01-01',periods=116,freq='MS')
        t=np.arange(116)/12
        y=-2*t+5*np.maximum(t-5,0)+np.cos(2*np.pi*t)
        y[29:40]=np.nan
        r=fit_shape(dates,y,'hinge')
        self.assertEqual(r['date'],'2020-01-01')
        self.assertAlmostEqual(r['pre_slope'],-2.,places=8)
        self.assertAlmostEqual(r['post_slope'],3.,places=8)

    def test_no_interval_with_one_region(self):
        d=pd.DataFrame(dict(grid_lon=[113.,115.],grid_lat=[35.,36.],region_tag=['south','south'],slope=[1.,2.]))
        r=spatial_contrast(d,n_boot=50)
        self.assertTrue(np.isnan(r['estimate']))
        self.assertTrue(np.isnan(r['ci_low']))
        self.assertEqual(r['valid_replicates'],0)


if __name__=='__main__': unittest.main()
