"""Numerical invariants, not evidence of real-world breakpoint accuracy."""
import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch
from ncp_signal_reconstruction import diagonal_mean,ssa_fill,reconstruct,improved_ensemble
from ncp_story_analysis import report_candidates


class ReconstructionTests(unittest.TestCase):
    def test_diagonal_average_recovers_linear_hankel(self):
        x=np.arange(80,dtype=float)
        self.assertTrue(np.allclose(diagonal_mean(np.lib.stride_tricks.sliding_window_view(x,20).T),x))

    def test_ssa_keeps_observations_and_calendar(self):
        x=.1*np.arange(120)+np.sin(2*np.pi*np.arange(120)/12)
        x[55:66]=np.nan
        filled,info=ssa_fill(x)
        self.assertEqual(len(filled),120)
        self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(filled[np.isfinite(x)],x[np.isfinite(x)])

    def test_ensemble_complete_and_deterministic(self):
        t=np.arange(120);x=.03*t+np.sin(2*np.pi*t/12)+.3*np.cos(2*np.pi*t/4)
        m,r=improved_ensemble(x,trials=10)
        np.testing.assert_allclose(m.sum(axis=0)+r,x,atol=1e-10)
        a,_=reconstruct(x,trials=10);b,_=reconstruct(x,trials=10)
        np.testing.assert_array_equal(a,b)

    def test_constant_signal_preserved(self):
        x=np.ones(100)*4
        t,_=reconstruct(x,trials=10)
        np.testing.assert_array_equal(x,t)

    def test_filled_values_do_not_create_observed_support(self):
        dates=pd.date_range('2003-01-01',periods=200,freq='MS')
        t=np.arange(200);y=-.1*t+.6*np.maximum(t-144,0)
        qualification=y.copy();qualification[170:]=np.nan
        result=SimpleNamespace(trend=SimpleNamespace(cp=np.array([2015.,2016+4/12]),cpPr=np.array([.8,.99]),Y=y))
        with patch('ncp_story_analysis.rb.beast',return_value=result):
            choices,candidates,_=report_candidates(dates,y,qualification)
        self.assertTrue(candidates[0]['qualifies'])
        self.assertFalse(candidates[1]['qualifies'])
        self.assertTrue(all(c['selected_date']=='2015-01-01' for c in choices))

    def test_high_probability_does_not_qualify_declining_trend(self):
        dates=pd.date_range('2003-01-01',periods=200,freq='MS');y=-.1*np.arange(200)
        result=SimpleNamespace(trend=SimpleNamespace(cp=np.array([2015.]),cpPr=np.array([1.]),Y=y))
        with patch('ncp_story_analysis.rb.beast',return_value=result):
            choices,_,_=report_candidates(dates,y,y)
        self.assertFalse(any(c['qualifies'] for c in choices))


if __name__=='__main__':unittest.main()
