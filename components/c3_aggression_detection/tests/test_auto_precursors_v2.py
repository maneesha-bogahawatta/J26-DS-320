import unittest
import numpy as np
from src.auto_annotate_precursors_v2 import TemporalCues, propose_intervals, parse_onset

class TestV2(unittest.TestCase):
    def person(self,pid,x):
        kp=np.zeros((17,2)); kp[:,1]=100
        conf=np.ones(17)
        return {'id':pid,'center':np.array([x,100.]),'height':100.,'kp':kp,'conf':conf}
    def test_parse(self):
        self.assertEqual(parse_onset({'intervals':'[[16,18]]'}),16)
    def test_no_single_rapid_motion_trigger(self):
        c=TemporalCues();c.update([self.person(1,0)],0.)
        r=c.update([self.person(1,100)],.2)
        self.assertNotIn('rapid_approach',r['signals'])
        self.assertFalse(r['dyadic'])
    def test_interval_clamped(self):
        obs=[{'timestamp':9.6,'score':.7,'dyadic':True},{'timestamp':9.8,'score':.7,'dyadic':True}]
        self.assertEqual(propose_intervals(obs,onset=10.),[[9.6,10.]])
    def test_gap_breaks(self):
        obs=[{'timestamp':5.,'score':.8,'dyadic':True},{'timestamp':6.,'score':.8,'dyadic':True}]
        self.assertEqual(propose_intervals(obs),[])
    def test_causal_past_unchanged(self):
        a=TemporalCues();b=TemporalCues()
        sequence=[[self.person(1,0),self.person(2,150)], [self.person(1,30),self.person(2,130)]]
        x=[a.update(p,t) for t,p in enumerate(sequence)]
        y=[b.update(p,t) for t,p in enumerate(sequence)]
        self.assertEqual(x,y)
if __name__=='__main__':unittest.main()
