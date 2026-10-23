import unittest

import numpy as np
import torch

from src.advanced_experiment import build_model
from src.advanced_stgcn import AdvancedSTGCN, pose_streams
from src.optimized_readout import OptimizedReadoutC3, feature_mask, normalize_features
from src.optimize_model import F9, active_loss, choose_threshold, cross_source_thresholds


class ModelOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        torch.manual_seed(73)
        sk=torch.rand(2,48,17,2,3)*.5;sk[...,2]=.8
        tm=torch.ones(2,48,dtype=torch.bool);jm=torch.ones(sk.shape[:-1],dtype=torch.bool)
        times=torch.arange(48).repeat(2,1).double()/15
        return sk,tm,jm,times

    def test_legacy_advanced_forward_is_exactly_preserved(self):
        sk,tm,jm,times=self.fixture();model=AdvancedSTGCN().eval()
        values,geometry,people=pose_streams(sk,jm,times);features=[]
        with torch.no_grad():
            for encoder,value in zip(model.streams,values):
                z=encoder(value.permute(0,3,4,1,2).reshape(4,3,48,17)).reshape(2,2,32)*people[...,None]
                features.extend([z.sum(1)/people.sum(1).clamp_min(1)[:,None],(z[:,0]-z[:,1]).abs()])
            expected=model.fusion(torch.cat(features+[model.geometry(geometry)],-1))
            torch.testing.assert_close(expected,model(sk,tm,jm,times)['logits'],rtol=0,atol=0)

    def test_short_window_does_not_read_earlier_history(self):
        sk,tm,jm,times=self.fixture();model=AdvancedSTGCN().eval()
        with torch.no_grad():
            expected=model.window_features(sk,tm,jm,times,24)
            sk[:,:24]=1000;times[:,:24]=-1000
            torch.testing.assert_close(expected,model.window_features(sk,tm,jm,times,24),rtol=0,atol=0)

    def test_padding_is_excluded_from_features(self):
        sk,tm,jm,times=self.fixture();tm[:,:10]=False;model=AdvancedSTGCN().eval()
        with torch.no_grad():
            before=model.window_features(sk,tm,jm,times)
            sk[:,:10]=float('nan');times[:,:10]=float('nan')
            torch.testing.assert_close(before,model.window_features(sk,tm,jm,times),rtol=0,atol=0)

    def test_group_masks_and_normalization(self):
        mask=feature_mask(('joint','bone'))
        self.assertEqual(mask.sum(),128)
        x=torch.randn(4,224);normalized=normalize_features(x,torch.zeros(224),torch.ones(224),mask)
        self.assertFalse(normalized[:,128:].any())
        with self.assertRaises(ValueError):feature_mask(('unknown',))

    def test_each_loss_has_finite_correct_direction_gradient(self):
        for name in ('bce','weighted','focal'):
            logits=torch.tensor([0.,0.],requires_grad=True)
            loss=active_loss(logits,torch.tensor([0.,1.]),name,torch.tensor(3.))
            loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad[0],0);self.assertLess(logits.grad[1],0)

    def test_threshold_constraints_and_tied_scores(self):
        y=np.array([0,0,0,1,1]);p=np.array([.1,.2,.6,.6,.8])
        point=choose_threshold(y,p,'fpr10',role='val')
        self.assertEqual(point['metrics']['fp'],0);self.assertEqual(point['metrics']['tp'],1)
        self.assertEqual(choose_threshold(y,p,'f1',role='val')['metrics']['tp'],2)
        with self.assertRaises(ValueError):choose_threshold(y,p,role='test')

    def test_infeasible_minimum_precision_abstains(self):
        point=choose_threshold([1,0,0],[.1,.2,.3],'precision50',role='val')
        self.assertFalse(point['feasible']);self.assertEqual(point['metrics']['tp']+point['metrics']['fp'],0)

    def test_source_held_out_labels_and_f9_never_tune_their_threshold(self):
        sources=np.array(['a','a','b','b','c','c',F9,F9])
        y=np.array([0,1]*4);p=np.array([.1,.8,.2,.7,.3,.6,.4,.9])
        report,_,thresholds=cross_source_thresholds(y,p,sources,'f1')
        changed=y.copy();changed[:2]=[1,0];changed[-2:]=[1,0]
        _,_,after=cross_source_thresholds(changed,p,sources,'f1')
        np.testing.assert_array_equal(thresholds[:2],after[:2])
        for fold in report['folds']:
            self.assertNotIn(fold['source'],fold['tuning_sources']);self.assertNotIn(F9,fold['tuning_sources'])

    def test_runtime_variant_strict_loading_and_onset_preservation(self):
        sk,tm,jm,times=self.fixture();model=OptimizedReadoutC3().eval()
        model.readout_weight.normal_(0,.01);model.readout_bias.copy_(torch.tensor([-.1,0,.1]))
        restored=build_model('optimized_readout').eval();restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            result=restored(sk,tm,jm,times)
            torch.testing.assert_close(result['logits'],model(sk,tm,jm,times)['logits'],rtol=0,atol=0)
            torch.testing.assert_close(result['logits'][:,0],model.onset(sk,tm,jm)['logits'][:,0],rtol=0,atol=0)
            empty=restored(sk,torch.zeros_like(tm),torch.zeros_like(jm),times)
            self.assertFalse(empty['valid_window'].any());self.assertTrue(torch.isfinite(empty['logits']).all())


if __name__=='__main__':unittest.main()
