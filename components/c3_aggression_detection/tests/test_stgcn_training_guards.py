import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from src.train_stgcn import fit_model, training_blockers


class TrainingGuardTests(unittest.TestCase):
    def test_no_positive_precursor_blocks_fit(self):
        empty=dict(supervised_windows_per_head=[0,10],positive_windows_per_head=[0,5])
        self.assertTrue(training_blockers(empty,empty,'early'))
        self.assertFalse(training_blockers(empty,empty,'active-fight'))

    def test_positive_without_negative_not_ready(self):
        positive=dict(supervised_windows_per_head=[4,0],positive_windows_per_head=[4,0])
        self.assertIn('No eligible negative early windows in train',training_blockers(positive,positive))

    def test_test_split_rejected_before_reading_samples(self):
        with self.assertRaisesRegex(ValueError,'test is prohibited'):
            fit_model(SimpleNamespace(split='test'),SimpleNamespace(split='val'),'unused.pt')

    def test_existing_risknet_trainer_requires_locked_manifest(self):
        from src.train import train
        with patch('src.lock_splits.load_locked',side_effect=ValueError('unlocked')):
            with self.assertRaisesRegex(ValueError,'unlocked'):
                train('video','unlocked.csv','unused.json','unused.pt')

    def test_different_cache_configurations_cannot_be_compared(self):
        train=SimpleNamespace(split='train',provenance={'feature_contract_sha256':'a'})
        val=SimpleNamespace(split='val',provenance={'feature_contract_sha256':'b'})
        with self.assertRaisesRegex(ValueError,'configurations differ'):
            fit_model(train,val,'unused.pt')

    def test_synthetic_training_loop_writes_only_new_checkpoint(self):
        class Synthetic:
            def __init__(self,split):self.split=split;self.provenance={'fixture':True,'split':split}
            def __len__(self):return 4
            def __getitem__(self,i):
                if i>=4:raise IndexError(i)
                return dict(sk=np.random.default_rng(i).normal(size=(8,17,2,3)).astype(np.float32),
                    joint_mask=np.ones((8,17,2),bool),observed_mask=np.ones((8,17,2),bool),
                    imputed_mask=np.zeros((8,17,2),bool),person_mask=np.ones((8,2),bool),
                    temporal_mask=np.ones(8,bool),frame_times=np.arange(8)/10,frame_indices=np.arange(8),
                    selected_ids=np.array([1,2]),quality=np.float32(.9),targets=np.array([i%2,i%2],np.float32),
                    target_mask=np.ones(2,bool),video_id=f'fixture_{i}',leakage_group=f'{self.split}_{i}',
                    split=self.split,cutoff_s=.7)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.pt'
            fit_model(Synthetic('train'),Synthetic('val'),path,epochs=1,batch_size=2)
            ck=torch.load(path,weights_only=True)
            self.assertEqual(ck['trained_heads'],['precursor'])
            self.assertFalse(ck['early_prediction_claim'])
            with self.assertRaises(FileExistsError):
                fit_model(Synthetic('train'),Synthetic('val'),path,epochs=1)


if __name__=='__main__':
    unittest.main()
