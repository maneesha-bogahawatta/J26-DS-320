import tempfile
import unittest
from pathlib import Path

from src.foundation_io import write_new
from src.lock_splits import assign, validate_locked


def sample(n=30):
    return [dict(video=f'F_{i}.mp4', group=f'g{i}', intervals='[[5,8]]' if i%2 else '[]',
                 split='', content_sha256=f'h{i}') for i in range(n)]


class SplitTests(unittest.TestCase):
    def test_reproducible_and_input_order_independent(self):
        rows = sample()
        self.assertEqual(assign(rows, {'f_0'}), assign(rows[::-1], {'f_0'}))
        self.assertEqual(assign(rows, {'f_0'})[0]['split'], 'test')

    def test_duplicate_test_companion_cannot_train(self):
        rows = sample()
        rows[1]['content_sha256'] = rows[0]['content_sha256']
        got = {r['video_id']:r for r in assign(rows, {'f_0'})}
        self.assertEqual(got['f_1']['split'], 'test')

    def test_transitive_known_groups(self):
        rows = sample()
        rows[1]['group'] = rows[0]['group']
        got = {r['video_id']:r for r in assign(rows, {'f_0'}, {'f_1':'scene', 'f_2':'scene'})}
        self.assertEqual(got['f_2']['split'], 'test')

    def test_conflict_fails_closed(self):
        rows = sample()
        rows[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            assign(rows, {'f_0'})

    def test_missing_test_or_ambiguous_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            assign(sample(), {'absent'})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            assign(sample()+[sample()[0]], set())

    def test_leakage_validator_rejects_tampering(self):
        rows = assign(sample(), {'f_0'})
        rows[1]['content_sha256'] = rows[0]['content_sha256']
        rows[1]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate_locked(rows, {'f_0'})

    def test_artifacts_never_overwrite_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'review.csv'
            write_new(path, 'reviewed')
            write_new(path, 'reviewed')
            with self.assertRaises(FileExistsError):
                write_new(path, 'changed')
            self.assertEqual(path.read_text(), 'reviewed')


if __name__ == '__main__':
    unittest.main()
