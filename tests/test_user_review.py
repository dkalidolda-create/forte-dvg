import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.datasets.prepare_user_review import prepare
from ml.event_schema import parse
from ml.rules import RULES, grounded_for_training
from ml.temporal_features import aggregate, binary_scores, fit_binary


class ReviewTests(unittest.TestCase):
    def fixture(self, folder, crossed=False):
        root = Path(folder)
        review = root/'review.json'
        review.write_text(json.dumps({'videos': [
            {'video_id': 'X-12', 'user_verdict': 'violation', 'categories': ['room_not_locked_on_departure'], 'original': 'review'},
            {'video_id': 'X-13', 'user_verdict': 'violation', 'categories': ['bag_unpacked_into_safe'], 'original': 'review'}]}))
        manifest = root/'manifest.csv'
        manifest.write_text('video_id,target,event_sec,split,exact_file_group,video_path,sha256\n'
                            'X-12,0,,train,same,a,abc\n'
                            f'X-13,1,4,{"validation" if crossed else "train"},same,b,abc\n')
        return review, manifest

    def test_conflicting_reference_is_preserved_and_quarantined(self):
        with tempfile.TemporaryDirectory() as folder:
            review, manifest = self.fixture(folder)
            output = Path(folder)/'out'
            prepare(review, manifest, output)
            rows = json.loads((output/'adjudication.json').read_text())['records']
            self.assertEqual(rows[0]['organizer_verdict'], 'no_violation')
            self.assertEqual(rows[0]['user_verdict'], 'violation')
            self.assertEqual(rows[0]['training_label_status'], 'quarantine_label_conflict')
            self.assertEqual(rows[1]['training_label_status'], 'observation_only_no_direct_prohibition')

    def test_exact_duplicate_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as folder:
            review, manifest = self.fixture(folder, crossed=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate group crosses splits'):
                prepare(review, manifest, Path(folder)/'out')

    def test_explicit_override_changes_training_target_without_erasing_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            review, manifest = self.fixture(folder)
            raw = json.loads(review.read_text())
            raw['videos'][0]['training_override'] = {'use_user_label': True, 'source': 'direct user instruction'}
            review.write_text(json.dumps(raw))
            output = Path(folder)/'out'
            prepare(review, manifest, output)
            row = json.loads((output/'adjudication.json').read_text())['records'][0]
            self.assertTrue(row['label_conflict'])
            self.assertEqual(row['organizer_verdict'], 'no_violation')
            self.assertEqual(row['training_verdict'], 'violation')
            self.assertEqual(row['training_label_status'], 'explicit_user_override_with_reference_preserved')

    def test_context_does_not_become_a_direct_prohibition(self):
        for category in ['bag_unpacked_into_safe', 'laptop_left_on', 'lights_left_on', 'client_personal_items_on_desk']:
            self.assertFalse(grounded_for_training(category))
        self.assertIn('работников', RULES['client_personal_items_on_desk'].explanation)
        self.assertIn('цифровой', RULES['identity_check_not_visible'].exclusions)


class TemporalTests(unittest.TestCase):
    def test_chronology_changes_features_and_nonfinite_is_rejected(self):
        frames = np.arange(36).reshape(6, 2, 3).astype(float)
        self.assertFalse(np.allclose(aggregate(frames), aggregate(frames[::-1])))
        frames[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            aggregate(frames)

    def test_head_is_fitted_from_features_with_both_classes(self):
        vectors = np.array([[1., 0], [.9, .1], [0, 1], [.1, .9]])
        support, dual = fit_binary(vectors, [1, 1, 0, 0])
        self.assertEqual((binary_scores(vectors, support, dual) >= 0).tolist(), [True, True, False, False])
        with self.assertRaisesRegex(ValueError, 'Both classes'):
            fit_binary(vectors, [1, 1, 1, 1])


class EventOutputTests(unittest.TestCase):
    def test_multiple_types_and_exact_frame_references(self):
        value = {'verdict': 'violation', 'events': [
            {'category': 'cash_left_after_service', 'frame': 3},
            {'category': 'safe_left_open', 'frame': 4}]}
        self.assertEqual(parse(json.dumps(value), 6), value)
        for bad in [True, -1, 6, 2.5]:
            value['events'][0]['frame'] = bad
            with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
                parse(json.dumps(value), 6)

    def test_inconsistent_verdict_and_hallucinated_rule_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            parse('{"verdict":"violation","events":[]}', 6)
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            parse('{"verdict":"violation","events":[{"category":"invented_rule","frame":0}]}', 6)

    def test_repeated_category_at_different_times_is_allowed_but_duplicate_is_not(self):
        value = {'verdict': 'violation', 'events': [
            {'category': 'cash_left_after_service', 'frame': 1},
            {'category': 'cash_left_after_service', 'frame': 4}]}
        self.assertEqual(len(parse(json.dumps(value), 6)['events']), 2)
        value['events'][1]['frame'] = 1
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            parse(json.dumps(value), 6)


if __name__ == '__main__':
    unittest.main()
