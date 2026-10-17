import json
import unittest

from ml.evaluation.score import score
from ml.grounded_cash import parse_label
from ml.local_cash import parse_decision


class EvaluationTests(unittest.TestCase):
    def test_grounded_decision_does_not_guess_from_explanation(self):
        self.assertEqual(parse_label(' B\n'), 0)
        self.assertEqual(parse_label('**A**'), 1)
        self.assertIsNone(parse_label('U'))
        for answer in ['A or B', 'B because service continues', '', 'AB']:
            with self.assertRaises(ValueError):
                parse_label(answer)

    def test_metrics_with_failures(self):
        truth = [{'video_id': str(i), 'target': t, 'event_sec': 10 if t else ''}
                 for i, t in enumerate([1, 1, 1, 0, 0, 0])]
        predictions = [
            {'video_id': '0', 'status': 'done', 'prediction': 1, 'event_times_sec': [12]},
            {'video_id': '1', 'status': 'done', 'prediction': 0},
            {'video_id': '2', 'status': 'failed', 'prediction': None},
            {'video_id': '3', 'status': 'done', 'prediction': 1},
            {'video_id': '4', 'status': 'done', 'prediction': 0},
            {'video_id': '5', 'status': 'done', 'prediction': None},
        ]
        metrics, rows = score(truth, predictions)
        self.assertAlmostEqual(metrics['accuracy_all_videos'], 2/6)
        self.assertAlmostEqual(metrics['accuracy_resolved_only'], .5)
        self.assertAlmostEqual(metrics['precision'], .5)
        self.assertAlmostEqual(metrics['recall_conservative'], 1/3)
        self.assertAlmostEqual(metrics['f1_conservative'], .4)
        self.assertAlmostEqual(metrics['specificity_conservative'], 1/3)
        self.assertAlmostEqual(metrics['balanced_accuracy_conservative'], 1/3)
        self.assertEqual(metrics['mean_time_error_sec_matched_only'], 2)
        self.assertEqual(len(rows), 6)

    def test_missing_prediction_is_not_dropped(self):
        metrics, _ = score([{'video_id': 'x', 'target': 1}], [])
        self.assertEqual(metrics['accuracy_all_videos'], 0)
        self.assertEqual(metrics['unresolved_positive'], 1)
        self.assertIsNone(metrics['precision'])

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            score([{'video_id': 'x', 'target': 1}] * 2, [])
        with self.assertRaises(ValueError):
            score([{'video_id': 'x', 'target': 1}],
                  [{'video_id': 'x', 'status': 'done', 'prediction': 0}] * 2)

    def test_invalid_model_response(self):
        response = {'decision': 'present', 'operation_end_visible': True,
                    'cash_visible_after_end': True, 'outside_storage': True,
                    'event_time_sec': 2.0}
        self.assertEqual(parse_decision(json.dumps(response), [0,2,4]).event_time_sec, 2)
        response['event_time_sec'] = 99.0
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(response), [0,2,4])
        response['event_time_sec'] = 2.0
        response['operation_end_visible'] = False
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(response), [0,2,4])


if __name__ == '__main__':
    unittest.main()
