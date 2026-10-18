import json
import unittest

from ml.training.label_mask import label_weights


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text],
                'offset_mapping': [(i, i+1) for i in range(len(text))]}


class LabelMaskTests(unittest.TestCase):
    def test_negative_supervises_only_first_decision_token(self):
        body = json.dumps({'verdict': 'no_violation', 'events': []})
        weights = label_weights(CharacterTokenizer(), body, [9]+list(map(ord, body))+[8])
        self.assertEqual(sum(weights), 1)
        self.assertEqual(weights[1+body.index('no_violation')], 1)
        self.assertEqual(sum(w > 0 for w in weights), 1)

    def test_multiple_categories_have_fixed_total_weight_and_no_frame_loss(self):
        body = json.dumps({'verdict': 'violation', 'events': [
            {'category': 'cash_left_after_service', 'frame': 3},
            {'category': 'safe_left_open', 'frame': 5}]})
        weights = label_weights(CharacterTokenizer(), body, list(map(ord, body)))
        self.assertAlmostEqual(sum(weights), 1.25)
        self.assertEqual(weights[body.index('3')], 0)
        self.assertEqual(weights[body.index('5')], 0)
        self.assertEqual(weights[body.index('violation')], 1)

    def test_misalignment_fails_instead_of_training_on_prompt(self):
        with self.assertRaises(ValueError):
            label_weights(CharacterTokenizer(), '{"verdict":"no_violation"}', [1, 2])
