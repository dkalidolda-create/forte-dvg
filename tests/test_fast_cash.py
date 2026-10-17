import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.fast_cash import confirmed_indexes, parse_indexed_occupancy, parse_occupancy
from ml.vision_crops import RidgeCropClassifier


class FastCashTests(unittest.TestCase):
    def test_missing_explicit_frame_is_uncertain_without_shifting(self):
        self.assertEqual(parse_indexed_occupancy('{"1":0,"2":1}', 3), [2, 0, 1])
        for invalid in ['{"3":1}', '{"0":true}', '[0,1,1]', '{}']:
            with self.assertRaises(ValueError):
                parse_indexed_occupancy(invalid, 3)

    def test_cash_during_staff_presence_is_not_confirmed(self):
        self.assertEqual(confirmed_indexes([True, True, True], [0, 0, 0]), [])
        self.assertEqual(confirmed_indexes([True, True, True], [0, 1, 1]), [1])

    def test_one_frame_or_uncertainty_does_not_confirm_departure(self):
        self.assertEqual(confirmed_indexes([True, False], [1, 1]), [])
        self.assertEqual(confirmed_indexes([True, True], [1, 2]), [])
        self.assertEqual(confirmed_indexes([True], [1]), [])

    def test_role_parser_rejects_missing_frames_or_bool(self):
        self.assertEqual(parse_occupancy('[0,1,2]', 3), [0, 1, 2])
        for values in [[0, 1], [0, True, 1], [0, 3, 0], {'0': 1}]:
            with self.assertRaises(ValueError):
                parse_occupancy(json.dumps(values), 3)

    def test_four_object_classes_survive_model_roundtrip(self):
        classes = ['banknotes', 'documents', 'cash_storage', 'collection_bag']
        vectors = np.eye(4)
        classifier = RidgeCropClassifier(classes).fit(vectors, np.arange(4))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'classifier.npz'
            classifier.save(path)
            restored = RidgeCropClassifier.load(path)
            self.assertEqual(restored.classes, classes)
            np.testing.assert_array_equal(restored.predict(vectors), np.arange(4))


if __name__ == '__main__':
    unittest.main()
