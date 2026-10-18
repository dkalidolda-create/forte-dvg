import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.temporal_events import TemporalEventHead


class TemporalEventTests(unittest.TestCase):
    def fixture(self):
        vectors = np.eye(6)
        binary = np.array([0, 1, 0, 1, 1, 0])
        labels = np.array([[0, 0], [1, 0], [0, 0], [0, 1], [1, 1], [0, 0]])
        times = labels*np.array([0.4, 0.8])
        return vectors, TemporalEventHead.fit(vectors, binary, labels, times, ['a', 'b'])

    def test_fits_two_event_combination_and_keeps_negative_empty(self):
        vectors, model = self.fixture()
        self.assertEqual(model.predict(vectors[0])['events'], [])
        self.assertEqual({e['category'] for e in model.predict(vectors[4])['events']}, {'a', 'b'})
        for e in model.predict(vectors[4])['events']:
            self.assertGreaterEqual(e['normalized_time'], 0)
            self.assertLess(e['normalized_time'], 1)

    def test_numeric_model_round_trip_without_reference_files(self):
        vectors, model = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.npz'
            model.save(path)
            loaded = TemporalEventHead.load(path)
            self.assertEqual(loaded.predict(vectors[3])['events'], model.predict(vectors[3])['events'])

    def test_rejects_nonfinite_features_and_inconsistent_targets(self):
        vectors, model = self.fixture()
        with self.assertRaises(ValueError):
            model.predict(np.full(6, np.nan))
        with self.assertRaises(ValueError):
            TemporalEventHead.fit(vectors, [0, 1, 0, 1, 1, 0], np.zeros((6, 2)), np.zeros((6, 2)), ['a', 'b'])
