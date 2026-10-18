import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.inference import analyze, result_to_candidates


class AdapterTests(unittest.TestCase):
    def scans(self):
        return {'timestamps': [0., 2., 4., 6., 8.], 'occupancy': [0, 1, 1, 1, 0],
                'scans': [{'cash_detected': True} for _ in range(5)]}

    def test_positive_episode_has_real_timestamp_and_one_to_three_evidence_times(self):
        result = {'prediction': 1, 'duration_sec': 10, 'event_times_sec': [2.]}
        candidates = result_to_candidates(result, self.scans())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].timecode_sec, 2.)
        self.assertEqual(candidates[0].evidence_timestamps, [0., 2., 4.])
        self.assertEqual(candidates[0].source_section, 'Пункт 20, абзац 2')

    def test_uncertainty_never_becomes_an_empty_list(self):
        with self.assertRaisesRegex(RuntimeError, 'analysis_uncertain'):
            result_to_candidates({'prediction': None}, {})
        self.assertEqual(result_to_candidates({'prediction': 0}, {}), [])
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            result_to_candidates({'prediction': False}, {})

    def test_positive_without_temporal_support_is_rejected(self):
        scans = self.scans()
        scans['occupancy'] = [0]*5
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            result_to_candidates({'prediction': 1, 'duration_sec': 10}, scans)
        scans = self.scans()
        scans['timestamps'][0] = -1.
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            result_to_candidates({'prediction': 1, 'duration_sec': 10}, scans)

    def test_adapter_passes_only_the_uploaded_path_to_the_detector(self):
        class Detector:
            def analyze(self, video_path, log_dir):
                self.path = video_path
                log_dir.mkdir()
                return {'prediction': 0, 'event_times_sec': []}
        detector = Detector()
        with tempfile.TemporaryDirectory() as folder, patch('ml.inference.load_detector', return_value=detector), patch.dict('os.environ', {'STORAGE_ROOT': folder}):
            path = Path(folder)/'opaque-upload.mov'
            self.assertEqual(analyze(path), [])
            self.assertEqual(detector.path, path)


if __name__ == '__main__':
    unittest.main()
