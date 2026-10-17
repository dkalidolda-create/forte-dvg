import copy
import unittest

from ml.datasets.import_grounding import validate_annotations
from ml.roi_cash import derive_roi


class GroundingValidationTests(unittest.TestCase):
    def setUp(self):
        self.review = {'manifest_hash': 'version', 'videos': [{'video_id': 'train-video',
                       'frames': [{'timestamp_sec': 2.0, 'image': 'train-video/001.jpg'}]}]}
        self.payload = {'schema_version': 1, 'split': 'train', 'manifest_sha256': 'version',
                        'annotations': [{'video_id': 'train-video', 'timestamp_sec': 2.0,
                        'operation_state': 'ended', 'boxes': [
                        {'object': 'banknotes', 'x': .1, 'y': .1, 'width': .2, 'height': .3}], 'note': ''}]}

    def test_accepts_manual_boxes_without_relabelling_video(self):
        result = validate_annotations(self.payload, self.review)
        self.assertEqual(result[0]['boxes'][0]['object'], 'banknotes')
        self.assertNotIn('target', result[0])

    def test_roi_uses_banknotes_from_train_only(self):
        roi = derive_roi(self.payload)
        self.assertAlmostEqual(roi[0], .06)
        self.assertAlmostEqual(roi[2], .34)
        self.payload['split'] = 'validation'
        with self.assertRaises(ValueError):
            derive_roi(self.payload)

    def test_rejects_validation_frame_or_another_dataset(self):
        for field, value in [('video_id', 'validation-video'), ('timestamp_sec', 3.0)]:
            invalid = copy.deepcopy(self.payload)
            invalid['annotations'][0][field] = value
            with self.assertRaises(ValueError):
                validate_annotations(invalid, self.review)
        self.payload['manifest_sha256'] = 'another-version'
        with self.assertRaises(ValueError):
            validate_annotations(self.payload, self.review)

    def test_rejects_invalid_coordinates_and_duplicates(self):
        for value in [float('nan'), float('inf'), -1, 2, True]:
            invalid = copy.deepcopy(self.payload)
            invalid['annotations'][0]['boxes'][0]['width'] = value
            with self.assertRaises(ValueError):
                validate_annotations(invalid, self.review)
        self.payload['annotations'] *= 2
        with self.assertRaises(ValueError):
            validate_annotations(self.payload, self.review)


if __name__ == '__main__':
    unittest.main()
