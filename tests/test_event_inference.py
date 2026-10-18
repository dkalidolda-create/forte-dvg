import unittest

from ml.event_inference import candidates_from_events


class EventInferenceTests(unittest.TestCase):
    def result(self, events):
        return {'verdict': 'violation' if events else 'no_violation', 'events': events,
                'sampled_timestamps': [0., 3., 6., 9., 12., 15.], 'duration_sec': 15.5}

    def test_combination_keeps_two_types_sources_and_real_frame_times(self):
        events = [{'category': 'cash_left_after_service', 'frame': 3},
                  {'category': 'room_not_locked_on_departure', 'frame': 5}]
        candidates = candidates_from_events(self.result(events))
        self.assertEqual([c.timecode_sec for c in candidates], [9., 15.])
        self.assertEqual(candidates[0].evidence_timestamps, [6., 9., 12.])
        self.assertEqual(candidates[1].evidence_timestamps, [12., 15.])
        self.assertTrue(candidates[0].source_url.endswith('V1900019680'))
        self.assertTrue(candidates[1].source_url.endswith('V2000020075'))

    def test_context_source_is_not_presented_as_a_direct_prohibition(self):
        candidate = candidates_from_events(self.result([{'category': 'laptop_left_on', 'frame': 5}]))[0]
        self.assertIn('Событие для проверки', candidate.type)
        self.assertIn('Прямой запрет', candidate.description)
        self.assertIn('не подтверждён', candidate.description)
        self.assertIn('не прямой запрет', candidate.source_section)

    def test_invalid_source_times_cannot_become_evidence(self):
        self.assertEqual(candidates_from_events(self.result([])), [])
        value = self.result([{'category': 'cash_left_after_service', 'frame': 0}])
        value['sampled_timestamps'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            candidates_from_events(value)


if __name__ == '__main__':
    unittest.main()
